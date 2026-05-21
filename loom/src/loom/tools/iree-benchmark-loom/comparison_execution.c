// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/comparison_execution.h"

#include <inttypes.h>
#include <string.h>

#include "iree/hal/api.h"
#include "loom/ir/module.h"
#include "loom/tooling/execution/hal/benchmark.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tools/iree-benchmark-loom/case_execution.h"
#include "loom/tools/iree-benchmark-loom/dispatch_benchmark.h"
#include "loom/tools/iree-benchmark-loom/hal_actual.h"
#include "loom/tools/iree-benchmark-loom/module_query.h"
#include "loom/tools/iree-benchmark-loom/report.h"
#include "loom/tools/iree-benchmark-loom/testbench.h"

static iree_status_t iree_benchmark_loom_validate_sample_flag(
    const iree_benchmark_loom_options_t* options, iree_host_size_t sample_count,
    iree_host_size_t* out_begin_sample, iree_host_size_t* out_end_sample) {
  if (options->sample_ordinal < 0) {
    *out_begin_sample = 0;
    *out_end_sample = sample_count;
    return iree_ok_status();
  }
  const iree_host_size_t sample_ordinal =
      (iree_host_size_t)options->sample_ordinal;
  if (sample_ordinal >= sample_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "--sample=%" PRIhsz
                            " exceeds selected benchmark sample count %" PRIhsz,
                            sample_ordinal, sample_count);
  }
  *out_begin_sample = sample_ordinal;
  *out_end_sample = sample_ordinal + 1;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_emit_dispatch_benchmark_result(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    const iree_benchmark_loom_event_sink_t* event_sink) {
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_event_sink_emit_benchmark_result(
      event_sink, run, candidate, IREE_BENCHMARK_LOOM_INDEX_INVALID,
      module_plan->module, benchmark_plan, case_plan, policy, benchmark_result,
      correctness_sample_count, correctness_failed_sample_count));
  return iree_benchmark_loom_event_sink_emit_profile(
      event_sink, run, candidate, IREE_BENCHMARK_LOOM_INDEX_INVALID,
      module_plan->module, benchmark_plan, case_plan, policy, benchmark_result);
}

static iree_status_t iree_benchmark_loom_comparison_candidate_record_timing(
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result) {
  if (!benchmark_result->executed || !benchmark_result->passed ||
      !benchmark_result->has_hal_benchmark) {
    return iree_ok_status();
  }
  if (candidate->sample_count >= candidate->sample_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "interleaved comparison collected more timing rows than planned");
  }
  const loom_run_benchmark_timing_stats_t* dispatch_timing =
      &benchmark_result->hal_benchmark.timing.operation_timing;
  candidate->p50_samples[candidate->sample_count] = dispatch_timing->p50_ns;
  candidate->p90_samples[candidate->sample_count] = dispatch_timing->p90_ns;
  ++candidate->sample_count;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_dispatch_comparison_sample_window(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const iree_benchmark_loom_options_t* options, bool has_fixed_sample,
    iree_host_size_t fixed_sample, iree_host_size_t* out_begin_sample,
    iree_host_size_t* out_end_sample) {
  if (has_fixed_sample) {
    *out_begin_sample = fixed_sample;
    *out_end_sample = fixed_sample + 1;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_validate_sample_flag(
      options, benchmark_plan->sample_count, out_begin_sample, out_end_sample));
  if (*out_end_sample != *out_begin_sample + 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "interleaved comparisons require exactly one concrete sample per "
        "candidate; use --sample= to select one of %" PRIhsz " samples",
        benchmark_plan->sample_count);
  }
  return iree_ok_status();
}

static bool iree_benchmark_loom_sample_attr_equal(loom_attribute_t lhs,
                                                  loom_attribute_t rhs) {
  if (lhs.kind != rhs.kind) {
    return false;
  }
  switch ((loom_attr_kind_t)lhs.kind) {
    case LOOM_ATTR_I64:
      return loom_attr_as_i64(lhs) == loom_attr_as_i64(rhs);
    case LOOM_ATTR_F64:
      return loom_attr_as_f64(lhs) == loom_attr_as_f64(rhs);
    case LOOM_ATTR_BOOL:
      return loom_attr_as_bool(lhs) == loom_attr_as_bool(rhs);
    case LOOM_ATTR_STRING:
      return loom_attr_as_string_id(lhs) == loom_attr_as_string_id(rhs);
    default:
      return lhs.raw == rhs.raw;
  }
}

static iree_status_t iree_benchmark_loom_validate_comparison_sample_parameters(
    const loom_module_t* module,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* baseline,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* candidate) {
  const loom_testbench_case_plan_t* baseline_case =
      baseline->selection->case_plan;
  const loom_testbench_case_plan_t* candidate_case =
      candidate->selection->case_plan;
  const iree_string_view_t baseline_name =
      baseline->selection->benchmark_plan->name;
  const iree_string_view_t candidate_name =
      candidate->selection->benchmark_plan->name;
  const iree_host_size_t baseline_case_sample =
      iree_benchmark_loom_case_sample_from_benchmark_sample(
          baseline->selection->benchmark_plan, baseline_case,
          baseline->begin_sample);
  const iree_host_size_t candidate_case_sample =
      iree_benchmark_loom_case_sample_from_benchmark_sample(
          candidate->selection->benchmark_plan, candidate_case,
          candidate->begin_sample);
  if (baseline_case->parameter_count != candidate_case->parameter_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "comparison candidate `%.*s` has %" PRIhsz
        " case parameters, but baseline `%.*s` has %" PRIhsz,
        (int)candidate_name.size, candidate_name.data,
        candidate_case->parameter_count, (int)baseline_name.size,
        baseline_name.data, baseline_case->parameter_count);
  }

  for (iree_host_size_t parameter_index = 0;
       parameter_index < baseline_case->parameter_count; ++parameter_index) {
    const loom_testbench_parameter_plan_t* baseline_parameter =
        &baseline_case->parameters[parameter_index];
    const loom_testbench_parameter_plan_t* candidate_parameter =
        &candidate_case->parameters[parameter_index];
    iree_string_view_t baseline_parameter_name = baseline_parameter->name;
    if (iree_string_view_is_empty(baseline_parameter_name)) {
      baseline_parameter_name =
          iree_benchmark_loom_value_name(module, baseline_parameter->value_id);
    }
    iree_string_view_t candidate_parameter_name = candidate_parameter->name;
    if (iree_string_view_is_empty(candidate_parameter_name)) {
      candidate_parameter_name =
          iree_benchmark_loom_value_name(module, candidate_parameter->value_id);
    }
    if (!iree_string_view_equal(baseline_parameter_name,
                                candidate_parameter_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "comparison candidate `%.*s` case parameter %" PRIhsz
          " is `%.*s`, but baseline `%.*s` uses `%.*s`",
          (int)candidate_name.size, candidate_name.data, parameter_index,
          (int)candidate_parameter_name.size, candidate_parameter_name.data,
          (int)baseline_name.size, baseline_name.data,
          (int)baseline_parameter_name.size, baseline_parameter_name.data);
    }
    if (baseline_parameter->kind != candidate_parameter->kind ||
        !loom_type_equal(baseline_parameter->type, candidate_parameter->type)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "comparison candidate `%.*s` case parameter `%.*s` does not match "
          "baseline `%.*s` parameter kind/type",
          (int)candidate_name.size, candidate_name.data,
          (int)baseline_parameter_name.size, baseline_parameter_name.data,
          (int)baseline_name.size, baseline_name.data);
    }

    const iree_host_size_t baseline_parameter_sample =
        loom_testbench_case_sample_parameter_ordinal(
            baseline_case, baseline_case_sample, parameter_index);
    const iree_host_size_t candidate_parameter_sample =
        loom_testbench_case_sample_parameter_ordinal(
            candidate_case, candidate_case_sample, parameter_index);
    loom_attribute_t baseline_value = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
        baseline_parameter, baseline_parameter_sample, &baseline_value));
    loom_attribute_t candidate_value = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
        candidate_parameter, candidate_parameter_sample, &candidate_value));
    if (!iree_benchmark_loom_sample_attr_equal(baseline_value,
                                               candidate_value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "comparison candidate `%.*s` case parameter `%.*s` sample differs "
          "from baseline `%.*s`; use --sample= to select matching samples",
          (int)candidate_name.size, candidate_name.data,
          (int)baseline_parameter_name.size, baseline_parameter_name.data,
          (int)baseline_name.size, baseline_name.data);
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_validate_comparison_samples(
    const loom_module_t* module,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* candidates,
    iree_host_size_t candidate_count) {
  if (candidate_count < 2) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 1; i < candidate_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_benchmark_loom_validate_comparison_sample_parameters(
            module, &candidates[0], &candidates[i]));
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_prepare_dispatch_comparison_candidate(
    const iree_benchmark_loom_comparison_execution_options_t* options,
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      candidate->selection;
  const loom_testbench_invocation_plan_t* actual_invocation = NULL;
  loom_testbench_case_execution_options_t candidate_execution_options =
      *options->case_execution_options;
  iree_benchmark_loom_reference_oracles_t reference_oracles = {0};

  iree_status_t status = loom_run_hal_testbench_select_actual_invocation(
      selection->case_plan, &actual_invocation);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_context_ensure_runtime(
        &options->hal_context->execution);
  }
  loom_testbench_requirement_result_t requirement_result = {0};
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_evaluate_case_requirements(
        options->hal_context->configuration, &options->hal_context->execution,
        options->module_plan, selection->case_plan, &requirement_result);
  }
  if (iree_status_is_ok(status) && requirement_result.skipped) {
    iree_benchmark_loom_benchmark_result_t benchmark_result = {
        .status = IREE_SV("skipped"),
        .sample_compilation = candidate->sample_compilation,
    };
    return iree_benchmark_loom_emit_dispatch_benchmark_result(
        options->run, &selection->identity, options->module_plan,
        selection->benchmark_plan, selection->case_plan, &selection->policy,
        &benchmark_result,
        /*correctness_sample_count=*/0,
        /*correctness_failed_sample_count=*/0, options->event_sink);
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t sample_constant_case_ordinal =
        candidate->has_sample_constant_ordinal
            ? iree_benchmark_loom_case_sample_from_benchmark_sample(
                  selection->benchmark_plan, selection->case_plan,
                  candidate->sample_constant_ordinal)
            : 0;
    status = iree_benchmark_loom_hal_actual_provider_initialize(
        options->hal_context, options->session, options->filename,
        options->source, options->benchmark_options->pipeline,
        options->module_plan->module, actual_invocation,
        candidate->sample_compilation, selection->case_plan,
        sample_constant_case_ordinal, candidate->has_sample_constant_ordinal,
        options->compile_report_options, &candidate->provider);
  }
  if (iree_status_is_ok(status)) {
    candidate->provider_initialized = true;
    candidate_execution_options.materializer.device =
        options->hal_context->execution.runtime.device;
    candidate_execution_options.materializer.device_allocator =
        iree_hal_device_allocator(
            options->hal_context->execution.runtime.device);
    candidate_execution_options.materializer.buffer_params =
        loom_run_hal_testbench_host_visible_buffer_params();
    candidate_execution_options.invocation.invoke_actual =
        (loom_testbench_invocation_callback_t){
            .fn = loom_run_hal_testbench_actual_invoke,
            .user_data = &candidate->provider.execution,
        };
    iree_benchmark_loom_configure_reference_oracles(
        &options->hal_context->execution,
        candidate_execution_options.materializer.host_allocator,
        &reference_oracles, &candidate_execution_options);
    candidate->benchmark_materializer =
        candidate_execution_options.materializer;
    candidate->benchmark_materializer.buffer_params =
        (iree_hal_buffer_params_t){0};
    status =
        iree_benchmark_loom_hal_actual_provider_compile(&candidate->provider);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_compiled_artifacts(
        options->run, &selection->identity, &candidate->provider,
        options->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_compile_report_artifact(
        options->run, &selection->identity, selection->benchmark_plan,
        selection->case_plan, &candidate->provider, options->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_device(
        options->event_sink, options->run, options->hal_context);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_compile(
        options->event_sink, options->run, &selection->identity,
        selection->benchmark_plan, selection->case_plan, &candidate->provider);
  }
  if (iree_status_is_ok(status) &&
      candidate->provider.execution.compile_rejected) {
    iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
    iree_benchmark_loom_benchmark_result_set_compile_rejection(
        &candidate->provider, &benchmark_result);
    ++*inout_failed_benchmark_count;
    return iree_benchmark_loom_emit_dispatch_benchmark_result(
        options->run, &selection->identity, options->module_plan,
        selection->benchmark_plan, selection->case_plan, &selection->policy,
        &benchmark_result,
        /*correctness_sample_count=*/0,
        /*correctness_failed_sample_count=*/0, options->event_sink);
  }

  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_run_case_correctness_range(
        options->run, &selection->identity, options->module_plan,
        selection->benchmark_plan, selection->benchmark_plan->case_index,
        &candidate_execution_options, candidate->sample_compilation,
        candidate->begin_sample, candidate->end_sample,
        options->execution_arena, options->event_sink,
        &candidate->correctness_sample_count,
        &candidate->correctness_failed_sample_count);
  }
  if (iree_status_is_ok(status)) {
    *inout_correctness_sample_count += candidate->correctness_sample_count;
    *inout_correctness_failed_sample_count +=
        candidate->correctness_failed_sample_count;
  }
  if (iree_status_is_ok(status) &&
      candidate->correctness_failed_sample_count != 0) {
    iree_benchmark_loom_benchmark_result_t benchmark_result = {
        .executed = false,
        .passed = false,
        .compile_report_artifact_path =
            candidate->provider.compile_report_artifact_path,
        .target_artifact_path = candidate->provider.target_artifact_path,
        .target_listing_path = candidate->provider.target_listing_path,
        .hal_executable_path = candidate->provider.hal_executable_path,
        .sample_compilation = candidate->sample_compilation,
        .has_sample_ordinal = true,
        .sample_ordinal = candidate->begin_sample,
        .samples_per_iteration = candidate->correctness_sample_count,
        .failed_sample_count = candidate->correctness_failed_sample_count,
    };
    ++*inout_failed_benchmark_count;
    return iree_benchmark_loom_emit_dispatch_benchmark_result(
        options->run, &selection->identity, options->module_plan,
        selection->benchmark_plan, selection->case_plan, &selection->policy,
        &benchmark_result, candidate->correctness_sample_count,
        candidate->correctness_failed_sample_count, options->event_sink);
  }
  if (iree_status_is_ok(status)) {
    candidate->runnable = true;
  }
  return status;
}

static iree_host_size_t iree_benchmark_loom_comparison_sample_capacity(
    iree_benchmark_loom_interleave_mode_t interleave_mode,
    iree_host_size_t candidate_count, iree_host_size_t candidate_index,
    iree_host_size_t repetitions) {
  if (interleave_mode == IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA) {
    return candidate_index == 0 ? (candidate_count - 1) * (repetitions + 1)
                                : repetitions;
  }
  return repetitions;
}

static iree_status_t
iree_benchmark_loom_initialize_dispatch_comparison_candidates(
    const iree_benchmark_loom_comparison_execution_options_t* options,
    iree_benchmark_loom_dispatch_comparison_candidate_t** out_candidates) {
  *out_candidates = NULL;
  if (options->sample_compilation_mode ==
      IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_BOTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "interleaved comparisons require one sample-compilation mode; use "
        "--sample_compilation=once or --sample_compilation=per_sample");
  }
  if (options->interleave_mode == IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA &&
      options->selection_count != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ABABA comparison requires exactly two selected "
                            "benchmarks; use --interleave=round_robin for "
                            "ABCD-style comparisons");
  }
  if (options->repetitions == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "interleaved comparison repetitions must be "
                            "positive");
  }
  if (options->selection_count > 26) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "interleaved comparison supports at most 26 "
                            "selected benchmarks; got %" PRIhsz,
                            options->selection_count);
  }

  iree_benchmark_loom_dispatch_comparison_candidate_t* candidates = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      options->host_allocator, options->selection_count, sizeof(*candidates),
      (void**)&candidates));
  memset(candidates, 0, options->selection_count * sizeof(*candidates));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < options->selection_count; ++i) {
    const iree_benchmark_loom_selected_benchmark_t* selection =
        &options->selections[i];
    if (selection->policy.measure_kind !=
        IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "interleaved comparison benchmark `%.*s` must use "
                           "measure = \"dispatch_complete\"",
                           (int)selection->benchmark_plan->name.size,
                           selection->benchmark_plan->name.data);
      break;
    }
    candidates[i].selection = selection;
    candidates[i].sample_compilation =
        iree_benchmark_loom_sample_compilation_mode_name(
            options->sample_compilation_mode);
    if (options->sample_compilation_mode ==
        IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_PER_SAMPLE) {
      iree_host_size_t begin_sample = 0;
      iree_host_size_t end_sample = 0;
      status = iree_benchmark_loom_validate_sample_flag(
          options->benchmark_options, selection->benchmark_plan->sample_count,
          &begin_sample, &end_sample);
      if (iree_status_is_ok(status) && end_sample != begin_sample + 1) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "per-sample interleaved comparison benchmark `%.*s` has %" PRIhsz
            " selected samples; use --sample= to select one sample",
            (int)selection->benchmark_plan->name.size,
            selection->benchmark_plan->name.data,
            selection->benchmark_plan->sample_count);
      }
      candidates[i].has_sample_constant_ordinal = true;
      candidates[i].sample_constant_ordinal = begin_sample;
    }
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_dispatch_comparison_sample_window(
          selection->benchmark_plan, options->benchmark_options,
          candidates[i].has_sample_constant_ordinal,
          candidates[i].sample_constant_ordinal, &candidates[i].begin_sample,
          &candidates[i].end_sample);
    }
    const iree_host_size_t sample_capacity =
        iree_benchmark_loom_comparison_sample_capacity(options->interleave_mode,
                                                       options->selection_count,
                                                       i, options->repetitions);
    candidates[i].sample_capacity = sample_capacity;
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc_array(options->host_allocator, sample_capacity,
                                      sizeof(*candidates[i].p50_samples),
                                      (void**)&candidates[i].p50_samples);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc_array(options->host_allocator, sample_capacity,
                                      sizeof(*candidates[i].p90_samples),
                                      (void**)&candidates[i].p90_samples);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_candidates = candidates;
  } else {
    for (iree_host_size_t i = 0; i < options->selection_count; ++i) {
      if (candidates[i].provider_initialized) {
        iree_benchmark_loom_hal_actual_provider_deinitialize(
            &candidates[i].provider);
      }
      iree_allocator_free(options->host_allocator, candidates[i].p50_samples);
      iree_allocator_free(options->host_allocator, candidates[i].p90_samples);
    }
    iree_allocator_free(options->host_allocator, candidates);
  }
  return status;
}

static void iree_benchmark_loom_dispatch_comparison_candidate_deinitialize(
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    iree_allocator_t allocator) {
  if (candidate->provider_initialized) {
    iree_benchmark_loom_hal_actual_provider_deinitialize(&candidate->provider);
  }
  iree_allocator_free(allocator, candidate->p50_samples);
  iree_allocator_free(allocator, candidate->p90_samples);
  *candidate = (iree_benchmark_loom_dispatch_comparison_candidate_t){0};
}

static void iree_benchmark_loom_dispatch_comparison_candidates_deinitialize(
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidates,
    iree_host_size_t candidate_count, iree_allocator_t allocator) {
  if (candidates == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    iree_benchmark_loom_dispatch_comparison_candidate_deinitialize(
        &candidates[i], allocator);
  }
  iree_allocator_free(allocator, candidates);
}

static iree_status_t iree_benchmark_loom_run_comparison_window(
    const iree_benchmark_loom_comparison_execution_options_t* options,
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    const iree_benchmark_loom_candidate_identity_t* baseline,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_host_size_t order_index, iree_host_size_t repetition_index,
    char schedule_token, iree_host_size_t* inout_failed_benchmark_count) {
  if (!candidate->runnable) {
    return iree_ok_status();
  }
  iree_benchmark_loom_benchmark_policy_t measurement_policy =
      candidate->selection->policy;
  const bool profile_suppressed =
      iree_any_bit_set(measurement_policy.hal_options.flags,
                       LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH);
  measurement_policy.hal_options.flags &=
      ~LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH;

  iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
  iree_status_t status = iree_benchmark_loom_run_hal_benchmark_sample(
      options->run, &candidate->selection->identity, options->module_plan,
      candidate->selection->benchmark_plan, candidate->selection->case_plan,
      &measurement_policy, options->benchmark_options, &candidate->provider,
      &candidate->benchmark_materializer, candidate->begin_sample,
      options->host_allocator, &benchmark_result);
  if (iree_status_is_ok(status) &&
      (!benchmark_result.executed || !benchmark_result.passed)) {
    ++*inout_failed_benchmark_count;
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_comparison_candidate_record_timing(
        candidate, &benchmark_result);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_benchmark_repetition(
        options->event_sink, options->run, candidate, baseline,
        comparison_group, method, order_index, repetition_index, schedule_token,
        profile_suppressed, &benchmark_result);
  }
  return status;
}

iree_status_t iree_benchmark_loom_run_dispatch_comparison(
    const iree_benchmark_loom_comparison_execution_options_t* options,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  iree_benchmark_loom_dispatch_comparison_candidate_t* candidates = NULL;
  iree_status_t status =
      iree_benchmark_loom_initialize_dispatch_comparison_candidates(
          options, &candidates);
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_validate_comparison_samples(
        options->module_plan->module, candidates, options->selection_count);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < options->selection_count; ++i) {
    status = iree_benchmark_loom_prepare_dispatch_comparison_candidate(
        options, &candidates[i], inout_correctness_sample_count,
        inout_correctness_failed_sample_count, inout_failed_benchmark_count);
  }

  const iree_benchmark_loom_candidate_identity_t* baseline =
      &options->selections[0].identity;
  const iree_string_view_t comparison_group =
      options->selections[0].benchmark_plan->name;
  const iree_string_view_t method =
      iree_benchmark_loom_interleave_mode_name(options->interleave_mode);
  iree_host_size_t order_index = 0;
  if (iree_status_is_ok(status) &&
      options->interleave_mode == IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA) {
    for (iree_host_size_t candidate_index = 1;
         iree_status_is_ok(status) &&
         candidate_index < options->selection_count;
         ++candidate_index) {
      status = iree_benchmark_loom_run_comparison_window(
          options, &candidates[0], baseline, comparison_group, method,
          order_index++, /*repetition_index=*/0, 'A',
          inout_failed_benchmark_count);
      for (iree_host_size_t repetition_index = 0;
           iree_status_is_ok(status) && repetition_index < options->repetitions;
           ++repetition_index) {
        status = iree_benchmark_loom_run_comparison_window(
            options, &candidates[candidate_index], baseline, comparison_group,
            method, order_index++, repetition_index,
            (char)('A' + candidate_index), inout_failed_benchmark_count);
        if (iree_status_is_ok(status)) {
          status = iree_benchmark_loom_run_comparison_window(
              options, &candidates[0], baseline, comparison_group, method,
              order_index++, repetition_index + 1, 'A',
              inout_failed_benchmark_count);
        }
      }
    }
  } else if (iree_status_is_ok(status) &&
             options->interleave_mode ==
                 IREE_BENCHMARK_LOOM_INTERLEAVE_ROUND_ROBIN) {
    for (iree_host_size_t repetition_index = 0;
         iree_status_is_ok(status) && repetition_index < options->repetitions;
         ++repetition_index) {
      for (iree_host_size_t candidate_index = 0;
           iree_status_is_ok(status) &&
           candidate_index < options->selection_count;
           ++candidate_index) {
        status = iree_benchmark_loom_run_comparison_window(
            options, &candidates[candidate_index], baseline, comparison_group,
            method, order_index++, repetition_index,
            (char)('A' + candidate_index), inout_failed_benchmark_count);
      }
    }
  }

  for (iree_host_size_t candidate_index = 1;
       iree_status_is_ok(status) && candidate_index < options->selection_count;
       ++candidate_index) {
    status = iree_benchmark_loom_event_sink_emit_comparison(
        options->event_sink, options->run, &candidates[0],
        &candidates[candidate_index], comparison_group, method);
  }

  iree_benchmark_loom_dispatch_comparison_candidates_deinitialize(
      candidates, options->selection_count, options->host_allocator);
  return status;
}
