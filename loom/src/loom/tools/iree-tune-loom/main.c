// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-tune-loom: correctness-gated benchmark runner for check.benchmark.

#include "loom/tools/iree-tune-loom/main.h"

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/special_values.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/hal/artifact.h"
#include "loom/tooling/execution/hal/benchmark.h"
#include "loom/tooling/execution/hal/candidate.h"
#include "loom/tooling/execution/hal/invocation.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/execution/one_shot.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tools/iree-tune-loom/diagnostics.h"
#include "loom/tools/iree-tune-loom/help.h"
#include "loom/tools/iree-tune-loom/manifest.h"
#include "loom/tools/iree-tune-loom/model.h"
#include "loom/tools/iree-tune-loom/module_query.h"
#include "loom/tools/iree-tune-loom/options.h"
#include "loom/tools/iree-tune-loom/output.h"
#include "loom/tools/iree-tune-loom/profile_report.h"
#include "loom/tools/iree-tune-loom/report.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

enum {
  IREE_TUNE_LOOM_DEFAULT_INPUT_RING_MIN_BYTES = 32 * 1024 * 1024,
};

IREE_FLAG(string, case, "",
          "Optional check.case symbol to benchmark, such as '@smoke'. Empty "
          "keeps all cases referenced by selected benchmarks.");
IREE_FLAG(string, benchmark, "",
          "Optional check.benchmark symbol to execute, such as '@smoke_time'. "
          "Empty executes all benchmarks in source order.");
IREE_FLAG(int32_t, sample, -1,
          "Optional concrete sample ordinal to execute for selected benchmark "
          "cases. Negative executes all planned samples.");
IREE_FLAG(int32_t, max_samples_per_case,
          LOOM_TESTBENCH_DEFAULT_MAX_SAMPLES_PER_CASE,
          "Maximum number of samples planned per check.case.");
IREE_FLAG(string, pipeline, "default",
          "Pass pipeline used before HAL candidate emission. Use 'default', "
          "'none', '@symbol', or a comma-separated pass list.");
IREE_FLAG(string, output, "",
          "Output path for the JSONL result stream. Empty or '-' writes to "
          "stdout.");
IREE_FLAG(string, file_output_dir, "",
          "Directory receiving check.file.write.* outputs. Empty uses "
          "$TMPDIR/iree-loom-tune/<source>_<hash>/ for this run.");
IREE_FLAG(string, artifact_bundle_dir, "",
          "Directory receiving a self-contained run bundle. When set and "
          "--output is empty, results are written to results.jsonl inside the "
          "bundle; check.file.write outputs and profile artifacts default to "
          "bundle subdirectories unless their explicit flags are set.");
IREE_FLAG(string, artifact_bundle_policy, "minimal",
          "Artifact bundle policy when --artifact_bundle_dir is set. Use "
          "'minimal', 'debug', or 'full'.");
IREE_FLAG(bool, dry_run, false,
          "Emits plan rows for selected benchmarks and stops before "
          "correctness, compilation, and measurement.");
IREE_FLAG(bool, agents_md, false,
          "Prints a compact Markdown snippet suitable for AGENTS.md and "
          "exits.");
IREE_FLAG(string, compile_report, "summary",
          "Structured compile report embedded in benchmark rows. Use "
          "'summary', 'details', or empty/'none'.");
IREE_FLAG(int32_t, compile_report_row_limit,
          LOOM_RUN_COMPILE_REPORT_DEFAULT_ROW_LIMIT,
          "Maximum rows per report row category to capture for "
          "--compile_report=details.");
IREE_FLAG(
    string, profile_data, "",
    "HAL profiling data families for the final profiled batch as a "
    "comma-separated list. Empty uses dispatch-events,executable-metadata. "
    "Accepted families match --device_profiling_mode: queue-events, "
    "host-execution, device-queue-events, dispatch-events, memory-events, "
    "device-metrics, command-region-events, counters, counter-ranges, "
    "executable-metadata, executable-traces.");
IREE_FLAG_LIST(
    string, profile_counter,
    "Implementation-specific hardware counter name to request during the final "
    "profiled batch. May be repeated and requires --profile_data to include "
    "counters or counter-ranges.");
IREE_FLAG(
    string, profile_artifacts_dir, "",
    "Directory receiving raw IREE HAL profile bundles from final profiled "
    "batches. Setting this implies --profile_final_batch=true unless that flag "
    "was explicitly set false.");
IREE_FLAG(string, shape_specialization, "dynamic",
          "Shape specialization mode for dispatch_complete benchmarks. Use "
          "'dynamic' to compile once and pass parameter values at dispatch "
          "time, 'per_sample' to compile each selected shape with concrete "
          "parameter facts, or 'both' to emit both result sets.");
IREE_FLAG(
    int64_t, input_ring_min_bytes, IREE_TUNE_LOOM_DEFAULT_INPUT_RING_MIN_BYTES,
    "Minimum total byte size of the device-buffer binding ring used by "
    "dispatch_complete benchmarks. The auto ring count is max(batch_size, "
    "ceil(input_ring_min_bytes / bytes_per_binding_set)); use 0 to record one "
    "hot-reuse binding set.");
IREE_FLAG(
    int32_t, input_ring_count, 0,
    "Exact number of physical device-buffer binding sets to rotate through "
    "dispatch_complete command buffers. Zero uses --input_ring_min_bytes. Use "
    "1 to force hot-reuse measurements.");
IREE_FLAG(string, compare, "",
          "Comma-separated check.benchmark symbols to compare in one "
          "interleaved dispatch_complete run, such as '@base,@variant'. "
          "Cannot be combined with --benchmark or --case.");
IREE_FLAG(string, interleave, "ABABA",
          "Interleaving schedule used with --compare. Use 'ABABA' for "
          "baseline-anchored pairwise windows or 'round_robin' for ABCD-style "
          "candidate rotation.");
IREE_FLAG(int32_t, repetitions, 2,
          "Interleaved comparison repetitions. With --interleave=ABABA this "
          "runs A then this many BA pairs; the default produces ABABA.");

static iree_tune_loom_i32_flag_t FLAG_iterations = {.value = 10};
IREE_FLAG_CALLBACK(iree_tune_loom_parse_i32_flag, iree_tune_loom_print_i32_flag,
                   &FLAG_iterations, iterations,
                   "Measured iterations. Overrides any benchmark "
                   "'iterations' attribute when provided.");
static iree_tune_loom_i32_flag_t FLAG_warmup_iterations = {.value = 1};
IREE_FLAG_CALLBACK(iree_tune_loom_parse_i32_flag, iree_tune_loom_print_i32_flag,
                   &FLAG_warmup_iterations, warmup_iterations,
                   "Warmup iterations. Overrides any benchmark "
                   "'warmup_iterations' attribute when provided.");
static iree_tune_loom_i32_flag_t FLAG_batch_size = {.value = 1};
IREE_FLAG_CALLBACK(iree_tune_loom_parse_i32_flag, iree_tune_loom_print_i32_flag,
                   &FLAG_batch_size, batch_size,
                   "Number of repeated dispatches recorded into each measured "
                   "HAL command buffer batch. Overrides any benchmark "
                   "'batch_size' attribute when provided.");
static iree_tune_loom_i32_flag_t FLAG_min_time_ms = {.value = 100};
IREE_FLAG_CALLBACK(iree_tune_loom_parse_i32_flag, iree_tune_loom_print_i32_flag,
                   &FLAG_min_time_ms, min_time_ms,
                   "Minimum measured duration for dispatch_complete "
                   "benchmarks. Overrides any benchmark 'min_time_ms' "
                   "attribute when provided.");
static iree_tune_loom_i32_flag_t FLAG_warmup_time_ms = {.value = 0};
IREE_FLAG_CALLBACK(iree_tune_loom_parse_i32_flag, iree_tune_loom_print_i32_flag,
                   &FLAG_warmup_time_ms, warmup_time_ms,
                   "Minimum warmup duration for dispatch_complete benchmarks. "
                   "Overrides any benchmark 'warmup_time_ms' attribute when "
                   "provided.");
static iree_tune_loom_i32_flag_t FLAG_max_batches = {.value = 1000};
IREE_FLAG_CALLBACK(iree_tune_loom_parse_i32_flag, iree_tune_loom_print_i32_flag,
                   &FLAG_max_batches, max_batches,
                   "Maximum measured command-buffer batches for "
                   "dispatch_complete benchmarks. Overrides any benchmark "
                   "'max_batches' attribute when provided.");
static iree_tune_loom_i32_flag_t FLAG_stable_p90_to_p50_ppm = {
    .value = 100000,
};
IREE_FLAG_CALLBACK(iree_tune_loom_parse_i32_flag, iree_tune_loom_print_i32_flag,
                   &FLAG_stable_p90_to_p50_ppm, stable_p90_to_p50_ppm,
                   "p90-to-p50 spread threshold in parts per million. Zero "
                   "stops after the minimum count and duration are reached. "
                   "Overrides any benchmark 'stable_p90_to_p50_ppm' attribute "
                   "when provided.");
static iree_tune_loom_bool_flag_t FLAG_profile_final_batch = {.value = false};
IREE_FLAG_CALLBACK(iree_tune_loom_parse_bool_flag,
                   iree_tune_loom_print_bool_flag, &FLAG_profile_final_batch,
                   profile_final_batch,
                   "Runs one final profiled HAL command-buffer batch after "
                   "measured dispatch_complete timing. Overrides any benchmark "
                   "'profile_final_batch' attribute when provided.");

static iree_status_t iree_tune_loom_register_context(void* user_data,
                                                     loom_context_t* context) {
  const iree_tune_loom_configuration_t* configuration =
      (const iree_tune_loom_configuration_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_op_registry_register_all_dialects(context));
  if (configuration->register_context.fn == NULL) {
    return iree_ok_status();
  }
  return configuration->register_context.fn(
      configuration->register_context.user_data, context);
}

// Formats a borrowed status as JSON. The caller retains status ownership.
static bool iree_tune_loom_case_matches_selection(
    const loom_testbench_case_plan_t* case_plan,
    iree_string_view_t selected_case_name) {
  return iree_string_view_is_empty(selected_case_name) ||
         iree_string_view_equal(case_plan->name, selected_case_name);
}

static bool iree_tune_loom_benchmark_matches_selection(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_string_view_t selected_benchmark_name) {
  return iree_string_view_is_empty(selected_benchmark_name) ||
         iree_string_view_equal(benchmark_plan->name, selected_benchmark_name);
}

static iree_status_t iree_tune_loom_duration_ms_to_ns(
    int64_t duration_ms, iree_duration_t* out_duration_ns) {
  if (duration_ms < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duration must be non-negative; got %" PRIi64,
                            duration_ms);
  }
  if (duration_ms > INT64_MAX / 1000000) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "duration %" PRIi64 "ms overflows nanoseconds",
                            duration_ms);
  }
  *out_duration_ns = (iree_duration_t)(duration_ms * 1000000);
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_policy_from_benchmark(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_tune_loom_benchmark_policy_t* out_policy) {
  memset(out_policy, 0, sizeof(*out_policy));
  loom_run_hal_benchmark_options_initialize(&out_policy->hal_options);

  int64_t iterations = 0;
  int64_t warmup_iterations = 0;
  IREE_RETURN_IF_ERROR(iree_tune_loom_read_i64_policy_attr(
      module_plan->module, benchmark_plan->attrs, IREE_SV("iterations"),
      &FLAG_iterations, &iterations));
  IREE_RETURN_IF_ERROR(iree_tune_loom_read_i64_policy_attr(
      module_plan->module, benchmark_plan->attrs, IREE_SV("warmup_iterations"),
      &FLAG_warmup_iterations, &warmup_iterations));
  if (iterations <= 0 || (uint64_t)iterations > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark `%.*s` iterations must be positive; "
                            "got %" PRIi64,
                            (int)benchmark_plan->name.size,
                            benchmark_plan->name.data, iterations);
  }
  if (warmup_iterations < 0 ||
      (uint64_t)warmup_iterations > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark `%.*s` warmup_iterations must be "
                            "non-negative; got %" PRIi64,
                            (int)benchmark_plan->name.size,
                            benchmark_plan->name.data, warmup_iterations);
  }

  iree_string_view_t measure = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_tune_loom_read_optional_string_attr(
      module_plan->module, benchmark_plan->attrs, IREE_SV("measure"),
      IREE_SV("case_end_to_end"), &measure));
  iree_tune_loom_measure_t measure_kind =
      IREE_TUNE_LOOM_MEASURE_CASE_END_TO_END;
  if (iree_string_view_equal(measure, IREE_SV("case_end_to_end")) ||
      iree_string_view_equal(measure, IREE_SV("end_to_end"))) {
    measure = IREE_SV("case_end_to_end");
    measure_kind = IREE_TUNE_LOOM_MEASURE_CASE_END_TO_END;
  } else if (iree_string_view_equal(measure, IREE_SV("dispatch_complete"))) {
    measure_kind = IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE;
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "benchmark `%.*s` requested unsupported V0 "
                            "measure '%.*s'",
                            (int)benchmark_plan->name.size,
                            benchmark_plan->name.data, (int)measure.size,
                            measure.data);
  }
  const iree_string_view_t profile_data_flag =
      iree_string_view_trim(iree_make_cstring_view(FLAG_profile_data));
  const bool profile_data_requested =
      !iree_string_view_is_empty(profile_data_flag) &&
      !iree_string_view_equal(profile_data_flag, IREE_SV("none"));
  const iree_flag_string_list_t profile_counters = FLAG_profile_counter_list();
  if (measure_kind != IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE &&
      (strlen(FLAG_profile_artifacts_dir) != 0 || profile_data_requested ||
       profile_counters.count != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "final-batch profile flags require benchmark `%.*s` "
        "to use measure = \"dispatch_complete\"",
        (int)benchmark_plan->name.size, benchmark_plan->name.data);
  }

  *out_policy = (iree_tune_loom_benchmark_policy_t){
      .measure_kind = measure_kind,
      .measure = measure,
      .warmup_iterations = (iree_host_size_t)warmup_iterations,
      .iterations = (iree_host_size_t)iterations,
  };
  loom_run_hal_benchmark_options_initialize(&out_policy->hal_options);
  if (measure_kind == IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE) {
    int64_t batch_size = 0;
    int64_t max_batches = 0;
    int64_t min_time_ms = 0;
    int64_t warmup_time_ms = 0;
    int64_t stable_p90_to_p50_ppm = 0;
    bool profile_final_batch = false;
    const bool profile_artifacts_requested =
        strlen(FLAG_profile_artifacts_dir) != 0;
    IREE_RETURN_IF_ERROR(iree_tune_loom_read_i64_policy_attr(
        module_plan->module, benchmark_plan->attrs, IREE_SV("batch_size"),
        &FLAG_batch_size, &batch_size));
    IREE_RETURN_IF_ERROR(iree_tune_loom_read_i64_policy_attr(
        module_plan->module, benchmark_plan->attrs, IREE_SV("max_batches"),
        &FLAG_max_batches, &max_batches));
    IREE_RETURN_IF_ERROR(iree_tune_loom_read_i64_policy_attr(
        module_plan->module, benchmark_plan->attrs, IREE_SV("min_time_ms"),
        &FLAG_min_time_ms, &min_time_ms));
    IREE_RETURN_IF_ERROR(iree_tune_loom_read_i64_policy_attr(
        module_plan->module, benchmark_plan->attrs, IREE_SV("warmup_time_ms"),
        &FLAG_warmup_time_ms, &warmup_time_ms));
    IREE_RETURN_IF_ERROR(iree_tune_loom_read_i64_policy_attr(
        module_plan->module, benchmark_plan->attrs,
        IREE_SV("stable_p90_to_p50_ppm"), &FLAG_stable_p90_to_p50_ppm,
        &stable_p90_to_p50_ppm));
    IREE_RETURN_IF_ERROR(iree_tune_loom_read_bool_policy_attr(
        module_plan->module, benchmark_plan->attrs,
        IREE_SV("profile_final_batch"), &FLAG_profile_final_batch,
        &profile_final_batch));
    if ((profile_artifacts_requested || profile_data_requested ||
         profile_counters.count != 0) &&
        !profile_final_batch) {
      if (FLAG_profile_final_batch.specified &&
          !FLAG_profile_final_batch.value) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--profile_final_batch=false conflicts with "
                                "requested final-batch profile data");
      }
      profile_final_batch = true;
    }
    if (batch_size <= 0 || (uint64_t)batch_size > IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "benchmark `%.*s` batch_size must be positive; "
                              "got %" PRIi64,
                              (int)benchmark_plan->name.size,
                              benchmark_plan->name.data, batch_size);
    }
    if (max_batches <= 0 || (uint64_t)max_batches > IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "benchmark `%.*s` max_batches must be positive; "
                              "got %" PRIi64,
                              (int)benchmark_plan->name.size,
                              benchmark_plan->name.data, max_batches);
    }
    if (stable_p90_to_p50_ppm < 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "benchmark `%.*s` stable_p90_to_p50_ppm must be non-negative; "
          "got %" PRIi64,
          (int)benchmark_plan->name.size, benchmark_plan->name.data,
          stable_p90_to_p50_ppm);
    }
    iree_duration_t min_duration_ns = 0;
    iree_duration_t warmup_min_duration_ns = 0;
    IREE_RETURN_IF_ERROR(
        iree_tune_loom_duration_ms_to_ns(min_time_ms, &min_duration_ns));
    IREE_RETURN_IF_ERROR(iree_tune_loom_duration_ms_to_ns(
        warmup_time_ms, &warmup_min_duration_ns));
    out_policy->hal_options.timing.batch_size = (iree_host_size_t)batch_size;
    out_policy->hal_options.timing.warmup_batch_count =
        (iree_host_size_t)warmup_iterations;
    out_policy->hal_options.timing.warmup_min_duration_ns =
        warmup_min_duration_ns;
    out_policy->hal_options.timing.min_batch_count =
        (iree_host_size_t)iterations;
    out_policy->hal_options.timing.min_duration_ns = min_duration_ns;
    out_policy->hal_options.timing.max_batch_count =
        (iree_host_size_t)max_batches;
    out_policy->hal_options.timing.stable_p90_to_p50_delta_ppm =
        (uint64_t)stable_p90_to_p50_ppm;
    out_policy->hal_options.dispatch_batch.dispatch_count =
        (iree_host_size_t)batch_size;
    IREE_RETURN_IF_ERROR(iree_tune_loom_parse_profile_data_families(
        profile_data_flag, &out_policy->hal_options.profile_data_families));
    const iree_hal_device_profiling_data_families_t counter_data =
        IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
        IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES;
    if (profile_counters.count != 0) {
      if (!iree_any_bit_set(out_policy->hal_options.profile_data_families,
                            counter_data)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "--profile_counter requires --profile_data to include counters or "
            "counter-ranges");
      }
      out_policy->profile_counter_set =
          (iree_hal_profile_counter_set_selection_t){
              .name = IREE_SV("iree-tune-loom"),
              .counter_name_count = profile_counters.count,
              .counter_names = profile_counters.values,
          };
      out_policy->hal_options.profile_counter_set_count = 1;
      out_policy->hal_options.profile_counter_sets =
          &out_policy->profile_counter_set;
    }
    if (profile_final_batch) {
      out_policy->hal_options.flags |=
          LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH;
    }
  }
  return iree_ok_status();
}

static iree_host_size_t iree_tune_loom_compare_token_count(
    iree_string_view_t compare_list) {
  iree_host_size_t count = 0;
  iree_string_view_t remaining = iree_string_view_trim(compare_list);
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      ++count;
    }
    remaining = iree_string_view_trim(remaining);
  }
  return count;
}

static iree_status_t iree_tune_loom_selected_benchmark_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_host_size_t candidate_index,
    iree_tune_loom_selected_benchmark_t* out_selection) {
  if (benchmark_plan->case_index >= module_plan->case_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "benchmark `%.*s` does not reference a planned check.case",
        (int)benchmark_plan->name.size, benchmark_plan->name.data);
  }
  memset(out_selection, 0, sizeof(*out_selection));
  snprintf(out_selection->candidate_id_storage,
           sizeof(out_selection->candidate_id_storage), "c%" PRIhsz,
           candidate_index);
  out_selection->identity = (iree_tune_loom_candidate_identity_t){
      .candidate_id =
          iree_make_cstring_view(out_selection->candidate_id_storage),
      .candidate_index = candidate_index,
  };
  out_selection->benchmark_plan = benchmark_plan;
  out_selection->case_plan = &module_plan->cases[benchmark_plan->case_index];
  return iree_tune_loom_policy_from_benchmark(module_plan, benchmark_plan,
                                              &out_selection->policy);
}

static iree_status_t iree_tune_loom_find_benchmark_by_name(
    const loom_testbench_module_plan_t* module_plan, iree_string_view_t name,
    const loom_testbench_benchmark_plan_t** out_benchmark) {
  *out_benchmark = NULL;
  for (iree_host_size_t i = 0; i < module_plan->benchmark_count; ++i) {
    const loom_testbench_benchmark_plan_t* benchmark =
        &module_plan->benchmarks[i];
    if (iree_string_view_equal(benchmark->name, name)) {
      *out_benchmark = benchmark;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "--compare benchmark '@%.*s' was not found",
                          (int)name.size, name.data);
}

static iree_status_t iree_tune_loom_select_compare_benchmarks(
    const loom_testbench_module_plan_t* module_plan, iree_string_view_t compare,
    iree_allocator_t allocator,
    iree_tune_loom_selected_benchmark_t** out_selections,
    iree_host_size_t* out_selection_count) {
  *out_selections = NULL;
  *out_selection_count = 0;
  compare = iree_string_view_trim(compare);
  const iree_host_size_t selection_count =
      iree_tune_loom_compare_token_count(compare);
  if (selection_count < 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compare requires at least two comma-separated check.benchmark "
        "symbols");
  }

  iree_tune_loom_selected_benchmark_t* selections = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, selection_count, sizeof(*selections), (void**)&selections));
  memset(selections, 0, selection_count * sizeof(*selections));

  iree_status_t status = iree_ok_status();
  iree_host_size_t selection_index = 0;
  iree_string_view_t remaining = compare;
  while (iree_status_is_ok(status) && !iree_string_view_is_empty(remaining)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &token, &remaining);
    token = iree_tune_loom_normalize_symbol_name(token);
    if (iree_string_view_is_empty(token)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--compare contains an empty benchmark symbol");
      break;
    }
    const loom_testbench_benchmark_plan_t* benchmark = NULL;
    status =
        iree_tune_loom_find_benchmark_by_name(module_plan, token, &benchmark);
    if (iree_status_is_ok(status)) {
      status = iree_tune_loom_selected_benchmark_initialize(
          module_plan, benchmark, selection_index,
          &selections[selection_index]);
    }
    ++selection_index;
    remaining = iree_string_view_trim(remaining);
  }
  if (iree_status_is_ok(status)) {
    *out_selections = selections;
    *out_selection_count = selection_count;
  } else {
    iree_allocator_free(allocator, selections);
  }
  return status;
}

static iree_status_t iree_tune_loom_compile_report_options_initialize(
    loom_run_compile_report_capture_options_t* out_options) {
  loom_run_compile_report_capture_options_initialize(out_options);
  IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_options_parse_request(
      iree_make_cstring_view(FLAG_compile_report), out_options));
  if (out_options->sink_format == LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_TEXT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-tune-loom emits structured JSONL reports; use "
        "--compile_report=summary, details, json-summary, or json-details");
  }
  if (FLAG_compile_report_row_limit < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compile_report_row_limit must be non-negative; got %d",
        (int)FLAG_compile_report_row_limit);
  }
  out_options->row_limit = (iree_host_size_t)FLAG_compile_report_row_limit;
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_validate_sample_flag(
    iree_host_size_t sample_count, iree_host_size_t* out_begin_sample,
    iree_host_size_t* out_end_sample) {
  if (FLAG_sample < 0) {
    *out_begin_sample = 0;
    *out_end_sample = sample_count;
    return iree_ok_status();
  }
  const iree_host_size_t sample_ordinal = (iree_host_size_t)FLAG_sample;
  if (sample_ordinal >= sample_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "--sample=%" PRIhsz
                            " exceeds selected case sample count %" PRIhsz,
                            sample_ordinal, sample_count);
  }
  *out_begin_sample = sample_ordinal;
  *out_end_sample = sample_ordinal + 1;
  return iree_ok_status();
}

static int iree_tune_loom_compare_duration(const void* lhs, const void* rhs) {
  const iree_duration_t lhs_duration = *(const iree_duration_t*)lhs;
  const iree_duration_t rhs_duration = *(const iree_duration_t*)rhs;
  return (lhs_duration > rhs_duration) - (lhs_duration < rhs_duration);
}

static iree_duration_t iree_tune_loom_nearest_rank_percentile(
    const iree_duration_t* sorted_durations, iree_host_size_t count,
    iree_host_size_t percentile) {
  IREE_ASSERT_ARGUMENT(sorted_durations);
  IREE_ASSERT(count > 0);
  const iree_host_size_t rank = (count * percentile + 99) / 100;
  iree_host_size_t index = rank == 0 ? 0 : rank - 1;
  if (index >= count) {
    index = count - 1;
  }
  return sorted_durations[index];
}

static void iree_tune_loom_compute_timing_stats(
    iree_duration_t* durations, iree_host_size_t count,
    iree_tune_loom_timing_stats_t* out_stats) {
  qsort(durations, count, sizeof(*durations), iree_tune_loom_compare_duration);
  int64_t total_ns = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    total_ns += durations[i];
  }
  *out_stats = (iree_tune_loom_timing_stats_t){
      .count = count,
      .total_ns = total_ns,
      .minimum_ns = durations[0],
      .maximum_ns = durations[count - 1],
      .mean_ns = (double)total_ns / (double)count,
      .p50_ns = iree_tune_loom_nearest_rank_percentile(durations, count, 50),
      .p90_ns = iree_tune_loom_nearest_rank_percentile(durations, count, 90),
  };
}

static void iree_tune_loom_hal_context_initialize(
    const iree_tune_loom_configuration_t* configuration,
    iree_allocator_t host_allocator,
    iree_tune_loom_hal_context_t* out_context) {
  *out_context = (iree_tune_loom_hal_context_t){
      .configuration = configuration,
  };
  loom_run_hal_testbench_context_initialize(
      configuration->hal_artifact_provider_registry, host_allocator,
      &out_context->execution);
}

static void iree_tune_loom_hal_context_deinitialize(
    iree_tune_loom_hal_context_t* context) {
  if (!context) {
    return;
  }
  loom_run_hal_testbench_context_deinitialize(&context->execution);
  *context = (iree_tune_loom_hal_context_t){0};
}

static iree_status_t iree_tune_loom_hal_actual_provider_initialize(
    iree_tune_loom_hal_context_t* context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    iree_string_view_t pipeline, const loom_module_t* test_module,
    const loom_testbench_invocation_plan_t* actual_invocation,
    iree_string_view_t specialization,
    const loom_testbench_case_plan_t* specialization_case_plan,
    iree_host_size_t specialization_sample_ordinal,
    bool has_specialization_sample_ordinal,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    iree_tune_loom_hal_actual_provider_t* out_provider) {
  *out_provider = (iree_tune_loom_hal_actual_provider_t){
      .context = context,
      .specialization = specialization,
  };
  iree_allocator_t host_allocator = context->execution.host_allocator;
  iree_tune_loom_diagnostic_capture_initialize(host_allocator,
                                               &out_provider->diagnostics);
  iree_status_t status = loom_run_compile_report_capture_initialize(
      compile_report_options, host_allocator,
      &out_provider->compile_report_capture);
  if (iree_status_is_ok(status)) {
    out_provider->compile_report_capture_initialized = true;
    loom_run_candidate_compile_options_t report_options = {0};
    loom_run_candidate_compile_options_initialize(&report_options);
    loom_run_compile_report_capture_configure_compile_options(
        &out_provider->compile_report_capture, &report_options);
    loom_run_candidate_artifact_flags_t artifact_flags = 0;
    if (context->artifact_bundle != NULL && context->artifact_bundle->enabled &&
        context->artifact_bundle->policy >=
            IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG) {
      artifact_flags |= LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_TARGET_LISTING;
    }
    loom_run_hal_testbench_actual_provider_options_t provider_options = {
        .context = &context->execution,
        .session = session,
        .target_environment = context->configuration->target_environment,
        .filename = filename,
        .source = source,
        .pipeline = pipeline,
        .test_module = test_module,
        .actual_invocation = actual_invocation,
        .specialization_case_plan = specialization_case_plan,
        .specialization_sample_ordinal = specialization_sample_ordinal,
        .has_specialization_sample_ordinal = has_specialization_sample_ordinal,
        .diagnostic_sink =
            (loom_diagnostic_sink_t){
                .fn = iree_tune_loom_diagnostic_capture_sink,
                .user_data = &out_provider->diagnostics,
            },
        .report = report_options.report,
        .report_row_storage = report_options.report_row_storage,
        .artifact_flags = artifact_flags,
    };
    loom_run_hal_testbench_actual_provider_initialize(&provider_options,
                                                      &out_provider->execution);
  } else {
    iree_tune_loom_diagnostic_capture_deinitialize(&out_provider->diagnostics);
    *out_provider = (iree_tune_loom_hal_actual_provider_t){0};
  }
  return status;
}

static void iree_tune_loom_hal_actual_provider_deinitialize(
    iree_tune_loom_hal_actual_provider_t* provider) {
  if (!provider) {
    return;
  }
  loom_run_hal_testbench_actual_provider_deinitialize(&provider->execution);
  if (provider->compile_report_capture_initialized) {
    loom_run_compile_report_capture_deinitialize(
        &provider->compile_report_capture);
  }
  iree_allocator_t host_allocator = provider->context->execution.host_allocator;
  iree_allocator_free(host_allocator, provider->hal_executable_path_storage);
  iree_allocator_free(host_allocator, provider->target_artifact_path_storage);
  iree_allocator_free(host_allocator, provider->target_listing_path_storage);
  iree_allocator_free(host_allocator,
                      provider->compile_report_artifact_path_storage);
  iree_tune_loom_diagnostic_capture_deinitialize(&provider->diagnostics);
  *provider = (iree_tune_loom_hal_actual_provider_t){0};
}

static void iree_tune_loom_dispatch_comparison_candidate_deinitialize(
    iree_tune_loom_dispatch_comparison_candidate_t* candidate,
    iree_allocator_t allocator) {
  if (!candidate) {
    return;
  }
  if (candidate->provider_initialized) {
    iree_tune_loom_hal_actual_provider_deinitialize(&candidate->provider);
  }
  if (candidate->p50_samples != NULL) {
    iree_allocator_free(allocator, candidate->p50_samples);
  }
  if (candidate->p90_samples != NULL) {
    iree_allocator_free(allocator, candidate->p90_samples);
  }
  *candidate = (iree_tune_loom_dispatch_comparison_candidate_t){0};
}

static void iree_tune_loom_dispatch_comparison_candidates_deinitialize(
    iree_tune_loom_dispatch_comparison_candidate_t* candidates,
    iree_host_size_t candidate_count, iree_allocator_t allocator) {
  if (candidates == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    iree_tune_loom_dispatch_comparison_candidate_deinitialize(&candidates[i],
                                                              allocator);
  }
  iree_allocator_free(allocator, candidates);
}

static iree_status_t iree_tune_loom_hal_actual_provider_compile(
    iree_tune_loom_hal_actual_provider_t* provider) {
  return loom_run_hal_testbench_actual_provider_compile(&provider->execution);
}

typedef struct iree_tune_loom_hal_input_ring_t {
  // Host allocator used for ring-owned arrays.
  iree_allocator_t host_allocator;
  // Ring-owned invocation plans materialized from check ops.
  loom_run_hal_invocation_plan_t* plans;
  // Borrowed binding-list pointers into |plans| for HAL benchmark setup.
  iree_vm_list_t** binding_lists;
  // Number of entries in |plans| and |binding_lists|.
  iree_host_size_t plan_count;
  // Data/cache summary derived while materializing the ring.
  iree_tune_loom_data_cache_summary_t summary;
} iree_tune_loom_hal_input_ring_t;

static void iree_tune_loom_hal_input_ring_deinitialize(
    iree_tune_loom_hal_input_ring_t* ring) {
  if (ring == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < ring->plan_count; ++i) {
    loom_run_hal_invocation_plan_deinitialize(&ring->plans[i]);
  }
  if (ring->binding_lists != NULL) {
    iree_allocator_free(ring->host_allocator, ring->binding_lists);
  }
  if (ring->plans != NULL) {
    iree_allocator_free(ring->host_allocator, ring->plans);
  }
  *ring = (iree_tune_loom_hal_input_ring_t){0};
}

static bool iree_tune_loom_hal_invocation_options_equal(
    const loom_run_hal_invocation_options_t* lhs,
    const loom_run_hal_invocation_options_t* rhs) {
  if (lhs->entry_point != rhs->entry_point ||
      lhs->constant_count != rhs->constant_count) {
    return false;
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(lhs->workgroup_count); ++i) {
    if (lhs->workgroup_count[i] != rhs->workgroup_count[i]) {
      return false;
    }
  }
  for (iree_host_size_t i = 0; i < lhs->constant_count; ++i) {
    if (lhs->constants[i] != rhs->constants[i]) {
      return false;
    }
  }
  return true;
}

static iree_status_t iree_tune_loom_hal_input_ring_count_for_sample(
    uint64_t binding_set_bytes, iree_host_size_t dispatches_per_batch,
    iree_host_size_t* out_ring_count) {
  *out_ring_count = 1;
  if (FLAG_input_ring_count > 0) {
    *out_ring_count = (iree_host_size_t)FLAG_input_ring_count;
    return iree_ok_status();
  }
  if (FLAG_input_ring_min_bytes == 0 || binding_set_bytes == 0) {
    return iree_ok_status();
  }
  const uint64_t requested_min_bytes = (uint64_t)FLAG_input_ring_min_bytes;
  const uint64_t byte_sized_count =
      requested_min_bytes / binding_set_bytes +
      (requested_min_bytes % binding_set_bytes == 0 ? 0 : 1);
  uint64_t ring_count = byte_sized_count;
  if (ring_count < dispatches_per_batch) {
    ring_count = dispatches_per_batch;
  }
  if (ring_count > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL benchmark input ring count %" PRIu64
                            " exceeds host size limits",
                            ring_count);
  }
  *out_ring_count = (iree_host_size_t)ring_count;
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_prepare_hal_invocation_plan_for_sample(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_hal_actual_provider_t* provider,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    loom_run_hal_invocation_plan_t* out_plan) {
  loom_run_hal_invocation_options_t invocation_options = {0};
  iree_vm_list_t* bindings = NULL;
  iree_status_t status =
      loom_run_hal_testbench_create_invocation_inputs_for_sample(
          module_plan->module, materializer_options, case_plan,
          provider->execution.actual_invocation, sample_ordinal,
          &provider->execution.invocation_options, allocator,
          &invocation_options, &bindings);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_plan_prepare_from_lists(
        &invocation_options, bindings, /*expected_bindings=*/NULL,
        /*max_output_element_count=*/0, out_plan);
  }
  iree_vm_list_release(bindings);
  return status;
}

static iree_status_t iree_tune_loom_hal_input_ring_prepare(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_hal_actual_provider_t* provider,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    iree_tune_loom_hal_input_ring_t* out_ring) {
  *out_ring = (iree_tune_loom_hal_input_ring_t){
      .host_allocator = allocator,
  };

  loom_run_hal_invocation_plan_t first_plan = {0};
  iree_status_t status = iree_tune_loom_prepare_hal_invocation_plan_for_sample(
      module_plan, case_plan, provider, materializer_options, sample_ordinal,
      allocator, &first_plan);
  uint64_t first_binding_set_bytes = 0;
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_binding_list_total_byte_length(
        first_plan.bindings, &first_binding_set_bytes);
  }

  iree_host_size_t ring_count = 1;
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_hal_input_ring_count_for_sample(
        first_binding_set_bytes, policy->hal_options.timing.batch_size,
        &ring_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, ring_count,
                                         sizeof(*out_ring->plans),
                                         (void**)&out_ring->plans);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, ring_count,
                                         sizeof(*out_ring->binding_lists),
                                         (void**)&out_ring->binding_lists);
  }
  if (iree_status_is_ok(status)) {
    out_ring->plan_count = ring_count;
    out_ring->plans[0] = first_plan;
    first_plan = (loom_run_hal_invocation_plan_t){0};
    out_ring->binding_lists[0] = out_ring->plans[0].bindings;
    out_ring->summary = (iree_tune_loom_data_cache_summary_t){
        .populated = true,
        .binding_count = iree_vm_list_size(out_ring->plans[0].bindings),
        .binding_ring_count = ring_count,
        .dispatches_per_batch = policy->hal_options.timing.batch_size,
        .requested_min_ring_bytes = (uint64_t)FLAG_input_ring_min_bytes,
        .binding_set_bytes = first_binding_set_bytes,
        .binding_ring_bytes = first_binding_set_bytes,
    };
  }
  for (iree_host_size_t i = 1; iree_status_is_ok(status) && i < ring_count;
       ++i) {
    status = iree_tune_loom_prepare_hal_invocation_plan_for_sample(
        module_plan, case_plan, provider, materializer_options, sample_ordinal,
        allocator, &out_ring->plans[i]);
    if (iree_status_is_ok(status) &&
        !iree_tune_loom_hal_invocation_options_equal(
            &out_ring->plans[0].options, &out_ring->plans[i].options)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL benchmark input ring materialization changed dispatch constants "
          "or geometry for sample %" PRIhsz,
          sample_ordinal);
    }
    uint64_t binding_set_bytes = 0;
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_binding_list_total_byte_length(
          out_ring->plans[i].bindings, &binding_set_bytes);
    }
    if (iree_status_is_ok(status)) {
      if (UINT64_MAX - out_ring->summary.binding_ring_bytes <
          binding_set_bytes) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "HAL benchmark input ring byte count "
                                  "overflowed uint64");
      } else {
        out_ring->binding_lists[i] = out_ring->plans[i].bindings;
        out_ring->summary.binding_ring_bytes += binding_set_bytes;
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_tune_loom_hal_input_ring_deinitialize(out_ring);
  }
  loom_run_hal_invocation_plan_deinitialize(&first_plan);
  return status;
}

static void iree_tune_loom_benchmark_result_set_compile_rejection(
    const iree_tune_loom_hal_actual_provider_t* provider,
    iree_tune_loom_benchmark_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  out_result->status = IREE_SV("compile_failed");
  out_result->has_failure = true;
  out_result->failure_stage = provider->execution.compile_failure_stage;
  out_result->failure_kind = provider->execution.compile_failure_kind;
  out_result->failure_message = provider->execution.compile_failure_message;
  out_result->diagnostic_error_count = provider->diagnostics.error_count;
  out_result->diagnostic_warning_count = provider->diagnostics.warning_count;
  out_result->diagnostic_remark_count = provider->diagnostics.remark_count;
  out_result->diagnostic_json =
      iree_string_builder_view(&provider->diagnostics.output);
  out_result->specialization = provider->specialization;
  if (provider->execution.has_specialization_sample_ordinal) {
    out_result->has_sample_ordinal = true;
    out_result->sample_ordinal =
        provider->execution.specialization_sample_ordinal;
    out_result->samples_per_iteration = 1;
  }
  if (provider->execution.compile_report_available) {
    out_result->compile_report_capture = &provider->compile_report_capture;
  }
  out_result->compile_report_artifact_path =
      provider->compile_report_artifact_path;
  out_result->target_artifact_path = provider->target_artifact_path;
  out_result->target_listing_path = provider->target_listing_path;
  out_result->hal_executable_path = provider->hal_executable_path;
}

static iree_status_t iree_tune_loom_append_effective_profile_artifacts_dir(
    const iree_tune_loom_run_identity_t* run,
    iree_hal_device_profiling_data_families_t profile_data_families,
    iree_string_builder_t* artifact_dir) {
  const iree_string_view_t explicit_artifacts_dir =
      iree_string_view_trim(iree_make_cstring_view(FLAG_profile_artifacts_dir));
  if (!iree_string_view_is_empty(explicit_artifacts_dir)) {
    return iree_string_builder_append_string(artifact_dir,
                                             explicit_artifacts_dir);
  }
  if (!iree_string_view_is_empty(run->profile_artifacts_dir)) {
    return iree_string_builder_append_string(artifact_dir,
                                             run->profile_artifacts_dir);
  }
  if (!iree_tune_loom_profile_data_needs_artifact_data(profile_data_families)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(artifact_dir, run->file_output_dir));
  if (!iree_string_view_ends_with(run->file_output_dir, IREE_SV("/"))) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(artifact_dir, "/"));
  }
  return iree_string_builder_append_cstring(artifact_dir, "profiles");
}

static iree_status_t iree_tune_loom_append_profile_artifact_path(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    iree_hal_device_profiling_data_families_t profile_data_families,
    iree_string_view_t specialization, iree_host_size_t sample_ordinal,
    iree_string_builder_t* artifact_path) {
  const iree_host_size_t initial_size = iree_string_builder_size(artifact_path);
  IREE_RETURN_IF_ERROR(iree_tune_loom_append_effective_profile_artifacts_dir(
      run, profile_data_families, artifact_path));
  if (iree_string_builder_size(artifact_path) == initial_size) {
    return iree_ok_status();
  }
  iree_string_view_t artifacts_dir = iree_string_builder_view(artifact_path);
  if (!iree_string_view_ends_with(artifacts_dir, IREE_SV("/"))) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(artifact_path, "/"));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(artifact_path, run->run_id));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(artifact_path, "_"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      artifact_path, candidate->candidate_id));
  if (!iree_string_view_is_empty(specialization)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(artifact_path, "_"));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(artifact_path, specialization));
  }
  return iree_string_builder_append_format(
      artifact_path, "_sample%" PRIhsz ".irpf", sample_ordinal);
}

static iree_status_t iree_tune_loom_run_hal_benchmark_sample(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    iree_tune_loom_hal_actual_provider_t* provider,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    iree_tune_loom_benchmark_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  out_result->specialization = provider->specialization;
  out_result->has_sample_ordinal = true;
  out_result->sample_ordinal = sample_ordinal;
  out_result->samples_per_iteration = 1;

  IREE_RETURN_IF_ERROR(iree_tune_loom_hal_actual_provider_compile(provider));
  if (provider->execution.compile_report_available) {
    out_result->compile_report_capture = &provider->compile_report_capture;
  }
  out_result->compile_report_artifact_path =
      provider->compile_report_artifact_path;
  out_result->target_artifact_path = provider->target_artifact_path;
  out_result->target_listing_path = provider->target_listing_path;
  out_result->hal_executable_path = provider->hal_executable_path;
  if (provider->execution.compile_rejected) {
    iree_tune_loom_benchmark_result_set_compile_rejection(provider, out_result);
    out_result->has_sample_ordinal = true;
    out_result->sample_ordinal = sample_ordinal;
    out_result->samples_per_iteration = 1;
    return iree_ok_status();
  }

  iree_tune_loom_hal_input_ring_t input_ring = {0};
  iree_status_t status = iree_tune_loom_hal_input_ring_prepare(
      module_plan, case_plan, policy, provider, materializer_options,
      sample_ordinal, allocator, &input_ring);
  if (iree_status_is_ok(status)) {
    out_result->has_hal_benchmark = true;
    out_result->data_cache = input_ring.summary;
    loom_run_hal_benchmark_options_t hal_options = policy->hal_options;
    iree_string_builder_t profile_artifact_path;
    iree_string_builder_initialize(allocator, &profile_artifact_path);
    status = iree_tune_loom_append_profile_artifact_path(
        run, candidate, policy->hal_options.profile_data_families,
        provider->specialization, sample_ordinal, &profile_artifact_path);
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      status = iree_tune_loom_create_parent_directory(
          iree_string_builder_view(&profile_artifact_path), allocator);
    }
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      status = iree_tune_loom_artifact_bundle_record_file(
          provider->context->artifact_bundle,
          IREE_TUNE_LOOM_BUNDLE_FILE_PROFILE,
          iree_string_builder_view(&profile_artifact_path));
    }
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      hal_options.profile_artifact_path =
          iree_string_builder_view(&profile_artifact_path);
    }
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_benchmark_dispatch_binding_ring(
          &provider->context->execution.runtime,
          &provider->execution.prepared_candidate, &input_ring.plans[0],
          input_ring.plan_count, input_ring.binding_lists, &hal_options,
          allocator, &out_result->hal_benchmark);
    }
    if (iree_status_is_ok(status)) {
      out_result->data_cache.command_buffer_ring_count =
          out_result->hal_benchmark.command_buffer_ring_count;
    }
    iree_string_builder_deinitialize(&profile_artifact_path);
  }
  if (iree_status_is_ok(status)) {
    out_result->executed = true;
    out_result->passed = true;
  }
  iree_tune_loom_hal_input_ring_deinitialize(&input_ring);
  return status;
}

static iree_status_t iree_tune_loom_prepare_case_executor(
    const loom_testbench_module_plan_t* module_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* arena,
    loom_testbench_prepared_case_t* out_prepared_case,
    loom_testbench_case_executor_t* out_executor, bool* out_initialized) {
  *out_initialized = false;
  IREE_RETURN_IF_ERROR(loom_testbench_prepare_case_execution(
      execution_options, module_plan, case_index, arena, out_prepared_case));
  IREE_RETURN_IF_ERROR(loom_testbench_case_executor_initialize(
      out_prepared_case, execution_options, out_executor));
  *out_initialized = true;
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_run_case_correctness_range(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_string_view_t specialization, iree_host_size_t begin_sample,
    iree_host_size_t end_sample, iree_arena_allocator_t* arena,
    iree_tune_loom_jsonl_sink_t* jsonl_sink, iree_host_size_t* out_sample_count,
    iree_host_size_t* out_failed_sample_count) {
  *out_sample_count = 0;
  *out_failed_sample_count = 0;

  const loom_testbench_case_plan_t* case_plan = &module_plan->cases[case_index];
  iree_status_t status = iree_ok_status();
  loom_testbench_prepared_case_t prepared_case = {0};
  loom_testbench_case_executor_t executor = {0};
  bool executor_initialized = false;
  status = iree_tune_loom_prepare_case_executor(
      module_plan, case_index, execution_options, arena, &prepared_case,
      &executor, &executor_initialized);

  for (iree_host_size_t sample_ordinal = begin_sample;
       iree_status_is_ok(status) && sample_ordinal < end_sample;
       ++sample_ordinal) {
    loom_testbench_case_sample_result_t sample_result = {0};
    status = loom_testbench_run_case_sample(&executor, sample_ordinal,
                                            &sample_result);
    if (iree_status_is_ok(status)) {
      loom_output_stream_t stream;
      loom_output_stream_for_builder(
          iree_tune_loom_jsonl_sink_begin(jsonl_sink), &stream);
      status = loom_output_stream_write_cstring(&stream, "{\"row\":\"sample\"");
      if (iree_status_is_ok(status)) {
        status = iree_tune_loom_write_run_id_field_json(run, &stream);
      }
      if (iree_status_is_ok(status)) {
        status =
            iree_tune_loom_write_candidate_identity_json(candidate, &stream);
      }
      if (iree_status_is_ok(status)) {
        status = iree_tune_loom_write_benchmark_metadata_json(
            module_plan->module, benchmark_plan, &stream);
      }
      if (iree_status_is_ok(status)) {
        status = iree_tune_loom_write_specialization_field_json(specialization,
                                                                &stream);
      }
      if (iree_status_is_ok(status)) {
        status = iree_tune_loom_write_shape_point_fields_json(
            module_plan->module, case_plan, sample_ordinal, &stream);
      }
      if (iree_status_is_ok(status)) {
        status = loom_output_stream_write_cstring(&stream, ",\"sample\":");
      }
      if (iree_status_is_ok(status)) {
        status = loom_testbench_case_sample_result_write_json(&sample_result,
                                                              &stream);
      }
      if (iree_status_is_ok(status)) {
        status = loom_output_stream_write_cstring(&stream, "}\n");
      }
      status = iree_tune_loom_jsonl_sink_end(jsonl_sink, status);
    }
    if (iree_status_is_ok(status)) {
      ++*out_sample_count;
      if (!sample_result.passed) {
        ++*out_failed_sample_count;
      }
    }
  }

  if (executor_initialized) {
    loom_testbench_case_executor_deinitialize(&executor);
  }
  iree_arena_reset(arena);
  return status;
}

static iree_status_t iree_tune_loom_run_case_correctness(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_string_view_t specialization, iree_arena_allocator_t* arena,
    iree_tune_loom_jsonl_sink_t* jsonl_sink, iree_host_size_t* out_sample_count,
    iree_host_size_t* out_failed_sample_count) {
  const loom_testbench_case_plan_t* case_plan = &module_plan->cases[case_index];
  iree_host_size_t begin_sample = 0;
  iree_host_size_t end_sample = 0;
  IREE_RETURN_IF_ERROR(iree_tune_loom_validate_sample_flag(
      case_plan->sample_count, &begin_sample, &end_sample));
  return iree_tune_loom_run_case_correctness_range(
      run, candidate, module_plan, benchmark_plan, case_index,
      execution_options, specialization, begin_sample, end_sample, arena,
      jsonl_sink, out_sample_count, out_failed_sample_count);
}

static iree_status_t iree_tune_loom_run_case_iteration(
    loom_testbench_case_executor_t* executor, iree_host_size_t begin_sample,
    iree_host_size_t end_sample, iree_host_size_t* inout_failed_sample_count) {
  for (iree_host_size_t sample_ordinal = begin_sample;
       sample_ordinal < end_sample; ++sample_ordinal) {
    loom_testbench_case_sample_result_t sample_result = {0};
    IREE_RETURN_IF_ERROR(loom_testbench_run_case_sample(
        executor, sample_ordinal, &sample_result));
    if (!sample_result.passed) {
      ++*inout_failed_sample_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_run_benchmark_iterations(
    const loom_testbench_module_plan_t* module_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* execution_options,
    const iree_tune_loom_benchmark_policy_t* policy,
    iree_arena_allocator_t* arena, iree_allocator_t host_allocator,
    iree_tune_loom_benchmark_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));

  const loom_testbench_case_plan_t* case_plan = &module_plan->cases[case_index];
  iree_host_size_t begin_sample = 0;
  iree_host_size_t end_sample = 0;
  IREE_RETURN_IF_ERROR(iree_tune_loom_validate_sample_flag(
      case_plan->sample_count, &begin_sample, &end_sample));
  out_result->samples_per_iteration = end_sample - begin_sample;
  if (end_sample == begin_sample + 1) {
    out_result->has_sample_ordinal = true;
    out_result->sample_ordinal = begin_sample;
  }

  iree_duration_t* durations = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, policy->iterations * sizeof(*durations),
      (void**)&durations));

  iree_status_t status = iree_ok_status();
  loom_testbench_prepared_case_t prepared_case = {0};
  loom_testbench_case_executor_t executor = {0};
  bool executor_initialized = false;
  status = iree_tune_loom_prepare_case_executor(
      module_plan, case_index, execution_options, arena, &prepared_case,
      &executor, &executor_initialized);

  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < policy->warmup_iterations; ++i) {
    status = iree_tune_loom_run_case_iteration(
        &executor, begin_sample, end_sample, &out_result->failed_sample_count);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < policy->iterations; ++i) {
    const iree_time_t start_time_ns = iree_time_now();
    status = iree_tune_loom_run_case_iteration(
        &executor, begin_sample, end_sample, &out_result->failed_sample_count);
    const iree_time_t end_time_ns = iree_time_now();
    durations[i] =
        end_time_ns >= start_time_ns ? end_time_ns - start_time_ns : 0;
  }

  if (iree_status_is_ok(status)) {
    out_result->executed = true;
    out_result->passed = out_result->failed_sample_count == 0;
    iree_tune_loom_compute_timing_stats(durations, policy->iterations,
                                        &out_result->timing);
  }

  if (executor_initialized) {
    loom_testbench_case_executor_deinitialize(&executor);
  }
  iree_arena_reset(arena);
  iree_allocator_free(host_allocator, durations);
  return status;
}

static iree_status_t iree_tune_loom_write_actual_invocation_plan_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    loom_output_stream_t* stream) {
  const loom_testbench_invocation_plan_t* actual_invocation = NULL;
  iree_host_size_t actual_invocation_count = 0;
  for (iree_host_size_t i = 0; i < case_plan->invocation_count; ++i) {
    const loom_testbench_invocation_plan_t* invocation =
        &case_plan->invocations[i];
    if (invocation->kind != LOOM_TESTBENCH_INVOCATION_ACTUAL) {
      continue;
    }
    actual_invocation = invocation;
    ++actual_invocation_count;
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"actual_invocation_count\":%" PRIhsz,
      actual_invocation_count));
  if (actual_invocation_count != 1) {
    return iree_ok_status();
  }

  iree_string_view_t actual_entry = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_tune_loom_module_symbol_name_from_ref(
      module, actual_invocation->callee_ref, &actual_entry));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"actual_entry\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, actual_entry));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"actual_input_count\":%" PRIhsz,
      actual_invocation->input_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"actual_result_count\":%" PRIhsz,
      actual_invocation->result_count));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_append_plan_row(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    iree_tune_loom_shape_specialization_mode_t shape_specialization_mode,
    iree_allocator_t allocator, iree_string_builder_t* plan_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(plan_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"plan\""));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_candidate_identity_json(candidate, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"benchmark\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"case\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, case_plan->name));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_benchmark_metadata_json(
      module, benchmark_plan, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"measure\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, policy->measure));
  if (policy->measure_kind == IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"shape_specialization\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, iree_tune_loom_shape_specialization_mode_name(
                     shape_specialization_mode)));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"warmup_iterations\":%" PRIhsz, policy->warmup_iterations));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"iterations\":%" PRIhsz, policy->iterations));
  if (FLAG_sample >= 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"selected_sample\":%" PRId32, FLAG_sample));
  }
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_actual_invocation_plan_json(
      module, case_plan, &stream));
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_shape_plan_fields_json(module, case_plan, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"cli_overrides\":{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, "\"iterations\":%s,\"warmup_iterations\":%s",
      FLAG_iterations.specified ? "true" : "false",
      FLAG_warmup_iterations.specified ? "true" : "false"));
  if (policy->measure_kind == IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        ",\"batch_size\":%s,\"min_time_ms\":%s,\"warmup_time_ms\":%s,"
        "\"max_batches\":%s,\"stable_p90_to_p50_ppm\":%s,"
        "\"profile_final_batch\":%s,\"profile_data\":%s,"
        "\"profile_counter\":%s,\"profile_artifacts_dir\":%s,"
        "\"input_ring_min_bytes\":%s,\"input_ring_count\":%s",
        FLAG_batch_size.specified ? "true" : "false",
        FLAG_min_time_ms.specified ? "true" : "false",
        FLAG_warmup_time_ms.specified ? "true" : "false",
        FLAG_max_batches.specified ? "true" : "false",
        FLAG_stable_p90_to_p50_ppm.specified ? "true" : "false",
        FLAG_profile_final_batch.specified ? "true" : "false",
        strlen(FLAG_profile_data) != 0 ? "true" : "false",
        FLAG_profile_counter_list().count != 0 ? "true" : "false",
        strlen(FLAG_profile_artifacts_dir) != 0 ? "true" : "false",
        FLAG_input_ring_min_bytes != IREE_TUNE_LOOM_DEFAULT_INPUT_RING_MIN_BYTES
            ? "true"
            : "false",
        FLAG_input_ring_count != 0 ? "true" : "false"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  if (policy->measure_kind == IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE) {
    const loom_run_benchmark_options_t* timing = &policy->hal_options.timing;
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"batch_size\":%" PRIhsz, timing->batch_size));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"min_time_ns\":%" PRIi64, timing->min_duration_ns));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_format(&stream, ",\"warmup_time_ns\":%" PRIi64,
                                        timing->warmup_min_duration_ns));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"max_batches\":%" PRIhsz, timing->max_batch_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"stable_p90_to_p50_ppm\":%" PRIu64,
        timing->stable_p90_to_p50_delta_ppm));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"profile_final_batch\":%s",
        iree_all_bits_set(policy->hal_options.flags,
                          LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH)
            ? "true"
            : "false"));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"data_cache_policy\":{"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, "\"validity\":\"check_ops\""));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"cache_policy\":\"binding_ring\""));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"input_ring_min_bytes\":%" PRIi64,
        FLAG_input_ring_min_bytes));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"input_ring_count\":%" PRId32, FLAG_input_ring_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
    if (iree_all_bits_set(policy->hal_options.flags,
                          LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH)) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream, ",\"profile_data_families\":%" PRIu64,
          policy->hal_options.profile_data_families));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
          &stream, ",\"profile_data_family_names\":"));
      IREE_RETURN_IF_ERROR(iree_tune_loom_write_profile_family_names_json(
          policy->hal_options.profile_data_families, &stream));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream, ",\"profile_counter_set_count\":%" PRIhsz,
          policy->hal_options.profile_counter_set_count));
      if (policy->hal_options.profile_counter_set_count != 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
            &stream, ",\"profile_counter_request\":"));
        IREE_RETURN_IF_ERROR(
            iree_tune_loom_write_profile_counter_request_json(policy, &stream));
      }
      iree_string_builder_t profile_artifacts_dir;
      iree_string_builder_initialize(allocator, &profile_artifacts_dir);
      iree_status_t profile_artifacts_status =
          iree_tune_loom_append_effective_profile_artifacts_dir(
              run, policy->hal_options.profile_data_families,
              &profile_artifacts_dir);
      if (iree_status_is_ok(profile_artifacts_status) &&
          iree_string_builder_size(&profile_artifacts_dir) != 0) {
        profile_artifacts_status = loom_output_stream_write_cstring(
            &stream, ",\"profile_artifacts_dir\":");
        if (iree_status_is_ok(profile_artifacts_status)) {
          profile_artifacts_status = loom_json_write_escaped_string(
              &stream, iree_string_builder_view(&profile_artifacts_dir));
        }
      }
      iree_string_builder_deinitialize(&profile_artifacts_dir);
      IREE_RETURN_IF_ERROR(profile_artifacts_status);
    }
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_append_dispatch_benchmark_result(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    iree_tune_loom_jsonl_sink_t* jsonl_sink, iree_allocator_t allocator) {
  IREE_RETURN_IF_ERROR(iree_tune_loom_jsonl_sink_end(
      jsonl_sink, iree_tune_loom_append_benchmark_result(
                      run, candidate, module_plan->module, benchmark_plan,
                      case_plan, policy, benchmark_result,
                      correctness_sample_count, correctness_failed_sample_count,
                      iree_tune_loom_jsonl_sink_begin(jsonl_sink))));
  return iree_tune_loom_jsonl_sink_end(
      jsonl_sink, iree_tune_loom_append_profile_row(
                      run, candidate, module_plan->module, benchmark_plan,
                      case_plan, policy, benchmark_result, allocator,
                      iree_tune_loom_jsonl_sink_begin(jsonl_sink)));
}

static iree_status_t iree_tune_loom_run_dispatch_complete_benchmark(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    iree_tune_loom_hal_context_t* hal_context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_string_view_t specialization, bool has_specialization_sample_ordinal,
    iree_host_size_t specialization_sample_ordinal,
    iree_tune_loom_device_row_state_t* device_row_state,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    iree_tune_loom_jsonl_sink_t* jsonl_sink,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const loom_testbench_invocation_plan_t* actual_invocation = NULL;
  iree_tune_loom_hal_actual_provider_t hal_provider = {0};
  bool hal_provider_initialized = false;
  loom_testbench_case_execution_options_t benchmark_execution_options =
      *execution_options;
  loom_testbench_value_materializer_options_t benchmark_materializer =
      execution_options->materializer;

  iree_status_t status = loom_run_hal_testbench_select_actual_invocation(
      case_plan, &actual_invocation);
  if (iree_status_is_ok(status)) {
    status =
        loom_run_hal_testbench_context_ensure_runtime(&hal_context->execution);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_hal_actual_provider_initialize(
        hal_context, session, filename, source,
        iree_make_cstring_view(FLAG_pipeline), module_plan->module,
        actual_invocation, specialization, case_plan,
        specialization_sample_ordinal, has_specialization_sample_ordinal,
        compile_report_options, &hal_provider);
  }
  if (iree_status_is_ok(status)) {
    hal_provider_initialized = true;
    benchmark_execution_options.materializer.device =
        hal_context->execution.runtime.device;
    benchmark_execution_options.materializer.device_allocator =
        iree_hal_device_allocator(hal_context->execution.runtime.device);
    benchmark_execution_options.materializer.buffer_params =
        loom_run_hal_testbench_host_visible_buffer_params();
    benchmark_execution_options.invocation.invoke_actual =
        (loom_testbench_invocation_callback_t){
            .fn = loom_run_hal_testbench_actual_invoke,
            .user_data = &hal_provider.execution,
        };
    benchmark_materializer = benchmark_execution_options.materializer;
    benchmark_materializer.buffer_params = (iree_hal_buffer_params_t){0};
    status = iree_tune_loom_hal_actual_provider_compile(&hal_provider);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_write_compiled_artifacts(run, candidate,
                                                     &hal_provider, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_write_compile_report_artifact(
        run, candidate, benchmark_plan, case_plan, &hal_provider, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_jsonl_sink_end(
        jsonl_sink, iree_tune_loom_append_device_row(
                        run, hal_context, device_row_state,
                        iree_tune_loom_jsonl_sink_begin(jsonl_sink)));
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_jsonl_sink_end(
        jsonl_sink,
        iree_tune_loom_append_compile_row(
            run, candidate, benchmark_plan, case_plan, &hal_provider,
            iree_tune_loom_jsonl_sink_begin(jsonl_sink)));
  }
  if (iree_status_is_ok(status) && hal_provider.execution.compile_rejected) {
    iree_tune_loom_benchmark_result_t benchmark_result = {0};
    iree_tune_loom_benchmark_result_set_compile_rejection(&hal_provider,
                                                          &benchmark_result);
    ++*inout_failed_benchmark_count;
    status = iree_tune_loom_append_dispatch_benchmark_result(
        run, candidate, module_plan, benchmark_plan, case_plan, policy,
        &benchmark_result, /*correctness_sample_count=*/0,
        /*correctness_failed_sample_count=*/0, jsonl_sink, allocator);
  }

  iree_host_size_t begin_sample = 0;
  iree_host_size_t end_sample = 0;
  if (iree_status_is_ok(status) && !hal_provider.execution.compile_rejected &&
      has_specialization_sample_ordinal) {
    begin_sample = specialization_sample_ordinal;
    end_sample = specialization_sample_ordinal + 1;
  } else if (iree_status_is_ok(status) &&
             !hal_provider.execution.compile_rejected) {
    status = iree_tune_loom_validate_sample_flag(case_plan->sample_count,
                                                 &begin_sample, &end_sample);
  }

  iree_host_size_t benchmark_correctness_sample_count = 0;
  iree_host_size_t benchmark_correctness_failed_sample_count = 0;
  if (iree_status_is_ok(status) && !hal_provider.execution.compile_rejected) {
    status = iree_tune_loom_run_case_correctness_range(
        run, candidate, module_plan, benchmark_plan, benchmark_plan->case_index,
        &benchmark_execution_options, specialization, begin_sample, end_sample,
        execution_arena, jsonl_sink, &benchmark_correctness_sample_count,
        &benchmark_correctness_failed_sample_count);
  }
  if (iree_status_is_ok(status) && !hal_provider.execution.compile_rejected) {
    *inout_correctness_sample_count += benchmark_correctness_sample_count;
    *inout_correctness_failed_sample_count +=
        benchmark_correctness_failed_sample_count;
  }

  if (iree_status_is_ok(status) && !hal_provider.execution.compile_rejected &&
      benchmark_correctness_failed_sample_count != 0) {
    iree_tune_loom_benchmark_result_t benchmark_result = {
        .executed = false,
        .passed = false,
        .compile_report_artifact_path =
            hal_provider.compile_report_artifact_path,
        .target_artifact_path = hal_provider.target_artifact_path,
        .target_listing_path = hal_provider.target_listing_path,
        .hal_executable_path = hal_provider.hal_executable_path,
        .specialization = specialization,
        .samples_per_iteration = benchmark_correctness_sample_count,
        .failed_sample_count = benchmark_correctness_failed_sample_count,
    };
    if (has_specialization_sample_ordinal || FLAG_sample >= 0) {
      benchmark_result.has_sample_ordinal = true;
      benchmark_result.sample_ordinal = has_specialization_sample_ordinal
                                            ? specialization_sample_ordinal
                                            : (iree_host_size_t)FLAG_sample;
    }
    ++*inout_failed_benchmark_count;
    status = iree_tune_loom_append_dispatch_benchmark_result(
        run, candidate, module_plan, benchmark_plan, case_plan, policy,
        &benchmark_result, benchmark_correctness_sample_count,
        benchmark_correctness_failed_sample_count, jsonl_sink, allocator);
  }

  for (iree_host_size_t sample_ordinal = begin_sample;
       iree_status_is_ok(status) && !hal_provider.execution.compile_rejected &&
       benchmark_correctness_failed_sample_count == 0 &&
       sample_ordinal < end_sample;
       ++sample_ordinal) {
    iree_tune_loom_benchmark_result_t benchmark_result = {0};
    status = iree_tune_loom_run_hal_benchmark_sample(
        run, candidate, module_plan, case_plan, policy, &hal_provider,
        &benchmark_materializer, sample_ordinal, allocator, &benchmark_result);
    if (iree_status_is_ok(status)) {
      if (!benchmark_result.executed || !benchmark_result.passed) {
        ++*inout_failed_benchmark_count;
      }
      status = iree_tune_loom_append_dispatch_benchmark_result(
          run, candidate, module_plan, benchmark_plan, case_plan, policy,
          &benchmark_result, benchmark_correctness_sample_count,
          benchmark_correctness_failed_sample_count, jsonl_sink, allocator);
    }
  }

  if (hal_provider_initialized) {
    iree_tune_loom_hal_actual_provider_deinitialize(&hal_provider);
  }
  return status;
}

static iree_status_t iree_tune_loom_comparison_candidate_record_timing(
    iree_tune_loom_dispatch_comparison_candidate_t* candidate,
    const iree_tune_loom_benchmark_result_t* benchmark_result) {
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

static iree_status_t iree_tune_loom_append_benchmark_repetition_row(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_dispatch_comparison_candidate_t* candidate,
    const iree_tune_loom_candidate_identity_t* baseline,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_host_size_t order_index, iree_host_size_t repetition_index,
    char schedule_token, bool profile_suppressed,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_string_builder_t* benchmark_output) {
  const iree_tune_loom_selected_benchmark_t* selection = candidate->selection;
  loom_output_stream_t stream;
  loom_output_stream_for_builder(benchmark_output, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "{\"row\":\"benchmark.repetition\""));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_candidate_identity_json(
      &selection->identity, &stream));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_benchmark_metadata_json(
      candidate->provider.execution.test_module, selection->benchmark_plan,
      &stream));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_specialization_field_json(
      candidate->specialization, &stream));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_shape_point_fields_json(
      candidate->provider.execution.test_module, selection->case_plan,
      benchmark_result->sample_ordinal, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"comparison_group\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, comparison_group));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"baseline_candidate_id\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, baseline->candidate_id));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"method\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, method));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"order_index\":%" PRIhsz, order_index));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"repetition_index\":%" PRIhsz, repetition_index));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"schedule_token\":\"%c\"", schedule_token));
  if (profile_suppressed) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"profile_suppressed_for_interleave\":true"));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"benchmark_result\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_benchmark_result_json(
      selection->benchmark_plan, selection->case_plan, &selection->policy,
      benchmark_result, candidate->correctness_sample_count,
      candidate->correctness_failed_sample_count, &stream));
  return loom_output_stream_write_cstring(&stream, "}\n");
}

static iree_status_t iree_tune_loom_append_comparison_row(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_dispatch_comparison_candidate_t* baseline,
    const iree_tune_loom_dispatch_comparison_candidate_t* candidate,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_string_builder_t* benchmark_output) {
  if (baseline->sample_count == 0 || candidate->sample_count == 0) {
    return iree_ok_status();
  }

  loom_run_benchmark_timing_stats_t baseline_p50 = {0};
  loom_run_benchmark_timing_stats_t candidate_p50 = {0};
  loom_run_benchmark_timing_stats_t baseline_p90 = {0};
  loom_run_benchmark_timing_stats_t candidate_p90 = {0};
  IREE_RETURN_IF_ERROR(loom_run_benchmark_compute_timing_stats(
      baseline->p50_samples, baseline->sample_count, &baseline_p50));
  IREE_RETURN_IF_ERROR(loom_run_benchmark_compute_timing_stats(
      candidate->p50_samples, candidate->sample_count, &candidate_p50));
  IREE_RETURN_IF_ERROR(loom_run_benchmark_compute_timing_stats(
      baseline->p90_samples, baseline->sample_count, &baseline_p90));
  IREE_RETURN_IF_ERROR(loom_run_benchmark_compute_timing_stats(
      candidate->p90_samples, candidate->sample_count, &candidate_p90));

  const double baseline_p50_ns = (double)baseline_p50.p50_ns;
  const double candidate_p50_ns = (double)candidate_p50.p50_ns;
  const double ratio_p50 =
      baseline_p50_ns == 0.0 ? 0.0 : candidate_p50_ns / baseline_p50_ns;
  const double speedup_p50 =
      candidate_p50_ns == 0.0 ? 0.0 : baseline_p50_ns / candidate_p50_ns;
  const double baseline_p90_ns = (double)baseline_p90.p50_ns;
  const double candidate_p90_ns = (double)candidate_p90.p50_ns;
  const double ratio_p90 =
      baseline_p90_ns == 0.0 ? 0.0 : candidate_p90_ns / baseline_p90_ns;
  const double speedup_p90 =
      candidate_p90_ns == 0.0 ? 0.0 : baseline_p90_ns / candidate_p90_ns;

  loom_output_stream_t stream;
  loom_output_stream_for_builder(benchmark_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"comparison\""));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"comparison_group\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, comparison_group));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"method\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, method));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"baseline_candidate_id\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, baseline->selection->identity.candidate_id));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"candidate_id\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, candidate->selection->identity.candidate_id));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"baseline_repetition_count\":%" PRIhsz,
      baseline->sample_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"candidate_repetition_count\":%" PRIhsz,
      candidate->sample_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"baseline_p50_ns\":%" PRIi64, baseline_p50.p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"candidate_p50_ns\":%" PRIi64, candidate_p50.p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"baseline_p90_ns\":%" PRIi64, baseline_p90.p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"candidate_p90_ns\":%" PRIi64, candidate_p90.p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"baseline_p50_spread_ppm\":%" PRIu64,
      baseline_p50.p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"candidate_p50_spread_ppm\":%" PRIu64,
      candidate_p50.p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"baseline_p90_spread_ppm\":%" PRIu64,
      baseline_p90.p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"candidate_p90_spread_ppm\":%" PRIu64,
      candidate_p90.p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"ratio_p50\":%.6f", ratio_p50));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"speedup_p50\":%.6f", speedup_p50));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"ratio_p90\":%.6f", ratio_p90));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"speedup_p90\":%.6f", speedup_p90));
  return loom_output_stream_write_cstring(&stream, "}\n");
}

static iree_status_t iree_tune_loom_dispatch_comparison_sample_window(
    const loom_testbench_case_plan_t* case_plan, bool has_specialization_sample,
    iree_host_size_t specialization_sample, iree_host_size_t* out_begin_sample,
    iree_host_size_t* out_end_sample) {
  if (has_specialization_sample) {
    *out_begin_sample = specialization_sample;
    *out_end_sample = specialization_sample + 1;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_tune_loom_validate_sample_flag(
      case_plan->sample_count, out_begin_sample, out_end_sample));
  if (*out_end_sample != *out_begin_sample + 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "interleaved comparisons require exactly one concrete shape sample per "
        "candidate; use --sample= to select one of %" PRIhsz " samples",
        case_plan->sample_count);
  }
  return iree_ok_status();
}

static bool iree_tune_loom_sample_attr_equal(loom_attribute_t lhs,
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

static iree_status_t iree_tune_loom_validate_comparison_shape(
    const loom_module_t* module,
    const iree_tune_loom_dispatch_comparison_candidate_t* baseline,
    const iree_tune_loom_dispatch_comparison_candidate_t* candidate) {
  const loom_testbench_case_plan_t* baseline_case =
      baseline->selection->case_plan;
  const loom_testbench_case_plan_t* candidate_case =
      candidate->selection->case_plan;
  const iree_string_view_t baseline_name =
      baseline->selection->benchmark_plan->name;
  const iree_string_view_t candidate_name =
      candidate->selection->benchmark_plan->name;
  if (baseline_case->parameter_count != candidate_case->parameter_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "comparison candidate `%.*s` has %" PRIhsz
        " shape parameters, but baseline `%.*s` has %" PRIhsz,
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
    const iree_string_view_t baseline_parameter_name =
        iree_tune_loom_value_name(module, baseline_parameter->value_id);
    const iree_string_view_t candidate_parameter_name =
        iree_tune_loom_value_name(module, candidate_parameter->value_id);
    if (!iree_string_view_equal(baseline_parameter_name,
                                candidate_parameter_name)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "comparison candidate `%.*s` shape parameter %" PRIhsz
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
          "comparison candidate `%.*s` shape parameter `%.*s` does not match "
          "baseline `%.*s` parameter kind/type",
          (int)candidate_name.size, candidate_name.data,
          (int)baseline_parameter_name.size, baseline_parameter_name.data,
          (int)baseline_name.size, baseline_name.data);
    }

    const iree_host_size_t baseline_parameter_sample =
        loom_testbench_case_sample_parameter_ordinal(
            baseline_case, baseline->begin_sample, parameter_index);
    const iree_host_size_t candidate_parameter_sample =
        loom_testbench_case_sample_parameter_ordinal(
            candidate_case, candidate->begin_sample, parameter_index);
    loom_attribute_t baseline_value = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
        baseline_parameter, baseline_parameter_sample, &baseline_value));
    loom_attribute_t candidate_value = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
        candidate_parameter, candidate_parameter_sample, &candidate_value));
    if (!iree_tune_loom_sample_attr_equal(baseline_value, candidate_value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "comparison candidate `%.*s` shape parameter `%.*s` sample differs "
          "from baseline `%.*s`; use --sample= to select matching shapes",
          (int)candidate_name.size, candidate_name.data,
          (int)baseline_parameter_name.size, baseline_parameter_name.data,
          (int)baseline_name.size, baseline_name.data);
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_tune_loom_validate_comparison_shapes(
    const loom_module_t* module,
    const iree_tune_loom_dispatch_comparison_candidate_t* candidates,
    iree_host_size_t candidate_count) {
  if (candidate_count < 2) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 1; i < candidate_count; ++i) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_validate_comparison_shape(
        module, &candidates[0], &candidates[i]));
  }
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_prepare_dispatch_comparison_candidate(
    const iree_tune_loom_run_identity_t* run,
    iree_tune_loom_dispatch_comparison_candidate_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    iree_tune_loom_hal_context_t* hal_context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_tune_loom_device_row_state_t* device_row_state,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    iree_tune_loom_jsonl_sink_t* jsonl_sink,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const iree_tune_loom_selected_benchmark_t* selection = candidate->selection;
  const loom_testbench_invocation_plan_t* actual_invocation = NULL;
  loom_testbench_case_execution_options_t candidate_execution_options =
      *execution_options;

  iree_status_t status = loom_run_hal_testbench_select_actual_invocation(
      selection->case_plan, &actual_invocation);
  if (iree_status_is_ok(status)) {
    status =
        loom_run_hal_testbench_context_ensure_runtime(&hal_context->execution);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_hal_actual_provider_initialize(
        hal_context, session, filename, source,
        iree_make_cstring_view(FLAG_pipeline), module_plan->module,
        actual_invocation, candidate->specialization, selection->case_plan,
        candidate->specialization_sample_ordinal,
        candidate->has_specialization_sample_ordinal, compile_report_options,
        &candidate->provider);
  }
  if (iree_status_is_ok(status)) {
    candidate->provider_initialized = true;
    candidate_execution_options.materializer.device =
        hal_context->execution.runtime.device;
    candidate_execution_options.materializer.device_allocator =
        iree_hal_device_allocator(hal_context->execution.runtime.device);
    candidate_execution_options.materializer.buffer_params =
        loom_run_hal_testbench_host_visible_buffer_params();
    candidate_execution_options.invocation.invoke_actual =
        (loom_testbench_invocation_callback_t){
            .fn = loom_run_hal_testbench_actual_invoke,
            .user_data = &candidate->provider.execution,
        };
    candidate->benchmark_materializer =
        candidate_execution_options.materializer;
    candidate->benchmark_materializer.buffer_params =
        (iree_hal_buffer_params_t){0};
    status = iree_tune_loom_hal_actual_provider_compile(&candidate->provider);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_write_compiled_artifacts(
        run, &selection->identity, &candidate->provider, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_write_compile_report_artifact(
        run, &selection->identity, selection->benchmark_plan,
        selection->case_plan, &candidate->provider, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_jsonl_sink_end(
        jsonl_sink, iree_tune_loom_append_device_row(
                        run, hal_context, device_row_state,
                        iree_tune_loom_jsonl_sink_begin(jsonl_sink)));
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_jsonl_sink_end(
        jsonl_sink, iree_tune_loom_append_compile_row(
                        run, &selection->identity, selection->benchmark_plan,
                        selection->case_plan, &candidate->provider,
                        iree_tune_loom_jsonl_sink_begin(jsonl_sink)));
  }
  if (iree_status_is_ok(status) &&
      candidate->provider.execution.compile_rejected) {
    iree_tune_loom_benchmark_result_t benchmark_result = {0};
    iree_tune_loom_benchmark_result_set_compile_rejection(&candidate->provider,
                                                          &benchmark_result);
    ++*inout_failed_benchmark_count;
    return iree_tune_loom_append_dispatch_benchmark_result(
        run, &selection->identity, module_plan, selection->benchmark_plan,
        selection->case_plan, &selection->policy, &benchmark_result,
        /*correctness_sample_count=*/0,
        /*correctness_failed_sample_count=*/0, jsonl_sink, allocator);
  }

  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_run_case_correctness_range(
        run, &selection->identity, module_plan, selection->benchmark_plan,
        selection->benchmark_plan->case_index, &candidate_execution_options,
        candidate->specialization, candidate->begin_sample,
        candidate->end_sample, execution_arena, jsonl_sink,
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
    iree_tune_loom_benchmark_result_t benchmark_result = {
        .executed = false,
        .passed = false,
        .compile_report_artifact_path =
            candidate->provider.compile_report_artifact_path,
        .target_artifact_path = candidate->provider.target_artifact_path,
        .target_listing_path = candidate->provider.target_listing_path,
        .hal_executable_path = candidate->provider.hal_executable_path,
        .specialization = candidate->specialization,
        .has_sample_ordinal = true,
        .sample_ordinal = candidate->begin_sample,
        .samples_per_iteration = candidate->correctness_sample_count,
        .failed_sample_count = candidate->correctness_failed_sample_count,
    };
    ++*inout_failed_benchmark_count;
    return iree_tune_loom_append_dispatch_benchmark_result(
        run, &selection->identity, module_plan, selection->benchmark_plan,
        selection->case_plan, &selection->policy, &benchmark_result,
        candidate->correctness_sample_count,
        candidate->correctness_failed_sample_count, jsonl_sink, allocator);
  }
  if (iree_status_is_ok(status)) {
    candidate->runnable = true;
  }
  return status;
}

static iree_host_size_t iree_tune_loom_comparison_sample_capacity(
    iree_tune_loom_interleave_mode_t interleave_mode,
    iree_host_size_t candidate_count, iree_host_size_t candidate_index,
    iree_host_size_t repetitions) {
  if (interleave_mode == IREE_TUNE_LOOM_INTERLEAVE_ABABA) {
    return candidate_index == 0 ? (candidate_count - 1) * (repetitions + 1)
                                : repetitions;
  }
  return repetitions;
}

static iree_status_t iree_tune_loom_initialize_dispatch_comparison_candidates(
    const iree_tune_loom_selected_benchmark_t* selections,
    iree_host_size_t selection_count,
    iree_tune_loom_shape_specialization_mode_t shape_specialization_mode,
    iree_tune_loom_interleave_mode_t interleave_mode,
    iree_host_size_t repetitions, iree_allocator_t allocator,
    iree_tune_loom_dispatch_comparison_candidate_t** out_candidates) {
  *out_candidates = NULL;
  if (shape_specialization_mode == IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_BOTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "interleaved comparisons require one specialization mode; use "
        "--shape_specialization=dynamic or --shape_specialization=per_sample");
  }
  if (interleave_mode == IREE_TUNE_LOOM_INTERLEAVE_ABABA &&
      selection_count != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ABABA comparison requires exactly two selected "
                            "benchmarks; use --interleave=round_robin for "
                            "ABCD-style comparisons");
  }
  if (repetitions == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "interleaved comparison repetitions must be "
                            "positive");
  }
  if (selection_count > 26) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "interleaved comparison supports at most 26 "
                            "selected benchmarks; got %" PRIhsz,
                            selection_count);
  }

  iree_tune_loom_dispatch_comparison_candidate_t* candidates = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, selection_count, sizeof(*candidates), (void**)&candidates));
  memset(candidates, 0, selection_count * sizeof(*candidates));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < selection_count;
       ++i) {
    if (selections[i].policy.measure_kind !=
        IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "interleaved comparison benchmark `%.*s` must use "
                           "measure = \"dispatch_complete\"",
                           (int)selections[i].benchmark_plan->name.size,
                           selections[i].benchmark_plan->name.data);
      break;
    }
    candidates[i].selection = &selections[i];
    candidates[i].specialization =
        iree_tune_loom_shape_specialization_mode_name(
            shape_specialization_mode);
    if (shape_specialization_mode ==
        IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_PER_SAMPLE) {
      iree_host_size_t begin_sample = 0;
      iree_host_size_t end_sample = 0;
      status = iree_tune_loom_validate_sample_flag(
          selections[i].case_plan->sample_count, &begin_sample, &end_sample);
      if (iree_status_is_ok(status) && end_sample != begin_sample + 1) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "per-sample interleaved comparison benchmark `%.*s` has %" PRIhsz
            " selected samples; use --sample= to select one shape",
            (int)selections[i].benchmark_plan->name.size,
            selections[i].benchmark_plan->name.data,
            selections[i].case_plan->sample_count);
      }
      candidates[i].has_specialization_sample_ordinal = true;
      candidates[i].specialization_sample_ordinal = begin_sample;
    }
    if (iree_status_is_ok(status)) {
      status = iree_tune_loom_dispatch_comparison_sample_window(
          selections[i].case_plan,
          candidates[i].has_specialization_sample_ordinal,
          candidates[i].specialization_sample_ordinal,
          &candidates[i].begin_sample, &candidates[i].end_sample);
    }
    const iree_host_size_t sample_capacity =
        iree_tune_loom_comparison_sample_capacity(
            interleave_mode, selection_count, i, repetitions);
    candidates[i].sample_capacity = sample_capacity;
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(allocator, sample_capacity,
                                           sizeof(*candidates[i].p50_samples),
                                           (void**)&candidates[i].p50_samples);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc_array(allocator, sample_capacity,
                                           sizeof(*candidates[i].p90_samples),
                                           (void**)&candidates[i].p90_samples);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_candidates = candidates;
  } else {
    iree_tune_loom_dispatch_comparison_candidates_deinitialize(
        candidates, selection_count, allocator);
  }
  return status;
}

static iree_status_t iree_tune_loom_run_comparison_window(
    const iree_tune_loom_run_identity_t* run,
    iree_tune_loom_dispatch_comparison_candidate_t* candidate,
    const iree_tune_loom_candidate_identity_t* baseline,
    const loom_testbench_module_plan_t* module_plan,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_host_size_t order_index, iree_host_size_t repetition_index,
    char schedule_token, iree_allocator_t allocator,
    iree_tune_loom_jsonl_sink_t* jsonl_sink,
    iree_host_size_t* inout_failed_benchmark_count) {
  if (!candidate->runnable) {
    return iree_ok_status();
  }
  iree_tune_loom_benchmark_policy_t measurement_policy =
      candidate->selection->policy;
  const bool profile_suppressed =
      iree_any_bit_set(measurement_policy.hal_options.flags,
                       LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH);
  measurement_policy.hal_options.flags &=
      ~LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH;

  iree_tune_loom_benchmark_result_t benchmark_result = {0};
  iree_status_t status = iree_tune_loom_run_hal_benchmark_sample(
      run, &candidate->selection->identity, module_plan,
      candidate->selection->case_plan, &measurement_policy,
      &candidate->provider, &candidate->benchmark_materializer,
      candidate->begin_sample, allocator, &benchmark_result);
  if (iree_status_is_ok(status) &&
      (!benchmark_result.executed || !benchmark_result.passed)) {
    ++*inout_failed_benchmark_count;
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_comparison_candidate_record_timing(
        candidate, &benchmark_result);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_jsonl_sink_end(
        jsonl_sink,
        iree_tune_loom_append_benchmark_repetition_row(
            run, candidate, baseline, comparison_group, method, order_index,
            repetition_index, schedule_token, profile_suppressed,
            &benchmark_result, iree_tune_loom_jsonl_sink_begin(jsonl_sink)));
  }
  return status;
}

static iree_status_t iree_tune_loom_run_dispatch_comparison(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_selected_benchmark_t* selections,
    iree_host_size_t selection_count,
    const loom_testbench_module_plan_t* module_plan,
    iree_tune_loom_shape_specialization_mode_t shape_specialization_mode,
    iree_tune_loom_interleave_mode_t interleave_mode,
    iree_host_size_t repetitions, iree_tune_loom_hal_context_t* hal_context,
    loom_run_session_t* session, iree_string_view_t filename,
    iree_string_view_t source,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_tune_loom_device_row_state_t* device_row_state,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    iree_tune_loom_jsonl_sink_t* jsonl_sink,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  iree_tune_loom_dispatch_comparison_candidate_t* candidates = NULL;
  iree_status_t status =
      iree_tune_loom_initialize_dispatch_comparison_candidates(
          selections, selection_count, shape_specialization_mode,
          interleave_mode, repetitions, allocator, &candidates);
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_validate_comparison_shapes(
        module_plan->module, candidates, selection_count);
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < selection_count;
       ++i) {
    status = iree_tune_loom_prepare_dispatch_comparison_candidate(
        run, &candidates[i], module_plan, hal_context, session, filename,
        source, compile_report_options, execution_options, device_row_state,
        execution_arena, allocator, jsonl_sink, inout_correctness_sample_count,
        inout_correctness_failed_sample_count, inout_failed_benchmark_count);
  }

  const iree_tune_loom_candidate_identity_t* baseline = &selections[0].identity;
  const iree_string_view_t comparison_group =
      selections[0].benchmark_plan->name;
  const iree_string_view_t method =
      iree_tune_loom_interleave_mode_name(interleave_mode);
  iree_host_size_t order_index = 0;
  if (iree_status_is_ok(status) &&
      interleave_mode == IREE_TUNE_LOOM_INTERLEAVE_ABABA) {
    for (iree_host_size_t candidate_index = 1;
         iree_status_is_ok(status) && candidate_index < selection_count;
         ++candidate_index) {
      status = iree_tune_loom_run_comparison_window(
          run, &candidates[0], baseline, module_plan, comparison_group, method,
          order_index++, /*repetition_index=*/0, 'A', allocator, jsonl_sink,
          inout_failed_benchmark_count);
      for (iree_host_size_t repetition_index = 0;
           iree_status_is_ok(status) && repetition_index < repetitions;
           ++repetition_index) {
        status = iree_tune_loom_run_comparison_window(
            run, &candidates[candidate_index], baseline, module_plan,
            comparison_group, method, order_index++, repetition_index,
            (char)('A' + candidate_index), allocator, jsonl_sink,
            inout_failed_benchmark_count);
        if (iree_status_is_ok(status)) {
          status = iree_tune_loom_run_comparison_window(
              run, &candidates[0], baseline, module_plan, comparison_group,
              method, order_index++, repetition_index + 1, 'A', allocator,
              jsonl_sink, inout_failed_benchmark_count);
        }
      }
    }
  } else if (iree_status_is_ok(status) &&
             interleave_mode == IREE_TUNE_LOOM_INTERLEAVE_ROUND_ROBIN) {
    for (iree_host_size_t repetition_index = 0;
         iree_status_is_ok(status) && repetition_index < repetitions;
         ++repetition_index) {
      for (iree_host_size_t candidate_index = 0;
           iree_status_is_ok(status) && candidate_index < selection_count;
           ++candidate_index) {
        status = iree_tune_loom_run_comparison_window(
            run, &candidates[candidate_index], baseline, module_plan,
            comparison_group, method, order_index++, repetition_index,
            (char)('A' + candidate_index), allocator, jsonl_sink,
            inout_failed_benchmark_count);
      }
    }
  }

  for (iree_host_size_t candidate_index = 1;
       iree_status_is_ok(status) && candidate_index < selection_count;
       ++candidate_index) {
    status = iree_tune_loom_jsonl_sink_end(
        jsonl_sink,
        iree_tune_loom_append_comparison_row(
            run, &candidates[0], &candidates[candidate_index], comparison_group,
            method, iree_tune_loom_jsonl_sink_begin(jsonl_sink)));
  }

  iree_tune_loom_dispatch_comparison_candidates_deinitialize(
      candidates, selection_count, allocator);
  return status;
}

int iree_tune_loom_main(int argc, char** argv,
                        const iree_tune_loom_configuration_t* configuration) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = iree_allocator_system();
  iree_string_builder_t command_line_json;
  iree_string_builder_initialize(allocator, &command_line_json);
  iree_status_t status =
      iree_tune_loom_append_command_line_json(argc, argv, &command_line_json);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    iree_string_builder_deinitialize(&command_line_json);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(1);
    return 1;
  }

  iree_tune_loom_set_usage(configuration->tool_name);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (FLAG_agents_md) {
    iree_tune_loom_print_agents_md(stdout);
    iree_string_builder_deinitialize(&command_line_json);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(0);
    return 0;
  }

  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  iree_tune_loom_artifact_bundle_t artifact_bundle = {0};
  iree_tune_loom_file_provider_t file_provider = {0};
  iree_tune_loom_hal_context_t hal_context = {0};
  iree_tune_loom_hal_context_initialize(configuration, allocator, &hal_context);
  iree_arena_allocator_t plan_arena;
  memset(&plan_arena, 0, sizeof(plan_arena));
  iree_arena_allocator_t execution_arena;
  memset(&execution_arena, 0, sizeof(execution_arena));
  iree_tune_loom_jsonl_sink_t jsonl_sink = {0};
  bool jsonl_sink_initialized = false;
  iree_tune_loom_diagnostic_capture_t source_diagnostics = {0};
  iree_tune_loom_diagnostic_capture_initialize(allocator, &source_diagnostics);
  iree_tune_loom_device_row_state_t device_row_state = {0};
  iree_host_size_t planned_case_count = 0;
  iree_host_size_t planned_benchmark_count = 0;
  iree_host_size_t selected_benchmark_count = 0;
  iree_host_size_t failure_count = 0;
  iree_host_size_t failed_benchmark_count = 0;
  iree_host_size_t correctness_sample_count = 0;
  iree_host_size_t correctness_failed_sample_count = 0;
  int exit_code = 0;

  status = iree_ok_status();
  iree_tune_loom_shape_specialization_mode_t shape_specialization_mode =
      IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_DYNAMIC;
  status = iree_tune_loom_parse_shape_specialization_mode(
      iree_make_cstring_view(FLAG_shape_specialization),
      &shape_specialization_mode);
  const iree_string_view_t compare_list =
      iree_string_view_trim(iree_make_cstring_view(FLAG_compare));
  const bool compare_requested = !iree_string_view_is_empty(compare_list);
  iree_tune_loom_interleave_mode_t interleave_mode =
      IREE_TUNE_LOOM_INTERLEAVE_NONE;
  if (iree_status_is_ok(status) && compare_requested) {
    status = iree_tune_loom_parse_interleave_mode(
        iree_make_cstring_view(FLAG_interleave), &interleave_mode);
  }
  if (iree_status_is_ok(status) && compare_requested && FLAG_repetitions <= 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--repetitions must be positive; got %d",
                              (int)FLAG_repetitions);
  }
  if (iree_status_is_ok(status) && compare_requested &&
      strlen(FLAG_benchmark) != 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compare selects benchmarks directly and cannot be combined with "
        "--benchmark");
  }
  if (iree_status_is_ok(status) && compare_requested &&
      strlen(FLAG_case) != 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compare selects benchmark/case pairs directly and cannot be "
        "combined with --case");
  }
  if (iree_status_is_ok(status) && compare_requested &&
      shape_specialization_mode == IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_BOTH) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compare requires one specialization mode; use "
        "--shape_specialization=dynamic or --shape_specialization=per_sample");
  }
  if (argc > 2) {
    status = iree_status_join(
        status,
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "iree-tune-loom accepts at most one input file or '-' "
                         "for stdin; got %d inputs",
                         argc - 1));
  }
  if (iree_status_is_ok(status) && FLAG_max_samples_per_case <= 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--max_samples_per_case must be positive; got "
                              "%d",
                              (int)FLAG_max_samples_per_case);
  }
  if (iree_status_is_ok(status) && FLAG_iterations.value <= 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--iterations must be positive; got %d",
                              (int)FLAG_iterations.value);
  }
  if (iree_status_is_ok(status) && FLAG_warmup_iterations.value < 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--warmup_iterations must be non-negative; got "
                              "%d",
                              (int)FLAG_warmup_iterations.value);
  }
  if (iree_status_is_ok(status) && FLAG_batch_size.value <= 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--batch_size must be positive; got %d",
                              (int)FLAG_batch_size.value);
  }
  if (iree_status_is_ok(status) && FLAG_min_time_ms.value < 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--min_time_ms must be non-negative; got %d",
                              (int)FLAG_min_time_ms.value);
  }
  if (iree_status_is_ok(status) && FLAG_warmup_time_ms.value < 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--warmup_time_ms must be non-negative; got %d",
                              (int)FLAG_warmup_time_ms.value);
  }
  if (iree_status_is_ok(status) && FLAG_max_batches.value <= 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--max_batches must be positive; got %d",
                              (int)FLAG_max_batches.value);
  }
  if (iree_status_is_ok(status) && FLAG_stable_p90_to_p50_ppm.value < 0) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "--stable_p90_to_p50_ppm must be non-negative; got %d",
                         (int)FLAG_stable_p90_to_p50_ppm.value);
  }
  if (iree_status_is_ok(status) && FLAG_input_ring_min_bytes < 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--input_ring_min_bytes must be non-negative; got %" PRIi64,
        FLAG_input_ring_min_bytes);
  }
  if (iree_status_is_ok(status) && FLAG_input_ring_count < 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--input_ring_count must be non-negative; got %d",
                              (int)FLAG_input_ring_count);
  }
  if (iree_status_is_ok(status) &&
      iree_string_view_equal(
          iree_string_view_trim(iree_make_cstring_view(FLAG_file_output_dir)),
          IREE_SV("-"))) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--file_output_dir must name a directory; '-' is reserved for stdout");
  }
  loom_run_compile_report_capture_options_t compile_report_options = {0};
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_compile_report_options_initialize(
        &compile_report_options);
  }
  if (iree_status_is_ok(status)) {
    iree_tune_loom_artifact_bundle_options_t artifact_bundle_options = {
        .dir = iree_string_view_trim(
            iree_make_cstring_view(FLAG_artifact_bundle_dir)),
    };
    status = iree_tune_loom_parse_artifact_bundle_policy(
        iree_make_cstring_view(FLAG_artifact_bundle_policy),
        &artifact_bundle_options.policy);
    if (iree_status_is_ok(status)) {
      status = iree_tune_loom_artifact_bundle_initialize(
          &artifact_bundle_options, allocator, &artifact_bundle);
      hal_context.artifact_bundle = &artifact_bundle;
    }
  }

  if (iree_status_is_ok(status)) {
    loom_run_session_options_t session_options = {0};
    loom_run_session_options_initialize(&session_options);
    session_options.host_allocator = allocator;
    session_options.register_context = (loom_run_register_context_callback_t){
        .fn = iree_tune_loom_register_context,
        .user_data = (void*)configuration,
    };
    session_options.initialize_low_descriptor_registry =
        configuration->initialize_low_descriptor_registry;
    status = loom_run_session_initialize(&session_options, &session);
  }
  const iree_string_view_t input_path =
      argc < 2 ? iree_string_view_empty() : iree_make_cstring_view(argv[1]);
  const iree_string_view_t filename =
      (argc < 2 || iree_string_view_equal(input_path, IREE_SV("-")))
          ? IREE_SV("<stdin>")
          : input_path;
  char run_id_storage[32];
  snprintf(run_id_storage, sizeof(run_id_storage), "r%016" PRIx64,
           (uint64_t)iree_time_now());
  iree_string_view_t run_id = iree_make_cstring_view(run_id_storage);
  const iree_string_view_t results_output_path =
      iree_tune_loom_effective_results_output_path(
          iree_make_cstring_view(FLAG_output), &artifact_bundle);
  const iree_string_view_t profile_artifacts_dir =
      iree_tune_loom_effective_profile_artifacts_dir(
          iree_make_cstring_view(FLAG_profile_artifacts_dir), &artifact_bundle);
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_file_provider_initialize(
        filename, run_id, iree_make_cstring_view(FLAG_file_output_dir),
        artifact_bundle.file_output_dir, &artifact_bundle, allocator,
        &file_provider);
  }
  const iree_tune_loom_run_identity_t run_identity = {
      .run_id = run_id,
      .source = filename,
      .results_path = iree_string_view_is_empty(results_output_path)
                          ? IREE_SV("-")
                          : results_output_path,
      .file_output_dir = file_provider.output_dir,
      .profile_artifacts_dir = profile_artifacts_dir,
      .artifact_bundle_dir = artifact_bundle.dir,
      .artifact_bundle_policy =
          iree_tune_loom_artifact_bundle_policy_name(artifact_bundle.policy),
  };
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_jsonl_sink_initialize(results_output_path,
                                                  allocator, &jsonl_sink);
    if (iree_status_is_ok(status)) {
      jsonl_sink_initialized = true;
      status = iree_tune_loom_jsonl_sink_end(
          &jsonl_sink,
          iree_tune_loom_append_run_row(
              &run_identity, FLAG_dry_run, shape_specialization_mode,
              iree_tune_loom_jsonl_sink_begin(&jsonl_sink)));
    }
  }
  iree_string_view_t source = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = loom_tooling_read_input_file(input_path, allocator, &contents);
    if (iree_status_is_ok(status)) {
      source = loom_tooling_file_contents_string_view(contents);
    }
  }
  if (iree_status_is_ok(status)) {
    loom_run_module_parse_options_t parse_options = {0};
    loom_run_module_parse_options_initialize(&parse_options);
    parse_options.filename = filename;
    parse_options.source = source;
    parse_options.diagnostic_sink = (loom_diagnostic_sink_t){
        .fn = iree_tune_loom_diagnostic_capture_sink,
        .user_data = &source_diagnostics,
    };
    status = loom_run_module_parse(&session, &parse_options, &run_module);
    if (!iree_status_is_ok(status) && source_diagnostics.error_count != 0) {
      // The diagnostic sink owns the input rejection evidence; the status only
      // carries the same non-infrastructure failure.
      iree_status_free(status);
      status = iree_tune_loom_jsonl_sink_end(
          &jsonl_sink,
          iree_tune_loom_append_failure_row(
              &run_identity, IREE_SV("parse"), IREE_SV("diagnostics"),
              IREE_SV("input module has parse errors"), &source_diagnostics,
              iree_tune_loom_jsonl_sink_begin(&jsonl_sink)));
      ++failure_count;
      exit_code = 1;
    }
  }

  if (iree_status_is_ok(status) && failure_count == 0) {
    iree_tune_loom_diagnostic_capture_deinitialize(&source_diagnostics);
    iree_tune_loom_diagnostic_capture_initialize(allocator,
                                                 &source_diagnostics);
    loom_verify_options_t verify_options = {0};
    verify_options.sink = (loom_diagnostic_sink_t){
        .fn = iree_tune_loom_diagnostic_capture_sink,
        .user_data = &source_diagnostics,
    };
    verify_options.max_errors = 20;
    verify_options.source_resolver =
        loom_run_module_source_resolver(&run_module);
    loom_verify_result_t verify_result = {0};
    status =
        loom_verify_module(run_module.module, &verify_options, &verify_result);
    if (iree_status_is_ok(status) && verify_result.error_count != 0) {
      status = iree_tune_loom_jsonl_sink_end(
          &jsonl_sink,
          iree_tune_loom_append_failure_row(
              &run_identity, IREE_SV("verify"), IREE_SV("diagnostics"),
              IREE_SV("input module failed verification"), &source_diagnostics,
              iree_tune_loom_jsonl_sink_begin(&jsonl_sink)));
      ++failure_count;
      exit_code = 1;
    }
  }

  if (iree_status_is_ok(status) && failure_count == 0) {
    iree_arena_initialize(loom_run_session_block_pool(&session), &plan_arena);
    iree_arena_initialize(loom_run_session_block_pool(&session),
                          &execution_arena);
    loom_testbench_plan_options_t plan_options = {0};
    loom_testbench_plan_options_initialize(&plan_options);
    plan_options.max_samples_per_case =
        (iree_host_size_t)FLAG_max_samples_per_case;
    loom_testbench_module_plan_t module_plan = {0};
    status = loom_testbench_plan_module(run_module.module, &plan_options,
                                        &plan_arena, &module_plan);
    if (iree_status_is_ok(status)) {
      planned_case_count = module_plan.case_count;
      planned_benchmark_count = module_plan.benchmark_count;
    }
    const iree_string_view_t selected_case_name =
        iree_tune_loom_normalize_symbol_name(iree_make_cstring_view(FLAG_case));
    const iree_string_view_t selected_benchmark_name =
        iree_tune_loom_normalize_symbol_name(
            iree_make_cstring_view(FLAG_benchmark));
    loom_testbench_case_execution_options_t execution_options = {0};
    loom_testbench_case_execution_options_initialize(&execution_options);
    execution_options.materializer.host_allocator = allocator;
    execution_options.materializer.open_read_file =
        (loom_testbench_file_open_callback_t){
            .fn = iree_tune_loom_open_file_for_read,
            .user_data = &file_provider,
        };
    execution_options.materializer.open_write_file =
        (loom_testbench_file_open_callback_t){
            .fn = iree_tune_loom_open_file_for_write,
            .user_data = &file_provider,
        };

    if (iree_status_is_ok(status) && compare_requested) {
      iree_tune_loom_selected_benchmark_t* selections = NULL;
      iree_host_size_t selection_count = 0;
      status = iree_tune_loom_select_compare_benchmarks(
          &module_plan, compare_list, allocator, &selections, &selection_count);
      if (iree_status_is_ok(status)) {
        selected_benchmark_count = selection_count;
      }
      for (iree_host_size_t i = 0; iree_status_is_ok(status) &&
                                   selections != NULL && i < selection_count;
           ++i) {
        if (selections[i].policy.measure_kind !=
            IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE) {
          status =
              iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                               "--compare benchmark `%.*s` must use measure = "
                               "\"dispatch_complete\"",
                               (int)selections[i].benchmark_plan->name.size,
                               selections[i].benchmark_plan->name.data);
          break;
        }
        status = iree_tune_loom_jsonl_sink_end(
            &jsonl_sink,
            iree_tune_loom_append_plan_row(
                &run_identity, &selections[i].identity, module_plan.module,
                selections[i].benchmark_plan, selections[i].case_plan,
                &selections[i].policy, shape_specialization_mode, allocator,
                iree_tune_loom_jsonl_sink_begin(&jsonl_sink)));
        if (iree_status_is_ok(status) && FLAG_dry_run) {
          status = iree_tune_loom_jsonl_sink_end(
              &jsonl_sink, iree_tune_loom_append_device_row(
                               &run_identity, &hal_context, &device_row_state,
                               iree_tune_loom_jsonl_sink_begin(&jsonl_sink)));
        }
      }
      if (iree_status_is_ok(status) && !FLAG_dry_run) {
        status = iree_tune_loom_run_dispatch_comparison(
            &run_identity, selections, selection_count, &module_plan,
            shape_specialization_mode, interleave_mode,
            (iree_host_size_t)FLAG_repetitions, &hal_context, &session,
            filename, source, &compile_report_options, &execution_options,
            &device_row_state, &execution_arena, allocator, &jsonl_sink,
            &correctness_sample_count, &correctness_failed_sample_count,
            &failed_benchmark_count);
      }
      iree_allocator_free(allocator, selections);
    }

    for (iree_host_size_t benchmark_index = 0;
         iree_status_is_ok(status) && !compare_requested &&
         benchmark_index < module_plan.benchmark_count;
         ++benchmark_index) {
      const loom_testbench_benchmark_plan_t* benchmark_plan =
          &module_plan.benchmarks[benchmark_index];
      if (!iree_tune_loom_benchmark_matches_selection(
              benchmark_plan, selected_benchmark_name)) {
        continue;
      }
      if (benchmark_plan->case_index >= module_plan.case_count) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "benchmark `%.*s` does not reference a planned check.case",
            (int)benchmark_plan->name.size, benchmark_plan->name.data);
        break;
      }
      const loom_testbench_case_plan_t* case_plan =
          &module_plan.cases[benchmark_plan->case_index];
      if (!iree_tune_loom_case_matches_selection(case_plan,
                                                 selected_case_name)) {
        continue;
      }

      const iree_host_size_t candidate_index = selected_benchmark_count;
      char candidate_id_storage[32];
      snprintf(candidate_id_storage, sizeof(candidate_id_storage), "c%" PRIhsz,
               candidate_index);
      const iree_tune_loom_candidate_identity_t candidate_identity = {
          .candidate_id = iree_make_cstring_view(candidate_id_storage),
          .candidate_index = candidate_index,
      };
      ++selected_benchmark_count;
      iree_tune_loom_benchmark_policy_t policy = {0};
      status = iree_tune_loom_policy_from_benchmark(&module_plan,
                                                    benchmark_plan, &policy);
      if (iree_status_is_ok(status)) {
        status = iree_tune_loom_jsonl_sink_end(
            &jsonl_sink,
            iree_tune_loom_append_plan_row(
                &run_identity, &candidate_identity, module_plan.module,
                benchmark_plan, case_plan, &policy, shape_specialization_mode,
                allocator, iree_tune_loom_jsonl_sink_begin(&jsonl_sink)));
      }
      if (iree_status_is_ok(status) && FLAG_dry_run &&
          policy.measure_kind == IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE) {
        status = iree_tune_loom_jsonl_sink_end(
            &jsonl_sink, iree_tune_loom_append_device_row(
                             &run_identity, &hal_context, &device_row_state,
                             iree_tune_loom_jsonl_sink_begin(&jsonl_sink)));
      }
      if (iree_status_is_ok(status) && FLAG_dry_run) {
        continue;
      }

      if (iree_status_is_ok(status) &&
          policy.measure_kind == IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE) {
        if (iree_tune_loom_shape_specialization_runs_dynamic(
                shape_specialization_mode)) {
          status = iree_tune_loom_run_dispatch_complete_benchmark(
              &run_identity, &candidate_identity, &module_plan, benchmark_plan,
              case_plan, &policy, &hal_context, &session, filename, source,
              &compile_report_options, &execution_options, IREE_SV("dynamic"),
              /*has_specialization_sample_ordinal=*/false,
              /*specialization_sample_ordinal=*/0, &device_row_state,
              &execution_arena, allocator, &jsonl_sink,
              &correctness_sample_count, &correctness_failed_sample_count,
              &failed_benchmark_count);
        }
        if (iree_status_is_ok(status) &&
            iree_tune_loom_shape_specialization_runs_per_sample(
                shape_specialization_mode)) {
          iree_host_size_t begin_sample = 0;
          iree_host_size_t end_sample = 0;
          status = iree_tune_loom_validate_sample_flag(
              case_plan->sample_count, &begin_sample, &end_sample);
          for (iree_host_size_t sample_ordinal = begin_sample;
               iree_status_is_ok(status) && sample_ordinal < end_sample;
               ++sample_ordinal) {
            status = iree_tune_loom_run_dispatch_complete_benchmark(
                &run_identity, &candidate_identity, &module_plan,
                benchmark_plan, case_plan, &policy, &hal_context, &session,
                filename, source, &compile_report_options, &execution_options,
                IREE_SV("per_sample"),
                /*has_specialization_sample_ordinal=*/true, sample_ordinal,
                &device_row_state, &execution_arena, allocator, &jsonl_sink,
                &correctness_sample_count, &correctness_failed_sample_count,
                &failed_benchmark_count);
          }
        }
        continue;
      }

      loom_testbench_case_execution_options_t benchmark_execution_options =
          execution_options;
      iree_host_size_t benchmark_correctness_sample_count = 0;
      iree_host_size_t benchmark_correctness_failed_sample_count = 0;
      if (iree_status_is_ok(status)) {
        status = iree_tune_loom_run_case_correctness(
            &run_identity, &candidate_identity, &module_plan, benchmark_plan,
            benchmark_plan->case_index, &benchmark_execution_options,
            iree_string_view_empty(), &execution_arena, &jsonl_sink,
            &benchmark_correctness_sample_count,
            &benchmark_correctness_failed_sample_count);
      }
      if (iree_status_is_ok(status)) {
        correctness_sample_count += benchmark_correctness_sample_count;
        correctness_failed_sample_count +=
            benchmark_correctness_failed_sample_count;
      }

      iree_tune_loom_benchmark_result_t benchmark_result = {0};
      if (iree_status_is_ok(status) &&
          benchmark_correctness_failed_sample_count == 0) {
        status = iree_tune_loom_run_benchmark_iterations(
            &module_plan, benchmark_plan->case_index,
            &benchmark_execution_options, &policy, &execution_arena, allocator,
            &benchmark_result);
      }
      if (iree_status_is_ok(status) &&
          benchmark_correctness_failed_sample_count != 0) {
        benchmark_result = (iree_tune_loom_benchmark_result_t){
            .executed = false,
            .passed = false,
            .samples_per_iteration = benchmark_correctness_sample_count,
            .failed_sample_count = benchmark_correctness_failed_sample_count,
        };
        if (FLAG_sample >= 0) {
          benchmark_result.has_sample_ordinal = true;
          benchmark_result.sample_ordinal = (iree_host_size_t)FLAG_sample;
        }
      }
      if (iree_status_is_ok(status)) {
        if (!benchmark_result.executed || !benchmark_result.passed) {
          ++failed_benchmark_count;
        }
        status = iree_tune_loom_jsonl_sink_end(
            &jsonl_sink,
            iree_tune_loom_append_benchmark_result(
                &run_identity, &candidate_identity, module_plan.module,
                benchmark_plan, case_plan, &policy, &benchmark_result,
                benchmark_correctness_sample_count,
                benchmark_correctness_failed_sample_count,
                iree_tune_loom_jsonl_sink_begin(&jsonl_sink)));
        if (iree_status_is_ok(status)) {
          status = iree_tune_loom_jsonl_sink_end(
              &jsonl_sink,
              iree_tune_loom_append_profile_row(
                  &run_identity, &candidate_identity, module_plan.module,
                  benchmark_plan, case_plan, &policy, &benchmark_result,
                  allocator, iree_tune_loom_jsonl_sink_begin(&jsonl_sink)));
        }
      }
    }
    if (iree_status_is_ok(status) && selected_benchmark_count == 0) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND, "no check.benchmark matched '%.*s'",
          (int)selected_benchmark_name.size, selected_benchmark_name.data);
    }
    if (iree_status_is_ok(status) &&
        (failed_benchmark_count != 0 || correctness_failed_sample_count != 0)) {
      exit_code = 1;
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_jsonl_sink_end(
        &jsonl_sink,
        iree_tune_loom_append_summary_row(
            &run_identity, &artifact_bundle, planned_case_count,
            planned_benchmark_count, selected_benchmark_count, failure_count,
            failed_benchmark_count, correctness_sample_count,
            correctness_failed_sample_count, FLAG_dry_run,
            shape_specialization_mode,
            iree_tune_loom_jsonl_sink_begin(&jsonl_sink)));
  }
  if (iree_status_is_ok(status) && jsonl_sink_initialized) {
    status = iree_tune_loom_jsonl_sink_close(&jsonl_sink);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_write_artifact_bundle_manifest(
        &artifact_bundle, &run_identity, &hal_context, source,
        iree_string_builder_view(&command_line_json), FLAG_dry_run,
        shape_specialization_mode, allocator);
  }
  if (iree_status_is_ok(status) && failure_count != 0) {
    exit_code = 1;
  }

  const bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = 1;
  }

  iree_tune_loom_diagnostic_capture_deinitialize(&source_diagnostics);
  if (jsonl_sink_initialized) {
    iree_tune_loom_jsonl_sink_deinitialize(&jsonl_sink);
  }
  iree_arena_deinitialize(&execution_arena);
  iree_arena_deinitialize(&plan_arena);
  loom_run_module_deinitialize(&run_module);
  iree_tune_loom_hal_context_deinitialize(&hal_context);
  iree_tune_loom_file_provider_deinitialize(&file_provider);
  iree_tune_loom_artifact_bundle_deinitialize(&artifact_bundle);
  iree_io_file_contents_free(contents);
  loom_run_session_deinitialize(&session);
  iree_string_builder_deinitialize(&command_line_json);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
