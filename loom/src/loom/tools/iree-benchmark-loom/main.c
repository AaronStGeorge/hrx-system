// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-benchmark-loom: correctness-gated benchmark runner for check.benchmark.

#include "loom/tools/iree-benchmark-loom/main.h"

#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
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
#include "loom/tools/iree-benchmark-loom/diagnostics.h"
#include "loom/tools/iree-benchmark-loom/event.h"
#include "loom/tools/iree-benchmark-loom/help.h"
#include "loom/tools/iree-benchmark-loom/manifest.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/module_query.h"
#include "loom/tools/iree-benchmark-loom/options.h"
#include "loom/tools/iree-benchmark-loom/output.h"
#include "loom/tools/iree-benchmark-loom/report.h"
#include "loom/tools/iree-benchmark-loom/testbench.h"
#include "loom/tools/iree-benchmark-loom/timing.h"
#include "loom/tools/iree-benchmark-loom/work_plan.h"
#include "loom/verify/verify.h"

IREE_FLAG(string, case, "",
          "Optional check.case symbol to benchmark, such as '@smoke'. Empty "
          "keeps all cases referenced by selected benchmarks.");
IREE_FLAG(string, benchmark, "",
          "Optional check.benchmark name to execute, such as '@smoke_time'. "
          "Empty executes all benchmarks in source order.");
IREE_FLAG(int32_t, sample, -1,
          "Optional concrete sample ordinal to execute for selected benchmark "
          "cases. Negative executes all planned samples.");
IREE_FLAG(string, measure, "case_end_to_end",
          "Measurement mode. Use 'case_end_to_end', 'end_to_end', or "
          "'dispatch_complete'.");
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
          "$TMPDIR/iree-loom-benchmark/<source>_<hash>/ for this run.");
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
IREE_FLAG(string, sample_compilation, "once",
          "Sample compilation mode for dispatch_complete benchmarks. Use "
          "'once' to compile once and pass parameter values at dispatch "
          "time, 'per_sample' to compile each selected sample with concrete "
          "parameter facts, or 'both' to emit both result sets.");
IREE_FLAG(
    int64_t, input_ring_min_bytes,
    IREE_BENCHMARK_LOOM_DEFAULT_INPUT_RING_MIN_BYTES,
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
          "Comma-separated check.benchmark names to compare in one "
          "interleaved dispatch_complete run, such as '@base,@variant'. "
          "Cannot be combined with --benchmark or --case.");
IREE_FLAG(string, interleave, "ABABA",
          "Interleaving schedule used with --compare. Use 'ABABA' for "
          "baseline-anchored pairwise windows or 'round_robin' for ABCD-style "
          "candidate rotation.");
IREE_FLAG(int32_t, repetitions, 2,
          "Interleaved comparison repetitions. With --interleave=ABABA this "
          "runs A then this many BA pairs; the default produces ABABA.");

static iree_benchmark_loom_i32_flag_t FLAG_iterations = {.value = 10};
IREE_FLAG_CALLBACK(iree_benchmark_loom_parse_i32_flag,
                   iree_benchmark_loom_print_i32_flag, &FLAG_iterations,
                   iterations, "Measured iterations.");
static iree_benchmark_loom_i32_flag_t FLAG_warmup_iterations = {.value = 1};
IREE_FLAG_CALLBACK(iree_benchmark_loom_parse_i32_flag,
                   iree_benchmark_loom_print_i32_flag, &FLAG_warmup_iterations,
                   warmup_iterations, "Warmup iterations.");
static iree_benchmark_loom_i32_flag_t FLAG_batch_size = {.value = 1};
IREE_FLAG_CALLBACK(iree_benchmark_loom_parse_i32_flag,
                   iree_benchmark_loom_print_i32_flag, &FLAG_batch_size,
                   batch_size,
                   "Number of repeated dispatches recorded into each measured "
                   "HAL command buffer batch.");
static iree_benchmark_loom_i32_flag_t FLAG_min_time_ms = {.value = 100};
IREE_FLAG_CALLBACK(iree_benchmark_loom_parse_i32_flag,
                   iree_benchmark_loom_print_i32_flag, &FLAG_min_time_ms,
                   min_time_ms,
                   "Minimum measured duration for dispatch_complete "
                   "benchmarks.");
static iree_benchmark_loom_i32_flag_t FLAG_warmup_time_ms = {.value = 0};
IREE_FLAG_CALLBACK(iree_benchmark_loom_parse_i32_flag,
                   iree_benchmark_loom_print_i32_flag, &FLAG_warmup_time_ms,
                   warmup_time_ms,
                   "Minimum warmup duration for dispatch_complete benchmarks.");
static iree_benchmark_loom_i32_flag_t FLAG_max_batches = {.value = 1000};
IREE_FLAG_CALLBACK(iree_benchmark_loom_parse_i32_flag,
                   iree_benchmark_loom_print_i32_flag, &FLAG_max_batches,
                   max_batches,
                   "Maximum measured command-buffer batches for "
                   "dispatch_complete benchmarks.");
static iree_benchmark_loom_i32_flag_t FLAG_stable_p90_to_p50_ppm = {
    .value = 100000,
};
IREE_FLAG_CALLBACK(iree_benchmark_loom_parse_i32_flag,
                   iree_benchmark_loom_print_i32_flag,
                   &FLAG_stable_p90_to_p50_ppm, stable_p90_to_p50_ppm,
                   "p90-to-p50 spread threshold in parts per million. Zero "
                   "stops after the minimum count and duration are reached.");
static iree_benchmark_loom_bool_flag_t FLAG_profile_final_batch = {.value =
                                                                       false};
IREE_FLAG_CALLBACK(iree_benchmark_loom_parse_bool_flag,
                   iree_benchmark_loom_print_bool_flag,
                   &FLAG_profile_final_batch, profile_final_batch,
                   "Runs one final profiled HAL command-buffer batch after "
                   "measured dispatch_complete timing.");

static iree_status_t iree_benchmark_loom_positive_i32_to_host_size(
    const char* flag_name, int32_t value, iree_host_size_t* out_value) {
  if (value <= 0 || (uint64_t)value > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--%s must be positive; got %d", flag_name,
                            (int)value);
  }
  *out_value = (iree_host_size_t)value;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_non_negative_i32_to_host_size(
    const char* flag_name, int32_t value, iree_host_size_t* out_value) {
  if (value < 0 || (uint64_t)value > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--%s must be non-negative; got %d", flag_name,
                            (int)value);
  }
  *out_value = (iree_host_size_t)value;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_options_from_flags(
    iree_benchmark_loom_options_t* out_options) {
  iree_benchmark_loom_options_initialize(out_options);
  out_options->selected_case = iree_benchmark_loom_normalize_selection_name(
      iree_make_cstring_view(FLAG_case));
  out_options->selected_benchmark =
      iree_benchmark_loom_normalize_selection_name(
          iree_make_cstring_view(FLAG_benchmark));
  out_options->sample_ordinal = FLAG_sample;
  out_options->pipeline = iree_make_cstring_view(FLAG_pipeline);
  out_options->output = iree_make_cstring_view(FLAG_output);
  out_options->file_output_dir = iree_make_cstring_view(FLAG_file_output_dir);
  out_options->artifact_bundle_dir =
      iree_string_view_trim(iree_make_cstring_view(FLAG_artifact_bundle_dir));
  out_options->dry_run = FLAG_dry_run;
  out_options->measure =
      iree_string_view_trim(iree_make_cstring_view(FLAG_measure));
  out_options->compile_report = iree_make_cstring_view(FLAG_compile_report);
  out_options->profile_data =
      iree_string_view_trim(iree_make_cstring_view(FLAG_profile_data));
  out_options->profile_artifacts_dir =
      iree_make_cstring_view(FLAG_profile_artifacts_dir);
  out_options->input_ring_min_bytes = FLAG_input_ring_min_bytes;
  out_options->input_ring_min_bytes_specified =
      FLAG_input_ring_min_bytes !=
      IREE_BENCHMARK_LOOM_DEFAULT_INPUT_RING_MIN_BYTES;
  out_options->compare =
      iree_string_view_trim(iree_make_cstring_view(FLAG_compare));
  out_options->profile_final_batch = FLAG_profile_final_batch.value;
  out_options->profile_final_batch_specified =
      FLAG_profile_final_batch.specified;
  out_options->iterations_specified = FLAG_iterations.specified;
  out_options->warmup_iterations_specified = FLAG_warmup_iterations.specified;
  out_options->batch_size_specified = FLAG_batch_size.specified;
  out_options->min_time_ms_specified = FLAG_min_time_ms.specified;
  out_options->warmup_time_ms_specified = FLAG_warmup_time_ms.specified;
  out_options->max_batches_specified = FLAG_max_batches.specified;
  out_options->stable_p90_to_p50_ppm_specified =
      FLAG_stable_p90_to_p50_ppm.specified;
  out_options->input_ring_count_specified = FLAG_input_ring_count != 0;

  IREE_RETURN_IF_ERROR(iree_benchmark_loom_parse_artifact_bundle_policy(
      iree_make_cstring_view(FLAG_artifact_bundle_policy),
      &out_options->artifact_bundle_policy));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_parse_sample_compilation_mode(
      iree_make_cstring_view(FLAG_sample_compilation),
      &out_options->sample_compilation_mode));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_positive_i32_to_host_size(
      "max_samples_per_case", FLAG_max_samples_per_case,
      &out_options->max_samples_per_case));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_positive_i32_to_host_size(
      "iterations", FLAG_iterations.value, &out_options->iterations));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_non_negative_i32_to_host_size(
      "warmup_iterations", FLAG_warmup_iterations.value,
      &out_options->warmup_iterations));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_positive_i32_to_host_size(
      "batch_size", FLAG_batch_size.value, &out_options->batch_size));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_positive_i32_to_host_size(
      "max_batches", FLAG_max_batches.value, &out_options->max_batches));

  if (FLAG_min_time_ms.value < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--min_time_ms must be non-negative; got %d",
                            (int)FLAG_min_time_ms.value);
  }
  out_options->min_time_ms = FLAG_min_time_ms.value;
  if (FLAG_warmup_time_ms.value < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--warmup_time_ms must be non-negative; got %d",
                            (int)FLAG_warmup_time_ms.value);
  }
  out_options->warmup_time_ms = FLAG_warmup_time_ms.value;
  if (FLAG_stable_p90_to_p50_ppm.value < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--stable_p90_to_p50_ppm must be non-negative; got %d",
        (int)FLAG_stable_p90_to_p50_ppm.value);
  }
  out_options->stable_p90_to_p50_ppm =
      (uint64_t)FLAG_stable_p90_to_p50_ppm.value;
  if (FLAG_input_ring_min_bytes < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--input_ring_min_bytes must be non-negative; got %" PRIi64,
        FLAG_input_ring_min_bytes);
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_non_negative_i32_to_host_size(
      "input_ring_count", FLAG_input_ring_count,
      &out_options->input_ring_count));
  if (FLAG_compile_report_row_limit < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compile_report_row_limit must be non-negative; got %d",
        (int)FLAG_compile_report_row_limit);
  }
  out_options->compile_report_row_limit =
      (iree_host_size_t)FLAG_compile_report_row_limit;

  const iree_flag_string_list_t profile_counters = FLAG_profile_counter_list();
  out_options->profile_counters = (iree_string_view_list_t){
      .count = profile_counters.count,
      .values = profile_counters.values,
  };
  out_options->profile_data_requested =
      !iree_string_view_is_empty(out_options->profile_data) &&
      !iree_string_view_equal(out_options->profile_data, IREE_SV("none"));

  const bool compare_requested =
      !iree_string_view_is_empty(out_options->compare);
  if (compare_requested) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_parse_interleave_mode(
        iree_make_cstring_view(FLAG_interleave),
        &out_options->interleave_mode));
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_positive_i32_to_host_size(
      "repetitions", FLAG_repetitions, &out_options->repetitions));
  if (compare_requested &&
      !iree_string_view_is_empty(out_options->selected_benchmark)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compare selects benchmarks directly and cannot be combined with "
        "--benchmark");
  }
  if (compare_requested &&
      !iree_string_view_is_empty(out_options->selected_case)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compare selects benchmark/case pairs directly and cannot be "
        "combined with --case");
  }
  if (compare_requested && out_options->sample_compilation_mode ==
                               IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_BOTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compare requires one sample-compilation mode; use "
        "--sample_compilation=once or --sample_compilation=per_sample");
  }
  if (iree_string_view_equal(
          iree_string_view_trim(out_options->file_output_dir), IREE_SV("-"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--file_output_dir must name a directory; '-' is reserved for stdout");
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_register_context(
    void* user_data, loom_context_t* context) {
  const iree_benchmark_loom_configuration_t* configuration =
      (const iree_benchmark_loom_configuration_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_op_registry_register_all_dialects(context));
  if (configuration->register_context.fn == NULL) {
    return iree_ok_status();
  }
  return configuration->register_context.fn(
      configuration->register_context.user_data, context);
}

static iree_status_t iree_benchmark_loom_compile_report_options_initialize(
    const iree_benchmark_loom_options_t* options,
    loom_run_compile_report_capture_options_t* out_options) {
  loom_run_compile_report_capture_options_initialize(out_options);
  IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_options_parse_request(
      options->compile_report, out_options));
  if (out_options->sink_format == LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_TEXT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-benchmark-loom emits structured JSONL reports; use "
        "--compile_report=summary, details, json-summary, or json-details");
  }
  out_options->row_limit = options->compile_report_row_limit;
  return iree_ok_status();
}

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

static iree_host_size_t iree_benchmark_loom_case_sample_from_benchmark_sample(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t benchmark_sample_ordinal) {
  return loom_testbench_benchmark_sample_case_ordinal(case_plan, benchmark_plan,
                                                      benchmark_sample_ordinal);
}

static void iree_benchmark_loom_hal_context_initialize(
    const iree_benchmark_loom_configuration_t* configuration,
    iree_allocator_t host_allocator,
    iree_benchmark_loom_hal_context_t* out_context) {
  *out_context = (iree_benchmark_loom_hal_context_t){
      .configuration = configuration,
  };
  loom_run_hal_testbench_context_initialize(
      configuration->hal_artifact_provider_registry, host_allocator,
      &out_context->execution);
}

static void iree_benchmark_loom_hal_context_deinitialize(
    iree_benchmark_loom_hal_context_t* context) {
  if (!context) {
    return;
  }
  loom_run_hal_testbench_context_deinitialize(&context->execution);
  *context = (iree_benchmark_loom_hal_context_t){0};
}

static iree_status_t iree_benchmark_loom_hal_actual_provider_initialize(
    iree_benchmark_loom_hal_context_t* context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    iree_string_view_t pipeline, const loom_module_t* test_module,
    const loom_testbench_invocation_plan_t* actual_invocation,
    iree_string_view_t sample_compilation,
    const loom_testbench_case_plan_t* sample_constant_case_plan,
    iree_host_size_t sample_constant_ordinal, bool has_sample_constant_ordinal,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    iree_benchmark_loom_hal_actual_provider_t* out_provider) {
  *out_provider = (iree_benchmark_loom_hal_actual_provider_t){
      .context = context,
      .sample_compilation = sample_compilation,
  };
  iree_allocator_t host_allocator = context->execution.host_allocator;
  iree_benchmark_loom_diagnostic_capture_initialize(host_allocator,
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
            IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG) {
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
        .sample_constant_case_plan = sample_constant_case_plan,
        .sample_constant_ordinal = sample_constant_ordinal,
        .has_sample_constant_ordinal = has_sample_constant_ordinal,
        .diagnostic_sink =
            (loom_diagnostic_sink_t){
                .fn = iree_benchmark_loom_diagnostic_capture_sink,
                .user_data = &out_provider->diagnostics,
            },
        .report = report_options.report,
        .report_row_storage = report_options.report_row_storage,
        .artifact_flags = artifact_flags,
    };
    loom_run_hal_testbench_actual_provider_initialize(&provider_options,
                                                      &out_provider->execution);
  } else {
    iree_benchmark_loom_diagnostic_capture_deinitialize(
        &out_provider->diagnostics);
    *out_provider = (iree_benchmark_loom_hal_actual_provider_t){0};
  }
  return status;
}

static void iree_benchmark_loom_hal_actual_provider_deinitialize(
    iree_benchmark_loom_hal_actual_provider_t* provider) {
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
  iree_benchmark_loom_diagnostic_capture_deinitialize(&provider->diagnostics);
  *provider = (iree_benchmark_loom_hal_actual_provider_t){0};
}

static iree_status_t iree_benchmark_loom_hal_actual_provider_compile(
    iree_benchmark_loom_hal_actual_provider_t* provider) {
  return loom_run_hal_testbench_actual_provider_compile(&provider->execution);
}

typedef struct iree_benchmark_loom_hal_input_ring_t {
  // Host allocator used for ring-owned arrays.
  iree_allocator_t host_allocator;
  // Ring-owned invocation plans materialized from check ops.
  loom_run_hal_invocation_plan_t* plans;
  // Borrowed binding-list pointers into |plans| for HAL benchmark setup.
  iree_vm_list_t** binding_lists;
  // Number of entries in |plans| and |binding_lists|.
  iree_host_size_t plan_count;
  // Data/cache summary derived while materializing the ring.
  iree_benchmark_loom_data_cache_summary_t summary;
} iree_benchmark_loom_hal_input_ring_t;

static bool iree_benchmark_loom_hal_invocation_options_equal(
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

static iree_status_t iree_benchmark_loom_hal_input_ring_count_for_sample(
    const iree_benchmark_loom_options_t* options, uint64_t binding_set_bytes,
    iree_host_size_t dispatches_per_batch, iree_host_size_t* out_ring_count) {
  *out_ring_count = 1;
  if (options->input_ring_count > 0) {
    *out_ring_count = options->input_ring_count;
    return iree_ok_status();
  }
  if (options->input_ring_min_bytes == 0 || binding_set_bytes == 0) {
    return iree_ok_status();
  }
  const uint64_t requested_min_bytes = (uint64_t)options->input_ring_min_bytes;
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

static iree_status_t iree_benchmark_loom_prepare_hal_invocation_plan_for_sample(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_hal_actual_provider_t* provider,
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

static iree_status_t iree_benchmark_loom_hal_input_ring_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_options_t* options,
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    iree_benchmark_loom_hal_input_ring_t* out_ring) {
  *out_ring = (iree_benchmark_loom_hal_input_ring_t){
      .host_allocator = allocator,
  };

  loom_run_hal_invocation_plan_t first_plan = {0};
  iree_status_t status =
      iree_benchmark_loom_prepare_hal_invocation_plan_for_sample(
          module_plan, case_plan, provider, materializer_options,
          sample_ordinal, allocator, &first_plan);
  uint64_t first_binding_set_bytes = 0;
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_binding_list_total_byte_length(
        first_plan.bindings, &first_binding_set_bytes);
  }

  iree_host_size_t ring_count = 1;
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_hal_input_ring_count_for_sample(
        options, first_binding_set_bytes, policy->hal_options.timing.batch_size,
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
    memset(out_ring->plans, 0, ring_count * sizeof(*out_ring->plans));
    out_ring->plan_count = ring_count;
    out_ring->plans[0] = first_plan;
    first_plan = (loom_run_hal_invocation_plan_t){0};
    out_ring->binding_lists[0] = out_ring->plans[0].bindings;
    out_ring->summary = (iree_benchmark_loom_data_cache_summary_t){
        .populated = true,
        .binding_count = iree_vm_list_size(out_ring->plans[0].bindings),
        .binding_ring_count = ring_count,
        .dispatches_per_batch = policy->hal_options.timing.batch_size,
        .requested_min_ring_bytes = (uint64_t)options->input_ring_min_bytes,
        .binding_set_bytes = first_binding_set_bytes,
        .binding_ring_bytes = first_binding_set_bytes,
    };
  }
  for (iree_host_size_t i = 1; iree_status_is_ok(status) && i < ring_count;
       ++i) {
    status = iree_benchmark_loom_prepare_hal_invocation_plan_for_sample(
        module_plan, case_plan, provider, materializer_options, sample_ordinal,
        allocator, &out_ring->plans[i]);
    if (iree_status_is_ok(status) &&
        !iree_benchmark_loom_hal_invocation_options_equal(
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
    for (iree_host_size_t i = 0; i < out_ring->plan_count; ++i) {
      loom_run_hal_invocation_plan_deinitialize(&out_ring->plans[i]);
    }
    iree_allocator_free(out_ring->host_allocator, out_ring->binding_lists);
    iree_allocator_free(out_ring->host_allocator, out_ring->plans);
    *out_ring = (iree_benchmark_loom_hal_input_ring_t){0};
  }
  loom_run_hal_invocation_plan_deinitialize(&first_plan);
  return status;
}

static void iree_benchmark_loom_hal_input_ring_deinitialize(
    iree_benchmark_loom_hal_input_ring_t* ring) {
  if (ring == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < ring->plan_count; ++i) {
    loom_run_hal_invocation_plan_deinitialize(&ring->plans[i]);
  }
  iree_allocator_free(ring->host_allocator, ring->binding_lists);
  iree_allocator_free(ring->host_allocator, ring->plans);
  *ring = (iree_benchmark_loom_hal_input_ring_t){0};
}

typedef struct iree_benchmark_loom_hal_sequence_input_ring_t {
  // Host allocator used for ring-owned arrays.
  iree_allocator_t host_allocator;
  // Ring-owned invocation plans in ring-slot, sequence-step order.
  loom_run_hal_invocation_plan_t* plans;
  // Borrowed plan pointers into |plans| for HAL benchmark setup.
  const loom_run_hal_invocation_plan_t** plan_ptrs;
  // Borrowed prepared candidates in sequence-step order.
  const loom_run_hal_prepared_candidate_t** candidates;
  // Number of logical ring slots in |plans|.
  iree_host_size_t plan_ring_count;
  // Number of actual dispatch invocations per logical ring slot.
  iree_host_size_t sequence_count;
  // Data/cache summary derived while materializing the ring.
  iree_benchmark_loom_data_cache_summary_t summary;
} iree_benchmark_loom_hal_sequence_input_ring_t;

static iree_host_size_t iree_benchmark_loom_hal_sequence_input_ring_plan_count(
    const iree_benchmark_loom_hal_sequence_input_ring_t* ring) {
  return ring->plan_ring_count * ring->sequence_count;
}

static iree_status_t iree_benchmark_loom_hal_sequence_plans_total_byte_length(
    const loom_run_hal_invocation_plan_t* plans, iree_host_size_t plan_count,
    uint64_t* out_byte_length) {
  *out_byte_length = 0;
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    uint64_t plan_byte_length = 0;
    IREE_RETURN_IF_ERROR(loom_run_hal_binding_list_total_byte_length(
        plans[i].bindings, &plan_byte_length));
    if (UINT64_MAX - *out_byte_length < plan_byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL benchmark sequence binding byte count "
                              "overflowed uint64");
    }
    *out_byte_length += plan_byte_length;
  }
  return iree_ok_status();
}

static iree_host_size_t iree_benchmark_loom_hal_sequence_binding_count(
    const loom_run_hal_invocation_plan_t* plans, iree_host_size_t plan_count) {
  iree_host_size_t binding_count = 0;
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    binding_count += iree_vm_list_size(plans[i].bindings);
  }
  return binding_count;
}

static iree_status_t iree_benchmark_loom_prepare_hal_sequence_plans_for_sample(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const loom_run_hal_testbench_actual_sequence_t* sequence,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    loom_run_hal_invocation_plan_t* out_plans) {
  loom_testbench_value_table_t table = {0};
  iree_status_t status = loom_testbench_value_table_initialize(
      module_plan->module, case_plan, allocator, &table);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_materialize_case_sample(
        materializer_options, case_plan, sample_ordinal, &table);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < sequence->provider_count; ++i) {
    const loom_run_hal_testbench_actual_provider_t* provider =
        &sequence->providers[i];
    loom_run_hal_invocation_options_t invocation_options = {0};
    iree_vm_list_t* bindings = NULL;
    status = loom_run_hal_testbench_create_invocation_inputs_from_table(
        &table, provider->actual_invocation, &provider->invocation_options,
        allocator, &invocation_options, &bindings);
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_invocation_plan_prepare_from_lists(
          &invocation_options, bindings, /*expected_bindings=*/NULL,
          /*max_output_element_count=*/0, &out_plans[i]);
    }
    iree_vm_list_release(bindings);
  }
  loom_testbench_value_table_deinitialize(&table);
  return status;
}

static iree_status_t iree_benchmark_loom_hal_sequence_input_ring_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_options_t* options,
    const loom_run_hal_testbench_actual_sequence_t* sequence,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    iree_benchmark_loom_hal_sequence_input_ring_t* out_ring) {
  *out_ring = (iree_benchmark_loom_hal_sequence_input_ring_t){
      .host_allocator = allocator,
      .sequence_count = sequence->provider_count,
  };

  loom_run_hal_invocation_plan_t* first_plans = NULL;
  iree_host_size_t first_plan_count = 0;
  iree_status_t status =
      iree_allocator_malloc_array(allocator, sequence->provider_count,
                                  sizeof(*first_plans), (void**)&first_plans);
  if (iree_status_is_ok(status)) {
    memset(first_plans, 0, sequence->provider_count * sizeof(*first_plans));
    first_plan_count = sequence->provider_count;
    status = iree_benchmark_loom_prepare_hal_sequence_plans_for_sample(
        module_plan, case_plan, sequence, materializer_options, sample_ordinal,
        allocator, first_plans);
  }
  uint64_t first_binding_set_bytes = 0;
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_hal_sequence_plans_total_byte_length(
        first_plans, sequence->provider_count, &first_binding_set_bytes);
  }

  iree_host_size_t ring_count = 1;
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_hal_input_ring_count_for_sample(
        options, first_binding_set_bytes, policy->hal_options.timing.batch_size,
        &ring_count);
  }
  iree_host_size_t plan_count = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_mul(ring_count, sequence->provider_count,
                                  &plan_count)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL benchmark sequence input ring plan count "
                              "overflowed host size limits");
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, plan_count,
                                         sizeof(*out_ring->plans),
                                         (void**)&out_ring->plans);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, plan_count,
                                         sizeof(*out_ring->plan_ptrs),
                                         (void**)&out_ring->plan_ptrs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, sequence->provider_count,
                                         sizeof(*out_ring->candidates),
                                         (void**)&out_ring->candidates);
  }
  if (iree_status_is_ok(status)) {
    memset(out_ring->plans, 0, plan_count * sizeof(*out_ring->plans));
    out_ring->plan_ring_count = ring_count;
    for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
      out_ring->plans[i] = first_plans[i];
      first_plans[i] = (loom_run_hal_invocation_plan_t){0};
      out_ring->plan_ptrs[i] = &out_ring->plans[i];
      out_ring->candidates[i] = &sequence->providers[i].prepared_candidate;
    }
    iree_host_size_t dispatches_per_batch = 0;
    if (!iree_host_size_checked_mul(policy->hal_options.timing.batch_size,
                                    sequence->provider_count,
                                    &dispatches_per_batch)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "HAL benchmark sequence dispatch count "
                                "overflowed host size limits");
    } else {
      out_ring->summary = (iree_benchmark_loom_data_cache_summary_t){
          .populated = true,
          .binding_count = iree_benchmark_loom_hal_sequence_binding_count(
              out_ring->plans, sequence->provider_count),
          .binding_ring_count = ring_count,
          .dispatches_per_batch = dispatches_per_batch,
          .requested_min_ring_bytes = (uint64_t)options->input_ring_min_bytes,
          .binding_set_bytes = first_binding_set_bytes,
          .binding_ring_bytes = first_binding_set_bytes,
      };
    }
  }
  for (iree_host_size_t ring_index = 1;
       iree_status_is_ok(status) && ring_index < ring_count; ++ring_index) {
    loom_run_hal_invocation_plan_t* ring_plans =
        &out_ring->plans[ring_index * sequence->provider_count];
    status = iree_benchmark_loom_prepare_hal_sequence_plans_for_sample(
        module_plan, case_plan, sequence, materializer_options, sample_ordinal,
        allocator, ring_plans);
    for (iree_host_size_t step_index = 0;
         iree_status_is_ok(status) && step_index < sequence->provider_count;
         ++step_index) {
      if (!iree_benchmark_loom_hal_invocation_options_equal(
              &out_ring->plans[step_index].options,
              &ring_plans[step_index].options)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HAL benchmark sequence input ring materialization changed "
            "dispatch constants or geometry for sample %" PRIhsz
            " step %" PRIhsz,
            sample_ordinal, step_index);
      }
    }
    uint64_t binding_set_bytes = 0;
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_hal_sequence_plans_total_byte_length(
          ring_plans, sequence->provider_count, &binding_set_bytes);
    }
    if (iree_status_is_ok(status)) {
      if (UINT64_MAX - out_ring->summary.binding_ring_bytes <
          binding_set_bytes) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "HAL benchmark sequence input ring byte "
                                  "count overflowed uint64");
      } else {
        out_ring->summary.binding_ring_bytes += binding_set_bytes;
      }
    }
    for (iree_host_size_t step_index = 0;
         iree_status_is_ok(status) && step_index < sequence->provider_count;
         ++step_index) {
      const iree_host_size_t plan_index =
          ring_index * sequence->provider_count + step_index;
      out_ring->plan_ptrs[plan_index] = &out_ring->plans[plan_index];
    }
  }
  if (!iree_status_is_ok(status)) {
    const iree_host_size_t plan_count =
        iree_benchmark_loom_hal_sequence_input_ring_plan_count(out_ring);
    for (iree_host_size_t i = 0; i < plan_count; ++i) {
      loom_run_hal_invocation_plan_deinitialize(&out_ring->plans[i]);
    }
    iree_allocator_free(out_ring->host_allocator, out_ring->candidates);
    iree_allocator_free(out_ring->host_allocator, out_ring->plan_ptrs);
    iree_allocator_free(out_ring->host_allocator, out_ring->plans);
    *out_ring = (iree_benchmark_loom_hal_sequence_input_ring_t){0};
  }
  for (iree_host_size_t i = 0; i < first_plan_count; ++i) {
    loom_run_hal_invocation_plan_deinitialize(&first_plans[i]);
  }
  iree_allocator_free(allocator, first_plans);
  return status;
}

static void iree_benchmark_loom_hal_sequence_input_ring_deinitialize(
    iree_benchmark_loom_hal_sequence_input_ring_t* ring) {
  if (ring == NULL) {
    return;
  }
  const iree_host_size_t plan_count =
      iree_benchmark_loom_hal_sequence_input_ring_plan_count(ring);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    loom_run_hal_invocation_plan_deinitialize(&ring->plans[i]);
  }
  iree_allocator_free(ring->host_allocator, ring->candidates);
  iree_allocator_free(ring->host_allocator, ring->plan_ptrs);
  iree_allocator_free(ring->host_allocator, ring->plans);
  *ring = (iree_benchmark_loom_hal_sequence_input_ring_t){0};
}

static void iree_benchmark_loom_benchmark_result_set_compile_rejection(
    const iree_benchmark_loom_hal_actual_provider_t* provider,
    iree_benchmark_loom_benchmark_result_t* out_result) {
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
  out_result->sample_compilation = provider->sample_compilation;
  if (provider->execution.has_sample_constant_ordinal) {
    out_result->has_sample_ordinal = true;
    out_result->sample_ordinal = provider->execution.sample_constant_ordinal;
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

static iree_status_t iree_benchmark_loom_append_profile_artifact_path(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_hal_device_profiling_data_families_t profile_data_families,
    iree_string_view_t sample_compilation, iree_host_size_t sample_ordinal,
    iree_string_builder_t* artifact_path) {
  const iree_host_size_t initial_size = iree_string_builder_size(artifact_path);
  IREE_RETURN_IF_ERROR(
      iree_benchmark_loom_append_effective_profile_artifacts_dir(
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
  if (!iree_string_view_is_empty(sample_compilation)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(artifact_path, "_"));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(artifact_path, sample_compilation));
  }
  return iree_string_builder_append_format(
      artifact_path, "_sample%" PRIhsz ".irpf", sample_ordinal);
}

static iree_status_t iree_benchmark_loom_run_hal_benchmark_sample(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_hal_actual_provider_t* provider,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t benchmark_sample_ordinal, iree_allocator_t allocator,
    iree_benchmark_loom_benchmark_result_t* out_result) {
  const iree_host_size_t case_sample_ordinal =
      iree_benchmark_loom_case_sample_from_benchmark_sample(
          benchmark_plan, case_plan, benchmark_sample_ordinal);
  memset(out_result, 0, sizeof(*out_result));
  out_result->sample_compilation = provider->sample_compilation;
  out_result->has_sample_ordinal = true;
  out_result->sample_ordinal = case_sample_ordinal;
  out_result->samples_per_iteration = 1;

  if (provider->execution.compile_report_available) {
    out_result->compile_report_capture = &provider->compile_report_capture;
  }
  out_result->compile_report_artifact_path =
      provider->compile_report_artifact_path;
  out_result->target_artifact_path = provider->target_artifact_path;
  out_result->target_listing_path = provider->target_listing_path;
  out_result->hal_executable_path = provider->hal_executable_path;
  if (provider->execution.compile_rejected) {
    iree_benchmark_loom_benchmark_result_set_compile_rejection(provider,
                                                               out_result);
    out_result->has_sample_ordinal = true;
    out_result->sample_ordinal = case_sample_ordinal;
    out_result->samples_per_iteration = 1;
    return iree_ok_status();
  }

  iree_benchmark_loom_hal_input_ring_t input_ring = {0};
  iree_status_t status = iree_benchmark_loom_hal_input_ring_initialize(
      module_plan, case_plan, policy, options, provider, materializer_options,
      case_sample_ordinal, allocator, &input_ring);
  if (iree_status_is_ok(status)) {
    out_result->has_hal_benchmark = true;
    out_result->data_cache = input_ring.summary;
    loom_run_hal_benchmark_options_t hal_options = policy->hal_options;
    iree_string_builder_t profile_artifact_path;
    iree_string_builder_initialize(allocator, &profile_artifact_path);
    status = iree_benchmark_loom_append_profile_artifact_path(
        run, candidate, policy->hal_options.profile_data_families,
        provider->sample_compilation, case_sample_ordinal,
        &profile_artifact_path);
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      status = iree_benchmark_loom_create_parent_directory(
          iree_string_builder_view(&profile_artifact_path), allocator);
    }
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      status = iree_benchmark_loom_artifact_bundle_record_file(
          provider->context->artifact_bundle,
          IREE_BENCHMARK_LOOM_BUNDLE_FILE_PROFILE,
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
  iree_benchmark_loom_hal_input_ring_deinitialize(&input_ring);
  return status;
}

static iree_status_t iree_benchmark_loom_hal_actual_sequence_compile(
    loom_run_hal_testbench_actual_sequence_t* sequence) {
  for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_run_hal_testbench_actual_provider_compile(
        &sequence->providers[i]));
  }
  return iree_ok_status();
}

static const loom_run_hal_testbench_actual_provider_t*
iree_benchmark_loom_hal_actual_sequence_first_rejection(
    const loom_run_hal_testbench_actual_sequence_t* sequence) {
  for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
    const loom_run_hal_testbench_actual_provider_t* provider =
        &sequence->providers[i];
    if (provider->compile_rejected) {
      return provider;
    }
  }
  return NULL;
}

static void iree_benchmark_loom_benchmark_result_set_sequence_compile_rejection(
    const loom_run_hal_testbench_actual_provider_t* provider,
    iree_string_view_t sample_compilation,
    iree_benchmark_loom_benchmark_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  out_result->status = IREE_SV("compile_failed");
  out_result->has_failure = true;
  out_result->failure_stage = provider->compile_failure_stage;
  out_result->failure_kind = provider->compile_failure_kind;
  out_result->failure_message = provider->compile_failure_message;
  out_result->diagnostic_error_count = provider->diagnostic_error_count;
  out_result->diagnostic_warning_count = provider->diagnostic_warning_count;
  out_result->diagnostic_remark_count = provider->diagnostic_remark_count;
  out_result->sample_compilation = sample_compilation;
}

static iree_status_t iree_benchmark_loom_run_hal_sequence_benchmark_sample(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_hal_context_t* hal_context,
    loom_run_hal_testbench_actual_sequence_t* sequence,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_string_view_t sample_compilation,
    iree_host_size_t benchmark_sample_ordinal, iree_allocator_t allocator,
    iree_benchmark_loom_benchmark_result_t* out_result) {
  const iree_host_size_t case_sample_ordinal =
      iree_benchmark_loom_case_sample_from_benchmark_sample(
          benchmark_plan, case_plan, benchmark_sample_ordinal);
  memset(out_result, 0, sizeof(*out_result));
  out_result->sample_compilation = sample_compilation;
  out_result->has_sample_ordinal = true;
  out_result->sample_ordinal = case_sample_ordinal;
  out_result->samples_per_iteration = 1;

  const loom_run_hal_testbench_actual_provider_t* rejected_provider =
      iree_benchmark_loom_hal_actual_sequence_first_rejection(sequence);
  if (rejected_provider != NULL) {
    iree_benchmark_loom_benchmark_result_set_sequence_compile_rejection(
        rejected_provider, sample_compilation, out_result);
    out_result->has_sample_ordinal = true;
    out_result->sample_ordinal = case_sample_ordinal;
    out_result->samples_per_iteration = 1;
    return iree_ok_status();
  }

  iree_benchmark_loom_hal_sequence_input_ring_t input_ring = {0};
  iree_status_t status = iree_benchmark_loom_hal_sequence_input_ring_initialize(
      module_plan, case_plan, policy, options, sequence, materializer_options,
      case_sample_ordinal, allocator, &input_ring);
  if (iree_status_is_ok(status)) {
    out_result->has_hal_benchmark = true;
    out_result->data_cache = input_ring.summary;
    loom_run_hal_benchmark_options_t hal_options = policy->hal_options;
    iree_string_builder_t profile_artifact_path;
    iree_string_builder_initialize(allocator, &profile_artifact_path);
    status = iree_benchmark_loom_append_profile_artifact_path(
        run, candidate, policy->hal_options.profile_data_families,
        sample_compilation, case_sample_ordinal, &profile_artifact_path);
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      status = iree_benchmark_loom_create_parent_directory(
          iree_string_builder_view(&profile_artifact_path), allocator);
    }
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      status = iree_benchmark_loom_artifact_bundle_record_file(
          hal_context->artifact_bundle, IREE_BENCHMARK_LOOM_BUNDLE_FILE_PROFILE,
          iree_string_builder_view(&profile_artifact_path));
    }
    if (iree_status_is_ok(status) &&
        iree_string_builder_size(&profile_artifact_path) != 0) {
      hal_options.profile_artifact_path =
          iree_string_builder_view(&profile_artifact_path);
    }
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_benchmark_dispatch_sequence_plan_ring(
          &hal_context->execution.runtime, sequence->provider_count,
          input_ring.candidates, input_ring.plan_ring_count,
          input_ring.plan_ptrs, &hal_options, allocator,
          &out_result->hal_benchmark);
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
  iree_benchmark_loom_hal_sequence_input_ring_deinitialize(&input_ring);
  return status;
}

static iree_status_t iree_benchmark_loom_prepare_case_executor(
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

static iree_status_t iree_benchmark_loom_run_case_correctness_range(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_string_view_t sample_compilation, iree_host_size_t begin_sample,
    iree_host_size_t end_sample, iree_arena_allocator_t* arena,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* out_sample_count,
    iree_host_size_t* out_failed_sample_count) {
  *out_sample_count = 0;
  *out_failed_sample_count = 0;

  const loom_testbench_case_plan_t* case_plan = &module_plan->cases[case_index];
  iree_status_t status = iree_ok_status();
  loom_testbench_prepared_case_t prepared_case = {0};
  loom_testbench_case_executor_t executor = {0};
  bool executor_initialized = false;
  status = iree_benchmark_loom_prepare_case_executor(
      module_plan, case_index, execution_options, arena, &prepared_case,
      &executor, &executor_initialized);

  for (iree_host_size_t benchmark_sample_ordinal = begin_sample;
       iree_status_is_ok(status) && benchmark_sample_ordinal < end_sample;
       ++benchmark_sample_ordinal) {
    const iree_host_size_t case_sample_ordinal =
        iree_benchmark_loom_case_sample_from_benchmark_sample(
            benchmark_plan, case_plan, benchmark_sample_ordinal);
    loom_testbench_case_sample_result_t sample_result = {0};
    status = loom_testbench_run_case_sample(&executor, case_sample_ordinal,
                                            &sample_result);
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_event_sink_emit_sample(
          event_sink, run, candidate, IREE_BENCHMARK_LOOM_INDEX_INVALID,
          module_plan->module, benchmark_plan, case_plan, sample_compilation,
          benchmark_sample_ordinal, case_sample_ordinal, &sample_result);
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

static iree_status_t iree_benchmark_loom_run_case_iteration(
    loom_testbench_case_executor_t* executor,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan, iree_host_size_t begin_sample,
    iree_host_size_t end_sample, iree_host_size_t* inout_failed_sample_count) {
  for (iree_host_size_t benchmark_sample_ordinal = begin_sample;
       benchmark_sample_ordinal < end_sample; ++benchmark_sample_ordinal) {
    const iree_host_size_t case_sample_ordinal =
        iree_benchmark_loom_case_sample_from_benchmark_sample(
            benchmark_plan, case_plan, benchmark_sample_ordinal);
    loom_testbench_case_sample_result_t sample_result = {0};
    IREE_RETURN_IF_ERROR(loom_testbench_run_case_sample(
        executor, case_sample_ordinal, &sample_result));
    if (!sample_result.passed) {
      ++*inout_failed_sample_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_run_benchmark_iterations(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_execution_options_t* execution_options,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    iree_host_size_t begin_sample, iree_host_size_t end_sample,
    iree_arena_allocator_t* arena, iree_allocator_t host_allocator,
    iree_benchmark_loom_benchmark_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));

  const loom_testbench_case_plan_t* case_plan =
      &module_plan->cases[benchmark_plan->case_index];
  out_result->samples_per_iteration = end_sample - begin_sample;
  if (end_sample == begin_sample + 1) {
    out_result->has_sample_ordinal = true;
    out_result->sample_ordinal =
        iree_benchmark_loom_case_sample_from_benchmark_sample(
            benchmark_plan, case_plan, begin_sample);
  }

  iree_duration_t* durations = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, policy->iterations * sizeof(*durations),
      (void**)&durations));

  iree_status_t status = iree_ok_status();
  loom_testbench_prepared_case_t prepared_case = {0};
  loom_testbench_case_executor_t executor = {0};
  bool executor_initialized = false;
  status = iree_benchmark_loom_prepare_case_executor(
      module_plan, benchmark_plan->case_index, execution_options, arena,
      &prepared_case, &executor, &executor_initialized);

  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < policy->warmup_iterations; ++i) {
    status = iree_benchmark_loom_run_case_iteration(
        &executor, benchmark_plan, case_plan, begin_sample, end_sample,
        &out_result->failed_sample_count);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < policy->iterations; ++i) {
    const iree_time_t start_time_ns = iree_time_now();
    status = iree_benchmark_loom_run_case_iteration(
        &executor, benchmark_plan, case_plan, begin_sample, end_sample,
        &out_result->failed_sample_count);
    const iree_time_t end_time_ns = iree_time_now();
    durations[i] =
        end_time_ns >= start_time_ns ? end_time_ns - start_time_ns : 0;
  }

  if (iree_status_is_ok(status)) {
    out_result->executed = true;
    out_result->passed = out_result->failed_sample_count == 0;
    iree_benchmark_loom_compute_timing_stats(durations, policy->iterations,
                                             &out_result->timing);
  }

  if (executor_initialized) {
    loom_testbench_case_executor_deinitialize(&executor);
  }
  iree_arena_reset(arena);
  iree_allocator_free(host_allocator, durations);
  return status;
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

static iree_status_t iree_benchmark_loom_emit_work_item_sample_aliases(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_work_item_t* work_item,
    iree_host_size_t sample_offset,
    const loom_testbench_case_sample_result_t* sample_result,
    const iree_benchmark_loom_event_sink_t* event_sink) {
  for (iree_host_size_t logical_sample_index = 0;
       logical_sample_index < work_plan->logical_sample_count;
       ++logical_sample_index) {
    const iree_benchmark_loom_logical_sample_t* logical_sample =
        &work_plan->logical_samples[logical_sample_index];
    if (logical_sample->work_item_index != work_item->work_item_index) {
      continue;
    }
    const iree_benchmark_loom_selected_benchmark_t* selection =
        &work_plan->selected_benchmarks[logical_sample->selection_index];
    const iree_host_size_t benchmark_sample_ordinal =
        logical_sample->begin_benchmark_sample + sample_offset;
    if (benchmark_sample_ordinal >= logical_sample->end_benchmark_sample) {
      continue;
    }
    const iree_host_size_t case_sample_ordinal =
        iree_benchmark_loom_case_sample_from_benchmark_sample(
            selection->benchmark_plan, selection->case_plan,
            benchmark_sample_ordinal);
    if (case_sample_ordinal != sample_result->sample_ordinal) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "benchmark work item alias for `%.*s` mapped benchmark sample "
          "%" PRIhsz " to case sample %" PRIhsz
          " but physical work item produced case sample %" PRIhsz,
          (int)selection->benchmark_plan->name.size,
          selection->benchmark_plan->name.data, benchmark_sample_ordinal,
          case_sample_ordinal, sample_result->sample_ordinal);
    }
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_event_sink_emit_sample(
        event_sink, run, &selection->identity, work_item->work_item_index,
        module_plan->module, selection->benchmark_plan, selection->case_plan,
        logical_sample->sample_compilation, benchmark_sample_ordinal,
        case_sample_ordinal, sample_result));
  }
  return iree_ok_status();
}

static bool iree_benchmark_loom_benchmark_result_counts_as_failed(
    const iree_benchmark_loom_benchmark_result_t* benchmark_result) {
  if (iree_string_view_equal(benchmark_result->status, IREE_SV("skipped"))) {
    return false;
  }
  return !benchmark_result->executed || !benchmark_result->passed;
}

static iree_status_t iree_benchmark_loom_emit_work_item_result_aliases(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_work_item_t* work_item,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* inout_failed_benchmark_count) {
  for (iree_host_size_t logical_sample_index = 0;
       logical_sample_index < work_plan->logical_sample_count;
       ++logical_sample_index) {
    const iree_benchmark_loom_logical_sample_t* logical_sample =
        &work_plan->logical_samples[logical_sample_index];
    if (logical_sample->work_item_index != work_item->work_item_index) {
      continue;
    }
    const iree_benchmark_loom_selected_benchmark_t* selection =
        &work_plan->selected_benchmarks[logical_sample->selection_index];
    iree_benchmark_loom_benchmark_result_t alias_result = *benchmark_result;
    if (logical_sample->has_case_sample_ordinal) {
      alias_result.has_sample_ordinal = true;
      alias_result.sample_ordinal = logical_sample->case_sample_ordinal;
    }
    if (alias_result.samples_per_iteration == 0 &&
        logical_sample->has_case_sample_ordinal) {
      alias_result.samples_per_iteration = 1;
    }
    if (iree_benchmark_loom_benchmark_result_counts_as_failed(&alias_result)) {
      ++*inout_failed_benchmark_count;
    }
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_event_sink_emit_benchmark_result(
        event_sink, run, &selection->identity, work_item->work_item_index,
        module_plan->module, selection->benchmark_plan, selection->case_plan,
        &selection->policy, &alias_result, correctness_sample_count,
        correctness_failed_sample_count));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_event_sink_emit_profile(
        event_sink, run, &selection->identity, work_item->work_item_index,
        module_plan->module, selection->benchmark_plan, selection->case_plan,
        &selection->policy, &alias_result));
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_run_work_item_correctness_range(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_work_item_t* work_item,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* arena,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* out_sample_count,
    iree_host_size_t* out_failed_sample_count) {
  *out_sample_count = 0;
  *out_failed_sample_count = 0;
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &work_plan
           ->selected_benchmarks[work_item->representative_selection_index];
  const loom_testbench_benchmark_plan_t* benchmark_plan =
      selection->benchmark_plan;
  const loom_testbench_case_plan_t* case_plan = selection->case_plan;

  loom_testbench_prepared_case_t prepared_case = {0};
  loom_testbench_case_executor_t executor = {0};
  bool executor_initialized = false;
  iree_status_t status = iree_benchmark_loom_prepare_case_executor(
      module_plan, benchmark_plan->case_index, execution_options, arena,
      &prepared_case, &executor, &executor_initialized);

  for (iree_host_size_t benchmark_sample_ordinal =
           work_item->begin_benchmark_sample;
       iree_status_is_ok(status) &&
       benchmark_sample_ordinal < work_item->end_benchmark_sample;
       ++benchmark_sample_ordinal) {
    const iree_host_size_t sample_offset =
        benchmark_sample_ordinal - work_item->begin_benchmark_sample;
    const iree_host_size_t case_sample_ordinal =
        iree_benchmark_loom_case_sample_from_benchmark_sample(
            benchmark_plan, case_plan, benchmark_sample_ordinal);
    loom_testbench_case_sample_result_t sample_result = {0};
    status = loom_testbench_run_case_sample(&executor, case_sample_ordinal,
                                            &sample_result);
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_emit_work_item_sample_aliases(
          run, module_plan, work_plan, work_item, sample_offset, &sample_result,
          event_sink);
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

typedef struct iree_benchmark_loom_dispatch_compile_context_t {
  // True once this compile item has been initialized or skipped.
  bool initialized;
  // True when target requirements skip every work item using this compile item.
  bool skipped;
  // True when this compile item executes a multi-actual sequence.
  bool uses_sequence;
  // Compiled multi-actual sequence reused by all work items in the compile
  // item.
  loom_run_hal_testbench_actual_sequence_t hal_sequence;
  // True when |hal_sequence| owns initialized state.
  bool hal_sequence_initialized;
  // Compiled single-actual provider reused by all work items in the compile
  // item.
  iree_benchmark_loom_hal_actual_provider_t hal_provider;
  // True when |hal_provider| owns initialized state.
  bool hal_provider_initialized;
  // First sequence provider that rejected compilation, or NULL when runnable.
  const loom_run_hal_testbench_actual_provider_t* rejected_sequence_provider;
  // Execution options with HAL actual and reference providers wired in.
  loom_testbench_case_execution_options_t execution_options;
  // Materializer options used by HAL benchmark timing batches.
  loom_testbench_value_materializer_options_t benchmark_materializer;
  // Reference oracle storage borrowed by |execution_options|.
  iree_benchmark_loom_reference_oracles_t reference_oracles;
} iree_benchmark_loom_dispatch_compile_context_t;

static iree_status_t iree_benchmark_loom_initialize_sequence_compile_context(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_dispatch_compile_item_t* compile_item,
    const iree_benchmark_loom_selected_benchmark_t* selection,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_hal_context_t* hal_context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    const loom_testbench_case_execution_options_t* execution_options,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_benchmark_loom_dispatch_compile_context_t* context) {
  const loom_testbench_case_plan_t* case_plan = selection->case_plan;
  context->uses_sequence = true;
  context->execution_options = *execution_options;
  context->benchmark_materializer = execution_options->materializer;

  iree_status_t status =
      loom_run_hal_testbench_context_ensure_runtime(&hal_context->execution);
  loom_testbench_requirement_result_t requirement_result = {0};
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_evaluate_case_requirements(
        hal_context->configuration, &hal_context->execution, module_plan,
        case_plan, &requirement_result);
  }
  if (iree_status_is_ok(status) && requirement_result.skipped) {
    context->skipped = true;
    return iree_ok_status();
  }

  if (iree_status_is_ok(status)) {
    context->execution_options.materializer.device =
        hal_context->execution.runtime.device;
    context->execution_options.materializer.device_allocator =
        iree_hal_device_allocator(hal_context->execution.runtime.device);
    context->execution_options.materializer.buffer_params =
        loom_run_hal_testbench_host_visible_buffer_params();
    const loom_run_hal_testbench_actual_sequence_options_t sequence_options = {
        .context = &hal_context->execution,
        .session = session,
        .target_environment = hal_context->configuration->target_environment,
        .filename = filename,
        .source = source,
        .pipeline = options->pipeline,
        .test_module = module_plan->module,
        .case_plan = case_plan,
        .sample_constant_case_plan = case_plan,
        .sample_constant_ordinal = compile_item->case_sample_ordinal,
        .has_sample_constant_ordinal = compile_item->has_case_sample_ordinal,
    };
    status = loom_run_hal_testbench_actual_sequence_initialize(
        &sequence_options, &context->hal_sequence);
  }
  if (iree_status_is_ok(status)) {
    context->hal_sequence_initialized = true;
    context->execution_options.invocation.invoke_actual =
        (loom_testbench_invocation_callback_t){
            .fn = loom_run_hal_testbench_actual_sequence_invoke,
            .user_data = &context->hal_sequence,
        };
    iree_benchmark_loom_configure_reference_oracles(
        &hal_context->execution,
        context->execution_options.materializer.host_allocator,
        &context->reference_oracles, &context->execution_options);
    context->benchmark_materializer = context->execution_options.materializer;
    context->benchmark_materializer.buffer_params =
        (iree_hal_buffer_params_t){0};
    status =
        iree_benchmark_loom_hal_actual_sequence_compile(&context->hal_sequence);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_device(event_sink, run,
                                                        hal_context);
  }
  if (iree_status_is_ok(status)) {
    context->rejected_sequence_provider =
        iree_benchmark_loom_hal_actual_sequence_first_rejection(
            &context->hal_sequence);
  }
  return status;
}

static iree_status_t iree_benchmark_loom_initialize_single_compile_context(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_dispatch_compile_item_t* compile_item,
    const iree_benchmark_loom_selected_benchmark_t* selection,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_hal_context_t* hal_context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_allocator_t allocator,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_benchmark_loom_dispatch_compile_context_t* context) {
  const loom_testbench_invocation_plan_t* actual_invocation = NULL;
  const loom_testbench_benchmark_plan_t* benchmark_plan =
      selection->benchmark_plan;
  const loom_testbench_case_plan_t* case_plan = selection->case_plan;
  context->execution_options = *execution_options;
  context->benchmark_materializer = execution_options->materializer;

  iree_status_t status = loom_run_hal_testbench_select_actual_invocation(
      case_plan, &actual_invocation);
  if (iree_status_is_ok(status)) {
    status =
        loom_run_hal_testbench_context_ensure_runtime(&hal_context->execution);
  }
  loom_testbench_requirement_result_t requirement_result = {0};
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_evaluate_case_requirements(
        hal_context->configuration, &hal_context->execution, module_plan,
        case_plan, &requirement_result);
  }
  if (iree_status_is_ok(status) && requirement_result.skipped) {
    context->skipped = true;
    return iree_ok_status();
  }

  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_hal_actual_provider_initialize(
        hal_context, session, filename, source, options->pipeline,
        module_plan->module, actual_invocation,
        compile_item->sample_compilation, case_plan,
        compile_item->case_sample_ordinal,
        compile_item->has_case_sample_ordinal, compile_report_options,
        &context->hal_provider);
  }
  if (iree_status_is_ok(status)) {
    context->hal_provider_initialized = true;
    context->execution_options.materializer.device =
        hal_context->execution.runtime.device;
    context->execution_options.materializer.device_allocator =
        iree_hal_device_allocator(hal_context->execution.runtime.device);
    context->execution_options.materializer.buffer_params =
        loom_run_hal_testbench_host_visible_buffer_params();
    context->execution_options.invocation.invoke_actual =
        (loom_testbench_invocation_callback_t){
            .fn = loom_run_hal_testbench_actual_invoke,
            .user_data = &context->hal_provider.execution,
        };
    iree_benchmark_loom_configure_reference_oracles(
        &hal_context->execution,
        context->execution_options.materializer.host_allocator,
        &context->reference_oracles, &context->execution_options);
    context->benchmark_materializer = context->execution_options.materializer;
    context->benchmark_materializer.buffer_params =
        (iree_hal_buffer_params_t){0};
    status =
        iree_benchmark_loom_hal_actual_provider_compile(&context->hal_provider);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_compiled_artifacts(
        run, &selection->identity, &context->hal_provider, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_compile_report_artifact(
        run, &selection->identity, benchmark_plan, case_plan,
        &context->hal_provider, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_device(event_sink, run,
                                                        hal_context);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_compile(
        event_sink, run, &selection->identity, benchmark_plan, case_plan,
        &context->hal_provider);
  }
  return status;
}

static iree_status_t iree_benchmark_loom_dispatch_compile_context_initialize(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_dispatch_compile_item_t* compile_item,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_hal_context_t* hal_context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_allocator_t allocator,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_benchmark_loom_dispatch_compile_context_t* context) {
  if (context->initialized) {
    return iree_ok_status();
  }
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &work_plan
           ->selected_benchmarks[compile_item->representative_selection_index];
  iree_host_size_t actual_invocation_count = 0;
  iree_status_t status = loom_run_hal_testbench_count_actual_invocations(
      selection->case_plan, &actual_invocation_count);
  if (iree_status_is_ok(status) && actual_invocation_count > 1) {
    status = iree_benchmark_loom_initialize_sequence_compile_context(
        run, module_plan, compile_item, selection, options, hal_context,
        session, filename, source, execution_options, event_sink, context);
  } else if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_initialize_single_compile_context(
        run, module_plan, compile_item, selection, options, hal_context,
        session, filename, source, compile_report_options, execution_options,
        allocator, event_sink, context);
  }
  if (iree_status_is_ok(status)) {
    context->initialized = true;
  } else {
    if (context->hal_sequence_initialized) {
      loom_run_hal_testbench_actual_sequence_deinitialize(
          &context->hal_sequence);
    }
    if (context->hal_provider_initialized) {
      iree_benchmark_loom_hal_actual_provider_deinitialize(
          &context->hal_provider);
    }
    *context = (iree_benchmark_loom_dispatch_compile_context_t){0};
  }
  return status;
}

static void iree_benchmark_loom_dispatch_compile_context_deinitialize(
    iree_benchmark_loom_dispatch_compile_context_t* context) {
  if (context->hal_sequence_initialized) {
    loom_run_hal_testbench_actual_sequence_deinitialize(&context->hal_sequence);
  }
  if (context->hal_provider_initialized) {
    iree_benchmark_loom_hal_actual_provider_deinitialize(
        &context->hal_provider);
  }
  *context = (iree_benchmark_loom_dispatch_compile_context_t){0};
}

static iree_status_t iree_benchmark_loom_run_dispatch_sample_work_item(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_work_item_t* work_item,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_hal_context_t* hal_context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_benchmark_loom_dispatch_compile_context_t* compile_context,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &work_plan
           ->selected_benchmarks[work_item->representative_selection_index];
  const iree_benchmark_loom_dispatch_compile_item_t* compile_item =
      &work_plan
           ->dispatch_compile_items[work_item->dispatch_compile_item_index];
  const loom_testbench_benchmark_plan_t* benchmark_plan =
      selection->benchmark_plan;
  const loom_testbench_case_plan_t* case_plan = selection->case_plan;
  const iree_benchmark_loom_benchmark_policy_t* policy = &selection->policy;

  IREE_RETURN_IF_ERROR(iree_benchmark_loom_dispatch_compile_context_initialize(
      run, module_plan, work_plan, compile_item, options, hal_context, session,
      filename, source, compile_report_options, execution_options, allocator,
      event_sink, compile_context));
  if (compile_context->skipped) {
    iree_benchmark_loom_benchmark_result_t benchmark_result = {
        .status = IREE_SV("skipped"),
        .sample_compilation = work_item->sample_compilation,
    };
    return iree_benchmark_loom_emit_work_item_result_aliases(
        run, module_plan, work_plan, work_item, &benchmark_result,
        /*correctness_sample_count=*/0,
        /*correctness_failed_sample_count=*/0, event_sink,
        inout_failed_benchmark_count);
  }

  if (compile_context->uses_sequence) {
    const loom_run_hal_testbench_actual_provider_t* rejected_provider =
        compile_context->rejected_sequence_provider;
    if (rejected_provider != NULL) {
      iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
      iree_benchmark_loom_benchmark_result_set_sequence_compile_rejection(
          rejected_provider, work_item->sample_compilation, &benchmark_result);
      benchmark_result.has_sample_ordinal = true;
      benchmark_result.sample_ordinal = work_item->case_sample_ordinal;
      benchmark_result.samples_per_iteration = 1;
      return iree_benchmark_loom_emit_work_item_result_aliases(
          run, module_plan, work_plan, work_item, &benchmark_result,
          /*correctness_sample_count=*/0,
          /*correctness_failed_sample_count=*/0, event_sink,
          inout_failed_benchmark_count);
    }

    iree_host_size_t correctness_sample_count = 0;
    iree_host_size_t correctness_failed_sample_count = 0;
    iree_status_t status = iree_benchmark_loom_run_work_item_correctness_range(
        run, module_plan, work_plan, work_item,
        &compile_context->execution_options, execution_arena, event_sink,
        &correctness_sample_count, &correctness_failed_sample_count);
    if (iree_status_is_ok(status)) {
      *inout_correctness_sample_count += correctness_sample_count;
      *inout_correctness_failed_sample_count += correctness_failed_sample_count;
    }
    if (iree_status_is_ok(status) && correctness_failed_sample_count != 0) {
      iree_benchmark_loom_benchmark_result_t benchmark_result = {
          .executed = false,
          .passed = false,
          .sample_compilation = work_item->sample_compilation,
          .has_sample_ordinal = true,
          .sample_ordinal = work_item->case_sample_ordinal,
          .samples_per_iteration = correctness_sample_count,
          .failed_sample_count = correctness_failed_sample_count,
      };
      status = iree_benchmark_loom_emit_work_item_result_aliases(
          run, module_plan, work_plan, work_item, &benchmark_result,
          correctness_sample_count, correctness_failed_sample_count, event_sink,
          inout_failed_benchmark_count);
    }
    if (iree_status_is_ok(status) && correctness_failed_sample_count == 0) {
      iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
      status = iree_benchmark_loom_run_hal_sequence_benchmark_sample(
          run, &selection->identity, module_plan, benchmark_plan, case_plan,
          policy, options, hal_context, &compile_context->hal_sequence,
          &compile_context->benchmark_materializer,
          work_item->sample_compilation, work_item->begin_benchmark_sample,
          allocator, &benchmark_result);
      if (iree_status_is_ok(status)) {
        status = iree_benchmark_loom_emit_work_item_result_aliases(
            run, module_plan, work_plan, work_item, &benchmark_result,
            correctness_sample_count, correctness_failed_sample_count,
            event_sink, inout_failed_benchmark_count);
      }
    }
    return status;
  }

  if (compile_context->hal_provider.execution.compile_rejected) {
    iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
    iree_benchmark_loom_benchmark_result_set_compile_rejection(
        &compile_context->hal_provider, &benchmark_result);
    benchmark_result.has_sample_ordinal = true;
    benchmark_result.sample_ordinal = work_item->case_sample_ordinal;
    benchmark_result.samples_per_iteration = 1;
    return iree_benchmark_loom_emit_work_item_result_aliases(
        run, module_plan, work_plan, work_item, &benchmark_result,
        /*correctness_sample_count=*/0,
        /*correctness_failed_sample_count=*/0, event_sink,
        inout_failed_benchmark_count);
  }

  iree_host_size_t correctness_sample_count = 0;
  iree_host_size_t correctness_failed_sample_count = 0;
  iree_status_t status = iree_benchmark_loom_run_work_item_correctness_range(
      run, module_plan, work_plan, work_item,
      &compile_context->execution_options, execution_arena, event_sink,
      &correctness_sample_count, &correctness_failed_sample_count);
  if (iree_status_is_ok(status)) {
    *inout_correctness_sample_count += correctness_sample_count;
    *inout_correctness_failed_sample_count += correctness_failed_sample_count;
  }
  if (iree_status_is_ok(status) && correctness_failed_sample_count != 0) {
    iree_benchmark_loom_benchmark_result_t benchmark_result = {
        .executed = false,
        .passed = false,
        .compile_report_artifact_path =
            compile_context->hal_provider.compile_report_artifact_path,
        .target_artifact_path =
            compile_context->hal_provider.target_artifact_path,
        .target_listing_path =
            compile_context->hal_provider.target_listing_path,
        .hal_executable_path =
            compile_context->hal_provider.hal_executable_path,
        .sample_compilation = work_item->sample_compilation,
        .has_sample_ordinal = true,
        .sample_ordinal = work_item->case_sample_ordinal,
        .samples_per_iteration = correctness_sample_count,
        .failed_sample_count = correctness_failed_sample_count,
    };
    status = iree_benchmark_loom_emit_work_item_result_aliases(
        run, module_plan, work_plan, work_item, &benchmark_result,
        correctness_sample_count, correctness_failed_sample_count, event_sink,
        inout_failed_benchmark_count);
  }
  if (iree_status_is_ok(status) && correctness_failed_sample_count == 0) {
    iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
    status = iree_benchmark_loom_run_hal_benchmark_sample(
        run, &selection->identity, module_plan, benchmark_plan, case_plan,
        policy, options, &compile_context->hal_provider,
        &compile_context->benchmark_materializer,
        work_item->begin_benchmark_sample, allocator, &benchmark_result);
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_emit_work_item_result_aliases(
          run, module_plan, work_plan, work_item, &benchmark_result,
          correctness_sample_count, correctness_failed_sample_count, event_sink,
          inout_failed_benchmark_count);
    }
  }
  return status;
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
    const iree_benchmark_loom_run_identity_t* run,
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_hal_context_t* hal_context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      candidate->selection;
  const loom_testbench_invocation_plan_t* actual_invocation = NULL;
  loom_testbench_case_execution_options_t candidate_execution_options =
      *execution_options;
  iree_benchmark_loom_reference_oracles_t reference_oracles = {0};

  iree_status_t status = loom_run_hal_testbench_select_actual_invocation(
      selection->case_plan, &actual_invocation);
  if (iree_status_is_ok(status)) {
    status =
        loom_run_hal_testbench_context_ensure_runtime(&hal_context->execution);
  }
  loom_testbench_requirement_result_t requirement_result = {0};
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_evaluate_case_requirements(
        hal_context->configuration, &hal_context->execution, module_plan,
        selection->case_plan, &requirement_result);
  }
  if (iree_status_is_ok(status) && requirement_result.skipped) {
    iree_benchmark_loom_benchmark_result_t benchmark_result = {
        .status = IREE_SV("skipped"),
        .sample_compilation = candidate->sample_compilation,
    };
    return iree_benchmark_loom_emit_dispatch_benchmark_result(
        run, &selection->identity, module_plan, selection->benchmark_plan,
        selection->case_plan, &selection->policy, &benchmark_result,
        /*correctness_sample_count=*/0,
        /*correctness_failed_sample_count=*/0, event_sink);
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t sample_constant_case_ordinal =
        candidate->has_sample_constant_ordinal
            ? iree_benchmark_loom_case_sample_from_benchmark_sample(
                  selection->benchmark_plan, selection->case_plan,
                  candidate->sample_constant_ordinal)
            : 0;
    status = iree_benchmark_loom_hal_actual_provider_initialize(
        hal_context, session, filename, source, options->pipeline,
        module_plan->module, actual_invocation, candidate->sample_compilation,
        selection->case_plan, sample_constant_case_ordinal,
        candidate->has_sample_constant_ordinal, compile_report_options,
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
    iree_benchmark_loom_configure_reference_oracles(
        &hal_context->execution,
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
        run, &selection->identity, &candidate->provider, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_compile_report_artifact(
        run, &selection->identity, selection->benchmark_plan,
        selection->case_plan, &candidate->provider, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_device(event_sink, run,
                                                        hal_context);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_compile(
        event_sink, run, &selection->identity, selection->benchmark_plan,
        selection->case_plan, &candidate->provider);
  }
  if (iree_status_is_ok(status) &&
      candidate->provider.execution.compile_rejected) {
    iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
    iree_benchmark_loom_benchmark_result_set_compile_rejection(
        &candidate->provider, &benchmark_result);
    ++*inout_failed_benchmark_count;
    return iree_benchmark_loom_emit_dispatch_benchmark_result(
        run, &selection->identity, module_plan, selection->benchmark_plan,
        selection->case_plan, &selection->policy, &benchmark_result,
        /*correctness_sample_count=*/0,
        /*correctness_failed_sample_count=*/0, event_sink);
  }

  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_run_case_correctness_range(
        run, &selection->identity, module_plan, selection->benchmark_plan,
        selection->benchmark_plan->case_index, &candidate_execution_options,
        candidate->sample_compilation, candidate->begin_sample,
        candidate->end_sample, execution_arena, event_sink,
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
        run, &selection->identity, module_plan, selection->benchmark_plan,
        selection->case_plan, &selection->policy, &benchmark_result,
        candidate->correctness_sample_count,
        candidate->correctness_failed_sample_count, event_sink);
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
    const iree_benchmark_loom_selected_benchmark_t* selections,
    iree_host_size_t selection_count,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode,
    iree_benchmark_loom_interleave_mode_t interleave_mode,
    iree_host_size_t repetitions, iree_allocator_t allocator,
    iree_benchmark_loom_dispatch_comparison_candidate_t** out_candidates) {
  *out_candidates = NULL;
  if (sample_compilation_mode == IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_BOTH) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "interleaved comparisons require one sample-compilation mode; use "
        "--sample_compilation=once or --sample_compilation=per_sample");
  }
  if (interleave_mode == IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA &&
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

  iree_benchmark_loom_dispatch_comparison_candidate_t* candidates = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, selection_count, sizeof(*candidates), (void**)&candidates));
  memset(candidates, 0, selection_count * sizeof(*candidates));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < selection_count;
       ++i) {
    if (selections[i].policy.measure_kind !=
        IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "interleaved comparison benchmark `%.*s` must use "
                           "measure = \"dispatch_complete\"",
                           (int)selections[i].benchmark_plan->name.size,
                           selections[i].benchmark_plan->name.data);
      break;
    }
    candidates[i].selection = &selections[i];
    candidates[i].sample_compilation =
        iree_benchmark_loom_sample_compilation_mode_name(
            sample_compilation_mode);
    if (sample_compilation_mode ==
        IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_PER_SAMPLE) {
      iree_host_size_t begin_sample = 0;
      iree_host_size_t end_sample = 0;
      status = iree_benchmark_loom_validate_sample_flag(
          options, selections[i].benchmark_plan->sample_count, &begin_sample,
          &end_sample);
      if (iree_status_is_ok(status) && end_sample != begin_sample + 1) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "per-sample interleaved comparison benchmark `%.*s` has %" PRIhsz
            " selected samples; use --sample= to select one sample",
            (int)selections[i].benchmark_plan->name.size,
            selections[i].benchmark_plan->name.data,
            selections[i].benchmark_plan->sample_count);
      }
      candidates[i].has_sample_constant_ordinal = true;
      candidates[i].sample_constant_ordinal = begin_sample;
    }
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_dispatch_comparison_sample_window(
          selections[i].benchmark_plan, options,
          candidates[i].has_sample_constant_ordinal,
          candidates[i].sample_constant_ordinal, &candidates[i].begin_sample,
          &candidates[i].end_sample);
    }
    const iree_host_size_t sample_capacity =
        iree_benchmark_loom_comparison_sample_capacity(
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
    for (iree_host_size_t i = 0; i < selection_count; ++i) {
      if (candidates[i].provider_initialized) {
        iree_benchmark_loom_hal_actual_provider_deinitialize(
            &candidates[i].provider);
      }
      iree_allocator_free(allocator, candidates[i].p50_samples);
      iree_allocator_free(allocator, candidates[i].p90_samples);
    }
    iree_allocator_free(allocator, candidates);
  }
  return status;
}

static void iree_benchmark_loom_dispatch_comparison_candidate_deinitialize(
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    iree_allocator_t allocator) {
  if (!candidate) {
    return;
  }
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
    const iree_benchmark_loom_run_identity_t* run,
    iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    const iree_benchmark_loom_candidate_identity_t* baseline,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_options_t* options,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_host_size_t order_index, iree_host_size_t repetition_index,
    char schedule_token, iree_allocator_t allocator,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* inout_failed_benchmark_count) {
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
      run, &candidate->selection->identity, module_plan,
      candidate->selection->benchmark_plan, candidate->selection->case_plan,
      &measurement_policy, options, &candidate->provider,
      &candidate->benchmark_materializer, candidate->begin_sample, allocator,
      &benchmark_result);
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
        event_sink, run, candidate, baseline, comparison_group, method,
        order_index, repetition_index, schedule_token, profile_suppressed,
        &benchmark_result);
  }
  return status;
}

static iree_status_t iree_benchmark_loom_run_dispatch_comparison(
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_selected_benchmark_t* selections,
    iree_host_size_t selection_count,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode,
    iree_benchmark_loom_interleave_mode_t interleave_mode,
    iree_host_size_t repetitions,
    iree_benchmark_loom_hal_context_t* hal_context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  iree_benchmark_loom_dispatch_comparison_candidate_t* candidates = NULL;
  iree_status_t status =
      iree_benchmark_loom_initialize_dispatch_comparison_candidates(
          selections, selection_count, options, sample_compilation_mode,
          interleave_mode, repetitions, allocator, &candidates);
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_validate_comparison_samples(
        module_plan->module, candidates, selection_count);
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < selection_count;
       ++i) {
    status = iree_benchmark_loom_prepare_dispatch_comparison_candidate(
        run, &candidates[i], module_plan, options, hal_context, session,
        filename, source, compile_report_options, execution_options,
        execution_arena, allocator, event_sink, inout_correctness_sample_count,
        inout_correctness_failed_sample_count, inout_failed_benchmark_count);
  }

  const iree_benchmark_loom_candidate_identity_t* baseline =
      &selections[0].identity;
  const iree_string_view_t comparison_group =
      selections[0].benchmark_plan->name;
  const iree_string_view_t method =
      iree_benchmark_loom_interleave_mode_name(interleave_mode);
  iree_host_size_t order_index = 0;
  if (iree_status_is_ok(status) &&
      interleave_mode == IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA) {
    for (iree_host_size_t candidate_index = 1;
         iree_status_is_ok(status) && candidate_index < selection_count;
         ++candidate_index) {
      status = iree_benchmark_loom_run_comparison_window(
          run, &candidates[0], baseline, module_plan, options, comparison_group,
          method, order_index++, /*repetition_index=*/0, 'A', allocator,
          event_sink, inout_failed_benchmark_count);
      for (iree_host_size_t repetition_index = 0;
           iree_status_is_ok(status) && repetition_index < repetitions;
           ++repetition_index) {
        status = iree_benchmark_loom_run_comparison_window(
            run, &candidates[candidate_index], baseline, module_plan, options,
            comparison_group, method, order_index++, repetition_index,
            (char)('A' + candidate_index), allocator, event_sink,
            inout_failed_benchmark_count);
        if (iree_status_is_ok(status)) {
          status = iree_benchmark_loom_run_comparison_window(
              run, &candidates[0], baseline, module_plan, options,
              comparison_group, method, order_index++, repetition_index + 1,
              'A', allocator, event_sink, inout_failed_benchmark_count);
        }
      }
    }
  } else if (iree_status_is_ok(status) &&
             interleave_mode == IREE_BENCHMARK_LOOM_INTERLEAVE_ROUND_ROBIN) {
    for (iree_host_size_t repetition_index = 0;
         iree_status_is_ok(status) && repetition_index < repetitions;
         ++repetition_index) {
      for (iree_host_size_t candidate_index = 0;
           iree_status_is_ok(status) && candidate_index < selection_count;
           ++candidate_index) {
        status = iree_benchmark_loom_run_comparison_window(
            run, &candidates[candidate_index], baseline, module_plan, options,
            comparison_group, method, order_index++, repetition_index,
            (char)('A' + candidate_index), allocator, event_sink,
            inout_failed_benchmark_count);
      }
    }
  }

  for (iree_host_size_t candidate_index = 1;
       iree_status_is_ok(status) && candidate_index < selection_count;
       ++candidate_index) {
    status = iree_benchmark_loom_event_sink_emit_comparison(
        event_sink, run, &candidates[0], &candidates[candidate_index],
        comparison_group, method);
  }

  iree_benchmark_loom_dispatch_comparison_candidates_deinitialize(
      candidates, selection_count, allocator);
  return status;
}

static iree_status_t iree_benchmark_loom_run_case_end_to_end_work_item(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_work_item_t* work_item,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &work_plan
           ->selected_benchmarks[work_item->representative_selection_index];
  const loom_testbench_benchmark_plan_t* benchmark_plan =
      selection->benchmark_plan;
  const iree_benchmark_loom_benchmark_policy_t* policy = &selection->policy;
  loom_testbench_case_execution_options_t benchmark_execution_options =
      *execution_options;

  iree_host_size_t correctness_sample_count = 0;
  iree_host_size_t correctness_failed_sample_count = 0;
  iree_status_t status = iree_benchmark_loom_run_work_item_correctness_range(
      run, module_plan, work_plan, work_item, &benchmark_execution_options,
      execution_arena, event_sink, &correctness_sample_count,
      &correctness_failed_sample_count);
  if (iree_status_is_ok(status)) {
    *inout_correctness_sample_count += correctness_sample_count;
    *inout_correctness_failed_sample_count += correctness_failed_sample_count;
  }

  iree_benchmark_loom_benchmark_result_t benchmark_result = {0};
  if (iree_status_is_ok(status) && correctness_failed_sample_count == 0) {
    status = iree_benchmark_loom_run_benchmark_iterations(
        module_plan, benchmark_plan, &benchmark_execution_options, policy,
        work_item->begin_benchmark_sample, work_item->end_benchmark_sample,
        execution_arena, allocator, &benchmark_result);
  }
  if (iree_status_is_ok(status) && correctness_failed_sample_count != 0) {
    benchmark_result = (iree_benchmark_loom_benchmark_result_t){
        .executed = false,
        .passed = false,
        .samples_per_iteration = correctness_sample_count,
        .failed_sample_count = correctness_failed_sample_count,
    };
    if (work_item->has_case_sample_ordinal) {
      benchmark_result.has_sample_ordinal = true;
      benchmark_result.sample_ordinal = work_item->case_sample_ordinal;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_emit_work_item_result_aliases(
        run, module_plan, work_plan, work_item, &benchmark_result,
        correctness_sample_count, correctness_failed_sample_count, event_sink,
        inout_failed_benchmark_count);
  }
  return status;
}

static iree_status_t iree_benchmark_loom_run_work_plan(
    const iree_benchmark_loom_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_hal_context_t* hal_context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    const iree_benchmark_loom_event_sink_t* event_sink,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  iree_benchmark_loom_dispatch_compile_context_t* dispatch_compile_contexts =
      NULL;
  iree_status_t status = iree_ok_status();
  if (work_plan->dispatch_compile_item_count != 0) {
    status = iree_allocator_malloc_array(
        allocator, work_plan->dispatch_compile_item_count,
        sizeof(*dispatch_compile_contexts), (void**)&dispatch_compile_contexts);
    if (iree_status_is_ok(status)) {
      memset(dispatch_compile_contexts, 0,
             work_plan->dispatch_compile_item_count *
                 sizeof(*dispatch_compile_contexts));
    }
  }

  for (iree_host_size_t work_item_index = 0;
       iree_status_is_ok(status) &&
       work_item_index < work_plan->work_item_count;
       ++work_item_index) {
    const iree_benchmark_loom_work_item_t* work_item =
        &work_plan->work_items[work_item_index];
    switch ((iree_benchmark_loom_work_item_kind_t)work_item->kind) {
      case IREE_BENCHMARK_LOOM_WORK_ITEM_CASE_END_TO_END: {
        status = iree_benchmark_loom_run_case_end_to_end_work_item(
            run, module_plan, work_plan, work_item, execution_options,
            execution_arena, allocator, event_sink,
            inout_correctness_sample_count,
            inout_correctness_failed_sample_count,
            inout_failed_benchmark_count);
        break;
      }
      case IREE_BENCHMARK_LOOM_WORK_ITEM_DISPATCH_SAMPLE: {
        if (work_item->dispatch_compile_item_index >=
            work_plan->dispatch_compile_item_count) {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "dispatch benchmark work item references compile item %" PRIhsz
              " but the work plan only has %" PRIhsz " compile items",
              work_item->dispatch_compile_item_index,
              work_plan->dispatch_compile_item_count);
          break;
        }
        status = iree_benchmark_loom_run_dispatch_sample_work_item(
            run, module_plan, work_plan, work_item, options, hal_context,
            session, filename, source, compile_report_options,
            execution_options, execution_arena, allocator, event_sink,
            &dispatch_compile_contexts[work_item->dispatch_compile_item_index],
            inout_correctness_sample_count,
            inout_correctness_failed_sample_count,
            inout_failed_benchmark_count);
        break;
      }
      case IREE_BENCHMARK_LOOM_WORK_ITEM_NONE:
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported benchmark work item kind %d",
                                  (int)work_item->kind);
        break;
    }
  }

  if (dispatch_compile_contexts != NULL) {
    for (iree_host_size_t i = 0; i < work_plan->dispatch_compile_item_count;
         ++i) {
      iree_benchmark_loom_dispatch_compile_context_deinitialize(
          &dispatch_compile_contexts[i]);
    }
  }
  iree_allocator_free(allocator, dispatch_compile_contexts);
  return status;
}

int iree_benchmark_loom_main(
    int argc, char** argv,
    const iree_benchmark_loom_configuration_t* configuration) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = iree_allocator_system();
  iree_string_builder_t command_line_json;
  iree_string_builder_initialize(allocator, &command_line_json);
  iree_status_t status = iree_benchmark_loom_append_command_line_json(
      argc, argv, &command_line_json);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    iree_string_builder_deinitialize(&command_line_json);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(1);
    return 1;
  }

  iree_benchmark_loom_set_usage(configuration->tool_name);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (FLAG_agents_md) {
    iree_benchmark_loom_print_agents_md(stdout);
    iree_string_builder_deinitialize(&command_line_json);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(0);
    return 0;
  }

  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  iree_benchmark_loom_artifact_bundle_t artifact_bundle = {0};
  iree_benchmark_loom_file_provider_t file_provider = {0};
  iree_benchmark_loom_hal_context_t hal_context = {0};
  iree_benchmark_loom_hal_context_initialize(configuration, allocator,
                                             &hal_context);
  iree_arena_allocator_t plan_arena;
  memset(&plan_arena, 0, sizeof(plan_arena));
  iree_arena_allocator_t execution_arena;
  memset(&execution_arena, 0, sizeof(execution_arena));
  iree_benchmark_loom_jsonl_sink_t jsonl_sink = {0};
  bool jsonl_sink_initialized = false;
  iree_benchmark_loom_jsonl_event_sink_t jsonl_event_sink = {0};
  iree_benchmark_loom_event_sink_t event_sink = {0};
  bool event_sink_initialized = false;
  iree_benchmark_loom_diagnostic_capture_t source_diagnostics = {0};
  iree_benchmark_loom_diagnostic_capture_initialize(allocator,
                                                    &source_diagnostics);
  iree_host_size_t planned_case_count = 0;
  iree_host_size_t planned_benchmark_count = 0;
  iree_host_size_t selected_benchmark_count = 0;
  iree_host_size_t logical_sample_count = 0;
  iree_host_size_t work_item_count = 0;
  iree_host_size_t failure_count = 0;
  iree_host_size_t failed_benchmark_count = 0;
  iree_host_size_t correctness_sample_count = 0;
  iree_host_size_t correctness_failed_sample_count = 0;
  int exit_code = 0;

  status = iree_ok_status();
  iree_benchmark_loom_options_t options = {0};
  status = iree_benchmark_loom_options_from_flags(&options);
  const bool compare_requested = !iree_string_view_is_empty(options.compare);
  if (argc > 2) {
    status = iree_status_join(
        status, iree_make_status(
                    IREE_STATUS_INVALID_ARGUMENT,
                    "iree-benchmark-loom accepts at most one input file or '-' "
                    "for stdin; got %d inputs",
                    argc - 1));
  }
  loom_run_compile_report_capture_options_t compile_report_options = {0};
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_compile_report_options_initialize(
        &options, &compile_report_options);
  }
  if (iree_status_is_ok(status)) {
    iree_benchmark_loom_artifact_bundle_options_t artifact_bundle_options = {
        .dir = options.artifact_bundle_dir,
        .policy = options.artifact_bundle_policy,
    };
    status = iree_benchmark_loom_artifact_bundle_initialize(
        &artifact_bundle_options, allocator, &artifact_bundle);
    if (iree_status_is_ok(status)) {
      hal_context.artifact_bundle = &artifact_bundle;
    }
  }

  if (iree_status_is_ok(status)) {
    loom_run_session_options_t session_options = {0};
    loom_run_session_options_initialize(&session_options);
    session_options.host_allocator = allocator;
    session_options.register_context = (loom_run_register_context_callback_t){
        .fn = iree_benchmark_loom_register_context,
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
      iree_benchmark_loom_effective_results_output_path(options.output,
                                                        &artifact_bundle);
  const iree_string_view_t profile_artifacts_dir =
      iree_benchmark_loom_effective_profile_artifacts_dir(
          options.profile_artifacts_dir, &artifact_bundle);
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_file_provider_initialize(
        filename, run_id, options.file_output_dir,
        artifact_bundle.file_output_dir, &artifact_bundle, allocator,
        &file_provider);
  }
  const iree_benchmark_loom_run_identity_t run_identity = {
      .run_id = run_id,
      .source = filename,
      .results_path = iree_string_view_is_empty(results_output_path)
                          ? IREE_SV("-")
                          : results_output_path,
      .file_output_dir = file_provider.output_dir,
      .profile_artifacts_dir = profile_artifacts_dir,
      .artifact_bundle_dir = artifact_bundle.dir,
      .artifact_bundle_policy = iree_benchmark_loom_artifact_bundle_policy_name(
          artifact_bundle.policy),
  };
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_jsonl_sink_initialize(results_output_path,
                                                       allocator, &jsonl_sink);
    if (iree_status_is_ok(status)) {
      jsonl_sink_initialized = true;
      iree_benchmark_loom_jsonl_event_sink_initialize(
          &jsonl_sink, &jsonl_event_sink, &event_sink);
      event_sink_initialized = true;
      status = iree_benchmark_loom_event_sink_emit_run(
          &event_sink, &run_identity, options.dry_run,
          options.sample_compilation_mode);
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
        .fn = iree_benchmark_loom_diagnostic_capture_sink,
        .user_data = &source_diagnostics,
    };
    status = loom_run_module_parse(&session, &parse_options, &run_module);
    if (!iree_status_is_ok(status) && source_diagnostics.error_count != 0) {
      // The diagnostic sink owns the input rejection evidence; the status only
      // carries the same non-infrastructure failure.
      iree_status_free(status);
      status = iree_benchmark_loom_event_sink_emit_failure(
          &event_sink, &run_identity, IREE_SV("parse"), IREE_SV("diagnostics"),
          IREE_SV("input module has parse errors"), &source_diagnostics);
      ++failure_count;
      exit_code = 1;
    }
  }

  if (iree_status_is_ok(status) && failure_count == 0) {
    iree_benchmark_loom_diagnostic_capture_deinitialize(&source_diagnostics);
    iree_benchmark_loom_diagnostic_capture_initialize(allocator,
                                                      &source_diagnostics);
    loom_verify_options_t verify_options = {0};
    verify_options.sink = (loom_diagnostic_sink_t){
        .fn = iree_benchmark_loom_diagnostic_capture_sink,
        .user_data = &source_diagnostics,
    };
    verify_options.max_errors = 20;
    verify_options.source_resolver =
        loom_run_module_source_resolver(&run_module);
    loom_verify_result_t verify_result = {0};
    status =
        loom_verify_module(run_module.module, &verify_options, &verify_result);
    if (iree_status_is_ok(status) && verify_result.error_count != 0) {
      status = iree_benchmark_loom_event_sink_emit_failure(
          &event_sink, &run_identity, IREE_SV("verify"), IREE_SV("diagnostics"),
          IREE_SV("input module failed verification"), &source_diagnostics);
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
    plan_options.max_samples_per_case = options.max_samples_per_case;
    loom_testbench_module_plan_t module_plan = {0};
    status = loom_testbench_plan_module(run_module.module, &plan_options,
                                        &plan_arena, &module_plan);
    if (iree_status_is_ok(status)) {
      planned_case_count = module_plan.case_count;
      planned_benchmark_count = module_plan.benchmark_count;
    }
    loom_testbench_case_execution_options_t execution_options = {0};
    loom_testbench_case_execution_options_initialize(&execution_options);
    execution_options.materializer.host_allocator = allocator;
    execution_options.materializer.open_read_file =
        (loom_testbench_file_open_callback_t){
            .fn = iree_benchmark_loom_open_file_for_read,
            .user_data = &file_provider,
        };
    execution_options.materializer.open_write_file =
        (loom_testbench_file_open_callback_t){
            .fn = iree_benchmark_loom_open_file_for_write,
            .user_data = &file_provider,
        };

    iree_benchmark_loom_work_plan_t work_plan = {0};
    bool work_plan_initialized = false;
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_work_plan_initialize(&module_plan, &options,
                                                        allocator, &work_plan);
      if (iree_status_is_ok(status)) {
        work_plan_initialized = true;
        selected_benchmark_count = work_plan.selected_benchmark_count;
        logical_sample_count = work_plan.logical_sample_count;
        work_item_count = work_plan.work_item_count;
      }
    }

    if (iree_status_is_ok(status) && compare_requested) {
      const iree_benchmark_loom_selected_benchmark_t* selections =
          work_plan.selected_benchmarks;
      const iree_host_size_t selection_count =
          work_plan.selected_benchmark_count;
      for (iree_host_size_t i = 0;
           iree_status_is_ok(status) && i < selection_count; ++i) {
        if (selections[i].policy.measure_kind !=
            IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
          status =
              iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                               "--compare benchmark `%.*s` must use measure = "
                               "\"dispatch_complete\"",
                               (int)selections[i].benchmark_plan->name.size,
                               selections[i].benchmark_plan->name.data);
          break;
        }
        status = iree_benchmark_loom_event_sink_emit_plan(
            &event_sink, &run_identity, module_plan.module, &selections[i],
            &options, options.sample_compilation_mode);
        if (iree_status_is_ok(status) && options.dry_run) {
          status = iree_benchmark_loom_event_sink_emit_device(
              &event_sink, &run_identity, &hal_context);
        }
      }
      if (iree_status_is_ok(status) && !options.dry_run) {
        status = iree_benchmark_loom_run_dispatch_comparison(
            &run_identity, selections, selection_count, &module_plan, &options,
            options.sample_compilation_mode, options.interleave_mode,
            options.repetitions, &hal_context, &session, filename, source,
            &compile_report_options, &execution_options, &execution_arena,
            allocator, &event_sink, &correctness_sample_count,
            &correctness_failed_sample_count, &failed_benchmark_count);
      }
    }

    for (iree_host_size_t selection_index = 0;
         iree_status_is_ok(status) && !compare_requested &&
         selection_index < work_plan.selected_benchmark_count;
         ++selection_index) {
      const iree_benchmark_loom_selected_benchmark_t* selection =
          &work_plan.selected_benchmarks[selection_index];
      status = iree_benchmark_loom_event_sink_emit_plan(
          &event_sink, &run_identity, module_plan.module, selection, &options,
          options.sample_compilation_mode);
      if (iree_status_is_ok(status) && options.dry_run &&
          selection->policy.measure_kind ==
              IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
        status = iree_benchmark_loom_event_sink_emit_device(
            &event_sink, &run_identity, &hal_context);
      }
    }
    if (iree_status_is_ok(status) && !compare_requested && !options.dry_run) {
      status = iree_benchmark_loom_run_work_plan(
          &run_identity, &module_plan, &work_plan, &options, &hal_context,
          &session, filename, source, &compile_report_options,
          &execution_options, &execution_arena, allocator, &event_sink,
          &correctness_sample_count, &correctness_failed_sample_count,
          &failed_benchmark_count);
    }
    if (iree_status_is_ok(status) &&
        (failed_benchmark_count != 0 || correctness_failed_sample_count != 0)) {
      exit_code = 1;
    }
    if (work_plan_initialized) {
      iree_benchmark_loom_work_plan_deinitialize(&work_plan);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_event_sink_emit_summary(
        &event_sink, &run_identity, &artifact_bundle, planned_case_count,
        planned_benchmark_count, selected_benchmark_count, logical_sample_count,
        work_item_count, failure_count, failed_benchmark_count,
        correctness_sample_count, correctness_failed_sample_count,
        options.dry_run, options.sample_compilation_mode);
  }
  if (iree_status_is_ok(status) && jsonl_sink_initialized) {
    status = iree_benchmark_loom_jsonl_sink_close(&jsonl_sink);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_write_artifact_bundle_manifest(
        &artifact_bundle, &run_identity, &hal_context, source,
        iree_string_builder_view(&command_line_json), options.dry_run,
        options.sample_compilation_mode, allocator);
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

  iree_benchmark_loom_diagnostic_capture_deinitialize(&source_diagnostics);
  if (event_sink_initialized) {
    iree_benchmark_loom_jsonl_event_sink_deinitialize(&jsonl_event_sink);
  }
  if (jsonl_sink_initialized) {
    iree_benchmark_loom_jsonl_sink_deinitialize(&jsonl_sink);
  }
  iree_arena_deinitialize(&execution_arena);
  iree_arena_deinitialize(&plan_arena);
  loom_run_module_deinitialize(&run_module);
  iree_benchmark_loom_hal_context_deinitialize(&hal_context);
  iree_benchmark_loom_file_provider_deinitialize(&file_provider);
  iree_benchmark_loom_artifact_bundle_deinitialize(&artifact_bundle);
  iree_io_file_contents_free(contents);
  loom_run_session_deinitialize(&session);
  iree_string_builder_deinitialize(&command_line_json);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
