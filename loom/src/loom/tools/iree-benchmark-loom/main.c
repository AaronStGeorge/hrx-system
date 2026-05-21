// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-benchmark-loom: correctness-gated benchmark runner for check.benchmark.

#include "loom/tools/iree-benchmark-loom/main.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "loom/ir/module.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tools/iree-benchmark-loom/comparison_execution.h"
#include "loom/tools/iree-benchmark-loom/context.h"
#include "loom/tools/iree-benchmark-loom/diagnostics.h"
#include "loom/tools/iree-benchmark-loom/event.h"
#include "loom/tools/iree-benchmark-loom/help.h"
#include "loom/tools/iree-benchmark-loom/manifest.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/module_query.h"
#include "loom/tools/iree-benchmark-loom/options.h"
#include "loom/tools/iree-benchmark-loom/output.h"
#include "loom/tools/iree-benchmark-loom/session.h"
#include "loom/tools/iree-benchmark-loom/snapshot.h"
#include "loom/tools/iree-benchmark-loom/work_execution.h"
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
          "Output path for benchmark results. Empty or '-' writes to stdout.");
IREE_FLAG(string, output_format, "snapshot",
          "Benchmark result output format. Use 'snapshot' for one compact JSON "
          "document or 'jsonl' for newline-delimited lifecycle events.");
IREE_FLAG(string, file_output_dir, "",
          "Directory receiving check.file.write.* outputs. Empty uses "
          "$TMPDIR/iree-loom-benchmark/<source>_<hash>/ for this run.");
IREE_FLAG(string, artifact_bundle_dir, "",
          "Directory receiving a self-contained run bundle. When set and "
          "--output is empty, results are written inside the bundle; "
          "check.file.write outputs and profile artifacts default to "
          "bundle subdirectories unless their explicit flags are set.");
IREE_FLAG(string, artifact_bundle_policy, "minimal",
          "Artifact bundle policy when --artifact_bundle_dir is set. Use "
          "'minimal', 'debug', or 'full'.");
IREE_FLAG(bool, dry_run, false,
          "Reports selected logical benchmarks and deduplicated physical work "
          "items without running correctness, compilation, or measurement.");
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
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_parse_output_format(
      iree_make_cstring_view(FLAG_output_format), &out_options->output_format));
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

static iree_status_t iree_benchmark_loom_compile_report_options_initialize(
    const iree_benchmark_loom_options_t* options,
    loom_run_compile_report_capture_options_t* out_options) {
  loom_run_compile_report_capture_options_initialize(out_options);
  IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_options_parse_request(
      options->compile_report, out_options));
  if (out_options->sink_format == LOOM_RUN_COMPILE_REPORT_SINK_FORMAT_TEXT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-benchmark-loom emits structured JSON reports; use "
        "--compile_report=summary, details, json-summary, or json-details");
  }
  out_options->row_limit = options->compile_report_row_limit;
  return iree_ok_status();
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
  iree_benchmark_loom_snapshot_sink_t snapshot_sink = {0};
  bool snapshot_sink_initialized = false;
  iree_benchmark_loom_event_sink_t event_sink = {0};
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
        .output_format = options.output_format,
    };
    status = iree_benchmark_loom_artifact_bundle_initialize(
        &artifact_bundle_options, allocator, &artifact_bundle);
    if (iree_status_is_ok(status)) {
      hal_context.artifact_bundle = &artifact_bundle;
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_session_initialize(configuration, allocator,
                                                    &session);
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
    switch (options.output_format) {
      case IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_SNAPSHOT:
        status = iree_benchmark_loom_snapshot_sink_initialize(allocator,
                                                              &snapshot_sink);
        if (iree_status_is_ok(status)) {
          snapshot_sink_initialized = true;
          iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot_sink,
                                                             &event_sink);
        }
        break;
      case IREE_BENCHMARK_LOOM_OUTPUT_FORMAT_JSONL:
        status = iree_benchmark_loom_jsonl_sink_initialize(
            results_output_path, allocator, &jsonl_sink);
        if (iree_status_is_ok(status)) {
          jsonl_sink_initialized = true;
          iree_benchmark_loom_jsonl_event_sink_initialize(
              &jsonl_sink, &jsonl_event_sink, &event_sink);
        }
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "unsupported benchmark output format %d",
                                  (int)options.output_format);
        break;
    }
    if (iree_status_is_ok(status)) {
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
        const iree_benchmark_loom_comparison_execution_options_t
            comparison_execution_options = {
                .run = &run_identity,
                .selections = selections,
                .selection_count = selection_count,
                .module_plan = &module_plan,
                .benchmark_options = &options,
                .sample_compilation_mode = options.sample_compilation_mode,
                .interleave_mode = options.interleave_mode,
                .repetitions = options.repetitions,
                .hal_context = &hal_context,
                .session = &session,
                .filename = filename,
                .source = source,
                .compile_report_options = &compile_report_options,
                .case_execution_options = &execution_options,
                .execution_arena = &execution_arena,
                .host_allocator = allocator,
                .event_sink = &event_sink,
            };
        status = iree_benchmark_loom_run_dispatch_comparison(
            &comparison_execution_options, &correctness_sample_count,
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
    if (iree_status_is_ok(status) && options.dry_run) {
      status = iree_benchmark_loom_event_sink_emit_work_plan(
          &event_sink, &run_identity, module_plan.module, &work_plan);
    }
    if (iree_status_is_ok(status) && !compare_requested && !options.dry_run) {
      const iree_benchmark_loom_work_plan_execution_options_t
          work_execution_options = {
              .run = &run_identity,
              .module_plan = &module_plan,
              .work_plan = &work_plan,
              .benchmark_options = &options,
              .hal_context = &hal_context,
              .session = &session,
              .filename = filename,
              .source = source,
              .compile_report_options = &compile_report_options,
              .case_execution_options = &execution_options,
              .execution_arena = &execution_arena,
              .host_allocator = allocator,
              .event_sink = &event_sink,
          };
      status = iree_benchmark_loom_run_work_plan(
          &work_execution_options, &correctness_sample_count,
          &correctness_failed_sample_count, &failed_benchmark_count);
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
  if (iree_status_is_ok(status) && snapshot_sink_initialized) {
    status = iree_benchmark_loom_snapshot_sink_write(&snapshot_sink,
                                                     results_output_path);
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
  if (jsonl_sink_initialized) {
    iree_benchmark_loom_jsonl_event_sink_deinitialize(&jsonl_event_sink);
  }
  if (snapshot_sink_initialized) {
    iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot_sink);
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
