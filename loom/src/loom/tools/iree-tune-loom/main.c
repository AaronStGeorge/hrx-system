// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-tune-loom: correctness-gated benchmark runner for check.benchmark.

#include "loom/tools/iree-tune-loom/main.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/tooling/device_util.h"
#include "iree/vm/api.h"
#include "loom/error/json_sink.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/hal_backend.h"
#include "loom/tooling/execution/hal_benchmark.h"
#include "loom/tooling/execution/hal_candidate.h"
#include "loom/tooling/execution/hal_invocation.h"
#include "loom/tooling/execution/hal_runtime.h"
#include "loom/tooling/execution/one_shot.h"
#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

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
IREE_FLAG(bool, dry_run, false,
          "Emits plan rows for selected benchmarks and stops before "
          "correctness, compilation, and measurement.");
IREE_FLAG(bool, agents_md, false,
          "Prints a compact Markdown snippet suitable for AGENTS.md and "
          "exits.");
IREE_FLAG(string, compile_report, "summary",
          "Compile report embedded in benchmark rows. Use 'summary', "
          "'details', or empty/'none'.");
IREE_FLAG(int32_t, compile_report_row_limit,
          LOOM_RUN_COMPILE_REPORT_DEFAULT_ROW_LIMIT,
          "Maximum pressure, spill, and source-low rows to capture for "
          "--compile_report=details.");

typedef struct iree_tune_loom_i32_flag_t {
  // Current flag value.
  int32_t value;
  // True when the flag was provided on the command line or in a flagfile.
  bool specified;
} iree_tune_loom_i32_flag_t;

typedef struct iree_tune_loom_bool_flag_t {
  // Current flag value.
  bool value;
  // True when the flag was provided on the command line or in a flagfile.
  bool specified;
} iree_tune_loom_bool_flag_t;

static iree_status_t iree_tune_loom_parse_i32_flag(iree_string_view_t flag_name,
                                                   void* storage,
                                                   iree_string_view_t value) {
  iree_tune_loom_i32_flag_t* flag = (iree_tune_loom_i32_flag_t*)storage;
  char* value_end = NULL;
  int64_t parsed_value = 0;
  if (!iree_string_view_is_empty(value)) {
    errno = 0;
    parsed_value = strtoll(value.data, &value_end, 10);
    if (errno == ERANGE) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "flag '--%.*s' value '%.*s' is outside int64 "
                              "range",
                              (int)flag_name.size, flag_name.data,
                              (int)value.size, value.data);
    }
    if (value_end != value.data + value.size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "flag '--%.*s' must be an int32 value; got '%.*s'",
          (int)flag_name.size, flag_name.data, (int)value.size, value.data);
    }
  }
  if (parsed_value < INT32_MIN || parsed_value > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "flag '--%.*s' value %" PRIi64
                            " is outside int32 range",
                            (int)flag_name.size, flag_name.data, parsed_value);
  }
  flag->value = (int32_t)parsed_value;
  flag->specified = true;
  return iree_ok_status();
}

static void iree_tune_loom_print_i32_flag(iree_string_view_t flag_name,
                                          void* storage, FILE* file) {
  const iree_tune_loom_i32_flag_t* flag =
      (const iree_tune_loom_i32_flag_t*)storage;
  fprintf(file, "--%.*s=%" PRId32 "\n", (int)flag_name.size, flag_name.data,
          flag->value);
}

static iree_status_t iree_tune_loom_parse_bool_flag(
    iree_string_view_t flag_name, void* storage, iree_string_view_t value) {
  iree_tune_loom_bool_flag_t* flag = (iree_tune_loom_bool_flag_t*)storage;
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("true")) ||
      iree_string_view_equal(value, IREE_SV("1"))) {
    flag->value = true;
  } else if (iree_string_view_equal(value, IREE_SV("false")) ||
             iree_string_view_equal(value, IREE_SV("0"))) {
    flag->value = false;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flag '--%.*s' must be a bool value; got '%.*s'",
                            (int)flag_name.size, flag_name.data,
                            (int)value.size, value.data);
  }
  flag->specified = true;
  return iree_ok_status();
}

static void iree_tune_loom_print_bool_flag(iree_string_view_t flag_name,
                                           void* storage, FILE* file) {
  const iree_tune_loom_bool_flag_t* flag =
      (const iree_tune_loom_bool_flag_t*)storage;
  fprintf(file, "--%.*s=%s\n", (int)flag_name.size, flag_name.data,
          flag->value ? "true" : "false");
}

static void iree_tune_loom_print_agents_md(FILE* file) {
  fprintf(file,
          "## iree-tune-loom\n"
          "\n"
          "Use `iree-tune-loom` when a `.loom` file declares "
          "`check.benchmark` records and you need correctness-gated timing, "
          "compile diagnostics, or HAL dispatch evidence for kernel tuning. "
          "The tool emits JSONL: every row has `row` and `run_id`; selected "
          "benchmarks also have `candidate_id` so rows can be joined across "
          "large sweeps.\n"
          "\n"
          "Examples:\n"
          "\n"
          "```bash\n"
          "iree-bazel-run //loom/src/loom/tools/iree-tune-loom -- "
          "path/to/file.loom --benchmark=@kernel_latency --dry_run "
          "--output=plan.jsonl\n"
          "```\n"
          "\n"
          "```bash\n"
          "iree-bazel-run //loom/src/loom/tools/iree-tune-loom -- "
          "path/to/file.loom --device=amdgpu --benchmark=@kernel_latency "
          "--batch_size=64 --iterations=16 --warmup_iterations=4 "
          "--min_time_ms=100 --profile_final_batch=true "
          "--output=results.jsonl\n"
          "```\n"
          "\n"
          "```bash\n"
          "jq 'select(.row==\"benchmark\") | .benchmark_result | "
          "{benchmark,status,p50:.dispatch_timing_ns.p50,stop_reason}' "
          "results.jsonl\n"
          "jq 'select(.row==\"compile\" and .status!=\"ok\") | "
          ".diagnostics[]?' results.jsonl\n"
          "```\n"
          "\n"
          "Use `iree-tune-loom --help` for the full flag list, JSONL row "
          "schema notes, and more `jq` recipes.\n");
}

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

typedef enum iree_tune_loom_measure_e {
  // Full testbench case execution wall time.
  IREE_TUNE_LOOM_MEASURE_CASE_END_TO_END = 0,
  // Prepared HAL dispatch completion timing with correctness outside timing.
  IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE = 1,
} iree_tune_loom_measure_t;

typedef struct iree_tune_loom_benchmark_policy_t {
  // Parsed measurement mode.
  iree_tune_loom_measure_t measure_kind;
  // Declared measurement mode.
  iree_string_view_t measure;
  // Number of warmup iterations run before measured timing.
  iree_host_size_t warmup_iterations;
  // Number of measured iterations.
  iree_host_size_t iterations;
  // HAL benchmark options for dispatch_complete measurement.
  loom_run_hal_benchmark_options_t hal_options;
} iree_tune_loom_benchmark_policy_t;

typedef struct iree_tune_loom_diagnostic_capture_t {
  // JSON array entries for diagnostics emitted by this candidate compile.
  iree_string_builder_t output;
  // Output stream backed by |output|.
  loom_output_stream_t stream;
  // True after |output| has been initialized.
  bool initialized;
  // True until the first diagnostic has been written.
  bool first_diagnostic;
  // Number of error diagnostics captured.
  iree_host_size_t error_count;
  // Number of warning diagnostics captured.
  iree_host_size_t warning_count;
  // Number of remark diagnostics captured.
  iree_host_size_t remark_count;
} iree_tune_loom_diagnostic_capture_t;

typedef struct iree_tune_loom_device_row_state_t {
  // True once a selected-device row has been appended.
  bool appended;
} iree_tune_loom_device_row_state_t;

typedef struct iree_tune_loom_run_identity_t {
  // Process-local run identifier copied into every JSONL row.
  iree_string_view_t run_id;
  // User-provided source path, or "<stdin>" when reading standard input.
  iree_string_view_t source;
} iree_tune_loom_run_identity_t;

typedef struct iree_tune_loom_candidate_identity_t {
  // Deterministic candidate identifier within the source/run selection.
  iree_string_view_t candidate_id;
  // Zero-based selected benchmark ordinal within this run.
  iree_host_size_t candidate_index;
} iree_tune_loom_candidate_identity_t;

typedef struct iree_tune_loom_timing_stats_t {
  // Number of measured timing samples.
  iree_host_size_t count;
  // Sum of measured durations in nanoseconds.
  int64_t total_ns;
  // Minimum measured duration in nanoseconds.
  int64_t minimum_ns;
  // Maximum measured duration in nanoseconds.
  int64_t maximum_ns;
  // Arithmetic mean of measured durations in nanoseconds.
  double mean_ns;
  // Nearest-rank median duration in nanoseconds.
  int64_t p50_ns;
  // Nearest-rank p90 duration in nanoseconds.
  int64_t p90_ns;
} iree_tune_loom_timing_stats_t;

typedef struct iree_tune_loom_benchmark_result_t {
  // Stable benchmark status string when it differs from the default.
  iree_string_view_t status;
  // True when failure fields below describe why the benchmark did not run.
  bool has_failure;
  // Product stage that failed: compile, prepare, benchmark, etc.
  iree_string_view_t failure_stage;
  // Failure kind within |failure_stage|.
  iree_string_view_t failure_kind;
  // Optional human-facing failure message.
  iree_string_view_t failure_message;
  // Number of error diagnostics associated with the failure.
  iree_host_size_t diagnostic_error_count;
  // Number of warning diagnostics associated with the failure.
  iree_host_size_t diagnostic_warning_count;
  // Number of remark diagnostics associated with the failure.
  iree_host_size_t diagnostic_remark_count;
  // JSON array entries for structured diagnostics associated with the failure.
  iree_string_view_t diagnostic_json;
  // Captured structured compile report for the benchmark candidate.
  const loom_run_compile_report_capture_t* compile_report_capture;
  // True when benchmark setup and timing completed.
  bool executed;
  // True when no measured or warmup sample failed expectations.
  bool passed;
  // True when |sample_ordinal| identifies the measured sample.
  bool has_sample_ordinal;
  // Concrete sample ordinal measured by dispatch_complete.
  iree_host_size_t sample_ordinal;
  // Number of case samples run per benchmark iteration.
  iree_host_size_t samples_per_iteration;
  // Failed sample executions observed during warmup and measured iterations.
  iree_host_size_t failed_sample_count;
  // Timing summary for measured iterations.
  iree_tune_loom_timing_stats_t timing;
  // True when |hal_benchmark| contains dispatch_complete evidence.
  bool has_hal_benchmark;
  // HAL batch benchmark evidence.
  loom_run_hal_benchmark_result_t hal_benchmark;
} iree_tune_loom_benchmark_result_t;

typedef struct iree_tune_loom_hal_context_t {
  // Tool configuration with linked backend registries.
  const iree_tune_loom_configuration_t* configuration;
  // Host allocator used for runtime and candidate storage.
  iree_allocator_t host_allocator;
  // Selected HAL backend for dispatch_complete benchmarks.
  const loom_run_hal_backend_t* backend;
  // Shared HAL runtime used by selected dispatch_complete benchmarks.
  loom_run_hal_runtime_t runtime;
  // True when |runtime| owns initialized HAL state.
  bool runtime_initialized;
} iree_tune_loom_hal_context_t;

typedef struct iree_tune_loom_hal_actual_provider_t {
  // Shared HAL runtime and backend state.
  iree_tune_loom_hal_context_t* context;
  // Execution session used to parse the compile copy.
  loom_run_session_t* session;
  // Source filename used for diagnostics.
  iree_string_view_t filename;
  // Source text used to parse a private compile copy.
  iree_string_view_t source;
  // User-selected pass pipeline.
  iree_string_view_t pipeline;
  // Module that owns the invocation plan passed by the testbench executor.
  const loom_module_t* test_module;
  // Actual invocation selected for this benchmark.
  const loom_testbench_invocation_plan_t* actual_invocation;
  // Parsed compile module owned by this provider.
  loom_run_module_t compile_module;
  // Backend-produced HAL executable candidate.
  loom_run_hal_candidate_t candidate;
  // Prepared executable retained for correctness and benchmark dispatches.
  loom_run_hal_prepared_candidate_t prepared_candidate;
  // Dispatch options derived from the compiled source entry.
  loom_run_hal_invocation_options_t invocation_options;
  // Structured diagnostics emitted while compiling this candidate.
  iree_tune_loom_diagnostic_capture_t diagnostics;
  // Structured compile report populated while emitting this candidate.
  loom_run_compile_report_capture_t compile_report_capture;
  // Pass diagnostic counts from the Loom compile pipeline.
  loom_pass_run_result_t pass_result;
  // True when compile completed with product diagnostics rather than an
  // infrastructure failure.
  bool compile_rejected;
  // Product stage that rejected the compile.
  iree_string_view_t compile_failure_stage;
  // Product reason for |compile_rejected|.
  iree_string_view_t compile_failure_kind;
  // Human-facing failure message when no structured diagnostic exists.
  iree_string_view_t compile_failure_message;
  // True when |compile_module| has been initialized.
  bool compile_module_initialized;
  // True when |candidate| has been initialized.
  bool candidate_initialized;
  // True when |prepared_candidate| has been initialized.
  bool prepared_candidate_initialized;
  // True when |compile_report_capture| owns initialized capture state.
  bool compile_report_capture_initialized;
  // True when HAL candidate emission populated |compile_report_capture|.
  bool compile_report_available;
} iree_tune_loom_hal_actual_provider_t;

static void iree_tune_loom_diagnostic_capture_initialize(
    iree_allocator_t allocator, iree_tune_loom_diagnostic_capture_t* capture) {
  *capture = (iree_tune_loom_diagnostic_capture_t){
      .initialized = true,
      .first_diagnostic = true,
  };
  iree_string_builder_initialize(allocator, &capture->output);
  loom_output_stream_for_builder(&capture->output, &capture->stream);
}

static void iree_tune_loom_diagnostic_capture_deinitialize(
    iree_tune_loom_diagnostic_capture_t* capture) {
  if (capture == NULL || !capture->initialized) {
    return;
  }
  iree_string_builder_deinitialize(&capture->output);
  *capture = (iree_tune_loom_diagnostic_capture_t){0};
}

static iree_string_view_t iree_tune_loom_diagnostic_capture_json(
    const iree_tune_loom_diagnostic_capture_t* capture) {
  if (!capture || !capture->initialized) {
    return iree_string_view_empty();
  }
  return iree_string_builder_view(&capture->output);
}

static iree_status_t iree_tune_loom_write_diagnostic_array_json(
    const iree_tune_loom_diagnostic_capture_t* capture,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(
      stream, iree_tune_loom_diagnostic_capture_json(capture)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_diagnostic_capture_sink(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  iree_tune_loom_diagnostic_capture_t* capture =
      (iree_tune_loom_diagnostic_capture_t*)user_data;
  if (!capture->first_diagnostic) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&capture->stream, ","));
  }
  IREE_RETURN_IF_ERROR(loom_diagnostic_json_write_object(
      &capture->stream, diagnostic,
      (loom_type_formatter_t){loom_type_format_minimal, NULL}));
  capture->first_diagnostic = false;
  switch (diagnostic->severity) {
    case LOOM_DIAGNOSTIC_ERROR:
      ++capture->error_count;
      break;
    case LOOM_DIAGNOSTIC_WARNING:
      ++capture->warning_count;
      break;
    case LOOM_DIAGNOSTIC_REMARK:
      ++capture->remark_count;
      break;
    default:
      break;
  }
  return iree_ok_status();
}

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

static iree_string_view_t iree_tune_loom_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t iree_tune_loom_normalize_symbol_name(
    iree_string_view_t symbol_name) {
  symbol_name = iree_string_view_trim(symbol_name);
  if (iree_string_view_starts_with(symbol_name, IREE_SV("@"))) {
    return iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);
  }
  return symbol_name;
}

static iree_string_view_t iree_tune_loom_device_uri_driver_name(
    iree_string_view_t device_uri) {
  iree_string_view_t driver_name = iree_string_view_empty();
  iree_string_view_split(device_uri, ':', &driver_name, NULL);
  return driver_name;
}

static iree_status_t iree_tune_loom_hal_context_select_backend(
    iree_tune_loom_hal_context_t* context);

static iree_string_view_t iree_tune_loom_selected_device_uri(
    const iree_tune_loom_hal_context_t* context) {
  const iree_string_view_list_t device_uris = iree_hal_device_flag_list();
  if (device_uris.count == 1) {
    return device_uris.values[0];
  }
  if (context->backend != NULL) {
    return context->backend->hal_driver_name;
  }
  return iree_string_view_empty();
}

static iree_status_t iree_tune_loom_write_status_object_json(
    iree_status_t status, loom_output_stream_t* stream) {
  const iree_status_code_t code = iree_status_code(status);
  char message[512];
  iree_host_size_t required_length = 0;
  if (!iree_status_format(status, sizeof(message), message, &required_length)) {
    const char* status_string = iree_status_code_string(code);
    iree_host_size_t index = 0;
    for (; status_string[index] != '\0' && index + 1 < sizeof(message);
         ++index) {
      message[index] = status_string[index];
    }
    message[index] = '\0';
    required_length = index;
  }
  if (required_length >= sizeof(message)) {
    required_length = sizeof(message) - 1;
  }

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(stream, "\"code\":%d", (int)code));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"status\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, iree_make_cstring_view(iree_status_code_string(code))));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"message\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, iree_make_string_view(message, required_length)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  iree_status_free(status);
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_write_optional_i64_query_json(
    iree_hal_device_t* device, iree_string_view_t category,
    iree_string_view_t key, const char* field_name,
    loom_output_stream_t* stream) {
  int64_t value = 0;
  iree_status_t status =
      iree_hal_device_query_i64(device, category, key, &value);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, field_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ":"));
  if (iree_status_is_ok(status)) {
    return loom_output_stream_write_format(stream, "%" PRIi64, value);
  }
  return iree_tune_loom_write_status_object_json(status, stream);
}

static iree_status_t iree_tune_loom_write_hex_bytes_json(
    const uint8_t* bytes, iree_host_size_t byte_count,
    loom_output_stream_t* stream) {
  static const char kHexDigits[] = "0123456789abcdef";
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\""));
  for (iree_host_size_t i = 0; i < byte_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_char(stream, kHexDigits[bytes[i] >> 4]));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_char(stream, kHexDigits[bytes[i] & 0x0F]));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\""));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_write_run_id_field_json(
    const iree_tune_loom_run_identity_t* run, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"run_id\":"));
  return loom_json_write_escaped_string(stream, run->run_id);
}

static iree_status_t iree_tune_loom_write_candidate_identity_json(
    const iree_tune_loom_candidate_identity_t* candidate,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"candidate_id\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, candidate->candidate_id));
  return loom_output_stream_write_format(
      stream, ",\"candidate_index\":%" PRIhsz, candidate->candidate_index);
}

static iree_status_t iree_tune_loom_append_run_row(
    const iree_tune_loom_run_identity_t* run, bool dry_run,
    iree_string_builder_t* output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"run\""));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, ",\"tool\":\"iree-tune-loom\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"source\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, run->source));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"dry_run\":%s", dry_run ? "true" : "false"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_append_device_row(
    const iree_tune_loom_run_identity_t* run,
    iree_tune_loom_hal_context_t* context,
    iree_tune_loom_device_row_state_t* row_state,
    iree_string_builder_t* device_output) {
  if (row_state->appended) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_tune_loom_hal_context_select_backend(context));

  loom_output_stream_t stream;
  loom_output_stream_for_builder(device_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"device\""));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"device_uri\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_tune_loom_selected_device_uri(context)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"driver\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, context->backend->hal_driver_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"backend\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, context->backend->name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target_family\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, context->backend->target_family_name));
  if (context->runtime_initialized && context->runtime.device != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"status\":\"created\""));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"device_id\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, iree_hal_device_id(context->runtime.device)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"queries\":{"));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, "\"attempted\":true"));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_optional_i64_query_json(
        context->runtime.device, IREE_SV("hal.device"), IREE_SV("concurrency"),
        "hal_device_concurrency", &stream));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_optional_i64_query_json(
        context->runtime.device, IREE_SV("hal.dispatch"),
        IREE_SV("concurrency"), "hal_dispatch_concurrency", &stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));

    iree_hal_device_capabilities_t capabilities = {0};
    iree_status_t capabilities_status = iree_hal_device_query_capabilities(
        context->runtime.device, &capabilities);
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"capabilities\":"));
    if (iree_status_is_ok(capabilities_status)) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream,
          "{\"flags\":%" PRIu64 ",\"numa_node\":%" PRIu8
          ",\"has_physical_device_uuid\":%s,\"device_group_index\":%" PRIu32
          ",\"has_device_group\":%s",
          capabilities.flags, capabilities.numa_node,
          capabilities.has_physical_device_uuid ? "true" : "false",
          capabilities.device_group_index,
          capabilities.has_device_group ? "true" : "false"));
      if (capabilities.has_physical_device_uuid) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
            &stream, ",\"physical_device_uuid\":"));
        IREE_RETURN_IF_ERROR(iree_tune_loom_write_hex_bytes_json(
            capabilities.physical_device_uuid,
            IREE_ARRAYSIZE(capabilities.physical_device_uuid), &stream));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
    } else {
      IREE_RETURN_IF_ERROR(iree_tune_loom_write_status_object_json(
          capabilities_status, &stream));
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"status\":\"planned\""));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  row_state->appended = true;
  return iree_ok_status();
}

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

static iree_status_t iree_tune_loom_module_symbol_name_from_ref(
    const loom_module_t* module, loom_symbol_ref_t ref,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol ref %u is outside the module symbol table",
                            (unsigned)ref.symbol_id);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol ref %u has an invalid name",
                            (unsigned)ref.symbol_id);
  }
  *out_name = module->strings.entries[symbol->name_id];
  return iree_ok_status();
}

static iree_hal_buffer_params_t iree_tune_loom_host_visible_buffer_params(
    void) {
  return (iree_hal_buffer_params_t){
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
  };
}

static const loom_named_attr_t* iree_tune_loom_find_named_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (iree_string_view_equal(
            iree_tune_loom_module_string(module, attr->name_id), name)) {
      return attr;
    }
  }
  return NULL;
}

static iree_status_t iree_tune_loom_read_optional_i64_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, int64_t default_value, int64_t* out_value) {
  const loom_named_attr_t* attr =
      iree_tune_loom_find_named_attr(module, attrs, name);
  if (!attr) {
    *out_value = default_value;
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark attr '%.*s' must be i64", (int)name.size,
                            name.data);
  }
  *out_value = attr->value.i64;
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_read_optional_bool_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, bool default_value, bool* out_value) {
  const loom_named_attr_t* attr =
      iree_tune_loom_find_named_attr(module, attrs, name);
  if (!attr) {
    *out_value = default_value;
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_BOOL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark attr '%.*s' must be bool",
                            (int)name.size, name.data);
  }
  *out_value = loom_attr_as_bool(attr->value);
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_read_i64_policy_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, const iree_tune_loom_i32_flag_t* command_line_flag,
    int64_t* out_value) {
  if (command_line_flag->specified) {
    *out_value = command_line_flag->value;
    return iree_ok_status();
  }
  return iree_tune_loom_read_optional_i64_attr(
      module, attrs, name, command_line_flag->value, out_value);
}

static iree_status_t iree_tune_loom_read_bool_policy_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name,
    const iree_tune_loom_bool_flag_t* command_line_flag, bool* out_value) {
  if (command_line_flag->specified) {
    *out_value = command_line_flag->value;
    return iree_ok_status();
  }
  return iree_tune_loom_read_optional_bool_attr(
      module, attrs, name, command_line_flag->value, out_value);
}

static iree_status_t iree_tune_loom_read_optional_string_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, iree_string_view_t default_value,
    iree_string_view_t* out_value) {
  const loom_named_attr_t* attr =
      iree_tune_loom_find_named_attr(module, attrs, name);
  if (!attr) {
    *out_value = default_value;
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark attr '%.*s' must be string",
                            (int)name.size, name.data);
  }
  *out_value =
      iree_tune_loom_module_string(module, loom_attr_as_string_id(attr->value));
  return iree_ok_status();
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
    if (profile_final_batch) {
      out_policy->hal_options.flags |=
          LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_compile_report_options_initialize(
    loom_run_compile_report_capture_options_t* out_options) {
  loom_run_compile_report_capture_options_initialize(out_options);
  IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_options_parse_mode(
      iree_make_cstring_view(FLAG_compile_report), out_options));
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
      .host_allocator = host_allocator,
  };
}

static void iree_tune_loom_hal_context_deinitialize(
    iree_tune_loom_hal_context_t* context) {
  if (!context) {
    return;
  }
  if (context->runtime_initialized) {
    loom_run_hal_runtime_deinitialize(&context->runtime);
  }
  *context = (iree_tune_loom_hal_context_t){0};
}

static iree_status_t iree_tune_loom_hal_context_select_backend(
    iree_tune_loom_hal_context_t* context) {
  if (context->backend != NULL) {
    return iree_ok_status();
  }
  const loom_run_hal_backend_registry_t* registry =
      context->configuration->hal_backend_registry;
  if (registry == NULL || registry->backend_count == 0) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "dispatch_complete benchmarks require a linked HAL backend");
  }

  const iree_string_view_list_t device_uris = iree_hal_device_flag_list();
  if (device_uris.count > 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "dispatch_complete V0 supports exactly one --device= URI; got %" PRIhsz,
        device_uris.count);
  }
  if (device_uris.count == 1) {
    const iree_string_view_t driver_name =
        iree_tune_loom_device_uri_driver_name(device_uris.values[0]);
    for (iree_host_size_t i = 0; i < registry->backend_count; ++i) {
      const loom_run_hal_backend_t* backend = registry->backends[i];
      if (iree_string_view_equal(backend->hal_driver_name, driver_name)) {
        context->backend = backend;
        return iree_ok_status();
      }
    }
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--device=%.*s selects HAL driver `%.*s`, but no linked Loom HAL "
        "backend can emit for that driver",
        (int)device_uris.values[0].size, device_uris.values[0].data,
        (int)driver_name.size, driver_name.data);
  }

  if (registry->backend_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch_complete benchmarks require --device= when %" PRIhsz
        " HAL backends are linked",
        registry->backend_count);
  }
  context->backend = registry->backends[0];
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_hal_context_ensure_runtime(
    iree_tune_loom_hal_context_t* context) {
  if (context->runtime_initialized) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_tune_loom_hal_context_select_backend(context));
  IREE_RETURN_IF_ERROR(loom_run_hal_runtime_initialize(
      context->backend, context->host_allocator, &context->runtime));
  context->runtime_initialized = true;
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_select_hal_actual_invocation(
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_invocation_plan_t** out_invocation) {
  *out_invocation = NULL;
  iree_host_size_t actual_invocation_count = 0;
  for (iree_host_size_t i = 0; i < case_plan->invocation_count; ++i) {
    const loom_testbench_invocation_plan_t* invocation =
        &case_plan->invocations[i];
    if (invocation->kind != LOOM_TESTBENCH_INVOCATION_ACTUAL) {
      continue;
    }
    *out_invocation = invocation;
    ++actual_invocation_count;
  }
  if (actual_invocation_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "dispatch_complete V0 requires exactly one actual invocation in "
        "check.case `%.*s`; found %" PRIhsz,
        (int)case_plan->name.size, case_plan->name.data,
        actual_invocation_count);
  }
  if ((*out_invocation)->result_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "dispatch_complete V0 supports in-place HAL buffer arguments only; "
        "actual invocation in `%.*s` has %" PRIhsz " results",
        (int)case_plan->name.size, case_plan->name.data,
        (*out_invocation)->result_count);
  }
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_hal_actual_provider_initialize(
    iree_tune_loom_hal_context_t* context, loom_run_session_t* session,
    iree_string_view_t filename, iree_string_view_t source,
    iree_string_view_t pipeline, const loom_module_t* test_module,
    const loom_testbench_invocation_plan_t* actual_invocation,
    const loom_run_compile_report_capture_options_t* compile_report_options,
    iree_tune_loom_hal_actual_provider_t* out_provider) {
  *out_provider = (iree_tune_loom_hal_actual_provider_t){
      .context = context,
      .session = session,
      .filename = filename,
      .source = source,
      .pipeline = pipeline,
      .test_module = test_module,
      .actual_invocation = actual_invocation,
  };
  loom_run_hal_invocation_options_initialize(&out_provider->invocation_options);
  iree_tune_loom_diagnostic_capture_initialize(context->host_allocator,
                                               &out_provider->diagnostics);
  iree_status_t status = loom_run_compile_report_capture_initialize(
      compile_report_options, context->host_allocator,
      &out_provider->compile_report_capture);
  if (iree_status_is_ok(status)) {
    out_provider->compile_report_capture_initialized = true;
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
  if (provider->prepared_candidate_initialized) {
    loom_run_hal_prepared_candidate_deinitialize(&provider->prepared_candidate);
  }
  if (provider->candidate_initialized) {
    loom_run_hal_candidate_deinitialize(&provider->candidate);
  }
  if (provider->compile_module_initialized) {
    loom_run_module_deinitialize(&provider->compile_module);
  }
  if (provider->compile_report_capture_initialized) {
    loom_run_compile_report_capture_deinitialize(
        &provider->compile_report_capture);
  }
  iree_tune_loom_diagnostic_capture_deinitialize(&provider->diagnostics);
  *provider = (iree_tune_loom_hal_actual_provider_t){0};
}

static iree_status_t iree_tune_loom_hal_actual_provider_compile(
    iree_tune_loom_hal_actual_provider_t* provider) {
  if (provider->prepared_candidate_initialized || provider->compile_rejected) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_hal_context_ensure_runtime(provider->context));

  iree_string_view_t entry_symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_tune_loom_module_symbol_name_from_ref(
      provider->test_module, provider->actual_invocation->callee_ref,
      &entry_symbol));

  loom_run_module_parse_options_t parse_options = {0};
  loom_run_module_parse_options_initialize(&parse_options);
  parse_options.filename = provider->filename;
  parse_options.source = provider->source;
  IREE_RETURN_IF_ERROR(loom_run_module_parse(provider->session, &parse_options,
                                             &provider->compile_module));
  provider->compile_module_initialized = true;

  loom_run_one_shot_options_t one_shot_options = {0};
  loom_run_one_shot_options_initialize(&one_shot_options);
  loom_run_one_shot_options_apply_static_hal_workgroup_count(
      provider->compile_module.module, entry_symbol, &one_shot_options);
  provider->invocation_options.entry_point = one_shot_options.hal_entry_point;
  provider->invocation_options.workgroup_count[0] =
      one_shot_options.hal_workgroup_count[0];
  provider->invocation_options.workgroup_count[1] =
      one_shot_options.hal_workgroup_count[1];
  provider->invocation_options.workgroup_count[2] =
      one_shot_options.hal_workgroup_count[2];

  if (provider->context->configuration->target_environment == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "dispatch_complete benchmarks require a target environment");
  }
  loom_compile_pipeline_options_t pipeline_options = {0};
  loom_compile_pipeline_options_initialize(&pipeline_options);
  pipeline_options.pipeline = provider->pipeline;
  pipeline_options.entry_symbol = entry_symbol;
  pipeline_options.target_environment =
      provider->context->configuration->target_environment;
  pipeline_options.low_descriptor_registry =
      loom_run_session_low_descriptor_registry(provider->session);
  pipeline_options.diagnostic_sink = (loom_diagnostic_sink_t){
      .fn = iree_tune_loom_diagnostic_capture_sink,
      .user_data = &provider->diagnostics,
  };
  pipeline_options.source_resolver =
      loom_run_module_source_resolver(&provider->compile_module);
  iree_status_t status = loom_compile_run_pipeline(
      provider->compile_module.module, &pipeline_options,
      loom_run_session_block_pool(provider->session), &provider->pass_result);
  if (!iree_status_is_ok(status)) {
    if (provider->diagnostics.error_count != 0) {
      iree_status_free(status);
      provider->compile_rejected = true;
      provider->compile_failure_stage = IREE_SV("compile");
      provider->compile_failure_kind = IREE_SV("pass_diagnostics");
      return iree_ok_status();
    }
    return status;
  }
  if (provider->pass_result.error_count != 0) {
    provider->compile_rejected = true;
    provider->compile_failure_stage = IREE_SV("compile");
    provider->compile_failure_kind = IREE_SV("pass_diagnostics");
    return iree_ok_status();
  }

  loom_run_candidate_compile_options_t compile_options = {0};
  loom_run_candidate_compile_options_initialize(&compile_options);
  compile_options.module_name = IREE_SV("loom");
  compile_options.entry_symbol = entry_symbol;
  compile_options.diagnostic_sink = (loom_diagnostic_sink_t){
      .fn = iree_tune_loom_diagnostic_capture_sink,
      .user_data = &provider->diagnostics,
  };
  compile_options.source_resolver =
      loom_run_module_source_resolver(&provider->compile_module);
  loom_run_compile_report_capture_configure_compile_options(
      &provider->compile_report_capture, &compile_options);
  provider->candidate_initialized = true;
  status = loom_run_hal_candidate_compile(
      provider->context->backend, &provider->context->runtime,
      &provider->compile_module, &compile_options,
      provider->context->host_allocator, &provider->candidate);
  provider->compile_report_available = true;
  if (!iree_status_is_ok(status)) {
    if (provider->diagnostics.error_count != 0) {
      iree_status_free(status);
      provider->compile_rejected = true;
      provider->compile_failure_stage = IREE_SV("emit");
      provider->compile_failure_kind = IREE_SV("emit_diagnostics");
      return iree_ok_status();
    }
    return status;
  }
  if (!provider->candidate.compiled) {
    provider->compile_rejected = true;
    provider->compile_failure_stage = IREE_SV("emit");
    provider->compile_failure_kind = IREE_SV("no_executable");
    provider->compile_failure_message =
        IREE_SV("HAL backend did not emit an executable");
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_run_hal_prepared_candidate_prepare(
      &provider->context->runtime, &provider->candidate.executable,
      &provider->prepared_candidate));
  provider->prepared_candidate_initialized = true;
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_variant_list_from_invocation_inputs(
    const iree_vm_variant_t* inputs, iree_host_size_t input_count,
    iree_allocator_t allocator, iree_vm_list_t** out_list) {
  *out_list = NULL;
  iree_vm_list_t* list = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                           input_count, allocator, &list));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < input_count;
       ++i) {
    status = iree_vm_list_push_variant_retain(list, &inputs[i]);
  }
  if (iree_status_is_ok(status)) {
    *out_list = list;
  } else {
    iree_vm_list_release(list);
  }
  return status;
}

static iree_status_t iree_tune_loom_hal_actual_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t input_count, const iree_vm_variant_t* inputs,
    iree_host_size_t result_count, iree_vm_variant_t* out_results) {
  (void)out_results;
  iree_tune_loom_hal_actual_provider_t* provider =
      (iree_tune_loom_hal_actual_provider_t*)user_data;
  if (invocation != provider->actual_invocation) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL actual provider received an unexpected invocation");
  }
  if (result_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "dispatch_complete V0 supports in-place HAL buffer arguments only");
  }
  IREE_RETURN_IF_ERROR(iree_tune_loom_hal_actual_provider_compile(provider));
  if (provider->compile_rejected) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL actual provider invoked after candidate "
                            "compile was rejected");
  }

  iree_vm_list_t* bindings = NULL;
  IREE_RETURN_IF_ERROR(iree_tune_loom_variant_list_from_invocation_inputs(
      inputs, input_count, provider->context->host_allocator, &bindings));

  loom_run_hal_invocation_plan_t plan = {0};
  loom_run_hal_iteration_t iteration = {0};
  iree_status_t status = loom_run_hal_invocation_plan_prepare_from_lists(
      &provider->invocation_options, bindings, /*expected_bindings=*/NULL,
      /*max_output_element_count=*/0, &plan);
  iree_vm_list_release(bindings);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_dispatch_plan(
        &provider->context->runtime, &provider->prepared_candidate, &plan,
        provider->context->host_allocator, &iteration);
  }
  loom_run_hal_iteration_deinitialize(&iteration);
  loom_run_hal_invocation_plan_deinitialize(&plan);
  return status;
}

static iree_status_t iree_tune_loom_create_hal_binding_list_for_sample(
    const loom_module_t* module,
    const loom_testbench_value_materializer_options_t* materializer_options,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    iree_vm_list_t** out_bindings) {
  *out_bindings = NULL;
  loom_testbench_value_table_t table = {0};
  iree_vm_list_t* bindings = NULL;
  iree_status_t status = loom_testbench_value_table_initialize(
      module, case_plan, allocator, &table);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_materialize_case_sample(
        materializer_options, case_plan, sample_ordinal, &table);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                 invocation->input_count, allocator, &bindings);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < invocation->input_count; ++i) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    status = loom_testbench_value_table_lookup_retain(
        &table, invocation->input_value_ids[i], &variant);
    if (iree_status_is_ok(status)) {
      status = iree_vm_list_push_variant_move(bindings, &variant);
    }
    iree_vm_variant_reset(&variant);
  }
  if (iree_status_is_ok(status)) {
    *out_bindings = bindings;
    bindings = NULL;
  }
  iree_vm_list_release(bindings);
  loom_testbench_value_table_deinitialize(&table);
  return status;
}

static void iree_tune_loom_benchmark_result_set_compile_rejection(
    const iree_tune_loom_hal_actual_provider_t* provider,
    iree_tune_loom_benchmark_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  out_result->status = IREE_SV("compile_failed");
  out_result->has_failure = true;
  out_result->failure_stage = provider->compile_failure_stage;
  out_result->failure_kind = provider->compile_failure_kind;
  out_result->failure_message = provider->compile_failure_message;
  out_result->diagnostic_error_count = provider->diagnostics.error_count;
  out_result->diagnostic_warning_count = provider->diagnostics.warning_count;
  out_result->diagnostic_remark_count = provider->diagnostics.remark_count;
  out_result->diagnostic_json =
      iree_string_builder_view(&provider->diagnostics.output);
  if (provider->compile_report_available) {
    out_result->compile_report_capture = &provider->compile_report_capture;
  }
}

static iree_status_t iree_tune_loom_run_hal_benchmark_sample(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    iree_tune_loom_hal_actual_provider_t* provider,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    iree_tune_loom_benchmark_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  out_result->has_sample_ordinal = true;
  out_result->sample_ordinal = sample_ordinal;
  out_result->samples_per_iteration = 1;

  IREE_RETURN_IF_ERROR(iree_tune_loom_hal_actual_provider_compile(provider));
  if (provider->compile_report_available) {
    out_result->compile_report_capture = &provider->compile_report_capture;
  }
  if (provider->compile_rejected) {
    iree_tune_loom_benchmark_result_set_compile_rejection(provider, out_result);
    out_result->has_sample_ordinal = true;
    out_result->sample_ordinal = sample_ordinal;
    out_result->samples_per_iteration = 1;
    return iree_ok_status();
  }

  iree_vm_list_t* bindings = NULL;
  iree_status_t status = iree_tune_loom_create_hal_binding_list_for_sample(
      module_plan->module, materializer_options, case_plan,
      provider->actual_invocation, sample_ordinal, allocator, &bindings);
  loom_run_hal_invocation_plan_t plan = {0};
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_plan_prepare_from_lists(
        &provider->invocation_options, bindings, /*expected_bindings=*/NULL,
        /*max_output_element_count=*/0, &plan);
  }
  iree_vm_list_release(bindings);
  if (iree_status_is_ok(status)) {
    out_result->has_hal_benchmark = true;
    status = loom_run_hal_benchmark_dispatch_plan(
        &provider->context->runtime, &provider->prepared_candidate, &plan,
        &policy->hal_options, allocator, &out_result->hal_benchmark);
  }
  if (iree_status_is_ok(status)) {
    out_result->executed = true;
    out_result->passed = true;
  }
  loom_run_hal_invocation_plan_deinitialize(&plan);
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

static iree_status_t iree_tune_loom_run_case_correctness(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    iree_host_size_t case_index,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* arena, iree_string_builder_t* sample_output,
    iree_host_size_t* out_sample_count,
    iree_host_size_t* out_failed_sample_count) {
  *out_sample_count = 0;
  *out_failed_sample_count = 0;

  const loom_testbench_case_plan_t* case_plan = &module_plan->cases[case_index];
  iree_host_size_t begin_sample = 0;
  iree_host_size_t end_sample = 0;
  IREE_RETURN_IF_ERROR(iree_tune_loom_validate_sample_flag(
      case_plan->sample_count, &begin_sample, &end_sample));

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
      loom_output_stream_for_builder(sample_output, &stream);
      status = loom_output_stream_write_cstring(&stream, "{\"row\":\"sample\"");
      if (iree_status_is_ok(status)) {
        status = iree_tune_loom_write_run_id_field_json(run, &stream);
      }
      if (iree_status_is_ok(status)) {
        status =
            iree_tune_loom_write_candidate_identity_json(candidate, &stream);
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

static iree_status_t iree_tune_loom_write_timing_stats_json(
    const iree_tune_loom_timing_stats_t* stats, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"count\":%" PRIhsz, stats->count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"total\":%" PRIi64, stats->total_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"min\":%" PRIi64, stats->minimum_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"max\":%" PRIi64, stats->maximum_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(stream, ",\"mean\":%.3f",
                                                       stats->mean_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p50\":%" PRIi64, stats->p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p90\":%" PRIi64, stats->p90_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_write_benchmark_timing_stats_json(
    const loom_run_benchmark_timing_stats_t* stats,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"count\":%" PRIhsz, stats->count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"total\":%" PRIi64, stats->total_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"min\":%" PRIi64, stats->minimum_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"max\":%" PRIi64, stats->maximum_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(stream, ",\"mean\":%.3f",
                                                       stats->mean_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p50\":%" PRIi64, stats->p50_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p90\":%" PRIi64, stats->p90_ns));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"p90_to_p50_delta_ppm\":%" PRIu64,
      stats->p90_to_p50_delta_ppm));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

static const char* iree_tune_loom_profile_statistics_row_type_name(
    iree_hal_profile_statistics_row_type_t row_type) {
  switch (row_type) {
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_EXPORT:
      return "dispatch_export";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_COMMAND_BUFFER:
      return "dispatch_command_buffer";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_DISPATCH_COMMAND_OPERATION:
      return "dispatch_command_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_DEVICE_OPERATION:
      return "queue_device_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_QUEUE_HOST_OPERATION:
      return "queue_host_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_EXPORT:
      return "host_execution_export";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_COMMAND_BUFFER:
      return "host_execution_command_buffer";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_COMMAND_OPERATION:
      return "host_execution_command_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_HOST_EXECUTION_QUEUE_OPERATION:
      return "host_execution_queue_operation";
    case IREE_HAL_PROFILE_STATISTICS_ROW_TYPE_MEMORY_LIFECYCLE:
      return "memory_lifecycle";
    default:
      return "unknown";
  }
}

static const char* iree_tune_loom_profile_statistics_time_domain_name(
    iree_hal_profile_statistics_time_domain_t time_domain) {
  switch (time_domain) {
    case IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_DEVICE_TICK:
      return "device_tick";
    case IREE_HAL_PROFILE_STATISTICS_TIME_DOMAIN_IREE_HOST_TIME_NS:
      return "iree_host_time_ns";
    default:
      return "none";
  }
}

static iree_status_t iree_tune_loom_write_profile_family_names_json(
    iree_hal_device_profiling_data_families_t data_families,
    loom_output_stream_t* stream) {
  typedef struct iree_tune_loom_profile_family_name_t {
    // Bit represented by |name|.
    iree_hal_device_profiling_data_families_t bit;
    // Stable JSON string used for the bit.
    const char* name;
  } iree_tune_loom_profile_family_name_t;
  static const iree_tune_loom_profile_family_name_t kNames[] = {
      {IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS, "queue_events"},
      {IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS,
       "host_execution_events"},
      {IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS,
       "device_queue_events"},
      {IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS, "dispatch_events"},
      {IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES, "counter_samples"},
      {IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA,
       "executable_metadata"},
      {IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES, "executable_traces"},
      {IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS, "memory_events"},
      {IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_METRICS, "device_metrics"},
      {IREE_HAL_DEVICE_PROFILING_DATA_COMMAND_REGION_EVENTS,
       "command_region_events"},
      {IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES, "counter_ranges"},
  };
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  bool first = true;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kNames); ++i) {
    if (!iree_all_bits_set(data_families, kNames[i].bit)) {
      continue;
    }
    if (!first) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_cstring(stream, kNames[i].name));
    first = false;
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_write_profile_flag_names_json(
    iree_hal_device_profiling_flags_t flags, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  if (iree_all_bits_set(
          flags, IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS)) {
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_cstring(stream, "lightweight_statistics"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_write_hal_profile_duration_json(
    const loom_run_hal_profile_row_summary_t* row,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"available\":%s",
      iree_all_bits_set(row->flags, IREE_HAL_PROFILE_STATISTICS_ROW_FLAG_TIMING)
          ? "true"
          : "false"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"time_domain\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream,
      iree_tune_loom_profile_statistics_time_domain_name(row->time_domain)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"first_start\":%" PRIu64 ",\"last_end\":%" PRIu64 ",\"total\":%" PRIu64
      ",\"min\":%" PRIu64 ",\"max\":%" PRIu64,
      row->first_start_time, row->last_end_time, row->total_duration,
      row->minimum_duration, row->maximum_duration));
  const uint64_t valid_sample_count =
      row->sample_count >= row->invalid_sample_count
          ? row->sample_count - row->invalid_sample_count
          : 0;
  if (valid_sample_count != 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"mean\":%" PRIu64,
        row->total_duration / valid_sample_count));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"scaled_ns\":%s",
      row->has_scaled_duration_ns ? "true" : "false"));
  if (row->has_scaled_duration_ns) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"total_ns\":%" PRIu64 ",\"min_ns\":%" PRIu64 ",\"max_ns\":%" PRIu64,
        row->total_duration_ns, row->minimum_duration_ns,
        row->maximum_duration_ns));
    if (valid_sample_count != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, ",\"mean_ns\":%" PRIu64,
          row->total_duration_ns / valid_sample_count));
    }
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_write_hal_profile_row_summary_json(
    const loom_run_hal_profile_row_summary_t* row,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\"type\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      stream, iree_tune_loom_profile_statistics_row_type_name(row->row_type)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"row_type\":%" PRIu32 ",\"flags\":%" PRIu32, row->row_type,
      row->flags));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"physical_device_ordinal\":%" PRIu32 ",\"queue_ordinal\":%" PRIu32
      ",\"event_type\":%" PRIu32,
      row->physical_device_ordinal, row->queue_ordinal, row->event_type));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"executable_id\":%" PRIu64 ",\"command_buffer_id\":%" PRIu64
      ",\"export_ordinal\":%" PRIu32 ",\"command_index\":%" PRIu32,
      row->executable_id, row->command_buffer_id, row->export_ordinal,
      row->command_index));
  if (row->export_name_length != 0) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"export_name\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream,
        iree_make_string_view(row->export_name, row->export_name_length)));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"sample_count\":%" PRIu64 ",\"invalid_sample_count\":%" PRIu64
      ",\"operation_count\":%" PRIu64 ",\"payload_bytes\":%" PRIu64
      ",\"tile_count\":%" PRIu64 ",\"tile_duration_sum_ns\":%" PRIu64,
      row->sample_count, row->invalid_sample_count, row->operation_count,
      row->payload_bytes, row->tile_count, row->tile_duration_sum_ns));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"timing\":"));
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_hal_profile_duration_json(row, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_write_hal_profile_summary_json(
    const loom_run_hal_profile_summary_t* profile,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"requested\":%s", profile->requested ? "true" : "false"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"executed\":%s", profile->executed ? "true" : "false"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"flags\":%" PRIu32, profile->flags));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"flag_names\":"));
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_profile_flag_names_json(profile->flags, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"data_families\":%" PRIu64, profile->data_families));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"data_family_names\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_profile_family_names_json(
      profile->data_families, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"row_count\":%" PRIhsz, profile->row_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"captured_row_count\":%" PRIhsz, profile->captured_row_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"truncated_row_count\":%" PRIhsz,
      profile->truncated_row_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"dropped_record_count\":%" PRIu64,
      profile->dropped_record_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"rows\":["));
  for (iree_host_size_t i = 0; i < profile->captured_row_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_hal_profile_row_summary_json(
        &profile->rows[i], stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  if (profile->has_error) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"error\":{"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "\"code\":%d", (int)profile->error_code));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"status\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream,
        iree_make_cstring_view(iree_status_code_string(profile->error_code))));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"message\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, iree_make_string_view(profile->error_message,
                                      profile->error_message_length)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

static iree_string_view_t iree_tune_loom_benchmark_result_status(
    const iree_tune_loom_benchmark_result_t* benchmark_result) {
  if (!iree_string_view_is_empty(benchmark_result->status)) {
    return benchmark_result->status;
  }
  if (benchmark_result->executed) {
    return benchmark_result->passed ? IREE_SV("ok") : IREE_SV("failed");
  }
  return IREE_SV("skipped");
}

static iree_status_t iree_tune_loom_write_benchmark_failure_json(
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream) {
  if (!benchmark_result->has_failure) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"failure\":{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\"stage\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, benchmark_result->failure_stage));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"kind\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, benchmark_result->failure_kind));
  if (!iree_string_view_is_empty(benchmark_result->failure_message)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"message\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, benchmark_result->failure_message));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"diagnostic_error_count\":%" PRIhsz,
      benchmark_result->diagnostic_error_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"diagnostic_warning_count\":%" PRIhsz,
      benchmark_result->diagnostic_warning_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"diagnostic_remark_count\":%" PRIhsz,
      benchmark_result->diagnostic_remark_count));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"diagnostics\":["));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write(stream, benchmark_result->diagnostic_json));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]}"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_write_compile_report_json(
    const loom_run_compile_report_capture_t* compile_report_capture,
    loom_output_stream_t* stream) {
  if (compile_report_capture == NULL ||
      compile_report_capture->options.mode ==
          LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"compile_report\":"));
  IREE_RETURN_IF_ERROR(loom_run_compile_report_capture_append_json(
      compile_report_capture, stream));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_write_benchmark_result_json(
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"benchmark\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"case\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, case_plan->name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"status\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, iree_tune_loom_benchmark_result_status(benchmark_result)));
  if (benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_format(stream, ",\"sample_ordinal\":%" PRIhsz,
                                        benchmark_result->sample_ordinal));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"measure\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, policy->measure));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"warmup_iterations\":%" PRIhsz, policy->warmup_iterations));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"iterations\":%" PRIhsz, policy->iterations));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"samples_per_iteration\":%" PRIhsz,
      benchmark_result->samples_per_iteration));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"correctness_sample_count\":%" PRIhsz,
      correctness_sample_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"correctness_failed_sample_count\":%" PRIhsz,
      correctness_failed_sample_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"benchmark_failed_sample_count\":%" PRIhsz,
      benchmark_result->failed_sample_count));
  if (benchmark_result->executed && !benchmark_result->has_hal_benchmark) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"timing_ns\":"));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_timing_stats_json(
        &benchmark_result->timing, stream));
  }
  if (benchmark_result->has_hal_benchmark) {
    const loom_run_benchmark_result_t* timing =
        &benchmark_result->hal_benchmark.timing;
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"batch_size\":%" PRIhsz, timing->batch_size));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"warmup_batch_count\":%" PRIhsz,
        timing->warmup_batch_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"warmup_duration_ns\":%" PRIi64,
        timing->warmup_duration_ns));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"measured_batch_count\":%" PRIhsz,
        timing->measured_batch_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"measured_dispatch_count\":%" PRIhsz,
        timing->measured_operation_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"measured_duration_ns\":%" PRIi64,
        timing->measured_duration_ns));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"stop_reason\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, loom_run_benchmark_stop_reason_name(timing->stop_reason)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"batch_timing_ns\":"));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_benchmark_timing_stats_json(
        &timing->batch_timing, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"dispatch_timing_ns\":"));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_benchmark_timing_stats_json(
        &timing->operation_timing, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"profile\":"));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_hal_profile_summary_json(
        &benchmark_result->hal_benchmark.profile, stream));
  }
  if (benchmark_result->compile_report_capture != NULL &&
      benchmark_result->compile_report_capture->options.mode !=
          LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_compile_report_json(
        benchmark_result->compile_report_capture, stream));
  }
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_benchmark_failure_json(benchmark_result, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_append_compile_row(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_hal_actual_provider_t* provider,
    iree_string_builder_t* compile_output) {
  iree_string_view_t entry_symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_tune_loom_module_symbol_name_from_ref(
      provider->test_module, provider->actual_invocation->callee_ref,
      &entry_symbol));

  loom_output_stream_t stream;
  loom_output_stream_for_builder(compile_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"compile\""));
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
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"entry\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, entry_symbol));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"status\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
      &stream, provider->compile_rejected ? "failed" : "ok"));
  if (provider->compile_rejected) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"stage\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->compile_failure_stage));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"kind\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->compile_failure_kind));
    if (!iree_string_view_is_empty(provider->compile_failure_message)) {
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(&stream, ",\"message\":"));
      IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
          &stream, provider->compile_failure_message));
    }
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"diagnostic_error_count\":%" PRIhsz,
      provider->diagnostics.error_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"diagnostic_warning_count\":%" PRIhsz,
      provider->diagnostics.warning_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"diagnostic_remark_count\":%" PRIhsz,
      provider->diagnostics.remark_count));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"diagnostics\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_diagnostic_array_json(
      &provider->diagnostics, &stream));
  if (provider->compile_report_available) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_compile_report_json(
        &provider->compile_report_capture, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_append_benchmark_result(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    iree_string_builder_t* benchmark_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(benchmark_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"benchmark\""));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_candidate_identity_json(candidate, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"benchmark_result\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_benchmark_result_json(
      benchmark_plan, case_plan, policy, benchmark_result,
      correctness_sample_count, correctness_failed_sample_count, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_append_profile_row(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_string_builder_t* profile_output) {
  if (!benchmark_result->has_hal_benchmark ||
      !benchmark_result->hal_benchmark.profile.requested) {
    return iree_ok_status();
  }
  loom_output_stream_t stream;
  loom_output_stream_for_builder(profile_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"profile\""));
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
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"batch_size\":%" PRIhsz,
      benchmark_result->hal_benchmark.timing.batch_size));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"profile\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_hal_profile_summary_json(
      &benchmark_result->hal_benchmark.profile, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
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
    iree_string_builder_t* plan_output) {
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
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"measure\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, policy->measure));
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
        "\"profile_final_batch\":%s",
        FLAG_batch_size.specified ? "true" : "false",
        FLAG_min_time_ms.specified ? "true" : "false",
        FLAG_warmup_time_ms.specified ? "true" : "false",
        FLAG_max_batches.specified ? "true" : "false",
        FLAG_stable_p90_to_p50_ppm.specified ? "true" : "false",
        FLAG_profile_final_batch.specified ? "true" : "false"));
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
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_append_failure_row(
    const iree_tune_loom_run_identity_t* run, iree_string_view_t stage,
    iree_string_view_t kind, iree_string_view_t message,
    const iree_tune_loom_diagnostic_capture_t* diagnostics,
    iree_string_builder_t* failure_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(failure_output, &stream);
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "{\"row\":\"failure\""));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"stage\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, stage));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"kind\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, kind));
  if (!iree_string_view_is_empty(message)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"message\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, message));
  }
  if (diagnostics != NULL) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"diagnostic_error_count\":%" PRIhsz,
        diagnostics->error_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"diagnostic_warning_count\":%" PRIhsz,
        diagnostics->warning_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"diagnostic_remark_count\":%" PRIhsz,
        diagnostics->remark_count));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"diagnostics\":"));
    IREE_RETURN_IF_ERROR(
        iree_tune_loom_write_diagnostic_array_json(diagnostics, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_write_report(
    const iree_tune_loom_run_identity_t* run,
    iree_host_size_t planned_case_count,
    iree_host_size_t planned_benchmark_count,
    iree_host_size_t selected_benchmark_count, iree_host_size_t failure_count,
    iree_host_size_t failed_benchmark_count,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count, bool dry_run,
    iree_string_view_t failures, iree_string_view_t plans,
    iree_string_view_t devices, iree_string_view_t compiles,
    iree_string_view_t samples, iree_string_view_t benchmarks,
    iree_string_view_t profiles, iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_tune_loom_append_run_row(run, dry_run, output));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, failures));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, plans));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, devices));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, compiles));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, samples));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, benchmarks));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, profiles));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, "{\"row\":\"summary\""));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"dry_run\":%s", dry_run ? "true" : "false"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"planned_case_count\":%" PRIhsz, planned_case_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"planned_benchmark_count\":%" PRIhsz,
      planned_benchmark_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"benchmark_count\":%" PRIhsz, selected_benchmark_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"failure_count\":%" PRIhsz, failure_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"failed_benchmark_count\":%" PRIhsz, failed_benchmark_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"correctness_sample_count\":%" PRIhsz,
      correctness_sample_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"correctness_failed_sample_count\":%" PRIhsz,
      correctness_failed_sample_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}\n"));
  return iree_ok_status();
}

int iree_tune_loom_main(int argc, char** argv,
                        const iree_tune_loom_configuration_t* configuration) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      configuration->tool_name,
      "Runs correctness-gated check.benchmark records from a Loom module.\n"
      "\n"
      "Usage:\n"
      "  iree-tune-loom file.loom --benchmark=@smoke_latency\n"
      "  iree-tune-loom file.loom --benchmark=@smoke_latency --dry_run "
      "--output=plan.jsonl\n"
      "  iree-tune-loom file.loom --device=amdgpu "
      "--benchmark=@kernel_latency --batch_size=64 "
      "--profile_final_batch=true --output=results.jsonl\n"
      "  cat module.loom | iree-tune-loom -\n"
      "  iree-tune-loom --agents_md\n"
      "\n"
      "JSONL rows:\n"
      "  run       first row; source path and process-local run_id\n"
      "  plan      selected benchmark/case, effective policy, CLI overrides\n"
      "  device    selected --device/driver/backend and cheap HAL identity\n"
      "  compile   per-candidate diagnostics and compile report payload\n"
      "  sample    correctness sample result from the testbench executor\n"
      "  benchmark timing/failure evidence; dispatch_timing_ns is the score\n"
      "  profile   optional final profiled-batch evidence outside timing\n"
      "  failure   parse/verify/planning failure diagnostics\n"
      "  summary   final totals row\n"
      "\n"
      "Every row carries run_id. Selected benchmark rows also carry "
      "candidate_id/candidate_index so agents can join plan, compile, sample, "
      "benchmark, and profile evidence across large JSONL sweeps. Compile "
      "diagnostics are data: a compile row with status!='ok' is a candidate "
      "rejection, not a tool infrastructure failure. Profile rows are emitted "
      "only when --profile_final_batch=true or a benchmark attr requests it; "
      "their profile.rows array contains HAL statistics rows such as "
      "dispatch_export, dispatch_command_buffer, and "
      "dispatch_command_operation when the backend provides them.\n"
      "\n"
      "jq recipes:\n"
      "  jq 'select(.row==\"run\" or .row==\"summary\")' results.jsonl\n"
      "  jq 'select(.row==\"summary\")' results.jsonl\n"
      "  jq 'select(.row==\"failure\" or .row==\"compile\") | "
      "select(.status!=\"ok\" or .row==\"failure\")' results.jsonl\n"
      "  jq 'select(.row==\"compile\") | .diagnostics[]?' results.jsonl\n"
      "  jq 'select(.row==\"benchmark\") | .benchmark_result | "
      "{benchmark,status,p50:.dispatch_timing_ns.p50}' results.jsonl\n"
      "  jq 'select(.row==\"profile\") as $r | $r.profile.rows[]? | "
      "select(.type|startswith(\"dispatch_\")) | "
      "{candidate_id:$r.candidate_id,type,export_name,"
      "timing:.timing.mean_ns}' "
      "results.jsonl\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (FLAG_agents_md) {
    iree_tune_loom_print_agents_md(stdout);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(0);
    return 0;
  }

  iree_allocator_t allocator = iree_allocator_system();
  iree_io_file_contents_t* contents = NULL;
  loom_run_session_t session = {0};
  loom_run_module_t run_module = {0};
  iree_tune_loom_hal_context_t hal_context = {0};
  iree_tune_loom_hal_context_initialize(configuration, allocator, &hal_context);
  iree_arena_allocator_t plan_arena;
  memset(&plan_arena, 0, sizeof(plan_arena));
  iree_arena_allocator_t execution_arena;
  memset(&execution_arena, 0, sizeof(execution_arena));
  iree_string_builder_t plan_output;
  iree_string_builder_initialize(allocator, &plan_output);
  iree_string_builder_t device_output;
  iree_string_builder_initialize(allocator, &device_output);
  iree_string_builder_t compile_output;
  iree_string_builder_initialize(allocator, &compile_output);
  iree_string_builder_t sample_output;
  iree_string_builder_initialize(allocator, &sample_output);
  iree_string_builder_t benchmark_output;
  iree_string_builder_initialize(allocator, &benchmark_output);
  iree_string_builder_t profile_output;
  iree_string_builder_initialize(allocator, &profile_output);
  iree_string_builder_t failure_output;
  iree_string_builder_initialize(allocator, &failure_output);
  iree_string_builder_t report_output;
  iree_string_builder_initialize(allocator, &report_output);
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

  iree_status_t status = iree_ok_status();
  if (argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "iree-tune-loom accepts at most one input file or '-' for stdin; got "
        "%d inputs",
        argc - 1);
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
  loom_run_compile_report_capture_options_t compile_report_options = {0};
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_compile_report_options_initialize(
        &compile_report_options);
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
  const iree_tune_loom_run_identity_t run_identity = {
      .run_id = iree_make_cstring_view(run_id_storage),
      .source = filename,
  };
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
      iree_status_free(status);
      status = iree_tune_loom_append_failure_row(
          &run_identity, IREE_SV("parse"), IREE_SV("diagnostics"),
          IREE_SV("input module has parse errors"), &source_diagnostics,
          &failure_output);
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
      status = iree_tune_loom_append_failure_row(
          &run_identity, IREE_SV("verify"), IREE_SV("diagnostics"),
          IREE_SV("input module failed verification"), &source_diagnostics,
          &failure_output);
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

    for (iree_host_size_t benchmark_index = 0;
         iree_status_is_ok(status) &&
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
        status = iree_tune_loom_append_plan_row(
            &run_identity, &candidate_identity, module_plan.module,
            benchmark_plan, case_plan, &policy, &plan_output);
      }
      if (iree_status_is_ok(status) && FLAG_dry_run &&
          policy.measure_kind == IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE) {
        status = iree_tune_loom_append_device_row(
            &run_identity, &hal_context, &device_row_state, &device_output);
      }
      if (iree_status_is_ok(status) && FLAG_dry_run) {
        continue;
      }

      loom_testbench_case_execution_options_t benchmark_execution_options =
          execution_options;
      loom_testbench_value_materializer_options_t benchmark_materializer =
          execution_options.materializer;
      iree_tune_loom_hal_actual_provider_t hal_provider = {0};
      bool hal_provider_initialized = false;
      bool benchmark_recorded = false;
      iree_host_size_t benchmark_correctness_sample_count = 0;
      iree_host_size_t benchmark_correctness_failed_sample_count = 0;
      if (iree_status_is_ok(status) &&
          policy.measure_kind == IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE) {
        const loom_testbench_invocation_plan_t* actual_invocation = NULL;
        status = iree_tune_loom_select_hal_actual_invocation(
            case_plan, &actual_invocation);
        if (iree_status_is_ok(status)) {
          status = iree_tune_loom_hal_context_ensure_runtime(&hal_context);
        }
        if (iree_status_is_ok(status)) {
          status = iree_tune_loom_hal_actual_provider_initialize(
              &hal_context, &session, filename, source,
              iree_make_cstring_view(FLAG_pipeline), module_plan.module,
              actual_invocation, &compile_report_options, &hal_provider);
        }
        if (iree_status_is_ok(status)) {
          hal_provider_initialized = true;
          benchmark_execution_options.materializer.device =
              hal_context.runtime.device;
          benchmark_execution_options.materializer.device_allocator =
              iree_hal_device_allocator(hal_context.runtime.device);
          benchmark_execution_options.materializer.buffer_params =
              iree_tune_loom_host_visible_buffer_params();
          benchmark_execution_options.invocation.invoke_actual =
              (loom_testbench_invocation_callback_t){
                  .fn = iree_tune_loom_hal_actual_invoke,
                  .user_data = &hal_provider,
              };
          benchmark_materializer = benchmark_execution_options.materializer;
          benchmark_materializer.buffer_params = (iree_hal_buffer_params_t){0};
          status = iree_tune_loom_hal_actual_provider_compile(&hal_provider);
        }
        if (iree_status_is_ok(status)) {
          status = iree_tune_loom_append_device_row(
              &run_identity, &hal_context, &device_row_state, &device_output);
        }
        if (iree_status_is_ok(status)) {
          status = iree_tune_loom_append_compile_row(
              &run_identity, &candidate_identity, benchmark_plan, case_plan,
              &hal_provider, &compile_output);
        }
        if (iree_status_is_ok(status) && hal_provider.compile_rejected) {
          iree_tune_loom_benchmark_result_t benchmark_result = {0};
          iree_tune_loom_benchmark_result_set_compile_rejection(
              &hal_provider, &benchmark_result);
          ++failed_benchmark_count;
          status = iree_tune_loom_append_benchmark_result(
              &run_identity, &candidate_identity, benchmark_plan, case_plan,
              &policy, &benchmark_result,
              /*correctness_sample_count=*/0,
              /*correctness_failed_sample_count=*/0, &benchmark_output);
          if (iree_status_is_ok(status)) {
            benchmark_recorded = true;
          }
        }
      }

      if (iree_status_is_ok(status) && !benchmark_recorded) {
        status = iree_tune_loom_run_case_correctness(
            &run_identity, &candidate_identity, &module_plan,
            benchmark_plan->case_index, &benchmark_execution_options,
            &execution_arena, &sample_output,
            &benchmark_correctness_sample_count,
            &benchmark_correctness_failed_sample_count);
      }
      if (iree_status_is_ok(status) && !benchmark_recorded) {
        correctness_sample_count += benchmark_correctness_sample_count;
        correctness_failed_sample_count +=
            benchmark_correctness_failed_sample_count;
      }

      if (iree_status_is_ok(status) && !benchmark_recorded) {
        if (benchmark_correctness_failed_sample_count == 0 &&
            policy.measure_kind == IREE_TUNE_LOOM_MEASURE_DISPATCH_COMPLETE) {
          iree_host_size_t begin_sample = 0;
          iree_host_size_t end_sample = 0;
          status = iree_tune_loom_validate_sample_flag(
              case_plan->sample_count, &begin_sample, &end_sample);
          for (iree_host_size_t sample_ordinal = begin_sample;
               iree_status_is_ok(status) && sample_ordinal < end_sample;
               ++sample_ordinal) {
            iree_tune_loom_benchmark_result_t benchmark_result = {0};
            status = iree_tune_loom_run_hal_benchmark_sample(
                &module_plan, case_plan, &policy, &hal_provider,
                &benchmark_materializer, sample_ordinal, allocator,
                &benchmark_result);
            if (iree_status_is_ok(status)) {
              if (!benchmark_result.executed || !benchmark_result.passed) {
                ++failed_benchmark_count;
              }
              status = iree_tune_loom_append_benchmark_result(
                  &run_identity, &candidate_identity, benchmark_plan, case_plan,
                  &policy, &benchmark_result,
                  benchmark_correctness_sample_count,
                  benchmark_correctness_failed_sample_count, &benchmark_output);
              if (iree_status_is_ok(status)) {
                status = iree_tune_loom_append_profile_row(
                    &run_identity, &candidate_identity, benchmark_plan,
                    case_plan, &benchmark_result, &profile_output);
              }
            }
          }
        } else {
          iree_tune_loom_benchmark_result_t benchmark_result = {0};
          if (iree_status_is_ok(status) &&
              benchmark_correctness_failed_sample_count == 0) {
            status = iree_tune_loom_run_benchmark_iterations(
                &module_plan, benchmark_plan->case_index,
                &benchmark_execution_options, &policy, &execution_arena,
                allocator, &benchmark_result);
          }
          if (iree_status_is_ok(status) &&
              benchmark_correctness_failed_sample_count != 0) {
            benchmark_result = (iree_tune_loom_benchmark_result_t){
                .executed = false,
                .passed = false,
                .samples_per_iteration = benchmark_correctness_sample_count,
                .failed_sample_count =
                    benchmark_correctness_failed_sample_count,
            };
          }
          if (iree_status_is_ok(status)) {
            if (!benchmark_result.executed || !benchmark_result.passed) {
              ++failed_benchmark_count;
            }
            status = iree_tune_loom_append_benchmark_result(
                &run_identity, &candidate_identity, benchmark_plan, case_plan,
                &policy, &benchmark_result, benchmark_correctness_sample_count,
                benchmark_correctness_failed_sample_count, &benchmark_output);
            if (iree_status_is_ok(status)) {
              status = iree_tune_loom_append_profile_row(
                  &run_identity, &candidate_identity, benchmark_plan, case_plan,
                  &benchmark_result, &profile_output);
            }
          }
        }
      }
      if (hal_provider_initialized) {
        iree_tune_loom_hal_actual_provider_deinitialize(&hal_provider);
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
    status = iree_tune_loom_write_report(
        &run_identity, planned_case_count, planned_benchmark_count,
        selected_benchmark_count, failure_count, failed_benchmark_count,
        correctness_sample_count, correctness_failed_sample_count, FLAG_dry_run,
        iree_string_builder_view(&failure_output),
        iree_string_builder_view(&plan_output),
        iree_string_builder_view(&device_output),
        iree_string_builder_view(&compile_output),
        iree_string_builder_view(&sample_output),
        iree_string_builder_view(&benchmark_output),
        iree_string_builder_view(&profile_output), &report_output);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        iree_make_cstring_view(FLAG_output),
        iree_string_builder_view(&report_output), allocator);
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

  iree_string_builder_deinitialize(&report_output);
  iree_tune_loom_diagnostic_capture_deinitialize(&source_diagnostics);
  iree_string_builder_deinitialize(&failure_output);
  iree_string_builder_deinitialize(&profile_output);
  iree_string_builder_deinitialize(&benchmark_output);
  iree_string_builder_deinitialize(&sample_output);
  iree_string_builder_deinitialize(&compile_output);
  iree_string_builder_deinitialize(&device_output);
  iree_string_builder_deinitialize(&plan_output);
  iree_arena_deinitialize(&execution_arena);
  iree_arena_deinitialize(&plan_arena);
  loom_run_module_deinitialize(&run_module);
  iree_tune_loom_hal_context_deinitialize(&hal_context);
  iree_io_file_contents_free(contents);
  loom_run_session_deinitialize(&session);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
