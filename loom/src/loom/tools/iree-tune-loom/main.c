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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#if defined(IREE_PLATFORM_WINDOWS)
#include <direct.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif  // defined(IREE_PLATFORM_WINDOWS)

#include "iree/base/api.h"
#include "iree/base/internal/json.h"
#include "iree/base/internal/path.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/stdio_stream.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/profile/counter.h"
#include "iree/tooling/profile/summary.h"
#include "iree/vm/api.h"
#include "loom/error/json_sink.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/special_values.h"
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
          "Compile report embedded in benchmark rows. Use 'summary', "
          "'details', or empty/'none'.");
IREE_FLAG(int32_t, compile_report_row_limit,
          LOOM_RUN_COMPILE_REPORT_DEFAULT_ROW_LIMIT,
          "Maximum pressure, spill, and source-low rows to capture for "
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

typedef enum iree_tune_loom_shape_specialization_mode_e {
  // Compile once and pass each shape's parameter values dynamically.
  IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_DYNAMIC = 0,
  // Compile a private candidate for each selected shape sample.
  IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_PER_SAMPLE = 1,
  // Run both dynamic and per-sample specialization modes.
  IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_BOTH = 2,
} iree_tune_loom_shape_specialization_mode_t;

typedef enum iree_tune_loom_artifact_bundle_policy_e {
  // Bundle policy has not been requested.
  IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_NONE = 0,
  // Capture the JSONL stream, manifest, source identity, and controlled paths.
  IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_MINIMAL = 1,
  // Reserve debug artifact locations such as profiles and compile artifacts.
  IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG = 2,
  // Reserve full artifact locations including future outputs and promotion
  // data.
  IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_FULL = 3,
} iree_tune_loom_artifact_bundle_policy_t;

typedef enum iree_tune_loom_interleave_mode_e {
  // No interleaved comparison was requested.
  IREE_TUNE_LOOM_INTERLEAVE_NONE = 0,
  // Runs a baseline-anchored pairwise schedule: A then N repetitions of BA.
  IREE_TUNE_LOOM_INTERLEAVE_ABABA = 1,
  // Runs all selected candidates in ABCD order for each repetition.
  IREE_TUNE_LOOM_INTERLEAVE_ROUND_ROBIN = 2,
} iree_tune_loom_interleave_mode_t;

static iree_string_view_t iree_tune_loom_artifact_bundle_policy_name(
    iree_tune_loom_artifact_bundle_policy_t policy) {
  switch (policy) {
    case IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_NONE:
      return IREE_SV("none");
    case IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_MINIMAL:
      return IREE_SV("minimal");
    case IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG:
      return IREE_SV("debug");
    case IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_FULL:
      return IREE_SV("full");
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t iree_tune_loom_parse_artifact_bundle_policy(
    iree_string_view_t value,
    iree_tune_loom_artifact_bundle_policy_t* out_policy) {
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    *out_policy = IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_NONE;
    return iree_ok_status();
  }
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("minimal"))) {
    *out_policy = IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_MINIMAL;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("debug"))) {
    *out_policy = IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("full"))) {
    *out_policy = IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_FULL;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--artifact_bundle_policy must be one of minimal, debug, full, or none; "
      "got '%.*s'",
      (int)value.size, value.data);
}

static iree_status_t iree_tune_loom_parse_shape_specialization_mode(
    iree_string_view_t value,
    iree_tune_loom_shape_specialization_mode_t* out_mode) {
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(value, IREE_SV("dynamic"))) {
    *out_mode = IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_DYNAMIC;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("per_sample")) ||
      iree_string_view_equal(value, IREE_SV("per-sample"))) {
    *out_mode = IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_PER_SAMPLE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("both"))) {
    *out_mode = IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_BOTH;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--shape_specialization must be one of dynamic, per_sample, or both; got "
      "'%.*s'",
      (int)value.size, value.data);
}

static iree_string_view_t iree_tune_loom_shape_specialization_mode_name(
    iree_tune_loom_shape_specialization_mode_t mode) {
  switch (mode) {
    case IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_DYNAMIC:
      return IREE_SV("dynamic");
    case IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_PER_SAMPLE:
      return IREE_SV("per_sample");
    case IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_BOTH:
      return IREE_SV("both");
    default:
      return IREE_SV("unknown");
  }
}

static bool iree_tune_loom_shape_specialization_runs_dynamic(
    iree_tune_loom_shape_specialization_mode_t mode) {
  return mode == IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_DYNAMIC ||
         mode == IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_BOTH;
}

static bool iree_tune_loom_shape_specialization_runs_per_sample(
    iree_tune_loom_shape_specialization_mode_t mode) {
  return mode == IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_PER_SAMPLE ||
         mode == IREE_TUNE_LOOM_SHAPE_SPECIALIZATION_BOTH;
}

static iree_status_t iree_tune_loom_parse_interleave_mode(
    iree_string_view_t value, iree_tune_loom_interleave_mode_t* out_mode) {
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("ABABA")) ||
      iree_string_view_equal(value, IREE_SV("ababa"))) {
    *out_mode = IREE_TUNE_LOOM_INTERLEAVE_ABABA;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("round_robin")) ||
      iree_string_view_equal(value, IREE_SV("round-robin"))) {
    *out_mode = IREE_TUNE_LOOM_INTERLEAVE_ROUND_ROBIN;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--interleave must be one of ABABA or round_robin; got '%.*s'",
      (int)value.size, value.data);
}

static iree_string_view_t iree_tune_loom_interleave_mode_name(
    iree_tune_loom_interleave_mode_t mode) {
  switch (mode) {
    case IREE_TUNE_LOOM_INTERLEAVE_NONE:
      return IREE_SV("none");
    case IREE_TUNE_LOOM_INTERLEAVE_ABABA:
      return IREE_SV("ABABA");
    case IREE_TUNE_LOOM_INTERLEAVE_ROUND_ROBIN:
      return IREE_SV("round_robin");
    default:
      return IREE_SV("unknown");
  }
}

typedef struct iree_tune_loom_profile_family_name_t {
  // HAL profiling data-family bit represented by these names.
  iree_hal_device_profiling_data_families_t bit;
  // Command-line family name accepted by --profile_data.
  const char* flag_name;
  // Stable JSON string used for this family.
  const char* json_name;
} iree_tune_loom_profile_family_name_t;

static const iree_tune_loom_profile_family_name_t
    iree_tune_loom_profile_family_names[] = {
        {IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS, "queue-events",
         "queue_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS, "host-execution",
         "host_execution_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS,
         "device-queue-events", "device_queue_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS, "dispatch-events",
         "dispatch_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES, "counters",
         "counter_samples"},
        {IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA,
         "executable-metadata", "executable_metadata"},
        {IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES, "executable-traces",
         "executable_traces"},
        {IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS, "memory-events",
         "memory_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_METRICS, "device-metrics",
         "device_metrics"},
        {IREE_HAL_DEVICE_PROFILING_DATA_COMMAND_REGION_EVENTS,
         "command-region-events", "command_region_events"},
        {IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES, "counter-ranges",
         "counter_ranges"},
};

static bool iree_tune_loom_profile_data_has_counter_data(
    iree_hal_device_profiling_data_families_t data_families) {
  return iree_any_bit_set(data_families,
                          IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
                              IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES);
}

static bool iree_tune_loom_profile_data_needs_artifact_data(
    iree_hal_device_profiling_data_families_t data_families) {
  return iree_tune_loom_profile_data_has_counter_data(data_families) ||
         iree_any_bit_set(data_families,
                          IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_METRICS |
                              IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_TRACES);
}

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
          "large sweeps. Rows are written and flushed as each lifecycle step "
          "finishes, so `tail -f results.jsonl` is useful while a long run is "
          "still active. The final `summary` row carries both flat totals and "
          "a nested `summary` object for agent consumers. Use "
          "`--profile_final_batch=true` for dispatch timing evidence outside "
          "the measured window; add "
          "`--profile_artifacts_dir=DIR` when you need the raw HAL profile "
          "bundle for counters or traces. Parameterized cases report "
          "`shape_id`, `shape_index`, and a concrete `shape.parameters` map on "
          "sample/benchmark/profile rows. For dispatch benchmarks, "
          "`--shape_specialization=dynamic` compiles once, "
          "`--shape_specialization=per_sample` compiles each selected shape "
          "with concrete parameter facts, and `--shape_specialization=both` "
          "emits both result sets. `check.file.read.npy` paths resolve "
          "relative to the `.loom` file; `check.file.write.npy` paths are "
          "relative to `--file_output_dir`, which defaults under "
          "`$TMPDIR/iree-loom-tune`. Dispatch benchmarks materialize valid "
          "check-op inputs into a device-buffer binding ring; "
          "`--input_ring_min_bytes=33554432` is the default cache-thwarting "
          "target and `--input_ring_count=1` forces hot-reuse timing when "
          "that is the experiment. Benchmark rows report `data_cache` with "
          "the effective ring count and bytes. "
          "Use `--compare=@baseline,@variant --interleave=ABABA` for a fair "
          "two-candidate dispatch comparison; `--interleave=round_robin` "
          "supports ABCD-style rotation. Comparison mode runs full benchmark "
          "windows for one concrete shape per candidate and emits "
          "`benchmark.repetition` plus `comparison` rows; selected candidates "
          "must have the same concrete parameter layout and values, so use "
          "`--sample=` when a case has multiple shape samples. "
          "When `--profile_data` includes "
          "`counters` or `counter-ranges`, decoded rows are emitted as "
          "`profile_counter` JSONL after the normal `profile` row. Each "
          "decode emits a `counter_decode_status` row carrying the requested "
          "families/sets and decoded row counts; raw-artifact-only and "
          "unavailable profiles use the same status row shape instead of "
          "failing the timing run. Raw `.irpf` profile bundles also emit "
          "`profile_summary` rows with bundle/device summaries, executable "
          "trace byte counts, device metric counts, and decode status. Use "
          "`--artifact_bundle_dir=DIR` to collect `results.jsonl`, "
          "`manifest.json`, file outputs, and profile artifacts under one "
          "directory; `--artifact_bundle_policy=debug|full` also writes "
          "per-candidate compile report sidecars under `compile_reports/`, "
          "target-native artifacts under `target_artifacts/`, target-owned "
          "assembly/listing text under `target_listings/`, and HAL executable "
          "packages under `hal_executables/`; compile and benchmark rows link "
          "them from `compile_report_path`, `target_artifact_path`, "
          "`target_listing_path`, and `hal_executable_path`. The manifest "
          "records command/path/source identity, path/size/mtime metadata "
          "for observed fixture/output/profile/compile-report/executable "
          "files, selected environment variables, and HAL device identity "
          "when a dispatch benchmark selected a backend. Source, fixture, "
          "and artifact files are identified by path/size/mtime; this CLI "
          "does not content-hash large files or act as a CAS. "
          "Benchmark attrs named `family`, `phase`, `strategy`, "
          "`knobs`, `problem`, or `reference_id` are copied into a "
          "`metadata` object on candidate rows. Compile rows and benchmark "
          "payloads with reports also carry `static_summary` for code size, "
          "instruction count, descriptor instruction mix, spills, memory, "
          "pressure, source-low counts, and move-cause totals.\n"
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
          "--input_ring_min_bytes=33554432 "
          "--shape_specialization=both "
          "--output=results.jsonl\n"
          "```\n"
          "\n"
          "```bash\n"
          "iree-bazel-run //loom/src/loom/tools/iree-tune-loom -- "
          "path/to/file.loom --device=amdgpu --benchmark=@kernel_latency "
          "--profile_data=counter-ranges --profile_counter=SQ_WAVES "
          "--artifact_bundle_dir=/tmp/loom-run\n"
          "```\n"
          "\n"
          "```bash\n"
          "iree-bazel-run //loom/src/loom/tools/iree-tune-loom -- "
          "path/to/file.loom --device=amdgpu "
          "--compare=@baseline_latency,@variant_latency --interleave=ABABA "
          "--repetitions=2 --sample=0 --profile_final_batch=false "
          "--output=ababa.jsonl\n"
          "```\n"
          "\n"
          "```bash\n"
          "jq 'select(.row==\"benchmark\") | .benchmark_result | "
          "{benchmark,status,p50:.dispatch_timing_ns.p50,stop_reason}' "
          "results.jsonl\n"
          "jq 'select(.row==\"compile\" and .status!=\"ok\") | "
          ".diagnostics[]?' results.jsonl\n"
          "jq 'select(.row==\"compile\" and .static_summary) | "
          "{candidate_id,code:.static_summary.code_byte_count,"
          "valu:.static_summary.vector_alu_count,"
          "spills:.static_summary.allocation_spill_count}' results.jsonl\n"
          "jq 'select(.row==\"compile\" and .compile_report_path) | "
          "{candidate_id,path:.compile_report_path}' results.jsonl\n"
          "jq 'select(.row==\"compile\" and .target_artifact_path) | "
          "{candidate_id,target:.target_artifact_path,"
          "listing:.target_listing_path,hal:.hal_executable_path}' "
          "results.jsonl\n"
          "jq 'select(.row==\"benchmark\" and .shape) | "
          "{candidate_id,shape_id,shape:.shape.parameters,"
          "p50:.benchmark_result.dispatch_timing_ns.p50}' results.jsonl\n"
          "jq 'select(.row==\"benchmark\") | .benchmark_result | "
          "{benchmark,p50:.dispatch_timing_ns.p50,"
          "data_cache}' results.jsonl\n"
          "jq 'select(.row==\"benchmark.repetition\") | "
          "{candidate_id,order_index,token:.schedule_token,"
          "p50:.benchmark_result.dispatch_timing_ns.p50}' results.jsonl\n"
          "jq 'select(.row==\"comparison\") | "
          "{candidate_id,baseline_candidate_id,ratio_p50,speedup_p50,"
          "ratio_p90,speedup_p90,candidate_p50_spread_ppm}' "
          "results.jsonl\n"
          "jq 'select(.row==\"profile_counter\") | .counter | "
          "select(.type==\"counter_group\") | {key,counter,avg,sum}' "
          "results.jsonl\n"
          "jq 'select(.row==\"profile_counter\" and "
          ".counter.type==\"counter_decode_status\") | "
          "{candidate_id,status:.counter.status,reason:.counter.reason,"
          "decoded:.counter.decoded_rows}' "
          "results.jsonl\n"
          "jq 'select(.row==\"profile_summary\" and "
          ".profile_summary.type==\"summary\") | "
          "{candidate_id,traces:.profile_summary.executable_trace_records,"
          "trace_bytes:.profile_summary.executable_trace_data_bytes,"
          "metric_values:.profile_summary.device_metric_values}' "
          "results.jsonl\n"
          "jq 'select(.row==\"profile_summary\" and "
          ".profile_summary.type==\"profile_summary_status\") | "
          "{candidate_id,status:.profile_summary.status,"
          "reason:.profile_summary.reason,"
          "decoded:.profile_summary.decoded_rows}' "
          "results.jsonl\n"
          "tail -f results.jsonl | jq -c 'select(.row==\"compile\" or "
          ".row==\"benchmark\" or .row==\"comparison\" or "
          ".row==\"summary\")'\n"
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
  // Counter-set selection storage referenced by |hal_options|.
  iree_hal_profile_counter_set_selection_t profile_counter_set;
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
  // Effective JSONL output path, or "-" when writing to stdout.
  iree_string_view_t results_path;
  // Directory receiving check.file.write.* outputs for this run.
  iree_string_view_t file_output_dir;
  // Directory receiving raw profile artifacts by default.
  iree_string_view_t profile_artifacts_dir;
  // Directory owning the artifact bundle, or empty when bundling is disabled.
  iree_string_view_t artifact_bundle_dir;
  // Active artifact bundle policy name, or "none" when bundling is disabled.
  iree_string_view_t artifact_bundle_policy;
} iree_tune_loom_run_identity_t;

typedef struct iree_tune_loom_jsonl_sink_t {
  // Host allocator used for scratch row storage.
  iree_allocator_t host_allocator;
  // Open FILE receiving the JSONL stream.
  FILE* file;
  // True when |file| is owned by this sink and must be closed.
  bool owns_file;
  // Scratch storage used to assemble complete rows before writing.
  iree_string_builder_t row_builder;
} iree_tune_loom_jsonl_sink_t;

typedef struct iree_tune_loom_candidate_identity_t {
  // Deterministic candidate identifier within the source/run selection.
  iree_string_view_t candidate_id;
  // Zero-based selected benchmark ordinal within this run.
  iree_host_size_t candidate_index;
} iree_tune_loom_candidate_identity_t;

typedef enum iree_tune_loom_bundle_file_kind_e {
  IREE_TUNE_LOOM_BUNDLE_FILE_FIXTURE_READ = 0,
  IREE_TUNE_LOOM_BUNDLE_FILE_OUTPUT = 1,
  IREE_TUNE_LOOM_BUNDLE_FILE_PROFILE = 2,
  IREE_TUNE_LOOM_BUNDLE_FILE_COMPILE_REPORT = 3,
  IREE_TUNE_LOOM_BUNDLE_FILE_TARGET_ARTIFACT = 4,
  IREE_TUNE_LOOM_BUNDLE_FILE_HAL_EXECUTABLE = 5,
  IREE_TUNE_LOOM_BUNDLE_FILE_TARGET_LISTING = 6,
} iree_tune_loom_bundle_file_kind_t;

typedef struct iree_tune_loom_bundle_file_entry_t {
  // Manifest bucket that owns this file reference.
  iree_tune_loom_bundle_file_kind_t kind;
  // Owned resolved filesystem path.
  char* path;
} iree_tune_loom_bundle_file_entry_t;

typedef struct iree_tune_loom_artifact_bundle_t
    iree_tune_loom_artifact_bundle_t;

typedef struct iree_tune_loom_selected_benchmark_t {
  // Stable candidate identity for rows produced by this selected benchmark.
  iree_tune_loom_candidate_identity_t identity;
  // Inline storage backing |identity.candidate_id|.
  char candidate_id_storage[32];
  // Borrowed benchmark plan selected from the module plan.
  const loom_testbench_benchmark_plan_t* benchmark_plan;
  // Borrowed case plan referenced by |benchmark_plan|.
  const loom_testbench_case_plan_t* case_plan;
  // Effective benchmark policy after command-line overrides.
  iree_tune_loom_benchmark_policy_t policy;
} iree_tune_loom_selected_benchmark_t;

typedef struct iree_tune_loom_file_provider_t {
  // Host allocator used only for path strings and stream-owned storage.
  iree_allocator_t host_allocator;
  // Optional bundle receiving opened fixture/output file references.
  iree_tune_loom_artifact_bundle_t* artifact_bundle;
  // Borrowed directory containing the input module for relative fixture reads.
  iree_string_view_t input_dir;
  // Borrowed view into |output_dir_storage|.
  iree_string_view_t output_dir;
  // Owned directory receiving relative file outputs.
  char* output_dir_storage;
} iree_tune_loom_file_provider_t;

struct iree_tune_loom_artifact_bundle_t {
  // Host allocator used for owned path strings.
  iree_allocator_t host_allocator;
  // True when --artifact_bundle_dir requested a bundle.
  bool enabled;
  // Parsed bundle policy controlling which artifact classes are expected.
  iree_tune_loom_artifact_bundle_policy_t policy;
  // Borrowed view into |dir_storage|.
  iree_string_view_t dir;
  // Owned bundle root directory.
  char* dir_storage;
  // Borrowed view into |results_path_storage|.
  iree_string_view_t results_path;
  // Owned default JSONL result path inside |dir|.
  char* results_path_storage;
  // Borrowed view into |manifest_path_storage|.
  iree_string_view_t manifest_path;
  // Owned manifest path inside |dir|.
  char* manifest_path_storage;
  // Borrowed view into |file_output_dir_storage|.
  iree_string_view_t file_output_dir;
  // Owned default file-output directory inside |dir|.
  char* file_output_dir_storage;
  // Borrowed view into |profile_artifacts_dir_storage|.
  iree_string_view_t profile_artifacts_dir;
  // Owned default profile-artifacts directory inside |dir|.
  char* profile_artifacts_dir_storage;
  // Borrowed view into |compile_report_dir_storage|.
  iree_string_view_t compile_report_dir;
  // Owned debug compile-report directory inside |dir|.
  char* compile_report_dir_storage;
  // Borrowed view into |target_artifact_dir_storage|.
  iree_string_view_t target_artifact_dir;
  // Owned debug target-native artifact directory inside |dir|.
  char* target_artifact_dir_storage;
  // Borrowed view into |target_listing_dir_storage|.
  iree_string_view_t target_listing_dir;
  // Owned debug target-native listing directory inside |dir|.
  char* target_listing_dir_storage;
  // Borrowed view into |hal_executable_dir_storage|.
  iree_string_view_t hal_executable_dir;
  // Owned debug HAL executable package directory inside |dir|.
  char* hal_executable_dir_storage;
  // Owned file references observed while the run executed.
  iree_tune_loom_bundle_file_entry_t* file_entries;
  // Number of populated entries in |file_entries|.
  iree_host_size_t file_entry_count;
  // Allocated capacity of |file_entries|.
  iree_host_size_t file_entry_capacity;
};

static bool iree_tune_loom_path_is_absolute(iree_string_view_t path) {
  if (iree_string_view_is_empty(path)) {
    return false;
  }
  if (path.data[0] == '/' || path.data[0] == '\\') {
    return true;
  }
  if (path.size >= 3 && path.data[1] == ':' &&
      (path.data[2] == '/' || path.data[2] == '\\')) {
    const char drive = path.data[0];
    return (drive >= 'a' && drive <= 'z') || (drive >= 'A' && drive <= 'Z');
  }
  return false;
}

static iree_string_view_t iree_tune_loom_path_as_output_relative(
    iree_string_view_t path) {
  return iree_string_view_trim(path);
}

static bool iree_tune_loom_is_safe_output_path_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-' || c == '/';
}

static iree_status_t iree_tune_loom_validate_file_output_path(
    iree_string_view_t path) {
  if (loom_tooling_file_path_is_stdio(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "check.file.write paths must name a file; '-' "
                            "would conflict with the JSONL output stream");
  }
  if (iree_tune_loom_path_is_absolute(path)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "check.file.write path '%.*s' must be relative to --file_output_dir",
        (int)path.size, path.data);
  }

  iree_string_view_t remaining =
      iree_tune_loom_path_as_output_relative(iree_string_view_trim(path));
  if (iree_string_view_is_empty(remaining)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "check.file.write path resolves to an empty "
                            "artifact name");
  }
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t component = iree_string_view_empty();
    iree_string_view_split(remaining, '/', &component, &remaining);
    if (iree_string_view_is_empty(component) ||
        iree_string_view_equal(component, IREE_SV("."))) {
      continue;
    }
    if (iree_string_view_equal(component, IREE_SV(".."))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "check.file.write path '%.*s' must stay within --file_output_dir",
          (int)path.size, path.data);
    }
    for (iree_host_size_t i = 0; i < component.size; ++i) {
      if (component.data[i] == '\\') {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "check.file.write path '%.*s' must use '/' separators",
            (int)path.size, path.data);
      }
    }
  }
  return iree_ok_status();
}

static uint64_t iree_tune_loom_hash_string_view(uint64_t hash,
                                                iree_string_view_t value) {
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    hash ^= (uint8_t)value.data[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

static iree_status_t iree_tune_loom_append_sanitized_path_component(
    iree_string_view_t value, iree_string_builder_t* builder) {
  if (iree_string_view_equal(value, IREE_SV("<stdin>"))) {
    return iree_string_builder_append_cstring(builder, "stdin");
  }

  bool appended = false;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    char c = value.data[i];
    if (!iree_tune_loom_is_safe_output_path_char(c) || c == '/') {
      c = '_';
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        builder, iree_make_string_view(&c, 1)));
    appended = true;
  }
  if (!appended) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "input"));
  }
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_join_path(iree_string_view_t lhs,
                                              iree_string_view_t rhs,
                                              iree_allocator_t allocator,
                                              char** out_path) {
  *out_path = NULL;
  return iree_file_path_join(lhs, rhs, allocator, out_path);
}

static iree_status_t iree_tune_loom_dup_string_view(iree_string_view_t value,
                                                    iree_allocator_t allocator,
                                                    char** out_value) {
  *out_value = NULL;
  iree_host_size_t storage_size = 0;
  if (!iree_host_size_checked_add(value.size, 1, &storage_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "string storage size overflow");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, storage_size, (void**)&storage));
  if (value.size != 0) {
    memcpy(storage, value.data, value.size);
  }
  storage[value.size] = '\0';
  *out_value = storage;
  return iree_ok_status();
}

static iree_string_view_t iree_tune_loom_tmp_dir(void) {
  const char* tmp_dir = getenv("TMPDIR");
  if (tmp_dir == NULL || tmp_dir[0] == '\0') {
    tmp_dir = getenv("TEMP");
  }
  if (tmp_dir == NULL || tmp_dir[0] == '\0') {
    tmp_dir = getenv("TMP");
  }
  if (tmp_dir == NULL || tmp_dir[0] == '\0') {
    tmp_dir = "/tmp";
  }
  return iree_make_cstring_view(tmp_dir);
}

static iree_status_t iree_tune_loom_make_default_file_output_dir(
    iree_string_view_t filename, iree_string_view_t run_id,
    iree_allocator_t allocator, char** out_path) {
  *out_path = NULL;

  uint64_t hash = 1469598103934665603ull;
  hash = iree_tune_loom_hash_string_view(hash, filename);
  hash = iree_tune_loom_hash_string_view(hash, run_id);

  iree_string_builder_t leaf_builder;
  iree_string_builder_initialize(allocator, &leaf_builder);
  iree_string_view_t source_name = iree_file_path_basename(filename);
  iree_status_t status = iree_tune_loom_append_sanitized_path_component(
      source_name, &leaf_builder);
  if (iree_status_is_ok(status)) {
    status =
        iree_string_builder_append_format(&leaf_builder, "_%016" PRIx64, hash);
  }
  iree_string_view_t leaf = iree_string_builder_view(&leaf_builder);

  char* root_path = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_join_path(iree_tune_loom_tmp_dir(),
                                      IREE_SV("iree-loom-tune"), allocator,
                                      &root_path);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_join_path(iree_make_cstring_view(root_path), leaf,
                                      allocator, out_path);
  }
  iree_allocator_free(allocator, root_path);
  iree_string_builder_deinitialize(&leaf_builder);
  return status;
}

static bool iree_tune_loom_artifact_bundle_has_file(
    const iree_tune_loom_artifact_bundle_t* bundle,
    iree_tune_loom_bundle_file_kind_t kind, iree_string_view_t path) {
  for (iree_host_size_t i = 0; i < bundle->file_entry_count; ++i) {
    const iree_tune_loom_bundle_file_entry_t* entry = &bundle->file_entries[i];
    if (entry->kind == kind &&
        iree_string_view_equal(iree_make_cstring_view(entry->path), path)) {
      return true;
    }
  }
  return false;
}

static iree_status_t iree_tune_loom_artifact_bundle_record_file(
    iree_tune_loom_artifact_bundle_t* bundle,
    iree_tune_loom_bundle_file_kind_t kind, iree_string_view_t path) {
  if (bundle == NULL || !bundle->enabled) {
    return iree_ok_status();
  }
  path = iree_string_view_trim(path);
  if (iree_string_view_is_empty(path) ||
      loom_tooling_file_path_is_stdio(path) ||
      iree_tune_loom_artifact_bundle_has_file(bundle, kind, path)) {
    return iree_ok_status();
  }

  if (bundle->file_entry_count == bundle->file_entry_capacity) {
    iree_host_size_t new_capacity = 8;
    if (bundle->file_entry_capacity != 0 &&
        !iree_host_size_checked_mul(bundle->file_entry_capacity, 2,
                                    &new_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "too many artifact bundle file entries");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
        bundle->host_allocator, new_capacity, sizeof(bundle->file_entries[0]),
        (void**)&bundle->file_entries));
    bundle->file_entry_capacity = new_capacity;
  }

  iree_tune_loom_bundle_file_entry_t* entry =
      &bundle->file_entries[bundle->file_entry_count];
  entry->kind = kind;
  IREE_RETURN_IF_ERROR(iree_tune_loom_dup_string_view(
      path, bundle->host_allocator, &entry->path));
  ++bundle->file_entry_count;
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_file_provider_initialize(
    iree_string_view_t filename, iree_string_view_t run_id,
    iree_string_view_t default_output_dir,
    iree_tune_loom_artifact_bundle_t* artifact_bundle,
    iree_allocator_t allocator, iree_tune_loom_file_provider_t* out_provider) {
  memset(out_provider, 0, sizeof(*out_provider));
  out_provider->host_allocator = allocator;
  out_provider->artifact_bundle = artifact_bundle;
  if (!iree_string_view_equal(filename, IREE_SV("<stdin>"))) {
    out_provider->input_dir = iree_file_path_dirname(filename);
  }

  iree_string_view_t output_dir =
      iree_string_view_trim(iree_make_cstring_view(FLAG_file_output_dir));
  if (!iree_string_view_is_empty(output_dir)) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_dup_string_view(
        output_dir, allocator, &out_provider->output_dir_storage));
  } else if (!iree_string_view_is_empty(default_output_dir)) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_dup_string_view(
        default_output_dir, allocator, &out_provider->output_dir_storage));
  } else {
    IREE_RETURN_IF_ERROR(iree_tune_loom_make_default_file_output_dir(
        filename, run_id, allocator, &out_provider->output_dir_storage));
  }
  out_provider->output_dir =
      iree_make_cstring_view(out_provider->output_dir_storage);
  return iree_ok_status();
}

static void iree_tune_loom_file_provider_deinitialize(
    iree_tune_loom_file_provider_t* provider) {
  if (!provider) {
    return;
  }
  iree_allocator_free(provider->host_allocator, provider->output_dir_storage);
  memset(provider, 0, sizeof(*provider));
}

static iree_status_t iree_tune_loom_resolve_file_read_path(
    const iree_tune_loom_file_provider_t* provider, iree_string_view_t path,
    char** out_path) {
  *out_path = NULL;
  if (loom_tooling_file_path_is_stdio(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "check.file.read paths must name a file");
  }
  if (iree_tune_loom_path_is_absolute(path) ||
      iree_string_view_is_empty(provider->input_dir)) {
    return iree_tune_loom_dup_string_view(path, provider->host_allocator,
                                          out_path);
  }
  return iree_tune_loom_join_path(provider->input_dir, path,
                                  provider->host_allocator, out_path);
}

static iree_status_t iree_tune_loom_resolve_file_write_path(
    const iree_tune_loom_file_provider_t* provider, iree_string_view_t path,
    char** out_path) {
  *out_path = NULL;
  IREE_RETURN_IF_ERROR(iree_tune_loom_validate_file_output_path(path));
  iree_string_view_t relative_path =
      iree_tune_loom_path_as_output_relative(iree_string_view_trim(path));
  return iree_tune_loom_join_path(provider->output_dir, relative_path,
                                  provider->host_allocator, out_path);
}

static iree_status_t iree_tune_loom_create_directory_if_needed(
    iree_string_view_t path, iree_allocator_t allocator) {
  path = iree_string_view_trim(path);
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }

  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_dup_string_view(path, allocator, &storage));
  iree_host_size_t length = strlen(storage);
  while (length > 1 && storage[length - 1] == '/') {
    storage[--length] = '\0';
  }
  if (length == 1 && storage[0] == '/') {
    iree_allocator_free(allocator, storage);
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 1; i < length && iree_status_is_ok(status); ++i) {
    if (storage[i] != '/') {
      continue;
    }
    storage[i] = '\0';
    if (!(i == 2 && storage[1] == ':') && strlen(storage) != 0) {
#if defined(IREE_PLATFORM_WINDOWS)
      const int result = _mkdir(storage);
#else
      const int result = mkdir(storage, 0700);
#endif  // defined(IREE_PLATFORM_WINDOWS)
      if (result != 0 && errno != EEXIST) {
        status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                  "failed to create directory `%s`: %s",
                                  storage, strerror(errno));
      }
    }
    storage[i] = '/';
  }
  if (iree_status_is_ok(status)) {
#if defined(IREE_PLATFORM_WINDOWS)
    const int result = _mkdir(storage);
#else
    const int result = mkdir(storage, 0700);
#endif  // defined(IREE_PLATFORM_WINDOWS)
    if (result != 0 && errno != EEXIST) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "failed to create directory `%s`: %s", storage,
                                strerror(errno));
    }
  }
  iree_allocator_free(allocator, storage);
  return status;
}

static iree_status_t iree_tune_loom_create_parent_directory(
    iree_string_view_t path, iree_allocator_t allocator) {
  iree_string_view_t parent = iree_file_path_dirname(path);
  if (iree_string_view_is_empty(parent)) {
    return iree_ok_status();
  }
  return iree_tune_loom_create_directory_if_needed(parent, allocator);
}

static iree_status_t iree_tune_loom_jsonl_sink_initialize(
    iree_string_view_t path, iree_allocator_t allocator,
    iree_tune_loom_jsonl_sink_t* out_sink) {
  memset(out_sink, 0, sizeof(*out_sink));
  out_sink->host_allocator = allocator;
  iree_string_builder_initialize(allocator, &out_sink->row_builder);

  path = iree_string_view_trim(path);
  if (loom_tooling_file_path_is_stdio(path)) {
    out_sink->file = stdout;
    return iree_ok_status();
  }

  iree_status_t status =
      iree_tune_loom_create_parent_directory(path, allocator);
  char* storage = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_dup_string_view(path, allocator, &storage);
  }
  if (iree_status_is_ok(status)) {
    out_sink->file = fopen(storage, "wb");
    if (out_sink->file == NULL) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "failed to open JSONL output `%s`: %s", storage,
                                strerror(errno));
    } else {
      out_sink->owns_file = true;
    }
  }
  iree_allocator_free(allocator, storage);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&out_sink->row_builder);
    memset(out_sink, 0, sizeof(*out_sink));
  }
  return status;
}

static void iree_tune_loom_jsonl_sink_deinitialize(
    iree_tune_loom_jsonl_sink_t* sink) {
  if (sink == NULL) {
    return;
  }
  if (sink->owns_file && sink->file != NULL) {
    fclose(sink->file);
  }
  iree_string_builder_deinitialize(&sink->row_builder);
  memset(sink, 0, sizeof(*sink));
}

static iree_string_builder_t* iree_tune_loom_jsonl_sink_begin(
    iree_tune_loom_jsonl_sink_t* sink) {
  iree_string_builder_reset(&sink->row_builder);
  return &sink->row_builder;
}

static iree_status_t iree_tune_loom_jsonl_sink_flush(
    iree_tune_loom_jsonl_sink_t* sink) {
  iree_string_view_t contents = iree_string_builder_view(&sink->row_builder);
  if (iree_string_view_is_empty(contents)) {
    return iree_ok_status();
  }
  const size_t written = fwrite(contents.data, 1, contents.size, sink->file);
  if (written != contents.size) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to write %" PRIhsz
                            " bytes to JSONL output: %s",
                            contents.size, strerror(errno));
  }
  if (fflush(sink->file) != 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to flush JSONL output: %s",
                            strerror(errno));
  }
  iree_string_builder_reset(&sink->row_builder);
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_jsonl_sink_close(
    iree_tune_loom_jsonl_sink_t* sink) {
  if (!sink->owns_file || sink->file == NULL) {
    return iree_ok_status();
  }
  if (fclose(sink->file) != 0) {
    sink->file = NULL;
    sink->owns_file = false;
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to close JSONL output: %s",
                            strerror(errno));
  }
  sink->file = NULL;
  sink->owns_file = false;
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_jsonl_sink_end(
    iree_tune_loom_jsonl_sink_t* sink, iree_status_t row_status) {
  if (!iree_status_is_ok(row_status)) {
    iree_string_builder_reset(&sink->row_builder);
    return row_status;
  }
  return iree_tune_loom_jsonl_sink_flush(sink);
}

static iree_status_t iree_tune_loom_join_bundle_path(
    iree_string_view_t bundle_dir, iree_string_view_t child,
    iree_allocator_t allocator, char** out_path) {
  return iree_tune_loom_join_path(bundle_dir, child, allocator, out_path);
}

static void iree_tune_loom_artifact_bundle_deinitialize(
    iree_tune_loom_artifact_bundle_t* bundle) {
  if (bundle == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < bundle->file_entry_count; ++i) {
    iree_allocator_free(bundle->host_allocator, bundle->file_entries[i].path);
  }
  iree_allocator_free(bundle->host_allocator, bundle->file_entries);
  iree_allocator_free(bundle->host_allocator,
                      bundle->hal_executable_dir_storage);
  iree_allocator_free(bundle->host_allocator,
                      bundle->target_artifact_dir_storage);
  iree_allocator_free(bundle->host_allocator,
                      bundle->target_listing_dir_storage);
  iree_allocator_free(bundle->host_allocator,
                      bundle->compile_report_dir_storage);
  iree_allocator_free(bundle->host_allocator,
                      bundle->profile_artifacts_dir_storage);
  iree_allocator_free(bundle->host_allocator, bundle->file_output_dir_storage);
  iree_allocator_free(bundle->host_allocator, bundle->manifest_path_storage);
  iree_allocator_free(bundle->host_allocator, bundle->results_path_storage);
  iree_allocator_free(bundle->host_allocator, bundle->dir_storage);
  *bundle = (iree_tune_loom_artifact_bundle_t){0};
}

static iree_status_t iree_tune_loom_artifact_bundle_initialize(
    iree_allocator_t allocator, iree_tune_loom_artifact_bundle_t* out_bundle) {
  memset(out_bundle, 0, sizeof(*out_bundle));
  out_bundle->host_allocator = allocator;

  const iree_string_view_t bundle_dir =
      iree_string_view_trim(iree_make_cstring_view(FLAG_artifact_bundle_dir));
  IREE_RETURN_IF_ERROR(iree_tune_loom_parse_artifact_bundle_policy(
      iree_make_cstring_view(FLAG_artifact_bundle_policy),
      &out_bundle->policy));
  if (iree_string_view_is_empty(bundle_dir)) {
    out_bundle->policy = IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_NONE;
    return iree_ok_status();
  }
  if (loom_tooling_file_path_is_stdio(bundle_dir)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--artifact_bundle_dir must name a directory");
  }
  if (out_bundle->policy == IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--artifact_bundle_policy=none conflicts with "
                            "--artifact_bundle_dir");
  }

  iree_status_t status = iree_tune_loom_dup_string_view(
      bundle_dir, allocator, &out_bundle->dir_storage);
  if (iree_status_is_ok(status)) {
    out_bundle->dir = iree_make_cstring_view(out_bundle->dir_storage);
    status =
        iree_tune_loom_create_directory_if_needed(out_bundle->dir, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("results.jsonl"), allocator,
        &out_bundle->results_path_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("manifest.json"), allocator,
        &out_bundle->manifest_path_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("outputs"), allocator,
        &out_bundle->file_output_dir_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("profiles"), allocator,
        &out_bundle->profile_artifacts_dir_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("compile_reports"), allocator,
        &out_bundle->compile_report_dir_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("target_artifacts"), allocator,
        &out_bundle->target_artifact_dir_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("target_listings"), allocator,
        &out_bundle->target_listing_dir_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_join_bundle_path(
        out_bundle->dir, IREE_SV("hal_executables"), allocator,
        &out_bundle->hal_executable_dir_storage);
  }
  if (iree_status_is_ok(status)) {
    out_bundle->results_path =
        iree_make_cstring_view(out_bundle->results_path_storage);
    out_bundle->manifest_path =
        iree_make_cstring_view(out_bundle->manifest_path_storage);
    out_bundle->file_output_dir =
        iree_make_cstring_view(out_bundle->file_output_dir_storage);
    out_bundle->profile_artifacts_dir =
        iree_make_cstring_view(out_bundle->profile_artifacts_dir_storage);
    out_bundle->compile_report_dir =
        iree_make_cstring_view(out_bundle->compile_report_dir_storage);
    out_bundle->target_artifact_dir =
        iree_make_cstring_view(out_bundle->target_artifact_dir_storage);
    out_bundle->target_listing_dir =
        iree_make_cstring_view(out_bundle->target_listing_dir_storage);
    out_bundle->hal_executable_dir =
        iree_make_cstring_view(out_bundle->hal_executable_dir_storage);
    out_bundle->enabled = true;
  } else {
    iree_tune_loom_artifact_bundle_deinitialize(out_bundle);
  }
  return status;
}

static iree_string_view_t iree_tune_loom_effective_results_output_path(
    const iree_tune_loom_artifact_bundle_t* bundle) {
  const iree_string_view_t explicit_output =
      iree_string_view_trim(iree_make_cstring_view(FLAG_output));
  if (!iree_string_view_is_empty(explicit_output)) {
    return explicit_output;
  }
  if (bundle->enabled) {
    return bundle->results_path;
  }
  return iree_string_view_empty();
}

static iree_string_view_t iree_tune_loom_effective_profile_artifacts_dir(
    const iree_tune_loom_artifact_bundle_t* bundle) {
  const iree_string_view_t explicit_artifacts_dir =
      iree_string_view_trim(iree_make_cstring_view(FLAG_profile_artifacts_dir));
  if (!iree_string_view_is_empty(explicit_artifacts_dir)) {
    return explicit_artifacts_dir;
  }
  return bundle->enabled ? bundle->profile_artifacts_dir
                         : iree_string_view_empty();
}

static iree_status_t iree_tune_loom_open_file_for_read(
    void* user_data, iree_string_view_t path, iree_io_stream_t** out_stream) {
  *out_stream = NULL;
  iree_tune_loom_file_provider_t* provider =
      (iree_tune_loom_file_provider_t*)user_data;
  char* resolved_path = NULL;
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_resolve_file_read_path(provider, path, &resolved_path));
  iree_status_t status = iree_io_stdio_stream_open(
      IREE_IO_STDIO_STREAM_MODE_READ, iree_make_cstring_view(resolved_path),
      provider->host_allocator, out_stream);
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_artifact_bundle_record_file(
        provider->artifact_bundle, IREE_TUNE_LOOM_BUNDLE_FILE_FIXTURE_READ,
        iree_make_cstring_view(resolved_path));
  }
  iree_allocator_free(provider->host_allocator, resolved_path);
  return status;
}

static iree_status_t iree_tune_loom_open_file_for_write(
    void* user_data, iree_string_view_t path, iree_io_stream_t** out_stream) {
  *out_stream = NULL;
  iree_tune_loom_file_provider_t* provider =
      (iree_tune_loom_file_provider_t*)user_data;
  char* resolved_path = NULL;
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_resolve_file_write_path(provider, path, &resolved_path));
  iree_string_view_t resolved_path_view = iree_make_cstring_view(resolved_path);
  iree_status_t status = iree_tune_loom_create_parent_directory(
      resolved_path_view, provider->host_allocator);
  if (iree_status_is_ok(status)) {
    status = iree_io_stdio_stream_open(
        IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_DISCARD,
        resolved_path_view, provider->host_allocator, out_stream);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_artifact_bundle_record_file(
        provider->artifact_bundle, IREE_TUNE_LOOM_BUNDLE_FILE_OUTPUT,
        resolved_path_view);
  }
  iree_allocator_free(provider->host_allocator, resolved_path);
  return status;
}

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

typedef struct iree_tune_loom_data_cache_summary_t {
  // True when the dispatch benchmark has populated this summary.
  bool populated;
  // Number of HAL binding references in each dispatch binding set.
  iree_host_size_t binding_count;
  // Number of physical binding sets materialized from check ops.
  iree_host_size_t binding_ring_count;
  // Number of pre-recorded command buffers rotated across benchmark batches.
  iree_host_size_t command_buffer_ring_count;
  // Number of dispatch slots recorded in each command buffer.
  iree_host_size_t dispatches_per_batch;
  // Requested minimum byte size for the physical binding ring.
  uint64_t requested_min_ring_bytes;
  // Byte length of the first materialized binding set.
  uint64_t binding_set_bytes;
  // Sum of byte lengths across materialized binding sets.
  uint64_t binding_ring_bytes;
} iree_tune_loom_data_cache_summary_t;

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
  // Sidecar compile report artifact path for debug/full bundles, if any.
  iree_string_view_t compile_report_artifact_path;
  // Target-native executable artifact path for debug/full bundles, if any.
  iree_string_view_t target_artifact_path;
  // Target-native textual listing path for debug/full bundles, if any.
  iree_string_view_t target_listing_path;
  // HAL executable package artifact path for debug/full bundles, if any.
  iree_string_view_t hal_executable_path;
  // True when benchmark setup and timing completed.
  bool executed;
  // True when no measured or warmup sample failed expectations.
  bool passed;
  // Shape specialization label for this benchmark result.
  iree_string_view_t specialization;
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
  // Device-buffer reuse shape used by the HAL batch benchmark.
  iree_tune_loom_data_cache_summary_t data_cache;
} iree_tune_loom_benchmark_result_t;

typedef struct iree_tune_loom_hal_context_t {
  // Tool configuration with linked backend registries.
  const iree_tune_loom_configuration_t* configuration;
  // Host allocator used for runtime and candidate storage.
  iree_allocator_t host_allocator;
  // Optional artifact bundle receiving HAL profile artifact references.
  iree_tune_loom_artifact_bundle_t* artifact_bundle;
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
  // Shape specialization label for rows emitted from this provider.
  iree_string_view_t specialization;
  // Case plan that owns parameter facts used for per-shape specialization.
  const loom_testbench_case_plan_t* specialization_case_plan;
  // Shape sample ordinal used to specialize this provider's compile module.
  iree_host_size_t specialization_sample_ordinal;
  // True when |specialization_sample_ordinal| is meaningful.
  bool has_specialization_sample_ordinal;
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
  // Borrowed view into |compile_report_artifact_path_storage|.
  iree_string_view_t compile_report_artifact_path;
  // Owned debug/full bundle compile-report artifact path.
  char* compile_report_artifact_path_storage;
  // Borrowed view into |target_artifact_path_storage|.
  iree_string_view_t target_artifact_path;
  // Owned debug/full bundle target-native artifact path.
  char* target_artifact_path_storage;
  // Borrowed view into |target_listing_path_storage|.
  iree_string_view_t target_listing_path;
  // Owned debug/full bundle target-native listing path.
  char* target_listing_path_storage;
  // Borrowed view into |hal_executable_path_storage|.
  iree_string_view_t hal_executable_path;
  // Owned debug/full bundle HAL executable package path.
  char* hal_executable_path_storage;
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
  // Number of function-like region arguments replaced by constants.
  iree_host_size_t specialized_argument_count;
} iree_tune_loom_hal_actual_provider_t;

typedef struct iree_tune_loom_dispatch_comparison_candidate_t {
  // Selected benchmark/case/policy identity for this comparison member.
  const iree_tune_loom_selected_benchmark_t* selection;
  // Shape specialization label for this prepared candidate.
  iree_string_view_t specialization;
  // True when |specialization_sample_ordinal| is the only comparable shape.
  bool has_specialization_sample_ordinal;
  // Concrete sample ordinal used for per-sample specialization.
  iree_host_size_t specialization_sample_ordinal;
  // First sample ordinal included in the comparison window.
  iree_host_size_t begin_sample;
  // One-past-end sample ordinal included in the comparison window.
  iree_host_size_t end_sample;
  // HAL provider owning the compiled executable and prepared candidate.
  iree_tune_loom_hal_actual_provider_t provider;
  // True when |provider| needs deinitialization.
  bool provider_initialized;
  // Materializer used to build device-buffer binding rings for timing windows.
  loom_testbench_value_materializer_options_t benchmark_materializer;
  // Number of correctness samples executed before interleaved timing.
  iree_host_size_t correctness_sample_count;
  // Number of failed correctness samples observed before interleaved timing.
  iree_host_size_t correctness_failed_sample_count;
  // True when compile and correctness succeeded and timing windows can run.
  bool runnable;
  // Per-repetition p50 dispatch timings collected for aggregate rows.
  iree_duration_t* p50_samples;
  // Per-repetition p90 dispatch timings collected for aggregate rows.
  iree_duration_t* p90_samples;
  // Number of entries populated in |p50_samples| and |p90_samples|.
  iree_host_size_t sample_count;
  // Capacity of |p50_samples| and |p90_samples|.
  iree_host_size_t sample_capacity;
} iree_tune_loom_dispatch_comparison_candidate_t;

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

// Formats a borrowed status as JSON. The caller retains status ownership.
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
  iree_status_t write_status =
      iree_tune_loom_write_status_object_json(status, stream);
  iree_status_free(status);
  return write_status;
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

static iree_status_t iree_tune_loom_write_specialization_field_json(
    iree_string_view_t specialization, loom_output_stream_t* stream) {
  if (iree_string_view_is_empty(specialization)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"specialization\":"));
  return loom_json_write_escaped_string(stream, specialization);
}

static const char* iree_tune_loom_parameter_kind_name(
    loom_testbench_parameter_kind_t kind) {
  switch (kind) {
    case LOOM_TESTBENCH_PARAMETER_RANGE:
      return "range";
    case LOOM_TESTBENCH_PARAMETER_CHOICE:
      return "choice";
    case LOOM_TESTBENCH_PARAMETER_SEED:
      return "seed";
    default:
      return "unknown";
  }
}

static iree_string_view_t iree_tune_loom_value_name(const loom_module_t* module,
                                                    loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return iree_string_view_empty();
  }
  const loom_string_id_t name_id = module->values.entries[value_id].name_id;
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[name_id];
}

static iree_status_t iree_tune_loom_write_parameter_name_json(
    const loom_module_t* module,
    const loom_testbench_parameter_plan_t* parameter,
    iree_host_size_t parameter_index, loom_output_stream_t* stream) {
  iree_string_view_t name =
      iree_tune_loom_value_name(module, parameter->value_id);
  if (!iree_string_view_is_empty(name)) {
    return loom_json_write_escaped_string(stream, name);
  }
  return loom_output_stream_write_format(stream, "\"param_%" PRIhsz "\"",
                                         parameter_index);
}

static iree_status_t iree_tune_loom_write_sample_attr_json(
    loom_attribute_t value, loom_output_stream_t* stream) {
  switch ((loom_attr_kind_t)value.kind) {
    case LOOM_ATTR_I64:
      return loom_output_stream_write_format(stream, "%" PRIi64,
                                             loom_attr_as_i64(value));
    case LOOM_ATTR_F64: {
      const double f64 = loom_attr_as_f64(value);
      if (!isfinite(f64)) {
        return loom_json_write_escaped_cstring(stream, "nonfinite");
      }
      char buffer[64];
      const int length = iree_snprintf(buffer, sizeof(buffer), "%.17g", f64);
      if (length <= 0 || (iree_host_size_t)length >= sizeof(buffer)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to format f64 sample value");
      }
      return loom_output_stream_write(stream,
                                      iree_make_string_view(buffer, length));
    }
    case LOOM_ATTR_BOOL:
      return loom_output_stream_write_cstring(
          stream, loom_attr_as_bool(value) ? "true" : "false");
    case LOOM_ATTR_STRING:
      return loom_json_write_escaped_cstring(stream, "<string>");
    default:
      return loom_json_write_escaped_cstring(stream, "<unsupported>");
  }
}

static iree_status_t iree_tune_loom_write_shape_parameter_map_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, bool write_ordinals,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  for (iree_host_size_t parameter_index = 0;
       parameter_index < case_plan->parameter_count; ++parameter_index) {
    const loom_testbench_parameter_plan_t* parameter =
        &case_plan->parameters[parameter_index];
    const iree_host_size_t parameter_sample_ordinal =
        loom_testbench_case_sample_parameter_ordinal(case_plan, sample_ordinal,
                                                     parameter_index);
    if (parameter_index != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_parameter_name_json(
        module, parameter, parameter_index, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ":"));
    if (write_ordinals) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream, "%" PRIhsz, parameter_sample_ordinal));
    } else {
      loom_attribute_t sample_value = loom_attr_absent();
      IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
          parameter, parameter_sample_ordinal, &sample_value));
      IREE_RETURN_IF_ERROR(
          iree_tune_loom_write_sample_attr_json(sample_value, stream));
    }
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_tune_loom_write_shape_point_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"sample_ordinal\":%" PRIhsz, sample_ordinal));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"parameter_count\":%" PRIhsz, case_plan->parameter_count));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"parameters\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_shape_parameter_map_json(
      module, case_plan, sample_ordinal, /*write_ordinals=*/false, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"parameter_ordinals\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_shape_parameter_map_json(
      module, case_plan, sample_ordinal, /*write_ordinals=*/true, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_tune_loom_write_shape_point_fields_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t sample_ordinal, loom_output_stream_t* stream) {
  if (case_plan->parameter_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"shape_id\":\"s%" PRIhsz "\",\"shape_index\":%" PRIhsz,
      sample_ordinal, sample_ordinal));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"shape\":"));
  return iree_tune_loom_write_shape_point_json(module, case_plan,
                                               sample_ordinal, stream);
}

static iree_status_t iree_tune_loom_write_shape_plan_fields_json(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    loom_output_stream_t* stream) {
  if (case_plan->parameter_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      ",\"shape_sample_count\":%" PRIhsz
      ",\"shape_cartesian_sample_count\":%" PRIhsz
      ",\"shape_sample_count_truncated\":%s",
      case_plan->sample_count, case_plan->cartesian_sample_count,
      case_plan->sample_count_truncated ? "true" : "false"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"shape_parameters\":["));
  for (iree_host_size_t parameter_index = 0;
       parameter_index < case_plan->parameter_count; ++parameter_index) {
    const loom_testbench_parameter_plan_t* parameter =
        &case_plan->parameters[parameter_index];
    if (parameter_index != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "{\"name\":"));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_parameter_name_json(
        module, parameter, parameter_index, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"kind\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
        stream, iree_tune_loom_parameter_kind_name(parameter->kind)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"sample_count\":%" PRIhsz, parameter->sample_count));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t iree_tune_loom_append_run_row(
    const iree_tune_loom_run_identity_t* run, bool dry_run,
    iree_tune_loom_shape_specialization_mode_t shape_specialization_mode,
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
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"results_path\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, run->results_path));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"file_output_dir\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, run->file_output_dir));
  if (!iree_string_view_is_empty(run->profile_artifacts_dir)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"profile_artifacts_dir\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, run->profile_artifacts_dir));
  }
  if (!iree_string_view_is_empty(run->artifact_bundle_dir)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"artifact_bundle\":{\"dir\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, run->artifact_bundle_dir));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"policy\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, run->artifact_bundle_policy));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"dry_run\":%s", dry_run ? "true" : "false"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"shape_specialization\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_tune_loom_shape_specialization_mode_name(
                   shape_specialization_mode)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_write_hal_context_identity_fields_json(
    const iree_tune_loom_hal_context_t* context, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"device_uri\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, iree_tune_loom_selected_device_uri(context)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"driver\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, context->backend->hal_driver_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"backend\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, context->backend->name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"target_family\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, context->backend->target_family_name));
  if (context->runtime_initialized && context->runtime.device != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"status\":\"created\""));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"device_id\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, iree_hal_device_id(context->runtime.device)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"queries\":{"));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "\"attempted\":true"));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_optional_i64_query_json(
        context->runtime.device, IREE_SV("hal.device"), IREE_SV("concurrency"),
        "hal_device_concurrency", stream));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_optional_i64_query_json(
        context->runtime.device, IREE_SV("hal.dispatch"),
        IREE_SV("concurrency"), "hal_dispatch_concurrency", stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));

    iree_hal_device_capabilities_t capabilities = {0};
    iree_status_t capabilities_status = iree_hal_device_query_capabilities(
        context->runtime.device, &capabilities);
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"capabilities\":"));
    if (iree_status_is_ok(capabilities_status)) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream,
          "{\"flags\":%" PRIu64 ",\"numa_node\":%" PRIu8
          ",\"has_physical_device_uuid\":%s,\"device_group_index\":%" PRIu32
          ",\"has_device_group\":%s",
          capabilities.flags, capabilities.numa_node,
          capabilities.has_physical_device_uuid ? "true" : "false",
          capabilities.device_group_index,
          capabilities.has_device_group ? "true" : "false"));
      if (capabilities.has_physical_device_uuid) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
            stream, ",\"physical_device_uuid\":"));
        IREE_RETURN_IF_ERROR(iree_tune_loom_write_hex_bytes_json(
            capabilities.physical_device_uuid,
            IREE_ARRAYSIZE(capabilities.physical_device_uuid), stream));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
    } else {
      iree_status_t write_status =
          iree_tune_loom_write_status_object_json(capabilities_status, stream);
      iree_status_free(capabilities_status);
      IREE_RETURN_IF_ERROR(write_status);
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"status\":\"planned\""));
  }
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
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_hal_context_identity_fields_json(context, &stream));
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

static iree_status_t iree_tune_loom_write_f64_json(
    double value, loom_output_stream_t* stream) {
  if (!isfinite(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "metadata f64 value is not finite");
  }
  char buffer[64];
  const int length = iree_snprintf(buffer, sizeof(buffer), "%.17g", value);
  if (length <= 0 || (iree_host_size_t)length >= sizeof(buffer)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to format f64 attribute value");
  }
  return loom_output_stream_write(stream,
                                  iree_make_string_view(buffer, length));
}

static iree_status_t iree_tune_loom_write_string_id_json(
    const loom_module_t* module, loom_string_id_t string_id,
    loom_output_stream_t* stream) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "metadata string id %u is outside the module "
                            "string table",
                            string_id);
  }
  return loom_json_write_escaped_string(stream,
                                        module->strings.entries[string_id]);
}

static iree_status_t iree_tune_loom_write_symbol_ref_json(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref,
    loom_output_stream_t* stream) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "metadata symbol reference is outside the source "
                            "module symbol table");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "metadata symbol name id %u is outside the module "
                            "string table",
                            symbol->name_id);
  }
  return loom_json_write_escaped_string(
      stream, module->strings.entries[symbol->name_id]);
}

static iree_status_t iree_tune_loom_write_predicate_arg_tag_json(
    uint8_t arg_tag, loom_output_stream_t* stream) {
  switch ((loom_predicate_arg_tag_t)arg_tag) {
    case LOOM_PRED_ARG_NONE:
      return loom_json_write_escaped_cstring(stream, "none");
    case LOOM_PRED_ARG_VALUE:
      return loom_json_write_escaped_cstring(stream, "value");
    case LOOM_PRED_ARG_CONST:
      return loom_json_write_escaped_cstring(stream, "const");
    case LOOM_PRED_ARG_COUNT_:
      break;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported predicate argument tag %u",
                          (unsigned)arg_tag);
}

static iree_status_t iree_tune_loom_write_predicate_json(
    const loom_predicate_t* predicate, loom_output_stream_t* stream) {
  const char* kind_name = loom_predicate_kind_name(predicate->kind);
  if (kind_name == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported predicate kind %u",
                            (unsigned)predicate->kind);
  }
  const uint8_t expected_arg_count =
      loom_predicate_kind_argument_count(predicate->kind);
  if (predicate->arg_count != expected_arg_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "predicate `%s` expected %u arguments but found %u",
                            kind_name, (unsigned)expected_arg_count,
                            (unsigned)predicate->arg_count);
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\"kind\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, kind_name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"args\":["));
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\"tag\":"));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_predicate_arg_tag_json(
        predicate->arg_tags[i], stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"value\":%" PRId64 "}", predicate->args[i]));
  }
  return loom_output_stream_write_cstring(stream, "]}");
}

static iree_status_t iree_tune_loom_write_attr_json(
    const loom_module_t* module, const loom_attribute_t* attr,
    loom_output_stream_t* stream, uint8_t depth) {
  if (depth >= LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "attribute nesting exceeds %u",
                            (unsigned)LOOM_ATTR_DICT_MAX_NESTING_DEPTH);
  }
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_ABSENT:
      return loom_output_stream_write_cstring(stream, "null");
    case LOOM_ATTR_I64:
      return loom_output_stream_write_format(stream, "%" PRId64, attr->i64);
    case LOOM_ATTR_F64:
      return iree_tune_loom_write_f64_json(attr->f64, stream);
    case LOOM_ATTR_STRING:
      return iree_tune_loom_write_string_id_json(module, attr->string_id,
                                                 stream);
    case LOOM_ATTR_BOOL:
      return loom_output_stream_write_cstring(stream,
                                              attr->raw ? "true" : "false");
    case LOOM_ATTR_ENUM:
      return loom_output_stream_write_format(stream, "%" PRIu64, attr->raw);
    case LOOM_ATTR_I64_ARRAY: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
      for (uint16_t i = 0; i < attr->count; ++i) {
        if (i != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
            stream, "%" PRId64, attr->i64_array[i]));
      }
      return loom_output_stream_write_cstring(stream, "]");
    }
    case LOOM_ATTR_SYMBOL:
      return iree_tune_loom_write_symbol_ref_json(module, attr->symbol, stream);
    case LOOM_ATTR_TYPE:
      return loom_output_stream_write_format(stream, "%" PRIu32, attr->type_id);
    case LOOM_ATTR_PREDICATE_LIST: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
      for (uint16_t i = 0; i < attr->count; ++i) {
        if (i != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        IREE_RETURN_IF_ERROR(iree_tune_loom_write_predicate_json(
            &attr->predicate_list[i], stream));
      }
      return loom_output_stream_write_cstring(stream, "]");
    }
    case LOOM_ATTR_DICT: {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
      for (uint16_t i = 0; i < attr->count; ++i) {
        if (i != 0) {
          IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
        }
        const loom_named_attr_t* entry = &attr->dict_entries[i];
        IREE_RETURN_IF_ERROR(iree_tune_loom_write_string_id_json(
            module, entry->name_id, stream));
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ":"));
        IREE_RETURN_IF_ERROR(iree_tune_loom_write_attr_json(
            module, &entry->value, stream, (uint8_t)(depth + 1)));
      }
      return loom_output_stream_write_cstring(stream, "}");
    }
    case LOOM_ATTR_ENCODING:
      return loom_output_stream_write_format(stream, "%" PRIu32,
                                             attr->encoding_id);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported metadata attribute kind %u",
                              (unsigned)attr->kind);
  }
}

static bool iree_tune_loom_benchmark_metadata_has_any(
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan) {
  static const char* metadata_keys[] = {
      "family", "phase", "strategy", "knobs", "problem", "reference_id",
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(metadata_keys); ++i) {
    const iree_string_view_t metadata_key =
        iree_make_cstring_view(metadata_keys[i]);
    if (iree_tune_loom_find_named_attr(module, benchmark_plan->attrs,
                                       metadata_key) != NULL) {
      return true;
    }
  }
  return false;
}

static iree_status_t iree_tune_loom_write_benchmark_metadata_json(
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    loom_output_stream_t* stream) {
  static const char* metadata_keys[] = {
      "family", "phase", "strategy", "knobs", "problem", "reference_id",
  };
  if (!iree_tune_loom_benchmark_metadata_has_any(module, benchmark_plan)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"metadata\":{"));
  bool first = true;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(metadata_keys); ++i) {
    const iree_string_view_t metadata_key =
        iree_make_cstring_view(metadata_keys[i]);
    const loom_named_attr_t* attr = iree_tune_loom_find_named_attr(
        module, benchmark_plan->attrs, metadata_key);
    if (attr == NULL) {
      continue;
    }
    if (!first) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, metadata_key));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ":"));
    IREE_RETURN_IF_ERROR(
        iree_tune_loom_write_attr_json(module, &attr->value, stream, 0));
    first = false;
  }
  return loom_output_stream_write_cstring(stream, "}");
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

static iree_status_t iree_tune_loom_parse_profile_data_families(
    iree_string_view_t value,
    iree_hal_device_profiling_data_families_t* out_data_families) {
  *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                       IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  iree_string_view_t remaining = iree_string_view_trim(value);
  if (iree_string_view_is_empty(remaining)) {
    return iree_ok_status();
  }
  *out_data_families = IREE_HAL_DEVICE_PROFILING_DATA_NONE;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t family_part = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &family_part, &remaining);
    family_part = iree_string_view_trim(family_part);
    if (iree_string_view_is_empty(family_part)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--profile_data contains an empty data family");
    }
    if (iree_string_view_equal(family_part, IREE_SV("none"))) {
      if (*out_data_families != IREE_HAL_DEVICE_PROFILING_DATA_NONE ||
          !iree_string_view_is_empty(iree_string_view_trim(remaining))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--profile_data=none cannot be combined with "
                                "other data families");
      }
      return iree_ok_status();
    }
    bool matched = false;
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(iree_tune_loom_profile_family_names); ++i) {
      const iree_tune_loom_profile_family_name_t* family =
          &iree_tune_loom_profile_family_names[i];
      if (iree_string_view_equal(family_part,
                                 iree_make_cstring_view(family->flag_name)) ||
          iree_string_view_equal(family_part,
                                 iree_make_cstring_view(family->json_name))) {
        *out_data_families |= family->bit;
        matched = true;
        break;
      }
    }
    if (!matched &&
        iree_string_view_equal(family_part, IREE_SV("pmc-ranges"))) {
      *out_data_families |= IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES;
      matched = true;
    }
    if (!matched) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported --profile_data family '%.*s'",
                              (int)family_part.size, family_part.data);
    }
    remaining = iree_string_view_trim(remaining);
  }
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
    iree_string_view_t specialization,
    const loom_testbench_case_plan_t* specialization_case_plan,
    iree_host_size_t specialization_sample_ordinal,
    bool has_specialization_sample_ordinal,
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
      .specialization = specialization,
      .specialization_case_plan = specialization_case_plan,
      .specialization_sample_ordinal = specialization_sample_ordinal,
      .has_specialization_sample_ordinal = has_specialization_sample_ordinal,
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
  iree_allocator_free(provider->context->host_allocator,
                      provider->hal_executable_path_storage);
  iree_allocator_free(provider->context->host_allocator,
                      provider->target_artifact_path_storage);
  iree_allocator_free(provider->context->host_allocator,
                      provider->target_listing_path_storage);
  iree_allocator_free(provider->context->host_allocator,
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

static bool iree_tune_loom_find_parameter_index_for_value(
    const loom_testbench_case_plan_t* case_plan, loom_value_id_t value_id,
    iree_host_size_t* out_parameter_index) {
  if (case_plan == NULL) {
    return false;
  }
  for (iree_host_size_t parameter_index = 0;
       parameter_index < case_plan->parameter_count; ++parameter_index) {
    if (case_plan->parameters[parameter_index].value_id == value_id) {
      *out_parameter_index = parameter_index;
      return true;
    }
  }
  return false;
}

static bool iree_tune_loom_value_facts_from_sample_attr(
    loom_attribute_t attr, loom_value_facts_t* out_facts) {
  switch ((loom_attr_kind_t)attr.kind) {
    case LOOM_ATTR_I64:
      *out_facts = loom_value_facts_exact_i64(loom_attr_as_i64(attr));
      return true;
    case LOOM_ATTR_F64:
      *out_facts = loom_value_facts_exact_f64(loom_attr_as_f64(attr));
      return true;
    case LOOM_ATTR_BOOL:
      *out_facts = loom_value_facts_exact_i64(loom_attr_as_bool(attr) ? 1 : 0);
      return true;
    default:
      *out_facts = loom_value_facts_unknown();
      return false;
  }
}

static iree_status_t iree_tune_loom_hal_actual_provider_resolve_compile_func(
    iree_tune_loom_hal_actual_provider_t* provider,
    iree_string_view_t entry_symbol, loom_func_like_t* out_func) {
  *out_func = (loom_func_like_t){0};
  loom_module_t* module = provider->compile_module.module;
  const loom_string_id_t entry_name_id =
      loom_module_lookup_string(module, entry_symbol);
  if (entry_name_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "entry symbol '@%.*s' was not found in compile "
                            "module string table",
                            (int)entry_symbol.size, entry_symbol.data);
  }
  const uint16_t symbol_id = loom_module_find_symbol(module, entry_name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID ||
      symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "entry symbol '@%.*s' was not found in compile "
                            "module symbol table",
                            (int)entry_symbol.size, entry_symbol.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  loom_func_like_t func = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(func)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry symbol '@%.*s' does not define a function",
                            (int)entry_symbol.size, entry_symbol.data);
  }
  *out_func = func;
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_specialize_func_region_argument(
    loom_module_t* module, loom_func_like_t func, uint8_t region_index,
    uint16_t argument_index, loom_value_facts_t facts,
    iree_host_size_t* inout_specialized_count) {
  loom_region_t* region = loom_func_like_region(func, region_index);
  if (region == NULL || region->block_count == 0) {
    return iree_ok_status();
  }
  loom_block_t* entry_block = loom_region_entry_block(region);
  if (argument_index >= entry_block->arg_count) {
    return iree_ok_status();
  }
  const loom_value_id_t argument_id =
      loom_block_arg_id(entry_block, argument_index);
  const loom_type_t argument_type = loom_module_value_type(module, argument_id);
  if (!loom_value_facts_can_materialize_constant(facts, argument_type)) {
    return iree_ok_status();
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, entry_block, &builder);
  if (entry_block->first_op != NULL) {
    loom_builder_set_before(&builder, entry_block->first_op);
  }
  const loom_location_id_t location = entry_block->first_op != NULL
                                          ? entry_block->first_op->location
                                          : func.op->location;
  loom_value_id_t replacement_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_constant_build(&builder, facts, argument_type,
                                           location, &replacement_id));
  IREE_RETURN_IF_ERROR(
      loom_value_replace_all_uses_with(module, argument_id, replacement_id));
  *inout_specialized_count += 1;
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_hal_actual_provider_specialize_sample(
    iree_tune_loom_hal_actual_provider_t* provider,
    iree_string_view_t entry_symbol) {
  if (!provider->has_specialization_sample_ordinal ||
      provider->specialization_case_plan == NULL) {
    return iree_ok_status();
  }

  loom_func_like_t func = {0};
  IREE_RETURN_IF_ERROR(iree_tune_loom_hal_actual_provider_resolve_compile_func(
      provider, entry_symbol, &func));
  loom_module_t* module = provider->compile_module.module;
  for (iree_host_size_t input_index = 0;
       input_index < provider->actual_invocation->input_count; ++input_index) {
    if (input_index > UINT16_MAX) {
      continue;
    }
    iree_host_size_t parameter_index = 0;
    if (!iree_tune_loom_find_parameter_index_for_value(
            provider->specialization_case_plan,
            provider->actual_invocation->input_value_ids[input_index],
            &parameter_index)) {
      continue;
    }
    const iree_host_size_t parameter_sample_ordinal =
        loom_testbench_case_sample_parameter_ordinal(
            provider->specialization_case_plan,
            provider->specialization_sample_ordinal, parameter_index);
    loom_attribute_t sample_value = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
        &provider->specialization_case_plan->parameters[parameter_index],
        parameter_sample_ordinal, &sample_value));
    loom_value_facts_t facts = loom_value_facts_unknown();
    if (!iree_tune_loom_value_facts_from_sample_attr(sample_value, &facts)) {
      continue;
    }

    const uint8_t region_count = loom_func_like_region_count(func);
    for (uint8_t region_index = 0; region_index < region_count;
         ++region_index) {
      if (!loom_func_like_region_is_body(func, region_index) &&
          !loom_func_like_region_projects_args(module, func, region_index)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(iree_tune_loom_specialize_func_region_argument(
          module, func, region_index, (uint16_t)input_index, facts,
          &provider->specialized_argument_count));
    }
  }
  return iree_ok_status();
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
  IREE_RETURN_IF_ERROR(iree_tune_loom_hal_actual_provider_specialize_sample(
      provider, entry_symbol));

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
      // The diagnostic sink owns the candidate rejection evidence; the status
      // only carries the same non-infrastructure failure.
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
  if (provider->context->artifact_bundle != NULL &&
      provider->context->artifact_bundle->enabled &&
      provider->context->artifact_bundle->policy >=
          IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG) {
    compile_options.artifact_flags |=
        LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_TARGET_LISTING;
  }
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
      // The diagnostic sink owns the candidate rejection evidence; the status
      // only carries the same non-infrastructure failure.
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

static iree_status_t iree_tune_loom_hal_invocation_options_push_constant(
    const iree_vm_variant_t* variant,
    loom_run_hal_invocation_options_t* options) {
  if (options->constant_count >= LOOM_RUN_HAL_MAX_CONSTANT_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL dispatch constant count exceeds capacity "
                            "%" PRIhsz,
                            (iree_host_size_t)LOOM_RUN_HAL_MAX_CONSTANT_COUNT);
  }
  const iree_vm_value_t value = iree_vm_variant_value(*variant);
  uint32_t raw_value = 0;
  switch (value.type) {
    case IREE_VM_VALUE_TYPE_I8:
      raw_value = (uint32_t)(int32_t)value.i8;
      break;
    case IREE_VM_VALUE_TYPE_I16:
      raw_value = (uint32_t)(int32_t)value.i16;
      break;
    case IREE_VM_VALUE_TYPE_I32:
      raw_value = (uint32_t)value.i32;
      break;
    case IREE_VM_VALUE_TYPE_I64:
      if (value.i64 < INT32_MIN || value.i64 > UINT32_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "HAL dispatch scalar i64 value %" PRIi64
            " does not fit in the current 32-bit direct argument ABI",
            value.i64);
      }
      raw_value = (uint32_t)value.i64;
      break;
    case IREE_VM_VALUE_TYPE_F32:
      memcpy(&raw_value, &value.f32, sizeof(raw_value));
      break;
    case IREE_VM_VALUE_TYPE_F64:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "HAL dispatch scalar f64 values are not supported by the current "
          "32-bit direct argument ABI");
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM value type %d cannot be passed as a HAL "
                              "dispatch constant",
                              (int)value.type);
  }
  options->constants[options->constant_count++] = raw_value;
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_borrowed_hal_input_append(
    iree_vm_list_t* bindings, const iree_vm_variant_t* input,
    loom_run_hal_invocation_options_t* options) {
  if (iree_vm_variant_is_ref(*input)) {
    return iree_vm_list_push_variant_retain(bindings, input);
  }
  if (iree_vm_variant_is_value(*input)) {
    return iree_tune_loom_hal_invocation_options_push_constant(input, options);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "HAL invocation input must be a buffer reference or "
                          "a scalar VM value");
}

static iree_status_t iree_tune_loom_owned_hal_input_append(
    iree_vm_list_t* bindings, iree_vm_variant_t* input,
    loom_run_hal_invocation_options_t* options) {
  if (iree_vm_variant_is_ref(*input)) {
    return iree_vm_list_push_variant_move(bindings, input);
  }
  if (iree_vm_variant_is_value(*input)) {
    return iree_tune_loom_hal_invocation_options_push_constant(input, options);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "HAL invocation input must be a buffer reference or "
                          "a scalar VM value");
}

static iree_status_t iree_tune_loom_hal_invocation_inputs_from_variants(
    const iree_vm_variant_t* inputs, iree_host_size_t input_count,
    loom_run_hal_invocation_options_t* options, iree_allocator_t allocator,
    iree_vm_list_t** out_bindings) {
  *out_bindings = NULL;
  iree_vm_list_t* bindings = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                           input_count, allocator, &bindings));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < input_count;
       ++i) {
    status =
        iree_tune_loom_borrowed_hal_input_append(bindings, &inputs[i], options);
  }
  if (iree_status_is_ok(status)) {
    *out_bindings = bindings;
  } else {
    iree_vm_list_release(bindings);
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

  loom_run_hal_invocation_options_t invocation_options =
      provider->invocation_options;
  iree_vm_list_t* bindings = NULL;
  IREE_RETURN_IF_ERROR(iree_tune_loom_hal_invocation_inputs_from_variants(
      inputs, input_count, &invocation_options,
      provider->context->host_allocator, &bindings));

  loom_run_hal_invocation_plan_t plan = {0};
  loom_run_hal_iteration_t iteration = {0};
  iree_status_t status = loom_run_hal_invocation_plan_prepare_from_lists(
      &invocation_options, bindings, /*expected_bindings=*/NULL,
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

static iree_status_t iree_tune_loom_create_hal_invocation_inputs_for_sample(
    const loom_module_t* module,
    const loom_testbench_value_materializer_options_t* materializer_options,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t sample_ordinal,
    const loom_run_hal_invocation_options_t* base_options,
    iree_allocator_t allocator, loom_run_hal_invocation_options_t* out_options,
    iree_vm_list_t** out_bindings) {
  *out_bindings = NULL;
  *out_options = *base_options;
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
      status = iree_tune_loom_owned_hal_input_append(bindings, &variant,
                                                     out_options);
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
  iree_status_t status = iree_tune_loom_create_hal_invocation_inputs_for_sample(
      module_plan->module, materializer_options, case_plan,
      provider->actual_invocation, sample_ordinal,
      &provider->invocation_options, allocator, &invocation_options, &bindings);
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
  out_result->failure_stage = provider->compile_failure_stage;
  out_result->failure_kind = provider->compile_failure_kind;
  out_result->failure_message = provider->compile_failure_message;
  out_result->diagnostic_error_count = provider->diagnostics.error_count;
  out_result->diagnostic_warning_count = provider->diagnostics.warning_count;
  out_result->diagnostic_remark_count = provider->diagnostics.remark_count;
  out_result->diagnostic_json =
      iree_string_builder_view(&provider->diagnostics.output);
  out_result->specialization = provider->specialization;
  if (provider->has_specialization_sample_ordinal) {
    out_result->has_sample_ordinal = true;
    out_result->sample_ordinal = provider->specialization_sample_ordinal;
    out_result->samples_per_iteration = 1;
  }
  if (provider->compile_report_available) {
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
  if (provider->compile_report_available) {
    out_result->compile_report_capture = &provider->compile_report_capture;
  }
  out_result->compile_report_artifact_path =
      provider->compile_report_artifact_path;
  out_result->target_artifact_path = provider->target_artifact_path;
  out_result->target_listing_path = provider->target_listing_path;
  out_result->hal_executable_path = provider->hal_executable_path;
  if (provider->compile_rejected) {
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
          &provider->context->runtime, &provider->prepared_candidate,
          &input_ring.plans[0], input_ring.plan_count, input_ring.binding_lists,
          &hal_options, allocator, &out_result->hal_benchmark);
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
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  bool first = true;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_tune_loom_profile_family_names); ++i) {
    const iree_tune_loom_profile_family_name_t* family =
        &iree_tune_loom_profile_family_names[i];
    if (!iree_all_bits_set(data_families, family->bit)) {
      continue;
    }
    if (!first) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_cstring(stream, family->json_name));
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
  if (profile->has_artifact_path) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"artifact_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, iree_make_string_view(profile->artifact_path,
                                      profile->artifact_path_length)));
  }
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

static iree_string_view_t iree_tune_loom_compile_artifact_kind_name(
    loom_target_compile_artifact_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_NONE:
      return IREE_SV("none");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE:
      return IREE_SV("vm-archive");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE:
      return IREE_SV("hal-executable");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_KERNEL_LIBRARY:
      return IREE_SV("hal-kernel-library");
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t iree_tune_loom_write_json_object_field_name(
    loom_output_stream_t* stream, bool* first_field, const char* name) {
  if (!*first_field) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  }
  *first_field = false;
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(stream, name));
  return loom_output_stream_write_cstring(stream, ":");
}

static iree_status_t iree_tune_loom_write_json_string_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_json_object_field_name(stream, first_field, name));
  return loom_json_write_escaped_string(stream, value);
}

static iree_status_t iree_tune_loom_write_json_optional_string_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) {
    return iree_ok_status();
  }
  return iree_tune_loom_write_json_string_field(stream, first_field, name,
                                                value);
}

static iree_status_t iree_tune_loom_write_json_u32_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    uint32_t value) {
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_json_object_field_name(stream, first_field, name));
  return loom_output_stream_write_format(stream, "%" PRIu32, value);
}

static iree_status_t iree_tune_loom_write_json_u64_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    uint64_t value) {
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_json_object_field_name(stream, first_field, name));
  return loom_output_stream_write_format(stream, "%" PRIu64, value);
}

static iree_status_t iree_tune_loom_write_json_size_field(
    loom_output_stream_t* stream, bool* first_field, const char* name,
    iree_host_size_t value) {
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_json_object_field_name(stream, first_field, name));
  return loom_output_stream_write_format(stream, "%" PRIhsz, value);
}

static void iree_tune_loom_compile_report_move_cause_totals(
    const loom_target_compile_report_t* report, uint64_t* out_kind_count,
    uint64_t* out_packet_count, uint64_t* out_unit_count) {
  *out_kind_count = 0;
  *out_packet_count = 0;
  *out_unit_count = 0;
  for (iree_host_size_t i = 1; i < LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT;
       ++i) {
    const loom_target_compile_report_move_cause_counts_t* counts =
        &report->move_causes[i];
    if (counts->packet_count == 0 && counts->unit_count == 0) {
      continue;
    }
    ++*out_kind_count;
    *out_packet_count += counts->packet_count;
    *out_unit_count += counts->unit_count;
  }
}

static iree_status_t iree_tune_loom_write_static_summary_json(
    const loom_run_compile_report_capture_t* compile_report_capture,
    loom_output_stream_t* stream) {
  if (compile_report_capture == NULL ||
      compile_report_capture->options.mode ==
          LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    return iree_ok_status();
  }
  const loom_target_compile_report_t* report = &compile_report_capture->report;
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"static_summary\":{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
      stream, &first_field, "artifact_kind",
      iree_tune_loom_compile_artifact_kind_name(report->artifact_kind)));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
      stream, &first_field, "status",
      iree_make_cstring_view(iree_status_code_string(report->status_code))));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u32_field(
      stream, &first_field, "detail_flags", report->detail_flags));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_optional_string_field(
      stream, &first_field, "backend", report->backend_name));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_optional_string_field(
      stream, &first_field, "target_family", report->target_family_name));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_optional_string_field(
      stream, &first_field, "target_key", report->target_key));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_optional_string_field(
      stream, &first_field, "target_bundle", report->target_bundle_name));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_optional_string_field(
      stream, &first_field, "target_export", report->target_export_name));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_optional_string_field(
      stream, &first_field, "target_config", report->target_config_name));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_optional_string_field(
      stream, &first_field, "lowered", report->lowered_symbol));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_optional_string_field(
      stream, &first_field, "executable_format", report->executable_format));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE)) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "artifact_size", report->artifact_size));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION)) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "instruction_count",
        report->emitted_instruction_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "code_byte_count",
        report->emitted_code_byte_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "code_storage_byte_count",
        report->emitted_code_storage_byte_count));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MEMORY)) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "private_memory_bytes",
        report->private_memory_bytes));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "local_memory_bytes",
        report->local_memory_bytes));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION)) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "allocation_assignment_count",
        report->allocation_assignment_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "allocation_spill_count",
        report->allocation_spill_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "allocation_spill_plan_count",
        report->allocation_spill_plan_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "allocation_coalesced_copy_count",
        report->allocation_coalesced_copy_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "allocation_materialized_copy_count",
        report->allocation_materialized_copy_count));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE)) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "schedule_node_count",
        report->schedule_node_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "scheduled_node_count",
        report->scheduled_node_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "register_pressure_summary_count",
        report->register_pressure_summary_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "register_pressure_peak_live_units",
        report->register_pressure_peak_live_units));
  }
  if (iree_any_bit_set(
          report->detail_flags,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_STATIC_INSTRUCTION_MIX)) {
    const loom_target_compile_report_static_instruction_mix_t* mix =
        &report->static_instruction_mix;
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "descriptor_count", mix->descriptor_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "unknown_descriptor_count", mix->unknown_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "scalar_alu_count", mix->scalar_alu_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "vector_alu_count", mix->vector_alu_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "matrix_count", mix->matrix_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "mfma_count", mix->mfma_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "wmma_count", mix->wmma_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "dot_count", mix->dot_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "global_memory_count", mix->global_memory_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "local_memory_count", mix->local_memory_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "scalar_memory_count", mix->scalar_memory_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "generic_memory_count",
        mix->generic_memory_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "atomic_count", mix->atomic_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "branch_count", mix->branch_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "barrier_count", mix->barrier_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "control_count", mix->control_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "conversion_count", mix->conversion_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "cache_count", mix->cache_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "register_move_count", mix->register_move_count));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "source_low_selected_op_count",
        report->source_low_selected_op_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "source_low_emitted_op_count",
        report->source_low_emitted_op_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
        stream, &first_field, "source_low_row_count",
        report->source_low_row_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
        stream, &first_field, "source_low_row_total_count",
        report->source_low_row_total_count));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
        stream, &first_field, "pressure_row_count",
        report->pressure_row_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
        stream, &first_field, "pressure_row_total_count",
        report->pressure_row_total_count));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
        stream, &first_field, "spill_row_count", report->spill_row_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
        stream, &first_field, "spill_row_total_count",
        report->spill_row_total_count));
  }
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES)) {
    uint64_t kind_count = 0;
    uint64_t packet_count = 0;
    uint64_t unit_count = 0;
    iree_tune_loom_compile_report_move_cause_totals(report, &kind_count,
                                                    &packet_count, &unit_count);
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "move_cause_kind_count", kind_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "move_cause_packet_count", packet_count));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
        stream, &first_field, "move_cause_unit_count", unit_count));
  }
  return loom_output_stream_write_cstring(stream, "}");
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

static bool iree_tune_loom_artifact_bundle_wants_debug_artifacts(
    const iree_tune_loom_artifact_bundle_t* bundle) {
  return bundle != NULL && bundle->enabled &&
         bundle->policy >= IREE_TUNE_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG;
}

static bool iree_tune_loom_artifact_bundle_wants_compile_reports(
    const iree_tune_loom_artifact_bundle_t* bundle) {
  return iree_tune_loom_artifact_bundle_wants_debug_artifacts(bundle);
}

static iree_status_t iree_tune_loom_append_candidate_artifact_stem(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const iree_tune_loom_hal_actual_provider_t* provider,
    iree_string_builder_t* stem) {
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_append_sanitized_path_component(run->run_id, stem));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(stem, "_"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_append_sanitized_path_component(
      candidate->candidate_id, stem));
  if (!iree_string_view_is_empty(provider->specialization)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(stem, "_"));
    IREE_RETURN_IF_ERROR(iree_tune_loom_append_sanitized_path_component(
        provider->specialization, stem));
  }
  if (provider->has_specialization_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        stem, "_sample%" PRIhsz, provider->specialization_sample_ordinal));
  }
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_append_compile_report_artifact_leaf(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const iree_tune_loom_hal_actual_provider_t* provider,
    iree_string_builder_t* leaf) {
  IREE_RETURN_IF_ERROR(iree_tune_loom_append_candidate_artifact_stem(
      run, candidate, provider, leaf));
  return iree_string_builder_append_cstring(leaf, "_compile_report.json");
}

static iree_status_t iree_tune_loom_append_artifact_extension(
    iree_string_view_t format, iree_string_view_t fallback_extension,
    iree_string_builder_t* leaf) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(leaf, "."));
  if (iree_string_view_is_empty(format)) {
    return iree_tune_loom_append_sanitized_path_component(fallback_extension,
                                                          leaf);
  }
  return iree_tune_loom_append_sanitized_path_component(format, leaf);
}

static iree_status_t iree_tune_loom_write_candidate_byte_artifact(
    iree_tune_loom_artifact_bundle_t* bundle,
    iree_tune_loom_bundle_file_kind_t kind, iree_string_view_t directory,
    iree_string_view_t leaf, iree_const_byte_span_t contents,
    iree_allocator_t allocator, char** inout_path_storage,
    iree_string_view_t* inout_path) {
  if (!iree_tune_loom_artifact_bundle_wants_debug_artifacts(bundle) ||
      !iree_string_view_is_empty(*inout_path)) {
    return iree_ok_status();
  }
  if (contents.data == NULL || contents.data_length == 0) {
    return iree_ok_status();
  }

  char* path_storage = NULL;
  iree_status_t status =
      iree_tune_loom_join_path(directory, leaf, allocator, &path_storage);
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_create_parent_directory(
        iree_make_cstring_view(path_storage), allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        iree_make_cstring_view(path_storage),
        iree_make_string_view((const char*)contents.data, contents.data_length),
        allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_artifact_bundle_record_file(
        bundle, kind, iree_make_cstring_view(path_storage));
  }
  if (iree_status_is_ok(status)) {
    iree_allocator_free(allocator, *inout_path_storage);
    *inout_path_storage = path_storage;
    *inout_path = iree_make_cstring_view(path_storage);
    path_storage = NULL;
  }
  iree_allocator_free(allocator, path_storage);
  return status;
}

static iree_status_t iree_tune_loom_write_compiled_artifacts(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    iree_tune_loom_hal_actual_provider_t* provider,
    iree_allocator_t allocator) {
  iree_tune_loom_artifact_bundle_t* bundle = provider->context->artifact_bundle;
  if (!iree_tune_loom_artifact_bundle_wants_debug_artifacts(bundle) ||
      !provider->candidate_initialized || !provider->candidate.compiled) {
    return iree_ok_status();
  }

  iree_string_builder_t leaf;
  iree_string_builder_initialize(allocator, &leaf);
  const loom_run_hal_executable_t* executable = &provider->candidate.executable;
  iree_status_t status = iree_tune_loom_append_candidate_artifact_stem(
      run, candidate, provider, &leaf);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&leaf, "_target");
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_append_artifact_extension(
        executable->target_artifact_format, IREE_SV("bin"), &leaf);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_write_candidate_byte_artifact(
        bundle, IREE_TUNE_LOOM_BUNDLE_FILE_TARGET_ARTIFACT,
        bundle->target_artifact_dir, iree_string_builder_view(&leaf),
        executable->target_artifact_data, allocator,
        &provider->target_artifact_path_storage,
        &provider->target_artifact_path);
  }

  iree_string_builder_reset(&leaf);
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_append_candidate_artifact_stem(run, candidate,
                                                           provider, &leaf);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&leaf, "_target_listing");
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_append_artifact_extension(
        executable->target_listing_format, IREE_SV("txt"), &leaf);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_write_candidate_byte_artifact(
        bundle, IREE_TUNE_LOOM_BUNDLE_FILE_TARGET_LISTING,
        bundle->target_listing_dir, iree_string_builder_view(&leaf),
        executable->target_listing_data, allocator,
        &provider->target_listing_path_storage, &provider->target_listing_path);
  }

  iree_string_builder_reset(&leaf);
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_append_candidate_artifact_stem(run, candidate,
                                                           provider, &leaf);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&leaf, "_hal_executable.hal");
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_write_candidate_byte_artifact(
        bundle, IREE_TUNE_LOOM_BUNDLE_FILE_HAL_EXECUTABLE,
        bundle->hal_executable_dir, iree_string_builder_view(&leaf),
        executable->executable_data, allocator,
        &provider->hal_executable_path_storage, &provider->hal_executable_path);
  }
  iree_string_builder_deinitialize(&leaf);
  return status;
}

static iree_status_t iree_tune_loom_append_compile_report_artifact_json(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_hal_actual_provider_t* provider,
    iree_string_builder_t* output) {
  iree_string_view_t entry_symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_tune_loom_module_symbol_name_from_ref(
      provider->test_module, provider->actual_invocation->callee_ref,
      &entry_symbol));

  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "{\"type\":\"compile_report\""));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_candidate_identity_json(candidate, &stream));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_specialization_field_json(
      provider->specialization, &stream));
  if (provider->has_specialization_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_shape_point_fields_json(
        provider->test_module, case_plan,
        provider->specialization_sample_ordinal, &stream));
  }
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
  if (!iree_string_view_is_empty(provider->target_artifact_path)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"target_artifact_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->target_artifact_path));
  }
  if (!iree_string_view_is_empty(provider->target_listing_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"target_listing_path\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, provider->target_listing_path));
  }
  if (!iree_string_view_is_empty(provider->hal_executable_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"hal_executable_path\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, provider->hal_executable_path));
  }
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
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"specialized_argument_count\":%" PRIhsz,
      provider->specialized_argument_count));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"diagnostics\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_diagnostic_array_json(
      &provider->diagnostics, &stream));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_static_summary_json(
      &provider->compile_report_capture, &stream));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_compile_report_json(
      &provider->compile_report_capture, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_write_compile_report_artifact(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_tune_loom_hal_actual_provider_t* provider,
    iree_allocator_t allocator) {
  if (!provider->compile_report_available ||
      provider->compile_report_capture.options.mode ==
          LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE ||
      !iree_string_view_is_empty(provider->compile_report_artifact_path)) {
    return iree_ok_status();
  }
  iree_tune_loom_artifact_bundle_t* bundle = provider->context->artifact_bundle;
  if (!iree_tune_loom_artifact_bundle_wants_compile_reports(bundle)) {
    return iree_ok_status();
  }

  iree_string_builder_t leaf;
  iree_string_builder_initialize(allocator, &leaf);
  char* path_storage = NULL;
  iree_status_t status = iree_tune_loom_append_compile_report_artifact_leaf(
      run, candidate, provider, &leaf);
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_join_path(bundle->compile_report_dir,
                                      iree_string_builder_view(&leaf),
                                      allocator, &path_storage);
  }

  iree_string_builder_t artifact;
  iree_string_builder_initialize(allocator, &artifact);
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_append_compile_report_artifact_json(
        run, candidate, benchmark_plan, case_plan, provider, &artifact);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_create_parent_directory(
        iree_make_cstring_view(path_storage), allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        iree_make_cstring_view(path_storage),
        iree_string_builder_view(&artifact), allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_artifact_bundle_record_file(
        bundle, IREE_TUNE_LOOM_BUNDLE_FILE_COMPILE_REPORT,
        iree_make_cstring_view(path_storage));
  }
  if (iree_status_is_ok(status)) {
    provider->compile_report_artifact_path_storage = path_storage;
    provider->compile_report_artifact_path =
        iree_make_cstring_view(path_storage);
    path_storage = NULL;
  }
  iree_string_builder_deinitialize(&artifact);
  iree_allocator_free(allocator, path_storage);
  iree_string_builder_deinitialize(&leaf);
  return status;
}

static iree_status_t iree_tune_loom_write_data_cache_summary_json(
    const iree_tune_loom_data_cache_summary_t* summary,
    loom_output_stream_t* stream) {
  if (!summary->populated) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"data_cache\":{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"validity\":\"check_ops\""));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, ",\"cache_policy\":\"binding_ring\""));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"binding_count\":%" PRIhsz, summary->binding_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"binding_ring_count\":%" PRIhsz, summary->binding_ring_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"command_buffer_ring_count\":%" PRIhsz,
      summary->command_buffer_ring_count));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"dispatches_per_batch\":%" PRIhsz,
      summary->dispatches_per_batch));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"requested_min_ring_bytes\":%" PRIu64,
      summary->requested_min_ring_bytes));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"binding_set_bytes\":%" PRIu64, summary->binding_set_bytes));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"binding_ring_bytes\":%" PRIu64, summary->binding_ring_bytes));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"hot_reuse\":"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, summary->binding_ring_count <= 1 ? "true" : "false"));
  return loom_output_stream_write_cstring(stream, "}");
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
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_specialization_field_json(
      benchmark_result->specialization, stream));
  if (benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_format(stream, ",\"sample_ordinal\":%" PRIhsz,
                                        benchmark_result->sample_ordinal));
  }
  if (!iree_string_view_is_empty(
          benchmark_result->compile_report_artifact_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"compile_report_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, benchmark_result->compile_report_artifact_path));
  }
  if (!iree_string_view_is_empty(benchmark_result->target_artifact_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"target_artifact_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, benchmark_result->target_artifact_path));
  }
  if (!iree_string_view_is_empty(benchmark_result->target_listing_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"target_listing_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, benchmark_result->target_listing_path));
  }
  if (!iree_string_view_is_empty(benchmark_result->hal_executable_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"hal_executable_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, benchmark_result->hal_executable_path));
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
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_data_cache_summary_json(
        &benchmark_result->data_cache, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"profile\":"));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_hal_profile_summary_json(
        &benchmark_result->hal_benchmark.profile, stream));
  }
  if (benchmark_result->compile_report_capture != NULL &&
      benchmark_result->compile_report_capture->options.mode !=
          LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_static_summary_json(
        benchmark_result->compile_report_capture, stream));
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
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_specialization_field_json(
      provider->specialization, &stream));
  if (provider->has_specialization_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_shape_point_fields_json(
        provider->test_module, case_plan,
        provider->specialization_sample_ordinal, &stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"benchmark\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"case\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, case_plan->name));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_benchmark_metadata_json(
      provider->test_module, benchmark_plan, &stream));
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
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"specialized_argument_count\":%" PRIhsz,
      provider->specialized_argument_count));
  if (!iree_string_view_is_empty(provider->compile_report_artifact_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"compile_report_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->compile_report_artifact_path));
  }
  if (!iree_string_view_is_empty(provider->target_artifact_path)) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        &stream, ",\"target_artifact_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        &stream, provider->target_artifact_path));
  }
  if (!iree_string_view_is_empty(provider->target_listing_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"target_listing_path\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, provider->target_listing_path));
  }
  if (!iree_string_view_is_empty(provider->hal_executable_path)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"hal_executable_path\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, provider->hal_executable_path));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"diagnostics\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_diagnostic_array_json(
      &provider->diagnostics, &stream));
  if (provider->compile_report_available) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_static_summary_json(
        &provider->compile_report_capture, &stream));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_compile_report_json(
        &provider->compile_report_capture, &stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_append_benchmark_result(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
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
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_specialization_field_json(
      benchmark_result->specialization, &stream));
  if (benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_shape_point_fields_json(
        module, case_plan, benchmark_result->sample_ordinal, &stream));
  }
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_benchmark_metadata_json(
      module, benchmark_plan, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"benchmark_result\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_benchmark_result_json(
      benchmark_plan, case_plan, policy, benchmark_result,
      correctness_sample_count, correctness_failed_sample_count, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_read_file_into_builder(
    FILE* file, iree_string_builder_t* output) {
  if (fflush(file) != 0) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to flush temporary report stream: %s",
                            strerror(errno));
  }
  if (fseek(file, 0, SEEK_SET) != 0) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to rewind temporary report stream: %s",
                            strerror(errno));
  }

  char buffer[4096];
  while (true) {
    const size_t read_count = fread(buffer, 1, sizeof(buffer), file);
    if (read_count != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
          output, iree_make_string_view(buffer, read_count)));
    }
    if (read_count < sizeof(buffer)) {
      break;
    }
  }
  if (ferror(file)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to read temporary report stream: %s",
                            strerror(errno));
  }
  return iree_ok_status();
}

typedef struct iree_tune_loom_status_json_error_t {
  // Terminal status code consumed into structured JSON.
  iree_status_code_t code;
  // Length of |message| in bytes.
  iree_host_size_t message_length;
  // Formatted status message, truncated when necessary.
  char message[512];
} iree_tune_loom_status_json_error_t;

static void iree_tune_loom_consume_status_json_error(
    iree_status_t status, iree_tune_loom_status_json_error_t* out_error) {
  out_error->code = iree_status_code(status);
  memset(out_error->message, 0, sizeof(out_error->message));
  iree_host_size_t required_length = 0;
  if (iree_status_format(status, sizeof(out_error->message), out_error->message,
                         &required_length)) {
    out_error->message_length = required_length;
    if (out_error->message_length >= sizeof(out_error->message)) {
      out_error->message_length = sizeof(out_error->message) - 1;
    }
  } else {
    const char* error_code_string = iree_status_code_string(out_error->code);
    iree_host_size_t index = 0;
    for (; error_code_string[index] != '\0' &&
           index + 1 < sizeof(out_error->message);
         ++index) {
      out_error->message[index] = error_code_string[index];
    }
    out_error->message[index] = '\0';
    out_error->message_length = index;
  }
  iree_status_free(status);
}

static iree_status_t iree_tune_loom_write_profile_artifact_identity_json(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_run_id_field_json(run, stream));
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_candidate_identity_json(candidate, stream));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_specialization_field_json(
      benchmark_result->specialization, stream));
  if (benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_shape_point_fields_json(
        module, case_plan, benchmark_result->sample_ordinal, stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"benchmark\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(stream, benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"case\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, case_plan->name));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_benchmark_metadata_json(
      module, benchmark_plan, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"batch_size\":%" PRIhsz,
      benchmark_result->hal_benchmark.timing.batch_size));
  const loom_run_hal_profile_summary_t* profile =
      &benchmark_result->hal_benchmark.profile;
  if (profile->has_artifact_path) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"artifact_path\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, iree_make_string_view(profile->artifact_path,
                                      profile->artifact_path_length)));
  }
  return iree_ok_status();
}

typedef struct iree_tune_loom_profile_counter_decode_summary_t {
  // Total non-empty JSONL rows decoded from the raw profile bundle.
  iree_host_size_t total_count;
  // Number of counter_summary rows decoded from the raw profile bundle.
  iree_host_size_t summary_count;
  // Number of counter_set metadata rows decoded from the raw profile bundle.
  iree_host_size_t counter_set_count;
  // Number of counter metadata rows decoded from the raw profile bundle.
  iree_host_size_t counter_count;
  // Number of aggregate counter_group rows decoded from the raw profile bundle.
  iree_host_size_t group_count;
  // Number of raw counter_sample rows decoded from the raw profile bundle.
  iree_host_size_t sample_count;
  // Number of decoded rows whose type is not known by this tuner build.
  iree_host_size_t unknown_count;
} iree_tune_loom_profile_counter_decode_summary_t;

typedef struct iree_tune_loom_profile_summary_decode_summary_t {
  // Total non-empty JSONL rows decoded from the raw profile bundle.
  iree_host_size_t total_count;
  // Number of bundle-level summary rows decoded from the raw profile bundle.
  iree_host_size_t summary_count;
  // Number of per-device summary rows decoded from the raw profile bundle.
  iree_host_size_t device_summary_count;
  // Number of decoded rows whose type is not known by this tuner build.
  iree_host_size_t unknown_count;
} iree_tune_loom_profile_summary_decode_summary_t;

static iree_status_t iree_tune_loom_summarize_profile_summary_line(
    iree_string_view_t line,
    iree_tune_loom_profile_summary_decode_summary_t* summary) {
  ++summary->total_count;

  char type_storage[64] = {0};
  iree_host_size_t type_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_string(
      line, IREE_SV("type"), IREE_SV("unknown"),
      iree_make_mutable_string_view(type_storage, sizeof(type_storage)),
      &type_length));
  const iree_string_view_t type =
      iree_make_string_view(type_storage, type_length);
  if (iree_string_view_equal(type, IREE_SV("summary"))) {
    ++summary->summary_count;
  } else if (iree_string_view_equal(type, IREE_SV("device_summary"))) {
    ++summary->device_summary_count;
  } else {
    ++summary->unknown_count;
  }
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_summarize_profile_summary_rows(
    iree_string_view_t summary_rows,
    iree_tune_loom_profile_summary_decode_summary_t* summary) {
  memset(summary, 0, sizeof(*summary));
  iree_string_view_t remaining = summary_rows;
  iree_status_t status = iree_ok_status();
  while (!iree_string_view_is_empty(remaining) && iree_status_is_ok(status)) {
    iree_string_view_t line = iree_string_view_empty();
    iree_string_view_split(remaining, '\n', &line, &remaining);
    line = iree_string_view_trim(line);
    if (iree_string_view_is_empty(line)) {
      continue;
    }
    status = iree_tune_loom_summarize_profile_summary_line(line, summary);
  }
  return status;
}

static iree_status_t iree_tune_loom_summarize_profile_counter_line(
    iree_string_view_t line,
    iree_tune_loom_profile_counter_decode_summary_t* summary) {
  ++summary->total_count;

  char type_storage[64] = {0};
  iree_host_size_t type_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_string(
      line, IREE_SV("type"), IREE_SV("unknown"),
      iree_make_mutable_string_view(type_storage, sizeof(type_storage)),
      &type_length));
  const iree_string_view_t type =
      iree_make_string_view(type_storage, type_length);
  if (iree_string_view_equal(type, IREE_SV("counter_summary"))) {
    ++summary->summary_count;
  } else if (iree_string_view_equal(type, IREE_SV("counter_set"))) {
    ++summary->counter_set_count;
  } else if (iree_string_view_equal(type, IREE_SV("counter"))) {
    ++summary->counter_count;
  } else if (iree_string_view_equal(type, IREE_SV("counter_group"))) {
    ++summary->group_count;
  } else if (iree_string_view_equal(type, IREE_SV("counter_sample"))) {
    ++summary->sample_count;
  } else {
    ++summary->unknown_count;
  }
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_summarize_profile_counter_rows(
    iree_string_view_t counter_rows,
    iree_tune_loom_profile_counter_decode_summary_t* summary) {
  memset(summary, 0, sizeof(*summary));
  iree_string_view_t remaining = counter_rows;
  iree_status_t status = iree_ok_status();
  while (!iree_string_view_is_empty(remaining) && iree_status_is_ok(status)) {
    iree_string_view_t line = iree_string_view_empty();
    iree_string_view_split(remaining, '\n', &line, &remaining);
    line = iree_string_view_trim(line);
    if (iree_string_view_is_empty(line)) {
      continue;
    }
    status = iree_tune_loom_summarize_profile_counter_line(line, summary);
  }
  return status;
}

static iree_status_t iree_tune_loom_write_profile_counter_request_json(
    const iree_tune_loom_benchmark_policy_t* policy,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_object_field_name(
      stream, &first_field, "data_family_names"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_profile_family_names_json(
      policy->hal_options.profile_data_families, stream));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
      stream, &first_field, "counter_set_count",
      policy->hal_options.profile_counter_set_count));
  if (policy->hal_options.profile_counter_set_count != 0) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_object_field_name(
        stream, &first_field, "counter_sets"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
    for (iree_host_size_t i = 0;
         i < policy->hal_options.profile_counter_set_count; ++i) {
      const iree_hal_profile_counter_set_selection_t* counter_set =
          &policy->hal_options.profile_counter_sets[i];
      if (i != 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
      bool first_counter_set_field = true;
      IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
          stream, &first_counter_set_field, "name", counter_set->name));
      IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
          stream, &first_counter_set_field, "counter_count",
          counter_set->counter_name_count));
      if (counter_set->counter_name_count != 0) {
        IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_object_field_name(
            stream, &first_counter_set_field, "counter_names"));
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
        for (iree_host_size_t j = 0; j < counter_set->counter_name_count; ++j) {
          if (j != 0) {
            IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
          }
          IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
              stream, counter_set->counter_names[j]));
        }
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
      }
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_tune_loom_write_profile_summary_request_json(
    const iree_tune_loom_benchmark_policy_t* policy,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_object_field_name(
      stream, &first_field, "data_family_names"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_profile_family_names_json(
      policy->hal_options.profile_data_families, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_tune_loom_write_profile_summary_decode_summary_json(
    const iree_tune_loom_profile_summary_decode_summary_t* summary,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
      stream, &first_field, "total", summary->total_count));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
      stream, &first_field, "summary", summary->summary_count));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
      stream, &first_field, "device_summary", summary->device_summary_count));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
      stream, &first_field, "unknown", summary->unknown_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_tune_loom_write_profile_summary_status_json(
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_profile_summary_decode_summary_t* decode_summary,
    iree_string_view_t status, iree_string_view_t reason,
    iree_status_code_t error_code, iree_string_view_t error_message,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
      stream, &first_field, "type", IREE_SV("profile_summary_status")));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
      stream, &first_field, "status", status));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
      stream, &first_field, "reason", reason));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_object_field_name(
      stream, &first_field, "request"));
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_profile_summary_request_json(policy, stream));
  if (decode_summary != NULL) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_object_field_name(
        stream, &first_field, "decoded_rows"));
    IREE_RETURN_IF_ERROR(
        iree_tune_loom_write_profile_summary_decode_summary_json(decode_summary,
                                                                 stream));
  }
  if (error_code != IREE_STATUS_OK ||
      !iree_string_view_is_empty(error_message)) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_object_field_name(
        stream, &first_field, "error"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
    bool first_error_field = true;
    if (error_code != IREE_STATUS_OK) {
      IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u32_field(
          stream, &first_error_field, "code", (uint32_t)error_code));
      IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
          stream, &first_error_field, "status",
          iree_make_cstring_view(iree_status_code_string(error_code))));
    }
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_optional_string_field(
        stream, &first_error_field, "message", error_message));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_tune_loom_write_profile_counter_decode_summary_json(
    const iree_tune_loom_profile_counter_decode_summary_t* summary,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
      stream, &first_field, "total", summary->total_count));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
      stream, &first_field, "summary", summary->summary_count));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
      stream, &first_field, "counter_sets", summary->counter_set_count));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
      stream, &first_field, "counters", summary->counter_count));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
      stream, &first_field, "groups", summary->group_count));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
      stream, &first_field, "samples", summary->sample_count));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_size_field(
      stream, &first_field, "unknown", summary->unknown_count));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_tune_loom_write_profile_counter_status_json(
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_profile_counter_decode_summary_t* decode_summary,
    iree_string_view_t status, iree_string_view_t reason,
    iree_status_code_t error_code, iree_string_view_t error_message,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
      stream, &first_field, "type", IREE_SV("counter_decode_status")));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
      stream, &first_field, "status", status));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
      stream, &first_field, "reason", reason));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_object_field_name(
      stream, &first_field, "request"));
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_profile_counter_request_json(policy, stream));
  if (decode_summary != NULL) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_object_field_name(
        stream, &first_field, "decoded_rows"));
    IREE_RETURN_IF_ERROR(
        iree_tune_loom_write_profile_counter_decode_summary_json(decode_summary,
                                                                 stream));
  }
  if (error_code != IREE_STATUS_OK ||
      !iree_string_view_is_empty(error_message)) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_object_field_name(
        stream, &first_field, "error"));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
    bool first_error_field = true;
    if (error_code != IREE_STATUS_OK) {
      IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u32_field(
          stream, &first_error_field, "code", (uint32_t)error_code));
      IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
          stream, &first_error_field, "status",
          iree_make_cstring_view(iree_status_code_string(error_code))));
    }
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_optional_string_field(
        stream, &first_error_field, "message", error_message));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_tune_loom_append_profile_summary_status_row(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    const iree_tune_loom_profile_summary_decode_summary_t* decode_summary,
    iree_string_view_t status, iree_string_view_t reason,
    iree_status_code_t error_code, iree_string_view_t error_message,
    iree_string_builder_t* profile_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(profile_output, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "{\"row\":\"profile_summary\""));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_profile_artifact_identity_json(
      run, candidate, module, benchmark_plan, case_plan, benchmark_result,
      &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"profile_summary\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_profile_summary_status_json(
      policy, decode_summary, status, reason, error_code, error_message,
      &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_append_profile_summary_status_from_error(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_string_view_t status, iree_string_view_t reason,
    iree_status_t error_status, iree_string_builder_t* profile_output) {
  iree_tune_loom_status_json_error_t error = {0};
  iree_tune_loom_consume_status_json_error(error_status, &error);
  return iree_tune_loom_append_profile_summary_status_row(
      run, candidate, module, benchmark_plan, case_plan, policy,
      benchmark_result, /*decode_summary=*/NULL, status, reason, error.code,
      iree_make_string_view(error.message, error.message_length),
      profile_output);
}

static iree_status_t iree_tune_loom_append_profile_summary_payload_row(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_string_view_t summary_json, iree_string_builder_t* profile_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(profile_output, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "{\"row\":\"profile_summary\""));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_profile_artifact_identity_json(
      run, candidate, module, benchmark_plan, case_plan, benchmark_result,
      &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"profile_summary\":"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, summary_json));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_append_profile_summary_rows(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_allocator_t allocator, iree_string_builder_t* profile_output) {
  const loom_run_hal_profile_summary_t* profile =
      &benchmark_result->hal_benchmark.profile;
  if (!benchmark_result->has_hal_benchmark || !profile->requested) {
    return iree_ok_status();
  }
  const bool summary_requested =
      profile->has_artifact_path ||
      iree_tune_loom_profile_data_needs_artifact_data(profile->data_families);
  if (!summary_requested) {
    return iree_ok_status();
  }
  if (!profile->executed) {
    if (profile->has_error) {
      return iree_tune_loom_append_profile_summary_status_row(
          run, candidate, module, benchmark_plan, case_plan, policy,
          benchmark_result, /*decode_summary=*/NULL, IREE_SV("unavailable"),
          IREE_SV("profile_error"), profile->error_code,
          iree_make_string_view(profile->error_message,
                                profile->error_message_length),
          profile_output);
    }
    return iree_tune_loom_append_profile_summary_status_row(
        run, candidate, module, benchmark_plan, case_plan, policy,
        benchmark_result, /*decode_summary=*/NULL, IREE_SV("unavailable"),
        IREE_SV("profile_not_executed"), IREE_STATUS_OK,
        iree_string_view_empty(), profile_output);
  }
  if (profile->has_error) {
    return iree_tune_loom_append_profile_summary_status_row(
        run, candidate, module, benchmark_plan, case_plan, policy,
        benchmark_result, /*decode_summary=*/NULL, IREE_SV("unavailable"),
        IREE_SV("profile_error"), profile->error_code,
        iree_make_string_view(profile->error_message,
                              profile->error_message_length),
        profile_output);
  }
  if (!profile->has_artifact_path) {
    return iree_tune_loom_append_profile_summary_status_row(
        run, candidate, module, benchmark_plan, case_plan, policy,
        benchmark_result, /*decode_summary=*/NULL, IREE_SV("unavailable"),
        IREE_SV("no_profile_artifact_path"), IREE_STATUS_OK,
        iree_string_view_empty(), profile_output);
  }

  FILE* summary_report_file = tmpfile();
  if (summary_report_file == NULL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to create temporary profile summary file: "
                            "%s",
                            strerror(errno));
  }

  const iree_string_view_t artifact_path = iree_make_string_view(
      profile->artifact_path, profile->artifact_path_length);
  iree_status_t decode_status = iree_profile_summary_file(
      artifact_path, IREE_SV("jsonl"), summary_report_file, allocator);
  iree_string_builder_t summary_rows;
  iree_string_builder_initialize(allocator, &summary_rows);
  if (!iree_status_is_ok(decode_status)) {
    fclose(summary_report_file);
    iree_string_builder_deinitialize(&summary_rows);
    return iree_tune_loom_append_profile_summary_status_from_error(
        run, candidate, module, benchmark_plan, case_plan, policy,
        benchmark_result, IREE_SV("raw_artifact_only"),
        IREE_SV("decode_failed"), decode_status, profile_output);
  }
  iree_status_t status =
      iree_tune_loom_read_file_into_builder(summary_report_file, &summary_rows);
  fclose(summary_report_file);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&summary_rows);
    return status;
  }

  const iree_string_view_t summary_rows_view =
      iree_string_builder_view(&summary_rows);
  iree_tune_loom_profile_summary_decode_summary_t decode_summary;
  status = iree_tune_loom_summarize_profile_summary_rows(summary_rows_view,
                                                         &decode_summary);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&summary_rows);
    return iree_tune_loom_append_profile_summary_status_from_error(
        run, candidate, module, benchmark_plan, case_plan, policy,
        benchmark_result, IREE_SV("raw_artifact_only"),
        IREE_SV("decode_summary_failed"), status, profile_output);
  }
  const iree_string_view_t reason =
      decode_summary.summary_count == 0 &&
              decode_summary.device_summary_count == 0
          ? IREE_SV("decoded_empty_profile_summary")
          : IREE_SV("decoded_profile_summary");
  status = iree_tune_loom_append_profile_summary_status_row(
      run, candidate, module, benchmark_plan, case_plan, policy,
      benchmark_result, &decode_summary, IREE_SV("decoded"), reason,
      IREE_STATUS_OK, iree_string_view_empty(), profile_output);

  iree_string_view_t remaining = iree_string_builder_view(&summary_rows);
  while (!iree_string_view_is_empty(remaining) && iree_status_is_ok(status)) {
    iree_string_view_t line = iree_string_view_empty();
    iree_string_view_split(remaining, '\n', &line, &remaining);
    line = iree_string_view_trim(line);
    if (iree_string_view_is_empty(line)) {
      continue;
    }
    status = iree_tune_loom_append_profile_summary_payload_row(
        run, candidate, module, benchmark_plan, case_plan, benchmark_result,
        line, profile_output);
  }
  iree_string_builder_deinitialize(&summary_rows);
  return status;
}

static iree_status_t iree_tune_loom_append_profile_counter_status_row(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    const iree_tune_loom_profile_counter_decode_summary_t* decode_summary,
    iree_string_view_t status, iree_string_view_t reason,
    iree_status_code_t error_code, iree_string_view_t error_message,
    iree_string_builder_t* profile_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(profile_output, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "{\"row\":\"profile_counter\""));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_profile_artifact_identity_json(
      run, candidate, module, benchmark_plan, case_plan, benchmark_result,
      &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"counter\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_profile_counter_status_json(
      policy, decode_summary, status, reason, error_code, error_message,
      &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_append_profile_counter_status_from_error(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_string_view_t status, iree_string_view_t reason,
    iree_status_t error_status, iree_string_builder_t* profile_output) {
  iree_tune_loom_status_json_error_t error = {0};
  iree_tune_loom_consume_status_json_error(error_status, &error);
  return iree_tune_loom_append_profile_counter_status_row(
      run, candidate, module, benchmark_plan, case_plan, policy,
      benchmark_result, /*decode_summary=*/NULL, status, reason, error.code,
      iree_make_string_view(error.message, error.message_length),
      profile_output);
}

static iree_status_t iree_tune_loom_append_profile_counter_payload_row(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_string_view_t counter_json, iree_string_builder_t* profile_output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(profile_output, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "{\"row\":\"profile_counter\""));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_profile_artifact_identity_json(
      run, candidate, module, benchmark_plan, case_plan, benchmark_result,
      &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"counter\":"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, counter_json));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_append_profile_counter_rows(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_allocator_t allocator, iree_string_builder_t* profile_output) {
  const loom_run_hal_profile_summary_t* profile =
      &benchmark_result->hal_benchmark.profile;
  if (!benchmark_result->has_hal_benchmark || !profile->requested ||
      !iree_tune_loom_profile_data_has_counter_data(profile->data_families)) {
    return iree_ok_status();
  }
  if (!profile->executed) {
    if (profile->has_error) {
      return iree_tune_loom_append_profile_counter_status_row(
          run, candidate, module, benchmark_plan, case_plan, policy,
          benchmark_result, /*decode_summary=*/NULL, IREE_SV("unavailable"),
          IREE_SV("profile_error"), profile->error_code,
          iree_make_string_view(profile->error_message,
                                profile->error_message_length),
          profile_output);
    }
    return iree_tune_loom_append_profile_counter_status_row(
        run, candidate, module, benchmark_plan, case_plan, policy,
        benchmark_result, /*decode_summary=*/NULL, IREE_SV("unavailable"),
        IREE_SV("profile_not_executed"), IREE_STATUS_OK,
        iree_string_view_empty(), profile_output);
  }
  if (profile->has_error) {
    return iree_tune_loom_append_profile_counter_status_row(
        run, candidate, module, benchmark_plan, case_plan, policy,
        benchmark_result, /*decode_summary=*/NULL, IREE_SV("unavailable"),
        IREE_SV("profile_error"), profile->error_code,
        iree_make_string_view(profile->error_message,
                              profile->error_message_length),
        profile_output);
  }
  if (!profile->has_artifact_path) {
    return iree_tune_loom_append_profile_counter_status_row(
        run, candidate, module, benchmark_plan, case_plan, policy,
        benchmark_result, /*decode_summary=*/NULL, IREE_SV("unavailable"),
        IREE_SV("no_profile_artifact_path"), IREE_STATUS_OK,
        iree_string_view_empty(), profile_output);
  }

  FILE* counter_report_file = tmpfile();
  if (counter_report_file == NULL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "failed to create temporary counter report file: "
                            "%s",
                            strerror(errno));
  }

  const iree_string_view_t artifact_path = iree_make_string_view(
      profile->artifact_path, profile->artifact_path_length);
  iree_status_t decode_status = iree_profile_counter_file(
      artifact_path, IREE_SV("jsonl"), iree_string_view_empty(),
      /*id_filter=*/-1,
      /*emit_samples=*/false, counter_report_file, allocator);
  iree_string_builder_t counter_rows;
  iree_string_builder_initialize(allocator, &counter_rows);
  if (!iree_status_is_ok(decode_status)) {
    fclose(counter_report_file);
    iree_string_builder_deinitialize(&counter_rows);
    return iree_tune_loom_append_profile_counter_status_from_error(
        run, candidate, module, benchmark_plan, case_plan, policy,
        benchmark_result, IREE_SV("raw_artifact_only"),
        IREE_SV("decode_failed"), decode_status, profile_output);
  }
  iree_status_t status =
      iree_tune_loom_read_file_into_builder(counter_report_file, &counter_rows);
  fclose(counter_report_file);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&counter_rows);
    return status;
  }

  const iree_string_view_t counter_rows_view =
      iree_string_builder_view(&counter_rows);
  iree_tune_loom_profile_counter_decode_summary_t decode_summary;
  status = iree_tune_loom_summarize_profile_counter_rows(counter_rows_view,
                                                         &decode_summary);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&counter_rows);
    return iree_tune_loom_append_profile_counter_status_from_error(
        run, candidate, module, benchmark_plan, case_plan, policy,
        benchmark_result, IREE_SV("raw_artifact_only"),
        IREE_SV("decode_summary_failed"), status, profile_output);
  }
  const iree_string_view_t reason =
      decode_summary.counter_count == 0 && decode_summary.group_count == 0 &&
              decode_summary.sample_count == 0
          ? IREE_SV("decoded_empty_counter_report")
          : IREE_SV("decoded_counter_report");
  status = iree_tune_loom_append_profile_counter_status_row(
      run, candidate, module, benchmark_plan, case_plan, policy,
      benchmark_result, &decode_summary, IREE_SV("decoded"), reason,
      IREE_STATUS_OK, iree_string_view_empty(), profile_output);

  iree_string_view_t remaining = iree_string_builder_view(&counter_rows);
  while (!iree_string_view_is_empty(remaining) && iree_status_is_ok(status)) {
    iree_string_view_t line = iree_string_view_empty();
    iree_string_view_split(remaining, '\n', &line, &remaining);
    line = iree_string_view_trim(line);
    if (iree_string_view_is_empty(line)) {
      continue;
    }
    status = iree_tune_loom_append_profile_counter_payload_row(
        run, candidate, module, benchmark_plan, case_plan, benchmark_result,
        line, profile_output);
  }
  iree_string_builder_deinitialize(&counter_rows);
  return status;
}

static iree_status_t iree_tune_loom_append_profile_row(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_allocator_t allocator, iree_string_builder_t* profile_output) {
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
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_specialization_field_json(
      benchmark_result->specialization, &stream));
  if (benchmark_result->has_sample_ordinal) {
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_shape_point_fields_json(
        module, case_plan, benchmark_result->sample_ordinal, &stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"benchmark\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, benchmark_plan->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"case\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, case_plan->name));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_benchmark_metadata_json(
      module, benchmark_plan, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"batch_size\":%" PRIhsz,
      benchmark_result->hal_benchmark.timing.batch_size));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"profile\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_hal_profile_summary_json(
      &benchmark_result->hal_benchmark.profile, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_append_profile_summary_rows(
      run, candidate, module, benchmark_plan, case_plan, policy,
      benchmark_result, allocator, profile_output));
  return iree_tune_loom_append_profile_counter_rows(
      run, candidate, module, benchmark_plan, case_plan, policy,
      benchmark_result, allocator, profile_output);
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

  iree_status_t status = iree_tune_loom_select_hal_actual_invocation(
      case_plan, &actual_invocation);
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_hal_context_ensure_runtime(hal_context);
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
        hal_context->runtime.device;
    benchmark_execution_options.materializer.device_allocator =
        iree_hal_device_allocator(hal_context->runtime.device);
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
  if (iree_status_is_ok(status) && hal_provider.compile_rejected) {
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
  if (iree_status_is_ok(status) && !hal_provider.compile_rejected &&
      has_specialization_sample_ordinal) {
    begin_sample = specialization_sample_ordinal;
    end_sample = specialization_sample_ordinal + 1;
  } else if (iree_status_is_ok(status) && !hal_provider.compile_rejected) {
    status = iree_tune_loom_validate_sample_flag(case_plan->sample_count,
                                                 &begin_sample, &end_sample);
  }

  iree_host_size_t benchmark_correctness_sample_count = 0;
  iree_host_size_t benchmark_correctness_failed_sample_count = 0;
  if (iree_status_is_ok(status) && !hal_provider.compile_rejected) {
    status = iree_tune_loom_run_case_correctness_range(
        run, candidate, module_plan, benchmark_plan, benchmark_plan->case_index,
        &benchmark_execution_options, specialization, begin_sample, end_sample,
        execution_arena, jsonl_sink, &benchmark_correctness_sample_count,
        &benchmark_correctness_failed_sample_count);
  }
  if (iree_status_is_ok(status) && !hal_provider.compile_rejected) {
    *inout_correctness_sample_count += benchmark_correctness_sample_count;
    *inout_correctness_failed_sample_count +=
        benchmark_correctness_failed_sample_count;
  }

  if (iree_status_is_ok(status) && !hal_provider.compile_rejected &&
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
       iree_status_is_ok(status) && !hal_provider.compile_rejected &&
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
      candidate->provider.test_module, selection->benchmark_plan, &stream));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_specialization_field_json(
      candidate->specialization, &stream));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_shape_point_fields_json(
      candidate->provider.test_module, selection->case_plan,
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

  iree_status_t status = iree_tune_loom_select_hal_actual_invocation(
      selection->case_plan, &actual_invocation);
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_hal_context_ensure_runtime(hal_context);
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
        hal_context->runtime.device;
    candidate_execution_options.materializer.device_allocator =
        iree_hal_device_allocator(hal_context->runtime.device);
    candidate_execution_options.materializer.buffer_params =
        iree_tune_loom_host_visible_buffer_params();
    candidate_execution_options.invocation.invoke_actual =
        (loom_testbench_invocation_callback_t){
            .fn = iree_tune_loom_hal_actual_invoke,
            .user_data = &candidate->provider,
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
  if (iree_status_is_ok(status) && candidate->provider.compile_rejected) {
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

static iree_status_t iree_tune_loom_append_summary_row(
    const iree_tune_loom_run_identity_t* run,
    iree_host_size_t planned_case_count,
    iree_host_size_t planned_benchmark_count,
    iree_host_size_t selected_benchmark_count, iree_host_size_t failure_count,
    iree_host_size_t failed_benchmark_count,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count, bool dry_run,
    iree_tune_loom_shape_specialization_mode_t shape_specialization_mode,
    iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, "{\"row\":\"summary\""));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_run_id_field_json(run, &stream));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"dry_run\":%s", dry_run ? "true" : "false"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"shape_specialization\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_tune_loom_shape_specialization_mode_name(
                   shape_specialization_mode)));
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
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"summary\":{"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "\"planned\":{\"case_count\":%" PRIhsz ",\"benchmark_count\":%" PRIhsz
      ",\"selected_benchmark_count\":%" PRIhsz "}",
      planned_case_count, planned_benchmark_count, selected_benchmark_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"failures\":{\"row_count\":%" PRIhsz
      ",\"failed_benchmark_count\":%" PRIhsz "}",
      failure_count, failed_benchmark_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"correctness\":{\"sample_count\":%" PRIhsz
      ",\"failed_sample_count\":%" PRIhsz "}}",
      correctness_sample_count, correctness_failed_sample_count));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}\n"));
  return iree_ok_status();
}

static const char* const iree_tune_loom_manifest_environment_variables[] = {
    "TMPDIR",
    "TEMP",
    "TMP",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "ONEAPI_DEVICE_SELECTOR",
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
    "HSA_ENABLE_SDMA",
    "HSA_TOOLS_LIB",
    "IREE_TRACY_CAPTURE",
};

static iree_status_t iree_tune_loom_write_manifest_environment_json(
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"environment\":{"));
  bool first_field = true;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_tune_loom_manifest_environment_variables); ++i) {
    const char* name = iree_tune_loom_manifest_environment_variables[i];
    const char* value = getenv(name);
    if (value == NULL || value[0] == '\0') {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
        stream, &first_field, name, iree_make_cstring_view(value)));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_tune_loom_write_file_stat_error_json(
    int error_number, loom_output_stream_t* stream) {
  const iree_status_code_t code = iree_status_code_from_errno(error_number);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
      stream, &first_field, "status", IREE_SV("stat_failed")));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u32_field(
      stream, &first_field, "code", (uint32_t)code));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
      stream, &first_field, "status_string",
      iree_make_cstring_view(iree_status_code_string(code))));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
      stream, &first_field, "message",
      iree_make_cstring_view(strerror(error_number))));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_tune_loom_write_file_identity_json(
    iree_string_view_t path, iree_allocator_t allocator,
    loom_output_stream_t* stream) {
  if (iree_string_view_equal(path, IREE_SV("<stdin>"))) {
    return loom_output_stream_write_cstring(stream, "{\"status\":\"stdin\"}");
  }
  if (loom_tooling_file_path_is_stdio(path)) {
    return loom_output_stream_write_cstring(stream, "{\"status\":\"stdio\"}");
  }

  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_dup_string_view(path, allocator, &storage));
#if defined(IREE_PLATFORM_WINDOWS)
  struct _stat64 file_stat = {0};
  const int stat_result = _stat64(storage, &file_stat);
  const bool is_regular_file =
      stat_result == 0 && (file_stat.st_mode & _S_IFMT) == _S_IFREG;
  const int64_t modified_time_seconds = (int64_t)file_stat.st_mtime;
  const int32_t modified_time_nanoseconds = 0;
#else
  struct stat file_stat = {0};
  const int stat_result = stat(storage, &file_stat);
  const bool is_regular_file = stat_result == 0 && S_ISREG(file_stat.st_mode);
#if defined(IREE_PLATFORM_APPLE)
  const int64_t modified_time_seconds = (int64_t)file_stat.st_mtimespec.tv_sec;
  const int32_t modified_time_nanoseconds =
      (int32_t)file_stat.st_mtimespec.tv_nsec;
#else
  const int64_t modified_time_seconds = (int64_t)file_stat.st_mtim.tv_sec;
  const int32_t modified_time_nanoseconds = (int32_t)file_stat.st_mtim.tv_nsec;
#endif  // defined(IREE_PLATFORM_APPLE)
#endif  // defined(IREE_PLATFORM_WINDOWS)
  const int stat_error_number = errno;
  iree_allocator_free(allocator, storage);

  if (stat_result != 0) {
    return iree_tune_loom_write_file_stat_error_json(stat_error_number, stream);
  }

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
      stream, &first_field, "status",
      is_regular_file ? IREE_SV("ok") : IREE_SV("not_regular")));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_u64_field(
      stream, &first_field, "byte_count", (uint64_t)file_stat.st_size));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_object_field_name(
      stream, &first_field, "modified_time"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"unix_seconds\":%" PRIi64 ",\"nanoseconds\":%" PRIi32 "}",
      modified_time_seconds, modified_time_nanoseconds));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_tune_loom_write_manifest_file_reference_json(
    iree_string_view_t path, iree_allocator_t allocator,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  bool first_field = true;
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_string_field(
      stream, &first_field, "path", path));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_json_object_field_name(
      stream, &first_field, "identity"));
  IREE_RETURN_IF_ERROR(
      iree_tune_loom_write_file_identity_json(path, allocator, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_tune_loom_write_manifest_file_array_json(
    const iree_tune_loom_artifact_bundle_t* bundle,
    iree_tune_loom_bundle_file_kind_t kind, iree_allocator_t allocator,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  bool first_entry = true;
  for (iree_host_size_t i = 0; i < bundle->file_entry_count; ++i) {
    const iree_tune_loom_bundle_file_entry_t* entry = &bundle->file_entries[i];
    if (entry->kind != kind) {
      continue;
    }
    if (!first_entry) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    first_entry = false;
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_manifest_file_reference_json(
        iree_make_cstring_view(entry->path), allocator, stream));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t iree_tune_loom_write_manifest_files_json(
    const iree_tune_loom_artifact_bundle_t* bundle,
    const iree_tune_loom_run_identity_t* run, iree_allocator_t allocator,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"files\":{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "\"results\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_manifest_file_reference_json(
      run->results_path, allocator, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"fixture_reads\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_manifest_file_array_json(
      bundle, IREE_TUNE_LOOM_BUNDLE_FILE_FIXTURE_READ, allocator, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"file_outputs\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_manifest_file_array_json(
      bundle, IREE_TUNE_LOOM_BUNDLE_FILE_OUTPUT, allocator, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"profiles\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_manifest_file_array_json(
      bundle, IREE_TUNE_LOOM_BUNDLE_FILE_PROFILE, allocator, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"compile_reports\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_manifest_file_array_json(
      bundle, IREE_TUNE_LOOM_BUNDLE_FILE_COMPILE_REPORT, allocator, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"target_artifacts\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_manifest_file_array_json(
      bundle, IREE_TUNE_LOOM_BUNDLE_FILE_TARGET_ARTIFACT, allocator, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"target_listings\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_manifest_file_array_json(
      bundle, IREE_TUNE_LOOM_BUNDLE_FILE_TARGET_LISTING, allocator, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"hal_executables\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_manifest_file_array_json(
      bundle, IREE_TUNE_LOOM_BUNDLE_FILE_HAL_EXECUTABLE, allocator, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t iree_tune_loom_append_artifact_bundle_manifest_json(
    const iree_tune_loom_artifact_bundle_t* bundle,
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_hal_context_t* hal_context,
    iree_string_view_t source_text, iree_string_view_t command_line_json,
    bool dry_run,
    iree_tune_loom_shape_specialization_mode_t shape_specialization_mode,
    iree_allocator_t allocator, iree_string_builder_t* manifest) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(manifest, &stream);

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "\"tool\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_cstring(&stream, "iree-tune-loom"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"run_id\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, run->run_id));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"source\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, run->source));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"source_identity\":{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "\"byte_count\":"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(&stream, "%" PRIhsz, source_text.size));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\"file\":"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_manifest_file_reference_json(
      run->source, allocator, &stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"policy\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_tune_loom_artifact_bundle_policy_name(bundle->policy)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"dry_run\":"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, dry_run ? "true" : "false"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"shape_specialization\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, iree_tune_loom_shape_specialization_mode_name(
                   shape_specialization_mode)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"paths\":{"));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "\"bundle\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(&stream, bundle->dir));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"results\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, run->results_path));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"manifest\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, bundle->manifest_path));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"file_outputs\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, run->file_output_dir));
  if (!iree_string_view_is_empty(run->profile_artifacts_dir)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"profiles\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, run->profile_artifacts_dir));
  }
  if (iree_tune_loom_artifact_bundle_wants_compile_reports(bundle)) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"compile_reports\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, bundle->compile_report_dir));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"target_artifacts\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, bundle->target_artifact_dir));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"target_listings\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, bundle->target_listing_dir));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"hal_executables\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, bundle->hal_executable_dir));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_manifest_files_json(
      bundle, run, allocator, &stream));
  if (hal_context->backend != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"device\":{"));
    IREE_RETURN_IF_ERROR(iree_tune_loom_write_hal_context_identity_fields_json(
        hal_context, &stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}"));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"command_line\":"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(&stream, command_line_json));
  IREE_RETURN_IF_ERROR(iree_tune_loom_write_manifest_environment_json(&stream));
  char cwd[4096] = {0};
#if defined(IREE_PLATFORM_WINDOWS)
  char* cwd_result = _getcwd(cwd, sizeof(cwd));
#else
  char* cwd_result = getcwd(cwd, sizeof(cwd));
#endif  // defined(IREE_PLATFORM_WINDOWS)
  if (cwd_result != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"cwd\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(&stream, cwd));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "}\n"));
  return iree_ok_status();
}

static iree_status_t iree_tune_loom_write_artifact_bundle_manifest(
    const iree_tune_loom_artifact_bundle_t* bundle,
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_hal_context_t* hal_context,
    iree_string_view_t source_text, iree_string_view_t command_line_json,
    bool dry_run,
    iree_tune_loom_shape_specialization_mode_t shape_specialization_mode,
    iree_allocator_t allocator) {
  if (!bundle->enabled) {
    return iree_ok_status();
  }

  iree_string_builder_t manifest;
  iree_string_builder_initialize(allocator, &manifest);
  iree_status_t status = iree_tune_loom_append_artifact_bundle_manifest_json(
      bundle, run, hal_context, source_text, command_line_json, dry_run,
      shape_specialization_mode, allocator, &manifest);
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        bundle->manifest_path, iree_string_builder_view(&manifest), allocator);
  }
  iree_string_builder_deinitialize(&manifest);
  return status;
}

static iree_status_t iree_tune_loom_append_command_line_json(
    int argc, char** argv, iree_string_builder_t* output) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "["));
  for (int i = 0; i < argc; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(&stream, argv[i]));
  }
  return loom_output_stream_write_cstring(&stream, "]");
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
      "  iree-tune-loom file.loom --device=amdgpu "
      "--benchmark=@kernel_latency --profile_final_batch=true "
      "--profile_artifacts_dir=.notes/profiles --output=results.jsonl\n"
      "  iree-tune-loom file.loom --device=amdgpu "
      "--benchmark=@kernel_latency --profile_data=counter-ranges "
      "--profile_counter=SQ_WAVES --file_output_dir=/tmp/loom-run "
      "--output=results.jsonl\n"
      "  iree-tune-loom file.loom --device=amdgpu "
      "--benchmark=@kernel_latency --artifact_bundle_dir=/tmp/loom-run\n"
      "  iree-tune-loom file.loom --device=amdgpu "
      "--benchmark=@kernel_latency "
      "--shape_specialization=dynamic|per_sample|both\n"
      "  iree-tune-loom file.loom --device=amdgpu "
      "--compare=@baseline,@variant --interleave=ABABA "
      "--repetitions=2 --profile_final_batch=false\n"
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
      "  benchmark.repetition one interleaved comparison timing window\n"
      "  comparison aggregate baseline/candidate timing ratio\n"
      "  profile   optional final profiled-batch evidence outside timing\n"
      "  profile_summary decoded raw profile bundle summary/device rows\n"
      "  profile_counter decoded counter status/summary/set/group rows\n"
      "  failure   parse/verify/planning failure diagnostics\n"
      "  summary   final totals row\n"
      "\n"
      "Rows are written and flushed as evidence becomes available, so an agent "
      "can `tail -f` an --output file during long sweeps. Every row carries "
      "run_id. Selected benchmark rows also carry "
      "candidate_id/candidate_index so agents can join plan, compile, sample, "
      "benchmark, and profile evidence across large JSONL sweeps. "
      "When --artifact_bundle_dir is set and --output is empty, results are "
      "written to results.jsonl in the bundle and manifest.json records "
      "command line, source identity, output, profile, file-output paths, "
      "path/size/mtime metadata for observed fixture/output/profile files, "
      "selected environment variables, and HAL device identity when a dispatch "
      "benchmark selected a backend. With --artifact_bundle_policy=debug|full, "
      "per-candidate compile report sidecars are written under "
      "compile_reports/, target-native artifacts under target_artifacts/, "
      "target-owned assembly/listing text under target_listings/, and HAL "
      "executable packages under hal_executables/; compile and benchmark rows "
      "link them from compile_report_path, target_artifact_path, "
      "target_listing_path, and hal_executable_path. Source, fixture, and "
      "artifact files are identified by path/size/mtime; this CLI does not "
      "content-hash large files or act as a CAS. "
      "`--shape_specialization=dynamic` compiles once, "
      "`--shape_specialization=per_sample` compiles each selected shape with "
      "concrete parameter facts, and `--shape_specialization=both` emits both "
      "views for comparing generic-vs-specialized kernels. "
      "Parameterized cases also carry shape_id/shape_index and a "
      "shape.parameters map on sample, benchmark, and profile rows so "
      "multi-size runs can be joined without filename conventions. Compile "
      "diagnostics are data: a compile row with status!='ok' is a candidate "
      "rejection, not a tool infrastructure failure. Profile rows are emitted "
      "only when --profile_final_batch=true or a benchmark attr requests it; "
      "their profile.rows array contains HAL statistics rows such as "
      "dispatch_export, dispatch_command_buffer, and "
      "dispatch_command_operation when the backend provides them. "
      "Dispatch benchmarks materialize device-buffer bindings from check ops "
      "into an input ring before recording command buffers; dispatch slot i "
      "uses a different physical binding set while the ring has capacity. "
      "The default auto ring targets at least 32MiB of bindings and at least "
      "one binding set per dispatch in the batch. Set --input_ring_count=1 "
      "for deliberate hot-reuse measurements. Benchmark rows include "
      "benchmark_result.data_cache with the effective ring shape and bytes. "
      "`--compare=@baseline,@variant` runs selected dispatch_complete "
      "benchmarks as prepared candidates in an interleaved schedule. ABABA "
      "requires exactly two benchmarks and runs A followed by --repetitions "
      "BA pairs; round_robin supports ABCD-style rotation. Comparison mode "
      "requires one concrete shape per candidate, so use --sample= when a "
      "case has multiple shape samples; selected candidates must have the "
      "same concrete parameter layout and values. Measured comparison windows "
      "suppress final profiling and report that as "
      "profile_suppressed_for_interleave so profiling evidence does not "
      "perturb interleaved timing. "
      "check.file.read.npy paths resolve relative to the input .loom file. "
      "check.file.write.npy paths must be relative and are rooted under "
      "--file_output_dir, which defaults under $TMPDIR/iree-loom-tune. "
      "Compile rows and benchmark payloads with reports also carry "
      "static_summary, a compact projection of static compile evidence such "
      "as instruction count, code bytes, descriptor instruction mix, spill "
      "counts, memory, pressure, source-low counts, and move-cause totals. "
      "--profile_data selects final-batch HAL profiling families and "
      "--profile_artifacts_dir writes raw .irpf bundles for heavyweight "
      "families such as counters, device metrics, and executable traces. Raw "
      "bundles emit profile_summary JSONL rows with bundle/device summaries, "
      "executable trace byte counts, device metric counts, and a "
      "profile_summary_status row. Counter families also emit "
      "profile_counter JSONL rows decoded from the raw profile bundle; each "
      "decode emits a counter_decode_status row with requested families/sets "
      "and decoded row counts, and decode failures or unavailable counter "
      "evidence use the same status row shape instead of failing the timing "
      "run. Without an explicit --profile_artifacts_dir, heavyweight bundles "
      "are staged under the run file_output_dir.\n"
      "Benchmark attrs named family, phase, strategy, knobs, problem, or "
      "reference_id are copied into a metadata object on candidate rows. "
      "The final summary row keeps the existing flat count fields and also "
      "carries a nested summary object grouping planned, failure, and "
      "correctness totals for compact agent consumption.\n"
      "\n"
      "jq recipes:\n"
      "  jq 'select(.row==\"run\" or .row==\"summary\")' results.jsonl\n"
      "  jq 'select(.row==\"summary\")' results.jsonl\n"
      "  jq 'select(.row==\"failure\" or .row==\"compile\") | "
      "select(.status!=\"ok\" or .row==\"failure\")' results.jsonl\n"
      "  jq 'select(.row==\"compile\") | .diagnostics[]?' results.jsonl\n"
      "  jq 'select(.row==\"benchmark\") | .benchmark_result | "
      "{benchmark,status,p50:.dispatch_timing_ns.p50}' results.jsonl\n"
      "  jq 'select(.row==\"benchmark\") | .benchmark_result | "
      "{benchmark,p50:.dispatch_timing_ns.p50,data_cache}' results.jsonl\n"
      "  jq 'select(.row==\"benchmark\" and .shape) | "
      "{candidate_id,shape_id,shape:.shape.parameters,"
      "p50:.benchmark_result.dispatch_timing_ns.p50}' results.jsonl\n"
      "  jq 'select(.row==\"benchmark.repetition\") | "
      "{candidate_id,order_index,token:.schedule_token,"
      "p50:.benchmark_result.dispatch_timing_ns.p50}' results.jsonl\n"
      "  jq 'select(.row==\"comparison\") | "
      "{candidate_id,baseline_candidate_id,ratio_p50,speedup_p50,"
      "ratio_p90,speedup_p90,candidate_p50_spread_ppm}' "
      "results.jsonl\n"
      "  jq 'select(.row==\"plan\" and .metadata) | "
      "{candidate_id,metadata}' results.jsonl\n"
      "  jq 'select(.row==\"compile\" and .static_summary) | "
      "{candidate_id,code:.static_summary.code_byte_count,"
      "spills:.static_summary.allocation_spill_count,"
      "valu:.static_summary.vector_alu_count,"
      "local:.static_summary.local_memory_bytes}' results.jsonl\n"
      "  jq 'select(.row==\"compile\" and .compile_report_path) | "
      "{candidate_id,path:.compile_report_path}' results.jsonl\n"
      "  jq 'select(.row==\"compile\" and .target_artifact_path) | "
      "{candidate_id,target:.target_artifact_path,"
      "listing:.target_listing_path,hal:.hal_executable_path}' "
      "results.jsonl\n"
      "  jq 'select(.row==\"profile\") as $r | $r.profile.rows[]? | "
      "select(.type|startswith(\"dispatch_\")) | "
      "{candidate_id:$r.candidate_id,type,export_name,"
      "timing:.timing.mean_ns}' "
      "results.jsonl\n"
      "  jq 'select(.row==\"profile_counter\") | .counter | "
      "select(.type==\"counter_group\") | {key,counter,avg,sum}' "
      "results.jsonl\n"
      "  jq 'select(.row==\"profile_counter\" and "
      ".counter.type==\"counter_decode_status\") | "
      "{candidate_id,status:.counter.status,reason:.counter.reason,"
      "decoded:.counter.decoded_rows,error:.counter.error.status}' "
      "results.jsonl\n"
      "  jq 'select(.row==\"profile_summary\" and "
      ".profile_summary.type==\"summary\") | "
      "{candidate_id,traces:.profile_summary.executable_trace_records,"
      "trace_bytes:.profile_summary.executable_trace_data_bytes,"
      "metric_values:.profile_summary.device_metric_values}' "
      "results.jsonl\n"
      "  jq 'select(.row==\"profile_summary\" and "
      ".profile_summary.type==\"profile_summary_status\") | "
      "{candidate_id,status:.profile_summary.status,"
      "reason:.profile_summary.reason,"
      "decoded:.profile_summary.decoded_rows}' results.jsonl\n"
      "  tail -f results.jsonl | jq -c 'select(.row==\"compile\" or "
      ".row==\"benchmark\" or .row==\"comparison\" or .row==\"summary\")'\n");
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
    status =
        iree_tune_loom_artifact_bundle_initialize(allocator, &artifact_bundle);
    hal_context.artifact_bundle = &artifact_bundle;
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
      iree_tune_loom_effective_results_output_path(&artifact_bundle);
  const iree_string_view_t profile_artifacts_dir =
      iree_tune_loom_effective_profile_artifacts_dir(&artifact_bundle);
  if (iree_status_is_ok(status)) {
    status = iree_tune_loom_file_provider_initialize(
        filename, run_id, artifact_bundle.file_output_dir, &artifact_bundle,
        allocator, &file_provider);
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
            &run_identity, planned_case_count, planned_benchmark_count,
            selected_benchmark_count, failure_count, failed_benchmark_count,
            correctness_sample_count, correctness_failed_sample_count,
            FLAG_dry_run, shape_specialization_mode,
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
