// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-line option parsing and mode names for loom-benchmark.

#ifndef LOOM_TOOLS_LOOM_BENCHMARK_OPTIONS_H_
#define LOOM_TOOLS_LOOM_BENCHMARK_OPTIONS_H_

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/sanitizer/options.h"
#include "loom/tooling/input/format.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_benchmark_artifact_bundle_policy_e {
  // Artifact bundling is disabled.
  LOOM_BENCHMARK_ARTIFACT_BUNDLE_POLICY_NONE = 0,
  // Bundle only results, manifest, file outputs, and required profile bundles.
  LOOM_BENCHMARK_ARTIFACT_BUNDLE_POLICY_MINIMAL = 1,
  // Bundle debug sidecars such as compile reports and target artifacts.
  LOOM_BENCHMARK_ARTIFACT_BUNDLE_POLICY_DEBUG = 2,
  // Bundle every currently supported artifact class.
  LOOM_BENCHMARK_ARTIFACT_BUNDLE_POLICY_FULL = 3,
} loom_benchmark_artifact_bundle_policy_t;

typedef enum loom_benchmark_interleave_mode_e {
  // No interleaved comparison was requested.
  LOOM_BENCHMARK_INTERLEAVE_NONE = 0,
  // Run A once, then repeated B/A pairs for each non-baseline candidate.
  LOOM_BENCHMARK_INTERLEAVE_ABABA = 1,
  // Rotate all comparison candidates in one repeated round-robin order.
  LOOM_BENCHMARK_INTERLEAVE_ROUND_ROBIN = 2,
} loom_benchmark_interleave_mode_t;

typedef enum loom_benchmark_output_format_e {
  // Emit one compact JSON document after the benchmark run completes.
  LOOM_BENCHMARK_OUTPUT_FORMAT_SNAPSHOT = 0,
  // Emit newline-delimited lifecycle event rows as the run proceeds.
  LOOM_BENCHMARK_OUTPUT_FORMAT_JSONL = 1,
} loom_benchmark_output_format_t;

enum {
  // Default reachable binding-ring byte target for reducing hot-cache effects.
  LOOM_BENCHMARK_DEFAULT_INPUT_RING_MIN_BYTES = 32 * 1024 * 1024,
};

typedef struct loom_benchmark_i32_flag_t {
  // Current flag value.
  int32_t value;
  // True when the flag was provided on the command line or in a flagfile.
  bool specified;
} loom_benchmark_i32_flag_t;

typedef struct loom_benchmark_bool_flag_t {
  // Current flag value.
  bool value;
  // True when the flag was provided on the command line or in a flagfile.
  bool specified;
} loom_benchmark_bool_flag_t;

typedef struct loom_benchmark_options_t {
  // Source format, provider options, and retained diagnostic path remapping.
  loom_input_options_t input;
  // Optional check.case or check.scenario symbol selected for execution.
  iree_string_view_t selected_case;
  // Optional check.benchmark symbol selected for execution.
  iree_string_view_t selected_benchmark;
  // Optional concrete benchmark sample ordinal, or negative to select all.
  int32_t sample_ordinal;
  // Maximum planned concrete samples retained for each check.case.
  iree_host_size_t max_samples_per_case;
  // Pass pipeline used before target artifact emission.
  iree_string_view_t pipeline;
  // Optional explicit `family:selector` compiler target.
  iree_string_view_t target;
  // Sanitizer checks inserted by the target pipeline.
  loom_sanitizer_options_t sanitizer;
  // Direct config bindings parsed from repeated --config=key=value flags.
  iree_string_view_list_t config_assignments;
  // Config object file paths parsed from repeated --config-file=path flags.
  iree_string_view_list_t config_files;
  // Explicit result output path, or empty to use stdout or bundle defaults.
  iree_string_view_t output;
  // Selected result output format.
  loom_benchmark_output_format_t output_format;
  // Explicit root for check.file.write.* output paths.
  iree_string_view_t file_output_root;
  // Explicit artifact bundle root directory, or empty when disabled.
  iree_string_view_t artifact_bundle_dir;
  // Artifact bundle policy controlling sidecar artifact classes.
  loom_benchmark_artifact_bundle_policy_t artifact_bundle_policy;
  // True when the run stops after planning selected benchmarks.
  bool dry_run;
  // True when selected case samples are captured as canonical compile requests.
  bool compile_only;
  // Check each selected scenario trial before timing; false skips oracle work.
  bool scenario_correctness;
  // Requested measurement mode, including automatic subject-based selection.
  iree_string_view_t measure;
  // Structured compile-report request.
  iree_string_view_t compile_report;
  // Structured artifact-manifest request.
  iree_string_view_t artifact_manifest;
  // Requested HAL profiling data-family list.
  iree_string_view_t profile_data;
  // Requested HAL profiling counter names.
  iree_string_view_list_t profile_counters;
  // Explicit directory receiving raw HAL profile artifacts.
  iree_string_view_t profile_artifacts_dir;
  // Minimum total reachable byte size for generated dispatch input rings.
  int64_t input_ring_min_bytes;
  // Exact dispatch input ring count, or zero for byte-size based selection.
  iree_host_size_t input_ring_count;
  // Benchmark list requested for interleaved comparison.
  iree_string_view_t compare;
  // Interleaving policy used when |compare| selects multiple benchmarks.
  loom_benchmark_interleave_mode_t interleave_mode;
  // Number of interleaved comparison repetitions.
  iree_host_size_t repetitions;
  // Number of measured benchmark iterations or batches.
  iree_host_size_t iterations;
  // True when |iterations| was provided by the CLI/user.
  bool iterations_specified;
  // Number of warmup iterations or batches before measurement.
  iree_host_size_t warmup_iterations;
  // True when |warmup_iterations| was provided by the CLI/user.
  bool warmup_iterations_specified;
  // Number of dispatches recorded into each HAL benchmark batch.
  iree_host_size_t batch_size;
  // True when |batch_size| was provided by the CLI/user.
  bool batch_size_specified;
  // Minimum measured duration in milliseconds for dispatch benchmarks.
  int64_t min_time_ms;
  // True when |min_time_ms| was provided by the CLI/user.
  bool min_time_ms_specified;
  // Minimum warmup duration in milliseconds for dispatch benchmarks.
  int64_t warmup_time_ms;
  // True when |warmup_time_ms| was provided by the CLI/user.
  bool warmup_time_ms_specified;
  // Maximum measured HAL command-buffer batches.
  iree_host_size_t max_batches;
  // True when |max_batches| was provided by the CLI/user.
  bool max_batches_specified;
  // Accepted p90-to-p50 spread threshold in parts per million.
  uint64_t stable_p90_to_p50_ppm;
  // True when |stable_p90_to_p50_ppm| was provided by the CLI/user.
  bool stable_p90_to_p50_ppm_specified;
  // True when one final profiled batch should run after timing.
  bool profile_final_batch;
  // True when |profile_final_batch| was provided by the CLI/user.
  bool profile_final_batch_specified;
  // True when |profile_data| was explicitly provided and is not "none".
  bool profile_data_requested;
  // True when |input_ring_min_bytes| differs from the default.
  bool input_ring_min_bytes_specified;
  // True when |input_ring_count| was provided by the CLI/user.
  bool input_ring_count_specified;
} loom_benchmark_options_t;

// Initializes benchmark runner options with production defaults.
void loom_benchmark_options_initialize(loom_benchmark_options_t* out_options);

// Returns true when the parsed CLI requested the AGENTS.md help snippet.
bool loom_benchmark_cli_flags_request_agents_md(void);

// Converts parsed command-line flags into structured benchmark options.
iree_status_t loom_benchmark_options_from_flags(
    loom_benchmark_options_t* out_options);

// Returns the stable JSON spelling for an artifact bundle policy.
iree_string_view_t loom_benchmark_artifact_bundle_policy_name(
    loom_benchmark_artifact_bundle_policy_t policy);

// Returns the stable flag and JSON spelling for an output format.
iree_string_view_t loom_benchmark_output_format_name(
    loom_benchmark_output_format_t format);

// Parses an output format flag value.
iree_status_t loom_benchmark_parse_output_format(
    iree_string_view_t value, loom_benchmark_output_format_t* out_format);

// Parses an artifact bundle policy flag value.
iree_status_t loom_benchmark_parse_artifact_bundle_policy(
    iree_string_view_t value,
    loom_benchmark_artifact_bundle_policy_t* out_policy);

// Parses a comparison interleave mode flag value.
iree_status_t loom_benchmark_parse_interleave_mode(
    iree_string_view_t value, loom_benchmark_interleave_mode_t* out_mode);

// Returns the stable JSON spelling for an interleave mode.
iree_string_view_t loom_benchmark_interleave_mode_name(
    loom_benchmark_interleave_mode_t mode);

// Returns true when |profile_data_families| includes counter payload data.
bool loom_benchmark_profile_data_has_counter_data(
    iree_hal_device_profiling_data_families_t profile_data_families);

// Returns true when |profile_data_families| requires raw profile artifacts.
bool loom_benchmark_profile_data_needs_artifact_data(
    iree_hal_device_profiling_data_families_t profile_data_families);

// Parses a comma-separated HAL profiling data family list.
iree_status_t loom_benchmark_parse_profile_data_families(
    iree_string_view_t value,
    iree_hal_device_profiling_data_families_t* out_profile_data_families);

// Writes selected profiling data-family JSON names as a JSON array.
iree_status_t loom_benchmark_write_profile_family_names_json(
    iree_hal_device_profiling_data_families_t profile_data_families,
    loom_output_stream_t* stream);

// Parses an int32 flag and records whether it was explicitly specified.
iree_status_t loom_benchmark_parse_i32_flag(iree_string_view_t flag_name,
                                            void* storage,
                                            iree_string_view_t value);

// Prints an int32 flag in IREE flagfile form.
void loom_benchmark_print_i32_flag(iree_string_view_t flag_name, void* storage,
                                   FILE* file);

// Parses a bool flag and records whether it was explicitly specified.
iree_status_t loom_benchmark_parse_bool_flag(iree_string_view_t flag_name,
                                             void* storage,
                                             iree_string_view_t value);

// Prints a bool flag in IREE flagfile form.
void loom_benchmark_print_bool_flag(iree_string_view_t flag_name, void* storage,
                                    FILE* file);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_BENCHMARK_OPTIONS_H_
