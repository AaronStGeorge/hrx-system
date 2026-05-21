// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-line option parsing and mode names for iree-benchmark-loom.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_OPTIONS_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_OPTIONS_H_

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum iree_benchmark_loom_sample_compilation_mode_e {
  // Compile once and pass each sample's parameter values dynamically.
  IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_ONCE = 0,
  // Compile a separate candidate for each concrete selected sample.
  IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_PER_SAMPLE = 1,
  // Run both once and per-sample compilation modes.
  IREE_BENCHMARK_LOOM_SAMPLE_COMPILATION_BOTH = 2,
} iree_benchmark_loom_sample_compilation_mode_t;

typedef enum iree_benchmark_loom_artifact_bundle_policy_e {
  // Artifact bundling is disabled.
  IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_NONE = 0,
  // Bundle only results, manifest, file outputs, and required profile bundles.
  IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_MINIMAL = 1,
  // Bundle debug sidecars such as compile reports and target artifacts.
  IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_DEBUG = 2,
  // Bundle every currently supported artifact class.
  IREE_BENCHMARK_LOOM_ARTIFACT_BUNDLE_POLICY_FULL = 3,
} iree_benchmark_loom_artifact_bundle_policy_t;

typedef enum iree_benchmark_loom_interleave_mode_e {
  // No interleaved comparison was requested.
  IREE_BENCHMARK_LOOM_INTERLEAVE_NONE = 0,
  // Run A once, then repeated B/A pairs for each non-baseline candidate.
  IREE_BENCHMARK_LOOM_INTERLEAVE_ABABA = 1,
  // Rotate all comparison candidates in one repeated round-robin order.
  IREE_BENCHMARK_LOOM_INTERLEAVE_ROUND_ROBIN = 2,
} iree_benchmark_loom_interleave_mode_t;

typedef struct iree_benchmark_loom_i32_flag_t {
  // Current flag value.
  int32_t value;
  // True when the flag was provided on the command line or in a flagfile.
  bool specified;
} iree_benchmark_loom_i32_flag_t;

typedef struct iree_benchmark_loom_bool_flag_t {
  // Current flag value.
  bool value;
  // True when the flag was provided on the command line or in a flagfile.
  bool specified;
} iree_benchmark_loom_bool_flag_t;

// Returns the stable JSON spelling for an artifact bundle policy.
iree_string_view_t iree_benchmark_loom_artifact_bundle_policy_name(
    iree_benchmark_loom_artifact_bundle_policy_t policy);

// Parses an artifact bundle policy flag value.
iree_status_t iree_benchmark_loom_parse_artifact_bundle_policy(
    iree_string_view_t value,
    iree_benchmark_loom_artifact_bundle_policy_t* out_policy);

// Parses a sample-compilation mode flag value.
iree_status_t iree_benchmark_loom_parse_sample_compilation_mode(
    iree_string_view_t value,
    iree_benchmark_loom_sample_compilation_mode_t* out_mode);

// Returns the stable JSON spelling for a sample-compilation mode.
iree_string_view_t iree_benchmark_loom_sample_compilation_mode_name(
    iree_benchmark_loom_sample_compilation_mode_t mode);

// Returns true when |mode| includes the compile-once pass.
bool iree_benchmark_loom_sample_compilation_runs_once(
    iree_benchmark_loom_sample_compilation_mode_t mode);

// Returns true when |mode| includes the per-sample compilation pass.
bool iree_benchmark_loom_sample_compilation_runs_per_sample(
    iree_benchmark_loom_sample_compilation_mode_t mode);

// Parses a comparison interleave mode flag value.
iree_status_t iree_benchmark_loom_parse_interleave_mode(
    iree_string_view_t value, iree_benchmark_loom_interleave_mode_t* out_mode);

// Returns the stable JSON spelling for an interleave mode.
iree_string_view_t iree_benchmark_loom_interleave_mode_name(
    iree_benchmark_loom_interleave_mode_t mode);

// Returns true when |profile_data_families| includes counter payload data.
bool iree_benchmark_loom_profile_data_has_counter_data(
    iree_hal_device_profiling_data_families_t profile_data_families);

// Returns true when |profile_data_families| requires raw profile artifacts.
bool iree_benchmark_loom_profile_data_needs_artifact_data(
    iree_hal_device_profiling_data_families_t profile_data_families);

// Parses a comma-separated HAL profiling data family list.
iree_status_t iree_benchmark_loom_parse_profile_data_families(
    iree_string_view_t value,
    iree_hal_device_profiling_data_families_t* out_profile_data_families);

// Writes selected profiling data-family JSON names as a JSON array.
iree_status_t iree_benchmark_loom_write_profile_family_names_json(
    iree_hal_device_profiling_data_families_t profile_data_families,
    loom_output_stream_t* stream);

// Parses an int32 flag and records whether it was explicitly specified.
iree_status_t iree_benchmark_loom_parse_i32_flag(iree_string_view_t flag_name,
                                                 void* storage,
                                                 iree_string_view_t value);

// Prints an int32 flag in IREE flagfile form.
void iree_benchmark_loom_print_i32_flag(iree_string_view_t flag_name,
                                        void* storage, FILE* file);

// Parses a bool flag and records whether it was explicitly specified.
iree_status_t iree_benchmark_loom_parse_bool_flag(iree_string_view_t flag_name,
                                                  void* storage,
                                                  iree_string_view_t value);

// Prints a bool flag in IREE flagfile form.
void iree_benchmark_loom_print_bool_flag(iree_string_view_t flag_name,
                                         void* storage, FILE* file);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_OPTIONS_H_
