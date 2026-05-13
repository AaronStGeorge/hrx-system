// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Batched HAL dispatch benchmarking for Loom execution sessions.

#ifndef LOOM_TOOLING_EXECUTION_HAL_BENCHMARK_H_
#define LOOM_TOOLING_EXECUTION_HAL_BENCHMARK_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/tooling/execution/benchmark.h"
#include "loom/tooling/execution/hal_invocation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t loom_run_hal_benchmark_flags_t;
enum loom_run_hal_benchmark_flag_bits_e {
  // Performs only the normal timing benchmark.
  LOOM_RUN_HAL_BENCHMARK_FLAG_NONE = 0u,
  // Runs one final profiled batch after the measured timing window.
  LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH = 1u << 0,
};

typedef struct loom_run_hal_profile_summary_t {
  // True when final-batch profiling was requested.
  bool requested;
  // True when the final profiled batch completed and emitted through the sink.
  bool executed;
  // Producer profiling flags requested for the final batch.
  iree_hal_device_profiling_flags_t flags;
  // Structured profiling data families requested for the final batch.
  iree_hal_device_profiling_data_families_t data_families;
  // Aggregate profile rows received by the statistics sink.
  iree_host_size_t row_count;
  // Source records reported as dropped by the profile producer.
  uint64_t dropped_record_count;
} loom_run_hal_profile_summary_t;

typedef struct loom_run_hal_benchmark_options_t {
  // Flags controlling optional HAL benchmark phases.
  loom_run_hal_benchmark_flags_t flags;
  // Generic warmup, measured timing, batch, and stability policy.
  loom_run_benchmark_options_t timing;
  // Command-buffer recording and queue execute options for the batch.
  loom_run_hal_dispatch_batch_options_t dispatch_batch;
  // Producer profiling flags used for a requested final profiled batch.
  iree_hal_device_profiling_flags_t profile_flags;
  // Structured profile families requested for a final profiled batch.
  iree_hal_device_profiling_data_families_t profile_data_families;
  // Optional capture filter used for expensive profile artifacts.
  iree_hal_profile_capture_filter_t profile_capture_filter;
  // Number of hardware/software counter selections in |profile_counter_sets|.
  iree_host_size_t profile_counter_set_count;
  // Borrowed counter selections used during profiling_begin.
  const iree_hal_profile_counter_set_selection_t* profile_counter_sets;
} loom_run_hal_benchmark_options_t;

typedef struct loom_run_hal_benchmark_result_t {
  // Generic host timing benchmark result.
  loom_run_benchmark_result_t timing;
  // Final profiled-batch summary.
  loom_run_hal_profile_summary_t profile;
} loom_run_hal_benchmark_result_t;

// Initializes HAL benchmark options for rigorous batched dispatch timing.
void loom_run_hal_benchmark_options_initialize(
    loom_run_hal_benchmark_options_t* out_options);

// Initializes an empty HAL benchmark result.
void loom_run_hal_benchmark_result_initialize(
    loom_run_hal_benchmark_result_t* out_result);

// Prepares and times a reusable HAL command buffer containing a dispatch batch.
iree_status_t loom_run_hal_benchmark_dispatch_plan(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan,
    const loom_run_hal_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_hal_benchmark_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_BENCHMARK_H_
