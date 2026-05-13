// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal_benchmark.h"

#include "iree/hal/utils/statistics_sink.h"

void loom_run_hal_benchmark_options_initialize(
    loom_run_hal_benchmark_options_t* out_options) {
  *out_options = (loom_run_hal_benchmark_options_t){0};
  loom_run_benchmark_options_initialize(&out_options->timing);
  loom_run_hal_dispatch_batch_options_initialize(&out_options->dispatch_batch);
  out_options->dispatch_batch.dispatch_count = out_options->timing.batch_size;
  out_options->profile_flags =
      IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
  out_options->profile_data_families = IREE_HAL_DEVICE_PROFILING_DATA_NONE;
  out_options->profile_capture_filter =
      iree_hal_profile_capture_filter_default();
}

void loom_run_hal_benchmark_result_initialize(
    loom_run_hal_benchmark_result_t* out_result) {
  *out_result = (loom_run_hal_benchmark_result_t){0};
}

static iree_status_t loom_run_hal_benchmark_options_validate(
    const loom_run_hal_benchmark_options_t* options) {
  const loom_run_hal_benchmark_flags_t known_flags =
      LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH;
  if (iree_any_bit_set(options->flags, ~known_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown HAL benchmark flags 0x%08X",
                            options->flags & ~known_flags);
  }
  if (options->timing.batch_size != options->dispatch_batch.dispatch_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL benchmark timing batch_size %" PRIhsz
                            " must match dispatch_count %" PRIhsz,
                            options->timing.batch_size,
                            options->dispatch_batch.dispatch_count);
  }
  return iree_ok_status();
}

typedef struct loom_run_hal_benchmark_batch_context_t {
  // HAL runtime that owns the device used for dispatch.
  const loom_run_hal_runtime_t* runtime;
  // Prepared reusable dispatch batch.
  loom_run_hal_dispatch_batch_t* batch;
} loom_run_hal_benchmark_batch_context_t;

static iree_status_t loom_run_hal_benchmark_execute_batch(void* user_data) {
  loom_run_hal_benchmark_batch_context_t* context =
      (loom_run_hal_benchmark_batch_context_t*)user_data;
  return loom_run_hal_dispatch_batch_execute(context->runtime, context->batch);
}

static iree_status_t loom_run_hal_benchmark_run_profiled_batch(
    const loom_run_hal_runtime_t* runtime, loom_run_hal_dispatch_batch_t* batch,
    const loom_run_hal_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_hal_profile_summary_t* out_profile) {
  *out_profile = (loom_run_hal_profile_summary_t){
      .requested = true,
      .flags = options->profile_flags,
      .data_families = options->profile_data_families,
  };

  iree_hal_profile_statistics_sink_t* statistics_sink = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_profile_statistics_sink_create(allocator, &statistics_sink));

  iree_hal_device_profiling_options_t profiling_options = {
      .flags = options->profile_flags,
      .data_families = options->profile_data_families,
      .sink = iree_hal_profile_statistics_sink_base(statistics_sink),
      .capture_filter = options->profile_capture_filter,
      .counter_set_count = options->profile_counter_set_count,
      .counter_sets = options->profile_counter_sets,
  };

  iree_status_t status =
      iree_hal_device_profiling_begin(runtime->device, &profiling_options);
  const bool profiling_began = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_dispatch_batch_execute(runtime, batch);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_profiling_flush(runtime->device);
  }
  if (profiling_began) {
    status = iree_status_join(status,
                              iree_hal_device_profiling_end(runtime->device));
  }
  if (iree_status_is_ok(status)) {
    out_profile->executed = true;
    out_profile->row_count =
        iree_hal_profile_statistics_sink_row_count(statistics_sink);
    out_profile->dropped_record_count =
        iree_hal_profile_statistics_sink_dropped_record_count(statistics_sink);
  }

  iree_hal_profile_statistics_sink_release(statistics_sink);
  return status;
}

iree_status_t loom_run_hal_benchmark_dispatch_plan(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan,
    const loom_run_hal_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_hal_benchmark_result_t* out_result) {
  loom_run_hal_benchmark_result_initialize(out_result);
  IREE_RETURN_IF_ERROR(loom_run_hal_benchmark_options_validate(options));

  loom_run_hal_dispatch_batch_t batch = {0};
  iree_status_t status = loom_run_hal_dispatch_batch_prepare(
      runtime, candidate, plan, &options->dispatch_batch, allocator, &batch);

  loom_run_hal_benchmark_batch_context_t context = {
      .runtime = runtime,
      .batch = &batch,
  };
  if (iree_status_is_ok(status)) {
    status = loom_run_benchmark_run_batches(
        (loom_run_benchmark_batch_callback_t){
            .fn = loom_run_hal_benchmark_execute_batch,
            .user_data = &context,
        },
        &options->timing, allocator, &out_result->timing);
  }
  if (iree_status_is_ok(status) &&
      iree_all_bits_set(options->flags,
                        LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH)) {
    status = loom_run_hal_benchmark_run_profiled_batch(
        runtime, &batch, options, allocator, &out_result->profile);
  }

  loom_run_hal_dispatch_batch_deinitialize(&batch);
  if (!iree_status_is_ok(status)) {
    loom_run_hal_benchmark_result_initialize(out_result);
  }
  return status;
}
