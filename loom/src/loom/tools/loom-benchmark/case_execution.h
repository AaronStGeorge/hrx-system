// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Correctness-gated check.case benchmark execution.

#ifndef LOOM_TOOLS_LOOM_BENCHMARK_CASE_EXECUTION_H_
#define LOOM_TOOLS_LOOM_BENCHMARK_CASE_EXECUTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tools/loom-benchmark/event.h"
#include "loom/tools/loom-benchmark/model.h"
#include "loom/tools/loom-benchmark/work_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Invoked immediately after one concrete case sample executes successfully.
typedef void(IREE_API_PTR* loom_benchmark_case_sample_observer_fn_t)(
    void* user_data, loom_testbench_case_sample_t sample);

typedef struct loom_benchmark_case_sample_observer_t {
  // Optional observer callback, or NULL when no sample evidence is required.
  loom_benchmark_case_sample_observer_fn_t fn;
  // Caller-owned context passed to |fn|.
  void* user_data;
} loom_benchmark_case_sample_observer_t;

// Emits result/profile-replay events for every logical benchmark alias of
// |work_item|.
iree_status_t loom_benchmark_emit_work_item_result_aliases(
    const loom_benchmark_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const loom_benchmark_work_plan_t* work_plan,
    const loom_benchmark_work_item_t* work_item,
    const loom_benchmark_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count,
    const loom_benchmark_event_sink_t* event_sink,
    iree_host_size_t* inout_failed_benchmark_count);

// Runs correctness samples for |work_item| and emits aliased sample events.
iree_status_t loom_benchmark_run_work_item_correctness_range(
    const loom_benchmark_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const loom_benchmark_work_plan_t* work_plan,
    const loom_benchmark_work_item_t* work_item,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* arena,
    loom_benchmark_case_sample_observer_t sample_observer,
    const loom_benchmark_event_sink_t* event_sink,
    iree_host_size_t* out_sample_count,
    iree_host_size_t* out_failed_sample_count);

// Measures one prepared case_end_to_end physical work item.
iree_status_t loom_benchmark_measure_case_end_to_end_work_item(
    const loom_testbench_module_plan_t* module_plan,
    const loom_benchmark_work_plan_t* work_plan,
    const loom_benchmark_work_item_t* work_item,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    loom_benchmark_benchmark_result_t* out_result);

// Runs one case_end_to_end physical work item and emits logical alias events.
iree_status_t loom_benchmark_run_case_end_to_end_work_item(
    const loom_benchmark_run_identity_t* run,
    const loom_testbench_module_plan_t* module_plan,
    const loom_benchmark_work_plan_t* work_plan,
    const loom_benchmark_work_item_t* work_item,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena, iree_allocator_t allocator,
    const loom_benchmark_event_sink_t* event_sink,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_BENCHMARK_CASE_EXECUTION_H_
