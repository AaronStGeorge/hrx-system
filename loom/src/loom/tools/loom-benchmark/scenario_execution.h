// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepared check.scenario benchmark execution.

#ifndef LOOM_TOOLS_LOOM_BENCHMARK_SCENARIO_EXECUTION_H_
#define LOOM_TOOLS_LOOM_BENCHMARK_SCENARIO_EXECUTION_H_

#include "iree/base/api.h"
#include "loom/tools/loom-benchmark/work_execution.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_benchmark_scenario_execution_t
    loom_benchmark_scenario_execution_t;

// Creates lazily prepared scenario benchmark execution state.
//
// When |entropy_domain| is non-empty, independently named scenarios derive
// generated values from that shared domain. Ordinary execution passes an empty
// domain and retains scenario-name isolation.
iree_status_t loom_benchmark_scenario_execution_allocate(
    const loom_benchmark_work_plan_execution_options_t* options,
    iree_string_view_t entropy_domain,
    loom_benchmark_scenario_execution_t** out_execution);

// Releases all prepared products and materialized values owned by |execution|.
void loom_benchmark_scenario_execution_free(
    loom_benchmark_scenario_execution_t* execution);

// Prepares a concrete trial's target-only input ring. When scenario correctness
// is enabled, first checks the trial and reports its untimed observations. A
// failed expectation leaves |out_runnable| false without allocating a timing
// ring.
iree_status_t loom_benchmark_scenario_execution_prepare(
    loom_benchmark_scenario_execution_t* execution,
    const loom_benchmark_selected_benchmark_t* selection,
    iree_host_size_t work_item_index, iree_host_size_t benchmark_sample_ordinal,
    loom_testbench_scenario_sample_coordinate_t coordinate, bool* out_runnable);

// Verifies that two prepared trials have identical invocation layouts, values,
// and buffer alias topology.
iree_status_t loom_benchmark_scenario_execution_validate_equivalent(
    loom_benchmark_scenario_execution_t* baseline,
    loom_benchmark_scenario_execution_t* candidate);

// Measures one prepared trial with |policy| and returns complete HAL evidence.
iree_status_t loom_benchmark_scenario_execution_measure(
    loom_benchmark_scenario_execution_t* execution,
    const loom_benchmark_benchmark_policy_t* policy,
    loom_benchmark_benchmark_result_t* out_result);

// Prepares and measures one scenario trial and emits every logical result
// alias.
iree_status_t loom_benchmark_run_scenario_work_item(
    loom_benchmark_scenario_execution_t* execution,
    const loom_benchmark_work_item_t* work_item,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_BENCHMARK_SCENARIO_EXECUTION_H_
