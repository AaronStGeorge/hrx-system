// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepared HAL execution for check.scenario command-program subjects.

#ifndef LOOM_TOOLING_EXECUTION_HAL_SCENARIO_COMMAND_PROGRAM_H_
#define LOOM_TOOLING_EXECUTION_HAL_SCENARIO_COMMAND_PROGRAM_H_

#include "loom/tooling/execution/hal/scenario_profile.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prepares one command-program subject against |profile|'s live HAL device.
iree_status_t loom_run_hal_testbench_command_scenario_product_prepare(
    loom_run_hal_testbench_scenario_profile_t* profile,
    const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* configuration,
    iree_allocator_t host_allocator,
    loom_testbench_prepared_product_t* out_product);

// Returns true when |prepared_product| is a command scenario product.
bool loom_run_hal_testbench_is_command_scenario_product(
    const loom_testbench_prepared_product_t* prepared_product);

// Benchmarks complete command-program calls over independent physical slots.
iree_status_t loom_run_hal_testbench_benchmark_command_scenario_product(
    const loom_testbench_prepared_product_t* prepared_product,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls,
    const loom_run_hal_benchmark_options_t* options,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_scenario_benchmark_result_t* out_result);

// Reports the authored input layout of one command-program call.
iree_status_t loom_run_hal_testbench_query_command_scenario_input_layout(
    const loom_testbench_prepared_product_t* prepared_product,
    const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_product_call_t* call,
    loom_run_hal_testbench_scenario_input_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_SCENARIO_COMMAND_PROGRAM_H_
