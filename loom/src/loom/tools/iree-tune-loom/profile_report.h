// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Raw HAL profile artifact and decoded profile JSONL row emission.

#ifndef LOOM_TOOLS_IREE_TUNE_LOOM_PROFILE_REPORT_H_
#define LOOM_TOOLS_IREE_TUNE_LOOM_PROFILE_REPORT_H_

#include "iree/base/api.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/tools/iree-tune-loom/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Writes the profile counter request object for plan and status rows.
iree_status_t iree_tune_loom_write_profile_counter_request_json(
    const iree_tune_loom_benchmark_policy_t* policy,
    loom_output_stream_t* stream);

// Appends decoded final-batch profile rows for |benchmark_result|.
iree_status_t iree_tune_loom_append_profile_row(
    const iree_tune_loom_run_identity_t* run,
    const iree_tune_loom_candidate_identity_t* candidate,
    const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_tune_loom_benchmark_policy_t* policy,
    const iree_tune_loom_benchmark_result_t* benchmark_result,
    iree_allocator_t allocator, iree_string_builder_t* profile_output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_TUNE_LOOM_PROFILE_REPORT_H_
