// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Static vector register-window extraction on x86.

#ifndef LOOM_TARGET_ARCH_X86_LOWER_VECTOR_SLICE_H_
#define LOOM_TARGET_ARCH_X86_LOWER_VECTOR_SLICE_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Queries ordinary rank-one static slices implemented as register windows.
iree_status_t loom_x86_query_vector_slice_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result);

// Returns true when |plan| is an x86 static register-window slice.
bool loom_x86_vector_slice_plan_isa(loom_low_lower_plan_t plan);

// Selects and retains the complete register-window recipe.
iree_status_t loom_x86_select_vector_slice_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Marks the source vector consumed by a selected slice.
void loom_x86_mark_vector_slice_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan);

// Describes the selected register-window mechanism for compiler reports.
void loom_x86_describe_vector_slice_plan(
    loom_low_lower_plan_t plan, loom_low_lower_plan_report_t* out_report);

// Emits the retained register-window recipe.
iree_status_t loom_x86_emit_vector_slice_plan(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_LOWER_VECTOR_SLICE_H_
