// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for arithmetic source operations that need target-owned
// planning beyond generated descriptor contracts.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_ARITHMETIC_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_ARITHMETIC_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects a mixed-source f32-result FMA plan for scalar.fmaf operands widened
// from f16 sources.
iree_status_t loom_amdgpu_select_scalar_fmaf_mix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fma_mix_plan_t* out_plan, bool* out_selected);

// Lowers scalar.fmaf to one mixed-source AMDGPU FMA/MAD descriptor packet.
iree_status_t loom_amdgpu_lower_scalar_fmaf_mix(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan);

// Marks the exact source values consumed by a selected mixed FMA plan.
void loom_amdgpu_mark_fma_mix_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fma_mix_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_ARITHMETIC_H_
