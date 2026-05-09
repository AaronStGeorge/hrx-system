// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering for kernel synchronization source operations.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SYNC_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SYNC_H_

#include "loom/codegen/low/lower.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects a workgroup barrier packet for a source kernel.barrier op.
iree_status_t loom_amdgpu_select_kernel_barrier_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan);

// Lowers a source kernel.barrier to an AMDGPU workgroup barrier packet.
iree_status_t loom_amdgpu_lower_kernel_barrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op);

// Verifies source kernel.barrier legality for AMDGPU target-low selection.
iree_status_t loom_amdgpu_low_legality_verify_kernel_barrier(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SYNC_H_
