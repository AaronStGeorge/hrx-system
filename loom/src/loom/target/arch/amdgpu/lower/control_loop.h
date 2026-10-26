// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU natural-loop lane retirement. Planning retains per-exit masks and
// payloads; emission restores the parent EXEC mask after every lane retires.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_LOOP_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_LOOP_H_

#include "loom/codegen/low/lower/control_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Claims a divergent selection that surrounds a nested natural-loop exit.
iree_status_t loom_amdgpu_loop_try_prepare_exit_region(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_prepared);

// Accounts one natural-loop exit and prepares shared retirement when needed.
iree_status_t loom_amdgpu_loop_try_prepare_exit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool is_divergent, iree_arena_allocator_t* analysis_arena,
    bool* out_handled, bool* out_active_uniform);

// Returns true when |plan| belongs to natural-loop retirement.
bool loom_amdgpu_loop_plan_isa(loom_low_lower_plan_t plan);

// Emits a previously planned divergent loop exit or nested exit region.
iree_status_t loom_amdgpu_loop_emit_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan, loom_value_id_t low_condition,
    loom_block_t* low_true_dest, loom_block_t* low_false_dest,
    loom_type_t condition_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_LOOP_H_
