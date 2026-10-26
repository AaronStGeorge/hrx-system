// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native EXEC-mask algebra and masked value merging shared by AMDGPU control
// planners. These helpers operate only on already-selected Low carriers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_MASK_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_MASK_H_

#include "loom/codegen/low/lower/control_plan.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_AMDGPU_MASK_VALUE_DYNAMIC = 0,
  LOOM_AMDGPU_MASK_VALUE_FALSE = 1,
  LOOM_AMDGPU_MASK_VALUE_TRUE = 2,
};

typedef struct loom_amdgpu_mask_merge_constants_t {
  // False arm's exact logical value, or DYNAMIC for its incoming native mask.
  uint8_t false_value;
  // True arm's exact logical value, or DYNAMIC for its incoming native mask.
  uint8_t true_value;
} loom_amdgpu_mask_merge_constants_t;

bool loom_amdgpu_low_type_is_native_i1_mask(loom_low_lower_context_t* context,
                                            loom_type_t low_type);

iree_status_t loom_amdgpu_prepare_masked_merge(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_block_t* source_merge_dest,
    loom_low_lower_block_ref_t merge_low_dest, loom_value_slice_t false_args,
    loom_value_slice_t true_args,
    const loom_amdgpu_mask_merge_constants_t** out_constants);

iree_status_t loom_amdgpu_emit_zero_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t value_type, loom_value_id_t* out_value);

iree_status_t loom_amdgpu_emit_zero_merge_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t value_type, loom_value_id_t* out_value);

iree_status_t loom_amdgpu_emit_native_i1_mask_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_result);

iree_status_t loom_amdgpu_emit_masked_merge_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_amdgpu_mask_merge_constants_t constants,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_condition, loom_type_t value_type,
    loom_value_id_t* out_merged_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CONTROL_MASK_H_
