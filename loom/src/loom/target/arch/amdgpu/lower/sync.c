// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/context.h"
#include "loom/target/arch/amdgpu/lower/internal.h"
#include "loom/target/arch/amdgpu/target_refs.h"

static bool loom_amdgpu_kernel_barrier_maps_to_s_barrier(
    const loom_op_t* source_op) {
  return loom_kernel_barrier_memory_space(source_op) ==
             LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
         loom_kernel_barrier_ordering(source_op) ==
             LOOM_ATOMIC_ORDERING_ACQ_REL &&
         loom_kernel_barrier_scope(source_op) == LOOM_ATOMIC_SCOPE_WORKGROUP;
}

iree_status_t loom_amdgpu_select_kernel_barrier_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (!loom_kernel_barrier_isa(source_op)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_kernel_barrier_maps_to_s_barrier(source_op)) {
    return iree_ok_status();
  }

  const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
      loom_low_lower_context_descriptor_set(context),
      LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_ok_status();
  }

  *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_kernel_barrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_op_t* low_op = NULL;
  return loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER,
      /*operands=*/NULL, /*operand_count=*/0,
      loom_make_named_attr_slice(NULL, 0), /*result_types=*/NULL,
      /*result_count=*/0, &low_op);
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_barrier(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (!loom_amdgpu_kernel_barrier_maps_to_s_barrier(op)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("barrier.workgroup_scope"));
  }

  const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
      loom_target_low_legality_descriptor_set(context),
      LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("descriptor.s_barrier"));
  }

  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_kernel_collective(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  return loom_amdgpu_low_legality_reject(context, op,
                                         IREE_SV("collective.packet_lowering"));
}
