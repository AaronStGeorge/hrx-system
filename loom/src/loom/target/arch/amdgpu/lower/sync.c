// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/context.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower/internal.h"

iree_status_t loom_amdgpu_select_kernel_barrier_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
  if (!loom_kernel_barrier_isa(source_op)) {
    return iree_ok_status();
  }

  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor_by_id(
          loom_low_lower_context_descriptor_set(context),
          LOOM_AMDGPU_DESCRIPTOR_ID_S_BARRIER);
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
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_BARRIER,
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

  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor_by_id(
          loom_target_low_legality_descriptor_set(context),
          LOOM_AMDGPU_DESCRIPTOR_ID_S_BARRIER);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("descriptor"), IREE_SV("s_barrier"),
        IREE_SV("selected descriptor set does not provide a workgroup barrier "
                "packet"));
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

  return loom_target_low_legality_reject(
      context, provider, op, IREE_SV("collective"),
      loom_op_name(loom_target_low_legality_module(context), op),
      IREE_SV("AMDGPU source-to-low needs target packet lowering for this "
              "collective form"));
}
