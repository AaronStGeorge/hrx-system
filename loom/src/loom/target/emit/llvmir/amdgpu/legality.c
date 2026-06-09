// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/amdgpu/legality.h"

#include "loom/ops/llvmir/ops.h"

static bool loom_llvmir_amdgpu_legality_intrinsic_is_workitem_id(
    iree_string_view_t kind) {
  return iree_string_view_equal(kind, IREE_SV("llvm.amdgcn.workitem.id.x")) ||
         iree_string_view_equal(kind, IREE_SV("llvm.amdgcn.workitem.id.y")) ||
         iree_string_view_equal(kind, IREE_SV("llvm.amdgcn.workitem.id.z"));
}

static bool loom_llvmir_amdgpu_legality_try_verify_op(
    const loom_llvmir_target_legality_provider_t* provider,
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  if (!loom_llvmir_intrinsic_isa(op)) return true;

  loom_string_id_t kind_id = loom_llvmir_intrinsic_kind(op);
  iree_string_view_t kind = iree_string_view_empty();
  if (!loom_llvmir_target_legality_string_attr(
          context, provider, op, IREE_SV("kind"), kind_id, &kind)) {
    return false;
  }
  if (!loom_llvmir_amdgpu_legality_intrinsic_is_workitem_id(kind)) {
    return true;
  }
  *out_handled = true;

  iree_string_view_t constraint_key = IREE_SV("amdgpu-workitem-id-signature");
  if (!loom_llvmir_target_legality_expect_intrinsic_shape(
          context, provider, op, 0, 1, constraint_key)) {
    return false;
  }
  return loom_llvmir_target_legality_expect_scalar_result(
      context, provider, op, LOOM_SCALAR_TYPE_I32, constraint_key);
}

static const loom_llvmir_target_legality_provider_t kAmdgpuLegalityProvider = {
    .name = IREE_SVL("amdgpu"),
    .try_verify_op = loom_llvmir_amdgpu_legality_try_verify_op,
};

const loom_llvmir_target_legality_provider_t*
loom_llvmir_amdgpu_legality_provider(void) {
  return &kAmdgpuLegalityProvider;
}
