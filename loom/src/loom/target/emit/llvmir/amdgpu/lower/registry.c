// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowering for AMDGPU-native Loom LLVMIR intrinsic contracts.

#include "loom/ops/llvmir/ops.h"
#include "loom/target/emit/llvmir/amdgpu/intrinsics.h"
#include "loom/target/emit/llvmir/amdgpu/lower.h"
#include "loom/target/emit/llvmir/lower/internal.h"

static bool loom_llvmir_amdgpu_target_profile_is_amdgpu(
    const loom_llvmir_lowering_state_t* state) {
  return iree_string_view_equal(
      state->target_profile->target_env->target_triple,
      IREE_SV("amdgcn-amd-amdhsa"));
}

static iree_status_t loom_llvmir_amdgpu_declare_workitem_id(
    loom_llvmir_lowering_state_t* state, iree_string_view_t kind,
    loom_llvmir_function_t** out_function) {
  if (iree_string_view_equal(kind, IREE_SV("llvm.amdgcn.workitem.id.x"))) {
    static const uint8_t kWorkitemXIntrinsicKey = 0;
    return loom_llvmir_lowering_declare_provider_intrinsic_cached(
        state, &kWorkitemXIntrinsicKey,
        loom_llvmir_declare_amdgcn_workitem_id_x, out_function);
  }
  if (iree_string_view_equal(kind, IREE_SV("llvm.amdgcn.workitem.id.y"))) {
    static const uint8_t kWorkitemYIntrinsicKey = 0;
    return loom_llvmir_lowering_declare_provider_intrinsic_cached(
        state, &kWorkitemYIntrinsicKey,
        loom_llvmir_declare_amdgcn_workitem_id_y, out_function);
  }
  static const uint8_t kWorkitemZIntrinsicKey = 0;
  return loom_llvmir_lowering_declare_provider_intrinsic_cached(
      state, &kWorkitemZIntrinsicKey, loom_llvmir_declare_amdgcn_workitem_id_z,
      out_function);
}

static bool loom_llvmir_amdgpu_intrinsic_is_workitem_id(
    iree_string_view_t kind) {
  return iree_string_view_equal(kind, IREE_SV("llvm.amdgcn.workitem.id.x")) ||
         iree_string_view_equal(kind, IREE_SV("llvm.amdgcn.workitem.id.y")) ||
         iree_string_view_equal(kind, IREE_SV("llvm.amdgcn.workitem.id.z"));
}

static iree_status_t loom_llvmir_amdgpu_try_lower_intrinsic(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, bool* out_handled) {
  loom_string_id_t kind_id = loom_llvmir_intrinsic_kind(op);
  iree_string_view_t kind = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_string_attr(
      state, op, IREE_SV("kind"), kind_id, &kind));
  if (!loom_llvmir_amdgpu_intrinsic_is_workitem_id(kind)) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (!loom_llvmir_amdgpu_target_profile_is_amdgpu(state)) {
    return loom_llvmir_lowering_unsupported_op(
        state, op,
        "AMDGPU llvmir.intrinsic requires an AMDGPU target environment");
  }

  char detail_storage[64];
  int detail_length =
      iree_snprintf(detail_storage, sizeof(detail_storage),
                    "%.*s expects () -> i32", (int)kind.size, kind.data);
  if (detail_length < 0 ||
      (iree_host_size_t)detail_length >= sizeof(detail_storage)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU intrinsic diagnostic overflow");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_expect_intrinsic_shape(
      state, op, 0, 1, detail_storage));
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_expect_scalar_result(
      state, op, LOOM_SCALAR_TYPE_I32, detail_storage));

  loom_llvmir_function_t* function = NULL;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_amdgpu_declare_workitem_id(state, kind, &function));
  return loom_llvmir_lowering_lower_declared_call(state, target_block, op,
                                                  function);
}

static iree_status_t loom_llvmir_amdgpu_try_lower_op(
    const loom_llvmir_lowering_provider_t* provider,
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, bool* out_handled) {
  (void)provider;
  *out_handled = false;
  switch (op->kind) {
    case LOOM_OP_LLVMIR_INTRINSIC:
      return loom_llvmir_amdgpu_try_lower_intrinsic(state, target_block, op,
                                                    out_handled);
    default:
      return iree_ok_status();
  }
}

static const loom_llvmir_lowering_provider_t kAmdgpuLoweringProvider = {
    .name = IREE_SVL("amdgpu"),
    .try_lower_op = loom_llvmir_amdgpu_try_lower_op,
};

const loom_llvmir_lowering_provider_t* loom_llvmir_amdgpu_lowering_provider(
    void) {
  return &kAmdgpuLoweringProvider;
}
