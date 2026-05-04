// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/x86/legality.h"

#include "loom/ops/llvmir/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/packed_dot_vector.h"

static bool loom_llvmir_x86_legality_intrinsic_is_x86(iree_string_view_t kind) {
  return iree_string_view_equal(kind, IREE_SV("llvm.x86.rdtsc")) ||
         iree_string_view_equal(kind, IREE_SV("llvm.x86.sse2.pause"));
}

static bool loom_llvmir_x86_legality_verify_intrinsic(
    const loom_llvmir_target_legality_provider_t* provider,
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  loom_string_id_t kind_id = loom_llvmir_intrinsic_kind(op);
  iree_string_view_t kind = iree_string_view_empty();
  if (!loom_llvmir_target_legality_string_attr(
          context, provider, op, IREE_SV("kind"), kind_id, &kind)) {
    return false;
  }
  if (!loom_llvmir_x86_legality_intrinsic_is_x86(kind)) {
    return true;
  }
  *out_handled = true;

  if (iree_string_view_equal(kind, IREE_SV("llvm.x86.rdtsc"))) {
    iree_string_view_t constraint_key = IREE_SV("llvm.x86.rdtsc.signature");
    if (!loom_llvmir_target_legality_expect_intrinsic_shape(
            context, provider, op, 0, 1, constraint_key)) {
      return false;
    }
    return loom_llvmir_target_legality_expect_scalar_result(
        context, provider, op, LOOM_SCALAR_TYPE_I64, constraint_key);
  }
  iree_string_view_t constraint_key = IREE_SV("llvm.x86.sse2.pause.signature");
  return loom_llvmir_target_legality_expect_intrinsic_shape(
      context, provider, op, 0, 0, constraint_key);
}

static iree_string_view_t loom_llvmir_x86_legality_rejection_name(
    loom_x86_packed_dot_rejection_bits_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_X86_PACKED_DOT_REJECTION_FEATURES)) {
    return IREE_SV("features");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD)) {
    return IREE_SV("payload");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_SHAPE)) {
    return IREE_SV("shape");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_FAMILY)) {
    return IREE_SV("family");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_FLAGS)) {
    return IREE_SV("flags");
  }
  return IREE_SV("invalid-request");
}

static bool loom_llvmir_x86_legality_verify_packed_dot(
    const loom_llvmir_target_legality_provider_t* provider,
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = true;

  loom_x86_packed_dot_match_request_t request = {0};
  if (!loom_x86_packed_dot_match_request_from_vector_op(
          loom_llvmir_target_legality_module(context), op, &request)) {
    return loom_llvmir_target_legality_fail(
        context, provider,
        LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TARGET_CONTRACT, op,
        IREE_SV("x86-packed-dot-request"), IREE_SV("invalid-request"));
  }
  const loom_llvmir_target_profile_t* profile =
      loom_llvmir_target_legality_profile(context);
  request.feature_bits =
      (loom_x86_packed_dot_feature_bits_t)profile->x86_packed_dot_feature_bits;

  loom_x86_packed_dot_match_diagnostic_t diagnostic = {0};
  const loom_x86_packed_dot_descriptor_t* descriptor =
      loom_x86_packed_dot_select(&request, &diagnostic);
  if (!descriptor) {
    return loom_llvmir_target_legality_fail(
        context, provider,
        LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TARGET_CONTRACT, op,
        IREE_SV("x86-packed-dot-descriptor"),
        loom_llvmir_x86_legality_rejection_name(diagnostic.rejection_bits));
  }

  return true;
}

static bool loom_llvmir_x86_legality_try_verify_op(
    const loom_llvmir_target_legality_provider_t* provider,
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  switch (op->kind) {
    case LOOM_OP_LLVMIR_INTRINSIC:
      return loom_llvmir_x86_legality_verify_intrinsic(provider, context, op,
                                                       out_handled);
    case LOOM_OP_VECTOR_DOT2F:
    case LOOM_OP_VECTOR_DOT4I:
      return loom_llvmir_x86_legality_verify_packed_dot(provider, context, op,
                                                        out_handled);
    default:
      return true;
  }
}

static const loom_llvmir_target_legality_provider_t kX86LegalityProvider = {
    .name = IREE_SVL("x86"),
    .try_verify_op = loom_llvmir_x86_legality_try_verify_op,
};

const loom_llvmir_target_legality_provider_t* loom_llvmir_x86_legality_provider(
    void) {
  return &kX86LegalityProvider;
}
