// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/x86/legality.h"

#include "loom/ops/llvmir/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/packed_dot_vector.h"

static bool loom_llvmir_x86_legality_target_is_x86(
    loom_llvmir_target_legality_context_t* context) {
  const loom_target_snapshot_t* snapshot =
      loom_llvmir_target_legality_snapshot(context);
  return iree_string_view_equal(snapshot->target_triple,
                                IREE_SV("x86_64-unknown-linux-gnu"));
}

static bool loom_llvmir_x86_legality_intrinsic_is_x86(iree_string_view_t kind) {
  return iree_string_view_equal(kind, IREE_SV("llvm.x86.rdtsc")) ||
         iree_string_view_equal(kind, IREE_SV("llvm.x86.sse2.pause"));
}

static iree_status_t loom_llvmir_x86_legality_verify_intrinsic(
    const loom_llvmir_target_legality_provider_t* provider,
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  loom_string_id_t kind_id = loom_llvmir_intrinsic_kind(op);
  iree_string_view_t kind = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_string_attr(
      context, provider, op, IREE_SV("kind"), kind_id, &kind));
  if (!loom_llvmir_x86_legality_intrinsic_is_x86(kind)) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (!loom_llvmir_x86_legality_target_is_x86(context)) {
    return loom_llvmir_target_legality_fail(
        context, provider, LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_INTRINSIC,
        op, IREE_SV("x86 llvmir.intrinsic requires an x86 target environment"),
        kind);
  }
  if (iree_string_view_equal(kind, IREE_SV("llvm.x86.rdtsc"))) {
    iree_string_view_t detail = IREE_SV("llvm.x86.rdtsc expects () -> i64");
    IREE_RETURN_IF_ERROR(loom_llvmir_target_legality_expect_intrinsic_shape(
        context, provider, op, 0, 1, detail));
    return loom_llvmir_target_legality_expect_scalar_result(
        context, provider, op, LOOM_SCALAR_TYPE_I64, detail);
  }
  iree_string_view_t detail = IREE_SV("llvm.x86.sse2.pause expects () -> ()");
  return loom_llvmir_target_legality_expect_intrinsic_shape(context, provider,
                                                            op, 0, 0, detail);
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

static iree_string_view_t loom_llvmir_x86_legality_rejection_detail(
    loom_x86_packed_dot_rejection_bits_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_X86_PACKED_DOT_REJECTION_FEATURES)) {
    return IREE_SV(
        "target profile does not enable a matching x86 packed-dot feature");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD)) {
    return IREE_SV(
        "no x86 packed-dot descriptor matches the vector dot payload types");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_SHAPE)) {
    return IREE_SV("no x86 packed-dot descriptor matches the vector dot shape");
  }
  return IREE_SV("no x86 packed-dot descriptor matches this vector dot op");
}

static iree_status_t loom_llvmir_x86_legality_verify_packed_dot(
    const loom_llvmir_target_legality_provider_t* provider,
    loom_llvmir_target_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  if (!loom_llvmir_x86_legality_target_is_x86(context)) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_x86_packed_dot_match_request_t request = {0};
  if (!loom_x86_packed_dot_match_request_from_vector_op(
          loom_llvmir_target_legality_module(context), op, &request)) {
    return loom_llvmir_target_legality_fail(
        context, provider,
        LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TARGET_CONTRACT, op,
        IREE_SV("no x86 packed-dot match request can be inferred"),
        IREE_SV("invalid-request"));
  }
  const loom_target_config_t* config =
      loom_llvmir_target_legality_config(context);
  request.feature_bits =
      (loom_x86_packed_dot_feature_bits_t)config->contract_feature_bits;

  loom_x86_packed_dot_match_diagnostic_t diagnostic = {0};
  const loom_x86_packed_dot_descriptor_t* descriptor =
      loom_x86_packed_dot_select(&request, &diagnostic);
  if (!descriptor) {
    return loom_llvmir_target_legality_fail(
        context, provider,
        LOOM_LLVMIR_TARGET_LEGALITY_UNSUPPORTED_TARGET_CONTRACT, op,
        loom_llvmir_x86_legality_rejection_detail(diagnostic.rejection_bits),
        loom_llvmir_x86_legality_rejection_name(diagnostic.rejection_bits));
  }

  loom_llvmir_target_legality_emit_event(
      context, provider,
      LOOM_LLVMIR_TARGET_LEGALITY_EVENT_TARGET_CONTRACT_SELECTED, op,
      IREE_SV("selected x86 packed-dot descriptor"), descriptor->name);
  return iree_ok_status();
}

static iree_status_t loom_llvmir_x86_legality_try_verify_op(
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
      return iree_ok_status();
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
