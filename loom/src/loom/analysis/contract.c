// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool loom_contract_operand_role_matches(
    loom_contract_operand_t operand, loom_contract_operand_role_t role) {
  return operand.role == role;
}

static bool loom_contract_operand_numeric_is_known(
    loom_contract_operand_t operand) {
  return operand.numeric_type != LOOM_CONTRACT_NUMERIC_UNKNOWN;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void loom_contract_request_initialize(loom_contract_request_t* out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  *out_request = (loom_contract_request_t){
      .scale_kind = LOOM_CONTRACT_SCALE_NONE,
      .policy = LOOM_LOWERING_POLICY_REFERENCE_ALLOWED,
  };
}

bool loom_contract_request_validate(
    const loom_contract_request_t* request,
    loom_contract_diagnostic_t* out_diagnostic) {
  IREE_ASSERT_ARGUMENT(request);
  if (out_diagnostic) {
    *out_diagnostic = (loom_contract_diagnostic_t){0};
  }

  loom_contract_rejection_bits_t rejection_bits = 0;
  if (request->kind == LOOM_CONTRACT_KIND_UNKNOWN ||
      request->arithmetic == LOOM_CONTRACT_ARITHMETIC_UNKNOWN) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_INVALID_REQUEST;
  }
  if (request->shape.m <= 0 || request->shape.n <= 0 || request->shape.k <= 0 ||
      request->k_group_size == 0) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_SHAPE;
  }
  if (!loom_contract_operand_role_matches(request->lhs,
                                          LOOM_CONTRACT_OPERAND_ROLE_LHS) ||
      !loom_contract_operand_role_matches(request->rhs,
                                          LOOM_CONTRACT_OPERAND_ROLE_RHS) ||
      !loom_contract_operand_role_matches(
          request->accumulator, LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR) ||
      !loom_contract_operand_role_matches(request->result,
                                          LOOM_CONTRACT_OPERAND_ROLE_RESULT)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_ROLE;
  }
  if (!loom_contract_operand_numeric_is_known(request->lhs) ||
      !loom_contract_operand_numeric_is_known(request->rhs) ||
      !loom_contract_operand_numeric_is_known(request->accumulator) ||
      !loom_contract_operand_numeric_is_known(request->result)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_NUMERIC;
  }
  if (request->capability_class == LOOM_CONTRACT_CAPABILITY_CLASS_UNKNOWN) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_CAPABILITY;
  }
  if (request->scale_kind == LOOM_CONTRACT_SCALE_UNKNOWN) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_CAPABILITY;
  }
  if (!iree_all_bits_set(request->available_capability_flags,
                         request->required_capability_flags)) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_CAPABILITY;
  }
  if (request->policy == LOOM_LOWERING_POLICY_UNKNOWN) {
    rejection_bits |= LOOM_CONTRACT_REJECTION_POLICY;
  }

  if (out_diagnostic) {
    out_diagnostic->rejection_bits = rejection_bits;
  }
  return rejection_bits == 0;
}

bool loom_contract_numeric_type_from_scalar(
    loom_scalar_type_t scalar_type, bool unsigned_integer,
    loom_contract_numeric_type_t* out_numeric_type) {
  IREE_ASSERT_ARGUMENT(out_numeric_type);
  *out_numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I8:
      *out_numeric_type = unsigned_integer ? LOOM_CONTRACT_NUMERIC_U8
                                           : LOOM_CONTRACT_NUMERIC_I8;
      return true;
    case LOOM_SCALAR_TYPE_I16:
      *out_numeric_type = unsigned_integer ? LOOM_CONTRACT_NUMERIC_U16
                                           : LOOM_CONTRACT_NUMERIC_I16;
      return true;
    case LOOM_SCALAR_TYPE_I32:
      *out_numeric_type = unsigned_integer ? LOOM_CONTRACT_NUMERIC_U32
                                           : LOOM_CONTRACT_NUMERIC_I32;
      return true;
    case LOOM_SCALAR_TYPE_F16:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_F16;
      return true;
    case LOOM_SCALAR_TYPE_BF16:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_BF16;
      return true;
    case LOOM_SCALAR_TYPE_F32:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
      return true;
    case LOOM_SCALAR_TYPE_F64:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_F64;
      return true;
    default:
      return false;
  }
}

iree_string_view_t loom_contract_numeric_type_name(
    loom_contract_numeric_type_t numeric_type) {
  switch (numeric_type) {
    case LOOM_CONTRACT_NUMERIC_I4:
      return IREE_SV("i4");
    case LOOM_CONTRACT_NUMERIC_U4:
      return IREE_SV("u4");
    case LOOM_CONTRACT_NUMERIC_I8:
      return IREE_SV("i8");
    case LOOM_CONTRACT_NUMERIC_U8:
      return IREE_SV("u8");
    case LOOM_CONTRACT_NUMERIC_I16:
      return IREE_SV("i16");
    case LOOM_CONTRACT_NUMERIC_U16:
      return IREE_SV("u16");
    case LOOM_CONTRACT_NUMERIC_I32:
      return IREE_SV("i32");
    case LOOM_CONTRACT_NUMERIC_U32:
      return IREE_SV("u32");
    case LOOM_CONTRACT_NUMERIC_F16:
      return IREE_SV("f16");
    case LOOM_CONTRACT_NUMERIC_BF16:
      return IREE_SV("bf16");
    case LOOM_CONTRACT_NUMERIC_F32:
      return IREE_SV("f32");
    case LOOM_CONTRACT_NUMERIC_F64:
      return IREE_SV("f64");
    case LOOM_CONTRACT_NUMERIC_FP8:
      return IREE_SV("fp8");
    case LOOM_CONTRACT_NUMERIC_BF8:
      return IREE_SV("bf8");
    case LOOM_CONTRACT_NUMERIC_FP6:
      return IREE_SV("fp6");
    case LOOM_CONTRACT_NUMERIC_BF6:
      return IREE_SV("bf6");
    case LOOM_CONTRACT_NUMERIC_FP4:
      return IREE_SV("fp4");
    case LOOM_CONTRACT_NUMERIC_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_contract_rejection_detail(
    loom_contract_rejection_bits_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_CONTRACT_REJECTION_INVALID_REQUEST)) {
    return IREE_SV("contract request is incomplete or invalid");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_SHAPE)) {
    return IREE_SV("contract shape or K grouping is not statically proven");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_ROLE)) {
    return IREE_SV("contract operand roles are missing or inconsistent");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_NUMERIC)) {
    return IREE_SV("contract numeric payload facts are missing or unsupported");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_FRAGMENT)) {
    return IREE_SV("contract fragment facts are insufficient");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_CAPABILITY)) {
    return IREE_SV("contract capability class is unsupported");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_CONTRACT_REJECTION_POLICY)) {
    return IREE_SV("contract lowering policy forbids the selected path");
  }
  return IREE_SV("contract request is not representable");
}
