// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract_vector.h"

#include "loom/ops/vector/ops.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool loom_contract_vector_fail(
    loom_contract_rejection_bits_t rejection_bits,
    loom_contract_diagnostic_t* out_diagnostic) {
  if (out_diagnostic) {
    out_diagnostic->rejection_bits = rejection_bits;
  }
  return false;
}

static bool loom_contract_vector_value_type(const loom_module_t* module,
                                            loom_value_id_t value_id,
                                            loom_type_t* out_type) {
  *out_type = loom_type_none();
  if (value_id >= module->values.count) {
    return false;
  }
  *out_type = loom_module_value_type(module, value_id);
  return true;
}

static bool loom_contract_vector_static_element_count(
    loom_type_t type, uint64_t* out_element_count) {
  *out_element_count = 0;
  if (!loom_type_is_vector(type)) {
    return false;
  }
  return loom_type_static_element_count(type, out_element_count);
}

static bool loom_contract_vector_assign_uint16(uint64_t value,
                                               uint16_t* out_value) {
  if (value > UINT16_MAX) {
    return false;
  }
  *out_value = (uint16_t)value;
  return true;
}

static bool loom_contract_vector_assign_scaled_uint16(uint64_t value,
                                                      uint64_t multiplier,
                                                      uint16_t* out_value) {
  if (multiplier == 0 || value > UINT16_MAX / multiplier) {
    return false;
  }
  *out_value = (uint16_t)(value * multiplier);
  return true;
}

static bool loom_contract_vector_static_grouped_shapes_match(
    loom_type_t source_type, loom_type_t result_type, int64_t group_size) {
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      !loom_type_is_all_static(source_type) ||
      !loom_type_is_all_static(result_type)) {
    return false;
  }
  const uint8_t rank = loom_type_rank(source_type);
  if (rank == 0 || loom_type_rank(result_type) != rank) {
    return false;
  }
  for (uint8_t i = 0; i + 1 < rank; ++i) {
    if (loom_type_dim_static_size_at(source_type, i) !=
        loom_type_dim_static_size_at(result_type, i)) {
      return false;
    }
  }
  const int64_t source_last_extent =
      loom_type_dim_static_size_at(source_type, rank - 1);
  const int64_t result_last_extent =
      loom_type_dim_static_size_at(result_type, rank - 1);
  return source_last_extent >= 0 && result_last_extent >= 0 &&
         source_last_extent % group_size == 0 &&
         source_last_extent / group_size == result_last_extent;
}

static loom_contract_operand_t loom_contract_vector_operand(
    loom_contract_operand_role_t role,
    loom_contract_numeric_type_t numeric_type) {
  return (loom_contract_operand_t){
      .role = role,
      .numeric_type = numeric_type,
  };
}

static bool loom_contract_vector_dot4i_numeric_types(
    loom_vector_dot4i_kind_t kind, loom_contract_numeric_type_t* out_lhs_type,
    loom_contract_numeric_type_t* out_rhs_type) {
  *out_lhs_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  *out_rhs_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  switch (kind) {
    case LOOM_VECTOR_DOT4I_KIND_S8S8:
      *out_lhs_type = LOOM_CONTRACT_NUMERIC_I8;
      *out_rhs_type = LOOM_CONTRACT_NUMERIC_I8;
      return true;
    case LOOM_VECTOR_DOT4I_KIND_U8S8:
      *out_lhs_type = LOOM_CONTRACT_NUMERIC_U8;
      *out_rhs_type = LOOM_CONTRACT_NUMERIC_I8;
      return true;
    case LOOM_VECTOR_DOT4I_KIND_S8U8:
      *out_lhs_type = LOOM_CONTRACT_NUMERIC_I8;
      *out_rhs_type = LOOM_CONTRACT_NUMERIC_U8;
      return true;
    case LOOM_VECTOR_DOT4I_KIND_U8U8:
      *out_lhs_type = LOOM_CONTRACT_NUMERIC_U8;
      *out_rhs_type = LOOM_CONTRACT_NUMERIC_U8;
      return true;
    case LOOM_VECTOR_DOT4I_KIND_COUNT_:
    default:
      return false;
  }
}

static bool loom_contract_vector_populate_fragment(
    uint64_t input_lane_count, uint64_t result_lane_count,
    uint64_t source_lane_bit_width, uint16_t k_group_size,
    loom_contract_request_t* request) {
  if (!loom_contract_vector_assign_uint16(
          input_lane_count, &request->fragment.source_lane_count) ||
      !loom_contract_vector_assign_uint16(
          result_lane_count, &request->fragment.result_lane_count) ||
      !loom_contract_vector_assign_scaled_uint16(
          input_lane_count, source_lane_bit_width,
          &request->fragment.vector_bit_width)) {
    return false;
  }
  request->fragment.atom_bits = LOOM_CONTRACT_FRAGMENT_VECTOR_LANE;
  request->shape.m = (int64_t)request->fragment.result_lane_count;
  request->shape.n = 1;
  request->shape.k = (int64_t)request->fragment.source_lane_count;
  request->k_group_size = k_group_size;
  return true;
}

static bool loom_contract_vector_finish(
    loom_lowering_policy_t policy, loom_contract_request_t* request,
    loom_contract_diagnostic_t* out_diagnostic) {
  request->capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT;
  request->policy = policy;
  if (!loom_contract_request_validate(request, out_diagnostic)) {
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Dot op adapters
//===----------------------------------------------------------------------===//

static bool loom_contract_request_from_dot2f_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_lowering_policy_t policy, loom_contract_request_t* request,
    loom_contract_diagnostic_t* out_diagnostic) {
  if (!loom_vector_dot2f_isa(op)) {
    return false;
  }

  loom_type_t lhs_type = loom_type_none();
  loom_type_t rhs_type = loom_type_none();
  loom_type_t accumulator_type = loom_type_none();
  loom_type_t result_type = loom_type_none();
  if (!loom_contract_vector_value_type(module, loom_vector_dot2f_lhs(op),
                                       &lhs_type) ||
      !loom_contract_vector_value_type(module, loom_vector_dot2f_rhs(op),
                                       &rhs_type) ||
      !loom_contract_vector_value_type(module, loom_vector_dot2f_acc(op),
                                       &accumulator_type) ||
      !loom_contract_vector_value_type(module, loom_vector_dot2f_result(op),
                                       &result_type)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_INVALID_REQUEST,
                                     out_diagnostic);
  }

  loom_contract_numeric_type_t source_numeric_type =
      LOOM_CONTRACT_NUMERIC_UNKNOWN;
  loom_contract_numeric_type_t rhs_numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  loom_contract_numeric_type_t accumulator_numeric_type =
      LOOM_CONTRACT_NUMERIC_UNKNOWN;
  loom_contract_numeric_type_t result_numeric_type =
      LOOM_CONTRACT_NUMERIC_UNKNOWN;
  if (!loom_contract_numeric_type_from_scalar(loom_type_element_type(lhs_type),
                                              false, &source_numeric_type) ||
      !loom_contract_numeric_type_from_scalar(loom_type_element_type(rhs_type),
                                              false, &rhs_numeric_type) ||
      source_numeric_type != rhs_numeric_type ||
      !loom_contract_numeric_type_from_scalar(
          loom_type_element_type(accumulator_type), false,
          &accumulator_numeric_type) ||
      !loom_contract_numeric_type_from_scalar(
          loom_type_element_type(result_type), false, &result_numeric_type) ||
      accumulator_numeric_type != LOOM_CONTRACT_NUMERIC_F32 ||
      result_numeric_type != LOOM_CONTRACT_NUMERIC_F32) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_NUMERIC,
                                     out_diagnostic);
  }
  if (!loom_type_shape_equals(lhs_type, rhs_type) ||
      !loom_type_shape_equals(accumulator_type, result_type) ||
      !loom_contract_vector_static_grouped_shapes_match(lhs_type, result_type,
                                                        2)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                     out_diagnostic);
  }

  uint64_t input_lane_count = 0;
  uint64_t rhs_lane_count = 0;
  uint64_t result_lane_count = 0;
  uint64_t accumulator_lane_count = 0;
  if (!loom_contract_vector_static_element_count(lhs_type, &input_lane_count) ||
      !loom_contract_vector_static_element_count(rhs_type, &rhs_lane_count) ||
      !loom_contract_vector_static_element_count(result_type,
                                                 &result_lane_count) ||
      !loom_contract_vector_static_element_count(accumulator_type,
                                                 &accumulator_lane_count) ||
      rhs_lane_count != input_lane_count ||
      accumulator_lane_count != result_lane_count) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                     out_diagnostic);
  }

  request->kind = LOOM_CONTRACT_KIND_VECTOR_DOT;
  request->arithmetic = LOOM_CONTRACT_ARITHMETIC_FLOAT_DOT;
  request->lhs = loom_contract_vector_operand(LOOM_CONTRACT_OPERAND_ROLE_LHS,
                                              source_numeric_type);
  request->rhs = loom_contract_vector_operand(LOOM_CONTRACT_OPERAND_ROLE_RHS,
                                              source_numeric_type);
  request->accumulator = loom_contract_vector_operand(
      LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, LOOM_CONTRACT_NUMERIC_F32);
  request->result = loom_contract_vector_operand(
      LOOM_CONTRACT_OPERAND_ROLE_RESULT, LOOM_CONTRACT_NUMERIC_F32);
  if (!loom_contract_vector_populate_fragment(
          input_lane_count, result_lane_count, 16, 2, request)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                     out_diagnostic);
  }
  return loom_contract_vector_finish(policy, request, out_diagnostic);
}

static bool loom_contract_request_from_dot4i_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_lowering_policy_t policy, loom_contract_request_t* request,
    loom_contract_diagnostic_t* out_diagnostic) {
  if (!loom_vector_dot4i_isa(op)) {
    return false;
  }

  loom_contract_numeric_type_t lhs_numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  loom_contract_numeric_type_t rhs_numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  if (!loom_contract_vector_dot4i_numeric_types(
          (loom_vector_dot4i_kind_t)loom_vector_dot4i_kind(op),
          &lhs_numeric_type, &rhs_numeric_type)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_NUMERIC,
                                     out_diagnostic);
  }

  loom_type_t lhs_type = loom_type_none();
  loom_type_t rhs_type = loom_type_none();
  loom_type_t accumulator_type = loom_type_none();
  loom_type_t result_type = loom_type_none();
  if (!loom_contract_vector_value_type(module, loom_vector_dot4i_lhs(op),
                                       &lhs_type) ||
      !loom_contract_vector_value_type(module, loom_vector_dot4i_rhs(op),
                                       &rhs_type) ||
      !loom_contract_vector_value_type(module, loom_vector_dot4i_acc(op),
                                       &accumulator_type) ||
      !loom_contract_vector_value_type(module, loom_vector_dot4i_result(op),
                                       &result_type)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_INVALID_REQUEST,
                                     out_diagnostic);
  }
  if (loom_type_element_type(lhs_type) != LOOM_SCALAR_TYPE_I8 ||
      loom_type_element_type(rhs_type) != LOOM_SCALAR_TYPE_I8 ||
      loom_type_element_type(accumulator_type) != LOOM_SCALAR_TYPE_I32 ||
      loom_type_element_type(result_type) != LOOM_SCALAR_TYPE_I32) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_NUMERIC,
                                     out_diagnostic);
  }
  if (!loom_type_shape_equals(lhs_type, rhs_type) ||
      !loom_type_shape_equals(accumulator_type, result_type) ||
      !loom_contract_vector_static_grouped_shapes_match(lhs_type, result_type,
                                                        4)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                     out_diagnostic);
  }

  uint64_t input_lane_count = 0;
  uint64_t rhs_lane_count = 0;
  uint64_t result_lane_count = 0;
  uint64_t accumulator_lane_count = 0;
  if (!loom_contract_vector_static_element_count(lhs_type, &input_lane_count) ||
      !loom_contract_vector_static_element_count(rhs_type, &rhs_lane_count) ||
      !loom_contract_vector_static_element_count(result_type,
                                                 &result_lane_count) ||
      !loom_contract_vector_static_element_count(accumulator_type,
                                                 &accumulator_lane_count) ||
      rhs_lane_count != input_lane_count ||
      accumulator_lane_count != result_lane_count) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                     out_diagnostic);
  }

  request->kind = LOOM_CONTRACT_KIND_VECTOR_DOT;
  request->arithmetic = LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT;
  request->lhs = loom_contract_vector_operand(LOOM_CONTRACT_OPERAND_ROLE_LHS,
                                              lhs_numeric_type);
  request->rhs = loom_contract_vector_operand(LOOM_CONTRACT_OPERAND_ROLE_RHS,
                                              rhs_numeric_type);
  request->accumulator = loom_contract_vector_operand(
      LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, LOOM_CONTRACT_NUMERIC_I32);
  request->result = loom_contract_vector_operand(
      LOOM_CONTRACT_OPERAND_ROLE_RESULT, LOOM_CONTRACT_NUMERIC_I32);
  if (!loom_contract_vector_populate_fragment(
          input_lane_count, result_lane_count, 8, 4, request)) {
    return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_SHAPE,
                                     out_diagnostic);
  }
  return loom_contract_vector_finish(policy, request, out_diagnostic);
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

bool loom_contract_request_from_vector_dot_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_lowering_policy_t policy, loom_contract_request_t* out_request,
    loom_contract_diagnostic_t* out_diagnostic) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(op);
  IREE_ASSERT_ARGUMENT(out_request);
  loom_contract_request_initialize(out_request);
  if (out_diagnostic) {
    *out_diagnostic = (loom_contract_diagnostic_t){0};
  }

  if (loom_vector_dot2f_isa(op)) {
    return loom_contract_request_from_dot2f_op(module, op, policy, out_request,
                                               out_diagnostic);
  }
  if (loom_vector_dot4i_isa(op)) {
    return loom_contract_request_from_dot4i_op(module, op, policy, out_request,
                                               out_diagnostic);
  }
  return loom_contract_vector_fail(LOOM_CONTRACT_REJECTION_INVALID_REQUEST,
                                   out_diagnostic);
}
