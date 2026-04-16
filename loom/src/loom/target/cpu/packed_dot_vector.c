// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/cpu/packed_dot_vector.h"

#include "loom/ops/vector/ops.h"

static bool loom_cpu_packed_dot_value_type(const loom_module_t* module,
                                           loom_value_id_t value_id,
                                           loom_type_t* out_type) {
  *out_type = loom_type_none();
  if (module == NULL || value_id >= module->values.count) return false;
  *out_type = loom_module_value_type(module, value_id);
  return true;
}

static bool loom_cpu_packed_dot_static_vector_element_count(
    loom_type_t type, uint64_t* out_element_count) {
  *out_element_count = 0;
  if (!loom_type_is_vector(type)) return false;
  return loom_type_static_element_count(type, out_element_count);
}

static bool loom_cpu_packed_dot_assign_uint16(uint64_t value,
                                              uint16_t* out_value) {
  if (value > UINT16_MAX) return false;
  *out_value = (uint16_t)value;
  return true;
}

static bool loom_cpu_packed_dot_assign_scaled_uint16(uint64_t value,
                                                     uint64_t multiplier,
                                                     uint16_t* out_value) {
  if (multiplier == 0 || value > UINT16_MAX / multiplier) return false;
  *out_value = (uint16_t)(value * multiplier);
  return true;
}

static bool loom_cpu_packed_dot_derive_grouped_shape(
    uint64_t input_lane_count, uint64_t result_lane_count,
    uint64_t source_lane_bit_width, loom_cpu_packed_dot_shape_t* out_shape) {
  *out_shape = (loom_cpu_packed_dot_shape_t){0};
  if (result_lane_count == 0 || input_lane_count % result_lane_count != 0) {
    return false;
  }
  if (!loom_cpu_packed_dot_assign_scaled_uint16(input_lane_count,
                                                source_lane_bit_width,
                                                &out_shape->vector_bit_width) ||
      !loom_cpu_packed_dot_assign_uint16(input_lane_count,
                                         &out_shape->input_lane_count) ||
      !loom_cpu_packed_dot_assign_uint16(result_lane_count,
                                         &out_shape->result_lane_count) ||
      !loom_cpu_packed_dot_assign_uint16(input_lane_count / result_lane_count,
                                         &out_shape->reduction_group_size)) {
    return false;
  }
  return true;
}

static bool loom_cpu_packed_dot_static_grouped_shapes_match(
    loom_type_t source_type, loom_type_t result_type, int64_t group_size) {
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      !loom_type_is_all_static(source_type) ||
      !loom_type_is_all_static(result_type)) {
    return false;
  }
  const uint8_t rank = loom_type_rank(source_type);
  if (rank == 0 || loom_type_rank(result_type) != rank) return false;
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

static bool loom_cpu_packed_dot_scalar_numeric_type(
    loom_scalar_type_t scalar_type,
    loom_cpu_packed_dot_numeric_type_t* out_numeric_type) {
  *out_numeric_type = LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN;
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F16:
      *out_numeric_type = LOOM_CPU_PACKED_DOT_NUMERIC_F16;
      return true;
    case LOOM_SCALAR_TYPE_BF16:
      *out_numeric_type = LOOM_CPU_PACKED_DOT_NUMERIC_BF16;
      return true;
    case LOOM_SCALAR_TYPE_I32:
      *out_numeric_type = LOOM_CPU_PACKED_DOT_NUMERIC_I32;
      return true;
    case LOOM_SCALAR_TYPE_F32:
      *out_numeric_type = LOOM_CPU_PACKED_DOT_NUMERIC_F32;
      return true;
    default:
      return false;
  }
}

static bool loom_cpu_packed_dot_dot4i_numeric_types(
    loom_vector_dot4i_kind_t kind,
    loom_cpu_packed_dot_numeric_type_t* out_lhs_type,
    loom_cpu_packed_dot_numeric_type_t* out_rhs_type) {
  *out_lhs_type = LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN;
  *out_rhs_type = LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN;
  switch (kind) {
    case LOOM_VECTOR_DOT4I_KIND_S8S8:
      *out_lhs_type = LOOM_CPU_PACKED_DOT_NUMERIC_I8;
      *out_rhs_type = LOOM_CPU_PACKED_DOT_NUMERIC_I8;
      return true;
    case LOOM_VECTOR_DOT4I_KIND_U8S8:
      *out_lhs_type = LOOM_CPU_PACKED_DOT_NUMERIC_U8;
      *out_rhs_type = LOOM_CPU_PACKED_DOT_NUMERIC_I8;
      return true;
    case LOOM_VECTOR_DOT4I_KIND_S8U8:
      *out_lhs_type = LOOM_CPU_PACKED_DOT_NUMERIC_I8;
      *out_rhs_type = LOOM_CPU_PACKED_DOT_NUMERIC_U8;
      return true;
    case LOOM_VECTOR_DOT4I_KIND_U8U8:
      *out_lhs_type = LOOM_CPU_PACKED_DOT_NUMERIC_U8;
      *out_rhs_type = LOOM_CPU_PACKED_DOT_NUMERIC_U8;
      return true;
    case LOOM_VECTOR_DOT4I_KIND_COUNT_:
    default:
      return false;
  }
}

static bool loom_cpu_packed_dot_match_request_from_dot2f_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_cpu_packed_dot_match_request_t* out_request) {
  if (!loom_vector_dot2f_isa(op)) return false;

  loom_type_t lhs_type = loom_type_none();
  loom_type_t rhs_type = loom_type_none();
  loom_type_t acc_type = loom_type_none();
  loom_type_t result_type = loom_type_none();
  if (!loom_cpu_packed_dot_value_type(module, loom_vector_dot2f_lhs(op),
                                      &lhs_type) ||
      !loom_cpu_packed_dot_value_type(module, loom_vector_dot2f_rhs(op),
                                      &rhs_type) ||
      !loom_cpu_packed_dot_value_type(module, loom_vector_dot2f_acc(op),
                                      &acc_type) ||
      !loom_cpu_packed_dot_value_type(module, loom_vector_dot2f_result(op),
                                      &result_type)) {
    return false;
  }

  loom_cpu_packed_dot_numeric_type_t source_numeric_type =
      LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN;
  loom_cpu_packed_dot_numeric_type_t rhs_numeric_type =
      LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN;
  loom_cpu_packed_dot_numeric_type_t acc_numeric_type =
      LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN;
  loom_cpu_packed_dot_numeric_type_t result_numeric_type =
      LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN;
  if (!loom_cpu_packed_dot_scalar_numeric_type(loom_type_element_type(lhs_type),
                                               &source_numeric_type) ||
      !loom_cpu_packed_dot_scalar_numeric_type(loom_type_element_type(rhs_type),
                                               &rhs_numeric_type) ||
      source_numeric_type != rhs_numeric_type ||
      !loom_cpu_packed_dot_scalar_numeric_type(loom_type_element_type(acc_type),
                                               &acc_numeric_type) ||
      !loom_cpu_packed_dot_scalar_numeric_type(
          loom_type_element_type(result_type), &result_numeric_type) ||
      acc_numeric_type != LOOM_CPU_PACKED_DOT_NUMERIC_F32 ||
      result_numeric_type != LOOM_CPU_PACKED_DOT_NUMERIC_F32 ||
      !loom_type_shape_equals(lhs_type, rhs_type) ||
      !loom_type_shape_equals(acc_type, result_type) ||
      !loom_cpu_packed_dot_static_grouped_shapes_match(lhs_type, result_type,
                                                       2)) {
    return false;
  }

  uint64_t input_lane_count = 0;
  uint64_t rhs_lane_count = 0;
  uint64_t result_lane_count = 0;
  uint64_t accumulator_lane_count = 0;
  if (!loom_cpu_packed_dot_static_vector_element_count(lhs_type,
                                                       &input_lane_count) ||
      !loom_cpu_packed_dot_static_vector_element_count(rhs_type,
                                                       &rhs_lane_count) ||
      !loom_cpu_packed_dot_static_vector_element_count(result_type,
                                                       &result_lane_count) ||
      !loom_cpu_packed_dot_static_vector_element_count(
          acc_type, &accumulator_lane_count) ||
      rhs_lane_count != input_lane_count ||
      accumulator_lane_count != result_lane_count) {
    return false;
  }

  if (!loom_cpu_packed_dot_derive_grouped_shape(
          input_lane_count, result_lane_count, 16, &out_request->shape)) {
    return false;
  }
  out_request->lhs_numeric_type = source_numeric_type;
  out_request->rhs_numeric_type = source_numeric_type;
  out_request->accumulator_numeric_type = LOOM_CPU_PACKED_DOT_NUMERIC_F32;
  out_request->result_numeric_type = LOOM_CPU_PACKED_DOT_NUMERIC_F32;
  return true;
}

static bool loom_cpu_packed_dot_match_request_from_dot4i_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_cpu_packed_dot_match_request_t* out_request) {
  if (!loom_vector_dot4i_isa(op)) return false;

  loom_cpu_packed_dot_numeric_type_t lhs_numeric_type =
      LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN;
  loom_cpu_packed_dot_numeric_type_t rhs_numeric_type =
      LOOM_CPU_PACKED_DOT_NUMERIC_UNKNOWN;
  if (!loom_cpu_packed_dot_dot4i_numeric_types(
          (loom_vector_dot4i_kind_t)loom_vector_dot4i_kind(op),
          &lhs_numeric_type, &rhs_numeric_type)) {
    return false;
  }

  loom_type_t lhs_type = loom_type_none();
  loom_type_t rhs_type = loom_type_none();
  loom_type_t acc_type = loom_type_none();
  loom_type_t result_type = loom_type_none();
  if (!loom_cpu_packed_dot_value_type(module, loom_vector_dot4i_lhs(op),
                                      &lhs_type) ||
      !loom_cpu_packed_dot_value_type(module, loom_vector_dot4i_rhs(op),
                                      &rhs_type) ||
      !loom_cpu_packed_dot_value_type(module, loom_vector_dot4i_acc(op),
                                      &acc_type) ||
      !loom_cpu_packed_dot_value_type(module, loom_vector_dot4i_result(op),
                                      &result_type)) {
    return false;
  }
  if (loom_type_element_type(lhs_type) != LOOM_SCALAR_TYPE_I8 ||
      loom_type_element_type(rhs_type) != LOOM_SCALAR_TYPE_I8 ||
      loom_type_element_type(acc_type) != LOOM_SCALAR_TYPE_I32 ||
      loom_type_element_type(result_type) != LOOM_SCALAR_TYPE_I32 ||
      !loom_type_shape_equals(lhs_type, rhs_type) ||
      !loom_type_shape_equals(acc_type, result_type) ||
      !loom_cpu_packed_dot_static_grouped_shapes_match(lhs_type, result_type,
                                                       4)) {
    return false;
  }

  uint64_t input_lane_count = 0;
  uint64_t rhs_lane_count = 0;
  uint64_t result_lane_count = 0;
  uint64_t accumulator_lane_count = 0;
  if (!loom_cpu_packed_dot_static_vector_element_count(lhs_type,
                                                       &input_lane_count) ||
      !loom_cpu_packed_dot_static_vector_element_count(rhs_type,
                                                       &rhs_lane_count) ||
      !loom_cpu_packed_dot_static_vector_element_count(result_type,
                                                       &result_lane_count) ||
      !loom_cpu_packed_dot_static_vector_element_count(
          acc_type, &accumulator_lane_count) ||
      rhs_lane_count != input_lane_count ||
      accumulator_lane_count != result_lane_count) {
    return false;
  }

  if (!loom_cpu_packed_dot_derive_grouped_shape(
          input_lane_count, result_lane_count, 8, &out_request->shape)) {
    return false;
  }
  out_request->lhs_numeric_type = lhs_numeric_type;
  out_request->rhs_numeric_type = rhs_numeric_type;
  out_request->accumulator_numeric_type = LOOM_CPU_PACKED_DOT_NUMERIC_I32;
  out_request->result_numeric_type = LOOM_CPU_PACKED_DOT_NUMERIC_I32;
  return true;
}

bool loom_cpu_packed_dot_match_request_from_vector_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_cpu_packed_dot_match_request_t* out_request) {
  if (out_request == NULL) return false;
  *out_request = (loom_cpu_packed_dot_match_request_t){0};
  if (module == NULL || op == NULL) return false;

  loom_cpu_packed_dot_match_request_t request = {0};
  if (loom_cpu_packed_dot_match_request_from_dot2f_op(module, op, &request) ||
      loom_cpu_packed_dot_match_request_from_dot4i_op(module, op, &request)) {
    *out_request = request;
    return true;
  }
  return false;
}
