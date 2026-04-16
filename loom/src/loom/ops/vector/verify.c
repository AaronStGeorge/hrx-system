// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <math.h>

#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/atomic.h"
#include "loom/ops/cache.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/encoding/roles.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"

#define LOOM_ASSERT_ATOMIC_KIND_VALUE(dialect_value, shared_value) \
  static_assert((int)(dialect_value) == (int)(shared_value),       \
                #dialect_value " must match " #shared_value)

LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_COUNT_, LOOM_ATOMIC_KIND_COUNT_);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_XCHGI, LOOM_ATOMIC_KIND_XCHGI);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_XCHGF, LOOM_ATOMIC_KIND_XCHGF);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_ADDI, LOOM_ATOMIC_KIND_ADDI);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_ADDF, LOOM_ATOMIC_KIND_ADDF);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_SUBI, LOOM_ATOMIC_KIND_SUBI);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_ANDI, LOOM_ATOMIC_KIND_ANDI);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_ORI, LOOM_ATOMIC_KIND_ORI);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_XORI, LOOM_ATOMIC_KIND_XORI);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_MINSI, LOOM_ATOMIC_KIND_MINSI);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_MAXSI, LOOM_ATOMIC_KIND_MAXSI);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_MINUI, LOOM_ATOMIC_KIND_MINUI);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_MAXUI, LOOM_ATOMIC_KIND_MAXUI);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_MINIMUMF,
                              LOOM_ATOMIC_KIND_MINIMUMF);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_MAXIMUMF,
                              LOOM_ATOMIC_KIND_MAXIMUMF);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_MINNUMF,
                              LOOM_ATOMIC_KIND_MINNUMF);
LOOM_ASSERT_ATOMIC_KIND_VALUE(LOOM_VECTOR_KIND_MAXNUMF,
                              LOOM_ATOMIC_KIND_MAXNUMF);

#undef LOOM_ASSERT_ATOMIC_KIND_VALUE

#define LOOM_ASSERT_CACHE_SCOPE_VALUE(dialect_value, shared_value) \
  static_assert((int)(dialect_value) == (int)(shared_value),       \
                #dialect_value " must match " #shared_value)

LOOM_ASSERT_CACHE_SCOPE_VALUE(LOOM_VECTOR_CACHE_SCOPE_COUNT_,
                              LOOM_CACHE_SCOPE_COUNT_);
LOOM_ASSERT_CACHE_SCOPE_VALUE(LOOM_VECTOR_CACHE_SCOPE_CU, LOOM_CACHE_SCOPE_CU);
LOOM_ASSERT_CACHE_SCOPE_VALUE(LOOM_VECTOR_CACHE_SCOPE_SE, LOOM_CACHE_SCOPE_SE);
LOOM_ASSERT_CACHE_SCOPE_VALUE(LOOM_VECTOR_CACHE_SCOPE_DEVICE,
                              LOOM_CACHE_SCOPE_DEVICE);
LOOM_ASSERT_CACHE_SCOPE_VALUE(LOOM_VECTOR_CACHE_SCOPE_SYSTEM,
                              LOOM_CACHE_SCOPE_SYSTEM);

#undef LOOM_ASSERT_CACHE_SCOPE_VALUE

#define LOOM_ASSERT_CACHE_TEMPORAL_VALUE(dialect_value, shared_value) \
  static_assert((int)(dialect_value) == (int)(shared_value),          \
                #dialect_value " must match " #shared_value)

LOOM_ASSERT_CACHE_TEMPORAL_VALUE(LOOM_VECTOR_CACHE_TEMPORAL_COUNT_,
                                 LOOM_CACHE_TEMPORAL_COUNT_);
LOOM_ASSERT_CACHE_TEMPORAL_VALUE(LOOM_VECTOR_CACHE_TEMPORAL_REGULAR,
                                 LOOM_CACHE_TEMPORAL_REGULAR);
LOOM_ASSERT_CACHE_TEMPORAL_VALUE(LOOM_VECTOR_CACHE_TEMPORAL_NON_TEMPORAL,
                                 LOOM_CACHE_TEMPORAL_NON_TEMPORAL);
LOOM_ASSERT_CACHE_TEMPORAL_VALUE(LOOM_VECTOR_CACHE_TEMPORAL_HIGH_TEMPORAL,
                                 LOOM_CACHE_TEMPORAL_HIGH_TEMPORAL);
LOOM_ASSERT_CACHE_TEMPORAL_VALUE(LOOM_VECTOR_CACHE_TEMPORAL_LAST_USE,
                                 LOOM_CACHE_TEMPORAL_LAST_USE);
LOOM_ASSERT_CACHE_TEMPORAL_VALUE(LOOM_VECTOR_CACHE_TEMPORAL_WRITEBACK,
                                 LOOM_CACHE_TEMPORAL_WRITEBACK);
LOOM_ASSERT_CACHE_TEMPORAL_VALUE(
    LOOM_VECTOR_CACHE_TEMPORAL_NON_TEMPORAL_REGULAR,
    LOOM_CACHE_TEMPORAL_NON_TEMPORAL_REGULAR);
LOOM_ASSERT_CACHE_TEMPORAL_VALUE(
    LOOM_VECTOR_CACHE_TEMPORAL_REGULAR_NON_TEMPORAL,
    LOOM_CACHE_TEMPORAL_REGULAR_NON_TEMPORAL);
LOOM_ASSERT_CACHE_TEMPORAL_VALUE(
    LOOM_VECTOR_CACHE_TEMPORAL_NON_TEMPORAL_HIGH_TEMPORAL,
    LOOM_CACHE_TEMPORAL_NON_TEMPORAL_HIGH_TEMPORAL);
LOOM_ASSERT_CACHE_TEMPORAL_VALUE(
    LOOM_VECTOR_CACHE_TEMPORAL_NON_TEMPORAL_WRITEBACK,
    LOOM_CACHE_TEMPORAL_NON_TEMPORAL_WRITEBACK);
LOOM_ASSERT_CACHE_TEMPORAL_VALUE(LOOM_VECTOR_CACHE_TEMPORAL_BYPASS,
                                 LOOM_CACHE_TEMPORAL_BYPASS);

#undef LOOM_ASSERT_CACHE_TEMPORAL_VALUE

static iree_status_t loom_vector_emit(iree_diagnostic_emitter_t emitter,
                                      const loom_op_t* op,
                                      const loom_error_def_t* error,
                                      const loom_diagnostic_param_t* params,
                                      iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static uint32_t loom_vector_saturating_count(uint64_t count) {
  if (count > UINT32_MAX) return UINT32_MAX;
  return (uint32_t)count;
}

static iree_status_t loom_vector_emit_attribute_kind_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, loom_attr_kind_t actual_kind,
    loom_attr_kind_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_u32(actual_kind),
      loom_param_u32(expected_kind),
  };
  return loom_vector_emit(emitter, op, &loom_err_type_005, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_attribute_value_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_vector_emit(emitter, op, &loom_err_structure_014, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_operand_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t operand_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(operand_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_vector_emit(emitter, op, &loom_err_type_003, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_result_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t result_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(result_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_vector_emit(emitter, op, &loom_err_type_004, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_result_code_capacity(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t result_name, loom_type_t actual_type,
    int64_t threshold_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(result_name),
      loom_param_type(actual_type),
      loom_param_i64(threshold_count),
      loom_param_i64(threshold_count),
  };
  return loom_vector_emit(emitter, op, &loom_err_type_011, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_static_threshold_order(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, uint32_t left_index, uint32_t right_index) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_u32(left_index),
      loom_param_u32(right_index),
  };
  return loom_vector_emit(emitter, op, &loom_err_structure_015, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_field_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    bool field_is_result, iree_string_view_t field_name,
    loom_type_t actual_type, iree_string_view_t expected_constraint) {
  if (field_is_result) {
    return loom_vector_emit_result_constraint(emitter, op, field_name,
                                              actual_type, expected_constraint);
  }
  return loom_vector_emit_operand_constraint(emitter, op, field_name,
                                             actual_type, expected_constraint);
}

static iree_status_t loom_vector_emit_rank_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, uint8_t actual_rank,
    iree_string_view_t expected_field_name, uint8_t expected_rank) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_i64(actual_rank),
      loom_param_string(expected_field_name),
      loom_param_i64(expected_rank),
  };
  return loom_vector_emit(emitter, op, &loom_err_shape_001, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_shape_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, iree_string_view_t expected_field_name) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_string(expected_field_name),
  };
  return loom_vector_emit(emitter, op, &loom_err_shape_002, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_emit_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, uint64_t actual_count,
    iree_string_view_t expected_field_name, uint64_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_u32(loom_vector_saturating_count(actual_count)),
      loom_param_string(expected_field_name),
      loom_param_u32(loom_vector_saturating_count(expected_count)),
  };
  return loom_vector_emit(emitter, op, &loom_err_structure_013, params,
                          IREE_ARRAYSIZE(params));
}

static uint16_t loom_vector_dynamic_sentinel_count(loom_attribute_t values) {
  uint16_t dynamic_count = 0;
  for (uint16_t i = 0; i < values.count; ++i) {
    if (values.i64_array[i] == INT64_MIN) ++dynamic_count;
  }
  return dynamic_count;
}

static bool loom_vector_static_index_in_bounds(loom_type_t source_type,
                                               uint8_t source_axis,
                                               int64_t static_index) {
  if (static_index < 0) return false;
  if (loom_type_dim_is_dynamic_at(source_type, source_axis)) return true;
  return static_index < loom_type_dim_static_size_at(source_type, source_axis);
}

static bool loom_vector_find_static_index_out_of_bounds(
    loom_attribute_t static_indices, loom_type_t source_type,
    uint16_t* out_axis, int64_t* out_static_index, int64_t* out_bound) {
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    int64_t static_index = static_indices.i64_array[i];
    if (static_index == INT64_MIN) continue;
    if (loom_vector_static_index_in_bounds(source_type, (uint8_t)i,
                                           static_index)) {
      continue;
    }
    *out_axis = i;
    *out_static_index = static_index;
    *out_bound = loom_type_dim_is_dynamic_at(source_type, i)
                     ? -1
                     : loom_type_dim_static_size_at(source_type, i);
    return true;
  }
  return false;
}

static iree_status_t loom_vector_emit_static_index_out_of_bounds(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t axis,
    int64_t static_index, int64_t bound) {
  int64_t total = static_index == INT64_MAX ? INT64_MAX : static_index + 1;
  loom_diagnostic_param_t params[] = {
      loom_param_i64(axis),  loom_param_i64(static_index), loom_param_i64(1),
      loom_param_i64(total), loom_param_i64(bound),
  };
  return loom_vector_emit(emitter, op, &loom_err_subrange_004, params,
                          IREE_ARRAYSIZE(params));
}

static int64_t loom_vector_saturating_add_i64(int64_t lhs, int64_t rhs) {
  if (rhs > 0 && lhs > INT64_MAX - rhs) return INT64_MAX;
  if (rhs < 0 && lhs < INT64_MIN - rhs) return INT64_MIN;
  return lhs + rhs;
}

static iree_status_t loom_vector_emit_static_access_out_of_bounds(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t axis,
    int64_t offset, int64_t size, int64_t bound) {
  loom_diagnostic_param_t params[] = {
      loom_param_i64(axis),
      loom_param_i64(offset),
      loom_param_i64(size),
      loom_param_i64(loom_vector_saturating_add_i64(offset, size)),
      loom_param_i64(bound),
  };
  return loom_vector_emit(emitter, op, &loom_err_subrange_004, params,
                          IREE_ARRAYSIZE(params));
}

static bool loom_vector_tail_shape_equals(loom_type_t source_type,
                                          uint8_t consumed_rank,
                                          loom_type_t tail_type) {
  uint8_t tail_rank = loom_type_rank(tail_type);
  for (uint8_t i = 0; i < tail_rank; ++i) {
    if (loom_type_dim(source_type, consumed_rank + i) !=
        loom_type_dim(tail_type, i)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_vector_verify_subvalue_type(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t value_name, loom_type_t value_type,
    iree_string_view_t expected_shape_name, loom_type_t source_type,
    uint8_t consumed_rank, bool value_is_result) {
  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t expected_rank = source_rank - consumed_rank;
  if (expected_rank == 0) {
    if (loom_type_is_scalar(value_type)) return iree_ok_status();
    return loom_vector_emit_field_constraint(emitter, op, value_is_result,
                                             value_name, value_type,
                                             IREE_SV("scalar tail value"));
  }

  if (!loom_type_is_vector(value_type)) {
    return loom_vector_emit_field_constraint(emitter, op, value_is_result,
                                             value_name, value_type,
                                             IREE_SV("vector tail value"));
  }
  uint8_t value_rank = loom_type_rank(value_type);
  if (value_rank != expected_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, value_name, value_rank,
                                          IREE_SV("source tail"),
                                          expected_rank);
  }
  if (loom_vector_tail_shape_equals(source_type, consumed_rank, value_type)) {
    return iree_ok_status();
  }
  return loom_vector_emit_shape_mismatch(emitter, op, value_name,
                                         expected_shape_name);
}

static bool loom_vector_reduce_kind_accepts_integer(uint8_t kind) {
  switch (kind) {
    case LOOM_VECTOR_REDUCE_KIND_ADDI:
    case LOOM_VECTOR_REDUCE_KIND_MULI:
    case LOOM_VECTOR_REDUCE_KIND_MINSI:
    case LOOM_VECTOR_REDUCE_KIND_MAXSI:
    case LOOM_VECTOR_REDUCE_KIND_MINUI:
    case LOOM_VECTOR_REDUCE_KIND_MAXUI:
    case LOOM_VECTOR_REDUCE_KIND_ANDI:
    case LOOM_VECTOR_REDUCE_KIND_ORI:
    case LOOM_VECTOR_REDUCE_KIND_XORI:
      return true;
    default:
      return false;
  }
}

static bool loom_vector_reduce_kind_accepts_float(uint8_t kind) {
  switch (kind) {
    case LOOM_VECTOR_REDUCE_KIND_ADDF:
    case LOOM_VECTOR_REDUCE_KIND_MULF:
    case LOOM_VECTOR_REDUCE_KIND_MINIMUMF:
    case LOOM_VECTOR_REDUCE_KIND_MAXIMUMF:
    case LOOM_VECTOR_REDUCE_KIND_MINNUMF:
    case LOOM_VECTOR_REDUCE_KIND_MAXNUMF:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_vector_verify_atomic_kind(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t value_name, loom_type_t value_type, uint8_t kind,
    bool allow_exchange) {
  if (!allow_exchange && loom_atomic_kind_is_exchange(kind)) {
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("kind"), kind,
        IREE_SV("non-exchange atomic reduce kind"));
  }
  if (!loom_atomic_kind_is_valid(kind)) return iree_ok_status();
  if (!loom_type_is_vector(value_type)) return iree_ok_status();

  loom_scalar_type_t element_type = loom_type_element_type(value_type);
  if (loom_scalar_type_is_integer(element_type) &&
      loom_atomic_kind_accepts_integer(kind)) {
    return iree_ok_status();
  }
  if (loom_scalar_type_is_float(element_type) &&
      loom_atomic_kind_accepts_float(kind)) {
    return iree_ok_status();
  }

  iree_string_view_t expected_constraint =
      loom_atomic_kind_accepts_integer(kind)
          ? IREE_SV("integer element type for atomic kind")
          : IREE_SV("floating-point element type for atomic kind");
  return loom_vector_emit_operand_constraint(emitter, op, value_name,
                                             value_type, expected_constraint);
}

static iree_status_t loom_vector_emit_offset_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t operand_name, uint16_t offset_count, uint8_t rank) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(operand_name),
      loom_param_u32(offset_count),
      loom_param_i64(rank),
  };
  return loom_vector_emit(emitter, op, &loom_err_subrange_001, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_vector_verify_dynamic_sentinel_count(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t dynamic_field_name, loom_attribute_t static_values,
    uint16_t dynamic_value_count) {
  uint16_t expected_dynamic_count =
      loom_vector_dynamic_sentinel_count(static_values);
  if (dynamic_value_count == expected_dynamic_count) return iree_ok_status();
  return loom_vector_emit_count_mismatch(
      emitter, op, dynamic_field_name, dynamic_value_count,
      IREE_SV("dynamic sentinels"), expected_dynamic_count);
}

static iree_status_t loom_vector_verify_dynamic_index_count(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_attribute_t static_indices, uint16_t dynamic_index_count) {
  return loom_vector_verify_dynamic_sentinel_count(
      emitter, op, IREE_SV("indices"), static_indices, dynamic_index_count);
}

static bool loom_vector_find_static_slice_out_of_bounds(
    loom_attribute_t static_offsets, loom_type_t source_type,
    loom_type_t result_type, uint16_t* out_axis, int64_t* out_offset,
    int64_t* out_extent, int64_t* out_bound) {
  uint8_t rank = loom_type_rank(source_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    int64_t offset = static_offsets.i64_array[axis];
    bool offset_is_static = offset != INT64_MIN;
    bool extent_is_static = !loom_type_dim_is_dynamic_at(result_type, axis);
    int64_t extent =
        extent_is_static ? loom_type_dim_static_size_at(result_type, axis) : 1;
    int64_t bound = loom_type_dim_is_dynamic_at(source_type, axis)
                        ? -1
                        : loom_type_dim_static_size_at(source_type, axis);

    if (offset_is_static && offset < 0) {
      *out_axis = axis;
      *out_offset = offset;
      *out_extent = extent;
      *out_bound = bound;
      return true;
    }

    if (bound < 0) continue;
    if (offset_is_static && offset > bound) {
      *out_axis = axis;
      *out_offset = offset;
      *out_extent = 0;
      *out_bound = bound;
      return true;
    }
    if (!extent_is_static) continue;
    if (extent > bound) {
      *out_axis = axis;
      *out_offset = offset_is_static ? offset : 0;
      *out_extent = extent;
      *out_bound = bound;
      return true;
    }
    if (offset_is_static && extent > bound - offset) {
      *out_axis = axis;
      *out_offset = offset;
      *out_extent = extent;
      *out_bound = bound;
      return true;
    }
  }
  return false;
}

static bool loom_vector_dim_equals(loom_type_t lhs_type, uint8_t lhs_axis,
                                   loom_type_t rhs_type, uint8_t rhs_axis) {
  return loom_type_dim(lhs_type, lhs_axis) == loom_type_dim(rhs_type, rhs_axis);
}

static bool loom_vector_shapes_match(loom_type_t lhs_type,
                                     loom_type_t rhs_type) {
  uint8_t rank = loom_type_rank(lhs_type);
  if (loom_type_rank(rhs_type) != rank) return false;
  for (uint8_t axis = 0; axis < rank; ++axis) {
    if (!loom_vector_dim_equals(lhs_type, axis, rhs_type, axis)) return false;
  }
  return true;
}

static iree_string_view_t loom_vector_grouped_last_axis_divisibility_constraint(
    int64_t group_size) {
  switch (group_size) {
    case 2:
      return IREE_SV("last axis extent divisible by 2");
    case 4:
      return IREE_SV("last axis extent divisible by 4");
    case 8:
      return IREE_SV("last axis extent divisible by 8");
    default:
      return IREE_SV("last axis extent divisible by group size");
  }
}

static iree_string_view_t loom_vector_grouped_last_axis_result_constraint(
    int64_t group_size) {
  switch (group_size) {
    case 2:
      return IREE_SV(
          "last axis extent equal to lhs last axis extent divided by 2");
    case 4:
      return IREE_SV(
          "last axis extent equal to lhs last axis extent divided by 4");
    case 8:
      return IREE_SV(
          "last axis extent equal to lhs last axis extent divided by 8");
    default:
      return IREE_SV(
          "last axis extent equal to lhs last axis extent divided by group "
          "size");
  }
}

static iree_status_t loom_vector_verify_grouped_last_axis_shape(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_type_t source_type, loom_type_t result_type, int64_t group_size) {
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(result_type);
  if (source_rank == 0 || result_rank == 0) return iree_ok_status();
  if (result_rank != source_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("result"),
                                          result_rank, IREE_SV("lhs"),
                                          source_rank);
  }

  uint8_t grouped_axis = source_rank - 1;
  for (uint8_t axis = 0; axis < grouped_axis; ++axis) {
    if (loom_vector_dim_equals(source_type, axis, result_type, axis)) continue;
    return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("result"),
                                           IREE_SV("lhs leading axes"));
  }

  if (loom_type_dim_is_dynamic_at(source_type, grouped_axis)) {
    return iree_ok_status();
  }

  int64_t source_axis_size =
      loom_type_dim_static_size_at(source_type, grouped_axis);
  if ((source_axis_size % group_size) != 0) {
    return loom_vector_emit_operand_constraint(
        emitter, op, IREE_SV("lhs"), source_type,
        loom_vector_grouped_last_axis_divisibility_constraint(group_size));
  }
  if (loom_type_dim_is_dynamic_at(result_type, grouped_axis)) {
    return iree_ok_status();
  }

  int64_t result_axis_size =
      loom_type_dim_static_size_at(result_type, grouped_axis);
  if (result_axis_size == source_axis_size / group_size) {
    return iree_ok_status();
  }
  return loom_vector_emit_result_constraint(
      emitter, op, IREE_SV("result"), result_type,
      loom_vector_grouped_last_axis_result_constraint(group_size));
}

static bool loom_vector_find_static_memory_access_out_of_bounds(
    const loom_vector_memory_access_t* access, loom_attribute_t static_indices,
    uint16_t* out_axis, int64_t* out_offset, int64_t* out_extent,
    int64_t* out_bound) {
  uint8_t view_rank = access->view_rank;
  for (uint8_t axis = 0; axis < view_rank; ++axis) {
    int64_t offset = static_indices.i64_array[axis];
    if (offset == INT64_MIN) continue;

    int64_t extent = 1;
    bool extent_is_static =
        loom_vector_memory_access_static_axis_extent(access, axis, &extent);
    if (offset < 0) {
      *out_axis = axis;
      *out_offset = offset;
      *out_extent = extent_is_static ? extent : 1;
      *out_bound = loom_type_dim_is_dynamic_at(access->view_type, axis)
                       ? -1
                       : loom_type_dim_static_size_at(access->view_type, axis);
      return true;
    }
    if (loom_type_dim_is_dynamic_at(access->view_type, axis)) {
      continue;
    }

    int64_t bound = loom_type_dim_static_size_at(access->view_type, axis);
    if (offset > bound) {
      *out_axis = axis;
      *out_offset = offset;
      *out_extent = 0;
      *out_bound = bound;
      return true;
    }

    if (!extent_is_static) continue;
    if (extent < 0 || extent > bound - offset) {
      *out_axis = axis;
      *out_offset = offset;
      *out_extent = extent;
      *out_bound = bound;
      return true;
    }
  }
  return false;
}

static bool loom_vector_type_has_index_or_non_i1_integer_element(
    loom_type_t type) {
  if (!loom_type_is_vector(type)) return false;
  loom_scalar_type_t element_type = loom_type_element_type(type);
  if (element_type == LOOM_SCALAR_TYPE_INDEX) return true;
  return element_type != LOOM_SCALAR_TYPE_I1 &&
         loom_scalar_type_is_integer(element_type);
}

static bool loom_vector_type_is_index_or_non_i1_integer_scalar(
    loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX) return true;
  return scalar_type != LOOM_SCALAR_TYPE_I1 &&
         loom_scalar_type_is_integer(scalar_type);
}

static iree_status_t loom_vector_verify_memory_access(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t vector_name, bool vector_is_result,
    loom_type_t view_type, loom_type_t vector_type,
    loom_attribute_t static_indices, uint16_t dynamic_index_count) {
  if (!loom_type_is_view(view_type) || !loom_type_is_vector(vector_type)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_verify_dynamic_index_count(
      emitter, op, static_indices, dynamic_index_count));

  uint8_t view_rank = loom_type_rank(view_type);
  if (static_indices.count != view_rank) {
    return loom_vector_emit_offset_count_mismatch(
        emitter, op, IREE_SV("view"), static_indices.count, view_rank);
  }

  uint8_t vector_rank = loom_type_rank(vector_type);
  if (vector_rank > view_rank) {
    return loom_vector_emit_field_constraint(
        emitter, op, vector_is_result, vector_name, vector_type,
        IREE_SV("rank no greater than view rank"));
  }

  loom_vector_memory_access_t access;
  if (!loom_vector_memory_access_describe(module, view_type, vector_type,
                                          &access)) {
    return iree_ok_status();
  }

  uint16_t out_of_bounds_axis = 0;
  int64_t out_of_bounds_offset = 0;
  int64_t out_of_bounds_extent = 0;
  int64_t out_of_bounds_bound = 0;
  if (loom_vector_find_static_memory_access_out_of_bounds(
          &access, static_indices, &out_of_bounds_axis, &out_of_bounds_offset,
          &out_of_bounds_extent, &out_of_bounds_bound)) {
    return loom_vector_emit_static_access_out_of_bounds(
        emitter, op, out_of_bounds_axis, out_of_bounds_offset,
        out_of_bounds_extent, out_of_bounds_bound);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_verify_gather_scatter_access(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_type_t view_type, loom_type_t offsets_type,
    loom_attribute_t static_indices, uint16_t dynamic_index_count) {
  if (!loom_type_is_view(view_type) || !loom_type_is_vector(offsets_type)) {
    return iree_ok_status();
  }

  if (!loom_vector_type_has_index_or_non_i1_integer_element(offsets_type)) {
    return loom_vector_emit_operand_constraint(
        emitter, op, IREE_SV("offsets"), offsets_type,
        IREE_SV("vector with index or non-i1 integer elements"));
  }

  IREE_RETURN_IF_ERROR(loom_vector_verify_dynamic_index_count(
      emitter, op, static_indices, dynamic_index_count));

  uint8_t view_rank = loom_type_rank(view_type);
  if (static_indices.count != view_rank) {
    return loom_vector_emit_offset_count_mismatch(
        emitter, op, IREE_SV("view"), static_indices.count, view_rank);
  }

  uint16_t out_of_bounds_axis = 0;
  int64_t out_of_bounds_index = 0;
  int64_t out_of_bounds_bound = 0;
  if (loom_vector_find_static_index_out_of_bounds(
          static_indices, view_type, &out_of_bounds_axis, &out_of_bounds_index,
          &out_of_bounds_bound)) {
    return loom_vector_emit_static_index_out_of_bounds(
        emitter, op, out_of_bounds_axis, out_of_bounds_index,
        out_of_bounds_bound);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_verify_optional_cache_policy(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    uint16_t cache_scope_attr_index, uint16_t cache_temporal_attr_index,
    loom_cache_policy_access_t access) {
  loom_attribute_t cache_scope_attr = loom_op_attrs(op)[cache_scope_attr_index];
  loom_attribute_t cache_temporal_attr =
      loom_op_attrs(op)[cache_temporal_attr_index];
  bool has_cache_scope = !loom_attr_is_absent(cache_scope_attr);
  bool has_cache_temporal = !loom_attr_is_absent(cache_temporal_attr);
  if (!has_cache_scope && !has_cache_temporal) return iree_ok_status();
  if (!has_cache_scope) {
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("cache_scope"), 0,
        IREE_SV("present when cache_temporal is present"));
  }
  if (!has_cache_temporal) {
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("cache_temporal"), 0,
        IREE_SV("present when cache_scope is present"));
  }
  if (cache_scope_attr.kind != LOOM_ATTR_ENUM ||
      cache_temporal_attr.kind != LOOM_ATTR_ENUM) {
    return iree_ok_status();
  }

  uint8_t cache_scope = loom_attr_as_enum(cache_scope_attr);
  uint8_t cache_temporal = loom_attr_as_enum(cache_temporal_attr);
  loom_cache_policy_error_t error =
      loom_cache_policy_validate(cache_scope, cache_temporal, access);
  if (error == LOOM_CACHE_POLICY_ERROR_NONE) return iree_ok_status();
  iree_string_view_t attr_name = loom_cache_policy_error_attr_name(error);
  int64_t actual_value =
      iree_string_view_equal(attr_name, IREE_SV("cache_scope"))
          ? cache_scope
          : cache_temporal;
  return loom_vector_emit_attribute_value_constraint(
      emitter, op, attr_name, actual_value,
      loom_cache_policy_error_expected_constraint(error));
}

static iree_status_t loom_vector_verify_element_width_relation(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_value_id_t input_id,
    loom_value_id_t result_id, bool result_must_be_wider,
    iree_string_view_t expected_constraint) {
  loom_type_t input_type = loom_module_value_type(module, input_id);
  loom_type_t result_type = loom_module_value_type(module, result_id);
  int32_t input_width =
      loom_scalar_type_bitwidth(loom_type_element_type(input_type));
  int32_t result_width =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  if (input_width == 0 || result_width == 0) return iree_ok_status();
  if (result_must_be_wider && result_width > input_width) {
    return iree_ok_status();
  }
  if (!result_must_be_wider && result_width < input_width) {
    return iree_ok_status();
  }
  return loom_vector_emit_result_constraint(emitter, op, IREE_SV("result"),
                                            result_type, expected_constraint);
}

iree_status_t loom_vector_constant_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_constant_result(op));
  if (!loom_type_is_vector(result_type)) return iree_ok_status();

  loom_attribute_t value = loom_vector_constant_value(op);
  loom_attr_kind_t expected_kind = LOOM_ATTR_ANY;
  if (loom_attr_matches_scalar_type(value, loom_type_element_type(result_type),
                                    &expected_kind)) {
    return iree_ok_status();
  }
  return loom_vector_emit_attribute_kind_mismatch(emitter, op, IREE_SV("value"),
                                                  (loom_attr_kind_t)value.kind,
                                                  expected_kind);
}

iree_status_t loom_vector_poison_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_poison_result(op));
  if (!loom_type_is_vector(result_type)) return iree_ok_status();
  if (!loom_type_has_static_zero_extent(result_type)) return iree_ok_status();
  return loom_vector_emit_result_constraint(
      emitter, op, IREE_SV("result"), result_type,
      IREE_SV("non-empty vector type; use vector.empty for static zero-lane "
              "values"));
}

iree_status_t loom_vector_empty_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_empty_result(op));
  if (!loom_type_is_vector(result_type)) return iree_ok_status();
  if (loom_type_has_static_zero_extent(result_type)) return iree_ok_status();
  return loom_vector_emit_result_constraint(
      emitter, op, IREE_SV("result"), result_type,
      IREE_SV("static zero-lane vector type"));
}

iree_status_t loom_vector_broadcast_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_broadcast_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_broadcast_result(op));
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(result_type);
  if (source_rank > result_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("source"),
                                          source_rank, IREE_SV("result"),
                                          result_rank);
  }

  uint8_t result_axis_offset = result_rank - source_rank;
  for (uint8_t source_axis = 0; source_axis < source_rank; ++source_axis) {
    uint8_t result_axis = result_axis_offset + source_axis;
    if (loom_type_dim_is_dynamic_at(source_type, source_axis) ||
        loom_type_dim_is_dynamic_at(result_type, result_axis)) {
      continue;
    }
    int64_t source_size =
        loom_type_dim_static_size_at(source_type, source_axis);
    int64_t result_size =
        loom_type_dim_static_size_at(result_type, result_axis);
    if (source_size == 1 || source_size == result_size) continue;
    loom_diagnostic_param_t params[] = {
        loom_param_u32(source_axis),
        loom_param_i64(source_size),
        loom_param_u32(result_axis),
        loom_param_i64(result_size),
    };
    return loom_vector_emit(emitter, op, &loom_err_shape_004, params,
                            IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

iree_status_t loom_vector_from_elements_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_from_elements_result(op));
  if (!loom_type_is_vector(result_type)) return iree_ok_status();
  if (!loom_type_satisfies_constraint(result_type,
                                      LOOM_TYPE_CONSTRAINT_ALL_STATIC_VECTOR)) {
    return iree_ok_status();
  }

  uint64_t expected_count = 0;
  bool element_count_is_static =
      loom_type_static_element_count(result_type, &expected_count);
  if (!element_count_is_static) {
    return loom_vector_emit_result_constraint(
        emitter, op, IREE_SV("result"), result_type,
        IREE_SV("representable static lane count"));
  }

  loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  if (elements.count == expected_count) return iree_ok_status();
  return loom_vector_emit_count_mismatch(
      emitter, op, IREE_SV("elements"), elements.count,
      IREE_SV("result element count"), expected_count);
}

iree_status_t loom_vector_iota_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  loom_type_t base_type =
      loom_module_value_type(module, loom_vector_iota_base(op));
  if (!loom_vector_type_is_index_or_non_i1_integer_scalar(base_type)) {
    return loom_vector_emit_operand_constraint(
        emitter, op, IREE_SV("base"), base_type,
        IREE_SV("index or non-i1 integer scalar"));
  }

  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_iota_result(op));
  if (loom_vector_type_has_index_or_non_i1_integer_element(result_type)) {
    return iree_ok_status();
  }
  return loom_vector_emit_result_constraint(
      emitter, op, IREE_SV("result"), result_type,
      IREE_SV("vector with index or non-i1 integer elements"));
}

iree_status_t loom_vector_mask_range_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  loom_type_t lower_bound_type =
      loom_module_value_type(module, loom_vector_mask_range_lower_bound(op));
  if (loom_vector_type_is_index_or_non_i1_integer_scalar(lower_bound_type)) {
    return iree_ok_status();
  }
  return loom_vector_emit_operand_constraint(
      emitter, op, IREE_SV("lower_bound"), lower_bound_type,
      IREE_SV("index or non-i1 integer scalar"));
}

iree_status_t loom_vector_load_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_load_view(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_load_result(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_memory_access(
      module, emitter, op, IREE_SV("result"), /*vector_is_result=*/true,
      view_type, result_type, loom_vector_load_static_indices(op),
      loom_vector_load_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_load_cache_scope_ATTR_INDEX,
      loom_vector_load_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_LOAD);
}

iree_status_t loom_vector_store_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_store_view(op));
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_store_value(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_memory_access(
      module, emitter, op, IREE_SV("value"), /*vector_is_result=*/false,
      view_type, value_type, loom_vector_store_static_indices(op),
      loom_vector_store_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_store_cache_scope_ATTR_INDEX,
      loom_vector_store_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_STORE);
}

iree_status_t loom_vector_load_mask_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_load_mask_view(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_load_mask_result(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_memory_access(
      module, emitter, op, IREE_SV("result"), /*vector_is_result=*/true,
      view_type, result_type, loom_vector_load_mask_static_indices(op),
      loom_vector_load_mask_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_load_mask_cache_scope_ATTR_INDEX,
      loom_vector_load_mask_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_LOAD);
}

iree_status_t loom_vector_store_mask_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_store_mask_view(op));
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_store_mask_value(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_memory_access(
      module, emitter, op, IREE_SV("value"), /*vector_is_result=*/false,
      view_type, value_type, loom_vector_store_mask_static_indices(op),
      loom_vector_store_mask_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_store_mask_cache_scope_ATTR_INDEX,
      loom_vector_store_mask_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_STORE);
}

iree_status_t loom_vector_load_expand_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_load_expand_view(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_load_expand_result(op));
  if (!loom_type_satisfies_constraint(result_type,
                                      LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_verify_memory_access(
      module, emitter, op, IREE_SV("result"), /*vector_is_result=*/true,
      view_type, result_type, loom_vector_load_expand_static_indices(op),
      loom_vector_load_expand_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_load_expand_cache_scope_ATTR_INDEX,
      loom_vector_load_expand_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_LOAD);
}

iree_status_t loom_vector_store_compress_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_store_compress_view(op));
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_store_compress_value(op));
  if (!loom_type_satisfies_constraint(value_type,
                                      LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_verify_memory_access(
      module, emitter, op, IREE_SV("value"), /*vector_is_result=*/false,
      view_type, value_type, loom_vector_store_compress_static_indices(op),
      loom_vector_store_compress_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_store_compress_cache_scope_ATTR_INDEX,
      loom_vector_store_compress_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_STORE);
}

iree_status_t loom_vector_gather_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_gather_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_gather_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_gather_static_indices(op),
      loom_vector_gather_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_gather_cache_scope_ATTR_INDEX,
      loom_vector_gather_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_LOAD);
}

iree_status_t loom_vector_scatter_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_scatter_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_scatter_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_scatter_static_indices(op),
      loom_vector_scatter_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_scatter_cache_scope_ATTR_INDEX,
      loom_vector_scatter_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_STORE);
}

iree_status_t loom_vector_gather_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_gather_mask_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_gather_mask_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_gather_mask_static_indices(op),
      loom_vector_gather_mask_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_gather_mask_cache_scope_ATTR_INDEX,
      loom_vector_gather_mask_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_LOAD);
}

iree_status_t loom_vector_scatter_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_scatter_mask_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_scatter_mask_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_scatter_mask_static_indices(op),
      loom_vector_scatter_mask_indices(op).count));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_scatter_mask_cache_scope_ATTR_INDEX,
      loom_vector_scatter_mask_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_STORE);
}

iree_status_t loom_vector_atomic_reduce_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_atomic_reduce_value(op));
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_atomic_reduce_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_atomic_reduce_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_atomic_reduce_static_indices(op),
      loom_vector_atomic_reduce_indices(op).count));
  IREE_RETURN_IF_ERROR(loom_vector_verify_atomic_kind(
      emitter, op, IREE_SV("value"), value_type,
      loom_vector_atomic_reduce_kind(op), /*allow_exchange=*/false));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_atomic_reduce_cache_scope_ATTR_INDEX,
      loom_vector_atomic_reduce_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_ATOMIC);
}

iree_status_t loom_vector_atomic_reduce_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_atomic_reduce_mask_value(op));
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_atomic_reduce_mask_view(op));
  loom_type_t offsets_type = loom_module_value_type(
      module, loom_vector_atomic_reduce_mask_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_atomic_reduce_mask_static_indices(op),
      loom_vector_atomic_reduce_mask_indices(op).count));
  IREE_RETURN_IF_ERROR(loom_vector_verify_atomic_kind(
      emitter, op, IREE_SV("value"), value_type,
      loom_vector_atomic_reduce_mask_kind(op), /*allow_exchange=*/false));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_atomic_reduce_mask_cache_scope_ATTR_INDEX,
      loom_vector_atomic_reduce_mask_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_ATOMIC);
}

iree_status_t loom_vector_atomic_rmw_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_atomic_rmw_value(op));
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_atomic_rmw_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_atomic_rmw_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_atomic_rmw_static_indices(op),
      loom_vector_atomic_rmw_indices(op).count));
  IREE_RETURN_IF_ERROR(loom_vector_verify_atomic_kind(
      emitter, op, IREE_SV("value"), value_type,
      loom_vector_atomic_rmw_kind(op), /*allow_exchange=*/true));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_atomic_rmw_cache_scope_ATTR_INDEX,
      loom_vector_atomic_rmw_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_ATOMIC);
}

iree_status_t loom_vector_atomic_rmw_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_atomic_rmw_mask_value(op));
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_atomic_rmw_mask_view(op));
  loom_type_t offsets_type =
      loom_module_value_type(module, loom_vector_atomic_rmw_mask_offsets(op));
  IREE_RETURN_IF_ERROR(loom_vector_verify_gather_scatter_access(
      emitter, op, view_type, offsets_type,
      loom_vector_atomic_rmw_mask_static_indices(op),
      loom_vector_atomic_rmw_mask_indices(op).count));
  IREE_RETURN_IF_ERROR(loom_vector_verify_atomic_kind(
      emitter, op, IREE_SV("value"), value_type,
      loom_vector_atomic_rmw_mask_kind(op), /*allow_exchange=*/true));
  return loom_vector_verify_optional_cache_policy(
      emitter, op, loom_vector_atomic_rmw_mask_cache_scope_ATTR_INDEX,
      loom_vector_atomic_rmw_mask_cache_temporal_ATTR_INDEX,
      LOOM_CACHE_POLICY_ACCESS_ATOMIC);
}

iree_status_t loom_vector_extract_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_extract_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_extract_result(op));
  if (!loom_type_is_vector(source_type)) return iree_ok_status();

  loom_attribute_t static_indices = loom_vector_extract_static_indices(op);
  uint16_t dynamic_index_count = loom_vector_extract_indices(op).count;
  uint16_t expected_dynamic_count =
      loom_vector_dynamic_sentinel_count(static_indices);
  if (dynamic_index_count != expected_dynamic_count) {
    return loom_vector_emit_count_mismatch(
        emitter, op, IREE_SV("indices"), dynamic_index_count,
        IREE_SV("dynamic sentinels"), expected_dynamic_count);
  }

  uint8_t source_rank = loom_type_rank(source_type);
  if (static_indices.count > source_rank) {
    return loom_vector_emit_count_mismatch(
        emitter, op, IREE_SV("static_indices"), static_indices.count,
        IREE_SV("source rank"), source_rank);
  }

  uint16_t out_of_bounds_axis = 0;
  int64_t out_of_bounds_index = 0;
  int64_t out_of_bounds_bound = 0;
  if (loom_vector_find_static_index_out_of_bounds(
          static_indices, source_type, &out_of_bounds_axis,
          &out_of_bounds_index, &out_of_bounds_bound)) {
    return loom_vector_emit_static_index_out_of_bounds(
        emitter, op, out_of_bounds_axis, out_of_bounds_index,
        out_of_bounds_bound);
  }
  return loom_vector_verify_subvalue_type(
      emitter, op, IREE_SV("result"), result_type, IREE_SV("source tail"),
      source_type, (uint8_t)static_indices.count, /*value_is_result=*/true);
}

iree_status_t loom_vector_insert_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_insert_value(op));
  loom_type_t dest_type =
      loom_module_value_type(module, loom_vector_insert_dest(op));
  if (!loom_type_is_vector(dest_type)) return iree_ok_status();

  loom_attribute_t static_indices = loom_vector_insert_static_indices(op);
  uint16_t dynamic_index_count = loom_vector_insert_indices(op).count;
  uint16_t expected_dynamic_count =
      loom_vector_dynamic_sentinel_count(static_indices);
  if (dynamic_index_count != expected_dynamic_count) {
    return loom_vector_emit_count_mismatch(
        emitter, op, IREE_SV("indices"), dynamic_index_count,
        IREE_SV("dynamic sentinels"), expected_dynamic_count);
  }

  uint8_t dest_rank = loom_type_rank(dest_type);
  if (static_indices.count > dest_rank) {
    return loom_vector_emit_count_mismatch(
        emitter, op, IREE_SV("static_indices"), static_indices.count,
        IREE_SV("dest rank"), dest_rank);
  }

  uint16_t out_of_bounds_axis = 0;
  int64_t out_of_bounds_index = 0;
  int64_t out_of_bounds_bound = 0;
  if (loom_vector_find_static_index_out_of_bounds(
          static_indices, dest_type, &out_of_bounds_axis, &out_of_bounds_index,
          &out_of_bounds_bound)) {
    return loom_vector_emit_static_index_out_of_bounds(
        emitter, op, out_of_bounds_axis, out_of_bounds_index,
        out_of_bounds_bound);
  }
  return loom_vector_verify_subvalue_type(
      emitter, op, IREE_SV("value"), value_type, IREE_SV("dest tail"),
      dest_type, (uint8_t)static_indices.count, /*value_is_result=*/false);
}

iree_status_t loom_vector_slice_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_slice_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_slice_result(op));
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(result_type);
  if (source_rank != result_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("result"),
                                          result_rank, IREE_SV("source"),
                                          source_rank);
  }

  loom_attribute_t static_offsets = loom_vector_slice_static_offsets(op);
  uint16_t dynamic_offset_count = loom_vector_slice_offsets(op).count;
  IREE_RETURN_IF_ERROR(loom_vector_verify_dynamic_sentinel_count(
      emitter, op, IREE_SV("offsets"), static_offsets, dynamic_offset_count));

  if (static_offsets.count != source_rank) {
    return loom_vector_emit_offset_count_mismatch(
        emitter, op, IREE_SV("source"), static_offsets.count, source_rank);
  }

  uint16_t out_of_bounds_axis = 0;
  int64_t out_of_bounds_offset = 0;
  int64_t out_of_bounds_extent = 0;
  int64_t out_of_bounds_bound = 0;
  if (loom_vector_find_static_slice_out_of_bounds(
          static_offsets, source_type, result_type, &out_of_bounds_axis,
          &out_of_bounds_offset, &out_of_bounds_extent, &out_of_bounds_bound)) {
    return loom_vector_emit_static_access_out_of_bounds(
        emitter, op, out_of_bounds_axis, out_of_bounds_offset,
        out_of_bounds_extent, out_of_bounds_bound);
  }
  return iree_ok_status();
}

iree_status_t loom_vector_concat_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_concat_result(op));
  if (!loom_type_is_vector(result_type)) return iree_ok_status();

  loom_value_slice_t inputs = loom_vector_concat_inputs(op);
  if (inputs.count == 0) {
    return loom_vector_emit_count_mismatch(emitter, op, IREE_SV("inputs"), 0,
                                           IREE_SV("minimum"), 1);
  }

  uint8_t result_rank = loom_type_rank(result_type);
  int64_t axis = loom_vector_concat_axis(op);
  if (axis < 0 || axis >= result_rank) {
    loom_diagnostic_param_t params[] = {
        loom_param_i64(axis),
        loom_param_i64(result_rank),
    };
    return loom_vector_emit(emitter, op, &loom_err_subrange_002, params,
                            IREE_ARRAYSIZE(params));
  }

  bool concat_axis_sum_is_static =
      !loom_type_dim_is_dynamic_at(result_type, (uint8_t)axis);
  int64_t concat_axis_sum = 0;
  for (uint16_t i = 0; i < inputs.count; ++i) {
    loom_type_t input_type = loom_module_value_type(module, inputs.values[i]);
    if (!loom_type_is_vector(input_type)) continue;

    uint8_t input_rank = loom_type_rank(input_type);
    if (input_rank != result_rank) {
      return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("inputs"),
                                            input_rank, IREE_SV("result"),
                                            result_rank);
    }

    for (uint8_t input_axis = 0; input_axis < input_rank; ++input_axis) {
      if (input_axis == (uint8_t)axis) continue;
      if (loom_vector_dim_equals(input_type, input_axis, result_type,
                                 input_axis)) {
        continue;
      }
      return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("inputs"),
                                             IREE_SV("result"));
    }

    if (loom_type_dim_is_dynamic_at(input_type, (uint8_t)axis)) {
      concat_axis_sum_is_static = false;
      continue;
    }
    if (!concat_axis_sum_is_static) continue;
    int64_t input_axis_size =
        loom_type_dim_static_size_at(input_type, (uint8_t)axis);
    if (!iree_checked_add_i64(concat_axis_sum, input_axis_size,
                              &concat_axis_sum)) {
      return loom_vector_emit_result_constraint(
          emitter, op, IREE_SV("result"), result_type,
          IREE_SV("representable static concat axis extent"));
    }
  }

  if (!concat_axis_sum_is_static) return iree_ok_status();
  int64_t result_axis_size =
      loom_type_dim_static_size_at(result_type, (uint8_t)axis);
  if (concat_axis_sum == result_axis_size) return iree_ok_status();
  return loom_vector_emit_result_constraint(
      emitter, op, IREE_SV("result"), result_type,
      IREE_SV("concat axis extent equal to sum of input extents"));
}

iree_status_t loom_vector_transpose_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_transpose_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_transpose_result(op));
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(result_type);
  if (source_rank != result_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("result"),
                                          result_rank, IREE_SV("source"),
                                          source_rank);
  }

  loom_attribute_t permutation = loom_vector_transpose_permutation(op);
  if (permutation.count != source_rank) {
    return loom_vector_emit_count_mismatch(emitter, op, IREE_SV("permutation"),
                                           permutation.count,
                                           IREE_SV("source rank"), source_rank);
  }

  uint32_t seen_axes = 0;
  for (uint16_t result_axis = 0; result_axis < permutation.count;
       ++result_axis) {
    int64_t source_axis = permutation.i64_array[result_axis];
    if (source_axis < 0 || source_axis >= source_rank) {
      loom_diagnostic_param_t params[] = {
          loom_param_i64(source_axis),
          loom_param_i64(source_rank),
      };
      return loom_vector_emit(emitter, op, &loom_err_subrange_002, params,
                              IREE_ARRAYSIZE(params));
    }

    uint32_t axis_bit = 1u << (uint32_t)source_axis;
    if (iree_all_bits_set(seen_axes, axis_bit)) {
      return loom_vector_emit_attribute_value_constraint(
          emitter, op, IREE_SV("permutation"), source_axis,
          IREE_SV("each source axis exactly once"));
    }
    seen_axes |= axis_bit;

    if (loom_vector_dim_equals(source_type, (uint8_t)source_axis, result_type,
                               (uint8_t)result_axis)) {
      continue;
    }
    return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("result"),
                                           IREE_SV("permutation(source)"));
  }
  return iree_ok_status();
}

iree_status_t loom_vector_shuffle_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_shuffle_source(op));
  if (!loom_type_is_vector(source_type)) return iree_ok_status();
  if (!loom_type_satisfies_constraint(
          source_type, LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR)) {
    return iree_ok_status();
  }

  int64_t source_lane_count = loom_type_dim_static_size_at(source_type, 0);
  loom_attribute_t source_lanes = loom_vector_shuffle_source_lanes(op);
  if (source_lanes.count != (uint64_t)source_lane_count) {
    return loom_vector_emit_count_mismatch(
        emitter, op, IREE_SV("source_lanes"), source_lanes.count,
        IREE_SV("source lane count"), (uint64_t)source_lane_count);
  }

  for (uint16_t result_lane = 0; result_lane < source_lanes.count;
       ++result_lane) {
    int64_t source_lane = source_lanes.i64_array[result_lane];
    if (source_lane >= 0 && source_lane < source_lane_count) continue;
    return loom_vector_emit_static_access_out_of_bounds(
        emitter, op, result_lane, source_lane, 1, source_lane_count);
  }
  return iree_ok_status();
}

iree_status_t loom_vector_interleave_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  loom_type_t even_type =
      loom_module_value_type(module, loom_vector_interleave_even(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_interleave_result(op));
  if (!loom_type_is_vector(even_type) || !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  uint8_t even_rank = loom_type_rank(even_type);
  uint8_t result_rank = loom_type_rank(result_type);
  if (result_rank != even_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("result"),
                                          result_rank, IREE_SV("even"),
                                          even_rank);
  }

  int64_t axis = loom_vector_interleave_axis(op);
  if (axis < 0 || axis >= even_rank) {
    loom_diagnostic_param_t params[] = {
        loom_param_i64(axis),
        loom_param_i64(even_rank),
    };
    return loom_vector_emit(emitter, op, &loom_err_subrange_002, params,
                            IREE_ARRAYSIZE(params));
  }

  for (uint8_t i = 0; i < even_rank; ++i) {
    if (i == (uint8_t)axis) continue;
    if (loom_vector_dim_equals(even_type, i, result_type, i)) continue;
    return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("result"),
                                           IREE_SV("even"));
  }

  if (loom_type_dim_is_dynamic_at(even_type, (uint8_t)axis) ||
      loom_type_dim_is_dynamic_at(result_type, (uint8_t)axis)) {
    return iree_ok_status();
  }

  int64_t expected_result_axis_size = 0;
  int64_t even_axis_size =
      loom_type_dim_static_size_at(even_type, (uint8_t)axis);
  if (!iree_checked_mul_i64(even_axis_size, 2, &expected_result_axis_size)) {
    return loom_vector_emit_result_constraint(
        emitter, op, IREE_SV("result"), result_type,
        IREE_SV("representable interleave axis extent"));
  }

  int64_t result_axis_size =
      loom_type_dim_static_size_at(result_type, (uint8_t)axis);
  if (result_axis_size == expected_result_axis_size) return iree_ok_status();
  return loom_vector_emit_result_constraint(
      emitter, op, IREE_SV("result"), result_type,
      IREE_SV("interleave axis extent twice input axis extent"));
}

iree_status_t loom_vector_deinterleave_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_deinterleave_source(op));
  loom_value_slice_t results = loom_vector_deinterleave_results(op);
  if (results.count != 2) {
    return loom_vector_emit_count_mismatch(emitter, op, IREE_SV("results"),
                                           results.count,
                                           IREE_SV("required result count"), 2);
  }
  loom_type_t even_type = loom_module_value_type(module, results.values[0]);
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(even_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t even_rank = loom_type_rank(even_type);
  if (even_rank != source_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("even"),
                                          even_rank, IREE_SV("source"),
                                          source_rank);
  }

  int64_t axis = loom_vector_deinterleave_axis(op);
  if (axis < 0 || axis >= source_rank) {
    loom_diagnostic_param_t params[] = {
        loom_param_i64(axis),
        loom_param_i64(source_rank),
    };
    return loom_vector_emit(emitter, op, &loom_err_subrange_002, params,
                            IREE_ARRAYSIZE(params));
  }

  for (uint8_t i = 0; i < source_rank; ++i) {
    if (i == (uint8_t)axis) continue;
    if (loom_vector_dim_equals(source_type, i, even_type, i)) continue;
    return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("even"),
                                           IREE_SV("source"));
  }

  bool source_axis_is_static =
      !loom_type_dim_is_dynamic_at(source_type, (uint8_t)axis);
  bool even_axis_is_static =
      !loom_type_dim_is_dynamic_at(even_type, (uint8_t)axis);

  int64_t source_axis_size =
      source_axis_is_static
          ? loom_type_dim_static_size_at(source_type, (uint8_t)axis)
          : 0;
  if (source_axis_is_static && (source_axis_size & 1) != 0) {
    return loom_vector_emit_operand_constraint(
        emitter, op, IREE_SV("source"), source_type,
        IREE_SV("even deinterleave axis extent"));
  }

  if (!even_axis_is_static) return iree_ok_status();
  int64_t expected_source_axis_size = 0;
  int64_t even_axis_size =
      loom_type_dim_static_size_at(even_type, (uint8_t)axis);
  if (!iree_checked_mul_i64(even_axis_size, 2, &expected_source_axis_size)) {
    return loom_vector_emit_result_constraint(
        emitter, op, IREE_SV("even"), even_type,
        IREE_SV("representable deinterleave source axis extent"));
  }

  if (!source_axis_is_static || source_axis_size == expected_source_axis_size) {
    return iree_ok_status();
  }
  return loom_vector_emit_result_constraint(
      emitter, op, IREE_SV("even"), even_type,
      IREE_SV("deinterleave axis extent half of source axis extent"));
}

static bool loom_vector_try_get_splat_i64_constant(const loom_module_t* module,
                                                   loom_value_id_t value_id,
                                                   int64_t* out_value) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op || !loom_vector_constant_isa(def_op)) return false;
  loom_attribute_t attr = loom_vector_constant_value(def_op);
  if (attr.kind != LOOM_ATTR_I64) return false;
  *out_value = loom_attr_as_i64(attr);
  return true;
}

static bool loom_vector_try_get_scalar_f64_constant(const loom_module_t* module,
                                                    loom_value_id_t value_id,
                                                    double* out_value) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op || !loom_scalar_constant_isa(def_op)) return false;
  loom_attribute_t attr = loom_scalar_constant_value(def_op);
  if (attr.kind != LOOM_ATTR_F64) return false;
  *out_value = loom_attr_as_f64(attr);
  return true;
}

static bool loom_vector_try_get_scalar_i64_constant(const loom_module_t* module,
                                                    loom_value_id_t value_id,
                                                    int64_t* out_value) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op) return false;
  if (!loom_scalar_constant_isa(def_op)) return false;
  loom_attribute_t attr = loom_scalar_constant_value(def_op);
  if (attr.kind != LOOM_ATTR_I64) return false;
  *out_value = loom_attr_as_i64(attr);
  return true;
}

static bool loom_vector_unsigned_code_capacity_covers(int32_t bitwidth,
                                                      int64_t max_code) {
  if (bitwidth <= 0 || max_code < 0) return false;
  if (bitwidth >= 63) return true;
  return (uint64_t)max_code < (UINT64_C(1) << bitwidth);
}

static iree_status_t loom_vector_verify_quantize_result_capacity(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    loom_type_t thresholds_type, loom_type_t result_type) {
  if (!loom_type_is_vector(thresholds_type) ||
      !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }
  if (loom_type_rank(thresholds_type) != 1 ||
      loom_type_dim_is_dynamic_at(thresholds_type, 0)) {
    return iree_ok_status();
  }

  loom_scalar_type_t result_element_type = loom_type_element_type(result_type);
  if (!loom_scalar_type_is_integer(result_element_type)) {
    return iree_ok_status();
  }

  int64_t threshold_count = loom_type_dim_static_size_at(thresholds_type, 0);
  int32_t result_bitwidth = loom_scalar_type_bitwidth(result_element_type);
  if (loom_vector_unsigned_code_capacity_covers(result_bitwidth,
                                                threshold_count)) {
    return iree_ok_status();
  }
  return loom_vector_emit_result_code_capacity(emitter, op, IREE_SV("result"),
                                               result_type, threshold_count);
}

static iree_status_t loom_vector_verify_static_threshold_order(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t thresholds_value) {
  const loom_value_t* value = loom_module_value(module, thresholds_value);
  if (loom_value_is_block_arg(value)) return iree_ok_status();
  const loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op) return iree_ok_status();

  if (loom_vector_constant_isa(def_op)) {
    loom_attribute_t attr = loom_vector_constant_value(def_op);
    if (attr.kind != LOOM_ATTR_F64 || !isnan(loom_attr_as_f64(attr))) {
      return iree_ok_status();
    }
    return loom_vector_emit_static_threshold_order(
        emitter, op, IREE_SV("thresholds"), /*left_index=*/0,
        /*right_index=*/0);
  }

  if (!loom_vector_from_elements_isa(def_op)) return iree_ok_status();

  loom_value_slice_t elements = loom_vector_from_elements_elements(def_op);
  double previous_value = 0.0;
  for (uint32_t i = 0; i < elements.count; ++i) {
    double value = 0.0;
    if (!loom_vector_try_get_scalar_f64_constant(module, elements.values[i],
                                                 &value)) {
      return iree_ok_status();
    }
    if (isnan(value)) {
      return loom_vector_emit_static_threshold_order(
          emitter, op, IREE_SV("thresholds"), i, i);
    }
    if (i > 0 && previous_value > value) {
      return loom_vector_emit_static_threshold_order(
          emitter, op, IREE_SV("thresholds"), i - 1, i);
    }
    previous_value = value;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_table_lookup_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_id_t table_value = loom_vector_table_lookup_table(op);
  loom_value_id_t indices_value = loom_vector_table_lookup_indices(op);
  loom_type_t table_type = loom_module_value_type(module, table_value);
  loom_type_t indices_type = loom_module_value_type(module, indices_value);
  if (!loom_type_is_vector(table_type) || !loom_type_is_vector(indices_type)) {
    return iree_ok_status();
  }
  if (!loom_type_satisfies_constraint(table_type,
                                      LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR)) {
    return iree_ok_status();
  }

  if (!loom_vector_type_has_index_or_non_i1_integer_element(indices_type)) {
    return loom_vector_emit_operand_constraint(
        emitter, op, IREE_SV("indices"), indices_type,
        IREE_SV("vector with index or non-i1 integer elements"));
  }
  if (loom_type_dim_is_dynamic_at(table_type, 0)) return iree_ok_status();

  int64_t table_lane_count = loom_type_dim_static_size_at(table_type, 0);
  int64_t splat_index = 0;
  if (!loom_vector_try_get_splat_i64_constant(module, indices_value,
                                              &splat_index)) {
    return iree_ok_status();
  }
  if (splat_index >= 0 && splat_index < table_lane_count) {
    return iree_ok_status();
  }
  return loom_vector_emit_static_index_out_of_bounds(
      emitter, op, /*axis=*/0, splat_index, table_lane_count);
}

iree_status_t loom_vector_table_quantize_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_id_t thresholds_value = loom_vector_table_quantize_thresholds(op);
  loom_type_t input_type =
      loom_module_value_type(module, loom_vector_table_quantize_input(op));
  loom_type_t thresholds_type =
      loom_module_value_type(module, thresholds_value);
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_table_quantize_result(op));
  if (!loom_type_is_vector(input_type) ||
      !loom_type_is_vector(thresholds_type) ||
      !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }
  if (!loom_type_satisfies_constraint(thresholds_type,
                                      LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_verify_quantize_result_capacity(
      emitter, op, thresholds_type, result_type));
  return loom_vector_verify_static_threshold_order(module, emitter, op,
                                                   thresholds_value);
}

typedef enum loom_vector_transform_family_e {
  LOOM_VECTOR_TRANSFORM_FAMILY_UNKNOWN = 0,
  LOOM_VECTOR_TRANSFORM_FAMILY_HADAMARD = 1,
  LOOM_VECTOR_TRANSFORM_FAMILY_HADAMARD_SIGN = 2,
  LOOM_VECTOR_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD = 3,
  LOOM_VECTOR_TRANSFORM_FAMILY_JL_DENSE = 4,
} loom_vector_transform_family_t;

static iree_string_view_t loom_vector_transform_family_param_name(void) {
  return IREE_SV("family");
}

static iree_string_view_t loom_vector_transform_input_elems_param_name(void) {
  return IREE_SV("input_elems");
}

static iree_string_view_t loom_vector_transform_output_elems_param_name(void) {
  return IREE_SV("output_elems");
}

static iree_string_view_t loom_vector_transform_signs_param_name(void) {
  return IREE_SV("signs");
}

static iree_string_view_t loom_vector_transform_permutation_param_name(void) {
  return IREE_SV("permutation");
}

static iree_string_view_t loom_vector_transform_matrix_param_name(void) {
  return IREE_SV("matrix");
}

static bool loom_vector_string_id_equal(const loom_module_t* module,
                                        loom_string_id_t string_id,
                                        iree_string_view_t expected) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[string_id], expected);
}

static const loom_named_attr_t* loom_vector_find_named_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    if (loom_vector_string_id_equal(module, entry->name_id, name)) {
      return entry;
    }
  }
  return NULL;
}

static bool loom_vector_encoding_dynamic_param_value(
    const loom_encoding_define_param_view_t* params,
    const loom_named_attr_t* name_entry, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (!name_entry || name_entry->value.kind != LOOM_ATTR_I64) return false;
  int64_t ordinal = name_entry->value.i64;
  if (ordinal < 0 || ordinal >= params->dynamic_values.count) return false;
  *out_value = params->dynamic_values.values[ordinal];
  return true;
}

static iree_status_t loom_vector_emit_encoding_dynamic_type_error(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t encoding_name,
    iree_string_view_t param_name, loom_value_id_t value_id,
    iree_string_view_t expected_type) {
  loom_type_t actual_type = loom_module_value_type(module, value_id);
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_type(actual_type),
      loom_param_string(expected_type),
  };
  return loom_vector_emit(emitter, op, &loom_err_encoding_009, params,
                          IREE_ARRAYSIZE(params));
}

static bool loom_vector_string_attr_value(const loom_module_t* module,
                                          loom_attribute_t attr,
                                          iree_string_view_t* out_value) {
  if (attr.kind != LOOM_ATTR_STRING ||
      attr.string_id == LOOM_STRING_ID_INVALID ||
      attr.string_id >= module->strings.count) {
    return false;
  }
  *out_value = module->strings.entries[attr.string_id];
  return true;
}

static loom_vector_transform_family_t loom_vector_transform_family_from_name(
    iree_string_view_t name) {
  if (iree_string_view_equal(name, IREE_SV("hadamard"))) {
    return LOOM_VECTOR_TRANSFORM_FAMILY_HADAMARD;
  }
  if (iree_string_view_equal(name, IREE_SV("hadamard_sign"))) {
    return LOOM_VECTOR_TRANSFORM_FAMILY_HADAMARD_SIGN;
  }
  if (iree_string_view_equal(name, IREE_SV("sign_permute_hadamard"))) {
    return LOOM_VECTOR_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD;
  }
  if (iree_string_view_equal(name, IREE_SV("jl_dense"))) {
    return LOOM_VECTOR_TRANSFORM_FAMILY_JL_DENSE;
  }
  return LOOM_VECTOR_TRANSFORM_FAMILY_UNKNOWN;
}

static bool loom_vector_transform_family_is_hadamard_like(
    loom_vector_transform_family_t family) {
  return family == LOOM_VECTOR_TRANSFORM_FAMILY_HADAMARD ||
         family == LOOM_VECTOR_TRANSFORM_FAMILY_HADAMARD_SIGN ||
         family == LOOM_VECTOR_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD;
}

static bool loom_vector_i64_is_power_of_two(int64_t value) {
  if (value <= 0) return false;
  return (value & (value - 1)) == 0;
}

static iree_status_t loom_vector_transform_verify_extent_param(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, const loom_encoding_define_param_view_t* params,
    iree_string_view_t param_name, iree_string_view_t field_name,
    loom_type_t field_type, uint8_t axis, bool field_is_result) {
  const loom_named_attr_t* static_param =
      loom_vector_find_named_attr(module, params->static_attrs, param_name);
  const loom_named_attr_t* dynamic_param =
      loom_vector_find_named_attr(module, params->dynamic_names, param_name);
  if (!static_param && !dynamic_param) return iree_ok_status();

  if (static_param) {
    if (static_param->value.kind != LOOM_ATTR_I64) {
      return iree_ok_status();
    }
    int64_t expected_extent = loom_attr_as_i64(static_param->value);
    if (expected_extent <= 0) {
      return iree_ok_status();
    }
    if (loom_type_dim_is_dynamic_at(field_type, axis)) {
      return loom_vector_emit_field_constraint(
          emitter, op, field_is_result, field_name, field_type,
          IREE_SV("last axis matching static transform extent"));
    }
    int64_t actual_extent = loom_type_dim_static_size_at(field_type, axis);
    if (actual_extent == expected_extent) return iree_ok_status();
    return loom_vector_emit_field_constraint(
        emitter, op, field_is_result, field_name, field_type,
        IREE_SV("last axis matching static transform extent"));
  }

  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_vector_encoding_dynamic_param_value(params, dynamic_param,
                                                &value_id)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "malformed encoding.define operand dictionary for parameter '%.*s'",
        (int)param_name.size, param_name.data);
  }
  if (!loom_type_equal(loom_module_value_type(module, value_id),
                       loom_type_scalar(LOOM_SCALAR_TYPE_INDEX))) {
    return iree_ok_status();
  }
  if (!loom_type_dim_is_dynamic_at(field_type, axis) ||
      loom_type_dim_value_id_at(field_type, axis) != value_id) {
    return loom_vector_emit_field_constraint(
        emitter, op, field_is_result, field_name, field_type,
        IREE_SV("last axis matching dynamic transform extent"));
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_transform_verify_permutation_lane(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint8_t source_axis,
    int64_t permutation_lane, int64_t source_lane_count) {
  if (permutation_lane >= 0 && permutation_lane < source_lane_count) {
    return iree_ok_status();
  }
  return loom_vector_emit_static_index_out_of_bounds(
      emitter, op, source_axis, permutation_lane, source_lane_count);
}

static iree_status_t loom_vector_transform_verify_static_permutation_values(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_type_t source_type,
    loom_value_id_t permutation_value) {
  uint8_t last_axis = (uint8_t)(loom_type_rank(source_type) - 1);
  if (loom_type_dim_is_dynamic_at(source_type, last_axis)) {
    return iree_ok_status();
  }
  int64_t source_lane_count =
      loom_type_dim_static_size_at(source_type, last_axis);
  const loom_value_t* value = loom_module_value(module, permutation_value);
  if (loom_value_is_block_arg(value)) return iree_ok_status();
  const loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op) return iree_ok_status();

  if (loom_vector_constant_isa(def_op)) {
    loom_attribute_t attr = loom_vector_constant_value(def_op);
    if (attr.kind != LOOM_ATTR_I64) return iree_ok_status();
    int64_t permutation_lane = loom_attr_as_i64(attr);
    IREE_RETURN_IF_ERROR(loom_vector_transform_verify_permutation_lane(
        emitter, op, last_axis, permutation_lane, source_lane_count));
    if (source_lane_count <= 1) return iree_ok_status();
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("permutation"), permutation_lane,
        IREE_SV("unique lane indices"));
  }

  if (!loom_vector_from_elements_isa(def_op)) return iree_ok_status();
  loom_value_slice_t elements = loom_vector_from_elements_elements(def_op);
  for (uint16_t i = 0; i < elements.count; ++i) {
    int64_t permutation_lane = 0;
    if (!loom_vector_try_get_scalar_i64_constant(module, elements.values[i],
                                                 &permutation_lane)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_vector_transform_verify_permutation_lane(
        emitter, op, last_axis, permutation_lane, source_lane_count));
    for (uint16_t j = 0; j < i; ++j) {
      int64_t previous_lane = 0;
      if (!loom_vector_try_get_scalar_i64_constant(module, elements.values[j],
                                                   &previous_lane)) {
        return iree_ok_status();
      }
      if (previous_lane != permutation_lane) continue;
      return loom_vector_emit_attribute_value_constraint(
          emitter, op, IREE_SV("permutation"), permutation_lane,
          IREE_SV("unique lane indices"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_transform_verify_dynamic_param_shapes(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, const loom_op_t* define_op,
    const loom_encoding_define_param_view_t* params,
    iree_string_view_t encoding_name, loom_type_t source_type,
    loom_type_t result_type) {
  const loom_named_attr_t* signs_param = loom_vector_find_named_attr(
      module, params->dynamic_names, loom_vector_transform_signs_param_name());
  if (signs_param) {
    loom_value_id_t signs_value = LOOM_VALUE_ID_INVALID;
    if (!loom_vector_encoding_dynamic_param_value(params, signs_param,
                                                  &signs_value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "malformed encoding.define operand dictionary for parameter 'signs'");
    }
    loom_type_t signs_type = loom_module_value_type(module, signs_value);
    if (loom_type_is_vector(signs_type) &&
        loom_type_element_type(signs_type) == LOOM_SCALAR_TYPE_I1 &&
        !loom_vector_shapes_match(signs_type, source_type)) {
      return loom_vector_emit_encoding_dynamic_type_error(
          module, emitter, define_op, encoding_name,
          loom_vector_transform_signs_param_name(), signs_value,
          IREE_SV("i1 vector with source shape"));
    }
  }

  const loom_named_attr_t* permutation_param = loom_vector_find_named_attr(
      module, params->dynamic_names,
      loom_vector_transform_permutation_param_name());
  if (permutation_param) {
    loom_value_id_t permutation_value = LOOM_VALUE_ID_INVALID;
    if (!loom_vector_encoding_dynamic_param_value(params, permutation_param,
                                                  &permutation_value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "malformed encoding.define operand dictionary for parameter "
          "'permutation'");
    }
    loom_type_t permutation_type =
        loom_module_value_type(module, permutation_value);
    if (loom_vector_type_has_index_or_non_i1_integer_element(
            permutation_type) &&
        !loom_vector_shapes_match(permutation_type, source_type)) {
      return loom_vector_emit_encoding_dynamic_type_error(
          module, emitter, define_op, encoding_name,
          loom_vector_transform_permutation_param_name(), permutation_value,
          IREE_SV("index or non-i1 integer vector with source shape"));
    }
    IREE_RETURN_IF_ERROR(loom_vector_transform_verify_static_permutation_values(
        module, emitter, op, source_type, permutation_value));
  }

  const loom_named_attr_t* matrix_param = loom_vector_find_named_attr(
      module, params->dynamic_names, loom_vector_transform_matrix_param_name());
  if (!matrix_param) return iree_ok_status();

  loom_value_id_t matrix_value = LOOM_VALUE_ID_INVALID;
  if (!loom_vector_encoding_dynamic_param_value(params, matrix_param,
                                                &matrix_value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "malformed encoding.define operand dictionary for parameter 'matrix'");
  }
  loom_type_t matrix_type = loom_module_value_type(module, matrix_value);
  if (!loom_type_is_vector(matrix_type) ||
      !loom_scalar_type_is_float(loom_type_element_type(matrix_type))) {
    return iree_ok_status();
  }

  const loom_named_attr_t* family_param = loom_vector_find_named_attr(
      module, params->static_attrs, loom_vector_transform_family_param_name());
  if (!family_param) return iree_ok_status();
  iree_string_view_t family_name = iree_string_view_empty();
  if (!loom_vector_string_attr_value(module, family_param->value,
                                     &family_name)) {
    return iree_ok_status();
  }
  if (loom_vector_transform_family_from_name(family_name) !=
      LOOM_VECTOR_TRANSFORM_FAMILY_JL_DENSE) {
    return iree_ok_status();
  }

  uint8_t matrix_rank = loom_type_rank(matrix_type);
  uint8_t source_last_axis = (uint8_t)(loom_type_rank(source_type) - 1);
  uint8_t result_last_axis = (uint8_t)(loom_type_rank(result_type) - 1);
  if (matrix_rank == 2 &&
      loom_vector_dim_equals(matrix_type, 0, result_type, result_last_axis) &&
      loom_vector_dim_equals(matrix_type, 1, source_type, source_last_axis)) {
    return iree_ok_status();
  }
  return loom_vector_emit_encoding_dynamic_type_error(
      module, emitter, define_op, encoding_name,
      loom_vector_transform_matrix_param_name(), matrix_value,
      IREE_SV("rank-2 floating-point matrix matching output x input extents"));
}

static iree_status_t loom_vector_transform_verify_family(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, loom_type_t source_type,
    loom_type_t result_type) {
  const loom_named_attr_t* family_param = loom_vector_find_named_attr(
      module, params->static_attrs, loom_vector_transform_family_param_name());
  if (!family_param) {
    return iree_ok_status();
  }

  iree_string_view_t family_name = iree_string_view_empty();
  if (!loom_vector_string_attr_value(module, family_param->value,
                                     &family_name)) {
    return iree_ok_status();
  }

  loom_vector_transform_family_t family =
      loom_vector_transform_family_from_name(family_name);
  if (family == LOOM_VECTOR_TRANSFORM_FAMILY_UNKNOWN) {
    return iree_ok_status();
  }

  if (!loom_vector_transform_family_is_hadamard_like(family)) {
    return iree_ok_status();
  }
  if (!loom_vector_shapes_match(source_type, result_type)) {
    return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("result"),
                                           IREE_SV("source"));
  }

  uint8_t last_axis = (uint8_t)(loom_type_rank(source_type) - 1);
  if (loom_type_dim_is_dynamic_at(source_type, last_axis)) {
    return iree_ok_status();
  }
  int64_t last_axis_size = loom_type_dim_static_size_at(source_type, last_axis);
  if (loom_vector_i64_is_power_of_two(last_axis_size)) return iree_ok_status();
  return loom_vector_emit_operand_constraint(
      emitter, op, IREE_SV("source"), source_type,
      IREE_SV("power-of-two last axis extent for Hadamard transform"));
}

iree_status_t loom_vector_transform_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  loom_value_id_t source_value = loom_vector_transform_source(op);
  loom_value_id_t transform_value = loom_vector_transform_transform(op);
  loom_value_id_t result_value = loom_vector_transform_result(op);
  loom_type_t source_type = loom_module_value_type(module, source_value);
  loom_type_t transform_type = loom_module_value_type(module, transform_value);
  loom_type_t result_type = loom_module_value_type(module, result_value);
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      !loom_type_is_encoding(transform_type)) {
    return iree_ok_status();
  }

  if (loom_type_encoding_role(transform_type) !=
      LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(result_type);
  if (source_rank == 0 || result_rank == 0) return iree_ok_status();
  if (source_rank != result_rank) {
    return loom_vector_emit_rank_mismatch(emitter, op, IREE_SV("result"),
                                          result_rank, IREE_SV("source"),
                                          source_rank);
  }

  uint8_t last_axis = (uint8_t)(source_rank - 1);
  for (uint8_t axis = 0; axis < last_axis; ++axis) {
    if (loom_vector_dim_equals(source_type, axis, result_type, axis)) continue;
    return loom_vector_emit_shape_mismatch(emitter, op, IREE_SV("result"),
                                           IREE_SV("source leading axes"));
  }

  const loom_value_t* transform = loom_module_value(module, transform_value);
  if (loom_value_is_block_arg(transform)) return iree_ok_status();
  const loom_op_t* define_op = loom_value_def_op(transform);
  if (!define_op || !loom_encoding_define_isa(define_op)) {
    return iree_ok_status();
  }

  loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(module, define_op);
  if (!params.spec || params.spec->name_id == LOOM_STRING_ID_INVALID ||
      params.spec->name_id >= module->strings.count) {
    return iree_ok_status();
  }
  iree_string_view_t encoding_name =
      module->strings.entries[params.spec->name_id];

  IREE_RETURN_IF_ERROR(loom_vector_transform_verify_extent_param(
      module, emitter, op, &params,
      loom_vector_transform_input_elems_param_name(), IREE_SV("source"),
      source_type, last_axis, /*field_is_result=*/false));
  IREE_RETURN_IF_ERROR(loom_vector_transform_verify_extent_param(
      module, emitter, op, &params,
      loom_vector_transform_output_elems_param_name(), IREE_SV("result"),
      result_type, last_axis, /*field_is_result=*/true));
  IREE_RETURN_IF_ERROR(loom_vector_transform_verify_dynamic_param_shapes(
      module, emitter, op, define_op, &params, encoding_name, source_type,
      result_type));

  return loom_vector_transform_verify_family(module, op, &params, emitter,
                                             source_type, result_type);
}

iree_status_t loom_vector_extf_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  return loom_vector_verify_element_width_relation(
      module, op, emitter, loom_vector_extf_input(op),
      loom_vector_extf_result(op), /*result_must_be_wider=*/true,
      IREE_SV("wider floating-point element type"));
}

iree_status_t loom_vector_fptrunc_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  return loom_vector_verify_element_width_relation(
      module, op, emitter, loom_vector_fptrunc_input(op),
      loom_vector_fptrunc_result(op), /*result_must_be_wider=*/false,
      IREE_SV("narrower floating-point element type"));
}

iree_status_t loom_vector_extsi_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  return loom_vector_verify_element_width_relation(
      module, op, emitter, loom_vector_extsi_input(op),
      loom_vector_extsi_result(op), /*result_must_be_wider=*/true,
      IREE_SV("wider integer element type"));
}

iree_status_t loom_vector_extui_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  return loom_vector_verify_element_width_relation(
      module, op, emitter, loom_vector_extui_input(op),
      loom_vector_extui_result(op), /*result_must_be_wider=*/true,
      IREE_SV("wider integer element type"));
}

iree_status_t loom_vector_trunci_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  return loom_vector_verify_element_width_relation(
      module, op, emitter, loom_vector_trunci_input(op),
      loom_vector_trunci_result(op), /*result_must_be_wider=*/false,
      IREE_SV("narrower integer element type"));
}

typedef struct loom_vector_total_bit_count_expr_t {
  // Product of all static dimensions and the element bit width.
  uint64_t static_factor;
  // Sorted dynamic dimension value IDs participating in the product.
  loom_value_id_t dynamic_dims[LOOM_TYPE_MAX_RANK];
  // Number of entries used in dynamic_dims.
  uint8_t dynamic_dim_count;
} loom_vector_total_bit_count_expr_t;

static void loom_vector_sort_dynamic_dims(loom_value_id_t* dims,
                                          uint8_t dim_count) {
  for (uint8_t i = 1; i < dim_count; ++i) {
    loom_value_id_t value = dims[i];
    uint8_t j = i;
    while (j > 0 && dims[j - 1] > value) {
      dims[j] = dims[j - 1];
      --j;
    }
    dims[j] = value;
  }
}

static bool loom_vector_total_bit_count_expr(
    loom_type_t type, int32_t element_width,
    loom_vector_total_bit_count_expr_t* out_expr) {
  *out_expr = (loom_vector_total_bit_count_expr_t){
      .static_factor = (uint64_t)element_width,
  };
  if (element_width <= 0 || !loom_type_is_vector(type)) return false;

  uint8_t rank = loom_type_rank(type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(type, i)) {
      if (out_expr->dynamic_dim_count >=
          IREE_ARRAYSIZE(out_expr->dynamic_dims)) {
        return false;
      }
      out_expr->dynamic_dims[out_expr->dynamic_dim_count++] =
          loom_type_dim_value_id_at(type, i);
      continue;
    }

    int64_t dimension_size = loom_type_dim_static_size_at(type, i);
    if (dimension_size < 0) return false;
    if (dimension_size == 0) {
      out_expr->static_factor = 0;
      out_expr->dynamic_dim_count = 0;
      return true;
    }
    uint64_t dimension_size_u64 = (uint64_t)dimension_size;
    if (out_expr->static_factor > UINT64_MAX / dimension_size_u64) {
      return false;
    }
    out_expr->static_factor *= dimension_size_u64;
  }

  loom_vector_sort_dynamic_dims(out_expr->dynamic_dims,
                                out_expr->dynamic_dim_count);
  return true;
}

static bool loom_vector_total_bit_count_expr_equal(
    const loom_vector_total_bit_count_expr_t* lhs,
    const loom_vector_total_bit_count_expr_t* rhs) {
  if (lhs->static_factor != rhs->static_factor ||
      lhs->dynamic_dim_count != rhs->dynamic_dim_count) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->dynamic_dim_count; ++i) {
    if (lhs->dynamic_dims[i] != rhs->dynamic_dims[i]) return false;
  }
  return true;
}

iree_status_t loom_vector_bitcast_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  loom_type_t input_type =
      loom_module_value_type(module, loom_vector_bitcast_input(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_bitcast_result(op));
  int32_t input_width =
      loom_scalar_type_bitwidth(loom_type_element_type(input_type));
  int32_t result_width =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  if (input_width == 0 || result_width == 0) return iree_ok_status();

  if (input_width == result_width &&
      loom_vector_shapes_match(input_type, result_type)) {
    return iree_ok_status();
  }

  loom_vector_total_bit_count_expr_t input_bit_count = {0};
  loom_vector_total_bit_count_expr_t result_bit_count = {0};
  if (loom_vector_total_bit_count_expr(input_type, input_width,
                                       &input_bit_count) &&
      loom_vector_total_bit_count_expr(result_type, result_width,
                                       &result_bit_count) &&
      loom_vector_total_bit_count_expr_equal(&input_bit_count,
                                             &result_bit_count)) {
    return iree_ok_status();
  }

  return loom_vector_emit_result_constraint(
      emitter, op, IREE_SV("result"), result_type,
      IREE_SV("provably same total bit count as input"));
}

static iree_status_t loom_vector_verify_bitfield_range(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, int64_t offset,
    int64_t width, int32_t storage_width) {
  if (offset < 0) {
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("offset"), offset,
        IREE_SV("non-negative bit offset"));
  }
  if (width <= 0) {
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("width"), width,
        IREE_SV("positive bitfield width"));
  }
  if (storage_width <= 0) return iree_ok_status();
  if (offset > storage_width || width > storage_width - offset) {
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("width"), width,
        IREE_SV("bitfield range within storage element width"));
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_verify_bitfield_extract(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_value_id_t source_value,
    loom_value_id_t result_value, int64_t offset, int64_t width) {
  loom_type_t source_type = loom_module_value_type(module, source_value);
  loom_type_t result_type = loom_module_value_type(module, result_value);
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  loom_scalar_type_t source_element_type = loom_type_element_type(source_type);
  loom_scalar_type_t result_element_type = loom_type_element_type(result_type);
  if (!loom_scalar_type_is_integer(source_element_type) ||
      !loom_scalar_type_is_integer(result_element_type)) {
    return iree_ok_status();
  }

  int32_t source_width = loom_scalar_type_bitwidth(source_element_type);
  IREE_RETURN_IF_ERROR(loom_vector_verify_bitfield_range(emitter, op, offset,
                                                         width, source_width));

  int32_t result_width = loom_scalar_type_bitwidth(result_element_type);
  if (result_width > 0 && width > result_width) {
    return loom_vector_emit_result_constraint(
        emitter, op, IREE_SV("result"), result_type,
        IREE_SV("integer element type at least bitfield width"));
  }
  return iree_ok_status();
}

iree_status_t loom_vector_bitfield_extractu_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_vector_verify_bitfield_extract(
      module, op, emitter, loom_vector_bitfield_extractu_source(op),
      loom_vector_bitfield_extractu_result(op),
      loom_vector_bitfield_extractu_offset(op),
      loom_vector_bitfield_extractu_width(op));
}

iree_status_t loom_vector_bitfield_extracts_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_vector_verify_bitfield_extract(
      module, op, emitter, loom_vector_bitfield_extracts_source(op),
      loom_vector_bitfield_extracts_result(op),
      loom_vector_bitfield_extracts_offset(op),
      loom_vector_bitfield_extracts_width(op));
}

iree_status_t loom_vector_bitfield_insert_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t field_type =
      loom_module_value_type(module, loom_vector_bitfield_insert_field(op));
  loom_type_t base_type =
      loom_module_value_type(module, loom_vector_bitfield_insert_base(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_bitfield_insert_result(op));
  if (!loom_type_is_vector(field_type) || !loom_type_is_vector(base_type) ||
      !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  loom_scalar_type_t field_element_type = loom_type_element_type(field_type);
  loom_scalar_type_t base_element_type = loom_type_element_type(base_type);
  if (!loom_scalar_type_is_integer(field_element_type) ||
      !loom_scalar_type_is_integer(base_element_type)) {
    return iree_ok_status();
  }

  int64_t offset = loom_vector_bitfield_insert_offset(op);
  int64_t width = loom_vector_bitfield_insert_width(op);
  int32_t base_width = loom_scalar_type_bitwidth(base_element_type);
  IREE_RETURN_IF_ERROR(loom_vector_verify_bitfield_range(emitter, op, offset,
                                                         width, base_width));

  int32_t field_width = loom_scalar_type_bitwidth(field_element_type);
  if (field_width > 0 && width > field_width) {
    return loom_vector_emit_operand_constraint(
        emitter, op, IREE_SV("field"), field_type,
        IREE_SV("integer element type at least bitfield width"));
  }
  return iree_ok_status();
}

static bool loom_vector_integer_element_width_is_less_than(loom_type_t type,
                                                           int64_t width) {
  if (!loom_type_is_vector(type)) return false;

  loom_scalar_type_t element_type = loom_type_element_type(type);
  if (!loom_scalar_type_is_integer(element_type)) return false;

  int32_t element_width = loom_scalar_type_bitwidth(element_type);
  return element_width > 0 && width > element_width;
}

static bool loom_vector_static_bit_count(loom_type_t type,
                                         int64_t bit_width_per_element,
                                         bool* out_is_static,
                                         uint64_t* out_bit_count) {
  *out_is_static = false;
  *out_bit_count = 0;
  if (bit_width_per_element < 0) return false;

  uint64_t element_count = 0;
  bool element_count_is_static =
      loom_type_static_element_count(type, &element_count);
  if (!element_count_is_static) return true;

  *out_is_static = true;
  if (bit_width_per_element == 0) return true;
  uint64_t bit_width = (uint64_t)bit_width_per_element;
  if (element_count > UINT64_MAX / bit_width) return false;
  *out_bit_count = element_count * bit_width;
  return true;
}

static iree_status_t loom_vector_verify_static_bit_count_relation(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t payload_name, loom_type_t payload_type,
    int64_t payload_bit_width, bool payload_is_result,
    iree_string_view_t storage_name, loom_type_t storage_type,
    int64_t storage_bit_width, bool storage_is_result,
    loom_type_t mismatch_result_type, iree_string_view_t expected_constraint) {
  bool payload_bit_count_is_static = false;
  uint64_t payload_bit_count = 0;
  if (!loom_vector_static_bit_count(payload_type, payload_bit_width,
                                    &payload_bit_count_is_static,
                                    &payload_bit_count)) {
    return loom_vector_emit_field_constraint(
        emitter, op, payload_is_result, payload_name, payload_type,
        IREE_SV("representable static payload bit count"));
  }

  bool storage_bit_count_is_static = false;
  uint64_t storage_bit_count = 0;
  if (!loom_vector_static_bit_count(storage_type, storage_bit_width,
                                    &storage_bit_count_is_static,
                                    &storage_bit_count)) {
    return loom_vector_emit_field_constraint(
        emitter, op, storage_is_result, storage_name, storage_type,
        IREE_SV("representable static storage bit count"));
  }

  if (!payload_bit_count_is_static || !storage_bit_count_is_static) {
    return iree_ok_status();
  }
  if (payload_bit_count == storage_bit_count) return iree_ok_status();
  return loom_vector_emit_result_constraint(emitter, op, IREE_SV("result"),
                                            mismatch_result_type,
                                            expected_constraint);
}

iree_status_t loom_vector_bitpack_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_bitpack_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_bitpack_result(op));
  int64_t width = loom_vector_bitpack_width(op);
  if (width <= 0) {
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("width"), width, IREE_SV("positive bit width"));
  }
  if (loom_vector_integer_element_width_is_less_than(source_type, width)) {
    return loom_vector_emit_operand_constraint(
        emitter, op, IREE_SV("source"), source_type,
        IREE_SV("integer element type at least bit width"));
  }

  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }
  loom_scalar_type_t result_element_type = loom_type_element_type(result_type);
  if (!loom_scalar_type_is_integer(result_element_type)) {
    return iree_ok_status();
  }

  int32_t result_element_width = loom_scalar_type_bitwidth(result_element_type);
  return loom_vector_verify_static_bit_count_relation(
      emitter, op, IREE_SV("source"), source_type, width,
      /*payload_is_result=*/false, IREE_SV("result"), result_type,
      result_element_width, /*storage_is_result=*/true, result_type,
      IREE_SV("packed payload bit count equal to result storage bit count"));
}

static iree_status_t loom_vector_verify_bitunpack(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_value_id_t source_value,
    loom_value_id_t result_value, int64_t width) {
  loom_type_t source_type = loom_module_value_type(module, source_value);
  loom_type_t result_type = loom_module_value_type(module, result_value);
  if (width <= 0) {
    return loom_vector_emit_attribute_value_constraint(
        emitter, op, IREE_SV("width"), width, IREE_SV("positive bit width"));
  }
  if (loom_vector_integer_element_width_is_less_than(result_type, width)) {
    return loom_vector_emit_result_constraint(
        emitter, op, IREE_SV("result"), result_type,
        IREE_SV("integer element type at least bit width"));
  }

  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }
  loom_scalar_type_t source_element_type = loom_type_element_type(source_type);
  if (!loom_scalar_type_is_integer(source_element_type)) {
    return iree_ok_status();
  }

  int32_t source_element_width = loom_scalar_type_bitwidth(source_element_type);
  return loom_vector_verify_static_bit_count_relation(
      emitter, op, IREE_SV("result"), result_type, width,
      /*payload_is_result=*/true, IREE_SV("source"), source_type,
      source_element_width, /*storage_is_result=*/false, result_type,
      IREE_SV("unpacked payload bit count equal to source storage bit count"));
}

iree_status_t loom_vector_bitunpacku_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  return loom_vector_verify_bitunpack(
      module, op, emitter, loom_vector_bitunpacku_source(op),
      loom_vector_bitunpacku_result(op), loom_vector_bitunpacku_width(op));
}

iree_status_t loom_vector_bitunpacks_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  return loom_vector_verify_bitunpack(
      module, op, emitter, loom_vector_bitunpacks_source(op),
      loom_vector_bitunpacks_result(op), loom_vector_bitunpacks_width(op));
}

iree_status_t loom_vector_dot2f_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_type_t lhs_type =
      loom_module_value_type(module, loom_vector_dot2f_lhs(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_dot2f_result(op));

  return loom_vector_verify_grouped_last_axis_shape(emitter, op, lhs_type,
                                                    result_type, 2);
}

iree_status_t loom_vector_dot4i_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_type_t lhs_type =
      loom_module_value_type(module, loom_vector_dot4i_lhs(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_dot4i_result(op));

  return loom_vector_verify_grouped_last_axis_shape(emitter, op, lhs_type,
                                                    result_type, 4);
}

iree_status_t loom_vector_reduce_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_type_t input_type =
      loom_module_value_type(module, loom_vector_reduce_input(op));
  if (!loom_type_is_vector(input_type)) return iree_ok_status();

  uint8_t kind = loom_vector_reduce_kind(op);
  loom_scalar_type_t element_type = loom_type_element_type(input_type);
  if (loom_scalar_type_is_integer(element_type) &&
      loom_vector_reduce_kind_accepts_integer(kind)) {
    return iree_ok_status();
  }
  if (loom_scalar_type_is_float(element_type) &&
      loom_vector_reduce_kind_accepts_float(kind)) {
    return iree_ok_status();
  }
  if (kind >= LOOM_VECTOR_REDUCE_KIND_COUNT_) return iree_ok_status();
  iree_string_view_t expected_constraint =
      loom_vector_reduce_kind_accepts_integer(kind)
          ? IREE_SV("integer element type for reduce kind")
          : IREE_SV("floating-point element type for reduce kind");
  return loom_vector_emit_operand_constraint(emitter, op, IREE_SV("input"),
                                             input_type, expected_constraint);
}
