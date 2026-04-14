// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/vector_to_scalar_internal.h"
#include "loom/util/math.h"

bool loom_vector_to_scalar_indices_are_dynamic(
    loom_vector_to_scalar_index_list_t indices) {
  return indices.dynamic_indices != NULL;
}

static iree_status_t loom_vector_to_scalar_copy_static_indices(
    loom_builder_t* builder, const int64_t* indices, uint8_t rank,
    int64_t** out_indices) {
  *out_indices = NULL;
  if (rank == 0) return iree_ok_status();
  int64_t* copied = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, rank, sizeof(int64_t), (void**)&copied));
  memcpy(copied, indices, (iree_host_size_t)rank * sizeof(int64_t));
  *out_indices = copied;
  return iree_ok_status();
}

loom_type_t loom_vector_to_scalar_lane_type(loom_type_t vector_type) {
  return loom_type_scalar(loom_type_element_type(vector_type));
}

static loom_attribute_t loom_vector_to_scalar_zero_attr(
    loom_scalar_type_t scalar_type) {
  if (scalar_type == LOOM_SCALAR_TYPE_I1) return loom_attr_bool(false);
  if (loom_scalar_type_is_float(scalar_type)) return loom_attr_f64(0.0);
  return loom_attr_i64(0);
}

iree_status_t loom_vector_to_scalar_build_scalar_constant(
    loom_builder_t* builder, loom_type_t result_type,
    loom_location_id_t location, int64_t integer_value,
    loom_value_id_t* out_value_id) {
  loom_scalar_type_t scalar_type = loom_type_element_type(result_type);
  loom_attribute_t attr = loom_scalar_type_is_float(scalar_type)
                              ? loom_attr_f64((double)integer_value)
                              : loom_attr_i64(integer_value);
  loom_op_t* constant_op = NULL;
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, attr, result_type,
                                                   location, &constant_op));
    *out_value_id = loom_index_constant_result(constant_op);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, attr, result_type,
                                                  location, &constant_op));
  *out_value_id = loom_scalar_constant_result(constant_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_attr_constant(
    loom_builder_t* builder, loom_type_t result_type,
    loom_location_id_t location, loom_attribute_t value,
    loom_value_id_t* out_value_id) {
  loom_scalar_type_t scalar_type = loom_type_element_type(result_type);
  loom_op_t* constant_op = NULL;
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, value, result_type,
                                                   location, &constant_op));
    *out_value_id = loom_index_constant_result(constant_op);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, value, result_type,
                                                  location, &constant_op));
  *out_value_id = loom_scalar_constant_result(constant_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_vector_zero(
    loom_vector_to_scalar_state_t* state, loom_type_t result_type,
    loom_value_id_t* out_value_id) {
  loom_scalar_type_t scalar_type = loom_type_element_type(result_type);
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      &state->rewriter->builder, loom_vector_to_scalar_zero_attr(scalar_type),
      result_type, state->location, &constant_op));
  *out_value_id = loom_vector_constant_result(constant_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_generic_scalar_op(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    uint8_t instance_flags, const loom_value_id_t* operands,
    uint16_t operand_count, const loom_attribute_t* attrs, uint8_t attr_count,
    loom_type_t result_type, loom_value_id_t* out_result) {
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(state->rewriter->module->context, kind);
  if (!vtable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no vtable registered for scalar op kind %u",
                            (unsigned)kind);
  }
  if (operand_count != vtable->fixed_operand_count ||
      attr_count != vtable->attribute_count ||
      vtable->fixed_result_count != 1 || vtable->region_count != 0 ||
      (vtable->vtable_flags & (LOOM_OP_VTABLE_VARIADIC_OPERANDS |
                               LOOM_OP_VTABLE_VARIADIC_RESULTS)) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "scalar op %.*s is not a fixed one-result lane op",
                            (int)loom_op_vtable_name(vtable).size,
                            loom_op_vtable_name(vtable).data);
  }

  loom_builder_t* builder = &state->rewriter->builder;
  loom_op_t* scalar_op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, kind, operand_count, 1,
                                                0, 0, attr_count,
                                                state->location, &scalar_op));
  scalar_op->instance_flags = instance_flags;
  loom_value_id_t* op_operands = loom_op_operands(scalar_op);
  for (uint16_t i = 0; i < operand_count; ++i) {
    op_operands[i] = operands[i];
  }
  IREE_RETURN_IF_ERROR(loom_builder_define_value(
      builder, result_type, &loom_op_results(scalar_op)[0]));
  loom_attribute_t* op_attrs = loom_op_attrs(scalar_op);
  for (uint8_t i = 0; i < attr_count; ++i) {
    op_attrs[i] = attrs[i];
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, scalar_op));
  *out_result = loom_op_results(scalar_op)[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_extract_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t vector_value,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t vector_type =
      loom_module_value_type(state->rewriter->module, vector_value);
  if (!loom_type_is_vector(vector_type)) {
    *out_lane = vector_value;
    return iree_ok_status();
  }
  loom_type_t lane_type = loom_vector_to_scalar_lane_type(vector_type);
  int64_t* static_indices = NULL;
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->rewriter->builder.arena, indices.rank,
                                  sizeof(int64_t), (void**)&static_indices));
    for (uint8_t i = 0; i < indices.rank; ++i) {
      static_indices[i] = INT64_MIN;
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_copy_static_indices(
        &state->rewriter->builder, indices.static_indices, indices.rank,
        &static_indices));
  }
  loom_op_t* extract_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &state->rewriter->builder, vector_value, indices.dynamic_indices,
      loom_vector_to_scalar_indices_are_dynamic(indices) ? indices.rank : 0,
      static_indices, indices.rank, lane_type, state->location, &extract_op));
  *out_lane = loom_vector_extract_result(extract_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_insert_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lane,
    loom_value_id_t aggregate, loom_type_t aggregate_type,
    loom_vector_to_scalar_index_list_t indices,
    loom_value_id_t* out_aggregate) {
  int64_t* static_indices = NULL;
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->rewriter->builder.arena, indices.rank,
                                  sizeof(int64_t), (void**)&static_indices));
    for (uint8_t i = 0; i < indices.rank; ++i) {
      static_indices[i] = INT64_MIN;
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_copy_static_indices(
        &state->rewriter->builder, indices.static_indices, indices.rank,
        &static_indices));
  }
  loom_op_t* insert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_insert_build(
      &state->rewriter->builder, lane, aggregate, indices.dynamic_indices,
      loom_vector_to_scalar_indices_are_dynamic(indices) ? indices.rank : 0,
      static_indices, indices.rank, aggregate_type, state->location,
      &insert_op));
  *out_aggregate = loom_vector_insert_result(insert_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_dim_bound(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type, uint8_t axis,
    loom_value_id_t* out_bound) {
  if (loom_type_dim_is_dynamic_at(vector_type, axis)) {
    *out_bound = loom_type_dim_value_id_at(vector_type, axis);
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, (int64_t)loom_type_dim_static_size_at(vector_type, axis),
      out_bound);
}

static iree_status_t loom_vector_to_scalar_build_index_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_result) {
  switch (kind) {
    case LOOM_OP_INDEX_ADD: {
      loom_op_t* op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          &state->rewriter->builder, lhs, rhs,
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->location, &op));
      *out_result = loom_index_add_result(op);
      return iree_ok_status();
    }
    case LOOM_OP_INDEX_SUB: {
      loom_op_t* op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_sub_build(
          &state->rewriter->builder, lhs, rhs,
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->location, &op));
      *out_result = loom_index_sub_result(op);
      return iree_ok_status();
    }
    case LOOM_OP_INDEX_MUL: {
      loom_op_t* op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_mul_build(
          &state->rewriter->builder, lhs, rhs,
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->location, &op));
      *out_result = loom_index_mul_result(op);
      return iree_ok_status();
    }
    case LOOM_OP_INDEX_DIV: {
      loom_op_t* op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_div_build(
          &state->rewriter->builder, lhs, rhs,
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->location, &op));
      *out_result = loom_index_div_result(op);
      return iree_ok_status();
    }
    case LOOM_OP_INDEX_REM: {
      loom_op_t* op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_rem_build(
          &state->rewriter->builder, lhs, rhs,
          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->location, &op));
      *out_result = loom_index_rem_result(op);
      return iree_ok_status();
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported index binary op kind %u",
                              (unsigned)kind);
  }
}

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_static_term(
    int64_t value) {
  return (loom_vector_to_scalar_index_term_t){
      .static_value = value,
  };
}

static loom_vector_to_scalar_index_term_t loom_vector_to_scalar_dynamic_term(
    loom_value_id_t value) {
  return (loom_vector_to_scalar_index_term_t){
      .dynamic_value = value,
      .is_dynamic = true,
  };
}

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_lane_term(
    loom_vector_to_scalar_index_list_t indices, uint8_t axis) {
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) {
    return loom_vector_to_scalar_dynamic_term(indices.dynamic_indices[axis]);
  }
  return loom_vector_to_scalar_static_term(indices.static_indices[axis]);
}

static bool loom_vector_to_scalar_term_is_static_value(
    loom_vector_to_scalar_index_term_t term, int64_t value) {
  return !term.is_dynamic && term.static_value == value;
}

iree_status_t loom_vector_to_scalar_term_value(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t term, loom_value_id_t* out_value) {
  if (term.is_dynamic) {
    *out_value = term.dynamic_value;
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, term.static_value, out_value);
}

iree_status_t loom_vector_to_scalar_build_term_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs,
    loom_vector_to_scalar_index_term_t* out_term) {
  if (!lhs.is_dynamic && !rhs.is_dynamic) {
    int64_t result = 0;
    switch (kind) {
      case LOOM_OP_INDEX_ADD:
        if (!loom_checked_add_i64(lhs.static_value, rhs.static_value,
                                  &result)) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "static index addition overflow");
        }
        *out_term = loom_vector_to_scalar_static_term(result);
        return iree_ok_status();
      case LOOM_OP_INDEX_SUB:
        if (!loom_checked_sub_i64(lhs.static_value, rhs.static_value,
                                  &result)) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "static index subtraction overflow");
        }
        *out_term = loom_vector_to_scalar_static_term(result);
        return iree_ok_status();
      case LOOM_OP_INDEX_MUL:
        if (!loom_checked_mul_i64(lhs.static_value, rhs.static_value,
                                  &result)) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "static index multiplication overflow");
        }
        *out_term = loom_vector_to_scalar_static_term(result);
        return iree_ok_status();
      case LOOM_OP_INDEX_DIV:
        if (lhs.static_value < 0 || rhs.static_value <= 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "static index division requires non-negative dividend and "
              "positive divisor");
        }
        *out_term = loom_vector_to_scalar_static_term(lhs.static_value /
                                                      rhs.static_value);
        return iree_ok_status();
      case LOOM_OP_INDEX_REM:
        if (lhs.static_value < 0 || rhs.static_value <= 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "static index remainder requires non-negative dividend and "
              "positive divisor");
        }
        *out_term = loom_vector_to_scalar_static_term(lhs.static_value %
                                                      rhs.static_value);
        return iree_ok_status();
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported static index term op kind %u",
                                (unsigned)kind);
    }
  }

  switch (kind) {
    case LOOM_OP_INDEX_ADD:
      if (loom_vector_to_scalar_term_is_static_value(lhs, 0)) {
        *out_term = rhs;
        return iree_ok_status();
      }
      if (loom_vector_to_scalar_term_is_static_value(rhs, 0)) {
        *out_term = lhs;
        return iree_ok_status();
      }
      break;
    case LOOM_OP_INDEX_SUB:
      if (loom_vector_to_scalar_term_is_static_value(rhs, 0)) {
        *out_term = lhs;
        return iree_ok_status();
      }
      break;
    case LOOM_OP_INDEX_MUL:
      if (loom_vector_to_scalar_term_is_static_value(lhs, 1)) {
        *out_term = rhs;
        return iree_ok_status();
      }
      if (loom_vector_to_scalar_term_is_static_value(rhs, 1)) {
        *out_term = lhs;
        return iree_ok_status();
      }
      break;
    case LOOM_OP_INDEX_DIV:
      if (loom_vector_to_scalar_term_is_static_value(rhs, 1)) {
        *out_term = lhs;
        return iree_ok_status();
      }
      break;
    default:
      break;
  }

  loom_value_id_t lhs_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_term_value(state, lhs, &lhs_value));
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_term_value(state, rhs, &rhs_value));
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_binary(
      state, kind, lhs_value, rhs_value, &result));
  *out_term = loom_vector_to_scalar_dynamic_term(result);
  return iree_ok_status();
}

bool loom_vector_to_scalar_terms_equal_static(
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs, bool* out_equal) {
  if (lhs.is_dynamic || rhs.is_dynamic) return false;
  *out_equal = lhs.static_value == rhs.static_value;
  return true;
}

static bool loom_vector_to_scalar_index_predicate_static_result(
    uint32_t predicate, int64_t lhs, int64_t rhs, bool* out_result) {
  switch (predicate) {
    case LOOM_INDEX_CMP_PREDICATE_EQ:
      *out_result = lhs == rhs;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SLT:
      *out_result = lhs < rhs;
      return true;
    default:
      return false;
  }
}

iree_status_t loom_vector_to_scalar_build_index_term_cmp(
    loom_vector_to_scalar_state_t* state, uint32_t predicate,
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs, loom_value_id_t* out_condition) {
  if (!lhs.is_dynamic && !rhs.is_dynamic) {
    bool result = false;
    if (loom_vector_to_scalar_index_predicate_static_result(
            predicate, lhs.static_value, rhs.static_value, &result)) {
      return loom_vector_to_scalar_build_scalar_constant(
          &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_I1),
          state->location, result ? 1 : 0, out_condition);
    }
  }

  loom_value_id_t lhs_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_term_value(state, lhs, &lhs_value));
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_term_value(state, rhs, &rhs_value));
  loom_op_t* cmp_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      &state->rewriter->builder, predicate, lhs_value, rhs_value,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), state->location, &cmp_op));
  *out_condition = loom_index_cmp_result(cmp_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_i1_and(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result) {
  loom_op_t* and_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
      &state->rewriter->builder, lhs, rhs,
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), state->location, &and_op));
  *out_result = loom_scalar_andi_result(and_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_scalar_select_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t condition,
    loom_value_id_t true_lane, loom_value_id_t false_lane,
    loom_value_id_t* out_lane) {
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_select_build(
      &state->rewriter->builder, condition, true_lane, false_lane,
      state->result_scalar_type, state->location, &select_op));
  *out_lane = loom_scalar_select_result(select_op);
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_terms_to_index_list(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_index_term_t* terms, uint8_t rank,
    loom_vector_to_scalar_index_list_t* out_indices) {
  bool has_dynamic = false;
  for (uint8_t i = 0; i < rank; ++i) {
    has_dynamic |= terms[i].is_dynamic;
  }
  if (!has_dynamic) {
    int64_t* static_indices = NULL;
    if (rank > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena,
                                                     rank, sizeof(int64_t),
                                                     (void**)&static_indices));
    }
    for (uint8_t i = 0; i < rank; ++i) {
      static_indices[i] = terms[i].static_value;
    }
    *out_indices = (loom_vector_to_scalar_index_list_t){
        .static_indices = static_indices,
        .rank = rank,
    };
    return iree_ok_status();
  }

  loom_value_id_t* dynamic_indices = NULL;
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena, rank,
                                                   sizeof(loom_value_id_t),
                                                   (void**)&dynamic_indices));
  }
  for (uint8_t i = 0; i < rank; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_term_value(state, terms[i], &dynamic_indices[i]));
  }
  *out_indices = (loom_vector_to_scalar_index_list_t){
      .dynamic_indices = dynamic_indices,
      .rank = rank,
  };
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_terms_from_explicit_indices(
    loom_vector_to_scalar_state_t* state, loom_attribute_t static_indices,
    loom_value_slice_t dynamic_indices,
    loom_vector_to_scalar_index_term_t** out_terms, uint8_t* out_count) {
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected i64 array index attribute");
  }
  if (static_indices.count > UINT8_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "index rank exceeds uint8_t range");
  }
  loom_vector_to_scalar_index_term_t* terms = NULL;
  if (static_indices.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, static_indices.count,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&terms));
  }
  uint16_t dynamic_ordinal = 0;
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    int64_t static_index = static_indices.i64_array[i];
    if (static_index == INT64_MIN) {
      if (dynamic_ordinal >= dynamic_indices.count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "dynamic index count does not match static sentinel count");
      }
      terms[i] = loom_vector_to_scalar_dynamic_term(
          loom_value_slice_get(dynamic_indices, dynamic_ordinal++));
    } else {
      terms[i] = loom_vector_to_scalar_static_term(static_index);
    }
  }
  if (dynamic_ordinal != dynamic_indices.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dynamic index count does not match static sentinel count");
  }
  *out_terms = terms;
  *out_count = (uint8_t)static_indices.count;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_linear_ordinal_dynamic(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_ordinal) {
  if (indices.rank == 1) {
    *out_ordinal = indices.dynamic_indices[0];
    return iree_ok_status();
  }

  loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &ordinal));
  for (uint8_t axis = 0; axis < indices.rank; ++axis) {
    if (axis > 0) {
      loom_value_id_t dim = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
          state, state->vector_type, axis, &dim));
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_binary(
          state, LOOM_OP_INDEX_MUL, ordinal, dim, &ordinal));
    }
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_binary(
        state, LOOM_OP_INDEX_ADD, ordinal, indices.dynamic_indices[axis],
        &ordinal));
  }
  *out_ordinal = ordinal;
  return iree_ok_status();
}

int64_t loom_vector_to_scalar_linear_ordinal_static(loom_type_t vector_type,
                                                    const int64_t* indices) {
  int64_t ordinal = 0;
  uint8_t rank = loom_type_rank(vector_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    if (axis > 0) {
      ordinal *= (int64_t)loom_type_dim_static_size_at(vector_type, axis);
    }
    ordinal += indices[axis];
  }
  return ordinal;
}

loom_vector_to_scalar_index_term_t loom_vector_to_scalar_dim_bound_term(
    loom_type_t vector_type, uint8_t axis) {
  if (loom_type_dim_is_dynamic_at(vector_type, axis)) {
    return loom_vector_to_scalar_dynamic_term(
        loom_type_dim_value_id_at(vector_type, axis));
  }
  return loom_vector_to_scalar_static_term(
      (int64_t)loom_type_dim_static_size_at(vector_type, axis));
}

iree_status_t loom_vector_to_scalar_linear_ordinal_term(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type,
    loom_vector_to_scalar_index_list_t indices,
    loom_vector_to_scalar_index_term_t* out_ordinal) {
  if (indices.rank != loom_type_rank(vector_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector lane index rank mismatch");
  }
  loom_vector_to_scalar_index_term_t ordinal =
      loom_vector_to_scalar_static_term(0);
  for (uint8_t axis = 0; axis < indices.rank; ++axis) {
    if (axis > 0) {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
          state, LOOM_OP_INDEX_MUL, ordinal,
          loom_vector_to_scalar_dim_bound_term(vector_type, axis), &ordinal));
    }
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_OP_INDEX_ADD, ordinal,
        loom_vector_to_scalar_lane_term(indices, axis), &ordinal));
  }
  *out_ordinal = ordinal;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_cast_index_to_scalar(
    loom_vector_to_scalar_state_t* state, loom_value_id_t index_value,
    loom_type_t result_type, loom_value_id_t* out_value) {
  if (loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_INDEX) {
    *out_value = index_value;
    return iree_ok_status();
  }
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_cast_build(&state->rewriter->builder, index_value,
                            loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                            result_type, state->location, &cast_op));
  *out_value = loom_index_cast_result(cast_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_ordinal_for_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_type_t result_type,
    loom_value_id_t* out_ordinal) {
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) {
    loom_value_id_t index_ordinal = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_linear_ordinal_dynamic(
        state, indices, &index_ordinal));
    return loom_vector_to_scalar_cast_index_to_scalar(state, index_ordinal,
                                                      result_type, out_ordinal);
  }
  int64_t ordinal = loom_vector_to_scalar_linear_ordinal_static(
      state->vector_type, indices.static_indices);
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, result_type, state->location, ordinal,
      out_ordinal);
}

static iree_status_t loom_vector_to_scalar_build_coordinate_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t integer_kind,
    loom_op_kind_t index_kind, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t type, loom_value_id_t* out_result) {
  if (loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX) {
    return loom_vector_to_scalar_build_index_binary(state, index_kind, lhs, rhs,
                                                    out_result);
  }
  return loom_vector_to_scalar_build_generic_scalar_op(
      state, integer_kind, 0, (loom_value_id_t[]){lhs, rhs}, 2, NULL, 0, type,
      out_result);
}

static iree_status_t loom_vector_to_scalar_build_iota_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t base = loom_vector_iota_base(state->op);
  loom_value_id_t step = loom_vector_iota_step(state->op);
  loom_type_t lane_type = state->result_scalar_type;
  loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_ordinal_for_lane(
      state, indices, lane_type, &ordinal));
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_coordinate_binary(
      state, LOOM_OP_SCALAR_MULI, LOOM_OP_INDEX_MUL, ordinal, step, lane_type,
      &scaled));
  return loom_vector_to_scalar_build_coordinate_binary(
      state, LOOM_OP_SCALAR_ADDI, LOOM_OP_INDEX_ADD, base, scaled, lane_type,
      out_lane);
}

static iree_status_t loom_vector_to_scalar_build_mask_range_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t lower_bound = loom_vector_mask_range_lower_bound(state->op);
  loom_value_id_t upper_bound = loom_vector_mask_range_upper_bound(state->op);
  loom_value_id_t step = loom_vector_mask_range_step(state->op);
  loom_type_t coordinate_type =
      loom_module_value_type(state->rewriter->module, lower_bound);

  loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_ordinal_for_lane(
      state, indices, coordinate_type, &ordinal));
  loom_value_id_t scaled = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_coordinate_binary(
      state, LOOM_OP_SCALAR_MULI, LOOM_OP_INDEX_MUL, ordinal, step,
      coordinate_type, &scaled));
  loom_value_id_t coordinate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_coordinate_binary(
      state, LOOM_OP_SCALAR_ADDI, LOOM_OP_INDEX_ADD, lower_bound, scaled,
      coordinate_type, &coordinate));

  if (loom_type_element_type(coordinate_type) == LOOM_SCALAR_TYPE_INDEX) {
    loom_op_t* cmp_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_cmp_build(
        &state->rewriter->builder, LOOM_INDEX_CMP_PREDICATE_SLT, coordinate,
        upper_bound, coordinate_type, loom_type_scalar(LOOM_SCALAR_TYPE_I1),
        state->location, &cmp_op));
    *out_lane = loom_index_cmp_result(cmp_op);
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_generic_scalar_op(
      state, LOOM_OP_SCALAR_CMPI, 0,
      (loom_value_id_t[]){coordinate, upper_bound}, 2,
      (loom_attribute_t[]){loom_attr_enum(LOOM_SCALAR_CMPI_PREDICATE_SLT)}, 1,
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), out_lane);
}

iree_status_t loom_vector_to_scalar_build_constant_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_lane) {
  return loom_vector_to_scalar_build_attr_constant(
      &state->rewriter->builder, state->result_scalar_type, state->location,
      loom_vector_constant_value(state->op), out_lane);
}

iree_status_t loom_vector_to_scalar_build_poison_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_lane) {
  loom_op_t* poison_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_poison_build(&state->rewriter->builder,
                                                state->result_scalar_type,
                                                state->location, &poison_op));
  *out_lane = loom_scalar_poison_result(poison_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_generic_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  const loom_vector_to_scalar_descriptor_t* descriptor = state->descriptor;
  if (descriptor->reject_instance_flags && state->op->instance_flags != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "vector-to-scalar cannot lower %.*s with instance flags because vector "
        "value-domain assumptions are not scalar fast-math flags",
        (int)loom_op_name(state->rewriter->module, state->op).size,
        loom_op_name(state->rewriter->module, state->op).data);
  }

  loom_value_id_t lane_operands[4] = {0};
  const loom_value_id_t* operands = loom_op_const_operands(state->op);
  for (uint8_t i = 0; i < descriptor->lane_operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, operands[i], indices, &lane_operands[i]));
  }

  uint8_t instance_flags =
      descriptor->forward_instance_flags ? state->op->instance_flags : 0;
  const loom_attribute_t* attrs = loom_op_attrs(state->op);
  return loom_vector_to_scalar_build_generic_scalar_op(
      state, descriptor->scalar_kind, instance_flags, lane_operands,
      descriptor->lane_operand_count, attrs, descriptor->copied_attr_count,
      state->result_scalar_type, out_lane);
}

iree_status_t loom_vector_to_scalar_build_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass,
                            LOOM_VECTOR_TO_SCALAR_STAT_LANES_MATERIALIZED, 1);
  }
  switch (state->descriptor->lane_kind) {
    case LOOM_VECTOR_TO_SCALAR_LANE_GENERIC:
      return loom_vector_to_scalar_build_generic_lane(state, indices, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_IOTA:
      return loom_vector_to_scalar_build_iota_lane(state, indices, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_MASK_RANGE:
      return loom_vector_to_scalar_build_mask_range_lane(state, indices,
                                                         out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_BROADCAST:
      return loom_vector_to_scalar_build_broadcast_lane(state, indices,
                                                        out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_EXTRACT:
      return loom_vector_to_scalar_build_extract_lane(state, indices, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_INSERT:
      return loom_vector_to_scalar_build_insert_lane(state, indices, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_SLICE:
      return loom_vector_to_scalar_build_slice_lane(state, indices, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_CONCAT:
      return loom_vector_to_scalar_build_concat_lane(state, indices, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_TRANSPOSE:
      return loom_vector_to_scalar_build_transpose_lane(state, indices,
                                                        out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_SHUFFLE:
      return loom_vector_to_scalar_build_shuffle_lane(state, indices, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_INTERLEAVE:
      return loom_vector_to_scalar_build_interleave_lane(state, indices,
                                                         out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_DEINTERLEAVE:
      return loom_vector_to_scalar_build_deinterleave_lane(state, indices,
                                                           out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_BITCAST:
      return loom_vector_to_scalar_build_bitcast_lane(state, indices, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_BITFIELD_EXTRACTU:
      return loom_vector_to_scalar_build_bitfield_extract_lane(
          state, indices, /*signed_extract=*/false, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_BITFIELD_EXTRACTS:
      return loom_vector_to_scalar_build_bitfield_extract_lane(
          state, indices, /*signed_extract=*/true, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_BITFIELD_INSERT:
      return loom_vector_to_scalar_build_bitfield_insert_lane(state, indices,
                                                              out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_DOT4I:
      return loom_vector_to_scalar_build_dot4i_lane(state, indices, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_BITPACK:
      return loom_vector_to_scalar_build_bitpack_lane(state, indices, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_BITUNPACKU:
      return loom_vector_to_scalar_build_bitunpack_lane(
          state, indices, /*signed_unpack=*/false, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_BITUNPACKS:
      return loom_vector_to_scalar_build_bitunpack_lane(
          state, indices, /*signed_unpack=*/true, out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_TABLE_LOOKUP:
      return loom_vector_to_scalar_build_table_lookup_lane(state, indices,
                                                           out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_TABLE_QUANTIZE:
      return loom_vector_to_scalar_build_table_quantize_lane(state, indices,
                                                             out_lane);
    case LOOM_VECTOR_TO_SCALAR_LANE_TRANSFORM:
      return loom_vector_to_scalar_build_transform_lane(state, indices,
                                                        out_lane);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown vector-to-scalar lane kind %u",
                              (unsigned)state->descriptor->lane_kind);
  }
}

static bool loom_vector_to_scalar_try_from_elements_lane(
    loom_vector_to_scalar_state_t* state, loom_op_t* def_op,
    loom_type_t vector_type, loom_vector_to_scalar_index_list_t indices,
    loom_value_id_t* out_lane) {
  if (!loom_vector_from_elements_isa(def_op)) return false;
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) return false;
  if (!loom_type_is_all_static(vector_type)) return false;
  int64_t ordinal = loom_vector_to_scalar_linear_ordinal_static(
      vector_type, indices.static_indices);
  if (ordinal < 0) return false;
  loom_value_slice_t elements = loom_vector_from_elements_elements(def_op);
  if ((uint64_t)ordinal >= elements.count) return false;
  (void)state;
  *out_lane = loom_value_slice_get(elements, (uint16_t)ordinal);
  return true;
}

static bool loom_vector_to_scalar_result_ordinal(loom_op_t* op,
                                                 loom_value_id_t value,
                                                 uint16_t* out_ordinal) {
  const loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == value) {
      *out_ordinal = i;
      return true;
    }
  }
  return false;
}

iree_status_t loom_vector_to_scalar_try_materialize_def_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_type_t vector_type, loom_vector_to_scalar_index_list_t indices,
    bool* out_materialized, loom_value_id_t* out_lane) {
  *out_materialized = false;
  loom_op_t* def_op =
      loom_value_def_op(loom_module_value(state->rewriter->module, value));
  if (!def_op) return iree_ok_status();
  if (loom_vector_to_scalar_try_from_elements_lane(state, def_op, vector_type,
                                                   indices, out_lane)) {
    *out_materialized = true;
    return iree_ok_status();
  }
  if (loom_vector_splat_isa(def_op)) {
    *out_lane = loom_vector_splat_scalar(def_op);
    *out_materialized = true;
    return iree_ok_status();
  }
  if (loom_vector_constant_isa(def_op)) {
    loom_vector_to_scalar_state_t def_state = {
        .pass = state->pass,
        .rewriter = state->rewriter,
        .op = def_op,
        .value_checkpoint = state->value_checkpoint,
        .vector_type = vector_type,
        .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
        .location = def_op->location,
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_build_constant_lane(&def_state, out_lane));
    *out_materialized = true;
    return iree_ok_status();
  }
  if (loom_vector_poison_isa(def_op)) {
    loom_vector_to_scalar_state_t def_state = {
        .pass = state->pass,
        .rewriter = state->rewriter,
        .op = def_op,
        .value_checkpoint = state->value_checkpoint,
        .vector_type = vector_type,
        .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
        .location = def_op->location,
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_build_poison_lane(&def_state, out_lane));
    *out_materialized = true;
    return iree_ok_status();
  }

  const loom_vector_to_scalar_descriptor_t* descriptor =
      loom_vector_to_scalar_find_descriptor(def_op->kind);
  if (!descriptor) return iree_ok_status();
  uint16_t result_ordinal = 0;
  if (!loom_vector_to_scalar_result_ordinal(def_op, value, &result_ordinal)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value is not a result of its defining op");
  }
  loom_type_t result_type = loom_module_value_type(
      state->rewriter->module, loom_op_results(def_op)[result_ordinal]);
  loom_vector_to_scalar_state_t def_state = {
      .pass = state->pass,
      .rewriter = state->rewriter,
      .op = def_op,
      .descriptor = descriptor,
      .value_checkpoint = state->value_checkpoint,
      .result_ordinal = result_ordinal,
      .vector_type = result_type,
      .result_scalar_type = descriptor->result_is_i1
                                ? loom_type_scalar(LOOM_SCALAR_TYPE_I1)
                                : loom_vector_to_scalar_lane_type(result_type),
      .location = def_op->location,
  };
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_lane(&def_state, indices, out_lane));
  *out_materialized = true;
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_materialize_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t type = loom_module_value_type(state->rewriter->module, value);
  if (!loom_type_is_vector(type)) {
    *out_lane = value;
    return iree_ok_status();
  }
  bool materialized = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_try_materialize_def_lane(
      state, value, type, indices, &materialized, out_lane));
  if (materialized) return iree_ok_status();
  return loom_vector_to_scalar_extract_lane(state, value, indices, out_lane);
}
