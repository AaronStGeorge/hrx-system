// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/vector/to_scalar_mma.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"

typedef enum loom_vector_to_scalar_mma_numeric_e {
  LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED = 0,
  LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_FLOAT = 1,
  LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_SIGNED_INTEGER = 2,
} loom_vector_to_scalar_mma_numeric_t;

typedef struct loom_vector_to_scalar_mma_fragment_t {
  // Dense logical payload value used for scalar lane extraction.
  loom_value_id_t payload;

  // Fragment facts attached to value.
  loom_vector_fragment_fact_t fact;

  // Flat row-major dense logical payload type.
  loom_type_t type;

  // Exact logical row count.
  int64_t rows;

  // Exact logical column count.
  int64_t columns;
} loom_vector_to_scalar_mma_fragment_t;

static bool loom_vector_to_scalar_mma_exact_positive_i64(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value_id,
    int64_t* out_value) {
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  loom_value_facts_t facts =
      loom_rewriter_value_facts(state->rewriter, value_id);
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts) ||
      facts.range_lo <= 0) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_vector_to_scalar_mma_fragment_has_auxiliary(
    const loom_vector_fragment_fact_t* fact) {
  return iree_any_bit_set(fact->flags,
                          LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_SCHEMA) ||
         fact->auxiliary.present_keys != 0;
}

static bool loom_vector_to_scalar_mma_query_fragment(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_vector_fragment_role_flags_t role_flags,
    loom_vector_to_scalar_mma_fragment_t* out_fragment) {
  *out_fragment = (loom_vector_to_scalar_mma_fragment_t){0};
  loom_value_facts_t facts = loom_rewriter_value_facts(state->rewriter, value);
  loom_vector_fragment_fact_t fact;
  if (!loom_vector_fragment_fact_query_value_facts(
          &state->rewriter->fact_table->context, facts, &fact) ||
      !iree_any_bit_set(fact.role_flags, role_flags) || fact.shape_rank != 2 ||
      loom_vector_to_scalar_mma_fragment_has_auxiliary(&fact)) {
    return false;
  }

  int64_t rows = 0;
  int64_t columns = 0;
  if (!loom_vector_to_scalar_mma_exact_positive_i64(
          state, fact.shape_value_ids[0], &rows) ||
      !loom_vector_to_scalar_mma_exact_positive_i64(
          state, fact.shape_value_ids[1], &columns)) {
    return false;
  }

  loom_value_id_t payload = value;
  loom_op_t* def_op =
      loom_vector_to_scalar_value_def_op(state->rewriter->module, value);
  if (def_op != NULL && loom_vector_fragment_isa(def_op)) {
    payload = loom_vector_fragment_data(def_op);
  }

  *out_fragment = (loom_vector_to_scalar_mma_fragment_t){
      .payload = payload,
      .fact = fact,
      .type = loom_module_value_type(state->rewriter->module, payload),
      .rows = rows,
      .columns = columns,
  };
  return true;
}

static bool loom_vector_to_scalar_mma_logical_element_count(
    int64_t rows, int64_t columns, uint64_t* out_element_count) {
  if (rows <= 0 || columns <= 0) return false;
  uint64_t row_count = (uint64_t)rows;
  uint64_t column_count = (uint64_t)columns;
  if (row_count > UINT64_MAX / column_count) return false;
  *out_element_count = row_count * column_count;
  return true;
}

static bool loom_vector_to_scalar_mma_product_count(uint64_t lhs, uint64_t rhs,
                                                    uint64_t* out_product) {
  if (lhs > UINT64_MAX / rhs) return false;
  *out_product = lhs * rhs;
  return true;
}

static bool loom_vector_to_scalar_mma_type_is_dense_payload(loom_type_t type,
                                                            int64_t rows,
                                                            int64_t columns) {
  uint64_t logical_element_count = 0;
  uint64_t physical_element_count = 0;
  return loom_vector_to_scalar_mma_logical_element_count(
             rows, columns, &logical_element_count) &&
         loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_static_element_count(type, &physical_element_count) &&
         physical_element_count == logical_element_count;
}

static bool loom_vector_to_scalar_mma_numeric_kind(
    loom_scalar_type_t lhs_type, loom_scalar_type_t rhs_type,
    loom_scalar_type_t accumulator_type,
    loom_vector_to_scalar_mma_numeric_t* out_numeric) {
  *out_numeric = LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  if (loom_scalar_type_is_float(lhs_type) &&
      loom_scalar_type_is_float(rhs_type) &&
      loom_scalar_type_is_float(accumulator_type)) {
    *out_numeric = LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_FLOAT;
    return true;
  }
  if (loom_scalar_type_is_integer(lhs_type) &&
      loom_scalar_type_is_integer(rhs_type) &&
      loom_scalar_type_is_integer(accumulator_type)) {
    *out_numeric = LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_SIGNED_INTEGER;
    return true;
  }
  return false;
}

static bool loom_vector_to_scalar_mma_cast_is_supported(
    loom_scalar_type_t input_type, loom_scalar_type_t result_type,
    loom_vector_to_scalar_mma_numeric_t numeric) {
  if (input_type == result_type) {
    return true;
  }
  const int32_t input_bitwidth = loom_scalar_type_bitwidth(input_type);
  const int32_t result_bitwidth = loom_scalar_type_bitwidth(result_type);
  if (input_bitwidth <= 0 || result_bitwidth <= 0 ||
      input_bitwidth >= result_bitwidth) {
    return false;
  }
  switch (numeric) {
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_FLOAT:
      return loom_scalar_type_is_float(input_type) &&
             loom_scalar_type_is_float(result_type);
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_SIGNED_INTEGER:
      return loom_scalar_type_is_integer(input_type) &&
             loom_scalar_type_is_integer(result_type);
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED:
    default:
      return false;
  }
}

static bool loom_vector_to_scalar_mma_shapes_match(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init,
    loom_vector_to_scalar_mma_numeric_t* out_numeric) {
  *out_numeric = LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  loom_type_t result_type = loom_module_value_type(
      state->rewriter->module, loom_vector_mma_result(state->op));
  if (lhs->columns != rhs->rows || lhs->rows != init->rows ||
      rhs->columns != init->columns ||
      !loom_vector_to_scalar_mma_type_is_dense_payload(lhs->type, lhs->rows,
                                                       lhs->columns) ||
      !loom_vector_to_scalar_mma_type_is_dense_payload(rhs->type, rhs->rows,
                                                       rhs->columns) ||
      !loom_vector_to_scalar_mma_type_is_dense_payload(init->type, init->rows,
                                                       init->columns) ||
      !loom_vector_to_scalar_mma_type_is_dense_payload(result_type, init->rows,
                                                       init->columns) ||
      !loom_type_equal(init->type, result_type)) {
    return false;
  }

  const loom_scalar_type_t lhs_element_type = loom_type_element_type(lhs->type);
  const loom_scalar_type_t rhs_element_type = loom_type_element_type(rhs->type);
  const loom_scalar_type_t accumulator_element_type =
      loom_type_element_type(init->type);
  loom_vector_to_scalar_mma_numeric_t numeric =
      LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  if (!loom_vector_to_scalar_mma_numeric_kind(
          lhs_element_type, rhs_element_type, accumulator_element_type,
          &numeric)) {
    return false;
  }
  if (!loom_vector_to_scalar_mma_cast_is_supported(
          lhs_element_type, accumulator_element_type, numeric) ||
      !loom_vector_to_scalar_mma_cast_is_supported(
          rhs_element_type, accumulator_element_type, numeric)) {
    return false;
  }
  *out_numeric = numeric;
  return true;
}

static iree_status_t loom_vector_to_scalar_mma_cast_float(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input,
    loom_scalar_type_t input_scalar_type, loom_type_t result_type,
    loom_value_id_t* out_result) {
  loom_type_t input_type = loom_type_scalar(input_scalar_type);
  if (loom_type_equal(input_type, result_type)) {
    *out_result = input;
    return iree_ok_status();
  }
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_extf_build(&state->rewriter->builder, input,
                                              input_type, result_type,
                                              state->location, &cast_op));
  *out_result = loom_scalar_extf_result(cast_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_cast_integer(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input,
    loom_scalar_type_t input_scalar_type, loom_type_t result_type,
    loom_value_id_t* out_result) {
  return loom_vector_to_scalar_cast_integer_lane(
      state, input, loom_type_scalar(input_scalar_type), result_type,
      /*signed_extend=*/true, out_result);
}

static iree_status_t loom_vector_to_scalar_mma_build_accumulate(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric, loom_value_id_t lhs,
    loom_scalar_type_t lhs_scalar_type, loom_value_id_t rhs,
    loom_scalar_type_t rhs_scalar_type, loom_value_id_t accumulator,
    loom_type_t accumulator_type, loom_value_id_t* out_next) {
  loom_value_id_t cast_lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t cast_rhs = LOOM_VALUE_ID_INVALID;
  switch (numeric) {
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_FLOAT: {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_cast_float(
          state, lhs, lhs_scalar_type, accumulator_type, &cast_lhs));
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_cast_float(
          state, rhs, rhs_scalar_type, accumulator_type, &cast_rhs));
      loom_op_t* fma_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_fmaf_build(
          &state->rewriter->builder, 0, cast_lhs, cast_rhs, accumulator,
          accumulator_type, state->location, &fma_op));
      *out_next = loom_scalar_fmaf_result(fma_op);
      return iree_ok_status();
    }
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_SIGNED_INTEGER: {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_cast_integer(
          state, lhs, lhs_scalar_type, accumulator_type, &cast_lhs));
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_cast_integer(
          state, rhs, rhs_scalar_type, accumulator_type, &cast_rhs));
      loom_op_t* product_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_muli_build(
          &state->rewriter->builder, 0, cast_lhs, cast_rhs, accumulator_type,
          state->location, &product_op));
      loom_op_t* add_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_scalar_addi_build(&state->rewriter->builder, 0, accumulator,
                                 loom_scalar_muli_result(product_op),
                                 accumulator_type, state->location, &add_op));
      *out_next = loom_scalar_addi_result(add_op);
      return iree_ok_status();
    }
    case LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED:
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported vector.mma scalar numeric kind");
  }
}

static iree_status_t loom_vector_to_scalar_mma_extract_matrix_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* fragment, int64_t row,
    int64_t column, loom_value_id_t* out_lane) {
  int64_t indices[1] = {row * fragment->columns + column};
  return loom_vector_to_scalar_materialize_lane(
      state, fragment->payload,
      (loom_vector_to_scalar_index_list_t){
          .static_indices = indices,
          .rank = 1,
      },
      out_lane);
}

static iree_status_t loom_vector_to_scalar_mma_build_result_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init, int64_t row,
    int64_t column, loom_value_id_t* out_lane) {
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_extract_matrix_lane(
      state, init, row, column, &accumulator));

  loom_type_t accumulator_type =
      loom_type_scalar(loom_type_element_type(init->type));
  const loom_scalar_type_t lhs_scalar_type = loom_type_element_type(lhs->type);
  const loom_scalar_type_t rhs_scalar_type = loom_type_element_type(rhs->type);
  for (int64_t k = 0; k < lhs->columns; ++k) {
    loom_value_id_t lhs_lane = LOOM_VALUE_ID_INVALID;
    loom_value_id_t rhs_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_extract_matrix_lane(
        state, lhs, row, k, &lhs_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_extract_matrix_lane(
        state, rhs, k, column, &rhs_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_accumulate(
        state, numeric, lhs_lane, lhs_scalar_type, rhs_lane, rhs_scalar_type,
        accumulator, accumulator_type, &accumulator));
    if (state->pass->statistics) {
      loom_pass_statistic_add(state->pass,
                              LOOM_VECTOR_TO_SCALAR_STAT_LANES_MATERIALIZED, 1);
    }
  }
  *out_lane = accumulator;
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_lower_mma(
    loom_vector_to_scalar_state_t* state, bool* out_handled,
    loom_value_id_t* out_replacement) {
  *out_handled = false;
  *out_replacement = LOOM_VALUE_ID_INVALID;
  if (!loom_vector_mma_isa(state->op) || state->rewriter->fact_table == NULL) {
    return iree_ok_status();
  }

  loom_vector_to_scalar_mma_fragment_t lhs = {0};
  loom_vector_to_scalar_mma_fragment_t rhs = {0};
  loom_vector_to_scalar_mma_fragment_t init = {0};
  if (!loom_vector_to_scalar_mma_query_fragment(
          state, loom_vector_mma_lhs(state->op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS, &lhs) ||
      !loom_vector_to_scalar_mma_query_fragment(
          state, loom_vector_mma_rhs(state->op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS, &rhs) ||
      !loom_vector_to_scalar_mma_query_fragment(
          state, loom_vector_mma_init(state->op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
              LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT,
          &init)) {
    return iree_ok_status();
  }

  loom_vector_to_scalar_mma_numeric_t numeric =
      LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  if (!loom_vector_to_scalar_mma_shapes_match(state, &lhs, &rhs, &init,
                                              &numeric)) {
    return iree_ok_status();
  }

  uint64_t element_count = 0;
  if (!loom_vector_to_scalar_mma_logical_element_count(init.rows, init.columns,
                                                       &element_count) ||
      element_count == 0 || element_count > UINT16_MAX) {
    return iree_ok_status();
  }
  uint64_t product_count = 0;
  if (!loom_vector_to_scalar_mma_product_count(
          element_count, (uint64_t)lhs.columns, &product_count) ||
      product_count > UINT16_MAX) {
    return iree_ok_status();
  }

  loom_value_id_t* elements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->rewriter->arena, (iree_host_size_t)element_count,
      sizeof(*elements), (void**)&elements));
  uint16_t ordinal = 0;
  for (int64_t row = 0; row < init.rows; ++row) {
    for (int64_t column = 0; column < init.columns; ++column) {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_result_lane(
          state, numeric, &lhs, &rhs, &init, row, column, &elements[ordinal]));
      ++ordinal;
    }
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &state->rewriter->builder, elements, (iree_host_size_t)element_count,
      init.type, state->location, &from_elements_op));
  loom_op_t* fragment_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_fragment_build(
      &state->rewriter->builder, LOOM_VECTOR_ROLE_RESULT,
      loom_vector_from_elements_result(from_elements_op),
      init.fact.shape_value_ids[0], init.fact.shape_value_ids[1], NULL, 0, NULL,
      0, init.type, state->location, &fragment_op));
  *out_replacement = loom_vector_fragment_result(fragment_op);
  *out_handled = true;
  return iree_ok_status();
}
