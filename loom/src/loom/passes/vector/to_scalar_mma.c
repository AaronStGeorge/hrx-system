// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/vector/to_scalar_mma.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"

typedef enum loom_vector_to_scalar_mma_numeric_e {
  LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED = 0,
  LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_FLOAT = 1,
  LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_SIGNED_INTEGER = 2,
} loom_vector_to_scalar_mma_numeric_t;

typedef enum loom_vector_to_scalar_mma_payload_layout_e {
  LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_UNSUPPORTED = 0,
  LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_FLAT_ROW_MAJOR = 1,
  LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_MATRIX_ROW_MAJOR = 2,
} loom_vector_to_scalar_mma_payload_layout_t;

typedef struct loom_vector_to_scalar_mma_fragment_t {
  // Dense logical payload value used for scalar lane extraction.
  loom_value_id_t payload;

  // Fragment facts attached to value.
  loom_vector_fragment_fact_t fact;

  // Dense logical payload type.
  loom_type_t type;

  // Logical row count for this fragment role.
  loom_vector_to_scalar_index_term_t rows;

  // Logical column count for this fragment role.
  loom_vector_to_scalar_index_term_t columns;

  // Physical-to-logical lane interpretation for dense fallback extraction.
  loom_vector_to_scalar_mma_payload_layout_t layout;
} loom_vector_to_scalar_mma_fragment_t;

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
      .rows = loom_vector_to_scalar_value_term(state, fact.shape_value_ids[0]),
      .columns =
          loom_vector_to_scalar_value_term(state, fact.shape_value_ids[1]),
  };
  return true;
}

static bool loom_vector_to_scalar_mma_logical_element_count(
    int64_t rows, int64_t columns, uint64_t* out_element_count) {
  if (rows < 0 || columns < 0) {
    return false;
  }
  uint64_t row_count = (uint64_t)rows;
  uint64_t column_count = (uint64_t)columns;
  if (column_count == 0) {
    *out_element_count = 0;
    return true;
  }
  if (row_count > UINT64_MAX / column_count) {
    return false;
  }
  *out_element_count = row_count * column_count;
  return true;
}

static bool loom_vector_to_scalar_mma_product_count(uint64_t lhs, uint64_t rhs,
                                                    uint64_t* out_product) {
  if (rhs == 0) {
    *out_product = 0;
    return true;
  }
  if (lhs > UINT64_MAX / rhs) {
    return false;
  }
  *out_product = lhs * rhs;
  return true;
}

static bool loom_vector_to_scalar_mma_term_is_static_non_negative(
    loom_vector_to_scalar_index_term_t term, int64_t* out_value) {
  if (term.is_dynamic || term.static_value < 0) {
    return false;
  }
  *out_value = term.static_value;
  return true;
}

static bool loom_vector_to_scalar_mma_term_is_static_value(
    loom_vector_to_scalar_index_term_t term, int64_t value) {
  return !term.is_dynamic && term.static_value == value;
}

static bool loom_vector_to_scalar_mma_terms_equal(
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs) {
  if (lhs.is_dynamic || rhs.is_dynamic) {
    return lhs.is_dynamic && rhs.is_dynamic &&
           lhs.dynamic_value == rhs.dynamic_value;
  }
  return lhs.static_value == rhs.static_value;
}

static bool loom_vector_to_scalar_mma_term_matches_product(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t candidate,
    loom_vector_to_scalar_index_term_t lhs,
    loom_vector_to_scalar_index_term_t rhs) {
  int64_t lhs_static = 0;
  int64_t rhs_static = 0;
  if (loom_vector_to_scalar_mma_term_is_static_non_negative(lhs, &lhs_static) &&
      loom_vector_to_scalar_mma_term_is_static_non_negative(rhs, &rhs_static)) {
    uint64_t product = 0;
    if (!loom_vector_to_scalar_mma_logical_element_count(lhs_static, rhs_static,
                                                         &product) ||
        product > INT64_MAX) {
      return false;
    }
    return loom_vector_to_scalar_mma_term_is_static_value(candidate,
                                                          (int64_t)product);
  }
  if (!candidate.is_dynamic) {
    return false;
  }
  loom_op_t* def_op = loom_vector_to_scalar_value_def_op(
      state->rewriter->module, candidate.dynamic_value);
  if (!def_op || !loom_index_mul_isa(def_op)) {
    return false;
  }
  loom_vector_to_scalar_index_term_t mul_lhs =
      loom_vector_to_scalar_value_term(state, loom_index_mul_lhs(def_op));
  loom_vector_to_scalar_index_term_t mul_rhs =
      loom_vector_to_scalar_value_term(state, loom_index_mul_rhs(def_op));
  return (loom_vector_to_scalar_mma_terms_equal(mul_lhs, lhs) &&
          loom_vector_to_scalar_mma_terms_equal(mul_rhs, rhs)) ||
         (loom_vector_to_scalar_mma_terms_equal(mul_lhs, rhs) &&
          loom_vector_to_scalar_mma_terms_equal(mul_rhs, lhs));
}

static bool loom_vector_to_scalar_mma_type_is_dense_payload(
    loom_vector_to_scalar_state_t* state, loom_type_t type,
    loom_vector_to_scalar_index_term_t rows,
    loom_vector_to_scalar_index_term_t columns,
    loom_vector_to_scalar_mma_payload_layout_t* out_layout) {
  *out_layout = LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_UNSUPPORTED;
  if (!loom_type_is_vector(type)) {
    return false;
  }
  if (loom_type_rank(type) == 2) {
    if (!loom_vector_to_scalar_mma_terms_equal(
            loom_vector_to_scalar_dim_bound_term(state, type, 0), rows) ||
        !loom_vector_to_scalar_mma_terms_equal(
            loom_vector_to_scalar_dim_bound_term(state, type, 1), columns)) {
      return false;
    }
    *out_layout = LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_MATRIX_ROW_MAJOR;
    return true;
  }
  if (loom_type_rank(type) != 1) {
    return false;
  }
  loom_vector_to_scalar_index_term_t flat_extent =
      loom_vector_to_scalar_dim_bound_term(state, type, 0);
  if ((loom_vector_to_scalar_mma_term_is_static_value(columns, 1) &&
       loom_vector_to_scalar_mma_terms_equal(flat_extent, rows)) ||
      (loom_vector_to_scalar_mma_term_is_static_value(rows, 1) &&
       loom_vector_to_scalar_mma_terms_equal(flat_extent, columns)) ||
      loom_vector_to_scalar_mma_term_matches_product(state, flat_extent, rows,
                                                     columns)) {
    *out_layout = LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_FLAT_ROW_MAJOR;
    return true;
  }

  int64_t static_rows = 0;
  int64_t static_columns = 0;
  uint64_t physical_element_count = 0;
  uint64_t logical_element_count = 0;
  if (loom_vector_to_scalar_mma_term_is_static_non_negative(rows,
                                                            &static_rows) &&
      loom_vector_to_scalar_mma_term_is_static_non_negative(columns,
                                                            &static_columns) &&
      loom_vector_to_scalar_mma_logical_element_count(
          static_rows, static_columns, &logical_element_count) &&
      loom_type_is_all_static(type) &&
      loom_type_static_element_count(type, &physical_element_count) &&
      physical_element_count == logical_element_count) {
    *out_layout = LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_FLAT_ROW_MAJOR;
    return true;
  }
  return false;
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
    loom_vector_to_scalar_mma_fragment_t* lhs,
    loom_vector_to_scalar_mma_fragment_t* rhs,
    loom_vector_to_scalar_mma_fragment_t* init,
    loom_vector_to_scalar_mma_numeric_t* out_numeric) {
  *out_numeric = LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  loom_type_t result_type = loom_module_value_type(
      state->rewriter->module, loom_vector_mma_result(state->op));
  loom_vector_to_scalar_mma_payload_layout_t result_layout =
      LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_UNSUPPORTED;
  if (!loom_vector_to_scalar_mma_terms_equal(lhs->columns, rhs->rows) ||
      !loom_vector_to_scalar_mma_terms_equal(lhs->rows, init->rows) ||
      !loom_vector_to_scalar_mma_terms_equal(rhs->columns, init->columns) ||
      !loom_vector_to_scalar_mma_type_is_dense_payload(
          state, lhs->type, lhs->rows, lhs->columns, &lhs->layout) ||
      !loom_vector_to_scalar_mma_type_is_dense_payload(
          state, rhs->type, rhs->rows, rhs->columns, &rhs->layout) ||
      !loom_vector_to_scalar_mma_type_is_dense_payload(
          state, init->type, init->rows, init->columns, &init->layout) ||
      !loom_vector_to_scalar_mma_type_is_dense_payload(
          state, result_type, init->rows, init->columns, &result_layout) ||
      init->layout != result_layout ||
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

static iree_status_t loom_vector_to_scalar_mma_build_matrix_indices(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* fragment,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column,
    loom_vector_to_scalar_index_list_t* out_indices) {
  switch (fragment->layout) {
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_FLAT_ROW_MAJOR: {
      loom_vector_to_scalar_index_term_t ordinal = {0};
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_MUL, row, fragment->columns,
          &ordinal));
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_ADD, ordinal, column,
          &ordinal));
      return loom_vector_to_scalar_terms_to_index_list(state, &ordinal, 1,
                                                       out_indices);
    }
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_MATRIX_ROW_MAJOR: {
      loom_vector_to_scalar_index_term_t terms[2] = {row, column};
      return loom_vector_to_scalar_terms_to_index_list(state, terms, 2,
                                                       out_indices);
    }
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_UNSUPPORTED:
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported vector.mma dense payload layout");
  }
}

static iree_status_t loom_vector_to_scalar_mma_materialize_matrix_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* fragment,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column, loom_value_id_t* out_lane) {
  loom_vector_to_scalar_index_list_t indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_matrix_indices(
      state, fragment, row, column, &indices));
  return loom_vector_to_scalar_materialize_lane(state, fragment->payload,
                                                indices, out_lane);
}

static iree_status_t loom_vector_to_scalar_mma_insert_matrix_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* fragment, loom_value_id_t lane,
    loom_value_id_t aggregate, loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column, loom_value_id_t* out_aggregate) {
  loom_vector_to_scalar_index_list_t indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_matrix_indices(
      state, fragment, row, column, &indices));
  return loom_vector_to_scalar_insert_lane(
      state, lane, aggregate, fragment->type, indices, out_aggregate);
}

static iree_status_t loom_vector_to_scalar_mma_extract_matrix_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* fragment, int64_t row,
    int64_t column, loom_value_id_t* out_lane) {
  return loom_vector_to_scalar_mma_materialize_matrix_lane(
      state, fragment, loom_vector_to_scalar_static_term(row),
      loom_vector_to_scalar_static_term(column), out_lane);
}

static iree_status_t loom_vector_to_scalar_mma_build_result_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init, int64_t row,
    int64_t column, int64_t k_count, loom_value_id_t* out_lane) {
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_extract_matrix_lane(
      state, init, row, column, &accumulator));

  loom_type_t accumulator_type =
      loom_type_scalar(loom_type_element_type(init->type));
  const loom_scalar_type_t lhs_scalar_type = loom_type_element_type(lhs->type);
  const loom_scalar_type_t rhs_scalar_type = loom_type_element_type(rhs->type);
  for (int64_t k = 0; k < k_count; ++k) {
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

static bool loom_vector_to_scalar_mma_shapes_are_static(
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init, int64_t* out_m,
    int64_t* out_n, int64_t* out_k) {
  int64_t lhs_rows = 0;
  int64_t lhs_columns = 0;
  int64_t rhs_rows = 0;
  int64_t rhs_columns = 0;
  int64_t init_rows = 0;
  int64_t init_columns = 0;
  if (!loom_vector_to_scalar_mma_term_is_static_non_negative(lhs->rows,
                                                             &lhs_rows) ||
      !loom_vector_to_scalar_mma_term_is_static_non_negative(lhs->columns,
                                                             &lhs_columns) ||
      !loom_vector_to_scalar_mma_term_is_static_non_negative(rhs->rows,
                                                             &rhs_rows) ||
      !loom_vector_to_scalar_mma_term_is_static_non_negative(rhs->columns,
                                                             &rhs_columns) ||
      !loom_vector_to_scalar_mma_term_is_static_non_negative(init->rows,
                                                             &init_rows) ||
      !loom_vector_to_scalar_mma_term_is_static_non_negative(init->columns,
                                                             &init_columns)) {
    return false;
  }
  if (lhs_rows != init_rows || lhs_columns != rhs_rows ||
      rhs_columns != init_columns) {
    return false;
  }
  *out_m = init_rows;
  *out_n = init_columns;
  *out_k = lhs_columns;
  return true;
}

static iree_status_t loom_vector_to_scalar_mma_build_accumulate_at(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column,
    loom_vector_to_scalar_index_term_t k, loom_value_id_t accumulator,
    loom_value_id_t* out_next) {
  loom_value_id_t lhs_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_materialize_matrix_lane(
      state, lhs, row, k, &lhs_lane));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_materialize_matrix_lane(
      state, rhs, k, column, &rhs_lane));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_accumulate(
      state, numeric, lhs_lane, loom_type_element_type(lhs->type), rhs_lane,
      loom_type_element_type(rhs->type), accumulator,
      loom_type_scalar(loom_type_element_type(state->vector_type)), out_next));
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass,
                            LOOM_VECTOR_TO_SCALAR_STAT_LANES_MATERIALIZED, 1);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_build_loop_bounds(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t upper_bound_term,
    loom_value_id_t* out_lower_bound, loom_value_id_t* out_upper_bound,
    loom_value_id_t* out_step) {
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, out_lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_term_value(state, upper_bound_term,
                                                        out_upper_bound));
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 1, out_step);
}

static iree_status_t loom_vector_to_scalar_mma_accumulator_loop(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column, loom_value_id_t init_lane,
    loom_value_id_t* out_lane) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_loop_bounds(
      state, lhs->columns, &lower_bound, &upper_bound, &step));

  loom_type_t accumulator_type =
      loom_type_scalar(loom_type_element_type(state->vector_type));
  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, lower_bound, upper_bound, step, &init_lane, 1,
      &accumulator_type, 1, NULL, 0, state->location, &loop));
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass,
                            LOOM_VECTOR_TO_SCALAR_STAT_LOOPS_CREATED, 1);
  }

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_vector_to_scalar_index_term_t k = loom_vector_to_scalar_dynamic_term(
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0));
  loom_value_id_t accumulator_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t next_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_accumulate_at(
      state, numeric, lhs, rhs, row, column, k, accumulator_arg,
      &next_accumulator));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &next_accumulator, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_lane = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_build_dynamic_result_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init,
    loom_vector_to_scalar_index_term_t row,
    loom_vector_to_scalar_index_term_t column, loom_value_id_t* out_lane) {
  loom_value_id_t init_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_materialize_matrix_lane(
      state, init, row, column, &init_lane));
  return loom_vector_to_scalar_mma_accumulator_loop(
      state, numeric, lhs, rhs, row, column, init_lane, out_lane);
}

static iree_status_t loom_vector_to_scalar_mma_column_loop(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init,
    loom_vector_to_scalar_index_term_t row, loom_value_id_t current_aggregate,
    loom_value_id_t* out_aggregate) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_loop_bounds(
      state, init->columns, &lower_bound, &upper_bound, &step));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, lower_bound, upper_bound, step,
      &current_aggregate, 1, &init->type, 1, NULL, 0, state->location, &loop));
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass,
                            LOOM_VECTOR_TO_SCALAR_STAT_LOOPS_CREATED, 1);
  }

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_vector_to_scalar_index_term_t column =
      loom_vector_to_scalar_dynamic_term(
          loom_region_entry_arg_id(loom_scf_for_body(loop), 0));
  loom_value_id_t aggregate_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_dynamic_result_lane(
      state, numeric, lhs, rhs, init, row, column, &lane));
  loom_value_id_t yielded_aggregate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_insert_matrix_lane(
      state, init, lane, aggregate_arg, row, column, &yielded_aggregate));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &yielded_aggregate, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_aggregate = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_row_loop(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init,
    loom_value_id_t* out_payload) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_loop_bounds(
      state, init->rows, &lower_bound, &upper_bound, &step));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->rewriter->builder, lower_bound, upper_bound, step, &init->payload,
      1, &init->type, 1, NULL, 0, state->location, &loop));
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass,
                            LOOM_VECTOR_TO_SCALAR_STAT_LOOPS_CREATED, 1);
  }

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, loop, loom_scf_for_body(loop));
  loom_vector_to_scalar_index_term_t row = loom_vector_to_scalar_dynamic_term(
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0));
  loom_value_id_t aggregate_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t yielded_aggregate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_column_loop(
      state, numeric, lhs, rhs, init, row, aggregate_arg, &yielded_aggregate));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&state->rewriter->builder,
                                            &yielded_aggregate, 1,
                                            state->location, &yield_op));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_payload = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_build_result_fragment(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* init, loom_value_id_t payload,
    loom_value_id_t* out_replacement) {
  loom_op_t* fragment_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_fragment_build(
      &state->rewriter->builder, LOOM_VECTOR_ROLE_RESULT, payload,
      init->fact.shape_value_ids[0], init->fact.shape_value_ids[1], NULL, 0,
      NULL, 0, init->type, state->location, &fragment_op));
  *out_replacement = loom_vector_fragment_result(fragment_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_lower_static(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init, int64_t m, int64_t n,
    int64_t k, loom_value_id_t* out_replacement, bool* out_handled) {
  uint64_t element_count = 0;
  if (!loom_vector_to_scalar_mma_logical_element_count(m, n, &element_count) ||
      element_count > UINT16_MAX) {
    return iree_ok_status();
  }
  uint64_t product_count = 0;
  if (!loom_vector_to_scalar_mma_product_count(element_count, (uint64_t)k,
                                               &product_count) ||
      product_count > UINT16_MAX) {
    return iree_ok_status();
  }

  if (element_count == 0) {
    loom_op_t* empty_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_empty_build(
        &state->rewriter->builder, init->type, state->location, &empty_op));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_result_fragment(
        state, init, loom_vector_empty_result(empty_op), out_replacement));
    *out_handled = true;
    return iree_ok_status();
  }

  loom_value_id_t* elements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->rewriter->arena, (iree_host_size_t)element_count,
      sizeof(*elements), (void**)&elements));
  uint16_t ordinal = 0;
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t column = 0; column < n; ++column) {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_result_lane(
          state, numeric, lhs, rhs, init, row, column, k, &elements[ordinal]));
      ++ordinal;
    }
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &state->rewriter->builder, elements, (iree_host_size_t)element_count,
      init->type, state->location, &from_elements_op));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_result_fragment(
      state, init, loom_vector_from_elements_result(from_elements_op),
      out_replacement));
  *out_handled = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_mma_lower_dynamic(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_numeric_t numeric,
    const loom_vector_to_scalar_mma_fragment_t* lhs,
    const loom_vector_to_scalar_mma_fragment_t* rhs,
    const loom_vector_to_scalar_mma_fragment_t* init,
    loom_value_id_t* out_replacement, bool* out_handled) {
  loom_value_id_t payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_row_loop(state, numeric, lhs,
                                                          rhs, init, &payload));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_result_fragment(
      state, init, payload, out_replacement));
  *out_handled = true;
  return iree_ok_status();
}

static bool loom_vector_to_scalar_mma_query_operands(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_mma_fragment_t* lhs,
    loom_vector_to_scalar_mma_fragment_t* rhs,
    loom_vector_to_scalar_mma_fragment_t* init,
    loom_vector_to_scalar_mma_numeric_t* out_numeric) {
  if (!loom_vector_to_scalar_mma_query_fragment(
          state, loom_vector_mma_lhs(state->op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS, lhs) ||
      !loom_vector_to_scalar_mma_query_fragment(
          state, loom_vector_mma_rhs(state->op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS, rhs) ||
      !loom_vector_to_scalar_mma_query_fragment(
          state, loom_vector_mma_init(state->op),
          LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
              LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT,
          init)) {
    return false;
  }
  return loom_vector_to_scalar_mma_shapes_match(state, lhs, rhs, init,
                                                out_numeric);
}

static iree_status_t loom_vector_to_scalar_mma_result_lane_terms(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_mma_fragment_t* init,
    loom_vector_to_scalar_index_list_t indices, bool* out_supported,
    loom_vector_to_scalar_index_term_t* out_row,
    loom_vector_to_scalar_index_term_t* out_column) {
  *out_supported = false;
  switch (init->layout) {
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_MATRIX_ROW_MAJOR:
      if (indices.rank != 2) {
        return iree_ok_status();
      }
      *out_row = loom_vector_to_scalar_lane_term(state, indices, 0);
      *out_column = loom_vector_to_scalar_lane_term(state, indices, 1);
      *out_supported = true;
      return iree_ok_status();
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_FLAT_ROW_MAJOR: {
      if (indices.rank != 1) {
        return iree_ok_status();
      }
      loom_vector_to_scalar_index_term_t ordinal =
          loom_vector_to_scalar_lane_term(state, indices, 0);
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_DIV, ordinal, init->columns,
          out_row));
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
          state, LOOM_VECTOR_TO_SCALAR_INDEX_BINARY_REM, ordinal, init->columns,
          out_column));
      *out_supported = true;
      return iree_ok_status();
    }
    case LOOM_VECTOR_TO_SCALAR_MMA_PAYLOAD_LAYOUT_UNSUPPORTED:
    default:
      return iree_ok_status();
  }
}

iree_status_t loom_vector_to_scalar_try_materialize_mma_lane(
    loom_vector_to_scalar_state_t* state, loom_op_t* op,
    loom_vector_to_scalar_index_list_t indices, bool* out_materialized,
    loom_value_id_t* out_lane) {
  *out_materialized = false;
  if (!loom_vector_mma_isa(op) || state->rewriter->fact_table == NULL) {
    return iree_ok_status();
  }

  loom_vector_to_scalar_state_t lane_state = *state;
  lane_state.op = op;
  lane_state.vector_type = loom_module_value_type(state->rewriter->module,
                                                  loom_vector_mma_result(op));
  lane_state.result_scalar_type =
      loom_vector_to_scalar_lane_type(lane_state.vector_type);
  lane_state.location = op->location;

  loom_vector_to_scalar_mma_fragment_t lhs = {0};
  loom_vector_to_scalar_mma_fragment_t rhs = {0};
  loom_vector_to_scalar_mma_fragment_t init = {0};
  loom_vector_to_scalar_mma_numeric_t numeric =
      LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  if (!loom_vector_to_scalar_mma_query_operands(&lane_state, &lhs, &rhs, &init,
                                                &numeric)) {
    return iree_ok_status();
  }

  loom_vector_to_scalar_index_term_t row = {0};
  loom_vector_to_scalar_index_term_t column = {0};
  bool lane_supported = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_result_lane_terms(
      &lane_state, &init, indices, &lane_supported, &row, &column));
  if (!lane_supported) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_mma_build_dynamic_result_lane(
      &lane_state, numeric, &lhs, &rhs, &init, row, column, out_lane));
  *out_materialized = true;
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
  loom_vector_to_scalar_mma_numeric_t numeric =
      LOOM_VECTOR_TO_SCALAR_MMA_NUMERIC_UNSUPPORTED;
  if (!loom_vector_to_scalar_mma_query_operands(state, &lhs, &rhs, &init,
                                                &numeric)) {
    return iree_ok_status();
  }

  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  if (loom_type_is_all_static(init.type) &&
      loom_vector_to_scalar_mma_shapes_are_static(&lhs, &rhs, &init, &m, &n,
                                                  &k)) {
    return loom_vector_to_scalar_mma_lower_static(state, numeric, &lhs, &rhs,
                                                  &init, m, n, k,
                                                  out_replacement, out_handled);
  }
  return loom_vector_to_scalar_mma_lower_dynamic(
      state, numeric, &lhs, &rhs, &init, out_replacement, out_handled);
}
