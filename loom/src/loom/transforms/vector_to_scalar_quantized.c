// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/vector_to_scalar_internal.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Quantized lane programs
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_build_scalar_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_type_t result_type,
    loom_value_id_t* out_result) {
  return loom_vector_to_scalar_build_generic_lane_op(
      state, kind, 0, (loom_value_id_t[]){lhs, rhs}, 2, NULL, 0, result_type,
      out_result);
}

static int64_t loom_vector_to_scalar_integer_mask_value(int32_t bit_width,
                                                        int64_t used_bits) {
  if (used_bits <= 0) return 0;
  if (used_bits >= bit_width) return bit_width == 1 ? 1 : -1;
  return (int64_t)((UINT64_C(1) << (uint64_t)used_bits) - 1);
}

static int64_t loom_vector_to_scalar_shifted_integer_mask_value(
    int32_t bit_width, int64_t offset, int64_t used_bits) {
  if (used_bits <= 0) return 0;
  uint64_t mask =
      used_bits >= 64 ? UINT64_MAX : ((UINT64_C(1) << (uint64_t)used_bits) - 1);
  mask <<= (uint64_t)offset;
  if (bit_width < 64) mask &= (UINT64_C(1) << (uint32_t)bit_width) - 1;
  if (bit_width == 1) return (mask & 1) ? 1 : 0;
  return (int64_t)mask;
}

static iree_status_t loom_vector_to_scalar_build_integer_mask(
    loom_vector_to_scalar_state_t* state, loom_type_t type, int64_t used_bits,
    loom_value_id_t* out_mask) {
  int32_t bit_width = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bit_width <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected fixed-width integer scalar type");
  }
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, type, state->location,
      loom_vector_to_scalar_integer_mask_value(bit_width, used_bits), out_mask);
}

static iree_status_t loom_vector_to_scalar_cast_integer_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type, bool signed_extend,
    loom_value_id_t* out_result) {
  if (loom_type_equal(input_type, result_type)) {
    *out_result = input;
    return iree_ok_status();
  }

  int32_t input_width =
      loom_scalar_type_bitwidth(loom_type_element_type(input_type));
  int32_t result_width =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  if (input_width <= 0 || result_width <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected fixed-width integer scalar types");
  }

  loom_op_t* cast_op = NULL;
  if (input_width < result_width) {
    if (signed_extend) {
      IREE_RETURN_IF_ERROR(
          loom_scalar_extsi_build(&state->rewriter->builder, input, input_type,
                                  result_type, state->location, &cast_op));
      *out_result = loom_scalar_extsi_result(cast_op);
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(&state->rewriter->builder,
                                                 input, input_type, result_type,
                                                 state->location, &cast_op));
    *out_result = loom_scalar_extui_result(cast_op);
    return iree_ok_status();
  }

  if (input_width > result_width) {
    IREE_RETURN_IF_ERROR(
        loom_scalar_trunci_build(&state->rewriter->builder, input, input_type,
                                 result_type, state->location, &cast_op));
    *out_result = loom_scalar_trunci_result(cast_op);
    return iree_ok_status();
  }

  loom_op_t* bitcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(&state->rewriter->builder,
                                                 input, input_type, result_type,
                                                 state->location, &bitcast_op));
  *out_result = loom_scalar_bitcast_result(bitcast_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_scalar_shift(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t input, loom_type_t type, int64_t amount,
    loom_value_id_t* out_result) {
  if (amount == 0) {
    *out_result = input;
    return iree_ok_status();
  }
  loom_value_id_t amount_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, type, state->location, amount, &amount_value));
  if (kind == LOOM_OP_SCALAR_SHLI) {
    loom_op_t* shift_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_shli_build(&state->rewriter->builder, 0,
                                                input, amount_value, type,
                                                state->location, &shift_op));
    *out_result = loom_scalar_shli_result(shift_op);
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_scalar_binary(
      state, kind, input, amount_value, type, out_result);
}

iree_status_t loom_vector_to_scalar_build_bitfield_extract_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, bool signed_extract,
    loom_value_id_t* out_lane) {
  loom_value_id_t source =
      signed_extract ? loom_vector_bitfield_extracts_source(state->op)
                     : loom_vector_bitfield_extractu_source(state->op);
  int64_t offset = signed_extract
                       ? loom_vector_bitfield_extracts_offset(state->op)
                       : loom_vector_bitfield_extractu_offset(state->op);
  int64_t width = signed_extract
                      ? loom_vector_bitfield_extracts_width(state->op)
                      : loom_vector_bitfield_extractu_width(state->op);
  loom_type_t source_type =
      loom_module_value_type(state->rewriter->module, source);
  loom_type_t source_scalar_type = loom_vector_to_scalar_lane_type(source_type);
  int32_t source_width =
      loom_scalar_type_bitwidth(loom_type_element_type(source_scalar_type));
  if (source_width <= 0 || offset < 0 || width <= 0 || offset > source_width ||
      width > source_width - offset) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid vector bitfield extract range");
  }

  loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, source, indices, &source_lane));

  loom_value_id_t extracted = LOOM_VALUE_ID_INVALID;
  if (signed_extract) {
    loom_value_id_t shifted_left = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
        state, LOOM_OP_SCALAR_SHLI, source_lane, source_scalar_type,
        source_width - offset - width, &shifted_left));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
        state, LOOM_OP_SCALAR_SHRSI, shifted_left, source_scalar_type,
        source_width - width, &extracted));
  } else {
    loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
        state, LOOM_OP_SCALAR_SHRUI, source_lane, source_scalar_type, offset,
        &shifted));
    loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
        state, source_scalar_type, width, &mask));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
        state, LOOM_OP_SCALAR_ANDI, shifted, mask, source_scalar_type,
        &extracted));
  }

  return loom_vector_to_scalar_cast_integer_lane(
      state, extracted, source_scalar_type, state->result_scalar_type,
      signed_extract, out_lane);
}

iree_status_t loom_vector_to_scalar_build_bitfield_insert_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t field = loom_vector_bitfield_insert_field(state->op);
  loom_value_id_t base = loom_vector_bitfield_insert_base(state->op);
  int64_t offset = loom_vector_bitfield_insert_offset(state->op);
  int64_t width = loom_vector_bitfield_insert_width(state->op);
  loom_type_t field_type =
      loom_module_value_type(state->rewriter->module, field);
  loom_type_t field_scalar_type = loom_vector_to_scalar_lane_type(field_type);
  loom_type_t base_scalar_type = state->result_scalar_type;
  int32_t base_width =
      loom_scalar_type_bitwidth(loom_type_element_type(base_scalar_type));
  if (base_width <= 0 || offset < 0 || width <= 0 || offset > base_width ||
      width > base_width - offset) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid vector bitfield insert range");
  }

  loom_value_id_t field_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, field, indices, &field_lane));
  loom_value_id_t base_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_materialize_lane(state, base, indices, &base_lane));

  loom_value_id_t field_in_base_type = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, field_lane, field_scalar_type, base_scalar_type,
      /*signed_extend=*/false, &field_in_base_type));

  loom_value_id_t field_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
      state, base_scalar_type, width, &field_mask));
  loom_value_id_t field_low_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ANDI, field_in_base_type, field_mask,
      base_scalar_type, &field_low_bits));
  loom_value_id_t shifted_field = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
      state, LOOM_OP_SCALAR_SHLI, field_low_bits, base_scalar_type, offset,
      &shifted_field));

  loom_value_id_t target_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, base_scalar_type, state->location,
      loom_vector_to_scalar_shifted_integer_mask_value(base_width, offset,
                                                       width),
      &target_mask));
  loom_value_id_t all_ones = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
      state, base_scalar_type, base_width, &all_ones));
  loom_value_id_t clear_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_XORI, target_mask, all_ones, base_scalar_type,
      &clear_mask));
  loom_value_id_t cleared_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ANDI, base_lane, clear_mask, base_scalar_type,
      &cleared_base));
  return loom_vector_to_scalar_build_scalar_binary(state, LOOM_OP_SCALAR_ORI,
                                                   cleared_base, shifted_field,
                                                   base_scalar_type, out_lane);
}

static bool loom_vector_to_scalar_dot4i_lhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT4I_KIND_S8S8 ||
         kind == LOOM_VECTOR_DOT4I_KIND_S8U8;
}

static bool loom_vector_to_scalar_dot4i_rhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT4I_KIND_S8S8 ||
         kind == LOOM_VECTOR_DOT4I_KIND_U8S8;
}

static iree_status_t loom_vector_to_scalar_build_dot4i_source_indices(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t result_indices,
    loom_vector_to_scalar_index_term_t grouped_axis_base, uint8_t group_lane,
    loom_vector_to_scalar_index_list_t* out_source_indices) {
  uint8_t rank = result_indices.rank;
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->rewriter->arena, rank, sizeof(loom_vector_to_scalar_index_term_t),
      (void**)&source_terms));
  for (uint8_t axis = 0; axis < rank; ++axis) {
    source_terms[axis] =
        loom_vector_to_scalar_lane_term(state, result_indices, axis);
  }

  uint8_t grouped_axis = (uint8_t)(rank - 1);
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_OP_INDEX_ADD, grouped_axis_base,
      loom_vector_to_scalar_static_term(group_lane),
      &source_terms[grouped_axis]));
  return loom_vector_to_scalar_terms_to_index_list(state, source_terms, rank,
                                                   out_source_indices);
}

iree_status_t loom_vector_to_scalar_build_dot4i_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  uint8_t kind = loom_vector_dot4i_kind(state->op);
  if (kind >= LOOM_VECTOR_DOT4I_KIND_COUNT_) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported vector.dot4i kind %u", (unsigned)kind);
  }
  if (indices.rank == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.dot4i requires rank-1-or-higher vectors");
  }

  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_dot4i_acc(state->op), indices, &accumulator));
  uint8_t grouped_axis = (uint8_t)(indices.rank - 1);
  loom_vector_to_scalar_index_term_t grouped_axis_base = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_OP_INDEX_MUL,
      loom_vector_to_scalar_lane_term(state, indices, grouped_axis),
      loom_vector_to_scalar_static_term(4), &grouped_axis_base));

  loom_type_t i8_type = loom_type_scalar(LOOM_SCALAR_TYPE_I8);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  for (uint8_t group_lane = 0; group_lane < 4; ++group_lane) {
    loom_vector_to_scalar_index_list_t source_indices = {0};
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_dot4i_source_indices(
        state, indices, grouped_axis_base, group_lane, &source_indices));

    loom_value_id_t lhs_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, loom_vector_dot4i_lhs(state->op), source_indices, &lhs_lane));
    loom_value_id_t rhs_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, loom_vector_dot4i_rhs(state->op), source_indices, &rhs_lane));

    loom_value_id_t lhs_i32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
        state, lhs_lane, i8_type, i32_type,
        loom_vector_to_scalar_dot4i_lhs_is_signed(kind), &lhs_i32));
    loom_value_id_t rhs_i32 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
        state, rhs_lane, i8_type, i32_type,
        loom_vector_to_scalar_dot4i_rhs_is_signed(kind), &rhs_i32));

    loom_value_id_t product = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
        state, LOOM_OP_SCALAR_MULI, 0, (loom_value_id_t[]){lhs_i32, rhs_i32}, 2,
        NULL, 0, i32_type, &product));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_generic_lane_op(
        state, LOOM_OP_SCALAR_ADDI, 0,
        (loom_value_id_t[]){accumulator, product}, 2, NULL, 0, i32_type,
        &accumulator));
  }
  *out_lane = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_index_term_as_scalar(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t term, loom_type_t result_type,
    loom_value_id_t* out_value) {
  if (!term.is_dynamic) {
    return loom_vector_to_scalar_build_scalar_constant(
        &state->rewriter->builder, result_type, state->location,
        term.static_value, out_value);
  }
  loom_value_id_t index_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_term_value(state, term, &index_value));
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_cast_build(&state->rewriter->builder, index_value,
                            loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                            result_type, state->location, &cast_op));
  *out_value = loom_index_cast_result(cast_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_scalar_shift_term(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t input, loom_type_t type,
    loom_vector_to_scalar_index_term_t amount, loom_value_id_t* out_result) {
  if (!amount.is_dynamic) {
    return loom_vector_to_scalar_build_scalar_shift(
        state, kind, input, type, amount.static_value, out_result);
  }
  loom_value_id_t amount_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_term_as_scalar(
      state, amount, type, &amount_value));
  if (kind == LOOM_OP_SCALAR_SHLI) {
    loom_op_t* shift_op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_shli_build(&state->rewriter->builder, 0,
                                                input, amount_value, type,
                                                state->location, &shift_op));
    *out_result = loom_scalar_shli_result(shift_op);
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_scalar_binary(
      state, kind, input, amount_value, type, out_result);
}

static iree_status_t loom_vector_to_scalar_checked_static_bit_position(
    int64_t ordinal, int64_t bit_width, int64_t* out_position) {
  if (!loom_checked_mul_i64(ordinal, bit_width, out_position)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "static bitstream position overflow");
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_checked_static_bit_end(
    int64_t start, int64_t bit_width, int64_t* out_end) {
  if (!loom_checked_add_i64(start, bit_width, out_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "static bitstream end overflow");
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_indices_from_ordinal_term(
    loom_vector_to_scalar_state_t* state, loom_type_t vector_type,
    loom_vector_to_scalar_index_term_t ordinal,
    loom_vector_to_scalar_index_list_t* out_indices) {
  uint8_t rank = loom_type_rank(vector_type);
  if (!ordinal.is_dynamic && ordinal.static_value < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "negative static vector lane ordinal");
  }
  if (!ordinal.is_dynamic && loom_type_is_all_static(vector_type)) {
    int64_t* static_indices = NULL;
    if (rank > 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena,
                                                     rank, sizeof(int64_t),
                                                     (void**)&static_indices));
    }
    loom_vector_to_scalar_indices_from_ordinal(
        vector_type, (iree_host_size_t)ordinal.static_value, static_indices);
    *out_indices = (loom_vector_to_scalar_index_list_t){
        .static_indices = static_indices,
        .rank = rank,
    };
    return iree_ok_status();
  }

  if (rank == 1) {
    loom_vector_to_scalar_index_term_t* terms = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, 1, sizeof(loom_vector_to_scalar_index_term_t),
        (void**)&terms));
    terms[0] = ordinal;
    return loom_vector_to_scalar_terms_to_index_list(state, terms, 1,
                                                     out_indices);
  }

  loom_vector_to_scalar_index_term_t* terms = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->rewriter->arena, rank, sizeof(loom_vector_to_scalar_index_term_t),
      (void**)&terms));
  loom_vector_to_scalar_index_term_t remaining = ordinal;
  for (uint8_t reverse_axis = 0; reverse_axis < rank; ++reverse_axis) {
    uint8_t axis = (uint8_t)(rank - reverse_axis - 1);
    loom_vector_to_scalar_index_term_t dim =
        loom_vector_to_scalar_dim_bound_term(state, vector_type, axis);
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_OP_INDEX_REM, remaining, dim, &terms[axis]));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_OP_INDEX_DIV, remaining, dim, &remaining));
  }
  return loom_vector_to_scalar_terms_to_index_list(state, terms, rank,
                                                   out_indices);
}

static iree_status_t loom_vector_to_scalar_materialize_linear_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t vector_value,
    loom_type_t vector_type, loom_vector_to_scalar_index_term_t ordinal,
    loom_value_id_t* out_lane) {
  loom_vector_to_scalar_index_list_t indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_indices_from_ordinal_term(
      state, vector_type, ordinal, &indices));
  return loom_vector_to_scalar_materialize_lane(state, vector_value, indices,
                                                out_lane);
}

static iree_status_t loom_vector_to_scalar_build_zero_lane(
    loom_vector_to_scalar_state_t* state, loom_type_t type,
    loom_value_id_t* out_value) {
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, type, state->location, 0, out_value);
}

static iree_status_t loom_vector_to_scalar_build_bitpack_piece(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source_lane,
    loom_type_t source_scalar_type, loom_type_t storage_scalar_type,
    int64_t source_shift, int64_t bit_count, int64_t dest_shift,
    loom_value_id_t accumulator, loom_value_id_t* out_accumulator) {
  loom_value_id_t shifted_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
      state, LOOM_OP_SCALAR_SHRUI, source_lane, source_scalar_type,
      source_shift, &shifted_source));
  loom_value_id_t source_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
      state, source_scalar_type, bit_count, &source_mask));
  loom_value_id_t source_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ANDI, shifted_source, source_mask,
      source_scalar_type, &source_bits));
  loom_value_id_t storage_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, source_bits, source_scalar_type, storage_scalar_type,
      /*signed_extend=*/false, &storage_bits));
  loom_value_id_t shifted_storage_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
      state, LOOM_OP_SCALAR_SHLI, storage_bits, storage_scalar_type, dest_shift,
      &shifted_storage_bits));
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ORI, accumulator, shifted_storage_bits,
      storage_scalar_type, out_accumulator);
}

static iree_status_t loom_vector_to_scalar_build_bitpack_static_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source,
    loom_type_t source_type, int64_t width,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t source_scalar_type = loom_vector_to_scalar_lane_type(source_type);
  loom_type_t storage_scalar_type = state->result_scalar_type;
  int32_t storage_width =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_scalar_type));
  int64_t storage_ordinal = loom_vector_to_scalar_linear_ordinal_static(
      state->vector_type, indices.static_indices);
  int64_t storage_start = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_checked_static_bit_position(
      storage_ordinal, storage_width, &storage_start));
  int64_t storage_end = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_checked_static_bit_end(
      storage_start, storage_width, &storage_end));
  int64_t first_source_ordinal = storage_start / width;
  int64_t last_source_ordinal = (storage_end - 1) / width;

  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_zero_lane(
      state, storage_scalar_type, &accumulator));
  uint8_t source_rank = loom_type_rank(source_type);
  int64_t* source_indices = NULL;
  if (source_rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena,
                                                   source_rank, sizeof(int64_t),
                                                   (void**)&source_indices));
  }
  for (int64_t source_ordinal = first_source_ordinal;
       source_ordinal <= last_source_ordinal; ++source_ordinal) {
    int64_t source_start = 0;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_checked_static_bit_position(
        source_ordinal, width, &source_start));
    int64_t source_end = 0;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_checked_static_bit_end(
        source_start, width, &source_end));
    int64_t overlap_start =
        storage_start > source_start ? storage_start : source_start;
    int64_t overlap_end = storage_end < source_end ? storage_end : source_end;
    int64_t bit_count = overlap_end - overlap_start;
    if (bit_count <= 0) continue;

    loom_vector_to_scalar_indices_from_ordinal(
        source_type, (iree_host_size_t)source_ordinal, source_indices);
    loom_vector_to_scalar_index_list_t source_index_list = {
        .static_indices = source_indices,
        .rank = source_rank,
    };
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, source, source_index_list, &source_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_bitpack_piece(
        state, source_lane, source_scalar_type, storage_scalar_type,
        overlap_start - source_start, bit_count, overlap_start - storage_start,
        accumulator, &accumulator));
  }
  *out_lane = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_bitpack_divisible_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source,
    loom_type_t source_type, int64_t width,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t source_scalar_type = loom_vector_to_scalar_lane_type(source_type);
  loom_type_t storage_scalar_type = state->result_scalar_type;
  int32_t storage_width =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_scalar_type));
  if (storage_width <= 0 || (storage_width % width) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid divisible bitpack widths");
  }
  int64_t fields_per_storage = storage_width / width;

  loom_vector_to_scalar_index_term_t storage_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_linear_ordinal_term(
      state, state->vector_type, indices, &storage_ordinal));
  loom_vector_to_scalar_index_term_t source_base = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_OP_INDEX_MUL, storage_ordinal,
      loom_vector_to_scalar_static_term(fields_per_storage), &source_base));

  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_zero_lane(
      state, storage_scalar_type, &accumulator));
  for (int64_t field_ordinal = 0; field_ordinal < fields_per_storage;
       ++field_ordinal) {
    loom_vector_to_scalar_index_term_t source_ordinal = {0};
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_OP_INDEX_ADD, source_base,
        loom_vector_to_scalar_static_term(field_ordinal), &source_ordinal));
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_linear_lane(
        state, source, source_type, source_ordinal, &source_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_bitpack_piece(
        state, source_lane, source_scalar_type, storage_scalar_type,
        /*source_shift=*/0, width, field_ordinal * width, accumulator,
        &accumulator));
  }
  *out_lane = accumulator;
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_bitpack_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t source = loom_vector_bitpack_source(state->op);
  loom_type_t source_type =
      loom_module_value_type(state->rewriter->module, source);
  int64_t width = loom_vector_bitpack_width(state->op);
  int32_t storage_width = loom_scalar_type_bitwidth(
      loom_type_element_type(state->result_scalar_type));
  if (width <= 0 || storage_width <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid vector.bitpack bit width");
  }

  if (!loom_vector_to_scalar_indices_are_dynamic(indices) &&
      loom_type_is_all_static(source_type)) {
    return loom_vector_to_scalar_build_bitpack_static_lane(
        state, source, source_type, width, indices, out_lane);
  }
  if ((storage_width % width) != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "vector-to-scalar can only lower dynamic vector.bitpack when field "
        "width divides storage element width");
  }
  return loom_vector_to_scalar_build_bitpack_divisible_lane(
      state, source, source_type, width, indices, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_bitunpack_piece(
    loom_vector_to_scalar_state_t* state, loom_value_id_t storage_lane,
    loom_type_t storage_scalar_type, loom_type_t result_scalar_type,
    int64_t source_shift, int64_t bit_count, int64_t dest_shift,
    loom_value_id_t accumulator, loom_value_id_t* out_accumulator) {
  loom_value_id_t shifted_storage = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
      state, LOOM_OP_SCALAR_SHRUI, storage_lane, storage_scalar_type,
      source_shift, &shifted_storage));
  loom_value_id_t storage_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
      state, storage_scalar_type, bit_count, &storage_mask));
  loom_value_id_t storage_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ANDI, shifted_storage, storage_mask,
      storage_scalar_type, &storage_bits));
  loom_value_id_t result_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, storage_bits, storage_scalar_type, result_scalar_type,
      /*signed_extend=*/false, &result_bits));
  loom_value_id_t shifted_result_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
      state, LOOM_OP_SCALAR_SHLI, result_bits, result_scalar_type, dest_shift,
      &shifted_result_bits));
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ORI, accumulator, shifted_result_bits,
      result_scalar_type, out_accumulator);
}

static iree_status_t loom_vector_to_scalar_finish_bitunpack_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t unsigned_lane,
    int64_t width, bool signed_unpack, loom_value_id_t* out_lane) {
  if (!signed_unpack) {
    *out_lane = unsigned_lane;
    return iree_ok_status();
  }
  int32_t result_width = loom_scalar_type_bitwidth(
      loom_type_element_type(state->result_scalar_type));
  if (result_width <= 0 || width > result_width) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid vector.bitunpack result width");
  }
  loom_value_id_t shifted_left = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift(
      state, LOOM_OP_SCALAR_SHLI, unsigned_lane, state->result_scalar_type,
      result_width - width, &shifted_left));
  return loom_vector_to_scalar_build_scalar_shift(
      state, LOOM_OP_SCALAR_SHRSI, shifted_left, state->result_scalar_type,
      result_width - width, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_bitunpack_static_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source,
    loom_type_t source_type, int64_t width, bool signed_unpack,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t storage_scalar_type =
      loom_vector_to_scalar_lane_type(source_type);
  loom_type_t result_scalar_type = state->result_scalar_type;
  int32_t storage_width =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_scalar_type));
  int64_t result_ordinal = loom_vector_to_scalar_linear_ordinal_static(
      state->vector_type, indices.static_indices);
  int64_t result_start = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_checked_static_bit_position(
      result_ordinal, width, &result_start));
  int64_t result_end = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_checked_static_bit_end(
      result_start, width, &result_end));
  int64_t first_storage_ordinal = result_start / storage_width;
  int64_t last_storage_ordinal = (result_end - 1) / storage_width;

  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_zero_lane(
      state, result_scalar_type, &accumulator));
  uint8_t source_rank = loom_type_rank(source_type);
  int64_t* source_indices = NULL;
  if (source_rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->rewriter->arena,
                                                   source_rank, sizeof(int64_t),
                                                   (void**)&source_indices));
  }
  for (int64_t storage_ordinal = first_storage_ordinal;
       storage_ordinal <= last_storage_ordinal; ++storage_ordinal) {
    int64_t storage_start = 0;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_checked_static_bit_position(
        storage_ordinal, storage_width, &storage_start));
    int64_t storage_end = 0;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_checked_static_bit_end(
        storage_start, storage_width, &storage_end));
    int64_t overlap_start =
        result_start > storage_start ? result_start : storage_start;
    int64_t overlap_end = result_end < storage_end ? result_end : storage_end;
    int64_t bit_count = overlap_end - overlap_start;
    if (bit_count <= 0) continue;

    loom_vector_to_scalar_indices_from_ordinal(
        source_type, (iree_host_size_t)storage_ordinal, source_indices);
    loom_vector_to_scalar_index_list_t source_index_list = {
        .static_indices = source_indices,
        .rank = source_rank,
    };
    loom_value_id_t storage_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, source, source_index_list, &storage_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_bitunpack_piece(
        state, storage_lane, storage_scalar_type, result_scalar_type,
        overlap_start - storage_start, bit_count, overlap_start - result_start,
        accumulator, &accumulator));
  }
  return loom_vector_to_scalar_finish_bitunpack_lane(state, accumulator, width,
                                                     signed_unpack, out_lane);
}

static iree_status_t loom_vector_to_scalar_build_bitunpack_divisible_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t source,
    loom_type_t source_type, int64_t width, bool signed_unpack,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t storage_scalar_type =
      loom_vector_to_scalar_lane_type(source_type);
  loom_type_t result_scalar_type = state->result_scalar_type;
  int32_t storage_width =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_scalar_type));
  if (storage_width <= 0 || (storage_width % width) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid divisible bitunpack widths");
  }
  int64_t fields_per_storage = storage_width / width;
  loom_vector_to_scalar_index_term_t result_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_linear_ordinal_term(
      state, state->vector_type, indices, &result_ordinal));
  loom_vector_to_scalar_index_term_t storage_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_OP_INDEX_DIV, result_ordinal,
      loom_vector_to_scalar_static_term(fields_per_storage), &storage_ordinal));
  loom_vector_to_scalar_index_term_t field_ordinal = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_OP_INDEX_REM, result_ordinal,
      loom_vector_to_scalar_static_term(fields_per_storage), &field_ordinal));
  loom_value_id_t storage_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_linear_lane(
      state, source, source_type, storage_ordinal, &storage_lane));
  loom_vector_to_scalar_index_term_t shift = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_OP_INDEX_MUL, field_ordinal,
      loom_vector_to_scalar_static_term(width), &shift));
  loom_value_id_t shifted_storage = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_shift_term(
      state, LOOM_OP_SCALAR_SHRUI, storage_lane, storage_scalar_type, shift,
      &shifted_storage));
  loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_integer_mask(
      state, storage_scalar_type, width, &mask));
  loom_value_id_t storage_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ANDI, shifted_storage, mask, storage_scalar_type,
      &storage_bits));
  loom_value_id_t result_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_integer_lane(
      state, storage_bits, storage_scalar_type, result_scalar_type,
      /*signed_extend=*/false, &result_bits));
  return loom_vector_to_scalar_finish_bitunpack_lane(state, result_bits, width,
                                                     signed_unpack, out_lane);
}

iree_status_t loom_vector_to_scalar_build_bitunpack_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, bool signed_unpack,
    loom_value_id_t* out_lane) {
  loom_value_id_t source = signed_unpack
                               ? loom_vector_bitunpacks_source(state->op)
                               : loom_vector_bitunpacku_source(state->op);
  loom_type_t source_type =
      loom_module_value_type(state->rewriter->module, source);
  int64_t width = signed_unpack ? loom_vector_bitunpacks_width(state->op)
                                : loom_vector_bitunpacku_width(state->op);
  loom_type_t storage_scalar_type =
      loom_vector_to_scalar_lane_type(source_type);
  int32_t storage_width =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_scalar_type));
  if (width <= 0 || storage_width <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid vector.bitunpack bit width");
  }

  if (!loom_vector_to_scalar_indices_are_dynamic(indices) &&
      loom_type_is_all_static(source_type)) {
    return loom_vector_to_scalar_build_bitunpack_static_lane(
        state, source, source_type, width, signed_unpack, indices, out_lane);
  }
  if ((storage_width % width) != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "vector-to-scalar can only lower dynamic vector.bitunpack when field "
        "width divides storage element width");
  }
  return loom_vector_to_scalar_build_bitunpack_divisible_lane(
      state, source, source_type, width, signed_unpack, indices, out_lane);
}
