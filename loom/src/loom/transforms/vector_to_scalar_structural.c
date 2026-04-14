// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/vector_to_scalar_internal.h"

//===----------------------------------------------------------------------===//
// Structural lane programs
//===----------------------------------------------------------------------===//

iree_status_t loom_vector_to_scalar_build_broadcast_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t source = loom_vector_broadcast_source(state->op);
  loom_type_t source_type =
      loom_module_value_type(state->rewriter->module, source);
  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(state->vector_type);
  if (source_rank > result_rank) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.broadcast source rank exceeds result rank");
  }

  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (source_rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, source_rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  uint8_t leading_broadcast_rank = (uint8_t)(result_rank - source_rank);
  for (uint8_t source_axis = 0; source_axis < source_rank; ++source_axis) {
    if (!loom_type_dim_is_dynamic_at(source_type, source_axis) &&
        loom_type_dim_static_size_at(source_type, source_axis) == 1) {
      source_terms[source_axis] = loom_vector_to_scalar_static_term(0);
    } else {
      uint8_t result_axis = (uint8_t)(leading_broadcast_rank + source_axis);
      source_terms[source_axis] =
          loom_vector_to_scalar_lane_term(indices, result_axis);
    }
  }

  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, source_rank, &source_indices));
  return loom_vector_to_scalar_materialize_lane(state, source, source_indices,
                                                out_lane);
}

iree_status_t loom_vector_to_scalar_build_extract_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t source = loom_vector_extract_source(state->op);
  loom_type_t source_type =
      loom_module_value_type(state->rewriter->module, source);
  loom_vector_to_scalar_index_term_t* explicit_terms = NULL;
  uint8_t explicit_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_from_explicit_indices(
      state, loom_vector_extract_static_indices(state->op),
      loom_vector_extract_indices(state->op), &explicit_terms,
      &explicit_count));

  uint8_t source_rank = loom_type_rank(source_type);
  if ((uint16_t)explicit_count + indices.rank != source_rank) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.extract index rank mismatch");
  }
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (source_rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, source_rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  for (uint8_t i = 0; i < explicit_count; ++i) {
    source_terms[i] = explicit_terms[i];
  }
  for (uint8_t i = 0; i < indices.rank; ++i) {
    source_terms[explicit_count + i] =
        loom_vector_to_scalar_lane_term(indices, i);
  }

  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, source_rank, &source_indices));
  return loom_vector_to_scalar_materialize_lane(state, source, source_indices,
                                                out_lane);
}

iree_status_t loom_vector_to_scalar_build_insert_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_vector_to_scalar_index_term_t* explicit_terms = NULL;
  uint8_t explicit_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_from_explicit_indices(
      state, loom_vector_insert_static_indices(state->op),
      loom_vector_insert_indices(state->op), &explicit_terms, &explicit_count));
  if (explicit_count > indices.rank) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.insert index rank mismatch");
  }

  bool condition_is_statically_true = true;
  loom_value_id_t dynamic_condition = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < explicit_count; ++i) {
    loom_vector_to_scalar_index_term_t lane_term =
        loom_vector_to_scalar_lane_term(indices, i);
    bool equal = false;
    if (loom_vector_to_scalar_terms_equal_static(lane_term, explicit_terms[i],
                                                 &equal)) {
      if (!equal) {
        return loom_vector_to_scalar_materialize_lane(
            state, loom_vector_insert_dest(state->op), indices, out_lane);
      }
      continue;
    }
    condition_is_statically_true = false;
    loom_value_id_t axis_equal = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_term_cmp(
        state, LOOM_INDEX_CMP_PREDICATE_EQ, lane_term, explicit_terms[i],
        &axis_equal));
    if (dynamic_condition == LOOM_VALUE_ID_INVALID) {
      dynamic_condition = axis_equal;
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_i1_and(
          state, dynamic_condition, axis_equal, &dynamic_condition));
    }
  }

  loom_vector_to_scalar_index_term_t* value_terms = NULL;
  uint8_t value_rank = (uint8_t)(indices.rank - explicit_count);
  if (value_rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, value_rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&value_terms));
  }
  for (uint8_t i = 0; i < value_rank; ++i) {
    value_terms[i] =
        loom_vector_to_scalar_lane_term(indices, (uint8_t)(explicit_count + i));
  }
  loom_vector_to_scalar_index_list_t value_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, value_terms, value_rank, &value_indices));
  loom_value_id_t value_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_insert_value(state->op), value_indices, &value_lane));
  if (condition_is_statically_true) {
    *out_lane = value_lane;
    return iree_ok_status();
  }

  loom_value_id_t dest_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_insert_dest(state->op), indices, &dest_lane));
  return loom_vector_to_scalar_build_scalar_select_lane(
      state, dynamic_condition, value_lane, dest_lane, out_lane);
}

iree_status_t loom_vector_to_scalar_build_slice_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_vector_to_scalar_index_term_t* offset_terms = NULL;
  uint8_t offset_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_from_explicit_indices(
      state, loom_vector_slice_static_offsets(state->op),
      loom_vector_slice_offsets(state->op), &offset_terms, &offset_count));
  if (offset_count != indices.rank) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.slice offset rank mismatch");
  }
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (indices.rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, indices.rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  for (uint8_t i = 0; i < indices.rank; ++i) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
        state, LOOM_OP_INDEX_ADD, offset_terms[i],
        loom_vector_to_scalar_lane_term(indices, i), &source_terms[i]));
  }

  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, indices.rank, &source_indices));
  return loom_vector_to_scalar_materialize_lane(
      state, loom_vector_slice_source(state->op), source_indices, out_lane);
}

static bool loom_vector_to_scalar_concat_axis_extents_are_static(
    const loom_module_t* module, loom_value_slice_t inputs, uint8_t axis) {
  for (uint16_t i = 0; i < inputs.count; ++i) {
    loom_type_t input_type =
        loom_module_value_type(module, loom_value_slice_get(inputs, i));
    if (loom_type_dim_is_dynamic_at(input_type, axis)) return false;
  }
  return true;
}

static iree_status_t loom_vector_to_scalar_build_concat_dynamic_lane(
    loom_vector_to_scalar_state_t* state, loom_value_slice_t inputs,
    uint16_t input_ordinal, loom_vector_to_scalar_index_term_t prefix,
    uint8_t axis, loom_vector_to_scalar_index_list_t indices,
    loom_value_id_t* out_lane) {
  if (input_ordinal >= inputs.count) {
    return loom_vector_to_scalar_build_poison_lane(state, out_lane);
  }

  loom_value_id_t input = loom_value_slice_get(inputs, input_ordinal);
  loom_type_t input_type =
      loom_module_value_type(state->rewriter->module, input);
  loom_vector_to_scalar_index_term_t axis_index =
      loom_vector_to_scalar_lane_term(indices, axis);
  loom_vector_to_scalar_index_term_t extent =
      loom_vector_to_scalar_dim_bound_term(input_type, axis);
  loom_vector_to_scalar_index_term_t end = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_OP_INDEX_ADD, prefix, extent, &end));
  loom_value_id_t within_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_term_cmp(
      state, LOOM_INDEX_CMP_PREDICATE_SLT, axis_index, end, &within_input));

  loom_vector_to_scalar_index_term_t source_axis = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_OP_INDEX_SUB, axis_index, prefix, &source_axis));
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (indices.rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, indices.rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  for (uint8_t i = 0; i < indices.rank; ++i) {
    source_terms[i] = loom_vector_to_scalar_lane_term(indices, i);
  }
  source_terms[axis] = source_axis;
  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, indices.rank, &source_indices));

  loom_op_t* if_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_if_build(
      &state->rewriter->builder, within_input, &state->result_scalar_type, 1,
      NULL, 0, state->location, &if_op));

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->rewriter->builder, if_op, loom_scf_if_then_region(if_op));
  loom_value_id_t then_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, input, source_indices, &then_lane));
  loom_op_t* then_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      &state->rewriter->builder, &then_lane, 1, state->location, &then_yield));
  loom_builder_restore(&state->rewriter->builder, saved);

  saved = loom_builder_enter_region(&state->rewriter->builder, if_op,
                                    loom_scf_if_else_region(if_op));
  loom_value_id_t else_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_concat_dynamic_lane(
      state, inputs, (uint16_t)(input_ordinal + 1), end, axis, indices,
      &else_lane));
  loom_op_t* else_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      &state->rewriter->builder, &else_lane, 1, state->location, &else_yield));
  loom_builder_restore(&state->rewriter->builder, saved);

  *out_lane = loom_scf_if_results(if_op).values[0];
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_concat_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  int64_t axis = loom_vector_concat_axis(state->op);
  if (axis < 0 || axis >= indices.rank) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.concat axis out of range");
  }
  uint8_t axis_u8 = (uint8_t)axis;
  loom_value_slice_t inputs = loom_vector_concat_inputs(state->op);
  if (loom_vector_to_scalar_indices_are_dynamic(indices) ||
      !loom_vector_to_scalar_concat_axis_extents_are_static(
          state->rewriter->module, inputs, axis_u8)) {
    return loom_vector_to_scalar_build_concat_dynamic_lane(
        state, inputs, 0, loom_vector_to_scalar_static_term(0), axis_u8,
        indices, out_lane);
  }

  int64_t axis_index = indices.static_indices[axis_u8];
  int64_t prefix = 0;
  for (uint16_t input_ordinal = 0; input_ordinal < inputs.count;
       ++input_ordinal) {
    loom_value_id_t input = loom_value_slice_get(inputs, input_ordinal);
    loom_type_t input_type =
        loom_module_value_type(state->rewriter->module, input);
    if (loom_type_dim_is_dynamic_at(input_type, axis_u8)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "vector-to-scalar cannot lower vector.concat with dynamic input "
          "axis extents yet");
    }
    int64_t extent = (int64_t)loom_type_dim_static_size_at(input_type, axis_u8);
    if (axis_index < prefix + extent) {
      loom_vector_to_scalar_index_term_t* source_terms = NULL;
      if (indices.rank > 0) {
        IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
            state->rewriter->arena, indices.rank,
            sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
      }
      for (uint8_t i = 0; i < indices.rank; ++i) {
        source_terms[i] =
            loom_vector_to_scalar_static_term(indices.static_indices[i]);
      }
      source_terms[axis_u8] =
          loom_vector_to_scalar_static_term(axis_index - prefix);
      loom_vector_to_scalar_index_list_t source_indices = {0};
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
          state, source_terms, indices.rank, &source_indices));
      return loom_vector_to_scalar_materialize_lane(state, input,
                                                    source_indices, out_lane);
    }
    prefix += extent;
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "vector.concat lane index is outside inputs");
}

iree_status_t loom_vector_to_scalar_build_transpose_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_attribute_t permutation = loom_vector_transpose_permutation(state->op);
  if (permutation.kind != LOOM_ATTR_I64_ARRAY ||
      permutation.count != indices.rank) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.transpose permutation rank mismatch");
  }
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (indices.rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, indices.rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  for (uint8_t result_axis = 0; result_axis < indices.rank; ++result_axis) {
    int64_t source_axis = permutation.i64_array[result_axis];
    if (source_axis < 0 || source_axis >= indices.rank) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "vector.transpose source axis out of range");
    }
    source_terms[source_axis] =
        loom_vector_to_scalar_lane_term(indices, result_axis);
  }
  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, indices.rank, &source_indices));
  return loom_vector_to_scalar_materialize_lane(
      state, loom_vector_transpose_source(state->op), source_indices, out_lane);
}

iree_status_t loom_vector_to_scalar_build_shuffle_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  if (loom_vector_to_scalar_indices_are_dynamic(indices)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "vector-to-scalar cannot lower dynamic "
                            "vector.shuffle lane selection");
  }
  loom_attribute_t source_lanes = loom_vector_shuffle_source_lanes(state->op);
  if (source_lanes.kind != LOOM_ATTR_I64_ARRAY || indices.rank != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.shuffle requires a rank-1 lane map");
  }
  int64_t result_lane = indices.static_indices[0];
  if (result_lane < 0 || result_lane >= source_lanes.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.shuffle result lane is out of range");
  }
  int64_t source_lane = source_lanes.i64_array[result_lane];
  loom_vector_to_scalar_index_list_t source_indices = {
      .static_indices = &source_lane,
      .rank = 1,
  };
  return loom_vector_to_scalar_materialize_lane(
      state, loom_vector_shuffle_source(state->op), source_indices, out_lane);
}

iree_status_t loom_vector_to_scalar_build_interleave_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  int64_t axis = loom_vector_interleave_axis(state->op);
  if (axis < 0 || axis >= indices.rank) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.interleave axis out of range");
  }
  uint8_t axis_u8 = (uint8_t)axis;
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (indices.rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, indices.rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  for (uint8_t i = 0; i < indices.rank; ++i) {
    source_terms[i] = loom_vector_to_scalar_lane_term(indices, i);
  }

  loom_vector_to_scalar_index_term_t axis_index = source_terms[axis_u8];
  loom_vector_to_scalar_index_term_t source_axis_index = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_OP_INDEX_DIV, axis_index,
      loom_vector_to_scalar_static_term(2), &source_axis_index));
  source_terms[axis_u8] = source_axis_index;

  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, indices.rank, &source_indices));

  if (!loom_vector_to_scalar_indices_are_dynamic(indices)) {
    loom_value_id_t source = (indices.static_indices[axis_u8] & 1) == 0
                                 ? loom_vector_interleave_even(state->op)
                                 : loom_vector_interleave_odd(state->op);
    return loom_vector_to_scalar_materialize_lane(state, source, source_indices,
                                                  out_lane);
  }

  loom_value_id_t even_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_interleave_even(state->op), source_indices,
      &even_lane));
  loom_value_id_t odd_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, loom_vector_interleave_odd(state->op), source_indices, &odd_lane));

  loom_vector_to_scalar_index_term_t remainder = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_OP_INDEX_REM, axis_index,
      loom_vector_to_scalar_static_term(2), &remainder));
  loom_value_id_t remainder_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_term_value(state, remainder, &remainder_value));
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      state->location, 0, &zero));
  loom_op_t* is_even_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      &state->rewriter->builder, LOOM_INDEX_CMP_PREDICATE_EQ, remainder_value,
      zero, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), state->location, &is_even_op));
  return loom_vector_to_scalar_build_scalar_select_lane(
      state, loom_index_cmp_result(is_even_op), even_lane, odd_lane, out_lane);
}

iree_status_t loom_vector_to_scalar_build_deinterleave_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  int64_t axis = loom_vector_deinterleave_axis(state->op);
  if (axis < 0 || axis >= indices.rank) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.deinterleave axis out of range");
  }
  uint8_t axis_u8 = (uint8_t)axis;
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  if (indices.rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->rewriter->arena, indices.rank,
        sizeof(loom_vector_to_scalar_index_term_t), (void**)&source_terms));
  }
  for (uint8_t i = 0; i < indices.rank; ++i) {
    source_terms[i] = loom_vector_to_scalar_lane_term(indices, i);
  }

  loom_vector_to_scalar_index_term_t scaled = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_OP_INDEX_MUL, source_terms[axis_u8],
      loom_vector_to_scalar_static_term(2), &scaled));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_term_binary(
      state, LOOM_OP_INDEX_ADD, scaled,
      loom_vector_to_scalar_static_term((int64_t)state->result_ordinal),
      &source_terms[axis_u8]));

  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      state, source_terms, indices.rank, &source_indices));
  return loom_vector_to_scalar_materialize_lane(
      state, loom_vector_deinterleave_source(state->op), source_indices,
      out_lane);
}

iree_status_t loom_vector_to_scalar_build_bitcast_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_value_id_t input = loom_vector_bitcast_input(state->op);
  loom_type_t input_type =
      loom_module_value_type(state->rewriter->module, input);
  if (!loom_type_shape_equals(input_type, state->vector_type)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "vector-to-scalar cannot lower shape-changing vector.bitcast yet");
  }
  loom_scalar_type_t input_element_type = loom_type_element_type(input_type);
  loom_scalar_type_t result_element_type =
      loom_type_element_type(state->vector_type);
  if (loom_scalar_type_bitwidth(input_element_type) !=
      loom_scalar_type_bitwidth(result_element_type)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "vector-to-scalar cannot lower element-width-changing vector.bitcast "
        "yet");
  }

  loom_value_id_t input_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, input, indices, &input_lane));
  if (input_element_type == result_element_type) {
    *out_lane = input_lane;
    return iree_ok_status();
  }
  loom_op_t* bitcast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
      &state->rewriter->builder, input_lane,
      loom_type_scalar(input_element_type), state->result_scalar_type,
      state->location, &bitcast_op));
  *out_lane = loom_scalar_bitcast_result(bitcast_op);
  return iree_ok_status();
}
