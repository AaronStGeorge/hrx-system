// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/combining.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/passes/vector_to_scalar_internal.h"

//===----------------------------------------------------------------------===//
// Reduction lowering
//===----------------------------------------------------------------------===//

static bool loom_vector_to_scalar_reduce_scalar_kind(
    loom_combining_kind_t reduce_kind, loom_op_kind_t* out_scalar_kind) {
  switch (reduce_kind) {
    case LOOM_COMBINING_KIND_ADDI:
      *out_scalar_kind = LOOM_OP_SCALAR_ADDI;
      return true;
    case LOOM_COMBINING_KIND_ADDF:
      *out_scalar_kind = LOOM_OP_SCALAR_ADDF;
      return true;
    case LOOM_COMBINING_KIND_MULI:
      *out_scalar_kind = LOOM_OP_SCALAR_MULI;
      return true;
    case LOOM_COMBINING_KIND_MULF:
      *out_scalar_kind = LOOM_OP_SCALAR_MULF;
      return true;
    case LOOM_COMBINING_KIND_MINSI:
      *out_scalar_kind = LOOM_OP_SCALAR_MINSI;
      return true;
    case LOOM_COMBINING_KIND_MAXSI:
      *out_scalar_kind = LOOM_OP_SCALAR_MAXSI;
      return true;
    case LOOM_COMBINING_KIND_MINUI:
      *out_scalar_kind = LOOM_OP_SCALAR_MINUI;
      return true;
    case LOOM_COMBINING_KIND_MAXUI:
      *out_scalar_kind = LOOM_OP_SCALAR_MAXUI;
      return true;
    case LOOM_COMBINING_KIND_ANDI:
      *out_scalar_kind = LOOM_OP_SCALAR_ANDI;
      return true;
    case LOOM_COMBINING_KIND_ORI:
      *out_scalar_kind = LOOM_OP_SCALAR_ORI;
      return true;
    case LOOM_COMBINING_KIND_XORI:
      *out_scalar_kind = LOOM_OP_SCALAR_XORI;
      return true;
    case LOOM_COMBINING_KIND_MINIMUMF:
      *out_scalar_kind = LOOM_OP_SCALAR_MINIMUMF;
      return true;
    case LOOM_COMBINING_KIND_MAXIMUMF:
      *out_scalar_kind = LOOM_OP_SCALAR_MAXIMUMF;
      return true;
    case LOOM_COMBINING_KIND_MINNUMF:
      *out_scalar_kind = LOOM_OP_SCALAR_MINNUMF;
      return true;
    case LOOM_COMBINING_KIND_MAXNUMF:
      *out_scalar_kind = LOOM_OP_SCALAR_MAXNUMF;
      return true;
    default:
      return false;
  }
}

typedef struct loom_vector_to_scalar_accumulator_state_t {
  loom_vector_to_scalar_state_t lane_state;
  loom_value_id_t input;
  loom_value_id_t rhs;
  loom_value_id_t init;
  loom_op_kind_t scalar_kind;
  bool use_fmaf;
} loom_vector_to_scalar_accumulator_state_t;

static iree_status_t loom_vector_to_scalar_build_accumulator_lane(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t accumulator,
    loom_value_id_t* out_next) {
  loom_value_id_t lhs_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      &state->lane_state, state->input, indices, &lhs_lane));
  if (state->use_fmaf) {
    loom_value_id_t rhs_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        &state->lane_state, state->rhs, indices, &rhs_lane));
    return loom_vector_to_scalar_build_generic_lane_op(
        &state->lane_state, LOOM_OP_SCALAR_FMAF, 0,
        (loom_value_id_t[]){lhs_lane, rhs_lane, accumulator}, 3, NULL, 0,
        state->lane_state.result_scalar_type, out_next);
  }
  return loom_vector_to_scalar_build_generic_lane_op(
      &state->lane_state, state->scalar_kind, 0,
      (loom_value_id_t[]){accumulator, lhs_lane}, 2, NULL, 0,
      state->lane_state.result_scalar_type, out_next);
}

static iree_status_t loom_vector_to_scalar_lower_static_accumulator(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_value_id_t* out_replacement) {
  iree_host_size_t element_count = 0;
  if (!loom_type_static_element_count(state->lane_state.vector_type,
                                      &element_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected all-static vector input type");
  }
  loom_value_id_t accumulator = state->init;
  uint8_t rank = loom_type_rank(state->lane_state.vector_type);
  int64_t* indices = NULL;
  if (rank > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->lane_state.rewriter->arena, rank,
                                  sizeof(int64_t), (void**)&indices));
  }
  for (iree_host_size_t ordinal = 0; ordinal < element_count; ++ordinal) {
    loom_vector_to_scalar_indices_from_ordinal(state->lane_state.vector_type,
                                               ordinal, indices);
    loom_vector_to_scalar_index_list_t index_list = {
        .static_indices = indices,
        .rank = rank,
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_accumulator_lane(
        state, index_list, accumulator, &accumulator));
    if (state->lane_state.pass->statistics) {
      loom_pass_statistic_add(state->lane_state.pass,
                              LOOM_VECTOR_TO_SCALAR_STAT_LANES_MATERIALIZED, 1);
    }
  }
  *out_replacement = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_accumulator_loop_axis(
    loom_vector_to_scalar_accumulator_state_t* state, uint8_t axis,
    loom_value_id_t current_accumulator, loom_value_id_t* dynamic_indices,
    loom_value_id_t* out_accumulator) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 0,
      &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 1,
      &step));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      &state->lane_state, state->lane_state.vector_type, axis, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->lane_state.rewriter->builder, lower_bound, upper_bound, step,
      &current_accumulator, 1, &state->lane_state.result_scalar_type, 1, NULL,
      0, state->lane_state.location, &loop));
  if (state->lane_state.pass->statistics) {
    loom_pass_statistic_add(state->lane_state.pass,
                            LOOM_VECTOR_TO_SCALAR_STAT_LOOPS_CREATED, 1);
  }

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->lane_state.rewriter->builder, loop, loom_scf_for_body(loop));
  dynamic_indices[axis] = loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t accumulator_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t yielded_accumulator = LOOM_VALUE_ID_INVALID;
  if (axis + 1 == loom_type_rank(state->lane_state.vector_type)) {
    loom_vector_to_scalar_index_list_t index_list = {
        .dynamic_indices = dynamic_indices,
        .rank = loom_type_rank(state->lane_state.vector_type),
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_accumulator_lane(
        state, index_list, accumulator_arg, &yielded_accumulator));
    if (state->lane_state.pass->statistics) {
      loom_pass_statistic_add(state->lane_state.pass,
                              LOOM_VECTOR_TO_SCALAR_STAT_LANES_MATERIALIZED, 1);
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_accumulator_loop_axis(
        state, (uint8_t)(axis + 1), accumulator_arg, dynamic_indices,
        &yielded_accumulator));
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      &state->lane_state.rewriter->builder, &yielded_accumulator, 1,
      state->lane_state.location, &yield_op));
  loom_builder_restore(&state->lane_state.rewriter->builder, saved);

  *out_accumulator = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_lower_dynamic_accumulator(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_value_id_t* out_replacement) {
  uint8_t rank = loom_type_rank(state->lane_state.vector_type);
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->lane_state.rewriter->arena, rank, sizeof(loom_value_id_t),
      (void**)&dynamic_indices));
  return loom_vector_to_scalar_accumulator_loop_axis(
      state, 0, state->init, dynamic_indices, out_replacement);
}

static iree_status_t loom_vector_to_scalar_lower_accumulator(
    loom_vector_to_scalar_accumulator_state_t* state,
    loom_value_id_t* out_replacement) {
  if (loom_type_is_all_static(state->lane_state.vector_type)) {
    return loom_vector_to_scalar_lower_static_accumulator(state,
                                                          out_replacement);
  }
  return loom_vector_to_scalar_lower_dynamic_accumulator(state,
                                                         out_replacement);
}

iree_status_t loom_vector_to_scalar_lower_reduce(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  loom_op_kind_t scalar_kind = LOOM_OP_KIND_UNKNOWN;
  if (!loom_vector_to_scalar_reduce_scalar_kind(
          loom_vector_reduce_kind(state->op), &scalar_kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported vector.reduce kind %u",
                            (unsigned)loom_vector_reduce_kind(state->op));
  }
  loom_vector_to_scalar_accumulator_state_t accumulator_state = {
      .lane_state = *state,
      .input = loom_vector_reduce_input(state->op),
      .init = loom_vector_reduce_init(state->op),
      .scalar_kind = scalar_kind,
  };
  return loom_vector_to_scalar_lower_accumulator(&accumulator_state,
                                                 out_replacement);
}

//===----------------------------------------------------------------------===//
// Axis-preserving reduction lowering
//===----------------------------------------------------------------------===//

typedef struct loom_vector_to_scalar_reduce_axes_state_t {
  loom_vector_to_scalar_state_t lane_state;
  loom_value_id_t input;
  loom_value_id_t init;
  loom_type_t input_type;
  loom_type_t result_type;
  loom_attribute_t axes;
  loom_op_kind_t scalar_kind;
} loom_vector_to_scalar_reduce_axes_state_t;

static void loom_vector_to_scalar_reduce_axes_source_terms(
    loom_vector_to_scalar_reduce_axes_state_t* state,
    loom_vector_to_scalar_index_list_t result_indices,
    const loom_vector_to_scalar_index_term_t* reduced_terms,
    loom_vector_to_scalar_index_term_t* source_terms) {
  uint16_t reduced_index = 0;
  uint8_t result_axis = 0;
  uint8_t input_rank = loom_type_rank(state->input_type);
  for (uint8_t input_axis = 0; input_axis < input_rank; ++input_axis) {
    if (reduced_index < state->axes.count &&
        state->axes.i64_array[reduced_index] == input_axis) {
      source_terms[input_axis] = reduced_terms[reduced_index++];
      continue;
    }
    source_terms[input_axis] = loom_vector_to_scalar_lane_term(
        &state->lane_state, result_indices, result_axis++);
  }
}

static iree_status_t loom_vector_to_scalar_reduce_axes_build_combiner(
    loom_vector_to_scalar_reduce_axes_state_t* state,
    loom_vector_to_scalar_index_list_t result_indices,
    const loom_vector_to_scalar_index_term_t* reduced_terms,
    loom_value_id_t accumulator, loom_value_id_t* out_accumulator) {
  loom_vector_to_scalar_index_term_t source_terms[LOOM_TYPE_MAX_RANK] = {{0}};
  loom_vector_to_scalar_reduce_axes_source_terms(state, result_indices,
                                                 reduced_terms, source_terms);

  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      &state->lane_state, source_terms, loom_type_rank(state->input_type),
      &source_indices));
  loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      &state->lane_state, state->input, source_indices, &lane));

  if (state->lane_state.pass->statistics) {
    loom_pass_statistic_add(state->lane_state.pass,
                            LOOM_VECTOR_TO_SCALAR_STAT_LANES_MATERIALIZED, 1);
  }
  return loom_vector_to_scalar_build_generic_lane_op(
      &state->lane_state, state->scalar_kind, 0,
      (loom_value_id_t[]){accumulator, lane}, 2, NULL, 0,
      state->lane_state.result_scalar_type, out_accumulator);
}

static iree_status_t loom_vector_to_scalar_reduce_axes_axis(
    loom_vector_to_scalar_reduce_axes_state_t* state, uint16_t axis_index,
    loom_vector_to_scalar_index_list_t result_indices,
    loom_vector_to_scalar_index_term_t* reduced_terms,
    loom_value_id_t current_accumulator, loom_value_id_t* out_accumulator) {
  if (axis_index == state->axes.count) {
    return loom_vector_to_scalar_reduce_axes_build_combiner(
        state, result_indices, reduced_terms, current_accumulator,
        out_accumulator);
  }

  uint8_t input_axis = (uint8_t)state->axes.i64_array[axis_index];
  if (!loom_type_dim_is_dynamic_at(state->input_type, input_axis)) {
    loom_value_id_t accumulator = current_accumulator;
    int64_t extent =
        (int64_t)loom_type_dim_static_size_at(state->input_type, input_axis);
    for (int64_t i = 0; i < extent; ++i) {
      reduced_terms[axis_index] = loom_vector_to_scalar_static_term(i);
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_reduce_axes_axis(
          state, (uint16_t)(axis_index + 1), result_indices, reduced_terms,
          accumulator, &accumulator));
    }
    *out_accumulator = accumulator;
    return iree_ok_status();
  }

  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 0,
      &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 1,
      &step));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      &state->lane_state, state->input_type, input_axis, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->lane_state.rewriter->builder, lower_bound, upper_bound, step,
      &current_accumulator, 1, &state->lane_state.result_scalar_type, 1, NULL,
      0, state->lane_state.location, &loop));
  if (state->lane_state.pass->statistics) {
    loom_pass_statistic_add(state->lane_state.pass,
                            LOOM_VECTOR_TO_SCALAR_STAT_LOOPS_CREATED, 1);
  }

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->lane_state.rewriter->builder, loop, loom_scf_for_body(loop));
  reduced_terms[axis_index] = loom_vector_to_scalar_value_term(
      &state->lane_state, loom_region_entry_arg_id(loom_scf_for_body(loop), 0));
  loom_value_id_t accumulator_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t yielded_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_reduce_axes_axis(
      state, (uint16_t)(axis_index + 1), result_indices, reduced_terms,
      accumulator_arg, &yielded_accumulator));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      &state->lane_state.rewriter->builder, &yielded_accumulator, 1,
      state->lane_state.location, &yield_op));
  loom_builder_restore(&state->lane_state.rewriter->builder, saved);

  *out_accumulator = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_reduce_axes_lane(
    loom_vector_to_scalar_reduce_axes_state_t* state,
    loom_vector_to_scalar_index_list_t result_indices,
    loom_value_id_t* out_lane) {
  loom_value_id_t accumulator = state->init;
  if (loom_type_is_vector(state->result_type)) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        &state->lane_state, state->init, result_indices, &accumulator));
  }

  loom_vector_to_scalar_index_term_t reduced_terms[LOOM_TYPE_MAX_RANK] = {{0}};
  return loom_vector_to_scalar_reduce_axes_axis(
      state, 0, result_indices, reduced_terms, accumulator, out_lane);
}

static iree_status_t loom_vector_to_scalar_reduce_axes_static_result(
    loom_vector_to_scalar_reduce_axes_state_t* state,
    loom_value_id_t* out_replacement) {
  iree_host_size_t element_count = 0;
  if (!loom_type_static_element_count(state->result_type, &element_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected all-static vector result type");
  }
  if (element_count == 0) {
    *out_replacement = state->init;
    return iree_ok_status();
  }

  loom_value_id_t* elements = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->lane_state.rewriter->arena, element_count, sizeof(loom_value_id_t),
      (void**)&elements));
  uint8_t result_rank = loom_type_rank(state->result_type);
  int64_t* result_indices = NULL;
  if (result_rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->lane_state.rewriter->arena, result_rank, sizeof(int64_t),
        (void**)&result_indices));
  }
  for (iree_host_size_t ordinal = 0; ordinal < element_count; ++ordinal) {
    loom_vector_to_scalar_indices_from_ordinal(state->result_type, ordinal,
                                               result_indices);
    loom_vector_to_scalar_index_list_t index_list = {
        .static_indices = result_indices,
        .rank = result_rank,
    };
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_reduce_axes_lane(
        state, index_list, &elements[ordinal]));
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &state->lane_state.rewriter->builder, elements, element_count,
      state->result_type, state->lane_state.location, &from_elements_op));
  *out_replacement = loom_vector_from_elements_result(from_elements_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_reduce_axes_result_loop_axis(
    loom_vector_to_scalar_reduce_axes_state_t* state, uint8_t axis,
    loom_value_id_t current_aggregate, loom_value_id_t* dynamic_indices,
    loom_value_id_t* out_aggregate) {
  loom_value_id_t lower_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t step = LOOM_VALUE_ID_INVALID;
  loom_value_id_t upper_bound = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 0,
      &lower_bound));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_constant(
      &state->lane_state.rewriter->builder,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->lane_state.location, 1,
      &step));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_dim_bound(
      &state->lane_state, state->result_type, axis, &upper_bound));

  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &state->lane_state.rewriter->builder, lower_bound, upper_bound, step,
      &current_aggregate, 1, &state->result_type, 1, NULL, 0,
      state->lane_state.location, &loop));
  if (state->lane_state.pass->statistics) {
    loom_pass_statistic_add(state->lane_state.pass,
                            LOOM_VECTOR_TO_SCALAR_STAT_LOOPS_CREATED, 1);
  }

  loom_builder_ip_t saved = loom_builder_enter_region(
      &state->lane_state.rewriter->builder, loop, loom_scf_for_body(loop));
  dynamic_indices[axis] = loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  loom_value_id_t aggregate_arg =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  loom_value_id_t yielded_aggregate = LOOM_VALUE_ID_INVALID;
  if (axis + 1 == loom_type_rank(state->result_type)) {
    loom_vector_to_scalar_index_list_t index_list = {
        .dynamic_indices = dynamic_indices,
        .rank = loom_type_rank(state->result_type),
    };
    loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_reduce_axes_lane(state, index_list, &lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_insert_lane(
        &state->lane_state, lane, aggregate_arg, state->result_type, index_list,
        &yielded_aggregate));
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_reduce_axes_result_loop_axis(
        state, (uint8_t)(axis + 1), aggregate_arg, dynamic_indices,
        &yielded_aggregate));
  }
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      &state->lane_state.rewriter->builder, &yielded_aggregate, 1,
      state->lane_state.location, &yield_op));
  loom_builder_restore(&state->lane_state.rewriter->builder, saved);

  *out_aggregate = loom_scf_for_results(loop).values[0];
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_reduce_axes_dynamic_result(
    loom_vector_to_scalar_reduce_axes_state_t* state,
    loom_value_id_t* out_replacement) {
  uint8_t result_rank = loom_type_rank(state->result_type);
  loom_value_id_t* dynamic_indices = NULL;
  if (result_rank > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->lane_state.rewriter->arena, result_rank, sizeof(loom_value_id_t),
        (void**)&dynamic_indices));
  }
  return loom_vector_to_scalar_reduce_axes_result_loop_axis(
      state, 0, state->init, dynamic_indices, out_replacement);
}

iree_status_t loom_vector_to_scalar_lower_reduce_axes(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  loom_op_kind_t scalar_kind = LOOM_OP_KIND_UNKNOWN;
  if (!loom_vector_to_scalar_reduce_scalar_kind(
          loom_vector_reduce_axes_kind(state->op), &scalar_kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported vector.reduce.axes kind %u",
                            (unsigned)loom_vector_reduce_axes_kind(state->op));
  }

  loom_type_t result_type = loom_module_value_type(
      state->rewriter->module, loom_vector_reduce_axes_result(state->op));
  loom_vector_to_scalar_reduce_axes_state_t reduce_state = {
      .lane_state = *state,
      .input = loom_vector_reduce_axes_input(state->op),
      .init = loom_vector_reduce_axes_init(state->op),
      .input_type = loom_module_value_type(
          state->rewriter->module, loom_vector_reduce_axes_input(state->op)),
      .result_type = result_type,
      .axes = loom_vector_reduce_axes_axes(state->op),
      .scalar_kind = scalar_kind,
  };
  if (loom_type_is_scalar(result_type)) {
    loom_vector_to_scalar_index_list_t index_list = {0};
    return loom_vector_to_scalar_reduce_axes_lane(&reduce_state, index_list,
                                                  out_replacement);
  }
  if (!loom_type_is_vector(result_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected scalar or vector reduce.axes result");
  }
  if (loom_type_is_all_static(result_type)) {
    return loom_vector_to_scalar_reduce_axes_static_result(&reduce_state,
                                                           out_replacement);
  }
  return loom_vector_to_scalar_reduce_axes_dynamic_result(&reduce_state,
                                                          out_replacement);
}

iree_status_t loom_vector_to_scalar_lower_dotf(
    loom_vector_to_scalar_state_t* state, loom_value_id_t* out_replacement) {
  loom_vector_to_scalar_accumulator_state_t accumulator_state = {
      .lane_state = *state,
      .input = loom_vector_dotf_lhs(state->op),
      .rhs = loom_vector_dotf_rhs(state->op),
      .init = loom_vector_dotf_init(state->op),
      .use_fmaf = true,
  };
  return loom_vector_to_scalar_lower_accumulator(&accumulator_state,
                                                 out_replacement);
}
