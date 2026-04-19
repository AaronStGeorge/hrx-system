// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/passes/vector_to_scalar_internal.h"

//===----------------------------------------------------------------------===//
// Reduction lowering
//===----------------------------------------------------------------------===//

static bool loom_vector_to_scalar_reduce_scalar_kind(
    uint8_t reduce_kind, loom_op_kind_t* out_scalar_kind) {
  switch ((loom_vector_reduce_kind_t)reduce_kind) {
    case LOOM_VECTOR_REDUCE_KIND_ADDI:
      *out_scalar_kind = LOOM_OP_SCALAR_ADDI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_ADDF:
      *out_scalar_kind = LOOM_OP_SCALAR_ADDF;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MULI:
      *out_scalar_kind = LOOM_OP_SCALAR_MULI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MULF:
      *out_scalar_kind = LOOM_OP_SCALAR_MULF;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MINSI:
      *out_scalar_kind = LOOM_OP_SCALAR_MINSI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MAXSI:
      *out_scalar_kind = LOOM_OP_SCALAR_MAXSI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MINUI:
      *out_scalar_kind = LOOM_OP_SCALAR_MINUI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MAXUI:
      *out_scalar_kind = LOOM_OP_SCALAR_MAXUI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_ANDI:
      *out_scalar_kind = LOOM_OP_SCALAR_ANDI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_ORI:
      *out_scalar_kind = LOOM_OP_SCALAR_ORI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_XORI:
      *out_scalar_kind = LOOM_OP_SCALAR_XORI;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MINIMUMF:
      *out_scalar_kind = LOOM_OP_SCALAR_MINIMUMF;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MAXIMUMF:
      *out_scalar_kind = LOOM_OP_SCALAR_MAXIMUMF;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MINNUMF:
      *out_scalar_kind = LOOM_OP_SCALAR_MINNUMF;
      return true;
    case LOOM_VECTOR_REDUCE_KIND_MAXNUMF:
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
