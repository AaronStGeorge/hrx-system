// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/passes/vector/to_scalar.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/passes/vector/to_scalar_internal.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

static const loom_pass_statistic_def_t kVectorToScalarStatistics[] = {
    {IREE_SVL("ops-lowered"), IREE_SVL("Number of vector ops lowered.")},
    {IREE_SVL("loops-created"), IREE_SVL("Number of scf.for loops created.")},
    {IREE_SVL("lanes-materialized"),
     IREE_SVL("Number of scalar lane programs materialized.")},
};

static const loom_pass_info_t loom_vector_to_scalar_pass_info_storage = {
    .name = IREE_SVL("vector-to-scalar"),
    .description =
        IREE_SVL("Expose vector lane semantics as scalar ops and scf loops."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_defs = kVectorToScalarStatistics,
    .statistic_count = IREE_ARRAYSIZE(kVectorToScalarStatistics),
};

const loom_pass_info_t* loom_vector_to_scalar_pass_info(void) {
  return &loom_vector_to_scalar_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_to_scalar_prepare_state(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_vector_to_scalar_descriptor_t* descriptor,
    uint16_t result_ordinal, loom_vector_to_scalar_state_t* out_state) {
  if (result_ordinal >= op->result_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "result ordinal %u out of range for %.*s",
        (unsigned)result_ordinal, (int)loom_op_name(rewriter->module, op).size,
        loom_op_name(rewriter->module, op).data);
  }
  loom_value_id_t result = loom_op_results(op)[result_ordinal];
  loom_type_t result_type = loom_module_value_type(rewriter->module, result);
  if (!loom_type_is_vector(result_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected vector result for %.*s",
                            (int)loom_op_name(rewriter->module, op).size,
                            loom_op_name(rewriter->module, op).data);
  }
  loom_type_t scalar_type = descriptor && descriptor->result_is_i1
                                ? loom_type_scalar(LOOM_SCALAR_TYPE_I1)
                                : loom_vector_to_scalar_lane_type(result_type);
  *out_state = (loom_vector_to_scalar_state_t){
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .descriptor = descriptor,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .result_ordinal = result_ordinal,
      .vector_type = result_type,
      .result_scalar_type = scalar_type,
      .location = op->location,
  };
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_replace_one_result(
    loom_vector_to_scalar_state_t* state, loom_value_id_t replacement) {
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      state->rewriter, state->op, &replacement, 1, state->value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      state->rewriter, state->op, &replacement, 1));
  loom_pass_mark_changed(state->pass);
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass, LOOM_VECTOR_TO_SCALAR_STAT_OPS_LOWERED,
                            1);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_replace_results(
    loom_vector_to_scalar_state_t* state, const loom_value_id_t* replacements,
    iree_host_size_t replacement_count) {
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      state->rewriter, state->op, replacements, replacement_count,
      state->value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      state->rewriter, state->op, replacements, replacement_count));
  loom_pass_mark_changed(state->pass);
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass, LOOM_VECTOR_TO_SCALAR_STAT_OPS_LOWERED,
                            1);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_erase_lowered_op(
    loom_vector_to_scalar_state_t* state) {
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(state->rewriter, state->op));
  loom_pass_mark_changed(state->pass);
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass, LOOM_VECTOR_TO_SCALAR_STAT_OPS_LOWERED,
                            1);
  }
  return iree_ok_status();
}

static bool loom_vector_to_scalar_memory_store_isa(loom_op_t* op) {
  return loom_vector_store_isa(op) || loom_vector_store_mask_isa(op) ||
         loom_vector_scatter_isa(op) || loom_vector_scatter_mask_isa(op);
}

static bool loom_vector_to_scalar_atomic_reduce_isa(loom_op_t* op) {
  return loom_vector_atomic_reduce_isa(op) ||
         loom_vector_atomic_reduce_mask_isa(op);
}

static bool loom_vector_to_scalar_atomic_rmw_isa(loom_op_t* op) {
  return loom_vector_atomic_rmw_isa(op) || loom_vector_atomic_rmw_mask_isa(op);
}

static loom_value_id_t loom_vector_to_scalar_memory_store_value(loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_VECTOR_STORE:
      return loom_vector_store_value(op);
    case LOOM_OP_VECTOR_STORE_MASK:
      return loom_vector_store_mask_value(op);
    case LOOM_OP_VECTOR_SCATTER:
      return loom_vector_scatter_value(op);
    case LOOM_OP_VECTOR_SCATTER_MASK:
      return loom_vector_scatter_mask_value(op);
    default:
      return LOOM_VALUE_ID_INVALID;
  }
}

static loom_value_id_t loom_vector_to_scalar_atomic_reduce_value(
    loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_VECTOR_ATOMIC_REDUCE:
      return loom_vector_atomic_reduce_value(op);
    case LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK:
      return loom_vector_atomic_reduce_mask_value(op);
    default:
      return LOOM_VALUE_ID_INVALID;
  }
}

static iree_status_t loom_vector_to_scalar_lower_memory_store_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  loom_value_id_t value = loom_vector_to_scalar_memory_store_value(op);
  loom_type_t vector_type = loom_module_value_type(rewriter->module, value);
  if (!loom_type_is_vector(vector_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected vector value for memory store op");
  }
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = vector_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
      .location = op->location,
  };
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_lower_memory_store(&state));
  return loom_vector_to_scalar_erase_lowered_op(&state);
}

static iree_status_t loom_vector_to_scalar_lower_store_compress_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  loom_value_id_t value = loom_vector_store_compress_value(op);
  loom_type_t vector_type = loom_module_value_type(rewriter->module, value);
  if (!loom_type_is_vector(vector_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected vector value for vector.store.compress");
  }
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = vector_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
      .location = op->location,
  };
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_memory_store_compress(&state));
  return loom_vector_to_scalar_erase_lowered_op(&state);
}

static iree_status_t loom_vector_to_scalar_lower_atomic_reduce_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  loom_value_id_t value = loom_vector_to_scalar_atomic_reduce_value(op);
  loom_type_t vector_type = loom_module_value_type(rewriter->module, value);
  if (!loom_type_is_vector(vector_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected vector value for vector.atomic.reduce");
  }
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = vector_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
      .location = op->location,
  };
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_memory_atomic_reduce(&state));
  return loom_vector_to_scalar_erase_lowered_op(&state);
}

static iree_status_t loom_vector_to_scalar_lower_atomic_rmw_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  loom_type_t vector_type =
      loom_module_value_type(rewriter->module, loom_op_results(op)[0]);
  if (!loom_type_is_vector(vector_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected vector result for vector.atomic.rmw");
  }
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = vector_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
      .location = op->location,
  };
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_memory_atomic_rmw(&state, &replacement));
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_atomic_cmpxchg_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  loom_type_t vector_type =
      loom_module_value_type(rewriter->module, loom_op_results(op)[0]);
  if (!loom_type_is_vector(vector_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected vector result for vector.atomic.cmpxchg");
  }
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = vector_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(vector_type),
      .location = op->location,
  };
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_memory_atomic_cmpxchg(&state, &replacement));
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_scalar_extract(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  loom_value_id_t result = loom_vector_extract_result(op);
  loom_type_t result_type = loom_module_value_type(rewriter->module, result);
  if (loom_type_is_vector(result_type)) return iree_ok_status();

  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .descriptor = loom_vector_to_scalar_find_descriptor(op->kind),
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .result_scalar_type = result_type,
      .location = op->location,
  };
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  loom_vector_to_scalar_index_term_t* explicit_terms = NULL;
  uint8_t explicit_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_from_explicit_indices(
      &state, loom_vector_extract_static_indices(op),
      loom_vector_extract_indices(op), &explicit_terms, &explicit_count));
  loom_vector_to_scalar_index_list_t source_indices = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_terms_to_index_list(
      &state, explicit_terms, explicit_count, &source_indices));
  loom_value_id_t source = loom_vector_extract_source(op);
  loom_type_t source_type = loom_module_value_type(rewriter->module, source);
  bool materialized = false;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_try_materialize_def_lane(
      &state, source, source_type, source_indices, &materialized,
      &replacement));
  if (!materialized) return iree_ok_status();
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_static_constant(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_vector_constant_result(op));
  if (!loom_type_is_all_static(result_type)) return iree_ok_status();
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = result_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(result_type),
      .location = op->location,
  };

  iree_host_size_t element_count = 0;
  if (!loom_type_static_element_count(result_type, &element_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected all-static vector.constant type");
  }
  loom_value_id_t* elements = NULL;
  if (element_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(rewriter->arena, element_count,
                                  sizeof(loom_value_id_t), (void**)&elements));
  }
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_build_constant_lane(&state, &elements[i]));
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &rewriter->builder, elements, element_count, result_type, op->location,
      &from_elements_op));
  loom_value_id_t replacement =
      loom_vector_from_elements_result(from_elements_op);
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_static_poison(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_vector_poison_result(op));
  if (!loom_type_is_all_static(result_type)) return iree_ok_status();
  loom_vector_to_scalar_state_t state = {
      .pass = pass,
      .rewriter = rewriter,
      .op = op,
      .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
      .vector_type = result_type,
      .result_scalar_type = loom_vector_to_scalar_lane_type(result_type),
      .location = op->location,
  };

  iree_host_size_t element_count = 0;
  if (!loom_type_static_element_count(result_type, &element_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected all-static vector.poison type");
  }
  loom_value_id_t* elements = NULL;
  if (element_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(rewriter->arena, element_count,
                                  sizeof(loom_value_id_t), (void**)&elements));
  }
  for (iree_host_size_t i = 0; i < element_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_build_poison_lane(&state, &elements[i]));
  }

  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &rewriter->builder, elements, element_count, result_type, op->location,
      &from_elements_op));
  loom_value_id_t replacement =
      loom_vector_from_elements_result(from_elements_op);
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

static iree_status_t loom_vector_to_scalar_lower_deinterleave(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op) {
  const loom_vector_to_scalar_descriptor_t* descriptor =
      loom_vector_to_scalar_find_descriptor(op->kind);
  if (!descriptor) return iree_ok_status();
  if (op->result_count != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.deinterleave must have two results");
  }

  loom_value_id_t replacements[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  loom_vector_to_scalar_state_t first_state = {0};
  for (uint16_t i = 0; i < 2; ++i) {
    loom_vector_to_scalar_state_t state = {0};
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_prepare_state(
        pass, rewriter, op, descriptor, i, &state));
    if (i == 0) first_state = state;
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_lower_aggregate(&state, &replacements[i]));
  }
  return loom_vector_to_scalar_replace_results(&first_state, replacements,
                                               IREE_ARRAYSIZE(replacements));
}

static iree_status_t loom_vector_to_scalar_lower_op(loom_pass_t* pass,
                                                    loom_rewriter_t* rewriter,
                                                    loom_op_t* op) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;

  if (loom_vector_deinterleave_isa(op)) {
    return loom_vector_to_scalar_lower_deinterleave(pass, rewriter, op);
  }

  if (loom_vector_to_scalar_memory_store_isa(op)) {
    return loom_vector_to_scalar_lower_memory_store_op(pass, rewriter, op);
  }

  if (loom_vector_store_compress_isa(op)) {
    return loom_vector_to_scalar_lower_store_compress_op(pass, rewriter, op);
  }

  if (loom_vector_to_scalar_atomic_reduce_isa(op)) {
    return loom_vector_to_scalar_lower_atomic_reduce_op(pass, rewriter, op);
  }

  if (loom_vector_to_scalar_atomic_rmw_isa(op)) {
    return loom_vector_to_scalar_lower_atomic_rmw_op(pass, rewriter, op);
  }

  if (loom_vector_atomic_cmpxchg_isa(op)) {
    return loom_vector_to_scalar_lower_atomic_cmpxchg_op(pass, rewriter, op);
  }

  if (op->result_count != 1) return iree_ok_status();

  if (loom_vector_constant_isa(op)) {
    return loom_vector_to_scalar_lower_static_constant(pass, rewriter, op);
  }

  if (loom_vector_poison_isa(op)) {
    return loom_vector_to_scalar_lower_static_poison(pass, rewriter, op);
  }

  if (loom_vector_empty_isa(op)) return iree_ok_status();

  if (loom_vector_extract_isa(op)) {
    loom_type_t extract_result_type = loom_module_value_type(
        rewriter->module, loom_vector_extract_result(op));
    if (!loom_type_is_vector(extract_result_type)) {
      return loom_vector_to_scalar_lower_scalar_extract(pass, rewriter, op);
    }
  }

  if (loom_vector_insert_isa(op)) {
    loom_type_t insert_result_type =
        loom_module_value_type(rewriter->module, loom_vector_insert_result(op));
    if (!loom_type_is_all_static(insert_result_type)) return iree_ok_status();
  }

  if (loom_vector_splat_isa(op)) {
    loom_vector_to_scalar_state_t state = {0};
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_prepare_state(pass, rewriter, op,
                                                             NULL, 0, &state));
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_lower_splat(&state, &replacement));
    return loom_vector_to_scalar_replace_one_result(&state, replacement);
  }

  if (loom_vector_reduce_isa(op)) {
    loom_vector_to_scalar_state_t state = {
        .pass = pass,
        .rewriter = rewriter,
        .op = op,
        .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
        .vector_type = loom_module_value_type(rewriter->module,
                                              loom_vector_reduce_input(op)),
        .result_scalar_type = loom_module_value_type(
            rewriter->module, loom_vector_reduce_init(op)),
        .location = op->location,
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_lower_reduce(&state, &replacement));
    return loom_vector_to_scalar_replace_one_result(&state, replacement);
  }

  if (loom_vector_reduce_axes_isa(op)) {
    loom_type_t result_type = loom_module_value_type(
        rewriter->module, loom_vector_reduce_axes_result(op));
    loom_type_t input_type = loom_module_value_type(
        rewriter->module, loom_vector_reduce_axes_input(op));
    loom_vector_to_scalar_state_t state = {
        .pass = pass,
        .rewriter = rewriter,
        .op = op,
        .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
        .vector_type =
            loom_type_is_vector(result_type) ? result_type : input_type,
        .result_scalar_type = loom_type_is_vector(result_type)
                                  ? loom_vector_to_scalar_lane_type(result_type)
                                  : result_type,
        .location = op->location,
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_lower_reduce_axes(&state, &replacement));
    return loom_vector_to_scalar_replace_one_result(&state, replacement);
  }

  if (loom_vector_dotf_isa(op)) {
    loom_vector_to_scalar_state_t state = {
        .pass = pass,
        .rewriter = rewriter,
        .op = op,
        .value_checkpoint = loom_rewriter_value_checkpoint(rewriter),
        .vector_type =
            loom_module_value_type(rewriter->module, loom_vector_dotf_lhs(op)),
        .result_scalar_type =
            loom_module_value_type(rewriter->module, loom_vector_dotf_init(op)),
        .location = op->location,
    };
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_lower_dotf(&state, &replacement));
    return loom_vector_to_scalar_replace_one_result(&state, replacement);
  }

  const loom_vector_to_scalar_descriptor_t* descriptor =
      loom_vector_to_scalar_find_descriptor(op->kind);
  if (!descriptor) return iree_ok_status();

  loom_vector_to_scalar_state_t state = {0};
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_prepare_state(
      pass, rewriter, op, descriptor, 0, &state));
  if (descriptor->lane_kind == LOOM_VECTOR_TO_SCALAR_LANE_TRANSFORM) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_validate_transform(&state));
  }
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_lower_aggregate(&state, &replacement));
  return loom_vector_to_scalar_replace_one_result(&state, replacement);
}

iree_status_t loom_vector_to_scalar_run(loom_pass_t* pass,
                                        loom_module_t* module,
                                        loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  loom_value_fact_table_t* facts = NULL;
  iree_status_t status = loom_pass_value_facts_prepare(
      pass, module, loom_pass_value_fact_scope_function(function), &facts);
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_enable_analysis(&rewriter, function, facts);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_seed_function(&rewriter, function);
  }
  while (iree_status_is_ok(status)) {
    loom_op_t* op = loom_rewriter_pop(&rewriter);
    if (!op) break;
    bool erased = false;
    status = loom_rewriter_erase_if_dead(&rewriter, op, &erased);
    if (!iree_status_is_ok(status)) continue;
    if (erased) {
      loom_pass_mark_changed(pass);
      continue;
    }
    status = loom_vector_to_scalar_lower_op(pass, &rewriter, op);
  }
  loom_rewriter_deinitialize(&rewriter);
  if (!iree_status_is_ok(status)) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
  }
  return status;
}
