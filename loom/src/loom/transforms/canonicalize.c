// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/canonicalize.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/special_values.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/rewriter.h"

//===----------------------------------------------------------------------===//
// Constant materialization
//===----------------------------------------------------------------------===//

// Materializes a typed constant from exact facts.
static iree_status_t loom_canonicalize_materialize_constant(
    loom_builder_t* builder, loom_value_facts_t facts, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value_id) {
  loom_attribute_t attr;
  if (loom_value_facts_is_float(facts)) {
    attr = loom_attr_f64(loom_value_facts_as_f64(facts));
  } else {
    attr = loom_attr_i64(facts.range_lo);
  }
  if (loom_type_is_scalar(result_type) &&
      (loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_INDEX ||
       loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_OFFSET)) {
    loom_op_t* constant_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, attr, result_type,
                                                   location, &constant_op));
    *out_value_id = loom_index_constant_result(constant_op);
    return iree_ok_status();
  }
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(builder, attr, result_type,
                                                  location, &constant_op));
  *out_value_id = loom_scalar_constant_result(constant_op);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

static const loom_pass_option_def_t kCanonicalizeOptions[] = {
    {IREE_SVL("max-iterations"),
     IREE_SVL("Maximum number of worklist iterations.")},
};

enum {
  LOOM_CANONICALIZE_STAT_OPS_MODIFIED = 0,
};

static const loom_pass_statistic_def_t kCanonicalizeStatistics[] = {
    {IREE_SVL("ops-modified"),
     IREE_SVL("Number of ops simplified by canonicalization.")},
};

static const loom_pass_info_t loom_canonicalize_pass_info_storage = {
    .name = IREE_SVL("canonicalize"),
    .description = IREE_SVL("Apply op-specific canonicalization patterns."),
    .kind = LOOM_PASS_FUNCTION,
    .option_defs = kCanonicalizeOptions,
    .option_count = 1,
    .statistic_defs = kCanonicalizeStatistics,
    .statistic_count = 1,
};

const loom_pass_info_t* loom_canonicalize_pass_info(void) {
  return &loom_canonicalize_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

#define LOOM_CANONICALIZE_DEFAULT_MAX_ITERATIONS 10

static bool loom_canonicalize_value_type_has_poison(loom_type_t type) {
  if (loom_type_is_scalar(type)) return true;
  if (loom_type_is_vector(type)) {
    return !loom_type_has_static_zero_extent(type);
  }
  return false;
}

static bool loom_canonicalize_type_is_static_empty_vector(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_has_static_zero_extent(type);
}

static bool loom_canonicalize_value_is_static_empty_vector(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  return loom_canonicalize_type_is_static_empty_vector(
      loom_module_value_type(module, value_id));
}

static bool loom_canonicalize_op_has_poison_operand(const loom_module_t* module,
                                                    const loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] == LOOM_VALUE_ID_INVALID) continue;
    if (loom_value_is_poison(module, operands[i])) return true;
  }
  return false;
}

static bool loom_canonicalize_can_replace_results_with_poison(
    const loom_module_t* module, const loom_op_t* op) {
  if (op->result_count == 0) return false;
  if (op->region_count != 0) return false;
  if (op->tied_result_count != 0) return false;
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) return false;
    loom_type_t type = loom_module_value_type(module, results[i]);
    if (!loom_canonicalize_value_type_has_poison(type)) return false;
  }
  return true;
}

static iree_status_t loom_canonicalize_try_propagate_poison(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_propagated) {
  *out_propagated = false;
  loom_trait_flags_t traits = loom_op_effective_traits(rewriter->module, op);
  if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE)) return iree_ok_status();
  if (!loom_canonicalize_op_has_poison_operand(rewriter->module, op)) {
    return iree_ok_status();
  }
  if (!loom_canonicalize_can_replace_results_with_poison(rewriter->module,
                                                         op)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_results_with_materialized_values_and_erase(
          rewriter, op, loom_poison_build));
  *out_propagated = true;
  return iree_ok_status();
}

static bool loom_canonicalize_extract_consumes_static_empty_axis(
    const loom_module_t* module, const loom_op_t* op) {
  if (!loom_vector_extract_isa(op)) return false;
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_extract_source(op));
  if (!loom_type_is_vector(source_type)) return false;

  loom_attribute_t static_indices = loom_vector_extract_static_indices(op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) return false;

  uint8_t source_rank = loom_type_rank(source_type);
  uint16_t consumed_rank = static_indices.count;
  if (consumed_rank > source_rank) consumed_rank = source_rank;
  for (uint16_t axis = 0; axis < consumed_rank; ++axis) {
    if (loom_type_dim_is_dynamic_at(source_type, axis)) continue;
    if (loom_type_dim_static_size_at(source_type, axis) == 0) return true;
  }
  return false;
}

static iree_status_t loom_canonicalize_try_replace_empty_extract_with_poison(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_replaced) {
  *out_replaced = false;
  if (!loom_canonicalize_extract_consumes_static_empty_axis(rewriter->module,
                                                            op)) {
    return iree_ok_status();
  }
  loom_value_id_t result = loom_vector_extract_result(op);
  if (result == LOOM_VALUE_ID_INVALID) return iree_ok_status();
  loom_type_t result_type = loom_module_value_type(rewriter->module, result);
  if (!loom_canonicalize_value_type_has_poison(result_type)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_results_with_materialized_values_and_erase(
          rewriter, op, loom_poison_build));
  *out_replaced = true;
  return iree_ok_status();
}

static iree_status_t loom_canonicalize_try_fold_empty_accumulator_op(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_folded) {
  *out_folded = false;

  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  loom_value_id_t init = LOOM_VALUE_ID_INVALID;
  if (loom_vector_reduce_isa(op)) {
    input = loom_vector_reduce_input(op);
    init = loom_vector_reduce_init(op);
  } else if (loom_vector_dotf_isa(op)) {
    input = loom_vector_dotf_lhs(op);
    init = loom_vector_dotf_init(op);
  } else {
    return iree_ok_status();
  }

  if (!loom_canonicalize_value_is_static_empty_vector(rewriter->module,
                                                      input)) {
    return iree_ok_status();
  }

  loom_value_id_t replacement = init;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_folded = true;
  return iree_ok_status();
}

static bool loom_canonicalize_can_replace_results_with_empty(
    const loom_module_t* module, const loom_op_t* op) {
  if (loom_op_is_empty(op) || loom_op_is_poison(op)) return false;
  if (op->result_count == 0) return false;
  if (op->region_count != 0) return false;
  if (op->tied_result_count != 0) return false;

  loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE)) return false;

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) return false;
    loom_type_t type = loom_module_value_type(module, results[i]);
    if (!loom_type_has_empty_materializer(type)) return false;
  }
  return true;
}

static bool loom_canonicalize_single_empty_result(const loom_module_t* module,
                                                  const loom_op_t* op) {
  if (op->result_count != 1) return false;
  return loom_canonicalize_value_is_static_empty_vector(
      module, loom_op_const_results(op)[0]);
}

static bool loom_canonicalize_empty_value(const loom_module_t* module,
                                          loom_value_id_t value_id) {
  return loom_canonicalize_value_is_static_empty_vector(module, value_id);
}

static iree_status_t loom_canonicalize_try_elide_empty_memory_effect(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_elided) {
  *out_elided = false;

  bool replace_result_with_empty = false;
  bool erase_op = false;
  switch (op->kind) {
    case LOOM_OP_VECTOR_LOAD:
      replace_result_with_empty =
          loom_canonicalize_single_empty_result(rewriter->module, op);
      break;
    case LOOM_OP_VECTOR_LOAD_MASK:
      replace_result_with_empty =
          loom_canonicalize_single_empty_result(rewriter->module, op) &&
          loom_canonicalize_empty_value(rewriter->module,
                                        loom_vector_load_mask_mask(op)) &&
          loom_canonicalize_empty_value(rewriter->module,
                                        loom_vector_load_mask_passthrough(op));
      break;
    case LOOM_OP_VECTOR_LOAD_EXPAND:
      replace_result_with_empty =
          loom_canonicalize_single_empty_result(rewriter->module, op) &&
          loom_canonicalize_empty_value(rewriter->module,
                                        loom_vector_load_expand_mask(op)) &&
          loom_canonicalize_empty_value(
              rewriter->module, loom_vector_load_expand_passthrough(op));
      break;
    case LOOM_OP_VECTOR_GATHER:
      replace_result_with_empty =
          loom_canonicalize_single_empty_result(rewriter->module, op) &&
          loom_canonicalize_empty_value(rewriter->module,
                                        loom_vector_gather_offsets(op));
      break;
    case LOOM_OP_VECTOR_GATHER_MASK:
      replace_result_with_empty =
          loom_canonicalize_single_empty_result(rewriter->module, op) &&
          loom_canonicalize_empty_value(rewriter->module,
                                        loom_vector_gather_mask_offsets(op)) &&
          loom_canonicalize_empty_value(rewriter->module,
                                        loom_vector_gather_mask_mask(op)) &&
          loom_canonicalize_empty_value(
              rewriter->module, loom_vector_gather_mask_passthrough(op));
      break;
    case LOOM_OP_VECTOR_ATOMIC_RMW:
      replace_result_with_empty =
          loom_canonicalize_single_empty_result(rewriter->module, op) &&
          loom_canonicalize_empty_value(rewriter->module,
                                        loom_vector_atomic_rmw_value(op)) &&
          loom_canonicalize_empty_value(rewriter->module,
                                        loom_vector_atomic_rmw_offsets(op));
      break;
    case LOOM_OP_VECTOR_ATOMIC_RMW_MASK:
      replace_result_with_empty =
          loom_canonicalize_single_empty_result(rewriter->module, op) &&
          loom_canonicalize_empty_value(
              rewriter->module, loom_vector_atomic_rmw_mask_value(op)) &&
          loom_canonicalize_empty_value(
              rewriter->module, loom_vector_atomic_rmw_mask_offsets(op)) &&
          loom_canonicalize_empty_value(rewriter->module,
                                        loom_vector_atomic_rmw_mask_mask(op)) &&
          loom_canonicalize_empty_value(
              rewriter->module, loom_vector_atomic_rmw_mask_passthrough(op));
      break;
    case LOOM_OP_VECTOR_STORE:
      erase_op = loom_canonicalize_empty_value(rewriter->module,
                                               loom_vector_store_value(op));
      break;
    case LOOM_OP_VECTOR_STORE_MASK:
      erase_op = loom_canonicalize_empty_value(
                     rewriter->module, loom_vector_store_mask_value(op)) &&
                 loom_canonicalize_empty_value(rewriter->module,
                                               loom_vector_store_mask_mask(op));
      break;
    case LOOM_OP_VECTOR_STORE_COMPRESS:
      erase_op = loom_canonicalize_empty_value(
                     rewriter->module, loom_vector_store_compress_value(op)) &&
                 loom_canonicalize_empty_value(
                     rewriter->module, loom_vector_store_compress_mask(op));
      break;
    case LOOM_OP_VECTOR_SCATTER:
      erase_op = loom_canonicalize_empty_value(rewriter->module,
                                               loom_vector_scatter_value(op)) &&
                 loom_canonicalize_empty_value(rewriter->module,
                                               loom_vector_scatter_offsets(op));
      break;
    case LOOM_OP_VECTOR_SCATTER_MASK:
      erase_op = loom_canonicalize_empty_value(
                     rewriter->module, loom_vector_scatter_mask_value(op)) &&
                 loom_canonicalize_empty_value(
                     rewriter->module, loom_vector_scatter_mask_offsets(op)) &&
                 loom_canonicalize_empty_value(
                     rewriter->module, loom_vector_scatter_mask_mask(op));
      break;
    case LOOM_OP_VECTOR_ATOMIC_REDUCE:
      erase_op = loom_canonicalize_empty_value(
                     rewriter->module, loom_vector_atomic_reduce_value(op)) &&
                 loom_canonicalize_empty_value(
                     rewriter->module, loom_vector_atomic_reduce_offsets(op));
      break;
    case LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK:
      erase_op =
          loom_canonicalize_empty_value(
              rewriter->module, loom_vector_atomic_reduce_mask_value(op)) &&
          loom_canonicalize_empty_value(
              rewriter->module, loom_vector_atomic_reduce_mask_offsets(op)) &&
          loom_canonicalize_empty_value(
              rewriter->module, loom_vector_atomic_reduce_mask_mask(op));
      break;
    default:
      return iree_ok_status();
  }

  if (replace_result_with_empty) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_replace_results_with_materialized_values_and_erase(
            rewriter, op, loom_empty_build));
    *out_elided = true;
    return iree_ok_status();
  }
  if (erase_op) {
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
    *out_elided = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_canonicalize_try_elide_empty_vector_op(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_elided) {
  *out_elided = false;

  IREE_RETURN_IF_ERROR(loom_canonicalize_try_replace_empty_extract_with_poison(
      rewriter, op, out_elided));
  if (*out_elided) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_canonicalize_try_fold_empty_accumulator_op(
      rewriter, op, out_elided));
  if (*out_elided) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_canonicalize_try_elide_empty_memory_effect(
      rewriter, op, out_elided));
  if (*out_elided) return iree_ok_status();

  if (!loom_canonicalize_can_replace_results_with_empty(rewriter->module, op)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_results_with_materialized_values_and_erase(
          rewriter, op, loom_empty_build));
  *out_elided = true;
  return iree_ok_status();
}

iree_status_t loom_canonicalize_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  uint32_t max_iterations = LOOM_CANONICALIZE_DEFAULT_MAX_ITERATIONS;

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  rewriter.materialize_constant = loom_canonicalize_materialize_constant;
  iree_status_t status = loom_rewriter_enable_analysis(&rewriter, function);

  for (uint32_t iteration = 0;
       iree_status_is_ok(status) && iteration < max_iterations; ++iteration) {
    status = loom_rewriter_seed_function(&rewriter, function);
    if (!iree_status_is_ok(status)) break;
    bool any_changed = false;

    loom_op_t* op = NULL;
    while ((op = loom_rewriter_pop(&rewriter)) != NULL) {
      // Mini-DCE: erase trivially dead ops before fold/canonicalize.
      bool erased = false;
      iree_status_t dce_status =
          loom_rewriter_erase_if_dead(&rewriter, op, &erased);
      if (!iree_status_is_ok(dce_status)) {
        loom_rewriter_deinitialize(&rewriter);
        return dce_status;
      }
      if (erased) {
        any_changed = true;
        continue;
      }

      // Static empty vectors have a valid empty aggregate value, not poison.
      // Handle them before poison propagation so zero-lane computations and
      // zero-footprint memory effects disappear without observing operands.
      bool empty_elided = false;
      status = loom_canonicalize_try_elide_empty_vector_op(&rewriter, op,
                                                           &empty_elided);
      if (!iree_status_is_ok(status)) {
        loom_rewriter_deinitialize(&rewriter);
        return status;
      }
      if (empty_elided) {
        any_changed = true;
        if (pass->statistics) {
          loom_pass_statistic_add(pass, LOOM_CANONICALIZE_STAT_OPS_MODIFIED, 1);
        }
        continue;
      }

      // Poison propagation: pure scalar/vector-result ops with any poison
      // operand become typed poison values. Boundary diagnostics decide later
      // whether the remaining poison is observable.
      bool poison_propagated = false;
      status = loom_canonicalize_try_propagate_poison(&rewriter, op,
                                                      &poison_propagated);
      if (!iree_status_is_ok(status)) {
        loom_rewriter_deinitialize(&rewriter);
        return status;
      }
      if (poison_propagated) {
        any_changed = true;
        if (pass->statistics) {
          loom_pass_statistic_add(pass, LOOM_CANONICALIZE_STAT_OPS_MODIFIED, 1);
        }
        continue;
      }

      // Try fold: constant fold via facts and replace with constants.
      bool folded = false;
      status = loom_rewriter_try_fold(&rewriter, op, &folded);
      if (!iree_status_is_ok(status)) {
        loom_rewriter_deinitialize(&rewriter);
        return status;
      }
      if (folded) {
        any_changed = true;
        if (pass->statistics) {
          loom_pass_statistic_add(pass, LOOM_CANONICALIZE_STAT_OPS_MODIFIED, 1);
        }
        continue;
      }

      // Structural canonicalization patterns.
      const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
      if (!vtable || !vtable->canonicalize) continue;

      rewriter.flags = 0;
      status = vtable->canonicalize(op, &rewriter);
      if (!iree_status_is_ok(status)) {
        loom_rewriter_deinitialize(&rewriter);
        return status;
      }
      if (rewriter.flags & LOOM_REWRITER_FLAG_CHANGED) {
        any_changed = true;
        if (pass->statistics) {
          loom_pass_statistic_add(pass, LOOM_CANONICALIZE_STAT_OPS_MODIFIED, 1);
        }
      }
    }

    if (!any_changed) break;
  }

  loom_rewriter_deinitialize(&rewriter);
  return status;
}
