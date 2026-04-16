// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/canonicalize.h"

#include <string.h>

#include "loom/analysis/symbolic_expr.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/special_values.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/rewriter.h"
#include "loom/transforms/type_propagation.h"

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
  } else if (loom_type_is_scalar(result_type) &&
             loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1) {
    attr = loom_attr_bool(facts.range_lo != 0);
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

static iree_status_t loom_canonicalize_replace_single_result_with_value(
    loom_rewriter_t* rewriter, loom_op_t* op, loom_value_id_t replacement) {
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_canonicalize_replace_single_result_with_exact_i64(
    loom_rewriter_t* rewriter, loom_op_t* op, int64_t value) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t result = loom_op_const_results(op)[0];
  loom_type_t result_type = loom_module_value_type(rewriter->module, result);

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_build_constant(rewriter, loom_value_facts_exact_i64(value),
                                   result_type, op->location, &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_canonicalize_replace_single_result_with_value(rewriter, op,
                                                            replacement);
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

static bool loom_canonicalize_symbolic_relation_from_index_predicate(
    loom_rewriter_t* rewriter, uint8_t predicate, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_symbolic_integer_relation_t* out_relation) {
  switch ((loom_index_cmp_predicate_t)predicate) {
    case LOOM_INDEX_CMP_PREDICATE_EQ:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_NE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_NE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SLT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SLE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SGT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SGE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_ULT:
    case LOOM_INDEX_CMP_PREDICATE_ULE:
    case LOOM_INDEX_CMP_PREDICATE_UGT:
    case LOOM_INDEX_CMP_PREDICATE_UGE:
      if (!loom_value_facts_is_non_negative(
              loom_rewriter_value_facts(rewriter, lhs)) ||
          !loom_value_facts_is_non_negative(
              loom_rewriter_value_facts(rewriter, rhs))) {
        return false;
      }
      if (predicate == LOOM_INDEX_CMP_PREDICATE_ULT) {
        *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      } else if (predicate == LOOM_INDEX_CMP_PREDICATE_ULE) {
        *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      } else if (predicate == LOOM_INDEX_CMP_PREDICATE_UGT) {
        *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      } else {
        *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      }
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_canonicalize_try_symbolic_index_sub(
    loom_rewriter_t* rewriter, loom_symbolic_expr_context_t* expression_context,
    loom_op_t* op, bool* out_changed) {
  *out_changed = false;
  if (!loom_index_sub_isa(op)) return iree_ok_status();

  loom_symbolic_value_difference_t difference = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_simplify_value_difference(
      expression_context, loom_index_sub_lhs(op), loom_index_sub_rhs(op),
      &difference));
  switch (difference.kind) {
    case LOOM_SYMBOLIC_VALUE_DIFFERENCE_CONSTANT: {
      IREE_RETURN_IF_ERROR(
          loom_canonicalize_replace_single_result_with_exact_i64(
              rewriter, op, difference.constant));
      *out_changed = true;
      return iree_ok_status();
    }
    case LOOM_SYMBOLIC_VALUE_DIFFERENCE_VALUE: {
      loom_type_t result_type = loom_module_value_type(
          rewriter->module, loom_op_const_results(op)[0]);
      loom_type_t replacement_type =
          loom_module_value_type(rewriter->module, difference.value_id);
      if (!loom_type_equal(result_type, replacement_type)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_canonicalize_replace_single_result_with_value(
          rewriter, op, difference.value_id));
      *out_changed = true;
      return iree_ok_status();
    }
    case LOOM_SYMBOLIC_VALUE_DIFFERENCE_UNKNOWN:
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_canonicalize_try_symbolic_index_cmp(
    loom_rewriter_t* rewriter, loom_symbolic_expr_context_t* expression_context,
    loom_op_t* op, bool* out_changed) {
  *out_changed = false;
  if (!loom_index_cmp_isa(op)) return iree_ok_status();

  loom_value_id_t lhs = loom_index_cmp_lhs(op);
  loom_value_id_t rhs = loom_index_cmp_rhs(op);
  loom_symbolic_integer_relation_t relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  if (!loom_canonicalize_symbolic_relation_from_index_predicate(
          rewriter, loom_index_cmp_predicate(op), lhs, rhs, &relation)) {
    return iree_ok_status();
  }

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_value_relation(
      expression_context, relation, lhs, rhs, &proof));
  if (proof == LOOM_SYMBOLIC_PROOF_UNKNOWN) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_canonicalize_replace_single_result_with_exact_i64(
      rewriter, op, proof == LOOM_SYMBOLIC_PROOF_TRUE ? 1 : 0));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_canonicalize_try_symbolic_index_cleanup(
    loom_rewriter_t* rewriter, loom_symbolic_expr_context_t* expression_context,
    loom_op_t* op, bool* out_changed) {
  *out_changed = false;

  IREE_RETURN_IF_ERROR(loom_canonicalize_try_symbolic_index_sub(
      rewriter, expression_context, op, out_changed));
  if (*out_changed) return iree_ok_status();

  return loom_canonicalize_try_symbolic_index_cmp(rewriter, expression_context,
                                                  op, out_changed);
}

struct loom_canonicalizer_state_t {
  // Rewriter state for the most recent run. Its fact table remains queryable
  // until the next run or deinitialize.
  loom_rewriter_t rewriter;

  // True when rewriter currently owns live worklist bits that must be cleared
  // before resetting scratch storage.
  bool rewriter_initialized;
};

static void loom_canonicalizer_reset_run_state(
    loom_canonicalizer_t* canonicalizer) {
  if (canonicalizer->state && canonicalizer->state->rewriter_initialized) {
    loom_rewriter_deinitialize(&canonicalizer->state->rewriter);
    canonicalizer->state->rewriter_initialized = false;
  }
  if (canonicalizer->scratch_arena_initialized) {
    iree_arena_reset(&canonicalizer->scratch_arena);
  }
}

static void loom_canonicalizer_record_rewriter_flags(
    loom_canonicalizer_result_t* result, const loom_rewriter_t* rewriter) {
  if (!result) return;
  if (iree_any_bit_set(rewriter->flags, LOOM_REWRITER_FLAG_FACTS_CHANGED)) {
    result->facts_changed = true;
  }
  if (iree_any_bit_set(rewriter->flags, LOOM_REWRITER_FLAG_TYPE_CHANGED)) {
    result->types_changed = true;
  }
}

static void loom_canonicalizer_record_rewrite(
    loom_canonicalizer_result_t* result, const loom_rewriter_t* rewriter,
    bool count_modified_op) {
  if (!result) return;
  result->changed = true;
  loom_canonicalizer_record_rewriter_flags(result, rewriter);
  if (count_modified_op) ++result->ops_modified;
}

iree_status_t loom_canonicalizer_initialize(
    loom_module_t* module, iree_arena_allocator_t* parent_arena,
    loom_canonicalizer_t* out_canonicalizer) {
  if (!module || !parent_arena || !parent_arena->block_pool ||
      !out_canonicalizer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, parent arena with block pool, and output canonicalizer are "
        "required");
  }
  memset(out_canonicalizer, 0, sizeof(*out_canonicalizer));
  loom_canonicalizer_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(parent_arena, sizeof(*state), (void**)&state));
  memset(state, 0, sizeof(*state));
  out_canonicalizer->module = module;
  out_canonicalizer->parent_arena = parent_arena;
  out_canonicalizer->state = state;
  iree_arena_initialize(parent_arena->block_pool,
                        &out_canonicalizer->scratch_arena);
  out_canonicalizer->scratch_arena_initialized = true;
  return iree_ok_status();
}

void loom_canonicalizer_deinitialize(loom_canonicalizer_t* canonicalizer) {
  if (!canonicalizer) return;
  loom_canonicalizer_reset_run_state(canonicalizer);
  if (canonicalizer->scratch_arena_initialized) {
    iree_arena_deinitialize(&canonicalizer->scratch_arena);
  }
  memset(canonicalizer, 0, sizeof(*canonicalizer));
}

const loom_value_fact_table_t* loom_canonicalizer_fact_table(
    const loom_canonicalizer_t* canonicalizer) {
  if (!canonicalizer || !canonicalizer->state ||
      !canonicalizer->state->rewriter_initialized) {
    return NULL;
  }
  if (!canonicalizer->state->rewriter.fact_table.entries) return NULL;
  return &canonicalizer->state->rewriter.fact_table;
}

iree_status_t loom_canonicalizer_run_function(
    loom_canonicalizer_t* canonicalizer, loom_func_like_t function,
    const loom_canonicalizer_options_t* options,
    loom_canonicalizer_result_t* out_result) {
  if (!canonicalizer || !canonicalizer->module || !canonicalizer->state) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "initialized canonicalizer required");
  }
  if (out_result) memset(out_result, 0, sizeof(*out_result));
  loom_canonicalizer_reset_run_state(canonicalizer);
  if (!loom_func_like_body(function)) return iree_ok_status();

  uint32_t max_iterations = options && options->max_iterations > 0
                                ? options->max_iterations
                                : LOOM_CANONICALIZER_DEFAULT_MAX_ITERATIONS;

  loom_rewriter_t* rewriter = &canonicalizer->state->rewriter;
  IREE_RETURN_IF_ERROR(loom_rewriter_initialize(rewriter, canonicalizer->module,
                                                &canonicalizer->scratch_arena));
  canonicalizer->state->rewriter_initialized = true;
  rewriter->materialize_constant = loom_canonicalize_materialize_constant;

  iree_status_t status = loom_rewriter_enable_analysis_with_seed_facts(
      rewriter, function, options ? options->seed_facts : NULL);
  loom_symbolic_expr_context_t expression_context;
  if (iree_status_is_ok(status)) {
    loom_symbolic_expr_context_initialize(
        canonicalizer->module, &rewriter->fact_table,
        &canonicalizer->scratch_arena, &expression_context);
  }
  loom_type_propagator_t* type_propagator = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_type_propagator_allocate(
        canonicalizer->module, &canonicalizer->scratch_arena, &type_propagator);
  }

  for (uint32_t iteration = 0;
       iree_status_is_ok(status) && iteration < max_iterations; ++iteration) {
    status = loom_type_propagator_prepare_function(type_propagator, function);
    if (!iree_status_is_ok(status)) break;
    status = loom_rewriter_seed_function(rewriter, function);
    if (!iree_status_is_ok(status)) break;
    bool any_changed = false;

    loom_op_t* op = NULL;
    while ((op = loom_rewriter_pop(rewriter)) != NULL) {
      // Mini-DCE: erase trivially dead ops before fold/canonicalize.
      bool erased = false;
      rewriter->flags = 0;
      iree_status_t dce_status =
          loom_rewriter_erase_if_dead(rewriter, op, &erased);
      if (!iree_status_is_ok(dce_status)) {
        status = dce_status;
        break;
      }
      if (erased) {
        any_changed = true;
        loom_canonicalizer_record_rewrite(out_result, rewriter,
                                          /*count_modified_op=*/false);
        loom_symbolic_expr_context_reset(&expression_context);
        continue;
      }

      // Static empty vectors have a valid empty aggregate value, not poison.
      // Handle them before poison propagation so zero-lane computations and
      // zero-footprint memory effects disappear without observing operands.
      bool empty_elided = false;
      rewriter->flags = 0;
      status = loom_canonicalize_try_elide_empty_vector_op(rewriter, op,
                                                           &empty_elided);
      if (!iree_status_is_ok(status)) break;
      if (empty_elided) {
        any_changed = true;
        loom_canonicalizer_record_rewrite(out_result, rewriter,
                                          /*count_modified_op=*/true);
        loom_symbolic_expr_context_reset(&expression_context);
        continue;
      }

      // Poison propagation: pure scalar/vector-result ops with any poison
      // operand become typed poison values. Boundary diagnostics decide later
      // whether the remaining poison is observable.
      bool poison_propagated = false;
      rewriter->flags = 0;
      status = loom_canonicalize_try_propagate_poison(rewriter, op,
                                                      &poison_propagated);
      if (!iree_status_is_ok(status)) break;
      if (poison_propagated) {
        any_changed = true;
        loom_canonicalizer_record_rewrite(out_result, rewriter,
                                          /*count_modified_op=*/true);
        loom_symbolic_expr_context_reset(&expression_context);
        continue;
      }

      // Try fold: constant fold via facts and replace with constants.
      bool folded = false;
      rewriter->flags = 0;
      status = loom_rewriter_try_fold(rewriter, op, &folded);
      loom_canonicalizer_record_rewriter_flags(out_result, rewriter);
      if (!iree_status_is_ok(status)) break;
      if (folded) {
        any_changed = true;
        loom_canonicalizer_record_rewrite(out_result, rewriter,
                                          /*count_modified_op=*/true);
        loom_symbolic_expr_context_reset(&expression_context);
        continue;
      }

      // Table-driven type propagation: generated equality constraints and
      // value facts can narrow dynamic shapes, encoding roles, and static
      // attachments without hand-writing one pattern per op family.
      bool types_propagated = false;
      rewriter->flags = 0;
      status = loom_type_propagator_apply_op(type_propagator, rewriter, op,
                                             &types_propagated);
      if (!iree_status_is_ok(status)) break;
      if (types_propagated) {
        any_changed = true;
        loom_canonicalizer_record_rewrite(out_result, rewriter,
                                          /*count_modified_op=*/true);
        loom_symbolic_expr_context_reset(&expression_context);
        continue;
      }

      // Symbolic address-domain cleanup uses the generic expression analysis
      // for exact linear cancellation and relation proofs that are awkward as
      // op-local patterns.
      bool symbolic_changed = false;
      rewriter->flags = 0;
      status = loom_canonicalize_try_symbolic_index_cleanup(
          rewriter, &expression_context, op, &symbolic_changed);
      if (!iree_status_is_ok(status)) break;
      if (symbolic_changed) {
        any_changed = true;
        loom_canonicalizer_record_rewrite(out_result, rewriter,
                                          /*count_modified_op=*/true);
        loom_symbolic_expr_context_reset(&expression_context);
        continue;
      }

      // Structural canonicalization patterns.
      const loom_op_vtable_t* vtable =
          loom_op_vtable(canonicalizer->module, op);
      if (!vtable || !vtable->canonicalize) continue;

      rewriter->flags = 0;
      status = vtable->canonicalize(op, rewriter);
      if (!iree_status_is_ok(status)) break;
      if (rewriter->flags & LOOM_REWRITER_FLAG_CHANGED) {
        any_changed = true;
        loom_canonicalizer_record_rewrite(out_result, rewriter,
                                          /*count_modified_op=*/true);
        loom_symbolic_expr_context_reset(&expression_context);
      } else {
        loom_canonicalizer_record_rewriter_flags(out_result, rewriter);
      }
    }

    if (!any_changed) break;
  }

  if (out_result) {
    out_result->boundary_maybe_changed = out_result->changed ||
                                         out_result->facts_changed ||
                                         out_result->types_changed;
  }
  if (!iree_status_is_ok(status)) {
    loom_rewriter_deinitialize(rewriter);
    canonicalizer->state->rewriter_initialized = false;
    iree_arena_reset(&canonicalizer->scratch_arena);
  }
  return status;
}

iree_status_t loom_canonicalize_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function) {
  loom_canonicalizer_t canonicalizer;
  IREE_RETURN_IF_ERROR(
      loom_canonicalizer_initialize(module, pass->arena, &canonicalizer));

  loom_canonicalizer_result_t result;
  iree_status_t status = loom_canonicalizer_run_function(
      &canonicalizer, function, /*options=*/NULL, &result);
  if (iree_status_is_ok(status) && pass->statistics) {
    loom_pass_statistic_add(pass, LOOM_CANONICALIZE_STAT_OPS_MODIFIED,
                            result.ops_modified);
  }
  loom_canonicalizer_deinitialize(&canonicalizer);
  return status;
}
