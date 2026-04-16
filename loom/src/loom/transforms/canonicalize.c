// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/canonicalize.h"

#include <string.h>

#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/special_values.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/rewriter.h"
#include "loom/transforms/type_propagation.h"
#include "loom/util/walk.h"

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

static iree_status_t loom_canonicalize_parse_option(void* user_data,
                                                    iree_string_view_t name,
                                                    iree_string_view_t value) {
  loom_canonicalizer_options_t* options =
      (loom_canonicalizer_options_t*)user_data;
  if (iree_string_view_equal(name, IREE_SV("max-iterations"))) {
    if (options->max_iterations != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate option 'max-iterations' for pass 'canonicalize'");
    }
    IREE_RETURN_IF_ERROR(loom_pass_option_parse_uint32(
        IREE_SV("canonicalize"), name, value, &options->max_iterations));
    if (options->max_iterations == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pass 'canonicalize' option 'max-iterations' must be greater than 0");
    }
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown option '%.*s' for pass 'canonicalize'",
                          (int)name.size, name.data);
}

iree_status_t loom_canonicalize_create(loom_pass_t* pass,
                                       iree_string_view_t options_string) {
  loom_canonicalizer_options_t* options = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena,
                                           sizeof(*options), (void**)&options));
  memset(options, 0, sizeof(*options));
  IREE_RETURN_IF_ERROR(loom_pass_options_parse(pass->info->name, options_string,
                                               loom_canonicalize_parse_option,
                                               options));
  pass->state = options;
  return iree_ok_status();
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

static iree_status_t loom_canonicalize_try_symbolic_integer_cmp(
    loom_rewriter_t* rewriter, loom_symbolic_expr_context_t* expression_context,
    loom_op_t* op, bool* out_changed) {
  *out_changed = false;
  if (!loom_index_cmp_isa(op) && !loom_scalar_cmpi_isa(op)) {
    return iree_ok_status();
  }

  loom_condition_integer_relation_t relation_storage[1];
  loom_condition_fact_set_t condition_facts;
  loom_condition_fact_set_initialize(
      relation_storage, IREE_ARRAYSIZE(relation_storage), &condition_facts);
  IREE_RETURN_IF_ERROR(loom_condition_facts_query(
      rewriter->module, &rewriter->fact_table, loom_op_const_results(op)[0],
      /*assumed_truth=*/true, &condition_facts));
  if (condition_facts.integer_relation_count != 1) {
    return iree_ok_status();
  }
  const loom_condition_integer_relation_t* relation =
      &condition_facts.integer_relations[0];
  if (relation->left.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE ||
      relation->right.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    return iree_ok_status();
  }

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_value_relation(
      expression_context, relation->relation, relation->left.value_id,
      relation->right.value_id, &proof));
  if (proof == LOOM_SYMBOLIC_PROOF_UNKNOWN) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_canonicalize_replace_single_result_with_exact_i64(
      rewriter, op, proof == LOOM_SYMBOLIC_PROOF_TRUE ? 1 : 0));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_canonicalize_try_symbolic_integer_cleanup(
    loom_rewriter_t* rewriter, loom_symbolic_expr_context_t* expression_context,
    loom_op_t* op, bool* out_changed) {
  *out_changed = false;

  IREE_RETURN_IF_ERROR(loom_canonicalize_try_symbolic_index_sub(
      rewriter, expression_context, op, out_changed));
  if (*out_changed) return iree_ok_status();

  return loom_canonicalize_try_symbolic_integer_cmp(
      rewriter, expression_context, op, out_changed);
}

//===----------------------------------------------------------------------===//
// Branch-edge fact materialization
//===----------------------------------------------------------------------===//

// Opportunistic branch-edge materialization has a local rewrite budget, not an
// IR legality limit. Over-budget facts remain true but unmaterialized; later
// canonicalization/fact passes still see the original program semantics.
#define LOOM_CANONICALIZE_EDGE_RELATION_CAPACITY 32
#define LOOM_CANONICALIZE_EDGE_CANDIDATE_CAPACITY 16
#define LOOM_CANONICALIZE_EDGE_PREDICATE_CAPACITY 64

typedef struct loom_canonicalize_edge_assume_candidate_t {
  // Value used before the branch whose edge-local replacement dominates the
  // region body.
  loom_value_id_t source;
  // Source type, reused for the assume result.
  loom_type_t type;
  // True when the source should be refined with index.assume instead of
  // scalar.assume.
  bool uses_index_assume;
  // First predicate in the flat predicate storage.
  uint16_t predicate_offset;
  // Number of predicates attached to the assume.
  uint16_t predicate_count;
  // True when the source has an operand use or rewritable type reference inside
  // the target region.
  bool has_region_use;
  // Region-entry assume op built for this source.
  loom_op_t* assume_op;
  // Result of assume_op that replaces region-local source uses.
  loom_value_id_t replacement;
} loom_canonicalize_edge_assume_candidate_t;

typedef struct loom_canonicalize_edge_assume_set_t {
  // Candidate sources requiring region-entry assumes.
  loom_canonicalize_edge_assume_candidate_t
      candidates[LOOM_CANONICALIZE_EDGE_CANDIDATE_CAPACITY];
  // Number of populated candidates.
  uint16_t candidate_count;
  // Flat predicate storage referenced by each candidate.
  loom_predicate_t predicates[LOOM_CANONICALIZE_EDGE_PREDICATE_CAPACITY];
  // Number of populated predicates.
  uint16_t predicate_count;
} loom_canonicalize_edge_assume_set_t;

static bool loom_canonicalize_type_uses_index_assume(loom_type_t type,
                                                     bool* out_uses_index) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    *out_uses_index = true;
    return true;
  }
  if (loom_scalar_type_is_integer(scalar_type)) {
    *out_uses_index = false;
    return true;
  }
  return false;
}

static bool loom_canonicalize_predicate_equal(const loom_predicate_t* lhs,
                                              const loom_predicate_t* rhs) {
  if (lhs->kind != rhs->kind || lhs->arg_count != rhs->arg_count) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->arg_count; ++i) {
    if (lhs->arg_tags[i] != rhs->arg_tags[i] || lhs->args[i] != rhs->args[i]) {
      return false;
    }
  }
  return true;
}

static bool loom_canonicalize_value_exact_integer(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id) {
  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, value_id);
  return loom_value_facts_is_exact(facts) && !loom_value_facts_is_float(facts);
}

static iree_status_t loom_canonicalize_edge_assume_set_append_predicate(
    loom_rewriter_t* rewriter, loom_canonicalize_edge_assume_set_t* assume_set,
    loom_value_id_t source, loom_predicate_t predicate) {
  if (source == LOOM_VALUE_ID_INVALID ||
      source >= rewriter->module->values.count) {
    return iree_ok_status();
  }
  if (loom_canonicalize_value_exact_integer(&rewriter->fact_table, source)) {
    return iree_ok_status();
  }

  loom_type_t source_type = loom_module_value_type(rewriter->module, source);
  bool uses_index_assume = false;
  if (!loom_canonicalize_type_uses_index_assume(source_type,
                                                &uses_index_assume)) {
    return iree_ok_status();
  }

  uint16_t candidate_index = 0;
  bool found_candidate = false;
  for (uint16_t i = 0; i < assume_set->candidate_count; ++i) {
    if (assume_set->candidates[i].source == source) {
      candidate_index = i;
      found_candidate = true;
      break;
    }
  }
  if (!found_candidate) {
    if (assume_set->candidate_count >=
        LOOM_CANONICALIZE_EDGE_CANDIDATE_CAPACITY) {
      return iree_ok_status();
    }
    candidate_index = assume_set->candidate_count++;
    assume_set->candidates[candidate_index] =
        (loom_canonicalize_edge_assume_candidate_t){
            .source = source,
            .type = source_type,
            .uses_index_assume = uses_index_assume,
            .predicate_offset = assume_set->predicate_count,
            .predicate_count = 0,
            .has_region_use = false,
            .assume_op = NULL,
            .replacement = LOOM_VALUE_ID_INVALID,
        };
  }

  loom_canonicalize_edge_assume_candidate_t* candidate =
      &assume_set->candidates[candidate_index];
  for (uint16_t i = 0; i < candidate->predicate_count; ++i) {
    const loom_predicate_t* existing =
        &assume_set->predicates[candidate->predicate_offset + i];
    if (loom_canonicalize_predicate_equal(existing, &predicate)) {
      return iree_ok_status();
    }
  }
  if (assume_set->predicate_count >=
      LOOM_CANONICALIZE_EDGE_PREDICATE_CAPACITY) {
    return iree_ok_status();
  }

  uint16_t insert_index =
      (uint16_t)(candidate->predicate_offset + candidate->predicate_count);
  if (insert_index < assume_set->predicate_count) {
    memmove(&assume_set->predicates[insert_index + 1],
            &assume_set->predicates[insert_index],
            ((iree_host_size_t)assume_set->predicate_count - insert_index) *
                sizeof(*assume_set->predicates));
    for (uint16_t i = 0; i < assume_set->candidate_count; ++i) {
      if (i != candidate_index &&
          assume_set->candidates[i].predicate_offset >= insert_index) {
        ++assume_set->candidates[i].predicate_offset;
      }
    }
  }
  assume_set->predicates[insert_index] = predicate;
  ++assume_set->predicate_count;
  ++candidate->predicate_count;
  return iree_ok_status();
}

static iree_status_t loom_canonicalize_edge_assume_set_append_relation(
    loom_rewriter_t* rewriter, loom_canonicalize_edge_assume_set_t* assume_set,
    const loom_condition_integer_relation_t* relation) {
  if (relation->left.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    loom_predicate_t predicate = {0};
    if (loom_condition_integer_relation_make_predicate_for_value(
            relation, &rewriter->fact_table, relation->left.value_id,
            &predicate)) {
      IREE_RETURN_IF_ERROR(loom_canonicalize_edge_assume_set_append_predicate(
          rewriter, assume_set, relation->left.value_id, predicate));
    }
  }
  if (relation->right.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    loom_predicate_t predicate = {0};
    if (loom_condition_integer_relation_make_predicate_for_value(
            relation, &rewriter->fact_table, relation->right.value_id,
            &predicate)) {
      IREE_RETURN_IF_ERROR(loom_canonicalize_edge_assume_set_append_predicate(
          rewriter, assume_set, relation->right.value_id, predicate));
    }
  }
  return iree_ok_status();
}

static bool loom_canonicalize_field_ref_matches_operand(
    const loom_op_vtable_t* vtable, loom_field_ref_t field_ref,
    uint16_t operand_index) {
  if (LOOM_FIELD_REF_CATEGORY(field_ref) != LOOM_FIELD_OPERAND) return false;
  uint8_t field_index = LOOM_FIELD_REF_INDEX(field_ref);
  if (field_index == operand_index) return true;
  if (!vtable || !vtable->operand_descriptors ||
      field_index < vtable->fixed_operand_count) {
    return false;
  }
  const loom_operand_descriptor_t* descriptor =
      &vtable->operand_descriptors[field_index];
  return iree_any_bit_set(descriptor->flags, LOOM_OPERAND_VARIADIC) &&
         operand_index >= field_index;
}

static bool loom_canonicalize_type_constraint_mentions_operand(
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint,
    uint16_t operand_index) {
  switch ((enum loom_constraint_property_e)constraint->property) {
    case LOOM_PROPERTY_TYPE:
    case LOOM_PROPERTY_ENCODING:
    case LOOM_PROPERTY_SHAPE:
      break;
    default:
      return false;
  }
  switch ((enum loom_constraint_relation_e)constraint->relation) {
    case LOOM_RELATION_PAIRWISE_EQ:
    case LOOM_RELATION_ALL_SAME:
    case LOOM_RELATION_REGION_ARG_MATCH:
    case LOOM_RELATION_YIELD_MATCH:
    case LOOM_RELATION_VARIADIC_MATCH:
      break;
    default:
      return false;
  }
  for (uint8_t i = 0; i < constraint->arg_count; ++i) {
    if (loom_canonicalize_field_ref_matches_operand(vtable, constraint->args[i],
                                                    operand_index)) {
      return true;
    }
  }
  return false;
}

static bool loom_canonicalize_op_type_constraints_mention_operand(
    const loom_op_vtable_t* vtable, uint16_t operand_index) {
  if (!vtable) return true;
  if (vtable->constraint_count > 0 && !vtable->constraints) return true;
  for (uint8_t i = 0; i < vtable->constraint_count; ++i) {
    if (loom_canonicalize_type_constraint_mentions_operand(
            vtable, &vtable->constraints[i], operand_index)) {
      return true;
    }
  }
  return false;
}

static bool loom_canonicalize_value_has_type_sensitive_use(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return true;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    loom_op_t* user = loom_use_user_op(uses[i]);
    if (!user || iree_any_bit_set(user->flags, LOOM_OP_FLAG_DEAD)) continue;
    const loom_op_vtable_t* vtable = loom_op_vtable(module, user);
    if (!vtable) return true;
    if (iree_any_bit_set(vtable->traits, LOOM_TRAIT_TERMINATOR)) {
      return true;
    }
    if (loom_canonicalize_op_type_constraints_mention_operand(
            vtable, loom_use_operand_index(uses[i]))) {
      return true;
    }
  }
  return false;
}

static bool loom_canonicalize_can_rewrite_edge_type_refs_for_result(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_id_t result_id) {
  if (result_id == LOOM_VALUE_ID_INVALID || result_id >= module->values.count) {
    return false;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !loom_traits_have_refinable_result_type_refs(vtable->traits)) {
    return false;
  }
  return !loom_canonicalize_value_has_type_sensitive_use(module, result_id);
}

static void loom_canonicalize_scan_type_for_edge_uses(
    loom_canonicalize_edge_assume_set_t* assume_set, loom_type_t type) {
  for (uint16_t candidate_index = 0;
       candidate_index < assume_set->candidate_count; ++candidate_index) {
    loom_canonicalize_edge_assume_candidate_t* candidate =
        &assume_set->candidates[candidate_index];
    if (loom_type_references_value(type, candidate->source)) {
      candidate->has_region_use = true;
    }
  }
}

typedef struct loom_canonicalize_region_use_scan_t {
  // Module owning the walked region.
  const loom_module_t* module;
  // Candidate set whose has_region_use flags are being populated.
  loom_canonicalize_edge_assume_set_t* assume_set;
} loom_canonicalize_region_use_scan_t;

static iree_status_t loom_canonicalize_scan_region_uses(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_canonicalize_region_use_scan_t* scan =
      (loom_canonicalize_region_use_scan_t*)user_data;
  if (loom_index_assume_isa(op) || loom_scalar_assume_isa(op)) {
    return iree_ok_status();
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t operand_index = 0; operand_index < op->operand_count;
       ++operand_index) {
    for (uint16_t candidate_index = 0;
         candidate_index < scan->assume_set->candidate_count;
         ++candidate_index) {
      loom_canonicalize_edge_assume_candidate_t* candidate =
          &scan->assume_set->candidates[candidate_index];
      if (operands[operand_index] == candidate->source) {
        candidate->has_region_use = true;
      }
    }
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t result_index = 0; result_index < op->result_count;
       ++result_index) {
    loom_value_id_t result = results[result_index];
    if (!loom_canonicalize_can_rewrite_edge_type_refs_for_result(scan->module,
                                                                 op, result)) {
      continue;
    }
    loom_canonicalize_scan_type_for_edge_uses(
        scan->assume_set, loom_module_value_type(scan->module, result));
  }
  return iree_ok_status();
}

static bool loom_canonicalize_is_edge_assume_op(
    const loom_canonicalize_edge_assume_set_t* assume_set,
    const loom_op_t* op) {
  for (uint16_t i = 0; i < assume_set->candidate_count; ++i) {
    if (assume_set->candidates[i].assume_op == op) return true;
  }
  return false;
}

typedef struct loom_canonicalize_region_replacement_t {
  // Rewriter used for operand updates.
  loom_rewriter_t* rewriter;
  // Candidate replacements to apply.
  loom_canonicalize_edge_assume_set_t* assume_set;
} loom_canonicalize_region_replacement_t;

static iree_status_t loom_canonicalize_rewrite_type_with_edge_assumes(
    loom_rewriter_t* rewriter,
    const loom_canonicalize_edge_assume_set_t* assume_set, loom_type_t type,
    loom_type_t* out_type, bool* out_changed) {
  *out_type = type;
  *out_changed = false;
  for (uint16_t candidate_index = 0;
       candidate_index < assume_set->candidate_count; ++candidate_index) {
    const loom_canonicalize_edge_assume_candidate_t* candidate =
        &assume_set->candidates[candidate_index];
    if (candidate->replacement == LOOM_VALUE_ID_INVALID) continue;
    loom_type_t rewritten_type = *out_type;
    bool candidate_changed = false;
    IREE_RETURN_IF_ERROR(loom_module_replace_type_value_references(
        rewriter->module, *out_type, candidate->source, candidate->replacement,
        &rewritten_type, &candidate_changed));
    if (candidate_changed) {
      *out_type = rewritten_type;
      *out_changed = true;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_canonicalize_replace_result_type_refs(
    loom_canonicalize_region_replacement_t* replacement, loom_op_t* op) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t result_index = 0; result_index < op->result_count;
       ++result_index) {
    loom_value_id_t result = results[result_index];
    if (!loom_canonicalize_can_rewrite_edge_type_refs_for_result(
            replacement->rewriter->module, op, result)) {
      continue;
    }
    loom_type_t old_type =
        loom_module_value_type(replacement->rewriter->module, result);
    loom_type_t new_type = old_type;
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_canonicalize_rewrite_type_with_edge_assumes(
        replacement->rewriter, replacement->assume_set, old_type, &new_type,
        &changed));
    if (changed) {
      IREE_RETURN_IF_ERROR(loom_rewriter_set_value_type(replacement->rewriter,
                                                        result, new_type));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_canonicalize_replace_region_uses(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_canonicalize_region_replacement_t* replacement =
      (loom_canonicalize_region_replacement_t*)user_data;
  if (loom_canonicalize_is_edge_assume_op(replacement->assume_set, op) ||
      loom_index_assume_isa(op) || loom_scalar_assume_isa(op)) {
    return iree_ok_status();
  }

  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t operand_index = 0; operand_index < op->operand_count;
       ++operand_index) {
    loom_value_id_t operand = operands[operand_index];
    for (uint16_t candidate_index = 0;
         candidate_index < replacement->assume_set->candidate_count;
         ++candidate_index) {
      const loom_canonicalize_edge_assume_candidate_t* candidate =
          &replacement->assume_set->candidates[candidate_index];
      if (operand == candidate->source &&
          candidate->replacement != LOOM_VALUE_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(
            replacement->rewriter, op, operand_index, candidate->replacement));
        break;
      }
    }
  }
  return loom_canonicalize_replace_result_type_refs(replacement, op);
}

static iree_status_t loom_canonicalize_materialize_edge_assumes(
    loom_rewriter_t* rewriter, loom_op_t* parent_op, loom_region_t* region,
    loom_canonicalize_edge_assume_set_t* assume_set, bool* out_changed) {
  *out_changed = false;
  if (!region || region->block_count == 0 || assume_set->candidate_count == 0) {
    return iree_ok_status();
  }

  loom_canonicalize_region_use_scan_t scan = {
      .module = rewriter->module,
      .assume_set = assume_set,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_region(
      rewriter->module, region, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_canonicalize_scan_region_uses, &scan},
      rewriter->arena, &walk_result));

  loom_block_t* entry_block = loom_region_entry_block(region);
  if (!entry_block || !entry_block->first_op) return iree_ok_status();

  iree_status_t status = iree_ok_status();
  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  loom_builder_set_before(&rewriter->builder, entry_block->first_op);
  for (uint16_t candidate_index = 0;
       iree_status_is_ok(status) &&
       candidate_index < assume_set->candidate_count;
       ++candidate_index) {
    loom_canonicalize_edge_assume_candidate_t* candidate =
        &assume_set->candidates[candidate_index];
    if (!candidate->has_region_use || candidate->predicate_count == 0) {
      continue;
    }
    loom_value_id_t value = candidate->source;
    loom_type_t result_type = candidate->type;
    const loom_predicate_t* source_predicates =
        &assume_set->predicates[candidate->predicate_offset];
    loom_predicate_t* predicates = NULL;
    status = iree_arena_allocate_array(
        &rewriter->module->arena, candidate->predicate_count,
        sizeof(loom_predicate_t), (void**)&predicates);
    if (!iree_status_is_ok(status)) break;
    memcpy(predicates, source_predicates,
           candidate->predicate_count * sizeof(loom_predicate_t));
    if (candidate->uses_index_assume) {
      status = loom_index_assume_build(
          &rewriter->builder, &value, 1, predicates, candidate->predicate_count,
          &result_type, 1, parent_op->location, &candidate->assume_op);
      if (!iree_status_is_ok(status)) break;
      candidate->replacement =
          loom_index_assume_results(candidate->assume_op).values[0];
    } else {
      status = loom_scalar_assume_build(
          &rewriter->builder, &value, 1, predicates, candidate->predicate_count,
          &result_type, 1, parent_op->location, &candidate->assume_op);
      if (!iree_status_is_ok(status)) break;
      candidate->replacement =
          loom_scalar_assume_results(candidate->assume_op).values[0];
    }
    *out_changed = true;
  }
  loom_builder_restore(&rewriter->builder, saved_ip);
  if (!iree_status_is_ok(status)) return status;

  if (!*out_changed) return iree_ok_status();
  loom_canonicalize_region_replacement_t replacement = {
      .rewriter = rewriter,
      .assume_set = assume_set,
  };
  return loom_walk_region(
      rewriter->module, region, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_canonicalize_replace_region_uses,
                             &replacement},
      rewriter->arena, &walk_result);
}

static iree_status_t loom_canonicalize_materialize_condition_facts_in_region(
    loom_rewriter_t* rewriter, loom_op_t* parent_op, loom_region_t* region,
    loom_value_id_t condition, bool assumed_truth, bool* out_changed) {
  *out_changed = false;
  loom_condition_integer_relation_t
      relation_storage[LOOM_CANONICALIZE_EDGE_RELATION_CAPACITY];
  loom_condition_fact_set_t condition_facts;
  loom_condition_fact_set_initialize(
      relation_storage, IREE_ARRAYSIZE(relation_storage), &condition_facts);
  IREE_RETURN_IF_ERROR(
      loom_condition_facts_query(rewriter->module, &rewriter->fact_table,
                                 condition, assumed_truth, &condition_facts));
  if (condition_facts.integer_relation_count == 0) return iree_ok_status();

  loom_canonicalize_edge_assume_set_t assume_set = {0};
  for (iree_host_size_t relation_index = 0;
       relation_index < condition_facts.integer_relation_count;
       ++relation_index) {
    IREE_RETURN_IF_ERROR(loom_canonicalize_edge_assume_set_append_relation(
        rewriter, &assume_set,
        &condition_facts.integer_relations[relation_index]));
  }
  return loom_canonicalize_materialize_edge_assumes(rewriter, parent_op, region,
                                                    &assume_set, out_changed);
}

static iree_status_t loom_canonicalize_materialize_selector_case_fact_in_region(
    loom_rewriter_t* rewriter, loom_op_t* parent_op, loom_region_t* region,
    loom_value_id_t selector, int64_t case_key, bool* out_changed) {
  *out_changed = false;
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_EQ,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_NONE},
      .args = {selector, case_key, 0},
  };
  loom_canonicalize_edge_assume_set_t assume_set = {0};
  IREE_RETURN_IF_ERROR(loom_canonicalize_edge_assume_set_append_predicate(
      rewriter, &assume_set, selector, predicate));
  return loom_canonicalize_materialize_edge_assumes(rewriter, parent_op, region,
                                                    &assume_set, out_changed);
}

static iree_status_t
loom_canonicalize_materialize_selector_default_facts_in_region(
    loom_rewriter_t* rewriter, loom_op_t* parent_op, loom_region_t* region,
    loom_value_id_t selector, loom_attribute_t case_keys, bool* out_changed) {
  *out_changed = false;
  loom_canonicalize_edge_assume_set_t assume_set = {0};
  for (uint16_t i = 0; i < case_keys.count; ++i) {
    loom_predicate_t predicate = {
        .kind = LOOM_PREDICATE_NE,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_NONE},
        .args = {selector, case_keys.i64_array[i], 0},
    };
    IREE_RETURN_IF_ERROR(loom_canonicalize_edge_assume_set_append_predicate(
        rewriter, &assume_set, selector, predicate));
  }
  return loom_canonicalize_materialize_edge_assumes(rewriter, parent_op, region,
                                                    &assume_set, out_changed);
}

static iree_status_t loom_canonicalize_try_materialize_branch_edge_facts(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_changed) {
  *out_changed = false;
  if (loom_scf_if_isa(op)) {
    bool then_changed = false;
    IREE_RETURN_IF_ERROR(
        loom_canonicalize_materialize_condition_facts_in_region(
            rewriter, op, loom_scf_if_then_region(op),
            loom_scf_if_condition(op), true, &then_changed));
    bool else_changed = false;
    IREE_RETURN_IF_ERROR(
        loom_canonicalize_materialize_condition_facts_in_region(
            rewriter, op, loom_scf_if_else_region(op),
            loom_scf_if_condition(op), false, &else_changed));
    *out_changed = then_changed || else_changed;
    return iree_ok_status();
  }
  if (!loom_scf_switch_isa(op)) return iree_ok_status();

  loom_attribute_t case_keys = loom_scf_switch_case_keys(op);
  if (case_keys.kind != LOOM_ATTR_I64_ARRAY ||
      (case_keys.count > 0 && !case_keys.i64_array)) {
    return iree_ok_status();
  }
  loom_region_slice_t case_regions = loom_scf_switch_case_regions(op);
  if (case_regions.count != case_keys.count) return iree_ok_status();

  loom_value_id_t selector = loom_scf_switch_selector(op);
  for (uint16_t i = 0; i < case_keys.count; ++i) {
    bool case_changed = false;
    IREE_RETURN_IF_ERROR(
        loom_canonicalize_materialize_selector_case_fact_in_region(
            rewriter, op, case_regions.regions[i], selector,
            case_keys.i64_array[i], &case_changed));
    *out_changed |= case_changed;
  }
  bool default_changed = false;
  IREE_RETURN_IF_ERROR(
      loom_canonicalize_materialize_selector_default_facts_in_region(
          rewriter, op, loom_scf_switch_default_region(op), selector, case_keys,
          &default_changed));
  *out_changed |= default_changed;
  return iree_ok_status();
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
  rewriter->materialize_constant = loom_constant_build;

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
      status = loom_canonicalize_try_symbolic_integer_cleanup(
          rewriter, &expression_context, op, &symbolic_changed);
      if (!iree_status_is_ok(status)) break;
      if (symbolic_changed) {
        any_changed = true;
        loom_canonicalizer_record_rewrite(out_result, rewriter,
                                          /*count_modified_op=*/true);
        loom_symbolic_expr_context_reset(&expression_context);
        continue;
      }

      // Branch-edge fact materialization makes predicate implications visible
      // as SSA assume values inside each region. Keeping this in the driver
      // lets the transform use index/scalar assume ops without making scf
      // depend on those dialects.
      bool edge_facts_materialized = false;
      rewriter->flags = 0;
      status = loom_canonicalize_try_materialize_branch_edge_facts(
          rewriter, op, &edge_facts_materialized);
      if (!iree_status_is_ok(status)) break;
      if (edge_facts_materialized) {
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
      &canonicalizer, function,
      (const loom_canonicalizer_options_t*)pass->state, &result);
  if (iree_status_is_ok(status) && pass->statistics) {
    loom_pass_statistic_add(pass, LOOM_CANONICALIZE_STAT_OPS_MODIFIED,
                            result.ops_modified);
  }
  loom_canonicalizer_deinitialize(&canonicalizer);
  return status;
}
