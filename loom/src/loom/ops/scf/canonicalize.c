// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/scf/ops.h"
#include "loom/transforms/rewriter.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool loom_scf_value_facts_are_exact_i64(loom_rewriter_t* rewriter,
                                               loom_value_id_t value,
                                               int64_t* out_value) {
  loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, value);
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_scf_value_facts_are_exact_bool(loom_rewriter_t* rewriter,
                                                loom_value_id_t value,
                                                bool* out_value) {
  int64_t integer_value = 0;
  if (!loom_scf_value_facts_are_exact_i64(rewriter, value, &integer_value)) {
    return false;
  }
  *out_value = integer_value != 0;
  return true;
}

static loom_op_t* loom_scf_region_terminator(loom_region_t* region) {
  if (!region || region->block_count != 1) return NULL;
  loom_block_t* block = loom_region_entry_block(region);
  if (!block || !block->last_op || !loom_scf_yield_isa(block->last_op)) {
    return NULL;
  }
  return block->last_op;
}

static iree_status_t loom_scf_replace_results_and_erase(
    loom_op_t* op, loom_rewriter_t* rewriter,
    const loom_value_id_t* replacements, uint16_t replacement_count) {
  if (replacement_count != op->result_count) return iree_ok_status();
  if (op->result_count == 0) return loom_rewriter_erase(rewriter, op);
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, replacements,
                                                  replacement_count);
}

//===----------------------------------------------------------------------===//
// scf.if
//===----------------------------------------------------------------------===//

iree_status_t loom_scf_if_canonicalize(loom_op_t* op,
                                       loom_rewriter_t* rewriter) {
  bool condition = false;
  if (!loom_scf_value_facts_are_exact_bool(rewriter, loom_scf_if_condition(op),
                                           &condition)) {
    return iree_ok_status();
  }

  loom_region_t* selected_region =
      condition ? loom_scf_if_then_region(op) : loom_scf_if_else_region(op);
  loom_op_t* yield = loom_scf_region_terminator(selected_region);
  if (!yield) return iree_ok_status();

  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  if (yielded_values.count != op->result_count) return iree_ok_status();

  loom_block_t* selected_block = loom_region_entry_block(selected_region);
  loom_op_t* child_op = selected_block->first_op;
  while (child_op && child_op != yield) {
    loom_op_t* next_child_op = child_op->next_op;
    IREE_RETURN_IF_ERROR(loom_rewriter_move_before(rewriter, child_op, op));
    child_op = next_child_op;
  }

  return loom_scf_replace_results_and_erase(op, rewriter, yielded_values.values,
                                            yielded_values.count);
}

//===----------------------------------------------------------------------===//
// scf.for
//===----------------------------------------------------------------------===//

static bool loom_scf_for_has_zero_trip_count(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  loom_value_facts_t lower_bound =
      loom_rewriter_value_facts(rewriter, loom_scf_for_lower_bound(op));
  loom_value_facts_t upper_bound =
      loom_rewriter_value_facts(rewriter, loom_scf_for_upper_bound(op));
  if (loom_value_facts_is_float(lower_bound) ||
      loom_value_facts_is_float(upper_bound)) {
    return false;
  }
  if (loom_value_facts_is_exact(lower_bound) &&
      loom_value_facts_is_exact(upper_bound) &&
      lower_bound.range_lo == upper_bound.range_lo) {
    return true;
  }

  loom_value_facts_t step =
      loom_rewriter_value_facts(rewriter, loom_scf_for_step(op));
  if (!loom_value_facts_is_positive(step)) return false;
  return lower_bound.range_lo >= upper_bound.range_hi;
}

static bool loom_scf_for_step_is_positive(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_facts_t step =
      loom_rewriter_value_facts(rewriter, loom_scf_for_step(op));
  return !loom_value_facts_is_float(step) && loom_value_facts_is_positive(step);
}

static bool loom_scf_for_yields_loop_carried_args(loom_op_t* op) {
  loom_region_t* body = loom_scf_for_body(op);
  loom_op_t* yield = loom_scf_region_terminator(body);
  if (!yield) return false;

  loom_block_t* block = loom_region_entry_block(body);
  if (block->first_op != yield) return false;
  if (op->result_count == 0) return true;

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  if (iter_args.count != op->result_count ||
      yielded_values.count != op->result_count) {
    return false;
  }
  if (block->arg_count < 1 + op->result_count) return false;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (yielded_values.values[i] != loom_block_arg_id(block, 1 + i)) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_scf_for_canonicalize(loom_op_t* op,
                                        loom_rewriter_t* rewriter) {
  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  if (loom_scf_for_has_zero_trip_count(op, rewriter)) {
    return loom_scf_replace_results_and_erase(op, rewriter, iter_args.values,
                                              iter_args.count);
  }
  if (loom_scf_for_step_is_positive(op, rewriter) &&
      loom_scf_for_yields_loop_carried_args(op)) {
    return loom_scf_replace_results_and_erase(op, rewriter, iter_args.values,
                                              iter_args.count);
  }
  return iree_ok_status();
}
