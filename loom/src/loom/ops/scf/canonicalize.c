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

static void loom_scf_preserve_value_name(loom_module_t* module,
                                         loom_value_id_t old_value_id,
                                         loom_value_id_t new_value_id) {
  if (old_value_id == LOOM_VALUE_ID_INVALID ||
      new_value_id == LOOM_VALUE_ID_INVALID) {
    return;
  }
  loom_value_t* old_value = loom_module_value(module, old_value_id);
  loom_value_t* new_value = loom_module_value(module, new_value_id);
  if (old_value->name_id == LOOM_STRING_ID_INVALID ||
      new_value->name_id != LOOM_STRING_ID_INVALID) {
    return;
  }
  new_value->name_id = old_value->name_id;
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

static iree_status_t loom_scf_for_adjust_tied_results(
    loom_op_t* op, const uint16_t* result_map, const uint16_t* iter_arg_map,
    loom_rewriter_t* rewriter, loom_tied_result_t** out_tied_results,
    uint16_t* out_tied_result_count, bool* out_supported) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  *out_supported = true;
  if (op->tied_result_count == 0) return iree_ok_status();

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->tied_result_count,
                                sizeof(*tied_results), (void**)&tied_results));

  const loom_tied_result_t* old_tied_results = loom_op_tied_results(op);
  uint16_t tied_result_count = 0;
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    loom_tied_result_t tied_result = old_tied_results[i];
    if (tied_result.result_index >= op->result_count) {
      *out_supported = false;
      return iree_ok_status();
    }
    if (tied_result.operand_index >= op->operand_count) {
      *out_supported = false;
      return iree_ok_status();
    }
    uint16_t new_result_index = result_map[tied_result.result_index];
    if (new_result_index == UINT16_MAX) continue;

    uint16_t new_operand_index = tied_result.operand_index;
    if (new_operand_index >= 3) {
      uint16_t old_iter_arg_index = (uint16_t)(new_operand_index - 3);
      if (old_iter_arg_index >= op->result_count) {
        *out_supported = false;
        return iree_ok_status();
      }
      uint16_t new_iter_arg_index = iter_arg_map[old_iter_arg_index];
      if (new_iter_arg_index == UINT16_MAX) {
        *out_supported = false;
        return iree_ok_status();
      }
      new_operand_index = (uint16_t)(3 + new_iter_arg_index);
    }

    tied_results[tied_result_count++] = (loom_tied_result_t){
        .result_index = new_result_index,
        .operand_index = new_operand_index,
        .has_type_change = tied_result.has_type_change,
    };
  }

  *out_tied_results = tied_results;
  *out_tied_result_count = tied_result_count;
  return iree_ok_status();
}

// Compacts loop-carried slots whose yield operand is exactly the corresponding
// iter_arg block argument. The loop is rebuilt because scf.for operands, body
// block arguments, results, and tied-result metadata all share the carried slot
// numbering.
static iree_status_t loom_scf_for_forward_loop_carried_results(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  if (!loom_scf_for_step_is_positive(op, rewriter) || op->result_count == 0) {
    return iree_ok_status();
  }

  loom_region_t* body = loom_scf_for_body(op);
  loom_op_t* yield = loom_scf_region_terminator(body);
  if (!yield) return iree_ok_status();

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  if (iter_args.count != op->result_count ||
      yielded_values.count != op->result_count) {
    return iree_ok_status();
  }

  loom_block_t* old_block = loom_region_entry_block(body);
  if (old_block->arg_count < 1 + op->result_count) return iree_ok_status();

  bool* forwarded_results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, op->result_count, sizeof(*forwarded_results),
      (void**)&forwarded_results));
  uint16_t forwarded_count = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t carried_arg =
        loom_block_arg_id(old_block, (uint16_t)(1 + i));
    forwarded_results[i] = yielded_values.values[i] == carried_arg;
    if (forwarded_results[i]) ++forwarded_count;
  }
  if (forwarded_count == 0) return iree_ok_status();

  uint16_t kept_count = (uint16_t)(op->result_count - forwarded_count);
  uint16_t* result_map = NULL;
  uint16_t* iter_arg_map = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*result_map), (void**)&result_map));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*iter_arg_map), (void**)&iter_arg_map));
  for (uint16_t i = 0; i < op->result_count; ++i) {
    result_map[i] = UINT16_MAX;
    iter_arg_map[i] = UINT16_MAX;
  }

  loom_value_id_t* kept_iter_args = NULL;
  loom_type_t* kept_result_types = NULL;
  loom_value_id_t* kept_yielded_values = NULL;
  if (kept_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, kept_count,
                                                   sizeof(*kept_iter_args),
                                                   (void**)&kept_iter_args));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, kept_count,
                                                   sizeof(*kept_result_types),
                                                   (void**)&kept_result_types));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, kept_count, sizeof(*kept_yielded_values),
        (void**)&kept_yielded_values));
  }

  const loom_value_id_t* old_results = loom_op_const_results(op);
  uint16_t kept_ordinal = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (forwarded_results[i]) continue;
    result_map[i] = kept_ordinal;
    iter_arg_map[i] = kept_ordinal;
    kept_iter_args[kept_ordinal] = iter_args.values[i];
    kept_result_types[kept_ordinal] =
        loom_module_value_type(rewriter->module, old_results[i]);
    kept_yielded_values[kept_ordinal] = yielded_values.values[i];
    ++kept_ordinal;
  }

  loom_tied_result_t* tied_results = NULL;
  uint16_t tied_result_count = 0;
  bool tied_results_supported = false;
  IREE_RETURN_IF_ERROR(loom_scf_for_adjust_tied_results(
      op, result_map, iter_arg_map, rewriter, &tied_results, &tied_result_count,
      &tied_results_supported));
  if (!tied_results_supported) return iree_ok_status();

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* new_loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      &rewriter->builder, loom_scf_for_lower_bound(op),
      loom_scf_for_upper_bound(op), loom_scf_for_step(op), kept_iter_args,
      kept_count, kept_result_types, kept_count, tied_results,
      tied_result_count, op->location, &new_loop));

  loom_region_t* new_body = loom_scf_for_body(new_loop);
  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&rewriter->builder, new_loop, new_body);
  loom_op_t* new_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&rewriter->builder,
                                            kept_yielded_values, kept_count,
                                            op->location, &new_yield));
  loom_builder_restore(&rewriter->builder, saved_ip);

  loom_block_t* new_block = loom_region_entry_block(new_body);
  loom_scf_preserve_value_name(rewriter->module,
                               loom_block_arg_id(old_block, 0),
                               loom_block_arg_id(new_block, 0));
  kept_ordinal = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (forwarded_results[i]) continue;
    loom_scf_preserve_value_name(
        rewriter->module, loom_block_arg_id(old_block, (uint16_t)(1 + i)),
        loom_block_arg_id(new_block, (uint16_t)(1 + kept_ordinal++)));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
      rewriter, loom_block_arg_id(old_block, 0),
      loom_block_arg_id(new_block, 0)));
  kept_ordinal = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t old_arg = loom_block_arg_id(old_block, (uint16_t)(1 + i));
    loom_value_id_t replacement =
        forwarded_results[i]
            ? iter_args.values[i]
            : loom_block_arg_id(new_block, (uint16_t)(1 + kept_ordinal++));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_replace_all_uses_with(rewriter, old_arg, replacement));
  }

  loom_op_t* child_op = old_block->first_op;
  while (child_op && child_op != yield) {
    loom_op_t* next_child_op = child_op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(rewriter, child_op, new_yield));
    child_op = next_child_op;
  }

  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*replacements), (void**)&replacements));
  kept_ordinal = 0;
  loom_value_slice_t new_results = loom_scf_for_results(new_loop);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    replacements[i] = forwarded_results[i] ? iter_args.values[i]
                                           : new_results.values[kept_ordinal++];
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, op->result_count, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, replacements,
                                                  op->result_count);
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
  return loom_scf_for_forward_loop_carried_results(op, rewriter);
}
