// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/scf/ops.h"
#include "loom/transforms/rewriter.h"
#include "loom/util/math.h"

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

static iree_status_t loom_scf_move_region_body_before_op(
    loom_rewriter_t* rewriter, loom_region_t* region, loom_op_t* old_yield,
    loom_op_t* before_op) {
  loom_block_t* block = loom_region_entry_block(region);
  if (!block) return iree_ok_status();
  loom_op_t* child_op = block->first_op;
  while (child_op && child_op != old_yield) {
    loom_op_t* next_child_op = child_op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(rewriter, child_op, before_op));
    child_op = next_child_op;
  }
  return iree_ok_status();
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
// scf.select
//===----------------------------------------------------------------------===//

iree_status_t loom_scf_select_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  loom_value_id_t true_value = loom_scf_select_true_value(op);
  loom_value_id_t false_value = loom_scf_select_false_value(op);
  if (true_value == false_value) {
    return loom_scf_replace_results_and_erase(op, rewriter, &true_value, 1);
  }

  bool condition = false;
  if (!loom_scf_value_facts_are_exact_bool(
          rewriter, loom_scf_select_condition(op), &condition)) {
    loom_value_id_t result = loom_scf_select_result(op);
    loom_type_t result_type = loom_module_value_type(rewriter->module, result);
    int64_t true_i64 = 0;
    int64_t false_i64 = 0;
    if (loom_type_is_scalar(result_type) &&
        loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1 &&
        loom_scf_value_facts_are_exact_i64(rewriter, true_value, &true_i64) &&
        true_i64 == 1 &&
        loom_scf_value_facts_are_exact_i64(rewriter, false_value, &false_i64) &&
        false_i64 == 0) {
      loom_value_id_t condition_value = loom_scf_select_condition(op);
      return loom_scf_replace_results_and_erase(op, rewriter, &condition_value,
                                                1);
    }
    return iree_ok_status();
  }
  loom_value_id_t replacement = condition ? true_value : false_value;
  return loom_scf_replace_results_and_erase(op, rewriter, &replacement, 1);
}

//===----------------------------------------------------------------------===//
// scf.if
//===----------------------------------------------------------------------===//

static bool loom_scf_if_regions_are_discardable(const loom_module_t* module,
                                                loom_op_t* op) {
  // Read-only branch work with no yielded observer is dead. Writes and hints
  // are retained because they affect memory or requested code-generation shape.
  if (loom_op_regions_have_write_effects(op)) return false;
  return !loom_op_regions_have_hints(module, op);
}

static bool loom_scf_value_has_no_uses(const loom_module_t* module,
                                       loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  return loom_value_has_no_uses(loom_module_value(module, value_id)) &&
         !loom_module_value_has_type_uses(module, value_id);
}

static iree_status_t loom_scf_if_erase_if_effect_free_resultless(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_erased) {
  *out_erased = false;
  if (op->result_count != 0) return iree_ok_status();
  if (!loom_scf_if_regions_are_discardable(rewriter->module, op)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  *out_erased = true;
  return iree_ok_status();
}

static iree_status_t loom_scf_if_compact_results(loom_op_t* op,
                                                 loom_rewriter_t* rewriter) {
  if (op->result_count == 0 || op->tied_result_count != 0) {
    return iree_ok_status();
  }

  loom_op_t* then_yield =
      loom_scf_region_terminator(loom_scf_if_then_region(op));
  loom_op_t* else_yield =
      loom_scf_region_terminator(loom_scf_if_else_region(op));
  if (!then_yield || !else_yield) return iree_ok_status();

  loom_value_slice_t then_values = loom_scf_yield_values(then_yield);
  loom_value_slice_t else_values = loom_scf_yield_values(else_yield);
  if (then_values.count != op->result_count ||
      else_values.count != op->result_count) {
    return iree_ok_status();
  }

  const loom_value_id_t* old_results = loom_op_const_results(op);
  bool* forwarded_results = NULL;
  bool* dropped_results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, op->result_count, sizeof(*forwarded_results),
      (void**)&forwarded_results));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, op->result_count, sizeof(*dropped_results),
      (void**)&dropped_results));
  uint16_t dropped_count = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    forwarded_results[i] = then_values.values[i] == else_values.values[i];
    dropped_results[i] =
        forwarded_results[i] ||
        loom_scf_value_has_no_uses(rewriter->module, old_results[i]);
    if (dropped_results[i]) ++dropped_count;
  }
  if (dropped_count == 0) return iree_ok_status();

  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*replacements), (void**)&replacements));

  if (dropped_count == op->result_count &&
      loom_scf_if_regions_are_discardable(rewriter->module, op)) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      replacements[i] =
          forwarded_results[i] ? then_values.values[i] : old_results[i];
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        rewriter, op, replacements, op->result_count));
    return iree_ok_status();
  }

  uint16_t kept_count = (uint16_t)(op->result_count - dropped_count);
  loom_type_t* kept_result_types = NULL;
  loom_value_id_t* kept_then_values = NULL;
  loom_value_id_t* kept_else_values = NULL;
  if (kept_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, kept_count,
                                                   sizeof(*kept_result_types),
                                                   (void**)&kept_result_types));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, kept_count,
                                                   sizeof(*kept_then_values),
                                                   (void**)&kept_then_values));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(rewriter->arena, kept_count,
                                                   sizeof(*kept_else_values),
                                                   (void**)&kept_else_values));
  }

  uint16_t kept_ordinal = 0;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (dropped_results[i]) continue;
    kept_result_types[kept_ordinal] =
        loom_module_value_type(rewriter->module, old_results[i]);
    kept_then_values[kept_ordinal] = then_values.values[i];
    kept_else_values[kept_ordinal] = else_values.values[i];
    ++kept_ordinal;
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* new_if = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_if_build(&rewriter->builder, loom_scf_if_condition(op),
                        kept_result_types, kept_count, /*tied_results=*/NULL,
                        /*tied_result_count=*/0, op->location, &new_if));

  loom_region_t* new_then_region = loom_scf_if_then_region(new_if);
  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&rewriter->builder, new_if, new_then_region);
  loom_op_t* new_then_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&rewriter->builder,
                                            kept_then_values, kept_count,
                                            op->location, &new_then_yield));
  loom_builder_restore(&rewriter->builder, saved_ip);

  loom_region_t* new_else_region = loom_scf_if_else_region(new_if);
  saved_ip =
      loom_builder_enter_region(&rewriter->builder, new_if, new_else_region);
  loom_op_t* new_else_yield = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(&rewriter->builder,
                                            kept_else_values, kept_count,
                                            op->location, &new_else_yield));
  loom_builder_restore(&rewriter->builder, saved_ip);

  IREE_RETURN_IF_ERROR(loom_scf_move_region_body_before_op(
      rewriter, loom_scf_if_then_region(op), then_yield, new_then_yield));
  IREE_RETURN_IF_ERROR(loom_scf_move_region_body_before_op(
      rewriter, loom_scf_if_else_region(op), else_yield, new_else_yield));

  kept_ordinal = 0;
  loom_value_slice_t new_results = loom_scf_if_results(new_if);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (!dropped_results[i]) {
      replacements[i] = new_results.values[kept_ordinal++];
    } else if (forwarded_results[i]) {
      replacements[i] = then_values.values[i];
    } else {
      replacements[i] = old_results[i];
    }
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, op->result_count, value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, op->result_count));
  return iree_ok_status();
}

static iree_status_t loom_scf_if_build_select(loom_op_t* op, loom_type_t type,
                                              loom_value_id_t true_value,
                                              loom_value_id_t false_value,
                                              loom_rewriter_t* rewriter,
                                              loom_value_id_t* out_result) {
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      &rewriter->builder, loom_scf_if_condition(op), true_value, false_value,
      type, op->location, &select_op));
  *out_result = loom_scf_select_result(select_op);
  return iree_ok_status();
}

static iree_status_t loom_scf_if_selectify_yield_only(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  if (op->result_count == 0 || op->tied_result_count != 0) {
    return iree_ok_status();
  }

  loom_region_t* then_region = loom_scf_if_then_region(op);
  loom_region_t* else_region = loom_scf_if_else_region(op);
  loom_op_t* then_yield = loom_scf_region_terminator(then_region);
  loom_op_t* else_yield = loom_scf_region_terminator(else_region);
  if (!then_yield || !else_yield) return iree_ok_status();

  loom_block_t* then_block = loom_region_entry_block(then_region);
  loom_block_t* else_block = loom_region_entry_block(else_region);
  if (then_block->first_op != then_yield ||
      else_block->first_op != else_yield) {
    return iree_ok_status();
  }

  loom_value_slice_t then_values = loom_scf_yield_values(then_yield);
  loom_value_slice_t else_values = loom_scf_yield_values(else_yield);
  if (then_values.count != op->result_count ||
      else_values.count != op->result_count) {
    return iree_ok_status();
  }

  const loom_value_id_t* old_results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, old_results[i]);
    if (!loom_type_equal(
            result_type,
            loom_module_value_type(rewriter->module, then_values.values[i])) ||
        !loom_type_equal(
            result_type,
            loom_module_value_type(rewriter->module, else_values.values[i]))) {
      return iree_ok_status();
    }
  }

  loom_value_id_t* replacements = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, op->result_count,
                                sizeof(*replacements), (void**)&replacements));
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, old_results[i]);
    IREE_RETURN_IF_ERROR(loom_scf_if_build_select(
        op, result_type, then_values.values[i], else_values.values[i], rewriter,
        &replacements[i]));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, op->result_count, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, replacements,
                                                  op->result_count);
}

iree_status_t loom_scf_if_canonicalize(loom_op_t* op,
                                       loom_rewriter_t* rewriter) {
  bool condition = false;
  if (!loom_scf_value_facts_are_exact_bool(rewriter, loom_scf_if_condition(op),
                                           &condition)) {
    bool erased = false;
    IREE_RETURN_IF_ERROR(
        loom_scf_if_erase_if_effect_free_resultless(op, rewriter, &erased));
    if (erased) return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_scf_if_compact_results(op, rewriter));
    if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
      return iree_ok_status();
    }
    return loom_scf_if_selectify_yield_only(op, rewriter);
  }

  loom_region_t* selected_region =
      condition ? loom_scf_if_then_region(op) : loom_scf_if_else_region(op);
  loom_op_t* yield = loom_scf_region_terminator(selected_region);
  if (!yield) return iree_ok_status();

  loom_value_slice_t yielded_values = loom_scf_yield_values(yield);
  if (yielded_values.count != op->result_count) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_scf_move_region_body_before_op(
      rewriter, selected_region, yield, op));

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

static bool loom_scf_for_has_single_trip_count(loom_op_t* op,
                                               loom_rewriter_t* rewriter) {
  loom_value_facts_t lower_bound =
      loom_rewriter_value_facts(rewriter, loom_scf_for_lower_bound(op));
  loom_value_facts_t upper_bound =
      loom_rewriter_value_facts(rewriter, loom_scf_for_upper_bound(op));
  loom_value_facts_t step =
      loom_rewriter_value_facts(rewriter, loom_scf_for_step(op));
  if (loom_value_facts_is_float(lower_bound) ||
      loom_value_facts_is_float(upper_bound) ||
      loom_value_facts_is_float(step) || !loom_value_facts_is_positive(step)) {
    return false;
  }

  if (lower_bound.range_hi >= upper_bound.range_lo) return false;
  int64_t next_iv_lower_bound = 0;
  if (!loom_checked_add_i64(lower_bound.range_lo, step.range_lo,
                            &next_iv_lower_bound)) {
    return false;
  }
  return next_iv_lower_bound >= upper_bound.range_hi;
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

static iree_status_t loom_scf_for_inline_single_trip(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  if (!loom_scf_for_has_single_trip_count(op, rewriter)) {
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

  loom_block_t* block = loom_region_entry_block(body);
  if (block->arg_count < 1 + iter_args.count) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
      rewriter, loom_block_arg_id(block, 0), loom_scf_for_lower_bound(op)));
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        rewriter, loom_block_arg_id(block, (uint16_t)(1 + i)),
        iter_args.values[i]));
  }

  IREE_RETURN_IF_ERROR(
      loom_scf_move_region_body_before_op(rewriter, body, yield, op));

  loom_value_id_t* replacements = NULL;
  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, op->result_count, sizeof(*replacements),
        (void**)&replacements));
    yielded_values = loom_scf_yield_values(yield);
    memcpy(replacements, yielded_values.values,
           (iree_host_size_t)op->result_count * sizeof(*replacements));
  }
  return loom_scf_replace_results_and_erase(op, rewriter, replacements,
                                            op->result_count);
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
  IREE_RETURN_IF_ERROR(loom_scf_for_inline_single_trip(op, rewriter));
  if (op->flags & LOOM_OP_FLAG_DEAD) return iree_ok_status();
  if (loom_scf_for_step_is_positive(op, rewriter) &&
      loom_scf_for_yields_loop_carried_args(op)) {
    return loom_scf_replace_results_and_erase(op, rewriter, iter_args.values,
                                              iter_args.count);
  }
  return loom_scf_for_forward_loop_carried_results(op, rewriter);
}
