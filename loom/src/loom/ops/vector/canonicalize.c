// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"

//===----------------------------------------------------------------------===//
// Fact queries
//===----------------------------------------------------------------------===//

static bool loom_vector_exact_facts_equal(loom_value_facts_t lhs,
                                          loom_value_facts_t rhs) {
  if (!loom_value_facts_is_exact(lhs) || !loom_value_facts_is_exact(rhs)) {
    return false;
  }
  return loom_value_facts_equal(lhs, rhs);
}

static bool loom_vector_exact_i64_facts_match(loom_value_facts_t facts,
                                              int64_t expected) {
  return loom_value_facts_is_exact(facts) &&
         !loom_value_facts_is_float(facts) && facts.range_lo == expected;
}

static bool loom_vector_query_all_equal_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_facts_t* out_element) {
  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(context, facts, &uniform)) {
    *out_element = uniform.element;
    return true;
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (!loom_value_facts_query_small_static_lanes(context, facts, &lanes) ||
      lanes.count == 0 || !loom_value_facts_is_exact(lanes.lanes[0])) {
    return false;
  }
  for (iree_host_size_t i = 1; i < lanes.count; ++i) {
    if (!loom_vector_exact_facts_equal(lanes.lanes[0], lanes.lanes[i])) {
      return false;
    }
  }
  *out_element = lanes.lanes[0];
  return true;
}

static bool loom_vector_value_is_all_exact_i64(const loom_rewriter_t* rewriter,
                                               loom_value_id_t value_id,
                                               int64_t expected_value) {
  loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, value_id);
  loom_value_facts_t element = {0};
  if (!loom_vector_query_all_equal_element(&rewriter->fact_table.context, facts,
                                           &element)) {
    return false;
  }
  return loom_value_facts_is_exact(element) &&
         !loom_value_facts_is_float(element) &&
         element.range_lo == expected_value;
}

static bool loom_vector_value_is_unit_stride_iota(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id,
    loom_type_t vector_type) {
  if (!loom_type_is_vector(vector_type) || loom_type_rank(vector_type) != 1) {
    return false;
  }
  loom_value_fact_vector_iota_t iota = {0};
  if (!loom_value_facts_query_vector_iota(
          &rewriter->fact_table.context,
          loom_rewriter_value_facts(rewriter, value_id), &iota)) {
    return false;
  }
  return loom_vector_exact_i64_facts_match(iota.base, 0) &&
         loom_vector_exact_i64_facts_match(iota.step, 1);
}

static bool loom_vector_value_is_all_bool(const loom_rewriter_t* rewriter,
                                          loom_value_id_t value_id,
                                          bool expected) {
  return loom_vector_value_is_all_exact_i64(rewriter, value_id,
                                            expected ? 1 : 0);
}

static bool loom_vector_facts_to_constant_attr(loom_value_facts_t facts,
                                               loom_scalar_type_t element_type,
                                               loom_attribute_t* out_attr) {
  if (!loom_value_facts_is_exact(facts)) return false;

  if (loom_scalar_type_is_float(element_type)) {
    if (!loom_value_facts_is_float(facts)) return false;
    *out_attr = loom_attr_f64(loom_value_facts_as_f64(facts));
    return true;
  }

  if (loom_value_facts_is_float(facts)) return false;
  if (element_type == LOOM_SCALAR_TYPE_I1) {
    if (facts.range_lo != 0 && facts.range_lo != 1) return false;
    *out_attr = loom_attr_bool(facts.range_lo != 0);
    return true;
  }

  *out_attr = loom_attr_i64(facts.range_lo);
  return true;
}

//===----------------------------------------------------------------------===//
// IR helpers
//===----------------------------------------------------------------------===//

static bool loom_vector_value_def_op(const loom_rewriter_t* rewriter,
                                     loom_value_id_t value_id,
                                     loom_op_t** out_def_op) {
  if (value_id >= rewriter->module->values.count) return false;
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op || (def_op->flags & LOOM_OP_FLAG_DEAD)) return false;
  *out_def_op = def_op;
  return true;
}

static iree_status_t loom_vector_replace_single_result_with_value(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t replacement) {
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_vector_replace_single_result_with_constant_attr(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_type_t result_type,
    loom_attribute_t attr) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      &rewriter->builder, attr, result_type, op->location, &constant_op));
  loom_value_id_t replacement = loom_vector_constant_result(constant_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_vector_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static iree_status_t loom_vector_replace_single_result_with_constant_facts(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_facts_t element_facts,
    loom_type_t result_type) {
  loom_attribute_t attr = {0};
  if (!loom_vector_facts_to_constant_attr(
          element_facts, loom_type_element_type(result_type), &attr)) {
    return iree_ok_status();
  }
  return loom_vector_replace_single_result_with_constant_attr(
      op, rewriter, result_type, attr);
}

static iree_status_t loom_vector_replace_single_result_with_splat(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t scalar,
    loom_type_t result_type) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* splat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      &rewriter->builder, scalar, result_type, op->location, &splat_op));
  loom_value_id_t replacement = loom_vector_splat_result(splat_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_vector_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

static bool loom_vector_get_single_result_type(const loom_rewriter_t* rewriter,
                                               const loom_op_t* op,
                                               loom_type_t* out_result_type) {
  if (op->result_count != 1) return false;
  loom_value_id_t result = loom_op_const_results(op)[0];
  if (result >= rewriter->module->values.count) return false;
  *out_result_type = loom_module_value_type(rewriter->module, result);
  return true;
}

static int64_t loom_vector_integer_all_ones(loom_scalar_type_t element_type) {
  return element_type == LOOM_SCALAR_TYPE_I1 ? 1 : -1;
}

static iree_status_t loom_vector_replace_single_result_with_integer_zero(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_type_t result_type) {
  loom_attribute_t attr =
      loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1
          ? loom_attr_bool(false)
          : loom_attr_i64(0);
  return loom_vector_replace_single_result_with_constant_attr(
      op, rewriter, result_type, attr);
}

static iree_status_t loom_vector_replace_single_result_with_new_op(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* new_op,
    loom_value_id_t value_checkpoint) {
  if (new_op->result_count != 1) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "replacement op must have exactly one result");
  }
  loom_value_id_t replacement = loom_op_const_results(new_op)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_vector_replace_single_result_with_value(op, rewriter,
                                                      replacement);
}

//===----------------------------------------------------------------------===//
// Generic vector fact materialization
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_canonicalize_uniform_result(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;
  if (op->kind == LOOM_OP_VECTOR_CONSTANT || op->kind == LOOM_OP_VECTOR_EMPTY ||
      op->kind == LOOM_OP_VECTOR_POISON) {
    return iree_ok_status();
  }
  loom_trait_flags_t traits = loom_op_effective_traits(rewriter->module, op);
  if (!iree_any_bit_set(traits, LOOM_TRAIT_PURE)) return iree_ok_status();

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type) ||
      !loom_type_is_vector(result_type)) {
    return iree_ok_status();
  }

  loom_value_id_t result = loom_op_const_results(op)[0];
  loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, result);
  loom_value_facts_t element = {0};
  if (!loom_vector_query_all_equal_element(&rewriter->fact_table.context, facts,
                                           &element) ||
      !loom_value_facts_is_exact(element)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_constant_facts(
      op, rewriter, element, result_type));
  *out_changed = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Construction and access
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_canonicalize_from_elements(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;
  loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  if (elements.count == 0) return iree_ok_status();

  loom_value_id_t first_element = elements.values[0];
  for (uint16_t i = 1; i < elements.count; ++i) {
    if (elements.values[i] != first_element) return iree_ok_status();
  }

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_splat(
      op, rewriter, first_element, result_type));
  *out_changed = true;
  return iree_ok_status();
}

static bool loom_vector_static_indices_are_proven_in_bounds(
    loom_type_t source_type, loom_attribute_t static_indices) {
  if (!loom_type_is_all_static(source_type)) return false;
  if (static_indices.count > loom_type_rank(source_type)) return false;
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    int64_t static_index = static_indices.i64_array[i];
    if (static_index < 0 || static_index == INT64_MIN) return false;
    if (static_index >= loom_type_dim_static_size_at(source_type, i)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_vector_canonicalize_extract_from_splat(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* source_def_op,
    loom_type_t source_type, loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  loom_attribute_t static_indices = loom_vector_extract_static_indices(op);
  if (!loom_vector_static_indices_are_proven_in_bounds(source_type,
                                                       static_indices)) {
    return iree_ok_status();
  }

  loom_value_id_t scalar = loom_vector_splat_scalar(source_def_op);
  if (loom_type_is_scalar(result_type)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, scalar));
  } else if (loom_type_is_vector(result_type)) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_splat(
        op, rewriter, scalar, result_type));
  } else {
    return iree_ok_status();
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_extract_from_elements(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_t* source_def_op,
    loom_type_t result_type, bool* out_changed) {
  *out_changed = false;
  if (!loom_type_is_scalar(result_type)) return iree_ok_status();

  loom_attribute_t static_indices = loom_vector_extract_static_indices(op);
  if (static_indices.count != 1 || static_indices.i64_array[0] < 0 ||
      static_indices.i64_array[0] == INT64_MIN) {
    return iree_ok_status();
  }
  loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_def_op);
  iree_host_size_t lane = (iree_host_size_t)static_indices.i64_array[0];
  if (lane >= elements.count) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
      op, rewriter, elements.values[lane]));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_extract(loom_op_t* op,
                                                      loom_rewriter_t* rewriter,
                                                      bool* out_changed) {
  *out_changed = false;

  loom_value_id_t source = loom_vector_extract_source(op);
  loom_op_t* source_def_op = NULL;
  if (!loom_vector_value_def_op(rewriter, source, &source_def_op)) {
    return iree_ok_status();
  }

  loom_type_t source_type = loom_module_value_type(rewriter->module, source);
  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }

  if (loom_vector_splat_isa(source_def_op)) {
    return loom_vector_canonicalize_extract_from_splat(
        op, rewriter, source_def_op, source_type, result_type, out_changed);
  }
  if (loom_vector_from_elements_isa(source_def_op)) {
    return loom_vector_canonicalize_extract_from_elements(
        op, rewriter, source_def_op, result_type, out_changed);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_shuffle(loom_op_t* op,
                                                      loom_rewriter_t* rewriter,
                                                      bool* out_changed) {
  *out_changed = false;

  loom_attribute_t source_lanes = loom_vector_shuffle_source_lanes(op);
  for (uint16_t i = 0; i < source_lanes.count; ++i) {
    if (source_lanes.i64_array[i] != (int64_t)i) return iree_ok_status();
  }

  loom_value_id_t source = loom_vector_shuffle_source(op);
  IREE_RETURN_IF_ERROR(
      loom_vector_replace_single_result_with_value(op, rewriter, source));
  *out_changed = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Selection and comparison
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_canonicalize_select(loom_op_t* op,
                                                     loom_rewriter_t* rewriter,
                                                     bool* out_changed) {
  *out_changed = false;
  loom_value_id_t true_value = loom_vector_select_true_value(op);
  loom_value_id_t false_value = loom_vector_select_false_value(op);
  if (true_value == false_value) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, true_value));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_value_id_t condition = loom_vector_select_condition(op);
  if (loom_vector_value_is_all_bool(rewriter, condition, true)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, true_value));
    *out_changed = true;
    return iree_ok_status();
  }
  if (loom_vector_value_is_all_bool(rewriter, condition, false)) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
        op, rewriter, false_value));
    *out_changed = true;
  }
  return iree_ok_status();
}

static bool loom_vector_cmpi_same_operand_result(uint8_t predicate,
                                                 bool* out_value) {
  switch ((loom_vector_cmpi_predicate_t)predicate) {
    case LOOM_VECTOR_CMPI_PREDICATE_EQ:
    case LOOM_VECTOR_CMPI_PREDICATE_SLE:
    case LOOM_VECTOR_CMPI_PREDICATE_SGE:
    case LOOM_VECTOR_CMPI_PREDICATE_ULE:
    case LOOM_VECTOR_CMPI_PREDICATE_UGE:
      *out_value = true;
      return true;
    case LOOM_VECTOR_CMPI_PREDICATE_NE:
    case LOOM_VECTOR_CMPI_PREDICATE_SLT:
    case LOOM_VECTOR_CMPI_PREDICATE_SGT:
    case LOOM_VECTOR_CMPI_PREDICATE_ULT:
    case LOOM_VECTOR_CMPI_PREDICATE_UGT:
      *out_value = false;
      return true;
    case LOOM_VECTOR_CMPI_PREDICATE_COUNT_:
      return false;
  }
  return false;
}

static bool loom_vector_cmpf_same_operand_result(uint8_t predicate,
                                                 bool* out_value) {
  switch ((loom_vector_cmpf_predicate_t)predicate) {
    case LOOM_VECTOR_CMPF_PREDICATE_OGT:
    case LOOM_VECTOR_CMPF_PREDICATE_OLT:
    case LOOM_VECTOR_CMPF_PREDICATE_ONE:
      *out_value = false;
      return true;
    case LOOM_VECTOR_CMPF_PREDICATE_UEQ:
    case LOOM_VECTOR_CMPF_PREDICATE_UGE:
    case LOOM_VECTOR_CMPF_PREDICATE_ULE:
      *out_value = true;
      return true;
    case LOOM_VECTOR_CMPF_PREDICATE_OEQ:
    case LOOM_VECTOR_CMPF_PREDICATE_OGE:
    case LOOM_VECTOR_CMPF_PREDICATE_OLE:
    case LOOM_VECTOR_CMPF_PREDICATE_ORD:
    case LOOM_VECTOR_CMPF_PREDICATE_UGT:
    case LOOM_VECTOR_CMPF_PREDICATE_ULT:
    case LOOM_VECTOR_CMPF_PREDICATE_UNE:
    case LOOM_VECTOR_CMPF_PREDICATE_UNO:
    case LOOM_VECTOR_CMPF_PREDICATE_COUNT_:
      return false;
  }
  return false;
}

static iree_status_t loom_vector_canonicalize_comparison(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;

  loom_value_id_t lhs = loom_vector_cmpi_isa(op) ? loom_vector_cmpi_lhs(op)
                                                 : loom_vector_cmpf_lhs(op);
  loom_value_id_t rhs = loom_vector_cmpi_isa(op) ? loom_vector_cmpi_rhs(op)
                                                 : loom_vector_cmpf_rhs(op);
  if (lhs != rhs) return iree_ok_status();

  bool value = false;
  bool has_result = loom_vector_cmpi_isa(op)
                        ? loom_vector_cmpi_same_operand_result(
                              loom_vector_cmpi_predicate(op), &value)
                        : loom_vector_cmpf_same_operand_result(
                              loom_vector_cmpf_predicate(op), &value);
  if (!has_result) return iree_ok_status();

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_constant_attr(
      op, rewriter, result_type, loom_attr_bool(value)));
  *out_changed = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lanewise integer arithmetic
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_canonicalize_binary_identity(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;

  loom_type_t result_type = {0};
  if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
    return iree_ok_status();
  }
  loom_scalar_type_t element_type = loom_type_element_type(result_type);
  int64_t all_ones = loom_vector_integer_all_ones(element_type);

  const loom_value_id_t* operands = loom_op_const_operands(op);
  loom_value_id_t lhs = operands[0];
  loom_value_id_t rhs = operands[1];
  if (lhs == rhs &&
      (op->kind == LOOM_OP_VECTOR_ANDI || op->kind == LOOM_OP_VECTOR_ORI ||
       op->kind == LOOM_OP_VECTOR_MINSI || op->kind == LOOM_OP_VECTOR_MAXSI ||
       op->kind == LOOM_OP_VECTOR_MINUI || op->kind == LOOM_OP_VECTOR_MAXUI)) {
    IREE_RETURN_IF_ERROR(
        loom_vector_replace_single_result_with_value(op, rewriter, lhs));
    *out_changed = true;
    return iree_ok_status();
  }

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  switch (op->kind) {
    case LOOM_OP_VECTOR_ADDI:
    case LOOM_OP_VECTOR_ORI:
      if (loom_vector_value_is_all_exact_i64(rewriter, lhs, 0)) {
        replacement = rhs;
      }
      if (loom_vector_value_is_all_exact_i64(rewriter, rhs, 0)) {
        replacement = lhs;
      }
      break;
    case LOOM_OP_VECTOR_SUBI:
      if (loom_vector_value_is_all_exact_i64(rewriter, rhs, 0)) {
        replacement = lhs;
      }
      break;
    case LOOM_OP_VECTOR_MULI:
      if (loom_vector_value_is_all_exact_i64(rewriter, lhs, 0)) {
        replacement = lhs;
      }
      if (loom_vector_value_is_all_exact_i64(rewriter, rhs, 0)) {
        replacement = rhs;
      }
      if (replacement == LOOM_VALUE_ID_INVALID &&
          loom_vector_value_is_all_exact_i64(rewriter, lhs, 1)) {
        replacement = rhs;
      }
      if (replacement == LOOM_VALUE_ID_INVALID &&
          loom_vector_value_is_all_exact_i64(rewriter, rhs, 1)) {
        replacement = lhs;
      }
      break;
    case LOOM_OP_VECTOR_ANDI:
      if (loom_vector_value_is_all_exact_i64(rewriter, lhs, 0)) {
        replacement = lhs;
      }
      if (loom_vector_value_is_all_exact_i64(rewriter, rhs, 0)) {
        replacement = rhs;
      }
      if (replacement == LOOM_VALUE_ID_INVALID &&
          loom_vector_value_is_all_exact_i64(rewriter, lhs, all_ones)) {
        replacement = rhs;
      }
      if (replacement == LOOM_VALUE_ID_INVALID &&
          loom_vector_value_is_all_exact_i64(rewriter, rhs, all_ones)) {
        replacement = lhs;
      }
      break;
    case LOOM_OP_VECTOR_XORI:
      if (loom_vector_value_is_all_exact_i64(rewriter, lhs, 0)) {
        replacement = rhs;
      }
      if (loom_vector_value_is_all_exact_i64(rewriter, rhs, 0)) {
        replacement = lhs;
      }
      break;
    default:
      break;
  }

  if (replacement != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
        op, rewriter, replacement));
    *out_changed = true;
    return iree_ok_status();
  }

  if (lhs == rhs &&
      (op->kind == LOOM_OP_VECTOR_SUBI || op->kind == LOOM_OP_VECTOR_XORI)) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_integer_zero(
        op, rewriter, result_type));
    *out_changed = true;
    return iree_ok_status();
  }

  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_reduce(loom_op_t* op,
                                                     loom_rewriter_t* rewriter,
                                                     bool* out_changed) {
  *out_changed = false;

  loom_value_id_t input = loom_vector_reduce_input(op);
  loom_value_id_t init = loom_vector_reduce_init(op);
  loom_type_t input_type = loom_module_value_type(rewriter->module, input);
  int64_t identity = 0;
  switch ((loom_vector_reduce_kind_t)loom_vector_reduce_kind(op)) {
    case LOOM_VECTOR_REDUCE_KIND_ADDI:
    case LOOM_VECTOR_REDUCE_KIND_ORI:
    case LOOM_VECTOR_REDUCE_KIND_XORI:
      identity = 0;
      break;
    case LOOM_VECTOR_REDUCE_KIND_MULI:
      identity = 1;
      break;
    case LOOM_VECTOR_REDUCE_KIND_ANDI:
      identity =
          loom_vector_integer_all_ones(loom_type_element_type(input_type));
      break;
    case LOOM_VECTOR_REDUCE_KIND_ADDF:
    case LOOM_VECTOR_REDUCE_KIND_MULF:
    case LOOM_VECTOR_REDUCE_KIND_MINSI:
    case LOOM_VECTOR_REDUCE_KIND_MAXSI:
    case LOOM_VECTOR_REDUCE_KIND_MINUI:
    case LOOM_VECTOR_REDUCE_KIND_MAXUI:
    case LOOM_VECTOR_REDUCE_KIND_MINIMUMF:
    case LOOM_VECTOR_REDUCE_KIND_MAXIMUMF:
    case LOOM_VECTOR_REDUCE_KIND_MINNUMF:
    case LOOM_VECTOR_REDUCE_KIND_MAXNUMF:
    case LOOM_VECTOR_REDUCE_KIND_COUNT_:
      return iree_ok_status();
  }

  if (!loom_vector_value_is_all_exact_i64(rewriter, input, identity)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_vector_replace_single_result_with_value(op, rewriter, init));
  *out_changed = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Masked memory
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_canonicalize_all_true_masked_memory(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;

  loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
  switch (op->kind) {
    case LOOM_OP_VECTOR_LOAD_MASK:
      mask = loom_vector_load_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_LOAD_EXPAND:
      mask = loom_vector_load_expand_mask(op);
      break;
    case LOOM_OP_VECTOR_GATHER_MASK:
      mask = loom_vector_gather_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_ATOMIC_RMW_MASK:
      mask = loom_vector_atomic_rmw_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_STORE_MASK:
      mask = loom_vector_store_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_STORE_COMPRESS:
      mask = loom_vector_store_compress_mask(op);
      break;
    case LOOM_OP_VECTOR_SCATTER_MASK:
      mask = loom_vector_scatter_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK:
      mask = loom_vector_atomic_reduce_mask_mask(op);
      break;
    default:
      return iree_ok_status();
  }
  if (!loom_vector_value_is_all_bool(rewriter, mask, true)) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* new_op = NULL;
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(rewriter->module, op,
                                               &cache_policy)) {
    return iree_ok_status();
  }
  switch (op->kind) {
    case LOOM_OP_VECTOR_LOAD_MASK: {
      loom_value_slice_t indices = loom_vector_load_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_load_mask_static_indices(op);
      loom_type_t result_type = {0};
      if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_vector_load_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_load_mask_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          cache_policy.cache_scope, cache_policy.cache_temporal, result_type,
          op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
          op, rewriter, new_op, value_checkpoint));
      break;
    }
    case LOOM_OP_VECTOR_LOAD_EXPAND: {
      loom_value_slice_t indices = loom_vector_load_expand_indices(op);
      loom_attribute_t static_indices =
          loom_vector_load_expand_static_indices(op);
      loom_type_t result_type = {0};
      if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_vector_load_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_load_expand_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          cache_policy.cache_scope, cache_policy.cache_temporal, result_type,
          op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
          op, rewriter, new_op, value_checkpoint));
      break;
    }
    case LOOM_OP_VECTOR_GATHER_MASK: {
      loom_value_slice_t indices = loom_vector_gather_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_gather_mask_static_indices(op);
      loom_type_t result_type = {0};
      if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_vector_gather_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_gather_mask_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          loom_vector_gather_mask_offsets(op), cache_policy.cache_scope,
          cache_policy.cache_temporal, result_type, op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
          op, rewriter, new_op, value_checkpoint));
      break;
    }
    case LOOM_OP_VECTOR_ATOMIC_RMW_MASK: {
      loom_value_slice_t indices = loom_vector_atomic_rmw_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_atomic_rmw_mask_static_indices(op);
      loom_type_t result_type = {0};
      if (!loom_vector_get_single_result_type(rewriter, op, &result_type)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_vector_atomic_rmw_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_atomic_rmw_mask_kind(op),
          loom_vector_atomic_rmw_mask_value(op),
          loom_vector_atomic_rmw_mask_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          loom_vector_atomic_rmw_mask_offsets(op),
          loom_vector_atomic_rmw_mask_ordering(op),
          loom_vector_atomic_rmw_mask_scope(op), cache_policy.cache_scope,
          cache_policy.cache_temporal, result_type, op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
          op, rewriter, new_op, value_checkpoint));
      break;
    }
    case LOOM_OP_VECTOR_STORE_MASK: {
      loom_value_slice_t indices = loom_vector_store_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_store_mask_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_store_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_store_mask_value(op), loom_vector_store_mask_view(op),
          indices.values, indices.count, static_indices.i64_array,
          static_indices.count, cache_policy.cache_scope,
          cache_policy.cache_temporal, op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
      break;
    }
    case LOOM_OP_VECTOR_STORE_COMPRESS: {
      loom_value_slice_t indices = loom_vector_store_compress_indices(op);
      loom_attribute_t static_indices =
          loom_vector_store_compress_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_store_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_store_compress_value(op),
          loom_vector_store_compress_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          cache_policy.cache_scope, cache_policy.cache_temporal, op->location,
          &new_op));
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
      break;
    }
    case LOOM_OP_VECTOR_SCATTER_MASK: {
      loom_value_slice_t indices = loom_vector_scatter_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_scatter_mask_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_scatter_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_scatter_mask_value(op), loom_vector_scatter_mask_view(op),
          indices.values, indices.count, static_indices.i64_array,
          static_indices.count, loom_vector_scatter_mask_offsets(op),
          cache_policy.cache_scope, cache_policy.cache_temporal, op->location,
          &new_op));
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
      break;
    }
    case LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK: {
      loom_value_slice_t indices = loom_vector_atomic_reduce_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_atomic_reduce_mask_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_atomic_reduce_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_atomic_reduce_mask_kind(op),
          loom_vector_atomic_reduce_mask_value(op),
          loom_vector_atomic_reduce_mask_view(op), indices.values,
          indices.count, static_indices.i64_array, static_indices.count,
          loom_vector_atomic_reduce_mask_offsets(op),
          loom_vector_atomic_reduce_mask_ordering(op),
          loom_vector_atomic_reduce_mask_scope(op), cache_policy.cache_scope,
          cache_policy.cache_temporal, op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
      break;
    }
    default:
      return iree_ok_status();
  }

  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_all_false_masked_memory(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;

  loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
  loom_value_id_t passthrough = LOOM_VALUE_ID_INVALID;
  bool has_passthrough_result = false;
  switch (op->kind) {
    case LOOM_OP_VECTOR_LOAD_MASK:
      mask = loom_vector_load_mask_mask(op);
      passthrough = loom_vector_load_mask_passthrough(op);
      has_passthrough_result = true;
      break;
    case LOOM_OP_VECTOR_LOAD_EXPAND:
      mask = loom_vector_load_expand_mask(op);
      passthrough = loom_vector_load_expand_passthrough(op);
      has_passthrough_result = true;
      break;
    case LOOM_OP_VECTOR_GATHER_MASK:
      mask = loom_vector_gather_mask_mask(op);
      passthrough = loom_vector_gather_mask_passthrough(op);
      has_passthrough_result = true;
      break;
    case LOOM_OP_VECTOR_ATOMIC_RMW_MASK:
      mask = loom_vector_atomic_rmw_mask_mask(op);
      passthrough = loom_vector_atomic_rmw_mask_passthrough(op);
      has_passthrough_result = true;
      break;
    case LOOM_OP_VECTOR_STORE_MASK:
      mask = loom_vector_store_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_STORE_COMPRESS:
      mask = loom_vector_store_compress_mask(op);
      break;
    case LOOM_OP_VECTOR_SCATTER_MASK:
      mask = loom_vector_scatter_mask_mask(op);
      break;
    case LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK:
      mask = loom_vector_atomic_reduce_mask_mask(op);
      break;
    default:
      return iree_ok_status();
  }

  if (!loom_vector_value_is_all_bool(rewriter, mask, false)) {
    return iree_ok_status();
  }

  if (has_passthrough_result) {
    IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_value(
        op, rewriter, passthrough));
  } else {
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_canonicalize_contiguous_gather_scatter(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed) {
  *out_changed = false;

  loom_value_id_t offsets = LOOM_VALUE_ID_INVALID;
  loom_value_id_t shaped_value = LOOM_VALUE_ID_INVALID;
  switch (op->kind) {
    case LOOM_OP_VECTOR_GATHER:
      offsets = loom_vector_gather_offsets(op);
      shaped_value = loom_vector_gather_result(op);
      break;
    case LOOM_OP_VECTOR_GATHER_MASK:
      offsets = loom_vector_gather_mask_offsets(op);
      shaped_value = loom_vector_gather_mask_result(op);
      break;
    case LOOM_OP_VECTOR_SCATTER:
      offsets = loom_vector_scatter_offsets(op);
      shaped_value = loom_vector_scatter_value(op);
      break;
    case LOOM_OP_VECTOR_SCATTER_MASK:
      offsets = loom_vector_scatter_mask_offsets(op);
      shaped_value = loom_vector_scatter_mask_value(op);
      break;
    default:
      return iree_ok_status();
  }

  loom_type_t offsets_type = loom_module_value_type(rewriter->module, offsets);
  loom_type_t shaped_type =
      loom_module_value_type(rewriter->module, shaped_value);
  if (!loom_type_is_vector(shaped_type) || loom_type_rank(shaped_type) != 1 ||
      !loom_vector_value_is_unit_stride_iota(rewriter, offsets, offsets_type)) {
    return iree_ok_status();
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* new_op = NULL;
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(rewriter->module, op,
                                               &cache_policy)) {
    return iree_ok_status();
  }
  switch (op->kind) {
    case LOOM_OP_VECTOR_GATHER: {
      loom_value_slice_t indices = loom_vector_gather_indices(op);
      loom_attribute_t static_indices = loom_vector_gather_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_load_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_gather_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          cache_policy.cache_scope, cache_policy.cache_temporal, shaped_type,
          op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
          op, rewriter, new_op, value_checkpoint));
      break;
    }
    case LOOM_OP_VECTOR_GATHER_MASK: {
      loom_value_slice_t indices = loom_vector_gather_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_gather_mask_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_load_mask_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_gather_mask_view(op), indices.values, indices.count,
          static_indices.i64_array, static_indices.count,
          loom_vector_gather_mask_mask(op),
          loom_vector_gather_mask_passthrough(op), cache_policy.cache_scope,
          cache_policy.cache_temporal, shaped_type, op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_vector_replace_single_result_with_new_op(
          op, rewriter, new_op, value_checkpoint));
      break;
    }
    case LOOM_OP_VECTOR_SCATTER: {
      loom_value_slice_t indices = loom_vector_scatter_indices(op);
      loom_attribute_t static_indices = loom_vector_scatter_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_store_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_scatter_value(op), loom_vector_scatter_view(op),
          indices.values, indices.count, static_indices.i64_array,
          static_indices.count, cache_policy.cache_scope,
          cache_policy.cache_temporal, op->location, &new_op));
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
      break;
    }
    case LOOM_OP_VECTOR_SCATTER_MASK: {
      loom_value_slice_t indices = loom_vector_scatter_mask_indices(op);
      loom_attribute_t static_indices =
          loom_vector_scatter_mask_static_indices(op);
      IREE_RETURN_IF_ERROR(loom_vector_store_mask_build(
          &rewriter->builder, cache_policy.build_flags,
          loom_vector_scatter_mask_value(op), loom_vector_scatter_mask_view(op),
          indices.values, indices.count, static_indices.i64_array,
          static_indices.count, loom_vector_scatter_mask_mask(op),
          cache_policy.cache_scope, cache_policy.cache_temporal, op->location,
          &new_op));
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
      break;
    }
    default:
      return iree_ok_status();
  }

  *out_changed = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Entry points
//===----------------------------------------------------------------------===//

typedef iree_status_t (*loom_vector_try_canonicalize_fn_t)(
    loom_op_t* op, loom_rewriter_t* rewriter, bool* out_changed);

iree_status_t loom_vector_uniform_result_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  bool changed = false;
  return loom_vector_canonicalize_uniform_result(op, rewriter, &changed);
}

static iree_status_t loom_vector_canonicalize_uniform_then(
    loom_op_t* op, loom_rewriter_t* rewriter,
    loom_vector_try_canonicalize_fn_t specific_canonicalize) {
  bool changed = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_canonicalize_uniform_result(op, rewriter, &changed));
  if (changed) return iree_ok_status();
  return specific_canonicalize(op, rewriter, &changed);
}

iree_status_t loom_vector_from_elements_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(
      op, rewriter, loom_vector_canonicalize_from_elements);
}

iree_status_t loom_vector_extract_canonicalize(loom_op_t* op,
                                               loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(
      op, rewriter, loom_vector_canonicalize_extract);
}

iree_status_t loom_vector_shuffle_canonicalize(loom_op_t* op,
                                               loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(
      op, rewriter, loom_vector_canonicalize_shuffle);
}

iree_status_t loom_vector_select_canonicalize(loom_op_t* op,
                                              loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(op, rewriter,
                                               loom_vector_canonicalize_select);
}

iree_status_t loom_vector_comparison_canonicalize(loom_op_t* op,
                                                  loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(
      op, rewriter, loom_vector_canonicalize_comparison);
}

iree_status_t loom_vector_binary_identity_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  return loom_vector_canonicalize_uniform_then(
      op, rewriter, loom_vector_canonicalize_binary_identity);
}

iree_status_t loom_vector_reduce_canonicalize(loom_op_t* op,
                                              loom_rewriter_t* rewriter) {
  bool changed = false;
  return loom_vector_canonicalize_reduce(op, rewriter, &changed);
}

iree_status_t loom_vector_gather_scatter_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  bool changed = false;
  return loom_vector_canonicalize_contiguous_gather_scatter(op, rewriter,
                                                            &changed);
}

iree_status_t loom_vector_masked_memory_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  bool changed = false;
  IREE_RETURN_IF_ERROR(
      loom_vector_canonicalize_all_false_masked_memory(op, rewriter, &changed));
  if (changed) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_vector_canonicalize_contiguous_gather_scatter(
      op, rewriter, &changed));
  if (changed) return iree_ok_status();
  return loom_vector_canonicalize_all_true_masked_memory(op, rewriter,
                                                         &changed);
}
