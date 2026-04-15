// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/transforms/rewriter.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool loom_index_value_facts_are_exact_i64(loom_rewriter_t* rewriter,
                                                 loom_value_id_t value,
                                                 int64_t expected_value) {
  loom_value_facts_t facts = loom_rewriter_value_facts(rewriter, value);
  return loom_value_facts_is_exact(facts) &&
         !loom_value_facts_is_float(facts) && facts.range_lo == expected_value;
}

static bool loom_index_value_facts_are_non_overlapping(
    loom_rewriter_t* rewriter, loom_value_id_t lhs, loom_value_id_t rhs) {
  loom_value_facts_t lhs_facts = loom_rewriter_value_facts(rewriter, lhs);
  loom_value_facts_t rhs_facts = loom_rewriter_value_facts(rewriter, rhs);
  return lhs_facts.range_hi < rhs_facts.range_lo ||
         rhs_facts.range_hi < lhs_facts.range_lo;
}

static iree_status_t loom_index_replace_single_result_with_value(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_value_id_t replacement) {
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_index_replace_single_result_with_index_constant(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_type_t result_type,
    int64_t value) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(&rewriter->builder, loom_attr_i64(value),
                                result_type, op->location, &constant_op));
  loom_value_id_t replacement = loom_index_constant_result(constant_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_index_replace_single_result_with_value(op, rewriter, replacement);
}

static iree_status_t loom_index_replace_single_result_with_i1_constant(
    loom_op_t* op, loom_rewriter_t* rewriter, bool value) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      &rewriter->builder, loom_attr_bool(value),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), op->location, &constant_op));
  loom_value_id_t replacement = loom_scalar_constant_result(constant_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_index_replace_single_result_with_value(op, rewriter, replacement);
}

static iree_status_t loom_index_replace_single_result_with_binary_op(
    loom_op_t* op, loom_rewriter_t* rewriter, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs) {
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);

  loom_op_t* replacement_op = NULL;
  switch (kind) {
    case LOOM_OP_INDEX_ADD: {
      IREE_RETURN_IF_ERROR(loom_index_add_build(&rewriter->builder, lhs, rhs,
                                                result_type, op->location,
                                                &replacement_op));
      break;
    }
    case LOOM_OP_INDEX_MUL: {
      IREE_RETURN_IF_ERROR(loom_index_mul_build(&rewriter->builder, lhs, rhs,
                                                result_type, op->location,
                                                &replacement_op));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported replacement index op kind %u",
                              (unsigned)kind);
  }

  loom_value_id_t replacement = loom_op_const_results(replacement_op)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_index_replace_single_result_with_value(op, rewriter, replacement);
}

//===----------------------------------------------------------------------===//
// Arithmetic
//===----------------------------------------------------------------------===//

iree_status_t loom_index_add_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_add_lhs(op);
  loom_value_id_t rhs = loom_index_add_rhs(op);
  if (loom_index_value_facts_are_exact_i64(rewriter, lhs, 0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, rhs);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }
  return iree_ok_status();
}

iree_status_t loom_index_sub_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_sub_lhs(op);
  loom_value_id_t rhs = loom_index_sub_rhs(op);
  if (loom_index_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }
  if (lhs == rhs) {
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, loom_index_sub_result(op));
    return loom_index_replace_single_result_with_index_constant(op, rewriter,
                                                                result_type, 0);
  }
  return iree_ok_status();
}

iree_status_t loom_index_mul_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_mul_lhs(op);
  loom_value_id_t rhs = loom_index_mul_rhs(op);
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_index_mul_result(op));
  if (loom_index_value_facts_are_exact_i64(rewriter, lhs, 0) ||
      loom_index_value_facts_are_exact_i64(rewriter, rhs, 0)) {
    return loom_index_replace_single_result_with_index_constant(op, rewriter,
                                                                result_type, 0);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, lhs, 1)) {
    return loom_index_replace_single_result_with_value(op, rewriter, rhs);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, rhs, 1)) {
    return loom_index_replace_single_result_with_value(op, rewriter, lhs);
  }
  return iree_ok_status();
}

iree_status_t loom_index_div_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  if (loom_index_value_facts_are_exact_i64(rewriter, loom_index_div_rhs(op),
                                           1)) {
    return loom_index_replace_single_result_with_value(op, rewriter,
                                                       loom_index_div_lhs(op));
  }
  return iree_ok_status();
}

iree_status_t loom_index_rem_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  if (loom_index_value_facts_are_exact_i64(rewriter, loom_index_rem_rhs(op),
                                           1)) {
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, loom_index_rem_result(op));
    return loom_index_replace_single_result_with_index_constant(op, rewriter,
                                                                result_type, 0);
  }
  return iree_ok_status();
}

iree_status_t loom_index_madd_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  loom_value_id_t a = loom_index_madd_a(op);
  loom_value_id_t b = loom_index_madd_b(op);
  loom_value_id_t c = loom_index_madd_c(op);
  if (loom_index_value_facts_are_exact_i64(rewriter, a, 0) ||
      loom_index_value_facts_are_exact_i64(rewriter, b, 0)) {
    return loom_index_replace_single_result_with_value(op, rewriter, c);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, c, 0)) {
    return loom_index_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_INDEX_MUL, a, b);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, a, 1)) {
    return loom_index_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_INDEX_ADD, b, c);
  }
  if (loom_index_value_facts_are_exact_i64(rewriter, b, 1)) {
    return loom_index_replace_single_result_with_binary_op(
        op, rewriter, LOOM_OP_INDEX_ADD, a, c);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Casts, comparisons, and selection
//===----------------------------------------------------------------------===//

iree_status_t loom_index_cast_canonicalize(loom_op_t* op,
                                           loom_rewriter_t* rewriter) {
  loom_type_t input_type =
      loom_module_value_type(rewriter->module, loom_index_cast_input(op));
  loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_index_cast_result(op));
  if (!loom_type_equal(input_type, result_type)) return iree_ok_status();
  return loom_index_replace_single_result_with_value(op, rewriter,
                                                     loom_index_cast_input(op));
}

static bool loom_index_same_operand_cmp_result(uint8_t predicate,
                                               bool* out_result) {
  switch ((loom_index_cmp_predicate_t)predicate) {
    case LOOM_INDEX_CMP_PREDICATE_EQ:
    case LOOM_INDEX_CMP_PREDICATE_SLE:
    case LOOM_INDEX_CMP_PREDICATE_SGE:
    case LOOM_INDEX_CMP_PREDICATE_ULE:
    case LOOM_INDEX_CMP_PREDICATE_UGE:
      *out_result = true;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_NE:
    case LOOM_INDEX_CMP_PREDICATE_SLT:
    case LOOM_INDEX_CMP_PREDICATE_SGT:
    case LOOM_INDEX_CMP_PREDICATE_ULT:
    case LOOM_INDEX_CMP_PREDICATE_UGT:
      *out_result = false;
      return true;
    default:
      return false;
  }
}

static bool loom_index_signed_range_cmp_result(loom_rewriter_t* rewriter,
                                               uint8_t predicate,
                                               loom_value_id_t lhs,
                                               loom_value_id_t rhs,
                                               bool* out_result) {
  loom_value_facts_t lhs_facts = loom_rewriter_value_facts(rewriter, lhs);
  loom_value_facts_t rhs_facts = loom_rewriter_value_facts(rewriter, rhs);
  switch ((loom_index_cmp_predicate_t)predicate) {
    case LOOM_INDEX_CMP_PREDICATE_EQ:
      if (loom_index_value_facts_are_non_overlapping(rewriter, lhs, rhs)) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_INDEX_CMP_PREDICATE_NE:
      if (loom_index_value_facts_are_non_overlapping(rewriter, lhs, rhs)) {
        *out_result = true;
        return true;
      }
      return false;
    case LOOM_INDEX_CMP_PREDICATE_SLT:
      if (lhs_facts.range_hi < rhs_facts.range_lo) {
        *out_result = true;
        return true;
      }
      if (lhs_facts.range_lo >= rhs_facts.range_hi) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_INDEX_CMP_PREDICATE_SLE:
      if (lhs_facts.range_hi <= rhs_facts.range_lo) {
        *out_result = true;
        return true;
      }
      if (lhs_facts.range_lo > rhs_facts.range_hi) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_INDEX_CMP_PREDICATE_SGT:
      if (lhs_facts.range_lo > rhs_facts.range_hi) {
        *out_result = true;
        return true;
      }
      if (lhs_facts.range_hi <= rhs_facts.range_lo) {
        *out_result = false;
        return true;
      }
      return false;
    case LOOM_INDEX_CMP_PREDICATE_SGE:
      if (lhs_facts.range_lo >= rhs_facts.range_hi) {
        *out_result = true;
        return true;
      }
      if (lhs_facts.range_hi < rhs_facts.range_lo) {
        *out_result = false;
        return true;
      }
      return false;
    default:
      return false;
  }
}

static bool loom_index_unsigned_range_cmp_result(loom_rewriter_t* rewriter,
                                                 uint8_t predicate,
                                                 loom_value_id_t lhs,
                                                 loom_value_id_t rhs,
                                                 bool* out_result) {
  loom_value_facts_t lhs_facts = loom_rewriter_value_facts(rewriter, lhs);
  loom_value_facts_t rhs_facts = loom_rewriter_value_facts(rewriter, rhs);
  if (!loom_value_facts_is_non_negative(lhs_facts) ||
      !loom_value_facts_is_non_negative(rhs_facts)) {
    return false;
  }
  switch ((loom_index_cmp_predicate_t)predicate) {
    case LOOM_INDEX_CMP_PREDICATE_ULT:
      return loom_index_signed_range_cmp_result(
          rewriter, LOOM_INDEX_CMP_PREDICATE_SLT, lhs, rhs, out_result);
    case LOOM_INDEX_CMP_PREDICATE_ULE:
      return loom_index_signed_range_cmp_result(
          rewriter, LOOM_INDEX_CMP_PREDICATE_SLE, lhs, rhs, out_result);
    case LOOM_INDEX_CMP_PREDICATE_UGT:
      return loom_index_signed_range_cmp_result(
          rewriter, LOOM_INDEX_CMP_PREDICATE_SGT, lhs, rhs, out_result);
    case LOOM_INDEX_CMP_PREDICATE_UGE:
      return loom_index_signed_range_cmp_result(
          rewriter, LOOM_INDEX_CMP_PREDICATE_SGE, lhs, rhs, out_result);
    default:
      return false;
  }
}

iree_status_t loom_index_cmp_canonicalize(loom_op_t* op,
                                          loom_rewriter_t* rewriter) {
  loom_value_id_t lhs = loom_index_cmp_lhs(op);
  loom_value_id_t rhs = loom_index_cmp_rhs(op);
  uint8_t predicate = loom_index_cmp_predicate(op);
  bool result = false;
  if (lhs == rhs && loom_index_same_operand_cmp_result(predicate, &result)) {
    return loom_index_replace_single_result_with_i1_constant(op, rewriter,
                                                             result);
  }
  if (loom_index_signed_range_cmp_result(rewriter, predicate, lhs, rhs,
                                         &result) ||
      loom_index_unsigned_range_cmp_result(rewriter, predicate, lhs, rhs,
                                           &result)) {
    return loom_index_replace_single_result_with_i1_constant(op, rewriter,
                                                             result);
  }
  return iree_ok_status();
}

iree_status_t loom_index_select_canonicalize(loom_op_t* op,
                                             loom_rewriter_t* rewriter) {
  loom_value_id_t true_value = loom_index_select_true_value(op);
  loom_value_id_t false_value = loom_index_select_false_value(op);
  if (true_value == false_value) {
    return loom_index_replace_single_result_with_value(op, rewriter,
                                                       true_value);
  }

  loom_value_facts_t condition_facts =
      loom_rewriter_value_facts(rewriter, loom_index_select_condition(op));
  if (!loom_value_facts_is_exact(condition_facts)) return iree_ok_status();
  return loom_index_replace_single_result_with_value(
      op, rewriter, condition_facts.range_lo ? true_value : false_value);
}
