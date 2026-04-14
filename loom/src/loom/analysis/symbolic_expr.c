// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_expr.h"

#include <stdlib.h>
#include <string.h>

#include "loom/ir/attribute.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Context storage
//===----------------------------------------------------------------------===//

enum loom_symbolic_expr_memo_state_e {
  LOOM_SYMBOLIC_EXPR_MEMO_EMPTY = 0,
  LOOM_SYMBOLIC_EXPR_MEMO_VISITING = 1,
  LOOM_SYMBOLIC_EXPR_MEMO_READY = 2,
};

struct loom_symbolic_expr_memo_entry_t {
  // Current memo state for this value ID.
  uint8_t state;

  // Cached expression when state is LOOM_SYMBOLIC_EXPR_MEMO_READY.
  loom_symbolic_expr_t expression;
};

void loom_symbolic_expr_context_initialize(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    iree_arena_allocator_t* arena, loom_symbolic_expr_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->module = module;
  out_context->fact_table = fact_table;
  out_context->arena = arena;
  out_context->maximum_term_count = LOOM_SYMBOLIC_EXPR_DEFAULT_TERM_LIMIT;
}

void loom_symbolic_expr_context_reset(loom_symbolic_expr_context_t* context) {
  if (context->memo_entries && context->memo_capacity > 0) {
    memset(context->memo_entries, 0,
           context->memo_capacity * sizeof(*context->memo_entries));
  }
}

static iree_status_t loom_symbolic_expr_ensure_memo_capacity(
    loom_symbolic_expr_context_t* context, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= context->memo_capacity) return iree_ok_status();
  iree_host_size_t old_capacity = context->memo_capacity;
  void* entries = context->memo_entries;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->arena, context->memo_capacity, minimum_capacity,
      sizeof(*context->memo_entries), &context->memo_capacity, &entries));
  context->memo_entries = (loom_symbolic_expr_memo_entry_t*)entries;
  memset(
      context->memo_entries + old_capacity, 0,
      (context->memo_capacity - old_capacity) * sizeof(*context->memo_entries));
  return iree_ok_status();
}

static iree_status_t loom_symbolic_expr_ensure_scratch_terms(
    loom_symbolic_expr_context_t* context, iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= context->scratch_term_capacity) {
    return iree_ok_status();
  }
  void* terms = context->scratch_terms;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->arena, 0, minimum_capacity, sizeof(*context->scratch_terms),
      &context->scratch_term_capacity, &terms));
  context->scratch_terms = (loom_symbolic_term_t*)terms;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Constructors and normalization
//===----------------------------------------------------------------------===//

static loom_value_facts_t loom_symbolic_expr_lookup_facts(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id) {
  if (!context->fact_table) return loom_value_facts_unknown();
  return loom_value_fact_table_lookup(context->fact_table, value_id);
}

static bool loom_symbolic_expr_exact_integer_facts(loom_value_facts_t facts,
                                                   int64_t* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_symbolic_expr_checked_term_count(iree_host_size_t left_count,
                                                  iree_host_size_t right_count,
                                                  iree_host_size_t* out_count) {
  if (left_count > (iree_host_size_t)-1 - right_count) return false;
  *out_count = left_count + right_count;
  return true;
}

void loom_symbolic_expr_unknown(loom_value_facts_t facts,
                                loom_symbolic_expr_t* out_expression) {
  *out_expression = (loom_symbolic_expr_t){
      .constant = 0,
      .terms = NULL,
      .term_count = 0,
      .facts = facts,
      .flags = 0,
  };
}

void loom_symbolic_expr_constant(int64_t value,
                                 loom_symbolic_expr_t* out_expression) {
  *out_expression = (loom_symbolic_expr_t){
      .constant = value,
      .terms = NULL,
      .term_count = 0,
      .facts = loom_value_facts_exact_i64(value),
      .flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR,
  };
}

static int loom_symbolic_expr_compare_terms(const void* left,
                                            const void* right) {
  const loom_symbolic_term_t* left_term = (const loom_symbolic_term_t*)left;
  const loom_symbolic_term_t* right_term = (const loom_symbolic_term_t*)right;
  if (left_term->value_id < right_term->value_id) return -1;
  if (left_term->value_id > right_term->value_id) return 1;
  return 0;
}

static iree_status_t loom_symbolic_expr_make_linear(
    loom_symbolic_expr_context_t* context, int64_t constant,
    loom_symbolic_term_t* terms, iree_host_size_t term_count,
    loom_value_facts_t facts, loom_symbolic_expr_t* out_expression) {
  if (term_count > context->maximum_term_count) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }
  if (term_count > 1) {
    qsort(terms, term_count, sizeof(*terms), loom_symbolic_expr_compare_terms);
  }

  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0; read_index < term_count;) {
    loom_value_id_t value_id = terms[read_index].value_id;
    int64_t coefficient = 0;
    while (read_index < term_count && terms[read_index].value_id == value_id) {
      int64_t new_coefficient = 0;
      if (!loom_checked_add_i64(coefficient, terms[read_index].coefficient,
                                &new_coefficient)) {
        loom_symbolic_expr_unknown(facts, out_expression);
        return iree_ok_status();
      }
      coefficient = new_coefficient;
      ++read_index;
    }
    if (coefficient == 0) continue;
    terms[write_index++] = (loom_symbolic_term_t){
        .coefficient = coefficient,
        .value_id = value_id,
    };
  }

  const loom_symbolic_term_t* retained_terms = NULL;
  if (write_index > 0) {
    loom_symbolic_term_t* copied_terms = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->arena, write_index,
                                                   sizeof(*copied_terms),
                                                   (void**)&copied_terms));
    memcpy(copied_terms, terms, write_index * sizeof(*copied_terms));
    retained_terms = copied_terms;
  }

  *out_expression = (loom_symbolic_expr_t){
      .constant = constant,
      .terms = retained_terms,
      .term_count = write_index,
      .facts = facts,
      .flags = LOOM_SYMBOLIC_EXPR_FLAG_LINEAR,
  };
  return iree_ok_status();
}

iree_status_t loom_symbolic_expr_value(loom_symbolic_expr_context_t* context,
                                       loom_value_id_t value_id,
                                       loom_symbolic_expr_t* out_expression) {
  loom_value_facts_t facts = loom_symbolic_expr_lookup_facts(context, value_id);
  int64_t exact_value = 0;
  if (loom_symbolic_expr_exact_integer_facts(facts, &exact_value)) {
    loom_symbolic_expr_constant(exact_value, out_expression);
    out_expression->facts = facts;
    return iree_ok_status();
  }
  if (!context->module || value_id >= context->module->values.count) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_ensure_scratch_terms(context, 1));
  context->scratch_terms[0] = (loom_symbolic_term_t){
      .coefficient = 1,
      .value_id = value_id,
  };
  return loom_symbolic_expr_make_linear(context, 0, context->scratch_terms, 1,
                                        facts, out_expression);
}

static void loom_symbolic_expr_override_facts(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_symbolic_expr_t* expression) {
  loom_value_facts_t facts = loom_symbolic_expr_lookup_facts(context, value_id);
  if (!loom_value_facts_is_unknown(facts)) expression->facts = facts;
}

static iree_status_t loom_symbolic_expr_add_or_sub(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression, bool subtract,
    loom_symbolic_expr_t* out_expression) {
  loom_value_facts_t facts = {0};
  if (subtract) {
    loom_value_facts_subi(&left_expression->facts, &right_expression->facts,
                          &facts);
  } else {
    loom_value_facts_addi(&left_expression->facts, &right_expression->facts,
                          &facts);
  }
  if (!loom_symbolic_expr_is_linear(left_expression) ||
      !loom_symbolic_expr_is_linear(right_expression)) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }

  int64_t constant = 0;
  bool constant_ok =
      subtract ? loom_checked_sub_i64(left_expression->constant,
                                      right_expression->constant, &constant)
               : loom_checked_add_i64(left_expression->constant,
                                      right_expression->constant, &constant);
  if (!constant_ok) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }

  iree_host_size_t term_count = 0;
  if (!loom_symbolic_expr_checked_term_count(left_expression->term_count,
                                             right_expression->term_count,
                                             &term_count)) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }
  if (term_count > context->maximum_term_count) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_ensure_scratch_terms(context, term_count));
  iree_host_size_t term_ordinal = 0;
  for (iree_host_size_t i = 0; i < left_expression->term_count; ++i) {
    context->scratch_terms[term_ordinal++] = left_expression->terms[i];
  }
  for (iree_host_size_t i = 0; i < right_expression->term_count; ++i) {
    int64_t coefficient = right_expression->terms[i].coefficient;
    if (subtract) {
      if (coefficient == INT64_MIN) {
        loom_symbolic_expr_unknown(facts, out_expression);
        return iree_ok_status();
      }
      coefficient = -coefficient;
    }
    context->scratch_terms[term_ordinal++] = (loom_symbolic_term_t){
        .coefficient = coefficient,
        .value_id = right_expression->terms[i].value_id,
    };
  }
  return loom_symbolic_expr_make_linear(context, constant,
                                        context->scratch_terms, term_ordinal,
                                        facts, out_expression);
}

iree_status_t loom_symbolic_expr_add(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_expr_t* out_expression) {
  return loom_symbolic_expr_add_or_sub(context, left_expression,
                                       right_expression, false, out_expression);
}

iree_status_t loom_symbolic_expr_sub(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_expr_t* out_expression) {
  return loom_symbolic_expr_add_or_sub(context, left_expression,
                                       right_expression, true, out_expression);
}

iree_status_t loom_symbolic_expr_mul_i64(loom_symbolic_expr_context_t* context,
                                         const loom_symbolic_expr_t* expression,
                                         int64_t multiplier,
                                         loom_symbolic_expr_t* out_expression) {
  loom_value_facts_t multiplier_facts = loom_value_facts_exact_i64(multiplier);
  loom_value_facts_t facts = {0};
  loom_value_facts_muli(&expression->facts, &multiplier_facts, &facts);
  if (!loom_symbolic_expr_is_linear(expression)) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }

  int64_t constant = 0;
  if (!loom_checked_mul_i64(expression->constant, multiplier, &constant)) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }
  if (expression->term_count > context->maximum_term_count) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_ensure_scratch_terms(context, expression->term_count));
  for (iree_host_size_t i = 0; i < expression->term_count; ++i) {
    int64_t coefficient = 0;
    if (!loom_checked_mul_i64(expression->terms[i].coefficient, multiplier,
                              &coefficient)) {
      loom_symbolic_expr_unknown(facts, out_expression);
      return iree_ok_status();
    }
    context->scratch_terms[i] = (loom_symbolic_term_t){
        .coefficient = coefficient,
        .value_id = expression->terms[i].value_id,
    };
  }
  return loom_symbolic_expr_make_linear(
      context, constant, context->scratch_terms, expression->term_count, facts,
      out_expression);
}

static bool loom_symbolic_expr_constant_value(
    const loom_symbolic_expr_t* expression, int64_t* out_value) {
  if (!loom_symbolic_expr_is_constant(expression)) return false;
  *out_value = expression->constant;
  return true;
}

//===----------------------------------------------------------------------===//
// Value expansion
//===----------------------------------------------------------------------===//

static iree_status_t loom_symbolic_expr_from_binary_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t left_value,
    loom_value_id_t right_value, bool subtract,
    loom_symbolic_expr_t* out_expression) {
  loom_symbolic_expr_t left_expression = {0};
  loom_symbolic_expr_t right_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, left_value, &left_expression));
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, right_value, &right_expression));
  return loom_symbolic_expr_add_or_sub(
      context, &left_expression, &right_expression, subtract, out_expression);
}

static iree_status_t loom_symbolic_expr_from_mul_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t result_value,
    loom_value_id_t left_value, loom_value_id_t right_value,
    loom_symbolic_expr_t* out_expression) {
  loom_symbolic_expr_t left_expression = {0};
  loom_symbolic_expr_t right_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, left_value, &left_expression));
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, right_value, &right_expression));
  int64_t left_constant = 0;
  int64_t right_constant = 0;
  if (loom_symbolic_expr_constant_value(&left_expression, &left_constant)) {
    return loom_symbolic_expr_mul_i64(context, &right_expression, left_constant,
                                      out_expression);
  }
  if (loom_symbolic_expr_constant_value(&right_expression, &right_constant)) {
    return loom_symbolic_expr_mul_i64(context, &left_expression, right_constant,
                                      out_expression);
  }
  return loom_symbolic_expr_value(context, result_value, out_expression);
}

static iree_status_t loom_symbolic_expr_from_madd_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t result_value,
    loom_value_id_t a_value, loom_value_id_t b_value, loom_value_id_t c_value,
    loom_symbolic_expr_t* out_expression) {
  loom_symbolic_expr_t product_expression = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_mul_value(
      context, result_value, a_value, b_value, &product_expression));
  if (product_expression.term_count == 1 &&
      product_expression.terms[0].value_id == result_value &&
      product_expression.constant == 0) {
    return loom_symbolic_expr_value(context, result_value, out_expression);
  }
  loom_symbolic_expr_t c_expression = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_from_value(context, c_value, &c_expression));
  return loom_symbolic_expr_add(context, &product_expression, &c_expression,
                                out_expression);
}

static iree_status_t loom_symbolic_expr_from_select_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t result_value,
    loom_value_id_t condition_value, loom_value_id_t true_value,
    loom_value_id_t false_value, loom_symbolic_expr_t* out_expression) {
  loom_symbolic_expr_t condition_expression = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(context, condition_value,
                                                     &condition_expression));
  int64_t condition = 0;
  if (loom_symbolic_expr_constant_value(&condition_expression, &condition)) {
    return loom_symbolic_expr_from_value(
        context, condition ? true_value : false_value, out_expression);
  }
  return loom_symbolic_expr_value(context, result_value, out_expression);
}

static iree_status_t loom_symbolic_expr_from_assume_value(
    loom_symbolic_expr_context_t* context, const loom_value_slice_t values,
    uint16_t result_index, loom_symbolic_expr_t* out_expression) {
  if (result_index >= values.count) {
    loom_symbolic_expr_unknown(loom_value_facts_unknown(), out_expression);
    return iree_ok_status();
  }
  return loom_symbolic_expr_from_value(context, values.values[result_index],
                                       out_expression);
}

static iree_status_t loom_symbolic_expr_from_value_uncached(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_symbolic_expr_t* out_expression) {
  loom_value_facts_t facts = loom_symbolic_expr_lookup_facts(context, value_id);
  int64_t exact_value = 0;
  if (loom_symbolic_expr_exact_integer_facts(facts, &exact_value)) {
    loom_symbolic_expr_constant(exact_value, out_expression);
    out_expression->facts = facts;
    return iree_ok_status();
  }
  if (!context->module || value_id >= context->module->values.count) {
    loom_symbolic_expr_unknown(facts, out_expression);
    return iree_ok_status();
  }

  const loom_value_t* value = loom_module_value(context->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_symbolic_expr_value(context, value_id, out_expression);
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    return loom_symbolic_expr_value(context, value_id, out_expression);
  }

  iree_status_t status = iree_ok_status();
  switch (defining_op->kind) {
    case LOOM_OP_INDEX_CONSTANT: {
      loom_attribute_t value_attr = loom_index_constant_value(defining_op);
      if (value_attr.kind == LOOM_ATTR_I64) {
        loom_symbolic_expr_constant(loom_attr_as_i64(value_attr),
                                    out_expression);
      } else {
        status = loom_symbolic_expr_value(context, value_id, out_expression);
      }
      break;
    }
    case LOOM_OP_INDEX_CAST:
      status = loom_symbolic_expr_from_value(
          context, loom_index_cast_input(defining_op), out_expression);
      break;
    case LOOM_OP_INDEX_ASSUME:
      status = loom_symbolic_expr_from_assume_value(
          context, loom_index_assume_values(defining_op),
          loom_value_def_index(value), out_expression);
      break;
    case LOOM_OP_INDEX_ADD:
      status = loom_symbolic_expr_from_binary_value(
          context, loom_index_add_lhs(defining_op),
          loom_index_add_rhs(defining_op), false, out_expression);
      break;
    case LOOM_OP_INDEX_SUB:
      status = loom_symbolic_expr_from_binary_value(
          context, loom_index_sub_lhs(defining_op),
          loom_index_sub_rhs(defining_op), true, out_expression);
      break;
    case LOOM_OP_INDEX_MUL:
      status = loom_symbolic_expr_from_mul_value(
          context, value_id, loom_index_mul_lhs(defining_op),
          loom_index_mul_rhs(defining_op), out_expression);
      break;
    case LOOM_OP_INDEX_MADD:
      status = loom_symbolic_expr_from_madd_value(
          context, value_id, loom_index_madd_a(defining_op),
          loom_index_madd_b(defining_op), loom_index_madd_c(defining_op),
          out_expression);
      break;
    case LOOM_OP_INDEX_SELECT:
      status = loom_symbolic_expr_from_select_value(
          context, value_id, loom_index_select_condition(defining_op),
          loom_index_select_true_value(defining_op),
          loom_index_select_false_value(defining_op), out_expression);
      break;
    case LOOM_OP_SCALAR_CONSTANT: {
      loom_attribute_t value_attr = loom_scalar_constant_value(defining_op);
      if (value_attr.kind == LOOM_ATTR_I64) {
        loom_symbolic_expr_constant(loom_attr_as_i64(value_attr),
                                    out_expression);
      } else {
        status = loom_symbolic_expr_value(context, value_id, out_expression);
      }
      break;
    }
    case LOOM_OP_SCALAR_ADDI:
      status = loom_symbolic_expr_from_binary_value(
          context, loom_scalar_addi_lhs(defining_op),
          loom_scalar_addi_rhs(defining_op), false, out_expression);
      break;
    case LOOM_OP_SCALAR_SUBI:
      status = loom_symbolic_expr_from_binary_value(
          context, loom_scalar_subi_lhs(defining_op),
          loom_scalar_subi_rhs(defining_op), true, out_expression);
      break;
    case LOOM_OP_SCALAR_MULI:
      status = loom_symbolic_expr_from_mul_value(
          context, value_id, loom_scalar_muli_lhs(defining_op),
          loom_scalar_muli_rhs(defining_op), out_expression);
      break;
    case LOOM_OP_SCALAR_NEGI: {
      loom_symbolic_expr_t input_expression = {0};
      status = loom_symbolic_expr_from_value(
          context, loom_scalar_negi_input(defining_op), &input_expression);
      if (iree_status_is_ok(status)) {
        status = loom_symbolic_expr_mul_i64(context, &input_expression, -1,
                                            out_expression);
      }
      break;
    }
    case LOOM_OP_SCALAR_FMAI:
      status = loom_symbolic_expr_from_madd_value(
          context, value_id, loom_scalar_fmai_a(defining_op),
          loom_scalar_fmai_b(defining_op), loom_scalar_fmai_c(defining_op),
          out_expression);
      break;
    case LOOM_OP_SCALAR_ASSUME:
      status = loom_symbolic_expr_from_assume_value(
          context, loom_scalar_assume_values(defining_op),
          loom_value_def_index(value), out_expression);
      break;
    case LOOM_OP_SCALAR_SELECT:
      status = loom_symbolic_expr_from_select_value(
          context, value_id, loom_scalar_select_condition(defining_op),
          loom_scalar_select_true_value(defining_op),
          loom_scalar_select_false_value(defining_op), out_expression);
      break;
    default:
      status = loom_symbolic_expr_value(context, value_id, out_expression);
      break;
  }
  if (iree_status_is_ok(status)) {
    if (!loom_symbolic_expr_is_linear(out_expression)) {
      status = loom_symbolic_expr_value(context, value_id, out_expression);
    }
  }
  if (iree_status_is_ok(status)) {
    loom_symbolic_expr_override_facts(context, value_id, out_expression);
  }
  return status;
}

iree_status_t loom_symbolic_expr_from_value(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    loom_symbolic_expr_t* out_expression) {
  if (!context->module || value_id >= context->module->values.count) {
    loom_symbolic_expr_unknown(loom_value_facts_unknown(), out_expression);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_ensure_memo_capacity(context, value_id + 1));
  loom_symbolic_expr_memo_entry_t* entry = &context->memo_entries[value_id];
  if (entry->state == LOOM_SYMBOLIC_EXPR_MEMO_READY) {
    *out_expression = entry->expression;
    return iree_ok_status();
  }
  if (entry->state == LOOM_SYMBOLIC_EXPR_MEMO_VISITING) {
    return loom_symbolic_expr_value(context, value_id, out_expression);
  }

  entry->state = LOOM_SYMBOLIC_EXPR_MEMO_VISITING;
  iree_status_t status =
      loom_symbolic_expr_from_value_uncached(context, value_id, out_expression);
  if (iree_status_is_ok(status)) {
    entry->expression = *out_expression;
    entry->state = LOOM_SYMBOLIC_EXPR_MEMO_READY;
  } else {
    entry->state = LOOM_SYMBOLIC_EXPR_MEMO_EMPTY;
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Proofs
//===----------------------------------------------------------------------===//

static loom_symbolic_proof_result_t loom_symbolic_expr_prove_le_by_facts(
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression) {
  if (loom_value_facts_is_float(left_expression->facts) ||
      loom_value_facts_is_float(right_expression->facts)) {
    return LOOM_SYMBOLIC_PROOF_UNKNOWN;
  }
  if (left_expression->facts.range_hi <= right_expression->facts.range_lo) {
    return LOOM_SYMBOLIC_PROOF_TRUE;
  }
  if (left_expression->facts.range_lo > right_expression->facts.range_hi) {
    return LOOM_SYMBOLIC_PROOF_FALSE;
  }
  return LOOM_SYMBOLIC_PROOF_UNKNOWN;
}

static bool loom_symbolic_expr_accumulate_checked(int64_t term_min,
                                                  int64_t term_max,
                                                  int64_t* inout_min,
                                                  int64_t* inout_max) {
  int64_t new_min = 0;
  int64_t new_max = 0;
  if (!loom_checked_add_i64(*inout_min, term_min, &new_min)) return false;
  if (!loom_checked_add_i64(*inout_max, term_max, &new_max)) return false;
  *inout_min = new_min;
  *inout_max = new_max;
  return true;
}

static bool loom_symbolic_expr_term_interval(
    const loom_symbolic_expr_context_t* context,
    const loom_symbolic_term_t term, int64_t* out_min, int64_t* out_max) {
  loom_value_facts_t facts =
      loom_symbolic_expr_lookup_facts(context, term.value_id);
  if (loom_value_facts_is_unknown(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  int64_t lower_product = 0;
  int64_t upper_product = 0;
  if (term.coefficient >= 0) {
    if (!loom_checked_mul_i64(term.coefficient, facts.range_lo,
                              &lower_product)) {
      return false;
    }
    if (!loom_checked_mul_i64(term.coefficient, facts.range_hi,
                              &upper_product)) {
      return false;
    }
  } else {
    if (!loom_checked_mul_i64(term.coefficient, facts.range_hi,
                              &lower_product)) {
      return false;
    }
    if (!loom_checked_mul_i64(term.coefficient, facts.range_lo,
                              &upper_product)) {
      return false;
    }
  }
  *out_min = lower_product;
  *out_max = upper_product;
  return true;
}

static iree_status_t loom_symbolic_expr_prove_le_linear(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result) {
  int64_t constant = 0;
  if (!loom_checked_sub_i64(left_expression->constant,
                            right_expression->constant, &constant)) {
    *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    return iree_ok_status();
  }

  iree_host_size_t term_count = 0;
  if (!loom_symbolic_expr_checked_term_count(left_expression->term_count,
                                             right_expression->term_count,
                                             &term_count)) {
    *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    return iree_ok_status();
  }
  if (term_count > context->maximum_term_count) {
    *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_ensure_scratch_terms(context, term_count));
  iree_host_size_t term_ordinal = 0;
  for (iree_host_size_t i = 0; i < left_expression->term_count; ++i) {
    context->scratch_terms[term_ordinal++] = left_expression->terms[i];
  }
  for (iree_host_size_t i = 0; i < right_expression->term_count; ++i) {
    int64_t coefficient = right_expression->terms[i].coefficient;
    if (coefficient == INT64_MIN) {
      *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
      return iree_ok_status();
    }
    context->scratch_terms[term_ordinal++] = (loom_symbolic_term_t){
        .coefficient = -coefficient,
        .value_id = right_expression->terms[i].value_id,
    };
  }
  if (term_ordinal > 1) {
    qsort(context->scratch_terms, term_ordinal, sizeof(*context->scratch_terms),
          loom_symbolic_expr_compare_terms);
  }

  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0; read_index < term_ordinal;) {
    loom_value_id_t value_id = context->scratch_terms[read_index].value_id;
    int64_t coefficient = 0;
    while (read_index < term_ordinal &&
           context->scratch_terms[read_index].value_id == value_id) {
      int64_t new_coefficient = 0;
      if (!loom_checked_add_i64(coefficient,
                                context->scratch_terms[read_index].coefficient,
                                &new_coefficient)) {
        *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
        return iree_ok_status();
      }
      coefficient = new_coefficient;
      ++read_index;
    }
    if (coefficient == 0) continue;
    context->scratch_terms[write_index++] = (loom_symbolic_term_t){
        .coefficient = coefficient,
        .value_id = value_id,
    };
  }

  if (write_index == 0) {
    *out_result =
        constant <= 0 ? LOOM_SYMBOLIC_PROOF_TRUE : LOOM_SYMBOLIC_PROOF_FALSE;
    return iree_ok_status();
  }

  int64_t minimum = constant;
  int64_t maximum = constant;
  for (iree_host_size_t i = 0; i < write_index; ++i) {
    int64_t term_minimum = 0;
    int64_t term_maximum = 0;
    if (!loom_symbolic_expr_term_interval(context, context->scratch_terms[i],
                                          &term_minimum, &term_maximum) ||
        !loom_symbolic_expr_accumulate_checked(term_minimum, term_maximum,
                                               &minimum, &maximum)) {
      *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
      return iree_ok_status();
    }
  }
  if (maximum <= 0) {
    *out_result = LOOM_SYMBOLIC_PROOF_TRUE;
  } else if (minimum > 0) {
    *out_result = LOOM_SYMBOLIC_PROOF_FALSE;
  } else {
    *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  }
  return iree_ok_status();
}

iree_status_t loom_symbolic_expr_prove_le(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_expr_t* left_expression,
    const loom_symbolic_expr_t* right_expression,
    loom_symbolic_proof_result_t* out_result) {
  *out_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  if (loom_symbolic_expr_is_linear(left_expression) &&
      loom_symbolic_expr_is_linear(right_expression)) {
    IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_le_linear(
        context, left_expression, right_expression, out_result));
    if (*out_result != LOOM_SYMBOLIC_PROOF_UNKNOWN) return iree_ok_status();
  }
  // Expanded expressions may still carry stronger facts from their defining
  // SSA value, such as index.assume range facts on a value that algebraically
  // expands back to its unconstrained source.
  *out_result =
      loom_symbolic_expr_prove_le_by_facts(left_expression, right_expression);
  return iree_ok_status();
}
