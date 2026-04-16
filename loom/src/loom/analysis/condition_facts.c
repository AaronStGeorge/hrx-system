// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_facts.h"

#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"

void loom_condition_fact_set_initialize(
    loom_condition_integer_relation_t* integer_relation_storage,
    iree_host_size_t integer_relation_capacity,
    loom_condition_fact_set_t* out_facts) {
  *out_facts = (loom_condition_fact_set_t){
      .integer_relations = integer_relation_storage,
      .integer_relation_count = 0,
      .integer_relation_capacity = integer_relation_capacity,
  };
}

void loom_condition_fact_set_reset(loom_condition_fact_set_t* facts) {
  facts->integer_relation_count = 0;
}

static loom_condition_integer_operand_t loom_condition_value_operand(
    loom_value_id_t value_id) {
  return (loom_condition_integer_operand_t){
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = value_id,
      .constant = 0,
  };
}

static iree_status_t loom_condition_fact_set_append_integer_relation(
    loom_condition_fact_set_t* facts,
    loom_condition_integer_relation_t relation) {
  if (!facts->integer_relations ||
      facts->integer_relation_count >= facts->integer_relation_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition integer relation capacity exceeded");
  }
  facts->integer_relations[facts->integer_relation_count++] = relation;
  return iree_ok_status();
}

static loom_value_facts_t loom_condition_lookup_facts(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id) {
  if (!fact_table) return loom_value_facts_unknown();
  return loom_value_fact_table_lookup(fact_table, value_id);
}

static bool loom_condition_value_exact_integer(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    int64_t* out_value) {
  loom_value_facts_t facts = loom_condition_lookup_facts(fact_table, value_id);
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_condition_values_are_non_negative(
    const loom_value_fact_table_t* fact_table, loom_value_id_t left_value,
    loom_value_id_t right_value) {
  return loom_value_facts_is_non_negative(
             loom_condition_lookup_facts(fact_table, left_value)) &&
         loom_value_facts_is_non_negative(
             loom_condition_lookup_facts(fact_table, right_value));
}

static bool loom_condition_index_predicate_relation(
    uint8_t predicate, const loom_value_fact_table_t* fact_table,
    loom_value_id_t left_value, loom_value_id_t right_value,
    loom_symbolic_integer_relation_t* out_relation) {
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
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_ULE:
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_UGT:
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_UGE:
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    default:
      return false;
  }
}

static bool loom_condition_scalar_cmpi_predicate_relation(
    uint8_t predicate, const loom_value_fact_table_t* fact_table,
    loom_value_id_t left_value, loom_value_id_t right_value,
    loom_symbolic_integer_relation_t* out_relation) {
  switch ((loom_scalar_cmpi_predicate_t)predicate) {
    case LOOM_SCALAR_CMPI_PREDICATE_EQ:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_NE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_NE;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_SLT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_SLE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_SGT:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_SGE:
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_ULT:
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_ULE:
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_LE;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_UGT:
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GT;
      return true;
    case LOOM_SCALAR_CMPI_PREDICATE_UGE:
      if (!loom_condition_values_are_non_negative(fact_table, left_value,
                                                  right_value)) {
        return false;
      }
      *out_relation = LOOM_SYMBOLIC_INTEGER_RELATION_GE;
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_condition_facts_query_integer_compare(
    loom_condition_fact_set_t* facts, const loom_value_fact_table_t* fact_table,
    loom_value_id_t left_value, loom_value_id_t right_value, uint8_t predicate,
    bool assumed_truth,
    bool (*predicate_relation)(uint8_t, const loom_value_fact_table_t*,
                               loom_value_id_t, loom_value_id_t,
                               loom_symbolic_integer_relation_t*)) {
  loom_symbolic_integer_relation_t relation = LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  if (!predicate_relation(predicate, fact_table, left_value, right_value,
                          &relation)) {
    return iree_ok_status();
  }
  if (!assumed_truth) {
    relation = loom_symbolic_integer_relation_invert(relation);
  }

  loom_condition_integer_relation_t assertion = {
      .relation = relation,
      .left = loom_condition_value_operand(left_value),
      .right = loom_condition_value_operand(right_value),
  };
  return loom_condition_fact_set_append_integer_relation(facts, assertion);
}

iree_status_t loom_condition_facts_query(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t condition_value, bool assumed_truth,
    loom_condition_fact_set_t* out_facts) {
  if (!out_facts) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "output condition fact set is required");
  }
  loom_condition_fact_set_reset(out_facts);
  if (!module || condition_value >= module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* value = loom_module_value(module, condition_value);
  if (loom_value_is_block_arg(value)) return iree_ok_status();
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) return iree_ok_status();

  switch (defining_op->kind) {
    case LOOM_OP_INDEX_CMP:
      return loom_condition_facts_query_integer_compare(
          out_facts, fact_table, loom_index_cmp_lhs(defining_op),
          loom_index_cmp_rhs(defining_op),
          loom_index_cmp_predicate(defining_op), assumed_truth,
          loom_condition_index_predicate_relation);
    case LOOM_OP_SCALAR_CMPI:
      return loom_condition_facts_query_integer_compare(
          out_facts, fact_table, loom_scalar_cmpi_lhs(defining_op),
          loom_scalar_cmpi_rhs(defining_op),
          loom_scalar_cmpi_predicate(defining_op), assumed_truth,
          loom_condition_scalar_cmpi_predicate_relation);
    default:
      return iree_ok_status();
  }
}

static bool loom_condition_relation_to_predicate_kind(
    loom_symbolic_integer_relation_t relation, uint8_t* out_kind) {
  switch (relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
      *out_kind = LOOM_PREDICATE_EQ;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      *out_kind = LOOM_PREDICATE_LT;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      *out_kind = LOOM_PREDICATE_LE;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      *out_kind = LOOM_PREDICATE_GT;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      *out_kind = LOOM_PREDICATE_GE;
      return true;
    case LOOM_SYMBOLIC_INTEGER_RELATION_NE:
    default:
      return false;
  }
}

static bool loom_condition_operand_matches_value(
    loom_condition_integer_operand_t operand, loom_value_id_t value_id) {
  return operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
         operand.value_id == value_id;
}

static bool loom_condition_operand_exact_integer(
    loom_condition_integer_operand_t operand,
    const loom_value_fact_table_t* fact_table, int64_t* out_value) {
  switch (operand.kind) {
    case LOOM_CONDITION_INTEGER_OPERAND_CONSTANT:
      *out_value = operand.constant;
      return true;
    case LOOM_CONDITION_INTEGER_OPERAND_VALUE:
      return loom_condition_value_exact_integer(fact_table, operand.value_id,
                                                out_value);
    default:
      return false;
  }
}

bool loom_condition_integer_relation_apply_to_value_facts(
    const loom_condition_integer_relation_t* relation,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_facts_t* inout_facts) {
  loom_symbolic_integer_relation_t normalized_relation = relation->relation;
  int64_t constant = 0;
  if (loom_condition_operand_matches_value(relation->left, value_id)) {
    if (!loom_condition_operand_exact_integer(relation->right, fact_table,
                                              &constant)) {
      return false;
    }
  } else if (loom_condition_operand_matches_value(relation->right, value_id)) {
    if (!loom_condition_operand_exact_integer(relation->left, fact_table,
                                              &constant)) {
      return false;
    }
    normalized_relation =
        loom_symbolic_integer_relation_swap(normalized_relation);
  } else {
    return false;
  }

  uint8_t predicate_kind = 0;
  if (!loom_condition_relation_to_predicate_kind(normalized_relation,
                                                 &predicate_kind)) {
    return false;
  }
  loom_predicate_t predicate = {
      .kind = predicate_kind,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST, 0},
      .args = {value_id, constant, 0},
  };
  loom_value_facts_apply_predicate(inout_facts, &predicate);
  return true;
}

bool loom_condition_fact_set_apply_to_value_facts(
    const loom_condition_fact_set_t* facts,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_facts_t* inout_facts) {
  bool applied = false;
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    applied |= loom_condition_integer_relation_apply_to_value_facts(
        &facts->integer_relations[i], fact_table, value_id, inout_facts);
  }
  return applied;
}
