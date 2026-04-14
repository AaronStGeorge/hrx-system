// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the index dialect.

#include "loom/ir/facts.h"

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"

#define LOOM_INDEX_BINARY_FACTS(name, transfer_fn)                       \
  iree_status_t name(loom_fact_context_t* context,                       \
                     const loom_module_t* module, const loom_op_t* op,   \
                     const loom_value_facts_t* operand_facts,            \
                     loom_value_facts_t* result_facts) {                 \
    transfer_fn(&operand_facts[0], &operand_facts[1], &result_facts[0]); \
    return iree_ok_status();                                             \
  }

iree_status_t loom_index_constant_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_attr_as_i64(loom_index_constant_value(op)));
  return iree_ok_status();
}

iree_status_t loom_index_cast_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  result_facts[0] = operand_facts[0];
  return iree_ok_status();
}

iree_status_t loom_index_assume_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  uint16_t fact_count = op->operand_count < op->result_count ? op->operand_count
                                                             : op->result_count;
  for (uint16_t i = 0; i < fact_count; ++i) {
    result_facts[i] = operand_facts[i];
  }
  for (uint16_t i = fact_count; i < op->result_count; ++i) {
    result_facts[i] = loom_value_facts_unknown();
  }
  loom_attribute_t pred_attr = loom_op_attrs(op)[0];
  const loom_predicate_t* predicates = pred_attr.predicate_list;
  uint16_t predicate_count = pred_attr.count;
  for (uint16_t predicate_ordinal = 0; predicate_ordinal < predicate_count;
       ++predicate_ordinal) {
    const loom_predicate_t* predicate = &predicates[predicate_ordinal];
    if (predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE) continue;
    loom_value_slice_t values = loom_index_assume_values(op);
    loom_value_id_t target_id = (loom_value_id_t)predicate->args[0];
    uint16_t target = 0;
    bool found = false;
    for (uint16_t i = 0; i < values.count; ++i) {
      if (values.values[i] == target_id) {
        target = i;
        found = true;
        break;
      }
    }
    if (!found) continue;
    if (target < fact_count) {
      loom_value_facts_apply_predicate(&result_facts[target], predicate);
    }
  }
  return iree_ok_status();
}

LOOM_INDEX_BINARY_FACTS(loom_index_add_facts, loom_value_facts_addi)
LOOM_INDEX_BINARY_FACTS(loom_index_sub_facts, loom_value_facts_subi)
LOOM_INDEX_BINARY_FACTS(loom_index_mul_facts, loom_value_facts_muli)

iree_status_t loom_index_madd_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  loom_value_facts_fmai(&operand_facts[0], &operand_facts[1], &operand_facts[2],
                        &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_index_cmp_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_make(0, 1, 1);
  return iree_ok_status();
}

iree_status_t loom_index_select_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] =
        operand_facts[0].range_lo ? operand_facts[1] : operand_facts[2];
    return iree_ok_status();
  }
  loom_value_facts_meet(&operand_facts[1], &operand_facts[2], &result_facts[0]);
  return iree_ok_status();
}

#undef LOOM_INDEX_BINARY_FACTS
