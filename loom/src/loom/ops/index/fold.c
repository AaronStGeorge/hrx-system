// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fold implementations for the index dialect.

#include "loom/ir/attribute.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"

#define LOOM_INDEX_BINARY_FOLD(name, transfer_fn)                        \
  void name(const loom_module_t* module, const loom_op_t* op,            \
            const loom_value_facts_t* operand_facts,                     \
            loom_value_facts_t* result_facts) {                          \
    transfer_fn(&operand_facts[0], &operand_facts[1], &result_facts[0]); \
  }

void loom_index_constant_fold(const loom_module_t* module, const loom_op_t* op,
                              const loom_value_facts_t* operand_facts,
                              loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_attr_as_i64(loom_index_constant_value(op)));
}

void loom_index_cast_fold(const loom_module_t* module, const loom_op_t* op,
                          const loom_value_facts_t* operand_facts,
                          loom_value_facts_t* result_facts) {
  result_facts[0] = operand_facts[0];
}

LOOM_INDEX_BINARY_FOLD(loom_index_add_fold, loom_value_facts_addi)
LOOM_INDEX_BINARY_FOLD(loom_index_sub_fold, loom_value_facts_subi)
LOOM_INDEX_BINARY_FOLD(loom_index_mul_fold, loom_value_facts_muli)

void loom_index_madd_fold(const loom_module_t* module, const loom_op_t* op,
                          const loom_value_facts_t* operand_facts,
                          loom_value_facts_t* result_facts) {
  loom_value_facts_fmai(&operand_facts[0], &operand_facts[1], &operand_facts[2],
                        &result_facts[0]);
}

void loom_index_cmp_fold(const loom_module_t* module, const loom_op_t* op,
                         const loom_value_facts_t* operand_facts,
                         loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_make(0, 1, 1);
}

void loom_index_select_fold(const loom_module_t* module, const loom_op_t* op,
                            const loom_value_facts_t* operand_facts,
                            loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] =
        operand_facts[0].range_lo ? operand_facts[1] : operand_facts[2];
    return;
  }
  loom_value_facts_meet(&operand_facts[1], &operand_facts[2], &result_facts[0]);
}

#undef LOOM_INDEX_BINARY_FOLD
