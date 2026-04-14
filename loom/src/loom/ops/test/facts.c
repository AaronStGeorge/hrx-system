// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the test dialect.
//
// test.constant: produces exact facts from the constant attribute.
// test.fact_*: expose individual analysis facts as observable values
// for testing. Each reads one field from its input's facts and returns
// it as an exact constant, which the rewriter materializes into a
// scalar.constant during canonicalization.

#include "loom/ir/facts.h"

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"

//===----------------------------------------------------------------------===//
// test.addi
//===----------------------------------------------------------------------===//

iree_status_t loom_test_addi_facts(loom_fact_context_t* context,
                                   const loom_module_t* module,
                                   const loom_op_t* op,
                                   const loom_value_facts_t* operand_facts,
                                   loom_value_facts_t* result_facts) {
  loom_value_facts_addi(&operand_facts[0], &operand_facts[1], &result_facts[0]);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// test.constant
//===----------------------------------------------------------------------===//

iree_status_t loom_test_constant_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  loom_attribute_t attr = loom_op_attrs(op)[0];
  loom_value_id_t result_id = loom_test_constant_result(op);
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_scalar_type_is_float(loom_type_element_type(result_type))) {
    result_facts[0] = loom_value_facts_exact_f64(loom_attr_as_f64(attr));
  } else {
    result_facts[0] = loom_value_facts_exact_i64(loom_attr_as_i64(attr));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// test.fact_* — value facts inspection ops
//===----------------------------------------------------------------------===//
//
// Each reads one property from operand_facts[0] and returns an exact
// value. The rewriter's try_fold sees the exact output and materializes
// a scalar.constant, making the analysis state observable in .loom-test
// fixtures.

iree_status_t loom_test_fact_range_lo_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(operand_facts[0].range_lo);
  return iree_ok_status();
}

iree_status_t loom_test_fact_range_hi_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(operand_facts[0].range_hi);
  return iree_ok_status();
}

iree_status_t loom_test_fact_divisor_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(operand_facts[0].known_divisor);
  return iree_ok_status();
}

iree_status_t loom_test_fact_non_negative_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_is_non_negative(operand_facts[0]) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_non_zero_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_is_non_zero(operand_facts[0]) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_positive_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_is_positive(operand_facts[0]) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_power_of_two_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_is_power_of_two(operand_facts[0]) ? 1 : 0);
  return iree_ok_status();
}
