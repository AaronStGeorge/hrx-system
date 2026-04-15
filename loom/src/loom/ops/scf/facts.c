// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the scf dialect.

#include "loom/ir/facts.h"

#include "loom/ir/attribute.h"
#include "loom/ops/scf/ops.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool loom_scf_lookup_key_matches_selector_facts(
    loom_value_facts_t selector_facts, int64_t key) {
  if (loom_value_facts_is_float(selector_facts)) return true;
  if (key < selector_facts.range_lo || key > selector_facts.range_hi) {
    return false;
  }
  int64_t divisor = selector_facts.known_divisor;
  return divisor <= 1 || key % divisor == 0;
}

static bool loom_scf_lookup_key_is_explicit(loom_attribute_t case_keys,
                                            int64_t key) {
  for (uint16_t i = 0; i < case_keys.count; ++i) {
    if (case_keys.i64_array[i] == key) return true;
    if (case_keys.i64_array[i] > key) return false;
  }
  return false;
}

static bool loom_scf_lookup_default_row_may_match(
    loom_value_facts_t selector_facts, loom_attribute_t case_keys) {
  if (loom_value_facts_is_float(selector_facts)) return true;
  if (loom_value_facts_is_exact(selector_facts)) {
    return !loom_scf_lookup_key_is_explicit(case_keys, selector_facts.range_lo);
  }

  int64_t span = 0;
  if (!loom_checked_sub_i64(selector_facts.range_hi, selector_facts.range_lo,
                            &span) ||
      span < 0 || span > 4096) {
    return true;
  }

  for (int64_t offset = 0; offset <= span; ++offset) {
    int64_t candidate = selector_facts.range_lo + offset;
    if (!loom_scf_lookup_key_matches_selector_facts(selector_facts,
                                                    candidate)) {
      continue;
    }
    if (!loom_scf_lookup_key_is_explicit(case_keys, candidate)) return true;
  }
  return false;
}

static void loom_scf_lookup_meet_row_facts(
    uint16_t result_count, iree_host_size_t row_index,
    const loom_value_facts_t* operand_facts, bool initialized_results,
    loom_value_facts_t* result_facts) {
  iree_host_size_t row_offset = 1 + row_index * result_count;
  for (uint16_t i = 0; i < result_count; ++i) {
    const loom_value_facts_t* candidate = &operand_facts[row_offset + i];
    if (!initialized_results) {
      result_facts[i] = *candidate;
    } else {
      loom_value_facts_meet(&result_facts[i], candidate, &result_facts[i]);
    }
  }
}

iree_status_t loom_scf_select_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] =
        operand_facts[0].range_lo ? operand_facts[1] : operand_facts[2];
    return iree_ok_status();
  }
  if (loom_value_facts_equal(operand_facts[1], operand_facts[2])) {
    result_facts[0] = operand_facts[1];
    return iree_ok_status();
  }
  loom_value_facts_meet(&operand_facts[1], &operand_facts[2], &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_scf_lookup_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  (void)context;
  (void)module;
  loom_attribute_t case_keys = loom_scf_lookup_case_keys(op);
  loom_value_slice_t values = loom_scf_lookup_values(op);
  if (op->result_count == 0 || case_keys.kind != LOOM_ATTR_I64_ARRAY ||
      (case_keys.count > 0 && !case_keys.i64_array)) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      result_facts[i] = loom_value_facts_unknown();
    }
    return iree_ok_status();
  }

  iree_host_size_t row_count = (iree_host_size_t)case_keys.count + 1;
  iree_host_size_t expected_value_count =
      row_count * (iree_host_size_t)op->result_count;
  if (values.count != expected_value_count) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      result_facts[i] = loom_value_facts_unknown();
    }
    return iree_ok_status();
  }

  loom_value_facts_t selector_facts = operand_facts[0];
  if (!loom_value_facts_is_float(selector_facts) &&
      loom_value_facts_is_exact(selector_facts)) {
    iree_host_size_t row_index = case_keys.count;
    for (uint16_t i = 0; i < case_keys.count; ++i) {
      if (case_keys.i64_array[i] == selector_facts.range_lo) {
        row_index = i;
        break;
      }
    }
    for (uint16_t i = 0; i < op->result_count; ++i) {
      result_facts[i] = operand_facts[1 + row_index * op->result_count + i];
    }
    return iree_ok_status();
  }

  bool initialized_results = false;
  for (uint16_t i = 0; i < case_keys.count; ++i) {
    if (!loom_scf_lookup_key_matches_selector_facts(selector_facts,
                                                    case_keys.i64_array[i])) {
      continue;
    }
    loom_scf_lookup_meet_row_facts(op->result_count, i, operand_facts,
                                   initialized_results, result_facts);
    initialized_results = true;
  }
  if (loom_scf_lookup_default_row_may_match(selector_facts, case_keys)) {
    loom_scf_lookup_meet_row_facts(op->result_count, case_keys.count,
                                   operand_facts, initialized_results,
                                   result_facts);
    initialized_results = true;
  }
  if (!initialized_results) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      result_facts[i] = loom_value_facts_unknown();
    }
  }
  return iree_ok_status();
}
