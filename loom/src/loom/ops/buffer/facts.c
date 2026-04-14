// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the buffer dialect.

#include "loom/ir/facts.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/view/reference.h"

static loom_value_facts_t loom_buffer_nonnegative_unknown_facts(void) {
  return loom_value_facts_make(0, INT64_MAX, 1);
}

static loom_value_facts_t loom_buffer_clamp_nonnegative(
    loom_value_facts_t facts) {
  if (loom_value_facts_is_float(facts) || facts.range_hi < 0) {
    return loom_buffer_nonnegative_unknown_facts();
  }
  int64_t lower_bound = facts.range_lo < 0 ? 0 : facts.range_lo;
  int64_t upper_bound = facts.range_hi < 0 ? 0 : facts.range_hi;
  int64_t divisor = facts.known_divisor > 0 ? facts.known_divisor : 1;
  return loom_value_facts_make(lower_bound, upper_bound, divisor);
}

static loom_value_fact_memory_space_t loom_buffer_memory_space_from_attr(
    uint8_t value) {
  switch ((loom_buffer_memory_space_t)value) {
    case LOOM_BUFFER_MEMORY_SPACE_GLOBAL:
      return LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
    case LOOM_BUFFER_MEMORY_SPACE_WORKGROUP:
      return LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
    case LOOM_BUFFER_MEMORY_SPACE_PRIVATE:
      return LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE;
    case LOOM_BUFFER_MEMORY_SPACE_CONSTANT:
      return LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT;
    case LOOM_BUFFER_MEMORY_SPACE_HOST:
      return LOOM_VALUE_FACT_MEMORY_SPACE_HOST;
    case LOOM_BUFFER_MEMORY_SPACE_DESCRIPTOR:
      return LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR;
    case LOOM_BUFFER_MEMORY_SPACE_UNKNOWN:
    default:
      return LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  }
}

static loom_value_fact_buffer_reference_t loom_buffer_default_reference(
    loom_value_id_t buffer_value_id) {
  return (loom_value_fact_buffer_reference_t){
      .maximum_byte_extent = loom_buffer_nonnegative_unknown_facts(),
      .minimum_alignment = 1,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = buffer_value_id,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
}

iree_status_t loom_buffer_alloca_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  int64_t base_alignment = loom_buffer_alloca_base_alignment(op);
  loom_value_fact_buffer_reference_t reference = {
      .maximum_byte_extent = loom_buffer_clamp_nonnegative(operand_facts[0]),
      .minimum_alignment = base_alignment > 0 ? (uint64_t)base_alignment : 1,
      .memory_space = loom_buffer_memory_space_from_attr(
          loom_buffer_alloca_memory_space(op)),
      .root_value_id = loom_buffer_alloca_result(op),
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_NON_NULL,
  };
  return loom_value_facts_make_buffer_reference(context, reference,
                                                &result_facts[0]);
}

iree_status_t loom_buffer_assume_memory_space_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_buffer_reference_t reference =
      loom_buffer_default_reference(loom_buffer_assume_memory_space_buffer(op));
  (void)loom_value_facts_query_buffer_reference(context, operand_facts[0],
                                                &reference);
  reference.memory_space = loom_buffer_memory_space_from_attr(
      loom_buffer_assume_memory_space_memory_space(op));
  return loom_value_facts_make_buffer_reference(context, reference,
                                                &result_facts[0]);
}

iree_status_t loom_buffer_view_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_buffer_view_result(op));
  return loom_view_reference_make_buffer_view(
      context, module, loom_buffer_view_buffer(op), operand_facts[0],
      operand_facts[1], result_type, &result_facts[0]);
}
