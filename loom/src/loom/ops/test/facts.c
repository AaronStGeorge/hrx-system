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
#include "loom/util/fact_table.h"

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

iree_status_t loom_test_fact_is_vector_iota_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_query_vector_iota(context, operand_facts[0], NULL) ? 1
                                                                          : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_is_vector_prefix_mask_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_query_vector_prefix_mask(context, operand_facts[0], NULL)
          ? 1
          : 0);
  return iree_ok_status();
}

static loom_value_fact_encoding_summary_t loom_test_encoding_summary_or_empty(
    const loom_fact_context_t* context, loom_value_facts_t facts) {
  loom_value_fact_encoding_summary_t summary = {0};
  (void)loom_value_facts_query_encoding_summary(context, facts, &summary);
  return summary;
}

iree_status_t loom_test_fact_encoding_layout_kind_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_encoding_summary_t summary =
      loom_test_encoding_summary_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64((int64_t)summary.address_layout.kind);
  return iree_ok_status();
}

iree_status_t loom_test_fact_encoding_layout_stride_hi_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_encoding_summary_t summary =
      loom_test_encoding_summary_or_empty(context, operand_facts[0]);
  int64_t axis = loom_attr_as_i64(loom_op_attrs(op)[0]);
  if (summary.address_layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED ||
      axis < 0 || axis >= summary.address_layout.rank ||
      !summary.address_layout.strides) {
    result_facts[0] = loom_value_facts_exact_i64(INT64_MIN);
    return iree_ok_status();
  }
  result_facts[0] =
      loom_value_facts_exact_i64(summary.address_layout.strides[axis].range_hi);
  return iree_ok_status();
}

iree_status_t loom_test_fact_is_buffer_reference_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_query_buffer_reference(context, operand_facts[0], NULL)
          ? 1
          : 0);
  return iree_ok_status();
}

iree_status_t loom_test_fact_is_view_reference_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_exact_i64(
      loom_value_facts_query_view_reference(context, operand_facts[0], NULL)
          ? 1
          : 0);
  return iree_ok_status();
}

static int64_t loom_test_memory_space_or_unknown(
    loom_value_fact_memory_space_t memory_space) {
  if (memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN) return -1;
  return (int64_t)memory_space;
}

iree_status_t loom_test_fact_buffer_memory_space_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_buffer_reference_t reference = {0};
  if (!loom_value_facts_query_buffer_reference(context, operand_facts[0],
                                               &reference)) {
    result_facts[0] = loom_value_facts_exact_i64(-1);
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_exact_i64(
      loom_test_memory_space_or_unknown(reference.memory_space));
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_memory_space_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference = {0};
  if (!loom_value_facts_query_view_reference(context, operand_facts[0],
                                             &reference)) {
    result_facts[0] = loom_value_facts_exact_i64(-1);
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_exact_i64(
      loom_test_memory_space_or_unknown(reference.memory_space));
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_root_matches_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_value_facts_query_view_reference(context, operand_facts[0],
                                             &view_reference)) {
    result_facts[0] = loom_value_facts_exact_i64(0);
    return iree_ok_status();
  }

  loom_value_id_t root_value_id = loom_op_const_operands(op)[1];
  loom_value_fact_buffer_reference_t buffer_reference = {0};
  loom_value_fact_view_reference_t other_view_reference = {0};
  if (loom_value_facts_query_buffer_reference(context, operand_facts[1],
                                              &buffer_reference)) {
    root_value_id = buffer_reference.root_value_id;
  } else if (loom_value_facts_query_view_reference(context, operand_facts[1],
                                                   &other_view_reference)) {
    root_value_id = other_view_reference.root_value_id;
  }

  result_facts[0] = loom_value_facts_exact_i64(
      view_reference.root_value_id == root_value_id ? 1 : 0);
  return iree_ok_status();
}

static loom_value_fact_view_reference_t loom_test_view_reference_or_empty(
    const loom_fact_context_t* context, loom_value_facts_t facts) {
  loom_value_fact_view_reference_t reference = {
      .base_byte_offset = loom_value_facts_exact_i64(INT64_MIN),
      .footprint_byte_length = loom_value_facts_exact_i64(INT64_MIN),
      .minimum_alignment = 0,
      .root_minimum_alignment = 0,
      .static_element_byte_count = -1,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = LOOM_VALUE_ID_INVALID,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
  (void)loom_value_facts_query_view_reference(context, facts, &reference);
  return reference;
}

iree_status_t loom_test_fact_view_byte_offset_lo_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64(reference.base_byte_offset.range_lo);
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_byte_offset_hi_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64(reference.base_byte_offset.range_hi);
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_byte_length_lo_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64(reference.footprint_byte_length.range_lo);
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_byte_length_hi_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64(reference.footprint_byte_length.range_hi);
  return iree_ok_status();
}

iree_status_t loom_test_fact_buffer_min_alignment_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_buffer_reference_t reference = {
      .maximum_byte_extent = loom_value_facts_exact_i64(INT64_MIN),
      .minimum_alignment = 0,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = LOOM_VALUE_ID_INVALID,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
  };
  (void)loom_value_facts_query_buffer_reference(context, operand_facts[0],
                                                &reference);
  result_facts[0] =
      loom_value_facts_exact_i64((int64_t)reference.minimum_alignment);
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_min_alignment_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64((int64_t)reference.minimum_alignment);
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_root_min_alignment_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64((int64_t)reference.root_minimum_alignment);
  return iree_ok_status();
}

iree_status_t loom_test_fact_view_element_bytes_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_fact_view_reference_t reference =
      loom_test_view_reference_or_empty(context, operand_facts[0]);
  result_facts[0] =
      loom_value_facts_exact_i64(reference.static_element_byte_count);
  return iree_ok_status();
}
