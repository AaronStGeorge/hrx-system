// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/matrix_facts.h"

static bool loom_amdgpu_matrix_numeric_type_from_fact(
    loom_value_fact_numeric_format_flags_t format,
    loom_amdgpu_matrix_numeric_type_t* out_numeric_type) {
  *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN;
  switch (format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_FP8;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_BF8:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_BF8;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_FP6;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_BF6:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_BF6;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_FP4;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN:
    default:
      return false;
  }
}

bool loom_amdgpu_matrix_format_selector_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_amdgpu_matrix_format_selector_t* out_selector) {
  IREE_ASSERT_ARGUMENT(out_selector);
  switch (schema.encoded_operand.element_format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3:
      *out_selector = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP8;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_BF8:
      *out_selector = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_BF8;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2:
      *out_selector = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP6;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_BF6:
      *out_selector = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_BF6;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1:
      *out_selector = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP4;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_UNKNOWN:
    default:
      return false;
  }
}

bool loom_amdgpu_matrix_payload_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_amdgpu_matrix_payload_shape_t* out_payload) {
  IREE_ASSERT_ARGUMENT(out_payload);
  *out_payload = (loom_amdgpu_matrix_payload_shape_t){0};
  loom_value_fact_encoded_operand_schema_t operand = schema.encoded_operand;
  if (!iree_any_bit_set(operand.payload_packing,
                        LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT) ||
      operand.payload_register_count == 0 ||
      operand.payload_element_count == 0) {
    return false;
  }
  loom_amdgpu_matrix_numeric_type_t numeric_type =
      LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN;
  if (!loom_amdgpu_matrix_numeric_type_from_fact(operand.element_format,
                                                 &numeric_type)) {
    return false;
  }
  out_payload->numeric_type = numeric_type;
  out_payload->register_count = operand.payload_register_count;
  out_payload->element_count = operand.payload_element_count;
  return true;
}

bool loom_amdgpu_matrix_scale_kind_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_amdgpu_matrix_scale_kind_t* out_scale_kind) {
  IREE_ASSERT_ARGUMENT(out_scale_kind);
  loom_value_fact_encoded_operand_schema_t operand = schema.encoded_operand;
  if (!loom_value_fact_encoded_operand_schema_has_scale(operand)) {
    *out_scale_kind = LOOM_AMDGPU_MATRIX_SCALE_NONE;
    return true;
  }
  if (operand.scale_topology == 0 || operand.scale_operand_count == 0) {
    return false;
  }
  switch (operand.scale_group_element_count) {
    case 32:
      *out_scale_kind = LOOM_AMDGPU_MATRIX_SCALE_32;
      return true;
    case 16:
      *out_scale_kind = LOOM_AMDGPU_MATRIX_SCALE_16;
      return true;
    default:
      return false;
  }
}

loom_amdgpu_matrix_contract_flags_t
loom_amdgpu_matrix_available_flags_from_storage_schema(
    loom_value_fact_storage_schema_t schema) {
  loom_amdgpu_matrix_contract_flags_t flags = 0;
  loom_amdgpu_matrix_format_selector_t selector =
      LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP8;
  if (loom_amdgpu_matrix_format_selector_from_storage_schema(schema,
                                                             &selector)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS;
  }
  loom_value_fact_encoded_operand_schema_t operand = schema.encoded_operand;
  if (loom_value_fact_encoded_operand_schema_has_scale(operand) &&
      operand.scale_operand_count != 0) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED;
  }
  if (operand.scale_format != 0 || operand.secondary_scale_format != 0) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS;
  }
  if (iree_any_bit_set(
          operand.flags,
          LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK;
  }
  if (iree_any_bit_set(operand.sparsity_policy,
                       LOOM_VALUE_FACT_SPARSITY_POLICY_ALL)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE;
  }
  return flags;
}
