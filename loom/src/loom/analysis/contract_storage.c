// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract_storage.h"

bool loom_contract_numeric_type_from_matrix_format(
    loom_value_fact_matrix_format_t format,
    loom_contract_numeric_type_t* out_numeric_type) {
  IREE_ASSERT_ARGUMENT(out_numeric_type);
  *out_numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  switch (format) {
    case LOOM_VALUE_FACT_MATRIX_FORMAT_FP8:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_FP8;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_BF8:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_BF8;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_FP6:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_FP6;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_BF6:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_BF6;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_FP4:
      *out_numeric_type = LOOM_CONTRACT_NUMERIC_FP4;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_UNKNOWN:
    default:
      return false;
  }
}

bool loom_contract_scale_kind_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_contract_scale_kind_t* out_scale_kind) {
  IREE_ASSERT_ARGUMENT(out_scale_kind);
  *out_scale_kind = LOOM_CONTRACT_SCALE_UNKNOWN;
  switch (schema.matrix.scale_kind) {
    case LOOM_VALUE_FACT_MATRIX_SCALE_NONE:
      *out_scale_kind = LOOM_CONTRACT_SCALE_NONE;
      return true;
    case LOOM_VALUE_FACT_MATRIX_SCALE_32:
      *out_scale_kind = LOOM_CONTRACT_SCALE_32;
      return true;
    case LOOM_VALUE_FACT_MATRIX_SCALE_16:
      *out_scale_kind = LOOM_CONTRACT_SCALE_16;
      return true;
    case LOOM_VALUE_FACT_MATRIX_SCALE_UNKNOWN:
    default:
      return false;
  }
}

loom_contract_capability_flags_t
loom_contract_capability_flags_from_storage_schema(
    loom_value_fact_storage_schema_t schema) {
  loom_contract_capability_flags_t flags = 0;
  loom_contract_numeric_type_t numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  if (loom_contract_numeric_type_from_matrix_format(schema.matrix.format,
                                                    &numeric_type)) {
    flags |= LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS;
  }
  if (schema.matrix.scale_kind != LOOM_VALUE_FACT_MATRIX_SCALE_UNKNOWN &&
      schema.matrix.scale_kind != LOOM_VALUE_FACT_MATRIX_SCALE_NONE) {
    flags |= LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;
  }
  if (schema.matrix.scale_format !=
          LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_UNKNOWN &&
      schema.matrix.scale_format != LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE) {
    flags |= LOOM_CONTRACT_CAPABILITY_SCALE_FORMAT_SELECTORS;
  }
  if (schema.matrix.zero_scale_fallback) {
    flags |= LOOM_CONTRACT_CAPABILITY_ZERO_SCALE_FALLBACK;
  }
  return flags;
}

bool loom_contract_operand_from_storage_schema(
    loom_contract_operand_role_t role, loom_value_fact_storage_schema_t schema,
    loom_contract_operand_t* out_operand) {
  IREE_ASSERT_ARGUMENT(out_operand);
  *out_operand = (loom_contract_operand_t){0};
  loom_contract_numeric_type_t numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  if (!loom_contract_numeric_type_from_matrix_format(schema.matrix.format,
                                                     &numeric_type)) {
    return false;
  }
  *out_operand = (loom_contract_operand_t){
      .role = role,
      .numeric_type = numeric_type,
      .payload_register_count = schema.matrix.packed_register_count,
      .payload_element_count = schema.matrix.packed_element_count,
  };
  return true;
}
