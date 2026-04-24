// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract_storage.h"

#include "iree/testing/gtest.h"

namespace {

loom_value_fact_storage_schema_t MatrixSchema(
    loom_value_fact_matrix_format_t format,
    loom_value_fact_matrix_scale_kind_t scale_kind,
    loom_value_fact_matrix_scale_format_t scale_format,
    uint16_t packed_register_count, uint16_t packed_element_count) {
  loom_value_fact_storage_schema_t schema = {};
  schema.matrix.format = format;
  schema.matrix.scale_kind = scale_kind;
  schema.matrix.scale_format = scale_format;
  schema.matrix.packed_register_count = packed_register_count;
  schema.matrix.packed_element_count = packed_element_count;
  schema.matrix.zero_scale_fallback =
      scale_kind != LOOM_VALUE_FACT_MATRIX_SCALE_NONE;
  return schema;
}

TEST(ContractStorageTest, MapsMatrixStorageSchemaToGenericOperand) {
  loom_value_fact_storage_schema_t schema = MatrixSchema(
      LOOM_VALUE_FACT_MATRIX_FORMAT_FP6, LOOM_VALUE_FACT_MATRIX_SCALE_32,
      LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE, 6, 32);

  loom_contract_operand_t operand = {};
  ASSERT_TRUE(loom_contract_operand_from_storage_schema(
      LOOM_CONTRACT_OPERAND_ROLE_LHS, schema, &operand));
  EXPECT_EQ(operand.role, LOOM_CONTRACT_OPERAND_ROLE_LHS);
  EXPECT_EQ(operand.numeric_type, LOOM_CONTRACT_NUMERIC_FP6);
  EXPECT_EQ(operand.payload_register_count, 6);
  EXPECT_EQ(operand.payload_element_count, 32);

  loom_contract_scale_kind_t scale_kind = LOOM_CONTRACT_SCALE_UNKNOWN;
  ASSERT_TRUE(
      loom_contract_scale_kind_from_storage_schema(schema, &scale_kind));
  EXPECT_EQ(scale_kind, LOOM_CONTRACT_SCALE_32);

  const loom_contract_capability_flags_t flags =
      loom_contract_capability_flags_from_storage_schema(schema);
  EXPECT_TRUE(
      iree_any_bit_set(flags, LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS));
  EXPECT_TRUE(iree_any_bit_set(flags, LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS));
  EXPECT_TRUE(
      iree_any_bit_set(flags, LOOM_CONTRACT_CAPABILITY_ZERO_SCALE_FALLBACK));
  EXPECT_FALSE(
      iree_any_bit_set(flags, LOOM_CONTRACT_CAPABILITY_SCALE_FORMAT_SELECTORS));
}

TEST(ContractStorageTest, RejectsUnknownMatrixFormat) {
  loom_value_fact_storage_schema_t schema = MatrixSchema(
      LOOM_VALUE_FACT_MATRIX_FORMAT_UNKNOWN, LOOM_VALUE_FACT_MATRIX_SCALE_NONE,
      LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE, 0, 0);

  loom_contract_operand_t operand = {};
  EXPECT_FALSE(loom_contract_operand_from_storage_schema(
      LOOM_CONTRACT_OPERAND_ROLE_LHS, schema, &operand));
  EXPECT_EQ(operand.numeric_type, LOOM_CONTRACT_NUMERIC_UNKNOWN);
}

}  // namespace
