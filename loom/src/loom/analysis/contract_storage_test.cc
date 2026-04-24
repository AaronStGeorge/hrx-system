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

loom_contract_view_payload_t PlainPayload(
    loom_contract_operand_role_t role,
    loom_contract_numeric_type_t numeric_type) {
  return (loom_contract_view_payload_t){
      .kind = LOOM_CONTRACT_VIEW_PAYLOAD_PLAIN_ELEMENT,
      .operand =
          (loom_contract_operand_t){
              .role = role,
              .numeric_type = numeric_type,
          },
      .scale_kind = LOOM_CONTRACT_SCALE_NONE,
  };
}

loom_contract_view_payload_t MatrixPayload(
    loom_contract_operand_role_t role,
    loom_value_fact_storage_schema_t schema) {
  loom_contract_view_payload_t payload = {
      .kind = LOOM_CONTRACT_VIEW_PAYLOAD_UNSUPPORTED_STORAGE_SCHEMA,
      .scale_kind = LOOM_CONTRACT_SCALE_NONE,
      .storage_schema = schema,
  };
  if (!loom_contract_operand_from_storage_schema(role, schema,
                                                 &payload.operand)) {
    return payload;
  }
  if (!loom_contract_scale_kind_from_storage_schema(schema,
                                                    &payload.scale_kind)) {
    payload.operand.numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
    return payload;
  }
  payload.kind = LOOM_CONTRACT_VIEW_PAYLOAD_MATRIX_STORAGE_SCHEMA;
  payload.available_capability_flags =
      loom_contract_capability_flags_from_storage_schema(schema);
  return payload;
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

TEST(ContractStorageTest, QueriesPlainViewPayloadFromElementType) {
  loom_type_t view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8, loom_dim_pack_static(32), 0);

  loom_contract_view_payload_t payload = {};
  ASSERT_TRUE(loom_contract_view_payload_from_type(
      nullptr, nullptr, view_type, LOOM_CONTRACT_OPERAND_ROLE_LHS,
      /*plain_integer_is_unsigned=*/true, &payload));
  EXPECT_EQ(payload.kind, LOOM_CONTRACT_VIEW_PAYLOAD_PLAIN_ELEMENT);
  EXPECT_EQ(payload.operand.role, LOOM_CONTRACT_OPERAND_ROLE_LHS);
  EXPECT_EQ(payload.operand.numeric_type, LOOM_CONTRACT_NUMERIC_U8);
  EXPECT_EQ(payload.scale_kind, LOOM_CONTRACT_SCALE_NONE);
  EXPECT_EQ(payload.available_capability_flags, 0u);
}

TEST(ContractStorageTest, BuildsMatrixRequestFromPayloadFacts) {
  loom_value_fact_storage_schema_t lhs_schema = MatrixSchema(
      LOOM_VALUE_FACT_MATRIX_FORMAT_FP6, LOOM_VALUE_FACT_MATRIX_SCALE_32,
      LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE, 6, 32);
  loom_value_fact_storage_schema_t rhs_schema = MatrixSchema(
      LOOM_VALUE_FACT_MATRIX_FORMAT_BF6, LOOM_VALUE_FACT_MATRIX_SCALE_32,
      LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE, 6, 32);

  loom_contract_matrix_request_options_t options = {};
  options.shape = (loom_contract_shape_t){.m = 16, .n = 16, .k = 128};
  options.k_group_size = 1;
  options.lhs = MatrixPayload(LOOM_CONTRACT_OPERAND_ROLE_LHS, lhs_schema);
  options.rhs = MatrixPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, rhs_schema);
  options.accumulator_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
  options.result_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
  options.arithmetic = LOOM_CONTRACT_ARITHMETIC_MIXED_DOT;
  options.fragment = (loom_contract_fragment_t){
      .atom_bits = LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE,
      .subgroup_size = 64,
  };
  options.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX;
  options.required_capability_flags = LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS |
                                      LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS;
  options.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;

  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  ASSERT_TRUE(loom_contract_request_from_matrix_payloads(&options, &request,
                                                         &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NONE);
  EXPECT_EQ(request.kind, LOOM_CONTRACT_KIND_MATRIX_MULTIPLY);
  EXPECT_EQ(request.shape.m, 16);
  EXPECT_EQ(request.shape.n, 16);
  EXPECT_EQ(request.shape.k, 128);
  EXPECT_EQ(request.lhs.numeric_type, LOOM_CONTRACT_NUMERIC_FP6);
  EXPECT_EQ(request.lhs.payload_register_count, 6);
  EXPECT_EQ(request.rhs.numeric_type, LOOM_CONTRACT_NUMERIC_BF6);
  EXPECT_EQ(request.accumulator.numeric_type, LOOM_CONTRACT_NUMERIC_F32);
  EXPECT_EQ(request.result.numeric_type, LOOM_CONTRACT_NUMERIC_F32);
  EXPECT_EQ(request.scale_kind, LOOM_CONTRACT_SCALE_32);
  EXPECT_TRUE(iree_all_bits_set(request.available_capability_flags,
                                LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS |
                                    LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS));
}

TEST(ContractStorageTest, BuildsPackedDotRequestFromPlainPayloadFacts) {
  loom_contract_matrix_request_options_t options = {};
  options.shape = (loom_contract_shape_t){.m = 8, .n = 1, .k = 32};
  options.k_group_size = 4;
  options.lhs =
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_LHS, LOOM_CONTRACT_NUMERIC_U8);
  options.rhs =
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_I8);
  options.accumulator_numeric_type = LOOM_CONTRACT_NUMERIC_I32;
  options.result_numeric_type = LOOM_CONTRACT_NUMERIC_I32;
  options.arithmetic = LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT;
  options.fragment = (loom_contract_fragment_t){
      .atom_bits = LOOM_CONTRACT_FRAGMENT_VECTOR_LANE,
      .vector_bit_width = 256,
      .source_lane_count = 32,
      .result_lane_count = 8,
  };
  options.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT;
  options.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;

  loom_contract_request_t request = {};
  ASSERT_TRUE(
      loom_contract_request_from_matrix_payloads(&options, &request, nullptr));
  EXPECT_EQ(request.kind, LOOM_CONTRACT_KIND_MATRIX_MULTIPLY);
  EXPECT_EQ(request.arithmetic, LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT);
  EXPECT_EQ(request.k_group_size, 4);
  EXPECT_EQ(request.lhs.numeric_type, LOOM_CONTRACT_NUMERIC_U8);
  EXPECT_EQ(request.rhs.numeric_type, LOOM_CONTRACT_NUMERIC_I8);
  EXPECT_EQ(request.fragment.atom_bits, LOOM_CONTRACT_FRAGMENT_VECTOR_LANE);
  EXPECT_EQ(request.fragment.vector_bit_width, 256);
}

TEST(ContractStorageTest, RejectsUnsupportedPayloadForOptimizedContract) {
  loom_contract_matrix_request_options_t options = {};
  options.shape = (loom_contract_shape_t){.m = 16, .n = 16, .k = 128};
  options.k_group_size = 1;
  options.lhs.kind = LOOM_CONTRACT_VIEW_PAYLOAD_UNSUPPORTED_STORAGE_SCHEMA;
  options.rhs =
      PlainPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_I8);
  options.accumulator_numeric_type = LOOM_CONTRACT_NUMERIC_I32;
  options.result_numeric_type = LOOM_CONTRACT_NUMERIC_I32;
  options.arithmetic = LOOM_CONTRACT_ARITHMETIC_MIXED_DOT;
  options.fragment.atom_bits = LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE;
  options.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX;
  options.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;

  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_from_matrix_payloads(&options, &request,
                                                          &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NUMERIC);
}

TEST(ContractStorageTest, RejectsMismatchedScalePayloads) {
  loom_value_fact_storage_schema_t lhs_schema = MatrixSchema(
      LOOM_VALUE_FACT_MATRIX_FORMAT_FP6, LOOM_VALUE_FACT_MATRIX_SCALE_32,
      LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE, 6, 32);
  loom_value_fact_storage_schema_t rhs_schema = MatrixSchema(
      LOOM_VALUE_FACT_MATRIX_FORMAT_BF6, LOOM_VALUE_FACT_MATRIX_SCALE_NONE,
      LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE, 6, 32);

  loom_contract_matrix_request_options_t options = {};
  options.shape = (loom_contract_shape_t){.m = 16, .n = 16, .k = 128};
  options.k_group_size = 1;
  options.lhs = MatrixPayload(LOOM_CONTRACT_OPERAND_ROLE_LHS, lhs_schema);
  options.rhs = MatrixPayload(LOOM_CONTRACT_OPERAND_ROLE_RHS, rhs_schema);
  options.accumulator_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
  options.result_numeric_type = LOOM_CONTRACT_NUMERIC_F32;
  options.arithmetic = LOOM_CONTRACT_ARITHMETIC_MIXED_DOT;
  options.fragment.atom_bits = LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE;
  options.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX;
  options.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;

  loom_contract_request_t request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_from_matrix_payloads(&options, &request,
                                                          &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_CAPABILITY);
}

}  // namespace
