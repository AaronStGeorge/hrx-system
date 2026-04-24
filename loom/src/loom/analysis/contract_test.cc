// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/contract.h"

#include "iree/testing/gtest.h"

namespace {

loom_contract_operand_t Operand(loom_contract_operand_role_t role,
                                loom_contract_numeric_type_t numeric_type) {
  return (loom_contract_operand_t){
      .role = role,
      .numeric_type = numeric_type,
  };
}

loom_contract_request_t CompletePackedDotRequest() {
  loom_contract_request_t request = {};
  loom_contract_request_initialize(&request);
  request.kind = LOOM_CONTRACT_KIND_MATRIX_MULTIPLY;
  request.arithmetic = LOOM_CONTRACT_ARITHMETIC_FLOAT_DOT;
  request.shape = {.m = 8, .n = 1, .k = 16};
  request.k_group_size = 2;
  request.lhs =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_LHS, LOOM_CONTRACT_NUMERIC_BF16);
  request.rhs =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_BF16);
  request.accumulator = Operand(LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,
                                LOOM_CONTRACT_NUMERIC_F32);
  request.result =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_RESULT, LOOM_CONTRACT_NUMERIC_F32);
  request.fragment.atom_bits = LOOM_CONTRACT_FRAGMENT_VECTOR_LANE;
  request.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT;
  request.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;
  return request;
}

TEST(ContractTest, ValidatesCompletePackedDotRequest) {
  loom_contract_request_t request = CompletePackedDotRequest();

  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_TRUE(loom_contract_request_validate(&request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NONE);
}

TEST(ContractTest, RejectsUnavailableRequiredCapability) {
  loom_contract_request_t request = CompletePackedDotRequest();
  request.required_capability_flags = LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;

  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_validate(&request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_CAPABILITY);
}

TEST(ContractTest, RejectsMissingShapeRoleAndCapability) {
  loom_contract_request_t request = {};
  loom_contract_request_initialize(&request);
  request.kind = LOOM_CONTRACT_KIND_MATRIX_MULTIPLY;
  request.arithmetic = LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT;
  request.shape = {.m = 16, .n = 16, .k = 0};
  request.k_group_size = 4;
  request.lhs =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_LHS, LOOM_CONTRACT_NUMERIC_U8);
  request.rhs =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_RHS, LOOM_CONTRACT_NUMERIC_I8);
  request.accumulator =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN, LOOM_CONTRACT_NUMERIC_I32);
  request.result =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_RESULT, LOOM_CONTRACT_NUMERIC_I32);
  request.required_capability_flags = LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS;

  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_contract_request_validate(&request, &diagnostic));
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_bits,
                               LOOM_CONTRACT_REJECTION_SHAPE));
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_bits,
                               LOOM_CONTRACT_REJECTION_ROLE));
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_bits,
                               LOOM_CONTRACT_REJECTION_CAPABILITY));
}

TEST(ContractTest, MapsScalarNumericTypes) {
  loom_contract_numeric_type_t numeric_type = LOOM_CONTRACT_NUMERIC_UNKNOWN;
  ASSERT_TRUE(loom_contract_numeric_type_from_scalar(LOOM_SCALAR_TYPE_I8, false,
                                                     &numeric_type));
  EXPECT_EQ(numeric_type, LOOM_CONTRACT_NUMERIC_I8);
  ASSERT_TRUE(loom_contract_numeric_type_from_scalar(LOOM_SCALAR_TYPE_I8, true,
                                                     &numeric_type));
  EXPECT_EQ(numeric_type, LOOM_CONTRACT_NUMERIC_U8);
  ASSERT_TRUE(loom_contract_numeric_type_from_scalar(LOOM_SCALAR_TYPE_BF16,
                                                     false, &numeric_type));
  EXPECT_EQ(numeric_type, LOOM_CONTRACT_NUMERIC_BF16);
}

}  // namespace
