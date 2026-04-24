// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/packed_dot_contract_projection.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

loom_contract_operand_t Operand(loom_contract_operand_role_t role,
                                loom_contract_numeric_type_t numeric_type) {
  return (loom_contract_operand_t){
      .role = role,
      .numeric_type = numeric_type,
  };
}

loom_contract_arithmetic_t ArithmeticForAccumulator(
    loom_contract_numeric_type_t accumulator_numeric_type) {
  switch (accumulator_numeric_type) {
    case LOOM_CONTRACT_NUMERIC_F16:
    case LOOM_CONTRACT_NUMERIC_BF16:
    case LOOM_CONTRACT_NUMERIC_F32:
    case LOOM_CONTRACT_NUMERIC_F64:
      return LOOM_CONTRACT_ARITHMETIC_FLOAT_DOT;
    default:
      return LOOM_CONTRACT_ARITHMETIC_INTEGER_DOT;
  }
}

loom_contract_request_t PackedDotRequest(
    loom_contract_numeric_type_t lhs_numeric_type,
    loom_contract_numeric_type_t rhs_numeric_type,
    loom_contract_numeric_type_t accumulator_numeric_type,
    loom_contract_numeric_type_t result_numeric_type, uint16_t vector_bit_width,
    uint16_t source_lane_count, uint16_t result_lane_count,
    uint16_t k_group_size) {
  loom_contract_request_t request = {};
  loom_contract_request_initialize(&request);
  request.kind = LOOM_CONTRACT_KIND_MATRIX_MULTIPLY;
  request.arithmetic = ArithmeticForAccumulator(accumulator_numeric_type);
  request.shape = {
      .m = result_lane_count,
      .n = 1,
      .k = source_lane_count,
  };
  request.k_group_size = k_group_size;
  request.lhs = Operand(LOOM_CONTRACT_OPERAND_ROLE_LHS, lhs_numeric_type);
  request.rhs = Operand(LOOM_CONTRACT_OPERAND_ROLE_RHS, rhs_numeric_type);
  request.accumulator =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, accumulator_numeric_type);
  request.result =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_RESULT, result_numeric_type);
  request.fragment = {
      .atom_bits = LOOM_CONTRACT_FRAGMENT_VECTOR_LANE,
      .vector_bit_width = vector_bit_width,
      .source_lane_count = source_lane_count,
      .result_lane_count = result_lane_count,
  };
  request.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_CPU_PACKED_DOT;
  request.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;
  return request;
}

TEST(PackedDotContractProjectionTest, ProjectsBf16ContractToAvx512Bf16) {
  loom_contract_request_t contract = PackedDotRequest(
      LOOM_CONTRACT_NUMERIC_BF16, LOOM_CONTRACT_NUMERIC_BF16,
      LOOM_CONTRACT_NUMERIC_F32, LOOM_CONTRACT_NUMERIC_F32, 256, 16, 8, 2);

  loom_x86_packed_dot_feature_bits_t feature_bits = 0;
  IREE_ASSERT_OK(loom_x86_packed_dot_feature_bits_for_name(
      IREE_SV("x86-avx512-bf16-vl"), &feature_bits));
  loom_x86_packed_dot_match_request_t x86_request = {};
  ASSERT_TRUE(loom_x86_packed_dot_match_request_from_contract(
      &contract, feature_bits, &x86_request, nullptr));

  const loom_x86_packed_dot_descriptor_t* descriptor =
      loom_x86_packed_dot_select(&x86_request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "x86.avx512_bf16.vdpbf16ps.ymm");
}

TEST(PackedDotContractProjectionTest, ProjectsU8S8ContractToAvxVnni) {
  loom_contract_request_t contract = PackedDotRequest(
      LOOM_CONTRACT_NUMERIC_U8, LOOM_CONTRACT_NUMERIC_I8,
      LOOM_CONTRACT_NUMERIC_I32, LOOM_CONTRACT_NUMERIC_I32, 256, 32, 8, 4);

  loom_x86_packed_dot_feature_bits_t feature_bits = 0;
  IREE_ASSERT_OK(loom_x86_packed_dot_feature_bits_for_name(
      IREE_SV("x86-avx-vnni"), &feature_bits));
  loom_x86_packed_dot_match_request_t x86_request = {};
  ASSERT_TRUE(loom_x86_packed_dot_match_request_from_contract(
      &contract, feature_bits, &x86_request, nullptr));

  const loom_x86_packed_dot_descriptor_t* descriptor =
      loom_x86_packed_dot_select(&x86_request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "x86.avx_vnni.vpdpbusd.ymm");
}

TEST(PackedDotContractProjectionTest, RejectsMissingVectorFragmentFacts) {
  loom_contract_request_t contract = PackedDotRequest(
      LOOM_CONTRACT_NUMERIC_BF16, LOOM_CONTRACT_NUMERIC_BF16,
      LOOM_CONTRACT_NUMERIC_F32, LOOM_CONTRACT_NUMERIC_F32, 256, 16, 8, 2);
  contract.fragment.atom_bits = 0;

  loom_x86_packed_dot_match_request_t x86_request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_x86_packed_dot_match_request_from_contract(
      &contract, 0, &x86_request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_FRAGMENT);
}

TEST(PackedDotContractProjectionTest, RejectsUnsupportedNumericType) {
  loom_contract_request_t contract = PackedDotRequest(
      LOOM_CONTRACT_NUMERIC_FP8, LOOM_CONTRACT_NUMERIC_BF8,
      LOOM_CONTRACT_NUMERIC_F32, LOOM_CONTRACT_NUMERIC_F32, 256, 16, 8, 2);

  loom_x86_packed_dot_match_request_t x86_request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_x86_packed_dot_match_request_from_contract(
      &contract, 0, &x86_request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_NUMERIC);
}

}  // namespace
