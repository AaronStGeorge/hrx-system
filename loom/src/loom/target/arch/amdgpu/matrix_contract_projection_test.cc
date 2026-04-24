// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/matrix_contract_projection.h"

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

loom_contract_request_t MatrixRequest(
    int64_t m, int64_t n, int64_t k,
    loom_contract_numeric_type_t lhs_numeric_type,
    loom_contract_numeric_type_t rhs_numeric_type,
    loom_contract_numeric_type_t accumulator_numeric_type,
    loom_contract_numeric_type_t result_numeric_type) {
  loom_contract_request_t request = {};
  loom_contract_request_initialize(&request);
  request.kind = LOOM_CONTRACT_KIND_MATRIX_MULTIPLY;
  request.arithmetic = LOOM_CONTRACT_ARITHMETIC_MIXED_DOT;
  request.shape = {
      .m = m,
      .n = n,
      .k = k,
  };
  request.k_group_size = 1;
  request.lhs = Operand(LOOM_CONTRACT_OPERAND_ROLE_LHS, lhs_numeric_type);
  request.rhs = Operand(LOOM_CONTRACT_OPERAND_ROLE_RHS, rhs_numeric_type);
  request.accumulator =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR, accumulator_numeric_type);
  request.result =
      Operand(LOOM_CONTRACT_OPERAND_ROLE_RESULT, result_numeric_type);
  request.fragment = {
      .atom_bits = LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE,
      .subgroup_size = 64,
  };
  request.capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX;
  request.policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED;
  return request;
}

TEST(MatrixContractProjectionTest, ProjectsI8ContractToGfx908Mfma) {
  loom_contract_request_t contract = MatrixRequest(
      16, 16, 16, LOOM_CONTRACT_NUMERIC_I8, LOOM_CONTRACT_NUMERIC_I8,
      LOOM_CONTRACT_NUMERIC_I32, LOOM_CONTRACT_NUMERIC_I32);

  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_for_processor(
      IREE_SV("gfx908"), &feature_bits));
  loom_amdgpu_matrix_contract_match_request_t amdgpu_request = {};
  ASSERT_TRUE(loom_amdgpu_matrix_contract_match_request_from_contract(
      &contract, feature_bits, 64, &amdgpu_request, nullptr));

  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&amdgpu_request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "mfma.i32.16x16x16.i8");
}

TEST(MatrixContractProjectionTest, ProjectsScaledFp6ContractToGfx950Mfma) {
  loom_contract_request_t contract = MatrixRequest(
      16, 16, 128, LOOM_CONTRACT_NUMERIC_FP6, LOOM_CONTRACT_NUMERIC_BF6,
      LOOM_CONTRACT_NUMERIC_F32, LOOM_CONTRACT_NUMERIC_F32);
  contract.lhs.payload_register_count = 6;
  contract.lhs.payload_element_count = 32;
  contract.rhs.payload_register_count = 6;
  contract.rhs.payload_element_count = 32;
  contract.scale_kind = LOOM_CONTRACT_SCALE_32;
  contract.available_capability_flags =
      LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS |
      LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS |
      LOOM_CONTRACT_CAPABILITY_ZERO_SCALE_FALLBACK;
  contract.required_capability_flags =
      LOOM_CONTRACT_CAPABILITY_SCALE_OPERANDS |
      LOOM_CONTRACT_CAPABILITY_FORMAT_SELECTORS;

  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  IREE_ASSERT_OK(loom_amdgpu_matrix_feature_bits_for_processor(
      IREE_SV("gfx950"), &feature_bits));
  loom_amdgpu_matrix_contract_match_request_t amdgpu_request = {};
  ASSERT_TRUE(loom_amdgpu_matrix_contract_match_request_from_contract(
      &contract, feature_bits, 64, &amdgpu_request, nullptr));

  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&amdgpu_request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "mfma.scale.f32.16x16x128.f8f6f4");
}

TEST(MatrixContractProjectionTest, RejectsMissingSubgroupFragmentFacts) {
  loom_contract_request_t contract = MatrixRequest(
      16, 16, 16, LOOM_CONTRACT_NUMERIC_I8, LOOM_CONTRACT_NUMERIC_I8,
      LOOM_CONTRACT_NUMERIC_I32, LOOM_CONTRACT_NUMERIC_I32);
  contract.fragment.atom_bits = 0;

  loom_amdgpu_matrix_contract_match_request_t amdgpu_request = {};
  loom_contract_diagnostic_t diagnostic = {};
  EXPECT_FALSE(loom_amdgpu_matrix_contract_match_request_from_contract(
      &contract, 0, 64, &amdgpu_request, &diagnostic));
  EXPECT_EQ(diagnostic.rejection_bits, LOOM_CONTRACT_REJECTION_FRAGMENT);
}

}  // namespace
