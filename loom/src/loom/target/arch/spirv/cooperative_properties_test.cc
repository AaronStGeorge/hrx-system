// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/cooperative_properties.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

void PreparePropertySet(loom_spirv_feature_bits_t feature_bits,
                        loom_spirv_cooperative_property_set_t* property_set) {
  loom_spirv_feature_set_t feature_set = {};
  IREE_ASSERT_OK(loom_spirv_feature_set_prepare(IREE_SV("test.spirv"),
                                                feature_bits, &feature_set));
  loom_spirv_cooperative_property_set_prepare(&feature_set, property_set);
}

loom_spirv_cooperative_matrix_query_t F16MatrixQuery(
    loom_lowering_policy_t policy) {
  return {
      .m_size = 16,
      .n_size = 16,
      .k_size = 16,
      .lhs_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT16,
      .rhs_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT16,
      .accumulator_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT32,
      .result_type = LOOM_SPIRV_COMPONENT_TYPE_FLOAT32,
      .scope = LOOM_SPIRV_SCOPE_SUBGROUP,
      .layout = LOOM_SPIRV_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR,
      .storage_class = LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      .policy = policy,
  };
}

loom_spirv_cooperative_vector_query_t PackedS8VectorQuery(
    loom_lowering_policy_t policy) {
  return {
      .m_size = 32,
      .k_size = 32,
      .input_type = LOOM_SPIRV_COMPONENT_TYPE_UINT32,
      .input_interpretation = LOOM_SPIRV_COMPONENT_TYPE_SINT8_PACKED,
      .matrix_interpretation = LOOM_SPIRV_COMPONENT_TYPE_SINT8,
      .bias_interpretation = LOOM_SPIRV_COMPONENT_TYPE_SINT32,
      .result_type = LOOM_SPIRV_COMPONENT_TYPE_SINT32,
      .matrix_layout =
          LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
      .storage_class = LOOM_SPIRV_STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER,
      .policy = policy,
  };
}

TEST(SpirvCooperativePropertiesTest, PreparesSelectedPropertyTables) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
                     &property_set);

  EXPECT_TRUE(iree_all_bits_set(property_set.feature_bits,
                                LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR));
  EXPECT_TRUE(iree_all_bits_set(property_set.feature_bits,
                                LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV));
  EXPECT_GT(property_set.matrix_property_count, 0u);
  EXPECT_GT(property_set.matrix_shape_span_count, 0u);
  EXPECT_GT(property_set.vector_property_count, 0u);
  EXPECT_GT(property_set.vector_shape_span_count, 0u);
}

TEST(SpirvCooperativePropertiesTest, SelectsCooperativeMatrixByShapeAndFacts) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
                     &property_set);
  const loom_spirv_cooperative_matrix_query_t query =
      F16MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_matrix_property_t* property =
      loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                    &diagnostic);

  ASSERT_NE(property, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      property->name,
      IREE_SV("khr.cooperative_matrix.f16.16x16x16.f32.subgroup")));
  EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);
}

TEST(SpirvCooperativePropertiesTest, RejectsMissingMatrixFeatureSeparately) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER, &property_set);
  const loom_spirv_cooperative_matrix_query_t query =
      F16MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                          &diagnostic),
            nullptr);

  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_FEATURE_MISSING);
  EXPECT_EQ(diagnostic.feature_atom,
            LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_MATRIX_KHR);
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_flags,
                               LOOM_SPIRV_COOPERATIVE_REJECTION_FEATURE));
}

TEST(SpirvCooperativePropertiesTest, RecordsMatrixPropertyMissAndFallback) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
                     &property_set);
  loom_spirv_cooperative_matrix_query_t query =
      F16MatrixQuery(LOOM_LOWERING_POLICY_REFERENCE_ALLOWED);
  query.k_size = 8;

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                          &diagnostic),
            nullptr);

  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_FALLBACK_PERMITTED);
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_flags,
                               LOOM_SPIRV_COOPERATIVE_REJECTION_SHAPE));
  EXPECT_TRUE(
      iree_any_bit_set(diagnostic.rejection_flags,
                       LOOM_SPIRV_COOPERATIVE_REJECTION_POLICY_FALLBACK));
}

TEST(SpirvCooperativePropertiesTest,
     RequiresExactCooperativeMatrixOperandFlags) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_MATRIX_KHR,
                     &property_set);
  loom_spirv_cooperative_matrix_query_t query =
      F16MatrixQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
  query.operand_flags =
      LOOM_SPIRV_COOPERATIVE_MATRIX_OPERAND_SATURATING_ACCUMULATION;

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_spirv_cooperative_matrix_property_select(&property_set, &query,
                                                          &diagnostic),
            nullptr);
  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_PROPERTY_MISSING);
  EXPECT_TRUE(iree_any_bit_set(diagnostic.rejection_flags,
                               LOOM_SPIRV_COOPERATIVE_REJECTION_OPERANDS));
}

TEST(SpirvCooperativePropertiesTest, SelectsCooperativeVectorInterpretation) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
                     &property_set);
  const loom_spirv_cooperative_vector_query_t query =
      PackedS8VectorQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_vector_property_t* property =
      loom_spirv_cooperative_vector_property_select(&property_set, &query,
                                                    &diagnostic);

  ASSERT_NE(property, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      property->name, IREE_SV("nv.cooperative_vector.u32.32x32.s8_packed")));
  EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);
}

TEST(SpirvCooperativePropertiesTest, RequiresTrainingFeatureForTrainingQuery) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV,
                     &property_set);
  loom_spirv_cooperative_vector_query_t query =
      PackedS8VectorQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
  query.matrix_layout =
      LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL;
  query.flags = LOOM_SPIRV_COOPERATIVE_VECTOR_FLAG_TRAINING;

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  EXPECT_EQ(loom_spirv_cooperative_vector_property_select(&property_set, &query,
                                                          &diagnostic),
            nullptr);

  EXPECT_EQ(diagnostic.status,
            LOOM_SPIRV_COOPERATIVE_SELECTION_REQUIRED_FEATURE_MISSING);
  EXPECT_EQ(diagnostic.feature_atom,
            LOOM_SPIRV_FEATURE_ATOM_COOPERATIVE_VECTOR_TRAINING_NV);
}

TEST(SpirvCooperativePropertiesTest, SelectsCooperativeVectorTrainingRow) {
  loom_spirv_cooperative_property_set_t property_set = {};
  PreparePropertySet(LOOM_SPIRV_FEATURE_VULKAN_SHADER |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_NV |
                         LOOM_SPIRV_FEATURE_COOPERATIVE_VECTOR_TRAINING_NV,
                     &property_set);
  loom_spirv_cooperative_vector_query_t query =
      PackedS8VectorQuery(LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED);
  query.matrix_layout =
      LOOM_SPIRV_COOPERATIVE_VECTOR_MATRIX_LAYOUT_TRAINING_OPTIMAL;
  query.flags = LOOM_SPIRV_COOPERATIVE_VECTOR_FLAG_TRAINING;

  loom_spirv_cooperative_diagnostic_t diagnostic = {};
  const loom_spirv_cooperative_vector_property_t* property =
      loom_spirv_cooperative_vector_property_select(&property_set, &query,
                                                    &diagnostic);

  ASSERT_NE(property, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      property->name,
      IREE_SV("nv.cooperative_vector.training.u32.32x32.s8_packed")));
  EXPECT_EQ(diagnostic.status, LOOM_SPIRV_COOPERATIVE_SELECTION_MATCHED);
}

}  // namespace
