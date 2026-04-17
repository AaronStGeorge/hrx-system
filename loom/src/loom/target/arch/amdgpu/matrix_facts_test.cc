// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/matrix_facts.h"

#include <string>

#include "iree/testing/gtest.h"

namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

loom_value_fact_storage_schema_t MatrixSchema(
    loom_value_fact_matrix_format_t format,
    loom_value_fact_matrix_scale_kind_t scale_kind,
    loom_value_fact_matrix_scale_format_t scale_format,
    uint16_t packed_register_count, uint16_t packed_element_count) {
  loom_value_fact_storage_schema_t schema = {};
  schema.matrix.format = format;
  schema.matrix.scale_kind = scale_kind;
  schema.matrix.scale_format = scale_format;
  schema.matrix.scale_placement =
      scale_kind == LOOM_VALUE_FACT_MATRIX_SCALE_NONE
          ? LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_NONE
          : LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_EXPLICIT;
  schema.matrix.scale_conversion =
      scale_kind == LOOM_VALUE_FACT_MATRIX_SCALE_NONE
          ? LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_NONE
          : LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_CONVERGENT;
  schema.matrix.packed_register_count = packed_register_count;
  schema.matrix.packed_element_count = packed_element_count;
  schema.matrix.zero_scale_fallback =
      scale_kind != LOOM_VALUE_FACT_MATRIX_SCALE_NONE;
  return schema;
}

loom_amdgpu_matrix_payload_shape_t PayloadShape(
    loom_amdgpu_matrix_numeric_type_t numeric_type) {
  loom_amdgpu_matrix_payload_shape_t payload = {};
  payload.numeric_type = numeric_type;
  return payload;
}

TEST(MatrixFactsTest, MapsSchemaToPayloadSelectorAndFlags) {
  loom_value_fact_storage_schema_t schema = MatrixSchema(
      LOOM_VALUE_FACT_MATRIX_FORMAT_FP6, LOOM_VALUE_FACT_MATRIX_SCALE_32,
      LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE, 6, 32);

  loom_amdgpu_matrix_payload_shape_t payload = {};
  EXPECT_TRUE(loom_amdgpu_matrix_payload_from_storage_schema(schema, &payload));
  EXPECT_EQ(payload.numeric_type, LOOM_AMDGPU_MATRIX_NUMERIC_FP6);
  EXPECT_EQ(payload.register_count, 6);
  EXPECT_EQ(payload.element_count, 32);

  loom_amdgpu_matrix_format_selector_t selector =
      LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP8;
  EXPECT_TRUE(loom_amdgpu_matrix_format_selector_from_storage_schema(
      schema, &selector));
  EXPECT_EQ(selector, LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP6);

  loom_amdgpu_matrix_scale_kind_t scale_kind = LOOM_AMDGPU_MATRIX_SCALE_NONE;
  EXPECT_TRUE(
      loom_amdgpu_matrix_scale_kind_from_storage_schema(schema, &scale_kind));
  EXPECT_EQ(scale_kind, LOOM_AMDGPU_MATRIX_SCALE_32);

  loom_amdgpu_matrix_contract_flags_t flags =
      loom_amdgpu_matrix_available_flags_from_storage_schema(schema);
  EXPECT_EQ(flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS);
  EXPECT_EQ(flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED);
  EXPECT_EQ(flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK);
  EXPECT_EQ(flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS, 0u);
}

TEST(MatrixFactsTest, SelectsScaledMfmaFromExactFp6SchemaFacts) {
  loom_value_fact_storage_schema_t lhs_schema = MatrixSchema(
      LOOM_VALUE_FACT_MATRIX_FORMAT_FP6, LOOM_VALUE_FACT_MATRIX_SCALE_32,
      LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE, 6, 32);
  loom_value_fact_storage_schema_t rhs_schema = MatrixSchema(
      LOOM_VALUE_FACT_MATRIX_FORMAT_BF6, LOOM_VALUE_FACT_MATRIX_SCALE_32,
      LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE, 6, 32);

  loom_amdgpu_matrix_contract_match_request_t request = {};
  request.family = LOOM_AMDGPU_MATRIX_FAMILY_MFMA;
  request.tile_shape.result_row_count = 16;
  request.tile_shape.result_column_count = 16;
  request.tile_shape.reduction_count = 128;
  ASSERT_TRUE(loom_amdgpu_matrix_payload_from_storage_schema(
      lhs_schema, &request.lhs_payload));
  ASSERT_TRUE(loom_amdgpu_matrix_payload_from_storage_schema(
      rhs_schema, &request.rhs_payload));
  request.accumulator_payload = PayloadShape(LOOM_AMDGPU_MATRIX_NUMERIC_F32);
  request.result_payload = PayloadShape(LOOM_AMDGPU_MATRIX_NUMERIC_F32);
  ASSERT_TRUE(loom_amdgpu_matrix_scale_kind_from_storage_schema(
      lhs_schema, &request.scale_kind));
  request.feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950 |
                         LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4;
  request.wave_size = 64;
  request.available_flags =
      loom_amdgpu_matrix_available_flags_from_storage_schema(lhs_schema) |
      loom_amdgpu_matrix_available_flags_from_storage_schema(rhs_schema);
  request.required_flags = LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
                           LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS;

  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "mfma.scale.f32.16x16x128.f8f6f4");
}

TEST(MatrixFactsTest, SelectsScaledWmmaF4FromScaleFormatFacts) {
  loom_value_fact_storage_schema_t lhs_schema = MatrixSchema(
      LOOM_VALUE_FACT_MATRIX_FORMAT_FP4, LOOM_VALUE_FACT_MATRIX_SCALE_32,
      LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_E4M3, 16, 128);
  loom_value_fact_storage_schema_t rhs_schema = MatrixSchema(
      LOOM_VALUE_FACT_MATRIX_FORMAT_FP4, LOOM_VALUE_FACT_MATRIX_SCALE_32,
      LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_E4M3, 8, 64);

  loom_amdgpu_matrix_contract_match_request_t request = {};
  request.family = LOOM_AMDGPU_MATRIX_FAMILY_WMMA;
  request.tile_shape.result_row_count = 32;
  request.tile_shape.result_column_count = 16;
  request.tile_shape.reduction_count = 128;
  ASSERT_TRUE(loom_amdgpu_matrix_payload_from_storage_schema(
      lhs_schema, &request.lhs_payload));
  ASSERT_TRUE(loom_amdgpu_matrix_payload_from_storage_schema(
      rhs_schema, &request.rhs_payload));
  request.accumulator_payload = PayloadShape(LOOM_AMDGPU_MATRIX_NUMERIC_F32);
  request.result_payload = PayloadShape(LOOM_AMDGPU_MATRIX_NUMERIC_F32);
  ASSERT_TRUE(loom_amdgpu_matrix_scale_kind_from_storage_schema(
      lhs_schema, &request.scale_kind));
  request.feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250 |
                         LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250_SCALE_F8F6F4;
  request.wave_size = 32;
  request.available_flags =
      loom_amdgpu_matrix_available_flags_from_storage_schema(lhs_schema) |
      loom_amdgpu_matrix_available_flags_from_storage_schema(rhs_schema) |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE;
  request.required_flags = LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
                           LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS;

  const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
      loom_amdgpu_matrix_contract_select(&request, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor->name), "wmma.scale.f32.32x16x128.f4");
}

}  // namespace
