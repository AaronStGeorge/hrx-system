// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/matrix_facts.h"

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

}  // namespace
