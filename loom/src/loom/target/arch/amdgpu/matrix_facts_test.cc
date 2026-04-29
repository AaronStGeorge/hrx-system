// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/matrix_facts.h"

#include "iree/testing/gtest.h"

namespace {

loom_value_fact_storage_schema_t EncodedSchema(
    loom_value_fact_numeric_format_flags_t element_format,
    uint16_t scale_group_element_count,
    loom_value_fact_numeric_format_flags_t scale_format,
    uint16_t payload_register_count, uint16_t payload_element_count) {
  loom_value_fact_storage_schema_t schema = {};
  schema.encoded_operand.element_format = element_format;
  schema.encoded_operand.payload_packing =
      LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT;
  schema.encoded_operand.scale_format = scale_format;
  schema.encoded_operand.scale_topology =
      scale_group_element_count == 0 ? LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE
                                     : LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D;
  schema.encoded_operand.payload_register_count = payload_register_count;
  schema.encoded_operand.payload_element_count = payload_element_count;
  schema.encoded_operand.scale_group_element_count = scale_group_element_count;
  schema.encoded_operand.scale_operand_count =
      scale_group_element_count == 0 ? 0 : 1;
  if (scale_group_element_count != 0) {
    schema.encoded_operand.flags |=
        LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK;
  }
  return schema;
}

TEST(MatrixFactsTest, MapsSchemaToPayloadSelectorAndFlags) {
  loom_value_fact_storage_schema_t schema =
      EncodedSchema(LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2, 32,
                    LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE, 6, 32);

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
