// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/gfx950_descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

const loom_low_descriptor_t* LookupDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  uint32_t ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_EXPECT_OK(
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, key, &ordinal));
  return loom_low_descriptor_set_descriptor_at(descriptor_set, ordinal);
}

TEST(AmdgpuDescriptorsTest, Gfx950CoreDescriptorSetVerifies) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx950_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(descriptor_set));

  iree_string_view_t set_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &set_key));
  EXPECT_TRUE(iree_string_view_equal(set_key, IREE_SV("amdgpu.gfx950.core")));

  EXPECT_GE(descriptor_set->descriptor_count, 8u);
  EXPECT_EQ(descriptor_set->descriptor_ref_count,
            descriptor_set->descriptor_count);
  EXPECT_EQ(descriptor_set->reg_class_count, 3u);
  EXPECT_GE(descriptor_set->schedule_class_count, 7u);
  EXPECT_GE(descriptor_set->resource_count, 7u);
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    EXPECT_NE(
        descriptor_set->reg_classes[i].flags & LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
        0u);
    EXPECT_GT(descriptor_set->reg_classes[i].physical_count, 0u);
  }
}

TEST(AmdgpuDescriptorsTest, Gfx950CoreDescriptorLookupUsesStableKeys) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx950_core_descriptor_set();

  const loom_low_descriptor_t* add_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.v_add_u32"));
  ASSERT_NE(add_descriptor, nullptr);
  iree_string_view_t add_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, add_descriptor->key_string_offset, &add_key));
  EXPECT_TRUE(iree_string_view_equal(add_key, IREE_SV("amdgpu.v_add_u32")));
  EXPECT_EQ(add_descriptor->operand_count, 3u);
  EXPECT_EQ(add_descriptor->result_count, 1u);
  EXPECT_EQ(add_descriptor->encoding_id, 52u);

  const loom_low_descriptor_t* load_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.buffer_load_dword"));
  ASSERT_NE(load_descriptor, nullptr);
  EXPECT_EQ(load_descriptor->operand_count, 4u);
  EXPECT_EQ(load_descriptor->result_count, 1u);
  EXPECT_EQ(load_descriptor->effect_count, 1u);
  EXPECT_EQ(load_descriptor->encoding_id, 20u);
  EXPECT_NE(load_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
            0u);

  const loom_low_descriptor_t* wait_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_waitcnt"));
  ASSERT_NE(wait_descriptor, nullptr);
  EXPECT_EQ(wait_descriptor->operand_count, 0u);
  EXPECT_EQ(wait_descriptor->immediate_count, 2u);
  EXPECT_EQ(wait_descriptor->effect_count, 1u);
  EXPECT_EQ(wait_descriptor->encoding_id, 12u);
  EXPECT_NE(wait_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
            0u);
}

TEST(AmdgpuDescriptorsTest, Gfx950MfmaPacketMatchesCdnaRegisterShape) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx950_core_descriptor_set();

  const loom_low_descriptor_t* mfma_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("amdgpu.v_mfma_f32_16x16x16_f16"));
  ASSERT_NE(mfma_descriptor, nullptr);
  EXPECT_EQ(mfma_descriptor->operand_count, 4u);
  EXPECT_EQ(mfma_descriptor->result_count, 1u);
  EXPECT_EQ(mfma_descriptor->encoding_id, 77u);

  const loom_low_operand_t* mfma_operands =
      &descriptor_set->operands[mfma_descriptor->operand_start];
  EXPECT_EQ(mfma_operands[0].unit_count, 4u);
  EXPECT_EQ(mfma_operands[1].unit_count, 2u);
  EXPECT_EQ(mfma_operands[2].unit_count, 2u);
  EXPECT_EQ(mfma_operands[3].unit_count, 4u);
  EXPECT_EQ(mfma_operands[0].reg_class_alt_count, 2u);
  EXPECT_EQ(mfma_operands[1].reg_class_alt_count, 2u);
  EXPECT_EQ(mfma_operands[2].reg_class_alt_count, 2u);
  EXPECT_EQ(mfma_operands[3].reg_class_alt_count, 3u);

  const loom_low_reg_class_alt_t* result_alts =
      &descriptor_set->reg_class_alts[mfma_operands[0].reg_class_alt_start];
  EXPECT_NE(result_alts[0].reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_NE(result_alts[1].reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  const loom_low_reg_class_alt_t* accumulator_alts =
      &descriptor_set->reg_class_alts[mfma_operands[3].reg_class_alt_start];
  EXPECT_EQ(accumulator_alts[0].reg_class_id, result_alts[0].reg_class_id);
  EXPECT_EQ(accumulator_alts[1].reg_class_id, result_alts[1].reg_class_id);
  EXPECT_EQ(accumulator_alts[2].reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_NE(accumulator_alts[2].flags & LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE,
            0u);
}

TEST(AmdgpuDescriptorsTest, ManifestNamesScalarVectorMemoryAndMatrixPackets) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx950_core_descriptor_set();

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      loom_low_descriptor_set_format_manifest_json(descriptor_set, &builder));
  std::string json(iree_string_builder_buffer(&builder),
                   iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_NE(json.find("\"key\":\"amdgpu.gfx950.core\""), std::string::npos);
  EXPECT_NE(json.find("\"abi_version\":5"), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.s_add_u32\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.v_add_u32\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.buffer_load_dword\""),
            std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.v_mfma_f32_16x16x16_f16\""),
            std::string::npos);
  EXPECT_NE(json.find("\"descriptor_refs\""), std::string::npos);
}

}  // namespace
}  // namespace loom
