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

  uint32_t add_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("amdgpu.v_add_u32"), &add_ordinal));
  EXPECT_NE(add_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  const loom_low_descriptor_t* add_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, add_ordinal);
  ASSERT_NE(add_descriptor, nullptr);
  iree_string_view_t add_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, add_descriptor->key_string_offset, &add_key));
  EXPECT_TRUE(iree_string_view_equal(add_key, IREE_SV("amdgpu.v_add_u32")));
  EXPECT_EQ(add_descriptor->operand_count, 3u);
  EXPECT_EQ(add_descriptor->result_count, 1u);

  uint32_t load_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("amdgpu.buffer_load_dword"), &load_ordinal));
  const loom_low_descriptor_t* load_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, load_ordinal);
  ASSERT_NE(load_descriptor, nullptr);
  EXPECT_EQ(load_descriptor->operand_count, 4u);
  EXPECT_EQ(load_descriptor->result_count, 1u);
  EXPECT_EQ(load_descriptor->effect_count, 1u);
  EXPECT_NE(load_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
            0u);

  uint32_t mfma_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("amdgpu.v_mfma_f32_16x16x16_f16"),
      &mfma_ordinal));
  const loom_low_descriptor_t* mfma_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, mfma_ordinal);
  ASSERT_NE(mfma_descriptor, nullptr);
  EXPECT_EQ(mfma_descriptor->operand_count, 4u);
  EXPECT_EQ(mfma_descriptor->result_count, 1u);
  const loom_low_operand_t* mfma_operands =
      &descriptor_set->operands[mfma_descriptor->operand_start];
  EXPECT_EQ(mfma_operands[0].unit_count, 4u);
  EXPECT_EQ(mfma_operands[1].unit_count, 2u);
  EXPECT_EQ(mfma_operands[2].unit_count, 2u);
  EXPECT_EQ(mfma_operands[3].unit_count, 4u);
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
