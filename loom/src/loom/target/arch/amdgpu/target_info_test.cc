// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/target_info.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(AmdgpuTargetInfoTest, LooksUpGfx11Processor) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx1100"), &processor));
  ASSERT_NE(processor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(processor->descriptor_set_key,
                                     IREE_SV("amdgpu.rdna3.core")));
  EXPECT_EQ(processor->descriptor_set_ordinal,
            LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3);
  EXPECT_NE(processor->elf_machine_flags, 0u);
  EXPECT_EQ(processor->default_wavefront_size, 32u);
  EXPECT_EQ(processor->kernel_descriptor_profile,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11);
  EXPECT_TRUE(processor->kernel_descriptor_has_architected_flat_scratch);
  EXPECT_TRUE(processor->kernel_descriptor_has_packed_workitem_id);
}

TEST(AmdgpuTargetInfoTest, IteratesProcessors) {
  const iree_host_size_t count = loom_amdgpu_target_info_processor_count();
  ASSERT_GT(count, 0u);
  EXPECT_EQ(loom_amdgpu_target_info_processor_at(count), nullptr);

  const loom_amdgpu_processor_info_t* first =
      loom_amdgpu_target_info_processor_at(0);
  ASSERT_NE(first, nullptr);
  EXPECT_FALSE(iree_string_view_is_empty(first->target_cpu));
}

TEST(AmdgpuTargetInfoTest, LooksUpDescriptorSetEncodingProfile) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_target_info_lookup_descriptor_set(
      IREE_SV("amdgpu.cdna4.core"), &descriptor_set));
  ASSERT_NE(descriptor_set, nullptr);
  ASSERT_EQ(descriptor_set->descriptor_set_ordinal,
            LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA4);
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_by_ordinal = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_target_info_lookup_descriptor_set_by_ordinal(
      descriptor_set->descriptor_set_ordinal, &descriptor_set_by_ordinal));
  EXPECT_EQ(descriptor_set_by_ordinal, descriptor_set);
  EXPECT_NE(descriptor_set->s_endpgm_opcode, 0u);
  EXPECT_TRUE(descriptor_set->supports_descriptor_packet_encoding);
  EXPECT_EQ(descriptor_set->buffer_resource_cache_swizzle,
            LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT);
  EXPECT_EQ(descriptor_set->vector_memory_cache_policy_encoding,
            LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1);
}

TEST(AmdgpuTargetInfoTest, DescriptorSetAtReturnsNullForUnknownOrdinal) {
  EXPECT_EQ(loom_amdgpu_target_info_descriptor_set_at(
                LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE),
            nullptr);
  EXPECT_EQ(loom_amdgpu_target_info_descriptor_set_at(UINT16_MAX - 1), nullptr);
}

TEST(AmdgpuTargetInfoTest, LooksUpMatrixOnlyProcessor) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx942"), &processor));
  ASSERT_NE(processor, nullptr);
  EXPECT_TRUE(iree_string_view_is_empty(processor->descriptor_set_key));
  EXPECT_EQ(processor->descriptor_set_ordinal,
            LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE);
  EXPECT_EQ(processor->elf_machine_flags, 0x04Cu);
  EXPECT_EQ(processor->default_wavefront_size, 64u);
  EXPECT_EQ(processor->matrix_feature_profile,
            LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940);
  EXPECT_TRUE(processor->kernel_descriptor_has_packed_workitem_id);
}

TEST(AmdgpuTargetInfoTest, LooksUpGfx1170AsMatrixOnlyProcessor) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx1170"), &processor));
  ASSERT_NE(processor, nullptr);
  EXPECT_TRUE(iree_string_view_is_empty(processor->descriptor_set_key));
  EXPECT_EQ(processor->descriptor_set_ordinal,
            LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE);
  EXPECT_EQ(processor->elf_machine_flags, 0x05Du);
  EXPECT_EQ(processor->default_wavefront_size, 32u);
  EXPECT_EQ(processor->matrix_feature_profile,
            LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12);
}

TEST(AmdgpuTargetInfoTest, ParsesAmdhsaTargetIdWithFeatureSuffix) {
  loom_amdgpu_amdhsa_target_id_t target_id = {};
  IREE_ASSERT_OK(loom_amdgpu_target_info_parse_amdhsa_target_id(
      IREE_SV("amdgcn-amd-amdhsa--gfx1100:sramecc+:xnack-"), &target_id));
  ASSERT_NE(target_id.processor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(target_id.processor->target_cpu,
                                     IREE_SV("gfx1100")));
  EXPECT_TRUE(iree_string_view_equal(target_id.feature_suffix,
                                     IREE_SV("sramecc+:xnack-")));
}

TEST(AmdgpuTargetInfoTest, RejectsUnknownProcessor) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx9999"), &processor));
  EXPECT_EQ(processor, nullptr);
}

TEST(AmdgpuTargetInfoTest, RejectsNonAmdhsaTargetId) {
  loom_amdgpu_amdhsa_target_id_t target_id = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_amdgpu_target_info_parse_amdhsa_target_id(
                            IREE_SV("amdgcn-amd-amdpal--gfx1100"), &target_id));
}

TEST(AmdgpuTargetInfoTest, RejectsEmptyFeatureSuffix) {
  loom_amdgpu_amdhsa_target_id_t target_id = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_target_info_parse_amdhsa_target_id(
          IREE_SV("amdgcn-amd-amdhsa--gfx1100:"), &target_id));
}

TEST(AmdgpuTargetInfoTest, RejectsUnsupportedTargetIdCharacters) {
  loom_amdgpu_amdhsa_target_id_t target_id = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_target_info_parse_amdhsa_target_id(
          IREE_SV("amdgcn-amd-amdhsa--gfx1100\\bad"), &target_id));
}

}  // namespace
