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
  EXPECT_TRUE(iree_string_view_equal(processor->low_preset_key,
                                     IREE_SV("amdgpu-gfx11")));
  EXPECT_TRUE(iree_string_view_equal(processor->descriptor_set_key,
                                     IREE_SV("amdgpu.gfx11.core")));
  EXPECT_NE(processor->descriptor_set_stable_id, 0u);
  EXPECT_NE(processor->elf_machine_flags, 0u);
  EXPECT_EQ(processor->default_wavefront_size, 32u);
  EXPECT_EQ(processor->kernel_descriptor_profile,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11);
  EXPECT_TRUE(processor->kernel_descriptor_has_architected_flat_scratch);
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
      IREE_SV("amdgpu.gfx950.core"), &descriptor_set));
  ASSERT_NE(descriptor_set, nullptr);
  EXPECT_TRUE(iree_string_view_equal(descriptor_set->low_preset_key,
                                     IREE_SV("amdgpu-gfx950")));
  ASSERT_NE(descriptor_set->descriptor_set_stable_id, 0u);
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_by_id = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_target_info_lookup_descriptor_set_by_id(
      descriptor_set->descriptor_set_stable_id, &descriptor_set_by_id));
  EXPECT_EQ(descriptor_set_by_id, descriptor_set);
  EXPECT_NE(descriptor_set->s_endpgm_opcode, 0u);
  EXPECT_TRUE(descriptor_set->supports_descriptor_packet_encoding);
}

TEST(AmdgpuTargetInfoTest, LooksUpMatrixOnlyProcessor) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx942"), &processor));
  ASSERT_NE(processor, nullptr);
  EXPECT_TRUE(iree_string_view_is_empty(processor->low_preset_key));
  EXPECT_TRUE(iree_string_view_is_empty(processor->descriptor_set_key));
  EXPECT_EQ(processor->descriptor_set_stable_id, 0u);
  EXPECT_EQ(processor->elf_machine_flags, 0x04Cu);
  EXPECT_EQ(processor->default_wavefront_size, 64u);
  EXPECT_EQ(processor->matrix_feature_profile,
            LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940);
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
