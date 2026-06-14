// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/target_info.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using SchedulingBits = loom_amdgpu_processor_scheduling_bits_t;

static void ExpectSchedulingBits(const loom_amdgpu_processor_info_t* processor,
                                 SchedulingBits expected_scheduling_bits) {
  EXPECT_EQ(processor->scheduling_bits, expected_scheduling_bits);
}

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
  EXPECT_TRUE(processor->kernel_descriptor_uses_gfx10_sgpr_encoding);
  EXPECT_FALSE(processor->kernel_descriptor_has_accum_offset);
  EXPECT_TRUE(processor->kernel_descriptor_has_dx10_clamp_and_ieee_mode);
  EXPECT_TRUE(processor->kernel_descriptor_has_packed_workitem_id);
  ExpectSchedulingBits(processor,
                       LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR);
}

TEST(AmdgpuTargetInfoTest, LooksUpGfx1150Processor) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx1150"), &processor));
  ASSERT_NE(processor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(processor->descriptor_set_key,
                                     IREE_SV("amdgpu.rdna3_5.core")));
  EXPECT_EQ(processor->descriptor_set_ordinal,
            LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3_5);
  EXPECT_EQ(processor->elf_machine_flags, 0x043u);
  EXPECT_EQ(processor->default_wavefront_size, 32u);
  EXPECT_EQ(processor->kernel_descriptor_profile,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11);
  EXPECT_EQ(processor->kernel_descriptor_vgpr_encoding_granule_wave32, 8u);
  EXPECT_EQ(processor->kernel_descriptor_vgpr_encoding_granule_wave64, 4u);
  EXPECT_TRUE(processor->kernel_descriptor_has_architected_flat_scratch);
  EXPECT_TRUE(processor->kernel_descriptor_uses_gfx10_sgpr_encoding);
  EXPECT_FALSE(processor->kernel_descriptor_has_accum_offset);
  EXPECT_TRUE(processor->kernel_descriptor_has_dx10_clamp_and_ieee_mode);
  EXPECT_TRUE(processor->kernel_descriptor_has_packed_workitem_id);
  EXPECT_EQ(processor->matrix_feature_profile,
            LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12);
  ExpectSchedulingBits(processor,
                       LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR);
}

TEST(AmdgpuTargetInfoTest, IteratesProcessors) {
  const iree_host_size_t count = loom_amdgpu_target_info_processor_count();
  ASSERT_GT(count, 0u);
  EXPECT_EQ(loom_amdgpu_target_info_processor_at(count), nullptr);

  const loom_amdgpu_processor_info_t* first =
      loom_amdgpu_target_info_processor_at(0);
  ASSERT_NE(first, nullptr);
  EXPECT_FALSE(iree_string_view_is_empty(first->processor));
}

TEST(AmdgpuTargetInfoTest, LooksUpDescriptorSetEncodingProfile) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_target_info_lookup_descriptor_set(
      IREE_SV("amdgpu.cdna3.core"), &descriptor_set));
  ASSERT_NE(descriptor_set, nullptr);
  ASSERT_EQ(descriptor_set->descriptor_set_ordinal,
            LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA3);
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_by_ordinal = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_target_info_lookup_descriptor_set_by_ordinal(
      descriptor_set->descriptor_set_ordinal, &descriptor_set_by_ordinal));
  EXPECT_EQ(descriptor_set_by_ordinal, descriptor_set);
  EXPECT_NE(descriptor_set->s_endpgm_opcode, 0u);
  EXPECT_TRUE(descriptor_set->supports_descriptor_packet_encoding);
  EXPECT_EQ(descriptor_set->buffer_resource_cache_swizzle,
            LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE);
  EXPECT_EQ(descriptor_set->vector_memory_cache_policy_encoding,
            LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1);
}

TEST(AmdgpuTargetInfoTest, DescriptorSetAtReturnsNullForUnknownOrdinal) {
  EXPECT_EQ(loom_amdgpu_target_info_descriptor_set_at(
                LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE),
            nullptr);
  EXPECT_EQ(loom_amdgpu_target_info_descriptor_set_at(UINT16_MAX - 1), nullptr);
}

TEST(AmdgpuTargetInfoTest, LooksUpGfx942Processor) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx942"), &processor));
  ASSERT_NE(processor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(processor->descriptor_set_key,
                                     IREE_SV("amdgpu.cdna3.core")));
  EXPECT_EQ(processor->descriptor_set_ordinal,
            LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA3);
  EXPECT_EQ(processor->elf_machine_flags, 0x04Cu);
  EXPECT_EQ(processor->elf_feature_flags, 0x500u);
  EXPECT_EQ(processor->default_wavefront_size, 64u);
  EXPECT_EQ(processor->kernel_descriptor_profile,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9);
  EXPECT_EQ(processor->matrix_feature_profile,
            LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940);
  EXPECT_EQ(processor->kernel_descriptor_vgpr_encoding_granule_wave32, 8u);
  EXPECT_EQ(processor->kernel_descriptor_vgpr_encoding_granule_wave64, 8u);
  EXPECT_TRUE(processor->kernel_descriptor_has_architected_flat_scratch);
  EXPECT_FALSE(processor->kernel_descriptor_uses_gfx10_sgpr_encoding);
  EXPECT_TRUE(processor->kernel_descriptor_has_accum_offset);
  EXPECT_TRUE(processor->kernel_descriptor_has_dx10_clamp_and_ieee_mode);
  EXPECT_TRUE(processor->kernel_descriptor_has_packed_workitem_id);
  ExpectSchedulingBits(
      processor,
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES |
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES |
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_SDWA_DST_SEL_WAIT_STATES);
}

TEST(AmdgpuTargetInfoTest, LooksUpGfx950Processor) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx950"), &processor));
  ASSERT_NE(processor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(processor->descriptor_set_key,
                                     IREE_SV("amdgpu.cdna4.core")));
  EXPECT_EQ(processor->descriptor_set_ordinal,
            LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA4);
  EXPECT_EQ(processor->kernel_descriptor_profile,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9);
  EXPECT_EQ(processor->matrix_feature_profile,
            LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950);
  EXPECT_EQ(processor->kernel_descriptor_vgpr_encoding_granule_wave32, 8u);
  EXPECT_EQ(processor->kernel_descriptor_vgpr_encoding_granule_wave64, 8u);
  EXPECT_TRUE(processor->kernel_descriptor_has_architected_flat_scratch);
  EXPECT_FALSE(processor->kernel_descriptor_uses_gfx10_sgpr_encoding);
  EXPECT_TRUE(processor->kernel_descriptor_has_accum_offset);
  EXPECT_TRUE(processor->kernel_descriptor_has_dx10_clamp_and_ieee_mode);
  EXPECT_TRUE(processor->kernel_descriptor_has_packed_workitem_id);
  ExpectSchedulingBits(
      processor,
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES |
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES |
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_SDWA_DST_SEL_WAIT_STATES);
}

TEST(AmdgpuTargetInfoTest, WavefrontSizeSupportMatchesGfxFamilies) {
  struct Case {
    iree_string_view_t processor_name;
    bool supports_wave32;
    bool supports_wave64;
  };
  static const Case cases[] = {
      {IREE_SV("gfx942"), false, true},  {IREE_SV("gfx950"), false, true},
      {IREE_SV("gfx1100"), true, true},  {IREE_SV("gfx1200"), true, true},
      {IREE_SV("gfx1250"), true, false},
  };
  for (const Case& c : cases) {
    const loom_amdgpu_processor_info_t* processor = nullptr;
    IREE_ASSERT_OK(
        loom_amdgpu_target_info_lookup_processor(c.processor_name, &processor));
    ASSERT_NE(processor, nullptr);
    EXPECT_EQ(loom_amdgpu_processor_supports_wavefront_size(processor, 32),
              c.supports_wave32)
        << std::string(c.processor_name.data, c.processor_name.size);
    EXPECT_EQ(loom_amdgpu_processor_supports_wavefront_size(processor, 64),
              c.supports_wave64)
        << std::string(c.processor_name.data, c.processor_name.size);
    EXPECT_FALSE(loom_amdgpu_processor_supports_wavefront_size(processor, 0));
  }
}

TEST(AmdgpuTargetInfoTest, LooksUpGfx1200Processor) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx1200"), &processor));
  ASSERT_NE(processor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(processor->descriptor_set_key,
                                     IREE_SV("amdgpu.rdna4.core")));
  EXPECT_EQ(processor->descriptor_set_ordinal,
            LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4);
  EXPECT_EQ(processor->kernel_descriptor_profile,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12);
  EXPECT_EQ(processor->matrix_feature_profile,
            LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12);
  EXPECT_EQ(processor->kernel_descriptor_vgpr_encoding_granule_wave32, 8u);
  EXPECT_EQ(processor->kernel_descriptor_vgpr_encoding_granule_wave64, 4u);
  EXPECT_TRUE(processor->kernel_descriptor_has_architected_flat_scratch);
  EXPECT_TRUE(processor->kernel_descriptor_uses_gfx10_sgpr_encoding);
  EXPECT_FALSE(processor->kernel_descriptor_has_accum_offset);
  EXPECT_FALSE(processor->kernel_descriptor_has_dx10_clamp_and_ieee_mode);
  EXPECT_TRUE(processor->kernel_descriptor_has_packed_workitem_id);
  ExpectSchedulingBits(processor,
                       LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR);
}

TEST(AmdgpuTargetInfoTest, LooksUpGfx1250Processor) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx1250"), &processor));
  ASSERT_NE(processor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(processor->descriptor_set_key,
                                     IREE_SV("amdgpu.rdna4.gfx125x.core")));
  EXPECT_EQ(processor->descriptor_set_ordinal,
            LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X);
  EXPECT_EQ(processor->kernel_descriptor_profile,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125);
  EXPECT_EQ(processor->matrix_feature_profile,
            LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250);
  EXPECT_EQ(processor->kernel_descriptor_vgpr_encoding_granule_wave32, 16u);
  EXPECT_EQ(processor->kernel_descriptor_vgpr_encoding_granule_wave64, 8u);
  EXPECT_TRUE(processor->kernel_descriptor_has_architected_flat_scratch);
  EXPECT_TRUE(processor->kernel_descriptor_uses_gfx10_sgpr_encoding);
  EXPECT_FALSE(processor->kernel_descriptor_has_accum_offset);
  EXPECT_FALSE(processor->kernel_descriptor_has_dx10_clamp_and_ieee_mode);
  EXPECT_TRUE(processor->kernel_descriptor_has_packed_workitem_id);
  ExpectSchedulingBits(processor, 0);
}

TEST(AmdgpuTargetInfoTest, MatchesAmdhsaGfx9PlusProcessorElfFlags) {
  const struct {
    iree_string_view_t processor;
    uint32_t elf_flags;
  } cases[] = {
      {IREE_SV("gfx900"), 0x12Cu},
      {IREE_SV("gfx902"), 0x12Du},
      {IREE_SV("gfx904"), 0x12Eu},
      {IREE_SV("gfx906"), 0x52Fu},
      {IREE_SV("gfx908"), 0x530u},
      {IREE_SV("gfx909"), 0x131u},
      {IREE_SV("gfx90a"), 0x53Fu},
      {IREE_SV("gfx90c"), 0x132u},
      {IREE_SV("gfx940"), 0x540u},
      {IREE_SV("gfx941"), 0x54Bu},
      {IREE_SV("gfx942"), 0x54Cu},
      {IREE_SV("gfx950"), 0x54Fu},
      {IREE_SV("gfx1010"), 0x133u},
      {IREE_SV("gfx1011"), 0x134u},
      {IREE_SV("gfx1012"), 0x135u},
      {IREE_SV("gfx1013"), 0x142u},
      {IREE_SV("gfx1030"), 0x036u},
      {IREE_SV("gfx1031"), 0x037u},
      {IREE_SV("gfx1032"), 0x038u},
      {IREE_SV("gfx1033"), 0x039u},
      {IREE_SV("gfx1034"), 0x03Eu},
      {IREE_SV("gfx1035"), 0x03Du},
      {IREE_SV("gfx1036"), 0x045u},
      {IREE_SV("gfx1100"), 0x041u},
      {IREE_SV("gfx1101"), 0x046u},
      {IREE_SV("gfx1102"), 0x047u},
      {IREE_SV("gfx1103"), 0x044u},
      {IREE_SV("gfx1150"), 0x043u},
      {IREE_SV("gfx1151"), 0x04Au},
      {IREE_SV("gfx1152"), 0x055u},
      {IREE_SV("gfx1153"), 0x058u},
      {IREE_SV("gfx1170"), 0x05Du},
      {IREE_SV("gfx1171"), 0x05Eu},
      {IREE_SV("gfx1172"), 0x05Cu},
      {IREE_SV("gfx1200"), 0x048u},
      {IREE_SV("gfx1201"), 0x04Eu},
      {IREE_SV("gfx1250"), 0x549u},
      {IREE_SV("gfx1251"), 0x55Au},
      {IREE_SV("gfx1310"), 0x050u},
      {IREE_SV("gfx9-generic"), 0x01000151u},
      {IREE_SV("gfx10-1-generic"), 0x01000152u},
      {IREE_SV("gfx10-3-generic"), 0x01000053u},
      {IREE_SV("gfx11-generic"), 0x01000054u},
      {IREE_SV("gfx12-generic"), 0x01000059u},
      {IREE_SV("gfx9-4-generic"), 0x0100055Fu},
      {IREE_SV("gfx12-5-generic"), 0x0100005Bu},
  };
  for (const auto& c : cases) {
    const loom_amdgpu_processor_info_t* processor = nullptr;
    IREE_ASSERT_OK(
        loom_amdgpu_target_info_lookup_processor(c.processor, &processor));
    ASSERT_NE(processor, nullptr);
    EXPECT_TRUE(iree_string_view_equal(processor->processor, c.processor));
    EXPECT_EQ(processor->elf_machine_flags | processor->elf_feature_flags,
              c.elf_flags);
  }
}

TEST(AmdgpuTargetInfoTest, LooksUpGfx1170Processor) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(
      loom_amdgpu_target_info_lookup_processor(IREE_SV("gfx1170"), &processor));
  ASSERT_NE(processor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(processor->descriptor_set_key,
                                     IREE_SV("amdgpu.rdna3_5.core")));
  EXPECT_EQ(processor->descriptor_set_ordinal,
            LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3_5);
  EXPECT_EQ(processor->elf_machine_flags, 0x05Du);
  EXPECT_EQ(processor->default_wavefront_size, 32u);
  EXPECT_EQ(processor->kernel_descriptor_profile,
            LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11);
  EXPECT_EQ(processor->kernel_descriptor_vgpr_encoding_granule_wave32, 8u);
  EXPECT_EQ(processor->kernel_descriptor_vgpr_encoding_granule_wave64, 4u);
  EXPECT_TRUE(processor->kernel_descriptor_has_architected_flat_scratch);
  EXPECT_TRUE(processor->kernel_descriptor_uses_gfx10_sgpr_encoding);
  EXPECT_FALSE(processor->kernel_descriptor_has_accum_offset);
  EXPECT_TRUE(processor->kernel_descriptor_has_dx10_clamp_and_ieee_mode);
  EXPECT_TRUE(processor->kernel_descriptor_has_packed_workitem_id);
  EXPECT_EQ(processor->matrix_feature_profile,
            LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12);
  ExpectSchedulingBits(processor,
                       LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR);
}

TEST(AmdgpuTargetInfoTest, LooksUpGfx94GenericSchedulingFacts) {
  const loom_amdgpu_processor_info_t* processor = nullptr;
  IREE_ASSERT_OK(loom_amdgpu_target_info_lookup_processor(
      IREE_SV("gfx9-4-generic"), &processor));
  ASSERT_NE(processor, nullptr);
  EXPECT_TRUE(iree_string_view_is_empty(processor->descriptor_set_key));
  EXPECT_EQ(processor->descriptor_set_ordinal,
            LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE);
  EXPECT_EQ(processor->matrix_feature_profile,
            LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940);
  ExpectSchedulingBits(
      processor,
      LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES |
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES |
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_SDWA_DST_SEL_WAIT_STATES);
}

TEST(AmdgpuTargetInfoTest, ParsesAmdhsaTargetIdWithFeatureSuffix) {
  loom_amdgpu_amdhsa_target_id_t target_id = {};
  IREE_ASSERT_OK(loom_amdgpu_target_info_parse_amdhsa_target_id(
      IREE_SV("amdgcn-amd-amdhsa--gfx1100:sramecc+:xnack-"), &target_id));
  ASSERT_NE(target_id.processor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(target_id.processor->processor,
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
