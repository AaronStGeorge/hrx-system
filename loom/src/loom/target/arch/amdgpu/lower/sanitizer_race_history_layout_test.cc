// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_history_layout.h"

#include <algorithm>
#include <cstdint>

#include "iree/testing/gtest.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"

namespace {

TEST(AmdgpuSanitizerRaceHistoryLayoutTest, LegacyLayoutMatchesDeviceAbi) {
  const loom_amdgpu_sanitizer_race_history_layout_t& layout =
      *loom_amdgpu_sanitizer_race_legacy_history_layout();

  EXPECT_EQ(layout.access_kind.shift,
            LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_SHIFT);
  EXPECT_EQ(layout.access_kind.bit_count,
            LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_BIT_COUNT);
  EXPECT_EQ(layout.workitem.shift,
            LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_SHIFT);
  EXPECT_EQ(layout.workitem.bit_count,
            LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_BIT_COUNT);
  EXPECT_EQ(layout.workgroup_epoch.shift,
            LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_SHIFT);
  EXPECT_EQ(layout.workgroup_epoch.bit_count,
            LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_BIT_COUNT);
  EXPECT_EQ(layout.subgroup_epoch.bit_count, 0u);
  EXPECT_EQ(layout.generation.shift,
            LOOM_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_SHIFT);
  EXPECT_EQ(layout.generation.bit_count,
            LOOM_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_BIT_COUNT);
  EXPECT_EQ(layout.site.shift, LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_SHIFT);
  EXPECT_EQ(layout.site.bit_count,
            LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_BIT_COUNT);
  EXPECT_EQ(layout.site_id_base, 0u);
  EXPECT_EQ(layout.format_tag, 0u);
  EXPECT_EQ(layout.wavefront_size_log2, 0u);
}

TEST(AmdgpuSanitizerRaceHistoryLayoutTest, SplitsFieldsAcrossHistoryLanes) {
  const loom_amdgpu_sanitizer_race_history_layout_t& layout =
      *loom_amdgpu_sanitizer_race_legacy_history_layout();

  EXPECT_EQ(loom_amdgpu_sanitizer_race_history_field_low_mask(layout.workitem),
            UINT32_C(0x00001ff8));
  EXPECT_EQ(loom_amdgpu_sanitizer_race_history_field_high_mask(layout.workitem),
            0u);
  EXPECT_EQ(
      loom_amdgpu_sanitizer_race_history_field_low_mask(layout.generation),
      UINT32_C(0xff800000));
  EXPECT_EQ(
      loom_amdgpu_sanitizer_race_history_field_high_mask(layout.generation),
      UINT32_C(0x000007ff));
  EXPECT_EQ(loom_amdgpu_sanitizer_race_history_field_mask(layout.site),
            UINT64_C(0xfffff80000000000));
}

TEST(AmdgpuSanitizerRaceHistoryLayoutTest,
     PacksOrdinarySubgroupFunctionWithoutGrowingHistory) {
  const loom_amdgpu_sanitizer_race_history_layout_config_t config = {
      .flat_workgroup_size = 64,
      .function_site_count = 3,
      .function_site_id_base = 17,
      .wavefront_size = 32,
  };
  loom_amdgpu_sanitizer_race_history_layout_t layout = {};
  EXPECT_EQ(loom_amdgpu_sanitizer_race_select_subgroup_history_layout(&config,
                                                                      &layout),
            LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_SUPPORTED);

  EXPECT_EQ(layout.access_kind.shift, 0u);
  EXPECT_EQ(layout.access_kind.bit_count, 3u);
  EXPECT_EQ(layout.workitem.shift, 3u);
  EXPECT_EQ(layout.workitem.bit_count, 6u);
  EXPECT_EQ(layout.workgroup_epoch.shift, 9u);
  EXPECT_EQ(layout.workgroup_epoch.bit_count, 10u);
  EXPECT_EQ(layout.subgroup_epoch.shift, 19u);
  EXPECT_EQ(layout.subgroup_epoch.bit_count, 23u);
  EXPECT_EQ(layout.generation.shift, 42u);
  EXPECT_EQ(layout.generation.bit_count, 20u);
  EXPECT_EQ(layout.site.shift, 62u);
  EXPECT_EQ(layout.site.bit_count, 2u);
  EXPECT_EQ(layout.site_id_base, 17u);
  EXPECT_NE(layout.format_tag, 0u);
  EXPECT_EQ(layout.wavefront_size_log2, 5u);
  EXPECT_EQ(
      loom_amdgpu_sanitizer_race_history_field_mask(layout.access_kind) |
          loom_amdgpu_sanitizer_race_history_field_mask(layout.workitem) |
          loom_amdgpu_sanitizer_race_history_field_mask(
              layout.workgroup_epoch) |
          loom_amdgpu_sanitizer_race_history_field_mask(layout.subgroup_epoch) |
          loom_amdgpu_sanitizer_race_history_field_mask(layout.generation) |
          loom_amdgpu_sanitizer_race_history_field_mask(layout.site),
      UINT64_MAX);
}

TEST(AmdgpuSanitizerRaceHistoryLayoutTest,
     PreservesMinimumEpochCapacityAtMaximumShape) {
  const loom_amdgpu_sanitizer_race_history_layout_config_t config = {
      .flat_workgroup_size = 1024,
      .function_site_count = 1u << 21,
      .function_site_id_base = 0,
      .wavefront_size = 64,
  };
  loom_amdgpu_sanitizer_race_history_layout_t layout = {};
  EXPECT_EQ(loom_amdgpu_sanitizer_race_select_subgroup_history_layout(&config,
                                                                      &layout),
            LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_SUPPORTED);

  EXPECT_EQ(layout.workitem.bit_count, 10u);
  EXPECT_EQ(layout.subgroup_epoch.bit_count,
            LOOM_AMDGPU_SANITIZER_RACE_MIN_SUBGROUP_EPOCH_BIT_COUNT);
  EXPECT_EQ(layout.generation.bit_count, 0u);
  EXPECT_EQ(layout.site.bit_count, 21u);
  EXPECT_EQ(layout.wavefront_size_log2, 6u);
}

TEST(AmdgpuSanitizerRaceHistoryLayoutTest,
     EncodesEveryRepresentableWidthCombination) {
  constexpr uint32_t kFormatPayloadMask = (1u << 21) - 1u;
  for (uint32_t wavefront_size : {32u, 64u}) {
    for (uint32_t workitem_bit_count = 0; workitem_bit_count <= 10;
         ++workitem_bit_count) {
      for (uint32_t site_bit_count = 0; site_bit_count <= 21;
           ++site_bit_count) {
        SCOPED_TRACE(::testing::Message()
                     << "wavefront_size=" << wavefront_size
                     << " workitem_bit_count=" << workitem_bit_count
                     << " site_bit_count=" << site_bit_count);
        const loom_amdgpu_sanitizer_race_history_layout_config_t config = {
            .flat_workgroup_size = 1u << workitem_bit_count,
            .function_site_count = 1u << site_bit_count,
            .function_site_id_base = 123,
            .wavefront_size = wavefront_size,
        };
        loom_amdgpu_sanitizer_race_history_layout_t layout = {};
        ASSERT_EQ(loom_amdgpu_sanitizer_race_select_subgroup_history_layout(
                      &config, &layout),
                  LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_SUPPORTED);

        const uint32_t variable_bit_count =
            31u - workitem_bit_count - site_bit_count;
        const uint32_t generation_bit_count = std::min(variable_bit_count, 20u);
        const uint32_t subgroup_epoch_bit_count =
            20u + variable_bit_count - generation_bit_count;
        EXPECT_EQ(layout.workitem.bit_count, workitem_bit_count);
        EXPECT_EQ(layout.site.bit_count, site_bit_count);
        EXPECT_EQ(layout.generation.bit_count, generation_bit_count);
        EXPECT_EQ(layout.subgroup_epoch.bit_count, subgroup_epoch_bit_count);
        EXPECT_EQ(layout.site_id_base, 123u);
        EXPECT_EQ(layout.site.shift + layout.site.bit_count, 64u);
        EXPECT_NE(layout.format_tag >> 21u, 0u);
        const uint32_t expected_payload =
            workitem_bit_count | (site_bit_count << 4u) |
            (subgroup_epoch_bit_count << 9u) | (generation_bit_count << 15u) |
            ((wavefront_size == 64u ? 1u : 0u) << 20u);
        EXPECT_EQ(layout.format_tag & kFormatPayloadMask, expected_payload);
      }
    }
  }
}

TEST(AmdgpuSanitizerRaceHistoryLayoutTest, RejectsUnrepresentableFamilies) {
  loom_amdgpu_sanitizer_race_history_layout_t layout = {};
  const loom_amdgpu_sanitizer_race_history_layout_config_t workitem_config = {
      .flat_workgroup_size = 1025,
      .function_site_count = 1,
      .wavefront_size = 32,
  };
  EXPECT_EQ(
      loom_amdgpu_sanitizer_race_select_subgroup_history_layout(
          &workitem_config, &layout),
      LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_WORKITEM_CAPACITY_EXCEEDED);
  const loom_amdgpu_sanitizer_race_history_layout_config_t site_config = {
      .flat_workgroup_size = 1,
      .function_site_count = (1u << 21) + 1u,
      .wavefront_size = 32,
  };
  EXPECT_EQ(loom_amdgpu_sanitizer_race_select_subgroup_history_layout(
                &site_config, &layout),
            LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_SITE_CAPACITY_EXCEEDED);
  const loom_amdgpu_sanitizer_race_history_layout_config_t wave_config = {
      .flat_workgroup_size = 1,
      .function_site_count = 1,
      .wavefront_size = 16,
  };
  EXPECT_EQ(loom_amdgpu_sanitizer_race_select_subgroup_history_layout(
                &wave_config, &layout),
            LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_WAVEFRONT_UNSUPPORTED);
}

}  // namespace
