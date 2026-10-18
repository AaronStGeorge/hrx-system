// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_history_layout.h"

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

}  // namespace
