// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/abi/tsan.h"

#include "iree/testing/gtest.h"

namespace {

TEST(AmdgpuTsanAbiTest, ShadowLayoutMarkerCannotAliasPayload) {
  const uint32_t marker = LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_LAYOUT_TAG_MARKER;
  const uint32_t chunk_state_mask =
      (1u << LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_BIT_COUNT) - 1u;
  EXPECT_EQ(marker & chunk_state_mask, 2u);

  const uint32_t access_kind_mask =
      (1u << LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_BIT_COUNT) - 1u;
  EXPECT_GT(marker & access_kind_mask,
            static_cast<uint32_t>(LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC));
}

}  // namespace
