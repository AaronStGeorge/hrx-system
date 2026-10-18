// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_history_layout.h"

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"

static const loom_amdgpu_sanitizer_race_history_layout_t
    loom_amdgpu_sanitizer_race_legacy_layout = {
        .access_kind =
            {
                .shift = LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_SHIFT,
                .bit_count =
                    LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_BIT_COUNT,
            },
        .workitem =
            {
                .shift = LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_SHIFT,
                .bit_count = LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_BIT_COUNT,
            },
        .workgroup_epoch =
            {
                .shift = LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_SHIFT,
                .bit_count = LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_BIT_COUNT,
            },
        .subgroup_epoch = {0},
        .generation =
            {
                .shift = LOOM_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_SHIFT,
                .bit_count = LOOM_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_BIT_COUNT,
            },
        .site =
            {
                .shift = LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_SHIFT,
                .bit_count = LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_BIT_COUNT,
            },
        .site_id_base = 0,
        .format_tag = 0,
        .wavefront_size_log2 = 0,
};

const loom_amdgpu_sanitizer_race_history_layout_t*
loom_amdgpu_sanitizer_race_legacy_history_layout(void) {
  return &loom_amdgpu_sanitizer_race_legacy_layout;
}

uint64_t loom_amdgpu_sanitizer_race_history_field_mask(
    loom_amdgpu_sanitizer_race_history_field_t field) {
  IREE_ASSERT_LE((uint32_t)field.shift + field.bit_count, 64u);
  if (field.bit_count == 0) {
    return 0;
  }
  const uint64_t unshifted_mask = field.bit_count == 64
                                      ? UINT64_MAX
                                      : (UINT64_C(1) << field.bit_count) - 1u;
  return unshifted_mask << field.shift;
}

uint32_t loom_amdgpu_sanitizer_race_history_field_low_mask(
    loom_amdgpu_sanitizer_race_history_field_t field) {
  return (uint32_t)loom_amdgpu_sanitizer_race_history_field_mask(field);
}

uint32_t loom_amdgpu_sanitizer_race_history_field_high_mask(
    loom_amdgpu_sanitizer_race_history_field_t field) {
  return (uint32_t)(loom_amdgpu_sanitizer_race_history_field_mask(field) >> 32);
}
