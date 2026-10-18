// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Packed AMDGPU TSAN access-history layout.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_HISTORY_LAYOUT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_HISTORY_LAYOUT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_sanitizer_race_history_field_t {
  // Least-significant bit of the field in the 64-bit history word.
  uint8_t shift;
  // Number of bits occupied by the field, or zero when it is absent.
  uint8_t bit_count;
} loom_amdgpu_sanitizer_race_history_field_t;

typedef struct loom_amdgpu_sanitizer_race_history_layout_t {
  // Encoded access-kind field.
  loom_amdgpu_sanitizer_race_history_field_t access_kind;
  // Encoded linear workitem identifier field.
  loom_amdgpu_sanitizer_race_history_field_t workitem;
  // Encoded workgroup synchronization epoch field.
  loom_amdgpu_sanitizer_race_history_field_t workgroup_epoch;
  // Encoded subgroup synchronization epoch field, when present.
  loom_amdgpu_sanitizer_race_history_field_t subgroup_epoch;
  // Encoded low dispatch-generation field.
  loom_amdgpu_sanitizer_race_history_field_t generation;
  // Encoded sanitizer-site identifier field.
  loom_amdgpu_sanitizer_race_history_field_t site;
  // Module-table site ID added to a decoded function-local site ordinal.
  uint32_t site_id_base;
  // Nonzero record-format identity, or zero for the legacy layout.
  uint32_t format_tag;
  // Log2 hardware wavefront width used to derive subgroup identity.
  uint8_t wavefront_size_log2;
} loom_amdgpu_sanitizer_race_history_layout_t;

// Returns the exact history layout used by workgroup-only instrumentation.
const loom_amdgpu_sanitizer_race_history_layout_t*
loom_amdgpu_sanitizer_race_legacy_history_layout(void);

// Returns the 64-bit mask selecting |field|.
uint64_t loom_amdgpu_sanitizer_race_history_field_mask(
    loom_amdgpu_sanitizer_race_history_field_t field);

// Returns the portions of |field| stored in each 32-bit history lane.
uint32_t loom_amdgpu_sanitizer_race_history_field_low_mask(
    loom_amdgpu_sanitizer_race_history_field_t field);
uint32_t loom_amdgpu_sanitizer_race_history_field_high_mask(
    loom_amdgpu_sanitizer_race_history_field_t field);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_HISTORY_LAYOUT_H_
