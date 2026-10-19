// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Packed AMDGPU TSAN access-history layout.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_HISTORY_LAYOUT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_HISTORY_LAYOUT_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"

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

enum loom_amdgpu_sanitizer_race_history_layout_constant_e {
  // Minimum subgroup epochs distinguished within one workgroup epoch.
  LOOM_AMDGPU_SANITIZER_RACE_MIN_SUBGROUP_EPOCH_BIT_COUNT = 20u,
};

typedef struct loom_amdgpu_sanitizer_race_history_layout_config_t {
  // Fixed flattened workgroup size.
  uint32_t flat_workgroup_size;
  // Number of densely assigned sanitizer sites in the source function.
  uint32_t function_site_count;
  // Module-table ID assigned to the first source-function site.
  uint32_t function_site_id_base;
  // Hardware wavefront width defining one subgroup.
  uint32_t wavefront_size;
} loom_amdgpu_sanitizer_race_history_layout_config_t;

typedef enum loom_amdgpu_sanitizer_race_history_layout_support_e {
  // The complete scope-aware history layout is representable.
  LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_SUPPORTED = 0,
  // The fixed workgroup has more workitems than histories can encode.
  LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_WORKITEM_CAPACITY_EXCEEDED,
  // The source function has more sites than histories can encode.
  LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_SITE_CAPACITY_EXCEEDED,
  // The target wavefront width cannot define packed subgroup identity.
  LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_WAVEFRONT_UNSUPPORTED,
} loom_amdgpu_sanitizer_race_history_layout_support_t;

// Returns the exact history layout used by workgroup-only instrumentation.
const loom_amdgpu_sanitizer_race_history_layout_t*
loom_amdgpu_sanitizer_race_legacy_history_layout(void);

// Selects a scope-aware packed layout for a function containing subgroup
// synchronization. The layout always retains at least 20 subgroup-epoch bits;
// fixed workitem and function-local site facts donate every remaining bit to
// dispatch generation first and then to subgroup epoch capacity.
loom_amdgpu_sanitizer_race_history_layout_support_t
loom_amdgpu_sanitizer_race_select_subgroup_history_layout(
    const loom_amdgpu_sanitizer_race_history_layout_config_t* config,
    loom_amdgpu_sanitizer_race_history_layout_t* out_layout);

// Returns whether |layout| carries subgroup synchronization epochs.
static inline bool loom_amdgpu_sanitizer_race_history_layout_has_subgroups(
    const loom_amdgpu_sanitizer_race_history_layout_t* layout) {
  return layout->subgroup_epoch.bit_count != 0;
}

// Returns the 64-bit mask selecting |field|.
uint64_t loom_amdgpu_sanitizer_race_history_field_mask(
    loom_amdgpu_sanitizer_race_history_field_t field);

// Returns the portions of |field| stored in each 32-bit history lane.
uint32_t loom_amdgpu_sanitizer_race_history_field_low_mask(
    loom_amdgpu_sanitizer_race_history_field_t field);
uint32_t loom_amdgpu_sanitizer_race_history_field_high_mask(
    loom_amdgpu_sanitizer_race_history_field_t field);

// Inserts one VGPR u32 value into |field| of the split 64-bit history. Existing
// bits in |inout_low| and |inout_high| must not overlap the field.
iree_status_t loom_amdgpu_sanitizer_race_history_insert_vgpr_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_history_field_t field, loom_value_id_t value,
    loom_value_id_t* inout_low, loom_value_id_t* inout_high);

// Extracts |field| from the split 64-bit history as a VGPR u32 value.
iree_status_t loom_amdgpu_sanitizer_race_history_extract_vgpr_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_history_field_t field, loom_value_id_t low,
    loom_value_id_t high, loom_value_id_t* out_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SANITIZER_RACE_HISTORY_LAYOUT_H_
