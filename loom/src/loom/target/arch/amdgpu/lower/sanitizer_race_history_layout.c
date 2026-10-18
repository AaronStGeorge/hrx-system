// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/sanitizer_race_history_layout.h"

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

enum loom_amdgpu_sanitizer_race_history_format_e {
  // Packed format version occupying the high 11 bits of the format tag.
  LOOM_AMDGPU_SANITIZER_RACE_HISTORY_FORMAT_VERSION = 1u,
  LOOM_AMDGPU_SANITIZER_RACE_HISTORY_FORMAT_VERSION_SHIFT = 21u,
};

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

static uint8_t loom_amdgpu_sanitizer_race_required_bit_count(uint32_t count) {
  IREE_ASSERT_NE(count, 0u);
  return count == 1
             ? 0
             : (uint8_t)(32u - iree_math_count_leading_zeros_u32(count - 1u));
}

loom_amdgpu_sanitizer_race_history_layout_support_t
loom_amdgpu_sanitizer_race_select_subgroup_history_layout(
    const loom_amdgpu_sanitizer_race_history_layout_config_t* config,
    loom_amdgpu_sanitizer_race_history_layout_t* out_layout) {
  *out_layout = (loom_amdgpu_sanitizer_race_history_layout_t){0};
  IREE_ASSERT_NE(config->flat_workgroup_size, 0u);
  IREE_ASSERT_NE(config->function_site_count, 0u);

  const uint8_t workitem_bit_count =
      loom_amdgpu_sanitizer_race_required_bit_count(
          config->flat_workgroup_size);
  if (workitem_bit_count > LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_BIT_COUNT) {
    return LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_WORKITEM_CAPACITY_EXCEEDED;
  }
  const uint8_t site_bit_count = loom_amdgpu_sanitizer_race_required_bit_count(
      config->function_site_count);
  if (site_bit_count > LOOM_AMDGPU_TSAN_SHADOW_ENTRY_SITE_BIT_COUNT) {
    return LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_SITE_CAPACITY_EXCEEDED;
  }
  if (config->wavefront_size != 32u && config->wavefront_size != 64u) {
    return LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_WAVEFRONT_UNSUPPORTED;
  }

  const uint8_t fixed_bit_count =
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_BIT_COUNT +
      LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_BIT_COUNT + workitem_bit_count +
      site_bit_count + LOOM_AMDGPU_SANITIZER_RACE_MIN_SUBGROUP_EPOCH_BIT_COUNT;
  IREE_ASSERT_LE(fixed_bit_count, 64u);
  const uint8_t remaining_bit_count = 64u - fixed_bit_count;
  const uint8_t generation_bit_count = (uint8_t)iree_min(
      remaining_bit_count, LOOM_AMDGPU_TSAN_SHADOW_ENTRY_GENERATION_BIT_COUNT);
  const uint8_t subgroup_epoch_bit_count =
      LOOM_AMDGPU_SANITIZER_RACE_MIN_SUBGROUP_EPOCH_BIT_COUNT +
      remaining_bit_count - generation_bit_count;

  uint8_t shift = 0;
  const loom_amdgpu_sanitizer_race_history_field_t access_kind = {
      .shift = shift,
      .bit_count = LOOM_AMDGPU_TSAN_SHADOW_ENTRY_ACCESS_KIND_BIT_COUNT,
  };
  shift += access_kind.bit_count;
  const loom_amdgpu_sanitizer_race_history_field_t workitem = {
      .shift = shift,
      .bit_count = workitem_bit_count,
  };
  shift += workitem.bit_count;
  const loom_amdgpu_sanitizer_race_history_field_t workgroup_epoch = {
      .shift = shift,
      .bit_count = LOOM_AMDGPU_TSAN_SHADOW_ENTRY_EPOCH_BIT_COUNT,
  };
  shift += workgroup_epoch.bit_count;
  const loom_amdgpu_sanitizer_race_history_field_t subgroup_epoch = {
      .shift = shift,
      .bit_count = subgroup_epoch_bit_count,
  };
  shift += subgroup_epoch.bit_count;
  const loom_amdgpu_sanitizer_race_history_field_t generation = {
      .shift = shift,
      .bit_count = generation_bit_count,
  };
  shift += generation.bit_count;
  const loom_amdgpu_sanitizer_race_history_field_t site = {
      .shift = shift,
      .bit_count = site_bit_count,
  };
  shift += site.bit_count;
  IREE_ASSERT_EQ(shift, 64u);

  const uint8_t wavefront_size_log2 = config->wavefront_size == 32u ? 5u : 6u;
  const uint32_t format_payload = (uint32_t)workitem_bit_count |
                                  ((uint32_t)site_bit_count << 4u) |
                                  ((uint32_t)subgroup_epoch_bit_count << 9u) |
                                  ((uint32_t)generation_bit_count << 15u) |
                                  ((uint32_t)(wavefront_size_log2 - 5u) << 20u);
  *out_layout = (loom_amdgpu_sanitizer_race_history_layout_t){
      .access_kind = access_kind,
      .workitem = workitem,
      .workgroup_epoch = workgroup_epoch,
      .subgroup_epoch = subgroup_epoch,
      .generation = generation,
      .site = site,
      .site_id_base = config->function_site_id_base,
      .format_tag =
          ((uint32_t)LOOM_AMDGPU_SANITIZER_RACE_HISTORY_FORMAT_VERSION
           << LOOM_AMDGPU_SANITIZER_RACE_HISTORY_FORMAT_VERSION_SHIFT) |
          format_payload,
      .wavefront_size_log2 = wavefront_size_log2,
  };
  return LOOM_AMDGPU_SANITIZER_RACE_HISTORY_LAYOUT_SUPPORTED;
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

static uint32_t loom_amdgpu_sanitizer_race_low_bit_mask(uint32_t bit_count) {
  return bit_count == 32u ? UINT32_MAX : (1u << bit_count) - 1u;
}

static iree_status_t loom_amdgpu_sanitizer_race_history_insert_piece(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, uint32_t source_shift, uint32_t bit_count,
    uint32_t destination_shift, loom_type_t vgpr_type,
    loom_value_id_t* inout_destination) {
  loom_value_id_t piece = value;
  if (source_shift != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        source_shift, piece, vgpr_type, &piece));
  }
  if (bit_count != 32u) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, piece,
        loom_amdgpu_sanitizer_race_low_bit_mask(bit_count), vgpr_type, &piece));
  }
  if (destination_shift != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        destination_shift, piece, vgpr_type, &piece));
  }
  return loom_amdgpu_emit_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
      *inout_destination, piece, vgpr_type, inout_destination);
}

iree_status_t loom_amdgpu_sanitizer_race_history_insert_vgpr_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_history_field_t field, loom_value_id_t value,
    loom_value_id_t* inout_low, loom_value_id_t* inout_high) {
  if (field.bit_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_LE(field.bit_count, 32u);
  IREE_ASSERT_LE((uint32_t)field.shift + field.bit_count, 64u);
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  const uint32_t low_bit_count =
      field.shift < 32u ? iree_min((uint32_t)field.bit_count, 32u - field.shift)
                        : 0u;
  if (low_bit_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_history_insert_piece(
        context, source_op, value, /*source_shift=*/0, low_bit_count,
        field.shift, vgpr_type, inout_low));
  }
  const uint32_t high_bit_count = field.bit_count - low_bit_count;
  if (high_bit_count != 0) {
    const uint32_t high_shift =
        field.shift < 32u ? 0u : (uint32_t)field.shift - 32u;
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_history_insert_piece(
        context, source_op, value, low_bit_count, high_bit_count, high_shift,
        vgpr_type, inout_high));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_sanitizer_race_history_extract_piece(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, uint32_t source_shift, uint32_t bit_count,
    uint32_t destination_shift, loom_type_t vgpr_type,
    loom_value_id_t* out_piece) {
  loom_value_id_t piece = source;
  if (source_shift != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        source_shift, piece, vgpr_type, &piece));
  }
  if (bit_count != 32u) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, piece,
        loom_amdgpu_sanitizer_race_low_bit_mask(bit_count), vgpr_type, &piece));
  }
  if (destination_shift != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        destination_shift, piece, vgpr_type, &piece));
  }
  *out_piece = piece;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_sanitizer_race_history_extract_vgpr_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_history_field_t field, loom_value_id_t low,
    loom_value_id_t high, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_NE(field.bit_count, 0u);
  IREE_ASSERT_LE(field.bit_count, 32u);
  IREE_ASSERT_LE((uint32_t)field.shift + field.bit_count, 64u);
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));

  const uint32_t low_bit_count =
      field.shift < 32u ? iree_min((uint32_t)field.bit_count, 32u - field.shift)
                        : 0u;
  loom_value_id_t low_piece = LOOM_VALUE_ID_INVALID;
  if (low_bit_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_history_extract_piece(
        context, source_op, low, field.shift, low_bit_count,
        /*destination_shift=*/0, vgpr_type, &low_piece));
  }
  const uint32_t high_bit_count = field.bit_count - low_bit_count;
  if (high_bit_count == 0) {
    *out_value = low_piece;
    return iree_ok_status();
  }
  loom_value_id_t high_piece = LOOM_VALUE_ID_INVALID;
  const uint32_t high_shift =
      field.shift < 32u ? 0u : (uint32_t)field.shift - 32u;
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_race_history_extract_piece(
      context, source_op, high, high_shift, high_bit_count, low_bit_count,
      vgpr_type, &high_piece));
  if (low_bit_count == 0) {
    *out_value = high_piece;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_binary(context, source_op,
                                 LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_piece,
                                 high_piece, vgpr_type, out_value);
}
