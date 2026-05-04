// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/matrix_contract.h"

#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/arch/amdgpu/target_refs.h"

#define MATRIX_PAYLOAD(numeric_type_value, register_count_value, \
                       element_count_value)                      \
  (loom_amdgpu_matrix_payload_shape_t) {                         \
    .numeric_type = (numeric_type_value),                        \
    .register_count = (register_count_value),                    \
    .element_count = (element_count_value),                      \
  }

#define MATRIX_TILE_SHAPE(row_count_value, column_count_value, \
                          reduction_count_value)               \
  (loom_amdgpu_matrix_tile_shape_t) {                          \
    .result_row_count = (row_count_value),                     \
    .result_column_count = (column_count_value),               \
    .reduction_count = (reduction_count_value),                \
  }

#define MATRIX_ROLE_LAYOUT(role_value, map_kind_value, register_count_value, \
                           elements_per_register_value,                      \
                           element_bit_count_value, coordinate_flags_value)  \
  (loom_amdgpu_matrix_fragment_role_layout_t) {                              \
    .role = (role_value), .map_kind = (map_kind_value),                      \
    .register_count = (register_count_value),                                \
    .elements_per_register = (elements_per_register_value),                  \
    .element_bit_count = (element_bit_count_value),                          \
    .coordinate_flags = (coordinate_flags_value),                            \
  }

#define MATRIX_DESCRIPTOR_WITH_LOW_ID_AND_LAYOUT(                             \
    name_value, low_descriptor_ref_value, intrinsic_name_value, family_value, \
    feature_bits_value, wave_bits_value, flags_value, row_count_value,        \
    column_count_value, reduction_count_value, lhs_type_value,                \
    lhs_registers_value, lhs_elements_value, rhs_type_value,                  \
    rhs_registers_value, rhs_elements_value, accumulator_type_value,          \
    accumulator_registers_value, accumulator_elements_value,                  \
    result_type_value, result_registers_value, result_elements_value,         \
    scale_kind_value, fragment_layout_kind_value)                             \
  {                                                                           \
      .name = IREE_SVL(name_value),                                           \
      .low_descriptor_ref = (low_descriptor_ref_value),                       \
      .llvm_intrinsic_name = IREE_SVL(intrinsic_name_value),                  \
      .family = (family_value),                                               \
      .required_feature_bits = (feature_bits_value),                          \
      .wave_size_bits = (wave_bits_value),                                    \
      .flags = (flags_value),                                                 \
      .tile_shape = MATRIX_TILE_SHAPE(row_count_value, column_count_value,    \
                                      reduction_count_value),                 \
      .lhs_payload = MATRIX_PAYLOAD(lhs_type_value, lhs_registers_value,      \
                                    lhs_elements_value),                      \
      .rhs_payload = MATRIX_PAYLOAD(rhs_type_value, rhs_registers_value,      \
                                    rhs_elements_value),                      \
      .accumulator_payload =                                                  \
          MATRIX_PAYLOAD(accumulator_type_value, accumulator_registers_value, \
                         accumulator_elements_value),                         \
      .result_payload = MATRIX_PAYLOAD(                                       \
          result_type_value, result_registers_value, result_elements_value),  \
      .scale_kind = (scale_kind_value),                                       \
      .fragment_layout_kind = (fragment_layout_kind_value),                   \
  }

#define MATRIX_DESCRIPTOR_WITH_LOW_ID(                                         \
    name_value, low_descriptor_ref_value, intrinsic_name_value, family_value,  \
    feature_bits_value, wave_bits_value, flags_value, row_count_value,         \
    column_count_value, reduction_count_value, lhs_type_value,                 \
    lhs_registers_value, lhs_elements_value, rhs_type_value,                   \
    rhs_registers_value, rhs_elements_value, accumulator_type_value,           \
    accumulator_registers_value, accumulator_elements_value,                   \
    result_type_value, result_registers_value, result_elements_value,          \
    scale_kind_value)                                                          \
  MATRIX_DESCRIPTOR_WITH_LOW_ID_AND_LAYOUT(                                    \
      name_value, low_descriptor_ref_value, intrinsic_name_value,              \
      family_value, feature_bits_value, wave_bits_value, flags_value,          \
      row_count_value, column_count_value, reduction_count_value,              \
      lhs_type_value, lhs_registers_value, lhs_elements_value, rhs_type_value, \
      rhs_registers_value, rhs_elements_value, accumulator_type_value,         \
      accumulator_registers_value, accumulator_elements_value,                 \
      result_type_value, result_registers_value, result_elements_value,        \
      scale_kind_value, LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN)

static const loom_amdgpu_matrix_fragment_layout_t kMatrixFragmentLayouts[] = {
    [LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_F16] =
        {
            .kind =
                LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_F16,
            .name = IREE_SVL("rdna3.wmmar3.f32.16x16x16.f16"),
            .wave_size = 32,
            .tile_shape = MATRIX_TILE_SHAPE(16, 16, 16),
            .lhs = MATRIX_ROLE_LAYOUT(
                LOOM_AMDGPU_MATRIX_OPERAND_ROLE_LHS,
                LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION,
                8, 2, 16,
                LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_ROW |
                    LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_REDUCTION),
            .rhs = MATRIX_ROLE_LAYOUT(
                LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RHS,
                LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION,
                8, 2, 16,
                LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_COLUMN |
                    LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_REDUCTION),
            .accumulator = MATRIX_ROLE_LAYOUT(
                LOOM_AMDGPU_MATRIX_OPERAND_ROLE_ACCUMULATOR,
                LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN,
                8, 1, 32,
                LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_ROW |
                    LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_COLUMN),
            .result = MATRIX_ROLE_LAYOUT(
                LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RESULT,
                LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN,
                8, 1, 32,
                LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_ROW |
                    LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_COLUMN),
        },
};

#define MATRIX_DESCRIPTOR(                                                     \
    name_value, intrinsic_name_value, family_value, feature_bits_value,        \
    wave_bits_value, flags_value, row_count_value, column_count_value,         \
    reduction_count_value, lhs_type_value, lhs_registers_value,                \
    lhs_elements_value, rhs_type_value, rhs_registers_value,                   \
    rhs_elements_value, accumulator_type_value, accumulator_registers_value,   \
    accumulator_elements_value, result_type_value, result_registers_value,     \
    result_elements_value, scale_kind_value)                                   \
  MATRIX_DESCRIPTOR_WITH_LOW_ID(                                               \
      name_value, LOOM_AMDGPU_MATRIX_LOW_DESCRIPTOR_REF_NONE,                  \
      intrinsic_name_value, family_value, feature_bits_value, wave_bits_value, \
      flags_value, row_count_value, column_count_value, reduction_count_value, \
      lhs_type_value, lhs_registers_value, lhs_elements_value, rhs_type_value, \
      rhs_registers_value, rhs_elements_value, accumulator_type_value,         \
      accumulator_registers_value, accumulator_elements_value,                 \
      result_type_value, result_registers_value, result_elements_value,        \
      scale_kind_value)

#define MFMA_GFX940_FP8_FEATURES (LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8)

#define MFMA_GFX950_FEATURES (LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950)

#define MFMA_GFX950_SCALE_FEATURES \
  (MFMA_GFX950_FEATURES | LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4)

#define SMFMAC_GFX940_FEATURES (LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940)

#define SMFMAC_GFX950_FEATURES (LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX950)

#define WMMA_GFX12_FEATURES (LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12)

#define WMMA_GFX1250_FEATURES (LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250)

#define WMMA_GFX1250_SCALE_FEATURES \
  (WMMA_GFX1250_FEATURES | LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250_SCALE_F8F6F4)

#define SWMMAC_GFX12_FEATURES (LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12)

#define SWMMAC_GFX1250_FEATURES (LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX1250)

#define SWMMAC_GFX12_IU_FLAGS                     \
  (LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |      \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT | \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP)

#define WMMA_GFX1250_MODS_ALL_FLAGS                \
  (LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS | \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER |   \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE)

#define WMMA_GFX1250_MODS_C_FLAGS                \
  (LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER | \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE)

#define WMMA_GFX1250_SCALE_F4_FLAGS                 \
  (LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |        \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER |    \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS | \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE |         \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK)

#define SWMMAC_GFX1250_AB_FLAGS                    \
  (LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |       \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS | \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE)

#define SWMMAC_GFX1250_IU8_FLAGS                   \
  (LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |       \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT |  \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS | \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE |        \
   LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP)

static const loom_amdgpu_matrix_contract_descriptor_t
    kMatrixContractDescriptors[] = {
        MATRIX_DESCRIPTOR("mfma.f32.16x16x1.f32",
                          "llvm.amdgcn.mfma.f32.16x16x1.f32",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 1, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 1, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.16x16x2.bf16",
                          "llvm.amdgcn.mfma.f32.16x16x2.bf16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 1, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 1, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.16x16x4.f16",
                          "llvm.amdgcn.mfma.f32.16x16x4.f16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.16x16x4.f32",
                          "llvm.amdgcn.mfma.f32.16x16x4.f32",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 1, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 1, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.16x16x8.bf16",
                          "llvm.amdgcn.mfma.f32.16x16x8.bf16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 1, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 1, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.16x16x16.f16",
                          "llvm.amdgcn.mfma.f32.16x16x16.f16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.16x16x16.bf16.1k",
                          "llvm.amdgcn.mfma.f32.16x16x16.bf16.1k",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.16x16x32.bf16",
                          "llvm.amdgcn.mfma.f32.16x16x32.bf16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA, MFMA_GFX950_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.16x16x32.bf8.bf8",
                          "llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          MFMA_GFX940_FP8_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.16x16x32.bf8.fp8",
                          "llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          MFMA_GFX940_FP8_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.16x16x32.f16",
                          "llvm.amdgcn.mfma.f32.16x16x32.f16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA, MFMA_GFX950_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.16x16x32.fp8.bf8",
                          "llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          MFMA_GFX940_FP8_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.16x16x32.fp8.fp8",
                          "llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          MFMA_GFX940_FP8_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.16x16x4.bf16.1k",
                          "llvm.amdgcn.mfma.f32.16x16x4.bf16.1k",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "mfma.f32.16x16x8.xf32", "llvm.amdgcn.mfma.f32.16x16x8.xf32",
            LOOM_AMDGPU_MATRIX_FAMILY_MFMA, MFMA_GFX940_FP8_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_XF32, 2, 2,
            LOOM_AMDGPU_MATRIX_NUMERIC_XF32, 2, 2,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x1.f32",
                          "llvm.amdgcn.mfma.f32.32x32x1.f32",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 1, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 1, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 32, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 32, 32,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x2.bf16",
                          "llvm.amdgcn.mfma.f32.32x32x2.bf16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 1, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 1, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 32, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 32, 32,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x2.f32",
                          "llvm.amdgcn.mfma.f32.32x32x2.f32",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 1, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 1, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x4.bf16",
                          "llvm.amdgcn.mfma.f32.32x32x4.bf16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 1, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 1, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x4.f16",
                          "llvm.amdgcn.mfma.f32.32x32x4.f16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 32, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 32, 32,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x8.f16",
                          "llvm.amdgcn.mfma.f32.32x32x8.f16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x16.bf16",
                          "llvm.amdgcn.mfma.f32.32x32x16.bf16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA, MFMA_GFX950_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x16.bf8.bf8",
                          "llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          MFMA_GFX940_FP8_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x16.bf8.fp8",
                          "llvm.amdgcn.mfma.f32.32x32x16.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          MFMA_GFX940_FP8_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x16.f16",
                          "llvm.amdgcn.mfma.f32.32x32x16.f16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA, MFMA_GFX950_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x16.fp8.bf8",
                          "llvm.amdgcn.mfma.f32.32x32x16.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          MFMA_GFX940_FP8_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x16.fp8.fp8",
                          "llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          MFMA_GFX940_FP8_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x4.bf16.1k",
                          "llvm.amdgcn.mfma.f32.32x32x4.bf16.1k",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 32, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 32, 32,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.32x32x8.bf16.1k",
                          "llvm.amdgcn.mfma.f32.32x32x8.bf16.1k",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "mfma.f32.32x32x4.xf32", "llvm.amdgcn.mfma.f32.32x32x4.xf32",
            LOOM_AMDGPU_MATRIX_FAMILY_MFMA, MFMA_GFX940_FP8_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_XF32, 2, 2,
            LOOM_AMDGPU_MATRIX_NUMERIC_XF32, 2, 2,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.4x4x1.f32",
                          "llvm.amdgcn.mfma.f32.4x4x1.f32",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 4, 4, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 1, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 1, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.4x4x2.bf16",
                          "llvm.amdgcn.mfma.f32.4x4x2.bf16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 4, 4, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 1, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 1, 2,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.4x4x4.f16",
                          "llvm.amdgcn.mfma.f32.4x4x4.f16",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 4, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f32.4x4x4.bf16.1k",
                          "llvm.amdgcn.mfma.f32.4x4x4.bf16.1k",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 4, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 2, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f64.16x16x4.f64",
                          "llvm.amdgcn.mfma.f64.16x16x4.f64",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F64, 2, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F64, 2, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F64, 8, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F64, 8, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.f64.4x4x4.f64",
                          "llvm.amdgcn.mfma.f64.4x4x4.f64",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 4, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F64, 2, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F64, 2, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F64, 2, 1,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F64, 2, 1,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("mfma.i32.4x4x4.i8", "llvm.amdgcn.mfma.i32.4x4x4.i8",
                          LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 4, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_I8, 1, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_I8, 1, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_I32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_I32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "mfma.i32.16x16x4.i8", "llvm.amdgcn.mfma.i32.16x16x4.i8",
            LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_I8, 1, 4, LOOM_AMDGPU_MATRIX_NUMERIC_I8,
            1, 4, LOOM_AMDGPU_MATRIX_NUMERIC_I32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "mfma.i32.16x16x16.i8", "llvm.amdgcn.mfma.i32.16x16x16.i8",
            LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_I8, 1, 4, LOOM_AMDGPU_MATRIX_NUMERIC_I8,
            1, 4, LOOM_AMDGPU_MATRIX_NUMERIC_I32, 4, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 4, 4,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "mfma.i32.16x16x32.i8", "llvm.amdgcn.mfma.i32.16x16x32.i8",
            LOOM_AMDGPU_MATRIX_FAMILY_MFMA, MFMA_GFX940_FP8_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_I8, 2, 8, LOOM_AMDGPU_MATRIX_NUMERIC_I8,
            2, 8, LOOM_AMDGPU_MATRIX_NUMERIC_I32, 4, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 4, 4,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "mfma.i32.16x16x64.i8", "llvm.amdgcn.mfma.i32.16x16x64.i8",
            LOOM_AMDGPU_MATRIX_FAMILY_MFMA, MFMA_GFX950_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 64,
            LOOM_AMDGPU_MATRIX_NUMERIC_I8, 4, 16, LOOM_AMDGPU_MATRIX_NUMERIC_I8,
            4, 16, LOOM_AMDGPU_MATRIX_NUMERIC_I32, 4, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 4, 4,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "mfma.i32.32x32x4.i8", "llvm.amdgcn.mfma.i32.32x32x4.i8",
            LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_I8, 1, 4, LOOM_AMDGPU_MATRIX_NUMERIC_I8,
            1, 4, LOOM_AMDGPU_MATRIX_NUMERIC_I32, 32, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 32, 32,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "mfma.i32.32x32x8.i8", "llvm.amdgcn.mfma.i32.32x32x8.i8",
            LOOM_AMDGPU_MATRIX_FAMILY_MFMA,
            LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_I8, 1, 4, LOOM_AMDGPU_MATRIX_NUMERIC_I8,
            1, 4, LOOM_AMDGPU_MATRIX_NUMERIC_I32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "mfma.i32.32x32x16.i8", "llvm.amdgcn.mfma.i32.32x32x16.i8",
            LOOM_AMDGPU_MATRIX_FAMILY_MFMA, MFMA_GFX940_FP8_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_I8, 2, 8, LOOM_AMDGPU_MATRIX_NUMERIC_I8,
            2, 8, LOOM_AMDGPU_MATRIX_NUMERIC_I32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "mfma.i32.32x32x32.i8", "llvm.amdgcn.mfma.i32.32x32x32.i8",
            LOOM_AMDGPU_MATRIX_FAMILY_MFMA, MFMA_GFX950_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 32, 32, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_I8, 4, 16, LOOM_AMDGPU_MATRIX_NUMERIC_I8,
            4, 16, LOOM_AMDGPU_MATRIX_NUMERIC_I32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "mfma.scale.f32.16x16x128.f8f6f4",
            "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4",
            LOOM_AMDGPU_MATRIX_FAMILY_MFMA, MFMA_GFX950_SCALE_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK,
            16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4, LOOM_AMDGPU_MATRIX_SCALE_32),
        MATRIX_DESCRIPTOR(
            "mfma.scale.f32.32x32x64.f8f6f4",
            "llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4",
            LOOM_AMDGPU_MATRIX_FAMILY_MFMA, MFMA_GFX950_SCALE_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK,
            32, 32, 64, LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_32),
        MATRIX_DESCRIPTOR("smfmac.f32.16x16x128.fp8.fp8",
                          "llvm.amdgcn.smfmac.f32.16x16x128.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX950_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 128,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.16x16x64.bf8.bf8",
                          "llvm.amdgcn.smfmac.f32.16x16x64.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX940_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.16x16x64.bf8.fp8",
                          "llvm.amdgcn.smfmac.f32.16x16x64.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX940_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "smfmac.f32.16x16x32.bf16", "llvm.amdgcn.smfmac.f32.16x16x32.bf16",
            LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC, SMFMAC_GFX940_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 2, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 4, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "smfmac.f32.16x16x32.f16", "llvm.amdgcn.smfmac.f32.16x16x32.f16",
            LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC, SMFMAC_GFX940_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 2, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "smfmac.f32.16x16x64.bf16", "llvm.amdgcn.smfmac.f32.16x16x64.bf16",
            LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC, SMFMAC_GFX950_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 64,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 4, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "smfmac.f32.16x16x64.f16", "llvm.amdgcn.smfmac.f32.16x16x64.f16",
            LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC, SMFMAC_GFX950_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 64,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.16x16x64.fp8.bf8",
                          "llvm.amdgcn.smfmac.f32.16x16x64.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX940_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.16x16x64.fp8.fp8",
                          "llvm.amdgcn.smfmac.f32.16x16x64.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX940_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.16x16x128.bf8.bf8",
                          "llvm.amdgcn.smfmac.f32.16x16x128.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX950_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 128,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.16x16x128.bf8.fp8",
                          "llvm.amdgcn.smfmac.f32.16x16x128.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX950_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 128,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.16x16x128.fp8.bf8",
                          "llvm.amdgcn.smfmac.f32.16x16x128.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX950_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 128,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 4, 4,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "smfmac.i32.16x16x64.i8", "llvm.amdgcn.smfmac.i32.16x16x64.i8",
            LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC, SMFMAC_GFX940_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 64,
            LOOM_AMDGPU_MATRIX_NUMERIC_I8, 2, 8, LOOM_AMDGPU_MATRIX_NUMERIC_I8,
            4, 16, LOOM_AMDGPU_MATRIX_NUMERIC_I32, 4, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 4, 4,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "smfmac.i32.16x16x128.i8", "llvm.amdgcn.smfmac.i32.16x16x128.i8",
            LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC, SMFMAC_GFX950_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 128,
            LOOM_AMDGPU_MATRIX_NUMERIC_I8, 4, 16, LOOM_AMDGPU_MATRIX_NUMERIC_I8,
            8, 32, LOOM_AMDGPU_MATRIX_NUMERIC_I32, 4, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 4, 4,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.32x32x64.fp8.fp8",
                          "llvm.amdgcn.smfmac.f32.32x32x64.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX950_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.32x32x32.bf8.bf8",
                          "llvm.amdgcn.smfmac.f32.32x32x32.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX940_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.32x32x32.bf8.fp8",
                          "llvm.amdgcn.smfmac.f32.32x32x32.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX940_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "smfmac.f32.32x32x16.bf16", "llvm.amdgcn.smfmac.f32.32x32x16.bf16",
            LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC, SMFMAC_GFX940_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 2, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 4, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "smfmac.f32.32x32x16.f16", "llvm.amdgcn.smfmac.f32.32x32x16.f16",
            LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC, SMFMAC_GFX940_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 2, 4,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "smfmac.f32.32x32x32.bf16", "llvm.amdgcn.smfmac.f32.32x32x32.bf16",
            LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC, SMFMAC_GFX950_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 4, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "smfmac.f32.32x32x32.f16", "llvm.amdgcn.smfmac.f32.32x32x32.f16",
            LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC, SMFMAC_GFX950_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.32x32x32.fp8.bf8",
                          "llvm.amdgcn.smfmac.f32.32x32x32.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX940_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "smfmac.i32.32x32x32.i8", "llvm.amdgcn.smfmac.i32.32x32x32.i8",
            LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC, SMFMAC_GFX940_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_I8, 2, 8, LOOM_AMDGPU_MATRIX_NUMERIC_I8,
            4, 16, LOOM_AMDGPU_MATRIX_NUMERIC_I32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "smfmac.i32.32x32x64.i8", "llvm.amdgcn.smfmac.i32.32x32x64.i8",
            LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC, SMFMAC_GFX950_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 64,
            LOOM_AMDGPU_MATRIX_NUMERIC_I8, 4, 16, LOOM_AMDGPU_MATRIX_NUMERIC_I8,
            8, 32, LOOM_AMDGPU_MATRIX_NUMERIC_I32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.32x32x32.fp8.fp8",
                          "llvm.amdgcn.smfmac.f32.32x32x32.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX940_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 2, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.32x32x64.bf8.bf8",
                          "llvm.amdgcn.smfmac.f32.32x32x64.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX950_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.32x32x64.bf8.fp8",
                          "llvm.amdgcn.smfmac.f32.32x32x64.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX950_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("smfmac.f32.32x32x64.fp8.bf8",
                          "llvm.amdgcn.smfmac.f32.32x32x64.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC,
                          SMFMAC_GFX950_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 32, 32, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 4, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("swmmac.f32.16x16x128.fp8.fp8",
                          "llvm.amdgcn.swmmac.f32.16x16x128.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC,
                          SWMMAC_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE,
                          16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("swmmac.f32.16x16x128.fp8.bf8",
                          "llvm.amdgcn.swmmac.f32.16x16x128.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC,
                          SWMMAC_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE,
                          16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("swmmac.f32.16x16x128.bf8.fp8",
                          "llvm.amdgcn.swmmac.f32.16x16x128.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC,
                          SWMMAC_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE,
                          16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("swmmac.f32.16x16x128.bf8.bf8",
                          "llvm.amdgcn.swmmac.f32.16x16x128.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC,
                          SWMMAC_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE,
                          16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("swmmac.f16.16x16x128.fp8.fp8",
                          "llvm.amdgcn.swmmac.f16.16x16x128.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC,
                          SWMMAC_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE,
                          16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("swmmac.f16.16x16x128.fp8.bf8",
                          "llvm.amdgcn.swmmac.f16.16x16x128.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC,
                          SWMMAC_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE,
                          16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("swmmac.f16.16x16x128.bf8.fp8",
                          "llvm.amdgcn.swmmac.f16.16x16x128.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC,
                          SWMMAC_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE,
                          16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("swmmac.f16.16x16x128.bf8.bf8",
                          "llvm.amdgcn.swmmac.f16.16x16x128.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC,
                          SWMMAC_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE,
                          16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "swmmac.f32.16x16x64.f16", "llvm.amdgcn.swmmac.f32.16x16x64.f16",
            LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC, SWMMAC_GFX1250_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, SWMMAC_GFX1250_AB_FLAGS, 16, 16,
            64, LOOM_AMDGPU_MATRIX_NUMERIC_F16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "swmmac.f32.16x16x64.bf16", "llvm.amdgcn.swmmac.f32.16x16x64.bf16",
            LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC, SWMMAC_GFX1250_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, SWMMAC_GFX1250_AB_FLAGS, 16, 16,
            64, LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "swmmac.f16.16x16x64.f16", "llvm.amdgcn.swmmac.f16.16x16x64.f16",
            LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC, SWMMAC_GFX1250_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, SWMMAC_GFX1250_AB_FLAGS, 16, 16,
            64, LOOM_AMDGPU_MATRIX_NUMERIC_F16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "swmmac.bf16.16x16x64.bf16",
            "llvm.amdgcn.swmmac.bf16.16x16x64.bf16",
            LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC, SWMMAC_GFX1250_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, SWMMAC_GFX1250_AB_FLAGS, 16, 16,
            64, LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 4, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 4, 8,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "swmmac.bf16f32.16x16x64.bf16",
            "llvm.amdgcn.swmmac.bf16f32.16x16x64.bf16",
            LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC, SWMMAC_GFX1250_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, SWMMAC_GFX1250_AB_FLAGS, 16, 16,
            64, LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "swmmac.i32.16x16x128.iu8", "llvm.amdgcn.swmmac.i32.16x16x128.iu8",
            LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC, SWMMAC_GFX1250_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, SWMMAC_GFX1250_IU8_FLAGS, 16, 16,
            128, LOOM_AMDGPU_MATRIX_NUMERIC_IU8, 8, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_IU8, 16, 64,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 8, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 8, 8,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "swmmac.f32.16x16x32.f16", "llvm.amdgcn.swmmac.f32.16x16x32.f16",
            LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC, SWMMAC_GFX12_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "swmmac.f32.16x16x32.bf16", "llvm.amdgcn.swmmac.f32.16x16x32.bf16",
            LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC, SWMMAC_GFX12_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "swmmac.f16.16x16x32.f16", "llvm.amdgcn.swmmac.f16.16x16x32.f16",
            LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC, SWMMAC_GFX12_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("swmmac.bf16.16x16x32.bf16",
                          "llvm.amdgcn.swmmac.bf16.16x16x32.bf16",
                          LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC,
                          SWMMAC_GFX12_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "swmmac.i32.16x16x32.iu8", "llvm.amdgcn.swmmac.i32.16x16x32.iu8",
            LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC, SWMMAC_GFX12_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, SWMMAC_GFX12_IU_FLAGS, 16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_IU8, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_IU8, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "swmmac.i32.16x16x32.iu4", "llvm.amdgcn.swmmac.i32.16x16x32.iu4",
            LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC, SWMMAC_GFX12_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, SWMMAC_GFX12_IU_FLAGS, 16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_IU4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_IU4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("swmmac.f32.16x16x32.fp8.fp8",
                          "llvm.amdgcn.swmmac.f32.16x16x32.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC,
                          SWMMAC_GFX12_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("swmmac.f32.16x16x32.bf8.bf8",
                          "llvm.amdgcn.swmmac.f32.16x16x32.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC,
                          SWMMAC_GFX12_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("swmmac.f32.16x16x32.bf8.fp8",
                          "llvm.amdgcn.swmmac.f32.16x16x32.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC,
                          SWMMAC_GFX12_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("swmmac.f32.16x16x32.fp8.bf8",
                          "llvm.amdgcn.swmmac.f32.16x16x32.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC,
                          SWMMAC_GFX12_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE, 16, 16, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "swmmac.i32.16x16x64.iu4", "llvm.amdgcn.swmmac.i32.16x16x64.iu4",
            LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC, SWMMAC_GFX12_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP,
            16, 16, 64, LOOM_AMDGPU_MATRIX_NUMERIC_IU4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_IU4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x128.f8f6f4",
                          "llvm.amdgcn.wmma.f32.16x16x128.f8f6f4",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER,
                          16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "wmma.f32.16x16x4.f32", "llvm.amdgcn.wmma.f32.16x16x4.f32",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, WMMA_GFX1250_MODS_ALL_FLAGS, 16,
            16, 4, LOOM_AMDGPU_MATRIX_NUMERIC_F32, 2, 2,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 2, 2,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "wmma.f32.16x16x32.bf16", "llvm.amdgcn.wmma.f32.16x16x32.bf16",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, WMMA_GFX1250_MODS_ALL_FLAGS, 16,
            16, 32, LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "wmma.f16.16x16x32.f16", "llvm.amdgcn.wmma.f16.16x16x32.f16",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, WMMA_GFX1250_MODS_ALL_FLAGS, 16,
            16, 32, LOOM_AMDGPU_MATRIX_NUMERIC_F16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "wmma.bf16.16x16x32.bf16", "llvm.amdgcn.wmma.bf16.16x16x32.bf16",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, WMMA_GFX1250_MODS_ALL_FLAGS, 16,
            16, 32, LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 4, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 4, 8,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.bf16f32.16x16x32.bf16",
                          "llvm.amdgcn.wmma.bf16f32.16x16x32.bf16",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_ALL_FLAGS, 16, 16, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 8, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 8, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 4, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x64.fp8.fp8",
                          "llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x64.fp8.bf8",
                          "llvm.amdgcn.wmma.f32.16x16x64.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x64.bf8.fp8",
                          "llvm.amdgcn.wmma.f32.16x16x64.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x64.bf8.bf8",
                          "llvm.amdgcn.wmma.f32.16x16x64.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f16.16x16x64.fp8.fp8",
                          "llvm.amdgcn.wmma.f16.16x16x64.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f16.16x16x64.fp8.bf8",
                          "llvm.amdgcn.wmma.f16.16x16x64.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f16.16x16x64.bf8.fp8",
                          "llvm.amdgcn.wmma.f16.16x16x64.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f16.16x16x64.bf8.bf8",
                          "llvm.amdgcn.wmma.f16.16x16x64.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f16.16x16x128.fp8.fp8",
                          "llvm.amdgcn.wmma.f16.16x16x128.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 128,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f16.16x16x128.fp8.bf8",
                          "llvm.amdgcn.wmma.f16.16x16x128.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 128,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f16.16x16x128.bf8.fp8",
                          "llvm.amdgcn.wmma.f16.16x16x128.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 128,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f16.16x16x128.bf8.bf8",
                          "llvm.amdgcn.wmma.f16.16x16x128.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 128,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 4, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x128.fp8.fp8",
                          "llvm.amdgcn.wmma.f32.16x16x128.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 128,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x128.fp8.bf8",
                          "llvm.amdgcn.wmma.f32.16x16x128.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 128,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x128.bf8.fp8",
                          "llvm.amdgcn.wmma.f32.16x16x128.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 128,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x128.bf8.bf8",
                          "llvm.amdgcn.wmma.f32.16x16x128.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          WMMA_GFX1250_MODS_C_FLAGS, 16, 16, 128,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 16, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.32x16x128.f4",
                          "llvm.amdgcn.wmma.f32.32x16x128.f4",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER, 32, 16,
                          128, LOOM_AMDGPU_MATRIX_NUMERIC_FP4, 16, 128,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP4, 8, 64,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "wmma.scale.f32.32x16x128.f4",
            "llvm.amdgcn.wmma.scale.f32.32x16x128.f4",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_SCALE_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, WMMA_GFX1250_SCALE_F4_FLAGS, 32,
            16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_FP4, 16, 128,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP4, 8, 64,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_32),
        MATRIX_DESCRIPTOR(
            "wmma.scale16.f32.32x16x128.f4",
            "llvm.amdgcn.wmma.scale16.f32.32x16x128.f4",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_SCALE_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, WMMA_GFX1250_SCALE_F4_FLAGS, 32,
            16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_FP4, 16, 128,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP4, 8, 64,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 16, 16,
            LOOM_AMDGPU_MATRIX_SCALE_16),
        MATRIX_DESCRIPTOR_WITH_LOW_ID(
            "wmma.f32.16x16x16.bf16",
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_BF16,
            "llvm.amdgcn.wmma.f32.16x16x16.bf16",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA,
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR_WITH_LOW_ID(
            "wmma.f16.16x16x16.f16",
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F16_16X16X16_F16,
            "llvm.amdgcn.wmma.f16.16x16x16.f16", LOOM_AMDGPU_MATRIX_FAMILY_WMMA,
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR_WITH_LOW_ID(
            "wmma.bf16.16x16x16.bf16",
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_BF16_16X16X16_BF16,
            "llvm.amdgcn.wmma.bf16.16x16x16.bf16",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA,
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR_WITH_LOW_ID(
            "wmma.f32.16x16x16.fp8.fp8",
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_FP8_FP8,
            "llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX12_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR_WITH_LOW_ID(
            "wmma.f32.16x16x16.bf8.bf8",
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_BF8_BF8,
            "llvm.amdgcn.wmma.f32.16x16x16.bf8.bf8",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX12_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR_WITH_LOW_ID(
            "wmma.f32.16x16x16.bf8.fp8",
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_BF8_FP8,
            "llvm.amdgcn.wmma.f32.16x16x16.bf8.fp8",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX12_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR_WITH_LOW_ID(
            "wmma.f32.16x16x16.fp8.bf8",
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_FP8_BF8,
            "llvm.amdgcn.wmma.f32.16x16x16.fp8.bf8",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX12_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR_WITH_LOW_ID_AND_LAYOUT(
            "wmma.f32.16x16x16.f16",
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X16_F16,
            "llvm.amdgcn.wmma.f32.16x16x16.f16", LOOM_AMDGPU_MATRIX_FAMILY_WMMA,
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_32, 0, 16, 16, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8, LOOM_AMDGPU_MATRIX_SCALE_NONE,
            LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_F16),
        MATRIX_DESCRIPTOR_WITH_LOW_ID(
            "wmma.i32.16x16x16.iu8",
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU8,
            "llvm.amdgcn.wmma.i32.16x16x16.iu8", LOOM_AMDGPU_MATRIX_FAMILY_WMMA,
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP,
            16, 16, 16, LOOM_AMDGPU_MATRIX_NUMERIC_IU8, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_IU8, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR_WITH_LOW_ID(
            "wmma.i32.16x16x16.iu4",
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X16_IU4,
            "llvm.amdgcn.wmma.i32.16x16x16.iu4", LOOM_AMDGPU_MATRIX_FAMILY_WMMA,
            LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP,
            16, 16, 16, LOOM_AMDGPU_MATRIX_NUMERIC_IU4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_IU4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR_WITH_LOW_ID(
            "wmma.f32.16x16x32.f16",
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_F32_16X16X32_F16,
            "llvm.amdgcn.wmma.f32.16x16x32.f16", LOOM_AMDGPU_MATRIX_FAMILY_WMMA,
            WMMA_GFX1250_FEATURES, LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            WMMA_GFX1250_MODS_ALL_FLAGS, 16, 16, 32,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F16, 8, 16,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR_WITH_LOW_ID(
            "wmma.i32.16x16x32.iu4",
            LOOM_AMDGPU_DESCRIPTOR_REF_V_WMMA_I32_16X16X32_IU4,
            "llvm.amdgcn.wmma.i32.16x16x32.iu4", LOOM_AMDGPU_MATRIX_FAMILY_WMMA,
            WMMA_GFX12_FEATURES, LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP,
            16, 16, 32, LOOM_AMDGPU_MATRIX_NUMERIC_IU4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_IU4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
            LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.i32.16x16x64.iu8",
                          "llvm.amdgcn.wmma.i32.16x16x64.iu8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE,
                          16, 16, 64, LOOM_AMDGPU_MATRIX_NUMERIC_IU8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_IU8, 8, 32,
                          LOOM_AMDGPU_MATRIX_NUMERIC_I32, 8, 8,
                          LOOM_AMDGPU_MATRIX_NUMERIC_I32, 8, 8,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "wmma.scale.f32.16x16x128.f8f6f4",
            "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_SCALE_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK,
            16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8, LOOM_AMDGPU_MATRIX_SCALE_32),
        MATRIX_DESCRIPTOR(
            "wmma.scale16.f32.16x16x128.f8f6f4",
            "llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_SCALE_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK,
            16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 8, 8, LOOM_AMDGPU_MATRIX_SCALE_16),
};

iree_string_view_t loom_amdgpu_matrix_family_name(
    loom_amdgpu_matrix_family_t family) {
  switch (family) {
    case LOOM_AMDGPU_MATRIX_FAMILY_UNKNOWN:
      return IREE_SV("unknown");
    case LOOM_AMDGPU_MATRIX_FAMILY_MFMA:
      return IREE_SV("mfma");
    case LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC:
      return IREE_SV("smfmac");
    case LOOM_AMDGPU_MATRIX_FAMILY_WMMA:
      return IREE_SV("wmma");
    case LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC:
      return IREE_SV("swmmac");
  }
  return IREE_SV("unknown");
}

iree_string_view_t loom_amdgpu_matrix_numeric_type_name(
    loom_amdgpu_matrix_numeric_type_t numeric_type) {
  switch (numeric_type) {
    case LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN:
      return IREE_SV("unknown");
    case LOOM_AMDGPU_MATRIX_NUMERIC_F64:
      return IREE_SV("f64");
    case LOOM_AMDGPU_MATRIX_NUMERIC_F32:
      return IREE_SV("f32");
    case LOOM_AMDGPU_MATRIX_NUMERIC_F16:
      return IREE_SV("f16");
    case LOOM_AMDGPU_MATRIX_NUMERIC_BF16:
      return IREE_SV("bf16");
    case LOOM_AMDGPU_MATRIX_NUMERIC_XF32:
      return IREE_SV("xf32");
    case LOOM_AMDGPU_MATRIX_NUMERIC_I32:
      return IREE_SV("i32");
    case LOOM_AMDGPU_MATRIX_NUMERIC_I8:
      return IREE_SV("i8");
    case LOOM_AMDGPU_MATRIX_NUMERIC_IU8:
      return IREE_SV("iu8");
    case LOOM_AMDGPU_MATRIX_NUMERIC_I4:
      return IREE_SV("i4");
    case LOOM_AMDGPU_MATRIX_NUMERIC_IU4:
      return IREE_SV("iu4");
    case LOOM_AMDGPU_MATRIX_NUMERIC_FP8:
      return IREE_SV("fp8");
    case LOOM_AMDGPU_MATRIX_NUMERIC_BF8:
      return IREE_SV("bf8");
    case LOOM_AMDGPU_MATRIX_NUMERIC_FP6:
      return IREE_SV("fp6");
    case LOOM_AMDGPU_MATRIX_NUMERIC_BF6:
      return IREE_SV("bf6");
    case LOOM_AMDGPU_MATRIX_NUMERIC_FP4:
      return IREE_SV("fp4");
    case LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4:
      return IREE_SV("f8f6f4");
  }
  return IREE_SV("unknown");
}

iree_string_view_t loom_amdgpu_matrix_scale_kind_name(
    loom_amdgpu_matrix_scale_kind_t scale_kind) {
  switch (scale_kind) {
    case LOOM_AMDGPU_MATRIX_SCALE_NONE:
      return IREE_SV("none");
    case LOOM_AMDGPU_MATRIX_SCALE_32:
      return IREE_SV("scale32");
    case LOOM_AMDGPU_MATRIX_SCALE_16:
      return IREE_SV("scale16");
  }
  return IREE_SV("unknown");
}

const loom_amdgpu_matrix_fragment_layout_t*
loom_amdgpu_matrix_fragment_layout_for_kind(
    loom_amdgpu_matrix_fragment_layout_kind_t kind) {
  if (kind <= LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_UNKNOWN ||
      (iree_host_size_t)kind >= IREE_ARRAYSIZE(kMatrixFragmentLayouts)) {
    return NULL;
  }
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      &kMatrixFragmentLayouts[kind];
  return layout->kind == kind ? layout : NULL;
}

const loom_amdgpu_matrix_fragment_layout_t*
loom_amdgpu_matrix_contract_descriptor_fragment_layout(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor) {
  if (descriptor == NULL) {
    return NULL;
  }
  return loom_amdgpu_matrix_fragment_layout_for_kind(
      descriptor->fragment_layout_kind);
}

const loom_amdgpu_matrix_fragment_role_layout_t*
loom_amdgpu_matrix_fragment_role_layout(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_amdgpu_matrix_operand_role_t role) {
  if (layout == NULL) {
    return NULL;
  }
  switch (role) {
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_LHS:
      return &layout->lhs;
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RHS:
      return &layout->rhs;
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_ACCUMULATOR:
      return &layout->accumulator;
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RESULT:
      return &layout->result;
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_UNKNOWN:
    default:
      return NULL;
  }
}

static bool loom_amdgpu_matrix_fragment_coordinate_from_role_layout(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout, uint16_t lane,
    uint16_t register_index, uint16_t element_index,
    loom_amdgpu_matrix_fragment_coordinate_t* out_coordinate) {
  const loom_amdgpu_matrix_tile_shape_t tile_shape = layout->tile_shape;
  if (tile_shape.result_row_count == 0 || tile_shape.result_column_count == 0 ||
      tile_shape.reduction_count == 0) {
    return false;
  }

  out_coordinate->coordinate_flags = role_layout->coordinate_flags;
  switch (role_layout->map_kind) {
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION: {
      const uint32_t reduction =
          (uint32_t)register_index * role_layout->elements_per_register +
          element_index;
      if (reduction >= tile_shape.reduction_count) {
        return false;
      }
      out_coordinate->row = lane % tile_shape.result_row_count;
      out_coordinate->reduction = (uint16_t)reduction;
      return true;
    }
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION: {
      const uint32_t reduction =
          (uint32_t)register_index * role_layout->elements_per_register +
          element_index;
      if (reduction >= tile_shape.reduction_count) {
        return false;
      }
      out_coordinate->column = lane % tile_shape.result_column_count;
      out_coordinate->reduction = (uint16_t)reduction;
      return true;
    }
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN: {
      const uint32_t row =
          (uint32_t)register_index * 2u + lane / tile_shape.result_column_count;
      if (row >= tile_shape.result_row_count) {
        return false;
      }
      out_coordinate->row = (uint16_t)row;
      out_coordinate->column = lane % tile_shape.result_column_count;
      return true;
    }
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_UNKNOWN:
    default:
      return false;
  }
}

bool loom_amdgpu_matrix_fragment_coordinate(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_amdgpu_matrix_operand_role_t role, uint16_t lane,
    uint16_t register_index, uint16_t element_index,
    loom_amdgpu_matrix_fragment_coordinate_t* out_coordinate) {
  *out_coordinate = (loom_amdgpu_matrix_fragment_coordinate_t){0};
  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, role);
  if (layout == NULL || role_layout == NULL || layout->wave_size == 0 ||
      lane >= layout->wave_size ||
      register_index >= role_layout->register_count ||
      element_index >= role_layout->elements_per_register) {
    return false;
  }

  return loom_amdgpu_matrix_fragment_coordinate_from_role_layout(
      layout, role_layout, lane, register_index, element_index, out_coordinate);
}

bool loom_amdgpu_matrix_feature_bits_for_profile(
    loom_amdgpu_matrix_feature_profile_t profile,
    loom_amdgpu_matrix_feature_bits_t* out_feature_bits) {
  *out_feature_bits = 0;
  switch (profile) {
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908:
      *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908;
      return true;
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A:
      *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908 |
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K |
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64;
      return true;
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940:
      *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908 |
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K |
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64 |
                          MFMA_GFX940_FP8_FEATURES |
                          LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940;
      return true;
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950:
      *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908 |
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K |
                          LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64 |
                          MFMA_GFX940_FP8_FEATURES |
                          MFMA_GFX950_SCALE_FEATURES |
                          LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940 |
                          LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX950;
      return true;
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11:
      *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11;
      return true;
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12:
      *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11 |
                          WMMA_GFX12_FEATURES |
                          LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12;
      return true;
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250:
      *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11 |
                          WMMA_GFX12_FEATURES | WMMA_GFX1250_SCALE_FEATURES |
                          LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12 |
                          LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX1250;
      return true;
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE:
      break;
  }
  return false;
}

iree_status_t loom_amdgpu_matrix_feature_bits_for_processor(
    iree_string_view_t processor,
    loom_amdgpu_matrix_feature_bits_t* out_feature_bits) {
  processor = iree_string_view_trim(processor);
  const loom_amdgpu_processor_info_t* processor_info = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_lookup_processor(processor, &processor_info));
  if (loom_amdgpu_matrix_feature_bits_for_profile(
          processor_info->matrix_feature_profile, out_feature_bits)) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU processor '%.*s' has no matrix feature profile",
      (int)processor.size, processor.data);
}

iree_host_size_t loom_amdgpu_matrix_contract_descriptor_count(void) {
  return IREE_ARRAYSIZE(kMatrixContractDescriptors);
}

const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_contract_descriptor_at(iree_host_size_t index) {
  if (index >= IREE_ARRAYSIZE(kMatrixContractDescriptors)) {
    return NULL;
  }
  return &kMatrixContractDescriptors[index];
}

bool loom_amdgpu_matrix_contract_is_available(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_amdgpu_matrix_feature_bits_t feature_bits, uint32_t wave_size) {
  if (descriptor == NULL) {
    return false;
  }
  if ((feature_bits & descriptor->required_feature_bits) !=
      descriptor->required_feature_bits) {
    return false;
  }
  if (wave_size == 0) {
    return true;
  }
  loom_amdgpu_matrix_wave_size_bits_t wave_size_bits = 0;
  if (wave_size == 32) {
    wave_size_bits = LOOM_AMDGPU_MATRIX_WAVE_SIZE_32;
  } else if (wave_size == 64) {
    wave_size_bits = LOOM_AMDGPU_MATRIX_WAVE_SIZE_64;
  } else {
    return false;
  }
  return (descriptor->wave_size_bits & wave_size_bits) != 0;
}

static bool loom_amdgpu_matrix_contract_family_matches(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    const loom_amdgpu_matrix_contract_match_request_t* request) {
  return request->family == LOOM_AMDGPU_MATRIX_FAMILY_UNKNOWN ||
         descriptor->family == request->family;
}

static bool loom_amdgpu_matrix_contract_tile_shape_matches(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    const loom_amdgpu_matrix_contract_match_request_t* request) {
  return descriptor->tile_shape.result_row_count ==
             request->tile_shape.result_row_count &&
         descriptor->tile_shape.result_column_count ==
             request->tile_shape.result_column_count &&
         descriptor->tile_shape.reduction_count ==
             request->tile_shape.reduction_count;
}

static bool loom_amdgpu_matrix_contract_payload_matches(
    loom_amdgpu_matrix_payload_shape_t descriptor_payload,
    loom_amdgpu_matrix_payload_shape_t request_payload) {
  if (request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN) {
    return false;
  }
  bool numeric_type_matches =
      descriptor_payload.numeric_type == request_payload.numeric_type;
  if (descriptor_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4) {
    // Selector-family descriptors accept exact low-bit requests when the
    // request also proves that matrix-format selector operands are available.
    numeric_type_matches =
        request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_FP8 ||
        request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_BF8 ||
        request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_FP6 ||
        request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_BF6 ||
        request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_FP4 ||
        request_payload.numeric_type == LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4;
  }
  if (!numeric_type_matches) {
    return false;
  }
  if (request_payload.register_count != 0 &&
      descriptor_payload.register_count != 0 &&
      descriptor_payload.register_count != request_payload.register_count) {
    return false;
  }
  if (request_payload.element_count != 0 &&
      descriptor_payload.element_count != 0 &&
      descriptor_payload.element_count != request_payload.element_count) {
    return false;
  }
  return true;
}

static loom_amdgpu_matrix_contract_rejection_bits_t
loom_amdgpu_matrix_contract_payload_rejection_bits(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    const loom_amdgpu_matrix_contract_match_request_t* request) {
  loom_amdgpu_matrix_contract_rejection_bits_t rejection_bits =
      LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE;
  if (!loom_amdgpu_matrix_contract_payload_matches(descriptor->lhs_payload,
                                                   request->lhs_payload)) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_LHS_PAYLOAD;
  }
  if (!loom_amdgpu_matrix_contract_payload_matches(descriptor->rhs_payload,
                                                   request->rhs_payload)) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_RHS_PAYLOAD;
  }
  if (!loom_amdgpu_matrix_contract_payload_matches(
          descriptor->accumulator_payload, request->accumulator_payload)) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_ACCUMULATOR_PAYLOAD;
  }
  if (!loom_amdgpu_matrix_contract_payload_matches(descriptor->result_payload,
                                                   request->result_payload)) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_RESULT_PAYLOAD;
  }
  return rejection_bits;
}

static loom_amdgpu_matrix_contract_rejection_bits_t
loom_amdgpu_matrix_contract_missing_flag_rejection_bits(
    loom_amdgpu_matrix_contract_flags_t missing_flags) {
  loom_amdgpu_matrix_contract_rejection_bits_t rejection_bits =
      LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE;
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SPARSE;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SCALE;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS) != 0) {
    rejection_bits |=
        LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_MATRIX_FORMATS;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_REUSE;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_CLAMP;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SIGN_SELECT;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS) != 0) {
    rejection_bits |=
        LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_AB_MODIFIERS;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_C_MODIFIER;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_OPSEL) != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_OPSEL;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS) != 0) {
    rejection_bits |=
        LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_MISSING_SCALE_FORMATS;
  }
  if ((missing_flags & LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK) !=
      0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_REQUIRED_FLAGS;
  }
  return rejection_bits;
}

static loom_amdgpu_matrix_contract_rejection_bits_t
loom_amdgpu_matrix_contract_flag_rejection_bits(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    const loom_amdgpu_matrix_contract_match_request_t* request) {
  const loom_amdgpu_matrix_contract_flags_t required_abi_flags =
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_OPSEL |
      LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS;
  const loom_amdgpu_matrix_contract_flags_t missing_available_flags =
      descriptor->flags & required_abi_flags & ~request->available_flags;
  loom_amdgpu_matrix_contract_rejection_bits_t rejection_bits =
      loom_amdgpu_matrix_contract_missing_flag_rejection_bits(
          missing_available_flags);
  const loom_amdgpu_matrix_contract_flags_t missing_required_flags =
      request->required_flags & ~descriptor->flags;
  if (missing_required_flags != 0) {
    rejection_bits |= LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_REQUIRED_FLAGS;
    rejection_bits |= loom_amdgpu_matrix_contract_missing_flag_rejection_bits(
        missing_required_flags);
  }
  return rejection_bits;
}

const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_contract_select(
    const loom_amdgpu_matrix_contract_match_request_t* request,
    loom_amdgpu_matrix_contract_match_diagnostic_t* out_diagnostic) {
  loom_amdgpu_matrix_contract_match_diagnostic_t diagnostic = {
      .descriptor_count = IREE_ARRAYSIZE(kMatrixContractDescriptors),
  };
  if (request == NULL) {
    diagnostic.rejection_bits =
        LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_INVALID_REQUEST;
    if (out_diagnostic != NULL) *out_diagnostic = diagnostic;
    return NULL;
  }

  loom_amdgpu_matrix_contract_rejection_bits_t payload_rejection_bits =
      LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE;
  loom_amdgpu_matrix_contract_rejection_bits_t flag_rejection_bits =
      LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kMatrixContractDescriptors);
       ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        &kMatrixContractDescriptors[i];
    if (!loom_amdgpu_matrix_contract_family_matches(descriptor, request)) {
      continue;
    }
    ++diagnostic.family_candidate_count;

    if (!loom_amdgpu_matrix_contract_tile_shape_matches(descriptor, request)) {
      continue;
    }
    ++diagnostic.shape_candidate_count;

    const loom_amdgpu_matrix_contract_rejection_bits_t payload_rejection =
        loom_amdgpu_matrix_contract_payload_rejection_bits(descriptor, request);
    if (payload_rejection != LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE) {
      payload_rejection_bits |= payload_rejection;
      continue;
    }
    ++diagnostic.payload_candidate_count;

    if (descriptor->scale_kind != request->scale_kind) {
      continue;
    }
    ++diagnostic.scale_candidate_count;

    const loom_amdgpu_matrix_contract_rejection_bits_t flag_rejection =
        loom_amdgpu_matrix_contract_flag_rejection_bits(descriptor, request);
    if (flag_rejection != LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_NONE) {
      flag_rejection_bits |= flag_rejection;
      continue;
    }
    ++diagnostic.flag_candidate_count;

    if ((request->feature_bits & descriptor->required_feature_bits) !=
        descriptor->required_feature_bits) {
      continue;
    }
    ++diagnostic.feature_candidate_count;

    if (!loom_amdgpu_matrix_contract_is_available(
            descriptor, request->feature_bits, request->wave_size)) {
      continue;
    }
    ++diagnostic.wave_candidate_count;

    if (out_diagnostic != NULL) *out_diagnostic = diagnostic;
    return descriptor;
  }

  if (diagnostic.family_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FAMILY;
  } else if (diagnostic.shape_candidate_count == 0) {
    diagnostic.rejection_bits =
        LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_TILE_SHAPE;
  } else if (diagnostic.payload_candidate_count == 0) {
    diagnostic.rejection_bits = payload_rejection_bits;
  } else if (diagnostic.scale_candidate_count == 0) {
    diagnostic.rejection_bits =
        LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_SCALE_KIND;
  } else if (diagnostic.flag_candidate_count == 0) {
    diagnostic.rejection_bits = flag_rejection_bits;
  } else if (diagnostic.feature_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_FEATURES;
  } else if (diagnostic.wave_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_AMDGPU_MATRIX_CONTRACT_REJECTION_WAVE_SIZE;
  }
  if (out_diagnostic != NULL) *out_diagnostic = diagnostic;
  return NULL;
}

#undef MATRIX_DESCRIPTOR
#undef MATRIX_PAYLOAD
#undef MATRIX_TILE_SHAPE
#undef MFMA_GFX940_FP8_FEATURES
#undef MFMA_GFX950_FEATURES
#undef MFMA_GFX950_SCALE_FEATURES
#undef SMFMAC_GFX940_FEATURES
#undef SMFMAC_GFX950_FEATURES
#undef WMMA_GFX12_FEATURES
#undef WMMA_GFX1250_FEATURES
#undef WMMA_GFX1250_SCALE_FEATURES
#undef SWMMAC_GFX12_FEATURES
#undef SWMMAC_GFX1250_FEATURES
#undef SWMMAC_GFX12_IU_FLAGS
#undef WMMA_GFX1250_MODS_ALL_FLAGS
#undef WMMA_GFX1250_MODS_C_FLAGS
#undef WMMA_GFX1250_SCALE_F4_FLAGS
#undef SWMMAC_GFX1250_AB_FLAGS
#undef SWMMAC_GFX1250_IU8_FLAGS
