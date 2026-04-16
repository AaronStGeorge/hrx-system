// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/amdgpu/matrix_contract.h"

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

#define MATRIX_DESCRIPTOR(                                                    \
    name_value, intrinsic_name_value, family_value, feature_bits_value,       \
    wave_bits_value, flags_value, row_count_value, column_count_value,        \
    reduction_count_value, lhs_type_value, lhs_registers_value,               \
    lhs_elements_value, rhs_type_value, rhs_registers_value,                  \
    rhs_elements_value, accumulator_type_value, accumulator_registers_value,  \
    accumulator_elements_value, result_type_value, result_registers_value,    \
    result_elements_value, scale_kind_value)                                  \
  {                                                                           \
      .name = IREE_SVL(name_value),                                           \
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
  }

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
                          16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
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
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x16.bf16",
                          "llvm.amdgcn.wmma.f32.16x16x16.bf16",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF16, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x16.fp8.fp8",
                          "llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX12_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x16.bf8.bf8",
                          "llvm.amdgcn.wmma.f32.16x16x16.bf8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX12_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x16.bf8.fp8",
                          "llvm.amdgcn.wmma.f32.16x16x16.bf8.fp8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX12_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x16.fp8.bf8",
                          "llvm.amdgcn.wmma.f32.16x16x16.fp8.bf8",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX12_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_FP8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_BF8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x16.f16",
                          "llvm.amdgcn.wmma.f32.16x16x16.f16",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA,
                          LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY, 0, 16, 16, 16,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.f32.16x16x32.f16",
                          "llvm.amdgcn.wmma.f32.16x16x32.f16",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
                          LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER |
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE,
                          16, 16, 32, LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F16, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR("wmma.i32.16x16x32.iu4",
                          "llvm.amdgcn.wmma.i32.16x16x32.iu4",
                          LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX12_FEATURES,
                          LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
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
                              LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS,
                          16, 16, 64, LOOM_AMDGPU_MATRIX_NUMERIC_IU8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_IU8, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
                          LOOM_AMDGPU_MATRIX_NUMERIC_I32, 0, 0,
                          LOOM_AMDGPU_MATRIX_SCALE_NONE),
        MATRIX_DESCRIPTOR(
            "wmma.scale.f32.16x16x128.f8f6f4",
            "llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_SCALE_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK,
            16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0, LOOM_AMDGPU_MATRIX_SCALE_32),
        MATRIX_DESCRIPTOR(
            "wmma.scale16.f32.16x16x128.f8f6f4",
            "llvm.amdgcn.wmma.scale16.f32.16x16x128.f8f6f4",
            LOOM_AMDGPU_MATRIX_FAMILY_WMMA, WMMA_GFX1250_SCALE_FEATURES,
            LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY,
            LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE |
                LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK,
            16, 16, 128, LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0,
            LOOM_AMDGPU_MATRIX_NUMERIC_F32, 0, 0, LOOM_AMDGPU_MATRIX_SCALE_16),
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

static bool loom_amdgpu_matrix_processor_is_any(iree_string_view_t processor,
                                                const iree_string_view_t* names,
                                                iree_host_size_t name_count) {
  for (iree_host_size_t i = 0; i < name_count; ++i) {
    if (iree_string_view_equal(processor, names[i])) return true;
  }
  return false;
}

iree_status_t loom_amdgpu_matrix_feature_bits_for_processor(
    iree_string_view_t processor,
    loom_amdgpu_matrix_feature_bits_t* out_feature_bits) {
  if (out_feature_bits == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU feature bit output is required");
  }
  *out_feature_bits = 0;
  processor = iree_string_view_trim(processor);
  static const iree_string_view_t gfx908_processors[] = {
      IREE_SVL("gfx908"),
  };
  static const iree_string_view_t gfx90a_processors[] = {
      IREE_SVL("gfx90a"),
  };
  static const iree_string_view_t gfx940_processors[] = {
      IREE_SVL("gfx940"),
      IREE_SVL("gfx941"),
      IREE_SVL("gfx942"),
  };
  static const iree_string_view_t gfx950_processors[] = {
      IREE_SVL("gfx950"),
  };
  static const iree_string_view_t gfx11_processors[] = {
      IREE_SVL("gfx1100"), IREE_SVL("gfx1101"), IREE_SVL("gfx1102"),
      IREE_SVL("gfx1103"), IREE_SVL("gfx1150"), IREE_SVL("gfx1151"),
  };
  static const iree_string_view_t gfx12_processors[] = {
      IREE_SVL("gfx1200"),
      IREE_SVL("gfx1201"),
  };
  static const iree_string_view_t gfx1250_processors[] = {
      IREE_SVL("gfx1250"),
      IREE_SVL("gfx1251"),
      IREE_SVL("gfx1252"),
  };

  if (loom_amdgpu_matrix_processor_is_any(processor, gfx908_processors,
                                          IREE_ARRAYSIZE(gfx908_processors))) {
    *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908;
    return iree_ok_status();
  }
  if (loom_amdgpu_matrix_processor_is_any(processor, gfx90a_processors,
                                          IREE_ARRAYSIZE(gfx90a_processors))) {
    *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908 |
                        LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K |
                        LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64;
    return iree_ok_status();
  }
  if (loom_amdgpu_matrix_processor_is_any(processor, gfx940_processors,
                                          IREE_ARRAYSIZE(gfx940_processors))) {
    *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908 |
                        LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K |
                        LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64 |
                        MFMA_GFX940_FP8_FEATURES |
                        LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940;
    return iree_ok_status();
  }
  if (loom_amdgpu_matrix_processor_is_any(processor, gfx950_processors,
                                          IREE_ARRAYSIZE(gfx950_processors))) {
    *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908 |
                        LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K |
                        LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64 |
                        MFMA_GFX940_FP8_FEATURES | MFMA_GFX950_SCALE_FEATURES |
                        LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940 |
                        LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX950;
    return iree_ok_status();
  }
  if (loom_amdgpu_matrix_processor_is_any(processor, gfx11_processors,
                                          IREE_ARRAYSIZE(gfx11_processors))) {
    *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11;
    return iree_ok_status();
  }
  if (loom_amdgpu_matrix_processor_is_any(processor, gfx12_processors,
                                          IREE_ARRAYSIZE(gfx12_processors))) {
    *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11 |
                        WMMA_GFX12_FEATURES |
                        LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12;
    return iree_ok_status();
  }
  if (loom_amdgpu_matrix_processor_is_any(processor, gfx1250_processors,
                                          IREE_ARRAYSIZE(gfx1250_processors))) {
    *out_feature_bits = LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11 |
                        WMMA_GFX12_FEATURES | WMMA_GFX1250_SCALE_FEATURES |
                        LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12 |
                        LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX1250;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown AMDGPU processor '%.*s'",
                          (int)processor.size, processor.data);
}

iree_host_size_t loom_amdgpu_matrix_contract_descriptor_count(void) {
  return IREE_ARRAYSIZE(kMatrixContractDescriptors);
}

const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_contract_descriptor_at(iree_host_size_t index) {
  if (index >= IREE_ARRAYSIZE(kMatrixContractDescriptors)) return NULL;
  return &kMatrixContractDescriptors[index];
}

const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_contract_find_by_name(iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kMatrixContractDescriptors);
       ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        &kMatrixContractDescriptors[i];
    if (iree_string_view_equal(name, descriptor->name)) return descriptor;
  }
  return NULL;
}

bool loom_amdgpu_matrix_contract_is_available(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_amdgpu_matrix_feature_bits_t feature_bits, uint32_t wave_size) {
  if (descriptor == NULL) return false;
  if ((feature_bits & descriptor->required_feature_bits) !=
      descriptor->required_feature_bits) {
    return false;
  }
  if (wave_size == 0) return true;
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
