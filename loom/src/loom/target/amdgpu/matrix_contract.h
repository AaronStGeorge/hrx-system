// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU matrix target contracts.
//
// This file describes the target-native matrix primitives that Loom can select
// from a higher-level tile.contract after shapes, encodings, layouts, and value
// facts are refined enough to make the choice structural. The descriptors are
// intentionally data-only: lowering code can query exact shape/type/feature
// requirements without hard-coding AMDGPU intrinsic names throughout generic
// tile/vector passes.

#ifndef LOOM_TARGET_AMDGPU_MATRIX_CONTRACT_H_
#define LOOM_TARGET_AMDGPU_MATRIX_CONTRACT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bitset of AMDGPU matrix features available on a selected processor.
typedef uint64_t loom_amdgpu_matrix_feature_bits_t;

// Processor supports gfx908-era MFMA instructions.
#define LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX908 (UINT64_C(1) << 0)
// Processor supports gfx90a BF16 1k MFMA variants.
#define LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_BF16_1K (UINT64_C(1) << 1)
// Processor supports gfx90a F64 MFMA variants.
#define LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX90A_F64 (UINT64_C(1) << 2)
// Processor supports gfx940 FP8/BF8 and XF32 MFMA variants.
#define LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX940_FP8 (UINT64_C(1) << 3)
// Processor supports gfx950 F16/BF16/I8 MFMA shape variants.
#define LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950 (UINT64_C(1) << 4)
// Processor supports gfx950 scaled F8/F6/F4 MFMA variants.
#define LOOM_AMDGPU_MATRIX_FEATURE_MFMA_GFX950_SCALE_F8F6F4 (UINT64_C(1) << 5)
// Processor supports gfx940 sparse MFMA accumulate variants.
#define LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX940 (UINT64_C(1) << 6)
// Processor supports gfx950 sparse MFMA accumulate variants.
#define LOOM_AMDGPU_MATRIX_FEATURE_SMFMAC_GFX950 (UINT64_C(1) << 7)
// Processor supports gfx11 WMMA variants.
#define LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11 (UINT64_C(1) << 8)
// Processor supports gfx12 WMMA FP8/BF8/IU4 variants.
#define LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX12 (UINT64_C(1) << 9)
// Processor supports gfx12 SWMMAC sparse variants.
#define LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX12 (UINT64_C(1) << 10)
// Processor supports gfx1250 WMMA modifier/reuse variants.
#define LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250 (UINT64_C(1) << 11)
// Processor supports gfx1250 WMMA scaled F8/F6/F4 and F4 variants.
#define LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX1250_SCALE_F8F6F4 (UINT64_C(1) << 12)
// Processor supports gfx1250 SWMMAC modifier/reuse variants.
#define LOOM_AMDGPU_MATRIX_FEATURE_SWMMAC_GFX1250 (UINT64_C(1) << 13)

// Bitset describing legal wave sizes for a matrix contract.
typedef uint32_t loom_amdgpu_matrix_wave_size_bits_t;

// Contract may be selected for wave32 code generation.
#define LOOM_AMDGPU_MATRIX_WAVE_SIZE_32 ((uint32_t)1u << 0)
// Contract may be selected for wave64 code generation.
#define LOOM_AMDGPU_MATRIX_WAVE_SIZE_64 ((uint32_t)1u << 1)
// Contract may be selected for either wave32 or wave64 code generation.
#define LOOM_AMDGPU_MATRIX_WAVE_SIZE_ANY \
  (LOOM_AMDGPU_MATRIX_WAVE_SIZE_32 | LOOM_AMDGPU_MATRIX_WAVE_SIZE_64)

typedef enum loom_amdgpu_matrix_family_e {
  // Unknown or uninitialized matrix contract family.
  LOOM_AMDGPU_MATRIX_FAMILY_UNKNOWN = 0,
  // Matrix FMA instruction family.
  LOOM_AMDGPU_MATRIX_FAMILY_MFMA = 1,
  // Sparse matrix FMA instruction family.
  LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC = 2,
  // Wave matrix multiply-accumulate instruction family.
  LOOM_AMDGPU_MATRIX_FAMILY_WMMA = 3,
  // Sparse wave matrix multiply-accumulate instruction family.
  LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC = 4,
} loom_amdgpu_matrix_family_t;

typedef enum loom_amdgpu_matrix_numeric_type_e {
  // Unknown or uninitialized numeric payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN = 0,
  // IEEE f64 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_F64 = 1,
  // IEEE f32 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_F32 = 2,
  // IEEE f16 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_F16 = 3,
  // BF16 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_BF16 = 4,
  // AMD XF32 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_XF32 = 5,
  // Signed i32 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_I32 = 6,
  // Signed i8 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_I8 = 7,
  // Per-operand sign-selected 8-bit integer payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_IU8 = 8,
  // Signed i4 payload packed into byte-like storage.
  LOOM_AMDGPU_MATRIX_NUMERIC_I4 = 9,
  // Per-operand sign-selected 4-bit integer payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_IU4 = 10,
  // AMD FP8 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_FP8 = 11,
  // AMD BF8 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_BF8 = 12,
  // AMD FP6 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_FP6 = 13,
  // AMD BF6 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_BF6 = 14,
  // AMD FP4 payload.
  LOOM_AMDGPU_MATRIX_NUMERIC_FP4 = 15,
  // Selector-driven AMD F8/F6/F4 payload family.
  LOOM_AMDGPU_MATRIX_NUMERIC_F8F6F4 = 16,
} loom_amdgpu_matrix_numeric_type_t;

typedef enum loom_amdgpu_matrix_scale_kind_e {
  // Contract has no explicit scale operands.
  LOOM_AMDGPU_MATRIX_SCALE_NONE = 0,
  // Contract uses 32-bit scale exponent operands.
  LOOM_AMDGPU_MATRIX_SCALE_32 = 1,
  // Contract uses 16-bit scale exponent operands packed into 64-bit operands.
  LOOM_AMDGPU_MATRIX_SCALE_16 = 2,
} loom_amdgpu_matrix_scale_kind_t;

typedef enum loom_amdgpu_matrix_format_selector_e {
  // LLVM selector value for FP8 matrix payloads.
  LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP8 = 0,
  // LLVM selector value for BF8 matrix payloads.
  LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_BF8 = 1,
  // LLVM selector value for FP6 matrix payloads.
  LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP6 = 2,
  // LLVM selector value for BF6 matrix payloads.
  LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_BF6 = 3,
  // LLVM selector value for FP4 matrix payloads.
  LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP4 = 4,
} loom_amdgpu_matrix_format_selector_t;

// Bitset of optional ABI operands/modifiers required by a contract.
typedef uint32_t loom_amdgpu_matrix_contract_flags_t;

// Contract consumes an explicit sparse index operand.
#define LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SPARSE ((uint32_t)1u << 0)
// Contract consumes explicit scale operands.
#define LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED ((uint32_t)1u << 1)
// Contract consumes matrix-format selector operands.
#define LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS ((uint32_t)1u << 2)
// Contract consumes matrix reuse immediate operands.
#define LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_REUSE ((uint32_t)1u << 3)
// Contract consumes a clamp immediate operand.
#define LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_CLAMP ((uint32_t)1u << 4)
// Contract consumes A/B sign-selection immediate operands.
#define LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SIGN_SELECT ((uint32_t)1u << 5)
// Contract consumes A/B operand modifier immediate operands.
#define LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_AB_MODIFIERS ((uint32_t)1u << 6)
// Contract consumes a C accumulator modifier immediate operand.
#define LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_C_MODIFIER ((uint32_t)1u << 7)
// Contract consumes a GFX11/GFX12 op_sel operand.
#define LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_OPSEL ((uint32_t)1u << 8)
// A zero scale can refine to an unscaled contract with the same shape.
#define LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK ((uint32_t)1u << 9)
// Contract consumes scale-format selector operands.
#define LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS ((uint32_t)1u << 10)

typedef struct loom_amdgpu_matrix_tile_shape_t {
  // Contracted result rows in the target-native tile.
  uint16_t result_row_count;
  // Contracted result columns in the target-native tile.
  uint16_t result_column_count;
  // Contracted K depth consumed by one target-native instruction.
  uint16_t reduction_count;
} loom_amdgpu_matrix_tile_shape_t;

typedef struct loom_amdgpu_matrix_payload_shape_t {
  // Logical numeric type represented by this operand or result.
  loom_amdgpu_matrix_numeric_type_t numeric_type;
  // Number of 32-bit VGPR payload registers in a fixed LLVM signature. Zero
  // means the descriptor needs a later wave- or format-specific fragment
  // signature.
  uint16_t register_count;
  // Number of logical scalar elements represented by the payload. Zero means
  // the descriptor needs a later wave- or format-specific fragment signature.
  uint16_t element_count;
} loom_amdgpu_matrix_payload_shape_t;

typedef struct loom_amdgpu_matrix_contract_descriptor_t {
  // Stable Loom descriptor name used by tests, diagnostics, and target logs.
  iree_string_view_t name;
  // LLVM AMDGPU intrinsic name selected by this descriptor.
  iree_string_view_t llvm_intrinsic_name;
  // AMDGPU instruction family used by this descriptor.
  loom_amdgpu_matrix_family_t family;
  // Processor feature bits required before this descriptor is legal.
  loom_amdgpu_matrix_feature_bits_t required_feature_bits;
  // Wave sizes for which this descriptor is legal.
  loom_amdgpu_matrix_wave_size_bits_t wave_size_bits;
  // Optional immediate operands or semantic decorations required by the call.
  loom_amdgpu_matrix_contract_flags_t flags;
  // Logical tile shape consumed and produced by one instruction.
  loom_amdgpu_matrix_tile_shape_t tile_shape;
  // Matrix A payload shape.
  loom_amdgpu_matrix_payload_shape_t lhs_payload;
  // Matrix B payload shape.
  loom_amdgpu_matrix_payload_shape_t rhs_payload;
  // Accumulator payload shape.
  loom_amdgpu_matrix_payload_shape_t accumulator_payload;
  // Result payload shape.
  loom_amdgpu_matrix_payload_shape_t result_payload;
  // Explicit scale operand kind.
  loom_amdgpu_matrix_scale_kind_t scale_kind;
} loom_amdgpu_matrix_contract_descriptor_t;

// Returns the stable display name for a matrix family.
iree_string_view_t loom_amdgpu_matrix_family_name(
    loom_amdgpu_matrix_family_t family);

// Returns the stable display name for a matrix numeric type.
iree_string_view_t loom_amdgpu_matrix_numeric_type_name(
    loom_amdgpu_matrix_numeric_type_t numeric_type);

// Returns the stable display name for a scale kind.
iree_string_view_t loom_amdgpu_matrix_scale_kind_name(
    loom_amdgpu_matrix_scale_kind_t scale_kind);

// Maps a processor name such as "gfx942" or "gfx1250" to matrix feature bits.
iree_status_t loom_amdgpu_matrix_feature_bits_for_processor(
    iree_string_view_t processor,
    loom_amdgpu_matrix_feature_bits_t* out_feature_bits);

// Returns the number of built-in AMDGPU matrix contract descriptors.
iree_host_size_t loom_amdgpu_matrix_contract_descriptor_count(void);

// Returns a built-in descriptor by ordinal, or NULL when |index| is out of
// range.
const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_contract_descriptor_at(iree_host_size_t index);

// Finds a built-in descriptor by stable descriptor name.
const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_matrix_contract_find_by_name(iree_string_view_t name);

// Returns whether a descriptor is legal for a processor feature set and wave
// size. Pass wave_size=0 to ignore wave-size filtering.
bool loom_amdgpu_matrix_contract_is_available(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_amdgpu_matrix_feature_bits_t feature_bits, uint32_t wave_size);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_AMDGPU_MATRIX_CONTRACT_H_
