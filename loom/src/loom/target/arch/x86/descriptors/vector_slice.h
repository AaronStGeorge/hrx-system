// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Instruction vocabulary for slicing x86 vector carriers in registers.

#ifndef LOOM_TARGET_ARCH_X86_DESCRIPTORS_VECTOR_SLICE_H_
#define LOOM_TARGET_ARCH_X86_DESCRIPTORS_VECTOR_SLICE_H_

#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_X86_VECTOR_SLICE_WIDTH_I8 = 0,
  LOOM_X86_VECTOR_SLICE_WIDTH_I16 = 1,
  LOOM_X86_VECTOR_SLICE_WIDTH_I32 = 2,
  LOOM_X86_VECTOR_SLICE_WIDTH_I64 = 3,
  LOOM_X86_VECTOR_SLICE_WIDTH_COUNT = 4,
};

enum {
  LOOM_X86_VECTOR_SLICE_CLASS_XMM = 0,
  LOOM_X86_VECTOR_SLICE_CLASS_YMM = 1,
  LOOM_X86_VECTOR_SLICE_CLASS_ZMM = 2,
  LOOM_X86_VECTOR_SLICE_CLASS_COUNT = 3,
};

enum {
  LOOM_X86_VECTOR_SLICE_WIDEN_I8_I16 = 0,
  LOOM_X86_VECTOR_SLICE_WIDEN_I8_I32 = 1,
  LOOM_X86_VECTOR_SLICE_WIDEN_I8_I64 = 2,
  LOOM_X86_VECTOR_SLICE_WIDEN_I16_I32 = 3,
  LOOM_X86_VECTOR_SLICE_WIDEN_I16_I64 = 4,
  LOOM_X86_VECTOR_SLICE_WIDEN_I32_I64 = 5,
  LOOM_X86_VECTOR_SLICE_WIDEN_COUNT = 6,
};

typedef struct loom_x86_vector_slice_descriptors_t {
  // Maximum logical vector width admitted by this descriptor view.
  uint16_t maximum_vector_bit_width;
  // True when the descriptor view provides AVX-512 mask registers.
  bool supports_mask;
  // Shifts one XMM carrier right by an immediate byte count.
  uint32_t shift_xmm;
  // Aligns a byte window across two adjacent XMM carriers.
  uint32_t align_xmm;
  // Extracts one 128-bit quarter from a YMM carrier.
  uint32_t extract_xmm_from_ymm;
  // Inserts one XMM carrier into a YMM carrier.
  uint32_t insert_xmm_into_ymm;
  // Aligns a ZMM carrier at a 32-bit lane boundary, or NONE without AVX-512.
  uint32_t align_dword_zmm;
  // Aligns a ZMM carrier at a 64-bit lane boundary, or NONE without AVX-512.
  uint32_t align_qword_zmm;
  // Extracts one 128-bit quarter from a ZMM carrier, or NONE without AVX-512.
  uint32_t extract_xmm_from_zmm;
  // Extracts one 256-bit half from a ZMM carrier, or NONE without AVX-512.
  uint32_t extract_ymm_from_zmm;
  // AVX2 signed SIMD predicate-representation conversion vocabulary.
  struct {
    // Signed widening indexed by LOOM_X86_VECTOR_SLICE_WIDEN_* and result
    // class XMM/YMM.
    uint32_t widen[LOOM_X86_VECTOR_SLICE_WIDEN_COUNT][2];
    // Packs i64 to i32, i32 to i16, or i16 to i8 in one XMM carrier.
    uint32_t pack[3];
  } simd_conversion;
  // AVX-512 native predicate-mask conversion vocabulary.
  struct {
    // Shifts a packed K mask right by an immediate logical lane count.
    uint32_t shift;
    // Compresses SIMD lanes to K, indexed by physical lane width and class.
    uint32_t compress[LOOM_X86_VECTOR_SLICE_WIDTH_COUNT]
                     [LOOM_X86_VECTOR_SLICE_CLASS_COUNT];
    // Expands K bits to SIMD lanes, indexed by physical lane width and class.
    uint32_t expand[LOOM_X86_VECTOR_SLICE_WIDTH_COUNT]
                   [LOOM_X86_VECTOR_SLICE_CLASS_COUNT];
    // Produces an empty ZMM carrier for a no-VL compression bridge.
    uint32_t zero_zmm;
    // Inserts an XMM carrier into an empty ZMM compression bridge.
    uint32_t insert_xmm_into_zmm;
    // Inserts a YMM carrier into an empty ZMM compression bridge.
    uint32_t insert_ymm_into_zmm;
  } mask_conversion;
} loom_x86_vector_slice_descriptors_t;

// Returns the register-slice vocabulary of an AVX2 or AVX-512 descriptor view.
// Other representation fragments return NULL.
const loom_x86_vector_slice_descriptors_t* loom_x86_vector_slice_descriptors(
    const loom_low_descriptor_set_t* descriptor_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_DESCRIPTORS_VECTOR_SLICE_H_
