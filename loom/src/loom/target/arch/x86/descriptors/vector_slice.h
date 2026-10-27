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

typedef struct loom_x86_vector_slice_descriptors_t {
  // Maximum logical vector width admitted by this descriptor view.
  uint16_t maximum_vector_bit_width;
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
} loom_x86_vector_slice_descriptors_t;

// Returns the register-slice vocabulary of an AVX2 or AVX-512 descriptor view.
// Other representation fragments return NULL.
const loom_x86_vector_slice_descriptors_t* loom_x86_vector_slice_descriptors(
    const loom_low_descriptor_set_t* descriptor_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_DESCRIPTORS_VECTOR_SLICE_H_
