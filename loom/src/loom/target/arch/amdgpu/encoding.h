// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target-owned low descriptor encoding format identifiers.
//
// These values are written into loom_low_descriptor_t::encoding_format_id by
// AMDGPU descriptor overlays. They are intentionally compact and stable within
// Loom; vendor XML encoding names stay in the generator layer instead of being
// linked into native emitters as strings.

#ifndef LOOM_TARGET_ARCH_AMDGPU_ENCODING_H_
#define LOOM_TARGET_ARCH_AMDGPU_ENCODING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_encoding_format_e {
  // Descriptor does not carry an AMDGPU machine encoding format.
  LOOM_AMDGPU_ENCODING_FORMAT_NONE = 0,
  // Scalar one-source instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SOP1 = 1,
  // Scalar two-source instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SOP2 = 2,
  // Scalar program-flow or wait instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SOPP = 3,
  // Vector two-source 32-bit instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP2 = 4,
  // VOP2 form with a mandatory literal payload.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP2_LITERAL = 5,
  // Vector three-source 64-bit instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP3 = 6,
  // Packed vector three-source 64-bit instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VOP3P = 7,
  // Scalar memory instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_SMEM = 8,
  // Memory buffer instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_MUBUF = 9,
  // RDNA4 VBUFFER instruction format.
  LOOM_AMDGPU_ENCODING_FORMAT_VBUFFER = 10,
} loom_amdgpu_encoding_format_t;

// Returns a short stable diagnostic name for |encoding_format|.
iree_string_view_t loom_amdgpu_encoding_format_name(uint16_t encoding_format);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_ENCODING_H_
