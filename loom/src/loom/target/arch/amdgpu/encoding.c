// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/encoding.h"

iree_string_view_t loom_amdgpu_encoding_format_name(uint16_t encoding_format) {
  switch (encoding_format) {
    case LOOM_AMDGPU_ENCODING_FORMAT_NONE:
      return IREE_SV("none");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOP1:
      return IREE_SV("sop1");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOP2:
      return IREE_SV("sop2");
    case LOOM_AMDGPU_ENCODING_FORMAT_SOPP:
      return IREE_SV("sopp");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP2:
      return IREE_SV("vop2");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP2_LITERAL:
      return IREE_SV("vop2_literal");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3:
      return IREE_SV("vop3");
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3P:
      return IREE_SV("vop3p");
    case LOOM_AMDGPU_ENCODING_FORMAT_SMEM:
      return IREE_SV("smem");
    case LOOM_AMDGPU_ENCODING_FORMAT_MUBUF:
      return IREE_SV("mubuf");
    case LOOM_AMDGPU_ENCODING_FORMAT_VBUFFER:
      return IREE_SV("vbuffer");
    default:
      return IREE_SV("unknown");
  }
}
