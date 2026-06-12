// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/low_aliases.h"

static const loom_amdgpu_low_blocked_alias_t kLoomAmdgpuLowBlockedAliases[] = {
    {
        .alias_name = IREE_SVL("amdgpu.v_fma_dx9_zero_f32"),
        .alias_mnemonic = IREE_SVL("v_fma_dx9_zero_f32"),
        .replacement_descriptor_name = IREE_SVL("amdgpu.v_fma_f32"),
        .replacement_mnemonic = IREE_SVL("v_fma_f32"),
    },
    {
        .alias_name = IREE_SVL("v_fma_dx9_zero_f32"),
        .alias_mnemonic = IREE_SVL("v_fma_dx9_zero_f32"),
        .replacement_descriptor_name = IREE_SVL("amdgpu.v_fma_f32"),
        .replacement_mnemonic = IREE_SVL("v_fma_f32"),
    },
    {
        .alias_name = IREE_SVL("amdgpu.v_fmac_dx9_zero_f32"),
        .alias_mnemonic = IREE_SVL("v_fmac_dx9_zero_f32"),
        .replacement_descriptor_name = IREE_SVL("amdgpu.v_fmac_f32"),
        .replacement_mnemonic = IREE_SVL("v_fmac_f32"),
    },
    {
        .alias_name = IREE_SVL("v_fmac_dx9_zero_f32"),
        .alias_mnemonic = IREE_SVL("v_fmac_dx9_zero_f32"),
        .replacement_descriptor_name = IREE_SVL("amdgpu.v_fmac_f32"),
        .replacement_mnemonic = IREE_SVL("v_fmac_f32"),
    },
};

const loom_amdgpu_low_blocked_alias_t* loom_amdgpu_low_blocked_alias_lookup(
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kLoomAmdgpuLowBlockedAliases);
       ++i) {
    const loom_amdgpu_low_blocked_alias_t* alias =
        &kLoomAmdgpuLowBlockedAliases[i];
    if (iree_string_view_equal(name, alias->alias_name)) {
      return alias;
    }
  }
  return NULL;
}
