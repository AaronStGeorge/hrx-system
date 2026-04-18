// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/low_descriptor_registry.h"

#include "loom/target/arch/amdgpu/gfx11_descriptors.h"
#include "loom/target/arch/amdgpu/gfx1250_descriptors.h"
#include "loom/target/arch/amdgpu/gfx12_descriptors.h"
#include "loom/target/arch/amdgpu/gfx950_descriptors.h"
#include "loom/target/arch/wasm/descriptors.h"
#include "loom/target/arch/x86/avx512_descriptors.h"
#include "loom/target/arch/x86/packed_dot_descriptors.h"
#include "loom/target/emit/ireevm/descriptors.h"

void loom_target_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(out_registry);
  *out_registry = (loom_target_low_descriptor_registry_t){0};
  out_registry->descriptor_sets[0] = loom_ireevm_core_descriptor_set();
  out_registry->descriptor_sets[1] = loom_wasm_core_simd128_descriptor_set();
  out_registry->descriptor_sets[2] = loom_x86_avx512_core_descriptor_set();
  out_registry->descriptor_sets[3] = loom_x86_packed_dot_core_descriptor_set();
  out_registry->descriptor_sets[4] = loom_amdgpu_gfx950_core_descriptor_set();
  out_registry->descriptor_sets[5] = loom_amdgpu_gfx11_core_descriptor_set();
  out_registry->descriptor_sets[6] = loom_amdgpu_gfx12_core_descriptor_set();
  out_registry->descriptor_sets[7] = loom_amdgpu_gfx1250_core_descriptor_set();
  out_registry->registry = (loom_low_descriptor_registry_t){
      .descriptor_sets = out_registry->descriptor_sets,
      .descriptor_set_count = IREE_ARRAYSIZE(out_registry->descriptor_sets),
  };
}

iree_status_t loom_target_low_descriptor_set_lookup(
    iree_string_view_t key,
    const loom_low_descriptor_set_t** out_descriptor_set) {
  loom_target_low_descriptor_registry_t registry;
  loom_target_low_descriptor_registry_initialize(&registry);
  return loom_low_descriptor_registry_lookup(&registry.registry, key,
                                             out_descriptor_set);
}
