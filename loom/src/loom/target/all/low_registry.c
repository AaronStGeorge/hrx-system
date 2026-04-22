// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/all/low_registry.h"

#include "loom/target/arch/amdgpu/gfx11_descriptors.h"
#include "loom/target/arch/amdgpu/gfx1250_descriptors.h"
#include "loom/target/arch/amdgpu/gfx12_descriptors.h"
#include "loom/target/arch/amdgpu/gfx950_descriptors.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/lower.h"
#include "loom/target/arch/wasm/descriptors.h"
#include "loom/target/arch/wasm/low_registry.h"
#include "loom/target/arch/x86/avx512_descriptors.h"
#include "loom/target/arch/x86/low_registry.h"
#include "loom/target/arch/x86/lower.h"
#include "loom/target/arch/x86/packed_dot_descriptors.h"
#include "loom/target/emit/ireevm/descriptors.h"
#include "loom/target/emit/ireevm/low_registry.h"
#include "loom/target/emit/ireevm/lower.h"
#include "loom/target/emit/wasm/lower.h"

static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
    loom_ireevm_core_descriptor_set,
    loom_wasm_core_simd128_descriptor_set,
    loom_x86_avx512_core_descriptor_set,
    loom_x86_packed_dot_core_descriptor_set,
    loom_amdgpu_gfx950_core_descriptor_set,
    loom_amdgpu_gfx11_core_descriptor_set,
    loom_amdgpu_gfx12_core_descriptor_set,
    loom_amdgpu_gfx1250_core_descriptor_set,
};

static const loom_target_bundle_t* const kLowTargetBundles[] = {
    &loom_ireevm_low_target_bundle_core,
    &loom_wasm_low_target_bundle_core_simd128,
    &loom_x86_low_target_bundle_avx512_core,
    &loom_x86_low_target_bundle_packed_dot_core,
    &loom_amdgpu_low_target_bundle_gfx950_core,
    &loom_amdgpu_low_target_bundle_gfx11_core,
    &loom_amdgpu_low_target_bundle_gfx12_core,
    &loom_amdgpu_low_target_bundle_gfx1250_core,
};

void loom_all_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kLowDescriptorSetProviders,
      IREE_ARRAYSIZE(kLowDescriptorSetProviders), kLowTargetBundles,
      IREE_ARRAYSIZE(kLowTargetBundles));
}

void loom_all_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static loom_low_lower_policy_registry_entry_t kLowLowerPolicyEntries[] = {
      {.contract_set_key = IREE_SVL("iree.vm.core")},
      {.contract_set_key = IREE_SVL("wasm.core.simd128")},
      {.contract_set_key = IREE_SVL("x86.avx512.core")},
      {.contract_set_key = IREE_SVL("x86.packed_dot.core")},
      {.contract_set_key = IREE_SVL("amdgpu.gfx950.core")},
      {.contract_set_key = IREE_SVL("amdgpu.gfx11.core")},
      {.contract_set_key = IREE_SVL("amdgpu.gfx12.core")},
      {.contract_set_key = IREE_SVL("amdgpu.gfx1250.core")},
  };
  kLowLowerPolicyEntries[0].policy = loom_ireevm_low_lower_policy();
  kLowLowerPolicyEntries[1].policy = loom_wasm_low_lower_policy();
  kLowLowerPolicyEntries[2].policy = loom_x86_avx512_low_lower_policy();
  kLowLowerPolicyEntries[3].policy = loom_x86_packed_dot_low_lower_policy();
  kLowLowerPolicyEntries[4].policy = loom_amdgpu_low_lower_policy();
  kLowLowerPolicyEntries[5].policy = loom_amdgpu_low_lower_policy();
  kLowLowerPolicyEntries[6].policy = loom_amdgpu_low_lower_policy();
  kLowLowerPolicyEntries[7].policy = loom_amdgpu_low_lower_policy();
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kLowLowerPolicyEntries,
      IREE_ARRAYSIZE(kLowLowerPolicyEntries));
}
