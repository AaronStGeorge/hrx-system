// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/low_registry.h"

#include "loom/target/arch/amdgpu/cdna3_descriptors.h"
#include "loom/target/arch/amdgpu/cdna4_descriptors.h"
#include "loom/target/arch/amdgpu/rdna3_descriptors.h"
#include "loom/target/arch/amdgpu/rdna4_descriptors.h"
#include "loom/target/arch/amdgpu/rdna4_gfx125x_descriptors.h"

// clang-format off
static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
  loom_amdgpu_cdna3_core_descriptor_set,
  loom_amdgpu_cdna4_core_descriptor_set,
  loom_amdgpu_rdna3_core_descriptor_set,
  loom_amdgpu_rdna4_core_descriptor_set,
  loom_amdgpu_rdna4_gfx125x_core_descriptor_set,
};

// clang-format on

void loom_amdgpu_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kLowDescriptorSetProviders,
      IREE_ARRAYSIZE(kLowDescriptorSetProviders));
}
