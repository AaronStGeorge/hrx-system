// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/low_registry.h"

#include "loom/target/arch/amdgpu/gfx11_descriptors.h"
#include "loom/target/arch/amdgpu/gfx1250_descriptors.h"
#include "loom/target/arch/amdgpu/gfx12_descriptors.h"
#include "loom/target/arch/amdgpu/gfx950_descriptors.h"

// clang-format off
static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
  loom_amdgpu_gfx950_core_descriptor_set,
  loom_amdgpu_gfx11_core_descriptor_set,
  loom_amdgpu_gfx12_core_descriptor_set,
  loom_amdgpu_gfx1250_core_descriptor_set,
};

// clang-format on

void loom_amdgpu_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kLowDescriptorSetProviders,
      IREE_ARRAYSIZE(kLowDescriptorSetProviders));
}
