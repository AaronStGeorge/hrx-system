// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/low_registry.h"

#include "loom/target/arch/x86/avx512_descriptors.h"
#include "loom/target/arch/x86/avx512_packed_dot_descriptors.h"
#include "loom/target/arch/x86/packed_dot_descriptors.h"
#include "loom/target/arch/x86/scalar_descriptors.h"

static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
    loom_x86_scalar_core_descriptor_set,
    loom_x86_avx512_core_descriptor_set,
    loom_x86_packed_dot_core_descriptor_set,
    loom_x86_avx512_packed_dot_core_descriptor_set,
};

void loom_x86_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kLowDescriptorSetProviders,
      IREE_ARRAYSIZE(kLowDescriptorSetProviders));
}
