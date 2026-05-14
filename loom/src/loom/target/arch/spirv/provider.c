// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/provider.h"

#include "loom/target/arch/spirv/low_registry.h"
#include "loom/target/arch/spirv/ops/registry.h"

const loom_target_provider_t loom_spirv_target_provider = {
    .register_context = loom_spirv_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_spirv_low_descriptor_registry_initialize,
};

static const loom_target_provider_t* const kLoomSpirvTargetProviders[] = {
    &loom_spirv_target_provider,
};

const loom_target_provider_set_t loom_spirv_target_provider_set = {
    .providers = kLoomSpirvTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomSpirvTargetProviders),
};
