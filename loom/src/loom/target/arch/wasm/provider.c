// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/provider.h"

#include "loom/target/arch/wasm/low_registry.h"
#include "loom/target/arch/wasm/math_policy.h"
#include "loom/target/arch/wasm/ops/registry.h"
#include "loom/target/emit/wasm/lower.h"

const loom_target_provider_t loom_wasm_target_provider = {
    .register_context = loom_wasm_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_wasm_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_wasm_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_wasm_math_policy_registry_initialize,
};

static const loom_target_provider_t* const kLoomWasmTargetProviders[] = {
    &loom_wasm_target_provider,
};

const loom_target_provider_set_t loom_wasm_target_provider_set = {
    .providers = kLoomWasmTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomWasmTargetProviders),
};
