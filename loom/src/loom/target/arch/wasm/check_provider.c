// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/check_provider.h"

#include "loom/target/arch/wasm/low_registry.h"
#include "loom/target/emit/wasm/loom_check.h"
#include "loom/target/emit/wasm/lower.h"

static const loom_check_emit_provider_t* const kLoomWasmCheckEmitProviders[] = {
    &loom_wasm_loom_check_emit_provider,
};

static const loom_check_requirement_provider_t* const
    kLoomWasmCheckRequirementProviders[] = {
        &loom_wasm_loom_check_requirement_provider,
};

const loom_check_provider_t loom_wasm_check_provider = {
    .name = IREE_SVL("wasm"),
    .initialize_low_descriptor_registry =
        loom_wasm_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_wasm_low_lower_policy_registry_initialize,
    .emit_providers = kLoomWasmCheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomWasmCheckEmitProviders),
    .requirement_providers = kLoomWasmCheckRequirementProviders,
    .requirement_provider_count =
        IREE_ARRAYSIZE(kLoomWasmCheckRequirementProviders),
};
