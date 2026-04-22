// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/check_provider.h"

#include "loom/target/arch/x86/low_registry.h"
#include "loom/target/arch/x86/lower.h"
#include "loom/target/emit/native/x86/loom_check.h"

static const loom_check_emit_provider_t* const kLoomX86CheckEmitProviders[] = {
    &loom_x86_native_loom_check_emit_provider,
};

static const loom_target_low_legality_provider_t* const
    kLoomX86LowLegalityProviders[] = {
        &loom_x86_low_legality_provider_storage,
};

const loom_check_provider_t loom_x86_check_provider = {
    .name = IREE_SVL("x86"),
    .initialize_low_descriptor_registry =
        loom_x86_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_x86_low_lower_policy_registry_initialize,
    .low_legality_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomX86LowLegalityProviders),
            .values = kLoomX86LowLegalityProviders,
        },
    .emit_providers = kLoomX86CheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomX86CheckEmitProviders),
};
