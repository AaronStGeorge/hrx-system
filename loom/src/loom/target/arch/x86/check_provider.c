// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/check_provider.h"

#include "loom/target/arch/x86/low_registry.h"
#include "loom/target/arch/x86/lower.h"

const loom_check_provider_t loom_x86_check_provider = {
    .name = IREE_SVL("x86"),
    .initialize_low_descriptor_registry =
        loom_x86_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_x86_low_lower_policy_registry_initialize,
};
