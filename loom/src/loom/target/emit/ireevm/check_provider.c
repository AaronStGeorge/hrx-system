// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/check_provider.h"

#include "loom/target/emit/ireevm/loom_check_run.h"
#include "loom/target/emit/ireevm/low_registry.h"
#include "loom/target/emit/ireevm/lower.h"

static const loom_check_run_provider_t* const kLoomIreeVmCheckRunProviders[] = {
    &loom_ireevm_loom_check_run_provider,
};

const loom_check_provider_t loom_ireevm_check_provider = {
    .name = IREE_SVL("ireevm"),
    .initialize_low_descriptor_registry =
        loom_ireevm_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_ireevm_low_lower_policy_registry_initialize,
    .run_providers = kLoomIreeVmCheckRunProviders,
    .run_provider_count = IREE_ARRAYSIZE(kLoomIreeVmCheckRunProviders),
};
