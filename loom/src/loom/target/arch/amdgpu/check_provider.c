// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/check_provider.h"

#include "loom/target/arch/amdgpu/loom_check_requirements.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/lower.h"
#include "loom/target/arch/amdgpu/amdgpu_hal_backend.h"
#include "loom/tools/loom-check/hal_run_provider.h"

static const loom_run_hal_backend_t* const kLoomAmdgpuCheckHalBackends[] = {
    &iree_run_loom_amdgpu_hal_backend,
};

static const loom_check_hal_run_provider_t kLoomAmdgpuCheckHalRunProvider = {
    .base =
        {
            .name = IREE_SVL("amdgpu-hal"),
            .match = loom_check_hal_run_provider_match,
            .execute = loom_check_hal_run_provider_execute,
            .append_names = loom_check_hal_run_provider_append_names,
        },
    .backend_registry =
        {
            .backends = kLoomAmdgpuCheckHalBackends,
            .backend_count = IREE_ARRAYSIZE(kLoomAmdgpuCheckHalBackends),
        },
};

static const loom_check_run_provider_t* const kLoomAmdgpuCheckRunProviders[] = {
    &kLoomAmdgpuCheckHalRunProvider.base,
};

static const loom_check_requirement_provider_t* const
    kLoomAmdgpuCheckRequirementProviders[] = {
        &loom_amdgpu_loom_check_requirement_provider,
};

const loom_check_provider_t loom_amdgpu_check_provider = {
    .name = IREE_SVL("amdgpu"),
    .initialize_low_descriptor_registry =
        loom_amdgpu_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_amdgpu_low_lower_policy_registry_initialize,
    .run_providers = kLoomAmdgpuCheckRunProviders,
    .run_provider_count = IREE_ARRAYSIZE(kLoomAmdgpuCheckRunProviders),
    .requirement_providers = kLoomAmdgpuCheckRequirementProviders,
    .requirement_provider_count =
        IREE_ARRAYSIZE(kLoomAmdgpuCheckRequirementProviders),
};
