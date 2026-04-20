// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check runner for AMDGPU target-owned .loom-test files.

#include "loom/target/arch/amdgpu/loom_check_requirements.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/lower.h"
#include "loom/target/arch/amdgpu/amdgpu_hal_backend.h"
#include "loom/tools/loom-check/hal_run_provider.h"
#include "loom/tools/loom-check/main.h"

static const loom_run_hal_backend_t* const kLoomCheckAmdgpuHalBackends[] = {
    &iree_run_loom_amdgpu_hal_backend,
};

static const loom_check_hal_run_provider_config_t
    kLoomCheckAmdgpuHalRunProviderConfig = {
        .backend_registry =
            {
                .backends = kLoomCheckAmdgpuHalBackends,
                .backend_count = IREE_ARRAYSIZE(kLoomCheckAmdgpuHalBackends),
            },
};

static const loom_check_run_provider_t kLoomCheckAmdgpuHalRunProvider = {
    .name = IREE_SVL("amdgpu-hal"),
    .user_data = &kLoomCheckAmdgpuHalRunProviderConfig,
    .match = loom_check_hal_run_provider_match,
    .execute = loom_check_hal_run_provider_execute,
    .append_names = loom_check_hal_run_provider_append_names,
};

static const loom_check_run_provider_t* const kLoomCheckAmdgpuRunProviders[] = {
    &kLoomCheckAmdgpuHalRunProvider,
};

static const loom_check_requirement_provider_t* const
    kLoomCheckAmdgpuRequirementProviders[] = {
        &loom_amdgpu_loom_check_requirement_provider,
};

static const loom_check_production_runner_t kLoomCheckAmdgpuRunner = {
    .initialize_low_descriptor_registry =
        loom_amdgpu_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_amdgpu_low_lower_policy_registry_initialize,
    .run_providers = kLoomCheckAmdgpuRunProviders,
    .run_provider_count = IREE_ARRAYSIZE(kLoomCheckAmdgpuRunProviders),
    .requirement_providers = kLoomCheckAmdgpuRequirementProviders,
    .requirement_provider_count =
        IREE_ARRAYSIZE(kLoomCheckAmdgpuRequirementProviders),
};

int main(int argc, char** argv) {
  return loom_check_production_main(argc, argv, &kLoomCheckAmdgpuRunner);
}
