// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/check_provider.h"

#include "loom/target/emit/native/amdgpu/loom_check.h"

#ifndef LOOM_AMDGPU_CHECK_HAVE_HAL_RUN_PROVIDER
#define LOOM_AMDGPU_CHECK_HAVE_HAL_RUN_PROVIDER 0
#endif  // LOOM_AMDGPU_CHECK_HAVE_HAL_RUN_PROVIDER

#if LOOM_AMDGPU_CHECK_HAVE_HAL_RUN_PROVIDER
#include "loom/target/arch/amdgpu/loom_check_requirements.h"
#endif  // LOOM_AMDGPU_CHECK_HAVE_HAL_RUN_PROVIDER
#include "loom/target/arch/amdgpu/coverage.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/lower.h"
#if LOOM_AMDGPU_CHECK_HAVE_HAL_RUN_PROVIDER
#include "loom/target/arch/amdgpu/amdgpu_hal_backend.h"
#include "loom/tools/loom-check/hal_run_provider.h"
#endif  // LOOM_AMDGPU_CHECK_HAVE_HAL_RUN_PROVIDER

#if LOOM_AMDGPU_CHECK_HAVE_HAL_RUN_PROVIDER
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
#endif  // LOOM_AMDGPU_CHECK_HAVE_HAL_RUN_PROVIDER

static const loom_check_emit_provider_t* const kLoomAmdgpuCheckEmitProviders[] =
    {
        &loom_amdgpu_native_loom_check_emit_provider,
};

static const loom_target_coverage_provider_t* const
    kLoomAmdgpuCoverageProviders[] = {
        &loom_amdgpu_target_coverage_provider,
};

static const loom_target_low_legality_provider_t* const
    kLoomAmdgpuLowLegalityProviders[] = {
        &loom_amdgpu_low_legality_provider_storage,
};

#if LOOM_AMDGPU_CHECK_HAVE_HAL_RUN_PROVIDER
static const loom_check_requirement_provider_t* const
    kLoomAmdgpuCheckRequirementProviders[] = {
        &loom_amdgpu_loom_check_requirement_provider,
};
#endif  // LOOM_AMDGPU_CHECK_HAVE_HAL_RUN_PROVIDER

const loom_check_provider_t loom_amdgpu_check_provider = {
    .name = IREE_SVL("amdgpu"),
    .initialize_low_descriptor_registry =
        loom_amdgpu_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_amdgpu_low_lower_policy_registry_initialize,
    .low_legality_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuLowLegalityProviders),
            .values = kLoomAmdgpuLowLegalityProviders,
        },
    .emit_providers = kLoomAmdgpuCheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomAmdgpuCheckEmitProviders),
    .coverage_providers = kLoomAmdgpuCoverageProviders,
    .coverage_provider_count = IREE_ARRAYSIZE(kLoomAmdgpuCoverageProviders),
#if LOOM_AMDGPU_CHECK_HAVE_HAL_RUN_PROVIDER
    .run_providers = kLoomAmdgpuCheckRunProviders,
    .run_provider_count = IREE_ARRAYSIZE(kLoomAmdgpuCheckRunProviders),
    .requirement_providers = kLoomAmdgpuCheckRequirementProviders,
    .requirement_provider_count =
        IREE_ARRAYSIZE(kLoomAmdgpuCheckRequirementProviders),
#endif  // LOOM_AMDGPU_CHECK_HAVE_HAL_RUN_PROVIDER
};
