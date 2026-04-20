// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/execution_provider.h"

#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/amdgpu_hal_backend.h"
#include "loom/tooling/execution/hal_execution_backend.h"

static const loom_run_hal_backend_t* const kLoomAmdgpuExecutionHalBackends[] = {
    &iree_run_loom_amdgpu_hal_backend,
};

static const loom_run_hal_execution_backend_t kLoomAmdgpuHalExecutionBackend = {
    .base =
        {
            .name = IREE_SVL("amdgpu-hal"),
            .flags = LOOM_RUN_EXECUTION_BACKEND_FLAG_HAL_OPTIONS,
            .probe = loom_run_hal_execution_backend_probe,
            .run_one_shot = loom_run_hal_execution_backend_run_one_shot,
        },
    .hal_backend = &iree_run_loom_amdgpu_hal_backend,
};

static const loom_run_execution_backend_t* const
    kLoomAmdgpuExecutionBackends[] = {
        &kLoomAmdgpuHalExecutionBackend.base,
};

const loom_run_execution_provider_t loom_amdgpu_target_provider = {
    .name = IREE_SVL("amdgpu"),
    .initialize_low_descriptor_registry =
        loom_amdgpu_low_descriptor_registry_initialize,
    .hal_backends = kLoomAmdgpuExecutionHalBackends,
    .hal_backend_count = IREE_ARRAYSIZE(kLoomAmdgpuExecutionHalBackends),
    .execution_backends = kLoomAmdgpuExecutionBackends,
    .execution_backend_count = IREE_ARRAYSIZE(kLoomAmdgpuExecutionBackends),
};
