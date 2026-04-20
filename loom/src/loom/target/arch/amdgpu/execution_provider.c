// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/execution_provider.h"

#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/amdgpu_hal_backend.h"

static const loom_run_hal_backend_t* const kLoomAmdgpuExecutionHalBackends[] = {
    &iree_run_loom_amdgpu_hal_backend,
};

const loom_run_execution_provider_t loom_amdgpu_target_provider = {
    .name = IREE_SVL("amdgpu"),
    .initialize_low_descriptor_registry =
        loom_amdgpu_low_descriptor_registry_initialize,
    .hal_backends = kLoomAmdgpuExecutionHalBackends,
    .hal_backend_count = IREE_ARRAYSIZE(kLoomAmdgpuExecutionHalBackends),
};
