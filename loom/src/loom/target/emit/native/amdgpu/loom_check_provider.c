// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/loom_check_provider.h"

#include "loom/target/emit/native/amdgpu/loom_check.h"

static const loom_check_emit_provider_t* const
    kLoomAmdgpuNativeCheckEmitProviders[] = {
        &loom_amdgpu_kernel_loom_check_emit_provider,
        &loom_amdgpu_native_loom_check_emit_provider,
};

const loom_check_provider_t loom_amdgpu_native_check_provider = {
    .name = IREE_SVL("amdgpu-native"),
    .emit_providers = kLoomAmdgpuNativeCheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomAmdgpuNativeCheckEmitProviders),
};
