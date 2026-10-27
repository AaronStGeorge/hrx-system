// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/check/provider.h"

#include "loom/target/arch/spirv/check/source_low.h"

static const loom_check_emit_provider_t* const kLoomSpirvCheckEmitProviders[] =
    {
        &loom_spirv_loom_check_lower_provider,
};

const loom_check_provider_t loom_spirv_arch_check_provider = {
    .name = IREE_SVL("spirv-arch"),
    .emit_providers = kLoomSpirvCheckEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomSpirvCheckEmitProviders),
};
