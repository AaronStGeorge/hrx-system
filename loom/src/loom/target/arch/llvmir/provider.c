// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/llvmir/provider.h"

#include "loom/ops/llvmir/registry.h"

const loom_target_provider_t loom_llvmir_target_provider = {
    .register_context = loom_llvmir_ops_register_dialect,
};

static const loom_target_provider_t* const kLoomLlvmirTargetProviders[] = {
    &loom_llvmir_target_provider,
};

const loom_target_provider_set_t loom_llvmir_target_provider_set = {
    .providers = kLoomLlvmirTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomLlvmirTargetProviders),
};
