// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/execution_provider.h"

#include "loom/target/emit/ireevm/low_registry.h"

const loom_run_execution_provider_t loom_ireevm_execution_provider = {
    .name = IREE_SVL("ireevm"),
    .initialize_low_descriptor_registry =
        loom_ireevm_low_descriptor_registry_initialize,
};
