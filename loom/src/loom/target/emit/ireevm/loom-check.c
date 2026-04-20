// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check runner for IREE VM target-owned .loom-test files.

#include "loom/target/emit/ireevm/loom_check_run.h"
#include "loom/target/emit/ireevm/low_registry.h"
#include "loom/target/emit/ireevm/lower.h"
#include "loom/tools/loom-check/main.h"

static const loom_check_run_provider_t* const kLoomCheckIreeVmRunProviders[] = {
    &loom_ireevm_loom_check_run_provider,
};

static const loom_check_production_runner_t kLoomCheckIreeVmRunner = {
    .initialize_low_descriptor_registry =
        loom_ireevm_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_ireevm_low_lower_policy_registry_initialize,
    .run_providers = kLoomCheckIreeVmRunProviders,
    .run_provider_count = IREE_ARRAYSIZE(kLoomCheckIreeVmRunProviders),
};

int main(int argc, char** argv) {
  return loom_check_production_main(argc, argv, &kLoomCheckIreeVmRunner);
}
