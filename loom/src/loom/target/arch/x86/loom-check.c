// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check runner for x86 target-owned .loom-test files.

#include "loom/target/arch/x86/low_registry.h"
#include "loom/target/arch/x86/lower.h"
#include "loom/tools/loom-check/main.h"

static const loom_check_production_runner_t kLoomCheckX86Runner = {
    .initialize_low_descriptor_registry =
        loom_x86_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_x86_low_lower_policy_registry_initialize,
};

int main(int argc, char** argv) {
  return loom_check_production_main(argc, argv, &kLoomCheckX86Runner);
}
