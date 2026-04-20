// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check runner for AMDGPU target-owned .loom-test files.

#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/lower.h"
#include "loom/tools/loom-check/main.h"

static const loom_check_production_runner_t kLoomCheckAmdgpuRunner = {
    .initialize_low_descriptor_registry =
        loom_amdgpu_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_amdgpu_low_lower_policy_registry_initialize,
};

int main(int argc, char** argv) {
  return loom_check_production_main(argc, argv, &kLoomCheckAmdgpuRunner);
}
