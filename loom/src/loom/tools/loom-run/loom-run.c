// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-run binary with build-selected compiler integrations.

#include <stddef.h>

#include "loom/tooling/input/loomc_configured.h"
#include "loom/tools/loom-run/main.h"
#include "loomc/target/configured.h"

#ifndef LOOM_RUN_HAVE_AMDGPU
#define LOOM_RUN_HAVE_AMDGPU 0
#endif  // LOOM_RUN_HAVE_AMDGPU
#ifndef LOOM_RUN_HAVE_SPIRV
#define LOOM_RUN_HAVE_SPIRV 0
#endif  // LOOM_RUN_HAVE_SPIRV
#ifndef LOOM_RUN_HAVE_TASK
#define LOOM_RUN_HAVE_TASK 0
#endif  // LOOM_RUN_HAVE_TASK

#define LOOM_RUN_HAVE_ANY_PROVIDER \
  (LOOM_RUN_HAVE_AMDGPU || LOOM_RUN_HAVE_SPIRV || LOOM_RUN_HAVE_TASK)

#if LOOM_RUN_HAVE_AMDGPU
#include "loomc/target/amdgpu/iree_hal.h"
#endif  // LOOM_RUN_HAVE_AMDGPU
#if LOOM_RUN_HAVE_SPIRV
#include "loomc/target/spirv/iree_hal.h"
#endif  // LOOM_RUN_HAVE_SPIRV
#if LOOM_RUN_HAVE_TASK
#include "loomc/target/cpu/iree_hal.h"
#endif  // LOOM_RUN_HAVE_TASK

#if LOOM_RUN_HAVE_ANY_PROVIDER
static const loomc_iree_hal_target_provider_t* kLoomRunTargetProviders[3];
#endif  // LOOM_RUN_HAVE_ANY_PROVIDER

int main(int argc, char** argv) {
  iree_host_size_t target_provider_count = 0;
#if LOOM_RUN_HAVE_AMDGPU
  kLoomRunTargetProviders[target_provider_count++] =
      loomc_amdgpu_iree_hal_target_provider();
#endif  // LOOM_RUN_HAVE_AMDGPU
#if LOOM_RUN_HAVE_SPIRV
  kLoomRunTargetProviders[target_provider_count++] =
      loomc_spirv_iree_hal_target_provider();
#endif  // LOOM_RUN_HAVE_SPIRV
#if LOOM_RUN_HAVE_TASK
  kLoomRunTargetProviders[target_provider_count++] =
      loomc_cpu_iree_hal_target_provider();
#endif  // LOOM_RUN_HAVE_TASK

  const loom_run_configuration_t configuration = {
      .input_formats = loom_configured_input_loomc_formats(),
      .tool_name = "loom-run",
      .create_target_environment = loomc_target_environment_create_configured,
      .import = loom_configured_input_loomc_importer(),
#if LOOM_RUN_HAVE_ANY_PROVIDER
      .hal_target_providers = kLoomRunTargetProviders,
#endif  // LOOM_RUN_HAVE_ANY_PROVIDER
      .hal_target_provider_count = target_provider_count,
  };
  return loom_run_main(argc, argv, &configuration);
}
