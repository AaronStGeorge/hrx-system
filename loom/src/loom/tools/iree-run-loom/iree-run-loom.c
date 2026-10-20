// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-run-loom binary with build-selected compiler integrations.

#include <stddef.h>

#include "loom/tooling/input/configured.h"
#include "loom/tooling/input/loomc_configured.h"
#include "loom/tools/iree-run-loom/main.h"
#include "loomc/target/configured.h"

#ifndef IREE_RUN_LOOM_HAVE_AMDGPU
#define IREE_RUN_LOOM_HAVE_AMDGPU 0
#endif  // IREE_RUN_LOOM_HAVE_AMDGPU
#ifndef IREE_RUN_LOOM_HAVE_SPIRV
#define IREE_RUN_LOOM_HAVE_SPIRV 0
#endif  // IREE_RUN_LOOM_HAVE_SPIRV
#ifndef IREE_RUN_LOOM_HAVE_TASK
#define IREE_RUN_LOOM_HAVE_TASK 0
#endif  // IREE_RUN_LOOM_HAVE_TASK

#define IREE_RUN_LOOM_HAVE_ANY_PROVIDER                     \
  (IREE_RUN_LOOM_HAVE_AMDGPU || IREE_RUN_LOOM_HAVE_SPIRV || \
   IREE_RUN_LOOM_HAVE_TASK)

#if IREE_RUN_LOOM_HAVE_AMDGPU
#include "loomc/target/amdgpu/iree_hal.h"
#endif  // IREE_RUN_LOOM_HAVE_AMDGPU
#if IREE_RUN_LOOM_HAVE_SPIRV
#include "loomc/target/spirv/iree_hal.h"
#endif  // IREE_RUN_LOOM_HAVE_SPIRV
#if IREE_RUN_LOOM_HAVE_TASK
#include "loomc/target/cpu/iree_hal.h"
#endif  // IREE_RUN_LOOM_HAVE_TASK

#if IREE_RUN_LOOM_HAVE_ANY_PROVIDER
static const loomc_iree_hal_target_provider_t* kIreeRunLoomTargetProviders[3];
#endif  // IREE_RUN_LOOM_HAVE_ANY_PROVIDER

int main(int argc, char** argv) {
  iree_host_size_t target_provider_count = 0;
#if IREE_RUN_LOOM_HAVE_AMDGPU
  kIreeRunLoomTargetProviders[target_provider_count++] =
      loomc_amdgpu_iree_hal_target_provider();
#endif  // IREE_RUN_LOOM_HAVE_AMDGPU
#if IREE_RUN_LOOM_HAVE_SPIRV
  kIreeRunLoomTargetProviders[target_provider_count++] =
      loomc_spirv_iree_hal_target_provider();
#endif  // IREE_RUN_LOOM_HAVE_SPIRV
#if IREE_RUN_LOOM_HAVE_TASK
  kIreeRunLoomTargetProviders[target_provider_count++] =
      loomc_cpu_iree_hal_target_provider();
#endif  // IREE_RUN_LOOM_HAVE_TASK

  const iree_run_loom_configuration_t configuration = {
      .input_providers = loom_configured_input_providers(),
      .tool_name = "iree-run-loom",
      .create_target_environment = loomc_target_environment_create_configured,
      .import = loom_configured_input_loomc_importer(),
#if IREE_RUN_LOOM_HAVE_ANY_PROVIDER
      .hal_target_providers = kIreeRunLoomTargetProviders,
#endif  // IREE_RUN_LOOM_HAVE_ANY_PROVIDER
      .hal_target_provider_count = target_provider_count,
  };
  return iree_run_loom_main(argc, argv, &configuration);
}
