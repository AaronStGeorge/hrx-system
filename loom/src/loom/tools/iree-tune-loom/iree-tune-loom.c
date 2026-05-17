// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-tune-loom binary with build-selected execution providers.

#include <stddef.h>
#include <stdio.h>

#include "loom/tooling/execution/execution_provider.h"
#include "loom/tooling/execution/hal/artifact.h"
#include "loom/tools/iree-tune-loom/main.h"

#ifndef IREE_TUNE_LOOM_HAVE_AMDGPU
#define IREE_TUNE_LOOM_HAVE_AMDGPU 0
#endif  // IREE_TUNE_LOOM_HAVE_AMDGPU
#ifndef IREE_TUNE_LOOM_HAVE_IREEVM
#define IREE_TUNE_LOOM_HAVE_IREEVM 0
#endif  // IREE_TUNE_LOOM_HAVE_IREEVM
#ifndef IREE_TUNE_LOOM_HAVE_SPIRV
#define IREE_TUNE_LOOM_HAVE_SPIRV 0
#endif  // IREE_TUNE_LOOM_HAVE_SPIRV

#define IREE_TUNE_LOOM_HAVE_ANY_PROVIDER                       \
  (IREE_TUNE_LOOM_HAVE_AMDGPU || IREE_TUNE_LOOM_HAVE_IREEVM || \
   IREE_TUNE_LOOM_HAVE_SPIRV)
#define IREE_TUNE_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER \
  (IREE_TUNE_LOOM_HAVE_AMDGPU || IREE_TUNE_LOOM_HAVE_SPIRV)

#if IREE_TUNE_LOOM_HAVE_AMDGPU
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/tooling/target/amdgpu/artifact_provider.h"
#endif  // IREE_TUNE_LOOM_HAVE_AMDGPU
#if IREE_TUNE_LOOM_HAVE_IREEVM
#include "loom/tooling/target/ireevm/provider.h"
#endif  // IREE_TUNE_LOOM_HAVE_IREEVM
#if IREE_TUNE_LOOM_HAVE_SPIRV
#include "loom/target/arch/spirv/provider.h"
#include "loom/tooling/target/spirv/artifact_provider.h"
#endif  // IREE_TUNE_LOOM_HAVE_SPIRV

#if IREE_TUNE_LOOM_HAVE_AMDGPU
static const loom_run_execution_provider_t kIreeTuneLoomAmdgpuProvider = {
    .name = IREE_SVL("amdgpu"),
    .target_provider = &loom_amdgpu_target_provider,
};
#endif  // IREE_TUNE_LOOM_HAVE_AMDGPU

#if IREE_TUNE_LOOM_HAVE_SPIRV
static const loom_run_execution_provider_t kIreeTuneLoomSpirvProvider = {
    .name = IREE_SVL("spirv"),
    .target_provider = &loom_spirv_target_provider,
};
#endif  // IREE_TUNE_LOOM_HAVE_SPIRV

#if IREE_TUNE_LOOM_HAVE_ANY_PROVIDER
static const loom_run_execution_provider_t* const kIreeTuneLoomProviders[] = {
#if IREE_TUNE_LOOM_HAVE_AMDGPU
    &kIreeTuneLoomAmdgpuProvider,
#endif  // IREE_TUNE_LOOM_HAVE_AMDGPU
#if IREE_TUNE_LOOM_HAVE_IREEVM
    &loom_ireevm_execution_provider,
#endif  // IREE_TUNE_LOOM_HAVE_IREEVM
#if IREE_TUNE_LOOM_HAVE_SPIRV
    &kIreeTuneLoomSpirvProvider,
#endif  // IREE_TUNE_LOOM_HAVE_SPIRV
};
#endif  // IREE_TUNE_LOOM_HAVE_ANY_PROVIDER

static const loom_run_execution_provider_set_t kIreeTuneLoomProviderSet = {
#if IREE_TUNE_LOOM_HAVE_ANY_PROVIDER
    .providers = kIreeTuneLoomProviders,
    .provider_count = IREE_ARRAYSIZE(kIreeTuneLoomProviders),
#else
    .providers = NULL,
    .provider_count = 0,
#endif  // IREE_TUNE_LOOM_HAVE_ANY_PROVIDER
};

#if IREE_TUNE_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER
static const loom_run_hal_artifact_provider_t* const
    kIreeTuneLoomHalArtifactProviders[] = {
#if IREE_TUNE_LOOM_HAVE_AMDGPU
        &loom_amdgpu_hal_artifact_provider,
#endif  // IREE_TUNE_LOOM_HAVE_AMDGPU
#if IREE_TUNE_LOOM_HAVE_SPIRV
        &loom_spirv_vulkan_hal_artifact_provider,
#endif  // IREE_TUNE_LOOM_HAVE_SPIRV
};
#endif  // IREE_TUNE_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER

static const loom_run_hal_artifact_provider_registry_t
    kIreeTuneLoomHalArtifactProviderRegistry = {
#if IREE_TUNE_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER
        .providers = kIreeTuneLoomHalArtifactProviders,
        .provider_count = IREE_ARRAYSIZE(kIreeTuneLoomHalArtifactProviders),
#else
        .providers = NULL,
        .provider_count = 0,
#endif  // IREE_TUNE_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER
};

int main(int argc, char** argv) {
  loom_run_execution_environment_t environment;
  iree_status_t status = loom_run_execution_environment_initialize(
      &kIreeTuneLoomProviderSet, &environment);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }

  const iree_tune_loom_configuration_t configuration = {
      .tool_name = "iree-tune-loom",
      .register_context =
          loom_run_execution_environment_register_context_callback(
              &environment),
      .target_environment =
          loom_run_execution_environment_target_environment(&environment),
      .hal_artifact_provider_registry =
          &kIreeTuneLoomHalArtifactProviderRegistry,
      .initialize_low_descriptor_registry =
          loom_run_execution_environment_low_descriptor_registry_callback(
              &environment),
  };
  int exit_code = iree_tune_loom_main(argc, argv, &configuration);
  loom_run_execution_environment_deinitialize(&environment);
  return exit_code;
}
