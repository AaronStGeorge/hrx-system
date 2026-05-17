// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-test-loom binary with build-selected execution providers.

#include <stddef.h>
#include <stdio.h>

#include "loom/tooling/execution/execution_provider.h"
#include "loom/tools/iree-test-loom/main.h"

#ifndef IREE_TEST_LOOM_HAVE_AMDGPU
#define IREE_TEST_LOOM_HAVE_AMDGPU 0
#endif  // IREE_TEST_LOOM_HAVE_AMDGPU
#ifndef IREE_TEST_LOOM_HAVE_IREEVM
#define IREE_TEST_LOOM_HAVE_IREEVM 0
#endif  // IREE_TEST_LOOM_HAVE_IREEVM
#ifndef IREE_TEST_LOOM_HAVE_SPIRV
#define IREE_TEST_LOOM_HAVE_SPIRV 0
#endif  // IREE_TEST_LOOM_HAVE_SPIRV

#define IREE_TEST_LOOM_HAVE_ANY_PROVIDER                       \
  (IREE_TEST_LOOM_HAVE_AMDGPU || IREE_TEST_LOOM_HAVE_IREEVM || \
   IREE_TEST_LOOM_HAVE_SPIRV)
#define IREE_TEST_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER \
  (IREE_TEST_LOOM_HAVE_AMDGPU || IREE_TEST_LOOM_HAVE_SPIRV)

#if IREE_TEST_LOOM_HAVE_AMDGPU
#include "loom/tooling/execution/hal/amdgpu/artifact_provider.h"
#include "loom/tooling/execution/hal/amdgpu/provider.h"
#endif  // IREE_TEST_LOOM_HAVE_AMDGPU
#if IREE_TEST_LOOM_HAVE_IREEVM
#include "loom/tooling/execution/ireevm/provider.h"
#endif  // IREE_TEST_LOOM_HAVE_IREEVM
#if IREE_TEST_LOOM_HAVE_SPIRV
#include "loom/tooling/execution/hal/spirv/artifact_provider.h"
#include "loom/tooling/execution/hal/spirv/provider.h"
#endif  // IREE_TEST_LOOM_HAVE_SPIRV

#if IREE_TEST_LOOM_HAVE_ANY_PROVIDER
static const loom_run_execution_provider_t* const kIreeTestLoomProviders[] = {
#if IREE_TEST_LOOM_HAVE_AMDGPU
    &loom_amdgpu_hal_execution_provider,
#endif  // IREE_TEST_LOOM_HAVE_AMDGPU
#if IREE_TEST_LOOM_HAVE_IREEVM
    &loom_ireevm_execution_provider,
#endif  // IREE_TEST_LOOM_HAVE_IREEVM
#if IREE_TEST_LOOM_HAVE_SPIRV
    &loom_spirv_vulkan_hal_execution_provider,
#endif  // IREE_TEST_LOOM_HAVE_SPIRV
};
#endif  // IREE_TEST_LOOM_HAVE_ANY_PROVIDER

static const loom_run_execution_provider_set_t kIreeTestLoomProviderSet = {
#if IREE_TEST_LOOM_HAVE_ANY_PROVIDER
    .providers = kIreeTestLoomProviders,
    .provider_count = IREE_ARRAYSIZE(kIreeTestLoomProviders),
#else
    .providers = NULL,
    .provider_count = 0,
#endif  // IREE_TEST_LOOM_HAVE_ANY_PROVIDER
};

#if IREE_TEST_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER
static const loom_run_hal_artifact_provider_t* const
    kIreeTestLoomHalArtifactProviders[] = {
#if IREE_TEST_LOOM_HAVE_AMDGPU
        &loom_amdgpu_hal_artifact_provider,
#endif  // IREE_TEST_LOOM_HAVE_AMDGPU
#if IREE_TEST_LOOM_HAVE_SPIRV
        &loom_spirv_vulkan_hal_artifact_provider,
#endif  // IREE_TEST_LOOM_HAVE_SPIRV
};
#endif  // IREE_TEST_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER

static const loom_run_hal_artifact_provider_registry_t
    kIreeTestLoomHalArtifactProviderRegistry = {
#if IREE_TEST_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER
        .providers = kIreeTestLoomHalArtifactProviders,
        .provider_count = IREE_ARRAYSIZE(kIreeTestLoomHalArtifactProviders),
#else
        .providers = NULL,
        .provider_count = 0,
#endif  // IREE_TEST_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER
};

int main(int argc, char** argv) {
  loom_run_execution_environment_t environment;
  iree_status_t status = loom_run_execution_environment_initialize(
      &kIreeTestLoomProviderSet, &environment);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }

  const iree_test_loom_configuration_t configuration = {
      .tool_name = "iree-test-loom",
      .register_context =
          loom_run_execution_environment_register_context_callback(
              &environment),
      .target_environment =
          loom_run_execution_environment_target_environment(&environment),
      .hal_artifact_provider_registry =
          &kIreeTestLoomHalArtifactProviderRegistry,
      .initialize_low_descriptor_registry =
          loom_run_execution_environment_low_descriptor_registry_callback(
              &environment),
  };
  int exit_code = iree_test_loom_main(argc, argv, &configuration);
  loom_run_execution_environment_deinitialize(&environment);
  return exit_code;
}
