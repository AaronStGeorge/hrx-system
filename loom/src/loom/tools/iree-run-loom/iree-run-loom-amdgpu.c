// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-run-loom binary with the AMDGPU HAL provider linked in.

#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/tools/iree-run-loom/amdgpu_hal_backend.h"
#include "loom/tools/iree-run-loom/main.h"

static iree_status_t iree_run_loom_initialize_amdgpu_low_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(out_registry);
  loom_amdgpu_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

int main(int argc, char** argv) {
  static const loom_run_hal_backend_t* const kHalBackends[] = {
      &iree_run_loom_amdgpu_hal_backend,
  };
  loom_run_hal_backend_registry_t hal_backend_registry;
  loom_run_hal_backend_registry_initialize_from_entries(
      kHalBackends, IREE_ARRAYSIZE(kHalBackends), &hal_backend_registry);
  const iree_run_loom_configuration_t configuration = {
      .tool_name = "iree-run-loom-amdgpu",
      .initialize_low_descriptor_registry =
          {
              .fn = iree_run_loom_initialize_amdgpu_low_registry,
          },
      .hal_backend_registry = hal_backend_registry,
  };
  return iree_run_loom_main(argc, argv, &configuration);
}
