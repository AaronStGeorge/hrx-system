// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared iree-run-loom command-line implementation.

#ifndef LOOM_TOOLS_IREE_RUN_LOOM_MAIN_H_
#define LOOM_TOOLS_IREE_RUN_LOOM_MAIN_H_

#include "iree/base/api.h"
#include "loom/tooling/input/input.h"
#include "loom/tooling/input/loomc.h"
#include "loomc/iree.h"
#include "loomc/target/iree_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates the compiler target environment linked into the final runner.
typedef loomc_status_t (*iree_run_loom_create_target_environment_fn_t)(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment);

typedef struct iree_run_loom_configuration_t {
  // Borrowed optional source importers selected by the final application.
  loom_input_provider_list_t input_providers;
  // Null-terminated executable name used in help and diagnostics.
  const char* tool_name;
  // Factory for the runner's artifact-capable target environment.
  iree_run_loom_create_target_environment_fn_t create_target_environment;
  // Optional foreign-source importer dispatch.
  loom_tooling_input_import_loomc_fn_t import;
  // Opaque state forwarded to |import|.
  void* import_user_data;
  // Ordered target providers that bind live HAL devices to compiler profiles.
  const loomc_iree_hal_target_provider_t* const* hal_target_providers;
  // Number of entries in |hal_target_providers|.
  loomc_host_size_t hal_target_provider_count;
} iree_run_loom_configuration_t;

// Runs the configured iree-run-loom command-line tool.
int iree_run_loom_main(int argc, char** argv,
                       const iree_run_loom_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_RUN_LOOM_MAIN_H_
