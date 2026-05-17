// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared iree-tune-loom command-line implementation.

#ifndef LOOM_TOOLS_IREE_TUNE_LOOM_MAIN_H_
#define LOOM_TOOLS_IREE_TUNE_LOOM_MAIN_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_artifact_provider_registry_t
    loom_run_hal_artifact_provider_registry_t;
typedef struct loom_target_environment_t loom_target_environment_t;

typedef struct iree_tune_loom_configuration_t {
  // Null-terminated executable name used in help and diagnostics.
  const char* tool_name;
  // Target-owned dialect and interface registration callback.
  loom_run_register_context_callback_t register_context;
  // Target environment linked into this runner.
  const loom_target_environment_t* target_environment;
  // HAL artifact provider registry linked into this runner.
  const loom_run_hal_artifact_provider_registry_t*
      hal_artifact_provider_registry;
  // Target-low descriptor registry package linked into this runner.
  loom_run_initialize_low_descriptor_registry_callback_t
      initialize_low_descriptor_registry;
} iree_tune_loom_configuration_t;

// Runs the configured iree-tune-loom command-line tool.
int iree_tune_loom_main(int argc, char** argv,
                        const iree_tune_loom_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_TUNE_LOOM_MAIN_H_
