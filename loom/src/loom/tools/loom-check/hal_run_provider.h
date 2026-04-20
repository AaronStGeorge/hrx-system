// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic HAL RUN: run provider callbacks for target-owned loom-check runners.

#ifndef LOOM_TOOLS_LOOM_CHECK_HAL_RUN_PROVIDER_H_
#define LOOM_TOOLS_LOOM_CHECK_HAL_RUN_PROVIDER_H_

#include "loom/tooling/execution/hal_backend.h"
#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_check_hal_run_provider_config_t {
  // HAL backends linked into the owning target runner.
  loom_run_hal_backend_registry_t backend_registry;
} loom_check_hal_run_provider_config_t;

// Returns true when provider->user_data names a HAL backend matching
// --loom_backend in |arguments|.
bool loom_check_hal_run_provider_match(
    const loom_check_run_provider_t* provider,
    const loom_check_run_arguments_t* arguments);

// Compiles and invokes a Loom module through the selected HAL backend.
iree_status_t loom_check_hal_run_provider_execute(
    const loom_check_run_provider_t* provider,
    const loom_check_run_provider_request_t* request);

// Appends the configured HAL backend names.
iree_status_t loom_check_hal_run_provider_append_names(
    const loom_check_run_provider_t* provider, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_HAL_RUN_PROVIDER_H_
