// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic HAL executable invocation helpers for Loom execution sessions.

#ifndef LOOM_TOOLING_EXECUTION_HAL_INVOCATION_H_
#define LOOM_TOOLING_EXECUTION_HAL_INVOCATION_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "loom/tooling/execution/hal_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Maximum number of HAL dispatch bindings accepted by the generic runner.
  LOOM_RUN_HAL_MAX_BINDING_COUNT = 64,
};

typedef struct loom_run_hal_invocation_options_t {
  // HAL executable entry point ordinal to dispatch.
  uint32_t entry_point;
  // Dispatch workgroup count in x, y, z order.
  uint32_t workgroup_count[3];
} loom_run_hal_invocation_options_t;

// Initializes invocation options to entry point 0 and a single workgroup.
void loom_run_hal_invocation_options_initialize(
    loom_run_hal_invocation_options_t* out_options);

// Prepares a HAL executable object from backend-produced executable bytes.
iree_status_t loom_run_hal_executable_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_executable_t* executable,
    iree_hal_executable_t** out_hal_executable);

// Dispatches a prepared HAL executable with |binding_list|.
iree_status_t loom_run_hal_dispatch(
    iree_hal_device_t* device, iree_hal_executable_t* executable,
    iree_vm_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options);

// Prepares and dispatches |executable| through |runtime|.
iree_status_t loom_run_hal_invocation_execute(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_executable_t* executable, iree_vm_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options);

// Transfers dispatch bindings back to host-visible storage for inspection.
iree_status_t loom_run_hal_transfer_bindings_to_host(
    const loom_run_hal_runtime_t* runtime, iree_vm_list_t* binding_list);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_INVOCATION_H_
