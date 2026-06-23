// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic HAL runtime setup for Loom execution sessions.

#ifndef LOOM_TOOLING_EXECUTION_HAL_RUNTIME_H_
#define LOOM_TOOLING_EXECUTION_HAL_RUNTIME_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared HAL runtime state created by the execution layer.
typedef struct loom_run_hal_runtime_t {
  // Selected HAL device used for executable preparation and dispatch.
  iree_hal_device_t* device;
  // Topology group assigning frontier state to |device|.
  iree_hal_device_group_t* device_group;
  // Executable cache owned by |device| and used for target probing/loading.
  iree_hal_executable_cache_t* executable_cache;
} loom_run_hal_runtime_t;

typedef struct loom_run_hal_runtime_options_t {
  // HAL driver component of the selected `--device=` URI.
  iree_string_view_t hal_driver_name;
  // Device event sink used for feedback emitted by the HAL device.
  iree_hal_device_event_sink_t event_sink;
} loom_run_hal_runtime_options_t;

// Initializes |out_options| with default HAL runtime creation policy.
void loom_run_hal_runtime_options_initialize(
    iree_string_view_t hal_driver_name,
    loom_run_hal_runtime_options_t* out_options);

// Initializes the HAL runtime state using |options|.
iree_status_t loom_run_hal_runtime_initialize(
    const loom_run_hal_runtime_options_t* options, iree_allocator_t allocator,
    loom_run_hal_runtime_t* out_runtime);

// Releases all resources owned by |runtime|.
void loom_run_hal_runtime_deinitialize(loom_run_hal_runtime_t* runtime);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_RUNTIME_H_
