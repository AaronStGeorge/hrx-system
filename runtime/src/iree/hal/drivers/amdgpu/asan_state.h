// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ASAN_STATE_H_
#define IREE_HAL_DRIVERS_AMDGPU_ASAN_STATE_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/util/shadow_map.h"

typedef struct iree_hal_amdgpu_physical_device_t
    iree_hal_amdgpu_physical_device_t;
typedef struct iree_hal_amdgpu_system_t iree_hal_amdgpu_system_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_asan_state_t
//===----------------------------------------------------------------------===//

// Logical-device ASAN state shared by allocator, executable, and queue paths.
typedef struct iree_hal_amdgpu_asan_state_t {
  // True when ASAN support is enabled for the logical device.
  uint32_t is_enabled : 1;

  // Sparse shadow map used by ASAN-instrumented device code.
  iree_hal_amdgpu_shadow_map_t shadow_map;
} iree_hal_amdgpu_asan_state_t;

// Initializes |out_state| from logical-device options.
//
// When ASAN is disabled this leaves |out_state| zeroed and returns OK. When
// enabled the state reserves the device-visible shadow virtual address range
// and configures later physical shadow slab mappings to use the first physical
// device's coarse-grained VRAM pool with shared topology access.
iree_status_t iree_hal_amdgpu_asan_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_allocator_t host_allocator, iree_hal_amdgpu_asan_state_t* out_state);

// Releases any ASAN shadow resources owned by |state|.
void iree_hal_amdgpu_asan_state_deinitialize(
    iree_hal_amdgpu_asan_state_t* state);

// Returns true when |state| owns enabled ASAN resources.
bool iree_hal_amdgpu_asan_state_is_enabled(
    const iree_hal_amdgpu_asan_state_t* state);

// Returns the mutable ASAN shadow map, or NULL when ASAN is disabled.
iree_hal_amdgpu_shadow_map_t* iree_hal_amdgpu_asan_state_shadow_map(
    iree_hal_amdgpu_asan_state_t* state);

// Populates |out_config| with the device-visible ASAN configuration.
//
// Callers must only call this when ASAN is enabled. Report fields remain zero
// until the report transport is attached to the logical-device ASAN state.
void iree_hal_amdgpu_asan_state_populate_config(
    const iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_config_t* out_config);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_ASAN_STATE_H_
