// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_TSAN_STATE_H_
#define IREE_HAL_DRIVERS_AMDGPU_TSAN_STATE_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

typedef struct iree_hal_amdgpu_physical_device_t
    iree_hal_amdgpu_physical_device_t;
typedef struct iree_hal_amdgpu_system_t iree_hal_amdgpu_system_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_tsan_state_t
//===----------------------------------------------------------------------===//

// Per-physical-device TSAN shadow state.
typedef struct iree_hal_amdgpu_tsan_device_state_t {
  // Physical device ordinal associated with this shadow state.
  iree_host_size_t physical_device_ordinal;

  // Host-accessible device-visible shadow allocation.
  void* shadow_base;

  // Shadow allocation byte length.
  iree_device_size_t shadow_size;

  // Device-visible config published into executable globals.
  iree_hal_amdgpu_tsan_config_t config;
} iree_hal_amdgpu_tsan_device_state_t;

// Logical-device TSAN state shared by executable and queue paths.
typedef struct iree_hal_amdgpu_tsan_state_t {
  // True when TSAN support is enabled for the logical device.
  uint32_t is_enabled : 1;

  // Borrowed HSA API table.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Host allocator used for state-owned allocations.
  iree_allocator_t host_allocator;

  // Number of entries in |device_states|.
  iree_host_size_t device_state_count;

  // Per-physical-device TSAN shadow allocations.
  iree_hal_amdgpu_tsan_device_state_t* device_states;
} iree_hal_amdgpu_tsan_state_t;

// Initializes |out_state| from logical-device options.
//
// When TSAN is disabled this leaves |out_state| zeroed and returns OK.
iree_status_t iree_hal_amdgpu_tsan_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_allocator_t host_allocator, iree_hal_amdgpu_tsan_state_t* out_state);

// Releases all TSAN resources owned by |state|.
void iree_hal_amdgpu_tsan_state_deinitialize(
    iree_hal_amdgpu_tsan_state_t* state);

// Returns true when |state| owns enabled TSAN resources.
bool iree_hal_amdgpu_tsan_state_is_enabled(
    const iree_hal_amdgpu_tsan_state_t* state);

// Populates |out_config| with the device-visible TSAN configuration.
//
// Callers must only call this when TSAN is enabled.
// |physical_device_ordinal| selects the physical device whose executable global
// is being assigned.
iree_status_t iree_hal_amdgpu_tsan_state_populate_config(
    const iree_hal_amdgpu_tsan_state_t* state,
    iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_tsan_config_t* out_config);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_TSAN_STATE_H_
