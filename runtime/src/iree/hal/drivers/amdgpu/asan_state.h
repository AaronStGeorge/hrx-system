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

typedef struct iree_hal_amdgpu_asan_application_range_t {
  // Device-visible application address assigned to this range.
  uint64_t address;

  // Length of the application range in bytes.
  iree_device_size_t length;

  // Next range in the sorted free list while owned by ASAN state.
  struct iree_hal_amdgpu_asan_application_range_t* next;
} iree_hal_amdgpu_asan_application_range_t;

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

  // Borrowed HSA API table used to release the application reservation.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Host allocator used for ASAN state-owned range nodes.
  iree_allocator_t host_allocator;

  // Base pointer of the ASAN-covered application virtual address reservation.
  IREE_AMDGPU_DEVICE_PTR void* application_reservation_base_ptr;

  // Size of the ASAN-covered application virtual address reservation.
  iree_device_size_t application_reservation_size;

  // Guards application free-list mutation.
  iree_slim_mutex_t application_mutex;

  // Sorted list of free application ranges within the ASAN-covered window.
  iree_hal_amdgpu_asan_application_range_t* application_free_ranges;
} iree_hal_amdgpu_asan_state_t;

// Initializes |out_state| from logical-device options.
//
// When ASAN is disabled this leaves |out_state| zeroed and returns OK. When
// enabled the state reserves a device-visible application virtual address
// window, reserves the corresponding shadow virtual address range, and
// configures later physical shadow slab mappings to use the first physical
// device's coarse-grained VRAM pool with shared topology access.
iree_status_t iree_hal_amdgpu_asan_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_allocator_t host_allocator, iree_hal_amdgpu_asan_state_t* out_state);

// Releases any ASAN application-window and shadow resources owned by |state|.
void iree_hal_amdgpu_asan_state_deinitialize(
    iree_hal_amdgpu_asan_state_t* state);

// Returns true when |state| owns enabled ASAN resources.
bool iree_hal_amdgpu_asan_state_is_enabled(
    const iree_hal_amdgpu_asan_state_t* state);

// Returns the mutable ASAN shadow map, or NULL when ASAN is disabled.
iree_hal_amdgpu_shadow_map_t* iree_hal_amdgpu_asan_state_shadow_map(
    iree_hal_amdgpu_asan_state_t* state);

// Ensures shadow slabs are mapped for the given device-visible application
// byte range.
//
// When ASAN is disabled this is a no-op. When enabled the range must fit within
// the configured application window; out-of-window allocations fail loudly
// because instrumented kernels would otherwise fault when loading shadow bytes.
iree_status_t iree_hal_amdgpu_asan_state_map_range(
    iree_hal_amdgpu_asan_state_t* state, uint64_t application_address,
    iree_device_size_t application_length);

// Marks an ASAN-owned mapped allocation as addressable followed by redzone.
//
// |accessible_length| bytes beginning at |application_address| become
// addressable. The remaining bytes up to |mapped_length| become poisoned
// redzone. The range must have been assigned by
// iree_hal_amdgpu_asan_state_reserve_application_range.
iree_status_t iree_hal_amdgpu_asan_state_publish_allocated_range(
    iree_hal_amdgpu_asan_state_t* state, uint64_t application_address,
    iree_device_size_t accessible_length, iree_device_size_t mapped_length);

// Re-poisons a previously published mapped allocation before teardown.
//
// This performs no host allocations and asserts HSA shadow writes succeed.
// |application_address| and |mapped_length| must match a range successfully
// published by iree_hal_amdgpu_asan_state_publish_allocated_range.
void iree_hal_amdgpu_asan_state_publish_released_range(
    iree_hal_amdgpu_asan_state_t* state, uint64_t application_address,
    iree_device_size_t mapped_length);

// Assigns an application virtual address range from the ASAN-covered window.
//
// The state owns the overall HSA VMM reservation. Callers must map physical HSA
// VMM handles into the returned subrange before exposing it to kernels. The
// returned address is aligned to |alignment| when non-zero. Callers must return
// |out_application_range| with
// iree_hal_amdgpu_asan_state_release_application_range once the range is no
// longer mapped.
iree_status_t iree_hal_amdgpu_asan_state_reserve_application_range(
    iree_hal_amdgpu_asan_state_t* state, iree_device_size_t length,
    iree_device_size_t alignment, uint64_t* out_application_address,
    iree_hal_amdgpu_asan_application_range_t** out_application_range);

// Releases an application range previously reserved from |state|.
//
// This performs no host allocations and is safe to call from cleanup paths that
// have no status channel. |application_range| is consumed by the ASAN state.
void iree_hal_amdgpu_asan_state_release_application_range(
    iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_application_range_t* application_range);

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
