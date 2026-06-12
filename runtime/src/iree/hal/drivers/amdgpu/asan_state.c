// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/asan_state.h"

#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"

static iree_status_t iree_hal_amdgpu_asan_state_initialize_shadow_map(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_shadow_map_t* out_shadow_map) {
  if (IREE_UNLIKELY(physical_device_count == 0 || !physical_devices ||
                    !physical_devices[0])) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU ASAN requires at least one initialized physical device");
  }

  iree_hal_amdgpu_physical_device_t* physical_device = physical_devices[0];
  hsa_amd_memory_pool_t shadow_memory_pool =
      physical_device->coarse_block_pools.large.memory_pool;
  if (IREE_UNLIKELY(!shadow_memory_pool.handle)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU ASAN requires a coarse-grained device-local memory pool");
  }

  hsa_amd_memory_access_desc_t access_descs[IREE_HAL_AMDGPU_MAX_CPU_AGENT +
                                            IREE_HAL_AMDGPU_MAX_GPU_AGENT];
  iree_host_size_t access_desc_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmem_build_access_descs_for_topology(
      &system->topology, physical_device->device_agent,
      IREE_HAL_AMDGPU_ACCESS_MODE_SHARED, IREE_ARRAYSIZE(access_descs),
      access_descs, &access_desc_count));

  iree_hal_amdgpu_shadow_map_hsa_params_t shadow_map_params = {
      .host_allocator = host_allocator,
      .libhsa = &system->libhsa,
      .memory_pool = shadow_memory_pool,
      .memory_type = IREE_HAL_AMDGPU_VMEM_MEMORY_TYPE_DEFAULT,
      .shadow_scale_shift = options->asan.shadow_scale_shift,
      .application_window_base = 0,
      .shadow_size = options->asan.shadow_size,
      .requested_slab_size = options->asan.shadow_slab_size,
      .access_desc_count = access_desc_count,
      .access_descs = access_descs,
  };
  return iree_hal_amdgpu_shadow_map_initialize_hsa(&shadow_map_params,
                                                   out_shadow_map);
}

iree_status_t iree_hal_amdgpu_asan_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_allocator_t host_allocator, iree_hal_amdgpu_asan_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(system);
  IREE_ASSERT_ARGUMENT(out_state);
  memset(out_state, 0, sizeof(*out_state));

  if (!options->asan.enabled) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_asan_state_initialize_shadow_map(
      options, system, physical_device_count, physical_devices, host_allocator,
      &out_state->shadow_map));
  out_state->is_enabled = true;
  return iree_ok_status();
}

void iree_hal_amdgpu_asan_state_deinitialize(
    iree_hal_amdgpu_asan_state_t* state) {
  if (!state || !state->is_enabled) return;
  iree_hal_amdgpu_shadow_map_deinitialize(&state->shadow_map);
  memset(state, 0, sizeof(*state));
}

bool iree_hal_amdgpu_asan_state_is_enabled(
    const iree_hal_amdgpu_asan_state_t* state) {
  return state && state->is_enabled;
}

iree_hal_amdgpu_shadow_map_t* iree_hal_amdgpu_asan_state_shadow_map(
    iree_hal_amdgpu_asan_state_t* state) {
  return iree_hal_amdgpu_asan_state_is_enabled(state) ? &state->shadow_map
                                                      : NULL;
}

void iree_hal_amdgpu_asan_state_populate_config(
    const iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_config_t* out_config) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(out_config);
  IREE_ASSERT(iree_hal_amdgpu_asan_state_is_enabled(state));

  const iree_hal_amdgpu_shadow_map_t* shadow_map = &state->shadow_map;
  *out_config = (iree_hal_amdgpu_asan_config_t){
      .record_length = sizeof(*out_config),
      .abi_version = IREE_HAL_AMDGPU_ASAN_CONFIG_ABI_VERSION_0,
      .flags = IREE_HAL_AMDGPU_ASAN_CONFIG_FLAG_ENABLED,
      .shadow_scale_shift = shadow_map->shadow_scale_shift,
      .shadow_base = shadow_map->shadow_base,
      .application_window_base = shadow_map->application_window_base,
      .application_window_size = shadow_map->application_window_size,
      .shadow_size = shadow_map->reservation_size,
      .shadow_slab_size = shadow_map->slab_size,
      .report_ring_base = 0,
      .report_ring_size = 0,
      .reserved = {0},
  };
}
