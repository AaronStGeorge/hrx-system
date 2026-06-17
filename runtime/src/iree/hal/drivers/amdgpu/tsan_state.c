// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/tsan_state.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"

static iree_status_t iree_hal_amdgpu_tsan_state_calculate_layout(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_device_size_t* out_workgroup_shadow_stride,
    iree_device_size_t* out_shadow_size) {
  *out_workgroup_shadow_stride = 0;
  *out_shadow_size = 0;

  const iree_device_size_t granule_size = (iree_device_size_t)1ull
                                          << options->tsan.memory_granule_shift;
  const iree_device_size_t workgroup_entry_count = iree_device_size_ceil_div(
      options->tsan.workgroup_local_memory_size, granule_size);
  iree_device_size_t workgroup_shadow_stride = 0;
  if (!iree_device_size_checked_mul(workgroup_entry_count,
                                    IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_SIZE,
                                    &workgroup_shadow_stride)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU TSAN workgroup shadow size overflow: entry_count=%" PRIu64,
        (uint64_t)workgroup_entry_count);
  }
  iree_device_size_t shadow_size = 0;
  if (!iree_device_size_checked_mul(workgroup_shadow_stride,
                                    options->tsan.workgroup_capacity,
                                    &shadow_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU TSAN dispatch shadow size overflow: "
        "workgroup_shadow_stride=%" PRIu64 ", workgroup_capacity=%u",
        (uint64_t)workgroup_shadow_stride, options->tsan.workgroup_capacity);
  }

  *out_workgroup_shadow_stride = workgroup_shadow_stride;
  *out_shadow_size = shadow_size;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_tsan_state_initialize_device(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system,
    iree_hal_amdgpu_physical_device_t* physical_device,
    iree_device_size_t workgroup_shadow_stride, iree_device_size_t shadow_size,
    iree_hal_amdgpu_tsan_device_state_t* out_device_state) {
  memset(out_device_state, 0, sizeof(*out_device_state));

  if (IREE_UNLIKELY(shadow_size == 0 ||
                    (iree_device_size_t)(size_t)shadow_size != shadow_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU TSAN shadow allocation size is invalid: %" PRIu64,
        (uint64_t)shadow_size);
  }

  hsa_amd_memory_pool_t shadow_memory_pool =
      physical_device->host_memory_pools.fine_pool;
  if (IREE_UNLIKELY(!shadow_memory_pool.handle)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU TSAN requires a host fine-grained shadow memory pool");
  }

  void* shadow_base = NULL;
  iree_status_t status = iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(&system->libhsa), shadow_memory_pool, (size_t)shadow_size,
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG, &shadow_base);
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_agents_allow_access(
        IREE_LIBHSA(&system->libhsa), /*num_agents=*/1,
        &physical_device->device_agent, /*flags=*/NULL, shadow_base);
  }
  if (iree_status_is_ok(status)) {
    memset(shadow_base, 0, (size_t)shadow_size);
    out_device_state->physical_device_ordinal = physical_device->device_ordinal;
    out_device_state->shadow_base = shadow_base;
    out_device_state->shadow_size = shadow_size;
    out_device_state->config = (iree_hal_amdgpu_tsan_config_t){
        .record_length = sizeof(out_device_state->config),
        .abi_version = IREE_HAL_AMDGPU_TSAN_CONFIG_ABI_VERSION_0,
        .flags = IREE_HAL_AMDGPU_TSAN_CONFIG_FLAG_ENABLED,
        .memory_granule_shift = options->tsan.memory_granule_shift,
        .shadow_base = (uint64_t)(uintptr_t)shadow_base,
        .shadow_size = shadow_size,
        .dispatch_shadow_stride = shadow_size,
        .workgroup_shadow_stride = workgroup_shadow_stride,
        .workgroup_capacity = options->tsan.workgroup_capacity,
        .shadow_entry_size = IREE_HAL_AMDGPU_TSAN_SHADOW_ENTRY_SIZE,
        .queue_aql_base = 0,
        .queue_aql_slot_mask = 0,
        .reserved = {0},
    };
  } else if (shadow_base) {
    status = iree_status_join(
        status, iree_hsa_amd_memory_pool_free(IREE_LIBHSA(&system->libhsa),
                                              shadow_base));
  }
  return status;
}

iree_status_t iree_hal_amdgpu_tsan_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_allocator_t host_allocator, iree_hal_amdgpu_tsan_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(out_state);
  memset(out_state, 0, sizeof(*out_state));

  if (!options->tsan.enabled) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(physical_device_count == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU TSAN requires at least one initialized physical device");
  }

  iree_device_size_t workgroup_shadow_stride = 0;
  iree_device_size_t shadow_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_tsan_state_calculate_layout(
      options, &workgroup_shadow_stride, &shadow_size));

  out_state->libhsa = &system->libhsa;
  out_state->host_allocator = host_allocator;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, physical_device_count,
      sizeof(out_state->device_states[0]), (void**)&out_state->device_states);
  for (iree_host_size_t i = 0;
       i < physical_device_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_amdgpu_tsan_state_initialize_device(
        options, system, physical_devices[i], workgroup_shadow_stride,
        shadow_size, &out_state->device_states[i]);
    if (iree_status_is_ok(status)) {
      ++out_state->device_state_count;
    }
  }

  if (iree_status_is_ok(status)) {
    out_state->is_enabled = true;
  } else {
    iree_hal_amdgpu_tsan_state_deinitialize(out_state);
  }
  return status;
}

void iree_hal_amdgpu_tsan_state_deinitialize(
    iree_hal_amdgpu_tsan_state_t* state) {
  if (!state || !state->libhsa) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < state->device_state_count; ++i) {
    iree_hal_amdgpu_tsan_device_state_t* device_state =
        &state->device_states[i];
    if (device_state->shadow_base) {
      iree_hal_amdgpu_hsa_cleanup_assert_success(
          iree_hsa_amd_memory_pool_free_raw(state->libhsa,
                                            device_state->shadow_base));
    }
  }
  iree_allocator_free(state->host_allocator, state->device_states);
  memset(state, 0, sizeof(*state));

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdgpu_tsan_state_is_enabled(
    const iree_hal_amdgpu_tsan_state_t* state) {
  return state && state->is_enabled;
}

iree_status_t iree_hal_amdgpu_tsan_state_populate_config(
    const iree_hal_amdgpu_tsan_state_t* state,
    iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_tsan_config_t* out_config) {
  IREE_ASSERT_ARGUMENT(out_config);
  memset(out_config, 0, sizeof(*out_config));

  if (IREE_UNLIKELY(!iree_hal_amdgpu_tsan_state_is_enabled(state))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU TSAN state is not enabled");
  }
  if (IREE_UNLIKELY(physical_device_ordinal >= state->device_state_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU TSAN physical device ordinal %" PRIhsz
                            " exceeds device count %" PRIhsz,
                            physical_device_ordinal, state->device_state_count);
  }

  *out_config = state->device_states[physical_device_ordinal].config;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_tsan_state_populate_queue_config(
    const iree_hal_amdgpu_tsan_state_t* state,
    iree_host_size_t physical_device_ordinal, uint64_t queue_aql_base,
    uint64_t queue_aql_slot_mask, iree_hal_amdgpu_tsan_config_t* out_config) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_tsan_state_populate_config(
      state, physical_device_ordinal, out_config));
  out_config->queue_aql_base = queue_aql_base;
  out_config->queue_aql_slot_mask = queue_aql_slot_mask;
  return iree_ok_status();
}
