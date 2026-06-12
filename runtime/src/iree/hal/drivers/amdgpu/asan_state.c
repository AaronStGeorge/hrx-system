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

static bool iree_hal_amdgpu_asan_align_address(uint64_t address,
                                               uint64_t alignment,
                                               uint64_t* out_address) {
  *out_address = address;
  if (alignment <= 1) return true;
  if (address > UINT64_MAX - (alignment - 1)) return false;
  *out_address = (address + (alignment - 1)) & ~(alignment - 1);
  return true;
}

static iree_status_t iree_hal_amdgpu_asan_application_range_allocate(
    iree_hal_amdgpu_asan_state_t* state, uint64_t address,
    iree_device_size_t length,
    iree_hal_amdgpu_asan_application_range_t** out_range) {
  IREE_ASSERT_ARGUMENT(out_range);
  *out_range = NULL;
  iree_hal_amdgpu_asan_application_range_t* range = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(state->host_allocator,
                                             sizeof(*range), (void**)&range));
  range->address = address;
  range->length = length;
  range->next = NULL;
  *out_range = range;
  return iree_ok_status();
}

static void iree_hal_amdgpu_asan_application_range_free(
    iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_application_range_t* range) {
  iree_allocator_free(state->host_allocator, range);
}

static void iree_hal_amdgpu_asan_application_range_free_list(
    iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_application_range_t* range) {
  while (range) {
    iree_hal_amdgpu_asan_application_range_t* next_range = range->next;
    iree_hal_amdgpu_asan_application_range_free(state, range);
    range = next_range;
  }
}

static iree_status_t iree_hal_amdgpu_asan_state_initialize_shadow_map(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    uint64_t application_window_base, iree_allocator_t host_allocator,
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
      .application_window_base = application_window_base,
      .shadow_size = options->asan.shadow_size,
      .requested_slab_size = options->asan.shadow_slab_size,
      .access_desc_count = access_desc_count,
      .access_descs = access_descs,
  };
  return iree_hal_amdgpu_shadow_map_initialize_hsa(&shadow_map_params,
                                                   out_shadow_map);
}

static iree_status_t iree_hal_amdgpu_asan_state_calculate_application_size(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_device_size_t* out_application_window_size) {
  *out_application_window_size = 0;
  if (IREE_UNLIKELY(
          options->asan.shadow_size == 0 ||
          !iree_device_size_is_power_of_two(options->asan.shadow_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ASAN shadow size %" PRIu64
                            " must be a non-zero power of two",
                            (uint64_t)options->asan.shadow_size);
  }
  if (IREE_UNLIKELY(
          options->asan.shadow_slab_size == 0 ||
          !iree_device_size_is_power_of_two(options->asan.shadow_slab_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU ASAN shadow slab size %" PRIu64
                            " must be a non-zero power of two",
                            (uint64_t)options->asan.shadow_slab_size);
  }
  if (IREE_UNLIKELY(options->asan.shadow_scale_shift >= 63)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU ASAN shadow scale shift %u is too large",
                            options->asan.shadow_scale_shift);
  }
  if (IREE_UNLIKELY(
          options->asan.shadow_size >
          (IREE_DEVICE_SIZE_MAX >> options->asan.shadow_scale_shift))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU ASAN application window size overflows: shadow_size=%" PRIu64
        ", scale_shift=%u",
        (uint64_t)options->asan.shadow_size, options->asan.shadow_scale_shift);
  }
  *out_application_window_size = options->asan.shadow_size
                                 << options->asan.shadow_scale_shift;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_asan_state_reserve_application_window(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_device_size_t application_window_size,
    iree_device_size_t application_window_alignment,
    IREE_AMDGPU_DEVICE_PTR void** out_base_ptr) {
  *out_base_ptr = NULL;
  return iree_hsa_amd_vmem_address_reserve_align(
      IREE_LIBHSA(libhsa), out_base_ptr, application_window_size,
      IREE_HAL_AMDGPU_ASAN_PREFERRED_APPLICATION_WINDOW_BASE,
      application_window_alignment, /*flags=*/0);
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

  out_state->host_allocator = host_allocator;
  iree_device_size_t application_window_size = 0;
  iree_status_t status = iree_hal_amdgpu_asan_state_calculate_application_size(
      options, &application_window_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_asan_state_reserve_application_window(
        &system->libhsa, application_window_size,
        options->asan.shadow_slab_size,
        &out_state->application_reservation_base_ptr);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_asan_state_initialize_shadow_map(
        options, system, physical_device_count, physical_devices,
        (uint64_t)(uintptr_t)out_state->application_reservation_base_ptr,
        host_allocator, &out_state->shadow_map);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_asan_application_range_allocate(
        out_state, out_state->shadow_map.application_window_base,
        out_state->shadow_map.application_window_size,
        &out_state->application_free_ranges);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_asan_application_range_free_list(
        out_state, out_state->application_free_ranges);
    if (out_state->application_reservation_base_ptr) {
      status = iree_status_join(status,
                                iree_hsa_amd_vmem_address_free(
                                    IREE_LIBHSA(&system->libhsa),
                                    out_state->application_reservation_base_ptr,
                                    application_window_size));
    }
    iree_hal_amdgpu_shadow_map_deinitialize(&out_state->shadow_map);
    memset(out_state, 0, sizeof(*out_state));
    return status;
  }

  out_state->libhsa = &system->libhsa;
  out_state->application_reservation_size = application_window_size;
  iree_slim_mutex_initialize(&out_state->application_mutex);
  out_state->is_enabled = true;
  return iree_ok_status();
}

void iree_hal_amdgpu_asan_state_deinitialize(
    iree_hal_amdgpu_asan_state_t* state) {
  if (!state || !state->is_enabled) return;
  if (state->application_reservation_base_ptr) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_vmem_address_free_raw(
            state->libhsa, state->application_reservation_base_ptr,
            state->application_reservation_size));
  }
  iree_hal_amdgpu_shadow_map_deinitialize(&state->shadow_map);
  iree_hal_amdgpu_asan_application_range_free_list(
      state, state->application_free_ranges);
  iree_slim_mutex_deinitialize(&state->application_mutex);
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

iree_status_t iree_hal_amdgpu_asan_state_map_range(
    iree_hal_amdgpu_asan_state_t* state, uint64_t application_address,
    iree_device_size_t application_length) {
  if (!iree_hal_amdgpu_asan_state_is_enabled(state)) return iree_ok_status();
  return iree_hal_amdgpu_shadow_map_map_range(
      &state->shadow_map, application_address, application_length,
      /*out_range=*/NULL);
}

iree_status_t iree_hal_amdgpu_asan_state_reserve_application_range(
    iree_hal_amdgpu_asan_state_t* state, iree_device_size_t length,
    iree_device_size_t alignment, uint64_t* out_application_address,
    iree_hal_amdgpu_asan_application_range_t** out_application_range) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(out_application_address);
  IREE_ASSERT_ARGUMENT(out_application_range);
  *out_application_address = 0;
  *out_application_range = NULL;
  if (IREE_UNLIKELY(!iree_hal_amdgpu_asan_state_is_enabled(state))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU ASAN application address reservation requires ASAN state");
  }
  if (IREE_UNLIKELY(length == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU ASAN application address reservation requires non-zero length");
  }
  if (IREE_UNLIKELY(!iree_device_size_is_valid_alignment(alignment))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU ASAN application address reservation alignment %" PRIu64
        " is not a power of two",
        (uint64_t)alignment);
  }
  if (alignment == 0) alignment = 1;

  iree_hal_amdgpu_shadow_map_t* shadow_map = &state->shadow_map;
  const uint64_t window_end =
      shadow_map->application_window_base + shadow_map->application_window_size;
  uint64_t application_address = 0;
  iree_hal_amdgpu_asan_application_range_t* application_range = NULL;
  iree_hal_amdgpu_asan_application_range_t* suffix_range = NULL;

  iree_slim_mutex_lock(&state->application_mutex);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_asan_application_range_t** current_range_ptr =
      &state->application_free_ranges;
  iree_hal_amdgpu_asan_application_range_t* current_range =
      state->application_free_ranges;
  while (current_range) {
    const uint64_t range_end = current_range->address + current_range->length;
    if (iree_hal_amdgpu_asan_align_address(current_range->address, alignment,
                                           &application_address) &&
        application_address <= range_end &&
        length <= range_end - application_address) {
      break;
    }
    current_range_ptr = &current_range->next;
    current_range = current_range->next;
  }

  if (!current_range) {
    status = iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU ASAN application window [0x%016" PRIx64 ", 0x%016" PRIx64
        ") has no space for length %" PRIu64 " with alignment %" PRIu64,
        shadow_map->application_window_base, window_end, (uint64_t)length,
        (uint64_t)alignment);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_asan_application_range_allocate(
        state, application_address, length, &application_range);
  }
  if (iree_status_is_ok(status)) {
    const uint64_t allocation_end = application_address + length;
    const iree_device_size_t prefix_length =
        application_address - current_range->address;
    const uint64_t current_range_end =
        current_range->address + current_range->length;
    const iree_device_size_t suffix_length = current_range_end - allocation_end;
    if (prefix_length > 0 && suffix_length > 0) {
      status = iree_hal_amdgpu_asan_application_range_allocate(
          state, allocation_end, suffix_length, &suffix_range);
    }
    if (iree_status_is_ok(status)) {
      if (prefix_length > 0 && suffix_length > 0) {
        current_range->length = prefix_length;
        suffix_range->next = current_range->next;
        current_range->next = suffix_range;
        suffix_range = NULL;
      } else if (prefix_length > 0) {
        current_range->length = prefix_length;
      } else if (suffix_length > 0) {
        current_range->address = allocation_end;
        current_range->length = suffix_length;
      } else {
        *current_range_ptr = current_range->next;
        iree_hal_amdgpu_asan_application_range_free(state, current_range);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_application_address = application_address;
    *out_application_range = application_range;
    application_range = NULL;
  }
  iree_slim_mutex_unlock(&state->application_mutex);
  iree_hal_amdgpu_asan_application_range_free(state, application_range);
  iree_hal_amdgpu_asan_application_range_free(state, suffix_range);
  return status;
}

void iree_hal_amdgpu_asan_state_release_application_range(
    iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_application_range_t* application_range) {
  if (!application_range) return;
  IREE_ASSERT(iree_hal_amdgpu_asan_state_is_enabled(state));

  iree_slim_mutex_lock(&state->application_mutex);
  iree_hal_amdgpu_asan_application_range_t** next_range_ptr =
      &state->application_free_ranges;
  iree_hal_amdgpu_asan_application_range_t* next_range =
      state->application_free_ranges;
  iree_hal_amdgpu_asan_application_range_t* previous_range = NULL;
  while (next_range && next_range->address < application_range->address) {
    previous_range = next_range;
    next_range_ptr = &next_range->next;
    next_range = next_range->next;
  }

  const uint64_t application_range_end =
      application_range->address + application_range->length;
  if (previous_range) {
    const uint64_t previous_range_end =
        previous_range->address + previous_range->length;
    IREE_ASSERT(previous_range_end <= application_range->address);
    if (previous_range_end == application_range->address) {
      previous_range->length += application_range->length;
      iree_hal_amdgpu_asan_application_range_free(state, application_range);
      application_range = previous_range;
    }
  }
  if (next_range) {
    IREE_ASSERT(application_range_end <= next_range->address);
    if (application_range_end == next_range->address) {
      if (application_range == previous_range) {
        application_range->length += next_range->length;
        application_range->next = next_range->next;
        iree_hal_amdgpu_asan_application_range_free(state, next_range);
      } else {
        next_range->address = application_range->address;
        next_range->length += application_range->length;
        iree_hal_amdgpu_asan_application_range_free(state, application_range);
        application_range = NULL;
      }
    }
  }
  if (application_range && application_range != previous_range) {
    application_range->next = next_range;
    *next_range_ptr = application_range;
  }
  iree_slim_mutex_unlock(&state->application_mutex);
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
      .reserved = {0},
  };
}
