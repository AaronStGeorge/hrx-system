// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/feedback_state.h"

#include <stdio.h>

#include "iree/base/threading/affinity.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"

static iree_status_t iree_hal_amdgpu_feedback_state_handle_packet(
    iree_host_size_t physical_device_ordinal,
    const iree_hal_amdgpu_feedback_packet_t* packet) {
  switch (packet->kind) {
    case IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_ASAN:
      return iree_make_status(
          IREE_STATUS_ABORTED,
          "AMDGPU ASAN feedback packet reported on physical device %" PRIhsz
          " workgroup_x=%u workitem_x=%u dispatch=0x%016" PRIx64,
          physical_device_ordinal, packet->source_workgroup_id_x,
          packet->source_workitem_id_x, packet->source_dispatch_ptr);
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "unhandled AMDGPU feedback packet kind %u on physical device %" PRIhsz
          " workgroup_x=%u workitem_x=%u dispatch=0x%016" PRIx64,
          (uint32_t)packet->kind, physical_device_ordinal,
          packet->source_workgroup_id_x, packet->source_workitem_id_x,
          packet->source_dispatch_ptr);
  }
}

static void iree_hal_amdgpu_feedback_state_report_error(
    iree_hal_amdgpu_feedback_state_t* state, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  if (state->error_handler) {
    state->error_handler(state->error_handler_user_data, status);
  } else {
    iree_status_free(status);
  }
}

static iree_status_t iree_hal_amdgpu_feedback_state_drain_packet(
    const iree_hal_amdgpu_feedback_packet_t* packet, void* user_data) {
  iree_hal_amdgpu_feedback_device_state_t* device_state =
      (iree_hal_amdgpu_feedback_device_state_t*)user_data;
  iree_status_t status = iree_hal_amdgpu_feedback_state_handle_packet(
      device_state->physical_device_ordinal, packet);
  iree_hal_amdgpu_feedback_state_report_error(device_state->parent, status);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_feedback_state_drain_device(
    iree_hal_amdgpu_feedback_device_state_t* device_state) {
  iree_host_size_t packet_count = 0;
  return iree_hal_amdgpu_feedback_channel_drain(
      &device_state->channel, IREE_HOST_SIZE_MAX,
      iree_hal_amdgpu_feedback_state_drain_packet, device_state, &packet_count);
}

static void iree_hal_amdgpu_feedback_state_request_device_stop(
    iree_hal_amdgpu_feedback_device_state_t* device_state) {
  if (device_state->stop_signal.handle) {
    iree_hsa_signal_store_screlease(IREE_LIBHSA(device_state->parent->libhsa),
                                    device_state->stop_signal, 1);
  }
}

static int iree_hal_amdgpu_feedback_state_service_thread_main(void* entry_arg) {
  {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z0, "iree_hal_amdgpu_feedback_state_service_thread_start");
    IREE_TRACE_ZONE_END(z0);
  }

  iree_hal_amdgpu_feedback_device_state_t* device_state =
      (iree_hal_amdgpu_feedback_device_state_t*)entry_arg;
  iree_hal_amdgpu_feedback_state_t* state = device_state->parent;
  const iree_hal_amdgpu_libhsa_t* libhsa = state->libhsa;

  enum {
    IREE_HAL_AMDGPU_FEEDBACK_WAIT_NOTIFY_SIGNAL = 0,
    IREE_HAL_AMDGPU_FEEDBACK_WAIT_STOP_SIGNAL = 1,
    IREE_HAL_AMDGPU_FEEDBACK_WAIT_SIGNAL_COUNT = 2,
  };

  bool keep_running = true;
  while (keep_running) {
    hsa_signal_t signals[IREE_HAL_AMDGPU_FEEDBACK_WAIT_SIGNAL_COUNT] = {
        device_state->channel.notify_signal,
        device_state->stop_signal,
    };
    hsa_signal_condition_t
        conditions[IREE_HAL_AMDGPU_FEEDBACK_WAIT_SIGNAL_COUNT] = {
            HSA_SIGNAL_CONDITION_NE,
            HSA_SIGNAL_CONDITION_NE,
        };
    hsa_signal_value_t values[IREE_HAL_AMDGPU_FEEDBACK_WAIT_SIGNAL_COUNT] = {
        0,
        0,
    };
    const uint32_t signal_index = iree_hsa_amd_signal_wait_any(
        IREE_LIBHSA(libhsa), IREE_HAL_AMDGPU_FEEDBACK_WAIT_SIGNAL_COUNT,
        signals, conditions, values, UINT64_MAX, HSA_WAIT_STATE_BLOCKED,
        /*satisfying_value=*/NULL);

    {
      IREE_TRACE_ZONE_BEGIN_NAMED(
          z0, "iree_hal_amdgpu_feedback_state_service_thread_pump");
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, signal_index);

      if (signal_index == IREE_HAL_AMDGPU_FEEDBACK_WAIT_NOTIFY_SIGNAL) {
        (void)iree_hsa_signal_exchange_scacquire(
            IREE_LIBHSA(libhsa), device_state->channel.notify_signal, 0);
        iree_status_t status =
            iree_hal_amdgpu_feedback_state_drain_device(device_state);
        iree_hal_amdgpu_feedback_state_report_error(state, status);
      }

      if (signal_index == IREE_HAL_AMDGPU_FEEDBACK_WAIT_STOP_SIGNAL) {
        keep_running = false;
      } else if (IREE_UNLIKELY(signal_index >=
                               IREE_HAL_AMDGPU_FEEDBACK_WAIT_SIGNAL_COUNT)) {
        iree_status_t status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "hsa_amd_signal_wait_any returned invalid signal index %u while "
            "waiting for AMDGPU feedback packets",
            signal_index);
        iree_hal_amdgpu_feedback_state_report_error(state, status);
        keep_running = false;
      }

      IREE_TRACE_ZONE_END(z0);
    }
  }

  iree_status_t status =
      iree_hal_amdgpu_feedback_state_drain_device(device_state);
  iree_hal_amdgpu_feedback_state_report_error(state, status);

  {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z0, "iree_hal_amdgpu_feedback_state_service_thread_exit");
    IREE_TRACE_ZONE_END(z0);
  }
  return 0;
}

static iree_status_t iree_hal_amdgpu_feedback_state_initialize_device(
    iree_hal_amdgpu_feedback_state_t* state, iree_hal_amdgpu_system_t* system,
    iree_hal_amdgpu_physical_device_t* physical_device,
    iree_hal_amdgpu_feedback_device_state_t* out_device_state) {
  if (IREE_UNLIKELY(!physical_device)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU feedback requires initialized physical devices");
  }

  hsa_amd_memory_pool_t control_memory_pool =
      physical_device->host_memory_pools.fine_pool;
  if (IREE_UNLIKELY(!control_memory_pool.handle)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU feedback requires a host fine-grained control memory pool");
  }

  hsa_amd_memory_pool_t ring_memory_pool =
      physical_device->fine_block_pools.large.memory_pool;
  if (IREE_UNLIKELY(!ring_memory_pool.handle)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU feedback requires a device fine-grained ring memory pool");
  }

  out_device_state->parent = state;
  out_device_state->physical_device_ordinal = physical_device->device_ordinal;

  const iree_hal_amdgpu_feedback_channel_params_t channel_params = {
      .libhsa = state->libhsa,
      .device_agent = physical_device->device_agent,
      .control_memory_pool = control_memory_pool,
      .ring_memory_pool = ring_memory_pool,
      .topology = &system->topology,
      .minimum_capacity = 0,
  };

  iree_status_t status = iree_hal_amdgpu_feedback_channel_initialize(
      &channel_params, &out_device_state->channel);
  if (iree_status_is_ok(status)) {
    status = iree_hsa_amd_signal_create(
        IREE_LIBHSA(state->libhsa), /*initial_value=*/0,
        /*num_consumers=*/0, /*consumers=*/NULL, /*attributes=*/0,
        &out_device_state->stop_signal);
  }
  if (iree_status_is_ok(status)) {
    iree_thread_create_params_t thread_params;
    memset(&thread_params, 0, sizeof(thread_params));
    char thread_name[32] = {0};
    snprintf(thread_name, IREE_ARRAYSIZE(thread_name), "amdgpu-d%u-feedback",
             (unsigned)physical_device->device_ordinal);
    thread_params.name = iree_make_cstring_view(thread_name);
    iree_thread_affinity_set_group_any(physical_device->host_numa_node,
                                       &thread_params.initial_affinity);
    status = iree_thread_create(
        iree_hal_amdgpu_feedback_state_service_thread_main, out_device_state,
        thread_params, state->host_allocator,
        &out_device_state->service_thread);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_feedback_state_request_device_stop(out_device_state);
    iree_thread_release(out_device_state->service_thread);
    out_device_state->service_thread = NULL;
    if (out_device_state->stop_signal.handle) {
      iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_signal_destroy_raw(
          state->libhsa, out_device_state->stop_signal));
    }
    iree_hal_amdgpu_feedback_channel_deinitialize(&out_device_state->channel);
    memset(out_device_state, 0, sizeof(*out_device_state));
  }
  return status;
}

iree_status_t iree_hal_amdgpu_feedback_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_hal_amdgpu_feedback_error_handler_fn_t error_handler,
    void* error_handler_user_data, iree_allocator_t host_allocator,
    iree_hal_amdgpu_feedback_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(system);
  IREE_ASSERT_ARGUMENT(out_state);
  memset(out_state, 0, sizeof(*out_state));

  if (!options->asan.enabled) return iree_ok_status();
  if (IREE_UNLIKELY(physical_device_count == 0 || !physical_devices)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU feedback requires at least one initialized physical device");
  }
  if (IREE_UNLIKELY(!error_handler)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU feedback requires a service-thread error handler");
  }

  out_state->libhsa = &system->libhsa;
  out_state->host_allocator = host_allocator;
  out_state->error_handler = error_handler;
  out_state->error_handler_user_data = error_handler_user_data;

  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, physical_device_count,
      sizeof(out_state->device_states[0]), (void**)&out_state->device_states);
  for (iree_host_size_t i = 0;
       i < physical_device_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_amdgpu_feedback_state_initialize_device(
        out_state, system, physical_devices[i], &out_state->device_states[i]);
    if (iree_status_is_ok(status)) {
      ++out_state->device_state_count;
    }
  }

  if (iree_status_is_ok(status)) {
    out_state->is_enabled = true;
  } else {
    iree_hal_amdgpu_feedback_state_deinitialize(out_state);
  }
  return status;
}

void iree_hal_amdgpu_feedback_state_deinitialize(
    iree_hal_amdgpu_feedback_state_t* state) {
  if (!state || !state->libhsa) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < state->device_state_count; ++i) {
    iree_hal_amdgpu_feedback_state_request_device_stop(
        &state->device_states[i]);
  }
  for (iree_host_size_t i = 0; i < state->device_state_count; ++i) {
    iree_hal_amdgpu_feedback_device_state_t* device_state =
        &state->device_states[i];
    iree_thread_release(device_state->service_thread);
    device_state->service_thread = NULL;
    if (device_state->stop_signal.handle) {
      iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_signal_destroy_raw(
          state->libhsa, device_state->stop_signal));
      device_state->stop_signal = iree_hsa_signal_null();
    }
    iree_hal_amdgpu_feedback_channel_deinitialize(&device_state->channel);
  }
  iree_allocator_free(state->host_allocator, state->device_states);
  memset(state, 0, sizeof(*state));

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdgpu_feedback_state_is_enabled(
    const iree_hal_amdgpu_feedback_state_t* state) {
  return state && state->is_enabled;
}

iree_status_t iree_hal_amdgpu_feedback_state_populate_config(
    const iree_hal_amdgpu_feedback_state_t* state,
    iree_host_size_t physical_device_ordinal,
    iree_hal_amdgpu_feedback_config_t* out_config) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(out_config);
  memset(out_config, 0, sizeof(*out_config));

  if (IREE_UNLIKELY(!iree_hal_amdgpu_feedback_state_is_enabled(state))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU feedback state is not enabled");
  }
  if (IREE_UNLIKELY(physical_device_ordinal >= state->device_state_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU feedback physical device ordinal %" PRIhsz
                            " exceeds device count %" PRIhsz,
                            physical_device_ordinal, state->device_state_count);
  }

  *out_config = state->device_states[physical_device_ordinal].channel.config;
  return iree_ok_status();
}
