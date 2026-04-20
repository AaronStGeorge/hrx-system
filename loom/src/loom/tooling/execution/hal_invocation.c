// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal_invocation.h"

#include "iree/modules/hal/types.h"
#include "iree/tooling/function_util.h"

void loom_run_hal_invocation_options_initialize(
    loom_run_hal_invocation_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_run_hal_invocation_options_t){
      .entry_point = 0,
      .workgroup_count = {1, 1, 1},
  };
}

iree_status_t loom_run_hal_executable_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_executable_t* executable,
    iree_hal_executable_t** out_hal_executable) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_hal_executable);
  *out_hal_executable = NULL;
  if (runtime->device == NULL || runtime->executable_cache == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }

  iree_hal_executable_params_t executable_params;
  iree_hal_executable_params_initialize(&executable_params);
  executable_params.caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION |
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
  executable_params.executable_format = executable->executable_format;
  executable_params.executable_data = executable->executable_data;
  return iree_hal_executable_cache_prepare_executable(
      runtime->executable_cache, &executable_params, out_hal_executable);
}

static iree_status_t loom_run_hal_binding_refs_from_list(
    iree_vm_list_t* binding_list, iree_hal_buffer_ref_t* binding_refs,
    iree_host_size_t binding_ref_capacity) {
  IREE_ASSERT_ARGUMENT(binding_list);
  IREE_ASSERT_ARGUMENT(binding_refs);
  const iree_host_size_t binding_count = iree_vm_list_size(binding_list);
  if (binding_count > binding_ref_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL binding count %" PRIhsz
                            " exceeds capacity %" PRIhsz,
                            binding_count, binding_ref_capacity);
  }
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    iree_vm_ref_t value = iree_vm_ref_null();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_ref_assign(binding_list, i, &value));
    iree_hal_buffer_t* buffer = NULL;
    if (iree_hal_buffer_isa(value)) {
      buffer = iree_hal_buffer_deref(value);
    } else if (iree_hal_buffer_view_isa(value)) {
      buffer = iree_hal_buffer_view_buffer(iree_hal_buffer_view_deref(value));
    } else {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL binding %" PRIhsz " is not a buffer or buffer view", i);
    }
    binding_refs[i] =
        iree_hal_make_buffer_ref(buffer, 0, IREE_HAL_WHOLE_BUFFER);
  }
  return iree_ok_status();
}

iree_status_t loom_run_hal_dispatch(
    iree_hal_device_t* device, iree_hal_executable_t* executable,
    iree_vm_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(binding_list);
  IREE_ASSERT_ARGUMENT(options);

  iree_hal_buffer_ref_t binding_refs[LOOM_RUN_HAL_MAX_BINDING_COUNT];
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_refs_from_list(
      binding_list, binding_refs, IREE_ARRAYSIZE(binding_refs)));

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_hal_semaphore_t* semaphore = NULL;
  uint64_t signal_value = 1;

  iree_status_t status = iree_hal_command_buffer_create(
      device,
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_ref_list_t bindings = {
        .count = iree_vm_list_size(binding_list),
        .values = binding_refs,
    };
    iree_hal_dispatch_config_t config = iree_hal_make_static_dispatch_config(
        options->workgroup_count[0], options->workgroup_count[1],
        options->workgroup_count[2]);
    status = iree_hal_command_buffer_dispatch(
        command_buffer, executable, options->entry_point, config,
        iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t wait_semaphores = iree_hal_semaphore_list_empty();
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = 1,
        .semaphores = &semaphore,
        .payload_values = &signal_value,
    };
    status = iree_hal_device_queue_execute(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_semaphores, signal_semaphores,
        command_buffer, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_EXECUTE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_hal_semaphore_release(semaphore);
  iree_hal_command_buffer_release(command_buffer);
  return status;
}

iree_status_t loom_run_hal_invocation_execute(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_executable_t* executable, iree_vm_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(binding_list);
  IREE_ASSERT_ARGUMENT(options);

  iree_hal_executable_t* hal_executable = NULL;
  iree_status_t status =
      loom_run_hal_executable_prepare(runtime, executable, &hal_executable);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_dispatch(runtime->device, hal_executable,
                                   binding_list, options);
  }
  iree_hal_executable_release(hal_executable);
  return status;
}

iree_status_t loom_run_hal_transfer_bindings_to_host(
    const loom_run_hal_runtime_t* runtime, iree_vm_list_t* binding_list) {
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(binding_list);
  if (runtime->device == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL runtime is not initialized");
  }

  iree_hal_allocator_t* device_allocator =
      iree_hal_device_allocator(runtime->device);
  iree_hal_buffer_params_t host_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 0,
  };
  return iree_tooling_transfer_variants(
      binding_list, runtime->device, device_allocator, host_params,
      /*wait_fence=*/NULL, /*signal_fence=*/NULL);
}
