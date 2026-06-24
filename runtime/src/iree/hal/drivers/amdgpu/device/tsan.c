// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/tsan.h"

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

void iree_hal_amdgpu_device_tsan_emplace_queue_initialize(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        queue_initialize_kernel_args,
    const iree_hal_amdgpu_tsan_queue_initialize_args_t* IREE_AMDGPU_RESTRICT
        queue_initialize_args,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  iree_hal_amdgpu_tsan_queue_initialize_args_t* IREE_AMDGPU_RESTRICT kernargs =
      (iree_hal_amdgpu_tsan_queue_initialize_args_t*)kernarg_ptr;
  *kernargs = *queue_initialize_args;
  kernargs->queue_state_template = &kernargs->queue_state_template_value;

  const uint64_t clear_size = queue_initialize_args->shadow_size;
  const uint32_t workgroup_size =
      queue_initialize_kernel_args->workgroup_size[0];
  const uint64_t max_workgroup_count = UINT32_MAX / workgroup_size;
  const uint64_t required_workgroup_count =
      IREE_AMDGPU_CEIL_DIV(clear_size, workgroup_size);
  const uint32_t workgroup_count[3] = {
      (uint32_t)IREE_AMDGPU_MAX(
          1ull, IREE_AMDGPU_MIN(required_workgroup_count, max_workgroup_count)),
      1, 1};
  iree_hal_amdgpu_device_dispatch_emplace_packet(
      queue_initialize_kernel_args, workgroup_count,
      /*dynamic_workgroup_local_memory=*/0, dispatch_packet, kernarg_ptr);
}

#if defined(IREE_AMDGPU_TARGET_DEVICE)

IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_tsan_initialize_queue_state(
    iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT queue_state,
    uint8_t* IREE_AMDGPU_RESTRICT shadow_base, uint64_t shadow_size,
    const iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT
        queue_state_template) {
  const uint64_t byte_index = iree_hal_amdgpu_device_global_linear_id_1d();
  const iree_hsa_kernel_dispatch_packet_t* dispatch_ptr =
      iree_amdgcn_dispatch_ptr();
  const uint64_t byte_stride = dispatch_ptr->grid_size[0];

  for (uint64_t i = byte_index; i < shadow_size; i += byte_stride) {
    shadow_base[i] = 0;
  }

  if (byte_index == 0) {
    *queue_state = *queue_state_template;
  }
}

#endif  // IREE_AMDGPU_TARGET_DEVICE
