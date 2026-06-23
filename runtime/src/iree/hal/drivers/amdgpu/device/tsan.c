// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/tsan.h"

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

void iree_hal_amdgpu_device_tsan_emplace_assignment(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        assignment_kernel_args,
    const iree_hal_amdgpu_tsan_assignment_plan_t* IREE_AMDGPU_RESTRICT plan,
    iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT queue_state,
    uint32_t assignment_record_count, uint64_t generation_epoch,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  iree_hal_amdgpu_tsan_assignment_args_t* IREE_AMDGPU_RESTRICT kernargs =
      (iree_hal_amdgpu_tsan_assignment_args_t*)kernarg_ptr;
  kernargs->plan_ptr = (uint64_t)(uintptr_t)plan;
  kernargs->queue_state_ptr = (uint64_t)(uintptr_t)queue_state;
  kernargs->generation_epoch = generation_epoch;

  const uint32_t workgroup_size = assignment_kernel_args->workgroup_size[0];
  const uint32_t workgroup_count[3] = {
      (uint32_t)IREE_AMDGPU_CEIL_DIV(assignment_record_count, workgroup_size),
      1, 1};
  iree_hal_amdgpu_device_dispatch_emplace_packet(
      assignment_kernel_args, workgroup_count,
      /*dynamic_workgroup_local_memory=*/0, dispatch_packet, kernarg_ptr);
}

#if defined(IREE_AMDGPU_TARGET_DEVICE)

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_tsan_assign(
    const iree_hal_amdgpu_tsan_assignment_plan_t* IREE_AMDGPU_RESTRICT plan,
    iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT queue_state,
    uint64_t generation_epoch) {
  const size_t record_index = iree_hal_amdgpu_device_global_linear_id_1d();
  if (record_index >= plan->record_count) return;

  const iree_hal_amdgpu_tsan_assignment_record_t* IREE_AMDGPU_RESTRICT records =
      (const iree_hal_amdgpu_tsan_assignment_record_t*)(const void*)(plan + 1);
  const iree_hal_amdgpu_tsan_assignment_record_t record = records[record_index];
  const uint64_t assignment_packet_address =
      (uint64_t)(uintptr_t)iree_amdgcn_dispatch_ptr();
  const uint64_t packet_offset =
      assignment_packet_address - queue_state->aql_ring_base;
  const uint32_t assignment_slot =
      (uint32_t)((packet_offset / sizeof(iree_hsa_kernel_dispatch_packet_t)) &
                 queue_state->aql_ring_mask);
  const uint32_t target_slot =
      (assignment_slot + record.packet_delta) & queue_state->aql_ring_mask;

  iree_hal_amdgpu_tsan_dispatch_state_t* IREE_AMDGPU_RESTRICT dispatch_states =
      (iree_hal_amdgpu_tsan_dispatch_state_t*)(uintptr_t)
          queue_state->dispatch_state_base;
  iree_hal_amdgpu_tsan_dispatch_state_t* IREE_AMDGPU_RESTRICT dispatch_state =
      &dispatch_states[target_slot];
  dispatch_state->generation =
      generation_epoch + plan->generation_base + record.generation_delta;
  dispatch_state->shadow_slot = record.shadow_slot;
  iree_amdgpu_scoped_atomic_store(
      (iree_amdgpu_scoped_atomic_uint32_t*)&dispatch_state->flags,
      IREE_HAL_AMDGPU_TSAN_DISPATCH_STATE_FLAG_ASSIGNED,
      iree_amdgpu_memory_order_release, iree_amdgpu_memory_scope_system);
}

#endif  // IREE_AMDGPU_TARGET_DEVICE
