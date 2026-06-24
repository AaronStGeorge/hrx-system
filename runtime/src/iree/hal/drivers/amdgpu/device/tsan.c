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

  const uint64_t clear_size = queue_initialize_args->dispatch_state_length +
                              queue_initialize_args->shadow_size;
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

void iree_hal_amdgpu_device_tsan_emplace_dispatch_setup(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        setup_kernel_args,
    iree_hal_amdgpu_device_tsan_dispatch_setup_flags_t setup_flags,
    const uint32_t* IREE_AMDGPU_RESTRICT workgroup_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    uint16_t dispatch_header, uint16_t dispatch_setup,
    iree_amdgpu_kernel_implicit_args_t* IREE_AMDGPU_RESTRICT implicit_args,
    iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT queue_state,
    uint64_t tsan_generation_epoch, uint32_t packet_delta,
    uint32_t generation_delta, uint32_t shadow_slot,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT setup_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  iree_hal_amdgpu_device_tsan_dispatch_setup_args_t* IREE_AMDGPU_RESTRICT
      kernargs =
          (iree_hal_amdgpu_device_tsan_dispatch_setup_args_t*)kernarg_ptr;
  kernargs->flags = setup_flags;
  kernargs->reserved0 = 0;
  kernargs->workgroup_count = workgroup_count;
  kernargs->dispatch_packet = dispatch_packet;
  kernargs->implicit_args = implicit_args;
  kernargs->tsan_queue_state = queue_state;
  kernargs->tsan_generation_epoch = tsan_generation_epoch;
  kernargs->dispatch_header_setup =
      (uint32_t)dispatch_header | ((uint32_t)dispatch_setup << 16);
  kernargs->packet_delta = packet_delta;
  kernargs->generation_delta = generation_delta;
  kernargs->shadow_slot = shadow_slot;

  const uint32_t workgroup_count_1d[3] = {1, 1, 1};
  iree_hal_amdgpu_device_dispatch_emplace_packet(
      setup_kernel_args, workgroup_count_1d,
      /*dynamic_workgroup_local_memory=*/0, setup_packet, kernarg_ptr);
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
  const uint64_t assignment_dispatch_id = iree_amdgcn_dispatch_id();
  const uint32_t target_slot =
      (uint32_t)((assignment_dispatch_id + record.packet_delta) &
                 queue_state->aql_ring_mask);

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

IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_tsan_initialize_queue_state(
    iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT queue_state,
    iree_hal_amdgpu_tsan_dispatch_state_t* IREE_AMDGPU_RESTRICT dispatch_states,
    uint8_t* IREE_AMDGPU_RESTRICT shadow_base, uint64_t dispatch_state_length,
    uint64_t shadow_size,
    const iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT
        queue_state_template) {
  const uint64_t byte_index = iree_hal_amdgpu_device_global_linear_id_1d();
  const iree_hsa_kernel_dispatch_packet_t* dispatch_ptr =
      iree_amdgcn_dispatch_ptr();
  const uint64_t byte_stride = dispatch_ptr->grid_size[0];

  uint8_t* IREE_AMDGPU_RESTRICT dispatch_bytes = (uint8_t*)dispatch_states;
  for (uint64_t i = byte_index; i < dispatch_state_length; i += byte_stride) {
    dispatch_bytes[i] = 0;
  }
  for (uint64_t i = byte_index; i < shadow_size; i += byte_stride) {
    shadow_base[i] = 0;
  }

  if (byte_index == 0) {
    *queue_state = *queue_state_template;
  }
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_tsan_setup_dispatch(
    iree_hal_amdgpu_device_tsan_dispatch_setup_flags_t setup_flags,
    const uint32_t* IREE_AMDGPU_RESTRICT workgroup_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    iree_amdgpu_kernel_implicit_args_t* IREE_AMDGPU_RESTRICT implicit_args,
    iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT queue_state,
    uint64_t tsan_generation_epoch, uint32_t dispatch_header_setup,
    uint32_t packet_delta, uint32_t generation_delta, uint32_t shadow_slot) {
  const bool patch_indirect_parameters =
      (setup_flags &
       IREE_HAL_AMDGPU_DEVICE_TSAN_DISPATCH_SETUP_FLAG_INDIRECT_PARAMETERS) !=
      0;
  if (patch_indirect_parameters) {
    dispatch_packet->grid_size[0] =
        workgroup_count[0] * dispatch_packet->workgroup_size[0];
    dispatch_packet->grid_size[1] =
        workgroup_count[1] * dispatch_packet->workgroup_size[1];
    dispatch_packet->grid_size[2] =
        workgroup_count[2] * dispatch_packet->workgroup_size[2];

    if (implicit_args) {
      implicit_args->block_count[0] = workgroup_count[0];
      implicit_args->block_count[1] = workgroup_count[1];
      implicit_args->block_count[2] = workgroup_count[2];
    }
  }

  const uint64_t setup_dispatch_id = iree_amdgcn_dispatch_id();
  const uint32_t target_slot = (uint32_t)((setup_dispatch_id + packet_delta) &
                                          queue_state->aql_ring_mask);

  iree_hal_amdgpu_tsan_dispatch_state_t* IREE_AMDGPU_RESTRICT dispatch_states =
      (iree_hal_amdgpu_tsan_dispatch_state_t*)(uintptr_t)
          queue_state->dispatch_state_base;
  iree_hal_amdgpu_tsan_dispatch_state_t* IREE_AMDGPU_RESTRICT dispatch_state =
      &dispatch_states[target_slot];
  dispatch_state->generation = tsan_generation_epoch + generation_delta;
  dispatch_state->shadow_slot = shadow_slot;
  iree_amdgpu_scoped_atomic_store(
      (iree_amdgpu_scoped_atomic_uint32_t*)&dispatch_state->flags,
      IREE_HAL_AMDGPU_TSAN_DISPATCH_STATE_FLAG_ASSIGNED,
      iree_amdgpu_memory_order_release, iree_amdgpu_memory_scope_system);

  if (patch_indirect_parameters) {
    iree_amdgpu_scoped_atomic_store(
        (iree_amdgpu_scoped_atomic_uint32_t*)dispatch_packet,
        dispatch_header_setup, iree_amdgpu_memory_order_release,
        iree_amdgpu_memory_scope_system);
  }
}

#endif  // IREE_AMDGPU_TARGET_DEVICE
