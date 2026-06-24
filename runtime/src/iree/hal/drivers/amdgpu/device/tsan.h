// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_TSAN_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_TSAN_H_

#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef uint32_t iree_hal_amdgpu_device_tsan_dispatch_setup_flags_t;
enum iree_hal_amdgpu_device_tsan_dispatch_setup_flag_bits_t {
  IREE_HAL_AMDGPU_DEVICE_TSAN_DISPATCH_SETUP_FLAG_NONE = 0u,
  // Read and apply indirect workgroup-count parameters before publishing the
  // target dispatch packet.
  IREE_HAL_AMDGPU_DEVICE_TSAN_DISPATCH_SETUP_FLAG_INDIRECT_PARAMETERS = 1u << 0,
};

// Kernel arguments for the builtin TSAN dispatch setup kernel.
typedef struct iree_hal_amdgpu_device_tsan_dispatch_setup_args_t {
  // Flags from iree_hal_amdgpu_device_tsan_dispatch_setup_flag_bits_t.
  iree_hal_amdgpu_device_tsan_dispatch_setup_flags_t flags;
  // Reserved padding that must be zero.
  uint32_t reserved0;
  // Device pointer to a uint32_t[3] workgroup-count parameter buffer.
  const uint32_t* workgroup_count;
  // Device pointer to the target dispatch packet when patching indirect
  // parameters; otherwise unused.
  iree_hsa_kernel_dispatch_packet_t* dispatch_packet;
  // Optional device pointer to the dispatch's implicit args suffix.
  iree_amdgpu_kernel_implicit_args_t* implicit_args;
  // Device pointer to queue-local TSAN state.
  iree_hal_amdgpu_tsan_queue_state_t* tsan_queue_state;
  // Base generation value used when publishing TSAN dispatch state.
  uint64_t tsan_generation_epoch;
  // Final 32-bit header/setup word used when patching indirect parameters.
  uint32_t dispatch_header_setup;
  // Distance from the setup packet to the target dispatch packet.
  uint32_t packet_delta;
  // Value added to |tsan_generation_epoch| for the target dispatch state.
  uint32_t generation_delta;
  // Queue-local shadow slot assigned to the target dispatch.
  uint32_t shadow_slot;
} iree_hal_amdgpu_device_tsan_dispatch_setup_args_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_device_tsan_dispatch_setup_args_t) == 64,
    "TSAN dispatch setup args must match the kernel ABI");

// Populates a builtin dispatch packet and kernargs that assigns queue-local
// TSAN dispatch state for a command-buffer block.
//
// |dispatch_packet| and |kernarg_ptr| must point to reserved queue storage.
// The caller owns packet header commit and barrier placement.
void iree_hal_amdgpu_device_tsan_emplace_assignment(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        assignment_kernel_args,
    const iree_hal_amdgpu_tsan_assignment_plan_t* IREE_AMDGPU_RESTRICT plan,
    iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT queue_state,
    uint32_t assignment_record_count, uint64_t generation_epoch,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Populates a builtin dispatch packet and kernargs that initializes queue-local
// TSAN state before user work can consume it.
//
// |dispatch_packet| and |kernarg_ptr| must point to reserved queue storage.
// The caller owns packet header commit and barrier placement.
void iree_hal_amdgpu_device_tsan_emplace_queue_initialize(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        queue_initialize_kernel_args,
    const iree_hal_amdgpu_tsan_queue_initialize_args_t* IREE_AMDGPU_RESTRICT
        queue_initialize_args,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Populates a builtin TSAN dispatch setup packet and kernargs.
//
// The target dispatch packet must already contain every non-header field. The
// setup dispatch publishes queue-local TSAN state for the target packet. When
// |setup_flags| includes INDIRECT_PARAMETERS, it also reads |workgroup_count|
// on device and updates the target packet's grid-size fields and optional
// implicit args before atomically publishing |dispatch_header|/|dispatch_setup|
// to the target packet. Without INDIRECT_PARAMETERS the host commits the target
// packet header and orders it after the setup dispatch with AQL barriers.
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
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Device builtin that publishes queue-local TSAN dispatch state for all
// instrumented dispatch packets in a command-buffer block.
IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_tsan_assign(
    const iree_hal_amdgpu_tsan_assignment_plan_t* IREE_AMDGPU_RESTRICT plan,
    iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT queue_state,
    uint64_t generation_epoch);

// Device builtin that initializes queue-local TSAN state.
IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_tsan_initialize_queue_state(
    iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT queue_state,
    iree_hal_amdgpu_tsan_dispatch_state_t* IREE_AMDGPU_RESTRICT dispatch_states,
    uint8_t* IREE_AMDGPU_RESTRICT shadow_base, uint64_t dispatch_state_length,
    uint64_t shadow_size,
    const iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT
        queue_state_template);

// Device builtin that assigns queue-local TSAN state for a following dispatch
// packet and optionally patches indirect workgroup-count parameters.
IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_tsan_setup_dispatch(
    iree_hal_amdgpu_device_tsan_dispatch_setup_flags_t setup_flags,
    const uint32_t* IREE_AMDGPU_RESTRICT workgroup_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    iree_amdgpu_kernel_implicit_args_t* IREE_AMDGPU_RESTRICT implicit_args,
    iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT queue_state,
    uint64_t tsan_generation_epoch, uint32_t dispatch_header_setup,
    uint32_t packet_delta, uint32_t generation_delta, uint32_t shadow_slot);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_TSAN_H_
