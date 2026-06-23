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

#if defined(IREE_AMDGPU_TARGET_DEVICE)

// Device builtin that publishes queue-local TSAN dispatch state for all
// instrumented dispatch packets in a command-buffer block.
IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_tsan_assign(
    const iree_hal_amdgpu_tsan_assignment_plan_t* IREE_AMDGPU_RESTRICT plan,
    iree_hal_amdgpu_tsan_queue_state_t* IREE_AMDGPU_RESTRICT queue_state,
    uint64_t generation_epoch);

#endif  // IREE_AMDGPU_TARGET_DEVICE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_TSAN_H_
