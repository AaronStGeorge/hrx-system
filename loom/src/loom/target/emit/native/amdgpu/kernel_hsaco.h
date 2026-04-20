// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HSA code-object emission from target-low native fragments.
//
// This layer consumes a scheduled and physically allocated low.func.def for one
// HAL kernel and writes a loadable HSACO ELF containing metadata, a kernel
// descriptor, and encoded native text.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_HSACO_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_HSACO_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule.h"
#include "loom/target/arch/amdgpu/wait_packets.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits complete AMDGPU HSACO for one ABI-lowered HAL kernel low.func.def.
//
// The output stream receives a self-contained ELF code object with metadata,
// one kernel descriptor, and one encoded text entry. Values must be physically
// allocated and unspilled.
iree_status_t loom_amdgpu_emit_kernel_hsaco(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena);

// Emits complete AMDGPU HSACO with planned wait packets inserted into the
// native text stream. |wait_packets| must be derived from |schedule| and remain
// alive for the duration of emission.
iree_status_t loom_amdgpu_emit_kernel_hsaco_with_wait_packets(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    iree_io_stream_t* stream, iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_HSACO_H_
