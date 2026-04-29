// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native assembly-fragment emission from target-low packet tables.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_ASSEMBLY_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_ASSEMBLY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/wait_packets.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits an AMDGPU assembly fragment for one scheduled and allocated AMDGPU
// target-low function. The fragment assumes exact physical-register inputs and
// outputs; it does not emit kernel metadata, PAL metadata, or an ELF code
// object envelope. Values must be physically allocated and unspilled.
iree_status_t loom_amdgpu_emit_assembly_fragment(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena);

// Emits an AMDGPU assembly fragment with planned wait packets inserted before
// their scheduled packet insertion points. |wait_packets| must be derived from
// |schedule| and remain alive for the duration of emission.
iree_status_t loom_amdgpu_emit_assembly_fragment_with_wait_packets(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_ASSEMBLY_H_
