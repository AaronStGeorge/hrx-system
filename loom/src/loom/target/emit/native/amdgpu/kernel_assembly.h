// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HSA kernel assembly envelopes over target-low native fragments.
//
// The fragment emitter owns instruction syntax. This layer owns the first
// loadable-kernel boundary: symbol selection, .amdgcn_target, AMDHSA kernel
// descriptor directives, and the strict subset that is safe before full HAL
// ABI lowering exists.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_ASSEMBLY_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_ASSEMBLY_H_

#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule.h"
#include "loom/target/arch/amdgpu/wait_packets.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits complete AMDGPU assembly for one ABI-lowered HAL kernel low.func.def.
// The output is assembler input containing a text function body and an AMDHSA
// kernel descriptor. It deliberately remains text: assembling, disassembling,
// loading, and launching are tool/runtime adapter responsibilities.
iree_status_t loom_amdgpu_emit_kernel_assembly(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena);

// Emits complete AMDGPU assembly with planned wait packets inserted into the
// native fragment before the kernel descriptor is appended. |wait_packets| must
// be derived from |schedule| and remain alive for the duration of emission.
// |scratch_arena| receives transient ABI layout and metadata adapter storage.
iree_status_t loom_amdgpu_emit_kernel_assembly_with_wait_packets(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_KERNEL_ASSEMBLY_H_
