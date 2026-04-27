// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native machine-code emission from target-low packet sidecars.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_ENCODING_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_ENCODING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule.h"
#include "loom/target/arch/amdgpu/wait_packets.h"

#ifdef __cplusplus
extern "C" {
#endif

// Encodes one scheduled and allocated AMDGPU target-low function into an
// arena-owned instruction byte stream. The returned bytes are only the
// executable text
// payload; kernel descriptors, metadata, ELF sections, and relocations are
// emitted by later native code-object layers. Values must be physically
// allocated and unspilled.
iree_status_t loom_amdgpu_encode_instruction_stream(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena);

// Encodes one instruction stream with planned wait packets inserted before
// their scheduled packet insertion points. |wait_packets| must be derived from
// |schedule| and remain alive for the duration of emission.
iree_status_t loom_amdgpu_encode_instruction_stream_with_wait_packets(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_ENCODING_H_
