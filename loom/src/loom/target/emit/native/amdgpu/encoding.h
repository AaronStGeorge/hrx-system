// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU native machine-code emission from target-low packet tables.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_ENCODING_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_ENCODING_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/emit/native/amdgpu/text_fixup.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_encode_instruction_stream_options_t {
  // Optional target-owned packet plan applied during native encoding.
  const struct loom_amdgpu_packet_plan_t* packet_plan;
} loom_amdgpu_encode_instruction_stream_options_t;

typedef struct loom_amdgpu_encoded_instruction_stream_t {
  // Encoded executable text bytes.
  iree_const_byte_span_t text;
  // Text literal fixups resolved after final HSACO section layout.
  const loom_amdgpu_hsaco_text_fixup_t* text_fixups;
  // Number of entries in |text_fixups|.
  iree_host_size_t text_fixup_count;
} loom_amdgpu_encoded_instruction_stream_t;

// Encodes one scheduled and allocated AMDGPU target-low function into an
// arena-owned instruction byte stream. The returned bytes are only the
// executable text payload; kernel descriptors, metadata, ELF sections, and
// relocations are emitted by later native code-object layers. Values must be
// physically allocated and unspilled.
iree_status_t loom_amdgpu_encode_instruction_stream(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena);

// Encodes one instruction stream with target-owned insertion plans.
iree_status_t loom_amdgpu_encode_instruction_stream_with_options(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_encode_instruction_stream_options_t* options,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena);

// Encodes one instruction stream and returns text plus relocation fixups.
iree_status_t loom_amdgpu_encode_instruction_stream_result(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    loom_amdgpu_encoded_instruction_stream_t* out_stream,
    iree_arena_allocator_t* arena);

// Encodes one instruction stream with target-owned insertion plans and returns
// text plus relocation fixups.
iree_status_t loom_amdgpu_encode_instruction_stream_result_with_options(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_encode_instruction_stream_options_t* options,
    loom_amdgpu_encoded_instruction_stream_t* out_stream,
    iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_ENCODING_H_
