// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU fixed-segment layout for function-owned low.slot records.
//
// This layout is the single contract shared by kernel metadata and native
// encoding. Each low.slot owned by a low.func.def is packed in module source
// order into the target segment selected by its space, with the slot's
// requested alignment applied before assigning its byte offset.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_SLOT_LAYOUT_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_SLOT_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/ops/low/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_slot_layout_segment_sizes_t {
  // Fixed bytes of workgroup LDS required by low.slot records.
  uint64_t group_segment_fixed_size;
  // Fixed bytes of invocation-private storage required by low.slot records.
  uint64_t private_segment_fixed_size;
} loom_amdgpu_slot_layout_segment_sizes_t;

typedef struct loom_amdgpu_slot_layout_slot_t {
  // Storage segment containing the resolved slot.
  loom_low_slot_space_t space;
  // Byte offset assigned to the slot within its storage segment.
  uint64_t byte_offset;
  // Slot size in bytes.
  uint64_t byte_size;
  // Slot alignment in bytes.
  uint64_t byte_alignment;
} loom_amdgpu_slot_layout_slot_t;

typedef struct loom_amdgpu_slot_layout_record_t {
  // Slot symbol resolved by this record.
  loom_symbol_ref_t slot_ref;
  // Segment placement assigned to the slot.
  loom_amdgpu_slot_layout_slot_t slot;
} loom_amdgpu_slot_layout_record_t;

typedef struct loom_amdgpu_slot_layout_t {
  // Fixed segment sizes after laying out all function-owned slots.
  loom_amdgpu_slot_layout_segment_sizes_t segment_sizes;
  // Arena-owned records in function slot layout order.
  const loom_amdgpu_slot_layout_record_t* records;
  // Number of records in |records|.
  iree_host_size_t record_count;
} loom_amdgpu_slot_layout_t;

// Computes fixed AMDGPU segment sizes for every low.slot owned by
// |function_op|. Stack and scratch slots are rejected until their ABI lowering
// is represented in the native path.
iree_status_t loom_amdgpu_slot_layout_collect_segment_sizes(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_slot_layout_segment_sizes_t* out_sizes);

// Builds the full fixed-segment layout for every low.slot owned by
// |function_op|. Records are arena-owned and stay valid for the arena lifetime.
iree_status_t loom_amdgpu_slot_layout_build(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_arena_allocator_t* arena, loom_amdgpu_slot_layout_t* out_layout);

// Looks up |slot_ref| in a previously built layout.
iree_status_t loom_amdgpu_slot_layout_lookup_slot(
    const loom_amdgpu_slot_layout_t* layout, loom_symbol_ref_t slot_ref,
    loom_amdgpu_slot_layout_slot_t* out_slot);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_SLOT_LAYOUT_H_
