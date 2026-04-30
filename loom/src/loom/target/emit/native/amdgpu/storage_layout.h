// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU fixed-segment layout for function-local storage.
//
// This layout is the single contract shared by kernel metadata and native
// encoding. Each low.storage.reserve inside a target-low function is packed in
// function body order into the target segment selected by its storage type,
// with the reservation's requested alignment applied before assigning its byte
// offset.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_STORAGE_LAYOUT_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_STORAGE_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/ops/low/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_storage_layout_segment_sizes_t {
  // Fixed bytes of workgroup storage required by low.storage.reserve ops.
  uint64_t group_segment_fixed_size;
  // Fixed bytes of invocation-private storage required by low.storage.reserve
  // ops.
  uint64_t private_segment_fixed_size;
} loom_amdgpu_storage_layout_segment_sizes_t;

typedef struct loom_amdgpu_storage_layout_reservation_t {
  // Storage segment containing the resolved reservation.
  loom_storage_space_t space;
  // Byte offset assigned to the reservation within its storage segment.
  uint64_t byte_offset;
  // Reservation size in bytes.
  uint64_t byte_size;
  // Reservation alignment in bytes.
  uint64_t byte_alignment;
} loom_amdgpu_storage_layout_reservation_t;

typedef struct loom_amdgpu_storage_layout_record_t {
  // SSA storage value resolved by this record.
  loom_value_id_t storage_value_id;
  // Segment placement assigned to the storage reservation.
  loom_amdgpu_storage_layout_reservation_t reservation;
} loom_amdgpu_storage_layout_record_t;

typedef struct loom_amdgpu_storage_layout_t {
  // Fixed segment sizes after laying out all function-local storage.
  loom_amdgpu_storage_layout_segment_sizes_t segment_sizes;
  // Arena-owned records in function storage layout order.
  const loom_amdgpu_storage_layout_record_t* records;
  // Number of records in |records|.
  iree_host_size_t record_count;
} loom_amdgpu_storage_layout_t;

// Computes fixed AMDGPU segment sizes for every low.storage.reserve in
// |function_op|. Stack and scratch storage are rejected until their ABI
// lowering is represented in the native path.
iree_status_t loom_amdgpu_storage_layout_collect_segment_sizes(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_storage_layout_segment_sizes_t* out_sizes);

// Builds the full fixed-segment layout for every low.storage.reserve in
// |function_op|. Records are arena-owned and stay valid for the arena lifetime.
iree_status_t loom_amdgpu_storage_layout_build(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_arena_allocator_t* arena, loom_amdgpu_storage_layout_t* out_layout);

// Looks up |storage_value_id| in a previously built layout.
iree_status_t loom_amdgpu_storage_layout_lookup(
    const loom_amdgpu_storage_layout_t* layout,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reservation_t* out_reservation);

// Resolves one storage reservation from |function_op| without materializing the
// complete layout record table.
iree_status_t loom_amdgpu_storage_layout_resolve(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reservation_t* out_reservation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_STORAGE_LAYOUT_H_
