// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent fixed layout for function-local low storage.
//
// low.storage.reserve ops declare byte-addressable storage owned by the current
// low function. This layout packs reservations by storage space in function
// body order, honoring each reservation's byte alignment before assigning its
// stable byte offset. Targets project the generic storage spaces onto their ABI
// frame, private, scratch, local, or workgroup storage mechanisms.

#ifndef LOOM_CODEGEN_LOW_STORAGE_LAYOUT_H_
#define LOOM_CODEGEN_LOW_STORAGE_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_storage_layout_space_sizes_t {
  // Bytes reserved in function stack-frame storage.
  uint64_t stack_bytes;
  // Bytes reserved in per-lane spill/scratch storage.
  uint64_t scratch_bytes;
  // Bytes reserved in target-private per-invocation storage.
  uint64_t private_bytes;
  // Bytes reserved in workgroup-local shared storage.
  uint64_t workgroup_bytes;
} loom_low_storage_layout_space_sizes_t;

typedef struct loom_low_storage_layout_reservation_t {
  // Function-local storage space containing this reservation.
  loom_storage_space_t space;
  // Byte offset assigned within |space|.
  uint64_t byte_offset;
  // Reservation size in bytes.
  uint64_t byte_size;
  // Reservation alignment in bytes.
  uint64_t byte_alignment;
} loom_low_storage_layout_reservation_t;

typedef struct loom_low_storage_layout_record_t {
  // SSA value produced by the low.storage.reserve op.
  loom_value_id_t storage_value_id;
  // Layout assigned to |storage_value_id|.
  loom_low_storage_layout_reservation_t reservation;
} loom_low_storage_layout_record_t;

typedef struct loom_low_storage_layout_t {
  // Total bytes reserved in each function-local storage space.
  loom_low_storage_layout_space_sizes_t space_sizes;
  // Arena-owned records in function body declaration order.
  const loom_low_storage_layout_record_t* records;
  // Number of entries in |records|.
  iree_host_size_t record_count;
} loom_low_storage_layout_t;

typedef struct loom_low_storage_layout_reference_t {
  // Root reservation containing the referenced storage bytes.
  loom_low_storage_layout_reservation_t reservation;
  // Byte offset from |reservation| to the referenced storage view.
  uint64_t byte_offset;
  // Static byte length visible through the referenced storage handle.
  uint64_t byte_length;
} loom_low_storage_layout_reference_t;

// Callback invoked for each packed low.storage.reserve reservation.
typedef iree_status_t (*loom_low_storage_layout_reservation_callback_t)(
    void* user_data, loom_value_id_t storage_value_id,
    const loom_low_storage_layout_reservation_t* reservation);

// Callback used to resolve a root low.storage.reserve placement while walking a
// low.storage.view chain.
typedef iree_status_t (*loom_low_storage_layout_reservation_lookup_fn_t)(
    void* user_data, loom_value_id_t storage_value_id,
    loom_low_storage_layout_reservation_t* out_reservation);

// Visits packed low.storage.reserve ops in |function_op| in function body
// declaration order.
iree_status_t loom_low_storage_layout_visit_reservations(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_low_storage_layout_reservation_callback_t callback, void* user_data);

// Computes packed byte usage for low.storage.reserve ops in |function_op|.
iree_status_t loom_low_storage_layout_collect_space_sizes(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_low_storage_layout_space_sizes_t* out_sizes);

// Builds the complete reservation layout for low.storage.reserve ops in
// |function_op|. Records are arena-owned and remain valid for |arena|'s
// lifetime.
iree_status_t loom_low_storage_layout_build(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_arena_allocator_t* arena, loom_low_storage_layout_t* out_layout);

// Looks up a low.storage.reserve result in a previously built layout.
iree_status_t loom_low_storage_layout_lookup_reservation(
    const loom_low_storage_layout_t* layout, loom_value_id_t storage_value_id,
    loom_low_storage_layout_reservation_t* out_reservation);

// Resolves a low.storage.reserve result by scanning |function_op| without
// materializing a full record table.
iree_status_t loom_low_storage_layout_resolve_reservation(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_value_id_t storage_value_id,
    loom_low_storage_layout_reservation_t* out_reservation);

// Resolves a low.storage.reserve or low.storage.view handle against |layout|.
iree_status_t loom_low_storage_layout_lookup_reference(
    const loom_low_storage_layout_t* layout, const loom_module_t* module,
    loom_value_id_t storage_value_id,
    loom_low_storage_layout_reference_t* out_reference);

// Resolves a low.storage.reserve or low.storage.view handle by asking
// |lookup_reservation| for the root low.storage.reserve placement. Targets that
// project generic storage spaces onto target-specific segments can use this to
// share storage-view validation while keeping target segment offsets.
iree_status_t loom_low_storage_layout_resolve_reference_from_reservations(
    const loom_module_t* module, loom_value_id_t storage_value_id,
    loom_low_storage_layout_reservation_lookup_fn_t lookup_reservation,
    void* lookup_user_data, loom_low_storage_layout_reference_t* out_reference);

// Resolves a low.storage.reserve or low.storage.view handle by scanning
// |function_op| without materializing a full record table.
iree_status_t loom_low_storage_layout_resolve_reference(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_value_id_t storage_value_id,
    loom_low_storage_layout_reference_t* out_reference);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_STORAGE_LAYOUT_H_
