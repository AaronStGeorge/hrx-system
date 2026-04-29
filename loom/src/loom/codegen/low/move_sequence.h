// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent sequencing for parallel target-low storage moves.

#ifndef LOOM_CODEGEN_LOW_MOVE_SEQUENCE_H_
#define LOOM_CODEGEN_LOW_MOVE_SEQUENCE_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation.h"

#ifdef __cplusplus
extern "C" {
#endif

// One target-visible allocation unit.
typedef struct loom_low_move_location_t {
  // Target-visible storage kind.
  loom_low_allocation_location_kind_t location_kind;
  // Storage class for the unit.
  loom_liveness_value_class_t value_class;
  // Descriptor-set-local register class ID for |value_class|.
  uint16_t descriptor_reg_class_id;
  // Physical register, target ID, or spill slot ordinal.
  uint32_t location;
} loom_low_move_location_t;

// One parallel move from an old source unit to a destination unit.
typedef struct loom_low_move_t {
  // Unit overwritten by the move.
  loom_low_move_location_t destination;
  // Unit read by the move.
  loom_low_move_location_t source;
} loom_low_move_t;

typedef iree_status_t (*loom_low_move_sequence_emit_fn_t)(
    void* user_data, const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source);

// Callback used to emit one already-sequenced scalar storage move.
typedef struct loom_low_move_sequence_emit_callback_t {
  // Emits |destination| <- |source|.
  loom_low_move_sequence_emit_fn_t fn;
  // Opaque user data passed to |fn|.
  void* user_data;
} loom_low_move_sequence_emit_callback_t;

// Options for lowering one parallel move set to a linear sequence.
typedef struct loom_low_move_sequence_options_t {
  // Scratch units available to break cycles.
  const loom_low_move_location_t* temporary_locations;
  // Number of entries in |temporary_locations|.
  iree_host_size_t temporary_location_count;
  // Required sequenced move emitter.
  loom_low_move_sequence_emit_callback_t emit_move;
} loom_low_move_sequence_options_t;

// Returns true when two move locations name the same target-visible unit.
bool loom_low_move_locations_equal(const loom_low_move_location_t* lhs,
                                   const loom_low_move_location_t* rhs);

// Returns the location for one assignment unit.
iree_status_t loom_low_move_location_from_assignment_unit(
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_index,
    loom_low_move_location_t* out_location);

// Counts scalar unit moves required by one allocation edge-copy group.
iree_status_t loom_low_move_sequence_count_edge_copy_units(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_edge_copy_group_t* group,
    iree_host_size_t* out_move_count);

// Populates |moves| from one allocation edge-copy group. |move_count| must
// match loom_low_move_sequence_count_edge_copy_units.
iree_status_t loom_low_move_sequence_populate_edge_copy_units(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_edge_copy_group_t* group, loom_low_move_t* moves,
    iree_host_size_t move_count);

// Populates |temporary_locations| from the scratch units planned for one
// allocation edge-copy group. |temporary_location_count| must match
// group->temporary_count.
iree_status_t loom_low_move_sequence_populate_edge_copy_temporaries(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_edge_copy_group_t* group,
    loom_low_move_location_t* temporary_locations,
    iree_host_size_t temporary_location_count);

// Counts scalar unit moves required to materialize one low.slice. Coalesced
// structural slices require zero moves.
iree_status_t loom_low_move_sequence_count_slice_units(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    iree_host_size_t* out_move_count);

// Populates |moves| from one low.slice. |move_count| must match
// loom_low_move_sequence_count_slice_units.
iree_status_t loom_low_move_sequence_populate_slice_units(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    loom_low_move_t* moves, iree_host_size_t move_count);

// Counts scalar unit moves required to materialize one low.concat. Coalesced
// structural concats require zero moves.
iree_status_t loom_low_move_sequence_count_concat_units(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    iree_host_size_t* out_move_count);

// Populates |moves| from one low.concat. |move_count| must match
// loom_low_move_sequence_count_concat_units.
iree_status_t loom_low_move_sequence_populate_concat_units(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op,
    loom_low_move_t* moves, iree_host_size_t move_count);

// Emits |moves| as a sequential move list that preserves parallel-copy
// semantics. Identity moves are elided. Acyclic overlap is handled by choosing
// a safe order. Cycles use a matching storage-class entry in
// |temporary_locations| and otherwise fail loud with
// IREE_STATUS_FAILED_PRECONDITION.
//
// |moves| is caller-owned scratch storage and is mutated during sequencing.
iree_status_t loom_low_move_sequence_emit(
    loom_low_move_t* moves, iree_host_size_t move_count,
    const loom_low_move_sequence_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_MOVE_SEQUENCE_H_
