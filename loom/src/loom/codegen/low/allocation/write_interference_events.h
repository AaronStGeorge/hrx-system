// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native retained-read propagation over physical accesses and control-flow
// spans.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_WRITE_INTERFERENCE_EVENTS_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_WRITE_INTERFERENCE_EVENTS_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_low_allocation_assignment_t;

typedef struct loom_low_write_range_t {
  // First element in the owning array.
  uint32_t start;
  // Number of elements in the range.
  uint32_t count;
} loom_low_write_range_t;

typedef struct loom_low_write_value_t {
  // Required tied-storage origin in the local ordinal domain.
  loom_value_ordinal_t origin;
  // Fixed physical base, or UINT32_MAX when unconstrained.
  uint32_t fixed_base;
  // Collection metadata retires before the query index is constructed.
  union {
    // Register extents used by operand collection and completion
    // classification.
    struct {
      // Width of the value's linear register range.
      uint32_t width;
      // First retained-read unit bit, or UINT32_MAX if never retained.
      uint32_t retained_start;
    } construction;
    // Indexed write constraints and copy equations mentioning this origin.
    loom_low_write_range_t constraints;
  };
} loom_low_write_value_t;

enum loom_low_write_access_flag_bits_e {
  LOOM_LOW_WRITE_ACCESS_RETAIN = 1u << 0,
  LOOM_LOW_WRITE_ACCESS_WRITE = 1u << 1,
  LOOM_LOW_WRITE_ACCESS_INTERFERE = 1u << 2,
};

typedef struct loom_low_write_access_t {
  // Storage origin read or written by the event.
  loom_value_ordinal_t value;
  // Structural source origin, or INVALID for an unconditional access.
  loom_value_ordinal_t source;
  // First accessed unit in |value|.
  uint32_t offset;
  // First copied unit in |source|.
  uint32_t source_offset;
  // Number of accessed units.
  uint32_t count;
  // Next access at this point, or UINT32_MAX.
  uint32_t next;
  // Read, overwrite and hazardous-write effects.
  uint8_t flags;
} loom_low_write_access_t;

typedef struct loom_low_write_event_t {
  // First access at this write point, or UINT32_MAX.
  uint32_t access;
  // Shared immutable snapshot before this point's writes, after its read reset.
  loom_low_write_range_t retained;
  // A replacing read clears the incoming retained domain before writes.
  bool reset;
} loom_low_write_event_t;

typedef struct loom_low_write_retained_t {
  // Canonical storage origin whose read is retained.
  loom_value_ordinal_t value;
  // First retained unit within the origin.
  uint32_t offset;
  // Number of consecutive retained units.
  uint32_t count;
} loom_low_write_retained_t;

// Collected physical accesses and their shared retained-read snapshots.
// The descriptor/placement producer supplies canonical storage identities.
// This state never owns IR or reconstructs control flow.
typedef struct loom_low_write_events_t {
  // One event per write point; point zero is not executed.
  loom_low_write_event_t* points;
  // Number of entries in |points|.
  uint32_t point_count;
  // Arena-owned operand and structural accesses.
  loom_low_write_access_t* accesses;
  // Number of initialized accesses.
  iree_host_size_t access_count;
  // Logical capacity of |accesses|, at most UINT32_MAX.
  iree_host_size_t access_capacity;
  // Immutable retained ranges shared by points with unchanged state.
  loom_low_write_retained_t* retained;
  // Number of initialized retained ranges.
  iree_host_size_t retained_count;
  // Logical capacity of |retained|, at most UINT32_MAX.
  iree_host_size_t retained_capacity;
} loom_low_write_events_t;

// A maximal linear span of write points, including structured entry/exit gaps.
typedef struct loom_low_write_flow_block_t {
  // First included write point.
  uint32_t begin;
  // Exclusive end write point.
  uint32_t end;
  // First outgoing successor in the graph's edge array.
  uint32_t successor_start;
  // Number of outgoing successors.
  uint32_t successor_count;
} loom_low_write_flow_block_t;

// Root CFG refined at structured branches and backedges. Constructed once from
// liveness's accepted operation points, without requiring a schedule table.
typedef struct loom_low_write_flow_t {
  // Arena-owned maximal linear point spans.
  loom_low_write_flow_block_t* blocks;
  // Number of spans.
  uint32_t block_count;
  // Arena-owned successor block indices.
  uint32_t* successors;
  // Recycled construction buffer with at least |block_count| uninitialized
  // entries for the solver's worklist; no graph facts remain in this storage.
  uint32_t* worklist;
} loom_low_write_flow_t;

// Appends a producer-classified access. Points start with access=UINT32_MAX;
// retained-value bit ranges have already been assigned by operand collection.
iree_status_t loom_low_write_events_append(loom_low_write_events_t* events,
                                           uint32_t point,
                                           loom_low_write_access_t access,
                                           iree_arena_allocator_t* arena);

// Returns the proved difference between canonical storage bases. Distinct
// origins have a known difference only when both physical bases are fixed.
static inline bool loom_low_write_known_delta(
    const loom_low_write_value_t* values, loom_value_ordinal_t destination,
    loom_value_ordinal_t source, int64_t* out_delta) {
  if (destination == source) {
    *out_delta = 0;
    return true;
  }
  const uint32_t destination_base = values[destination].fixed_base;
  const uint32_t source_base = values[source].fixed_base;
  if (destination_base == UINT32_MAX || source_base == UINT32_MAX) {
    return false;
  }
  *out_delta = (int64_t)destination_base - source_base;
  return true;
}

// Solves retained reads and counts the corresponding hazardous-write rows.
// |values| supplies collection extents and fixed bases; its retained bit ranges
// partition |retained_unit_count|. |flow| has at least one span and supplies
// reusable worklist storage. Snapshots use |arena|; the decoder and dataflow
// bitmaps use |scratch_arena|. The caller restores scratch together with the
// flow after solving, on success or failure. The arenas must be distinct.
// This runs once, before collection metadata becomes the constraint index.
iree_status_t loom_low_write_events_solve(
    loom_low_write_events_t* events, const loom_low_write_value_t* values,
    iree_host_size_t value_count, uint32_t retained_unit_count,
    const loom_low_write_flow_t* flow, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    iree_host_size_t* out_constraint_count);

// Tests one physical temporary in the tracked bank against a point's retained
// units. Every retained origin has a final, spill-free assignment in the
// supplied origin-indexed lookup; no speculative inference participates.
bool loom_low_write_events_temporary_conflicts(
    const loom_low_write_events_t* events,
    const struct loom_low_allocation_assignment_t* assignments,
    const uint32_t* assignment_indices_by_origin, uint32_t write_point,
    uint32_t temporary_location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_WRITE_INTERFERENCE_EVENTS_H_
