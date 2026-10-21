// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical writes constrained by retained instruction reads.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_WRITE_INTERFERENCE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_WRITE_INTERFERENCE_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/assignment_map.h"
#include "loom/codegen/low/allocation/move.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/placement.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Arena-owned retained-read events and indexed physical-write constraints.
// Unit-liveness owns operand collection; fixed-binding resolution completes
// the table before any assignment. Queries never inspect IR or rescan events.
typedef struct loom_low_allocation_write_interference_t
    loom_low_allocation_write_interference_t;

// Dense identity of one canonical storage owner in a retained-write table.
// Original SSA values map into this domain; entries beyond that prefix do not
// require module value IDs.
typedef uint32_t loom_low_allocation_storage_origin_t;
#define LOOM_LOW_ALLOCATION_STORAGE_ORIGIN_INVALID \
  ((loom_low_allocation_storage_origin_t)UINT32_MAX)

// One register value occurrence bound to a descriptor operand. Planning
// domains supply storage identities directly; no SSA value is required.
typedef struct loom_low_allocation_write_occurrence_t {
  // Canonical storage identity observed or written by the instruction.
  loom_low_allocation_storage_origin_t origin;
  // Number of consecutive allocation units occupied by the occurrence.
  uint32_t unit_count;
  // Descriptor-local register class of the occurrence.
  uint16_t register_class;
} loom_low_allocation_write_occurrence_t;

// Borrowed assignment lookup indexed by canonical storage origin. Allocation
// over original SSA values may project its value-ordinal index directly;
// extended allocation domains supply their own index over the same records.
typedef struct loom_low_allocation_write_assignment_map_t {
  // Published assignment records.
  const loom_low_allocation_assignment_t* assignments;
  // Assignment indices by storage origin, or UINT32_MAX when unpublished.
  const uint32_t* assignment_indices_by_origin;
} loom_low_allocation_write_assignment_map_t;

// Projects an ordinary local-value assignment map into its origin-compatible
// lookup view. Canonical original origins retain their value ordinals.
static inline loom_low_allocation_write_assignment_map_t
loom_low_allocation_write_assignment_map_from_value_map(
    const loom_low_allocation_assignment_map_t* map) {
  return (loom_low_allocation_write_assignment_map_t){
      .assignments = map->assignments,
      .assignment_indices_by_origin = map->assignment_indices_by_value_ordinal,
  };
}

// Creates arena-owned collection state for targets with an active
// read-retention rule. Other targets leave |out_interference| NULL and allocate
// no state. Unit-liveness has already bounded the sum of register interval
// widths to UINT32_MAX; retained and copy-unit domains are subsets of it.
iree_status_t loom_low_allocation_write_interference_allocate(
    const loom_low_resolved_target_t* target,
    const loom_low_placement_table_t* placement,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_allocation_write_interference_t** out_interference);

// Records one bound descriptor-operand occurrence. Pass NULL for an implicit
// state operand. Explicit non-register values and descriptor rows that do not
// map to packet values are omitted by the caller. |point| is the instruction's
// write point, after its ordinary operand uses.
iree_status_t loom_low_allocation_write_interference_note_occurrence(
    loom_low_allocation_write_interference_t* interference,
    const loom_low_descriptor_set_t* descriptors,
    const loom_low_descriptor_t* descriptor, uint16_t operand_index,
    const loom_low_allocation_write_occurrence_t* occurrence, uint32_t point,
    iree_arena_allocator_t* arena);

// Retains the resolved fixed location of a value in the tracked bank. Required
// tied aliases supply their canonical origin; other banks contribute no facts.
void loom_low_allocation_write_interference_note_fixed(
    loom_low_allocation_write_interference_t* interference,
    loom_low_allocation_storage_origin_t origin,
    const loom_low_allocation_assignment_t* assignment);

// Finalizes path-sensitive constraints and per-point temporary exclusions.
// This consumes canonical liveness order, root CFG, structural placements and
// completed per-unit physical lifetimes; the same operation covers scheduled
// and source-order allocation. Required ties retain storage independently of
// the semantic lifetime of their first SSA name.
// The completed |unit_liveness| is borrowed through all subsequent queries;
// it and its per-unit ends remain unchanged for that lifetime.
// Persistent indexes use |arena|; construction scratch borrows and restores
// |scratch_arena|'s tail. The two arenas must be distinct.
iree_status_t loom_low_allocation_write_interference_finalize(
    loom_low_allocation_write_interference_t* interference,
    const loom_liveness_analysis_t* liveness, const loom_cfg_graph_t* cfg_graph,
    const loom_low_placement_table_t* placement,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena);

// Records a published assignment before the next candidate query. A successful
// query's inferred locations remain a valid positive witness when publication
// agrees with them. Other tracked physical assignments invalidate that witness;
// provisional spills only remove constraints.
void loom_low_allocation_write_interference_note_assignment(
    loom_low_allocation_write_interference_t* interference,
    loom_low_allocation_storage_origin_t origin,
    const loom_low_allocation_assignment_t* assignment);

// Ends an assignment attempt's inference lifetime on success or failure. The
// immutable constraints remain available for a new coloring or relocation.
void loom_low_allocation_write_interference_reset_inference(
    loom_low_allocation_write_interference_t* interference);

// Returns the canonical storage origin of one original local value.
loom_low_allocation_storage_origin_t
loom_low_allocation_write_interference_storage_origin_for_value_ordinal(
    const loom_low_allocation_write_interference_t* interference,
    loom_value_ordinal_t value_ordinal);

// Returns the retained origin rejecting a candidate, or INVALID. The search
// owner skips this query when the optional finalized table is absent. Future
// fixed assignments participate. Producer-proved zero-copy equations propagate
// in both directions using reusable inference scratch, without assigning their
// endpoints. Successful inference remains reusable until assignment publication
// invalidates it or the owning attempt ends. Provisional spills end
// propagation: spill plans rebuild allocation before emission. Required aliases
// query their canonical storage origin. Hypothetical recoloring of an already
// located root does not publish its implications for queries rooted at other
// values. |unit_end_points| selects the current assignment attempt's physical
// lifetime view; it may shorten canonical source observations after a covering
// structural value is materialized.
loom_low_allocation_storage_origin_t
loom_low_allocation_write_interference_conflicting_read(
    loom_low_allocation_write_interference_t* interference,
    const loom_low_allocation_write_assignment_map_t* assignments,
    const uint32_t* unit_end_points,
    loom_low_allocation_storage_origin_t candidate_origin,
    const loom_low_allocation_assignment_t* candidate);

// A complete simultaneous location proposal, indexed by storage origin.
// UINT32_MAX entries preserve the current assignment. Affected origins are
// listed once so transaction checks visit only their indexed constraints.
typedef struct loom_low_allocation_write_proposal_t {
  // Proposed base per canonical origin, or UINT32_MAX when unchanged.
  uint32_t* bases;
  // Canonical origins whose locations change together.
  loom_low_allocation_storage_origin_t* origins;
  // Number of initialized entries in |origins|.
  iree_host_size_t count;
} loom_low_allocation_write_proposal_t;

// Allocates reusable transaction scratch only when a rule is active.
iree_status_t loom_low_allocation_write_proposal_initialize(
    const loom_low_allocation_write_interference_t* interference,
    iree_arena_allocator_t* arena,
    loom_low_allocation_write_proposal_t* out_proposal);

// Clears the preceding proposal without walking unaffected values.
void loom_low_allocation_write_proposal_reset(
    loom_low_allocation_write_proposal_t* proposal);

// Records one member of a complete coalesced relocation group or an eviction.
void loom_low_allocation_write_proposal_add(
    const loom_low_allocation_write_interference_t* interference,
    loom_low_allocation_storage_origin_t origin,
    const loom_low_allocation_assignment_t* assignment, uint32_t base,
    loom_low_allocation_write_proposal_t* proposal);

// Validates all endpoints against the simultaneous proposal before mutation.
bool loom_low_allocation_write_proposal_conflicts(
    loom_low_allocation_write_interference_t* interference,
    const loom_low_allocation_write_assignment_map_t* assignments,
    const loom_low_allocation_write_proposal_t* proposal);

// Returns whether a copy-cycle temporary would overwrite a retained unit at
// |write_point|. Such a temporary has no SSA lifetime of its own. Move planning
// supplies final spill-free assignments, with physical locations for every
// retained origin in the tracked bank.
bool loom_low_allocation_write_interference_temporary_conflicts(
    const loom_low_allocation_write_interference_t* interference,
    const loom_low_allocation_write_assignment_map_t* assignments,
    uint32_t write_point, const loom_low_move_location_t* temporary);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_WRITE_INTERFERENCE_H_
