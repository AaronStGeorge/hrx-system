// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prospective concat placement using retained allocation and storage facts.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_CONCAT_RESERVATION_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_CONCAT_RESERVATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/codegen/low/allocation/search.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_low_allocation_concat_source_constraint_e {
  // No fixed or assigned concat source constrains the result reservation.
  LOOM_LOW_ALLOCATION_CONCAT_SOURCE_CONSTRAINT_NONE = 0,
  // Placed sources cannot occupy the corresponding slices of one result.
  LOOM_LOW_ALLOCATION_CONCAT_SOURCE_CONSTRAINT_INCOMPATIBLE = 1,
  // Placed linear sources require one exact result base.
  LOOM_LOW_ALLOCATION_CONCAT_SOURCE_CONSTRAINT_LINEAR_LOCATION = 2,
  // Explicit physical views require validation for each result candidate.
  LOOM_LOW_ALLOCATION_CONCAT_SOURCE_CONSTRAINT_EXPLICIT_CANDIDATE = 3,
} loom_low_allocation_concat_source_constraint_kind_t;

// Fixed or assigned source feasibility retained while collecting a concat's
// ignored source set. Linear storage has at most one compatible aggregate base;
// explicit physical views may contain the same source in several candidates.
typedef struct loom_low_allocation_concat_source_constraint_t {
  // Form of source-placement constraint on the result reservation.
  loom_low_allocation_concat_source_constraint_kind_t kind;
  // Required result base when |kind| is LINEAR_LOCATION.
  uint32_t linear_result_location_base;
} loom_low_allocation_concat_source_constraint_t;

// Source facts retained while preparing one prospective concat reservation.
typedef struct loom_low_allocation_concat_reservation_sources_t {
  // Sources whose storage may alias the aggregate after source constraints
  // establish that each committed location matches its result subrange.
  const loom_value_id_t* ignored_value_ids;
  // Number of entries in |ignored_value_ids|.
  uint16_t ignored_value_count;
  // Feasibility constraint imposed by fixed or committed source assignments.
  loom_low_allocation_concat_source_constraint_t source_constraint;
} loom_low_allocation_concat_reservation_sources_t;

// Retains source feasibility and the corresponding ignored source set for one
// prospective reservation. Fixed and already assigned sources constrain the
// aggregate's location; committed sources with competing affinities remain
// ordinary interference. Uses caller-owned inline storage when it fits, or
// allocates from |arena|. Both storage lifetimes must extend through
// reservation search and publication. The source set and constraint describe
// one snapshot of the current assignments and must be prepared again after
// assignments change.
iree_status_t loom_low_allocation_concat_reservation_prepare_sources(
    const loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* result_interval,
    const loom_low_placement_relation_range_t* range,
    loom_value_id_t* inline_ignored_value_ids,
    iree_host_size_t inline_ignored_value_capacity,
    iree_arena_allocator_t* arena,
    loom_low_allocation_concat_reservation_sources_t* out_sources);

// Selects a future concat result reservation or a current source assignment
// when that avoids a lower residency tier. The source and result intervals
// share a value class; |result_range| contains the assembly's retained
// placement relations. |sources| retains both the concat source set and its
// placement feasibility so candidate search does not rescan the aggregate.
// |source_has_unique_affinity_path| proves that allocation derived this
// relation from one producer path even when the source has no authored
// placement relation. The relation's alias flag distinguishes a storage
// handoff from an early reservation trigger whose source remains live.
// The result may instead be the assembly's exclusive whole-value
// edge destination when the caller proves that source storage is consumed
// before destination acquisition.
//
// A value_id of INVALID means no placement was selected. This query does not
// publish the assignment: the allocator retains ownership of active storage,
// liveness and leases. Result reservations may share leases with the ignored
// sources; a source assignment requires no ignored leases.
// Optional reservations preserve outstanding source leases. Ordinary placement
// owns the decision to release those leases for capacity or residency; result
// leases marked releasable for pressure remain available to reservations.
iree_status_t loom_low_allocation_concat_reservation_find(
    loom_low_allocation_search_context_t* context,
    const loom_liveness_interval_t* source_interval,
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* result_interval,
    const loom_low_placement_relation_range_t* result_range,
    const loom_low_allocation_concat_reservation_sources_t* sources,
    bool source_has_unique_affinity_path,
    loom_low_allocation_assignment_t* out_assignment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_CONCAT_RESERVATION_H_
