// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-owned construction of placement and fixed-location facts.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_PLACEMENT_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_PLACEMENT_H_

#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/placement.h"

#ifdef __cplusplus
extern "C" {
#endif

// Collects placement constraints, normalizes fixed locations and chooses
// transport sources before publishing the single endpoint-indexed table.
// Fixed location facts belong to |target_constraints|. Complete assignments and
// their conflict index are published by finalize_fixed_values after
// unit-lifetime refinement. |arena| owns the placement table;
// |preference_arena| owns working preferences that allocation releases after
// final register numbering.
iree_status_t loom_low_allocation_placement_build(
    loom_low_allocation_target_constraints_t* target_constraints,
    const loom_region_t* region, const loom_local_value_domain_t* value_domain,
    const loom_value_ordinal_t* storage_origins,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_fixed_value_t* fixed_values,
    iree_host_size_t fixed_value_count,
    loom_low_placement_pair_use_list_t pair_uses,
    loom_low_placement_instruction_preferences_t instruction_preferences,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* preference_arena,
    loom_low_placement_table_t* out_table,
    loom_low_placement_preference_index_t* out_preferences);

typedef struct loom_low_allocation_concat_affinity_t
    loom_low_allocation_concat_affinity_t;

// Retains unique aliasable concat chains ending in a single control-flow
// transfer after destructive-reuse refinement. Construction visits the
// placement's users-before-sources order once. Captured components and
// ambiguous users preclude a transitive preference, so placement never scans
// a destination fanout once per component of a nested concat. The pooled
// index belongs to the allocation decision arena, not the retained IR;
// functions without concats or control-flow transfers return NULL.
iree_status_t loom_low_allocation_concat_affinity_build(
    const loom_low_placement_table_t* placement, iree_arena_allocator_t* arena,
    loom_low_allocation_concat_affinity_t** out_affinity);

// Projects a concat relation's destination span through its retained unique
// concat chain in constant time. Returns false and leaves |relation| unchanged
// when no chain covers its units. This is a location preference, not a storage
// identity proof; the allocator must still check the complete lifetime.
bool loom_low_allocation_concat_affinity_project(
    const loom_low_allocation_concat_affinity_t* affinity,
    loom_low_placement_relation_t* relation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_PLACEMENT_H_
