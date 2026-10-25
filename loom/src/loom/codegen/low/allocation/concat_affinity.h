// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-owned concat placement affinities.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_CONCAT_AFFINITY_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_CONCAT_AFFINITY_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/placement.h"

#ifdef __cplusplus
extern "C" {
#endif

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

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_CONCAT_AFFINITY_H_
