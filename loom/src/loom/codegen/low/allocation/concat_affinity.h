// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-owned concat placement affinities.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_CONCAT_AFFINITY_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_CONCAT_AFFINITY_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/placement.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_concat_affinity_t
    loom_low_allocation_concat_affinity_t;

// Retains unique concat destinations after destructive-reuse refinement.
// Construction first composes authored concat chains in placement's
// users-before-sources order. It then walks retained liveness operations in
// reverse for register classes with explicit physical-register views. A unique
// equal-width producer becomes actionable when its physical last-use handoff
// has competing direct uses, or when its storage remains observable and it has
// exactly one direct operation use. The former inherits the concat slice; the
// latter carries a reservation-only path. Contiguous register classes and
// linear last-use chains need no early aggregate commitment. Captured
// components and ambiguous destinations or producers stop propagation. The
// pooled index belongs to the allocation decision arena, not the retained IR;
// functions without concats return NULL.
iree_status_t loom_low_allocation_concat_affinity_build(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_placement_table_t* placement,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    iree_arena_allocator_t* arena,
    loom_low_allocation_concat_affinity_t** out_affinity);

// Returns the unique concat span retained for |source_ordinal| as a preferred
// relation. CAN_ALIAS_STORAGE identifies a physical last-use handoff; its
// absence identifies an early reservation trigger whose source storage remains
// observable. Callers use this only for values without an authored concat
// relation so direct placement policy remains authoritative.
bool loom_low_allocation_concat_affinity_relation_for_source(
    const loom_low_allocation_concat_affinity_t* affinity,
    loom_value_ordinal_t source_ordinal,
    loom_low_placement_relation_t* out_relation);

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
