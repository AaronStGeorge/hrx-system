// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source register-choice affinities induced by control-flow destinations.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_EDGE_AFFINITY_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_EDGE_AFFINITY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// One whole-value source/destination correspondence on a logical edge.
typedef struct loom_low_edge_affinity_input_t {
  // Destination's mandatory storage-owner ordinal.
  loom_value_ordinal_t destination_origin;
  // Source's mandatory storage-owner ordinal.
  loom_value_ordinal_t source_origin;
  // Identity shared by inputs forwarded on the same logical edge.
  uint32_t edge_group;
  // Stable placement-relation position used to retain provenance.
  uint32_t sequence;
  // Operation forwarding the source.
  const loom_op_t* op;
} loom_low_edge_affinity_input_t;

// One unordered hard register-ordinal pair between destination storage owners.
typedef struct loom_low_edge_affinity_constraint_t {
  // First destination storage-owner ordinal.
  loom_value_ordinal_t first;
  // Second destination storage-owner ordinal.
  loom_value_ordinal_t second;
} loom_low_edge_affinity_constraint_t;

// One preferred register-ordinal pair between control-flow sources.
typedef struct loom_low_edge_affinity_t {
  // First source storage-owner ordinal.
  loom_value_ordinal_t first;
  // Second source storage-owner ordinal.
  loom_value_ordinal_t second;
  // Number of distinct logical edges benefiting from this pairing.
  uint16_t priority;
  // Earliest forwarding operation contributing this pairing.
  const loom_op_t* op;
} loom_low_edge_affinity_t;

// Arena-owned source affinity rows sorted by source ordinal pair.
typedef struct loom_low_edge_affinity_list_t {
  // Immutable affinity rows.
  const loom_low_edge_affinity_t* values;
  // Number of rows in values.
  iree_host_size_t count;
} loom_low_edge_affinity_list_t;

// Composes hard destination register-ordinal pairs backward through matching
// logical control-flow edges. Duplicate constraints and inputs on one edge are
// collapsed; repeated edges increase the preference priority. Source pairs
// that are already hard-constrained are omitted. The input and constraint
// arrays are working storage and may be reordered.
iree_status_t loom_low_edge_affinity_build(
    loom_low_edge_affinity_input_t* inputs, iree_host_size_t input_count,
    loom_low_edge_affinity_constraint_t* constraints,
    iree_host_size_t constraint_count, loom_value_ordinal_t value_count,
    iree_arena_allocator_t* arena,
    loom_low_edge_affinity_list_t* out_affinities);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_EDGE_AFFINITY_H_
