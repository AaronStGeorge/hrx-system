// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Dominance queries for loom IR.
//
// For structured regions (no LOOM_REGION_INSTANCE_FLAG_CFG), dominance
// is read directly from the IR structure via parent_op chains — no
// precomputation required. Each query walks the ancestry of the two
// ops to determine their relationship:
//
//   Same block:      a dominates b if a comes before b (array index).
//   Ancestor scope:  an op in an enclosing region dominates all ops
//                    in nested regions, unless an isolated-from-above
//                    boundary intervenes.
//   Entry block:     block 0 of a multi-block region dominates all
//                    sibling blocks.
//
// For CFG regions (future), the dominance info struct will cache a
// computed dominator tree from predecessor edges. The query API is
// identical — callers don't need to know which strategy is in use.
//
// Usage:
//
//   loom_dominance_info_t dom_info;
//   loom_dominance_info_initialize(module, &arena, &dom_info);
//
//   if (loom_dominates_op(&dom_info, producer, consumer)) {
//     // producer's results are visible at consumer.
//   }
//
//   if (loom_dominates_value(&dom_info, value_id, use_op)) {
//     // value_id is in scope at use_op.
//   }

#ifndef LOOM_UTIL_DOMINANCE_H_
#define LOOM_UTIL_DOMINANCE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dominance info
//===----------------------------------------------------------------------===//

// Dominance analysis state. For structured IR this is lightweight
// (just a module reference). For CFG regions it will cache dominator
// trees allocated from the arena.
typedef struct loom_dominance_info_t {
  const loom_module_t* module;
  iree_arena_allocator_t* arena;
} loom_dominance_info_t;

// Initializes dominance info for the given module.
void loom_dominance_info_initialize(const loom_module_t* module,
                                    iree_arena_allocator_t* arena,
                                    loom_dominance_info_t* out_info);

//===----------------------------------------------------------------------===//
// Dominance queries
//===----------------------------------------------------------------------===//

// Returns the nesting depth of |op| by counting parent_op hops.
// Top-level ops (parent_op == NULL) have depth 0.
uint16_t loom_op_nesting_depth(const loom_op_t* op);

// Returns the ancestor of |op| at the given |target_depth|.
// Walks parent_op pointers |depth - target_depth| times.
// Returns |op| itself if already at the target depth.
const loom_op_t* loom_op_ancestor_at_depth(const loom_op_t* op,
                                           uint16_t target_depth);

// Does op |a| dominate op |b|?
//
// Dominance rules for structured IR:
//   - Self-dominance: a dominates a.
//   - Same block: a dominates b if a appears before b in the block's
//     ops array.
//   - Ancestor scope: a dominates b if a is in an enclosing scope
//     (reachable via b's parent_op chain) and no isolated-from-above
//     boundary is crossed.
//   - Entry block dominance: an op in block 0 of a multi-block region
//     dominates ops in sibling blocks of the same region.
//
// Both ops must have valid parent_op/parent_block pointers (set by
// the builder or loom_module_compute_uses).
bool loom_dominates_op(const loom_dominance_info_t* info, const loom_op_t* a,
                       const loom_op_t* b);

// Does the value defined by |value_id| dominate op |use_op|?
//
// For op results: equivalent to loom_dominates_op(defining_op, use_op).
// For block arguments: the value dominates all ops in its block and
// all ops in nested regions (subject to isolation boundaries).
bool loom_dominates_value(const loom_dominance_info_t* info,
                          loom_value_id_t value_id, const loom_op_t* use_op);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_DOMINANCE_H_
