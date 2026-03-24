// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/dominance.h"

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Dominance info
//===----------------------------------------------------------------------===//

void loom_dominance_info_initialize(const loom_module_t* module,
                                    iree_arena_allocator_t* arena,
                                    loom_dominance_info_t* out_info) {
  out_info->module = module;
  out_info->arena = arena;
}

//===----------------------------------------------------------------------===//
// Ancestry helpers
//===----------------------------------------------------------------------===//

uint16_t loom_op_nesting_depth(const loom_op_t* op) {
  uint16_t depth = 0;
  const loom_op_t* current = op->parent_op;
  while (current) {
    ++depth;
    current = current->parent_op;
  }
  return depth;
}

const loom_op_t* loom_op_ancestor_at_depth(const loom_op_t* op,
                                           uint16_t target_depth) {
  uint16_t depth = loom_op_nesting_depth(op);
  IREE_ASSERT(target_depth <= depth);
  while (depth > target_depth) {
    op = op->parent_op;
    --depth;
  }
  return op;
}

// Returns true if blocks |a_block| and |b_block| are siblings in the
// same region, AND |a_block| is the entry block (block 0). This means
// a_block dominates b_block in structured control flow.
static bool loom_entry_block_dominates(const loom_op_t* parent_op,
                                       const loom_block_t* a_block,
                                       const loom_block_t* b_block) {
  if (!parent_op) return false;
  loom_region_t** regions = loom_op_regions(parent_op);
  for (uint8_t r = 0; r < parent_op->region_count; ++r) {
    loom_region_t* region = regions[r];
    if (!region || region->block_count < 2) continue;
    if (&region->blocks[0] == a_block) {
      // a is the entry block. Check if b is a sibling.
      for (uint16_t b = 1; b < region->block_count; ++b) {
        if (&region->blocks[b] == b_block) return true;
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Dominance queries
//===----------------------------------------------------------------------===//

bool loom_dominates_op(const loom_dominance_info_t* info, const loom_op_t* a,
                       const loom_op_t* b) {
  // Self-dominance.
  if (a == b) return true;

  // Same block: compare positions in the ops array.
  if (a->parent_block == b->parent_block) {
    uint16_t index_a = loom_block_find_op(a->parent_block, a);
    uint16_t index_b = loom_block_find_op(b->parent_block, b);
    return index_a < index_b;
  }

  // Same parent op, different blocks: entry block dominance.
  if (a->parent_op == b->parent_op && a->parent_op != NULL) {
    return loom_entry_block_dominates(a->parent_op, a->parent_block,
                                      b->parent_block);
  }

  // Different nesting depths or different parent ops.
  // Walk ancestry chains to find the relationship.
  uint16_t depth_a = loom_op_nesting_depth(a);
  uint16_t depth_b = loom_op_nesting_depth(b);

  // If a is deeper than b, a cannot dominate b (inner doesn't
  // dominate outer).
  if (depth_a > depth_b) return false;

  // Check if a is an ancestor of b: walk b up to a's depth and
  // see if we reach a's parent_op at a's level. If a's parent_op
  // is on b's ancestry chain and a comes before the descendant
  // in the same block, a dominates b.
  //
  // Walk b up to depth_a. At each step, the op we're looking at
  // is the child at that nesting level.
  const loom_op_t* b_at_a_depth = b;
  uint16_t current_depth = depth_b;
  while (current_depth > depth_a) {
    // Check for isolation boundary on b's path.
    if (b_at_a_depth->parent_op) {
      const loom_op_vtable_t* vtable =
          loom_op_vtable(info->module, b_at_a_depth->parent_op);
      if (vtable) {
        loom_trait_flags_t traits =
            vtable->effective_traits
                ? vtable->effective_traits(b_at_a_depth->parent_op)
                : vtable->traits;
        if (loom_traits_is_isolated(traits)) return false;
      }
    }
    b_at_a_depth = b_at_a_depth->parent_op;
    --current_depth;
  }

  // Now b_at_a_depth is at the same depth as a.
  //
  // If a == b_at_a_depth, then a is the op whose region contains b.
  // An op dominates everything in its own regions, so a dominates b.
  // (This is distinct from self-dominance, which was handled above —
  // here a != b but a is b's ancestor.)
  if (a == b_at_a_depth) return true;

  // Different ops at the same depth in the same block: compare positions.
  if (a->parent_block == b_at_a_depth->parent_block) {
    uint16_t index_a = loom_block_find_op(a->parent_block, a);
    uint16_t index_b =
        loom_block_find_op(b_at_a_depth->parent_block, b_at_a_depth);
    return index_a < index_b;
  }

  // Same parent op at the same depth, different blocks: entry block
  // dominance.
  if (a->parent_op == b_at_a_depth->parent_op && a->parent_op != NULL) {
    return loom_entry_block_dominates(a->parent_op, a->parent_block,
                                      b_at_a_depth->parent_block);
  }

  // Different parent ops at the same depth: neither dominates.
  return false;
}

bool loom_dominates_value(const loom_dominance_info_t* info,
                          loom_value_id_t value_id, const loom_op_t* use_op) {
  const loom_value_t* value = &info->module->values.entries[value_id];

  if (value->flags & LOOM_VALUE_FLAG_BLOCK_ARG) {
    // Block argument: dominates all ops in the block and all ops
    // in nested regions (subject to isolation boundaries).
    const loom_block_t* def_block = loom_value_def_block(value);
    if (!def_block) return false;

    // Same block: block args dominate all ops in the block.
    if (use_op->parent_block == def_block) return true;

    // Check if use_op is nested inside the def block's region.
    // Walk use_op's ancestry looking for the def block.
    const loom_op_t* current = use_op;
    while (current->parent_op) {
      // Check isolation on the parent.
      const loom_op_vtable_t* vtable =
          loom_op_vtable(info->module, current->parent_op);
      if (vtable) {
        loom_trait_flags_t traits =
            vtable->effective_traits
                ? vtable->effective_traits(current->parent_op)
                : vtable->traits;
        if (loom_traits_is_isolated(traits)) return false;
      }
      if (current->parent_op->parent_block == def_block) return true;
      current = current->parent_op;
    }
    return false;
  }

  // Op result: the defining op must dominate the use op.
  const loom_op_t* def_op = loom_value_def_op(value);
  if (!def_op) return false;
  return loom_dominates_op(info, def_op, use_op);
}
