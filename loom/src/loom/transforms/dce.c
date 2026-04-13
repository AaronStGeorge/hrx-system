// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/dce.h"

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

enum {
  LOOM_DCE_STAT_OPS_ELIMINATED = 0,
};

static const loom_pass_statistic_def_t kDCEStatistics[] = {
    {IREE_SVL("ops-eliminated"),
     IREE_SVL("Number of dead operations removed.")},
};

static const loom_pass_info_t loom_dce_pass_info_storage = {
    .name = IREE_SVL("dce"),
    .description = IREE_SVL("Remove operations with unused results."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_defs = kDCEStatistics,
    .statistic_count = 1,
};

const loom_pass_info_t* loom_dce_pass_info(void) {
  return &loom_dce_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Block collection
//===----------------------------------------------------------------------===//
//
// Collects all blocks reachable from a function into a flat array,
// including blocks inside nested regions. Uses an iterative region
// stack to avoid recursion on deeply nested untrusted input.

#define LOOM_DCE_INITIAL_CAPACITY 32

static iree_status_t loom_dce_collect_blocks(iree_arena_allocator_t* arena,
                                             loom_region_t* body,
                                             loom_block_t*** out_blocks,
                                             iree_host_size_t* out_count) {
  // Region stack for iterative DFS.
  loom_region_t** region_stack = NULL;
  iree_host_size_t stack_count = 0;
  iree_host_size_t stack_capacity = LOOM_DCE_INITIAL_CAPACITY;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, stack_capacity, sizeof(loom_region_t*), (void**)&region_stack));

  // Output block array.
  loom_block_t** blocks = NULL;
  iree_host_size_t block_count = 0;
  iree_host_size_t block_capacity = LOOM_DCE_INITIAL_CAPACITY;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, block_capacity, sizeof(loom_block_t*), (void**)&blocks));

  // Seed with the function body.
  region_stack[stack_count++] = body;

  while (stack_count > 0) {
    loom_region_t* region = region_stack[--stack_count];

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      // Append block to output array.
      if (block_count >= block_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            arena, block_count, block_count + 1, sizeof(loom_block_t*),
            &block_capacity, (void**)&blocks));
      }
      blocks[block_count++] = block;

      // Push nested regions from ops in this block.
      for (uint16_t i = 0; i < block->op_count; ++i) {
        loom_op_t* op = loom_block_op(block, i);
        if (op->flags & LOOM_OP_FLAG_DEAD) continue;
        if (op->region_count == 0) continue;
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t r = 0; r < op->region_count; ++r) {
          if (!regions[r] || regions[r]->block_count == 0) continue;
          if (stack_count >= stack_capacity) {
            IREE_RETURN_IF_ERROR(iree_arena_grow_array(
                arena, stack_count, stack_count + 1, sizeof(loom_region_t*),
                &stack_capacity, (void**)&region_stack));
          }
          region_stack[stack_count++] = regions[r];
        }
      }
    }
  }

  *out_blocks = blocks;
  *out_count = block_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

iree_status_t loom_dce_run(loom_pass_t* pass, loom_module_t* module,
                           loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  // Collect all blocks (including nested regions) into a flat array.
  loom_block_t** all_blocks = NULL;
  iree_host_size_t block_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_dce_collect_blocks(pass->arena, body, &all_blocks, &block_count));

  // Walk all blocks. Within each block, walk ops in reverse so that
  // erasing a dead op may make its operand-producing ops newly dead
  // (their use count drops, and we haven't visited them yet).
  // The outer fixed-point loop handles cascading across blocks: erasing
  // an op in one block may make ops in other blocks dead.
  bool changed = true;
  while (changed) {
    changed = false;
    for (iree_host_size_t b = 0; b < block_count; ++b) {
      loom_block_t* block = all_blocks[b];
      for (int32_t i = (int32_t)block->op_count - 1; i >= 0; --i) {
        loom_op_t* op = loom_block_op(block, (uint16_t)i);
        if (op->flags & LOOM_OP_FLAG_DEAD) continue;
        if (loom_op_is_trivially_dead(module, op)) {
          IREE_RETURN_IF_ERROR(loom_op_erase(module, op));
          if (pass->statistics) {
            loom_pass_statistic_add(pass, LOOM_DCE_STAT_OPS_ELIMINATED, 1);
          }
          changed = true;
        }
      }
    }
  }
  return iree_ok_status();
}
