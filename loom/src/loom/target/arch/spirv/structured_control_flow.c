// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/structured_control_flow.h"

#include "loom/ops/low/ops.h"

static bool loom_spirv_low_block_backedges_to(
    const loom_block_t* block, const loom_block_t* header_block,
    const loom_op_t** out_backedge_op) {
  const loom_op_t* terminator = block->last_op;
  if (!terminator || !loom_low_br_isa(terminator) ||
      loom_low_br_dest(terminator) != header_block) {
    *out_backedge_op = NULL;
    return false;
  }
  *out_backedge_op = terminator;
  return true;
}

bool loom_spirv_low_select_merge_block(const loom_op_t* op,
                                       const loom_block_t** out_merge_block) {
  *out_merge_block = NULL;
  const loom_block_t* true_dest = loom_low_cond_br_true_dest(op);
  const loom_block_t* false_dest = loom_low_cond_br_false_dest(op);
  const uint16_t header_index = loom_block_region_index(op->parent_block);
  const uint16_t true_index = loom_block_region_index(true_dest);
  const uint16_t false_index = loom_block_region_index(false_dest);
  if (true_dest == false_dest && true_index > header_index) {
    *out_merge_block = true_dest;
    return true;
  }

  const loom_op_t* true_terminator = true_dest->last_op;
  const loom_op_t* false_terminator = false_dest->last_op;
  if (true_terminator && loom_low_br_isa(true_terminator) &&
      loom_low_br_dest(true_terminator) == false_dest &&
      true_index > header_index && false_index > true_index) {
    *out_merge_block = false_dest;
    return true;
  }
  if (false_terminator && loom_low_br_isa(false_terminator) &&
      loom_low_br_dest(false_terminator) == true_dest &&
      false_index > header_index && true_index > false_index) {
    *out_merge_block = true_dest;
    return true;
  }
  if (!true_terminator || !false_terminator ||
      !loom_low_br_isa(true_terminator) || !loom_low_br_isa(false_terminator)) {
    return false;
  }

  const loom_block_t* merge_block = loom_low_br_dest(true_terminator);
  if (merge_block != loom_low_br_dest(false_terminator)) return false;
  const uint16_t merge_index = loom_block_region_index(merge_block);
  if (true_index <= header_index || false_index <= header_index ||
      merge_index <= true_index || merge_index <= false_index) {
    return false;
  }
  *out_merge_block = merge_block;
  return true;
}

static bool loom_spirv_low_try_loop_successor_pair(
    const loom_op_t* op, const loom_block_t* body_block,
    const loom_block_t* merge_block, loom_spirv_low_loop_shape_t* out_loop) {
  const loom_block_t* header_block = op->parent_block;
  const uint16_t header_index = loom_block_region_index(header_block);
  const uint16_t body_index = loom_block_region_index(body_block);
  const uint16_t merge_index = loom_block_region_index(merge_block);
  const loom_op_t* backedge_op = NULL;
  if (body_index <= header_index || merge_index <= body_index ||
      !loom_spirv_low_block_backedges_to(body_block, header_block,
                                         &backedge_op)) {
    return false;
  }
  *out_loop = (loom_spirv_low_loop_shape_t){
      .header_block = header_block,
      .body_block = body_block,
      .merge_block = merge_block,
      .backedge_op = backedge_op,
  };
  return true;
}

bool loom_spirv_low_loop_shape(const loom_op_t* op,
                               loom_spirv_low_loop_shape_t* out_loop) {
  *out_loop = (loom_spirv_low_loop_shape_t){0};
  const loom_block_t* true_dest = loom_low_cond_br_true_dest(op);
  const loom_block_t* false_dest = loom_low_cond_br_false_dest(op);
  if (loom_spirv_low_try_loop_successor_pair(op, true_dest, false_dest,
                                             out_loop)) {
    return true;
  }
  return loom_spirv_low_try_loop_successor_pair(op, false_dest, true_dest,
                                                out_loop);
}

bool loom_spirv_low_br_is_loop_backedge(const loom_op_t* op) {
  const loom_block_t* header_block = loom_low_br_dest(op);
  const loom_op_t* header_terminator = header_block->last_op;
  if (!header_terminator || !loom_low_cond_br_isa(header_terminator)) {
    return false;
  }
  loom_spirv_low_loop_shape_t loop = {0};
  return loom_spirv_low_loop_shape(header_terminator, &loop) &&
         loop.backedge_op == op;
}
