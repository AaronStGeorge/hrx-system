// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/structured_control_flow.h"

#include "loom/ops/low/ops.h"

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
