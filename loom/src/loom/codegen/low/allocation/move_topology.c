// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/move_topology.h"

#include "loom/ops/low/ops.h"

iree_host_size_t loom_low_allocation_move_topology_count_copy_ops(
    const loom_region_t* body) {
  IREE_ASSERT_ARGUMENT(body);
  iree_host_size_t copy_count = 0;
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_copy_isa(op)) {
        ++copy_count;
      }
    }
  }
  return copy_count;
}

const loom_op_t* loom_low_allocation_move_topology_value_defining_concat(
    const loom_module_t* module, loom_value_id_t value_id) {
  IREE_ASSERT_ARGUMENT(module);
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return NULL;
  }
  const loom_op_t* def_op = loom_value_def_op(value);
  return def_op && loom_low_concat_isa(def_op) ? def_op : NULL;
}

static iree_status_t
loom_low_allocation_move_topology_count_branch_payload_edge_copies(
    const loom_module_t* module, loom_value_id_t payload_value_id,
    iree_host_size_t* inout_copy_count) {
  iree_host_size_t payload_copy_count = 1;
  const loom_op_t* concat_op =
      loom_low_allocation_move_topology_value_defining_concat(module,
                                                              payload_value_id);
  if (concat_op) {
    payload_copy_count = loom_low_concat_sources(concat_op).count;
  }
  if (payload_copy_count > IREE_HOST_SIZE_MAX - *inout_copy_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low.br edge-copy count exceeds host size");
  }
  *inout_copy_count += payload_copy_count;
  return iree_ok_status();
}

iree_status_t loom_low_allocation_move_topology_count_edge_copy_groups(
    const loom_module_t* module, const loom_region_t* body,
    iree_host_size_t* out_group_count, iree_host_size_t* out_copy_count) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(body);
  IREE_ASSERT_ARGUMENT(out_group_count);
  IREE_ASSERT_ARGUMENT(out_copy_count);
  *out_group_count = 0;
  *out_copy_count = 0;
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_br_isa(op)) {
        continue;
      }
      loom_value_slice_t args = loom_low_br_args(op);
      if (args.count == 0) {
        continue;
      }
      if (*out_group_count == IREE_HOST_SIZE_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "low.br edge-copy count exceeds host size");
      }
      ++*out_group_count;
      for (uint16_t i = 0; i < args.count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_move_topology_count_branch_payload_edge_copies(
                module, args.values[i], out_copy_count));
      }
    }
  }
  return iree_ok_status();
}

bool loom_low_allocation_move_topology_concat_requires_packet_materialization_for_module(
    const loom_module_t* module, const loom_op_t* op) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(op);
  if (!loom_low_concat_isa(op)) {
    return true;
  }
  const loom_value_t* result =
      loom_module_value(module, loom_low_concat_result(op));
  const loom_use_t* use = NULL;
  loom_value_for_each_use(result, use) {
    if (!loom_low_br_isa(loom_use_user_op(*use))) {
      return true;
    }
  }
  return false;
}

bool loom_low_allocation_move_topology_concat_requires_packet_materialization(
    const loom_low_allocation_table_t* table, const loom_op_t* op) {
  IREE_ASSERT_ARGUMENT(table);
  return loom_low_allocation_move_topology_concat_requires_packet_materialization_for_module(
      table->module, op);
}

bool loom_low_allocation_move_topology_op_has_packet_moves(
    const loom_op_t* op) {
  IREE_ASSERT_ARGUMENT(op);
  return loom_low_copy_isa(op) || loom_low_slice_isa(op) ||
         loom_low_concat_isa(op);
}
