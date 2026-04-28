// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packetization.h"

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/schedule/run.h"
#include "loom/ops/low/ops.h"

static iree_status_t loom_low_packetization_liveness_order_from_schedule(
    loom_op_t* low_func_op, const loom_low_schedule_sidecar_t* schedule,
    iree_arena_allocator_t* arena, loom_liveness_order_t* out_order) {
  IREE_ASSERT_ARGUMENT(low_func_op);
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_order);
  *out_order = loom_liveness_order_empty();

  loom_region_t* body = loom_low_function_body(low_func_op);
  if (!body) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low function body is required");
  }
  if (schedule->function_op != low_func_op ||
      schedule->block_count != body->block_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low packetization schedule must describe the low function body");
  }
  if (schedule->scheduled_node_count != 0 &&
      schedule->scheduled_node_indices == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low packetization schedule has packets but no packet index table");
  }

  loom_liveness_block_order_t* block_orders = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, body->block_count, sizeof(*block_orders), (void**)&block_orders));
  iree_host_size_t total_scheduled_nodes = 0;
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_low_schedule_block_t* schedule_block =
        &schedule->blocks[block_index];
    if (schedule_block->block != block ||
        schedule_block->scheduled_node_count != block->op_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low packetization schedule block %u does not "
                              "match the function body",
                              block_index);
    }
    const loom_op_t** ops = NULL;
    if (schedule_block->scheduled_node_count != 0) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(arena, schedule_block->scheduled_node_count,
                                    sizeof(*ops), (void**)&ops));
    }
    for (uint32_t scheduled_ordinal = 0;
         scheduled_ordinal < schedule_block->scheduled_node_count;
         ++scheduled_ordinal) {
      const iree_host_size_t packet_index =
          (iree_host_size_t)schedule_block->scheduled_node_start +
          scheduled_ordinal;
      if (packet_index >= schedule->scheduled_node_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low packetization schedule block %u references packet %" PRIhsz
            " outside the schedule",
            block_index, packet_index);
      }
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      if (node_index >= schedule->node_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "low packetization schedule packet %" PRIhsz
                                " references node %" PRIu32,
                                packet_index, node_index);
      }
      const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
      if (node->block != block ||
          node->scheduled_ordinal != scheduled_ordinal) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low packetization schedule node does not match its block order");
      }
      ops[scheduled_ordinal] = node->op;
      ++total_scheduled_nodes;
    }
    block_orders[block_index] = (loom_liveness_block_order_t){
        .block = block,
        .ops = ops,
        .op_count = schedule_block->scheduled_node_count,
    };
  }
  if (total_scheduled_nodes != schedule->scheduled_node_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low packetization schedule has %" PRIhsz
        " scheduled packet(s) but body blocks cover %" PRIhsz,
        schedule->scheduled_node_count, total_scheduled_nodes);
  }
  *out_order = (loom_liveness_order_t){
      .blocks = block_orders,
      .block_count = body->block_count,
  };
  return iree_ok_status();
}

iree_status_t loom_low_packetize_function(
    loom_module_t* module, loom_op_t* low_func_op,
    const loom_low_packetization_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_packetization_t* out_packetization) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(low_func_op);
  IREE_ASSERT_ARGUMENT(options && options->descriptor_registry);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_packetization);
  if (options->allocation_budget_count > 0 && !options->allocation_budgets) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low allocation budgets are required when allocation_budget_count is "
        "non-zero");
  }
  if (options->allocation_fixed_value_count > 0 &&
      !options->allocation_fixed_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low allocation fixed values are required when "
                            "allocation_fixed_value_count is non-zero");
  }
  if (options->allocation_reserved_range_count > 0 &&
      !options->allocation_reserved_ranges) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low allocation reserved ranges are required when "
                            "allocation_reserved_range_count is non-zero");
  }
  if (!loom_low_function_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def or low.kernel.def");
  }

  *out_packetization = (loom_low_packetization_t){0};

  loom_low_schedule_options_t schedule_options = {
      .descriptor_registry = options->descriptor_registry,
      .memory_access_table = options->memory_access_table,
      .pressure_cliffs = options->schedule_pressure_cliffs,
      .emitter = options->emitter,
      .diagnostic_flags = options->schedule_diagnostic_flags,
      .strategy = options->schedule_strategy,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_function(module, low_func_op, &schedule_options, arena,
                                 &out_packetization->schedule));

  loom_liveness_order_t liveness_order = loom_liveness_order_empty();
  IREE_RETURN_IF_ERROR(loom_low_packetization_liveness_order_from_schedule(
      low_func_op, &out_packetization->schedule, arena, &liveness_order));
  loom_low_allocation_options_t allocation_options = {
      .liveness_order = liveness_order,
      .descriptor_registry = options->descriptor_registry,
      .budgets = options->allocation_budgets,
      .budget_count = options->allocation_budget_count,
      .fixed_values = options->allocation_fixed_values,
      .fixed_value_count = options->allocation_fixed_value_count,
      .reserved_ranges = options->allocation_reserved_ranges,
      .reserved_range_count = options->allocation_reserved_range_count,
      .emitter = options->emitter,
      .diagnostic_flags = options->allocation_diagnostic_flags,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_allocate_function(module, low_func_op, &allocation_options,
                                 arena, &out_packetization->allocation));

  return loom_low_packet_validate_sidecars(&out_packetization->schedule,
                                           &out_packetization->allocation);
}
