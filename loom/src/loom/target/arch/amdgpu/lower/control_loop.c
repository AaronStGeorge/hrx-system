// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/control_loop.h"

#include <string.h>

#include "iree/base/bitmap.h"
#include "loom/codegen/low/lower/control_carriers.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/control_mask.h"
#include "loom/target/arch/amdgpu/lower/control_plan.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

typedef struct loom_amdgpu_loop_plan_t loom_amdgpu_loop_plan_t;
typedef struct loom_amdgpu_loop_exit_plan_t loom_amdgpu_loop_exit_plan_t;

typedef struct loom_amdgpu_loop_exit_region_adapter_t {
  // Low-only block that retains one clean-arm continuation tuple.
  loom_low_lower_block_ref_t block;
} loom_amdgpu_loop_exit_region_adapter_t;

// One divergent selection whose second scheduled arm may retire lanes from an
// enclosing natural loop. The clean arm is always scheduled first, independent
// of its source successor position. Its values and surviving mask remain live
// while the exit-bearing arm executes, then both arms reconverge before the
// loop's shared retirement machinery observes the completed iteration state.
typedef struct loom_amdgpu_loop_exit_region_plan_t {
  // Shared natural-loop retirement plan owning the nested exit.
  loom_amdgpu_loop_plan_t* loop;
  // Direct loop exit at the entry of the second scheduled arm.
  loom_amdgpu_loop_exit_plan_t* exit;
  // Source selection whose arms are scheduled by this plan.
  const loom_op_t* source_op;
  // Source successor index of the first, nonretiring arm.
  uint8_t clean_successor_index;
  // Shared source continuation after both arms complete.
  uint16_t continuation_index;
  // Planned Low identity of the shared continuation.
  loom_low_lower_block_ref_t continuation;
  // Number of values carried to the shared continuation.
  uint16_t merge_arg_count;
  // Number of authored and realization values requiring lane-masked merge.
  // Supplemental control carriers follow these values and are immutable
  // across both arms.
  uint16_t source_merge_arg_count;
  // Low-only adapters retaining clean-arm continuation tuples.
  loom_amdgpu_loop_exit_region_adapter_t* clean_adapters;
  // Number of populated clean-arm adapters.
  uint16_t clean_adapter_count;
  // Shared clean-arm tuple plus its surviving EXEC mask.
  loom_low_lower_block_ref_t arm_dispatch_block;
  // Source path used when the clean arm has no active lanes.
  loom_low_lower_block_ref_t no_clean_block;
  // Original entry of the exit-bearing arm.
  loom_low_lower_block_ref_t arm_body_block;
  // Direct continuation when the exit-bearing arm has no active lanes.
  loom_low_lower_block_ref_t clean_only_block;
  // Shared merge for every ordinary exit from the second arm.
  loom_low_lower_block_ref_t merge_block;
  // Transfers captured return values to the natural-loop capture block.
  loom_low_lower_block_ref_t capture_forward_block;
  // Continues directly when no lane returned from the second arm.
  loom_low_lower_block_ref_t direct_forward_block;
  // Transfers an all-retired second arm with only clean-arm survivors.
  loom_low_lower_block_ref_t no_repeat_block;
  // Per-argument logical constants at the arm merge.
  const loom_amdgpu_mask_merge_constants_t* merge_constants;
  // Entry EXEC saved while the source selection executes.
  loom_value_id_t saved_exec;
  // Original source predicate used to merge the authored result tuple.
  loom_value_id_t source_condition;
} loom_amdgpu_loop_exit_region_plan_t;

typedef struct loom_amdgpu_loop_exit_adapter_t {
  // Low-only block that captures one source exit edge's result tuple.
  loom_low_lower_block_ref_t block;
} loom_amdgpu_loop_exit_adapter_t;

typedef enum loom_amdgpu_loop_validation_e {
  LOOM_AMDGPU_LOOP_VALIDATION_UNCHECKED = 0,
  LOOM_AMDGPU_LOOP_VALIDATION_SUPPORTED,
  LOOM_AMDGPU_LOOP_VALIDATION_REJECTED,
} loom_amdgpu_loop_validation_t;

struct loom_amdgpu_loop_exit_plan_t {
  // Shared saved-mask lifetime for the natural loop.
  loom_amdgpu_loop_plan_t* loop;
  // Source conditional terminator owning the exit successors.
  const loom_op_t* source_op;
  // Bit i is set when successor i exits the loop.
  uint8_t exit_successor_mask;
  // True when the exit selector is lane-varying.
  bool is_divergent;
  // Dense selector among divergent exits that may resume the loop.
  uint16_t selector;
  // Low-only block restoring the active mask when no lane takes this exit.
  loom_low_lower_block_ref_t no_exit_block;
  // Original in-loop destination reached when lanes remain active.
  loom_low_lower_block_ref_t repeat_dest;
  // Source-local repeat destination before nested reconvergence.
  loom_low_lower_block_ref_t source_repeat_dest;
  // Selector dispatch block for the remaining active lanes.
  loom_low_lower_block_ref_t dispatch_block;
  // Argument-bearing continuation reached after selector dispatch.
  loom_low_lower_block_ref_t resume_block;
  // Local result capture used before a nested arm reconverges.
  loom_low_lower_block_ref_t local_capture_block;
  // Nested branch scheduler owning this exit, or NULL for a direct exit.
  loom_amdgpu_loop_exit_region_plan_t* region;
  // First capture-block argument carrying this exit's resume tuple.
  uint16_t resume_arg_offset;
  // Number of resume arguments consumed by this exit's dispatch block.
  uint16_t resume_arg_count;
  // Exit-edge adapters carrying result values into the shared capture block.
  loom_amdgpu_loop_exit_adapter_t* adapters;
  // Number of populated exit-edge adapters.
  uint16_t adapter_count;
};

struct loom_amdgpu_loop_plan_t {
  // Low-only block that saves parent EXEC before the loop header.
  loom_low_lower_block_ref_t entry_block;
  // Original low loop header destination reached after entry_block.
  loom_low_lower_block_ref_t header_dest;
  // Shared block that accumulates return payloads under each exiting mask.
  loom_low_lower_block_ref_t capture_block;
  // Low-only block that restores parent EXEC after all lanes leave the loop.
  loom_low_lower_block_ref_t restore_block;
  // Shared source continuation reached after restore_block.
  loom_low_lower_block_ref_t restore_dest;
  // Shared semantic continuation reached after all loop-exit regions.
  uint16_t continuation_index;
  // Low value produced by entry_block, or INVALID before emission.
  loom_value_id_t saved_exec;
  // Native EXEC-mask carrier appended to the shared capture signature.
  loom_type_t exec_mask_type;
  // SGPR selector carrier appended when more than one resume site exists.
  loom_type_t selector_type;
  // True when exiting lanes must execute an exit region or retain values.
  bool capture_exits;
  // Number of authored continuation arguments preceding private capture state.
  uint16_t capture_value_count;
  // Number of divergent exits assigned selector dispatch blocks.
  uint16_t divergent_exit_count;
  // First shared-capture argument reserved for nested resume tuples.
  uint16_t resume_arg_offset;
  // Total nested resume arguments appended to the shared capture tuple.
  uint16_t resume_arg_count;
  // Number of direct exit edges collected during branch preparation.
  uint32_t exit_edge_count;
  // Number of populated records in exit_plans.
  uint32_t exit_plan_count;
  // Boundary validation state for shared retirement.
  loom_amdgpu_loop_validation_t validation;
  // Direct exit records, allocated inline to the loop's direct exit count.
  loom_amdgpu_loop_exit_plan_t exit_plans[];
};

static int loom_amdgpu_loop_plan_table_key;
static int loom_amdgpu_loop_exit_region_table_key;

typedef struct loom_amdgpu_zero_placeholder_t {
  loom_type_t type;
  loom_value_id_t value;
} loom_amdgpu_zero_placeholder_t;

typedef struct loom_amdgpu_loop_exit_region_match_t {
  uint16_t loop_index;
  uint16_t continuation_index;
  uint16_t exit_entry_index;
  uint8_t clean_successor_index;
  const loom_cfg_region_t* clean_region;
  loom_cfg_edge_index_span_t exit_local_edges;
} loom_amdgpu_loop_exit_region_match_t;

static bool loom_amdgpu_try_find_loop_exit(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           uint16_t* out_loop_index,
                                           uint8_t* out_exit_successor_mask) {
  *out_loop_index = LOOM_CFG_LOOP_NEST_NONE;
  *out_exit_successor_mask = 0;
  const loom_cfg_loop_nest_t* loops =
      &loom_low_lower_context_cfg(context)->loops;
  const loom_cfg_graph_t* graph = loops->graph;
  const uint16_t block_index =
      (uint16_t)loom_cfg_graph_block_index(graph, source_op->parent_block);
  const uint16_t loop_index = loom_cfg_loop_nest_innermost(loops, block_index);
  if (loop_index == LOOM_CFG_LOOP_NEST_NONE) {
    return false;
  }
  const loom_cfg_edge_index_span_t outgoing =
      loom_cfg_graph_successor_edges(graph, block_index);
  const loom_cfg_edge_info_t* true_edge = &graph->edges[outgoing.values[0]];
  const loom_cfg_edge_info_t* false_edge = &graph->edges[outgoing.values[1]];
  const bool true_inside = loom_cfg_loop_nest_contains(
      loops, loop_index, true_edge->target_block_index);
  const bool false_inside = loom_cfg_loop_nest_contains(
      loops, loop_index, false_edge->target_block_index);
  if (true_inside && false_inside) {
    return false;
  }
  *out_loop_index = loop_index;
  *out_exit_successor_mask = (true_inside ? 0 : 1) | (false_inside ? 0 : 2);
  return true;
}

static iree_status_t loom_amdgpu_allocate_loop_plan(
    loom_low_lower_context_t* context, uint32_t exit_capacity,
    loom_amdgpu_loop_plan_t** out_plan) {
  *out_plan = NULL;
  const iree_host_size_t plan_length =
      sizeof(**out_plan) + exit_capacity * sizeof(loom_amdgpu_loop_exit_plan_t);
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(context, plan_length,
                                                         (void**)out_plan));
  **out_plan = (loom_amdgpu_loop_plan_t){
      .saved_exec = LOOM_VALUE_ID_INVALID,
      .continuation_index = LOOM_CFG_LOOP_CONTINUATION_NONE,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_get_or_allocate_loop_plan(
    loom_low_lower_context_t* context, const loom_cfg_loop_nest_t* loops,
    uint16_t loop_index, loom_amdgpu_loop_plan_t** out_plan) {
  *out_plan = NULL;
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  if (loop->exits.count == 1) {
    return loom_amdgpu_allocate_loop_plan(context, loop->direct_exit_count,
                                          out_plan);
  }

  loom_amdgpu_loop_plan_t** plans = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_loop_plan_table_key,
      loops->loop_count * sizeof(*plans), (void**)&plans));
  if (plans[loop_index] == NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_allocate_loop_plan(
        context, loop->direct_exit_count, &plans[loop_index]));
  }
  *out_plan = plans[loop_index];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_get_loop_exit_region_table(
    loom_low_lower_context_t* context,
    loom_amdgpu_loop_exit_region_plan_t*** out_table) {
  const iree_host_size_t block_count =
      loom_low_lower_context_cfg(context)->graph.block_count;
  return loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_loop_exit_region_table_key,
      block_count * sizeof(**out_table), (void**)out_table);
}

static bool loom_amdgpu_try_loop_exit_region_orientation(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t loop_index, uint8_t clean_successor_index,
    loom_amdgpu_loop_exit_region_match_t* out_match) {
  const loom_value_fact_cfg_region_t* facts =
      loom_low_lower_context_cfg(context);
  const loom_cfg_loop_nest_t* loops = &facts->loops;
  const loom_cfg_graph_t* graph = &facts->graph;
  const loom_block_t* clean_entry =
      loom_op_const_successors(source_op)[clean_successor_index];
  const loom_block_t* exit_entry =
      loom_op_const_successors(source_op)[clean_successor_index ^ 1u];
  const uint16_t guard_index = source_op->parent_block->region_index;
  const uint16_t clean_entry_index = clean_entry->region_index;
  const uint16_t exit_entry_index = exit_entry->region_index;
  if (facts->dominance.entry_predecessors[clean_entry_index] != guard_index ||
      facts->dominance.entry_predecessors[exit_entry_index] != guard_index) {
    return false;
  }

  const loom_cfg_region_t* clean_region =
      &facts->regions.blocks[clean_entry_index];
  if (clean_region->continuation_index == LOOM_CFG_REGION_CONTINUATION_NONE ||
      !clean_region->exit_count ||
      clean_region->exit_count != clean_region->exit_edges.count ||
      !loom_cfg_loop_nest_contains(loops, loop_index,
                                   clean_region->continuation_index)) {
    return false;
  }
  const uint16_t continuation_index = clean_region->continuation_index;
  if (continuation_index == guard_index) {
    return false;
  }

  const loom_op_t* exit_terminator = loom_block_const_last_op(exit_entry);
  if (!loom_cfg_cond_br_isa(exit_terminator)) {
    return false;
  }
  uint16_t exit_loop_index = LOOM_CFG_LOOP_NEST_NONE;
  uint8_t exit_successor_mask = 0;
  if (!loom_amdgpu_try_find_loop_exit(context, exit_terminator,
                                      &exit_loop_index, &exit_successor_mask) ||
      exit_loop_index != loop_index ||
      (exit_successor_mask != 1 && exit_successor_mask != 2)) {
    return false;
  }

  const loom_cfg_region_t* exit_region =
      &facts->regions.blocks[exit_entry_index];
  const loom_cfg_edge_index_span_t exit_local_edges =
      loom_cfg_regions_edges_to(graph, &facts->dominance, &facts->regions,
                                exit_entry_index, continuation_index);
  if (!exit_local_edges.count ||
      exit_region->exit_count != exit_local_edges.count + 1) {
    return false;
  }

  *out_match = (loom_amdgpu_loop_exit_region_match_t){
      .loop_index = loop_index,
      .continuation_index = continuation_index,
      .exit_entry_index = exit_entry_index,
      .clean_successor_index = clean_successor_index,
      .clean_region = clean_region,
      .exit_local_edges = exit_local_edges,
  };
  return true;
}

static iree_status_t loom_amdgpu_try_prepare_loop_exit_region(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_prepared) {
  *out_prepared = false;
  const loom_value_fact_cfg_region_t* facts =
      loom_low_lower_context_cfg(context);
  const loom_cfg_loop_nest_t* loops = &facts->loops;
  const loom_cfg_graph_t* graph = &facts->graph;
  const uint16_t guard_index = source_op->parent_block->region_index;
  const uint16_t loop_index = loom_cfg_loop_nest_innermost(loops, guard_index);
  if (loop_index == LOOM_CFG_LOOP_NEST_NONE) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < 2; ++i) {
    if (!loom_cfg_loop_nest_contains(
            loops, loop_index,
            loom_op_const_successors(source_op)[i]->region_index)) {
      return iree_ok_status();
    }
  }

  loom_amdgpu_loop_exit_region_match_t match = {0};
  if (!loom_amdgpu_try_loop_exit_region_orientation(
          context, source_op, loop_index, /*clean_successor_index=*/0,
          &match) &&
      !loom_amdgpu_try_loop_exit_region_orientation(
          context, source_op, loop_index, /*clean_successor_index=*/1,
          &match)) {
    return iree_ok_status();
  }

  const loom_low_lower_block_ref_t continuation =
      loom_low_lower_control_source_block(
          graph->blocks[match.continuation_index].block);
  const uint16_t merge_arg_count =
      loom_low_lower_control_argument_count(context, continuation);
  const uint16_t source_merge_arg_count =
      loom_low_lower_control_carrier_argument_offset(context, continuation);
  const uint32_t previous_error_count =
      loom_low_lower_context_error_count(context);
  const loom_amdgpu_mask_merge_constants_t* merge_constants = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_prepare_masked_merge(
      context, source_op, graph->blocks[match.continuation_index].block,
      continuation, (loom_value_slice_t){0}, (loom_value_slice_t){0},
      &merge_constants));
  if (loom_low_lower_context_error_count(context) != previous_error_count) {
    return iree_ok_status();
  }

  loom_amdgpu_loop_plan_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_or_allocate_loop_plan(
      context, loops, match.loop_index, &loop));
  loom_amdgpu_loop_exit_region_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_amdgpu_loop_exit_region_plan_t){
      .loop = loop,
      .source_op = source_op,
      .clean_successor_index = match.clean_successor_index,
      .continuation_index = match.continuation_index,
      .continuation = continuation,
      .merge_arg_count = merge_arg_count,
      .source_merge_arg_count = source_merge_arg_count,
      .merge_constants = merge_constants,
      .saved_exec = LOOM_VALUE_ID_INVALID,
      .source_condition = LOOM_VALUE_ID_INVALID,
  };

  loom_type_t exec_mask_type = loom_low_lower_value_binding_type(
      context, loom_cfg_cond_br_condition(source_op));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block_with_appended_arguments(
      context, /*before=*/0, continuation, &exec_mask_type, 1,
      &plan->arm_dispatch_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, /*signature=*/0, &plan->no_clean_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, /*signature=*/0, &plan->clean_only_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, continuation, &plan->merge_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, /*signature=*/0, &plan->capture_forward_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, /*signature=*/0, &plan->direct_forward_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, /*signature=*/0, &plan->no_repeat_block));

  IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
      context, source_op, match.clean_successor_index ^ 1u,
      plan->no_clean_block, &plan->arm_body_block));

  const iree_host_size_t clean_exit_count =
      match.clean_region->exit_edges.count;
  IREE_ASSERT_LE(clean_exit_count, UINT16_MAX);
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, clean_exit_count * sizeof(*plan->clean_adapters),
      (void**)&plan->clean_adapters));
  for (iree_host_size_t i = 0; i < clean_exit_count; ++i) {
    const loom_cfg_edge_info_t* edge =
        &graph->edges[match.clean_region->exit_edges.values[i]];
    loom_low_lower_block_ref_t adapter = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
        context, /*before=*/0, continuation, &adapter));
    loom_low_lower_block_ref_t previous_dest = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
        context, edge->terminator, (uint8_t)edge->successor_index, adapter,
        &previous_dest));
    if (previous_dest != continuation) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
    plan->clean_adapters[plan->clean_adapter_count++].block = adapter;
  }

  for (iree_host_size_t i = 0; i < match.exit_local_edges.count; ++i) {
    const loom_cfg_edge_info_t* edge =
        &graph->edges[match.exit_local_edges.values[i]];
    loom_low_lower_block_ref_t previous_dest = 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
        context, edge->terminator, (uint8_t)edge->successor_index,
        plan->merge_block, &previous_dest));
    if (previous_dest != continuation) {
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_exit_edge_interposition"));
    }
  }

  loom_amdgpu_loop_exit_region_plan_t** table = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_loop_exit_region_table(context, &table));
  IREE_ASSERT(table[match.exit_entry_index] == NULL);
  table[match.exit_entry_index] = plan;
  loom_low_lower_set_branch_plan(
      context, source_op,
      loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_LOOP_EXIT_REGION, plan));
  *out_prepared = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_validate_divergent_loop(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_fact_cfg_region_t* facts, uint16_t loop_index,
    loom_amdgpu_loop_plan_t* plan) {
  const loom_cfg_loop_nest_t* loops = &facts->loops;
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  if (loop->entries.count != 1) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("divergent_loop_single_entry"));
  }
  uint32_t continuation = loop->continuation_index;
  if (continuation == LOOM_CFG_LOOP_CONTINUATION_NONE &&
      facts->control_structure.postdominance.available) {
    continuation =
        facts->control_structure.postdominance.nodes[loop->header_index]
            .immediate_postdominator;
  }
  if (continuation >= facts->graph.block_count ||
      loom_cfg_loop_nest_contains(loops, loop_index, (uint16_t)continuation)) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("divergent_loop_single_continuation"));
  }
  if (loop->direct_exit_count != loop->exits.count) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("divergent_loop_direct_exits"));
  }
  plan->continuation_index = (uint16_t)continuation;
  return iree_ok_status();
}

static bool loom_amdgpu_loop_guard_is_active_uniform(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_fact_cfg_region_t* facts, uint16_t loop_index) {
  const loom_cfg_natural_loop_t* loop = &facts->loops.loops[loop_index];
  if (source_op->parent_block->region_index != loop->header_index) {
    return false;
  }
  const loom_value_fact_induction_t* induction = &facts->inductions[loop_index];
  if (induction->value == LOOM_VALUE_ID_INVALID ||
      induction->step.value == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  const loom_value_fact_table_t* table =
      loom_low_lower_context_fact_table(context);
  return loom_value_facts_is_subgroup_uniform(
             loom_value_fact_recurrence_operand_facts(
                 table, induction->initial_value)) &&
         loom_value_facts_is_subgroup_uniform(
             loom_value_fact_recurrence_operand_facts(table,
                                                      induction->bound)) &&
         loom_value_facts_is_subgroup_uniform(
             loom_value_fact_recurrence_operand_facts(table, induction->step));
}

static iree_status_t loom_amdgpu_validate_loop_exit_regions(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_fact_cfg_region_t* facts, loom_amdgpu_loop_plan_t* plan) {
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    const loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    loom_cfg_edge_index_span_t outgoing = loom_cfg_graph_successor_edges(
        &facts->graph, exit->source_op->parent_block->region_index);
    for (uint8_t successor_index = 0; successor_index < 2; ++successor_index) {
      if ((exit->exit_successor_mask & (1u << successor_index)) == 0) {
        continue;
      }
      const uint16_t target =
          facts->graph.edges[outgoing.values[successor_index]]
              .target_block_index;
      if (target == plan->continuation_index) {
        continue;
      }
      if (facts->regions.blocks[target].continuation_index !=
          plan->continuation_index) {
        plan->validation = LOOM_AMDGPU_LOOP_VALIDATION_REJECTED;
        return loom_low_lower_emit_branch_constraint(
            context, source_op, IREE_SV("divergent_loop_single_continuation"));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_plan_loop_exit_adapter(
    loom_low_lower_context_t* context,
    const loom_value_fact_cfg_region_t* facts, loom_cfg_edge_index_t edge_index,
    loom_low_lower_block_ref_t continuation,
    loom_amdgpu_loop_exit_plan_t* exit) {
  const loom_cfg_edge_info_t* edge = &facts->graph.edges[edge_index];
  loom_low_lower_block_ref_t adapter = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, continuation, &adapter));
  loom_low_lower_block_ref_t previous_dest = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
      context, edge->terminator, (uint8_t)edge->successor_index, adapter,
      &previous_dest));
  if (previous_dest != continuation) {
    return loom_low_lower_emit_branch_constraint(
        context, exit->source_op,
        IREE_SV("divergent_loop_single_continuation"));
  }
  exit->adapters[exit->adapter_count++].block = adapter;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_finalize_loop_plan(
    loom_low_lower_context_t* context,
    const loom_value_fact_cfg_region_t* facts, uint16_t loop_index,
    loom_amdgpu_loop_plan_t* plan, iree_arena_allocator_t* analysis_arena) {
  const loom_cfg_loop_nest_t* loops = &facts->loops;
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  const loom_cfg_graph_t* graph = &facts->graph;
  IREE_RETURN_IF_ERROR(loom_amdgpu_validate_loop_exit_regions(
      context, plan->exit_plans[plan->exit_plan_count - 1].source_op, facts,
      plan));
  if (plan->validation == LOOM_AMDGPU_LOOP_VALIDATION_REJECTED) {
    return iree_ok_status();
  }
  const loom_cfg_edge_info_t* entry_edge =
      &graph->edges[loop->entries.unique_index];
  const loom_block_t* header_block = graph->blocks[loop->header_index].block;
  const loom_low_lower_block_ref_t low_continuation =
      loom_low_lower_control_source_block(
          graph->blocks[plan->continuation_index].block);

  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    if (plan->exit_plans[i].is_divergent) {
      ++plan->divergent_exit_count;
    }
  }
  IREE_ASSERT_GT(plan->divergent_exit_count, 0);
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    const loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (!exit->is_divergent) {
      continue;
    }
    plan->exec_mask_type = loom_low_lower_value_binding_type(
        context, loom_cfg_cond_br_condition(exit->source_op));
    break;
  }
  IREE_ASSERT(
      loom_amdgpu_low_type_is_native_i1_mask(context, plan->exec_mask_type));
  plan->capture_value_count =
      loom_low_lower_control_argument_count(context, low_continuation);
  plan->capture_exits = plan->capture_value_count != 0;
  for (uint32_t i = 0; i < plan->exit_plan_count && !plan->capture_exits; ++i) {
    const loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    const loom_cfg_edge_index_span_t outgoing = loom_cfg_graph_successor_edges(
        graph, exit->source_op->parent_block->region_index);
    for (uint8_t successor_index = 0; successor_index < 2; ++successor_index) {
      if ((exit->exit_successor_mask & (1u << successor_index)) != 0 &&
          graph->edges[outgoing.values[successor_index]].target_block_index !=
              plan->continuation_index) {
        plan->capture_exits = true;
        break;
      }
    }
  }

  const loom_low_lower_block_ref_t low_header =
      loom_low_lower_control_source_block(header_block);
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(context, 0, low_header,
                                                        &plan->entry_block));
  IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
      context, entry_edge->terminator, (uint8_t)entry_edge->successor_index,
      plan->entry_block, &plan->header_dest));

  if (!plan->capture_exits) {
    IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
        context, /*before=*/0, /*signature=*/0, &plan->restore_block));
    for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
      const loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
      for (uint8_t successor_index = 0; successor_index < 2;
           ++successor_index) {
        if ((exit->exit_successor_mask & (1u << successor_index)) == 0) {
          continue;
        }
        loom_low_lower_block_ref_t previous_dest = 0;
        IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_successor(
            context, exit->source_op, successor_index, plan->restore_block,
            &previous_dest));
        if (plan->restore_dest == 0) {
          plan->restore_dest = previous_dest;
        } else {
          IREE_ASSERT_EQ(plan->restore_dest, previous_dest);
        }
      }
    }
    IREE_ASSERT_NE(plan->restore_dest, 0);
    return iree_ok_status();
  }

  IREE_ASSERT_LE(plan->exit_plan_count, UINT16_MAX);
  uint16_t* capture_sources = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      analysis_arena, plan->exit_plan_count, sizeof(*capture_sources),
      (void**)&capture_sources));
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    capture_sources[i] =
        plan->exit_plans[i].source_op->parent_block->region_index;
  }
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (!exit->is_divergent) {
      continue;
    }
    const uint8_t repeat_successor = exit->exit_successor_mask == 1 ? 1 : 0;
    const loom_block_t* repeat_block =
        exit->region != NULL
            ? graph->blocks[exit->region->continuation_index].block
            : loom_op_const_successors(exit->source_op)[repeat_successor];
    IREE_RETURN_IF_ERROR(loom_low_lower_control_retain_resume_closure(
        context, repeat_block, capture_sources,
        (uint16_t)plan->exit_plan_count));
  }
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    loom_amdgpu_loop_exit_region_plan_t* region = plan->exit_plans[i].region;
    if (region != NULL) {
      region->merge_arg_count =
          loom_low_lower_control_argument_count(context, region->continuation);
    }
  }
  plan->capture_value_count =
      loom_low_lower_control_argument_count(context, low_continuation);

  uint32_t resume_arg_count = 0;
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    const loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (!exit->is_divergent) {
      continue;
    }
    const uint8_t repeat_successor = exit->exit_successor_mask == 1 ? 1 : 0;
    const loom_block_t* repeat_block =
        exit->region != NULL
            ? graph->blocks[exit->region->continuation_index].block
            : loom_op_const_successors(exit->source_op)[repeat_successor];
    resume_arg_count +=
        exit->region != NULL
            ? exit->region->merge_arg_count
            : loom_low_lower_control_source_carriers(context, repeat_block)
                  .count;
  }
  if (resume_arg_count > UINT16_MAX) {
    return loom_low_lower_emit_branch_constraint(
        context, plan->exit_plans[0].source_op,
        IREE_SV("divergent_loop_resume_tuple_u16"));
  }
  plan->resume_arg_count = (uint16_t)resume_arg_count;

  uint16_t private_type_count = 1;
  if (plan->divergent_exit_count > 1) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_type(context, &plan->selector_type));
    ++private_type_count;
  }
  if ((uint32_t)private_type_count + plan->resume_arg_count > UINT16_MAX) {
    return loom_low_lower_emit_branch_constraint(
        context, plan->exit_plans[0].source_op,
        IREE_SV("divergent_loop_resume_tuple_u16"));
  }
  private_type_count += plan->resume_arg_count;
  loom_type_t* private_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      analysis_arena, private_type_count, sizeof(*private_types),
      (void**)&private_types));
  uint16_t private_index = 0;
  private_types[private_index++] = plan->exec_mask_type;
  if (plan->divergent_exit_count > 1) {
    private_types[private_index++] = plan->selector_type;
  }
  const uint16_t resume_private_offset = private_index;
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (!exit->is_divergent) {
      continue;
    }
    const uint8_t repeat_successor = exit->exit_successor_mask == 1 ? 1 : 0;
    const loom_block_t* repeat_block =
        exit->region != NULL
            ? graph->blocks[exit->region->continuation_index].block
            : loom_op_const_successors(exit->source_op)[repeat_successor];
    const loom_low_lower_block_ref_t resume_signature =
        loom_low_lower_control_source_block(repeat_block);
    exit->resume_arg_count =
        exit->region != NULL
            ? exit->region->merge_arg_count
            : loom_low_lower_control_source_carriers(context, repeat_block)
                  .count;
    if (exit->region != NULL) {
      for (uint16_t j = 0; j < exit->resume_arg_count; ++j) {
        private_types[private_index++] =
            loom_low_lower_control_argument_type(context, resume_signature, j);
      }
    } else {
      const loom_low_lower_control_carrier_slice_t carriers =
          loom_low_lower_control_source_carriers(context, repeat_block);
      for (uint16_t j = 0; j < carriers.count; ++j) {
        private_types[private_index++] =
            loom_low_lower_control_carrier_type(context, carriers.values[j]);
      }
    }
  }
  IREE_ASSERT_EQ(private_index, private_type_count);
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block_with_appended_arguments(
      context, /*before=*/0, low_continuation, private_types,
      private_type_count, &plan->capture_block));
  plan->resume_arg_offset =
      (uint16_t)(plan->capture_value_count + resume_private_offset);
  private_index = resume_private_offset;
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (!exit->is_divergent) {
      continue;
    }
    exit->resume_arg_offset =
        (uint16_t)(plan->capture_value_count + private_index);
    private_index += exit->resume_arg_count;
  }
  IREE_ASSERT_EQ(private_index, private_type_count);
  IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
      context, /*before=*/0, /*signature=*/0, &plan->restore_block));
  plan->restore_dest = low_continuation;

  iree_bitmap_t visited_targets = {.bit_count = graph->block_count};
  const iree_host_size_t visited_word_count =
      iree_bitmap_calculate_words(visited_targets.bit_count);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      analysis_arena, visited_word_count, sizeof(*visited_targets.words),
      (void**)&visited_targets.words));
  iree_bitmap_reset_all(visited_targets);
  uint16_t next_selector = 0;
  for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
    loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (exit->region != NULL) {
      const loom_type_t local_types[] = {
          plan->exec_mask_type,
          plan->exec_mask_type,
      };
      IREE_RETURN_IF_ERROR(
          loom_low_lower_control_add_block_with_appended_arguments(
              context, /*before=*/0, low_continuation, local_types,
              IREE_ARRAYSIZE(local_types), &exit->local_capture_block));
    }
    loom_cfg_edge_index_span_t outgoing = loom_cfg_graph_successor_edges(
        graph, exit->source_op->parent_block->region_index);
    iree_host_size_t adapter_capacity = 0;
    for (uint8_t successor_index = 0; successor_index < 2; ++successor_index) {
      if ((exit->exit_successor_mask & (1u << successor_index)) == 0) {
        continue;
      }
      const loom_cfg_edge_index_t edge_index = outgoing.values[successor_index];
      const uint16_t target = graph->edges[edge_index].target_block_index;
      adapter_capacity += target == plan->continuation_index
                              ? 1
                              : facts->regions.blocks[target].exit_edges.count;
    }
    IREE_ASSERT_LE(adapter_capacity, UINT16_MAX);
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, adapter_capacity * sizeof(*exit->adapters),
        (void**)&exit->adapters));

    for (uint8_t successor_index = 0; successor_index < 2; ++successor_index) {
      if ((exit->exit_successor_mask & (1u << successor_index)) == 0) {
        continue;
      }
      const loom_cfg_edge_index_t edge_index = outgoing.values[successor_index];
      const uint16_t target = graph->edges[edge_index].target_block_index;
      if (target == plan->continuation_index) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_plan_loop_exit_adapter(
            context, facts, edge_index, low_continuation, exit));
        continue;
      }
      if (iree_bitmap_test(visited_targets, target)) {
        return loom_low_lower_emit_branch_constraint(
            context, exit->source_op,
            IREE_SV("divergent_loop_exit_region_ownership"));
      }
      iree_bitmap_set(visited_targets, target);
      loom_cfg_edge_index_span_t retirement_edges =
          facts->regions.blocks[target].exit_edges;
      for (iree_host_size_t j = 0; j < retirement_edges.count; ++j) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_plan_loop_exit_adapter(
            context, facts, retirement_edges.values[j], low_continuation,
            exit));
      }
    }
    IREE_ASSERT_EQ(exit->adapter_count, adapter_capacity);

    if (!exit->is_divergent) {
      continue;
    }
    if (exit->exit_successor_mask == 3) {
      return loom_low_lower_emit_branch_constraint(
          context, exit->source_op, IREE_SV("divergent_loop_repeat_successor"));
    }
    exit->selector = next_selector++;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
        context, /*before=*/0, /*signature=*/0, &exit->dispatch_block));
    if (exit->resume_arg_count != 0) {
      IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
          context, /*before=*/0, /*signature=*/0, &exit->resume_block));
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
        context, /*before=*/0, /*signature=*/0, &exit->no_exit_block));
    const uint8_t repeat_successor = exit->exit_successor_mask == 1 ? 1 : 0;
    IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_argless_successor(
        context, exit->source_op, repeat_successor, exit->no_exit_block,
        &exit->source_repeat_dest));
    exit->repeat_dest =
        exit->region != NULL
            ? loom_low_lower_control_source_block(
                  graph->blocks[exit->region->continuation_index].block)
            : exit->source_repeat_dest;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_try_prepare_loop_exit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool is_divergent, iree_arena_allocator_t* analysis_arena,
    bool* out_handled, bool* out_active_uniform) {
  *out_handled = false;
  *out_active_uniform = false;
  const loom_value_fact_cfg_region_t* facts =
      loom_low_lower_context_cfg(context);
  const loom_cfg_loop_nest_t* loops = &facts->loops;
  uint16_t loop_index = LOOM_CFG_LOOP_NEST_NONE;
  uint8_t exit_successor_mask = 0;
  if (!loom_amdgpu_try_find_loop_exit(context, source_op, &loop_index,
                                      &exit_successor_mask)) {
    return iree_ok_status();
  }
  *out_handled = true;
  const loom_cfg_natural_loop_t* loop = &loops->loops[loop_index];
  if (is_divergent && loom_amdgpu_loop_guard_is_active_uniform(
                          context, source_op, facts, loop_index)) {
    is_divergent = false;
    *out_active_uniform = true;
  }
  if (loop->exits.count == 1 && !is_divergent) {
    return iree_ok_status();
  }

  loom_amdgpu_loop_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_get_or_allocate_loop_plan(context, loops, loop_index, &plan));
  if (is_divergent &&
      plan->validation == LOOM_AMDGPU_LOOP_VALIDATION_UNCHECKED) {
    const uint32_t previous_error_count =
        loom_low_lower_context_error_count(context);
    IREE_RETURN_IF_ERROR(loom_amdgpu_validate_divergent_loop(
        context, source_op, facts, loop_index, plan));
    plan->validation =
        loom_low_lower_context_error_count(context) == previous_error_count
            ? LOOM_AMDGPU_LOOP_VALIDATION_SUPPORTED
            : LOOM_AMDGPU_LOOP_VALIDATION_REJECTED;
  }
  if (plan->validation == LOOM_AMDGPU_LOOP_VALIDATION_REJECTED) {
    return iree_ok_status();
  }

  IREE_ASSERT_LT(plan->exit_plan_count, loop->direct_exit_count);
  loom_amdgpu_loop_exit_plan_t* exit_plan =
      &plan->exit_plans[plan->exit_plan_count++];
  *exit_plan = (loom_amdgpu_loop_exit_plan_t){
      .loop = plan,
      .source_op = source_op,
      .exit_successor_mask = exit_successor_mask,
      .is_divergent = is_divergent,
  };
  const iree_host_size_t block_count = facts->graph.block_count;
  loom_amdgpu_loop_exit_region_plan_t* const* region_table =
      (loom_amdgpu_loop_exit_region_plan_t* const*)
          loom_low_lower_lookup_target_state(
              context, &loom_amdgpu_loop_exit_region_table_key,
              block_count * sizeof(*region_table));
  loom_amdgpu_loop_exit_region_plan_t* region =
      region_table ? region_table[source_op->parent_block->region_index] : NULL;
  if (region != NULL) {
    IREE_ASSERT_EQ(region->loop, plan);
    if (!is_divergent) {
      plan->validation = LOOM_AMDGPU_LOOP_VALIDATION_REJECTED;
      return loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("divergent_loop_nested_exit_condition"));
    }
    IREE_ASSERT(region->exit == NULL);
    region->exit = exit_plan;
    exit_plan->region = region;
  }
  plan->exit_edge_count += exit_successor_mask == 3 ? 2 : 1;
  IREE_ASSERT_LE(plan->exit_edge_count, loop->direct_exit_count);
  if (is_divergent) {
    loom_low_lower_set_branch_plan(
        context, source_op,
        loom_low_lower_plan_make(LOOM_AMDGPU_BRANCH_PLAN_DIVERGENT_LOOP,
                                 exit_plan));
  }
  if (plan->exit_edge_count == loop->exits.count &&
      plan->validation == LOOM_AMDGPU_LOOP_VALIDATION_SUPPORTED) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_finalize_loop_plan(
        context, facts, loop_index, plan, analysis_arena));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_loop_emit_exec_restore_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_block_t* restore_block,
    loom_block_t* restore_dest) {
  IREE_ASSERT_EQ(restore_block->op_count, 0);
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, restore_block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, restore_dest, /*args=*/NULL, 0,
                               source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_capture_branch(
    loom_low_lower_context_t* context, const loom_amdgpu_loop_exit_plan_t* exit,
    const loom_op_t* source_op, const loom_value_id_t* payload_args,
    uint16_t payload_count, loom_value_id_t repeat_mask,
    const loom_value_id_t* resume_args, uint16_t resume_arg_count) {
  const loom_amdgpu_loop_plan_t* plan = exit->loop;
  loom_block_t* capture_block =
      loom_low_lower_control_block(context, plan->capture_block);
  IREE_ASSERT(capture_block != NULL);
  IREE_ASSERT_EQ(payload_count, plan->capture_value_count);

  loom_value_id_t* direct_resume_args = NULL;
  if (exit->is_divergent && exit->region == NULL &&
      exit->resume_arg_count != 0 && resume_args == NULL) {
    const loom_low_lower_control_carrier_slice_t carriers =
        loom_low_lower_control_block_carriers(context, exit->repeat_dest);
    IREE_ASSERT_EQ(carriers.count, exit->resume_arg_count);
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, carriers.count, sizeof(*direct_resume_args),
        (void**)&direct_resume_args));
    for (uint16_t i = 0; i < carriers.count; ++i) {
      direct_resume_args[i] =
          loom_low_lower_lookup_value(context, carriers.values[i]);
    }
    resume_args = direct_resume_args;
    resume_arg_count = carriers.count;
  }
  IREE_ASSERT_EQ(resume_arg_count, exit->resume_arg_count);

  const uint16_t capture_arg_count = capture_block->arg_count;
  loom_value_id_t* capture_args = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, capture_arg_count, sizeof(*capture_args),
      (void**)&capture_args));
  if (payload_count != 0) {
    memcpy(capture_args, payload_args, payload_count * sizeof(*capture_args));
  }
  capture_args[plan->capture_value_count] = repeat_mask;
  uint16_t private_index = plan->capture_value_count + 1;
  if (plan->divergent_exit_count > 1) {
    loom_value_id_t selector = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        exit->selector, plan->selector_type, &selector));
    capture_args[private_index++] = selector;
  }
  IREE_ASSERT_EQ(private_index, plan->resume_arg_offset);

  loom_amdgpu_zero_placeholder_t* placeholders = NULL;
  if (plan->resume_arg_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, plan->resume_arg_count, sizeof(*placeholders),
        (void**)&placeholders));
  }
  uint16_t placeholder_count = 0;
  loom_module_t* module = loom_low_lower_context_module(context);
  const uint16_t resume_limit =
      plan->resume_arg_offset + plan->resume_arg_count;
  for (uint16_t i = plan->resume_arg_offset; i < resume_limit; ++i) {
    if (i >= exit->resume_arg_offset &&
        i < exit->resume_arg_offset + resume_arg_count) {
      capture_args[i] = resume_args[i - exit->resume_arg_offset];
      continue;
    }
    const loom_type_t value_type =
        loom_module_value_type(module, loom_block_arg_id(capture_block, i));
    capture_args[i] = LOOM_VALUE_ID_INVALID;
    for (uint16_t j = 0; j < placeholder_count; ++j) {
      if (loom_type_equal(placeholders[j].type, value_type)) {
        capture_args[i] = placeholders[j].value;
        break;
      }
    }
    if (capture_args[i] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_zero_merge_value(
        context, source_op, value_type, &capture_args[i]));
    placeholders[placeholder_count++] = (loom_amdgpu_zero_placeholder_t){
        .type = value_type,
        .value = capture_args[i],
    };
  }

  loom_op_t* branch_op = NULL;
  uint16_t identity_start = exit->resume_arg_offset;
  uint16_t identity_count = exit->resume_arg_count;
  if (exit->region != NULL) {
    IREE_ASSERT_LE(exit->region->source_merge_arg_count,
                   exit->region->merge_arg_count);
    identity_start += exit->region->source_merge_arg_count;
    identity_count =
        exit->region->merge_arg_count - exit->region->source_merge_arg_count;
  }
  if (identity_count != 0) {
    return loom_low_br_identity_build(
        loom_low_lower_context_builder(context), identity_start, identity_count,
        capture_block, capture_args, capture_arg_count, source_op->location,
        &branch_op);
  }
  return loom_low_br_build(loom_low_lower_context_builder(context),
                           capture_block, capture_args, capture_arg_count,
                           source_op->location, &branch_op);
}

static iree_status_t loom_amdgpu_emit_loop_exit_region_clean_adapters(
    loom_low_lower_context_t* context,
    const loom_amdgpu_loop_exit_region_plan_t* plan,
    loom_type_t exec_mask_type) {
  loom_block_t* dispatch =
      loom_low_lower_control_block(context, plan->arm_dispatch_block);
  IREE_ASSERT(dispatch != NULL);
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0;
       i < plan->clean_adapter_count && iree_status_is_ok(status); ++i) {
    loom_block_t* adapter =
        loom_low_lower_control_block(context, plan->clean_adapters[i].block);
    IREE_ASSERT(adapter != NULL);
    IREE_ASSERT_EQ(adapter->op_count, 0);
    IREE_ASSERT_EQ(adapter->arg_count, plan->merge_arg_count);
    loom_builder_set_block(builder, adapter);

    loom_op_t* read_exec_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, plan->source_op,
        LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
        /*operands=*/NULL, 0, loom_named_attr_slice_empty(), &exec_mask_type, 1,
        &read_exec_op);
    loom_value_id_t* args = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_allocate_emission_array(
          context, dispatch->arg_count, sizeof(*args), (void**)&args);
    }
    if (iree_status_is_ok(status) && plan->merge_arg_count != 0) {
      memcpy(args, adapter->arg_ids, plan->merge_arg_count * sizeof(*args));
    }
    if (iree_status_is_ok(status)) {
      args[plan->merge_arg_count] = loom_op_const_results(read_exec_op)[0];
      loom_op_t* branch_op = NULL;
      status = loom_low_br_build(builder, dispatch, args, dispatch->arg_count,
                                 plan->source_op->location, &branch_op);
    }
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_exit_region_no_clean_block(
    loom_low_lower_context_t* context,
    const loom_amdgpu_loop_exit_region_plan_t* plan,
    loom_type_t exec_mask_type) {
  loom_block_t* block =
      loom_low_lower_control_block(context, plan->no_clean_block);
  loom_block_t* dispatch =
      loom_low_lower_control_block(context, plan->arm_dispatch_block);
  IREE_ASSERT(block != NULL);
  IREE_ASSERT(dispatch != NULL);
  IREE_ASSERT_EQ(block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &plan->saved_exec, 1, loom_named_attr_slice_empty(),
      /*result_types=*/NULL, 0, &restore_op);

  loom_value_id_t* args = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_emission_array(
        context, dispatch->arg_count, sizeof(*args), (void**)&args);
  }
  loom_amdgpu_zero_placeholder_t* placeholders = NULL;
  if (iree_status_is_ok(status) && plan->source_merge_arg_count != 0) {
    status = loom_low_lower_allocate_emission_array(
        context, plan->source_merge_arg_count, sizeof(*placeholders),
        (void**)&placeholders);
  }
  uint16_t placeholder_count = 0;
  loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0;
       i < plan->source_merge_arg_count && iree_status_is_ok(status); ++i) {
    const loom_type_t type =
        loom_module_value_type(module, loom_block_arg_id(dispatch, i));
    args[i] = LOOM_VALUE_ID_INVALID;
    for (uint16_t j = 0; j < placeholder_count; ++j) {
      if (loom_type_equal(placeholders[j].type, type)) {
        args[i] = placeholders[j].value;
        break;
      }
    }
    if (args[i] == LOOM_VALUE_ID_INVALID) {
      status = loom_amdgpu_emit_zero_merge_value(context, plan->source_op, type,
                                                 &args[i]);
      if (iree_status_is_ok(status)) {
        placeholders[placeholder_count++] = (loom_amdgpu_zero_placeholder_t){
            .type = type,
            .value = args[i],
        };
      }
    }
  }
  const loom_low_lower_control_carrier_slice_t carriers =
      loom_low_lower_control_block_carriers(context, plan->continuation);
  IREE_ASSERT_EQ(
      (uint16_t)(plan->merge_arg_count - plan->source_merge_arg_count),
      carriers.count);
  for (uint16_t i = 0; i < carriers.count && iree_status_is_ok(status); ++i) {
    args[plan->source_merge_arg_count + i] =
        loom_low_lower_lookup_value(context, carriers.values[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_emit_zero_native_i1_mask(
        context, plan->source_op, exec_mask_type, &args[plan->merge_arg_count]);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, dispatch, args, dispatch->arg_count,
                               plan->source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_exit_region_dispatch_blocks(
    loom_low_lower_context_t* context,
    const loom_amdgpu_loop_exit_region_plan_t* plan, loom_type_t active_type) {
  loom_block_t* dispatch =
      loom_low_lower_control_block(context, plan->arm_dispatch_block);
  loom_block_t* arm_body =
      loom_low_lower_control_block(context, plan->arm_body_block);
  loom_block_t* clean_only =
      loom_low_lower_control_block(context, plan->clean_only_block);
  loom_block_t* continuation =
      loom_low_lower_control_block(context, plan->continuation);
  IREE_ASSERT(dispatch != NULL);
  IREE_ASSERT(arm_body != NULL);
  IREE_ASSERT(clean_only != NULL);
  IREE_ASSERT(continuation != NULL);
  IREE_ASSERT_EQ(dispatch->op_count, 0);
  IREE_ASSERT_EQ(clean_only->op_count, 0);

  const loom_value_id_t clean_mask =
      loom_block_arg_id(dispatch, plan->merge_arg_count);
  const loom_value_id_t arm_mask_operands[] = {
      plan->saved_exec,
      clean_mask,
  };
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, dispatch);
  loom_op_t* arm_active_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ANDN2_B64_EXEC,
      arm_mask_operands, IREE_ARRAYSIZE(arm_mask_operands),
      loom_named_attr_slice_empty(), &active_type, 1, &arm_active_op);
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_cond_br_build(
        builder, loom_op_const_results(arm_active_op)[0], arm_body, clean_only,
        plan->source_op->location, &branch_op);
  }

  if (iree_status_is_ok(status)) {
    loom_builder_set_block(builder, clean_only);
    loom_op_t* restore_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &clean_mask, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
        &restore_op);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, continuation, dispatch->arg_ids,
                               plan->merge_arg_count, plan->source_op->location,
                               &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_exit_region_merge_blocks(
    loom_low_lower_context_t* context,
    const loom_amdgpu_loop_exit_region_plan_t* plan, loom_type_t exec_mask_type,
    loom_type_t active_type) {
  const loom_amdgpu_loop_exit_plan_t* exit = plan->exit;
  const loom_amdgpu_loop_plan_t* loop = exit->loop;
  loom_block_t* dispatch =
      loom_low_lower_control_block(context, plan->arm_dispatch_block);
  loom_block_t* merge =
      loom_low_lower_control_block(context, plan->merge_block);
  loom_block_t* capture_forward =
      loom_low_lower_control_block(context, plan->capture_forward_block);
  loom_block_t* direct_forward =
      loom_low_lower_control_block(context, plan->direct_forward_block);
  loom_block_t* no_repeat =
      loom_low_lower_control_block(context, plan->no_repeat_block);
  loom_block_t* local_capture =
      loom_low_lower_control_block(context, exit->local_capture_block);
  loom_block_t* continuation =
      loom_low_lower_control_block(context, exit->repeat_dest);
  IREE_ASSERT(dispatch != NULL);
  IREE_ASSERT(merge != NULL);
  IREE_ASSERT(capture_forward != NULL);
  IREE_ASSERT(direct_forward != NULL);
  IREE_ASSERT(no_repeat != NULL);
  IREE_ASSERT(local_capture != NULL);
  IREE_ASSERT(continuation != NULL);
  IREE_ASSERT_EQ(merge->op_count, 0);
  IREE_ASSERT_EQ(capture_forward->op_count, 0);
  IREE_ASSERT_EQ(direct_forward->op_count, 0);
  IREE_ASSERT_EQ(no_repeat->op_count, 0);

  const loom_value_id_t clean_mask =
      loom_block_arg_id(dispatch, plan->merge_arg_count);
  const loom_value_id_t exit_mask =
      loom_block_arg_id(local_capture, loop->capture_value_count);
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, merge);

  loom_op_t* read_exec_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      /*operands=*/NULL, 0, loom_named_attr_slice_empty(), &exec_mask_type, 1,
      &read_exec_op);
  loom_value_id_t survivors = LOOM_VALUE_ID_INVALID;
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_emit_native_i1_mask_binary(
        context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
        loom_op_const_results(read_exec_op)[0], clean_mask, exec_mask_type,
        &survivors);
  }

  loom_value_id_t* merged_args = NULL;
  if (iree_status_is_ok(status) && plan->merge_arg_count != 0) {
    status = loom_low_lower_allocate_emission_array(
        context, plan->merge_arg_count, sizeof(*merged_args),
        (void**)&merged_args);
  }
  loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0;
       i < plan->source_merge_arg_count && iree_status_is_ok(status); ++i) {
    const loom_value_id_t clean_value = loom_block_arg_id(dispatch, i);
    const loom_value_id_t exit_value = loom_block_arg_id(merge, i);
    const loom_value_id_t false_value =
        plan->clean_successor_index == 0 ? exit_value : clean_value;
    const loom_value_id_t true_value =
        plan->clean_successor_index == 0 ? clean_value : exit_value;
    status = loom_amdgpu_emit_masked_merge_value(
        context, plan->source_op, plan->saved_exec,
        plan->merge_constants ? plan->merge_constants[i]
                              : (loom_amdgpu_mask_merge_constants_t){0},
        false_value, true_value, plan->source_condition,
        loom_module_value_type(module, clean_value), &merged_args[i]);
  }
  for (uint16_t i = plan->source_merge_arg_count; i < plan->merge_arg_count;
       ++i) {
    merged_args[i] = loom_block_arg_id(merge, i);
  }

  loom_op_t* returned_op = NULL;
  if (iree_status_is_ok(status)) {
    const loom_value_id_t returned_operands[] = {exit_mask, exit_mask};
    const loom_type_t returned_types[] = {exec_mask_type, active_type};
    status = loom_amdgpu_emit_low_op(
        context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64_SCC,
        returned_operands, IREE_ARRAYSIZE(returned_operands),
        loom_named_attr_slice_empty(), returned_types,
        IREE_ARRAYSIZE(returned_types), &returned_op);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_cond_br_build(
        builder, loom_op_const_results(returned_op)[1], capture_forward,
        direct_forward, plan->source_op->location, &branch_op);
  }

  if (iree_status_is_ok(status)) {
    loom_builder_set_block(builder, direct_forward);
    loom_op_t* set_exec_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &survivors, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
        &set_exec_op);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, continuation, merged_args,
                               plan->merge_arg_count, plan->source_op->location,
                               &branch_op);
  }

  if (iree_status_is_ok(status)) {
    loom_builder_set_block(builder, capture_forward);
    loom_op_t* set_exec_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &exit_mask, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
        &set_exec_op);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_emit_loop_capture_branch(
        context, exit, plan->source_op, local_capture->arg_ids,
        loop->capture_value_count, survivors, merged_args,
        plan->merge_arg_count);
  }

  if (iree_status_is_ok(status)) {
    loom_builder_set_block(builder, no_repeat);
    loom_op_t* set_exec_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &exit_mask, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
        &set_exec_op);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_emit_loop_capture_branch(
        context, exit, plan->source_op, local_capture->arg_ids,
        loop->capture_value_count, clean_mask, dispatch->arg_ids,
        plan->merge_arg_count);
  }

  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_exit_region_blocks(
    loom_low_lower_context_t* context,
    loom_amdgpu_loop_exit_region_plan_t* plan, loom_value_id_t saved_exec,
    loom_value_id_t source_condition, loom_type_t exec_mask_type,
    loom_type_t active_type) {
  IREE_ASSERT(plan->exit != NULL);
  plan->saved_exec = saved_exec;
  plan->source_condition = source_condition;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_loop_exit_region_clean_adapters(
      context, plan, exec_mask_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_loop_exit_region_no_clean_block(
      context, plan, exec_mask_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_loop_exit_region_dispatch_blocks(
      context, plan, active_type));
  return loom_amdgpu_emit_loop_exit_region_merge_blocks(
      context, plan, exec_mask_type, active_type);
}

static iree_status_t loom_amdgpu_emit_loop_exit_adapters(
    loom_low_lower_context_t* context, const loom_amdgpu_loop_exit_plan_t* exit,
    loom_value_id_t saved_active) {
  const loom_amdgpu_loop_plan_t* plan = exit->loop;
  loom_block_t* capture_block = loom_low_lower_control_block(
      context,
      exit->region != NULL ? exit->local_capture_block : plan->capture_block);
  IREE_ASSERT(capture_block != NULL);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < exit->adapter_count && iree_status_is_ok(status);
       ++i) {
    loom_block_t* adapter =
        loom_low_lower_control_block(context, exit->adapters[i].block);
    IREE_ASSERT(adapter != NULL);
    IREE_ASSERT_EQ(adapter->op_count, 0);
    IREE_ASSERT_EQ(adapter->arg_count, plan->capture_value_count);
    loom_builder_set_block(builder, adapter);

    loom_value_id_t exit_mask = LOOM_VALUE_ID_INVALID;
    loom_value_id_t repeat_mask = LOOM_VALUE_ID_INVALID;
    if (exit->is_divergent) {
      loom_op_t* read_exec_op = NULL;
      status = loom_amdgpu_emit_low_op(
          context, exit->source_op,
          LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
          /*operands=*/NULL, 0, loom_named_attr_slice_empty(),
          &plan->exec_mask_type, 1, &read_exec_op);
      if (iree_status_is_ok(status)) {
        exit_mask = loom_value_slice_get(loom_low_op_results(read_exec_op), 0);
        status = loom_amdgpu_emit_native_i1_mask_binary(
            context, exit->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
            saved_active, exit_mask, plan->exec_mask_type, &repeat_mask);
      }
    } else {
      status = loom_amdgpu_emit_zero_native_i1_mask(
          context, exit->source_op, plan->exec_mask_type, &repeat_mask);
    }

    if (iree_status_is_ok(status) && exit->region != NULL) {
      IREE_ASSERT_NE(exit_mask, LOOM_VALUE_ID_INVALID);
      const uint16_t local_arg_count = capture_block->arg_count;
      loom_value_id_t* local_args = NULL;
      status = loom_low_lower_allocate_emission_array(
          context, local_arg_count, sizeof(*local_args), (void**)&local_args);
      if (iree_status_is_ok(status) && plan->capture_value_count != 0) {
        memcpy(local_args, adapter->arg_ids,
               plan->capture_value_count * sizeof(*local_args));
      }
      if (iree_status_is_ok(status)) {
        local_args[plan->capture_value_count] = exit_mask;
        local_args[plan->capture_value_count + 1] = repeat_mask;
        loom_op_t* branch_op = NULL;
        status = loom_low_br_build(builder, capture_block, local_args,
                                   local_arg_count, exit->source_op->location,
                                   &branch_op);
      }
    } else if (iree_status_is_ok(status)) {
      status = loom_amdgpu_emit_loop_capture_branch(
          context, exit, exit->source_op, adapter->arg_ids,
          plan->capture_value_count, repeat_mask, /*resume_args=*/NULL,
          /*resume_arg_count=*/0);
    }
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_no_exit_block(
    loom_low_lower_context_t* context, const loom_amdgpu_loop_exit_plan_t* exit,
    loom_value_id_t saved_active) {
  loom_block_t* no_exit_block =
      loom_low_lower_control_block(context, exit->no_exit_block);
  loom_block_t* repeat_dest = loom_low_lower_control_block(
      context,
      exit->region != NULL ? exit->local_capture_block : exit->repeat_dest);
  IREE_ASSERT(no_exit_block != NULL);
  IREE_ASSERT(repeat_dest != NULL);
  IREE_ASSERT_EQ(no_exit_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, no_exit_block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, exit->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_active, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);
  if (iree_status_is_ok(status)) {
    if (exit->region == NULL) {
      loom_value_slice_t repeat_args = {0};
      status = loom_low_lower_control_append_carrier_arguments(
          context, exit->repeat_dest, /*prefix_values=*/NULL,
          /*prefix_count=*/0, &repeat_args);
      loom_op_t* branch_op = NULL;
      if (iree_status_is_ok(status)) {
        status = loom_low_br_build(builder, repeat_dest, repeat_args.values,
                                   repeat_args.count, exit->source_op->location,
                                   &branch_op);
      }
    } else {
      const loom_amdgpu_loop_plan_t* plan = exit->loop;
      const uint16_t local_arg_count = repeat_dest->arg_count;
      loom_value_id_t* local_args = NULL;
      status = loom_low_lower_allocate_emission_array(
          context, local_arg_count, sizeof(*local_args), (void**)&local_args);
      loom_amdgpu_zero_placeholder_t* placeholders = NULL;
      if (iree_status_is_ok(status) && plan->capture_value_count != 0) {
        status = loom_low_lower_allocate_emission_array(
            context, plan->capture_value_count, sizeof(*placeholders),
            (void**)&placeholders);
      }
      uint16_t placeholder_count = 0;
      loom_module_t* module = loom_low_lower_context_module(context);
      for (uint16_t i = 0;
           i < plan->capture_value_count && iree_status_is_ok(status); ++i) {
        const loom_type_t type =
            loom_module_value_type(module, loom_block_arg_id(repeat_dest, i));
        local_args[i] = LOOM_VALUE_ID_INVALID;
        for (uint16_t j = 0; j < placeholder_count; ++j) {
          if (loom_type_equal(placeholders[j].type, type)) {
            local_args[i] = placeholders[j].value;
            break;
          }
        }
        if (local_args[i] == LOOM_VALUE_ID_INVALID) {
          status = loom_amdgpu_emit_zero_merge_value(context, exit->source_op,
                                                     type, &local_args[i]);
          if (iree_status_is_ok(status)) {
            placeholders[placeholder_count++] =
                (loom_amdgpu_zero_placeholder_t){
                    .type = type,
                    .value = local_args[i],
                };
          }
        }
      }
      if (iree_status_is_ok(status)) {
        status = loom_amdgpu_emit_zero_native_i1_mask(
            context, exit->source_op, plan->exec_mask_type,
            &local_args[plan->capture_value_count]);
      }
      if (iree_status_is_ok(status)) {
        local_args[plan->capture_value_count + 1] = saved_active;
        loom_op_t* branch_op = NULL;
        status =
            loom_low_br_build(builder, repeat_dest, local_args, local_arg_count,
                              exit->source_op->location, &branch_op);
      }
    }
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_local_capture_block(
    loom_low_lower_context_t* context,
    const loom_amdgpu_loop_exit_plan_t* exit) {
  IREE_ASSERT(exit->region != NULL);
  const loom_amdgpu_loop_plan_t* plan = exit->loop;
  loom_block_t* local_capture =
      loom_low_lower_control_block(context, exit->local_capture_block);
  loom_block_t* repeat_dest =
      loom_low_lower_control_block(context, exit->source_repeat_dest);
  loom_block_t* no_repeat =
      loom_low_lower_control_block(context, exit->region->no_repeat_block);
  IREE_ASSERT(local_capture != NULL);
  IREE_ASSERT(repeat_dest != NULL);
  IREE_ASSERT(no_repeat != NULL);
  IREE_ASSERT_EQ(local_capture->op_count, 0);

  const loom_value_id_t repeat_mask =
      loom_block_arg_id(local_capture, plan->capture_value_count + 1);
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_value_id_t active_operands[] = {repeat_mask, repeat_mask};
  const loom_type_t active_result_types[] = {plan->exec_mask_type, scc_type};

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, local_capture);
  loom_op_t* active_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, exit->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64_SCC,
      active_operands, IREE_ARRAYSIZE(active_operands),
      loom_named_attr_slice_empty(), active_result_types,
      IREE_ARRAYSIZE(active_result_types), &active_op);
  if (iree_status_is_ok(status)) {
    loom_op_t* set_exec_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, exit->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        loom_op_const_results(active_op), 1, loom_named_attr_slice_empty(),
        /*result_types=*/NULL, 0, &set_exec_op);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_cond_br_build(
        builder, loom_op_const_results(active_op)[1], repeat_dest, no_repeat,
        exit->source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_capture_blocks(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_loop_plan_t* plan, loom_value_id_t saved_exec) {
  loom_block_t* capture_block =
      loom_low_lower_control_block(context, plan->capture_block);
  loom_block_t* restore_block =
      loom_low_lower_control_block(context, plan->restore_block);
  loom_block_t* restore_dest =
      loom_low_lower_control_block(context, plan->restore_dest);
  IREE_ASSERT(capture_block != NULL);
  IREE_ASSERT(restore_block != NULL);
  IREE_ASSERT(restore_dest != NULL);
  IREE_ASSERT_EQ(capture_block->op_count, 0);
  IREE_ASSERT_EQ(restore_block->op_count, 0);

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_value_id_t repeat_mask =
      loom_block_arg_id(capture_block, plan->capture_value_count);
  const loom_value_id_t active_operands[] = {repeat_mask, repeat_mask};
  const loom_type_t active_result_types[] = {plan->exec_mask_type, scc_type};

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, capture_block);
  loom_op_t* active_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64_SCC,
      active_operands, IREE_ARRAYSIZE(active_operands),
      loom_named_attr_slice_empty(), active_result_types,
      IREE_ARRAYSIZE(active_result_types), &active_op);
  if (iree_status_is_ok(status)) {
    const loom_value_id_t active_mask = loom_op_const_results(active_op)[0];
    loom_op_t* set_exec_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &active_mask, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL,
        0, &set_exec_op);
  }
  if (iree_status_is_ok(status)) {
    const loom_amdgpu_loop_exit_plan_t* first = NULL;
    for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
      if (plan->exit_plans[i].is_divergent) {
        first = &plan->exit_plans[i];
        break;
      }
    }
    IREE_ASSERT(first != NULL);
    loom_op_t* branch_op = NULL;
    status = loom_low_cond_br_build(
        builder, loom_op_const_results(active_op)[1],
        loom_low_lower_control_block(context, first->dispatch_block),
        restore_block, source_op->location, &branch_op);
  }

  for (uint32_t i = 0; i < plan->exit_plan_count && iree_status_is_ok(status);
       ++i) {
    const loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (!exit->is_divergent) {
      continue;
    }
    loom_block_t* dispatch =
        loom_low_lower_control_block(context, exit->dispatch_block);
    IREE_ASSERT(dispatch != NULL);
    IREE_ASSERT_EQ(dispatch->op_count, 0);
    loom_builder_set_block(builder, dispatch);

    loom_block_t* resume_dest =
        loom_low_lower_control_block(context, exit->repeat_dest);
    if (exit->resume_block != 0) {
      loom_block_t* resume =
          loom_low_lower_control_block(context, exit->resume_block);
      IREE_ASSERT(resume != NULL);
      IREE_ASSERT_EQ(resume->op_count, 0);
      loom_builder_set_block(builder, resume);
      loom_op_t* branch_op = NULL;
      status = loom_low_br_build(
          builder, resume_dest,
          capture_block->arg_ids + exit->resume_arg_offset,
          exit->resume_arg_count, source_op->location, &branch_op);
      if (!iree_status_is_ok(status)) {
        break;
      }
      resume_dest = resume;
      loom_builder_set_block(builder, dispatch);
    }

    const bool is_last = exit->selector + 1 == plan->divergent_exit_count;
    if (is_last) {
      loom_op_t* branch_op = NULL;
      status = loom_low_br_build(builder, resume_dest, /*args=*/NULL, 0,
                                 source_op->location, &branch_op);
      continue;
    }

    loom_value_id_t selector_constant = LOOM_VALUE_ID_INVALID;
    status = loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        exit->selector, plan->selector_type, &selector_constant);
    if (!iree_status_is_ok(status)) {
      break;
    }
    const loom_value_id_t selector =
        loom_block_arg_id(capture_block, plan->capture_value_count + 1);
    const loom_value_id_t compare_operands[] = {selector, selector_constant};
    loom_op_t* compare_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32,
        compare_operands, IREE_ARRAYSIZE(compare_operands),
        loom_named_attr_slice_empty(), &scc_type, 1, &compare_op);
    if (iree_status_is_ok(status)) {
      const loom_amdgpu_loop_exit_plan_t* next = NULL;
      for (uint32_t j = i + 1; j < plan->exit_plan_count; ++j) {
        if (plan->exit_plans[j].is_divergent) {
          next = &plan->exit_plans[j];
          break;
        }
      }
      IREE_ASSERT(next != NULL);
      loom_op_t* branch_op = NULL;
      status = loom_low_cond_br_build(
          builder, loom_op_const_results(compare_op)[0], resume_dest,
          loom_low_lower_control_block(context, next->dispatch_block),
          source_op->location, &branch_op);
    }
  }

  if (iree_status_is_ok(status)) {
    loom_builder_set_block(builder, restore_block);
    loom_op_t* restore_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
        &restore_op);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, restore_dest, capture_block->arg_ids,
                               plan->capture_value_count, source_op->location,
                               &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_divergent_loop_blocks(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t exec_mask_type, loom_amdgpu_loop_plan_t* plan,
    loom_value_id_t* out_saved_exec) {
  loom_block_t* entry_block =
      loom_low_lower_control_block(context, plan->entry_block);
  loom_block_t* header_dest =
      loom_low_lower_control_block(context, plan->header_dest);
  *out_saved_exec = plan->saved_exec;
  if (plan->saved_exec != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_ASSERT(entry_block != NULL);
  IREE_ASSERT(header_dest != NULL);
  IREE_ASSERT_EQ(entry_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, entry_block);

  loom_op_t* read_exec_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      /*operands=*/NULL, 0, loom_named_attr_slice_empty(), &exec_mask_type, 1,
      &read_exec_op);
  if (iree_status_is_ok(status)) {
    *out_saved_exec =
        loom_value_slice_get(loom_low_op_results(read_exec_op), 0);
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, header_dest, entry_block->arg_ids,
                               entry_block->arg_count, source_op->location,
                               &branch_op);
  }

  loom_builder_restore(builder, saved_ip);
  if (iree_status_is_ok(status) && plan->capture_exits) {
    status = loom_amdgpu_emit_loop_capture_blocks(context, source_op, plan,
                                                  *out_saved_exec);
  }
  if (iree_status_is_ok(status) && plan->capture_exits) {
    for (uint32_t i = 0; i < plan->exit_plan_count && iree_status_is_ok(status);
         ++i) {
      if (!plan->exit_plans[i].is_divergent) {
        status = loom_amdgpu_emit_loop_exit_adapters(
            context, &plan->exit_plans[i], LOOM_VALUE_ID_INVALID);
      }
    }
  }
  if (iree_status_is_ok(status) && !plan->capture_exits) {
    status = loom_amdgpu_loop_emit_exec_restore_block(
        context, source_op, *out_saved_exec,
        loom_low_lower_control_block(context, plan->restore_block),
        loom_low_lower_control_block(context, plan->restore_dest));
  }
  if (iree_status_is_ok(status)) {
    plan->saved_exec = *out_saved_exec;
  }
  return status;
}

iree_status_t loom_amdgpu_loop_try_prepare_exit_region(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool* out_prepared) {
  return loom_amdgpu_try_prepare_loop_exit_region(context, source_op,
                                                  out_prepared);
}

iree_status_t loom_amdgpu_loop_try_prepare_exit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    bool is_divergent, iree_arena_allocator_t* analysis_arena,
    bool* out_handled, bool* out_active_uniform) {
  return loom_amdgpu_try_prepare_loop_exit(context, source_op, is_divergent,
                                           analysis_arena, out_handled,
                                           out_active_uniform);
}

bool loom_amdgpu_loop_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_AMDGPU_BRANCH_PLAN_DIVERGENT_LOOP ||
         plan.id == LOOM_AMDGPU_BRANCH_PLAN_LOOP_EXIT_REGION;
}

iree_status_t loom_amdgpu_loop_emit_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan, loom_value_id_t low_condition,
    loom_block_t* low_true_dest, loom_block_t* low_false_dest,
    loom_type_t condition_type) {
  IREE_ASSERT(loom_amdgpu_loop_plan_isa(plan));
  const loom_amdgpu_loop_exit_plan_t* loop_exit = NULL;
  loom_amdgpu_loop_exit_region_plan_t* exit_region = NULL;
  if (plan.id == LOOM_AMDGPU_BRANCH_PLAN_DIVERGENT_LOOP) {
    loop_exit = (const loom_amdgpu_loop_exit_plan_t*)plan.target_data;
    IREE_ASSERT(loop_exit != NULL);
  } else {
    exit_region = (loom_amdgpu_loop_exit_region_plan_t*)plan.target_data;
    IREE_ASSERT(exit_region != NULL);
  }

  loom_type_t active_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &active_type));
  const loom_type_t result_types[] = {condition_type, active_type};

  if (exit_region != NULL) {
    const loom_value_id_t source_condition = low_condition;
    if (exit_region->clean_successor_index != 0) {
      loom_op_t* read_exec_op = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
          /*operands=*/NULL, 0, loom_named_attr_slice_empty(), &condition_type,
          1, &read_exec_op));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
          loom_op_const_results(read_exec_op)[0], low_condition, condition_type,
          &low_condition));
    }

    loom_op_t* saveexec_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
        &low_condition, 1, loom_named_attr_slice_empty(), result_types,
        IREE_ARRAYSIZE(result_types), &saveexec_op));
    const loom_value_id_t saved_exec = loom_op_const_results(saveexec_op)[0];
    const loom_value_id_t active = loom_op_const_results(saveexec_op)[1];
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_loop_exit_region_blocks(
        context, exit_region, saved_exec, source_condition, condition_type,
        active_type));

    loom_block_t* clean_dest = exit_region->clean_successor_index == 0
                                   ? low_true_dest
                                   : low_false_dest;
    loom_block_t* no_clean_dest = exit_region->clean_successor_index == 0
                                      ? low_false_dest
                                      : low_true_dest;
    return loom_low_lower_control_emit_cond_branch(context, source_op, active,
                                                   clean_dest, no_clean_dest);
  }

  loom_value_id_t loop_saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_divergent_loop_blocks(
      context, source_op, condition_type, loop_exit->loop, &loop_saved_exec));
  IREE_ASSERT_NE(loop_saved_exec, LOOM_VALUE_ID_INVALID);

  if (!loop_exit->loop->capture_exits) {
    if (loop_exit->exit_successor_mask == 3) {
      IREE_ASSERT_EQ(low_true_dest, low_false_dest);
      loom_op_t* branch_op = NULL;
      return loom_low_br_build(loom_low_lower_context_builder(context),
                               low_true_dest, /*args=*/NULL, 0,
                               source_op->location, &branch_op);
    }
    if (loop_exit->exit_successor_mask == 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
          loop_saved_exec, low_condition, condition_type, &low_condition));
      loom_block_t* repeat_dest = low_false_dest;
      low_false_dest = low_true_dest;
      low_true_dest = repeat_dest;
    }
  } else if (loop_exit->exit_successor_mask == 2) {
    // Select the false lanes from the currently active loop lanes. Lanes
    // retired on earlier iterations remain absent from both masks.
    loom_op_t* read_exec_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
        /*operands=*/NULL, 0, loom_named_attr_slice_empty(), &condition_type, 1,
        &read_exec_op));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
        loom_op_const_results(read_exec_op)[0], low_condition, condition_type,
        &low_condition));
  } else {
    IREE_ASSERT_NE(loop_exit->exit_successor_mask, 3);
  }

  if (loop_exit->loop->capture_exits) {
    loom_op_t* saveexec_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
        &low_condition, 1, loom_named_attr_slice_empty(), result_types,
        IREE_ARRAYSIZE(result_types), &saveexec_op));
    const loom_value_id_t saved_active = loom_op_const_results(saveexec_op)[0];
    const loom_value_id_t exit_active = loom_op_const_results(saveexec_op)[1];
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_loop_exit_adapters(context, loop_exit, saved_active));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_loop_no_exit_block(context, loop_exit, saved_active));
    if (loop_exit->region != NULL) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_loop_local_capture_block(context, loop_exit));
    }

    loom_block_t* exit_dest =
        loop_exit->exit_successor_mask == 1 ? low_true_dest : low_false_dest;
    loom_block_t* no_exit_dest =
        loop_exit->exit_successor_mask == 1 ? low_false_dest : low_true_dest;
    return loom_low_lower_control_emit_cond_branch(
        context, source_op, exit_active, exit_dest, no_exit_dest);
  }

  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      &low_condition, 1, loom_named_attr_slice_empty(), result_types,
      IREE_ARRAYSIZE(result_types), &saveexec_op));
  return loom_low_lower_control_emit_cond_branch(
      context, source_op, loom_op_const_results(saveexec_op)[1], low_true_dest,
      low_false_dest);
}
