// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/loop_placement.h"

#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"

static int loom_low_loop_placement_compare_value_ids(const void* lhs_ptr,
                                                     const void* rhs_ptr) {
  const loom_value_id_t lhs = *(const loom_value_id_t*)lhs_ptr;
  const loom_value_id_t rhs = *(const loom_value_id_t*)rhs_ptr;
  return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
}

static bool loom_low_loop_placement_contains_value_id(
    const loom_value_id_t* sorted_value_ids, iree_host_size_t value_count,
    loom_value_id_t value_id) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = value_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (sorted_value_ids[middle] < value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin < value_count && sorted_value_ids[begin] == value_id;
}

static bool loom_low_loop_placement_contains_candidate_node(
    const loom_low_loop_placement_candidate_t* candidates,
    iree_host_size_t candidate_count, uint32_t node_index) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = candidate_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (candidates[middle].source_node_index < node_index) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return (begin < candidate_count &&
          candidates[begin].source_node_index == node_index) ||
         (begin > 0 &&
          candidates[begin - 1].source_node_index + 1 == node_index);
}

static bool loom_low_loop_placement_descriptor_reads_state(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        loom_low_descriptor_operand(descriptor_set, descriptor, i);
    if (iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_STATE_READ)) {
      return true;
    }
  }
  return false;
}

typedef struct loom_low_loop_placement_planner_t {
  const loom_low_schedule_table_t* schedule;
  const loom_low_allocation_table_t* allocation;
  loom_low_loop_placement_cost_query_fn_t cost_query;
  void* cost_query_user_data;
  iree_arena_allocator_t* arena;
  loom_low_loop_execution_path_index_t execution_index;
  bool execution_index_built;
} loom_low_loop_placement_planner_t;

static iree_status_t loom_low_loop_placement_match_candidate(
    loom_low_loop_placement_planner_t* planner, uint32_t source_node_index,
    const loom_low_schedule_node_t* copy_node,
    const loom_low_schedule_node_t* descriptor_node,
    loom_low_loop_placement_candidate_t* out_candidate, bool* out_matched) {
  *out_matched = false;
  const loom_low_schedule_table_t* schedule = planner->schedule;
  const loom_low_allocation_table_t* allocation = planner->allocation;
  const loom_op_t* copy_op = copy_node->op;
  const loom_op_t* descriptor_op = descriptor_node->op;
  const loom_low_descriptor_t* descriptor = descriptor_node->descriptor;
  if (copy_node->block_index != descriptor_node->block_index ||
      copy_op->next_op != descriptor_op || copy_op->kind != LOOM_OP_LOW_COPY ||
      copy_node->operand_count != 1 || copy_node->result_count != 1 ||
      descriptor == NULL || descriptor_node->result_count != 1 ||
      descriptor->result_count != 1 || descriptor_op->next_op == NULL ||
      !iree_any_bit_set(descriptor->flags,
                        LOOM_LOW_DESCRIPTOR_FLAG_SAFE_TO_SPECULATE) ||
      loom_low_loop_placement_descriptor_reads_state(
          schedule->target.descriptor_set, descriptor)) {
    return iree_ok_status();
  }

  const loom_value_id_t copy_result_value_id =
      loom_op_const_results(copy_op)[0];
  const loom_value_id_t descriptor_result_value_id =
      loom_op_const_results(descriptor_op)[0];
  const loom_value_t* copy_result =
      loom_module_value(schedule->module, copy_result_value_id);
  const loom_value_t* descriptor_result =
      loom_module_value(schedule->module, descriptor_result_value_id);
  if (!loom_value_has_single_use(copy_result) ||
      loom_use_user_op(*loom_value_single_use(copy_result)) != descriptor_op ||
      loom_value_has_attribute_uses(copy_result) ||
      loom_module_value_has_type_uses(schedule->module, copy_result_value_id) ||
      !loom_value_has_single_use(descriptor_result) ||
      loom_value_has_attribute_uses(descriptor_result) ||
      loom_module_value_has_type_uses(schedule->module,
                                      descriptor_result_value_id)) {
    return iree_ok_status();
  }

  const loom_value_ordinal_t copy_result_ordinal =
      loom_low_schedule_node_const_result_ordinals(copy_node)[0];
  const loom_value_ordinal_t descriptor_result_ordinal =
      loom_low_schedule_node_const_result_ordinals(descriptor_node)[0];
  const loom_low_allocation_assignment_t* descriptor_assignment =
      loom_low_allocation_assignment_for_value_ordinal(
          allocation, descriptor_result_ordinal, NULL);
  if (descriptor_assignment == NULL ||
      !loom_low_allocation_assignment_is_spill_slot(descriptor_assignment)) {
    return iree_ok_status();
  }

  const loom_low_placement_relation_range_t result_relations =
      loom_low_placement_relation_range_for_value_ordinal(
          &allocation->placement, descriptor_result_ordinal);
  if (result_relations.count == 0 ||
      allocation->placement.relations[result_relations.start].cause !=
          LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT ||
      loom_low_placement_tied_source_for_value_ordinal(
          &allocation->placement, descriptor_result_ordinal) !=
          copy_result_ordinal) {
    return iree_ok_status();
  }
  const loom_low_placement_relation_t* copy_relation =
      loom_low_placement_defining_transfer_for_value_ordinal(
          &allocation->placement, copy_result_ordinal);
  if (copy_relation == NULL || copy_relation->op != copy_op ||
      copy_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY) {
    return iree_ok_status();
  }

  const loom_op_t* destination_op =
      loom_use_user_op(*loom_value_single_use(descriptor_result));
  const loom_low_schedule_node_t* destination_node =
      loom_low_schedule_node_for_op(schedule, destination_op);
  if (destination_node == NULL ||
      destination_node->block_index == copy_node->block_index) {
    return iree_ok_status();
  }

  loom_low_loop_placement_cost_t cost = {0};
  if (!planner->cost_query(planner->cost_query_user_data, descriptor, &cost) ||
      cost.maximum_schedule_cycle_count == 0) {
    return iree_ok_status();
  }
  if (!planner->execution_index_built) {
    IREE_RETURN_IF_ERROR(loom_low_loop_execution_path_index_build(
        &schedule->loop_forest, &schedule->loop_execution, planner->arena,
        &planner->execution_index));
    planner->execution_index_built = true;
  }
  uint64_t execution_multiplier = 0;
  if (!loom_low_loop_execution_path_multiplier(
          &planner->execution_index, copy_node->block_index,
          destination_node->block_index, &execution_multiplier) ||
      execution_multiplier > cost.maximum_execution_multiplier) {
    return iree_ok_status();
  }

  *out_candidate = (loom_low_loop_placement_candidate_t){
      .copy_op = (loom_op_t*)copy_op,
      .descriptor_op = (loom_op_t*)descriptor_op,
      .destination_op = (loom_op_t*)destination_op,
      .source_anchor_op = descriptor_op->next_op,
      .source_node_index = source_node_index,
      .destination_node_index = (uint32_t)(destination_node - schedule->nodes),
      .maximum_schedule_cycle_count = cost.maximum_schedule_cycle_count,
  };
  *out_matched = true;
  return iree_ok_status();
}

iree_status_t loom_low_loop_placement_plan(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    loom_low_loop_placement_cost_query_fn_t cost_query,
    void* cost_query_user_data, iree_arena_allocator_t* arena,
    loom_low_loop_placement_plan_t* out_plan) {
  *out_plan = (loom_low_loop_placement_plan_t){0};
  if (schedule->loop_execution.facts_by_block == NULL || cost_query == NULL ||
      allocation->spill_plan_count == 0 || schedule->block_count < 2 ||
      schedule->scopes.scope_count != 0 ||
      loom_low_function_schedule(schedule->function_op) ==
          LOOM_LOW_SCHEDULE_LOCKED ||
      loom_low_function_allocation(schedule->function_op) ==
          LOOM_LOW_ALLOCATION_FIXED) {
    return iree_ok_status();
  }

  loom_low_loop_placement_planner_t planner = {
      .schedule = schedule,
      .allocation = allocation,
      .cost_query = cost_query,
      .cost_query_user_data = cost_query_user_data,
      .arena = arena,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t node_index = 0;
       iree_status_is_ok(status) && node_index + 1 < schedule->node_count;
       ++node_index) {
    loom_low_loop_placement_candidate_t candidate = {0};
    bool matched = false;
    status = loom_low_loop_placement_match_candidate(
        &planner, (uint32_t)node_index, &schedule->nodes[node_index],
        &schedule->nodes[node_index + 1], &candidate, &matched);
    if (!iree_status_is_ok(status) || !matched) {
      continue;
    }
    if (out_plan->candidates == NULL) {
      const iree_host_size_t candidate_capacity = allocation->spill_plan_count;
      status = iree_arena_allocate_array(arena, candidate_capacity,
                                         sizeof(*out_plan->candidates),
                                         (void**)&out_plan->candidates);
      if (iree_status_is_ok(status)) {
        status = iree_arena_allocate_array(
            arena, allocation->spill_plan_count,
            sizeof(*out_plan->baseline_spill_value_ids),
            (void**)&out_plan->baseline_spill_value_ids);
      }
      if (iree_status_is_ok(status)) {
        status = iree_arena_allocate_array(
            arena, schedule->block_count,
            sizeof(*out_plan->schedule_cycle_allowances_by_block),
            (void**)&out_plan->schedule_cycle_allowances_by_block);
        if (iree_status_is_ok(status)) {
          memset(out_plan->schedule_cycle_allowances_by_block, 0,
                 schedule->block_count *
                     sizeof(*out_plan->schedule_cycle_allowances_by_block));
        }
      }
      if (iree_status_is_ok(status)) {
        const iree_host_size_t word_count =
            iree_bitmap_calculate_words(schedule->block_count);
        status = iree_arena_allocate_array(
            arena, word_count, sizeof(*out_plan->changed_blocks.words),
            (void**)&out_plan->changed_blocks.words);
        if (iree_status_is_ok(status)) {
          memset(out_plan->changed_blocks.words, 0,
                 word_count * sizeof(*out_plan->changed_blocks.words));
          out_plan->changed_blocks.bit_count = schedule->block_count;
        }
      }
      if (iree_status_is_ok(status)) {
        out_plan->baseline_spill_value_count = allocation->spill_plan_count;
        for (iree_host_size_t i = 0; i < allocation->spill_plan_count; ++i) {
          out_plan->baseline_spill_value_ids[i] =
              allocation->spill_plans[i].value_id;
        }
        qsort(out_plan->baseline_spill_value_ids,
              out_plan->baseline_spill_value_count,
              sizeof(*out_plan->baseline_spill_value_ids),
              loom_low_loop_placement_compare_value_ids);
      }
    }
    if (iree_status_is_ok(status)) {
      IREE_ASSERT_LT(out_plan->candidate_count, allocation->spill_plan_count);
      out_plan->candidates[out_plan->candidate_count++] = candidate;
      ++node_index;
    }
  }
  if (!iree_status_is_ok(status) || out_plan->candidate_count == 0) {
    return status;
  }

  // A candidate destination that is itself moved by another candidate would
  // no longer name the placement point priced above. Retain only independent
  // chains; downstream chains remain eligible and the complete trial still
  // prices any resulting lifetime change in their inputs.
  if (out_plan->candidate_count > 1) {
    for (iree_host_size_t i = 0; i < out_plan->candidate_count; ++i) {
      loom_low_loop_placement_candidate_t* candidate = &out_plan->candidates[i];
      if (loom_low_loop_placement_contains_candidate_node(
              out_plan->candidates, out_plan->candidate_count,
              candidate->destination_node_index)) {
        candidate->maximum_schedule_cycle_count = 0;
      }
    }
    iree_host_size_t retained_count = 0;
    for (iree_host_size_t i = 0; i < out_plan->candidate_count; ++i) {
      const loom_low_loop_placement_candidate_t candidate =
          out_plan->candidates[i];
      if (candidate.maximum_schedule_cycle_count != 0) {
        out_plan->candidates[retained_count++] = candidate;
      }
    }
    out_plan->candidate_count = retained_count;
  }

  const iree_host_size_t changed_word_count =
      iree_bitmap_calculate_words(schedule->block_count);
  memset(out_plan->changed_blocks.words, 0,
         changed_word_count * sizeof(*out_plan->changed_blocks.words));
  for (iree_host_size_t i = 0; i < out_plan->candidate_count; ++i) {
    const loom_low_loop_placement_candidate_t* candidate =
        &out_plan->candidates[i];
    iree_bitmap_set(out_plan->changed_blocks,
                    schedule->nodes[candidate->source_node_index].block_index);
    const uint32_t destination_block =
        schedule->nodes[candidate->destination_node_index].block_index;
    iree_bitmap_set(out_plan->changed_blocks, destination_block);
    out_plan->schedule_cycle_allowances_by_block[destination_block] =
        iree_math_saturating_add_u32(
            out_plan->schedule_cycle_allowances_by_block[destination_block],
            candidate->maximum_schedule_cycle_count);
  }
  return status;
}

iree_status_t loom_low_loop_placement_apply(
    loom_rewriter_t* rewriter, const loom_low_loop_placement_plan_t* plan) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < plan->candidate_count; ++i) {
    const loom_low_loop_placement_candidate_t* candidate = &plan->candidates[i];
    status = loom_rewriter_move_before(rewriter, candidate->copy_op,
                                       candidate->destination_op);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_move_before(rewriter, candidate->descriptor_op,
                                         candidate->destination_op);
    }
  }
  return status;
}

iree_status_t loom_low_loop_placement_rollback(
    loom_rewriter_t* rewriter, const loom_low_loop_placement_plan_t* plan) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = plan->candidate_count;
       iree_status_is_ok(status) && i > 0;) {
    const loom_low_loop_placement_candidate_t* candidate =
        &plan->candidates[--i];
    status = loom_rewriter_move_before(rewriter, candidate->copy_op,
                                       candidate->source_anchor_op);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_move_before(rewriter, candidate->descriptor_op,
                                         candidate->source_anchor_op);
    }
  }
  return status;
}

static void loom_low_loop_placement_spill_traffic(
    const loom_low_allocation_table_t* allocation, uint64_t* out_store_count,
    uint64_t* out_reload_count) {
  uint64_t store_count = 0;
  uint64_t reload_count = 0;
  for (iree_host_size_t i = 0; i < allocation->spill_plan_count; ++i) {
    store_count += allocation->spill_plans[i].store_count;
    reload_count += allocation->spill_plans[i].reload_count;
  }
  *out_store_count = store_count;
  *out_reload_count = reload_count;
}

static bool loom_low_loop_placement_result_is_register_like(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation, const loom_op_t* op) {
  const loom_low_schedule_node_t* node =
      loom_low_schedule_node_for_op(schedule, op);
  if (node == NULL || node->result_count != 1) {
    return false;
  }
  const loom_value_ordinal_t result_ordinal =
      loom_low_schedule_node_const_result_ordinals(node)[0];
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_assignment_for_value_ordinal(allocation,
                                                       result_ordinal, NULL);
  return assignment != NULL &&
         loom_low_allocation_assignment_is_register_like(assignment);
}

bool loom_low_loop_placement_improves_allocation(
    const loom_low_loop_placement_plan_t* plan,
    const loom_low_allocation_table_t* baseline,
    const loom_low_schedule_table_t* trial_schedule,
    const loom_low_allocation_table_t* trial_allocation) {
  if (trial_allocation->spill_plan_count >= baseline->spill_plan_count ||
      trial_allocation->spill_count >= baseline->spill_count ||
      trial_allocation->packet_move_count > baseline->packet_move_count) {
    return false;
  }
  for (iree_host_size_t i = 0; i < trial_allocation->spill_plan_count; ++i) {
    if (!loom_low_loop_placement_contains_value_id(
            plan->baseline_spill_value_ids, plan->baseline_spill_value_count,
            trial_allocation->spill_plans[i].value_id)) {
      return false;
    }
  }

  uint64_t baseline_store_count = 0;
  uint64_t baseline_reload_count = 0;
  uint64_t trial_store_count = 0;
  uint64_t trial_reload_count = 0;
  loom_low_loop_placement_spill_traffic(baseline, &baseline_store_count,
                                        &baseline_reload_count);
  loom_low_loop_placement_spill_traffic(trial_allocation, &trial_store_count,
                                        &trial_reload_count);
  if (trial_store_count > baseline_store_count ||
      trial_reload_count > baseline_reload_count) {
    return false;
  }

  for (iree_host_size_t i = 0; i < plan->candidate_count; ++i) {
    const loom_low_loop_placement_candidate_t* candidate = &plan->candidates[i];
    if (!loom_low_loop_placement_result_is_register_like(
            trial_schedule, trial_allocation, candidate->copy_op) ||
        !loom_low_loop_placement_result_is_register_like(
            trial_schedule, trial_allocation, candidate->descriptor_op)) {
      return false;
    }
  }
  return true;
}

static uint32_t loom_low_loop_placement_block_extent(
    const loom_low_schedule_table_t* schedule, uint32_t block_index) {
  const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
  return block->issue_group_count == 0
             ? 0
             : schedule
                   ->issue_groups[block->issue_group_start +
                                  block->issue_group_count - 1]
                   .issue_cycle;
}

bool loom_low_loop_placement_bounds_schedule(
    const loom_low_loop_placement_plan_t* plan,
    const loom_low_schedule_table_t* baseline,
    const loom_low_schedule_table_t* trial) {
  if (trial->block_count != baseline->block_count) {
    return false;
  }

  for (uint32_t i = 0; i < baseline->block_count; ++i) {
    const uint32_t before = loom_low_loop_placement_block_extent(baseline, i);
    const uint32_t after = loom_low_loop_placement_block_extent(trial, i);
    if (after > before &&
        after - before > plan->schedule_cycle_allowances_by_block[i]) {
      return false;
    }
  }
  return true;
}
