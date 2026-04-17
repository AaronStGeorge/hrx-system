// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg_simplify.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/transforms/rewriter.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/dominance.h"
#include "loom/util/fact_table.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

enum {
  LOOM_CFG_SIMPLIFY_STAT_BRANCHES_FOLDED = 0,
  LOOM_CFG_SIMPLIFY_STAT_EDGES_FORWARDED = 1,
  LOOM_CFG_SIMPLIFY_STAT_BLOCKS_REMOVED = 2,
  LOOM_CFG_SIMPLIFY_STAT_BLOCK_ARGS_REMOVED = 3,
  LOOM_CFG_SIMPLIFY_STAT_BLOCKS_FUSED = 4,
  LOOM_CFG_SIMPLIFY_STAT_DUPLICATE_BLOCKS_MERGED = 5,
};

static const loom_pass_statistic_def_t kCfgSimplifyStatistics[] = {
    {IREE_SVL("branches-folded"),
     IREE_SVL("Number of conditional branches folded to direct branches.")},
    {IREE_SVL("edges-forwarded"),
     IREE_SVL("Number of predecessor edges forwarded through trivial blocks.")},
    {IREE_SVL("blocks-removed"),
     IREE_SVL("Number of unreachable CFG blocks removed.")},
    {IREE_SVL("block-args-removed"),
     IREE_SVL("Number of redundant CFG block arguments removed.")},
    {IREE_SVL("blocks-fused"),
     IREE_SVL("Number of single-predecessor CFG blocks fused into their "
              "predecessors.")},
    {IREE_SVL("duplicate-blocks-merged"),
     IREE_SVL("Number of duplicate terminal CFG blocks merged.")},
};

static const loom_pass_info_t loom_cfg_simplify_pass_info_storage = {
    .name = IREE_SVL("cfg-simplify"),
    .description = IREE_SVL("Simplify explicit CFG block structure."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_defs = kCfgSimplifyStatistics,
    .statistic_count = IREE_ARRAYSIZE(kCfgSimplifyStatistics),
};

const loom_pass_info_t* loom_cfg_simplify_pass_info(void) {
  return &loom_cfg_simplify_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Worklist
//===----------------------------------------------------------------------===//

#define LOOM_CFG_SIMPLIFY_INITIAL_REGION_STACK_CAPACITY 16

typedef struct loom_cfg_simplify_region_stack_t {
  // Regions still waiting for simplification.
  loom_region_t** regions;
  // Number of queued regions.
  iree_host_size_t count;
  // Capacity of regions.
  iree_host_size_t capacity;
} loom_cfg_simplify_region_stack_t;

typedef struct loom_cfg_simplify_state_t {
  // Active pass instance for statistics and scratch allocation.
  loom_pass_t* pass;
  // Module being rewritten.
  loom_module_t* module;
  // Rewriter used for use-list preserving IR edits.
  loom_rewriter_t* rewriter;
  // Per-iteration analysis and temporary allocation arena.
  iree_arena_allocator_t* analysis_arena;
  // Facts computed for the current fixed-point iteration.
  const loom_value_fact_table_t* fact_table;
  // Dominance computed for the current fixed-point iteration.
  const loom_dominance_info_t* dominance;
  // DFS stack for nested regions.
  loom_cfg_simplify_region_stack_t region_stack;
} loom_cfg_simplify_state_t;

static iree_status_t loom_cfg_simplify_region_stack_initialize(
    iree_arena_allocator_t* arena, loom_cfg_simplify_region_stack_t* stack) {
  stack->count = 0;
  stack->capacity = LOOM_CFG_SIMPLIFY_INITIAL_REGION_STACK_CAPACITY;
  return iree_arena_allocate_array(
      arena, stack->capacity, sizeof(*stack->regions), (void**)&stack->regions);
}

static iree_status_t loom_cfg_simplify_region_stack_push(
    iree_arena_allocator_t* arena, loom_cfg_simplify_region_stack_t* stack,
    loom_region_t* region) {
  if (!region) return iree_ok_status();
  if (stack->count >= stack->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, stack->count, stack->count + 1, sizeof(*stack->regions),
        &stack->capacity, (void**)&stack->regions));
  }
  stack->regions[stack->count++] = region;
  return iree_ok_status();
}

static loom_region_t* loom_cfg_simplify_region_stack_pop(
    loom_cfg_simplify_region_stack_t* stack) {
  return stack->count == 0 ? NULL : stack->regions[--stack->count];
}

static iree_status_t loom_cfg_simplify_push_child_regions(
    loom_cfg_simplify_state_t* state, const loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_region_stack_push(
        state->pass->arena, &state->region_stack, regions[region_index]));
  }
  return iree_ok_status();
}

static bool loom_cfg_simplify_region_has_successors(loom_region_t* region) {
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (op->successor_count > 0) return true;
    }
  }
  return false;
}

static iree_status_t loom_cfg_simplify_mark_cfg_regions(
    loom_region_t* root_region, iree_arena_allocator_t* arena) {
  if (!root_region) return iree_ok_status();

  iree_host_size_t stack_capacity =
      LOOM_CFG_SIMPLIFY_INITIAL_REGION_STACK_CAPACITY;
  loom_region_t** stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, stack_capacity, sizeof(*stack), (void**)&stack));
  iree_host_size_t stack_count = 0;
  stack[stack_count++] = root_region;

  while (stack_count > 0) {
    loom_region_t* region = stack[--stack_count];
    if (loom_cfg_simplify_region_has_successors(region)) {
      region->flags |= LOOM_REGION_INSTANCE_FLAG_CFG;
    } else {
      region->flags &= ~LOOM_REGION_INSTANCE_FLAG_CFG;
    }

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t region_index = 0; region_index < op->region_count;
             ++region_index) {
          if (!regions[region_index]) continue;
          if (stack_count >= stack_capacity) {
            IREE_RETURN_IF_ERROR(iree_arena_grow_array(
                arena, stack_count, stack_count + 1, sizeof(*stack),
                &stack_capacity, (void**)&stack));
          }
          stack[stack_count++] = regions[region_index];
        }
      }
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Branch rewriting
//===----------------------------------------------------------------------===//

static iree_status_t loom_cfg_simplify_replace_br(
    loom_cfg_simplify_state_t* state, loom_op_t* old_br, loom_block_t* dest,
    const loom_value_id_t* args, iree_host_size_t arg_count) {
  loom_builder_ip_t saved_ip = loom_builder_save(&state->rewriter->builder);
  loom_builder_set_before(&state->rewriter->builder, old_br);
  loom_op_t* new_br = NULL;
  iree_status_t status =
      loom_cfg_br_build(&state->rewriter->builder, dest, args, arg_count,
                        old_br->location, &new_br);
  loom_builder_restore(&state->rewriter->builder, saved_ip);
  if (!iree_status_is_ok(status)) return status;
  return loom_rewriter_erase(state->rewriter, old_br);
}

static iree_status_t loom_cfg_simplify_replace_cond_br_with_br(
    loom_cfg_simplify_state_t* state, loom_op_t* cond_br, loom_block_t* dest) {
  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_replace_br(state, cond_br, dest, NULL, 0));
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass, LOOM_CFG_SIMPLIFY_STAT_BRANCHES_FOLDED,
                            1);
  }
  return iree_ok_status();
}

static bool loom_cfg_simplify_exact_bool(loom_value_facts_t facts,
                                         bool* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo != 0;
  return true;
}

static iree_status_t loom_cfg_simplify_fold_cond_br(
    loom_cfg_simplify_state_t* state, loom_op_t* op, bool* out_changed) {
  if (!loom_cfg_cond_br_isa(op)) return iree_ok_status();
  loom_block_t* true_dest = loom_cfg_cond_br_true_dest(op);
  loom_block_t* false_dest = loom_cfg_cond_br_false_dest(op);
  if (true_dest == false_dest && true_dest && true_dest->arg_count == 0) {
    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_replace_cond_br_with_br(state, op, true_dest));
    *out_changed = true;
    return iree_ok_status();
  }

  bool condition = false;
  loom_value_facts_t facts = loom_value_fact_table_lookup(
      state->fact_table, loom_cfg_cond_br_condition(op));
  if (!loom_cfg_simplify_exact_bool(facts, &condition)) {
    return iree_ok_status();
  }
  loom_block_t* chosen_dest = condition ? true_dest : false_dest;
  if (!chosen_dest || chosen_dest->arg_count != 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_replace_cond_br_with_br(state, op, chosen_dest));
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_fold_block_branches(
    loom_cfg_simplify_state_t* state, loom_block_t* block, bool* out_changed) {
  loom_op_t* op = block->first_op;
  while (op) {
    loom_op_t* next_op = op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_fold_cond_br(state, op, out_changed));
    if (*out_changed) return iree_ok_status();
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_push_child_regions(state, op));
    op = next_op;
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Unreachable block removal
//===----------------------------------------------------------------------===//

static iree_status_t loom_cfg_simplify_remove_unreachable_blocks(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    bool* out_changed) {
  if (graph->malformed || graph->block_count <= 1) return iree_ok_status();

  bool* remove_blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->analysis_arena, graph->block_count, sizeof(*remove_blocks),
      (void**)&remove_blocks));
  memset(remove_blocks, 0, graph->block_count * sizeof(*remove_blocks));

  bool any_removed = false;
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (loom_cfg_graph_block_is_reachable(graph, block_index)) continue;
    remove_blocks[block_index] = true;
    any_removed = true;
  }
  if (!any_removed) return iree_ok_status();

  uint16_t removed_count = 0;
  IREE_RETURN_IF_ERROR(loom_region_remove_blocks(
      state->module, (loom_region_t*)graph->region, remove_blocks,
      graph->block_count, &removed_count));
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass, LOOM_CFG_SIMPLIFY_STAT_BLOCKS_REMOVED,
                            removed_count);
  }
  state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  *out_changed = removed_count != 0;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Trivial block forwarding
//===----------------------------------------------------------------------===//

static bool loom_cfg_simplify_match_trivial_forward_block(
    const loom_block_t* block, loom_block_t** out_dest,
    loom_value_slice_t* out_args) {
  *out_dest = NULL;
  *out_args = (loom_value_slice_t){0};
  if (!block || block->first_op != block->last_op || !block->first_op ||
      !loom_cfg_br_isa(block->first_op)) {
    return false;
  }
  loom_block_t* dest = loom_cfg_br_dest(block->first_op);
  if (!dest || dest == block) return false;
  *out_dest = dest;
  *out_args = loom_cfg_br_args(block->first_op);
  if (out_args->count != dest->arg_count) return false;
  return true;
}

static bool loom_cfg_simplify_map_forward_arg(
    const loom_cfg_simplify_state_t* state, const loom_block_t* forward_block,
    loom_value_slice_t predecessor_args, loom_value_id_t forward_arg,
    loom_value_id_t* out_arg) {
  *out_arg = LOOM_VALUE_ID_INVALID;
  if (forward_arg == LOOM_VALUE_ID_INVALID ||
      forward_arg >= state->module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(state->module, forward_arg);
  if (!loom_value_is_block_arg(value) ||
      loom_value_def_block(value) != forward_block) {
    *out_arg = forward_arg;
    return true;
  }

  uint16_t arg_index = loom_value_def_index(value);
  if (arg_index >= predecessor_args.count) return false;
  *out_arg = predecessor_args.values[arg_index];
  return *out_arg != LOOM_VALUE_ID_INVALID;
}

static bool loom_cfg_simplify_branch_args_available_before(
    const loom_cfg_simplify_state_t* state, loom_value_slice_t args,
    const loom_op_t* before_op) {
  for (uint16_t i = 0; i < args.count; ++i) {
    loom_value_id_t arg = args.values[i];
    if (!loom_value_is_available_before_op(state->dominance, arg, before_op) ||
        !loom_value_type_is_available_before_op(state->dominance, arg,
                                                before_op)) {
      return false;
    }
  }
  return true;
}

static bool loom_cfg_simplify_branch_args_match_dest(
    const loom_cfg_simplify_state_t* state, const loom_block_t* dest,
    loom_value_slice_t args) {
  if (!dest || args.count != dest->arg_count) return false;
  loom_type_value_remap_t remap = {
      .source_values = dest->arg_ids,
      .target_values = args.values,
      .count = dest->arg_count,
  };
  for (uint16_t i = 0; i < dest->arg_count; ++i) {
    loom_value_id_t actual_id = args.values[i];
    loom_value_id_t expected_id = loom_block_arg_id(dest, i);
    if (actual_id == LOOM_VALUE_ID_INVALID ||
        expected_id == LOOM_VALUE_ID_INVALID ||
        actual_id >= state->module->values.count ||
        expected_id >= state->module->values.count) {
      return false;
    }
    loom_type_t actual_type = loom_module_value_type(state->module, actual_id);
    loom_type_t expected_type =
        loom_module_value_type(state->module, expected_id);
    if (!loom_type_equal_after_value_remap(expected_type, actual_type,
                                           &remap)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_cfg_simplify_compose_forward_args(
    const loom_cfg_simplify_state_t* state, const loom_block_t* forward_block,
    loom_op_t* predecessor_br, const loom_block_t* new_dest,
    loom_value_slice_t forward_args, loom_value_slice_t* out_args,
    bool* out_valid) {
  *out_args = (loom_value_slice_t){0};
  *out_valid = false;

  loom_value_slice_t predecessor_args = loom_cfg_br_args(predecessor_br);
  if (predecessor_args.count != forward_block->arg_count) {
    return iree_ok_status();
  }

  loom_value_id_t* composed_args = NULL;
  if (forward_args.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->analysis_arena, forward_args.count, sizeof(*composed_args),
        (void**)&composed_args));
  }
  for (uint16_t i = 0; i < forward_args.count; ++i) {
    if (!loom_cfg_simplify_map_forward_arg(
            state, forward_block, predecessor_args, forward_args.values[i],
            &composed_args[i])) {
      return iree_ok_status();
    }
  }

  loom_value_slice_t composed = {
      .values = composed_args,
      .count = forward_args.count,
  };
  if (!loom_cfg_simplify_branch_args_available_before(state, composed,
                                                      predecessor_br) ||
      !loom_cfg_simplify_branch_args_match_dest(state, new_dest, composed)) {
    return iree_ok_status();
  }

  *out_args = composed;
  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_forward_branch_edge(
    loom_cfg_simplify_state_t* state, loom_op_t* predecessor_br,
    loom_block_t* old_dest, loom_block_t* new_dest, loom_value_slice_t new_args,
    bool* out_changed) {
  if (!loom_cfg_br_isa(predecessor_br) ||
      loom_cfg_br_dest(predecessor_br) != old_dest) {
    return iree_ok_status();
  }

  loom_value_slice_t composed_args = {0};
  bool valid_args = false;
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_compose_forward_args(
      state, old_dest, predecessor_br, new_dest, new_args, &composed_args,
      &valid_args));
  if (!valid_args) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_replace_br(state, predecessor_br, new_dest,
                                   composed_args.values, composed_args.count));
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass, LOOM_CFG_SIMPLIFY_STAT_EDGES_FORWARDED,
                            1);
  }
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_forward_cond_br_edge(
    loom_cfg_simplify_state_t* state, loom_op_t* predecessor_cond_br,
    loom_block_t* old_dest, loom_block_t* new_dest, loom_value_slice_t new_args,
    bool* out_changed) {
  if (!loom_cfg_cond_br_isa(predecessor_cond_br) || old_dest->arg_count != 0 ||
      new_args.count != 0) {
    return iree_ok_status();
  }
  if (!new_dest || new_dest->arg_count != 0) return iree_ok_status();
  loom_block_t** successors = loom_op_successors(predecessor_cond_br);
  for (uint8_t successor_index = 0; successor_index < 2; ++successor_index) {
    if (successors[successor_index] != old_dest) continue;
    successors[successor_index] = new_dest;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_to_worklist(state->rewriter, predecessor_cond_br));
    state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
    if (state->pass->statistics) {
      loom_pass_statistic_add(state->pass,
                              LOOM_CFG_SIMPLIFY_STAT_EDGES_FORWARDED, 1);
    }
    *out_changed = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_forward_trivial_blocks(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    bool* out_changed) {
  if (graph->malformed) return iree_ok_status();
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index)) continue;
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    loom_block_t* dest = NULL;
    loom_value_slice_t args = {0};
    if (!loom_cfg_simplify_match_trivial_forward_block(block, &dest, &args)) {
      continue;
    }

    loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(graph, block_index);
    for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
      const loom_cfg_block_info_t* predecessor_info =
          &graph->blocks[predecessors.values[i]];
      loom_op_t* terminator = ((loom_block_t*)predecessor_info->block)->last_op;
      if (!terminator) continue;
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_forward_branch_edge(
          state, terminator, block, dest, args, out_changed));
      if (*out_changed) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_forward_cond_br_edge(
          state, terminator, block, dest, args, out_changed));
      if (*out_changed) return iree_ok_status();
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Block fusion
//===----------------------------------------------------------------------===//

static bool loom_cfg_simplify_find_fusable_predecessor(
    const loom_cfg_graph_t* graph, uint16_t block_index,
    loom_op_t** out_predecessor_br, loom_value_slice_t* out_args) {
  *out_predecessor_br = NULL;
  *out_args = (loom_value_slice_t){0};
  if (block_index == 0 ||
      !loom_cfg_graph_block_is_reachable(graph, block_index)) {
    return false;
  }

  const loom_block_t* block = graph->blocks[block_index].block;
  loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  if (predecessors.count != 1) return false;

  const loom_block_t* predecessor = graph->blocks[predecessors.values[0]].block;
  if (!predecessor || predecessor == block) return false;

  loom_op_t* terminator = ((loom_block_t*)predecessor)->last_op;
  if (!terminator || !loom_cfg_br_isa(terminator) ||
      loom_cfg_br_dest(terminator) != block) {
    return false;
  }

  loom_value_slice_t args = loom_cfg_br_args(terminator);
  if (args.count != block->arg_count) return false;
  *out_predecessor_br = terminator;
  *out_args = args;
  return true;
}

static iree_status_t loom_cfg_simplify_validate_fused_block_args(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* predecessor_br, loom_value_slice_t replacements,
    bool* out_valid) {
  *out_valid = false;
  if (replacements.count != block->arg_count) return iree_ok_status();
  if (block->arg_count == 0) {
    *out_valid = true;
    return iree_ok_status();
  }

  loom_value_id_t* old_args = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->analysis_arena, block->arg_count,
                                sizeof(*old_args), (void**)&old_args));
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    old_args[i] = loom_block_arg_id(block, i);
  }

  loom_type_value_remap_t remap = {
      .source_values = old_args,
      .target_values = replacements.values,
      .count = block->arg_count,
  };
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    loom_value_id_t old_arg = old_args[i];
    loom_value_id_t replacement = replacements.values[i];
    if (old_arg == LOOM_VALUE_ID_INVALID ||
        replacement == LOOM_VALUE_ID_INVALID ||
        old_arg >= state->module->values.count ||
        replacement >= state->module->values.count) {
      return iree_ok_status();
    }
    if (!loom_value_is_available_before_op(state->dominance, replacement,
                                           predecessor_br) ||
        !loom_value_type_is_available_before_op(state->dominance, replacement,
                                                predecessor_br)) {
      return iree_ok_status();
    }
    loom_type_t old_type = loom_module_value_type(state->module, old_arg);
    loom_type_t replacement_type =
        loom_module_value_type(state->module, replacement);
    if (!loom_type_equal_after_value_remap(old_type, replacement_type,
                                           &remap)) {
      return iree_ok_status();
    }
  }

  *out_valid = true;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_replace_block_args(
    loom_cfg_simplify_state_t* state, loom_block_t* block,
    loom_value_slice_t replacements) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        state->rewriter, loom_block_arg_id(block, i), replacements.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_move_block_ops_before(
    loom_cfg_simplify_state_t* state, loom_block_t* block,
    loom_op_t* before_op) {
  loom_op_t* op = block->first_op;
  while (op) {
    loom_op_t* next_op = op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_move_before(state->rewriter, op, before_op));
    op = next_op;
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_remove_cfg_block(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index) {
  bool* remove_blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->analysis_arena, graph->block_count, sizeof(*remove_blocks),
      (void**)&remove_blocks));
  memset(remove_blocks, 0, graph->block_count * sizeof(*remove_blocks));
  remove_blocks[block_index] = true;

  uint16_t removed_count = 0;
  IREE_RETURN_IF_ERROR(loom_region_remove_blocks(
      state->module, (loom_region_t*)graph->region, remove_blocks,
      (uint16_t)graph->block_count, &removed_count));
  return removed_count == 1
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "expected to remove one fused block");
}

static iree_status_t loom_cfg_simplify_fuse_single_predecessor_blocks(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    bool* out_changed) {
  if (graph->malformed) return iree_ok_status();
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    if (!block || !block->first_op) continue;

    loom_op_t* predecessor_br = NULL;
    loom_value_slice_t replacements = {0};
    if (!loom_cfg_simplify_find_fusable_predecessor(
            graph, block_index, &predecessor_br, &replacements)) {
      continue;
    }

    bool valid_replacements = false;
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_validate_fused_block_args(
        state, block, predecessor_br, replacements, &valid_replacements));
    if (!valid_replacements) continue;

    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_replace_block_args(state, block, replacements));
    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_move_block_ops_before(state, block, predecessor_br));
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(state->rewriter, predecessor_br));
    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_remove_cfg_block(state, graph, block_index));

    if (state->pass->statistics) {
      loom_pass_statistic_add(state->pass, LOOM_CFG_SIMPLIFY_STAT_BLOCKS_FUSED,
                              1);
    }
    state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
    *out_changed = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Duplicate block merging
//===----------------------------------------------------------------------===//

static bool loom_cfg_simplify_is_mergeable_terminal_block(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index) {
  if (block_index == 0 ||
      !loom_cfg_graph_block_is_reachable(graph, block_index)) {
    return false;
  }
  const loom_block_t* block = graph->blocks[block_index].block;
  if (!block || block->arg_count != 0 || block->first_op != block->last_op ||
      !block->first_op) {
    return false;
  }

  const loom_op_t* terminator = block->first_op;
  loom_trait_flags_t traits =
      loom_op_effective_traits(state->module, terminator);
  return iree_all_bits_set(traits, LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_PURE) &&
         terminator->successor_count == 0 && terminator->result_count == 0 &&
         terminator->region_count == 0;
}

static bool loom_cfg_simplify_terminal_ops_equal(const loom_op_t* lhs,
                                                 const loom_op_t* rhs) {
  if (lhs->kind != rhs->kind || lhs->operand_count != rhs->operand_count ||
      lhs->attribute_count != rhs->attribute_count ||
      lhs->instance_flags != rhs->instance_flags) {
    return false;
  }

  const loom_value_id_t* lhs_operands = loom_op_operands((loom_op_t*)lhs);
  const loom_value_id_t* rhs_operands = loom_op_operands((loom_op_t*)rhs);
  if (memcmp(lhs_operands, rhs_operands,
             (iree_host_size_t)lhs->operand_count * sizeof(*lhs_operands)) !=
      0) {
    return false;
  }

  const loom_attribute_t* lhs_attrs = loom_op_attrs((loom_op_t*)lhs);
  const loom_attribute_t* rhs_attrs = loom_op_attrs((loom_op_t*)rhs);
  for (uint8_t i = 0; i < lhs->attribute_count; ++i) {
    if (!loom_attribute_equal(&lhs_attrs[i], &rhs_attrs[i])) return false;
  }
  return true;
}

static bool loom_cfg_simplify_find_duplicate_terminal_block(
    const loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index, loom_block_t** out_canonical_block) {
  *out_canonical_block = NULL;
  if (!loom_cfg_simplify_is_mergeable_terminal_block(state, graph,
                                                     block_index)) {
    return false;
  }
  const loom_block_t* block = graph->blocks[block_index].block;
  const loom_op_t* terminator = block->first_op;
  for (uint16_t canonical_index = 1; canonical_index < block_index;
       ++canonical_index) {
    if (!loom_cfg_simplify_is_mergeable_terminal_block(state, graph,
                                                       canonical_index)) {
      continue;
    }
    loom_block_t* canonical_block =
        (loom_block_t*)graph->blocks[canonical_index].block;
    if (!loom_cfg_simplify_terminal_ops_equal(canonical_block->first_op,
                                              terminator)) {
      continue;
    }
    *out_canonical_block = canonical_block;
    return true;
  }
  return false;
}

static bool loom_cfg_simplify_can_redirect_successor(
    const loom_op_t* terminator, const loom_block_t* old_dest,
    const loom_block_t* new_dest) {
  if (!new_dest || new_dest->arg_count != 0) return false;
  if (loom_cfg_br_isa(terminator)) {
    return loom_cfg_br_dest(terminator) == old_dest &&
           loom_cfg_br_args(terminator).count == 0;
  }
  if (loom_cfg_cond_br_isa(terminator)) {
    return loom_cfg_cond_br_true_dest(terminator) == old_dest ||
           loom_cfg_cond_br_false_dest(terminator) == old_dest;
  }
  return false;
}

static bool loom_cfg_simplify_can_redirect_block_predecessors(
    const loom_cfg_graph_t* graph, uint16_t block_index,
    loom_block_t* new_dest) {
  loom_block_t* old_dest = (loom_block_t*)graph->blocks[block_index].block;
  loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  if (predecessors.count == 0) return false;
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    const loom_block_t* predecessor =
        graph->blocks[predecessors.values[i]].block;
    if (!predecessor) return false;
    loom_op_t* terminator = ((loom_block_t*)predecessor)->last_op;
    if (!terminator || !loom_cfg_simplify_can_redirect_successor(
                           terminator, old_dest, new_dest)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_cfg_simplify_redirect_successor(
    loom_cfg_simplify_state_t* state, loom_op_t* terminator,
    loom_block_t* old_dest, loom_block_t* new_dest) {
  if (loom_cfg_br_isa(terminator)) {
    return loom_cfg_simplify_replace_br(state, terminator, new_dest, NULL, 0);
  }

  loom_block_t** successors = loom_op_successors(terminator);
  bool changed = false;
  for (uint8_t successor_index = 0;
       successor_index < terminator->successor_count; ++successor_index) {
    if (successors[successor_index] != old_dest) continue;
    successors[successor_index] = new_dest;
    changed = true;
  }
  if (!changed) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_to_worklist(state->rewriter, terminator));
  state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_redirect_block_predecessors(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    uint16_t block_index, loom_block_t* new_dest) {
  loom_block_t* old_dest = (loom_block_t*)graph->blocks[block_index].block;
  loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    loom_block_t* predecessor =
        (loom_block_t*)graph->blocks[predecessors.values[i]].block;
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_redirect_successor(
        state, predecessor->last_op, old_dest, new_dest));
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_merge_duplicate_terminal_blocks(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    bool* out_changed) {
  if (graph->malformed) return iree_ok_status();
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    loom_block_t* canonical_block = NULL;
    if (!loom_cfg_simplify_find_duplicate_terminal_block(
            state, graph, block_index, &canonical_block)) {
      continue;
    }

    if (!loom_cfg_simplify_can_redirect_block_predecessors(graph, block_index,
                                                           canonical_block)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_redirect_block_predecessors(
        state, graph, block_index, canonical_block));
    IREE_RETURN_IF_ERROR(
        loom_cfg_simplify_remove_cfg_block(state, graph, block_index));
    if (state->pass->statistics) {
      loom_pass_statistic_add(
          state->pass, LOOM_CFG_SIMPLIFY_STAT_DUPLICATE_BLOCKS_MERGED, 1);
    }
    state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
    *out_changed = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Block argument removal
//===----------------------------------------------------------------------===//

static bool loom_cfg_simplify_type_allows_replacement(
    const loom_module_t* module, loom_value_id_t old_value,
    loom_value_id_t replacement) {
  loom_type_t old_type = loom_module_value_type(module, old_value);
  loom_type_t replacement_type = loom_module_value_type(module, replacement);
  loom_type_value_remap_t remap = {
      .source_values = &old_value,
      .target_values = &replacement,
      .count = 1,
  };
  return loom_type_equal_after_value_remap(old_type, replacement_type, &remap);
}

static bool loom_cfg_simplify_pred_branches_to_block(
    const loom_cfg_graph_t* graph, uint16_t block_index,
    loom_op_t** pred_branches) {
  const loom_block_t* block = graph->blocks[block_index].block;
  loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    const loom_block_t* predecessor =
        graph->blocks[predecessors.values[i]].block;
    loom_op_t* terminator = ((loom_block_t*)predecessor)->last_op;
    if (!terminator || !loom_cfg_br_isa(terminator) ||
        loom_cfg_br_dest(terminator) != block) {
      return false;
    }
    loom_value_slice_t args = loom_cfg_br_args(terminator);
    if (args.count != block->arg_count) return false;
    pred_branches[i] = terminator;
  }
  return true;
}

static bool loom_cfg_simplify_incoming_slots_match(
    loom_op_t* const* pred_branches, iree_host_size_t predecessor_count,
    uint16_t lhs_index, uint16_t rhs_index) {
  for (iree_host_size_t i = 0; i < predecessor_count; ++i) {
    loom_value_slice_t args = loom_cfg_br_args(pred_branches[i]);
    if (args.values[lhs_index] != args.values[rhs_index]) return false;
  }
  return true;
}

static bool loom_cfg_simplify_find_duplicate_arg_replacement(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* const* pred_branches, iree_host_size_t predecessor_count,
    uint16_t arg_index, loom_value_id_t* out_replacement) {
  loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
  for (uint16_t candidate_index = 0; candidate_index < arg_index;
       ++candidate_index) {
    loom_value_id_t candidate = loom_block_arg_id(block, candidate_index);
    if (!loom_cfg_simplify_incoming_slots_match(
            pred_branches, predecessor_count, arg_index, candidate_index)) {
      continue;
    }
    if (!loom_cfg_simplify_type_allows_replacement(state->module, old_arg,
                                                   candidate)) {
      continue;
    }
    *out_replacement = candidate;
    return true;
  }
  return false;
}

static bool loom_cfg_simplify_find_forwarded_arg_replacement(
    const loom_cfg_simplify_state_t* state, const loom_block_t* block,
    loom_op_t* const* pred_branches, iree_host_size_t predecessor_count,
    uint16_t arg_index, loom_value_id_t* out_replacement) {
  if (predecessor_count == 0) return false;
  loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t i = 0; i < predecessor_count; ++i) {
    loom_value_id_t incoming =
        loom_cfg_br_args(pred_branches[i]).values[arg_index];
    if (incoming == old_arg) continue;
    if (replacement == LOOM_VALUE_ID_INVALID) {
      replacement = incoming;
      continue;
    }
    if (incoming != replacement) {
      return false;
    }
  }
  if (replacement == LOOM_VALUE_ID_INVALID) return false;
  if (!loom_cfg_simplify_type_allows_replacement(state->module, old_arg,
                                                 replacement)) {
    return false;
  }

  const loom_op_t* anchor = block->first_op ? block->first_op : block->last_op;
  if (!anchor) return false;
  if (!loom_value_is_available_before_op(state->dominance, replacement,
                                         anchor) ||
      !loom_value_type_is_available_before_op(state->dominance, replacement,
                                              anchor)) {
    return false;
  }
  *out_replacement = replacement;
  return true;
}

static iree_status_t loom_cfg_simplify_rebuild_br_without_arg(
    loom_cfg_simplify_state_t* state, loom_op_t* br, uint16_t removed_index) {
  loom_value_slice_t args = loom_cfg_br_args(br);
  if (removed_index >= args.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "removed branch argument index is out of range");
  }
  loom_value_id_t* kept_args = NULL;
  iree_host_size_t kept_count = args.count - 1;
  if (kept_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->analysis_arena, kept_count,
                                  sizeof(*kept_args), (void**)&kept_args));
  }
  iree_host_size_t kept_index = 0;
  for (uint16_t i = 0; i < args.count; ++i) {
    if (i == removed_index) continue;
    kept_args[kept_index++] = args.values[i];
  }
  return loom_cfg_simplify_replace_br(state, br, loom_cfg_br_dest(br),
                                      kept_args, kept_count);
}

static iree_status_t loom_cfg_simplify_remove_block_arg(
    loom_cfg_simplify_state_t* state, loom_block_t* block,
    loom_op_t** pred_branches, iree_host_size_t predecessor_count,
    uint16_t arg_index, loom_value_id_t replacement) {
  loom_value_id_t old_arg = loom_block_arg_id(block, arg_index);
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
      state->rewriter, old_arg, replacement));
  for (iree_host_size_t i = 0; i < predecessor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_cfg_simplify_rebuild_br_without_arg(
        state, pred_branches[i], arg_index));
  }
  IREE_RETURN_IF_ERROR(loom_block_remove_arg(state->module, block, arg_index));
  if (state->pass->statistics) {
    loom_pass_statistic_add(state->pass,
                            LOOM_CFG_SIMPLIFY_STAT_BLOCK_ARGS_REMOVED, 1);
  }
  state->rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

static iree_status_t loom_cfg_simplify_remove_redundant_block_args(
    loom_cfg_simplify_state_t* state, const loom_cfg_graph_t* graph,
    bool* out_changed) {
  if (graph->malformed) return iree_ok_status();
  for (uint16_t block_index = 1; block_index < graph->block_count;
       ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(graph, block_index)) continue;
    loom_block_t* block = (loom_block_t*)graph->blocks[block_index].block;
    if (block->arg_count == 0) continue;

    loom_cfg_block_index_span_t predecessors =
        loom_cfg_graph_predecessors(graph, block_index);
    if (predecessors.count == 0) continue;
    loom_op_t** pred_branches = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->analysis_arena, predecessors.count, sizeof(*pred_branches),
        (void**)&pred_branches));
    if (!loom_cfg_simplify_pred_branches_to_block(graph, block_index,
                                                  pred_branches)) {
      continue;
    }

    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
      bool found_replacement =
          loom_cfg_simplify_find_duplicate_arg_replacement(
              state, block, pred_branches, predecessors.count, arg_index,
              &replacement) ||
          loom_cfg_simplify_find_forwarded_arg_replacement(
              state, block, pred_branches, predecessors.count, arg_index,
              &replacement);
      if (!found_replacement) continue;
      IREE_RETURN_IF_ERROR(loom_cfg_simplify_remove_block_arg(
          state, block, pred_branches, predecessors.count, arg_index,
          replacement));
      *out_changed = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

static iree_status_t loom_cfg_simplify_process_cfg_region(
    loom_cfg_simplify_state_t* state, loom_region_t* region,
    bool* out_changed) {
  loom_cfg_graph_t graph = {0};
  IREE_RETURN_IF_ERROR(
      loom_cfg_graph_build(region, state->analysis_arena, &graph));
  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_remove_unreachable_blocks(state, &graph, out_changed));
  if (*out_changed) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      loom_cfg_simplify_forward_trivial_blocks(state, &graph, out_changed));
  if (*out_changed) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_fuse_single_predecessor_blocks(
      state, &graph, out_changed));
  if (*out_changed) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_merge_duplicate_terminal_blocks(
      state, &graph, out_changed));
  if (*out_changed) return iree_ok_status();
  return loom_cfg_simplify_remove_redundant_block_args(state, &graph,
                                                       out_changed);
}

static iree_status_t loom_cfg_simplify_process_function_once(
    loom_cfg_simplify_state_t* state, loom_func_like_t function,
    bool* out_changed) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  state->region_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_cfg_simplify_region_stack_push(
      state->pass->arena, &state->region_stack, body));
  while (true) {
    loom_region_t* region =
        loom_cfg_simplify_region_stack_pop(&state->region_stack);
    if (!region) break;

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      IREE_RETURN_IF_ERROR(
          loom_cfg_simplify_fold_block_branches(state, block, out_changed));
      if (*out_changed) return iree_ok_status();
    }

    if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
      IREE_RETURN_IF_ERROR(
          loom_cfg_simplify_process_cfg_region(state, region, out_changed));
      if (*out_changed) return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t loom_cfg_simplify_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));

  iree_arena_allocator_t analysis_arena = {0};
  iree_arena_initialize(pass->arena->block_pool, &analysis_arena);

  loom_cfg_simplify_state_t state = {
      .pass = pass,
      .module = module,
      .rewriter = &rewriter,
      .analysis_arena = &analysis_arena,
  };

  iree_status_t status = loom_cfg_simplify_region_stack_initialize(
      pass->arena, &state.region_stack);
  bool changed = true;
  while (iree_status_is_ok(status) && changed) {
    changed = false;
    iree_arena_reset(&analysis_arena);

    loom_value_fact_table_t fact_table = {0};
    status = loom_value_fact_table_initialize(&fact_table, &analysis_arena,
                                              module->values.count);
    if (!iree_status_is_ok(status)) break;
    status = loom_value_fact_table_compute(&fact_table, module, function);
    if (!iree_status_is_ok(status)) break;

    status = loom_cfg_simplify_mark_cfg_regions(loom_func_like_body(function),
                                                &analysis_arena);
    if (!iree_status_is_ok(status)) break;

    loom_dominance_info_t dominance = {0};
    status =
        loom_dominance_info_initialize(module, &analysis_arena, &dominance);
    if (!iree_status_is_ok(status)) break;

    state.fact_table = &fact_table;
    state.dominance = &dominance;
    status =
        loom_cfg_simplify_process_function_once(&state, function, &changed);
  }

  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&analysis_arena);
  return status;
}
