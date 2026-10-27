// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/control_carriers.h"

#include <string.h>

#include "iree/base/bitmap.h"
#include "loom/analysis/liveness_dataflow.h"
#include "loom/codegen/low/lower/bindings.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/realization.h"
#include "loom/codegen/low/lower/source_value_storage.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/low/ops.h"

typedef struct loom_low_lower_control_carrier_block_t {
  // Source values in region-local ordinal order.
  loom_value_id_t* source_values;
  // Canonical bindings saved while this block's overrides are active.
  loom_value_id_t* saved_values;
  // Number of populated entries in source_values and saved_values.
  uint16_t count;
  // Allocated source_values capacity.
  iree_host_size_t source_capacity;
  // Allocated saved_values capacity.
  iree_host_size_t saved_capacity;
  // First supplemental carrier argument on the emitted Low block.
  uint16_t argument_offset;
  // True after supplemental Low block arguments have been created.
  bool mapped;
  // True while this block's source bindings are active.
  bool active;
} loom_low_lower_control_carrier_block_t;

typedef struct loom_low_lower_control_carrier_edge_t {
  // Argument-free block reached by the source conditional branch.
  loom_low_lower_block_ref_t adapter;
  // Previous destination receiving the supplemental carrier tuple.
  loom_low_lower_block_ref_t destination;
} loom_low_lower_control_carrier_edge_t;

typedef struct loom_low_lower_control_carrier_source_t {
  // Conditional-edge forwarding records in successor order.
  loom_low_lower_control_carrier_edge_t* edges;
  // Number of populated edge records.
  iree_host_size_t edge_count;
  // Allocated edge record capacity.
  iree_host_size_t edge_capacity;
} loom_low_lower_control_carrier_source_t;

typedef struct loom_low_lower_control_carrier_pending_edge_t {
  // Source terminator whose active carrier bindings dominate the edge.
  const loom_op_t* source_terminator;
  // Target-owned retained edge plan updated during finalization.
  loom_low_lower_control_carrier_edge_plan_t* plan;
} loom_low_lower_control_carrier_pending_edge_t;

struct loom_low_lower_control_carriers_t {
  // Per-authored-block supplemental carrier tuples.
  loom_low_lower_control_carrier_block_t* blocks;
  // Per-authored-source conditional-edge forwarding recipes.
  loom_low_lower_control_carrier_source_t* sources;
  // Target-created conditional edges awaiting frozen carrier signatures.
  loom_low_lower_control_carrier_pending_edge_t* pending_edges;
  // Number of populated target-created conditional edges.
  iree_host_size_t pending_edge_count;
  // Allocated target-created conditional edge capacity.
  iree_host_size_t pending_edge_capacity;
  // Canonical source liveness, owned by the planning analysis arena.
  loom_liveness_dataflow_t liveness;
  // True after liveness has been computed for the immutable source CFG.
  bool liveness_available;
  // True after closure construction has ended and edge adapters are frozen.
  bool finalized;
};

static iree_status_t loom_low_lower_control_carriers_append_edge(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    loom_low_lower_block_ref_t adapter,
    loom_low_lower_block_ref_t destination) {
  loom_low_lower_control_carriers_t* state =
      context->lowering->control_carriers;
  loom_low_lower_control_carrier_source_t* source =
      &state->sources[source_terminator->parent_block->region_index];
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->function_arena, source->edge_count, source->edge_count + 1,
      sizeof(*source->edges), &source->edge_capacity, (void**)&source->edges));
  source->edges[source->edge_count++] = (loom_low_lower_control_carrier_edge_t){
      .adapter = adapter,
      .destination = destination,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_lower_control_carriers_require_state(
    loom_low_lower_context_t* context,
    loom_low_lower_control_carriers_t** out_state) {
  *out_state = context->lowering->control_carriers;
  if (*out_state != NULL) {
    return iree_ok_status();
  }
  loom_low_lower_control_carriers_t* state = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*state), (void**)&state));
  *state = (loom_low_lower_control_carriers_t){0};
  context->lowering->control_carriers = state;
  *out_state = state;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_control_carriers_require_tables(
    loom_low_lower_context_t* context,
    loom_low_lower_control_carriers_t* state) {
  if (state->blocks != NULL) {
    return iree_ok_status();
  }
  const uint16_t block_count =
      loom_func_like_body(context->source_function)->block_count;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, block_count, sizeof(*state->blocks), (void**)&state->blocks));
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, block_count, sizeof(*state->sources), (void**)&state->sources));
  memset(state->blocks, 0, block_count * sizeof(*state->blocks));
  memset(state->sources, 0, block_count * sizeof(*state->sources));
  return iree_ok_status();
}

static iree_status_t loom_low_lower_control_carriers_require_liveness(
    loom_low_lower_context_t* context,
    loom_low_lower_control_carriers_t* state) {
  if (state->liveness_available) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_low_lower_control_carriers_require_tables(context, state));
  const loom_value_fact_cfg_region_t* cfg = loom_low_lower_context_cfg(context);
  IREE_RETURN_IF_ERROR(loom_liveness_dataflow_analyze(
      &context->lowering->value_domain, &cfg->graph, LOOM_LIVENESS_VALUES_ALL,
      &context->analysis_arena, &state->liveness));
  state->liveness_available = true;
  return iree_ok_status();
}

iree_status_t loom_low_lower_control_plan_carrier_conditional_edge(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    loom_low_lower_block_ref_t destination,
    loom_low_lower_control_carrier_edge_plan_t* out_plan) {
  IREE_ASSERT(source_terminator != NULL);
  IREE_ASSERT_NE(destination, 0);
  IREE_ASSERT(out_plan != NULL);
  loom_low_lower_control_carriers_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_control_carriers_require_state(context, &state));
  IREE_ASSERT(!state->finalized);
  *out_plan = (loom_low_lower_control_carrier_edge_plan_t){
      .destination = destination,
      .branch_destination = destination,
  };
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->function_arena, state->pending_edge_count,
      state->pending_edge_count + 1, sizeof(*state->pending_edges),
      &state->pending_edge_capacity, (void**)&state->pending_edges));
  state->pending_edges[state->pending_edge_count++] =
      (loom_low_lower_control_carrier_pending_edge_t){
          .source_terminator = source_terminator,
          .plan = out_plan,
      };
  return iree_ok_status();
}

static bool loom_low_lower_control_live_in_contains(
    const loom_low_lower_context_t* context,
    const loom_liveness_block_relation_t* relation,
    loom_value_id_t source_value_id) {
  const loom_value_ordinal_t target =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  iree_host_size_t low = 0;
  iree_host_size_t high = relation->live_in_count;
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    const loom_value_ordinal_t candidate =
        loom_low_lowering_frame_value_ordinal(context->lowering,
                                              relation->live_in_values[middle]);
    if (candidate < target) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low < relation->live_in_count &&
         relation->live_in_values[low] == source_value_id;
}

static iree_status_t loom_low_lower_control_insert_carrier(
    loom_low_lower_context_t* context, uint16_t block_index,
    loom_value_id_t source_value_id) {
  loom_low_lower_control_carrier_block_t* block =
      &context->lowering->control_carriers->blocks[block_index];
  const loom_value_ordinal_t ordinal =
      loom_low_lowering_frame_value_ordinal(context->lowering, source_value_id);
  uint16_t insertion_index = 0;
  while (insertion_index < block->count) {
    const loom_value_id_t candidate = block->source_values[insertion_index];
    const loom_value_ordinal_t candidate_ordinal =
        loom_low_lowering_frame_value_ordinal(context->lowering, candidate);
    if (candidate_ordinal >= ordinal) {
      if (candidate == source_value_id) {
        return iree_ok_status();
      }
      break;
    }
    ++insertion_index;
  }
  const loom_block_t* source_block = loom_region_const_block(
      loom_func_like_body(context->source_function), block_index);
  const uint16_t authored_count =
      block_index == 0 ? context->lowering->boundary.argument_count
                       : source_block->arg_count;
  const uint16_t realization_count =
      loom_low_lower_realization_block_argument_count(context, source_block);
  if ((uint32_t)authored_count + realization_count + block->count >=
      UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "control block argument count exceeds u16");
  }
  const uint16_t required_count = (uint16_t)(block->count + 1);
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->function_arena, block->count, required_count,
      sizeof(*block->source_values), &block->source_capacity,
      (void**)&block->source_values));
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->function_arena, block->count, required_count,
      sizeof(*block->saved_values), &block->saved_capacity,
      (void**)&block->saved_values));
  const iree_host_size_t trailing_count = block->count - insertion_index;
  if (trailing_count != 0) {
    memmove(block->source_values + insertion_index + 1,
            block->source_values + insertion_index,
            trailing_count * sizeof(*block->source_values));
    memmove(block->saved_values + insertion_index + 1,
            block->saved_values + insertion_index,
            trailing_count * sizeof(*block->saved_values));
  }
  block->source_values[insertion_index] = source_value_id;
  block->saved_values[insertion_index] = LOOM_VALUE_ID_INVALID;
  block->count = required_count;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_control_propagate_carrier(
    loom_low_lower_context_t* context, uint16_t destination_index,
    loom_value_id_t source_value_id) {
  loom_low_lower_control_carriers_t* state =
      context->lowering->control_carriers;
  const loom_cfg_graph_t* graph = &loom_low_lower_context_cfg(context)->graph;
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(context->module->arena.block_pool, &scratch_arena);
  iree_bitmap_t visited = {.bit_count = graph->block_count};
  const iree_host_size_t visited_word_count =
      iree_bitmap_calculate_words(visited.bit_count);
  uint16_t* queue = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&scratch_arena, visited_word_count,
                                sizeof(*visited.words), (void**)&visited.words);
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch_arena, graph->block_count,
                                       sizeof(*queue), (void**)&queue);
  }
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&scratch_arena);
    return status;
  }
  iree_bitmap_reset_all(visited);
  iree_host_size_t read_index = 0;
  iree_host_size_t write_index = 0;
  queue[write_index++] = destination_index;
  iree_bitmap_set(visited, destination_index);
  while (read_index < write_index && iree_status_is_ok(status)) {
    const uint16_t block_index = queue[read_index++];
    const loom_liveness_block_relation_t* relation =
        &state->liveness.blocks[block_index];
    if (!loom_low_lower_control_live_in_contains(context, relation,
                                                 source_value_id)) {
      continue;
    }
    status = loom_low_lower_control_insert_carrier(context, block_index,
                                                   source_value_id);
    const loom_cfg_block_index_span_t successors =
        loom_cfg_graph_successors(graph, block_index);
    for (iree_host_size_t i = 0;
         i < successors.count && iree_status_is_ok(status); ++i) {
      const uint16_t successor = successors.values[i];
      if (!iree_bitmap_test(visited, successor) &&
          loom_low_lower_control_live_in_contains(
              context, &state->liveness.blocks[successor], source_value_id)) {
        iree_bitmap_set(visited, successor);
        queue[write_index++] = successor;
      }
    }
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

static bool loom_low_lower_control_value_dominates_capture_sources(
    const loom_low_lower_context_t* context,
    const loom_value_fact_cfg_region_t* cfg, loom_value_id_t source_value_id,
    const uint16_t* capture_sources, uint16_t capture_source_count) {
  const loom_value_t* value =
      loom_module_value(context->module, source_value_id);
  const loom_block_t* definition_block =
      loom_value_is_block_arg(value) ? loom_value_def_block(value)
                                     : loom_value_def_op(value)->parent_block;
  const iree_host_size_t definition_index =
      loom_cfg_graph_block_index(&cfg->graph, definition_block);
  IREE_ASSERT_NE(definition_index, IREE_HOST_SIZE_MAX);
  for (uint16_t i = 0; i < capture_source_count; ++i) {
    if (!loom_cfg_dominance_block_dominates(
            &cfg->dominance, (uint16_t)definition_index, capture_sources[i])) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_low_lower_control_retain_resume_closure(
    loom_low_lower_context_t* context, const loom_block_t* destination,
    const uint16_t* capture_sources, uint16_t capture_source_count) {
  IREE_ASSERT(context->low_func_op == NULL);
  IREE_ASSERT(destination != NULL);
  IREE_ASSERT_GT(capture_source_count, 0);
  IREE_ASSERT(capture_sources != NULL);
  loom_low_lower_control_carriers_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_control_carriers_require_state(context, &state));
  IREE_ASSERT(!state->finalized);
  IREE_RETURN_IF_ERROR(
      loom_low_lower_control_carriers_require_liveness(context, state));

  const loom_value_fact_cfg_region_t* cfg = loom_low_lower_context_cfg(context);
  const iree_host_size_t destination_index =
      loom_cfg_graph_block_index(&cfg->graph, destination);
  IREE_ASSERT_NE(destination_index, IREE_HOST_SIZE_MAX);
  const loom_liveness_block_relation_t* relation =
      &state->liveness.blocks[destination_index];
  for (iree_host_size_t i = 0; i < relation->live_in_count; ++i) {
    const loom_value_id_t source_value_id = relation->live_in_values[i];
    if (!loom_low_lower_source_value_storage_result_required(context,
                                                             source_value_id) ||
        loom_low_lower_control_value_dominates_capture_sources(
            context, cfg, source_value_id, capture_sources,
            capture_source_count)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_control_propagate_carrier(
        context, (uint16_t)destination_index, source_value_id));
  }
  return iree_ok_status();
}

loom_low_lower_control_carrier_slice_t loom_low_lower_control_source_carriers(
    const loom_low_lower_context_t* context, const loom_block_t* source_block) {
  const loom_low_lower_control_carriers_t* state =
      context->lowering->control_carriers;
  if (state == NULL || state->blocks == NULL || source_block == NULL) {
    return (loom_low_lower_control_carrier_slice_t){0};
  }
  const loom_low_lower_control_carrier_block_t* block =
      &state->blocks[source_block->region_index];
  return (loom_low_lower_control_carrier_slice_t){
      .values = block->source_values,
      .count = block->count,
  };
}

loom_low_lower_control_carrier_slice_t loom_low_lower_control_block_carriers(
    const loom_low_lower_context_t* context, loom_low_lower_block_ref_t block) {
  return loom_low_lower_control_source_carriers(
      context, loom_low_lower_control_signature_source_block(context, block));
}

loom_type_t loom_low_lower_control_carrier_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id) {
  return loom_low_lower_value_binding_type(context, source_value_id);
}

iree_status_t loom_low_lower_control_carriers_finalize(
    loom_low_lower_context_t* context) {
  loom_low_lower_control_carriers_t* state =
      context->lowering->control_carriers;
  if (state == NULL || state->finalized) {
    return iree_ok_status();
  }
  if (state->blocks == NULL) {
    state->finalized = true;
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < state->pending_edge_count; ++i) {
    loom_low_lower_control_carrier_pending_edge_t* pending =
        &state->pending_edges[i];
    loom_low_lower_control_carrier_edge_plan_t* plan = pending->plan;
    const loom_low_lower_control_carrier_slice_t carriers =
        loom_low_lower_control_block_carriers(context, plan->destination);
    if (carriers.count == 0) {
      continue;
    }
    IREE_ASSERT_EQ(loom_low_lower_control_carrier_argument_offset(
                       context, plan->destination),
                   0);
    IREE_ASSERT_EQ(
        loom_low_lower_control_argument_count(context, plan->destination),
        carriers.count);
    IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
        context, /*before=*/0, /*signature=*/0, &plan->branch_destination));
    IREE_RETURN_IF_ERROR(loom_low_lower_control_carriers_append_edge(
        context, pending->source_terminator, plan->branch_destination,
        plan->destination));
  }
  loom_region_t* body = loom_func_like_body(context->source_function);
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_op_t* terminator =
        loom_block_const_last_op(loom_region_block(body, block_index));
    if (terminator == NULL || terminator->successor_count == 0) {
      continue;
    }
    for (uint8_t successor_index = 0;
         successor_index < terminator->successor_count; ++successor_index) {
      const loom_low_lower_block_ref_t destination =
          loom_low_lower_control_successor(context, terminator,
                                           successor_index);
      if (loom_low_lower_control_block_carriers(context, destination).count ==
          0) {
        continue;
      }
      if (loom_cfg_br_isa(terminator)) {
        continue;
      }
      if (!loom_cfg_cond_br_isa(terminator)) {
        return loom_low_lower_emit_branch_constraint(
            context, terminator, IREE_SV("supplemental_control_edge_payload"));
      }
      const loom_low_lower_control_carrier_slice_t carriers =
          loom_low_lower_control_block_carriers(context, destination);
      if (loom_low_lower_control_carrier_argument_offset(context,
                                                         destination) != 0 ||
          loom_low_lower_control_argument_count(context, destination) !=
              carriers.count) {
        return loom_low_lower_emit_branch_constraint(
            context, terminator, IREE_SV("supplemental_control_edge_payload"));
      }
      loom_low_lower_block_ref_t adapter = 0;
      IREE_RETURN_IF_ERROR(loom_low_lower_control_add_block(
          context, /*before=*/0, /*signature=*/0, &adapter));
      loom_low_lower_block_ref_t previous = 0;
      IREE_RETURN_IF_ERROR(loom_low_lower_control_interpose_argless_successor(
          context, terminator, successor_index, adapter, &previous));
      IREE_ASSERT_EQ(previous, destination);

      IREE_RETURN_IF_ERROR(loom_low_lower_control_carriers_append_edge(
          context, terminator, adapter, destination));
    }
  }
  state->liveness = (loom_liveness_dataflow_t){0};
  state->liveness_available = false;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_control_validate_argument_counts(context));
  state->finalized = true;
  return iree_ok_status();
}

iree_status_t loom_low_lower_control_carriers_map_blocks(
    loom_low_lower_context_t* context) {
  loom_low_lower_control_carriers_t* state =
      context->lowering->control_carriers;
  if (state == NULL || state->blocks == NULL) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_func_like_body(context->source_function);
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    loom_low_lower_control_carrier_block_t* carrier_block =
        &state->blocks[block_index];
    if (carrier_block->count == 0) {
      continue;
    }
    loom_block_t* low_block = context->block_map[block_index];
    carrier_block->argument_offset = low_block->arg_count;
    for (uint16_t i = 0; i < carrier_block->count; ++i) {
      const loom_value_id_t source_value_id = carrier_block->source_values[i];
      loom_value_id_t low_argument = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
          &context->builder, low_block,
          loom_low_lower_control_carrier_type(context, source_value_id),
          &low_argument));
      IREE_RETURN_IF_ERROR(loom_low_lower_copy_value_name(
          context, source_value_id, low_argument));
    }
    carrier_block->mapped = true;
  }
  return iree_ok_status();
}

void loom_low_lower_control_carriers_enter_block(
    loom_low_lower_context_t* context, const loom_block_t* source_block) {
  loom_low_lower_control_carriers_t* state =
      context->lowering->control_carriers;
  if (state == NULL || state->blocks == NULL) {
    return;
  }
  loom_low_lower_control_carrier_block_t* carrier_block =
      &state->blocks[source_block->region_index];
  if (carrier_block->count == 0) {
    return;
  }
  IREE_ASSERT(carrier_block->mapped);
  IREE_ASSERT(!carrier_block->active);
  const loom_block_t* low_block =
      context->block_map[source_block->region_index];
  for (uint16_t i = 0; i < carrier_block->count; ++i) {
    carrier_block->saved_values[i] = loom_low_lower_swap_value_binding(
        context, carrier_block->source_values[i],
        loom_block_arg_id(low_block,
                          (uint16_t)(carrier_block->argument_offset + i)));
  }
  carrier_block->active = true;
}

void loom_low_lower_control_carriers_leave_block(
    loom_low_lower_context_t* context, const loom_block_t* source_block) {
  loom_low_lower_control_carriers_t* state =
      context->lowering->control_carriers;
  if (state == NULL || state->blocks == NULL) {
    return;
  }
  loom_low_lower_control_carrier_block_t* carrier_block =
      &state->blocks[source_block->region_index];
  if (carrier_block->count == 0) {
    return;
  }
  IREE_ASSERT(carrier_block->active);
  for (uint16_t i = 0; i < carrier_block->count; ++i) {
    const loom_value_id_t previous = loom_low_lower_swap_value_binding(
        context, carrier_block->source_values[i],
        carrier_block->saved_values[i]);
    IREE_ASSERT_EQ(
        previous,
        loom_block_arg_id(context->block_map[source_block->region_index],
                          (uint16_t)(carrier_block->argument_offset + i)));
    carrier_block->saved_values[i] = LOOM_VALUE_ID_INVALID;
  }
  carrier_block->active = false;
}

iree_status_t loom_low_lower_control_append_carrier_arguments(
    loom_low_lower_context_t* context, loom_low_lower_block_ref_t destination,
    const loom_value_id_t* prefix_values, uint16_t prefix_count,
    loom_value_slice_t* out_values) {
  *out_values = (loom_value_slice_t){0};
  const loom_low_lower_control_carrier_slice_t carriers =
      loom_low_lower_control_block_carriers(context, destination);
  const uint32_t total_count = (uint32_t)prefix_count + carriers.count;
  IREE_ASSERT_LE(total_count, UINT16_MAX);
  if (total_count == 0) {
    return iree_ok_status();
  }
  loom_value_id_t* values = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, total_count, sizeof(*values), (void**)&values));
  if (prefix_count != 0) {
    memcpy(values, prefix_values, prefix_count * sizeof(*values));
  }
  for (uint16_t i = 0; i < carriers.count; ++i) {
    values[prefix_count + i] =
        loom_low_lower_lookup_value(context, carriers.values[i]);
  }
  IREE_ASSERT_EQ(loom_low_lower_control_argument_count(context, destination),
                 total_count);
  *out_values = (loom_value_slice_t){
      .values = values,
      .count = total_count,
  };
  return iree_ok_status();
}

iree_status_t loom_low_lower_control_carriers_emit_edge_forwards(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator) {
  const loom_low_lower_control_carriers_t* state =
      context->lowering->control_carriers;
  if (state == NULL || state->sources == NULL) {
    return iree_ok_status();
  }
  const loom_low_lower_control_carrier_source_t* source =
      &state->sources[source_terminator->parent_block->region_index];
  if (source->edge_count == 0) {
    return iree_ok_status();
  }
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < source->edge_count && iree_status_is_ok(status); ++i) {
    const loom_low_lower_control_carrier_edge_t* edge = &source->edges[i];
    loom_block_t* adapter =
        loom_low_lower_control_block(context, edge->adapter);
    loom_block_t* destination =
        loom_low_lower_control_block(context, edge->destination);
    IREE_ASSERT_EQ(adapter->op_count, 0);
    const loom_low_lower_control_carrier_slice_t carriers =
        loom_low_lower_control_block_carriers(context, edge->destination);
    IREE_ASSERT_EQ(destination->arg_count, carriers.count);
    loom_builder_set_block(builder, adapter);
    loom_value_slice_t args = {0};
    status = loom_low_lower_control_append_carrier_arguments(
        context, edge->destination, /*prefix_values=*/NULL,
        /*prefix_count=*/0, &args);
    if (iree_status_is_ok(status)) {
      loom_op_t* branch_op = NULL;
      status = loom_low_br_build(builder, destination, args.values, args.count,
                                 source_terminator->location, &branch_op);
    }
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}
