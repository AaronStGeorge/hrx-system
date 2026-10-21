// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/fact_propagation.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_cfg.h"
#include "loom/util/fact_loop.h"

bool loom_rewriter_op_summarizes_nested_regions(
    const loom_op_vtable_t* vtable) {
  return vtable && (vtable->loop_like || vtable->region_branch);
}

static bool loom_rewriter_summary_region_is_ready(
    const loom_op_t* op, const loom_op_vtable_t* vtable, uint8_t region_index) {
  if (region_index >= op->region_count) {
    return false;
  }
  const loom_region_descriptor_t* descriptor =
      loom_op_vtable_region_descriptor(vtable, region_index);
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!descriptor || !region || region->block_count != 1) {
    return false;
  }
  const loom_block_t* block = loom_region_const_entry_block(region);
  const loom_op_t* terminator = block->last_op;
  if (!terminator) {
    return false;
  }
  return loom_region_descriptor_matches_terminator(descriptor,
                                                   terminator->kind);
}

bool loom_rewriter_nested_region_summary_is_ready(
    const loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_op_vtable_t* vtable) {
  if (!vtable) {
    return true;
  }
  if (vtable->loop_like) {
    if (!loom_rewriter_summary_region_is_ready(
            op, vtable, vtable->loop_like->body_region_index)) {
      return false;
    }
    const uint8_t condition_region_index =
        vtable->loop_like->condition_region_index;
    if (condition_region_index != LOOM_REGION_INDEX_NONE &&
        !loom_rewriter_summary_region_is_ready(op, vtable,
                                               condition_region_index)) {
      return false;
    }
  }
  if (vtable->region_branch) {
    loom_region_branch_t branch = loom_region_branch_cast(rewriter->module, op);
    for (uint8_t region_index = 0; region_index < op->region_count;
         ++region_index) {
      if (!loom_region_branch_region(rewriter->module, branch, region_index)) {
        continue;
      }
      if (!loom_rewriter_summary_region_is_ready(op, vtable, region_index)) {
        return false;
      }
    }
  }
  return true;
}

// Recomputes direct result facts. Callers own result-user and parent-summary
// scheduling because semantic mutations can matter even when direct facts
// remain equal.
iree_status_t loom_rewriter_recompute_op_facts(loom_rewriter_t* rewriter,
                                               loom_op_t* op) {
  if (!rewriter->fact_table) {
    return iree_ok_status();
  }

  bool facts_changed = false;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_op_and_report(
      rewriter->fact_table, rewriter->module, op, &facts_changed));
  if (!facts_changed) {
    return iree_ok_status();
  }

  rewriter->flags |= LOOM_REWRITER_FLAG_FACTS_CHANGED;
  return iree_ok_status();
}

// Region owners publish inherited context before their children are built or
// revisited. Cyclic summaries refresh it again when control facts change.
iree_status_t loom_rewriter_seed_nested_execution_facts(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  if (!rewriter->fact_table || !op->region_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_nested_target_scopes(
      rewriter->fact_table, rewriter->module, op));
  const loom_value_facts_t scope = loom_value_fact_table_block_temporal_scope(
      rewriter->fact_table, op->parent_block);
  const bool may_repeat =
      loom_loop_like_isa(loom_loop_like_cast(rewriter->module, op)) ||
      loom_value_fact_table_block_may_repeat(rewriter->fact_table,
                                             op->parent_block);
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_set_region_temporal_scope(
          rewriter->fact_table, regions[i], scope, may_repeat));
      if (regions[i]->block_count) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_loop_iv_arg(
            rewriter->fact_table, rewriter->module,
            loom_region_entry_block(regions[i]), op));
      }
    }
  }
  return iree_ok_status();
}

struct loom_rewriter_cfg_region_t {
  // Region represented by the published snapshot.
  const loom_region_t* region;
  // Owned resettable snapshot storage.
  iree_arena_allocator_t arena;
  // Next entry in the address hash bucket.
  loom_rewriter_cfg_region_t* next_bucket;
  // Next entry in the complete list for cleanup.
  loom_rewriter_cfg_region_t* next_entry;
};

static iree_host_size_t loom_rewriter_cfg_region_hash(
    const loom_region_t* region) {
  uintptr_t bits = (uintptr_t)region;
  bits ^= bits >> 17;
  bits *= (uintptr_t)0xed5ad4bbU;
  return bits ^ (bits >> 11);
}

void loom_rewriter_release_cfg_facts(loom_rewriter_t* rewriter) {
  for (loom_rewriter_cfg_region_t* entry = rewriter->cfg_facts.entries; entry;
       entry = entry->next_entry) {
    loom_value_fact_table_forget_cfg_region(rewriter->fact_table,
                                            entry->region);
    iree_arena_deinitialize(&entry->arena);
  }
  memset(&rewriter->cfg_facts, 0, sizeof(rewriter->cfg_facts));
}

static iree_status_t loom_rewriter_cfg_region_storage(
    loom_rewriter_t* rewriter, const loom_region_t* region,
    loom_rewriter_cfg_region_t** out_entry) {
  if (rewriter->cfg_facts.bucket_count) {
    iree_host_size_t bucket = loom_rewriter_cfg_region_hash(region) &
                              (rewriter->cfg_facts.bucket_count - 1);
    for (loom_rewriter_cfg_region_t* entry =
             rewriter->cfg_facts.buckets[bucket];
         entry; entry = entry->next_bucket) {
      if (entry->region == region) {
        *out_entry = entry;
        return iree_ok_status();
      }
    }
  }
  if (rewriter->cfg_facts.count >= rewriter->cfg_facts.bucket_count / 2) {
    iree_host_size_t bucket_count = rewriter->cfg_facts.bucket_count
                                        ? rewriter->cfg_facts.bucket_count * 2
                                        : 8;
    loom_rewriter_cfg_region_t** buckets = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, bucket_count, sizeof(*buckets), (void**)&buckets));
    memset(buckets, 0, bucket_count * sizeof(*buckets));
    for (loom_rewriter_cfg_region_t* entry = rewriter->cfg_facts.entries; entry;
         entry = entry->next_entry) {
      iree_host_size_t bucket =
          loom_rewriter_cfg_region_hash(entry->region) & (bucket_count - 1);
      entry->next_bucket = buckets[bucket];
      buckets[bucket] = entry;
    }
    rewriter->cfg_facts.buckets = buckets;
    rewriter->cfg_facts.bucket_count = bucket_count;
  }
  loom_rewriter_cfg_region_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(rewriter->arena, sizeof(*entry), (void**)&entry));
  memset(entry, 0, sizeof(*entry));
  entry->region = region;
  iree_arena_initialize(rewriter->arena->block_pool, &entry->arena);
  iree_host_size_t bucket = loom_rewriter_cfg_region_hash(region) &
                            (rewriter->cfg_facts.bucket_count - 1);
  entry->next_bucket = rewriter->cfg_facts.buckets[bucket];
  rewriter->cfg_facts.buckets[bucket] = entry;
  entry->next_entry = rewriter->cfg_facts.entries;
  rewriter->cfg_facts.entries = entry;
  ++rewriter->cfg_facts.count;
  *out_entry = entry;
  return iree_ok_status();
}

void loom_rewriter_attach_value_facts(loom_rewriter_t* rewriter,
                                      loom_value_fact_table_t* facts) {
  IREE_ASSERT(!facts || !facts->has_conditioned_results);
  loom_rewriter_release_cfg_facts(rewriter);
  rewriter->fact_table = facts;
}

static iree_status_t loom_rewriter_cfg_argument_changed(
    void* user_data, loom_value_id_t value_id) {
  loom_rewriter_t* rewriter = user_data;
  rewriter->flags |= LOOM_REWRITER_FLAG_FACTS_CHANGED;
  return loom_rewriter_add_users_to_worklist(rewriter, value_id, /*flags=*/0);
}

static iree_status_t loom_rewriter_publish_cfg_control_changes(
    loom_rewriter_t* rewriter, const loom_value_fact_cfg_region_t* structure) {
  uint16_t block_index = 0;
  while (loom_value_fact_control_take_changed_block(structure->control,
                                                    &block_index)) {
    loom_op_t* terminator = structure->graph.blocks[block_index].block->last_op;
    if (!terminator || !terminator->successor_count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, terminator));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_summary_ops_to_worklist(rewriter, terminator));
  }
  return iree_ok_status();
}

static iree_status_t loom_rewriter_update_cfg_block_facts(
    loom_rewriter_t* rewriter, const loom_value_fact_cfg_region_t* structure,
    uint16_t block_index) {
  if (structure->control_flow.components.count != 0 &&
      loom_cfg_graph_block_is_reachable(&structure->graph, block_index)) {
    const loom_scc_t* component =
        &structure->control_flow.components
             .values[structure->graph.blocks[block_index].component];
    if (component->is_cycle) {
      iree_arena_allocator_t scratch_arena;
      iree_arena_initialize(rewriter->arena->block_pool, &scratch_arena);
      iree_status_t status = loom_value_fact_table_recompute_cfg_component(
          rewriter->fact_table, rewriter->module, structure, component,
          &scratch_arena, loom_rewriter_cfg_argument_changed, rewriter);
      iree_arena_deinitialize(&scratch_arena);
      if (iree_status_is_ok(status)) {
        status = loom_rewriter_publish_cfg_control_changes(rewriter, structure);
      }
      iree_bitmap_reset(structure->control_flow.dirty,
                        structure->graph.blocks[block_index].component);
      return status;
    }
  }
  if (loom_cfg_graph_block_is_reachable(&structure->graph, block_index)) {
    iree_bitmap_reset(structure->control_flow.dirty,
                      structure->graph.blocks[block_index].component);
  }
  return loom_value_fact_table_update_cfg_block_args(
      rewriter->fact_table, rewriter->module, structure, block_index,
      loom_rewriter_cfg_argument_changed, rewriter);
}

// Compare retained edge identities and reachability, not the mutable fields of
// their IR terminators. Operand edits are separately scheduled by the rewriter.
static bool loom_rewriter_cfg_predecessors_equal(
    const loom_value_fact_cfg_region_t* old_structure, uint16_t old_index,
    const loom_value_fact_cfg_region_t* new_structure, uint16_t new_index) {
  const loom_cfg_graph_t* old_graph = &old_structure->graph;
  const loom_cfg_graph_t* new_graph = &new_structure->graph;
  loom_cfg_edge_index_span_t old_edges =
      loom_cfg_graph_predecessor_edges(old_graph, old_index);
  loom_cfg_edge_index_span_t new_edges =
      loom_cfg_graph_predecessor_edges(new_graph, new_index);
  iree_host_size_t old_position = 0;
  iree_host_size_t new_position = 0;
  while (true) {
    while (old_position < old_edges.count &&
           !old_graph
                ->blocks[old_graph->edges[old_edges.values[old_position]]
                             .source_block_index]
                .reachable) {
      ++old_position;
    }
    while (new_position < new_edges.count &&
           !new_graph
                ->blocks[new_graph->edges[new_edges.values[new_position]]
                             .source_block_index]
                .reachable) {
      ++new_position;
    }
    if (old_position == old_edges.count || new_position == new_edges.count) {
      return old_position == old_edges.count && new_position == new_edges.count;
    }
    const loom_cfg_edge_info_t* old_edge =
        &old_graph->edges[old_edges.values[old_position++]];
    const loom_cfg_edge_info_t* new_edge =
        &new_graph->edges[new_edges.values[new_position++]];
    if (old_edge->terminator != new_edge->terminator ||
        old_edge->successor_index != new_edge->successor_index ||
        old_edge->selector_value_id != new_edge->selector_value_id ||
        old_graph->blocks[old_edge->source_block_index].block !=
            new_graph->blocks[new_edge->source_block_index].block ||
        loom_value_fact_control_execution(old_structure->control,
                                          old_edge->source_block_index)
                .flags !=
            loom_value_fact_control_execution(new_structure->control,
                                              new_edge->source_block_index)
                .flags) {
      return false;
    }
  }
}

static iree_status_t loom_rewriter_refresh_cfg_block_facts(
    loom_rewriter_t* rewriter,
    const loom_value_fact_cfg_region_t* old_structure,
    const loom_value_fact_cfg_region_t* structure,
    iree_arena_allocator_t* arena) {
  iree_host_size_t* old_indices = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, structure->graph.block_count,
                                sizeof(*old_indices), (void**)&old_indices));
  memset(old_indices, 0xFF,
         structure->graph.block_count * sizeof(*old_indices));
  if (old_structure) {
    for (iree_host_size_t i = 0; i < old_structure->graph.block_count; ++i) {
      iree_host_size_t new_index = loom_cfg_graph_block_index(
          &structure->graph, old_structure->graph.blocks[i].block);
      if (new_index != IREE_HOST_SIZE_MAX) {
        old_indices[new_index] = i;
      }
    }
  }
  bool* updated_components = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, structure->control_flow.components.count,
      sizeof(*updated_components), (void**)&updated_components));
  memset(
      updated_components, 0,
      structure->control_flow.components.count * sizeof(*updated_components));
  for (uint16_t i = 0; i < structure->graph.block_count; ++i) {
    if (!structure->graph.blocks[i].reachable) {
      continue;
    }
    if (!structure->graph.blocks[i].component_is_cyclic &&
        old_indices[i] != IREE_HOST_SIZE_MAX &&
        old_structure->graph.blocks[old_indices[i]].component_is_cyclic) {
      // Breaking a cycle removes a temporal constraint without necessarily
      // changing any block argument. Refresh direct and nested observers.
      loom_op_t* op = NULL;
      loom_block_for_each_op((loom_block_t*)structure->graph.blocks[i].block,
                             op) {
        IREE_RETURN_IF_ERROR(loom_rewriter_record_subtree(rewriter, op));
      }
    }
    if (structure->control_flow.components.count) {
      iree_host_size_t component_index = structure->graph.blocks[i].component;
      const loom_scc_t* component =
          &structure->control_flow.components.values[component_index];
      if (component->is_cycle) {
        if (updated_components[component_index]) {
          continue;
        }
        bool was_dirty = false;
        if (old_indices[i] != IREE_HOST_SIZE_MAX &&
            old_structure->control_flow.components.count &&
            old_structure->graph.blocks[old_indices[i]].reachable) {
          was_dirty = iree_bitmap_test(
              old_structure->control_flow.dirty,
              old_structure->graph.blocks[old_indices[i]].component);
        }
        if (!was_dirty && old_indices[i] != IREE_HOST_SIZE_MAX &&
            loom_rewriter_cfg_predecessors_equal(old_structure, old_indices[i],
                                                 structure, i)) {
          continue;
        }
        updated_components[component_index] = true;
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_rewriter_update_cfg_block_facts(rewriter, structure, i));
  }
  return loom_rewriter_publish_cfg_control_changes(rewriter, structure);
}

iree_status_t loom_rewriter_refresh_cfg_facts(loom_rewriter_t* rewriter,
                                              loom_region_t* region) {
  if (!rewriter->fact_table) {
    return iree_ok_status();
  }
  loom_rewriter_cfg_region_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_cfg_region_storage(rewriter, region, &storage));
  const loom_value_fact_cfg_region_t* old_structure =
      loom_value_fact_table_lookup_cfg_region(rewriter->fact_table, region);
  iree_arena_allocator_t next_arena;
  iree_arena_initialize(rewriter->arena->block_pool, &next_arena);
  loom_value_fact_cfg_region_t* structure = NULL;
  iree_status_t status =
      iree_arena_allocate(&next_arena, sizeof(*structure), (void**)&structure);
  if (iree_status_is_ok(status)) {
    status = loom_value_fact_cfg_region_initialize(rewriter->module, region,
                                                   &next_arena, structure);
  }
  if (iree_status_is_ok(status)) {
    status = loom_value_fact_table_set_cfg_region(
        rewriter->fact_table, rewriter->module, region, structure);
  }
  if (iree_status_is_ok(status)) {
    iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(&next_arena);
    status = loom_rewriter_refresh_cfg_block_facts(rewriter, old_structure,
                                                   structure, &next_arena);
    iree_arena_checkpoint_restore(&checkpoint);
    iree_arena_deinitialize(&storage->arena);
    storage->arena = next_arena;
  } else {
    iree_arena_deinitialize(&next_arena);
  }
  return status;
}

iree_status_t loom_rewriter_update_successor_facts(loom_rewriter_t* rewriter,
                                                   loom_op_t* op) {
  if (!op->successor_count) {
    return iree_ok_status();
  }
  const loom_value_fact_cfg_region_t* structure = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_get_or_build_cfg_region(
      rewriter->fact_table, rewriter->module, op->parent_block->parent_region,
      &structure));
  loom_value_fact_cfg_update_control(rewriter->fact_table, structure,
                                     op->parent_block->region_index);
  IREE_RETURN_IF_ERROR(
      loom_rewriter_publish_cfg_control_changes(rewriter, structure));
  const uint16_t block_index = op->parent_block->region_index;
  if (structure->graph.blocks[block_index].component_is_cyclic) {
    const iree_host_size_t component_index =
        structure->graph.blocks[block_index].component;
    if (structure->control_flow.anchors[component_index] == op &&
        iree_bitmap_test(structure->control_flow.dirty, component_index)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_update_cfg_block_facts(
          rewriter, structure, block_index));
    }
  }
  if (op->successor_count != 1) {
    return iree_ok_status();
  }
  loom_block_t* successor = loom_op_successors(op)[0];
  if (!successor->arg_count) {
    return iree_ok_status();
  }
  const loom_cfg_block_info_t* successor_info =
      &structure->graph.blocks[successor->region_index];
  if (!successor_info->reachable ||
      !iree_bitmap_test(structure->control_flow.dirty,
                        successor_info->component)) {
    return iree_ok_status();
  }
  return loom_rewriter_update_cfg_block_facts(rewriter, structure,
                                              successor->region_index);
}

iree_status_t loom_rewriter_enable_region_analysis(
    loom_rewriter_t* rewriter, loom_func_like_t function, loom_region_t* region,
    loom_op_t* parent_op, loom_value_fact_table_t* facts) {
  loom_rewriter_attach_value_facts(rewriter, facts);
  return loom_value_fact_table_compute_region(
      rewriter->fact_table, rewriter->module, function, region, parent_op);
}

iree_status_t loom_rewriter_enable_analysis(loom_rewriter_t* rewriter,
                                            loom_func_like_t function,
                                            loom_value_fact_table_t* facts) {
  return loom_rewriter_enable_region_analysis(
      rewriter, function, loom_func_like_body(function), function.op, facts);
}

static iree_status_t loom_rewriter_add_cfg_summary_to_worklist(
    loom_rewriter_t* rewriter, const loom_block_t* block,
    bool arguments_changed) {
  if (!rewriter->fact_table || !block || !block->parent_region) {
    return iree_ok_status();
  }
  const loom_value_fact_cfg_region_t* structure =
      loom_value_fact_table_lookup_cfg_region(rewriter->fact_table,
                                              block->parent_region);
  if (!structure) {
    return iree_ok_status();
  }
  iree_host_size_t block_index =
      loom_cfg_graph_block_index(&structure->graph, block);
  if (block_index == IREE_HOST_SIZE_MAX ||
      !structure->graph.blocks[block_index].reachable) {
    return iree_ok_status();
  }
  iree_host_size_t component_index =
      structure->graph.blocks[block_index].component;
  if (!structure->graph.blocks[block_index].component_is_cyclic) {
    // Incoming terminators are already queued by the mutation or control
    // publication. Their first visit rejoins all inputs; the rest reuse it.
    if (arguments_changed && block->arg_count) {
      iree_bitmap_set(structure->control_flow.dirty, component_index);
    }
    return iree_ok_status();
  }
  loom_op_t* anchor = structure->control_flow.anchors[component_index];
  if (anchor) {
    iree_bitmap_set(structure->control_flow.dirty, component_index);
    IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, anchor));
  }
  return iree_ok_status();
}

// Branch operand identities define the forwarding graph independently of the
// facts carried by those operands. Structural CFG edits publish a new snapshot;
// payload edits only invalidate this successor's retained partition.
void loom_rewriter_invalidate_cfg_forwarding(loom_rewriter_t* rewriter,
                                             const loom_op_t* op) {
  if (!rewriter->fact_table || op->successor_count != 1) {
    return;
  }
  const loom_block_t* successor = loom_op_successors(op)[0];
  if (!successor->arg_count) {
    return;
  }
  const loom_value_fact_cfg_region_t* structure =
      loom_value_fact_table_lookup_cfg_region(rewriter->fact_table,
                                              successor->parent_region);
  if (!structure || !structure->control_flow.components.count) {
    return;
  }
  iree_host_size_t block_index =
      loom_cfg_graph_block_index(&structure->graph, successor);
  if (block_index == IREE_HOST_SIZE_MAX ||
      !structure->graph.blocks[block_index].component_is_cyclic) {
    return;
  }
  structure->control_flow
      .forwarding[structure->graph.blocks[block_index].component]
      .dirty = true;
}

// Structured parents own the summaries of their regions. Inserting or erasing
// an unused definition does not change an equation in its own CFG component.
iree_status_t loom_rewriter_add_parent_summary_ops_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  if (!rewriter->worklist_chunk && !rewriter->fact_table) {
    return iree_ok_status();
  }
  bool has_cfg_facts =
      rewriter->fact_table && rewriter->fact_table->regions.cfg_count;
  for (loom_op_t* parent = op ? op->parent_op : NULL; parent;
       parent = parent->parent_op) {
    if (has_cfg_facts) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_cfg_summary_to_worklist(
          rewriter, parent->parent_block, /*arguments_changed=*/false));
    }
    const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, parent);
    if (loom_rewriter_op_summarizes_nested_regions(vtable)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, parent));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_rewriter_add_summary_ops_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  if (op && rewriter->fact_table && rewriter->fact_table->regions.cfg_count) {
    const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, op);
    // Opaque operations define facts from their result types, independently of
    // their operands. Operand rewrites still revisit the operation, but cannot
    // change a cyclic fact equation through it. Terminators feed block
    // arguments or structured summaries even without an inference callback.
    if ((vtable && vtable->infer_facts) ||
        loom_rewriter_op_summarizes_nested_regions(vtable) ||
        iree_any_bit_set(op->traits, LOOM_TRAIT_TERMINATOR)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_cfg_summary_to_worklist(
          rewriter, op->parent_block, /*arguments_changed=*/false));
    }
    for (uint8_t i = 0; i < op->successor_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_cfg_summary_to_worklist(
          rewriter, loom_op_successors(op)[i], /*arguments_changed=*/true));
    }
  }
  return loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op);
}

// Attribute membership includes TYPE and predicate dependencies even when the
// provider is absent from the owner's ordinary operands.
static iree_status_t loom_rewriter_add_attribute_users_to_worklist(
    loom_rewriter_t* rewriter, loom_value_id_t value_id) {
  loom_type_use_iterator_t users;
  loom_attribute_users_begin(&rewriter->module->type_uses, value_id, &users);
  for (loom_attribute_user_t user = loom_attribute_users_next(&users); user.op;
       user = loom_attribute_users_next(&users)) {
    IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, user.op));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_summary_ops_to_worklist(rewriter, user.op));
  }
  return iree_ok_status();
}

// Adds operand, attribute, and value-type users to the worklist.
iree_status_t loom_rewriter_add_users_to_worklist(
    loom_rewriter_t* rewriter, loom_value_id_t value_id,
    loom_rewriter_user_change_flags_t flags) {
  if (!rewriter->worklist_chunk && !rewriter->fact_table) {
    return iree_ok_status();
  }
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (iree_any_bit_set(flags, LOOM_REWRITER_USER_CHANGE_FLAG_REPLACED)) {
      loom_rewriter_invalidate_cfg_forwarding(rewriter, user_op);
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, user_op));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_summary_ops_to_worklist(rewriter, user_op));
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_attribute_users_to_worklist(rewriter, value_id));
  loom_type_use_iterator_t type_users;
  loom_module_value_type_users(rewriter->module, value_id, &type_users);
  for (loom_value_id_t user_value_id = loom_type_users_next(&type_users);
       user_value_id != LOOM_VALUE_ID_INVALID;
       user_value_id = loom_type_users_next(&type_users)) {
    loom_value_t* user_value =
        loom_module_value(rewriter->module, user_value_id);
    if (!loom_value_is_block_arg(user_value)) {
      loom_op_t* def = loom_value_def_op(user_value);
      if (def) {
        IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, def));
        IREE_RETURN_IF_ERROR(
            loom_rewriter_add_summary_ops_to_worklist(rewriter, def));
      }
    }
    const loom_use_t* user_value_uses = loom_value_uses(user_value);
    for (uint32_t i = 0; i < user_value->use_count; ++i) {
      loom_op_t* user_op = loom_use_user_op(user_value_uses[i]);
      IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, user_op));
      IREE_RETURN_IF_ERROR(
          loom_rewriter_add_summary_ops_to_worklist(rewriter, user_op));
    }
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_attribute_users_to_worklist(rewriter, user_value_id));
  }
  return iree_ok_status();
}

// Adds users of all results of |op| to the worklist.
iree_status_t loom_rewriter_add_result_users_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_users_to_worklist(
          rewriter, results[i], /*flags=*/0));
    }
  }
  return iree_ok_status();
}
