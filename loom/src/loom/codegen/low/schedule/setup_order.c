// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/setup_order.h"

#include <string.h>

#include "loom/codegen/low/descriptor_traits.h"
#include "loom/codegen/low/schedule/context.h"
#include "loom/codegen/low/schedule/pressure.h"

iree_status_t loom_low_schedule_setup_order_initialize(
    uint32_t node_count, iree_arena_allocator_t* arena,
    loom_low_schedule_setup_order_t* out_order) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_count, sizeof(*out_order->completion_nodes),
      (void**)&out_order->completion_nodes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_count, sizeof(*out_order->entry_nodes),
      (void**)&out_order->entry_nodes));
  memset(out_order->completion_nodes, 0xFF,
         node_count * sizeof(*out_order->completion_nodes));
  memset(out_order->entry_nodes, 0xFF,
         node_count * sizeof(*out_order->entry_nodes));
  return iree_ok_status();
}

static bool loom_low_schedule_setup_order_has_per_user_placement(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node, bool* out_requires_register) {
  *out_requires_register = false;
  const iree_bitmap_t per_user_placement_values =
      state->options->per_user_placement_values;
  if (node->result_count != 1 || per_user_placement_values.bit_count == 0) {
    return false;
  }
  const loom_value_id_t value_id =
      state->value_domain
          ->value_ids[loom_low_schedule_node_const_result_ordinals(node)[0]];
  if (value_id >= per_user_placement_values.bit_count ||
      !iree_bitmap_test(per_user_placement_values, value_id)) {
    return false;
  }
  *out_requires_register =
      value_id < state->options->required_register_values.bit_count &&
      iree_bitmap_test(state->options->required_register_values, value_id);
  return true;
}

static bool loom_low_schedule_setup_order_is_member(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node, bool order_structural_setup) {
  if (node->kind == LOOM_LOW_SCHEDULE_NODE_STRUCTURAL) {
    if (order_structural_setup) {
      return true;
    }
    bool requires_register = false;
    if (loom_low_schedule_setup_order_has_per_user_placement(
            state, node, &requires_register) &&
        requires_register) {
      return true;
    }
    if (!iree_any_bit_set(node->flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP) ||
        node->result_count != 1) {
      return false;
    }
    // A singleton cannot hold a later setup while its consumer's prerequisites
    // still need that location. Other storage remains available for pressure
    // and issue-slot scheduling instead of acquiring these ordering edges.
    const loom_low_schedule_value_record_t* result =
        &state->values[loom_low_schedule_node_const_result_ordinals(node)[0]];
    if (!loom_low_schedule_value_requires_register(result)) {
      return false;
    }
    const uint16_t domain = loom_low_schedule_unspillable_completion_domain_id(
        state, result->register_class_id);
    return domain != UINT16_MAX && result->unit_count == 1 &&
           state->pressure_limits.unspillable_completion_domains[domain]
                   .capacity == 1;
  }
  if (node->result_count != 1) {
    return false;
  }
  // Input-free rematerializable values remain pressure-scheduled. A required
  // private value cannot be cloned and instead retains the complete target
  // setup chain that produced it.
  bool requires_register = false;
  const bool has_per_user_placement =
      loom_low_schedule_setup_order_has_per_user_placement(state, node,
                                                           &requires_register);
  return has_per_user_placement &&
         (node->operand_count != 0 || requires_register);
}

// Artificial prerequisites must preserve the footprint of storage that
// allocation cannot spill. Keep conversions and contractions available
// to pressure scheduling so those inputs can retire before independent work.
static bool loom_low_schedule_setup_order_preserves_required_pressure(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node) {
  if (node->result_count != 1) {
    return false;
  }
  const loom_low_schedule_value_record_t* result =
      &state->values[loom_low_schedule_node_const_result_ordinals(node)[0]];
  const uint16_t result_pressure_set =
      result->unit_count != 0
          ? loom_low_reg_class_pressure_set_id(
                &state->target.descriptor_set
                     ->reg_classes[result->register_class_id])
          : 0;
  uint64_t operand_units = 0;
  const loom_value_ordinal_t* operands =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t i = 0; i < node->operand_count; ++i) {
    const loom_low_schedule_value_record_t* operand =
        &state->values[operands[i]];
    if (operand->unit_count == 0 ||
        !loom_low_schedule_value_requires_register(operand)) {
      continue;
    }
    if (operand->register_class_id != result->register_class_id &&
        (result_pressure_set == 0 ||
         loom_low_reg_class_pressure_set_id(
             &state->target.descriptor_set
                  ->reg_classes[operand->register_class_id]) !=
             result_pressure_set)) {
      // A singleton may constrain the same storage more tightly than its
      // source class. Retaining the flexible source does not add storage, and
      // the fixed destination still needs its exclusion ordering.
      const loom_low_descriptor_set_t* descriptors =
          state->target.descriptor_set;
      if (result->unit_count != 1 ||
          !iree_all_bits_set(
              descriptors->reg_classes[result->register_class_id].flags,
              LOOM_LOW_REG_CLASS_FLAG_EXPLICIT_PHYSICAL_REGISTERS) ||
          descriptors->reg_classes[result->register_class_id]
                  .allocatable_count != 1 ||
          !loom_low_descriptor_set_find_physical_register_candidate(
              descriptors, operand->register_class_id,
              loom_low_descriptor_set_physical_register_candidate(
                  descriptors, result->register_class_id, 0),
              NULL)) {
        return false;
      }
    }
    operand_units += operand->unit_count;
  }
  if (operand_units == 0) {
    return true;
  }
  return loom_low_schedule_value_requires_register(result) &&
         operand_units <= result->unit_count &&
         !iree_any_bit_set(node->flags,
                           LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER |
                               LOOM_LOW_SCHEDULE_NODE_FLAG_LATE_READS);
}

static bool loom_low_schedule_setup_order_mark_node(
    loom_low_schedule_build_state_t* state, loom_low_schedule_node_t* node) {
  if (!loom_low_schedule_setup_order_preserves_required_pressure(state, node)) {
    return false;
  }
  node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
  state->setup_order.has_members = true;
  return true;
}

static void loom_low_schedule_setup_order_mark_continuation_source(
    loom_low_schedule_build_state_t* state, loom_value_ordinal_t source_ordinal,
    uint32_t producer) {
  loom_low_schedule_node_t* node = &state->nodes[producer];
  if (node->descriptor == NULL ||
      iree_any_bit_set(node->traits, LOOM_TRAIT_OBSERVABLE_EFFECT) ||
      !loom_low_schedule_value_requires_register(
          &state->values[source_ordinal])) {
    return;
  }
  // Replayability permits moving the producer, but does not justify keeping
  // larger inputs alive in another pressure domain. The partial write obeys
  // the same storage-footprint contract as the rest of its setup chain.
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t result_index = 0; result_index < node->result_count;
       ++result_index) {
    if (result_ordinals[result_index] == source_ordinal &&
        loom_low_descriptor_result_can_rematerialize(
            state->target.descriptor_set, node->descriptor, result_index)) {
      (void)loom_low_schedule_setup_order_mark_node(state, node);
      return;
    }
  }
}

// A storage continuation writes one register part while preserving a disjoint
// part supplied by its tied source. Keep pressure-preserving partial writes
// in the setup chain ending at the completed value's sole consumer. Other
// prerequisites close before these writes occupy required physical storage;
// contracting producers remain available to retire their larger inputs.
static void loom_low_schedule_setup_order_classify_storage_continuation(
    loom_low_schedule_build_state_t* state, loom_low_schedule_node_t* node) {
  if (node->descriptor == NULL || node->op->tied_result_count == 0) {
    return;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  const loom_tied_result_t* tied_results = loom_op_tied_results(node->op);
  for (uint16_t i = 0; i < node->op->tied_result_count; ++i) {
    const loom_tied_result_t tied = tied_results[i];
    const loom_low_operand_t* result = loom_low_descriptor_operand(
        descriptor_set, node->descriptor, tied.result_index);
    if (!iree_any_bit_set(result->flags,
                          LOOM_LOW_OPERAND_FLAG_STORAGE_CONTINUATION)) {
      continue;
    }
    const loom_value_ordinal_t source_ordinal =
        operand_ordinals[tied.operand_index];
    const uint32_t producer = state->value_producer_nodes[source_ordinal];
    // Input-free constant recipes remain available to pressure scheduling.
    // Forcing their partial writes behind consumer prerequisites can retain
    // those inputs while the constant is constructed. Explicit private
    // placement still orders these nodes through ordinary setup membership.
    if (node->operand_count == 1 &&
        loom_low_descriptor_result_can_rematerialize(
            descriptor_set, node->descriptor, tied.result_index) &&
        producer != LOOM_LOW_SCHEDULE_NODE_NONE &&
        state->nodes[producer].descriptor != NULL &&
        loom_low_descriptor_layout(descriptor_set,
                                   state->nodes[producer].descriptor)
                ->op_kind == LOOM_LOW_DESCRIPTOR_OP_KIND_CONST) {
      continue;
    }
    if (!loom_low_schedule_setup_order_mark_node(state, node)) {
      continue;
    }
    if (producer != LOOM_LOW_SCHEDULE_NODE_NONE &&
        state->nodes[producer].block_index == node->block_index) {
      loom_low_schedule_setup_order_mark_continuation_source(
          state, source_ordinal, producer);
    }
  }
}

void loom_low_schedule_setup_order_classify_node(
    loom_low_schedule_build_state_t* state, loom_low_schedule_node_t* node,
    bool order_structural_setup) {
  loom_low_schedule_setup_order_classify_storage_continuation(state, node);
  if (loom_low_schedule_setup_order_is_member(state, node,
                                              order_structural_setup)) {
    (void)loom_low_schedule_setup_order_mark_node(state, node);
  }
}

static iree_status_t loom_low_schedule_setup_order_append(
    loom_low_schedule_build_state_t* state, uint32_t producer,
    uint32_t consumer) {
  if (state->dependencies.count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low schedule dependency count exceeds uint32_t index capacity");
  }
  // The semantic graph is complete: these edges only close its retained setup
  // chains and do not contribute another successor observation.
  return loom_low_schedule_dependency_graph_append(
      &state->dependencies,
      (loom_low_schedule_dependency_t){
          .producer_node = producer,
          .consumer_node = consumer,
          .producer_attachment_index = LOOM_LOW_ID_NONE,
          .consumer_attachment_index = LOOM_LOW_ID_NONE,
          .producer_event_id = LOOM_LOW_TIMING_EVENT_NONE,
          .consumer_event_id = LOOM_LOW_TIMING_EVENT_NONE,
          .value_operand_index = LOOM_LOW_ID_NONE,
          .kind = LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER,
      },
      state->dependency_arena);
}

// A private producer P can retain its sole value consumer S even when it
// also precedes Q, if S already precedes Q. The dependency remains in the
// graph with its original timing; only the setup ownership query consumes
// this implication. Grouping incoming edges by Q proves every such relation
// with one stamp lookup, without transitive closure or per-producer searches.
static iree_status_t loom_low_schedule_setup_order_resolve_successors(
    loom_low_schedule_build_state_t* state, uint32_t node_count,
    uint32_t dependency_count) {
  loom_low_schedule_setup_order_t* order = &state->setup_order;
  bool has_ambiguous_member = false;
  for (uint32_t producer = 0; producer < node_count; ++producer) {
    const uint32_t successor = order->entry_nodes[producer];
    if (order->completion_nodes[producer] == producer &&
        successor != LOOM_LOW_SCHEDULE_NODE_NONE && successor != producer &&
        iree_any_bit_set(state->nodes[producer].flags,
                         LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP)) {
      order->completion_nodes[producer] = successor;
      has_ambiguous_member = true;
    }
  }
  if (!has_ambiguous_member) {
    return iree_ok_status();
  }
  uint32_t* incoming_heads = NULL;
  uint32_t* incoming_next = NULL;
  uint32_t* predecessor_stamps = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, node_count, sizeof(*incoming_heads),
      (void**)&incoming_heads));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, dependency_count, sizeof(*incoming_next),
      (void**)&incoming_next));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, node_count, sizeof(*predecessor_stamps),
      (void**)&predecessor_stamps));
  memset(incoming_heads, 0xFF, node_count * sizeof(*incoming_heads));
  memset(predecessor_stamps, 0xFF, node_count * sizeof(*predecessor_stamps));
  for (uint32_t i = 0; i < dependency_count; ++i) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&state->dependencies, i);
    incoming_next[i] = incoming_heads[dependency->consumer_node];
    incoming_heads[dependency->consumer_node] = i;
  }
  for (uint32_t consumer = 0; consumer < node_count; ++consumer) {
    for (uint32_t i = incoming_heads[consumer];
         i != LOOM_LOW_SCHEDULE_NODE_NONE; i = incoming_next[i]) {
      const loom_low_schedule_dependency_t* dependency =
          loom_low_schedule_dependency_graph_at(&state->dependencies, i);
      predecessor_stamps[dependency->producer_node] = consumer;
    }
    for (uint32_t i = incoming_heads[consumer];
         i != LOOM_LOW_SCHEDULE_NODE_NONE; i = incoming_next[i]) {
      const loom_low_schedule_dependency_t* dependency =
          loom_low_schedule_dependency_graph_at(&state->dependencies, i);
      const uint32_t producer = dependency->producer_node;
      const uint32_t successor = order->completion_nodes[producer];
      if (successor != LOOM_LOW_SCHEDULE_NODE_NONE && successor != producer &&
          successor != consumer && predecessor_stamps[successor] != consumer) {
        order->completion_nodes[producer] = producer;
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_schedule_setup_order_finish(
    loom_low_schedule_build_state_t* state, uint32_t node_count) {
  loom_low_schedule_setup_order_t* order = &state->setup_order;
  if (order->completion_nodes == NULL) {
    return iree_ok_status();
  }
  const uint32_t dependency_count = (uint32_t)state->dependencies.count;
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(state->options->flags,
                       LOOM_LOW_SCHEDULE_FLAG_REPAIR_PRIVATE_SETUP)) {
    const iree_arena_checkpoint_t proof_checkpoint =
        iree_arena_checkpoint_save(state->scratch_arena);
    status = loom_low_schedule_setup_order_resolve_successors(state, node_count,
                                                              dependency_count);
    iree_arena_checkpoint_restore(&proof_checkpoint);
    IREE_RETURN_IF_ERROR(status);
  }
  for (uint16_t block_index = 0;
       block_index < state->body->block_count && iree_status_is_ok(status);
       ++block_index) {
    const loom_low_schedule_block_t* block = &state->blocks[block_index];
    uint32_t range_end = block->node_start + block->node_count;
    for (uint32_t i = range_end;
         i > block->node_start && iree_status_is_ok(status); --i) {
      const uint32_t node_index = i - 1;
      const loom_low_schedule_node_t* node = &state->nodes[node_index];
      const uint32_t successor = order->completion_nodes[node_index];
      const uint32_t value_successor = order->entry_nodes[node_index];
      order->completion_nodes[node_index] = node_index;
      order->entry_nodes[node_index] = node_index;
      if (iree_any_bit_set(node->flags,
                           LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY)) {
        range_end = node_index;
        continue;
      }
      // An order-only successor does not consume the produced value. Grouping
      // unrelated readers around their next state writer would retain all of
      // their inputs together instead of completing each private setup.
      if (successor != value_successor || successor <= node_index ||
          successor >= range_end ||
          !iree_any_bit_set(node->flags,
                            LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP)) {
        continue;
      }
      const uint32_t completion = order->completion_nodes[successor];
      const uint32_t next_member = order->entry_nodes[completion];
      order->completion_nodes[node_index] = completion;
      order->entry_nodes[completion] = node_index;
      if (next_member != successor) {
        status = loom_low_schedule_setup_order_append(state, node_index,
                                                      next_member);
      }
    }
  }
  // Emit prerequisites from retained membership. Each original edge adds at
  // most one order edge in this linear pass. The ordinary dependency index
  // owns deduplication and all subsequent scheduling.
  for (uint32_t i = 0; i < dependency_count && iree_status_is_ok(status); ++i) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&state->dependencies, i);
    const uint32_t producer = dependency->producer_node;
    const uint32_t consumer = dependency->consumer_node;
    const uint32_t completion = order->completion_nodes[consumer];
    const uint32_t entry = order->entry_nodes[completion];
    if (entry != consumer && order->completion_nodes[producer] != completion) {
      status = loom_low_schedule_setup_order_append(state, producer, entry);
    }
  }
  return status;
}
