// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/graph.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/schedule/diagnostics.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"

typedef struct loom_low_schedule_effect_frontier_t {
  // Latest ordered effect node that every later dependency effect must follow.
  uint32_t ordered_node;
  // Latest write node by normalized memory-space bucket.
  uint32_t last_write_nodes[LOOM_LOW_SCHEDULE_EFFECT_MEMORY_SPACE_COUNT];
  // Latest write summary by normalized memory-space bucket.
  loom_low_memory_access_summary_t
      last_write_summaries[LOOM_LOW_SCHEDULE_EFFECT_MEMORY_SPACE_COUNT];
  // Outstanding read nodes not yet subsumed by a later write or ordered effect.
  uint32_t* read_nodes;
  // Access summary for each outstanding read node.
  loom_low_memory_access_summary_t* read_summaries;
  // Number of outstanding read entries.
  iree_host_size_t read_count;
} loom_low_schedule_effect_frontier_t;

static bool loom_low_schedule_op_is_descriptor_packet(const loom_op_t* op) {
  return loom_low_op_isa(op) || loom_low_const_isa(op);
}

static bool loom_low_schedule_op_is_terminator(const loom_module_t* module,
                                               const loom_op_t* op) {
  return iree_any_bit_set(loom_op_effective_traits(module, op),
                          LOOM_TRAIT_TERMINATOR);
}

static bool loom_low_schedule_node_has_effects(
    const loom_low_schedule_node_t* node,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor) {
    return descriptor->effect_count != 0 ||
           iree_any_bit_set(descriptor->flags,
                            LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                                LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR);
  }
  return iree_any_bit_set(node->traits, LOOM_TRAIT_READS_MEMORY |
                                            LOOM_TRAIT_WRITES_MEMORY |
                                            LOOM_TRAIT_NON_DETERMINISTIC |
                                            LOOM_TRAIT_UNKNOWN_EFFECTS);
}

static iree_status_t loom_low_schedule_resolve_descriptor(
    loom_low_schedule_build_state_t* state, const loom_op_t* op,
    loom_low_schedule_node_t* node,
    const loom_low_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  if (!loom_low_schedule_op_is_descriptor_packet(op)) {
    return iree_ok_status();
  }

  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
      state->module, &state->target, op, &packet));
  if (packet.descriptor == NULL) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_missing_descriptor(state, op, packet.key));
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low schedule descriptor '%.*s' is not available",
                            (int)packet.key.size, packet.key.data);
  }

  node->descriptor = packet.descriptor;
  node->effect_count = packet.descriptor->effect_count;
  node->schedule_class_id = packet.descriptor->schedule_class_id;
  const loom_low_schedule_class_t* schedule_class =
      &state->target.descriptor_set
           ->schedule_classes[packet.descriptor->schedule_class_id];
  node->latency_cycles = schedule_class->latency_cycles;
  node->latency_kind = schedule_class->latency_kind;
  node->model_quality = schedule_class->model_quality;
  node->issue_use_count = schedule_class->issue_use_count;
  node->hazard_count = schedule_class->hazard_count;
  node->schedule_class_name = loom_low_descriptor_set_string(
      state->target.descriptor_set, schedule_class->name_string_offset);
  *out_descriptor = packet.descriptor;
  return iree_ok_status();
}

static int loom_low_schedule_compare_memory_access_position(
    const loom_low_memory_access_position_t* position, uint16_t block_index,
    uint64_t block_ordinal) {
  const loom_low_memory_access_position_t key = {
      .block_index = block_index,
      .block_ordinal = block_ordinal,
  };
  return loom_low_memory_access_position_compare_order(position, &key);
}

static void loom_low_schedule_bind_memory_access_record(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t block_index, const loom_op_t* op) {
  while (state->memory_access_record_bind_index <
         state->memory_access_record_count) {
    const loom_low_memory_access_record_t* record =
        &state->memory_access_records[state->memory_access_record_bind_index];
    const int compare = loom_low_schedule_compare_memory_access_position(
        &record->position, block_index, op->block_ordinal);
    if (compare > 0) {
      return;
    }
    if (compare == 0) {
      state->nodes[node_index].memory_access_record_index =
          (uint32_t)state->memory_access_record_bind_index++;
      return;
    }
    ++state->memory_access_record_bind_index;
  }
}

static bool loom_low_schedule_dependency_equal(
    const loom_low_schedule_dependency_t* dependency, uint32_t producer_node,
    uint32_t consumer_node, loom_low_schedule_dependency_kind_t kind,
    uint32_t operand_index) {
  return dependency->producer_node == producer_node &&
         dependency->consumer_node == consumer_node &&
         dependency->kind == kind && dependency->operand_index == operand_index;
}

static iree_status_t loom_low_schedule_add_dependency(
    loom_low_schedule_build_state_t* state, uint32_t producer_node,
    uint32_t consumer_node, loom_low_schedule_dependency_kind_t kind,
    uint32_t operand_index) {
  if (producer_node == consumer_node) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < state->dependency_count; ++i) {
    if (loom_low_schedule_dependency_equal(&state->dependencies[i],
                                           producer_node, consumer_node, kind,
                                           operand_index)) {
      return iree_ok_status();
    }
  }
  if (state->dependency_count >= state->dependency_capacity) {
    iree_host_size_t new_capacity =
        state->dependency_capacity == 0 ? 16 : state->dependency_capacity * 2;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(state->arena, state->dependency_count,
                              new_capacity, sizeof(*state->dependencies),
                              &new_capacity, (void**)&state->dependencies));
    state->dependency_capacity = new_capacity;
  }
  state->dependencies[state->dependency_count++] =
      (loom_low_schedule_dependency_t){
          .producer_node = producer_node,
          .consumer_node = consumer_node,
          .kind = kind,
          .operand_index = operand_index,
      };
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_descriptor_operand_reg_class_id(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t descriptor_operand_index,
    uint16_t* out_reg_class_id) {
  *out_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  const uint32_t operand_row =
      descriptor->operand_start + descriptor_operand_index;
  if (operand_row >= descriptor_set->operand_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low schedule descriptor state operand row is out of range");
  }
  const loom_low_operand_t* operand = &descriptor_set->operands[operand_row];
  if (operand->reg_class_alt_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule state operand must have one register-class alternative");
  }
  const uint32_t alt_index = operand->reg_class_alt_start;
  if (alt_index >= descriptor_set->reg_class_alt_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low schedule state operand register-class alternative is out of "
        "range");
  }
  const loom_low_reg_class_alt_t* alt =
      &descriptor_set->reg_class_alts[alt_index];
  if (iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE) ||
      alt->reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      alt->reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule state operand must name a concrete register class");
  }
  *out_reg_class_id = alt->reg_class_id;
  return iree_ok_status();
}

static bool loom_low_schedule_reg_class_is_state(
    const loom_low_schedule_build_state_t* state, uint16_t reg_class_id) {
  return reg_class_id < state->target.descriptor_set->reg_class_count &&
         state->reg_class_state_flags != NULL &&
         state->reg_class_state_flags[reg_class_id] != 0;
}

static iree_status_t loom_low_schedule_note_state_read(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t reg_class_id) {
  if (!loom_low_schedule_reg_class_is_state(state, reg_class_id)) {
    return iree_ok_status();
  }
  const uint32_t last_write = state->state_last_write_nodes[reg_class_id];
  if (last_write != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, last_write, node_index, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
        UINT32_MAX));
  }
  const uint32_t last_read = state->state_last_read_nodes[reg_class_id];
  if (last_read != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, last_read, node_index, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
        UINT32_MAX));
  }
  state->state_last_read_nodes[reg_class_id] = node_index;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_state_write(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t reg_class_id) {
  if (!loom_low_schedule_reg_class_is_state(state, reg_class_id)) {
    return iree_ok_status();
  }
  const uint32_t last_write = state->state_last_write_nodes[reg_class_id];
  if (last_write != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, last_write, node_index, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
        UINT32_MAX));
  }
  const uint32_t last_read = state->state_last_read_nodes[reg_class_id];
  if (last_read != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, last_read, node_index, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
        UINT32_MAX));
  }
  state->state_last_write_nodes[reg_class_id] = node_index;
  state->state_last_read_nodes[reg_class_id] = LOOM_LOW_SCHEDULE_NODE_NONE;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_verify_state_value_not_clobbered(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    loom_value_ordinal_t value_ordinal, uint16_t reg_class_id) {
  const loom_low_schedule_value_record_t* value = &state->values[value_ordinal];
  const uint32_t producer_node = value->producer_node;
  if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
      state->nodes[producer_node].block != state->nodes[node_index].block) {
    return iree_ok_status();
  }
  const uint32_t last_write = state->state_last_write_nodes[reg_class_id];
  if (last_write == LOOM_LOW_SCHEDULE_NODE_NONE ||
      last_write == producer_node) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "low schedule state value %" PRIu32
                          " is clobbered before use by node %" PRIu32,
                          value->value_id, last_write);
}

static iree_status_t loom_low_schedule_note_state_value_read(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    loom_value_ordinal_t value_ordinal) {
  const uint16_t reg_class_id = state->values[value_ordinal].register_class_id;
  if (!loom_low_schedule_reg_class_is_state(state, reg_class_id)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_verify_state_value_not_clobbered(
      state, node_index, value_ordinal, reg_class_id));
  return loom_low_schedule_note_state_read(state, node_index, reg_class_id);
}

static iree_status_t loom_low_schedule_note_descriptor_state_accesses(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor == NULL) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const uint32_t operand_row = descriptor->operand_start + i;
    const loom_low_operand_t* operand = &descriptor_set->operands[operand_row];
    const uint16_t state_flags =
        operand->flags &
        (LOOM_LOW_OPERAND_FLAG_STATE_READ | LOOM_LOW_OPERAND_FLAG_STATE_WRITE);
    if (state_flags == 0) {
      continue;
    }
    uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    IREE_RETURN_IF_ERROR(loom_low_schedule_descriptor_operand_reg_class_id(
        descriptor_set, descriptor, i, &reg_class_id));
    if (iree_any_bit_set(state_flags, LOOM_LOW_OPERAND_FLAG_STATE_READ)) {
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_note_state_read(state, node_index, reg_class_id));
    }
    if (iree_any_bit_set(state_flags, LOOM_LOW_OPERAND_FLAG_STATE_WRITE)) {
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_note_state_write(state, node_index, reg_class_id));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_initialize_node_value_ordinals(
    loom_low_schedule_build_state_t* state, loom_low_schedule_node_t* node) {
  const loom_op_t* op = node->op;
  node->operand_count = op->operand_count;
  node->result_count = op->result_count;
  const uint32_t total_value_count =
      (uint32_t)op->operand_count + (uint32_t)op->result_count;
  if (total_value_count >
      LOOM_LOW_SCHEDULE_NODE_INLINE_VALUE_ORDINAL_CAPACITY) {
    node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_VALUE_ORDINALS_OVERFLOW;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, total_value_count,
        sizeof(*node->value_ordinals.overflow_value_ordinals),
        (void**)&node->value_ordinals.overflow_value_ordinals));
  }
  loom_value_ordinal_t* value_ordinals =
      loom_low_schedule_node_value_ordinals(node);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    value_ordinals[i] =
        loom_local_value_domain_ordinal(state->value_domain, operands[i]);
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  loom_value_ordinal_t* result_ordinals = value_ordinals + op->operand_count;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    result_ordinals[i] =
        loom_local_value_domain_ordinal(state->value_domain, results[i]);
  }
  return iree_ok_status();
}

iree_status_t loom_low_schedule_fill_nodes(
    loom_low_schedule_build_state_t* state) {
  uint32_t next_node_index = 0;
  for (uint16_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    loom_block_t* block = state->body->blocks[block_index];
    if (!block) {
      continue;
    }
    state->blocks[block_index] = (loom_low_schedule_block_t){
        .block = block,
        .node_start = next_node_index,
        .node_count = block->op_count,
        .scheduled_node_start = next_node_index,
        .scheduled_node_count = block->op_count,
    };

    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_low_schedule_node_t* node = &state->nodes[next_node_index];
      *node = (loom_low_schedule_node_t){
          .op = op,
          .block = block,
          .block_index = block_index,
          .source_ordinal = next_node_index,
          .scheduled_ordinal = LOOM_LOW_SCHEDULE_NODE_NONE,
          .kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL,
          .traits = loom_op_effective_traits(state->module, op),
          .descriptor = NULL,
          .memory_access_record_index =
              LOOM_LOW_SCHEDULE_MEMORY_ACCESS_RECORD_NONE,
          .schedule_class_id = LOOM_LOW_SCHEDULE_CLASS_NONE,
      };
      if (loom_low_schedule_op_is_terminator(state->module, op)) {
        node->kind = LOOM_LOW_SCHEDULE_NODE_TERMINATOR;
      } else if (loom_low_schedule_op_is_descriptor_packet(op)) {
        node->kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
      }

      const loom_low_descriptor_t* descriptor = NULL;
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_resolve_descriptor(state, op, node, &descriptor));
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_initialize_node_value_ordinals(state, node));
      loom_low_schedule_bind_memory_access_record(state, next_node_index,
                                                  block_index, op);

      const loom_value_ordinal_t* result_ordinals =
          loom_low_schedule_node_const_result_ordinals(node);
      for (uint16_t result_index = 0; result_index < node->result_count;
           ++result_index) {
        state->values[result_ordinals[result_index]].producer_node =
            next_node_index;
      }
      ++next_node_index;
    }
  }
  return iree_ok_status();
}

static void loom_low_schedule_effect_frontier_reset(
    loom_low_schedule_effect_frontier_t* frontier, uint32_t ordered_node) {
  frontier->ordered_node = ordered_node;
  frontier->read_count = 0;
  memset(frontier->last_write_nodes, 0xFF, sizeof(frontier->last_write_nodes));
  memset(frontier->last_write_summaries, 0,
         sizeof(frontier->last_write_summaries));
}

static void loom_low_schedule_effect_frontier_initialize(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* out_frontier) {
  *out_frontier = (loom_low_schedule_effect_frontier_t){
      .read_nodes = state->effect_read_nodes,
      .read_summaries = state->effect_read_summaries,
  };
  loom_low_schedule_effect_frontier_reset(out_frontier,
                                          LOOM_LOW_SCHEDULE_NODE_NONE);
}

static iree_status_t loom_low_schedule_effect_frontier_depend_on_ordered(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index) {
  if (frontier->ordered_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return iree_ok_status();
  }
  return loom_low_schedule_add_dependency(
      state, frontier->ordered_node, node_index,
      LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT32_MAX);
}

static iree_status_t loom_low_schedule_effect_frontier_note_read(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_memory_access_summary_t* summary) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_depend_on_ordered(
      state, frontier, node_index));
  for (uint32_t i = 0; i < LOOM_LOW_SCHEDULE_EFFECT_MEMORY_SPACE_COUNT; ++i) {
    const uint32_t write_node = frontier->last_write_nodes[i];
    if (write_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
      continue;
    }
    if (!loom_low_memory_access_summaries_may_alias(
            summary, &frontier->last_write_summaries[i])) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, write_node, node_index, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT,
        UINT32_MAX));
  }
  if (frontier->read_count >= state->effect_read_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "low schedule exceeded precomputed effect-frontier read capacity");
  }
  frontier->read_nodes[frontier->read_count] = node_index;
  frontier->read_summaries[frontier->read_count] = *summary;
  ++frontier->read_count;
  return iree_ok_status();
}

static void loom_low_schedule_effect_frontier_note_write_complete(
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_memory_access_summary_t* summary) {
  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0; read_index < frontier->read_count;
       ++read_index) {
    const loom_low_memory_access_summary_t* read_summary =
        &frontier->read_summaries[read_index];
    if (loom_low_memory_access_write_subsumes_read(summary, read_summary)) {
      continue;
    }
    frontier->read_nodes[write_index] = frontier->read_nodes[read_index];
    frontier->read_summaries[write_index] = *read_summary;
    ++write_index;
  }
  frontier->read_count = write_index;
  if (summary->memory_space == LOOM_LOW_MEMORY_SPACE_GENERIC) {
    memset(frontier->last_write_nodes, 0xFF,
           sizeof(frontier->last_write_nodes));
    memset(frontier->last_write_summaries, 0,
           sizeof(frontier->last_write_summaries));
  }
  frontier->last_write_nodes[(uint32_t)summary->memory_space] = node_index;
  frontier->last_write_summaries[(uint32_t)summary->memory_space] = *summary;
}

static iree_status_t loom_low_schedule_effect_frontier_note_write(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_memory_access_summary_t* summary) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_depend_on_ordered(
      state, frontier, node_index));
  for (uint32_t i = 0; i < LOOM_LOW_SCHEDULE_EFFECT_MEMORY_SPACE_COUNT; ++i) {
    const uint32_t write_node = frontier->last_write_nodes[i];
    if (write_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
      continue;
    }
    if (!loom_low_memory_access_summaries_may_alias(
            summary, &frontier->last_write_summaries[i])) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, write_node, node_index, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT,
        UINT32_MAX));
  }
  for (iree_host_size_t i = 0; i < frontier->read_count; ++i) {
    if (!loom_low_memory_access_summaries_may_alias(
            summary, &frontier->read_summaries[i])) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, frontier->read_nodes[i], node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT32_MAX));
  }
  loom_low_schedule_effect_frontier_note_write_complete(frontier, node_index,
                                                        summary);
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_effect_frontier_note_ordered(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_depend_on_ordered(
      state, frontier, node_index));
  for (uint32_t i = 0; i < LOOM_LOW_SCHEDULE_EFFECT_MEMORY_SPACE_COUNT; ++i) {
    const uint32_t write_node = frontier->last_write_nodes[i];
    if (write_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, write_node, node_index, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT,
        UINT32_MAX));
  }
  for (iree_host_size_t i = 0; i < frontier->read_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, frontier->read_nodes[i], node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT32_MAX));
  }
  loom_low_schedule_effect_frontier_reset(frontier, node_index);
  return iree_ok_status();
}

static bool loom_low_schedule_effect_is_ordered(
    const loom_low_effect_t* effect) {
  if (iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_ORDERED)) {
    return true;
  }
  switch (effect->kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
    case LOOM_LOW_EFFECT_KIND_WRITE:
      return false;
    case LOOM_LOW_EFFECT_KIND_UNKNOWN:
    case LOOM_LOW_EFFECT_KIND_CALL:
    case LOOM_LOW_EFFECT_KIND_BARRIER:
    case LOOM_LOW_EFFECT_KIND_COUNTER:
    case LOOM_LOW_EFFECT_KIND_CONVERGENT:
    case LOOM_LOW_EFFECT_KIND_CONTROL:
    default:
      return true;
  }
}

static uint16_t loom_low_schedule_descriptor_dependency_memory_effect_count(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  uint16_t count = 0;
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (!iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY)) {
      continue;
    }
    if (effect->kind == LOOM_LOW_EFFECT_KIND_READ ||
        effect->kind == LOOM_LOW_EFFECT_KIND_WRITE) {
      ++count;
    }
  }
  return count;
}

static const loom_low_memory_access_summary_t*
loom_low_schedule_lookup_memory_access_summary(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    const loom_low_descriptor_t* descriptor) {
  const uint32_t record_index =
      state->nodes[node_index].memory_access_record_index;
  if (record_index == LOOM_LOW_SCHEDULE_MEMORY_ACCESS_RECORD_NONE) {
    return NULL;
  }
  if (record_index >= state->memory_access_record_count) {
    return NULL;
  }
  if (loom_low_schedule_descriptor_dependency_memory_effect_count(
          state->target.descriptor_set, descriptor) != 1) {
    return NULL;
  }
  const loom_low_memory_access_record_t* record =
      &state->memory_access_records[record_index];
  return &record->summary;
}

static iree_status_t loom_low_schedule_note_descriptor_effects(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_descriptor_t* descriptor) {
  const bool side_effecting = iree_any_bit_set(
      descriptor->flags, LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                             LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR);
  if (descriptor->effect_count == 0) {
    return side_effecting ? loom_low_schedule_effect_frontier_note_ordered(
                                state, frontier, node_index)
                          : iree_ok_status();
  }
  const loom_low_memory_access_summary_t* source_summary =
      loom_low_schedule_lookup_memory_access_summary(state, node_index,
                                                     descriptor);
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (loom_low_schedule_effect_is_ordered(effect)) {
      return loom_low_schedule_effect_frontier_note_ordered(state, frontier,
                                                            node_index);
    }
  }
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (!iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY)) {
      continue;
    }
    switch (effect->kind) {
      case LOOM_LOW_EFFECT_KIND_READ: {
        loom_low_memory_access_summary_t summary =
            loom_low_memory_access_summary_from_effect(effect);
        if (source_summary != NULL) {
          summary = *source_summary;
        }
        IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_note_read(
            state, frontier, node_index, &summary));
        break;
      }
      case LOOM_LOW_EFFECT_KIND_WRITE: {
        loom_low_memory_access_summary_t summary =
            loom_low_memory_access_summary_from_effect(effect);
        if (source_summary != NULL) {
          summary = *source_summary;
        }
        IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_note_write(
            state, frontier, node_index, &summary));
        break;
      }
      default:
        return loom_low_schedule_effect_frontier_note_ordered(state, frontier,
                                                              node_index);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_structural_effects(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_NON_DETERMINISTIC |
                                         LOOM_TRAIT_UNKNOWN_EFFECTS |
                                         LOOM_TRAIT_CONVERGENT)) {
    return loom_low_schedule_effect_frontier_note_ordered(state, frontier,
                                                          node_index);
  }
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_WRITES_MEMORY)) {
    loom_low_memory_access_summary_t summary =
        loom_low_memory_access_summary_synthetic(LOOM_LOW_MEMORY_SPACE_GENERIC);
    return loom_low_schedule_effect_frontier_note_write(state, frontier,
                                                        node_index, &summary);
  }
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_READS_MEMORY)) {
    loom_low_memory_access_summary_t summary =
        loom_low_memory_access_summary_synthetic(LOOM_LOW_MEMORY_SPACE_GENERIC);
    return loom_low_schedule_effect_frontier_note_read(state, frontier,
                                                       node_index, &summary);
  }
  return iree_ok_status();
}

iree_status_t loom_low_schedule_build_dependencies(
    loom_low_schedule_build_state_t* state) {
  for (iree_host_size_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block_record = &state->blocks[block_index];
    loom_low_schedule_effect_frontier_t effect_frontier;
    loom_low_schedule_effect_frontier_initialize(state, &effect_frontier);
    uint32_t last_live_in_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    bool live_in_preamble_open = true;
    if (state->target.descriptor_set->reg_class_count != 0) {
      memset(state->state_last_write_nodes, 0xFF,
             state->target.descriptor_set->reg_class_count *
                 sizeof(*state->state_last_write_nodes));
      memset(state->state_last_read_nodes, 0xFF,
             state->target.descriptor_set->reg_class_count *
                 sizeof(*state->state_last_read_nodes));
    }
    const uint32_t block_node_end =
        block_record->node_start + block_record->node_count;
    for (uint32_t node_index = block_record->node_start;
         node_index < block_node_end; ++node_index) {
      const loom_low_schedule_node_t* node = &state->nodes[node_index];
      const loom_op_t* op = node->op;
      if (loom_low_live_in_isa(op)) {
        if (block_index != 0 || !live_in_preamble_open) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "low schedule requires low.live_in packets in the entry "
              "preamble");
        }
        if (last_live_in_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
              state, last_live_in_node, node_index,
              LOOM_LOW_SCHEDULE_DEPENDENCY_ANCHOR, UINT32_MAX));
        }
        last_live_in_node = node_index;
      } else {
        live_in_preamble_open = false;
        if (last_live_in_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
              state, last_live_in_node, node_index,
              LOOM_LOW_SCHEDULE_DEPENDENCY_ANCHOR, UINT32_MAX));
        }
      }

      const loom_value_ordinal_t* operand_ordinals =
          loom_low_schedule_node_const_operand_ordinals(node);
      for (uint16_t operand_index = 0; operand_index < node->operand_count;
           ++operand_index) {
        const loom_value_ordinal_t operand_ordinal =
            operand_ordinals[operand_index];
        const uint32_t producer_node =
            state->values[operand_ordinal].producer_node;
        if (producer_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
            state->nodes[producer_node].block == node->block) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
              state, producer_node, node_index,
              LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, operand_index));
        }
        IREE_RETURN_IF_ERROR(loom_low_schedule_note_state_value_read(
            state, node_index, operand_ordinal));
      }

      const loom_low_descriptor_t* descriptor = node->descriptor;
      IREE_RETURN_IF_ERROR(loom_low_schedule_note_descriptor_state_accesses(
          state, node_index, descriptor));
      if (descriptor != NULL) {
        IREE_RETURN_IF_ERROR(loom_low_schedule_note_descriptor_effects(
            state, &effect_frontier, node_index, descriptor));
      } else if (loom_low_schedule_node_has_effects(node, NULL)) {
        IREE_RETURN_IF_ERROR(loom_low_schedule_note_structural_effects(
            state, &effect_frontier, node_index));
      }

      if (node->kind == LOOM_LOW_SCHEDULE_NODE_TERMINATOR) {
        for (uint32_t predecessor_node = block_record->node_start;
             predecessor_node < node_index; ++predecessor_node) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
              state, predecessor_node, node_index,
              LOOM_LOW_SCHEDULE_DEPENDENCY_CONTROL, UINT32_MAX));
        }
      }
    }
  }
  return iree_ok_status();
}
