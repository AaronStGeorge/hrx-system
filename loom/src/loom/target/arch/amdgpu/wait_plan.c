// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/wait_plan.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

#define LOOM_AMDGPU_WAIT_COUNTER_COUNT 5

typedef struct loom_amdgpu_wait_node_state_t {
  // Counters observed on WAIT_COUNTER hazard rows for this node.
  uint32_t hazard_counter_mask;
  // Counters drained by explicit counter effects on this node.
  uint32_t explicit_wait_counter_mask;
  // Counters produced by dependency-participating memory reads on this node.
  uint32_t read_counter_mask;
  // Counters produced by dependency-participating memory writes on this node.
  uint32_t write_counter_mask;
  // Write counters whose effects are visible to workgroup-memory barriers.
  uint32_t workgroup_write_counter_mask;
  // Epoch for each counter produced by this node.
  uint32_t produced_counter_epoch[LOOM_AMDGPU_WAIT_COUNTER_COUNT];
  // Read counters whose result was drained before control left the producing
  // block.
  uint32_t drained_after_production_counter_mask;
  // Counters that must be drained before this barrier node executes.
  uint32_t barrier_counter_mask;
  // Workgroup-memory write counters drained before this barrier executes.
  uint32_t workgroup_barrier_counter_mask;
  // Whether the node has a counter effect without a concrete counter id.
  bool has_generic_counter_effect;
  // Whether a memory read effect uses the target's default read counter.
  bool has_default_dependency_read;
  // Whether a memory write effect uses the target's default write counter.
  bool has_default_dependency_write;
  // Whether a workgroup write effect uses the target's default write counter.
  bool has_default_workgroup_write;
} loom_amdgpu_wait_node_state_t;

typedef struct loom_amdgpu_wait_dependency_link_t {
  // Producer node for the SSA dependency.
  uint32_t producer_node;
  // Next dependency link for the same consumer node.
  uint32_t next_link;
  // Counters produced by |producer_node| and needed by this use.
  uint32_t counter_mask;
} loom_amdgpu_wait_dependency_link_t;

typedef struct loom_amdgpu_wait_plan_builder_t {
  // Schedule table being analyzed.
  const loom_low_schedule_table_t* schedule;
  // Arena that owns all output and scratch arrays.
  iree_arena_allocator_t* arena;
  // Per-node counter classification.
  loom_amdgpu_wait_node_state_t* node_states;
  // First relevant SSA dependency link per consumer node.
  uint32_t* first_dependency_link_by_consumer;
  // Relevant SSA dependency links.
  loom_amdgpu_wait_dependency_link_t* dependency_links;
  // Number of populated dependency links.
  iree_host_size_t dependency_link_count;
  // Allocated dependency link capacity.
  iree_host_size_t dependency_link_capacity;
  // Output action rows.
  loom_amdgpu_wait_plan_action_t* actions;
  // Number of populated action rows.
  iree_host_size_t action_count;
  // Allocated action row capacity.
  iree_host_size_t action_capacity;
  // Current epoch per wait counter.
  uint32_t counter_epochs[LOOM_AMDGPU_WAIT_COUNTER_COUNT];
  // Outstanding packet count per wait counter.
  uint32_t outstanding_counts[LOOM_AMDGPU_WAIT_COUNTER_COUNT];
  // Outstanding packet count per wait counter for memory writes.
  uint32_t outstanding_write_counts[LOOM_AMDGPU_WAIT_COUNTER_COUNT];
  // Outstanding packet count per wait counter for workgroup memory writes.
  uint32_t outstanding_workgroup_write_counts[LOOM_AMDGPU_WAIT_COUNTER_COUNT];
} loom_amdgpu_wait_plan_builder_t;

iree_string_view_t loom_amdgpu_wait_counter_name(uint16_t counter_id) {
  switch (counter_id) {
    case LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD:
      return IREE_SV("vmem_load");
    case LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE:
      return IREE_SV("vmem_store");
    case LOOM_AMDGPU_WAIT_COUNTER_LDS:
      return IREE_SV("lds");
    case LOOM_AMDGPU_WAIT_COUNTER_SMEM:
      return IREE_SV("smem");
    case LOOM_AMDGPU_WAIT_COUNTER_ALU:
      return IREE_SV("alu");
    case LOOM_AMDGPU_WAIT_COUNTER_NONE:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_amdgpu_wait_plan_action_kind_name(
    loom_amdgpu_wait_plan_action_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT:
      return IREE_SV("explicit");
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED:
      return IREE_SV("planned");
    case LOOM_AMDGPU_WAIT_PLAN_ACTION_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_amdgpu_wait_plan_reason_name(
    loom_amdgpu_wait_plan_reason_t reason) {
  switch (reason) {
    case LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET:
      return IREE_SV("explicit_packet");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE:
      return IREE_SV("ssa_use");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_BARRIER:
      return IREE_SV("barrier");
    case LOOM_AMDGPU_WAIT_PLAN_REASON_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

iree_status_t loom_amdgpu_wait_counter_mask(uint16_t counter_id,
                                            uint32_t* out_mask) {
  switch (counter_id) {
    case LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD:
      *out_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_LOAD;
      return iree_ok_status();
    case LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE:
      *out_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE;
      return iree_ok_status();
    case LOOM_AMDGPU_WAIT_COUNTER_LDS:
      *out_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_LDS;
      return iree_ok_status();
    case LOOM_AMDGPU_WAIT_COUNTER_SMEM:
      *out_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_SMEM;
      return iree_ok_status();
    case LOOM_AMDGPU_WAIT_COUNTER_ALU:
      *out_mask = LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU;
      return iree_ok_status();
    default:
      *out_mask = 0;
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown AMDGPU wait counter id %" PRIu16,
                              counter_id);
  }
}

static uint16_t loom_amdgpu_wait_counter_id_from_slot(uint32_t slot) {
  return (uint16_t)(slot + 1);
}

static uint32_t loom_amdgpu_wait_counter_mask_from_slot(uint32_t slot) {
  return 1u << slot;
}

static bool loom_amdgpu_wait_effect_is_dependency_memory(
    const loom_low_schedule_effect_use_t* effect_use) {
  if ((effect_use->effect_flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY) == 0) {
    return false;
  }
  switch (effect_use->memory_space) {
    case LOOM_LOW_MEMORY_SPACE_GENERIC:
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_wait_effect_counter_mask(
    const loom_low_schedule_effect_use_t* effect, uint32_t* out_counter_mask) {
  if (effect->counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE) {
    *out_counter_mask = 0;
    return iree_ok_status();
  }
  return loom_amdgpu_wait_counter_mask(effect->counter_id, out_counter_mask);
}

static iree_status_t loom_amdgpu_wait_plan_allocate(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  if (schedule->node_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, schedule->node_count, sizeof(*builder->node_states),
        (void**)&builder->node_states));
    memset(builder->node_states, 0,
           schedule->node_count * sizeof(*builder->node_states));

    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, schedule->node_count,
        sizeof(*builder->first_dependency_link_by_consumer),
        (void**)&builder->first_dependency_link_by_consumer));
    for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
      builder->first_dependency_link_by_consumer[i] =
          LOOM_LOW_SCHEDULE_NODE_NONE;
    }
  }

  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    builder->dependency_link_capacity += schedule->nodes[i].operand_count;
  }
  if (builder->dependency_link_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->dependency_link_capacity,
        sizeof(*builder->dependency_links),
        (void**)&builder->dependency_links));
  }

  iree_host_size_t action_input_capacity = 0;
  if (!iree_host_size_checked_add(builder->dependency_link_capacity,
                                  schedule->effect_use_count,
                                  &action_input_capacity) ||
      !iree_host_size_checked_mul(action_input_capacity,
                                  LOOM_AMDGPU_WAIT_COUNTER_COUNT,
                                  &builder->action_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU wait-plan action capacity overflows");
  }
  if (builder->action_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->action_capacity, sizeof(*builder->actions),
        (void**)&builder->actions));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_append_action(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_t action) {
  if (builder->action_count >= builder->action_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU wait plan exceeded precomputed action capacity");
  }
  builder->actions[builder->action_count++] = action;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_append_dependency_link(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t producer_node,
    uint32_t consumer_node, uint32_t counter_mask) {
  if (builder->dependency_link_count >= builder->dependency_link_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU wait plan exceeded precomputed dependency capacity");
  }
  loom_amdgpu_wait_dependency_link_t* link =
      &builder->dependency_links[builder->dependency_link_count];
  *link = (loom_amdgpu_wait_dependency_link_t){
      .producer_node = producer_node,
      .next_link = builder->first_dependency_link_by_consumer[consumer_node],
      .counter_mask = counter_mask,
  };
  builder->first_dependency_link_by_consumer[consumer_node] =
      (uint32_t)builder->dependency_link_count++;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_classify_hazards(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  for (iree_host_size_t i = 0; i < schedule->hazard_use_count; ++i) {
    const loom_low_schedule_hazard_use_t* hazard = &schedule->hazard_uses[i];
    if (hazard->kind != LOOM_LOW_HAZARD_KIND_WAIT_COUNTER) {
      continue;
    }
    if (hazard->reference_kind != LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU wait-counter hazard on node %" PRIu32
                              " does not reference a counter",
                              hazard->node_index);
    }
    if (hazard->node_index >= schedule->node_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait-counter hazard references node "
                              "%" PRIu32 " but schedule has %zu node(s)",
                              hazard->node_index, schedule->node_count);
    }
    uint32_t counter_mask = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_counter_mask(hazard->reference_id, &counter_mask));
    builder->node_states[hazard->node_index].hazard_counter_mask |=
        counter_mask;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_classify_effects(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  for (iree_host_size_t i = 0; i < schedule->effect_use_count; ++i) {
    const loom_low_schedule_effect_use_t* effect = &schedule->effect_uses[i];
    if (effect->node_index >= schedule->node_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait effect references node %" PRIu32
                              " but schedule has %zu node(s)",
                              effect->node_index, schedule->node_count);
    }
    loom_amdgpu_wait_node_state_t* node_state =
        &builder->node_states[effect->node_index];
    switch (effect->kind) {
      case LOOM_LOW_EFFECT_KIND_READ: {
        if (!loom_amdgpu_wait_effect_is_dependency_memory(effect)) {
          break;
        }
        uint32_t counter_mask = 0;
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_effect_counter_mask(effect, &counter_mask));
        if (counter_mask == 0) {
          node_state->has_default_dependency_read = true;
        } else {
          node_state->read_counter_mask |= counter_mask;
        }
        break;
      }
      case LOOM_LOW_EFFECT_KIND_WRITE: {
        if (!loom_amdgpu_wait_effect_is_dependency_memory(effect)) {
          break;
        }
        uint32_t counter_mask = 0;
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_effect_counter_mask(effect, &counter_mask));
        if (counter_mask == 0) {
          node_state->has_default_dependency_write = true;
          if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_WORKGROUP) {
            node_state->has_default_workgroup_write = true;
          }
        } else {
          node_state->write_counter_mask |= counter_mask;
          if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_WORKGROUP) {
            node_state->workgroup_write_counter_mask |= counter_mask;
          }
        }
        break;
      }
      case LOOM_LOW_EFFECT_KIND_BARRIER:
        if (loom_amdgpu_wait_effect_is_dependency_memory(effect)) {
          uint32_t counter_mask = 0;
          IREE_RETURN_IF_ERROR(
              loom_amdgpu_wait_effect_counter_mask(effect, &counter_mask));
          if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_WORKGROUP) {
            node_state->workgroup_barrier_counter_mask |=
                counter_mask == 0 ? LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY
                                  : counter_mask;
          } else {
            node_state->barrier_counter_mask |=
                counter_mask == 0 ? LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY
                                  : counter_mask;
          }
        }
        break;
      case LOOM_LOW_EFFECT_KIND_COUNTER: {
        if (effect->counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE) {
          node_state->has_generic_counter_effect = true;
          break;
        }
        uint32_t counter_mask = 0;
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_counter_mask(effect->counter_id, &counter_mask));
        node_state->explicit_wait_counter_mask |= counter_mask;
        break;
      }
      default:
        break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_finish_node_classification(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    loom_amdgpu_wait_node_state_t* node_state = &builder->node_states[i];
    if (node_state->has_generic_counter_effect) {
      node_state->explicit_wait_counter_mask |= node_state->hazard_counter_mask;
    }
    if (node_state->has_generic_counter_effect &&
        node_state->explicit_wait_counter_mask == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU generic counter-effect node %zu has no "
                              "wait-counter hazard rows",
                              i);
    }
    if (node_state->explicit_wait_counter_mask != 0 &&
        node_state->hazard_counter_mask == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU counter-effect node %zu has no "
                              "wait-counter hazard rows",
                              i);
    }
    if (node_state->has_default_dependency_read) {
      const uint32_t default_read_counter_mask =
          node_state->hazard_counter_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_READ;
      if (default_read_counter_mask == 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU dependency memory read node %zu has no read counter hazard",
            i);
      }
      node_state->read_counter_mask |= default_read_counter_mask;
    }
    if (iree_any_bit_set(node_state->read_counter_mask,
                         ~node_state->hazard_counter_mask)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU dependency memory read node %zu references a counter without "
          "a wait-counter hazard row",
          i);
    }
    if (node_state->has_default_dependency_write) {
      const uint32_t default_write_counter_mask =
          node_state->hazard_counter_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_WRITE;
      if (default_write_counter_mask == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AMDGPU dependency memory write node %zu has "
                                "no write counter hazard",
                                i);
      }
      node_state->write_counter_mask |= default_write_counter_mask;
      if (node_state->has_default_workgroup_write) {
        node_state->workgroup_write_counter_mask |= default_write_counter_mask;
      }
    }
    if (iree_any_bit_set(node_state->write_counter_mask,
                         ~node_state->hazard_counter_mask)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU dependency memory write node %zu "
                              "references a counter without "
                              "a wait-counter hazard row",
                              i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_build_dependency_links(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  const iree_host_size_t value_count = schedule->liveness.value_count;
  uint32_t* producer_nodes = NULL;
  if (value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(builder->arena, value_count,
                                                   sizeof(*producer_nodes),
                                                   (void**)&producer_nodes));
    for (iree_host_size_t i = 0; i < value_count; ++i) {
      producer_nodes[i] = LOOM_LOW_SCHEDULE_NODE_NONE;
    }
  }

  for (uint32_t node_index = 0; node_index < schedule->node_count;
       ++node_index) {
    const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
    const loom_value_ordinal_t* result_ordinals =
        loom_low_schedule_node_const_result_ordinals(node);
    for (uint16_t i = 0; i < node->result_count; ++i) {
      const loom_value_ordinal_t result_ordinal = result_ordinals[i];
      if (result_ordinal >= value_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU wait dependency result ordinal exceeds liveness domain");
      }
      producer_nodes[result_ordinal] = node_index;
    }
  }

  for (uint32_t consumer_node = 0; consumer_node < schedule->node_count;
       ++consumer_node) {
    const loom_low_schedule_node_t* node = &schedule->nodes[consumer_node];
    const loom_value_ordinal_t* operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(node);
    for (uint16_t i = 0; i < node->operand_count; ++i) {
      const loom_value_ordinal_t operand_ordinal = operand_ordinals[i];
      if (operand_ordinal >= value_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU wait dependency operand ordinal exceeds liveness domain");
      }
      const uint32_t producer_node = producer_nodes[operand_ordinal];
      if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
          producer_node == consumer_node) {
        continue;
      }
      const uint32_t counter_mask =
          builder->node_states[producer_node].read_counter_mask;
      if (counter_mask == 0) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_append_dependency_link(
          builder, producer_node, consumer_node, counter_mask));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_classify_nodes(
    loom_amdgpu_wait_plan_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_classify_hazards(builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_classify_effects(builder));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_finish_node_classification(builder));
  return loom_amdgpu_wait_plan_build_dependency_links(builder);
}

static iree_status_t loom_amdgpu_wait_plan_drain_counter(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_kind_t kind,
    loom_amdgpu_wait_plan_reason_t reason, uint32_t node_index,
    uint32_t producer_node, uint16_t counter_id) {
  const loom_low_schedule_node_t* node = &builder->schedule->nodes[node_index];
  if (counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE ||
      counter_id > LOOM_AMDGPU_WAIT_COUNTER_ALU) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown AMDGPU wait counter id %" PRIu16,
                            counter_id);
  }
  const uint32_t slot = counter_id - 1;
  const uint32_t outstanding_before = builder->outstanding_counts[slot];
  const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
  const uint32_t block_index = node->block_index;
  const loom_low_schedule_block_t* block =
      &builder->schedule->blocks[block_index];
  for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
    const uint32_t packet_index = block->scheduled_node_start + i;
    if (packet_index == node->scheduled_ordinal) {
      break;
    }
    const uint32_t prior_node_index =
        builder->schedule->scheduled_node_indices[packet_index];
    loom_amdgpu_wait_node_state_t* prior_state =
        &builder->node_states[prior_node_index];
    if ((prior_state->read_counter_mask & counter_mask) != 0 &&
        prior_state->produced_counter_epoch[slot] ==
            builder->counter_epochs[slot]) {
      prior_state->drained_after_production_counter_mask |= counter_mask;
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_append_action(
      builder,
      (loom_amdgpu_wait_plan_action_t){
          .kind = kind,
          .reason = reason,
          .counter_id = counter_id,
          .target_count = 0,
          .block_index = node->block_index,
          .node_index = node_index,
          .scheduled_ordinal = node->scheduled_ordinal,
          .producer_node = producer_node,
          .consumer_node = reason == LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE
                               ? node_index
                               : LOOM_LOW_SCHEDULE_NODE_NONE,
          .outstanding_before = outstanding_before,
      }));
  ++builder->counter_epochs[slot];
  builder->outstanding_counts[slot] = 0;
  builder->outstanding_write_counts[slot] = 0;
  builder->outstanding_workgroup_write_counts[slot] = 0;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_drain_mask(
    loom_amdgpu_wait_plan_builder_t* builder,
    loom_amdgpu_wait_plan_action_kind_t kind,
    loom_amdgpu_wait_plan_reason_t reason, uint32_t node_index,
    uint32_t producer_node, uint32_t counter_mask) {
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_COUNT; ++slot) {
    if ((counter_mask & loom_amdgpu_wait_counter_mask_from_slot(slot)) == 0) {
      continue;
    }
    const uint16_t counter_id = loom_amdgpu_wait_counter_id_from_slot(slot);
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_drain_counter(
        builder, kind, reason, node_index, producer_node, counter_id));
  }
  return iree_ok_status();
}

static uint32_t loom_amdgpu_wait_plan_outstanding_counter_mask(
    const uint32_t outstanding_counts[LOOM_AMDGPU_WAIT_COUNTER_COUNT],
    uint32_t counter_mask) {
  uint32_t outstanding_counter_mask = 0;
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_COUNT; ++slot) {
    const uint32_t slot_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    if ((counter_mask & slot_mask) != 0 && outstanding_counts[slot] != 0) {
      outstanding_counter_mask |= slot_mask;
    }
  }
  return outstanding_counter_mask;
}

static iree_status_t loom_amdgpu_wait_plan_handle_consumer(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  uint32_t active_counter_mask = 0;
  uint32_t active_producers[LOOM_AMDGPU_WAIT_COUNTER_COUNT] = {
      LOOM_LOW_SCHEDULE_NODE_NONE, LOOM_LOW_SCHEDULE_NODE_NONE,
      LOOM_LOW_SCHEDULE_NODE_NONE, LOOM_LOW_SCHEDULE_NODE_NONE,
      LOOM_LOW_SCHEDULE_NODE_NONE,
  };
  for (uint32_t link_index =
           builder->first_dependency_link_by_consumer[node_index];
       link_index != LOOM_LOW_SCHEDULE_NODE_NONE;) {
    const loom_amdgpu_wait_dependency_link_t* link =
        &builder->dependency_links[link_index];
    const loom_amdgpu_wait_node_state_t* producer_state =
        &builder->node_states[link->producer_node];
    for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_COUNT; ++slot) {
      const uint32_t counter_mask =
          loom_amdgpu_wait_counter_mask_from_slot(slot);
      if ((link->counter_mask & counter_mask) == 0) {
        continue;
      }
      const uint32_t producer_block =
          builder->schedule->nodes[link->producer_node].block_index;
      const uint32_t consumer_block =
          builder->schedule->nodes[node_index].block_index;
      if (producer_block == consumer_block) {
        // Epochs are block-local because outstanding counts reset at block
        // entry. Within one block, a newer epoch means an earlier wait already
        // drained the producer.
        if (producer_state->produced_counter_epoch[slot] !=
            builder->counter_epochs[slot]) {
          continue;
        }
      } else {
        // Across block boundaries, the producer is safe only if a wait in its
        // own block drained it before control could reach the consumer block.
        if ((producer_state->drained_after_production_counter_mask &
             counter_mask) != 0) {
          continue;
        }
      }
      active_counter_mask |= counter_mask;
      if (active_producers[slot] == LOOM_LOW_SCHEDULE_NODE_NONE) {
        active_producers[slot] = link->producer_node;
      }
    }
    link_index = link->next_link;
  }

  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_COUNT; ++slot) {
    const uint32_t counter_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    if ((active_counter_mask & counter_mask) == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_drain_counter(
        builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
        LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE, node_index,
        active_producers[slot], loom_amdgpu_wait_counter_id_from_slot(slot)));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_handle_barrier(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  const loom_amdgpu_wait_node_state_t* node_state =
      &builder->node_states[node_index];
  uint32_t outstanding_counter_mask =
      loom_amdgpu_wait_plan_outstanding_counter_mask(
          builder->outstanding_counts, node_state->barrier_counter_mask);
  outstanding_counter_mask |= loom_amdgpu_wait_plan_outstanding_counter_mask(
      builder->outstanding_workgroup_write_counts,
      node_state->workgroup_barrier_counter_mask);
  if (outstanding_counter_mask == 0) {
    return iree_ok_status();
  }
  return loom_amdgpu_wait_plan_drain_mask(
      builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
      LOOM_AMDGPU_WAIT_PLAN_REASON_BARRIER, node_index,
      LOOM_LOW_SCHEDULE_NODE_NONE, outstanding_counter_mask);
}

static iree_status_t loom_amdgpu_wait_plan_increment_outstanding_counts(
    uint32_t outstanding_counts[LOOM_AMDGPU_WAIT_COUNTER_COUNT],
    uint32_t counter_mask) {
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_COUNT; ++slot) {
    const uint32_t slot_mask = loom_amdgpu_wait_counter_mask_from_slot(slot);
    if ((counter_mask & slot_mask) == 0) {
      continue;
    }
    if (outstanding_counts[slot] == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "AMDGPU wait outstanding count overflows");
    }
    ++outstanding_counts[slot];
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_note_producer(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  loom_amdgpu_wait_node_state_t* node_state = &builder->node_states[node_index];
  const uint32_t counter_mask =
      node_state->read_counter_mask | node_state->write_counter_mask;
  for (uint32_t slot = 0; slot < LOOM_AMDGPU_WAIT_COUNTER_COUNT; ++slot) {
    if ((counter_mask & loom_amdgpu_wait_counter_mask_from_slot(slot)) == 0) {
      continue;
    }
    node_state->produced_counter_epoch[slot] = builder->counter_epochs[slot];
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_increment_outstanding_counts(
      builder->outstanding_counts, counter_mask));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_increment_outstanding_counts(
      builder->outstanding_write_counts, node_state->write_counter_mask));
  return loom_amdgpu_wait_plan_increment_outstanding_counts(
      builder->outstanding_workgroup_write_counts,
      node_state->workgroup_write_counter_mask);
}

static iree_status_t loom_amdgpu_wait_plan_process_node(
    loom_amdgpu_wait_plan_builder_t* builder, uint32_t node_index) {
  loom_amdgpu_wait_node_state_t* node_state = &builder->node_states[node_index];

  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_handle_consumer(builder, node_index));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_handle_barrier(builder, node_index));
  if (node_state->explicit_wait_counter_mask != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_drain_mask(
        builder, LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT,
        LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET, node_index,
        LOOM_LOW_SCHEDULE_NODE_NONE, node_state->explicit_wait_counter_mask));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_plan_note_producer(builder, node_index));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_plan_build_actions(
    loom_amdgpu_wait_plan_builder_t* builder) {
  const loom_low_schedule_table_t* schedule = builder->schedule;
  for (iree_host_size_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    memset(builder->counter_epochs, 0, sizeof(builder->counter_epochs));
    memset(builder->outstanding_counts, 0, sizeof(builder->outstanding_counts));
    memset(builder->outstanding_write_counts, 0,
           sizeof(builder->outstanding_write_counts));
    memset(builder->outstanding_workgroup_write_counts, 0,
           sizeof(builder->outstanding_workgroup_write_counts));
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      const uint32_t packet_index = block->scheduled_node_start + i;
      if (packet_index >= schedule->scheduled_node_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU wait plan block references scheduled node outside stream");
      }
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      if (node_index >= schedule->node_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU wait plan stream references node outside schedule");
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_wait_plan_process_node(builder, node_index));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_wait_plan_build(
    const loom_low_schedule_table_t* schedule, iree_arena_allocator_t* arena,
    loom_amdgpu_wait_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_wait_plan_t){0};
  loom_amdgpu_wait_plan_builder_t builder = {
      .schedule = schedule,
      .arena = arena,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_allocate(&builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_classify_nodes(&builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_build_actions(&builder));
  *out_plan = (loom_amdgpu_wait_plan_t){
      .schedule = schedule,
      .actions = builder.actions,
      .action_count = builder.action_count,
  };
  return iree_ok_status();
}

static iree_string_view_t loom_amdgpu_wait_plan_json_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<unnamed>");
  }
  return module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_amdgpu_wait_plan_json_function_name(
    const loom_low_schedule_table_t* schedule) {
  if (loom_low_function_def_isa(schedule->function_op)) {
    return loom_amdgpu_wait_plan_json_symbol_name(
        schedule->module, loom_low_function_callee(schedule->function_op));
  }
  return IREE_SV("<unnamed>");
}

iree_status_t loom_amdgpu_wait_plan_format_json(
    const loom_amdgpu_wait_plan_t* plan, iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  const loom_low_schedule_table_t* schedule = plan->schedule;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "\"format\":\"loom.amdgpu.wait_plan.v0\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"function\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, loom_amdgpu_wait_plan_json_function_name(schedule)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, schedule->target.target_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"descriptor_set\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, schedule->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"action_count\":%zu,\"actions\":[", plan->action_count));
  for (iree_host_size_t i = 0; i < plan->action_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    const loom_amdgpu_wait_plan_action_t* action = &plan->actions[i];
    const iree_string_view_t kind_name =
        loom_amdgpu_wait_plan_action_kind_name(action->kind);
    const iree_string_view_t reason_name =
        loom_amdgpu_wait_plan_reason_name(action->reason);
    const iree_string_view_t counter_name =
        loom_amdgpu_wait_counter_name(action->counter_id);
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        "{\"kind\":%u,\"kind_name\":\"%.*s\",\"reason\":%u"
        ",\"reason_name\":\"%.*s\",\"counter\":%" PRIu16
        ",\"counter_name\":\"%.*s\",\"target_count\":%" PRIu16
        ",\"block\":%" PRIu32 ",\"node\":%" PRIu32
        ",\"scheduled_ordinal\":%" PRIu32 ",\"producer_node\":",
        (unsigned)action->kind, (int)kind_name.size, kind_name.data,
        (unsigned)action->reason, (int)reason_name.size, reason_name.data,
        action->counter_id, (int)counter_name.size, counter_name.data,
        action->target_count, action->block_index, action->node_index,
        action->scheduled_ordinal));
    if (action->producer_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "null"));
    } else {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream, "%" PRIu32, action->producer_node));
    }
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, ",\"consumer_node\":"));
    if (action->consumer_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "null"));
    } else {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream, "%" PRIu32, action->consumer_node));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream, ",\"outstanding_before\":%" PRIu32 "}",
        action->outstanding_before));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]}"));
  return iree_ok_status();
}
