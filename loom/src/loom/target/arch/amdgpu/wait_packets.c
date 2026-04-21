// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/wait_packets.h"

#include <inttypes.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

#define LOOM_AMDGPU_WAIT_PACKET_TARGET_COUNT_NONE UINT16_MAX

typedef enum loom_amdgpu_wait_packet_kind_e {
  LOOM_AMDGPU_WAIT_PACKET_KIND_COMBINED_MEMORY = 0,
  LOOM_AMDGPU_WAIT_PACKET_KIND_LOAD = 1,
  LOOM_AMDGPU_WAIT_PACKET_KIND_STORE = 2,
  LOOM_AMDGPU_WAIT_PACKET_KIND_ALU = 3,
} loom_amdgpu_wait_packet_kind_t;

typedef struct loom_amdgpu_wait_packet_descriptor_t {
  // True when the descriptor exists in the target descriptor set.
  bool available;
  // Descriptor ordinal in the target descriptor set.
  uint32_t descriptor_ordinal;
  // Borrowed descriptor row.
  const loom_low_descriptor_t* descriptor;
  // Borrowed descriptor key string.
  iree_string_view_t key;
} loom_amdgpu_wait_packet_descriptor_t;

typedef struct loom_amdgpu_wait_packet_target_t {
  // Combined memory wait descriptor, such as amdgpu.s_waitcnt.
  loom_amdgpu_wait_packet_descriptor_t combined_memory_wait;
  // Split load wait descriptor, such as amdgpu.s_wait_loadcnt.
  loom_amdgpu_wait_packet_descriptor_t load_wait;
  // Split store wait descriptor, such as amdgpu.s_wait_storecnt.
  loom_amdgpu_wait_packet_descriptor_t store_wait;
  // ALU dependency wait descriptor, such as amdgpu.s_wait_alu.
  loom_amdgpu_wait_packet_descriptor_t alu_wait;
} loom_amdgpu_wait_packet_target_t;

typedef struct loom_amdgpu_wait_packet_group_t {
  // Region block containing the insertion point.
  uint32_t block_index;
  // Schedule node before which wait packets are inserted.
  uint32_t node_index;
  // Scheduled ordinal before which wait packets are inserted.
  uint32_t scheduled_ordinal;
  // First input action in this coalescing group.
  iree_host_size_t source_action_start;
  // Number of input actions in this coalescing group.
  iree_host_size_t source_action_count;
  // Logical counters required by this group.
  uint32_t counter_mask;
  // Target wait count per logical wait-counter slot.
  uint16_t target_counts[3];
} loom_amdgpu_wait_packet_group_t;

typedef struct loom_amdgpu_wait_packet_builder_t {
  // Logical wait plan being materialized.
  const loom_amdgpu_wait_plan_t* wait_plan;
  // Descriptor set selected by the scheduled low function.
  const loom_low_descriptor_set_t* descriptor_set;
  // Arena owning all output rows.
  iree_arena_allocator_t* arena;
  // Concrete wait packet descriptors available on this descriptor set.
  loom_amdgpu_wait_packet_target_t target;
  // Output concrete packet rows.
  loom_amdgpu_wait_packet_t* packets;
  // Number of populated concrete packet rows.
  iree_host_size_t packet_count;
  // Allocated concrete packet row capacity.
  iree_host_size_t packet_capacity;
  // Output immediate rows.
  loom_amdgpu_wait_packet_immediate_t* immediates;
  // Number of populated immediate rows.
  iree_host_size_t immediate_count;
  // Allocated immediate row capacity.
  iree_host_size_t immediate_capacity;
} loom_amdgpu_wait_packet_builder_t;

static uint16_t loom_amdgpu_wait_packet_counter_id_from_slot(uint32_t slot) {
  return (uint16_t)(slot + 1);
}

static iree_status_t loom_amdgpu_wait_packet_counter_slot(uint16_t counter_id,
                                                          uint32_t* out_slot) {
  IREE_ASSERT_ARGUMENT(out_slot);
  if (counter_id < LOOM_AMDGPU_WAIT_COUNTER_LOAD ||
      counter_id > LOOM_AMDGPU_WAIT_COUNTER_ALU) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown AMDGPU wait counter id %" PRIu16,
                            counter_id);
  }
  *out_slot = (uint32_t)(counter_id - 1);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_target_count(
    const loom_amdgpu_wait_packet_group_t* group, uint32_t counter_mask,
    uint16_t* out_target_count) {
  IREE_ASSERT_ARGUMENT(out_target_count);
  uint16_t target_count = LOOM_AMDGPU_WAIT_PACKET_TARGET_COUNT_NONE;
  for (uint32_t slot = 0; slot < IREE_ARRAYSIZE(group->target_counts); ++slot) {
    const uint32_t slot_mask = 1u << slot;
    if ((counter_mask & slot_mask) == 0 ||
        group->target_counts[slot] ==
            LOOM_AMDGPU_WAIT_PACKET_TARGET_COUNT_NONE) {
      continue;
    }
    if (group->target_counts[slot] < target_count) {
      target_count = group->target_counts[slot];
    }
  }
  if (target_count == LOOM_AMDGPU_WAIT_PACKET_TARGET_COUNT_NONE) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU wait packet group has no target count for counter mask 0x%x",
        counter_mask);
  }
  *out_target_count = target_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_verify_target(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU wait packet materialization requires a descriptor set");
  }
  if (descriptor_set->target_stable_id != LOOM_AMDGPU_TARGET_STABLE_ID) {
    iree_string_view_t target_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
        descriptor_set, descriptor_set->target_key_string_offset, &target_key));
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU wait packet materialization received target '%.*s'",
        (int)target_key.size, target_key.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_descriptor_counter_effect(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, bool* out_has_counter_effect,
    uint16_t* out_counter_id) {
  *out_has_counter_effect = false;
  *out_counter_id = LOOM_AMDGPU_WAIT_COUNTER_NONE;
  if (descriptor->effect_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->effect_start > descriptor_set->effect_count ||
      descriptor->effect_count >
          descriptor_set->effect_count - descriptor->effect_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait descriptor effect range is out of range");
  }
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (effect->kind != LOOM_LOW_EFFECT_KIND_COUNTER) {
      continue;
    }
    if (*out_has_counter_effect) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU wait descriptor carries multiple counter effects");
    }
    *out_has_counter_effect = true;
    *out_counter_id = effect->counter_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_assign_descriptor(
    loom_amdgpu_wait_packet_builder_t* builder, const char* packet_kind,
    uint32_t descriptor_ordinal, const loom_low_descriptor_t* descriptor,
    loom_amdgpu_wait_packet_descriptor_t* target_descriptor) {
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      builder->descriptor_set, descriptor->key_string_offset, &key));
  if (target_descriptor->available) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "AMDGPU descriptor set has multiple %s wait packets: '%.*s' and "
        "'%.*s'",
        packet_kind, (int)target_descriptor->key.size,
        target_descriptor->key.data, (int)key.size, key.data);
  }
  *target_descriptor = (loom_amdgpu_wait_packet_descriptor_t){
      .available = true,
      .descriptor_ordinal = descriptor_ordinal,
      .descriptor = descriptor,
      .key = key,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_classify_descriptor(
    loom_amdgpu_wait_packet_builder_t* builder, uint32_t descriptor_ordinal,
    const loom_low_descriptor_t* descriptor) {
  bool has_counter_effect = false;
  uint16_t counter_id = LOOM_AMDGPU_WAIT_COUNTER_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_descriptor_counter_effect(
      builder->descriptor_set, descriptor, &has_counter_effect, &counter_id));
  if (!has_counter_effect) {
    return iree_ok_status();
  }
  if (descriptor->immediate_count == 2 &&
      counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE) {
    return loom_amdgpu_wait_packet_assign_descriptor(
        builder, "combined memory", descriptor_ordinal, descriptor,
        &builder->target.combined_memory_wait);
  }
  if (descriptor->immediate_count != 1) {
    return iree_ok_status();
  }
  switch (counter_id) {
    case LOOM_AMDGPU_WAIT_COUNTER_LOAD:
      return loom_amdgpu_wait_packet_assign_descriptor(
          builder, "load", descriptor_ordinal, descriptor,
          &builder->target.load_wait);
    case LOOM_AMDGPU_WAIT_COUNTER_STORE:
      return loom_amdgpu_wait_packet_assign_descriptor(
          builder, "store", descriptor_ordinal, descriptor,
          &builder->target.store_wait);
    case LOOM_AMDGPU_WAIT_COUNTER_ALU:
      return loom_amdgpu_wait_packet_assign_descriptor(
          builder, "ALU", descriptor_ordinal, descriptor,
          &builder->target.alu_wait);
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_wait_packet_analyze_target(
    loom_amdgpu_wait_packet_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_packet_verify_target(builder->descriptor_set));
  const loom_amdgpu_wait_packet_descriptor_t unavailable = {
      .descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
  };
  builder->target = (loom_amdgpu_wait_packet_target_t){
      .combined_memory_wait = unavailable,
      .load_wait = unavailable,
      .store_wait = unavailable,
      .alu_wait = unavailable,
  };
  for (uint32_t i = 0; i < builder->descriptor_set->descriptor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_classify_descriptor(
        builder, i, &builder->descriptor_set->descriptors[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_allocate(
    loom_amdgpu_wait_packet_builder_t* builder) {
  iree_host_size_t planned_action_count = 0;
  for (iree_host_size_t i = 0; i < builder->wait_plan->action_count; ++i) {
    if (builder->wait_plan->actions[i].kind ==
        LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED) {
      ++planned_action_count;
    }
  }
  builder->packet_capacity = planned_action_count;
  if (builder->packet_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->packet_capacity, sizeof(*builder->packets),
        (void**)&builder->packets));
  }

  if (!iree_host_size_checked_mul(planned_action_count, 2,
                                  &builder->immediate_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU wait packet immediate capacity overflows");
  }
  if (builder->immediate_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->immediate_capacity,
        sizeof(*builder->immediates), (void**)&builder->immediates));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_append_immediate(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_amdgpu_wait_packet_descriptor_t* packet_descriptor,
    uint16_t descriptor_immediate_index, iree_string_view_t expected_name,
    uint16_t value) {
  if (builder->immediate_count >= builder->immediate_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU wait packet plan exceeded precomputed immediate capacity");
  }
  if (descriptor_immediate_index >=
      packet_descriptor->descriptor->immediate_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait descriptor '%.*s' immediate index %" PRIu16
        " is out of range",
        (int)packet_descriptor->key.size, packet_descriptor->key.data,
        descriptor_immediate_index);
  }
  const uint32_t immediate_row =
      packet_descriptor->descriptor->immediate_start +
      descriptor_immediate_index;
  if (immediate_row >= builder->descriptor_set->immediate_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait descriptor '%.*s' immediate row %" PRIu32
        " is out of range",
        (int)packet_descriptor->key.size, packet_descriptor->key.data,
        immediate_row);
  }
  const loom_low_immediate_t* immediate =
      &builder->descriptor_set->immediates[immediate_row];
  if (immediate->kind != LOOM_LOW_IMMEDIATE_KIND_UNSIGNED) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU wait descriptor '%.*s' immediate %" PRIu16 " must be unsigned",
        (int)packet_descriptor->key.size, packet_descriptor->key.data,
        descriptor_immediate_index);
  }
  if (value > immediate->unsigned_max) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait descriptor '%.*s' immediate %" PRIu16 " value %" PRIu16
        " exceeds maximum %" PRIu64,
        (int)packet_descriptor->key.size, packet_descriptor->key.data,
        descriptor_immediate_index, value, immediate->unsigned_max);
  }

  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      builder->descriptor_set, immediate->field_name_string_offset, &name));
  if (!iree_string_view_equal(name, expected_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU wait descriptor '%.*s' immediate %" PRIu16
        " is '%.*s' but expected '%.*s'",
        (int)packet_descriptor->key.size, packet_descriptor->key.data,
        descriptor_immediate_index, (int)name.size, name.data,
        (int)expected_name.size, expected_name.data);
  }

  builder->immediates[builder->immediate_count++] =
      (loom_amdgpu_wait_packet_immediate_t){
          .descriptor_immediate_index = descriptor_immediate_index,
          .name = name,
          .value = value,
      };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_append_single_immediate(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_amdgpu_wait_packet_descriptor_t* packet_descriptor,
    iree_string_view_t expected_name, uint16_t value) {
  if (packet_descriptor->descriptor->immediate_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU wait descriptor '%.*s' expected 1 immediate but has %" PRIu16,
        (int)packet_descriptor->key.size, packet_descriptor->key.data,
        packet_descriptor->descriptor->immediate_count);
  }
  return loom_amdgpu_wait_packet_append_immediate(
      builder, packet_descriptor, /*descriptor_immediate_index=*/0,
      expected_name, value);
}

static iree_status_t loom_amdgpu_wait_packet_append_combined_immediates(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_amdgpu_wait_packet_descriptor_t* packet_descriptor,
    uint16_t target_count) {
  if (packet_descriptor->descriptor->immediate_count != 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU wait descriptor '%.*s' expected 2 immediates but has %" PRIu16,
        (int)packet_descriptor->key.size, packet_descriptor->key.data,
        packet_descriptor->descriptor->immediate_count);
  }
  bool saw_vmcnt = false;
  bool saw_lgkmcnt = false;
  for (uint16_t i = 0; i < packet_descriptor->descriptor->immediate_count;
       ++i) {
    const uint32_t immediate_row =
        packet_descriptor->descriptor->immediate_start + i;
    if (immediate_row >= builder->descriptor_set->immediate_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait descriptor immediate row %" PRIu32
                              " is out of range",
                              immediate_row);
    }
    const loom_low_immediate_t* immediate =
        &builder->descriptor_set->immediates[immediate_row];
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
        builder->descriptor_set, immediate->field_name_string_offset, &name));
    if (iree_string_view_equal(name, IREE_SV("vmcnt"))) {
      saw_vmcnt = true;
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_append_immediate(
          builder, packet_descriptor, i, IREE_SV("vmcnt"), target_count));
      continue;
    }
    if (iree_string_view_equal(name, IREE_SV("lgkmcnt"))) {
      saw_lgkmcnt = true;
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_append_immediate(
          builder, packet_descriptor, i, IREE_SV("lgkmcnt"), target_count));
      continue;
    }
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU wait descriptor '%.*s' has unsupported immediate '%.*s'",
        (int)packet_descriptor->key.size, packet_descriptor->key.data,
        (int)name.size, name.data);
  }
  if (!saw_vmcnt || !saw_lgkmcnt) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU wait descriptor '%.*s' must carry vmcnt and lgkmcnt",
        (int)packet_descriptor->key.size, packet_descriptor->key.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_append_packet(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_amdgpu_wait_packet_group_t* group,
    const loom_amdgpu_wait_packet_descriptor_t* packet_descriptor,
    loom_amdgpu_wait_packet_kind_t packet_kind, uint32_t counter_mask) {
  if (builder->packet_count >= builder->packet_capacity) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU wait packet plan exceeded precomputed packet capacity");
  }
  if (!packet_descriptor->available) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU target cannot materialize wait packet for "
                            "counter mask 0x%x",
                            counter_mask);
  }

  const iree_host_size_t immediate_start = builder->immediate_count;
  uint16_t target_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_packet_target_count(group, counter_mask, &target_count));
  switch (packet_kind) {
    case LOOM_AMDGPU_WAIT_PACKET_KIND_COMBINED_MEMORY: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_append_combined_immediates(
          builder, packet_descriptor, target_count));
      break;
    }
    case LOOM_AMDGPU_WAIT_PACKET_KIND_LOAD: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_append_single_immediate(
          builder, packet_descriptor, IREE_SV("loadcnt"), target_count));
      break;
    }
    case LOOM_AMDGPU_WAIT_PACKET_KIND_STORE: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_append_single_immediate(
          builder, packet_descriptor, IREE_SV("storecnt"), target_count));
      break;
    }
    case LOOM_AMDGPU_WAIT_PACKET_KIND_ALU: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_append_single_immediate(
          builder, packet_descriptor, IREE_SV("depctr"), target_count));
      break;
    }
  }
  builder->packets[builder->packet_count++] = (loom_amdgpu_wait_packet_t){
      .descriptor_ordinal = packet_descriptor->descriptor_ordinal,
      .descriptor_key = packet_descriptor->key,
      .block_index = group->block_index,
      .node_index = group->node_index,
      .scheduled_ordinal = group->scheduled_ordinal,
      .counter_mask = counter_mask,
      .source_action_start = group->source_action_start,
      .source_action_count = group->source_action_count,
      .immediate_start = immediate_start,
      .immediate_count = builder->immediate_count - immediate_start,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_materialize_memory_group(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_amdgpu_wait_packet_group_t* group, uint32_t memory_mask) {
  const bool can_split =
      ((memory_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_LOAD) == 0 ||
       builder->target.load_wait.available) &&
      ((memory_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_STORE) == 0 ||
       builder->target.store_wait.available);
  if (can_split) {
    if ((memory_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_LOAD) != 0) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_append_packet(
          builder, group, &builder->target.load_wait,
          LOOM_AMDGPU_WAIT_PACKET_KIND_LOAD,
          LOOM_AMDGPU_WAIT_COUNTER_MASK_LOAD));
    }
    if ((memory_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_STORE) != 0) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_append_packet(
          builder, group, &builder->target.store_wait,
          LOOM_AMDGPU_WAIT_PACKET_KIND_STORE,
          LOOM_AMDGPU_WAIT_COUNTER_MASK_STORE));
    }
    return iree_ok_status();
  }
  if (builder->target.combined_memory_wait.available) {
    return loom_amdgpu_wait_packet_append_packet(
        builder, group, &builder->target.combined_memory_wait,
        LOOM_AMDGPU_WAIT_PACKET_KIND_COMBINED_MEMORY,
        LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY);
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU target cannot materialize memory wait for counter mask 0x%x",
      memory_mask);
}

static iree_status_t loom_amdgpu_wait_packet_materialize_group(
    loom_amdgpu_wait_packet_builder_t* builder,
    const loom_amdgpu_wait_packet_group_t* group) {
  const uint32_t unknown_mask =
      group->counter_mask & ~LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL;
  if (unknown_mask != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait packet group has unknown counter "
                            "mask 0x%x",
                            unknown_mask);
  }
  const uint32_t memory_mask =
      group->counter_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY;
  if (memory_mask != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_materialize_memory_group(
        builder, group, memory_mask));
  }
  if ((group->counter_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU) != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_append_packet(
        builder, group, &builder->target.alu_wait,
        LOOM_AMDGPU_WAIT_PACKET_KIND_ALU, LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_wait_packet_same_insertion_point(
    const loom_amdgpu_wait_plan_action_t* lhs,
    const loom_amdgpu_wait_plan_action_t* rhs) {
  return lhs->block_index == rhs->block_index &&
         lhs->node_index == rhs->node_index &&
         lhs->scheduled_ordinal == rhs->scheduled_ordinal;
}

static void loom_amdgpu_wait_packet_group_initialize(
    const loom_amdgpu_wait_plan_t* wait_plan, iree_host_size_t action_index,
    loom_amdgpu_wait_packet_group_t* out_group) {
  IREE_ASSERT_ARGUMENT(out_group);
  const loom_amdgpu_wait_plan_action_t* action =
      &wait_plan->actions[action_index];
  *out_group = (loom_amdgpu_wait_packet_group_t){
      .block_index = action->block_index,
      .node_index = action->node_index,
      .scheduled_ordinal = action->scheduled_ordinal,
      .source_action_start = action_index,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(out_group->target_counts); ++i) {
    out_group->target_counts[i] = LOOM_AMDGPU_WAIT_PACKET_TARGET_COUNT_NONE;
  }
}

static iree_status_t loom_amdgpu_wait_packet_group_accumulate(
    loom_amdgpu_wait_packet_group_t* group,
    const loom_amdgpu_wait_plan_action_t* action) {
  uint32_t counter_mask = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_counter_mask(action->counter_id, &counter_mask));
  uint32_t slot = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_packet_counter_slot(action->counter_id, &slot));
  group->counter_mask |= counter_mask;
  if (action->target_count < group->target_counts[slot]) {
    group->target_counts[slot] = action->target_count;
  }
  ++group->source_action_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_packet_build_packets(
    loom_amdgpu_wait_packet_builder_t* builder) {
  const loom_amdgpu_wait_plan_t* wait_plan = builder->wait_plan;
  for (iree_host_size_t i = 0; i < wait_plan->action_count;) {
    const loom_amdgpu_wait_plan_action_t* action = &wait_plan->actions[i];
    if (action->kind != LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED) {
      ++i;
      continue;
    }
    loom_amdgpu_wait_packet_group_t group = {0};
    loom_amdgpu_wait_packet_group_initialize(wait_plan, i, &group);
    iree_host_size_t next_action_index = i;
    while (next_action_index < wait_plan->action_count &&
           loom_amdgpu_wait_packet_same_insertion_point(
               action, &wait_plan->actions[next_action_index])) {
      const loom_amdgpu_wait_plan_action_t* group_action =
          &wait_plan->actions[next_action_index];
      if (group_action->kind == LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_wait_packet_group_accumulate(&group, group_action));
      }
      ++next_action_index;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_packet_materialize_group(builder, &group));
    i = next_action_index;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_wait_packet_plan_build(
    const loom_amdgpu_wait_plan_t* wait_plan, iree_arena_allocator_t* arena,
    loom_amdgpu_wait_packet_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_amdgpu_wait_packet_plan_t){0};
  if (wait_plan == NULL || wait_plan->schedule == NULL || arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "wait plan, schedule, and arena are required for AMDGPU wait packet "
        "materialization");
  }
  loom_amdgpu_wait_packet_builder_t builder = {
      .wait_plan = wait_plan,
      .descriptor_set = wait_plan->schedule->target.descriptor_set,
      .arena = arena,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_analyze_target(&builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_allocate(&builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_build_packets(&builder));
  *out_plan = (loom_amdgpu_wait_packet_plan_t){
      .wait_plan = wait_plan,
      .packets = builder.packets,
      .packet_count = builder.packet_count,
      .immediates = builder.immediates,
      .immediate_count = builder.immediate_count,
  };
  return iree_ok_status();
}

static iree_string_view_t loom_amdgpu_wait_packet_json_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (module == NULL || !loom_symbol_ref_is_valid(symbol_ref) ||
      symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<unnamed>");
  }
  return module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_amdgpu_wait_packet_json_function_name(
    const loom_amdgpu_wait_packet_plan_t* plan) {
  const loom_low_schedule_sidecar_t* schedule = plan->wait_plan->schedule;
  if (loom_low_func_def_isa(schedule->function_op)) {
    return loom_amdgpu_wait_packet_json_symbol_name(
        schedule->module, loom_low_func_def_callee(schedule->function_op));
  }
  return IREE_SV("<unnamed>");
}

static iree_status_t loom_amdgpu_wait_packet_json_write_counters(
    loom_output_stream_t* stream, uint32_t counter_mask) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "["));
  bool needs_comma = false;
  for (uint32_t slot = 0; slot < 3; ++slot) {
    const uint32_t slot_mask = 1u << slot;
    if ((counter_mask & slot_mask) == 0) {
      continue;
    }
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    }
    const uint16_t counter_id =
        loom_amdgpu_wait_packet_counter_id_from_slot(slot);
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, loom_amdgpu_wait_counter_name(counter_id)));
    needs_comma = true;
  }
  return loom_output_stream_write_cstring(stream, "]");
}

iree_status_t loom_amdgpu_wait_packet_plan_format_json(
    const loom_amdgpu_wait_packet_plan_t* plan,
    iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(builder);
  if (plan == NULL || plan->wait_plan == NULL ||
      plan->wait_plan->schedule == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU wait packet plan with schedule is required");
  }
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  const loom_low_schedule_sidecar_t* schedule = plan->wait_plan->schedule;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "\"format\":\"loom.amdgpu.wait_packet_plan.v0\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"function\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, loom_amdgpu_wait_packet_json_function_name(plan)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, schedule->target.target_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"descriptor_set\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, schedule->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream, ",\"packet_count\":%zu,\"packets\":[", plan->packet_count));
  for (iree_host_size_t i = 0; i < plan->packet_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
    }
    const loom_amdgpu_wait_packet_t* packet = &plan->packets[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "{"));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(&stream, "\"descriptor\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, packet->descriptor_key));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        ",\"descriptor_ordinal\":%" PRIu32 ",\"block\":%" PRIu32
        ",\"node\":%" PRIu32 ",\"scheduled_ordinal\":%" PRIu32
        ",\"counter_mask\":%" PRIu32 ",\"counters\":",
        packet->descriptor_ordinal, packet->block_index, packet->node_index,
        packet->scheduled_ordinal, packet->counter_mask));
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_json_write_counters(
        &stream, packet->counter_mask));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        &stream,
        ",\"source_action_start\":%zu,\"source_action_count\":%zu"
        ",\"immediates\":[",
        packet->source_action_start, packet->source_action_count));
    for (iree_host_size_t j = 0; j < packet->immediate_count; ++j) {
      if (j > 0) {
        IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ","));
      }
      const iree_host_size_t immediate_index = packet->immediate_start + j;
      if (immediate_index >= plan->immediate_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU wait packet immediate index %zu is out of range",
            immediate_index);
      }
      const loom_amdgpu_wait_packet_immediate_t* immediate =
          &plan->immediates[immediate_index];
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream, "{\"index\":%" PRIu16 ",\"name\":",
          immediate->descriptor_immediate_index));
      IREE_RETURN_IF_ERROR(
          loom_json_write_escaped_string(&stream, immediate->name));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          &stream, ",\"value\":%" PRIu16 "}", immediate->value));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, "]}"));
  }
  return loom_output_stream_write_cstring(&stream, "]}");
}
