// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet.h"

#include <inttypes.h>
#include <string.h>

iree_status_t loom_low_packet_validate_tables(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation) {
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(allocation);
  if (schedule->module == NULL || schedule->function_op == NULL ||
      allocation->module == NULL || allocation->function_op == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "schedule and allocation tables must name a low function");
  }
  if (schedule->module != allocation->module ||
      schedule->function_op != allocation->function_op) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "schedule and allocation tables must describe the same low function");
  }
  if (schedule->target.descriptor_set != allocation->target.descriptor_set) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "schedule and allocation tables must use the same descriptor set");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_packet_validate_asm_form_ordinal(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, uint32_t asm_form_ordinal) {
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "selected asm form has no descriptor set");
  }
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_set_asm_form_at(descriptor_set, asm_form_ordinal);
  if (asm_form == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "selected asm form ordinal %" PRIu32
                            " is out of range",
                            asm_form_ordinal);
  }
  if (asm_form->descriptor_ordinal != descriptor_ordinal) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected asm form ordinal %" PRIu32 " belongs to descriptor %" PRIu32
        " instead of descriptor %" PRIu32,
        asm_form_ordinal, asm_form->descriptor_ordinal, descriptor_ordinal);
  }
  return iree_ok_status();
}

iree_status_t loom_low_packet_validate_asm_form_table(
    const loom_low_schedule_table_t* schedule,
    const loom_low_packet_asm_form_table_t* asm_forms) {
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(asm_forms);
  if (schedule->module != asm_forms->module ||
      schedule->function_op != asm_forms->function_op) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected asm-form table must describe the scheduled low function");
  }
  if (schedule->target.descriptor_set != asm_forms->target.descriptor_set) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected asm-form table must use the schedule descriptor set");
  }
  if (asm_forms->asm_form_ordinal_count != schedule->scheduled_node_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "selected asm-form table has %" PRIhsz
                            " packet entries for %" PRIhsz " scheduled packets",
                            asm_forms->asm_form_ordinal_count,
                            schedule->scheduled_node_count);
  }
  if (asm_forms->asm_form_ordinal_count > 0 &&
      asm_forms->asm_form_ordinals == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected asm-form table entries are required for non-empty table");
  }
  for (iree_host_size_t packet_index = 0;
       packet_index < asm_forms->asm_form_ordinal_count; ++packet_index) {
    const uint32_t asm_form_ordinal =
        asm_forms->asm_form_ordinals[packet_index];
    if (asm_form_ordinal == LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
      continue;
    }
    uint32_t node_index = LOOM_LOW_SCHEDULE_NODE_NONE;
    IREE_RETURN_IF_ERROR(
        loom_low_packet_node_index_at(schedule, packet_index, &node_index));
    if (node_index >= schedule->node_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "selected asm-form packet %" PRIhsz
                              " references node %" PRIu32,
                              packet_index, node_index);
    }
    const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
    if (node->descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "selected asm form ordinal %" PRIu32
                              " was provided for structural "
                              "packet %" PRIhsz,
                              asm_form_ordinal, packet_index);
    }
    IREE_RETURN_IF_ERROR(loom_low_packet_validate_asm_form_ordinal(
        schedule->target.descriptor_set, node->descriptor_ordinal,
        asm_form_ordinal));
  }
  return iree_ok_status();
}

iree_host_size_t loom_low_packet_count(
    const loom_low_schedule_table_t* schedule) {
  return schedule ? schedule->scheduled_node_count : 0;
}

iree_status_t loom_low_packet_node_index_at(
    const loom_low_schedule_table_t* schedule, iree_host_size_t packet_index,
    uint32_t* out_node_index) {
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(out_node_index);
  *out_node_index = LOOM_LOW_SCHEDULE_NODE_NONE;
  if (packet_index >= schedule->scheduled_node_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "packet index %" PRIhsz
                            " is out of range for %" PRIhsz " packet(s)",
                            packet_index, schedule->scheduled_node_count);
  }
  if (!schedule->scheduled_node_indices) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "schedule has packets but no packet index table");
  }
  *out_node_index = schedule->scheduled_node_indices[packet_index];
  return iree_ok_status();
}

iree_status_t loom_low_packet_view_at(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_host_size_t packet_index, loom_low_packet_view_t* out_packet) {
  IREE_ASSERT_ARGUMENT(out_packet);
  memset(out_packet, 0, sizeof(*out_packet));
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(schedule, allocation));

  uint32_t node_index = LOOM_LOW_SCHEDULE_NODE_NONE;
  IREE_RETURN_IF_ERROR(
      loom_low_packet_node_index_at(schedule, packet_index, &node_index));
  if (node_index >= schedule->node_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "packet index %" PRIhsz " references node %" PRIu32
                            " but schedule has %" PRIhsz " node(s)",
                            packet_index, node_index, schedule->node_count);
  }

  const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
  const loom_low_descriptor_t* descriptor = NULL;
  if (node->descriptor_ordinal != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    if (!schedule->target.descriptor_set) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "descriptor-backed packet has no descriptor set");
    }
    descriptor = loom_low_descriptor_set_descriptor_at(
        schedule->target.descriptor_set, node->descriptor_ordinal);
    if (!descriptor) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "packet index %" PRIhsz
                              " references descriptor ordinal %" PRIu32,
                              packet_index, node->descriptor_ordinal);
    }
  }

  *out_packet = (loom_low_packet_view_t){
      .packet_index = packet_index,
      .node_index = node_index,
      .node = node,
      .descriptor = descriptor,
  };
  return iree_ok_status();
}

iree_status_t loom_low_packet_lookup_asm_form(
    const loom_low_schedule_table_t* schedule,
    const loom_low_packet_asm_form_table_t* asm_forms,
    const loom_low_packet_view_t* packet, uint32_t* out_asm_form_ordinal) {
  IREE_ASSERT_ARGUMENT(out_asm_form_ordinal);
  *out_asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  IREE_ASSERT_ARGUMENT(schedule);
  IREE_ASSERT_ARGUMENT(packet);
  if (packet->node == NULL || packet->descriptor == NULL ||
      packet->node->descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "packet has no descriptor-backed asm form");
  }
  if (asm_forms != NULL) {
    if (schedule->module != asm_forms->module ||
        schedule->function_op != asm_forms->function_op ||
        schedule->target.descriptor_set != asm_forms->target.descriptor_set) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "selected asm-form table must match the scheduled low function");
    }
    if (packet->packet_index >= asm_forms->asm_form_ordinal_count ||
        asm_forms->asm_form_ordinals == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "selected asm-form table does not cover packet %" PRIhsz,
          packet->packet_index);
    }
    const uint32_t selected_asm_form_ordinal =
        asm_forms->asm_form_ordinals[packet->packet_index];
    if (selected_asm_form_ordinal != LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
      IREE_RETURN_IF_ERROR(loom_low_packet_validate_asm_form_ordinal(
          schedule->target.descriptor_set, packet->node->descriptor_ordinal,
          selected_asm_form_ordinal));
      *out_asm_form_ordinal = selected_asm_form_ordinal;
      return iree_ok_status();
    }
  }
  *out_asm_form_ordinal = loom_low_descriptor_set_lookup_canonical_asm_form(
      schedule->target.descriptor_set, packet->node->descriptor_ordinal);
  return iree_ok_status();
}

uint32_t loom_low_packet_block_index(const loom_low_schedule_table_t* schedule,
                                     const loom_block_t* block) {
  if (!schedule || !block) {
    return LOOM_LOW_PACKET_INDEX_NONE;
  }
  uint16_t block_index = 0;
  if (!loom_region_try_block_index(block->parent_region, block, &block_index) ||
      block_index >= schedule->block_count) {
    return LOOM_LOW_PACKET_INDEX_NONE;
  }
  return schedule->blocks[block_index].block == block
             ? block_index
             : LOOM_LOW_PACKET_INDEX_NONE;
}

uint32_t loom_low_packet_hazard_gap_packet_index(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_hazard_gap_t* hazard_gap,
    uint32_t scheduled_ordinal) {
  if (!schedule || !hazard_gap ||
      hazard_gap->block_index >= schedule->block_count) {
    return LOOM_LOW_PACKET_INDEX_NONE;
  }
  const loom_low_schedule_block_t* block =
      &schedule->blocks[hazard_gap->block_index];
  const uint64_t packet_index =
      (uint64_t)block->scheduled_node_start + scheduled_ordinal;
  return packet_index < LOOM_LOW_PACKET_INDEX_NONE ? (uint32_t)packet_index
                                                   : LOOM_LOW_PACKET_INDEX_NONE;
}
