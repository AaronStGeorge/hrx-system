// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet.h"

#include <inttypes.h>
#include <string.h>

iree_status_t loom_low_packet_validate_sidecars(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation) {
  if (!schedule || !allocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "schedule and allocation sidecars are required");
  }
  if (schedule->module != allocation->module ||
      schedule->function_op != allocation->function_op) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "schedule and allocation sidecars must describe the same low function");
  }
  if (schedule->target.descriptor_set != allocation->target.descriptor_set) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "schedule and allocation sidecars must use the same descriptor set");
  }
  return iree_ok_status();
}

iree_host_size_t loom_low_packet_count(
    const loom_low_schedule_sidecar_t* schedule) {
  return schedule ? schedule->scheduled_node_count : 0;
}

iree_status_t loom_low_packet_node_index_at(
    const loom_low_schedule_sidecar_t* schedule, iree_host_size_t packet_index,
    uint32_t* out_node_index) {
  if (!schedule || !out_node_index) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "schedule and out_node_index are required");
  }
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
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    iree_host_size_t packet_index, loom_low_packet_view_t* out_packet) {
  if (!out_packet) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_packet is required");
  }
  memset(out_packet, 0, sizeof(*out_packet));
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_sidecars(schedule, allocation));

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

const loom_low_allocation_assignment_t* loom_low_packet_find_assignment(
    const loom_low_allocation_sidecar_t* allocation, loom_value_id_t value_id,
    iree_host_size_t* out_assignment_index) {
  if (out_assignment_index) {
    *out_assignment_index = IREE_HOST_SIZE_MAX;
  }
  if (!allocation) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    if (allocation->assignments[i].value_id == value_id) {
      if (out_assignment_index) {
        *out_assignment_index = i;
      }
      return &allocation->assignments[i];
    }
  }
  return NULL;
}

uint32_t loom_low_packet_block_index(
    const loom_low_schedule_sidecar_t* schedule, const loom_block_t* block) {
  if (!schedule || !block) {
    return LOOM_LOW_PACKET_INDEX_NONE;
  }
  for (iree_host_size_t i = 0; i < schedule->block_count; ++i) {
    if (schedule->blocks[i].block == block) {
      return (uint32_t)i;
    }
  }
  return LOOM_LOW_PACKET_INDEX_NONE;
}

uint32_t loom_low_packet_hazard_gap_packet_index(
    const loom_low_schedule_sidecar_t* schedule,
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
