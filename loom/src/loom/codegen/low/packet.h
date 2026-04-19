// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent packet views over scheduled and allocated low functions.
//
// The packet layer is the emitter-facing join between schedule and allocation
// sidecars. It intentionally does not serialize, print types, or know about any
// target backend. Native, VM, SPIR-V, and diagnostic emitters can all consume
// this view without copying the schedule/allocation join logic into each
// backend or routing through JSON.

#ifndef LOOM_CODEGEN_LOW_PACKET_H_
#define LOOM_CODEGEN_LOW_PACKET_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for absent packet indices.
#define LOOM_LOW_PACKET_INDEX_NONE UINT32_MAX

// One scheduled packet in emitter order.
typedef struct loom_low_packet_view_t {
  // Packet ordinal in the final scheduled stream.
  iree_host_size_t packet_index;
  // Schedule-node index represented by this packet.
  uint32_t node_index;
  // Schedule node represented by this packet.
  const loom_low_schedule_node_t* node;
  // Descriptor row for descriptor-backed packets, or NULL for structural ops.
  const loom_low_descriptor_t* descriptor;
} loom_low_packet_view_t;

// Verifies that |schedule| and |allocation| describe the same low function and
// target descriptor set. This is an emitter contract check, not target
// legality.
iree_status_t loom_low_packet_validate_sidecars(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation);

// Returns the number of scheduled packets in |schedule|. NULL schedules have no
// packets.
iree_host_size_t loom_low_packet_count(
    const loom_low_schedule_sidecar_t* schedule);

// Returns the schedule-node index at |packet_index|.
iree_status_t loom_low_packet_node_index_at(
    const loom_low_schedule_sidecar_t* schedule, iree_host_size_t packet_index,
    uint32_t* out_node_index);

// Returns the packet view at |packet_index|.
iree_status_t loom_low_packet_view_at(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    iree_host_size_t packet_index, loom_low_packet_view_t* out_packet);

// Finds the allocation assignment for |value_id|. Returns NULL when the value
// has no assignment. |out_assignment_index| is optional.
const loom_low_allocation_assignment_t* loom_low_packet_find_assignment(
    const loom_low_allocation_sidecar_t* allocation, loom_value_id_t value_id,
    iree_host_size_t* out_assignment_index);

// Returns the region-block index for |block|, or LOOM_LOW_PACKET_INDEX_NONE
// when |block| does not belong to |schedule|.
uint32_t loom_low_packet_block_index(
    const loom_low_schedule_sidecar_t* schedule, const loom_block_t* block);

// Maps a hazard-gap scheduled ordinal within the gap block to a packet index.
// Returns LOOM_LOW_PACKET_INDEX_NONE when the gap block is invalid or the
// computed packet index cannot fit in the packet-index sentinel domain.
uint32_t loom_low_packet_hazard_gap_packet_index(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_schedule_hazard_gap_t* hazard_gap,
    uint32_t scheduled_ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PACKET_H_
