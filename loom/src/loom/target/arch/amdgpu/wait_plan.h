// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU wait-counter planning over scheduled target-low functions.
//
// The shared low scheduler records target-neutral descriptor facts in scheduled
// order. This layer owns the AMDGPU interpretation of those facts: memory
// packets create outstanding wait-counter work, explicit wait packets drain
// counters, and missing waits are reported as planned insertions before the
// packet that needs the wait. The plan is a sidecar only; IR materialization is
// a later target-owned pass.

#ifndef LOOM_TARGET_ARCH_AMDGPU_WAIT_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_WAIT_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

// AMDGPU target counter ids used by the current descriptor overlays.
enum loom_amdgpu_wait_counter_e {
  // Descriptor did not name a concrete AMDGPU wait counter.
  LOOM_AMDGPU_WAIT_COUNTER_NONE = 0,
  // VMEM/SMEM load-result dependency counter.
  LOOM_AMDGPU_WAIT_COUNTER_LOAD = 1,
  // VMEM/global store completion counter.
  LOOM_AMDGPU_WAIT_COUNTER_STORE = 2,
  // ALU dependency counter used by depctr-style wait packets.
  LOOM_AMDGPU_WAIT_COUNTER_ALU = 3,
};

// Bit masks for AMDGPU wait counters. These are descriptor-overlay ids, not
// native instruction bit encodings.
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_LOAD ((uint32_t)1u << 0)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_STORE ((uint32_t)1u << 1)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU ((uint32_t)1u << 2)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_MEMORY \
  (LOOM_AMDGPU_WAIT_COUNTER_MASK_LOAD | LOOM_AMDGPU_WAIT_COUNTER_MASK_STORE)
#define LOOM_AMDGPU_WAIT_COUNTER_MASK_ALL                                     \
  (LOOM_AMDGPU_WAIT_COUNTER_MASK_LOAD | LOOM_AMDGPU_WAIT_COUNTER_MASK_STORE | \
   LOOM_AMDGPU_WAIT_COUNTER_MASK_ALU)

typedef enum loom_amdgpu_wait_plan_action_kind_e {
  // Unknown or uninitialized action kind.
  LOOM_AMDGPU_WAIT_PLAN_ACTION_UNKNOWN = 0,
  // Wait already exists in the scheduled low stream.
  LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT = 1,
  // Wait must be inserted before final emission.
  LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED = 2,
} loom_amdgpu_wait_plan_action_kind_t;

typedef enum loom_amdgpu_wait_plan_reason_e {
  // Unknown or uninitialized wait reason.
  LOOM_AMDGPU_WAIT_PLAN_REASON_UNKNOWN = 0,
  // Explicit wait packet in the low stream drains this counter.
  LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET = 1,
  // A consumer uses a value produced by an outstanding memory load.
  LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE = 2,
  // Block terminator exits with outstanding stores.
  LOOM_AMDGPU_WAIT_PLAN_REASON_BLOCK_EXIT = 3,
} loom_amdgpu_wait_plan_reason_t;

// One AMDGPU wait-counter action in scheduled packet order.
typedef struct loom_amdgpu_wait_plan_action_t {
  // Whether the action is present in the IR or must be inserted.
  loom_amdgpu_wait_plan_action_kind_t kind;
  // Why this wait action exists.
  loom_amdgpu_wait_plan_reason_t reason;
  // AMDGPU wait counter affected by the action.
  uint16_t counter_id;
  // Wait target value. The first planning slice always drains to zero.
  uint16_t target_count;
  // Region block containing the insertion point or explicit wait.
  uint32_t block_index;
  // Node before which a planned wait is inserted, or explicit wait node.
  uint32_t node_index;
  // Scheduled ordinal before which a planned wait is inserted, or the explicit
  // wait node's scheduled ordinal.
  uint32_t scheduled_ordinal;
  // Producer node that forced the wait, or LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t producer_node;
  // Consumer node that needs the wait, or LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t consumer_node;
  // Outstanding packet count for this counter before the wait action.
  uint32_t outstanding_before;
} loom_amdgpu_wait_plan_action_t;

// AMDGPU wait-counter sidecar for one scheduled low function.
typedef struct loom_amdgpu_wait_plan_t {
  // Schedule sidecar this plan was built from.
  const loom_low_schedule_sidecar_t* schedule;
  // Wait actions in scheduled packet order.
  const loom_amdgpu_wait_plan_action_t* actions;
  // Number of action records.
  iree_host_size_t action_count;
} loom_amdgpu_wait_plan_t;

// Returns the stable diagnostic spelling for an AMDGPU wait counter id.
iree_string_view_t loom_amdgpu_wait_counter_name(uint16_t counter_id);

// Returns the bit mask for one AMDGPU wait counter id.
iree_status_t loom_amdgpu_wait_counter_mask(uint16_t counter_id,
                                            uint32_t* out_mask);

// Returns the stable diagnostic spelling for a wait-plan action kind.
iree_string_view_t loom_amdgpu_wait_plan_action_kind_name(
    loom_amdgpu_wait_plan_action_kind_t kind);

// Returns the stable diagnostic spelling for a wait-plan reason.
iree_string_view_t loom_amdgpu_wait_plan_reason_name(
    loom_amdgpu_wait_plan_reason_t reason);

// Builds an AMDGPU wait-counter plan from a scheduled low function. The caller
// must keep |schedule| immutable and |arena| alive for as long as |out_plan| is
// used.
iree_status_t loom_amdgpu_wait_plan_build(
    const loom_low_schedule_sidecar_t* schedule, iree_arena_allocator_t* arena,
    loom_amdgpu_wait_plan_t* out_plan);

// Appends a compact JSON representation of |plan| to |builder|.
iree_status_t loom_amdgpu_wait_plan_format_json(
    const loom_amdgpu_wait_plan_t* plan, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_WAIT_PLAN_H_
