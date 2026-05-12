// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU fixed wait-state planning over scheduled target-low functions.
//
// AMDGPU has hazards that are not modeled by wait counters. CDNA MFMA/SMFMAC
// packets need fixed scalar no-op cycles both after matrix results before
// ordinary VGPR consumers and after legacy VALU writes before matrix source
// reads. GFX940-family transcendental VALU results also need a fixed wait
// before dependent non-transcendental VALU consumers, and nearby VALU reads of
// VALU-written SGPRs need fixed waits because the hardware does not interlock
// those dependencies. This table records target-owned insertion points after
// scheduling and allocation, where physical register identity is known.

#ifndef LOOM_TARGET_ARCH_AMDGPU_WAIT_STATES_H_
#define LOOM_TARGET_ARCH_AMDGPU_WAIT_STATES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of cycles a single `s_nop` packet can wait. The SOPP SIMM16
// operand is encoded as wait_cycles - 1, and current AMDGPU targets use a
// 4-bit no-op payload.
#define LOOM_AMDGPU_WAIT_STATE_MAX_S_NOP_CYCLES 16u

typedef enum loom_amdgpu_wait_state_reason_e {
  // Unknown or uninitialized wait-state reason.
  LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN = 0,
  // A packet consumes VGPR storage produced by an outstanding MFMA/SMFMAC.
  LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE = 1,
  // An MFMA/SMFMAC consumes VGPR storage written by a legacy VALU packet.
  LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE = 2,
  // A non-transcendental VALU consumes a GFX940-family transcendental result.
  LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE = 3,
  // A VALU consumes SGPR storage written by a recent VALU packet.
  LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ = 4,
} loom_amdgpu_wait_state_reason_t;

// One fixed wait-state action in scheduled packet order.
typedef struct loom_amdgpu_wait_state_t {
  // Why this wait state exists.
  loom_amdgpu_wait_state_reason_t reason;
  // Region block containing the insertion point.
  uint32_t block_index;
  // Schedule node before which the no-op wait is inserted.
  uint32_t node_index;
  // Scheduled ordinal before which the no-op wait is inserted.
  uint32_t scheduled_ordinal;
  // Producer node that forced the wait.
  uint32_t producer_node;
  // Consumer node that needs the wait.
  uint32_t consumer_node;
  // Number of cycles to wait before |consumer_node|.
  uint16_t cycle_count;
} loom_amdgpu_wait_state_t;

// AMDGPU fixed wait-state table for one scheduled and allocated low function.
typedef struct loom_amdgpu_wait_state_plan_t {
  // Schedule table this plan was built from.
  const loom_low_schedule_table_t* schedule;
  // Allocation table this plan was built from.
  const loom_low_allocation_table_t* allocation;
  // Wait states in scheduled packet order.
  const loom_amdgpu_wait_state_t* states;
  // Number of wait-state records.
  iree_host_size_t state_count;
} loom_amdgpu_wait_state_plan_t;

// Returns the stable spelling for a wait-state reason.
iree_string_view_t loom_amdgpu_wait_state_reason_name(
    loom_amdgpu_wait_state_reason_t reason);

// Builds fixed AMDGPU wait-state insertions from a scheduled and allocated low
// function. The caller must keep |schedule| and |allocation| immutable and
// |arena| alive for as long as |out_plan| is used.
iree_status_t loom_amdgpu_wait_state_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_amdgpu_wait_state_plan_t* out_plan);

// Returns the number of concrete `s_nop` instructions needed by |plan|.
uint64_t loom_amdgpu_wait_state_plan_instruction_count(
    const loom_amdgpu_wait_state_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_WAIT_STATES_H_
