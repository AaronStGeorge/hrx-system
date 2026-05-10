// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU VOPD packetization over scheduled target-low functions.
//
// VOPD is a target-owned post-allocation packetization decision: two VALU
// packets that remain adjacent in the emitted native instruction stream may
// become one native dual-issue packet when their descriptors, physical
// registers, and insertion points satisfy the architectural constraints. The
// plan records that final emission decision without changing the
// target-independent low schedule.

#ifndef LOOM_TARGET_ARCH_AMDGPU_VOPD_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_VOPD_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/wait_packets.h"
#include "loom/target/arch/amdgpu/wait_states.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for scheduled packets that do not belong to a VOPD pair.
#define LOOM_AMDGPU_VOPD_PAIR_NONE UINT32_MAX

// Component opcode for v_fmac_f32 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_FMAC_F32 UINT16_C(0)
// Component opcode for v_fmaak_f32 in a VOPD X/Y slot.
#define LOOM_AMDGPU_VOPD_OP_FMAAK_F32 UINT16_C(1)

typedef enum loom_amdgpu_vopd_packet_role_e {
  // Packet is not part of a VOPD pair.
  LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE = 0,
  // Packet is the X component and emission point for a VOPD pair.
  LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST = 1,
  // Packet is the Y component consumed by the previous VOPD pair.
  LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND = 2,
} loom_amdgpu_vopd_packet_role_t;

typedef enum loom_amdgpu_vopd_pair_reason_e {
  // Unknown or uninitialized VOPD pair reason.
  LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN = 0,
  // Two independent v_fmac_f32 packets were fused into v_dual_fmac_f32.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_FMAC_F32 = 1,
  // Two independent v_fmaak_f32 packets were fused into v_dual_fmaak_f32.
  LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_FMAAK_F32 = 2,
} loom_amdgpu_vopd_pair_reason_t;

typedef enum loom_amdgpu_vopd_pair_flag_bits_e {
  // VOPD pair has no additional payload flags.
  LOOM_AMDGPU_VOPD_PAIR_FLAG_NONE = 0u,
  // VOPD pair uses the shared 32-bit literal payload word.
  LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL = 1u << 0,
} loom_amdgpu_vopd_pair_flag_bits_t;
typedef uint32_t loom_amdgpu_vopd_pair_flags_t;

// One scheduled packet's membership in a planned VOPD pair.
typedef struct loom_amdgpu_vopd_packet_t {
  // Role this scheduled packet plays in a VOPD pair.
  loom_amdgpu_vopd_packet_role_t role;
  // VOPD pair index, or LOOM_AMDGPU_VOPD_PAIR_NONE.
  uint32_t pair_index;
} loom_amdgpu_vopd_packet_t;

// One native VOPD packet replacing two schedule-visible component packets.
typedef struct loom_amdgpu_vopd_pair_t {
  // Why this VOPD pair was formed.
  loom_amdgpu_vopd_pair_reason_t reason;
  // Region block containing both component packets.
  uint32_t block_index;
  // Scheduled packet index for the X component.
  uint32_t first_packet_index;
  // Scheduled packet index for the Y component.
  uint32_t second_packet_index;
  // Schedule node index for the X component.
  uint32_t first_node_index;
  // Schedule node index for the Y component.
  uint32_t second_node_index;
  // VOPD operation id encoded in the X slot.
  uint16_t op_x;
  // VOPD operation id encoded in the Y slot.
  uint16_t op_y;
  // Pair-local payload and encoding flags.
  loom_amdgpu_vopd_pair_flags_t flags;
  // Shared literal payload when LOOM_AMDGPU_VOPD_PAIR_FLAG_LITERAL is set.
  uint32_t literal_u32;
} loom_amdgpu_vopd_pair_t;

// AMDGPU VOPD packetization table for one scheduled and allocated low function.
typedef struct loom_amdgpu_vopd_plan_t {
  // Schedule table this plan was built from.
  const loom_low_schedule_table_t* schedule;
  // Allocation table this plan was built from.
  const loom_low_allocation_table_t* allocation;
  // VOPD pairs in scheduled order.
  const loom_amdgpu_vopd_pair_t* pairs;
  // Number of VOPD pair records.
  iree_host_size_t pair_count;
  // Per-scheduled-packet VOPD membership records.
  const loom_amdgpu_vopd_packet_t* packets;
  // Number of packet membership records.
  iree_host_size_t packet_count;
} loom_amdgpu_vopd_plan_t;

// Returns the stable spelling for a VOPD packet role.
iree_string_view_t loom_amdgpu_vopd_packet_role_name(
    loom_amdgpu_vopd_packet_role_t role);

// Returns the stable spelling for a VOPD pair reason.
iree_string_view_t loom_amdgpu_vopd_pair_reason_name(
    loom_amdgpu_vopd_pair_reason_t reason);

// Builds conservative AMDGPU VOPD pairings from a scheduled and allocated low
// function. Optional wait packet/state plans suppress pairs that would consume
// an insertion point before the second component. The caller must keep
// |schedule|, |allocation|, and |arena| immutable/alive for as long as
// |out_plan| is used.
iree_status_t loom_amdgpu_vopd_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    const loom_amdgpu_wait_state_plan_t* wait_states,
    iree_arena_allocator_t* arena, loom_amdgpu_vopd_plan_t* out_plan);

// Verifies that |plan| describes |schedule| and |allocation|.
iree_status_t loom_amdgpu_vopd_plan_verify(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_vopd_plan_t* plan);

// Verifies that wait insertions do not target the second component of any VOPD
// pair. Emission cannot preserve such an insertion without breaking the dual
// packet.
iree_status_t loom_amdgpu_vopd_plan_verify_wait_insertions(
    const loom_amdgpu_vopd_plan_t* plan,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    const loom_amdgpu_wait_state_plan_t* wait_states);

// Returns the VOPD membership record for |packet_index|, or NULL.
const loom_amdgpu_vopd_packet_t* loom_amdgpu_vopd_plan_packet_at(
    const loom_amdgpu_vopd_plan_t* plan, iree_host_size_t packet_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_VOPD_PLAN_H_
