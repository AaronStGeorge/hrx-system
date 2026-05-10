// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/vopd_plan.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/move_sequence.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/ir.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/arch/amdgpu/target_refs.h"

#define LOOM_AMDGPU_VOPD_OP_FMAC_F32 0u

typedef struct loom_amdgpu_vopd_candidate_registers_t {
  // X component destination and tied accumulator VGPR.
  uint16_t vdst_x;
  // X component first explicit source VGPR.
  uint16_t src0_x;
  // X component second explicit source VGPR.
  uint16_t vsrc1_x;
  // Y component destination and tied accumulator VGPR.
  uint16_t vdst_y;
  // Y component first explicit source VGPR.
  uint16_t src0_y;
  // Y component second explicit source VGPR.
  uint16_t vsrc1_y;
} loom_amdgpu_vopd_candidate_registers_t;

typedef struct loom_amdgpu_vopd_plan_builder_t {
  // Schedule table being analyzed.
  const loom_low_schedule_table_t* schedule;
  // Allocation table supplying physical register assignments.
  const loom_low_allocation_table_t* allocation;
  // Optional planned wait packets that block second-component fusion.
  const loom_amdgpu_wait_packet_plan_t* wait_packets;
  // Optional planned wait states that block second-component fusion.
  const loom_amdgpu_wait_state_plan_t* wait_states;
  // Arena owning all output and scratch arrays.
  iree_arena_allocator_t* arena;
  // Scheduled packets with wait insertions before them.
  bool* insertion_blocked_packets;
  // Output VOPD pair records.
  loom_amdgpu_vopd_pair_t* pairs;
  // Number of populated VOPD pair records.
  iree_host_size_t pair_count;
  // Allocated VOPD pair capacity.
  iree_host_size_t pair_capacity;
  // Output per-packet membership records.
  loom_amdgpu_vopd_packet_t* packets;
} loom_amdgpu_vopd_plan_builder_t;

iree_string_view_t loom_amdgpu_vopd_packet_role_name(
    loom_amdgpu_vopd_packet_role_t role) {
  switch (role) {
    case LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE:
      return IREE_SV("none");
    case LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST:
      return IREE_SV("first");
    case LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND:
      return IREE_SV("second");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_amdgpu_vopd_pair_reason_name(
    loom_amdgpu_vopd_pair_reason_t reason) {
  switch (reason) {
    case LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_FMAC_F32:
      return IREE_SV("dual_fmac_f32");
    case LOOM_AMDGPU_VOPD_PAIR_REASON_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

const loom_amdgpu_vopd_packet_t* loom_amdgpu_vopd_plan_packet_at(
    const loom_amdgpu_vopd_plan_t* plan, iree_host_size_t packet_index) {
  if (plan == NULL || packet_index >= plan->packet_count) {
    return NULL;
  }
  const loom_amdgpu_vopd_packet_t* packet = &plan->packets[packet_index];
  return packet->role == LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE ? NULL : packet;
}

static bool loom_amdgpu_vopd_target_supports_base_vopd(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set == NULL ||
      descriptor_set->target_stable_id != LOOM_AMDGPU_TARGET_STABLE_ID) {
    return false;
  }
  switch (descriptor_set->descriptor_set_ordinal) {
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3:
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4:
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_vopd_verify_wait_packet_plan(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_wait_packet_plan_t* wait_packets) {
  if (wait_packets == NULL) {
    return iree_ok_status();
  }
  if (wait_packets->wait_plan == NULL ||
      wait_packets->wait_plan->schedule != schedule) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD wait packets must be derived from "
                            "the planned schedule");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_verify_wait_state_plan(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_state_plan_t* wait_states) {
  if (wait_states != NULL && (wait_states->schedule != schedule ||
                              wait_states->allocation != allocation)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD wait states must be derived from the "
                            "planned schedule and allocation");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_packet_index_for_insertion(
    const loom_low_schedule_table_t* schedule, uint32_t block_index,
    uint32_t node_index, uint32_t scheduled_ordinal,
    uint32_t* out_packet_index) {
  *out_packet_index = LOOM_LOW_PACKET_INDEX_NONE;
  if (block_index >= schedule->block_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU VOPD insertion block index %" PRIu32
                            " is out of range",
                            block_index);
  }
  const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
  if (scheduled_ordinal >= block->scheduled_node_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU VOPD insertion scheduled ordinal %" PRIu32
                            " is out of range",
                            scheduled_ordinal);
  }
  const uint32_t packet_index = block->scheduled_node_start + scheduled_ordinal;
  uint32_t packet_node_index = LOOM_LOW_SCHEDULE_NODE_NONE;
  IREE_RETURN_IF_ERROR(loom_low_packet_node_index_at(schedule, packet_index,
                                                     &packet_node_index));
  if (packet_node_index != node_index) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD insertion node %" PRIu32
                            " does not match scheduled packet node %" PRIu32,
                            node_index, packet_node_index);
  }
  *out_packet_index = packet_index;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_mark_wait_packet_insertions(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  if (builder->wait_packets == NULL) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < builder->wait_packets->packet_count; ++i) {
    const loom_amdgpu_wait_packet_t* wait_packet =
        &builder->wait_packets->packets[i];
    uint32_t packet_index = LOOM_LOW_PACKET_INDEX_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_packet_index_for_insertion(
        builder->schedule, wait_packet->block_index, wait_packet->node_index,
        wait_packet->scheduled_ordinal, &packet_index));
    builder->insertion_blocked_packets[packet_index] = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_mark_wait_state_insertions(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  if (builder->wait_states == NULL) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < builder->wait_states->state_count; ++i) {
    const loom_amdgpu_wait_state_t* wait_state =
        &builder->wait_states->states[i];
    uint32_t packet_index = LOOM_LOW_PACKET_INDEX_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_packet_index_for_insertion(
        builder->schedule, wait_state->block_index, wait_state->node_index,
        wait_state->scheduled_ordinal, &packet_index));
    builder->insertion_blocked_packets[packet_index] = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_plan_allocate(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  const iree_host_size_t packet_count = builder->schedule->scheduled_node_count;
  builder->pair_capacity = packet_count / 2;
  if (packet_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(builder->arena, packet_count,
                                                   sizeof(*builder->packets),
                                                   (void**)&builder->packets));
    for (iree_host_size_t i = 0; i < packet_count; ++i) {
      builder->packets[i] = (loom_amdgpu_vopd_packet_t){
          .role = LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE,
          .pair_index = LOOM_AMDGPU_VOPD_PAIR_NONE,
      };
    }
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->arena, packet_count,
                                  sizeof(*builder->insertion_blocked_packets),
                                  (void**)&builder->insertion_blocked_packets));
    memset(builder->insertion_blocked_packets, 0,
           packet_count * sizeof(*builder->insertion_blocked_packets));
  }
  if (builder->pair_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->pair_capacity, sizeof(*builder->pairs),
        (void**)&builder->pairs));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_mark_wait_packet_insertions(builder));
  return loom_amdgpu_vopd_mark_wait_state_insertions(builder);
}

static const loom_low_allocation_assignment_t* loom_amdgpu_vopd_map_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id) {
  return loom_low_allocation_try_map_active_value_assignment(allocation,
                                                             value_id, NULL);
}

static bool loom_amdgpu_vopd_assignment_single_physical_vgpr(
    const loom_low_allocation_assignment_t* assignment,
    uint16_t* out_register) {
  *out_register = 0;
  if (assignment == NULL ||
      assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
      assignment->location_count != 1 || assignment->location_base > 255) {
    return false;
  }
  *out_register = (uint16_t)assignment->location_base;
  return true;
}

static bool loom_amdgpu_vopd_assignments_match(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs != NULL && rhs != NULL &&
         lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

static bool loom_amdgpu_vopd_component_result_is_used_by(
    const loom_low_packet_view_t* producer,
    const loom_low_packet_view_t* consumer) {
  const loom_op_t* producer_op = producer->node->op;
  const loom_op_t* consumer_op = consumer->node->op;
  const loom_value_id_t* results = loom_op_const_results(producer_op);
  const loom_value_id_t* operands = loom_op_const_operands(consumer_op);
  for (iree_host_size_t result_index = 0;
       result_index < producer_op->result_count; ++result_index) {
    for (iree_host_size_t operand_index = 0;
         operand_index < consumer_op->operand_count; ++operand_index) {
      if (results[result_index] == operands[operand_index]) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_amdgpu_vopd_cross_component_destinations_are_independent(
    const loom_amdgpu_vopd_candidate_registers_t* registers) {
  if (registers->vdst_x == registers->src0_y ||
      registers->vdst_x == registers->vsrc1_y) {
    return false;
  }
  if (registers->vdst_y == registers->src0_x ||
      registers->vdst_y == registers->vsrc1_x) {
    return false;
  }
  return true;
}

static bool loom_amdgpu_vopd_bank_compatible(uint16_t x_register,
                                             uint16_t y_register,
                                             uint16_t bank_mask) {
  return (x_register & bank_mask) != (y_register & bank_mask);
}

static bool loom_amdgpu_vopd_registers_satisfy_base_constraints(
    const loom_amdgpu_vopd_candidate_registers_t* registers) {
  if (((registers->vdst_x ^ registers->vdst_y) & 1u) == 0) {
    return false;
  }
  if (!loom_amdgpu_vopd_bank_compatible(registers->src0_x, registers->src0_y,
                                        3)) {
    return false;
  }
  if (!loom_amdgpu_vopd_bank_compatible(registers->vsrc1_x, registers->vsrc1_y,
                                        3)) {
    return false;
  }
  return loom_amdgpu_vopd_cross_component_destinations_are_independent(
      registers);
}

static iree_status_t loom_amdgpu_vopd_read_fmac_registers(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* packet, uint16_t* out_vdst,
    uint16_t* out_src0, uint16_t* out_vsrc1, bool* out_eligible) {
  *out_vdst = 0;
  *out_src0 = 0;
  *out_vsrc1 = 0;
  *out_eligible = false;

  const loom_op_t* op = packet->node->op;
  if (op->result_count != 1 || op->operand_count != 3) {
    return iree_ok_status();
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, results[0]);
  const loom_low_allocation_assignment_t* accumulator_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, operands[0]);
  const loom_low_allocation_assignment_t* src0_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, operands[1]);
  const loom_low_allocation_assignment_t* vsrc1_assignment =
      loom_amdgpu_vopd_map_assignment(builder->allocation, operands[2]);
  if (!loom_amdgpu_vopd_assignments_match(result_assignment,
                                          accumulator_assignment)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(result_assignment,
                                                        out_vdst)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(src0_assignment,
                                                        out_src0)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_vopd_assignment_single_physical_vgpr(vsrc1_assignment,
                                                        out_vsrc1)) {
    return iree_ok_status();
  }
  *out_eligible = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_try_match_fmac_pair(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* first, const loom_low_packet_view_t* second,
    bool* out_matched) {
  *out_matched = false;
  const loom_low_descriptor_t* fmac_descriptor =
      loom_amdgpu_descriptor_ref_descriptor(
          builder->schedule->target.descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_FMAC_F32);
  if (fmac_descriptor == NULL || first->descriptor != fmac_descriptor ||
      second->descriptor != fmac_descriptor) {
    return iree_ok_status();
  }
  if (loom_amdgpu_vopd_component_result_is_used_by(first, second)) {
    return iree_ok_status();
  }

  loom_amdgpu_vopd_candidate_registers_t registers = {0};
  bool first_eligible = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_read_fmac_registers(
      builder, first, &registers.vdst_x, &registers.src0_x, &registers.vsrc1_x,
      &first_eligible));
  bool second_eligible = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_read_fmac_registers(
      builder, second, &registers.vdst_y, &registers.src0_y, &registers.vsrc1_y,
      &second_eligible));
  if (!first_eligible || !second_eligible) {
    return iree_ok_status();
  }
  *out_matched =
      loom_amdgpu_vopd_registers_satisfy_base_constraints(&registers);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_append_pair(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_packet_view_t* first, const loom_low_packet_view_t* second) {
  if (builder->pair_count >= builder->pair_capacity ||
      builder->pair_count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU VOPD plan exceeded precomputed pair capacity");
  }
  const uint32_t pair_index = (uint32_t)builder->pair_count;
  builder->pairs[builder->pair_count++] = (loom_amdgpu_vopd_pair_t){
      .reason = LOOM_AMDGPU_VOPD_PAIR_REASON_DUAL_FMAC_F32,
      .block_index = first->node->block_index,
      .first_packet_index = (uint32_t)first->packet_index,
      .second_packet_index = (uint32_t)second->packet_index,
      .first_node_index = first->node_index,
      .second_node_index = second->node_index,
      .op_x = LOOM_AMDGPU_VOPD_OP_FMAC_F32,
      .op_y = LOOM_AMDGPU_VOPD_OP_FMAC_F32,
  };
  builder->packets[first->packet_index] = (loom_amdgpu_vopd_packet_t){
      .role = LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST,
      .pair_index = pair_index,
  };
  builder->packets[second->packet_index] = (loom_amdgpu_vopd_packet_t){
      .role = LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND,
      .pair_index = pair_index,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_packet_is_transparent(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    iree_host_size_t packet_index, bool* out_transparent) {
  *out_transparent = false;
  if (builder->insertion_blocked_packets[packet_index]) {
    return iree_ok_status();
  }

  loom_low_packet_view_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_packet_view_at(
      builder->schedule, builder->allocation, packet_index, &packet));
  if (packet.descriptor != NULL) {
    return iree_ok_status();
  }

  iree_host_size_t move_count = 0;
  const loom_op_t* op = packet.node->op;
  switch (op->kind) {
    case LOOM_OP_LOW_SLICE: {
      IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_slice_units(
          builder->allocation, op, &move_count));
      break;
    }
    case LOOM_OP_LOW_CONCAT: {
      IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_concat_units(
          builder->allocation, op, &move_count));
      break;
    }
    default:
      return iree_ok_status();
  }
  *out_transparent = move_count == 0;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_find_visible_packet(
    const loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_schedule_block_t* block,
    iree_host_size_t search_packet_index, iree_host_size_t* out_packet_index,
    bool* out_found) {
  *out_packet_index = LOOM_LOW_PACKET_INDEX_NONE;
  *out_found = false;

  const iree_host_size_t block_end =
      block->scheduled_node_start + block->scheduled_node_count;
  for (iree_host_size_t packet_index = search_packet_index;
       packet_index < block_end; ++packet_index) {
    bool transparent = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_packet_is_transparent(
        builder, packet_index, &transparent));
    if (transparent) {
      continue;
    }
    *out_packet_index = packet_index;
    *out_found = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_plan_block(
    loom_amdgpu_vopd_plan_builder_t* builder,
    const loom_low_schedule_block_t* block) {
  const iree_host_size_t block_end =
      block->scheduled_node_start + block->scheduled_node_count;
  for (iree_host_size_t search_packet_index = block->scheduled_node_start;
       search_packet_index < block_end;) {
    iree_host_size_t first_packet_index = LOOM_LOW_PACKET_INDEX_NONE;
    bool found_first = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_find_visible_packet(
        builder, block, search_packet_index, &first_packet_index,
        &found_first));
    if (!found_first) {
      break;
    }

    iree_host_size_t second_packet_index = LOOM_LOW_PACKET_INDEX_NONE;
    bool found_second = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_find_visible_packet(
        builder, block, first_packet_index + 1, &second_packet_index,
        &found_second));
    if (!found_second) {
      break;
    }
    if (builder->insertion_blocked_packets[second_packet_index]) {
      search_packet_index = first_packet_index + 1;
      continue;
    }

    loom_low_packet_view_t first = {0};
    IREE_RETURN_IF_ERROR(loom_low_packet_view_at(
        builder->schedule, builder->allocation, first_packet_index, &first));
    loom_low_packet_view_t second = {0};
    IREE_RETURN_IF_ERROR(loom_low_packet_view_at(
        builder->schedule, builder->allocation, second_packet_index, &second));
    bool matched = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_try_match_fmac_pair(
        builder, &first, &second, &matched));
    if (!matched) {
      search_packet_index = first_packet_index + 1;
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vopd_append_pair(builder, &first, &second));
    search_packet_index = second_packet_index + 1;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_plan_build_pairs(
    loom_amdgpu_vopd_plan_builder_t* builder) {
  loom_low_allocation_value_scratch_t scratch = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_acquire_value_scratch(builder->allocation, &scratch));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < builder->schedule->block_count && iree_status_is_ok(status); ++i) {
    status =
        loom_amdgpu_vopd_plan_block(builder, &builder->schedule->blocks[i]);
  }
  loom_low_allocation_release_value_scratch(&scratch);
  return status;
}

iree_status_t loom_amdgpu_vopd_plan_verify(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_vopd_plan_t* plan) {
  if (plan == NULL) {
    return iree_ok_status();
  }
  if (plan->schedule != schedule || plan->allocation != allocation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD plan must be derived from the emitted "
                            "schedule and allocation");
  }
  if (plan->packet_count != 0 &&
      plan->packet_count != schedule->scheduled_node_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD plan has %" PRIhsz
                            " packet records for %" PRIhsz " scheduled packets",
                            plan->packet_count, schedule->scheduled_node_count);
  }
  if (plan->pair_count != 0 && (plan->pairs == NULL || plan->packets == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU VOPD plan pair records are incomplete");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vopd_verify_insertion_packet(
    const loom_amdgpu_vopd_plan_t* plan, uint32_t block_index,
    uint32_t node_index, uint32_t scheduled_ordinal,
    iree_string_view_t insertion_kind) {
  if (plan == NULL || plan->packet_count == 0) {
    return iree_ok_status();
  }
  uint32_t packet_index = LOOM_LOW_PACKET_INDEX_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_packet_index_for_insertion(
      plan->schedule, block_index, node_index, scheduled_ordinal,
      &packet_index));
  const loom_amdgpu_vopd_packet_t* packet =
      loom_amdgpu_vopd_plan_packet_at(plan, packet_index);
  if (packet != NULL && packet->role == LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU VOPD plan cannot preserve %.*s insertion before fused second "
        "packet %" PRIu32,
        (int)insertion_kind.size, insertion_kind.data, packet_index);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_vopd_plan_verify_wait_insertions(
    const loom_amdgpu_vopd_plan_t* plan,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    const loom_amdgpu_wait_state_plan_t* wait_states) {
  if (plan == NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vopd_verify_wait_packet_plan(plan->schedule, wait_packets));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_verify_wait_state_plan(
      plan->schedule, plan->allocation, wait_states));
  if (wait_packets != NULL) {
    for (iree_host_size_t i = 0; i < wait_packets->packet_count; ++i) {
      const loom_amdgpu_wait_packet_t* wait_packet = &wait_packets->packets[i];
      IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_verify_insertion_packet(
          plan, wait_packet->block_index, wait_packet->node_index,
          wait_packet->scheduled_ordinal, IREE_SV("wait-packet")));
    }
  }
  if (wait_states != NULL) {
    for (iree_host_size_t i = 0; i < wait_states->state_count; ++i) {
      const loom_amdgpu_wait_state_t* wait_state = &wait_states->states[i];
      IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_verify_insertion_packet(
          plan, wait_state->block_index, wait_state->node_index,
          wait_state->scheduled_ordinal, IREE_SV("wait-state")));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_vopd_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    const loom_amdgpu_wait_state_plan_t* wait_states,
    iree_arena_allocator_t* arena, loom_amdgpu_vopd_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vopd_plan_t){0};
  if (schedule == NULL || allocation == NULL || arena == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "schedule, allocation, and arena are required for "
                            "AMDGPU VOPD planning");
  }
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(schedule, allocation));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vopd_verify_wait_packet_plan(schedule, wait_packets));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_verify_wait_state_plan(
      schedule, allocation, wait_states));
  if (!loom_amdgpu_vopd_target_supports_base_vopd(
          schedule->target.descriptor_set)) {
    *out_plan = (loom_amdgpu_vopd_plan_t){
        .schedule = schedule,
        .allocation = allocation,
    };
    return iree_ok_status();
  }

  loom_amdgpu_vopd_plan_builder_t builder = {
      .schedule = schedule,
      .allocation = allocation,
      .wait_packets = wait_packets,
      .wait_states = wait_states,
      .arena = arena,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_allocate(&builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vopd_plan_build_pairs(&builder));
  *out_plan = (loom_amdgpu_vopd_plan_t){
      .schedule = schedule,
      .allocation = allocation,
      .pairs = builder.pairs,
      .pair_count = builder.pair_count,
      .packets = builder.packets,
      .packet_count = schedule->scheduled_node_count,
  };
  return iree_ok_status();
}
