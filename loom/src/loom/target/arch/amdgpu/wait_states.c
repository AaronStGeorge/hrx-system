// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/wait_states.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/move_sequence.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/descriptor_semantics.h"
#include "loom/target/arch/amdgpu/matrix_contract.h"
#include "loom/target/arch/amdgpu/target_id.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/arch/amdgpu/target_refs.h"

#define LOOM_AMDGPU_WAIT_STATE_VALU_TO_MATRIX_CYCLES 2u
#define LOOM_AMDGPU_WAIT_STATE_TRANS_RESULT_USE_CYCLES 1u
#define LOOM_AMDGPU_WAIT_STATE_REASON_COUNT 4u

typedef enum loom_amdgpu_wait_state_vgpr_flag_bits_e {
  LOOM_AMDGPU_WAIT_STATE_VGPR_FLAG_VALID = 1u << 0,
} loom_amdgpu_wait_state_vgpr_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_state_vgpr_flags_t;

typedef enum loom_amdgpu_wait_state_reason_flag_bits_e {
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_MATRIX_RESULT_USE =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_TO_MATRIX_USE =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE,
} loom_amdgpu_wait_state_reason_flag_bits_t;
typedef uint32_t loom_amdgpu_wait_state_reason_flags_t;

typedef struct loom_amdgpu_wait_state_hazard_t {
  // Active-state flags for this physical VGPR.
  loom_amdgpu_wait_state_vgpr_flags_t flags;
  // Hazard reason associated with this outstanding write.
  loom_amdgpu_wait_state_reason_t reason;
  // Schedule node that produced the outstanding hazard.
  uint32_t producer_node;
  // Instruction-position immediately after the producer packet.
  uint64_t producer_end_position;
  // Required wait cycles before the matching consumer reads the VGPR.
  uint16_t required_cycle_count;
} loom_amdgpu_wait_state_hazard_t;

typedef struct loom_amdgpu_wait_state_vgpr_t {
  // Per-reason outstanding fixed-wait hazard state for this physical VGPR.
  loom_amdgpu_wait_state_hazard_t hazards[LOOM_AMDGPU_WAIT_STATE_REASON_COUNT];
} loom_amdgpu_wait_state_vgpr_t;

typedef struct loom_amdgpu_wait_state_match_t {
  // Reason responsible for the largest unsatisfied wait.
  loom_amdgpu_wait_state_reason_t reason;
  // Producer node responsible for the largest unsatisfied wait.
  uint32_t producer_node;
  // Additional cycles required before the current consumer.
  uint16_t cycle_count;
} loom_amdgpu_wait_state_match_t;

typedef struct loom_amdgpu_wait_state_builder_t {
  // Schedule table being analyzed.
  const loom_low_schedule_table_t* schedule;
  // Allocation table being analyzed.
  const loom_low_allocation_table_t* allocation;
  // Arena that owns all output and scratch arrays.
  iree_arena_allocator_t* arena;
  // Processor facts selected by the low target, or NULL if unavailable.
  const loom_amdgpu_processor_info_t* processor;
  // Per-physical-VGPR outstanding fixed-wait hazard state.
  loom_amdgpu_wait_state_vgpr_t* vgprs;
  // Number of entries in |vgprs|.
  iree_host_size_t vgpr_count;
  // Output wait-state rows.
  loom_amdgpu_wait_state_t* states;
  // Number of populated output wait-state rows.
  iree_host_size_t state_count;
  // Allocated output wait-state capacity.
  iree_host_size_t state_capacity;
  // Current ordinary instruction position in the active block.
  uint64_t current_position;
} loom_amdgpu_wait_state_builder_t;

iree_string_view_t loom_amdgpu_wait_state_reason_name(
    loom_amdgpu_wait_state_reason_t reason) {
  switch (reason) {
    case LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE:
      return IREE_SV("matrix_result_use");
    case LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE:
      return IREE_SV("valu_to_matrix_use");
    case LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE:
      return IREE_SV("trans_result_use");
    case LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

static loom_amdgpu_wait_state_reason_flags_t loom_amdgpu_wait_state_reason_flag(
    loom_amdgpu_wait_state_reason_t reason) {
  switch (reason) {
    case LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE:
      return LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_MATRIX_RESULT_USE;
    case LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE:
      return LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_TO_MATRIX_USE;
    case LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE:
      return LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE;
    case LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN:
    default:
      return 0;
  }
}

static bool loom_amdgpu_wait_state_reason_is_valid(
    loom_amdgpu_wait_state_reason_t reason) {
  return reason > LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN &&
         reason < LOOM_AMDGPU_WAIT_STATE_REASON_COUNT;
}

static bool loom_amdgpu_wait_state_assignment_is_physical_vgpr(
    const loom_low_allocation_assignment_t* assignment) {
  return assignment != NULL &&
         assignment->location_kind ==
             LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER &&
         assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR;
}

static bool loom_amdgpu_wait_state_assignments_match(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

static const loom_low_allocation_assignment_t*
loom_amdgpu_wait_state_assignment(const loom_low_allocation_table_t* allocation,
                                  loom_value_id_t value_id) {
  return loom_low_allocation_try_map_active_value_assignment(allocation,
                                                             value_id, NULL);
}

static iree_status_t loom_amdgpu_wait_state_allocate(
    loom_amdgpu_wait_state_builder_t* builder) {
  const loom_low_allocation_table_t* allocation = builder->allocation;
  iree_host_size_t vgpr_count = 0;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (!loom_amdgpu_wait_state_assignment_is_physical_vgpr(assignment)) {
      continue;
    }
    const uint64_t end =
        (uint64_t)assignment->location_base + assignment->location_count;
    if (end > IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU VGPR range exceeds host size");
    }
    if ((iree_host_size_t)end > vgpr_count) {
      vgpr_count = (iree_host_size_t)end;
    }
  }
  if (vgpr_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(builder->arena, vgpr_count,
                                                   sizeof(*builder->vgprs),
                                                   (void**)&builder->vgprs));
    memset(builder->vgprs, 0, vgpr_count * sizeof(*builder->vgprs));
    builder->vgpr_count = vgpr_count;
  }

  builder->state_capacity = builder->schedule->scheduled_node_count;
  if (builder->state_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->state_capacity, sizeof(*builder->states),
        (void**)&builder->states));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_wait_state_descriptor_matches_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  return descriptor ==
         loom_amdgpu_descriptor_ref_descriptor(descriptor_set, descriptor_ref);
}

static bool loom_amdgpu_wait_state_contract_for_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    const loom_amdgpu_matrix_contract_descriptor_t** out_contract) {
  *out_contract = NULL;
  for (iree_host_size_t i = 0;
       i < loom_amdgpu_matrix_contract_descriptor_count(); ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* contract =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    if (contract == NULL ||
        contract->low_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
      continue;
    }
    if (!loom_amdgpu_wait_state_descriptor_matches_ref(
            descriptor_set, descriptor, contract->low_descriptor_ref)) {
      continue;
    }
    *out_contract = contract;
    return true;
  }
  return false;
}

static bool loom_amdgpu_wait_state_matrix_wait_cycles(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_descriptor_t* descriptor, uint16_t* out_cycle_count) {
  *out_cycle_count = 0;
  const loom_amdgpu_matrix_contract_descriptor_t* contract = NULL;
  if (!loom_amdgpu_wait_state_contract_for_descriptor(
          builder->schedule->target.descriptor_set, descriptor, &contract)) {
    return false;
  }
  switch (contract->family) {
    case LOOM_AMDGPU_MATRIX_FAMILY_MFMA:
    case LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC:
      break;
    default:
      return false;
  }
  if (contract->result_payload.register_count == 0) {
    return false;
  }

  uint16_t cycle_count =
      (uint16_t)(contract->result_payload.register_count + 3u);
  if (builder->processor != NULL &&
      builder->processor->matrix_feature_profile ==
          LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950 &&
      contract->result_payload.register_count != 2) {
    ++cycle_count;
  }
  *out_cycle_count = cycle_count;
  return true;
}

static bool loom_amdgpu_wait_state_matrix_reads_valu_results(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_descriptor_t* descriptor) {
  const loom_amdgpu_matrix_contract_descriptor_t* contract = NULL;
  if (!loom_amdgpu_wait_state_contract_for_descriptor(
          builder->schedule->target.descriptor_set, descriptor, &contract)) {
    return false;
  }
  switch (contract->family) {
    case LOOM_AMDGPU_MATRIX_FAMILY_MFMA:
    case LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_wait_state_descriptor_uses_vector_alu(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_descriptor_t* descriptor) {
  return loom_amdgpu_descriptor_uses_vector_alu(
      builder->schedule->target.descriptor_set, descriptor);
}

static bool loom_amdgpu_wait_state_processor_has_trans_forwarding_hazard(
    const loom_amdgpu_processor_info_t* processor) {
  if (processor == NULL) {
    return false;
  }
  switch (processor->matrix_feature_profile) {
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940:
    case LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_wait_state_descriptor_is_transcendental(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_descriptor_t* descriptor) {
  return loom_amdgpu_descriptor_is_transcendental(
      builder->schedule->target.descriptor_set, descriptor);
}

static bool loom_amdgpu_wait_state_descriptor_is_trans_forwarding_consumer(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_descriptor_t* descriptor) {
  return loom_amdgpu_wait_state_processor_has_trans_forwarding_hazard(
             builder->processor) &&
         loom_amdgpu_wait_state_descriptor_uses_vector_alu(builder,
                                                           descriptor) &&
         !loom_amdgpu_wait_state_descriptor_is_transcendental(builder,
                                                              descriptor);
}

static void loom_amdgpu_wait_state_clear_assignment(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment) {
  if (!loom_amdgpu_wait_state_assignment_is_physical_vgpr(assignment)) {
    return;
  }
  const uint64_t end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (end > builder->vgpr_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    builder->vgprs[assignment->location_base + i] =
        (loom_amdgpu_wait_state_vgpr_t){0};
  }
}

static void loom_amdgpu_wait_state_record_assignment(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_wait_state_reason_t reason, uint32_t producer_node,
    uint16_t cycle_count, uint64_t producer_end_position) {
  if (!loom_amdgpu_wait_state_reason_is_valid(reason)) {
    return;
  }
  if (!loom_amdgpu_wait_state_assignment_is_physical_vgpr(assignment)) {
    return;
  }
  const uint64_t end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (end > builder->vgpr_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    loom_amdgpu_wait_state_hazard_t* hazard =
        &builder->vgprs[assignment->location_base + i].hazards[reason];
    *hazard = (loom_amdgpu_wait_state_hazard_t){
        .flags = LOOM_AMDGPU_WAIT_STATE_VGPR_FLAG_VALID,
        .reason = reason,
        .producer_node = producer_node,
        .producer_end_position = producer_end_position,
        .required_cycle_count = cycle_count,
    };
  }
}

static void loom_amdgpu_wait_state_match_assignment(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  if (!loom_amdgpu_wait_state_assignment_is_physical_vgpr(assignment)) {
    return;
  }
  const uint64_t end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (end > builder->vgpr_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    const loom_amdgpu_wait_state_vgpr_t* vgpr_state =
        &builder->vgprs[assignment->location_base + i];
    for (uint32_t reason = LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN + 1;
         reason < LOOM_AMDGPU_WAIT_STATE_REASON_COUNT; ++reason) {
      const loom_amdgpu_wait_state_hazard_t* hazard =
          &vgpr_state->hazards[reason];
      if (!iree_any_bit_set(hazard->flags,
                            LOOM_AMDGPU_WAIT_STATE_VGPR_FLAG_VALID)) {
        continue;
      }
      if (!iree_any_bit_set(allowed_reasons, loom_amdgpu_wait_state_reason_flag(
                                                 hazard->reason))) {
        continue;
      }
      const uint64_t elapsed =
          builder->current_position >= hazard->producer_end_position
              ? builder->current_position - hazard->producer_end_position
              : 0;
      if (elapsed >= hazard->required_cycle_count) {
        continue;
      }
      const uint16_t remaining =
          (uint16_t)(hazard->required_cycle_count - elapsed);
      if (remaining > match->cycle_count) {
        *match = (loom_amdgpu_wait_state_match_t){
            .reason = hazard->reason,
            .producer_node = hazard->producer_node,
            .cycle_count = remaining,
        };
      }
    }
  }
}

static void loom_amdgpu_wait_state_match_value(
    const loom_amdgpu_wait_state_builder_t* builder, loom_value_id_t value_id,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_wait_state_assignment(builder->allocation, value_id);
  loom_amdgpu_wait_state_match_assignment(builder, assignment, allowed_reasons,
                                          match);
}

static iree_status_t loom_amdgpu_wait_state_append(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_wait_state_match_t* match) {
  if (match->cycle_count == 0) {
    return iree_ok_status();
  }
  if (builder->state_count >= builder->state_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU wait-state plan capacity exhausted");
  }
  builder->states[builder->state_count++] = (loom_amdgpu_wait_state_t){
      .reason = match->reason,
      .block_index = packet->node->block_index,
      .node_index = packet->node_index,
      .scheduled_ordinal = packet->node->scheduled_ordinal,
      .producer_node = match->producer_node,
      .consumer_node = packet->node->source_ordinal,
      .cycle_count = match->cycle_count,
  };
  builder->current_position += match->cycle_count;
  return iree_ok_status();
}

static void loom_amdgpu_wait_state_match_descriptor_operands(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_op_t* op = packet->node->op;
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_amdgpu_wait_state_match_value(builder, loom_op_const_operands(op)[i],
                                       allowed_reasons, match);
  }
}

static void loom_amdgpu_wait_state_clear_results(
    loom_amdgpu_wait_state_builder_t* builder, const loom_op_t* op) {
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_state_assignment(builder->allocation,
                                          loom_op_const_results(op)[i]);
    loom_amdgpu_wait_state_clear_assignment(builder, assignment);
  }
}

static void loom_amdgpu_wait_state_record_results(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_reason_t reason, uint16_t cycle_count,
    uint64_t instruction_count) {
  const loom_op_t* op = packet->node->op;
  const uint64_t producer_end_position =
      builder->current_position + instruction_count;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_state_assignment(builder->allocation,
                                          loom_op_const_results(op)[i]);
    loom_amdgpu_wait_state_record_assignment(builder, assignment, reason,
                                             packet->node_index, cycle_count,
                                             producer_end_position);
  }
}

static iree_status_t loom_amdgpu_wait_state_copy_materializes(
    const loom_amdgpu_wait_state_builder_t* builder, const loom_op_t* op,
    bool* out_materializes) {
  *out_materializes = false;
  const loom_low_allocation_assignment_t* source_assignment =
      loom_amdgpu_wait_state_assignment(builder->allocation,
                                        loom_low_copy_source(op));
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_wait_state_assignment(builder->allocation,
                                        loom_low_copy_result(op));
  if (source_assignment == NULL || result_assignment == NULL) {
    return iree_ok_status();
  }
  if (source_assignment->location_count != result_assignment->location_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU low.copy allocation is malformed");
  }
  *out_materializes = !loom_amdgpu_wait_state_assignments_match(
      source_assignment, result_assignment);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_state_structural_materializes(
    const loom_amdgpu_wait_state_builder_t* builder, const loom_op_t* op,
    bool* out_materializes) {
  *out_materializes = false;
  if (loom_low_copy_isa(op)) {
    return loom_amdgpu_wait_state_copy_materializes(builder, op,
                                                    out_materializes);
  }
  if (loom_low_slice_isa(op)) {
    iree_host_size_t move_count = 0;
    IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_slice_units(
        builder->allocation, op, &move_count));
    *out_materializes = move_count != 0;
    return iree_ok_status();
  }
  if (loom_low_concat_isa(op)) {
    iree_host_size_t move_count = 0;
    IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_concat_units(
        builder->allocation, op, &move_count));
    *out_materializes = move_count != 0;
    return iree_ok_status();
  }
  if (loom_low_live_in_isa(op) || loom_low_storage_reserve_isa(op)) {
    return iree_ok_status();
  }
  *out_materializes = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_state_structural_instruction_count(
    const loom_amdgpu_wait_state_builder_t* builder, const loom_op_t* op,
    uint64_t* out_instruction_count) {
  *out_instruction_count = 0;
  if (loom_low_copy_isa(op)) {
    bool materializes = false;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_state_copy_materializes(builder, op, &materializes));
    if (materializes) {
      const loom_low_allocation_assignment_t* assignment =
          loom_amdgpu_wait_state_assignment(builder->allocation,
                                            loom_low_copy_result(op));
      *out_instruction_count = assignment->location_count;
    }
    return iree_ok_status();
  }
  if (loom_low_slice_isa(op)) {
    iree_host_size_t move_count = 0;
    IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_slice_units(
        builder->allocation, op, &move_count));
    *out_instruction_count = move_count;
    return iree_ok_status();
  }
  if (loom_low_concat_isa(op)) {
    iree_host_size_t move_count = 0;
    IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_concat_units(
        builder->allocation, op, &move_count));
    *out_instruction_count = move_count;
    return iree_ok_status();
  }
  bool materializes = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_structural_materializes(
      builder, op, &materializes));
  *out_instruction_count = materializes ? 1 : 0;
  return iree_ok_status();
}

static bool loom_amdgpu_wait_state_structural_writes_valu(
    const loom_op_t* op, uint64_t instruction_count) {
  if (instruction_count == 0) {
    return false;
  }
  return loom_low_copy_isa(op) || loom_low_slice_isa(op) ||
         loom_low_concat_isa(op);
}

static iree_status_t loom_amdgpu_wait_state_match_structural_operands(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_op_t* op = packet->node->op;
  bool materializes = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_structural_materializes(
      builder, op, &materializes));
  if (loom_low_br_isa(op)) {
    loom_value_slice_t args = loom_low_br_args(op);
    for (uint16_t i = 0; i < args.count; ++i) {
      loom_amdgpu_wait_state_match_value(builder, args.values[i],
                                         allowed_reasons, match);
    }
    return iree_ok_status();
  }
  if (!materializes) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_amdgpu_wait_state_match_value(builder, loom_op_const_operands(op)[i],
                                       allowed_reasons, match);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_state_apply_packet(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet) {
  uint16_t matrix_wait_cycles = 0;
  const bool is_matrix = packet->descriptor != NULL &&
                         loom_amdgpu_wait_state_matrix_wait_cycles(
                             builder, packet->descriptor, &matrix_wait_cycles);
  const bool matrix_reads_valu =
      packet->descriptor != NULL &&
      loom_amdgpu_wait_state_matrix_reads_valu_results(builder,
                                                       packet->descriptor);
  const bool trans_producer =
      packet->descriptor != NULL &&
      loom_amdgpu_wait_state_processor_has_trans_forwarding_hazard(
          builder->processor) &&
      loom_amdgpu_wait_state_descriptor_is_transcendental(builder,
                                                          packet->descriptor);
  const bool trans_forwarding_consumer =
      packet->descriptor != NULL &&
      loom_amdgpu_wait_state_descriptor_is_trans_forwarding_consumer(
          builder, packet->descriptor);

  loom_amdgpu_wait_state_match_t match = {0};
  if (matrix_reads_valu) {
    loom_amdgpu_wait_state_match_descriptor_operands(
        builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_TO_MATRIX_USE,
        &match);
  } else if (packet->descriptor != NULL) {
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons =
        LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_MATRIX_RESULT_USE;
    if (trans_forwarding_consumer) {
      allowed_reasons |= LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE;
    }
    loom_amdgpu_wait_state_match_descriptor_operands(builder, packet,
                                                     allowed_reasons, &match);
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_match_structural_operands(
        builder, packet,
        LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_MATRIX_RESULT_USE |
            LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE,
        &match));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_append(builder, packet, &match));

  uint64_t instruction_count = 1;
  if (packet->descriptor == NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_structural_instruction_count(
        builder, packet->node->op, &instruction_count));
  }
  if (is_matrix) {
    loom_amdgpu_wait_state_clear_results(builder, packet->node->op);
    loom_amdgpu_wait_state_record_results(
        builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE,
        matrix_wait_cycles, instruction_count);
  } else if (packet->descriptor != NULL &&
             loom_amdgpu_wait_state_descriptor_uses_vector_alu(
                 builder, packet->descriptor)) {
    loom_amdgpu_wait_state_clear_results(builder, packet->node->op);
    loom_amdgpu_wait_state_record_results(
        builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE,
        LOOM_AMDGPU_WAIT_STATE_VALU_TO_MATRIX_CYCLES, instruction_count);
    if (trans_producer) {
      loom_amdgpu_wait_state_record_results(
          builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE,
          LOOM_AMDGPU_WAIT_STATE_TRANS_RESULT_USE_CYCLES, instruction_count);
    }
  } else if (packet->descriptor != NULL) {
    loom_amdgpu_wait_state_clear_results(builder, packet->node->op);
  } else {
    if (loom_amdgpu_wait_state_structural_writes_valu(packet->node->op,
                                                      instruction_count)) {
      loom_amdgpu_wait_state_clear_results(builder, packet->node->op);
      loom_amdgpu_wait_state_record_results(
          builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE,
          LOOM_AMDGPU_WAIT_STATE_VALU_TO_MATRIX_CYCLES, instruction_count);
    } else if (instruction_count != 0) {
      loom_amdgpu_wait_state_clear_results(builder, packet->node->op);
    }
  }
  builder->current_position += instruction_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_state_plan_build_with_scratch(
    loom_amdgpu_wait_state_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_allocate(builder));
  builder->processor = loom_amdgpu_target_processor_from_resolved_target(
      builder->schedule->module, &builder->schedule->target);
  for (iree_host_size_t block_index = 0;
       block_index < builder->schedule->block_count; ++block_index) {
    if (builder->vgpr_count != 0) {
      memset(builder->vgprs, 0, builder->vgpr_count * sizeof(*builder->vgprs));
    }
    builder->current_position = 0;
    const loom_low_schedule_block_t* block =
        &builder->schedule->blocks[block_index];
    for (uint32_t scheduled_ordinal = 0;
         scheduled_ordinal < block->scheduled_node_count; ++scheduled_ordinal) {
      const iree_host_size_t packet_index =
          block->scheduled_node_start + scheduled_ordinal;
      loom_low_packet_view_t packet = {0};
      IREE_RETURN_IF_ERROR(loom_low_packet_view_at(
          builder->schedule, builder->allocation, packet_index, &packet));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_wait_state_apply_packet(builder, &packet));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_wait_state_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, loom_amdgpu_wait_state_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_wait_state_plan_t){0};
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_tables(schedule, allocation));

  loom_low_allocation_value_scratch_t scratch = {0};
  iree_status_t status =
      loom_low_allocation_acquire_value_scratch(allocation, &scratch);
  loom_amdgpu_wait_state_builder_t builder = {
      .schedule = schedule,
      .allocation = allocation,
      .arena = arena,
  };
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_state_plan_build_with_scratch(&builder);
  }
  if (iree_status_is_ok(status)) {
    *out_plan = (loom_amdgpu_wait_state_plan_t){
        .schedule = schedule,
        .allocation = allocation,
        .states = builder.states,
        .state_count = builder.state_count,
    };
  }
  loom_low_allocation_release_value_scratch(&scratch);
  return status;
}

uint64_t loom_amdgpu_wait_state_plan_instruction_count(
    const loom_amdgpu_wait_state_plan_t* plan) {
  if (plan == NULL) {
    return 0;
  }
  uint64_t instruction_count = 0;
  for (iree_host_size_t i = 0; i < plan->state_count; ++i) {
    const uint16_t cycle_count = plan->states[i].cycle_count;
    instruction_count +=
        (cycle_count + LOOM_AMDGPU_WAIT_STATE_MAX_S_NOP_CYCLES - 1u) /
        LOOM_AMDGPU_WAIT_STATE_MAX_S_NOP_CYCLES;
  }
  return instruction_count;
}
