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
#include "loom/util/json.h"
#include "loom/util/stream.h"

#define LOOM_AMDGPU_WAIT_STATE_VALU_TO_MATRIX_CYCLES 2u
#define LOOM_AMDGPU_WAIT_STATE_TRANS_RESULT_USE_CYCLES 1u
#define LOOM_AMDGPU_WAIT_STATE_VALU_SGPR_READ_CYCLES 2u
#define LOOM_AMDGPU_WAIT_STATE_REASON_COUNT 5u

enum {
  LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_INSTRUCTION_SLOT = 1,
};

typedef enum loom_amdgpu_wait_state_vgpr_flag_bits_e {
  LOOM_AMDGPU_WAIT_STATE_VGPR_FLAG_VALID = 1u << 0,
} loom_amdgpu_wait_state_vgpr_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_state_vgpr_flags_t;

typedef enum loom_amdgpu_wait_state_sgpr_flag_bits_e {
  LOOM_AMDGPU_WAIT_STATE_SGPR_FLAG_VALID = 1u << 0,
} loom_amdgpu_wait_state_sgpr_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_state_sgpr_flags_t;

typedef enum loom_amdgpu_wait_state_reason_flag_bits_e {
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_MATRIX_RESULT_USE =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_TO_MATRIX_USE =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_SGPR_READ =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ,
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

typedef struct loom_amdgpu_wait_state_sgpr_t {
  // Per-reason outstanding fixed-wait hazard state for this physical SGPR.
  loom_amdgpu_wait_state_hazard_t hazards[LOOM_AMDGPU_WAIT_STATE_REASON_COUNT];
} loom_amdgpu_wait_state_sgpr_t;

typedef struct loom_amdgpu_wait_state_match_t {
  // Reason responsible for the largest unsatisfied wait.
  loom_amdgpu_wait_state_reason_t reason;
  // Producer node responsible for the largest unsatisfied wait.
  uint32_t producer_node;
  // Required target progress before the current consumer.
  uint16_t required_cycle_count;
  // Target progress already supplied before the current consumer.
  uint16_t observed_cycle_count;
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
  // Per-physical-SGPR outstanding fixed-wait hazard state.
  loom_amdgpu_wait_state_sgpr_t* sgprs;
  // Number of entries in |sgprs|.
  iree_host_size_t sgpr_count;
  // Output wait-state rows.
  loom_amdgpu_wait_state_t* states;
  // Number of populated output wait-state rows.
  iree_host_size_t state_count;
  // Allocated output wait-state capacity.
  iree_host_size_t state_capacity;
  // Target progress facts for scheduled native instruction slots.
  loom_low_packet_progress_table_t progress;
  // Common residual hazard records for emitted wait states.
  loom_low_packet_hazard_plan_t hazard_plan;
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
    case LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ:
      return IREE_SV("valu_sgpr_read");
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
    case LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ:
      return LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_SGPR_READ;
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
  return loom_low_allocation_assignment_is_physical_register_class(
      assignment, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
}

static bool loom_amdgpu_wait_state_assignment_is_physical_sgpr(
    const loom_low_allocation_assignment_t* assignment) {
  return loom_low_allocation_assignment_is_physical_register_class(
      assignment, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
}

static bool loom_amdgpu_wait_state_assignments_match(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return loom_low_allocation_assignments_match(lhs, rhs);
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
  iree_host_size_t sgpr_count = 0;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (!loom_amdgpu_wait_state_assignment_is_physical_vgpr(assignment) &&
        !loom_amdgpu_wait_state_assignment_is_physical_sgpr(assignment)) {
      continue;
    }
    uint64_t end = 0;
    if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                               &end)) {
      continue;
    }
    if (end > IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU register range exceeds host size");
    }
    if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR &&
        (iree_host_size_t)end > vgpr_count) {
      vgpr_count = (iree_host_size_t)end;
    }
    if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR &&
        (iree_host_size_t)end > sgpr_count) {
      sgpr_count = (iree_host_size_t)end;
    }
  }
  if (vgpr_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(builder->arena, vgpr_count,
                                                   sizeof(*builder->vgprs),
                                                   (void**)&builder->vgprs));
    memset(builder->vgprs, 0, vgpr_count * sizeof(*builder->vgprs));
    builder->vgpr_count = vgpr_count;
  }
  if (sgpr_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(builder->arena, sgpr_count,
                                                   sizeof(*builder->sgprs),
                                                   (void**)&builder->sgprs));
    memset(builder->sgprs, 0, sgpr_count * sizeof(*builder->sgprs));
    builder->sgpr_count = sgpr_count;
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

static bool loom_amdgpu_wait_state_processor_has_valu_sgpr_read_hazard(
    const loom_amdgpu_processor_info_t* processor) {
  return processor != NULL && processor->has_valu_sgpr_read_wait_states;
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
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_vgpr(assignment)) {
    if (end > builder->vgpr_count) {
      return;
    }
    for (uint32_t i = 0; i < assignment->location_count; ++i) {
      builder->vgprs[assignment->location_base + i] =
          (loom_amdgpu_wait_state_vgpr_t){0};
    }
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_sgpr(assignment)) {
    if (end > builder->sgpr_count) {
      return;
    }
    for (uint32_t i = 0; i < assignment->location_count; ++i) {
      builder->sgprs[assignment->location_base + i] =
          (loom_amdgpu_wait_state_sgpr_t){0};
    }
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
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
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

static void loom_amdgpu_wait_state_record_sgpr_assignment(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_wait_state_reason_t reason, uint32_t producer_node,
    uint16_t cycle_count, uint64_t producer_end_position) {
  if (!loom_amdgpu_wait_state_reason_is_valid(reason)) {
    return;
  }
  if (!loom_amdgpu_wait_state_assignment_is_physical_sgpr(assignment)) {
    return;
  }
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
  if (end > builder->sgpr_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    loom_amdgpu_wait_state_hazard_t* hazard =
        &builder->sgprs[assignment->location_base + i].hazards[reason];
    *hazard = (loom_amdgpu_wait_state_hazard_t){
        .flags = LOOM_AMDGPU_WAIT_STATE_SGPR_FLAG_VALID,
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
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
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
            .required_cycle_count = hazard->required_cycle_count,
            .observed_cycle_count = (uint16_t)elapsed,
            .cycle_count = remaining,
        };
      }
    }
  }
}

static void loom_amdgpu_wait_state_match_sgpr_assignment(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  if (!loom_amdgpu_wait_state_assignment_is_physical_sgpr(assignment)) {
    return;
  }
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
  if (end > builder->sgpr_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    const loom_amdgpu_wait_state_sgpr_t* sgpr_state =
        &builder->sgprs[assignment->location_base + i];
    for (uint32_t reason = LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN + 1;
         reason < LOOM_AMDGPU_WAIT_STATE_REASON_COUNT; ++reason) {
      const loom_amdgpu_wait_state_hazard_t* hazard =
          &sgpr_state->hazards[reason];
      if (!iree_any_bit_set(hazard->flags,
                            LOOM_AMDGPU_WAIT_STATE_SGPR_FLAG_VALID)) {
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
            .required_cycle_count = hazard->required_cycle_count,
            .observed_cycle_count = (uint16_t)elapsed,
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

static void loom_amdgpu_wait_state_match_sgpr_value(
    const loom_amdgpu_wait_state_builder_t* builder, loom_value_id_t value_id,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_wait_state_assignment(builder->allocation, value_id);
  loom_amdgpu_wait_state_match_sgpr_assignment(builder, assignment,
                                               allowed_reasons, match);
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
      .required_cycle_count = match->required_cycle_count,
      .observed_cycle_count = match->observed_cycle_count,
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

static void loom_amdgpu_wait_state_match_descriptor_sgpr_operands(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_op_t* op = packet->node->op;
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_amdgpu_wait_state_match_sgpr_value(
        builder, loom_op_const_operands(op)[i], allowed_reasons, match);
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

static void loom_amdgpu_wait_state_record_sgpr_results(
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
    loom_amdgpu_wait_state_record_sgpr_assignment(
        builder, assignment, reason, packet->node_index, cycle_count,
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

static iree_status_t loom_amdgpu_wait_state_packet_instruction_count(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet, uint64_t* out_instruction_count) {
  if (packet->descriptor != NULL) {
    *out_instruction_count = 1;
    return iree_ok_status();
  }
  return loom_amdgpu_wait_state_structural_instruction_count(
      builder, packet->node->op, out_instruction_count);
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
  const bool valu_sgpr_read_hazard =
      packet->descriptor != NULL &&
      loom_amdgpu_wait_state_processor_has_valu_sgpr_read_hazard(
          builder->processor) &&
      loom_amdgpu_wait_state_descriptor_uses_vector_alu(builder,
                                                        packet->descriptor);

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
    if (valu_sgpr_read_hazard) {
      loom_amdgpu_wait_state_match_descriptor_sgpr_operands(
          builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_SGPR_READ,
          &match);
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_match_structural_operands(
        builder, packet,
        LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_MATRIX_RESULT_USE |
            LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE,
        &match));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_append(builder, packet, &match));

  uint64_t instruction_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_packet_instruction_count(
      builder, packet, &instruction_count));
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
    if (valu_sgpr_read_hazard) {
      loom_amdgpu_wait_state_record_sgpr_results(
          builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ,
          LOOM_AMDGPU_WAIT_STATE_VALU_SGPR_READ_CYCLES, instruction_count);
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

static iree_status_t loom_amdgpu_wait_state_progress_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet,
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data) {
  (void)schedule;
  (void)allocation;
  const loom_amdgpu_wait_state_builder_t* builder =
      (const loom_amdgpu_wait_state_builder_t*)user_data;
  uint64_t instruction_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_packet_instruction_count(
      builder, packet, &instruction_count));
  if (instruction_count == 0) {
    return iree_ok_status();
  }
  if (instruction_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait-state progress exceeds uint32_t units");
  }
  const loom_low_packet_progress_event_t event = {
      .progress_class_id =
          LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_INSTRUCTION_SLOT,
      .progress_class_name = IREE_SV("amdgpu.instruction_slot"),
      .action = LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE,
      .units = (uint32_t)instruction_count,
  };
  return emit(emit_user_data, &event);
}

static iree_status_t loom_amdgpu_wait_state_build_progress(
    loom_amdgpu_wait_state_builder_t* builder) {
  const loom_low_packet_progress_provider_t provider = {
      .user_data = builder,
      .query = loom_amdgpu_wait_state_progress_query,
  };
  return loom_low_packet_progress_build(builder->schedule, builder->allocation,
                                        &provider, builder->arena,
                                        &builder->progress);
}

static bool loom_amdgpu_wait_state_matches_packet(
    const loom_amdgpu_wait_state_t* wait_state,
    const loom_low_packet_view_t* packet) {
  return wait_state->block_index == packet->node->block_index &&
         wait_state->scheduled_ordinal == packet->node->scheduled_ordinal &&
         wait_state->node_index == packet->node_index;
}

static iree_status_t loom_amdgpu_wait_state_hazard_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_view_t* packet,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  (void)schedule;
  (void)allocation;
  (void)progress;
  const loom_amdgpu_wait_state_builder_t* builder =
      (const loom_amdgpu_wait_state_builder_t*)user_data;
  for (iree_host_size_t i = 0; i < builder->state_count; ++i) {
    const loom_amdgpu_wait_state_t* wait_state = &builder->states[i];
    if (!loom_amdgpu_wait_state_matches_packet(wait_state, packet)) {
      continue;
    }
    const loom_low_packet_hazard_plan_event_t event = {
        .kind = LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION,
        .reason_id = (uint16_t)wait_state->reason,
        .reason_name = loom_amdgpu_wait_state_reason_name(wait_state->reason),
        .producer_node_index = wait_state->producer_node,
        .progress_class_id =
            LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_INSTRUCTION_SLOT,
        .progress_class_name = IREE_SV("amdgpu.instruction_slot"),
        .required_progress = wait_state->required_cycle_count,
        .observed_progress = wait_state->observed_cycle_count,
        .residual_progress = wait_state->cycle_count,
    };
    IREE_RETURN_IF_ERROR(emit(emit_user_data, &event));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_state_build_hazard_plan(
    loom_amdgpu_wait_state_builder_t* builder) {
  const loom_low_packet_hazard_plan_provider_t provider = {
      .user_data = builder,
      .query = loom_amdgpu_wait_state_hazard_query,
  };
  return loom_low_packet_hazard_plan_build(
      builder->schedule, builder->allocation, &builder->progress, &provider,
      builder->arena, &builder->hazard_plan);
}

static iree_status_t loom_amdgpu_wait_state_plan_build_with_scratch(
    loom_amdgpu_wait_state_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_allocate(builder));
  builder->processor = loom_amdgpu_target_processor_from_resolved_target(
      builder->schedule->module, &builder->schedule->target);
  for (iree_host_size_t block_index = 0;
       block_index < builder->schedule->block_count; ++block_index) {
    if (block_index > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait-state block index exceeds uint32_t");
    }
    if (builder->vgpr_count != 0) {
      memset(builder->vgprs, 0, builder->vgpr_count * sizeof(*builder->vgprs));
    }
    if (builder->sgpr_count != 0) {
      memset(builder->sgprs, 0, builder->sgpr_count * sizeof(*builder->sgprs));
    }
    builder->current_position = 0;
    const loom_low_schedule_block_t* block =
        &builder->schedule->blocks[block_index];
    for (uint32_t scheduled_ordinal = 0;
         scheduled_ordinal < block->scheduled_node_count; ++scheduled_ordinal) {
      loom_low_packet_view_t packet = {0};
      IREE_RETURN_IF_ERROR(loom_low_packet_view_at_block_ordinal(
          builder->schedule, builder->allocation, (uint32_t)block_index,
          scheduled_ordinal, &packet));
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_wait_state_apply_packet(builder, &packet));
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_build_progress(builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_build_hazard_plan(builder));
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
        .progress = builder.progress,
        .hazard_plan = builder.hazard_plan,
        .states = builder.states,
        .state_count = builder.state_count,
    };
    out_plan->hazard_plan.progress = &out_plan->progress;
  }
  loom_low_allocation_release_value_scratch(&scratch);
  return status;
}

static iree_string_view_t loom_amdgpu_wait_state_progress_action_name(
    loom_low_packet_progress_action_t action) {
  switch (action) {
    case LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE:
      return IREE_SV("advance");
    case LOOM_LOW_PACKET_PROGRESS_ACTION_RESET:
      return IREE_SV("reset");
    case LOOM_LOW_PACKET_PROGRESS_ACTION_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_amdgpu_wait_state_hazard_kind_name(
    loom_low_packet_hazard_plan_record_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION:
      return IREE_SV("action");
    case LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_MISSING_TARGET_DATA:
      return IREE_SV("missing_target_data");
    case LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_UNSUPPORTED_PRE_ALLOCATION:
      return IREE_SV("unsupported_pre_allocation");
    case LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_IMPOSSIBLE_SATISFACTION:
      return IREE_SV("impossible_satisfaction");
    case LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t loom_amdgpu_wait_state_write_states_json(
    const loom_amdgpu_wait_state_plan_t* plan, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
  for (iree_host_size_t i = 0; i < plan->state_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    const loom_amdgpu_wait_state_t* state = &plan->states[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "{\"index\":%zu,\"reason\":%" PRIu32 ",\"reason_name\":", i,
        (uint32_t)state->reason));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, loom_amdgpu_wait_state_reason_name(state->reason)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"block\":%" PRIu32 ",\"node\":%" PRIu32
        ",\"scheduled_ordinal\":%" PRIu32 ",\"producer_node\":%" PRIu32
        ",\"consumer_node\":%" PRIu32 ",\"required\":%" PRIu16
        ",\"observed\":%" PRIu16 ",\"residual\":%" PRIu16 "}",
        state->block_index, state->node_index, state->scheduled_ordinal,
        state->producer_node, state->consumer_node, state->required_cycle_count,
        state->observed_cycle_count, state->cycle_count));
  }
  return loom_output_stream_write_char(stream, ']');
}

static iree_status_t loom_amdgpu_wait_state_write_progress_json(
    const loom_low_packet_progress_table_t* progress,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
  for (iree_host_size_t i = 0; i < progress->record_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    const loom_low_packet_progress_record_t* record = &progress->records[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        "{\"index\":%zu,\"packet\":%zu,\"node\":%" PRIu32 ",\"block\":%" PRIu32
        ",\"scheduled_ordinal\":%" PRIu32 ",\"class_id\":%" PRIu16
        ",\"class_name\":",
        i, record->packet_index, record->node_index, record->block_index,
        record->scheduled_ordinal, record->progress_class_id));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(stream, record->progress_class_name));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"action\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, loom_amdgpu_wait_state_progress_action_name(record->action)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"units\":%" PRIu32 "}", record->units));
  }
  return loom_output_stream_write_char(stream, ']');
}

static iree_status_t loom_amdgpu_wait_state_write_hazards_json(
    const loom_low_packet_hazard_plan_t* hazard_plan,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
  for (iree_host_size_t i = 0; i < hazard_plan->record_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    const loom_low_packet_hazard_plan_record_t* record =
        &hazard_plan->records[i];
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_format(stream, "{\"index\":%zu,\"kind\":", i));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, loom_amdgpu_wait_state_hazard_kind_name(record->kind)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"reason_id\":%" PRIu16 ",\"reason_name\":", record->reason_id));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(stream, record->reason_name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"producer_node\":%" PRIu32
        ",\"producer_packet\":%zu"
        ",\"producer_scheduled_ordinal\":%" PRIu32 ",\"consumer_node\":%" PRIu32
        ",\"insertion_packet\":%zu"
        ",\"block\":%" PRIu32 ",\"scheduled_ordinal\":%" PRIu32
        ",\"progress_class_id\":%" PRIu16 ",\"progress_class_name\":",
        record->producer_node_index, record->producer_packet_index,
        record->producer_scheduled_ordinal, record->consumer_node_index,
        record->insertion_packet_index, record->block_index,
        record->scheduled_ordinal, record->progress_class_id));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(stream, record->progress_class_name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"required\":%" PRIu32 ",\"observed\":%" PRIu32
        ",\"residual\":%" PRIu32 "}",
        record->required_progress, record->observed_progress,
        record->residual_progress));
  }
  return loom_output_stream_write_char(stream, ']');
}

iree_status_t loom_amdgpu_wait_state_plan_format_json(
    const loom_amdgpu_wait_state_plan_t* plan, iree_string_builder_t* builder) {
  if (plan == NULL || builder == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait-state plan and builder are required");
  }
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      "{\"format\":\"loom.amdgpu.wait_state_plan.v0\",\"state_count\":%zu"
      ",\"progress_count\":%zu,\"hazard_count\":%zu,\"states\":",
      plan->state_count, plan->progress.record_count,
      plan->hazard_plan.record_count));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_write_states_json(plan, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"progress\":"));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_state_write_progress_json(&plan->progress, &stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"hazards\":"));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_state_write_hazards_json(&plan->hazard_plan, &stream));
  return loom_output_stream_write_char(&stream, '}');
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
