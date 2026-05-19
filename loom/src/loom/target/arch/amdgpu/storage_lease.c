// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/storage_lease.h"

#include <inttypes.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/target/arch/amdgpu/target_refs.h"
#include "loom/target/arch/amdgpu/wait_plan.h"

typedef struct loom_amdgpu_storage_lease_counter_masks_t {
  // Wait counters produced by dependency-participating memory reads.
  uint32_t read_mask;
  // Wait counters produced by dependency-participating memory writes.
  uint32_t write_mask;
} loom_amdgpu_storage_lease_counter_masks_t;

static bool loom_amdgpu_storage_lease_effect_is_dependency_memory(
    const loom_low_effect_t* effect) {
  if (!iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY)) {
    return false;
  }
  switch (effect->memory_space) {
    case LOOM_LOW_MEMORY_SPACE_GENERIC:
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
    case LOOM_LOW_MEMORY_SPACE_STACK:
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_storage_lease_descriptor_hazard_counter_mask(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t* out_counter_mask) {
  *out_counter_mask = 0;
  if (descriptor->schedule_class_id >= descriptor_set->schedule_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU descriptor schedule class id %" PRIu16
                            " exceeds schedule-class table size %" PRIu32,
                            descriptor->schedule_class_id,
                            descriptor_set->schedule_class_count);
  }
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[descriptor->schedule_class_id];
  if (schedule_class->hazard_start > descriptor_set->hazard_count ||
      schedule_class->hazard_count >
          descriptor_set->hazard_count - schedule_class->hazard_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU descriptor hazard range is out of bounds");
  }
  for (uint16_t i = 0; i < schedule_class->hazard_count; ++i) {
    const loom_low_hazard_t* hazard =
        &descriptor_set->hazards[schedule_class->hazard_start + i];
    if (hazard->kind != LOOM_LOW_HAZARD_KIND_WAIT_COUNTER) {
      continue;
    }
    if (hazard->reference_kind != LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU wait-counter hazard does not reference "
                              "a counter");
    }
    uint32_t counter_mask = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_counter_mask(hazard->reference_id, &counter_mask));
    *out_counter_mask |= counter_mask;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_storage_lease_effect_counter_mask(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, const loom_low_effect_t* effect,
    uint32_t default_counter_mask, uint32_t allowed_counter_mask,
    uint32_t* out_counter_mask) {
  if (effect->counter_id == LOOM_AMDGPU_WAIT_COUNTER_NONE) {
    const uint32_t counter_mask = default_counter_mask & allowed_counter_mask;
    if (counter_mask == 0) {
      iree_string_view_t descriptor_key = loom_low_descriptor_set_string(
          descriptor_set, descriptor->key_string_offset);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU dependency memory effect on descriptor "
                              "'%.*s' has no matching wait-counter hazard",
                              (int)descriptor_key.size, descriptor_key.data);
    }
    *out_counter_mask = counter_mask;
    return iree_ok_status();
  }
  uint32_t counter_mask = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_counter_mask(effect->counter_id, &counter_mask));
  *out_counter_mask = counter_mask;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_storage_lease_counter_masks(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    loom_amdgpu_storage_lease_counter_masks_t* out_masks) {
  *out_masks = (loom_amdgpu_storage_lease_counter_masks_t){0};
  if (descriptor->effect_start > descriptor_set->effect_count ||
      descriptor->effect_count >
          descriptor_set->effect_count - descriptor->effect_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU descriptor effect range is out of bounds");
  }
  uint32_t hazard_counter_mask = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_lease_descriptor_hazard_counter_mask(
      descriptor_set, descriptor, &hazard_counter_mask));
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (!loom_amdgpu_storage_lease_effect_is_dependency_memory(effect)) {
      continue;
    }
    switch (effect->kind) {
      case LOOM_LOW_EFFECT_KIND_READ: {
        uint32_t counter_mask = 0;
        IREE_RETURN_IF_ERROR(loom_amdgpu_storage_lease_effect_counter_mask(
            descriptor_set, descriptor, effect, hazard_counter_mask,
            LOOM_AMDGPU_WAIT_COUNTER_MASK_READ, &counter_mask));
        out_masks->read_mask |= counter_mask;
        break;
      }
      case LOOM_LOW_EFFECT_KIND_WRITE: {
        uint32_t counter_mask = 0;
        IREE_RETURN_IF_ERROR(loom_amdgpu_storage_lease_effect_counter_mask(
            descriptor_set, descriptor, effect, hazard_counter_mask,
            LOOM_AMDGPU_WAIT_COUNTER_MASK_WRITE, &counter_mask));
        out_masks->write_mask |= counter_mask;
        break;
      }
      default:
        break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_storage_lease_operand_accepts_vgpr(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, bool* out_accepts_vgpr) {
  *out_accepts_vgpr = false;
  if (operand->reg_class_alt_start > descriptor_set->reg_class_alt_count ||
      operand->reg_class_alt_count >
          descriptor_set->reg_class_alt_count - operand->reg_class_alt_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU descriptor operand register-class range is out of bounds");
  }
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[operand->reg_class_alt_start + i];
    if (alt->reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR &&
        !iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      *out_accepts_vgpr = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_storage_lease_emit_event(
    loom_low_storage_lease_emit_fn_t emit, void* emit_user_data,
    loom_low_storage_lease_kind_t kind,
    loom_low_storage_lease_attachment_t attachment, uint16_t attachment_index,
    uint32_t unit_count, uint16_t release_class_id,
    loom_amdgpu_wait_plan_reason_t reason) {
  const loom_low_storage_lease_event_t event = {
      .kind = kind,
      .attachment = attachment,
      .attachment_index = attachment_index,
      .unit_offset = 0,
      .unit_count = unit_count,
      .release_scope = LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS,
      .release_class_id = release_class_id,
      .release_class_name =
          loom_amdgpu_wait_counter_progress_class_name(release_class_id),
      .release_action_id = LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET,
      .release_action_name = loom_amdgpu_wait_plan_residual_action_name(
          LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET),
      .release_reason_id = (uint16_t)reason,
      .release_reason_name = loom_amdgpu_wait_plan_reason_name(reason),
      .flags = LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE |
               LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_BEFORE_BOUNDARY,
  };
  return emit(emit_user_data, &event);
}

static iree_status_t loom_amdgpu_storage_lease_emit_result_leases(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_node_t* node, uint32_t read_counter_mask,
    loom_low_storage_lease_emit_fn_t emit, void* emit_user_data) {
  const loom_low_descriptor_t* descriptor = node->descriptor;
  if (read_counter_mask == 0 || node->result_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->operand_start > descriptor_set->operand_count ||
      descriptor->result_count >
          descriptor_set->operand_count - descriptor->operand_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU descriptor result range is out of bounds");
  }
  for (uint16_t result_index = 0; result_index < node->result_count &&
                                  result_index < descriptor->result_count;
       ++result_index) {
    const loom_low_operand_t* result =
        &descriptor_set->operands[descriptor->operand_start + result_index];
    if (result->unit_count == 0) {
      continue;
    }
    for (uint16_t counter_id = LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD;
         counter_id <= LOOM_AMDGPU_WAIT_COUNTER_ALU; ++counter_id) {
      uint32_t counter_mask = 0;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_wait_counter_mask(counter_id, &counter_mask));
      if ((read_counter_mask & counter_mask) == 0) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_storage_lease_emit_event(
          emit, emit_user_data, LOOM_LOW_STORAGE_LEASE_RESULT_WRITE,
          LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT, result_index,
          result->unit_count, counter_id,
          LOOM_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_storage_lease_emit_store_source_leases(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_node_t* node, uint32_t write_counter_mask,
    loom_low_storage_lease_emit_fn_t emit, void* emit_user_data) {
  const loom_low_descriptor_t* descriptor = node->descriptor;
  if ((write_counter_mask & LOOM_AMDGPU_WAIT_COUNTER_MASK_VMEM_STORE) == 0 ||
      node->operand_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->operand_start > descriptor_set->operand_count ||
      descriptor->operand_count >
          descriptor_set->operand_count - descriptor->operand_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU descriptor operand range is out of bounds");
  }
  uint16_t packet_operand_index = 0;
  for (uint16_t descriptor_operand_index = descriptor->result_count;
       descriptor_operand_index < descriptor->operand_count;
       ++descriptor_operand_index) {
    const loom_low_operand_t* operand =
        &descriptor_set
             ->operands[descriptor->operand_start + descriptor_operand_index];
    if (!loom_low_operand_role_is_packet_operand(operand->role)) {
      continue;
    }
    const uint16_t current_packet_operand_index = packet_operand_index++;
    bool accepts_vgpr = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_storage_lease_operand_accepts_vgpr(
        descriptor_set, operand, &accepts_vgpr));
    if (!accepts_vgpr || current_packet_operand_index >= node->operand_count ||
        operand->unit_count == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_storage_lease_emit_event(
        emit, emit_user_data, LOOM_LOW_STORAGE_LEASE_SOURCE_READ,
        LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND, current_packet_operand_index,
        operand->unit_count, LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE,
        LOOM_AMDGPU_WAIT_PLAN_REASON_STORE_SOURCE_REUSE));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_storage_lease_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node, loom_low_storage_lease_emit_fn_t emit,
    void* emit_user_data) {
  (void)user_data;
  if (schedule == NULL || node == NULL || node->descriptor == NULL ||
      schedule->target.descriptor_set == NULL) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  if (descriptor_set->target_stable_id != LOOM_AMDGPU_TARGET_STABLE_ID) {
    return iree_ok_status();
  }

  loom_amdgpu_storage_lease_counter_masks_t counter_masks = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_lease_counter_masks(
      descriptor_set, node->descriptor, &counter_masks));
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_lease_emit_result_leases(
      descriptor_set, node, counter_masks.read_mask, emit, emit_user_data));
  return loom_amdgpu_storage_lease_emit_store_source_leases(
      descriptor_set, node, counter_masks.write_mask, emit, emit_user_data);
}

void loom_amdgpu_storage_lease_provider(
    loom_low_storage_lease_provider_t* out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = (loom_low_storage_lease_provider_t){
      .query = loom_amdgpu_storage_lease_query,
  };
}
