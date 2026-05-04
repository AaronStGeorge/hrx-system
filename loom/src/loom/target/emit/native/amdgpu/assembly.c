// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/assembly.h"

#include <inttypes.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/move_sequence.h"
#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/encoding.h"
#include "loom/target/arch/amdgpu/register_class.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/arch/amdgpu/target_refs.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"
#include "loom/target/emit/native/assembly.h"

typedef struct loom_amdgpu_wait_packet_emit_state_t {
  // Wait-packet plan consumed in scheduled insertion order.
  const loom_amdgpu_wait_packet_plan_t* wait_packets;
  // Next wait-packet row to compare with the current scheduled packet.
  iree_host_size_t next_packet_index;
} loom_amdgpu_wait_packet_emit_state_t;

static iree_status_t loom_amdgpu_descriptor_key(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t* out_key) {
  return loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->key_string_offset, out_key);
}

static iree_status_t loom_amdgpu_append_mnemonic(
    const loom_native_assembly_packet_context_t* context) {
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      context->schedule->target.descriptor_set,
      context->packet->descriptor->mnemonic_string_offset, &mnemonic));
  return iree_string_builder_append_string(context->builder, mnemonic);
}

static const loom_low_allocation_assignment_t* loom_amdgpu_map_assignment(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t value_id) {
  return loom_low_allocation_map_active_value_assignment(context->allocation,
                                                         value_id, NULL);
}

static bool loom_amdgpu_assignments_match(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

static iree_status_t loom_amdgpu_append_register_range(
    const loom_native_assembly_packet_context_t* context, const char* prefix,
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment->location_count == 1) {
    return iree_string_builder_append_format(context->builder, "%s%" PRIu32,
                                             prefix, assignment->location_base);
  }
  const uint32_t last_register =
      assignment->location_base + assignment->location_count - 1;
  return iree_string_builder_append_format(
      context->builder, "%s[%" PRIu32 ":%" PRIu32 "]", prefix,
      assignment->location_base, last_register);
}

static iree_status_t loom_amdgpu_append_assignment(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return loom_amdgpu_append_register_range(context, "s", assignment);
  }
  if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return loom_amdgpu_append_register_range(context, "v", assignment);
  }
  if (loom_amdgpu_register_class_is_agpr(
          context->allocation->target.descriptor_set,
          assignment->descriptor_reg_class_id)) {
    return loom_amdgpu_append_register_range(context, "acc", assignment);
  }
  iree_string_view_t register_class = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
      context->allocation, assignment, &register_class));
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU assembly register class '%.*s' is unsupported",
      (int)register_class.size, register_class.data);
}

static iree_status_t loom_amdgpu_append_move_location(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_move_location_t* location) {
  if (location->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return iree_string_builder_append_format(context->builder, "s%" PRIu32,
                                             location->location);
  }
  if (location->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return iree_string_builder_append_format(context->builder, "v%" PRIu32,
                                             location->location);
  }
  if (loom_amdgpu_register_class_is_agpr(
          context->allocation->target.descriptor_set,
          location->descriptor_reg_class_id)) {
    return iree_string_builder_append_format(context->builder, "acc%" PRIu32,
                                             location->location);
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU assembly descriptor register class ID %" PRIu16 " is unsupported",
      location->descriptor_reg_class_id);
}

static iree_status_t loom_amdgpu_append_value(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t value_id) {
  const loom_low_allocation_assignment_t* assignment = NULL;
  assignment = loom_amdgpu_map_assignment(context, value_id);
  return loom_amdgpu_append_assignment(context, assignment);
}

static iree_status_t loom_amdgpu_append_result(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_index) {
  const loom_op_t* op = context->packet->node->op;
  if (result_index >= op->result_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU assembly result index is out of range");
  }
  return loom_amdgpu_append_value(context,
                                  loom_op_const_results(op)[result_index]);
}

static iree_status_t loom_amdgpu_append_operand(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t operand_index) {
  const loom_op_t* op = context->packet->node->op;
  if (operand_index >= op->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU assembly operand index is out of range");
  }
  return loom_amdgpu_append_value(context,
                                  loom_op_const_operands(op)[operand_index]);
}

static loom_named_attr_slice_t loom_amdgpu_packet_attrs(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  if (loom_low_op_isa(op)) {
    return loom_low_op_attrs(op);
  }
  if (loom_low_const_isa(op)) {
    return loom_low_const_attrs(op);
  }
  return loom_make_named_attr_slice(NULL, 0);
}

static iree_status_t loom_amdgpu_read_packet_i64_attr(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t name, int64_t* out_value) {
  return loom_native_assembly_read_i64_attr(context->schedule->module,
                                            loom_amdgpu_packet_attrs(context),
                                            name, out_value);
}

static iree_status_t loom_amdgpu_read_packet_immediate_i64(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_immediate_t* immediate, int64_t* out_value) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  iree_string_view_t name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      descriptor_set, immediate->field_name_string_offset, &name));
  const loom_named_attr_t* attr = loom_native_assembly_find_attr(
      context->schedule->module, loom_amdgpu_packet_attrs(context), name);
  if (attr == NULL) {
    if (iree_all_bits_set(immediate->flags,
                          LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
      *out_value = immediate->default_value;
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly requires attribute '%.*s'",
                            (int)name.size, name.data);
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly attribute '%.*s' must be i64",
                            (int)name.size, name.data);
  }
  *out_value = loom_attr_as_i64(attr->value);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_find_packet_immediate(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t field_name, const loom_low_immediate_t** out_immediate) {
  *out_immediate = NULL;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->immediate_start > descriptor_set->immediate_count ||
      descriptor->immediate_count >
          descriptor_set->immediate_count - descriptor->immediate_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly descriptor immediate range is out of range");
  }
  if (descriptor->immediate_count != 0 && descriptor_set->immediates == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly descriptor immediates table is "
                            "missing");
  }
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
        descriptor_set, immediate->field_name_string_offset, &name));
    if (iree_string_view_equal(name, field_name)) {
      *out_immediate = immediate;
      return iree_ok_status();
    }
  }
  iree_string_view_t descriptor_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_key(context, &descriptor_key));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "AMDGPU assembly descriptor '%.*s' has no immediate "
                          "attribute '%.*s'",
                          (int)descriptor_key.size, descriptor_key.data,
                          (int)field_name.size, field_name.data);
}

static iree_status_t loom_amdgpu_descriptor_has_effect(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_effect_kind_t kind,
    bool* out_has_effect) {
  *out_has_effect = false;
  if (descriptor->effect_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->effect_start > descriptor_set->effect_count ||
      descriptor->effect_count >
          descriptor_set->effect_count - descriptor->effect_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU descriptor effect range is out of range");
  }
  if (descriptor_set->effects == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor effects table is missing");
  }
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (effect->kind == kind) {
      *out_has_effect = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_descriptor_has_memory_effect(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_effect_kind_t kind,
    loom_low_memory_space_t memory_space, bool* out_has_effect) {
  *out_has_effect = false;
  if (descriptor->effect_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->effect_start > descriptor_set->effect_count ||
      descriptor->effect_count >
          descriptor_set->effect_count - descriptor->effect_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU descriptor effect range is out of range");
  }
  if (descriptor_set->effects == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor effects table is missing");
  }
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (effect->kind == kind && effect->memory_space == memory_space) {
      *out_has_effect = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_descriptor_is_global_to_lds(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, bool* out_is_global_to_lds) {
  bool has_global_read = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_has_memory_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_READ,
      LOOM_LOW_MEMORY_SPACE_GLOBAL, &has_global_read));
  bool has_workgroup_write = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_has_memory_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_WRITE,
      LOOM_LOW_MEMORY_SPACE_WORKGROUP, &has_workgroup_write));
  *out_is_global_to_lds = has_global_read && has_workgroup_write;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_descriptor_uses_resource_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_low_resource_kind_t kind,
    bool* out_uses_resource_kind) {
  *out_uses_resource_kind = false;
  if (descriptor->schedule_class_id >= descriptor_set->schedule_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU descriptor schedule class is out of range");
  }
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[descriptor->schedule_class_id];
  if (schedule_class->issue_use_count == 0) {
    return iree_ok_status();
  }
  if (schedule_class->issue_use_start > descriptor_set->issue_use_count ||
      schedule_class->issue_use_count >
          descriptor_set->issue_use_count - schedule_class->issue_use_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU descriptor schedule-class issue-use range is out of range");
  }
  if (descriptor_set->issue_uses == NULL || descriptor_set->resources == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor schedule resources are missing");
  }
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &descriptor_set->issue_uses[schedule_class->issue_use_start + i];
    if (issue_use->resource_id >= descriptor_set->resource_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU descriptor issue-use resource is out of "
                              "range");
    }
    const loom_low_resource_t* resource =
        &descriptor_set->resources[issue_use->resource_id];
    if (resource->kind == kind) {
      *out_uses_resource_kind = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_comma(
    const loom_native_assembly_packet_context_t* context) {
  return iree_string_builder_append_cstring(context->builder, ", ");
}

static iree_status_t loom_amdgpu_append_result_operand_list(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_count, iree_host_size_t operand_count) {
  bool needs_comma = false;
  for (iree_host_size_t i = 0; i < result_count; ++i) {
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(context, i));
    needs_comma = true;
  }
  for (iree_host_size_t i = 0; i < operand_count; ++i) {
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, i));
    needs_comma = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_canonical_asm_form(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_asm_form_t** out_form) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_key(context, &key));
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU assembly descriptor '%.*s' has no canonical asm form",
        (int)key.size, key.data);
  }
  *out_form = loom_low_descriptor_set_asm_form_at(
      descriptor_set, descriptor->canonical_asm_form_ordinal);
  if (*out_form == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU assembly canonical asm form is out of "
                            "range");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_descriptor_value_list(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->operand_start > descriptor_set->operand_count ||
      descriptor->operand_count >
          descriptor_set->operand_count - descriptor->operand_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU descriptor operand range is out of range");
  }
  if (descriptor->operand_count == 0) {
    return iree_ok_status();
  }
  if (descriptor_set->operands == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor operands table is missing");
  }

  bool needs_comma = false;
  for (uint16_t i = 0; i < descriptor->result_count; ++i) {
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    } else {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(context->builder, " "));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(context, i));
    needs_comma = true;
  }

  uint16_t packet_operand_index = 0;
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    const loom_low_operand_t* operand = &operands[i];
    if (operand->role == LOOM_LOW_OPERAND_ROLE_IMPLICIT) {
      continue;
    }
    if (!loom_low_operand_role_is_packet_operand(operand->role)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU descriptor operand row %" PRIu16
          " has role %u that cannot map to a packet operand",
          i, (unsigned)operand->role);
    }
    const uint16_t current_packet_operand_index = packet_operand_index++;
    if (iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      continue;
    }
    if (needs_comma) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    } else {
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(context->builder, " "));
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_operand(context, current_packet_operand_index));
    needs_comma = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_asm_form_separator(
    const loom_native_assembly_packet_context_t* context, bool* in_list) {
  if (*in_list) {
    return loom_amdgpu_append_comma(context);
  }
  *in_list = true;
  return iree_string_builder_append_cstring(context->builder, " ");
}

static iree_status_t loom_amdgpu_append_asm_form_value(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_descriptor_t* descriptor, uint16_t descriptor_operand_index,
    bool is_result) {
  const loom_op_t* op = context->packet->node->op;
  if (descriptor_operand_index >= descriptor->operand_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU asm-form operand index is outside the descriptor");
  }
  if (is_result) {
    if (descriptor_operand_index >= descriptor->result_count ||
        descriptor_operand_index >= op->result_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU asm-form result field does not name an "
                              "emitted result");
    }
    return loom_amdgpu_append_result(context, descriptor_operand_index);
  }
  if (descriptor_operand_index < descriptor->result_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU asm-form operand field unexpectedly names "
                            "a descriptor result");
  }
  const uint16_t operand_index =
      descriptor_operand_index - descriptor->result_count;
  if (operand_index >= op->operand_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU asm-form operand field does not name an "
                            "emitted operand");
  }
  return loom_amdgpu_append_operand(context, operand_index);
}

static iree_status_t loom_amdgpu_append_asm_form_values(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_descriptor_t* descriptor, uint32_t start, uint16_t count,
    bool is_result, bool* in_list) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  for (uint16_t i = 0; i < count; ++i) {
    const uint32_t asm_operand_index = start + i;
    if (asm_operand_index >= descriptor_set->asm_operand_index_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU asm-form operand row is outside the descriptor set");
    }
    const uint16_t descriptor_operand_index =
        descriptor_set->asm_operand_indices[asm_operand_index];
    if (descriptor_operand_index >= descriptor->operand_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU asm-form operand index is outside the descriptor");
    }
    const loom_low_operand_t* descriptor_operand =
        &descriptor_set
             ->operands[descriptor->operand_start + descriptor_operand_index];
    if (iree_any_bit_set(descriptor_operand->flags,
                         LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_asm_form_separator(context, in_list));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_asm_form_value(
        context, descriptor, descriptor_operand_index, is_result));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_asm_form_immediates(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_descriptor_t* descriptor, const loom_low_asm_form_t* form,
    bool* in_list) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  for (uint16_t i = 0; i < form->immediate_count; ++i) {
    const uint32_t asm_immediate_index = form->immediate_start + i;
    if (asm_immediate_index >= descriptor_set->asm_immediate_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU asm-form immediate row is outside the descriptor set");
    }
    const loom_low_asm_immediate_t* asm_immediate =
        &descriptor_set->asm_immediates[asm_immediate_index];
    if (asm_immediate->immediate_index >= descriptor->immediate_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU asm-form immediate references an invalid descriptor field");
    }
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start +
                                    asm_immediate->immediate_index];
    int64_t value = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_read_packet_immediate_i64(context, immediate, &value));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_asm_form_separator(context, in_list));
    if (asm_immediate->name_string_offset != LOOM_LOW_STRING_OFFSET_NONE) {
      iree_string_view_t spelling = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
          descriptor_set, asm_immediate->name_string_offset, &spelling));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          context->builder, "%.*s(%" PRId64 ")", (int)spelling.size,
          spelling.data, value));
    } else {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          context->builder, "%" PRId64, value));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_canonical_asm_form_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const loom_low_asm_form_t* form = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_canonical_asm_form(context, &form));
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      descriptor_set, form->mnemonic_string_offset, &mnemonic));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, mnemonic));
  bool in_list = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_asm_form_values(
      context, descriptor, form->result_operand_index_start,
      form->result_operand_index_count, /*is_result=*/true, &in_list));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_asm_form_values(
      context, descriptor, form->operand_index_start, form->operand_index_count,
      /*is_result=*/false, &in_list));
  return loom_amdgpu_append_asm_form_immediates(context, descriptor, form,
                                                &in_list);
}

static iree_status_t loom_amdgpu_append_basic_packet(
    const loom_native_assembly_packet_context_t* context,
    iree_host_size_t result_count, iree_host_size_t operand_count) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  if (result_count + operand_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  return loom_amdgpu_append_result_operand_list(context, result_count,
                                                operand_count);
}

static iree_status_t loom_amdgpu_append_memory_immediate_suffixes(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->immediate_count == 0) {
    return iree_ok_status();
  }
  if (descriptor->immediate_start > descriptor_set->immediate_count ||
      descriptor->immediate_count >
          descriptor_set->immediate_count - descriptor->immediate_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly descriptor immediate range is out of range");
  }
  if (descriptor_set->immediates == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly descriptor immediates table is "
                            "missing");
  }
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
        descriptor_set, immediate->field_name_string_offset, &name));
    int64_t value = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_read_packet_immediate_i64(context, immediate, &value));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, " %.*s:%" PRId64, (int)name.size, name.data, value));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_memory_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_descriptor_value_list(context));
  } else {
    const loom_low_asm_form_t* form = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_canonical_asm_form(context, &form));
    iree_string_view_t mnemonic = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
        descriptor_set, form->mnemonic_string_offset, &mnemonic));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(context->builder, mnemonic));
    bool in_list = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_asm_form_values(
        context, descriptor, form->result_operand_index_start,
        form->result_operand_index_count, /*is_result=*/true, &in_list));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_asm_form_values(
        context, descriptor, form->operand_index_start,
        form->operand_index_count, /*is_result=*/false, &in_list));
  }
  return loom_amdgpu_append_memory_immediate_suffixes(context);
}

static iree_status_t loom_amdgpu_append_offset_suffix(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_immediate_t* immediate = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_find_packet_immediate(
      context, IREE_SV("offset"), &immediate));
  int64_t offset = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_immediate_i64(context, immediate, &offset));
  return iree_string_builder_append_format(context->builder, " offset:%" PRId64,
                                           offset);
}

static iree_status_t loom_amdgpu_append_mubuf_load_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  if (descriptor ==
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_DWORD_OFF_ZERO)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "off"));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 0));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "0"));
    return loom_amdgpu_append_offset_suffix(context);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 2));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " offen"));
  return loom_amdgpu_append_offset_suffix(context);
}

static iree_status_t loom_amdgpu_append_mubuf_store_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  if (descriptor ==
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_STORE_DWORD_OFF_ZERO)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "off"));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 1));
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "0"));
    return loom_amdgpu_append_offset_suffix(context);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 3));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " offen"));
  return loom_amdgpu_append_offset_suffix(context);
}

static iree_status_t loom_amdgpu_append_buffer_atomic_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  if (descriptor->result_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(context, 0));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 0));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 3));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " offen"));
  return loom_amdgpu_append_memory_immediate_suffixes(context);
}

static bool loom_amdgpu_descriptor_uses_global_pointer_format(
    const loom_low_descriptor_t* descriptor) {
  switch (descriptor->encoding_format_id) {
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT:
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLBL:
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLOBAL:
    case LOOM_AMDGPU_ENCODING_FORMAT_VFLAT:
    case LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_descriptor_uses_data_share_format(
    const loom_low_descriptor_t* descriptor) {
  switch (descriptor->encoding_format_id) {
    case LOOM_AMDGPU_ENCODING_FORMAT_DS:
    case LOOM_AMDGPU_ENCODING_FORMAT_VDS:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_descriptor_uses_global_scalar_base_format(
    const loom_native_assembly_packet_context_t* context) {
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  const uint16_t address_operand_index = descriptor->result_count;
  if (address_operand_index >= descriptor->operand_count) {
    return false;
  }
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  return operands[address_operand_index].unit_count == 1;
}

static iree_status_t loom_amdgpu_append_global_load_packet(
    const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_result(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  if (loom_amdgpu_descriptor_uses_global_scalar_base_format(context)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 1));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "off"));
  }
  return loom_amdgpu_append_offset_suffix(context);
}

static iree_status_t loom_amdgpu_append_global_load_lds_packet(
    const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  if (loom_amdgpu_descriptor_uses_global_scalar_base_format(context)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 1));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "off"));
  }
  return loom_amdgpu_append_offset_suffix(context);
}

static iree_status_t loom_amdgpu_append_global_store_packet(
    const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 0));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  if (loom_amdgpu_descriptor_uses_global_scalar_base_format(context)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_operand(context, 2));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "off"));
  }
  return loom_amdgpu_append_offset_suffix(context);
}

static iree_status_t loom_amdgpu_append_waitcnt_packet(
    const loom_native_assembly_packet_context_t* context) {
  int64_t vmcnt = 0;
  int64_t lgkmcnt = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_i64_attr(context, IREE_SV("vmcnt"), &vmcnt));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_packet_i64_attr(context, IREE_SV("lgkmcnt"), &lgkmcnt));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_mnemonic(context));
  return iree_string_builder_append_format(
      context->builder, " vmcnt(%" PRId64 ") lgkmcnt(%" PRId64 ")", vmcnt,
      lgkmcnt);
}

static iree_status_t loom_amdgpu_append_materialized_wait_packet(
    const loom_native_assembly_packet_context_t* context,
    const loom_amdgpu_wait_packet_t* wait_packet,
    const loom_amdgpu_wait_packet_plan_t* wait_packets) {
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(descriptor_set,
                                                 wait_packet->descriptor);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait packet descriptor row does not belong to the selected "
        "descriptor set");
  }
  if (wait_packet->immediate_start > wait_packets->immediate_count ||
      wait_packet->immediate_count >
          wait_packets->immediate_count - wait_packet->immediate_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait packet immediate range is out of range");
  }
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
      descriptor_set, descriptor->mnemonic_string_offset, &mnemonic));

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "  "));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, mnemonic));
  for (iree_host_size_t i = 0; i < wait_packet->immediate_count; ++i) {
    const iree_host_size_t immediate_index = wait_packet->immediate_start + i;
    const loom_amdgpu_wait_packet_immediate_t* immediate =
        &wait_packets->immediates[immediate_index];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        context->builder, " %.*s(%" PRIu16 ")", (int)immediate->name.size,
        immediate->name.data, immediate->value));
  }
  return iree_string_builder_append_cstring(context->builder, "\n");
}

static bool loom_amdgpu_wait_packet_is_before_node(
    const loom_amdgpu_wait_packet_t* wait_packet,
    const loom_low_schedule_node_t* node) {
  return wait_packet->block_index < node->block_index ||
         (wait_packet->block_index == node->block_index &&
          wait_packet->scheduled_ordinal < node->scheduled_ordinal);
}

static bool loom_amdgpu_wait_packet_matches_packet(
    const loom_amdgpu_wait_packet_t* wait_packet,
    const loom_low_packet_view_t* packet) {
  const loom_low_schedule_node_t* node = packet->node;
  return wait_packet->block_index == node->block_index &&
         wait_packet->scheduled_ordinal == node->scheduled_ordinal &&
         wait_packet->node_index == packet->node_index;
}

static iree_status_t loom_amdgpu_append_wait_packets_before_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  loom_amdgpu_wait_packet_emit_state_t* state =
      (loom_amdgpu_wait_packet_emit_state_t*)user_data;
  const loom_low_schedule_node_t* node = context->packet->node;
  while (state->next_packet_index < state->wait_packets->packet_count) {
    const loom_amdgpu_wait_packet_t* wait_packet =
        &state->wait_packets->packets[state->next_packet_index];
    if (loom_amdgpu_wait_packet_is_before_node(wait_packet, node)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU wait packet plan contains an insertion point before the "
          "current scheduled packet");
    }
    if (!loom_amdgpu_wait_packet_matches_packet(wait_packet, context->packet)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_materialized_wait_packet(
        context, wait_packet, state->wait_packets));
    ++state->next_packet_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_copy_mnemonic(
    uint16_t descriptor_reg_class_id, iree_string_view_t* out_mnemonic) {
  if (descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    *out_mnemonic = IREE_SV("s_mov_b32");
    return iree_ok_status();
  }
  if (descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    *out_mnemonic = IREE_SV("v_mov_b32");
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDGPU assembly copy descriptor register class ID %" PRIu16
      " is unsupported",
      descriptor_reg_class_id);
}

typedef struct loom_amdgpu_assembly_move_state_t {
  // Packet context receiving assembly text.
  const loom_native_assembly_packet_context_t* context;
  // Target move mnemonic used for each emitted unit move.
  iree_string_view_t mnemonic;
  // Number of non-identity moves emitted so far.
  uint32_t emitted_count;
} loom_amdgpu_assembly_move_state_t;

static iree_status_t loom_amdgpu_append_move(
    void* user_data, const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source) {
  loom_amdgpu_assembly_move_state_t* state =
      (loom_amdgpu_assembly_move_state_t*)user_data;
  const loom_native_assembly_packet_context_t* context = state->context;
  if (state->emitted_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "\n  "));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(context->builder, state->mnemonic));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, " "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_move_location(context, destination));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_move_location(context, source));
  ++state->emitted_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_move_sequence(
    const loom_native_assembly_packet_context_t* context,
    iree_string_view_t mnemonic, iree_host_size_t move_count) {
  loom_amdgpu_assembly_move_state_t move_state = {
      .context = context,
      .mnemonic = mnemonic,
  };
  loom_low_move_sequence_options_t options = {
      .emit_move =
          {
              .fn = loom_amdgpu_append_move,
              .user_data = &move_state,
          },
  };
  return loom_low_move_sequence_emit(context->move_scratch, move_count,
                                     &options);
}

static iree_status_t loom_amdgpu_append_edge_move(
    void* user_data, const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source) {
  loom_amdgpu_assembly_move_state_t* state =
      (loom_amdgpu_assembly_move_state_t*)user_data;
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_copy_mnemonic(
      destination->descriptor_reg_class_id, &mnemonic));
  state->mnemonic = mnemonic;
  return loom_amdgpu_append_move(user_data, destination, source);
}

static iree_status_t loom_amdgpu_emit_edge_copy_group(
    const loom_native_assembly_packet_context_t* context,
    const loom_low_allocation_edge_copy_group_t* group) {
  iree_host_size_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_edge_copy_units(
      context->allocation, group, &move_count));
  if (move_count == 0) {
    return iree_ok_status();
  }
  loom_low_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_moves(
      context->move_scratch, move_count, &moves));
  loom_low_move_location_t* temporaries = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_temporaries(
      context->move_scratch, group->temporary_count, &temporaries));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_edge_copy_units(
      context->allocation, group, moves, move_count));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_edge_copy_temporaries(
      context->allocation, group, temporaries, group->temporary_count));
  loom_amdgpu_assembly_move_state_t move_state = {
      .context = context,
  };
  loom_low_move_sequence_options_t options = {
      .temporary_locations = temporaries,
      .temporary_location_count = group->temporary_count,
      .emit_move =
          {
              .fn = loom_amdgpu_append_edge_move,
              .user_data = &move_state,
          },
  };
  IREE_RETURN_IF_ERROR(
      loom_low_move_sequence_emit(context->move_scratch, move_count, &options));
  if (move_state.emitted_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(context->builder, "\n  "));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_copy_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  const loom_op_t* op = context->packet->node->op;
  const loom_low_allocation_assignment_t* source_assignment =
      loom_amdgpu_map_assignment(context, loom_low_copy_source(op));
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_map_assignment(context, loom_low_copy_result(op));
  if (loom_amdgpu_assignments_match(source_assignment, result_assignment)) {
    return iree_ok_status();
  }
  if (source_assignment->location_count != result_assignment->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU assembly copy requires matching register ranges");
  }
  const uint32_t register_count = source_assignment->location_count;
  const uint32_t last_register_offset = register_count - 1;
  if (source_assignment->location_base > UINT32_MAX - last_register_offset ||
      result_assignment->location_base > UINT32_MAX - last_register_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly copy register range exceeds uint32_t");
  }

  if (source_assignment->descriptor_reg_class_id !=
      result_assignment->descriptor_reg_class_id) {
    iree_string_view_t source_register_class = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
        context->allocation, source_assignment, &source_register_class));
    iree_string_view_t result_register_class = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
        context->allocation, result_assignment, &result_register_class));
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU assembly copy between register classes '%.*s' and '%.*s' is "
        "unsupported",
        (int)source_register_class.size, source_register_class.data,
        (int)result_register_class.size, result_register_class.data);
  }

  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_copy_mnemonic(
      result_assignment->descriptor_reg_class_id, &mnemonic));

  loom_low_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_moves(
      context->move_scratch, register_count, &moves));
  for (uint32_t i = 0; i < register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
        result_assignment, i, &moves[i].destination));
    IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
        source_assignment, i, &moves[i].source));
  }
  return loom_amdgpu_emit_move_sequence(context, mnemonic, register_count);
}

static iree_status_t loom_amdgpu_append_slice_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  const loom_op_t* op = context->packet->node->op;
  iree_host_size_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_slice_units(
      context->allocation, op, &move_count));
  if (move_count == 0) {
    return iree_ok_status();
  }

  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_map_assignment(context, loom_low_slice_result(op));
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_copy_mnemonic(
      result_assignment->descriptor_reg_class_id, &mnemonic));

  loom_low_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_moves(
      context->move_scratch, move_count, &moves));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_slice_units(
      context->allocation, op, moves, move_count));
  return loom_amdgpu_emit_move_sequence(context, mnemonic, move_count);
}

static iree_status_t loom_amdgpu_append_concat_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  const loom_op_t* op = context->packet->node->op;
  iree_host_size_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_concat_units(
      context->allocation, op, &move_count));
  if (move_count == 0) {
    return iree_ok_status();
  }

  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_map_assignment(context, loom_low_concat_result(op));
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_copy_mnemonic(
      result_assignment->descriptor_reg_class_id, &mnemonic));

  loom_low_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_moves(
      context->move_scratch, move_count, &moves));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_concat_units(
      context->allocation, op, moves, move_count));
  return loom_amdgpu_emit_move_sequence(context, mnemonic, move_count);
}

static iree_status_t loom_amdgpu_append_storage_address_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  const loom_op_t* op = context->packet->node->op;
  loom_amdgpu_storage_layout_reservation_t reservation;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_resolve(
      context->schedule->module, context->schedule->function_op,
      loom_low_storage_address_storage(op), &reservation));
  if (reservation.space != LOOM_STORAGE_SPACE_WORKGROUP) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU assembly only supports "
                            "low.storage.address for workgroup storage");
  }
  const int64_t signed_offset = loom_low_storage_address_offset(op);
  if (signed_offset < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU assembly low.storage.address offset must be non-negative");
  }
  const uint64_t offset = (uint64_t)signed_offset;
  if (reservation.byte_offset > UINT32_MAX ||
      offset > UINT32_MAX - reservation.byte_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU assembly low.storage.address byte offset exceeds u32");
  }

  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_map_assignment(context, loom_low_storage_address_result(op));
  if (assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
      assignment->location_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU assembly low.storage.address result must "
                            "be one VGPR");
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "v_mov_b32 "));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_assignment(context, assignment));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return iree_string_builder_append_format(
      context->builder, "%" PRIu32,
      (uint32_t)(reservation.byte_offset + offset));
}

static iree_status_t loom_amdgpu_append_matrix_packet(
    const loom_native_assembly_packet_context_t* context) {
  const loom_op_t* op = context->packet->node->op;
  uint16_t accumulator_operand_index = UINT16_MAX;
  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    if (tied_results[i].result_index == 0) {
      accumulator_operand_index = tied_results[i].operand_index;
      break;
    }
  }
  if (accumulator_operand_index == UINT16_MAX ||
      accumulator_operand_index >= op->operand_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU matrix result must be tied to an accumulator operand");
  }
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_map_assignment(context, loom_op_const_results(op)[0]);
  const loom_low_allocation_assignment_t* accumulator_assignment =
      loom_amdgpu_map_assignment(
          context, loom_op_const_operands(op)[accumulator_operand_index]);
  if (!loom_amdgpu_assignments_match(result_assignment,
                                     accumulator_assignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU matrix result must share the accumulator physical register");
  }
  return loom_amdgpu_append_canonical_asm_form_packet(context);
}

static iree_status_t loom_amdgpu_append_exec_restore_packet(
    const loom_native_assembly_packet_context_t* context) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "s_mov_b64 exec"));
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_comma(context));
  return loom_amdgpu_append_operand(context, 0);
}

static iree_status_t loom_amdgpu_append_descriptor_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  const loom_low_descriptor_set_t* descriptor_set =
      context->schedule->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = context->packet->descriptor;
  if (descriptor ==
      loom_amdgpu_descriptor_ref_descriptor(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC)) {
    return loom_amdgpu_append_exec_restore_packet(context);
  }
  bool has_read_effect = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_has_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_READ, &has_read_effect));
  bool has_write_effect = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_has_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_WRITE,
      &has_write_effect));
  if (has_read_effect && has_write_effect) {
    bool is_global_to_lds = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_is_global_to_lds(
        descriptor_set, descriptor, &is_global_to_lds));
    if (is_global_to_lds &&
        loom_amdgpu_descriptor_uses_global_pointer_format(descriptor)) {
      return loom_amdgpu_append_global_load_lds_packet(context);
    }
    if (descriptor->encoding_format_id == LOOM_AMDGPU_ENCODING_FORMAT_MUBUF ||
        descriptor->encoding_format_id == LOOM_AMDGPU_ENCODING_FORMAT_VBUFFER) {
      return loom_amdgpu_append_buffer_atomic_packet(context);
    }
    if (loom_amdgpu_descriptor_uses_global_pointer_format(descriptor) ||
        loom_amdgpu_descriptor_uses_data_share_format(descriptor)) {
      return loom_amdgpu_append_memory_packet(context);
    }
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_key(context, &key));
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU assembly descriptor '%.*s' has both read and write effects",
        (int)key.size, key.data);
  }
  if (has_read_effect || has_write_effect) {
    if (descriptor->encoding_format_id == LOOM_AMDGPU_ENCODING_FORMAT_MUBUF ||
        descriptor->encoding_format_id == LOOM_AMDGPU_ENCODING_FORMAT_VBUFFER) {
      return has_read_effect ? loom_amdgpu_append_mubuf_load_packet(context)
                             : loom_amdgpu_append_mubuf_store_packet(context);
    }
    if (loom_amdgpu_descriptor_uses_global_pointer_format(descriptor)) {
      return has_read_effect ? loom_amdgpu_append_global_load_packet(context)
                             : loom_amdgpu_append_global_store_packet(context);
    }
    return loom_amdgpu_append_memory_packet(context);
  }
  if (loom_amdgpu_descriptor_uses_data_share_format(descriptor)) {
    return loom_amdgpu_append_memory_packet(context);
  }
  bool uses_matrix_resource = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_uses_resource_kind(
      descriptor_set, descriptor, LOOM_LOW_RESOURCE_KIND_MATRIX,
      &uses_matrix_resource));
  if (uses_matrix_resource) {
    return loom_amdgpu_append_matrix_packet(context);
  }
  bool has_counter_effect = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_has_effect(
      descriptor_set, descriptor, LOOM_LOW_EFFECT_KIND_COUNTER,
      &has_counter_effect));
  if (has_counter_effect && descriptor->immediate_count == 2) {
    return loom_amdgpu_append_waitcnt_packet(context);
  }
  return loom_amdgpu_append_canonical_asm_form_packet(context);
}

static iree_status_t loom_amdgpu_append_return_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  return iree_string_builder_append_cstring(context->builder, "s_endpgm");
}

static iree_status_t loom_amdgpu_append_branch_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  const loom_op_t* op = context->packet->node->op;
  loom_value_slice_t args = loom_low_br_args(op);
  if (args.count != 0) {
    const loom_low_allocation_edge_copy_group_t* group =
        loom_low_allocation_find_edge_copy_group_by_source_ordinal(
            context->allocation, context->packet->node->source_ordinal);
    if (!group) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU assembly branch edge copies are missing from allocation");
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_edge_copy_group(context, group));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "s_branch "));
  return loom_native_assembly_append_block_label(
      context->schedule, loom_low_br_dest(op), context->builder);
}

static iree_status_t loom_amdgpu_verify_scc_condition_assignment(
    const loom_native_assembly_packet_context_t* context,
    loom_value_id_t condition_value_id) {
  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_map_assignment(context, condition_value_id);
  if (assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_SCC) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly conditional branch condition must "
                            "be allocated to SCC");
  }
  if (assignment->location_base != 0 || assignment->location_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly SCC condition must use the single "
                            "architectural SCC register");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_cond_branch_packet(
    void* user_data, const loom_native_assembly_packet_context_t* context) {
  (void)user_data;
  const loom_op_t* op = context->packet->node->op;
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_scc_condition_assignment(
      context, loom_low_cond_br_condition(op)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "s_cbranch_scc1 "));
  IREE_RETURN_IF_ERROR(loom_native_assembly_append_block_label(
      context->schedule, loom_low_cond_br_true_dest(op), context->builder));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(context->builder, "\n  s_branch "));
  return loom_native_assembly_append_block_label(
      context->schedule, loom_low_cond_br_false_dest(op), context->builder);
}

static iree_status_t loom_amdgpu_verify_assembly_target(
    const loom_low_schedule_table_t* schedule) {
  if (schedule->target.descriptor_set->target_stable_id !=
      LOOM_AMDGPU_TARGET_STABLE_ID) {
    iree_string_view_t target_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_native_assembly_descriptor_string(
        schedule->target.descriptor_set,
        schedule->target.descriptor_set->target_key_string_offset,
        &target_key));
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU assembly emitter received target '%.*s'",
                            (int)target_key.size, target_key.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_verify_wait_packet_plan(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_wait_packet_plan_t* wait_packets) {
  if (wait_packets->wait_plan->schedule != schedule) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU assembly wait packets must be derived from the emitted "
        "schedule");
  }
  return iree_ok_status();
}

static const loom_native_assembly_structural_packet_callback_t
    kLoomAmdgpuAssemblyStructuralPacketCallbacks[] = {
        {
            .op_kind = LOOM_OP_LOW_COPY,
            .append_packet =
                {
                    .fn = loom_amdgpu_append_copy_packet,
                    .user_data = NULL,
                },
        },
        {
            .op_kind = LOOM_OP_LOW_SLICE,
            .append_packet =
                {
                    .fn = loom_amdgpu_append_slice_packet,
                    .user_data = NULL,
                },
        },
        {
            .op_kind = LOOM_OP_LOW_CONCAT,
            .append_packet =
                {
                    .fn = loom_amdgpu_append_concat_packet,
                    .user_data = NULL,
                },
        },
        {
            .op_kind = LOOM_OP_LOW_STORAGE_ADDRESS,
            .append_packet =
                {
                    .fn = loom_amdgpu_append_storage_address_packet,
                    .user_data = NULL,
                },
        },
        {
            .op_kind = LOOM_OP_LOW_RETURN,
            .append_packet =
                {
                    .fn = loom_amdgpu_append_return_packet,
                    .user_data = NULL,
                },
        },
        {
            .op_kind = LOOM_OP_LOW_BR,
            .append_packet =
                {
                    .fn = loom_amdgpu_append_branch_packet,
                    .user_data = NULL,
                },
        },
        {
            .op_kind = LOOM_OP_LOW_COND_BR,
            .append_packet =
                {
                    .fn = loom_amdgpu_append_cond_branch_packet,
                    .user_data = NULL,
                },
        },
};

static loom_native_assembly_format_options_t loom_amdgpu_assembly_options(
    loom_native_assembly_append_packet_callback_t append_before_packet) {
  return (loom_native_assembly_format_options_t){
      .append_before_packet = append_before_packet,
      .append_descriptor_packet =
          {
              .fn = loom_amdgpu_append_descriptor_packet,
              .user_data = NULL,
          },
      .structural_packet_callbacks =
          kLoomAmdgpuAssemblyStructuralPacketCallbacks,
      .structural_packet_callback_count =
          IREE_ARRAYSIZE(kLoomAmdgpuAssemblyStructuralPacketCallbacks),
  };
}

iree_status_t loom_amdgpu_emit_assembly_fragment(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_assembly_target(schedule));
  const loom_native_assembly_format_options_t options =
      loom_amdgpu_assembly_options(
          (loom_native_assembly_append_packet_callback_t){0});
  return loom_native_assembly_format_fragment(schedule, allocation, &options,
                                              builder, scratch_arena);
}

iree_status_t loom_amdgpu_emit_assembly_fragment_with_wait_packets(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    iree_string_builder_t* builder, iree_arena_allocator_t* scratch_arena) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_assembly_target(schedule));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_verify_wait_packet_plan(schedule, wait_packets));

  loom_amdgpu_wait_packet_emit_state_t wait_state = {
      .wait_packets = wait_packets,
  };
  const loom_native_assembly_format_options_t options =
      loom_amdgpu_assembly_options(
          (loom_native_assembly_append_packet_callback_t){
              .fn = loom_amdgpu_append_wait_packets_before_packet,
              .user_data = &wait_state,
          });
  IREE_RETURN_IF_ERROR(loom_native_assembly_format_fragment(
      schedule, allocation, &options, builder, scratch_arena));
  if (wait_state.next_packet_index != wait_packets->packet_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU assembly wait packet plan contains an unmatched insertion "
        "point");
  }
  return iree_ok_status();
}
