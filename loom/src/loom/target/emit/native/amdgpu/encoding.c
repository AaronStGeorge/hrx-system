// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/encoding.h"

#include <inttypes.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/move_sequence.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/encoding.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"
#include "loom/target/emit/native/fragment.h"

#define LOOM_AMDGPU_MAX_PACKET_FIELD_VALUES 32u
typedef struct loom_amdgpu_encode_state_t {
  // Schedule table being encoded.
  const loom_low_schedule_table_t* schedule;
  // Allocation table supplying physical locations.
  const loom_low_allocation_table_t* allocation;
  // Resolved native target encoding profile.
  const loom_amdgpu_descriptor_set_info_t* target;
  // Resolved bit-encoding table for target descriptor packets, or NULL when
  // this descriptor set has no native encoding table.
  const loom_amdgpu_encoding_table_t* encoding_table;
  // Fixed-segment layout shared by low.storage.address packets.
  const loom_amdgpu_storage_layout_t* storage_layout;
  // Module string IDs for descriptor immediate rows, indexed by descriptor-set
  // immediate ordinal.
  const loom_string_id_t* immediate_name_ids;
  // Number of entries in immediate_name_ids.
  iree_host_size_t immediate_name_id_count;
  // Optional planned wait packets consumed in scheduled order.
  const loom_amdgpu_wait_packet_plan_t* wait_packets;
  // Arena used for transient encoding scratch and final byte storage.
  iree_arena_allocator_t* arena;
  // Reusable scratch for structural parallel-copy sequencing.
  loom_low_move_sequence_scratch_t* move_scratch;
  // Destination byte storage, or NULL during the sizing pass.
  uint8_t* data;
  // Capacity of |data| in bytes, or zero during the sizing pass.
  iree_host_size_t capacity;
  // Number of bytes planned or written so far.
  iree_host_size_t length;
  // Next wait-packet row to compare with the current scheduled packet.
  iree_host_size_t next_wait_packet_index;
  // Planned byte offset for each scheduled block.
  iree_host_size_t* block_offsets;
} loom_amdgpu_encode_state_t;

static iree_status_t loom_amdgpu_descriptor_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, iree_string_view_t* out_string) {
  *out_string = loom_low_descriptor_set_string(descriptor_set, string_offset);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_append_u32(loom_amdgpu_encode_state_t* state,
                                            uint32_t value) {
  iree_host_size_t next_length = 0;
  if (!iree_host_size_checked_add(state->length, sizeof(value), &next_length)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU encoded instruction stream length overflowed");
  }
  if (state->data != NULL) {
    if (state->capacity < state->length ||
        state->capacity - state->length < sizeof(value)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU encoded instruction stream overflowed "
                              "its planned byte length");
    }
    state->data[state->length + 0] = (uint8_t)(value & 0xFFu);
    state->data[state->length + 1] = (uint8_t)((value >> 8) & 0xFFu);
    state->data[state->length + 2] = (uint8_t)((value >> 16) & 0xFFu);
    state->data[state->length + 3] = (uint8_t)((value >> 24) & 0xFFu);
  }
  state->length = next_length;
  return iree_ok_status();
}

static uint64_t loom_amdgpu_low_bit_mask(uint16_t bit_count) {
  if (bit_count >= 64) {
    return UINT64_MAX;
  }
  return (UINT64_C(1) << bit_count) - 1;
}

static loom_named_attr_slice_t loom_amdgpu_packet_attrs(
    const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  if (loom_low_op_isa(op)) {
    return loom_low_op_attrs(op);
  }
  if (loom_low_const_isa(op)) {
    return loom_low_const_attrs(op);
  }
  return loom_make_named_attr_slice(NULL, 0);
}

static const loom_named_attr_t* loom_amdgpu_find_packet_attr_by_name_id(
    loom_named_attr_slice_t attrs, loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id == name_id) {
      return attr;
    }
  }
  return NULL;
}

static const loom_low_allocation_assignment_t* loom_amdgpu_map_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id) {
  return loom_low_allocation_map_active_value_assignment(allocation, value_id,
                                                         NULL);
}

static iree_status_t loom_amdgpu_assignment_sgpr(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* assignment,
    uint16_t* out_register) {
  *out_register = 0;
  (void)allocation;
  if (assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native encoding value %" PRIu32
                            " must be an SGPR",
                            assignment->value_id);
  }
  if (assignment->location_count != 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU native encoding SGPR value %" PRIu32
                            " requires %" PRIu32
                            " registers; only scalar registers are supported",
                            assignment->value_id, assignment->location_count);
  }
  if (assignment->location_base > 127) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding SGPR index %" PRIu32
                            " is out of range",
                            assignment->location_base);
  }
  *out_register = (uint16_t)assignment->location_base;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_assignment_vgpr(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* assignment,
    uint16_t* out_register) {
  *out_register = 0;
  (void)allocation;
  if (assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native encoding value %" PRIu32
                            " must be a VGPR",
                            assignment->value_id);
  }
  if (assignment->location_count != 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU native encoding VGPR value %" PRIu32
                            " requires %" PRIu32
                            " registers; only scalar registers are supported",
                            assignment->value_id, assignment->location_count);
  }
  if (assignment->location_base > 255) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding VGPR index %" PRIu32
                            " is out of range",
                            assignment->location_base);
  }
  *out_register = (uint16_t)assignment->location_base;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_verify_scc_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id) {
  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_map_assignment(allocation, value_id);
  if (assignment->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_SCC) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding conditional branch condition must be SCC");
  }
  if (assignment->location_base != 0 || assignment->location_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU native encoding SCC condition must use "
                            "the single architectural SCC register");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_move_location_sgpr(
    const loom_low_allocation_table_t* allocation,
    const loom_low_move_location_t* location, uint16_t* out_register) {
  *out_register = 0;
  (void)allocation;
  if (location->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native encoding move location must be an "
                            "SGPR");
  }
  if (location->location > 127) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding SGPR index %" PRIu32
                            " is out of range",
                            location->location);
  }
  *out_register = (uint16_t)location->location;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_move_location_vgpr(
    const loom_low_allocation_table_t* allocation,
    const loom_low_move_location_t* location, uint16_t* out_register) {
  *out_register = 0;
  (void)allocation;
  if (location->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native encoding move location must be a "
                            "VGPR");
  }
  if (location->location > 255) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding VGPR index %" PRIu32
                            " is out of range",
                            location->location);
  }
  *out_register = (uint16_t)location->location;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_packet_result_sgpr(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, iree_host_size_t result_index,
    uint16_t* out_register) {
  if (result_index >= packet->node->op->result_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding result index is out of "
                            "range");
  }
  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_map_assignment(
          state->allocation,
          loom_op_const_results(packet->node->op)[result_index]);
  return loom_amdgpu_assignment_sgpr(state->allocation, assignment,
                                     out_register);
}

static iree_status_t loom_amdgpu_packet_result_vgpr(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, iree_host_size_t result_index,
    uint16_t* out_register) {
  if (result_index >= packet->node->op->result_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding result index is out of "
                            "range");
  }
  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_map_assignment(
          state->allocation,
          loom_op_const_results(packet->node->op)[result_index]);
  return loom_amdgpu_assignment_vgpr(state->allocation, assignment,
                                     out_register);
}

static iree_status_t loom_amdgpu_append_encoding_packet(
    loom_amdgpu_encode_state_t* state,
    const loom_amdgpu_encoding_packet_t* packet) {
  for (uint16_t i = 0; i < packet->word_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_u32(state, packet->words[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_packet_descriptor_operand_assignment(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_operand_index,
    const loom_low_allocation_assignment_t** out_assignment) {
  *out_assignment = NULL;
  const loom_op_t* op = packet->node->op;
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  if (descriptor_operand_index < packet->descriptor->result_count) {
    if (descriptor_operand_index >= op->result_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU native encoding result descriptor operand %" PRIu16
          " has no matching SSA result",
          descriptor_operand_index);
    }
    value_id = loom_op_const_results(op)[descriptor_operand_index];
  } else {
    const uint16_t operand_index =
        descriptor_operand_index - packet->descriptor->result_count;
    if (operand_index >= op->operand_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU native encoding descriptor operand %" PRIu16
          " has no matching SSA operand",
          descriptor_operand_index);
    }
    value_id = loom_op_const_operands(op)[operand_index];
  }
  *out_assignment = loom_amdgpu_map_assignment(state->allocation, value_id);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_assignment_field_value(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_allocation_assignment_t* assignment,
    const loom_low_operand_t* operand, uint64_t* out_value) {
  *out_value = 0;
  if (assignment->location_count != operand->unit_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU native encoding value %" PRIu32 " has %" PRIu32
        " assigned registers but descriptor field needs %" PRIu16,
        assignment->value_id, assignment->location_count, operand->unit_count);
  }
  const uint64_t last_register =
      (uint64_t)assignment->location_base + assignment->location_count - 1u;
  if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    if (last_register > 127) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU native encoding SGPR range s[%" PRIu32
                              ":%" PRIu64 "] is out of range",
                              assignment->location_base, last_register);
    }
    *out_value = assignment->location_base;
    return iree_ok_status();
  }
  if (assignment->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    if (last_register >= state->encoding_table->vector_source_vgpr_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU native encoding VGPR range v[%" PRIu32
                              ":%" PRIu64 "] is out of range",
                              assignment->location_base, last_register);
    }
    *out_value = assignment->location_base;
    if (loom_amdgpu_encoding_field_uses_unified_source(
            operand->encoding_field_id)) {
      *out_value += state->encoding_table->vector_source_vgpr0;
    }
    return iree_ok_status();
  }
  iree_string_view_t register_class = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_register_class_name(
      state->allocation, assignment, &register_class));
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU native encoding register class '%.*s' is not "
                          "supported by the generic packet encoder",
                          (int)register_class.size, register_class.data);
}

static iree_status_t loom_amdgpu_push_encoding_field_value(
    loom_amdgpu_encoding_field_value_t* field_values,
    iree_host_size_t* field_value_count, uint16_t field_id, uint64_t value) {
  if (field_id == 0) {
    return iree_ok_status();
  }
  if (*field_value_count >= LOOM_AMDGPU_MAX_PACKET_FIELD_VALUES) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AMDGPU native encoding field value buffer exceeded %u entries",
        LOOM_AMDGPU_MAX_PACKET_FIELD_VALUES);
  }
  field_values[*field_value_count] = (loom_amdgpu_encoding_field_value_t){
      .field_id = field_id,
      .reserved = 0,
      .value = value,
  };
  ++*field_value_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_descriptor_operand_field_already_encoded(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t operand_index,
    const loom_low_operand_t* operand, uint64_t value,
    bool* out_already_encoded) {
  *out_already_encoded = false;
  const loom_low_descriptor_set_t* descriptor_set =
      state->schedule->target.descriptor_set;
  const loom_low_operand_t* operands =
      &descriptor_set->operands[packet->descriptor->operand_start];
  for (uint16_t previous_index = 0; previous_index < operand_index;
       ++previous_index) {
    const loom_low_operand_t* previous = &operands[previous_index];
    if (previous->encoding_field_id != operand->encoding_field_id) {
      continue;
    }
    if (!loom_low_descriptor_operands_are_tied(descriptor_set,
                                               packet->descriptor,
                                               previous_index, operand_index)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU native encoding descriptor has repeated field id %" PRIu16
          " without a tied constraint",
          operand->encoding_field_id);
    }
    const loom_low_allocation_assignment_t* previous_assignment = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_packet_descriptor_operand_assignment(
        state, packet, previous_index, &previous_assignment));
    uint64_t previous_value = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_assignment_field_value(
        state, previous_assignment, previous, &previous_value));
    if (previous_value != value) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU native encoding tied descriptor operands for field id "
          "%" PRIu16 " were assigned different registers",
          operand->encoding_field_id);
    }
    *out_already_encoded = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_read_immediate_field_value(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index,
    int64_t* out_value) {
  *out_value = 0;
  if (descriptor_immediate_index >= packet->descriptor->immediate_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native encoding descriptor immediate index %" PRIu16
        " is out of range",
        descriptor_immediate_index);
  }
  const uint32_t immediate_row =
      packet->descriptor->immediate_start + descriptor_immediate_index;
  if (immediate_row >=
      state->schedule->target.descriptor_set->immediate_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native encoding descriptor immediate row %" PRIu32
        " is out of range",
        immediate_row);
  }
  const loom_low_immediate_t* immediate =
      &state->schedule->target.descriptor_set->immediates[immediate_row];
  const loom_string_id_t field_name_id =
      immediate_row < state->immediate_name_id_count
          ? state->immediate_name_ids[immediate_row]
          : LOOM_STRING_ID_INVALID;
  const loom_named_attr_t* attr = loom_amdgpu_find_packet_attr_by_name_id(
      loom_amdgpu_packet_attrs(packet), field_name_id);
  if (attr != NULL && attr->value.kind == LOOM_ATTR_I64) {
    *out_value = attr->value.i64;
    return iree_ok_status();
  }

  iree_string_view_t field_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_string(
      state->schedule->target.descriptor_set,
      immediate->field_name_string_offset, &field_name));
  if (attr == NULL) {
    if (iree_any_bit_set(immediate->flags,
                         LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
      *out_value = immediate->default_value;
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native encoding immediate '%.*s' is "
                            "required",
                            (int)field_name.size, field_name.data);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "AMDGPU native encoding immediate '%.*s' must be an "
                          "i64",
                          (int)field_name.size, field_name.data);
}

static iree_status_t loom_amdgpu_read_immediate_field_name(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index,
    iree_string_view_t* out_field_name) {
  *out_field_name = iree_string_view_empty();
  const uint32_t immediate_row =
      packet->descriptor->immediate_start + descriptor_immediate_index;
  const loom_low_immediate_t* immediate =
      &state->schedule->target.descriptor_set->immediates[immediate_row];
  return loom_amdgpu_descriptor_string(state->schedule->target.descriptor_set,
                                       immediate->field_name_string_offset,
                                       out_field_name);
}

static iree_status_t loom_amdgpu_read_immediate_u32(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index,
    uint32_t* out_value) {
  *out_value = 0;
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_field_value(
      state, packet, descriptor_immediate_index, &value));
  if (value < 0 || value > UINT32_MAX) {
    iree_string_view_t field_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_field_name(
        state, packet, descriptor_immediate_index, &field_name));
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding immediate '%.*s' value "
                            "%" PRId64 " is not a u32",
                            (int)field_name.size, field_name.data, value);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_read_immediate_u16(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index,
    uint16_t* out_value) {
  *out_value = 0;
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_field_value(
      state, packet, descriptor_immediate_index, &value));
  if (value < 0 || value > UINT16_MAX) {
    iree_string_view_t field_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_field_name(
        state, packet, descriptor_immediate_index, &field_name));
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding immediate '%.*s' value "
                            "%" PRId64 " is not a u16",
                            (int)field_name.size, field_name.data, value);
  }
  *out_value = (uint16_t)value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_read_immediate_encoding_field_value(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, uint16_t descriptor_immediate_index,
    uint64_t* out_value) {
  *out_value = 0;
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_field_value(
      state, packet, descriptor_immediate_index, &value));
  const uint32_t immediate_row =
      packet->descriptor->immediate_start + descriptor_immediate_index;
  const loom_low_immediate_t* immediate =
      &state->schedule->target.descriptor_set->immediates[immediate_row];
  if (immediate->kind == LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    const int64_t maximum = immediate->unsigned_max > INT64_MAX
                                ? INT64_MAX
                                : (int64_t)immediate->unsigned_max;
    if (value < immediate->signed_min || value > maximum) {
      iree_string_view_t field_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_string(
          state->schedule->target.descriptor_set,
          immediate->field_name_string_offset, &field_name));
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU native encoding signed immediate '%.*s' "
                              "value %" PRId64 " is out of range",
                              (int)field_name.size, field_name.data, value);
    }
    if (immediate->bit_width == 0 || immediate->bit_width > 64) {
      iree_string_view_t field_name = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_string(
          state->schedule->target.descriptor_set,
          immediate->field_name_string_offset, &field_name));
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU native encoding signed immediate '%.*s' has invalid bit "
          "width %" PRIu16,
          (int)field_name.size, field_name.data, immediate->bit_width);
    }
    *out_value =
        (uint64_t)value & loom_amdgpu_low_bit_mask(immediate->bit_width);
    return iree_ok_status();
  }
  if (value < 0) {
    iree_string_view_t field_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_string(
        state->schedule->target.descriptor_set,
        immediate->field_name_string_offset, &field_name));
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding immediate '%.*s' value "
                            "%" PRId64 " is negative",
                            (int)field_name.size, field_name.data, value);
  }
  *out_value = (uint64_t)value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_sop1_s_mov_b32(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  IREE_ASSERT(packet->descriptor->stable_id ==
              LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32);
  uint16_t sdst = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_result_sgpr(state, packet, 0, &sdst));
  uint32_t imm32 = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_read_immediate_u32(state, packet, 0, &imm32));
  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_s_mov_b32_u32(
      state->encoding_table, sdst, imm32, &encoded_packet));
  return loom_amdgpu_append_encoding_packet(state, &encoded_packet);
}

static iree_status_t loom_amdgpu_encode_s_mov_b32_register(
    loom_amdgpu_encode_state_t* state, uint16_t sdst, uint16_t ssrc0) {
  if (sdst == ssrc0) {
    return iree_ok_status();
  }
  if (sdst > 127 || ssrc0 > 127) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native encoding s_mov_b32 register operands must be SGPRs");
  }
  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_s_mov_b32_sgpr(
      state->encoding_table, sdst, ssrc0, &encoded_packet));
  return loom_amdgpu_append_encoding_packet(state, &encoded_packet);
}

static iree_status_t loom_amdgpu_encode_v_mov_b32_register(
    loom_amdgpu_encode_state_t* state, uint16_t vdst, uint16_t src0) {
  if (vdst == src0) {
    return iree_ok_status();
  }
  if (state->encoding_table == NULL) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU native encoding descriptor set '%.*s' has no encoding table "
        "for v_mov_b32 register moves",
        (int)state->target->descriptor_set_key.size,
        state->target->descriptor_set_key.data);
  }

  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_v_mov_b32_vgpr(
      state->encoding_table, vdst, src0, &encoded_packet));
  return loom_amdgpu_append_encoding_packet(state, &encoded_packet);
}

static iree_status_t loom_amdgpu_encode_v_mov_b32_u32(
    loom_amdgpu_encode_state_t* state, uint16_t vdst, uint32_t imm32) {
  if (state->encoding_table == NULL) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU native encoding descriptor set '%.*s' has no encoding table "
        "for v_mov_b32 literal moves",
        (int)state->target->descriptor_set_key.size,
        state->target->descriptor_set_key.data);
  }

  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_v_mov_b32_u32(
      state->encoding_table, vdst, imm32, &encoded_packet));
  return loom_amdgpu_append_encoding_packet(state, &encoded_packet);
}

static iree_status_t loom_amdgpu_encode_move(
    void* user_data, const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source) {
  loom_amdgpu_encode_state_t* state = (loom_amdgpu_encode_state_t*)user_data;
  if (destination->descriptor_reg_class_id != source->descriptor_reg_class_id) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU native encoding move between descriptor register class IDs "
        "%" PRIu16 " and %" PRIu16 " is unsupported",
        destination->descriptor_reg_class_id, source->descriptor_reg_class_id);
  }
  if (destination->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    uint16_t vdst = 0;
    uint16_t src0 = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_move_location_vgpr(state->allocation, destination, &vdst));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_move_location_vgpr(state->allocation, source, &src0));
    return loom_amdgpu_encode_v_mov_b32_register(state, vdst, src0);
  }
  if (destination->descriptor_reg_class_id != LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU native encoding move for descriptor register class ID %" PRIu16
        " is unsupported",
        destination->descriptor_reg_class_id);
  }
  uint16_t sdst = 0;
  uint16_t ssrc0 = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_move_location_sgpr(state->allocation, destination, &sdst));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_move_location_sgpr(state->allocation, source, &ssrc0));
  return loom_amdgpu_encode_s_mov_b32_register(state, sdst, ssrc0);
}

static iree_status_t loom_amdgpu_encode_move_sequence(
    loom_amdgpu_encode_state_t* state, iree_host_size_t move_count,
    const loom_low_move_location_t* temporary_locations,
    iree_host_size_t temporary_location_count) {
  loom_low_move_sequence_options_t options = {
      .temporary_locations = temporary_locations,
      .temporary_location_count = temporary_location_count,
      .emit_move =
          {
              .fn = loom_amdgpu_encode_move,
              .user_data = state,
          },
  };
  return loom_low_move_sequence_emit(state->move_scratch, move_count, &options);
}

static iree_status_t loom_amdgpu_encode_edge_copy_group(
    loom_amdgpu_encode_state_t* state,
    const loom_low_allocation_edge_copy_group_t* group) {
  iree_host_size_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_edge_copy_units(
      state->allocation, group, &move_count));
  if (move_count == 0) {
    return iree_ok_status();
  }
  loom_low_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_moves(
      state->move_scratch, move_count, &moves));
  loom_low_move_location_t* temporaries = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_temporaries(
      state->move_scratch, group->temporary_count, &temporaries));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_edge_copy_units(
      state->allocation, group, moves, move_count));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_edge_copy_temporaries(
      state->allocation, group, temporaries, group->temporary_count));
  return loom_amdgpu_encode_move_sequence(state, move_count, temporaries,
                                          group->temporary_count);
}

static iree_status_t loom_amdgpu_encode_copy_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  const loom_low_allocation_assignment_t* source_assignment =
      loom_amdgpu_map_assignment(state->allocation, loom_low_copy_source(op));
  const loom_low_allocation_assignment_t* result_assignment =
      loom_amdgpu_map_assignment(state->allocation, loom_low_copy_result(op));
  if (source_assignment->location_count != result_assignment->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding copy requires matching register ranges");
  }
  const uint32_t register_count = source_assignment->location_count;
  loom_low_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_moves(
      state->move_scratch, register_count, &moves));
  for (uint32_t i = 0; i < register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
        result_assignment, i, &moves[i].destination));
    IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
        source_assignment, i, &moves[i].source));
  }
  return loom_amdgpu_encode_move_sequence(state, register_count, NULL, 0);
}

static iree_status_t loom_amdgpu_encode_slice_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  iree_host_size_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_slice_units(
      state->allocation, op, &move_count));
  if (move_count == 0) {
    return iree_ok_status();
  }

  loom_low_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_moves(
      state->move_scratch, move_count, &moves));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_slice_units(
      state->allocation, op, moves, move_count));
  return loom_amdgpu_encode_move_sequence(state, move_count, NULL, 0);
}

static iree_status_t loom_amdgpu_encode_concat_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  iree_host_size_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_concat_units(
      state->allocation, op, &move_count));
  if (move_count == 0) {
    return iree_ok_status();
  }

  loom_low_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_scratch_reserve_moves(
      state->move_scratch, move_count, &moves));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_populate_concat_units(
      state->allocation, op, moves, move_count));
  return loom_amdgpu_encode_move_sequence(state, move_count, NULL, 0);
}

static iree_status_t loom_amdgpu_encode_sopp_simm16(
    loom_amdgpu_encode_state_t* state, uint16_t opcode, uint16_t immediate) {
  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_sopp_simm16(
      state->encoding_table, opcode, immediate, &encoded_packet));
  return loom_amdgpu_append_encoding_packet(state, &encoded_packet);
}

static iree_status_t loom_amdgpu_push_immediate_encoding_field_values(
    loom_amdgpu_encode_state_t* state, const loom_low_immediate_t* immediate,
    uint64_t value, loom_amdgpu_encoding_field_value_t* field_values,
    iree_host_size_t* field_value_count) {
  if (immediate->encoding_slice_count == 0) {
    return loom_amdgpu_push_encoding_field_value(
        field_values, field_value_count, immediate->encoding_field_id, value);
  }

  const loom_low_descriptor_set_t* descriptor_set =
      state->schedule->target.descriptor_set;
  for (uint16_t i = 0; i < immediate->encoding_slice_count; ++i) {
    const loom_low_immediate_encoding_slice_t* slice =
        &descriptor_set
             ->immediate_encoding_slices[immediate->encoding_slice_start + i];
    const uint64_t field_value = (value >> slice->source_bit_offset) &
                                 loom_amdgpu_low_bit_mask(slice->bit_count);
    IREE_RETURN_IF_ERROR(loom_amdgpu_push_encoding_field_value(
        field_values, field_value_count, slice->encoding_field_id,
        field_value));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_generic_descriptor_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  if (state->encoding_table == NULL) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU native encoding has no bit table for "
                            "descriptor set '%.*s'",
                            (int)state->target->descriptor_set_key.size,
                            state->target->descriptor_set_key.data);
  }

  loom_amdgpu_encoding_field_value_t
      field_values[LOOM_AMDGPU_MAX_PACKET_FIELD_VALUES];
  iree_host_size_t field_value_count = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      state->schedule->target.descriptor_set;
  for (uint16_t i = 0; i < packet->descriptor->encoding_field_value_count;
       ++i) {
    const loom_low_encoding_field_value_t* field_value =
        &descriptor_set->encoding_field_values
             [packet->descriptor->encoding_field_value_start + i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_push_encoding_field_value(
        field_values, &field_value_count, field_value->encoding_field_id,
        field_value->value));
  }

  for (uint16_t i = 0; i < packet->descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[packet->descriptor->operand_start + i];
    if (operand->encoding_field_id == 0) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_packet_descriptor_operand_assignment(
        state, packet, i, &assignment));
    uint64_t value = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_assignment_field_value(state, assignment, operand, &value));
    bool already_encoded = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_operand_field_already_encoded(
        state, packet, i, operand, value, &already_encoded));
    if (already_encoded) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_push_encoding_field_value(
        field_values, &field_value_count, operand->encoding_field_id, value));
  }

  for (uint16_t i = 0; i < packet->descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[packet->descriptor->immediate_start + i];
    if (immediate->encoding_field_id == 0 &&
        immediate->encoding_slice_count == 0) {
      continue;
    }
    uint64_t value = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_encoding_field_value(
        state, packet, i, &value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_push_immediate_encoding_field_values(
        state, immediate, value, field_values, &field_value_count));
  }

  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack(
      state->encoding_table, packet->descriptor->encoding_format_id,
      packet->descriptor->encoding_id, field_values, field_value_count,
      &encoded_packet));
  return loom_amdgpu_append_encoding_packet(state, &encoded_packet);
}

static iree_status_t loom_amdgpu_encode_descriptor_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  if (!state->target->supports_descriptor_packet_encoding) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU descriptor packet encoding for descriptor "
                            "set '%.*s' is not supported yet",
                            (int)state->target->descriptor_set_key.size,
                            state->target->descriptor_set_key.data);
  }
  switch (packet->descriptor->encoding_format_id) {
    case LOOM_AMDGPU_ENCODING_FORMAT_SOP1:
      if (packet->descriptor->stable_id ==
          LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32) {
        return loom_amdgpu_encode_sop1_s_mov_b32(state, packet);
      }
      return loom_amdgpu_encode_generic_descriptor_packet(state, packet);
    case LOOM_AMDGPU_ENCODING_FORMAT_SOPP:
      return loom_amdgpu_encode_generic_descriptor_packet(state, packet);
    case LOOM_AMDGPU_ENCODING_FORMAT_SOP2:
    case LOOM_AMDGPU_ENCODING_FORMAT_SOPC:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP2:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP2_LITERAL:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3P:
    case LOOM_AMDGPU_ENCODING_FORMAT_SOPK:
    case LOOM_AMDGPU_ENCODING_FORMAT_SMEM:
    case LOOM_AMDGPU_ENCODING_FORMAT_MUBUF:
    case LOOM_AMDGPU_ENCODING_FORMAT_VBUFFER:
    case LOOM_AMDGPU_ENCODING_FORMAT_DS:
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLBL:
    case LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLOBAL:
    case LOOM_AMDGPU_ENCODING_FORMAT_VDS:
    case LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1_LITERAL:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3P_MFMA:
      return loom_amdgpu_encode_generic_descriptor_packet(state, packet);
    default: {
      iree_string_view_t key = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_string(
          state->schedule->target.descriptor_set,
          packet->descriptor->key_string_offset, &key));
      iree_string_view_t format_name = loom_amdgpu_encoding_format_name(
          packet->descriptor->encoding_format_id);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU descriptor '%.*s' uses unsupported native encoding format "
          "'%.*s'",
          (int)key.size, key.data, (int)format_name.size, format_name.data);
    }
  }
}

static iree_status_t loom_amdgpu_encode_return_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  (void)packet;
  return loom_amdgpu_encode_sopp_simm16(state, state->target->s_endpgm_opcode,
                                        0);
}

static iree_status_t loom_amdgpu_encode_branch_offset(
    loom_amdgpu_encode_state_t* state, const loom_block_t* target_block,
    uint16_t* out_immediate) {
  *out_immediate = 0;
  const uint32_t target_block_index =
      loom_low_packet_block_index(state->schedule, target_block);
  if (target_block_index == LOOM_LOW_PACKET_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native encoding branch target block is not "
                            "in the scheduled low function");
  }
  if (state->data == NULL) {
    return iree_ok_status();
  }
  IREE_ASSERT(state->block_offsets != NULL);
  if (state->length > (iree_host_size_t)INT64_MAX - 4 ||
      state->block_offsets[target_block_index] > (iree_host_size_t)INT64_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native encoding branch target byte offset exceeds int64_t");
  }

  const int64_t branch_base_offset = (int64_t)state->length + 4;
  const int64_t target_offset =
      (int64_t)state->block_offsets[target_block_index];
  const int64_t relative_byte_offset = target_offset - branch_base_offset;
  if ((relative_byte_offset % 4) != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding branch target offset is not dword aligned");
  }
  const int64_t relative_dword_offset = relative_byte_offset / 4;
  if (relative_dword_offset < INT16_MIN || relative_dword_offset > INT16_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native encoding branch target offset %" PRId64
        " dword(s) does not fit a signed 16-bit SOPP label",
        relative_dword_offset);
  }
  *out_immediate =
      (uint16_t)((uint32_t)relative_dword_offset & UINT32_C(0xFFFF));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_branch_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  loom_value_slice_t args = loom_low_br_args(op);
  if (args.count != 0) {
    const loom_low_allocation_edge_copy_group_t* group =
        loom_low_allocation_find_edge_copy_group_by_source_ordinal(
            state->allocation, packet->node->source_ordinal);
    if (!group) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU native encoding branch edge copies are missing from "
          "allocation");
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_encode_edge_copy_group(state, group));
  }
  uint16_t immediate = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encode_branch_offset(
      state, loom_low_br_dest(op), &immediate));
  return loom_amdgpu_encode_sopp_simm16(state, state->target->s_branch_opcode,
                                        immediate);
}

static iree_status_t loom_amdgpu_encode_cond_branch_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  IREE_RETURN_IF_ERROR(loom_amdgpu_verify_scc_assignment(
      state->allocation, loom_low_cond_br_condition(op)));
  uint16_t true_immediate = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encode_branch_offset(
      state, loom_low_cond_br_true_dest(op), &true_immediate));
  IREE_RETURN_IF_ERROR(loom_amdgpu_encode_sopp_simm16(
      state, state->target->s_cbranch_scc1_opcode, true_immediate));
  uint16_t false_immediate = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encode_branch_offset(
      state, loom_low_cond_br_false_dest(op), &false_immediate));
  return loom_amdgpu_encode_sopp_simm16(state, state->target->s_branch_opcode,
                                        false_immediate);
}

static iree_status_t loom_amdgpu_encode_live_in_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  (void)state;
  (void)packet;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_storage_reserve_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  (void)state;
  (void)packet;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_storage_address_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  loom_amdgpu_storage_layout_reservation_t reservation;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_lookup(
      state->storage_layout, loom_low_storage_address_storage(op),
      &reservation));
  if (reservation.space != LOOM_STORAGE_SPACE_WORKGROUP) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU native encoding only supports "
                            "low.storage.address for workgroup storage");
  }
  const int64_t signed_offset = loom_low_storage_address_offset(op);
  if (signed_offset < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU native encoding low.storage.address offset must be "
        "non-negative");
  }
  const uint64_t offset = (uint64_t)signed_offset;
  if (reservation.byte_offset > UINT32_MAX ||
      offset > UINT32_MAX - reservation.byte_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native encoding low.storage.address byte offset exceeds u32");
  }
  uint16_t vdst = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_result_vgpr(state, packet, 0, &vdst));
  return loom_amdgpu_encode_v_mov_b32_u32(
      state, vdst, (uint32_t)(reservation.byte_offset + offset));
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

static iree_status_t loom_amdgpu_wait_packet_immediate_value(
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    const loom_amdgpu_wait_packet_t* wait_packet,
    uint16_t descriptor_immediate_index, uint16_t default_value,
    uint16_t* out_value) {
  *out_value = default_value;
  for (iree_host_size_t i = 0; i < wait_packet->immediate_count; ++i) {
    const iree_host_size_t immediate_index = wait_packet->immediate_start + i;
    const loom_amdgpu_wait_packet_immediate_t* immediate =
        &wait_packets->immediates[immediate_index];
    if (immediate->descriptor_immediate_index == descriptor_immediate_index) {
      *out_value = immediate->value;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_generic_wait_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_descriptor_t* descriptor,
    const loom_amdgpu_wait_packet_t* wait_packet) {
  if (state->encoding_table == NULL) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU native encoding has no bit table for "
                            "descriptor set '%.*s'",
                            (int)state->target->descriptor_set_key.size,
                            state->target->descriptor_set_key.data);
  }

  loom_amdgpu_encoding_field_value_t
      field_values[LOOM_AMDGPU_MAX_PACKET_FIELD_VALUES];
  iree_host_size_t field_value_count = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      state->schedule->target.descriptor_set;
  for (uint16_t i = 0; i < descriptor->encoding_field_value_count; ++i) {
    const loom_low_encoding_field_value_t* field_value =
        &descriptor_set
             ->encoding_field_values[descriptor->encoding_field_value_start +
                                     i];
    IREE_RETURN_IF_ERROR(loom_amdgpu_push_encoding_field_value(
        field_values, &field_value_count, field_value->encoding_field_id,
        field_value->value));
  }

  for (iree_host_size_t i = 0; i < wait_packet->immediate_count; ++i) {
    const iree_host_size_t immediate_index = wait_packet->immediate_start + i;
    const loom_amdgpu_wait_packet_immediate_t* packet_immediate =
        &state->wait_packets->immediates[immediate_index];
    if (packet_immediate->descriptor_immediate_index >=
        descriptor->immediate_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait packet immediate index %" PRIu16
                              " is out of range",
                              packet_immediate->descriptor_immediate_index);
    }
  }

  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    const uint32_t descriptor_immediate_row = descriptor->immediate_start + i;
    if (descriptor_immediate_row >= descriptor_set->immediate_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU wait packet descriptor immediate row %" PRIu32
          " is out of range",
          descriptor_immediate_row);
    }
    const loom_low_immediate_t* descriptor_immediate =
        &descriptor_set->immediates[descriptor_immediate_row];
    if (descriptor_immediate->encoding_field_id == 0) {
      continue;
    }
    uint16_t default_value = 0;
    if (iree_any_bit_set(descriptor_immediate->flags,
                         LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
      if (descriptor_immediate->default_value < 0 ||
          descriptor_immediate->default_value > UINT16_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "AMDGPU wait packet default immediate value is not a u16");
      }
      default_value = (uint16_t)descriptor_immediate->default_value;
    }
    uint16_t value = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_immediate_value(
        state->wait_packets, wait_packet, i, default_value, &value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_push_encoding_field_value(
        field_values, &field_value_count,
        descriptor_immediate->encoding_field_id, value));
  }

  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack(
      state->encoding_table, descriptor->encoding_format_id,
      descriptor->encoding_id, field_values, field_value_count,
      &encoded_packet));
  return loom_amdgpu_append_encoding_packet(state, &encoded_packet);
}

static iree_status_t loom_amdgpu_encode_wait_packet(
    loom_amdgpu_encode_state_t* state,
    const loom_amdgpu_wait_packet_t* wait_packet) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->schedule->target.descriptor_set;
  if (wait_packet->descriptor_ordinal >= descriptor_set->descriptor_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU wait packet descriptor ordinal %" PRIu32
                            " is out of range",
                            wait_packet->descriptor_ordinal);
  }
  if (wait_packet->immediate_start > state->wait_packets->immediate_count ||
      wait_packet->immediate_count >
          state->wait_packets->immediate_count - wait_packet->immediate_start) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait packet immediate range is out of range");
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[wait_packet->descriptor_ordinal];

  if (descriptor->encoding_format_id == LOOM_AMDGPU_ENCODING_FORMAT_SOPP ||
      descriptor->encoding_format_id == LOOM_AMDGPU_ENCODING_FORMAT_SOPK) {
    return loom_amdgpu_encode_generic_wait_packet(state, descriptor,
                                                  wait_packet);
  }
  iree_string_view_t format_name =
      loom_amdgpu_encoding_format_name(descriptor->encoding_format_id);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU wait packet uses unsupported native encoding "
                          "format '%.*s'",
                          (int)format_name.size, format_name.data);
}

static iree_status_t loom_amdgpu_encode_wait_packets_before_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  if (state->wait_packets == NULL) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = packet->node;
  while (state->next_wait_packet_index < state->wait_packets->packet_count) {
    const loom_amdgpu_wait_packet_t* wait_packet =
        &state->wait_packets->packets[state->next_wait_packet_index];
    if (loom_amdgpu_wait_packet_is_before_node(wait_packet, node)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU wait packet plan contains an insertion point before the "
          "current scheduled packet");
    }
    if (!loom_amdgpu_wait_packet_matches_packet(wait_packet, packet)) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_encode_wait_packet(state, wait_packet));
    ++state->next_wait_packet_index;
  }
  return iree_ok_status();
}

typedef iree_status_t (*loom_amdgpu_encode_structural_packet_fn_t)(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet);

typedef struct loom_amdgpu_encode_structural_dispatch_t {
  // Structural op handled by this row.
  loom_op_kind_t op_kind;
  // Encoder for the structural packet.
  loom_amdgpu_encode_structural_packet_fn_t encode;
} loom_amdgpu_encode_structural_dispatch_t;

static const loom_amdgpu_encode_structural_dispatch_t
    kLoomAmdgpuEncodeStructuralDispatch[] = {
        {LOOM_OP_LOW_LIVE_IN, loom_amdgpu_encode_live_in_packet},
        {LOOM_OP_LOW_STORAGE_RESERVE,
         loom_amdgpu_encode_storage_reserve_packet},
        {LOOM_OP_LOW_COPY, loom_amdgpu_encode_copy_packet},
        {LOOM_OP_LOW_SLICE, loom_amdgpu_encode_slice_packet},
        {LOOM_OP_LOW_CONCAT, loom_amdgpu_encode_concat_packet},
        {LOOM_OP_LOW_STORAGE_ADDRESS,
         loom_amdgpu_encode_storage_address_packet},
        {LOOM_OP_LOW_BR, loom_amdgpu_encode_branch_packet},
        {LOOM_OP_LOW_COND_BR, loom_amdgpu_encode_cond_branch_packet},
        {LOOM_OP_LOW_RETURN, loom_amdgpu_encode_return_packet},
};

static iree_status_t loom_amdgpu_encode_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_wait_packets_before_packet(state, packet));
  if (packet->descriptor != NULL) {
    return loom_amdgpu_encode_descriptor_packet(state, packet);
  }
  const loom_op_t* op = packet->node->op;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuEncodeStructuralDispatch); ++i) {
    const loom_amdgpu_encode_structural_dispatch_t* row =
        &kLoomAmdgpuEncodeStructuralDispatch[i];
    if (op->kind != row->op_kind) {
      continue;
    }
    return row->encode(state, packet);
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(state->schedule->module, op);
  iree_string_view_t op_name =
      vtable ? loom_bstring_view(vtable->name) : IREE_SV("<unknown>");
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU native encoding does not support "
                          "structural op %.*s",
                          (int)op_name.size, op_name.data);
}

static iree_status_t loom_amdgpu_resolve_encoding_target(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_descriptor_set_info_t** out_target) {
  *out_target = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_lookup_descriptor_set_by_ordinal(
      schedule->target.descriptor_set->descriptor_set_ordinal, out_target));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_verify_wait_packet_plan(
    const loom_low_schedule_table_t* schedule,
    const loom_amdgpu_wait_packet_plan_t* wait_packets) {
  if (wait_packets->wait_plan->schedule != schedule) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU native encoding wait packets must be derived from the encoded "
        "schedule");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_resolve_immediate_name_ids(
    const loom_low_schedule_table_t* schedule,
    const loom_string_id_t** out_immediate_name_ids,
    iree_host_size_t* out_immediate_name_id_count,
    iree_arena_allocator_t* arena) {
  *out_immediate_name_ids = NULL;
  *out_immediate_name_id_count = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  if (descriptor_set->immediate_count == 0) {
    return iree_ok_status();
  }

  loom_string_id_t* immediate_name_ids = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, descriptor_set->immediate_count, sizeof(*immediate_name_ids),
      (void**)&immediate_name_ids));
  for (iree_host_size_t i = 0; i < descriptor_set->immediate_count; ++i) {
    const loom_low_immediate_t* immediate = &descriptor_set->immediates[i];
    iree_string_view_t field_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_string(
        descriptor_set, immediate->field_name_string_offset, &field_name));
    immediate_name_ids[i] =
        loom_module_lookup_string(schedule->module, field_name);
  }
  *out_immediate_name_ids = immediate_name_ids;
  *out_immediate_name_id_count = descriptor_set->immediate_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_instruction_stream_into_state(
    loom_amdgpu_encode_state_t* state) {
  state->length = 0;
  state->next_wait_packet_index = 0;
  for (iree_host_size_t block_index = 0;
       block_index < state->schedule->block_count; ++block_index) {
    const loom_low_schedule_block_t* block =
        &state->schedule->blocks[block_index];
    if (state->block_offsets != NULL) {
      state->block_offsets[block_index] = state->length;
    }
    for (uint32_t scheduled_ordinal = 0;
         scheduled_ordinal < block->scheduled_node_count; ++scheduled_ordinal) {
      const iree_host_size_t packet_index =
          block->scheduled_node_start + scheduled_ordinal;
      loom_low_packet_view_t packet = {0};
      IREE_RETURN_IF_ERROR(loom_low_packet_view_at(
          state->schedule, state->allocation, packet_index, &packet));
      IREE_RETURN_IF_ERROR(loom_amdgpu_encode_packet(state, &packet));
    }
  }
  if (state->wait_packets != NULL &&
      state->next_wait_packet_index != state->wait_packets->packet_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding wait packet plan contains an unmatched "
        "insertion point");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_instruction_stream_internal(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena) {
  *out_text = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      loom_native_fragment_validate_emission_inputs(schedule, allocation));
  const loom_amdgpu_descriptor_set_info_t* target = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_encoding_target(schedule, &target));
  const loom_amdgpu_encoding_table_t* encoding_table =
      loom_amdgpu_encoding_table_for_descriptor_set_ordinal(
          target->descriptor_set_ordinal);
  const loom_string_id_t* immediate_name_ids = NULL;
  iree_host_size_t immediate_name_id_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_immediate_name_ids(
      schedule, &immediate_name_ids, &immediate_name_id_count, arena));
  loom_amdgpu_storage_layout_t storage_layout;
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_build(
      schedule->module, schedule->function_op, arena, &storage_layout));
  if (wait_packets != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_verify_wait_packet_plan(schedule, wait_packets));
  }
  iree_host_size_t* block_offsets = NULL;
  if (schedule->block_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, schedule->block_count,
                                                   sizeof(*block_offsets),
                                                   (void**)&block_offsets));
  }

  loom_low_allocation_value_scratch_t scratch = {0};
  iree_status_t status =
      loom_low_allocation_acquire_value_scratch(allocation, &scratch);
  loom_low_move_sequence_scratch_t move_scratch = {0};
  loom_low_move_sequence_scratch_initialize(arena, &move_scratch);
  loom_amdgpu_encode_state_t sizing_state = {
      .schedule = schedule,
      .allocation = allocation,
      .target = target,
      .encoding_table = encoding_table,
      .storage_layout = &storage_layout,
      .immediate_name_ids = immediate_name_ids,
      .immediate_name_id_count = immediate_name_id_count,
      .wait_packets = wait_packets,
      .arena = arena,
      .move_scratch = &move_scratch,
      .block_offsets = block_offsets,
  };
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_encode_instruction_stream_into_state(&sizing_state);
  }

  uint8_t* data = NULL;
  if (iree_status_is_ok(status) && sizing_state.length != 0) {
    status = iree_arena_allocate(arena, sizing_state.length, (void**)&data);
  }
  loom_amdgpu_encode_state_t writing_state = {
      .schedule = schedule,
      .allocation = allocation,
      .target = target,
      .encoding_table = encoding_table,
      .storage_layout = &storage_layout,
      .immediate_name_ids = immediate_name_ids,
      .immediate_name_id_count = immediate_name_id_count,
      .wait_packets = wait_packets,
      .arena = arena,
      .move_scratch = &move_scratch,
      .data = data,
      .capacity = sizing_state.length,
      .block_offsets = block_offsets,
  };
  if (iree_status_is_ok(status) && sizing_state.length != 0) {
    status = loom_amdgpu_encode_instruction_stream_into_state(&writing_state);
  }
  if (iree_status_is_ok(status) &&
      writing_state.length != sizing_state.length) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU native encoding wrote %" PRIhsz
                              " bytes after planning %" PRIhsz,
                              writing_state.length, sizing_state.length);
  }
  if (iree_status_is_ok(status) && sizing_state.length != 0) {
    *out_text = iree_make_const_byte_span(data, writing_state.length);
  }
  loom_low_allocation_release_value_scratch(&scratch);
  return status;
}

iree_status_t loom_amdgpu_encode_instruction_stream(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena) {
  return loom_amdgpu_encode_instruction_stream_internal(schedule, allocation,
                                                        NULL, out_text, arena);
}

iree_status_t loom_amdgpu_encode_instruction_stream_with_wait_packets(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena) {
  return loom_amdgpu_encode_instruction_stream_internal(
      schedule, allocation, wait_packets, out_text, arena);
}
