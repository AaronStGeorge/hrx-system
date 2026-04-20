// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/encoding.h"

#include <inttypes.h>

#include "loom/codegen/low/move_sequence.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/context.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/encoding.h"
#include "loom/target/arch/amdgpu/target_info.h"

#define LOOM_AMDGPU_SOP1_BASE UINT32_C(0xBE800000)
#define LOOM_AMDGPU_SOPP_BASE UINT32_C(0xBF800000)
#define LOOM_AMDGPU_SOP1_S_MOV_B32_OPCODE UINT16_C(0x00)
#define LOOM_AMDGPU_SOPP_S_WAITCNT_OPCODE UINT16_C(0x09)
#define LOOM_AMDGPU_SOPP_S_WAITCNT_DEPCTR_OPCODE UINT16_C(0x08)
#define LOOM_AMDGPU_SOPP_S_WAIT_IDLE_OPCODE UINT16_C(0x0A)
#define LOOM_AMDGPU_SISRC_LITERAL UINT16_C(255)
#define LOOM_AMDGPU_VGPR_SRC_BASE UINT16_C(0x100)
#define LOOM_AMDGPU_MAX_PACKET_FIELD_VALUES 32u
#define LOOM_AMDGPU_INLINE_MOVE_COUNT 16u

typedef struct loom_amdgpu_encode_state_t {
  // Schedule sidecar being encoded.
  const loom_low_schedule_sidecar_t* schedule;
  // Allocation sidecar supplying physical locations.
  const loom_low_allocation_sidecar_t* allocation;
  // Resolved native target encoding profile.
  const loom_amdgpu_descriptor_set_info_t* target;
  // Optional planned wait packets consumed in scheduled order.
  const loom_amdgpu_wait_packet_plan_t* wait_packets;
  // Arena used for transient encoding scratch and final byte storage.
  iree_arena_allocator_t* arena;
  // Destination byte storage, or NULL during the sizing pass.
  uint8_t* data;
  // Capacity of |data| in bytes, or zero during the sizing pass.
  iree_host_size_t capacity;
  // Number of bytes planned or written so far.
  iree_host_size_t length;
  // Next wait-packet row to compare with the current scheduled packet.
  iree_host_size_t next_wait_packet_index;
} loom_amdgpu_encode_state_t;

static iree_string_view_t loom_amdgpu_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (module == NULL || string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_status_t loom_amdgpu_descriptor_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, iree_string_view_t* out_string) {
  IREE_ASSERT_ARGUMENT(out_string);
  *out_string = iree_string_view_empty();
  return loom_low_descriptor_set_string(descriptor_set, string_offset,
                                        out_string);
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

static iree_status_t loom_amdgpu_read_i64_attr(const loom_module_t* module,
                                               loom_named_attr_slice_t attrs,
                                               iree_string_view_t name,
                                               int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (!iree_string_view_equal(
            loom_amdgpu_module_string(module, attr->name_id), name)) {
      continue;
    }
    if (attr->value.kind != LOOM_ATTR_I64) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU native encoding attribute '%.*s' must be an i64",
          (int)name.size, name.data);
    }
    *out_value = attr->value.i64;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "AMDGPU native encoding attribute '%.*s' is required",
                          (int)name.size, name.data);
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

static iree_status_t loom_amdgpu_find_assignment(
    const loom_low_allocation_sidecar_t* allocation, loom_value_id_t value_id,
    const loom_low_allocation_assignment_t** out_assignment) {
  IREE_ASSERT_ARGUMENT(out_assignment);
  *out_assignment = NULL;
  const loom_low_allocation_assignment_t* assignment =
      loom_low_packet_find_assignment(allocation, value_id, NULL);
  if (assignment == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU native encoding value %" PRIu32
                            " has no allocation assignment",
                            value_id);
  }
  if (assignment->location_kind !=
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU native encoding value %" PRIu32
                            " is not physically allocated",
                            value_id);
  }
  *out_assignment = assignment;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_assignment_register_class(
    const loom_low_allocation_sidecar_t* allocation,
    const loom_low_allocation_assignment_t* assignment,
    iree_string_view_t* out_register_class) {
  IREE_ASSERT_ARGUMENT(out_register_class);
  *out_register_class = loom_amdgpu_module_string(
      allocation->module, assignment->value_class.register_class_id);
  if (iree_string_view_is_empty(*out_register_class)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU native encoding value %" PRIu32
                            " has no register class",
                            assignment->value_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_assignment_sgpr(
    const loom_low_allocation_sidecar_t* allocation,
    const loom_low_allocation_assignment_t* assignment,
    uint16_t* out_register) {
  IREE_ASSERT_ARGUMENT(out_register);
  *out_register = 0;
  iree_string_view_t register_class = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_assignment_register_class(
      allocation, assignment, &register_class));
  if (!iree_string_view_equal(register_class, IREE_SV("amdgpu.sgpr"))) {
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

static iree_status_t loom_amdgpu_move_location_sgpr(
    const loom_low_allocation_sidecar_t* allocation,
    const loom_low_move_location_t* location, uint16_t* out_register) {
  IREE_ASSERT_ARGUMENT(out_register);
  *out_register = 0;
  if (location->location_kind !=
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding move location is not physically allocated");
  }
  iree_string_view_t register_class = loom_amdgpu_module_string(
      allocation->module, location->value_class.register_class_id);
  if (!iree_string_view_equal(register_class, IREE_SV("amdgpu.sgpr"))) {
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
    const loom_low_allocation_sidecar_t* allocation,
    const loom_low_move_location_t* location, uint16_t* out_register) {
  IREE_ASSERT_ARGUMENT(out_register);
  *out_register = 0;
  if (location->location_kind !=
      LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding move location is not physically allocated");
  }
  iree_string_view_t register_class = loom_amdgpu_module_string(
      allocation->module, location->value_class.register_class_id);
  if (!iree_string_view_equal(register_class, IREE_SV("amdgpu.vgpr"))) {
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
  const loom_low_allocation_assignment_t* assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_find_assignment(
      state->allocation, loom_op_const_results(packet->node->op)[result_index],
      &assignment));
  return loom_amdgpu_assignment_sgpr(state->allocation, assignment,
                                     out_register);
}

static iree_status_t loom_amdgpu_read_u32_attr(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_packet_view_t* packet, iree_string_view_t name,
    uint32_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_i64_attr(
      schedule->module, loom_amdgpu_packet_attrs(packet), name, &value));
  if (value < 0 || value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding attribute '%.*s' value "
                            "%" PRId64 " is not a u32",
                            (int)name.size, name.data, value);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_read_u16_attr(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_packet_view_t* packet, iree_string_view_t name,
    uint16_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_i64_attr(
      schedule->module, loom_amdgpu_packet_attrs(packet), name, &value));
  if (value < 0 || value > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU native encoding attribute '%.*s' value "
                            "%" PRId64 " is not a u16",
                            (int)name.size, name.data, value);
  }
  *out_value = (uint16_t)value;
  return iree_ok_status();
}

static bool loom_amdgpu_sisrc_inline_u32(uint32_t value, uint16_t* out_sisrc) {
  IREE_ASSERT_ARGUMENT(out_sisrc);
  if (value <= 64) {
    *out_sisrc = (uint16_t)(0x80u + value);
    return true;
  }
  return false;
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
  IREE_ASSERT_ARGUMENT(out_assignment);
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
  return loom_amdgpu_find_assignment(state->allocation, value_id,
                                     out_assignment);
}

static iree_status_t loom_amdgpu_assignment_field_value(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_allocation_assignment_t* assignment,
    const loom_low_operand_t* operand, uint64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  iree_string_view_t register_class = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_assignment_register_class(
      state->allocation, assignment, &register_class));
  if (assignment->location_count != operand->unit_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU native encoding value %" PRIu32 " has %" PRIu32
        " assigned registers but descriptor field needs %" PRIu16,
        assignment->value_id, assignment->location_count, operand->unit_count);
  }
  const uint64_t last_register =
      (uint64_t)assignment->location_base + assignment->location_count - 1u;
  if (iree_string_view_equal(register_class, IREE_SV("amdgpu.sgpr"))) {
    if (last_register > 127) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU native encoding SGPR range s[%" PRIu32
                              ":%" PRIu64 "] is out of range",
                              assignment->location_base, last_register);
    }
    *out_value = assignment->location_base;
    return iree_ok_status();
  }
  if (iree_string_view_equal(register_class, IREE_SV("amdgpu.vgpr"))) {
    if (last_register > 255) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU native encoding VGPR range v[%" PRIu32
                              ":%" PRIu64 "] is out of range",
                              assignment->location_base, last_register);
    }
    *out_value = assignment->location_base;
    if (loom_amdgpu_encoding_field_uses_unified_source(
            operand->encoding_field_id)) {
      *out_value += LOOM_AMDGPU_VGPR_SRC_BASE;
    }
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU native encoding register class '%.*s' is not "
                          "supported by the generic packet encoder",
                          (int)register_class.size, register_class.data);
}

static iree_status_t loom_amdgpu_push_encoding_field_value(
    loom_amdgpu_encoding_field_value_t* field_values,
    iree_host_size_t* field_value_count, uint16_t field_id, uint64_t value) {
  IREE_ASSERT_ARGUMENT(field_value_count);
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

static iree_status_t loom_amdgpu_read_immediate_field_value(
    const loom_amdgpu_encode_state_t* state,
    const loom_low_packet_view_t* packet, const loom_low_immediate_t* immediate,
    uint64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  iree_string_view_t field_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_string(
      state->schedule->target.descriptor_set,
      immediate->field_name_string_offset, &field_name));
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_i64_attr(
      state->schedule->module, loom_amdgpu_packet_attrs(packet), field_name,
      &value));
  if (value < 0) {
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
  if (packet->descriptor->encoding_id != LOOM_AMDGPU_SOP1_S_MOV_B32_OPCODE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU SOP1 opcode %" PRIu16
                            " is not supported by native encoding",
                            packet->descriptor->encoding_id);
  }
  uint16_t sdst = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_packet_result_sgpr(state, packet, 0, &sdst));
  uint32_t imm32 = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_read_u32_attr(state->schedule, packet,
                                                 IREE_SV("imm32"), &imm32));

  uint16_t ssrc0 = 0;
  const bool is_inline = loom_amdgpu_sisrc_inline_u32(imm32, &ssrc0);
  if (!is_inline) {
    ssrc0 = LOOM_AMDGPU_SISRC_LITERAL;
  }
  const uint32_t word = LOOM_AMDGPU_SOP1_BASE |
                        ((uint32_t)packet->descriptor->encoding_id << 8) |
                        ((uint32_t)sdst << 16) | ssrc0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_append_u32(state, word));
  if (!is_inline) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_append_u32(state, imm32));
  }
  return iree_ok_status();
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
  const uint32_t word = LOOM_AMDGPU_SOP1_BASE |
                        ((uint32_t)LOOM_AMDGPU_SOP1_S_MOV_B32_OPCODE << 8) |
                        ((uint32_t)sdst << 16) | ssrc0;
  return loom_amdgpu_append_u32(state, word);
}

static iree_status_t loom_amdgpu_v_mov_b32_opcode(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t* out_opcode) {
  IREE_ASSERT_ARGUMENT(out_opcode);
  *out_opcode = 0;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("amdgpu.v_mov_b32"), &descriptor_ordinal));
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  if (descriptor == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "AMDGPU native encoding descriptor "
                            "'amdgpu.v_mov_b32' is missing");
  }
  *out_opcode = descriptor->encoding_id;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_v_mov_b32_register(
    loom_amdgpu_encode_state_t* state, uint16_t vdst, uint16_t src0) {
  if (vdst == src0) {
    return iree_ok_status();
  }
  const loom_amdgpu_encoding_table_t* encoding_table =
      loom_amdgpu_encoding_table_for_descriptor_set(
          state->target->descriptor_set_key);
  if (encoding_table == NULL) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU native encoding descriptor set '%.*s' has no encoding table "
        "for v_mov_b32 register moves",
        (int)state->target->descriptor_set_key.size,
        state->target->descriptor_set_key.data);
  }

  uint16_t opcode = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_v_mov_b32_opcode(
      state->schedule->target.descriptor_set, &opcode));
  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack_v_mov_b32_vgpr(
      encoding_table, opcode, vdst, src0, &encoded_packet));
  return loom_amdgpu_append_encoding_packet(state, &encoded_packet);
}

static iree_status_t loom_amdgpu_encode_move(
    void* user_data, const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source) {
  loom_amdgpu_encode_state_t* state = (loom_amdgpu_encode_state_t*)user_data;
  iree_string_view_t destination_class = loom_amdgpu_module_string(
      state->allocation->module, destination->value_class.register_class_id);
  iree_string_view_t source_class = loom_amdgpu_module_string(
      state->allocation->module, source->value_class.register_class_id);
  if (!iree_string_view_equal(destination_class, source_class)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU native encoding move between register classes '%.*s' and "
        "'%.*s' is unsupported",
        (int)destination_class.size, destination_class.data,
        (int)source_class.size, source_class.data);
  }
  if (iree_string_view_equal(destination_class, IREE_SV("amdgpu.vgpr"))) {
    uint16_t vdst = 0;
    uint16_t src0 = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_move_location_vgpr(state->allocation, destination, &vdst));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_move_location_vgpr(state->allocation, source, &src0));
    return loom_amdgpu_encode_v_mov_b32_register(state, vdst, src0);
  }
  if (!iree_string_view_equal(destination_class, IREE_SV("amdgpu.sgpr"))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU native encoding move for register class '%.*s' is unsupported",
        (int)destination_class.size, destination_class.data);
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
    loom_amdgpu_encode_state_t* state, loom_low_move_t* moves,
    iree_host_size_t move_count) {
  loom_low_move_sequence_options_t options = {
      .emit_move =
          {
              .fn = loom_amdgpu_encode_move,
              .user_data = state,
          },
  };
  return loom_low_move_sequence_emit(moves, move_count, &options);
}

static iree_status_t loom_amdgpu_encode_copy_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  const loom_low_allocation_assignment_t* source_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_find_assignment(
      state->allocation, loom_low_copy_source(op), &source_assignment));
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_find_assignment(
      state->allocation, loom_low_copy_result(op), &result_assignment));
  if (source_assignment->location_count == 0 ||
      source_assignment->location_count != result_assignment->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding copy requires non-empty matching register "
        "ranges");
  }
  const uint32_t register_count = source_assignment->location_count;
  loom_low_move_t inline_moves[LOOM_AMDGPU_INLINE_MOVE_COUNT];
  loom_low_move_t* moves = inline_moves;
  if (register_count > IREE_ARRAYSIZE(inline_moves)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, register_count, sizeof(*moves), (void**)&moves));
  }
  for (uint32_t i = 0; i < register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
        result_assignment, i, &moves[i].destination));
    IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
        source_assignment, i, &moves[i].source));
  }
  return loom_amdgpu_encode_move_sequence(state, moves, register_count);
}

static iree_status_t loom_amdgpu_encode_slice_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  const loom_low_allocation_assignment_t* source_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_find_assignment(
      state->allocation, loom_low_slice_source(op), &source_assignment));
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_find_assignment(
      state->allocation, loom_low_slice_result(op), &result_assignment));
  if (source_assignment->location_count == 0 ||
      result_assignment->location_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding slice requires non-empty register ranges");
  }
  const int64_t offset = loom_low_slice_offset(op);
  if (offset < 0 || offset > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU native encoding slice offset must fit a non-negative uint32_t");
  }
  const uint32_t source_offset = (uint32_t)offset;
  if (source_offset > source_assignment->location_count ||
      result_assignment->location_count >
          source_assignment->location_count - source_offset) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding slice result range exceeds the source range");
  }

  const uint32_t move_count = result_assignment->location_count;
  loom_low_move_t inline_moves[LOOM_AMDGPU_INLINE_MOVE_COUNT];
  loom_low_move_t* moves = inline_moves;
  if (move_count > IREE_ARRAYSIZE(inline_moves)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, move_count, sizeof(*moves), (void**)&moves));
  }
  for (uint32_t i = 0; i < move_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
        result_assignment, i, &moves[i].destination));
    IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
        source_assignment, source_offset + i, &moves[i].source));
  }
  return loom_amdgpu_encode_move_sequence(state, moves, move_count);
}

static iree_status_t loom_amdgpu_encode_concat_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_find_assignment(
      state->allocation, loom_low_concat_result(op), &result_assignment));
  if (result_assignment->location_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding concat requires a non-empty result range");
  }
  loom_low_move_t inline_moves[LOOM_AMDGPU_INLINE_MOVE_COUNT];
  loom_low_move_t* moves = inline_moves;
  if (result_assignment->location_count > IREE_ARRAYSIZE(inline_moves)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, result_assignment->location_count, sizeof(*moves),
        (void**)&moves));
  }

  uint32_t result_register_index = 0;
  loom_value_slice_t sources = loom_low_concat_sources(op);
  for (uint16_t i = 0; i < sources.count; ++i) {
    const loom_low_allocation_assignment_t* source_assignment = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_find_assignment(
        state->allocation, sources.values[i], &source_assignment));
    if (source_assignment->location_count == 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU native encoding concat requires non-empty source ranges");
    }
    if (result_register_index > result_assignment->location_count ||
        source_assignment->location_count >
            result_assignment->location_count - result_register_index) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU native encoding concat source ranges exceed the result "
          "range");
    }
    for (uint32_t source_register_index = 0;
         source_register_index < source_assignment->location_count;
         ++source_register_index) {
      IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
          result_assignment, result_register_index,
          &moves[result_register_index].destination));
      IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
          source_assignment, source_register_index,
          &moves[result_register_index].source));
      ++result_register_index;
    }
  }
  if (result_register_index != result_assignment->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU native encoding concat source ranges do not fill the result "
        "range");
  }
  return loom_amdgpu_encode_move_sequence(state, moves,
                                          result_assignment->location_count);
}

static iree_status_t loom_amdgpu_encode_sopp_word(
    loom_amdgpu_encode_state_t* state, uint16_t opcode, uint16_t immediate) {
  if (opcode > 0x7F) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU SOPP opcode %" PRIu16 " is out of range",
                            opcode);
  }
  const uint32_t word =
      LOOM_AMDGPU_SOPP_BASE | ((uint32_t)opcode << 16) | immediate;
  return loom_amdgpu_append_u32(state, word);
}

static iree_status_t loom_amdgpu_encode_gfx11_waitcnt_immediate(
    uint16_t vmcnt, uint16_t lgkmcnt, uint16_t* out_immediate) {
  IREE_ASSERT_ARGUMENT(out_immediate);
  *out_immediate = 0;
  if (vmcnt > 63 || lgkmcnt > 63) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU GFX11 waitcnt values must fit vmcnt/lgkmcnt 6-bit fields");
  }
  *out_immediate = (uint16_t)(UINT16_C(0x0007) | ((uint16_t)vmcnt << 10) |
                              ((uint16_t)lgkmcnt << 4));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_sopp_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  const uint16_t opcode = packet->descriptor->encoding_id;
  if (opcode == LOOM_AMDGPU_SOPP_S_WAITCNT_OPCODE) {
    uint16_t vmcnt = 0;
    uint16_t lgkmcnt = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_read_u16_attr(state->schedule, packet,
                                                   IREE_SV("vmcnt"), &vmcnt));
    IREE_RETURN_IF_ERROR(loom_amdgpu_read_u16_attr(
        state->schedule, packet, IREE_SV("lgkmcnt"), &lgkmcnt));
    uint16_t immediate = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_encode_gfx11_waitcnt_immediate(vmcnt, lgkmcnt, &immediate));
    return loom_amdgpu_encode_sopp_word(state, opcode, immediate);
  }
  if (opcode == LOOM_AMDGPU_SOPP_S_WAITCNT_DEPCTR_OPCODE) {
    uint16_t depctr = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_read_u16_attr(state->schedule, packet,
                                                   IREE_SV("depctr"), &depctr));
    return loom_amdgpu_encode_sopp_word(state, opcode, depctr);
  }
  if (opcode == LOOM_AMDGPU_SOPP_S_WAIT_IDLE_OPCODE) {
    return loom_amdgpu_encode_sopp_word(state, opcode, 0);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU SOPP opcode %" PRIu16
                          " is not supported by native encoding",
                          opcode);
}

static iree_status_t loom_amdgpu_encode_generic_descriptor_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  iree_string_view_t descriptor_set_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_string(
      state->schedule->target.descriptor_set,
      state->schedule->target.descriptor_set->key_string_offset,
      &descriptor_set_key));
  const loom_amdgpu_encoding_table_t* encoding_table =
      loom_amdgpu_encoding_table_for_descriptor_set(descriptor_set_key);
  if (encoding_table == NULL) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU native encoding has no bit table for "
                            "descriptor set '%.*s'",
                            (int)descriptor_set_key.size,
                            descriptor_set_key.data);
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
    IREE_RETURN_IF_ERROR(loom_amdgpu_push_encoding_field_value(
        field_values, &field_value_count, operand->encoding_field_id, value));
  }

  for (uint16_t i = 0; i < packet->descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[packet->descriptor->immediate_start + i];
    if (immediate->encoding_field_id == 0) {
      continue;
    }
    uint64_t value = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_read_immediate_field_value(
        state, packet, immediate, &value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_push_encoding_field_value(
        field_values, &field_value_count, immediate->encoding_field_id, value));
  }

  loom_amdgpu_encoding_packet_t encoded_packet;
  IREE_RETURN_IF_ERROR(loom_amdgpu_encoding_pack(
      encoding_table, packet->descriptor->encoding_format_id,
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
      return loom_amdgpu_encode_sop1_s_mov_b32(state, packet);
    case LOOM_AMDGPU_ENCODING_FORMAT_SOPP:
      return loom_amdgpu_encode_sopp_packet(state, packet);
    case LOOM_AMDGPU_ENCODING_FORMAT_SOP2:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP2:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP3:
    case LOOM_AMDGPU_ENCODING_FORMAT_SMEM:
    case LOOM_AMDGPU_ENCODING_FORMAT_MUBUF:
    case LOOM_AMDGPU_ENCODING_FORMAT_VOP1_LITERAL:
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
  loom_value_slice_t values = loom_low_return_values(packet->node->op);
  if (values.count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU native encoding return values require ABI lowering");
  }
  return loom_amdgpu_encode_sopp_word(state, state->target->s_endpgm_opcode, 0);
}

static iree_status_t loom_amdgpu_encode_live_in_packet(
    loom_amdgpu_encode_state_t* state, const loom_low_packet_view_t* packet) {
  (void)state;
  (void)packet;
  return iree_ok_status();
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
    const loom_amdgpu_wait_packet_t* wait_packet, iree_string_view_t name,
    uint16_t default_value, uint16_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = default_value;
  for (iree_host_size_t i = 0; i < wait_packet->immediate_count; ++i) {
    const iree_host_size_t immediate_index = wait_packet->immediate_start + i;
    const loom_amdgpu_wait_packet_immediate_t* immediate =
        &wait_packets->immediates[immediate_index];
    if (iree_string_view_equal(immediate->name, name)) {
      *out_value = immediate->value;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
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
  if (descriptor->encoding_format_id != LOOM_AMDGPU_ENCODING_FORMAT_SOPP) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU wait packet descriptor is not a SOPP instruction");
  }

  const uint16_t opcode = descriptor->encoding_id;
  if (opcode == LOOM_AMDGPU_SOPP_S_WAITCNT_OPCODE) {
    uint16_t vmcnt = 63;
    uint16_t lgkmcnt = 63;
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_immediate_value(
        state->wait_packets, wait_packet, IREE_SV("vmcnt"), vmcnt, &vmcnt));
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_immediate_value(
        state->wait_packets, wait_packet, IREE_SV("lgkmcnt"), lgkmcnt,
        &lgkmcnt));
    uint16_t immediate = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_encode_gfx11_waitcnt_immediate(vmcnt, lgkmcnt, &immediate));
    return loom_amdgpu_encode_sopp_word(state, opcode, immediate);
  }
  if (opcode == LOOM_AMDGPU_SOPP_S_WAITCNT_DEPCTR_OPCODE) {
    uint16_t depctr = 0;
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_immediate_value(
        state->wait_packets, wait_packet, IREE_SV("depctr"), depctr, &depctr));
    return loom_amdgpu_encode_sopp_word(state, opcode, depctr);
  }
  if (opcode == LOOM_AMDGPU_SOPP_S_WAIT_IDLE_OPCODE) {
    return loom_amdgpu_encode_sopp_word(state, opcode, 0);
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU wait packet SOPP opcode %" PRIu16
                          " is not supported by native encoding",
                          opcode);
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
        {LOOM_OP_LOW_COPY, loom_amdgpu_encode_copy_packet},
        {LOOM_OP_LOW_SLICE, loom_amdgpu_encode_slice_packet},
        {LOOM_OP_LOW_CONCAT, loom_amdgpu_encode_concat_packet},
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
    const loom_low_schedule_sidecar_t* schedule,
    const loom_amdgpu_descriptor_set_info_t** out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = NULL;
  if (schedule == NULL || schedule->target.descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native encoding schedule target is "
                            "required");
  }
  iree_string_view_t target_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_string(
      schedule->target.descriptor_set,
      schedule->target.descriptor_set->target_key_string_offset, &target_key));
  if (!iree_string_view_equal(target_key, IREE_SV("amdgpu"))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU native encoder received target '%.*s'",
                            (int)target_key.size, target_key.data);
  }
  iree_string_view_t descriptor_set_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_descriptor_string(
      schedule->target.descriptor_set,
      schedule->target.descriptor_set->key_string_offset, &descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_lookup_descriptor_set(
      descriptor_set_key, out_target));
  if (schedule->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU native encoding requires a single low function block");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_verify_wait_packet_plan(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_amdgpu_wait_packet_plan_t* wait_packets) {
  if (wait_packets == NULL || wait_packets->wait_plan == NULL ||
      wait_packets->wait_plan->schedule != schedule) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU native encoding wait packets must be derived from the encoded "
        "schedule");
  }
  if (wait_packets->packet_count != 0 && wait_packets->packets == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU native encoding wait packet rows are "
                            "required");
  }
  if (wait_packets->immediate_count != 0 && wait_packets->immediates == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU native encoding wait packet immediate rows are required");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_encode_instruction_stream_into_state(
    loom_amdgpu_encode_state_t* state) {
  state->length = 0;
  state->next_wait_packet_index = 0;
  const iree_host_size_t packet_count = loom_low_packet_count(state->schedule);
  for (iree_host_size_t packet_index = 0; packet_index < packet_count;
       ++packet_index) {
    loom_low_packet_view_t packet = {0};
    IREE_RETURN_IF_ERROR(loom_low_packet_view_at(
        state->schedule, state->allocation, packet_index, &packet));
    IREE_RETURN_IF_ERROR(loom_amdgpu_encode_packet(state, &packet));
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
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(out_text);
  IREE_ASSERT_ARGUMENT(arena);
  *out_text = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(loom_low_packet_validate_sidecars(schedule, allocation));
  const loom_amdgpu_descriptor_set_info_t* target = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_encoding_target(schedule, &target));
  if (wait_packets != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_verify_wait_packet_plan(schedule, wait_packets));
  }

  loom_amdgpu_encode_state_t sizing_state = {
      .schedule = schedule,
      .allocation = allocation,
      .target = target,
      .wait_packets = wait_packets,
      .arena = arena,
  };
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_instruction_stream_into_state(&sizing_state));
  if (sizing_state.length == 0) {
    return iree_ok_status();
  }

  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizing_state.length, (void**)&data));
  loom_amdgpu_encode_state_t writing_state = {
      .schedule = schedule,
      .allocation = allocation,
      .target = target,
      .wait_packets = wait_packets,
      .arena = arena,
      .data = data,
      .capacity = sizing_state.length,
  };
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_encode_instruction_stream_into_state(&writing_state));
  if (writing_state.length != sizing_state.length) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU native encoding wrote %" PRIhsz
                            " bytes after planning %" PRIhsz,
                            writing_state.length, sizing_state.length);
  }
  *out_text = iree_make_const_byte_span(data, writing_state.length);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_encode_instruction_stream(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena) {
  return loom_amdgpu_encode_instruction_stream_internal(schedule, allocation,
                                                        NULL, out_text, arena);
}

iree_status_t loom_amdgpu_encode_instruction_stream_with_wait_packets(
    const loom_low_schedule_sidecar_t* schedule,
    const loom_low_allocation_sidecar_t* allocation,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    iree_const_byte_span_t* out_text, iree_arena_allocator_t* arena) {
  return loom_amdgpu_encode_instruction_stream_internal(
      schedule, allocation, wait_packets, out_text, arena);
}
