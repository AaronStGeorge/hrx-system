// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/feedback.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/feedback_abi.h"
#include "loom/target/arch/amdgpu/lower/data_symbol.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/signal.h"
#include "loom/target/arch/amdgpu/lower/system_memory.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/registers.h"

static loom_amdgpu_feedback_config_values_t
loom_amdgpu_feedback_config_values_empty(void) {
  return (loom_amdgpu_feedback_config_values_t){
      .address = LOOM_VALUE_ID_INVALID,
      .flags = LOOM_VALUE_ID_INVALID,
      .channel_base = LOOM_VALUE_ID_INVALID,
      .notify_signal = LOOM_VALUE_ID_INVALID,
  };
}

static loom_amdgpu_feedback_channel_header_values_t
loom_amdgpu_feedback_channel_header_values_empty(void) {
  return (loom_amdgpu_feedback_channel_header_values_t){
      .address = LOOM_VALUE_ID_INVALID,
      .record_length = LOOM_VALUE_ID_INVALID,
      .abi_version = LOOM_VALUE_ID_INVALID,
      .flags = LOOM_VALUE_ID_INVALID,
      .ring_base = LOOM_VALUE_ID_INVALID,
      .ring_capacity = LOOM_VALUE_ID_INVALID,
  };
}

static loom_amdgpu_feedback_packet_address_t
loom_amdgpu_feedback_packet_address_empty(void) {
  return (loom_amdgpu_feedback_packet_address_t){
      .base = LOOM_VALUE_ID_INVALID,
      .byte_offset = LOOM_VALUE_ID_INVALID,
  };
}

static loom_amdgpu_feedback_reservation_attempt_t
loom_amdgpu_feedback_reservation_attempt_empty(void) {
  return (loom_amdgpu_feedback_reservation_attempt_t){
      .expected_head = LOOM_VALUE_ID_INVALID,
      .next_head = LOOM_VALUE_ID_INVALID,
      .observed_head = LOOM_VALUE_ID_INVALID,
      .cas_succeeded = LOOM_VALUE_ID_INVALID,
  };
}

typedef uint32_t loom_amdgpu_feedback_global_load_flags_t;

enum loom_amdgpu_feedback_global_load_flag_bits_e {
  LOOM_AMDGPU_FEEDBACK_GLOBAL_LOAD_FLAG_NONE = 0u,
  LOOM_AMDGPU_FEEDBACK_GLOBAL_LOAD_FLAG_ACQUIRE = 1u << 0,
};

static bool loom_amdgpu_feedback_packet_record_length_is_valid(
    uint32_t record_length) {
  return record_length >= LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH &&
         (record_length & (LOOM_AMDGPU_FEEDBACK_PACKET_ALIGNMENT - 1u)) == 0 &&
         record_length <= loom_amdgpu_feedback_packet_length(
                              LOOM_AMDGPU_FEEDBACK_PACKET_MAX_PAYLOAD_LENGTH);
}

static iree_status_t loom_amdgpu_feedback_build_u32_attr(
    loom_builder_t* builder, iree_string_view_t name, uint32_t value,
    loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, name, &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_offset_attr(
    loom_builder_t* builder, uint32_t byte_offset,
    loom_named_attr_t* out_attr) {
  return loom_amdgpu_feedback_build_u32_attr(builder, IREE_SV("offset"),
                                             byte_offset, out_attr);
}

static bool loom_amdgpu_feedback_type_is_register_class(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t type,
    uint16_t reg_class_id) {
  return loom_low_type_is_register(type) &&
         loom_low_register_type_descriptor_set_stable_id(type) ==
             descriptor_set->stable_id &&
         loom_low_register_type_class_id(type) == reg_class_id;
}

static iree_status_t loom_amdgpu_feedback_require_register_class(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint16_t reg_class_id, uint32_t unit_count) {
  if (value >= builder->module->values.count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU feedback builder received an invalid low value");
  }
  const loom_type_t type = loom_module_value_type(builder->module, value);
  if (!loom_amdgpu_feedback_type_is_register_class(descriptor_set, type,
                                                   reg_class_id) ||
      loom_low_register_type_unit_count(type) != unit_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU feedback builder received a low value with an unsupported "
        "register shape");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_descriptor_op(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_location_id_t location, loom_op_t** out_op) {
  *out_op = NULL;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, operands, operand_count,
      loom_make_named_attr_slice(NULL, 0), result_types, result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, out_op);
}

static iree_status_t loom_amdgpu_feedback_build_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t value,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));

  loom_named_attr_t imm32_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_u32_attr(
      builder, IREE_SV("imm32"), value, &imm32_attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor, opcode_id,
      loom_make_named_attr_slice(&imm32_attr, 1), result_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_descriptor_operand_type(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t descriptor_operand_index,
    loom_type_t* out_type) {
  *out_type = loom_type_none();
  if (descriptor_operand_index >= descriptor->operand_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU feedback descriptor operand index is out of range");
  }
  const uint32_t operand_index =
      (uint32_t)descriptor->operand_start + descriptor_operand_index;
  if (operand_index >= descriptor_set->operand_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU feedback descriptor operand row is out of range");
  }
  const loom_low_operand_t* operand = &descriptor_set->operands[operand_index];
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const uint32_t alt_index = operand->reg_class_alt_start + i;
    if (alt_index >= descriptor_set->reg_class_alt_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU feedback descriptor operand register-class alternative is "
          "out of range");
    }
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[alt_index];
    if (iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    return loom_low_build_register_type(descriptor_set, alt->reg_class_id,
                                        operand->unit_count, out_type);
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU feedback descriptor operand has no register alternative");
}

static iree_status_t loom_amdgpu_feedback_asm_operand_type(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t asm_operand_index,
    loom_type_t* out_type) {
  *out_type = loom_type_none();
  if (descriptor->canonical_asm_form_ordinal >=
      descriptor_set->asm_form_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU feedback descriptor has no canonical asm form");
  }
  const loom_low_asm_form_t* asm_form =
      &descriptor_set->asm_forms[descriptor->canonical_asm_form_ordinal];
  if (asm_operand_index >= asm_form->operand_index_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU feedback asm operand index is out of "
                            "range");
  }
  const uint32_t operand_index =
      (uint32_t)asm_form->operand_index_start + asm_operand_index;
  if (operand_index >= descriptor_set->asm_operand_index_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU feedback asm operand row is out of range");
  }
  return loom_amdgpu_feedback_descriptor_operand_type(
      descriptor_set, descriptor,
      descriptor_set->asm_operand_indices[operand_index], out_type);
}

static iree_status_t loom_amdgpu_feedback_build_m0_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* consumer_descriptor,
    uint16_t consumer_asm_operand_index, uint32_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_asm_operand_type(
      descriptor_set, consumer_descriptor, consumer_asm_operand_index,
      &m0_type));
  return loom_amdgpu_feedback_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM,
      value, m0_type, location, out_value);
}

static iree_status_t loom_amdgpu_feedback_build_vgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  return loom_amdgpu_feedback_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
      vgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_feedback_build_vgpr_u64_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, (uint32_t)value, location, &low_value_lo));
  loom_value_id_t low_value_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, (uint32_t)(value >> 32), location,
      &low_value_hi));

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));
  const loom_value_id_t parts[] = {low_value_lo, low_value_hi};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), vgpr_x2_type,
                            location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_register_lane(
    loom_builder_t* builder, loom_value_id_t source, uint32_t lane_index,
    loom_type_t lane_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, source, lane_index,
                                            lane_type, location, &slice_op));
  *out_value = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_vgpr64_add(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_sum) {
  *out_sum = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, lhs, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, rhs, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));

  loom_value_id_t lhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, lhs, /*lane_index=*/0, vgpr_type, location, &lhs_lo));
  loom_value_id_t lhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, lhs, /*lane_index=*/1, vgpr_type, location, &lhs_hi));
  loom_value_id_t rhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, rhs, /*lane_index=*/0, vgpr_type, location, &rhs_lo));
  loom_value_id_t rhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, rhs, /*lane_index=*/1, vgpr_type, location, &rhs_hi));

  const loom_value_id_t add_lo_operands[] = {lhs_lo, rhs_lo};
  const loom_type_t add_result_types[] = {vgpr_type, sgpr_x2_type};
  loom_op_t* add_lo_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_U32,
      add_lo_operands, IREE_ARRAYSIZE(add_lo_operands), add_result_types,
      IREE_ARRAYSIZE(add_result_types), location, &add_lo_op));
  const loom_value_id_t sum_lo =
      loom_value_slice_get(loom_low_op_results(add_lo_op), 0);
  const loom_value_id_t carry =
      loom_value_slice_get(loom_low_op_results(add_lo_op), 1);

  const loom_value_id_t add_hi_operands[] = {lhs_hi, rhs_hi, carry};
  loom_op_t* add_hi_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_CI_U32,
      add_hi_operands, IREE_ARRAYSIZE(add_hi_operands), add_result_types,
      IREE_ARRAYSIZE(add_result_types), location, &add_hi_op));
  const loom_value_id_t sum_hi =
      loom_value_slice_get(loom_low_op_results(add_hi_op), 0);

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));
  const loom_value_id_t parts[] = {sum_lo, sum_hi};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), vgpr_x2_type,
                            location, &concat_op));
  *out_sum = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_vgpr64_equal_mask(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, lhs, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, rhs, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &mask_type));

  loom_value_id_t lhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, lhs, /*lane_index=*/0, vgpr_type, location, &lhs_lo));
  loom_value_id_t lhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, lhs, /*lane_index=*/1, vgpr_type, location, &lhs_hi));
  loom_value_id_t rhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, rhs, /*lane_index=*/0, vgpr_type, location, &rhs_lo));
  loom_value_id_t rhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_register_lane(
      builder, rhs, /*lane_index=*/1, vgpr_type, location, &rhs_hi));

  const loom_value_id_t compare_hi_operands[] = {lhs_hi, rhs_hi};
  loom_op_t* compare_hi_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      compare_hi_operands, IREE_ARRAYSIZE(compare_hi_operands), &mask_type,
      /*result_count=*/1, location, &compare_hi_op));
  const loom_value_id_t high_mask =
      loom_value_slice_get(loom_low_op_results(compare_hi_op), 0);

  const loom_value_id_t compare_lo_operands[] = {lhs_lo, rhs_lo};
  loom_op_t* compare_lo_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      compare_lo_operands, IREE_ARRAYSIZE(compare_lo_operands), &mask_type,
      /*result_count=*/1, location, &compare_lo_op));
  const loom_value_id_t low_mask =
      loom_value_slice_get(loom_low_op_results(compare_lo_op), 0);

  const loom_value_id_t and_operands[] = {high_mask, low_mask};
  loom_op_t* and_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_descriptor_op(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
      and_operands, IREE_ARRAYSIZE(and_operands), &mask_type,
      /*result_count=*/1, location, &and_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(and_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_vgpr_pair(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t low_value, loom_value_id_t high_value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_x4_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 4, &vgpr_x4_type));
  const loom_value_id_t parts[] = {low_value, high_value};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, parts, IREE_ARRAYSIZE(parts), vgpr_x4_type,
                            location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_vgpr_b32_copy(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
      &descriptor, &opcode_id));
  loom_value_id_t operands[] = {source};
  loom_op_t* copy_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &vgpr_type,
      /*result_count=*/1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, &copy_op));
  *out_value = loom_value_slice_get(loom_low_op_results(copy_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_materialize_vgpr_registers(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, uint32_t expected_unit_count,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = source;
  if (source >= builder->module->values.count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU feedback builder received an invalid low value");
  }
  const loom_type_t source_type =
      loom_module_value_type(builder->module, source);
  if (!loom_low_type_is_register(source_type) ||
      loom_low_register_type_unit_count(source_type) != expected_unit_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU feedback builder cannot materialize value with unsupported "
        "register shape");
  }
  if (loom_amdgpu_feedback_type_is_register_class(
          descriptor_set, source_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_feedback_type_is_register_class(
          descriptor_set, source_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU feedback builder cannot materialize non-SGPR value into VGPR");
  }
  if (expected_unit_count > 2) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU feedback builder cannot materialize wide VGPR value");
  }
  if (expected_unit_count == 1) {
    return loom_amdgpu_feedback_build_vgpr_b32_copy(
        builder, descriptor_set, source, location, out_value);
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_value_id_t lanes[2];
  for (uint32_t i = 0; i < expected_unit_count; ++i) {
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, source, i, sgpr_type,
                                              location, &slice_op));
    const loom_value_id_t sgpr_lane = loom_low_slice_result(slice_op);
    IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_b32_copy(
        builder, descriptor_set, sgpr_lane, location, &lanes[i]));
  }

  loom_type_t vgpr_range_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, expected_unit_count,
      &vgpr_range_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, lanes, expected_unit_count,
                            vgpr_range_type, location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_global_store(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, loom_value_id_t value, uint32_t value_unit_count,
    const loom_named_attr_t* extra_attrs, iree_host_size_t extra_attr_count,
    loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, packet_address->base,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, packet_address->byte_offset,
      LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1));

  loom_value_id_t vgpr_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_materialize_vgpr_registers(
      builder, descriptor_set, value, value_unit_count, location, &vgpr_value));

  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));
  if (descriptor->canonical_asm_form_ordinal >=
      descriptor_set->asm_form_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU feedback global store descriptor has no canonical asm form");
  }
  const loom_low_asm_form_t* asm_form =
      &descriptor_set->asm_forms[descriptor->canonical_asm_form_ordinal];
  if (asm_form->operand_index_count != 3 &&
      asm_form->operand_index_count != 4) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU feedback global store descriptor has an unsupported packet "
        "operand count");
  }
  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_offset_attr(
      builder, byte_offset, &attrs[attr_count++]));
  if (extra_attr_count > IREE_ARRAYSIZE(attrs) - attr_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU feedback global store attr capacity "
                            "exceeded");
  }
  for (iree_host_size_t i = 0; i < extra_attr_count; ++i) {
    attrs[attr_count++] = extra_attrs[i];
  }
  loom_value_id_t operands[4] = {packet_address->byte_offset, vgpr_value,
                                 packet_address->base, LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 3;
  if (asm_form->operand_index_count == 4) {
    loom_value_id_t m0_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_m0_const_u32(
        builder, descriptor_set, descriptor,
        /*consumer_asm_operand_index=*/3, 0, location, &m0_value));
    operands[operand_count++] = m0_value;
  }
  loom_op_t* store_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &store_op);
}

static iree_status_t loom_amdgpu_feedback_build_global_store_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, loom_value_id_t value, loom_location_id_t location) {
  return loom_amdgpu_feedback_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR, packet_address,
      byte_offset, value, /*value_unit_count=*/1,
      /*extra_attrs=*/NULL, /*extra_attr_count=*/0, location);
}

static iree_status_t loom_amdgpu_feedback_build_global_store_b64(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    uint32_t byte_offset, loom_value_id_t value, loom_location_id_t location) {
  return loom_amdgpu_feedback_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR, packet_address,
      byte_offset, value, /*value_unit_count=*/2,
      /*extra_attrs=*/NULL, /*extra_attr_count=*/0, location);
}

static iree_status_t loom_amdgpu_feedback_build_publish_state_store(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    loom_value_id_t ready_value, loom_location_id_t location) {
  loom_named_attr_t extra_attrs[2] = {0};
  iree_host_size_t extra_attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_release_store_attrs(
      builder, descriptor_set, extra_attrs, IREE_ARRAYSIZE(extra_attrs),
      &extra_attr_count));
  return loom_amdgpu_feedback_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET, ready_value,
      /*value_unit_count=*/1, extra_attrs, extra_attr_count, location);
}

static iree_status_t loom_amdgpu_feedback_resolve_global_memory_descriptor(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    iree_string_view_t operation_name,
    const loom_low_descriptor_t** out_descriptor,
    loom_string_id_t* out_opcode_id, const loom_low_asm_form_t** out_asm_form) {
  *out_descriptor = NULL;
  *out_opcode_id = LOOM_STRING_ID_INVALID;
  *out_asm_form = NULL;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));
  if (descriptor->canonical_asm_form_ordinal >=
      descriptor_set->asm_form_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU feedback %.*s descriptor has no canonical "
                            "asm form",
                            (int)operation_name.size, operation_name.data);
  }
  const loom_low_asm_form_t* asm_form =
      &descriptor_set->asm_forms[descriptor->canonical_asm_form_ordinal];
  if (asm_form->operand_index_count != 2 &&
      asm_form->operand_index_count != 3 &&
      asm_form->operand_index_count != 4) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU feedback %.*s descriptor has an unsupported packet operand "
        "count",
        (int)operation_name.size, operation_name.data);
  }
  *out_descriptor = descriptor;
  *out_opcode_id = opcode_id;
  *out_asm_form = asm_form;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_global_load_b64_system(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, uint32_t byte_offset,
    loom_amdgpu_feedback_global_load_flags_t flags, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, channel_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2));

  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));

  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  const loom_low_asm_form_t* asm_form = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_resolve_global_memory_descriptor(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR,
      IREE_SV("load"), &descriptor, &opcode_id, &asm_form));
  if (asm_form->operand_index_count != 2 &&
      asm_form->operand_index_count != 3) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU feedback load descriptor has an unsupported packet operand "
        "count");
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &result_type));
  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_offset_attr(
      builder, byte_offset, &attrs[attr_count++]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_load_attrs(
      builder, descriptor_set, attrs, IREE_ARRAYSIZE(attrs), &attr_count));

  loom_value_id_t operands[3] = {zero_vaddr, channel_base,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 2;
  if (asm_form->operand_index_count == 3) {
    loom_value_id_t m0_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_m0_const_u32(
        builder, descriptor_set, descriptor,
        /*consumer_asm_operand_index=*/2, 0, location, &m0_value));
    operands[operand_count++] = m0_value;
  }
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  const loom_value_id_t value =
      loom_value_slice_get(loom_low_op_results(op), 0);
  if (iree_any_bit_set(flags, LOOM_AMDGPU_FEEDBACK_GLOBAL_LOAD_FLAG_ACQUIRE)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_acquire_ordering(
        builder, descriptor_set, location));
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_feedback_build_scalar_load(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t base_address,
    uint32_t byte_offset, loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));

  loom_named_attr_t offset_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_offset_attr(
      builder, byte_offset, &offset_attr));
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, &base_address,
      /*operand_count=*/1, loom_make_named_attr_slice(&offset_attr, 1),
      &result_type, /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &load_op));
  *out_value = loom_value_slice_get(loom_low_op_results(load_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_config_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t config_symbol, loom_location_id_t location,
    loom_amdgpu_feedback_config_values_t* out_values) {
  IREE_ASSERT_ARGUMENT(out_values);
  *out_values = loom_amdgpu_feedback_config_values_empty();

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));

  loom_amdgpu_feedback_config_values_t values =
      loom_amdgpu_feedback_config_values_empty();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_build_data_symbol_address(builder, descriptor_set,
                                            (loom_amdgpu_data_symbol_address_t){
                                                .symbol = config_symbol,
                                                .byte_offset = 0,
                                            },
                                            location, &values.address));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CONFIG_FLAGS_OFFSET, sgpr_type, location,
      &values.flags));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CONFIG_CHANNEL_BASE_OFFSET, sgpr_x2_type, location,
      &values.channel_base));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CONFIG_NOTIFY_SIGNAL_OFFSET, sgpr_x2_type, location,
      &values.notify_signal));

  *out_values = values;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_uniform_packet_address(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t packet_base, loom_location_id_t location,
    loom_amdgpu_feedback_packet_address_t* out_address) {
  IREE_ASSERT_ARGUMENT(out_address);
  *out_address = loom_amdgpu_feedback_packet_address_empty();

  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, packet_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2));

  loom_amdgpu_feedback_packet_address_t address =
      loom_amdgpu_feedback_packet_address_empty();
  address.base = packet_base;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &address.byte_offset));
  *out_address = address;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_dropped_packet_count_increment(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, channel_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2));

  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));
  loom_value_id_t one64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u64_const(
      builder, descriptor_set, 1, location, &one64));

  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR, &descriptor,
      &opcode_id));
  if (descriptor->canonical_asm_form_ordinal >=
      descriptor_set->asm_form_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU feedback atomic descriptor has no "
                            "canonical asm form");
  }
  const loom_low_asm_form_t* asm_form =
      &descriptor_set->asm_forms[descriptor->canonical_asm_form_ordinal];
  if (asm_form->operand_index_count != 3 &&
      asm_form->operand_index_count != 4) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU feedback atomic descriptor has an unsupported packet operand "
        "count");
  }

  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_offset_attr(
      builder, LOOM_AMDGPU_FEEDBACK_CHANNEL_DROPPED_PACKET_COUNT_OFFSET,
      &attrs[attr_count++]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_no_return_atomic_attrs(
      builder, descriptor_set, attrs, IREE_ARRAYSIZE(attrs), &attr_count));

  loom_value_id_t operands[4] = {zero_vaddr, one64, channel_base,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 3;
  if (asm_form->operand_index_count == 4) {
    loom_value_id_t m0_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_m0_const_u32(
        builder, descriptor_set, descriptor,
        /*consumer_asm_operand_index=*/3, 0, location, &m0_value));
    operands[operand_count++] = m0_value;
  }
  loom_op_t* op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op);
}

iree_status_t loom_amdgpu_build_feedback_reservation_head_load(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location,
    loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  return loom_amdgpu_feedback_build_global_load_b64_system(
      builder, descriptor_set, channel_base,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET,
      LOOM_AMDGPU_FEEDBACK_GLOBAL_LOAD_FLAG_NONE, location, out_value);
}

iree_status_t loom_amdgpu_build_feedback_read_tail_acquire_load(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location,
    loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  return loom_amdgpu_feedback_build_global_load_b64_system(
      builder, descriptor_set, channel_base,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET,
      LOOM_AMDGPU_FEEDBACK_GLOBAL_LOAD_FLAG_ACQUIRE, location, out_value);
}

iree_status_t
loom_amdgpu_build_feedback_reservation_head_compare_exchange_acq_rel(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_value_id_t expected_head,
    loom_value_id_t desired_head, loom_location_id_t location,
    loom_value_id_t* out_old_head) {
  IREE_ASSERT_ARGUMENT(out_old_head);
  *out_old_head = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, channel_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2));

  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));
  loom_value_id_t expected_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_materialize_vgpr_registers(
      builder, descriptor_set, expected_head, /*expected_unit_count=*/2,
      location, &expected_vgpr));
  loom_value_id_t desired_vgpr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_materialize_vgpr_registers(
      builder, descriptor_set, desired_head, /*expected_unit_count=*/2,
      location, &desired_vgpr));
  loom_value_id_t compare_exchange_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_pair(
      builder, descriptor_set, expected_vgpr, desired_vgpr, location,
      &compare_exchange_pair));

  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  const loom_low_asm_form_t* asm_form = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_resolve_global_memory_descriptor(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR,
      IREE_SV("compare-exchange"), &descriptor, &opcode_id, &asm_form));
  if (asm_form->operand_index_count != 3 &&
      asm_form->operand_index_count != 4) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU feedback compare-exchange descriptor has "
                            "an unsupported packet operand count");
  }

  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_offset_attr(
      builder, LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET,
      &attrs[attr_count++]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_append_return_atomic_attrs(
      builder, descriptor_set, attrs, IREE_ARRAYSIZE(attrs), &attr_count));

  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_release_ordering(
      builder, descriptor_set, location));

  loom_value_id_t operands[4] = {zero_vaddr, compare_exchange_pair,
                                 channel_base, LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 3;
  if (asm_form->operand_index_count == 4) {
    loom_value_id_t m0_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_m0_const_u32(
        builder, descriptor_set, descriptor,
        /*consumer_asm_operand_index=*/3, 0, location, &m0_value));
    operands[operand_count++] = m0_value;
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &result_type));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  const loom_value_id_t old_head =
      loom_value_slice_get(loom_low_op_results(op), 0);
  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_acquire_ordering(
      builder, descriptor_set, location));
  *out_old_head = old_head;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_reservation_attempt(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_value_id_t reservation_head,
    uint32_t packet_length, loom_location_id_t location,
    loom_amdgpu_feedback_reservation_attempt_t* out_attempt) {
  IREE_ASSERT_ARGUMENT(out_attempt);
  *out_attempt = loom_amdgpu_feedback_reservation_attempt_empty();
  if (!loom_amdgpu_feedback_packet_record_length_is_valid(packet_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU feedback reservation packet length violates the feedback ABI");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, channel_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, reservation_head, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
      2));

  loom_value_id_t packet_length_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u64_const(
      builder, descriptor_set, packet_length, location, &packet_length_value));
  loom_value_id_t next_head = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr64_add(
      builder, descriptor_set, reservation_head, packet_length_value, location,
      &next_head));

  loom_value_id_t observed_head = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_build_feedback_reservation_head_compare_exchange_acq_rel(
          builder, descriptor_set, channel_base, reservation_head, next_head,
          location, &observed_head));

  loom_value_id_t cas_succeeded = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr64_equal_mask(
      builder, descriptor_set, observed_head, reservation_head, location,
      &cas_succeeded));

  *out_attempt = (loom_amdgpu_feedback_reservation_attempt_t){
      .expected_head = reservation_head,
      .next_head = next_head,
      .observed_head = observed_head,
      .cas_succeeded = cas_succeeded,
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_feedback_packet_header(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    const loom_amdgpu_feedback_packet_header_t* header,
    loom_location_id_t location) {
  if (!loom_amdgpu_feedback_packet_record_length_is_valid(
          header->record_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU feedback packet header record length "
                            "violates the feedback ABI");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, packet_address->base,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, packet_address->byte_offset,
      LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1));

  loom_value_id_t zero32 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero32));
  loom_value_id_t zero64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u64_const(
      builder, descriptor_set, 0, location, &zero64));

  loom_value_id_t record_length = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, header->record_length, location,
      &record_length));
  const uint32_t packed_header_kind =
      LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH | ((uint32_t)header->kind << 16);
  loom_value_id_t header_kind = LOOM_VALUE_ID_INVALID;
  // The 16-bit header length and 16-bit kind occupy one little-endian dword.
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, packed_header_kind, location, &header_kind));
  loom_value_id_t packet_flags = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, header->flags, location, &packet_flags));

  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_RECORD_LENGTH_OFFSET, record_length,
      location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_HEADER_LENGTH_OFFSET, header_kind, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_FLAGS_OFFSET, packet_flags, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET, zero32, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_SEQUENCE_OFFSET, header->sequence, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_DISPATCH_PTR_OFFSET,
      header->source_dispatch_ptr, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKGROUP_ID_X_OFFSET,
      header->source_workgroup_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKITEM_ID_X_OFFSET,
      header->source_workitem_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED0_OFFSET, zero32, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED1_OFFSET, zero32, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_0_OFFSET, zero64, location));
  return loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, packet_address,
      LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_1_OFFSET, zero64, location);
}

iree_status_t loom_amdgpu_build_feedback_publish_packet_state(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, packet_address->base,
      LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, packet_address->byte_offset,
      LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1));

  IREE_RETURN_IF_ERROR(loom_amdgpu_system_memory_build_release_ordering(
      builder, descriptor_set, location));

  loom_value_id_t ready_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, LOOM_AMDGPU_FEEDBACK_PACKET_STATE_READY,
      location, &ready_value));
  return loom_amdgpu_feedback_build_publish_state_store(
      builder, descriptor_set, packet_address, ready_value, location);
}

iree_status_t loom_amdgpu_build_feedback_notify_host(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t notify_signal, loom_location_id_t location) {
  loom_amdgpu_signal_values_t signal_values = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_signal_values(
      builder, descriptor_set, notify_signal, location, &signal_values));
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_signal_add_one_release(
      builder, descriptor_set, signal_values.address, location));
  return loom_amdgpu_build_signal_poke_mailbox(
      builder, descriptor_set, signal_values.event_mailbox_ptr,
      signal_values.event_id, location);
}

iree_status_t loom_amdgpu_build_feedback_publish_packet(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_feedback_packet_address_t* packet_address,
    loom_value_id_t notify_signal, loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_feedback_publish_packet_state(
      builder, descriptor_set, packet_address, location));
  return loom_amdgpu_build_feedback_notify_host(builder, descriptor_set,
                                                notify_signal, location);
}

iree_status_t loom_amdgpu_build_feedback_channel_header_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t channel_base, loom_location_id_t location,
    loom_amdgpu_feedback_channel_header_values_t* out_values) {
  IREE_ASSERT_ARGUMENT(out_values);
  *out_values = loom_amdgpu_feedback_channel_header_values_empty();

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));

  loom_amdgpu_feedback_channel_header_values_t values =
      loom_amdgpu_feedback_channel_header_values_empty();
  values.address = channel_base;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RECORD_LENGTH_OFFSET, sgpr_type, location,
      &values.record_length));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_ABI_VERSION_OFFSET, sgpr_type, location,
      &values.abi_version));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_FLAGS_OFFSET, sgpr_type, location,
      &values.flags));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RING_BASE_OFFSET, sgpr_x2_type, location,
      &values.ring_base));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_FEEDBACK_CHANNEL_RING_CAPACITY_OFFSET, sgpr_x2_type, location,
      &values.ring_capacity));

  *out_values = values;
  return iree_ok_status();
}
