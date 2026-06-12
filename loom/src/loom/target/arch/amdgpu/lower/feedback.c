// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/feedback.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/cache.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/feedback_abi.h"
#include "loom/target/arch/amdgpu/lower/data_symbol.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
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

static iree_status_t loom_amdgpu_feedback_build_explicit_packet(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_named_attr_slice_t attrs,
    loom_location_id_t location) {
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, descriptor_ref, &descriptor, &opcode_id));
  loom_op_t* op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, /*operands=*/NULL,
      /*operand_count=*/0, attrs, /*result_types=*/NULL, /*result_count=*/0,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, &op);
}

static loom_amdgpu_vector_memory_cache_policy_encoding_t
loom_amdgpu_feedback_vector_cache_encoding(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  return descriptor_set_info == NULL
             ? LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE
             : descriptor_set_info->vector_memory_cache_policy_encoding;
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
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t zero_vaddr,
    loom_value_id_t packet_base, uint32_t byte_offset, loom_value_id_t value,
    uint32_t value_unit_count, const loom_named_attr_t* extra_attrs,
    iree_host_size_t extra_attr_count, loom_location_id_t location) {
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
  loom_value_id_t operands[4] = {zero_vaddr, vgpr_value, packet_base,
                                 LOOM_VALUE_ID_INVALID};
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
    loom_value_id_t zero_vaddr, loom_value_id_t packet_base,
    uint32_t byte_offset, loom_value_id_t value, loom_location_id_t location) {
  return loom_amdgpu_feedback_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR, zero_vaddr,
      packet_base, byte_offset, value, /*value_unit_count=*/1,
      /*extra_attrs=*/NULL, /*extra_attr_count=*/0, location);
}

static iree_status_t loom_amdgpu_feedback_build_global_store_b64(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t zero_vaddr, loom_value_id_t packet_base,
    uint32_t byte_offset, loom_value_id_t value, loom_location_id_t location) {
  return loom_amdgpu_feedback_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR, zero_vaddr,
      packet_base, byte_offset, value, /*value_unit_count=*/2,
      /*extra_attrs=*/NULL, /*extra_attr_count=*/0, location);
}

static iree_status_t loom_amdgpu_feedback_build_publish_state_store(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t zero_vaddr, loom_value_id_t packet_base,
    loom_value_id_t ready_value,
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding,
    loom_location_id_t location) {
  loom_named_attr_t extra_attrs[2] = {0};
  iree_host_size_t extra_attr_count = 0;
  switch (encoding) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_u32_attr(
          builder, IREE_SV("sc0"), 1, &extra_attrs[extra_attr_count++]));
      IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_u32_attr(
          builder, IREE_SV("sc1"), 1, &extra_attrs[extra_attr_count++]));
      break;
    }
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_u32_attr(
          builder, IREE_SV("scope"), LOOM_CACHE_SCOPE_SYSTEM,
          &extra_attrs[extra_attr_count++]));
      break;
    }
    default:
      break;
  }
  return loom_amdgpu_feedback_build_global_store(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR, zero_vaddr,
      packet_base, LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET, ready_value,
      /*value_unit_count=*/1, extra_attrs, extra_attr_count, location);
}

static iree_status_t loom_amdgpu_feedback_build_release_ordering(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding,
    loom_location_id_t location) {
  switch (encoding) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC: {
      loom_named_attr_t waitcnt_attrs[2] = {0};
      IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_u32_attr(
          builder, IREE_SV("vmcnt"), 0, &waitcnt_attrs[0]));
      IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_u32_attr(
          builder, IREE_SV("lgkmcnt"), 15, &waitcnt_attrs[1]));
      IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_explicit_packet(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT,
          loom_make_named_attr_slice(waitcnt_attrs,
                                     IREE_ARRAYSIZE(waitcnt_attrs)),
          location));

      loom_named_attr_t vscnt_attr = {0};
      IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_u32_attr(
          builder, IREE_SV("vscnt"), 0, &vscnt_attr));
      return loom_amdgpu_feedback_build_explicit_packet(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT,
          loom_make_named_attr_slice(&vscnt_attr, 1), location);
    }
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return loom_amdgpu_feedback_build_explicit_packet(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2,
          loom_make_named_attr_slice(NULL, 0), location);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH: {
      loom_named_attr_t scope_attr = {0};
      IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_u32_attr(
          builder, IREE_SV("scope"), LOOM_CACHE_SCOPE_SYSTEM, &scope_attr));
      return loom_amdgpu_feedback_build_explicit_packet(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB,
          loom_make_named_attr_slice(&scope_attr, 1), location);
    }
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU feedback release publication is not "
                              "supported by the descriptor set");
  }
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

iree_status_t loom_amdgpu_build_feedback_packet_header(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t packet_base,
    const loom_amdgpu_feedback_packet_header_t* header,
    loom_location_id_t location) {
  if (header->record_length < LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH ||
      (header->record_length & (LOOM_AMDGPU_FEEDBACK_PACKET_ALIGNMENT - 1u)) !=
          0 ||
      header->record_length >
          loom_amdgpu_feedback_packet_length(
              LOOM_AMDGPU_FEEDBACK_PACKET_MAX_PAYLOAD_LENGTH)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU feedback packet header record length "
                            "violates the feedback ABI");
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, packet_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2));

  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));
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
      builder, descriptor_set, zero_vaddr, packet_base,
      LOOM_AMDGPU_FEEDBACK_PACKET_RECORD_LENGTH_OFFSET, record_length,
      location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, zero_vaddr, packet_base,
      LOOM_AMDGPU_FEEDBACK_PACKET_HEADER_LENGTH_OFFSET, header_kind, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, zero_vaddr, packet_base,
      LOOM_AMDGPU_FEEDBACK_PACKET_FLAGS_OFFSET, packet_flags, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, zero_vaddr, packet_base,
      LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET, zero_vaddr, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, zero_vaddr, packet_base,
      LOOM_AMDGPU_FEEDBACK_PACKET_SEQUENCE_OFFSET, header->sequence, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, zero_vaddr, packet_base,
      LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_DISPATCH_PTR_OFFSET,
      header->source_dispatch_ptr, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, zero_vaddr, packet_base,
      LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKGROUP_ID_X_OFFSET,
      header->source_workgroup_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, zero_vaddr, packet_base,
      LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKITEM_ID_X_OFFSET,
      header->source_workitem_id_x, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, zero_vaddr, packet_base,
      LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED0_OFFSET, zero_vaddr, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b32(
      builder, descriptor_set, zero_vaddr, packet_base,
      LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED1_OFFSET, zero_vaddr, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, zero_vaddr, packet_base,
      LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_0_OFFSET, zero64, location));
  return loom_amdgpu_feedback_build_global_store_b64(
      builder, descriptor_set, zero_vaddr, packet_base,
      LOOM_AMDGPU_FEEDBACK_PACKET_RESERVED_ARRAY_1_OFFSET, zero64, location);
}

iree_status_t loom_amdgpu_build_feedback_publish_packet_state(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t packet_base, loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_require_register_class(
      builder, descriptor_set, packet_base, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2));

  const loom_amdgpu_vector_memory_cache_policy_encoding_t encoding =
      loom_amdgpu_feedback_vector_cache_encoding(descriptor_set);
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_release_ordering(
      builder, descriptor_set, encoding, location));

  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));
  loom_value_id_t ready_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_feedback_build_vgpr_u32_const(
      builder, descriptor_set, LOOM_AMDGPU_FEEDBACK_PACKET_STATE_READY,
      location, &ready_value));
  return loom_amdgpu_feedback_build_publish_state_store(
      builder, descriptor_set, zero_vaddr, packet_base, ready_value, encoding,
      location);
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
