// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/signal.h"

#include <stdint.h>

#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/cache.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/lower/control_packet.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/registers.h"

static loom_amdgpu_signal_values_t loom_amdgpu_signal_values_empty(void) {
  return (loom_amdgpu_signal_values_t){
      .address = LOOM_VALUE_ID_INVALID,
      .event_mailbox_ptr = LOOM_VALUE_ID_INVALID,
      .event_id = LOOM_VALUE_ID_INVALID,
  };
}

static iree_status_t loom_amdgpu_signal_build_u32_attr(
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

static iree_status_t loom_amdgpu_signal_build_offset_attr(
    loom_builder_t* builder, uint32_t byte_offset,
    loom_named_attr_t* out_attr) {
  return loom_amdgpu_signal_build_u32_attr(builder, IREE_SV("offset"),
                                           byte_offset, out_attr);
}

static iree_status_t loom_amdgpu_signal_build_explicit_packet(
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
loom_amdgpu_signal_vector_cache_encoding(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  return descriptor_set_info == NULL
             ? LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE
             : descriptor_set_info->vector_memory_cache_policy_encoding;
}

static bool loom_amdgpu_signal_type_is_register_class(
    const loom_low_descriptor_set_t* descriptor_set, loom_type_t type,
    uint16_t reg_class_id) {
  return loom_low_type_is_register(type) &&
         loom_low_register_type_descriptor_set_stable_id(type) ==
             descriptor_set->stable_id &&
         loom_low_register_type_class_id(type) == reg_class_id;
}

static iree_status_t loom_amdgpu_signal_require_register_class(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t value, uint16_t reg_class_id, uint32_t unit_count) {
  if (value >= builder->module->values.count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU signal builder received an invalid low value");
  }
  const loom_type_t type = loom_module_value_type(builder->module, value);
  if (!loom_amdgpu_signal_type_is_register_class(descriptor_set, type,
                                                 reg_class_id) ||
      loom_low_register_type_unit_count(type) != unit_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU signal builder received a low value with an unsupported "
        "register shape");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_descriptor_operand_type(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t descriptor_operand_index,
    loom_type_t* out_type) {
  *out_type = loom_type_none();
  if (descriptor_operand_index >= descriptor->operand_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU signal descriptor operand index is out of range");
  }
  const uint32_t operand_index =
      (uint32_t)descriptor->operand_start + descriptor_operand_index;
  if (operand_index >= descriptor_set->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU signal descriptor operand row is out of "
                            "range");
  }
  const loom_low_operand_t* operand = &descriptor_set->operands[operand_index];
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const uint32_t alt_index = operand->reg_class_alt_start + i;
    if (alt_index >= descriptor_set->reg_class_alt_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU signal descriptor operand register-class alternative is out "
          "of range");
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
      "AMDGPU signal descriptor operand has no register alternative");
}

static iree_status_t loom_amdgpu_signal_asm_form(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    const loom_low_asm_form_t** out_asm_form) {
  *out_asm_form = NULL;
  if (descriptor->canonical_asm_form_ordinal >=
      descriptor_set->asm_form_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU signal descriptor has no canonical asm form");
  }
  *out_asm_form =
      &descriptor_set->asm_forms[descriptor->canonical_asm_form_ordinal];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_asm_operand_type(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t asm_operand_index,
    loom_type_t* out_type) {
  *out_type = loom_type_none();
  const loom_low_asm_form_t* asm_form = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_signal_asm_form(descriptor_set, descriptor, &asm_form));
  if (asm_operand_index >= asm_form->operand_index_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU signal asm operand index is out of range");
  }
  const uint32_t operand_index =
      (uint32_t)asm_form->operand_index_start + asm_operand_index;
  if (operand_index >= descriptor_set->asm_operand_index_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU signal asm operand row is out of range");
  }
  return loom_amdgpu_signal_descriptor_operand_type(
      descriptor_set, descriptor,
      descriptor_set->asm_operand_indices[operand_index], out_type);
}

static iree_status_t loom_amdgpu_signal_descriptor_result_type(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t result_index,
    loom_type_t* out_type) {
  if (result_index >= descriptor->result_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU signal descriptor result index is out of range");
  }
  return loom_amdgpu_signal_descriptor_operand_type(descriptor_set, descriptor,
                                                    result_index, out_type);
}

static iree_status_t loom_amdgpu_signal_build_const_u32(
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_u32_attr(
      builder, IREE_SV("imm32"), value, &imm32_attr));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      builder, descriptor_set, descriptor, opcode_id,
      loom_make_named_attr_slice(&imm32_attr, 1), result_type, location,
      &const_op));
  *out_value = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_build_sgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  return loom_amdgpu_signal_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, value,
      sgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_signal_build_vgpr_u32_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint32_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, &vgpr_type));
  return loom_amdgpu_signal_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, value,
      vgpr_type, location, out_value);
}

static iree_status_t loom_amdgpu_signal_build_m0_const_u32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* consumer_descriptor,
    uint16_t consumer_asm_operand_index, uint32_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_asm_operand_type(
      descriptor_set, consumer_descriptor, consumer_asm_operand_index,
      &m0_type));
  return loom_amdgpu_signal_build_const_u32(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM,
      value, m0_type, location, out_value);
}

static iree_status_t loom_amdgpu_signal_build_vgpr_b32_copy(
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
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &copy_op));
  *out_value = loom_value_slice_get(loom_low_op_results(copy_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_materialize_vgpr_b32(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = source;
  if (source >= builder->module->values.count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU signal builder received an invalid low value");
  }
  const loom_type_t source_type =
      loom_module_value_type(builder->module, source);
  if (!loom_low_type_is_register(source_type) ||
      loom_low_register_type_unit_count(source_type) != 1) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU signal builder cannot materialize value with unsupported "
        "register shape");
  }
  if (loom_amdgpu_signal_type_is_register_class(
          descriptor_set, source_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_signal_type_is_register_class(
          descriptor_set, source_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU signal builder cannot materialize non-SGPR value into VGPR");
  }
  return loom_amdgpu_signal_build_vgpr_b32_copy(builder, descriptor_set, source,
                                                location, out_value);
}

static iree_status_t loom_amdgpu_signal_build_vgpr_u64_zero_extend(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_materialize_vgpr_b32(
      builder, descriptor_set, source, location, &low_source));
  loom_value_id_t low_zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &low_zero));

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));
  loom_value_id_t sources[] = {low_source, low_zero};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, sources, IREE_ARRAYSIZE(sources),
                            vgpr_x2_type, location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_build_vgpr_u64_const(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    uint64_t value, loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u32_const(
      builder, descriptor_set, (uint32_t)value, location, &low_value_lo));
  loom_value_id_t low_value_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u32_const(
      builder, descriptor_set, (uint32_t)(value >> 32), location,
      &low_value_hi));

  loom_type_t vgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2, &vgpr_x2_type));
  loom_value_id_t sources[] = {low_value_lo, low_value_hi};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(builder, sources, IREE_ARRAYSIZE(sources),
                            vgpr_x2_type, location, &concat_op));
  *out_value = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_packet_operand_count(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t* out_operand_count) {
  const loom_low_asm_form_t* asm_form = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_signal_asm_form(descriptor_set, descriptor, &asm_form));
  *out_operand_count = asm_form->operand_index_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_append_optional_m0_operand(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, loom_location_id_t location,
    loom_value_id_t* operands, iree_host_size_t operand_capacity,
    iree_host_size_t* inout_operand_count) {
  uint16_t packet_operand_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_packet_operand_count(
      descriptor_set, descriptor, &packet_operand_count));
  if (packet_operand_count == *inout_operand_count) {
    return iree_ok_status();
  }
  if (packet_operand_count != *inout_operand_count + 1 ||
      *inout_operand_count >= operand_capacity) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU signal descriptor has an unsupported packet operand count");
  }
  loom_value_id_t m0_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_m0_const_u32(
      builder, descriptor_set, descriptor, (uint16_t)*inout_operand_count, 0,
      location, &m0_value));
  operands[*inout_operand_count] = m0_value;
  *inout_operand_count += 1;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_build_release_ordering(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding,
    loom_location_id_t location) {
  switch (encoding) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC: {
      loom_named_attr_t waitcnt_attrs[2] = {0};
      IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_u32_attr(
          builder, IREE_SV("vmcnt"), 0, &waitcnt_attrs[0]));
      IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_u32_attr(
          builder, IREE_SV("lgkmcnt"), 15, &waitcnt_attrs[1]));
      IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_explicit_packet(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT,
          loom_make_named_attr_slice(waitcnt_attrs,
                                     IREE_ARRAYSIZE(waitcnt_attrs)),
          location));

      loom_named_attr_t vscnt_attr = {0};
      IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_u32_attr(
          builder, IREE_SV("vscnt"), 0, &vscnt_attr));
      return loom_amdgpu_signal_build_explicit_packet(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT,
          loom_make_named_attr_slice(&vscnt_attr, 1), location);
    }
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return loom_amdgpu_signal_build_explicit_packet(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_WBL2,
          loom_make_named_attr_slice(NULL, 0), location);
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH: {
      loom_named_attr_t scope_attr = {0};
      IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_u32_attr(
          builder, IREE_SV("scope"), LOOM_CACHE_SCOPE_SYSTEM, &scope_attr));
      return loom_amdgpu_signal_build_explicit_packet(
          builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_WB,
          loom_make_named_attr_slice(&scope_attr, 1), location);
    }
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU signal release operation is not supported by the descriptor "
          "set");
  }
}

static iree_status_t loom_amdgpu_signal_build_scalar_load(
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
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_signal_build_offset_attr(builder, byte_offset, &offset_attr));
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, &base_address,
      /*operand_count=*/1, loom_make_named_attr_slice(&offset_attr, 1),
      &result_type, /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &load_op));
  *out_value = loom_value_slice_get(loom_low_op_results(load_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_build_release_store_attrs(
    loom_builder_t* builder,
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* inout_attr_count) {
  switch (encoding) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      if (*inout_attr_count + 2 > attr_capacity) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "AMDGPU signal store attr capacity exceeded");
      }
      loom_named_attr_t sc0_attr = {0};
      IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_u32_attr(
          builder, IREE_SV("sc0"), 1, &sc0_attr));
      attrs[(*inout_attr_count)++] = sc0_attr;
      loom_named_attr_t sc1_attr = {0};
      IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_u32_attr(
          builder, IREE_SV("sc1"), 1, &sc1_attr));
      attrs[(*inout_attr_count)++] = sc1_attr;
      return iree_ok_status();
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      if (*inout_attr_count + 1 > attr_capacity) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "AMDGPU signal store attr capacity exceeded");
      }
      loom_named_attr_t scope_attr = {0};
      IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_u32_attr(
          builder, IREE_SV("scope"), LOOM_CACHE_SCOPE_SYSTEM, &scope_attr));
      attrs[(*inout_attr_count)++] = scope_attr;
      return iree_ok_status();
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_signal_build_global_store_b64(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t zero_vaddr, loom_value_id_t saddr, loom_value_id_t value,
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding,
    loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_require_register_class(
      builder, descriptor_set, zero_vaddr, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1));
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_require_register_class(
      builder, descriptor_set, value, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_require_register_class(
      builder, descriptor_set, saddr, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2));

  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR, &descriptor,
      &opcode_id));

  loom_named_attr_t attrs[3] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_signal_build_offset_attr(builder, 0, &attrs[attr_count++]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_release_store_attrs(
      builder, encoding, attrs, IREE_ARRAYSIZE(attrs), &attr_count));

  loom_value_id_t operands[4] = {zero_vaddr, value, saddr,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 3;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_append_optional_m0_operand(
      builder, descriptor_set, descriptor, location, operands,
      IREE_ARRAYSIZE(operands), &operand_count));
  loom_op_t* store_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &store_op);
}

static iree_status_t loom_amdgpu_signal_build_atomic_attrs(
    loom_builder_t* builder,
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding,
    loom_named_attr_t* attrs, iree_host_size_t attr_capacity,
    iree_host_size_t* out_attr_count) {
  *out_attr_count = 0;
  loom_named_attr_t offset_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_offset_attr(
      builder, LOOM_AMDGPU_SIGNAL_VALUE_OFFSET, &offset_attr));
  attrs[(*out_attr_count)++] = offset_attr;
  if (encoding !=
      LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH) {
    return iree_ok_status();
  }
  if (*out_attr_count >= attr_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU signal atomic attr capacity exceeded");
  }
  loom_named_attr_t scope_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_u32_attr(
      builder, IREE_SV("scope"), LOOM_CACHE_SCOPE_SYSTEM, &scope_attr));
  attrs[(*out_attr_count)++] = scope_attr;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_signal_build_m0_from_sgpr(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t source, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_require_register_class(
      builder, descriptor_set, source, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1));
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0,
      &descriptor, &opcode_id));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_descriptor_result_type(
      descriptor_set, descriptor, /*result_index=*/0, &result_type));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, &source,
      /*operand_count=*/1, loom_make_named_attr_slice(NULL, 0), &result_type,
      /*result_count=*/1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op));
  *out_value = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_signal_values(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t signal_address, loom_location_id_t location,
    loom_amdgpu_signal_values_t* out_values) {
  IREE_ASSERT_ARGUMENT(out_values);
  *out_values = loom_amdgpu_signal_values_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_require_register_class(
      builder, descriptor_set, signal_address, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      2));

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_type_t sgpr_x2_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, &sgpr_x2_type));

  loom_amdgpu_signal_values_t values = loom_amdgpu_signal_values_empty();
  values.address = signal_address;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORDX2_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_SIGNAL_EVENT_MAILBOX_PTR_OFFSET, sgpr_x2_type, location,
      &values.event_mailbox_ptr));
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_scalar_load(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY, values.address,
      LOOM_AMDGPU_SIGNAL_EVENT_ID_OFFSET, sgpr_type, location,
      &values.event_id));
  *out_values = values;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_signal_add_one_release(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t signal_address, loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_require_register_class(
      builder, descriptor_set, signal_address, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      2));

  const loom_amdgpu_vector_memory_cache_policy_encoding_t encoding =
      loom_amdgpu_signal_vector_cache_encoding(descriptor_set);
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_release_ordering(
      builder, descriptor_set, encoding, location));

  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));
  loom_value_id_t one64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u64_const(
      builder, descriptor_set, 1, location, &one64));

  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_SADDR, &descriptor,
      &opcode_id));
  loom_named_attr_t attrs[2] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_atomic_attrs(
      builder, encoding, attrs, IREE_ARRAYSIZE(attrs), &attr_count));
  loom_value_id_t operands[4] = {zero_vaddr, one64, signal_address,
                                 LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 3;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_append_optional_m0_operand(
      builder, descriptor_set, descriptor, location, operands,
      IREE_ARRAYSIZE(operands), &operand_count));
  loom_op_t* op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &op);
}

iree_status_t loom_amdgpu_build_signal_mailbox_message_id(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t event_id, loom_location_id_t location,
    loom_value_id_t* out_message_id) {
  IREE_ASSERT_ARGUMENT(out_message_id);
  *out_message_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_require_register_class(
      builder, descriptor_set, event_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1));

  switch (descriptor_set->descriptor_set_ordinal) {
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA3:
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA4:
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3:
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3_5:
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4:
    case LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU signal mailbox message-id mask is not known for the "
          "descriptor set");
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      descriptor_set, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, &sgpr_type));
  loom_value_id_t mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_sgpr_u32_const(
      builder, descriptor_set,
      LOOM_AMDGPU_SIGNAL_MAILBOX_MESSAGE_ID_GFX9_11_12_MASK, location, &mask));
  loom_value_id_t operands[] = {event_id, mask};
  const loom_low_descriptor_t* descriptor = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_descriptor_ref(
      builder, descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32,
      &descriptor, &opcode_id));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, descriptor_set, descriptor, opcode_id, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &sgpr_type,
      /*result_count=*/1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &op));
  *out_message_id = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_build_signal_poke_mailbox(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_value_id_t event_mailbox_ptr, loom_value_id_t event_id,
    loom_location_id_t location) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_require_register_class(
      builder, descriptor_set, event_mailbox_ptr, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      2));
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_require_register_class(
      builder, descriptor_set, event_id, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1));

  const loom_amdgpu_vector_memory_cache_policy_encoding_t encoding =
      loom_amdgpu_signal_vector_cache_encoding(descriptor_set);
  loom_value_id_t zero_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u32_const(
      builder, descriptor_set, 0, location, &zero_vaddr));
  loom_value_id_t event_id64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_vgpr_u64_zero_extend(
      builder, descriptor_set, event_id, location, &event_id64));
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_release_ordering(
      builder, descriptor_set, encoding, location));
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_global_store_b64(
      builder, descriptor_set, zero_vaddr, event_mailbox_ptr, event_id64,
      encoding, location));

  loom_value_id_t message_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_signal_mailbox_message_id(
      builder, descriptor_set, event_id, location, &message_id));
  loom_value_id_t m0_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_signal_build_m0_from_sgpr(
      builder, descriptor_set, message_id, location, &m0_payload));
  return loom_amdgpu_build_control_packet_send_message_with_m0(
      builder, descriptor_set, LOOM_AMDGPU_SIGNAL_INTERRUPT_SENDMSG, m0_payload,
      location);
}
