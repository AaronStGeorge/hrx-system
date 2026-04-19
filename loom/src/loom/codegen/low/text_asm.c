// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/text_asm.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static iree_status_t loom_low_descriptor_text_asm_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, iree_string_view_t* out_string) {
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, string_offset, out_string));
  if (iree_string_view_is_empty(*out_string)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm descriptor table has an empty required "
                            "string");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_immediate_info(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    const loom_low_asm_immediate_t* asm_immediate,
    loom_text_low_asm_immediate_descriptor_t* out_immediate) {
  if (asm_immediate->immediate_index >= descriptor->immediate_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm immediate index is out of range");
  }
  const uint32_t immediate_index =
      descriptor->immediate_start + asm_immediate->immediate_index;
  if (immediate_index >= descriptor_set->immediate_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm immediate field is out of range");
  }
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_index];
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_string(
      descriptor_set, immediate->field_name_string_offset,
      &out_immediate->field_name));
  if (asm_immediate->name_string_offset != LOOM_LOW_STRING_OFFSET_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_string(
        descriptor_set, asm_immediate->name_string_offset,
        &out_immediate->spelling));
  } else {
    out_immediate->spelling = out_immediate->field_name;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_lookup_descriptor_set(
    void* user_data, iree_string_view_t key, const void** out_descriptor_set) {
  *out_descriptor_set = NULL;
  const loom_low_descriptor_registry_t* registry =
      (const loom_low_descriptor_registry_t*)user_data;
  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_registry_lookup(registry, key, &descriptor_set));
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "low descriptor set was not found");
  }
  *out_descriptor_set = descriptor_set;
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_lookup_packet(
    void* user_data, const void* descriptor_set_opaque,
    iree_string_view_t mnemonic,
    loom_text_low_asm_packet_descriptor_t* out_packet) {
  (void)user_data;
  *out_packet = (loom_text_low_asm_packet_descriptor_t){0};
  const loom_low_descriptor_set_t* descriptor_set =
      (const loom_low_descriptor_set_t*)descriptor_set_opaque;

  uint32_t asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_lookup_asm_form(
      descriptor_set, mnemonic, &asm_form_ordinal));
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_set_asm_form_at(descriptor_set, asm_form_ordinal);
  if (asm_form == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm form ordinal is out of range");
  }

  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set,
                                            asm_form->descriptor_ordinal);
  if (descriptor == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm form references an invalid descriptor");
  }

  iree_string_view_t opcode_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_string(
      descriptor_set, descriptor->key_string_offset, &opcode_key));

  if (asm_form->immediate_start > descriptor_set->asm_immediate_count ||
      asm_form->immediate_count >
          descriptor_set->asm_immediate_count - asm_form->immediate_start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm form immediate span is out of range");
  }
  bool has_named_immediates = false;
  for (uint16_t i = 0; i < asm_form->immediate_count; ++i) {
    const loom_low_asm_immediate_t* asm_immediate =
        &descriptor_set
             ->asm_immediates[asm_form->immediate_start + (uint32_t)i];
    if (asm_immediate->name_string_offset != LOOM_LOW_STRING_OFFSET_NONE) {
      has_named_immediates = true;
      break;
    }
  }

  *out_packet = (loom_text_low_asm_packet_descriptor_t){
      .descriptor_set = descriptor_set,
      .form = asm_form,
      .descriptor = descriptor,
      .opcode_key = opcode_key,
      .result_count = asm_form->result_operand_index_count,
      .operand_count = asm_form->operand_index_count,
      .immediate_count = asm_form->immediate_count,
      .has_named_immediates = has_named_immediates,
      .builds_as_const =
          asm_form->operand_index_count == 0 &&
          asm_form->result_operand_index_count == 1 &&
          asm_form->immediate_count > 0 &&
          !iree_all_bits_set(descriptor->flags,
                             LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING),
  };
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_infer_result_type(
    void* user_data, const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t result_index, loom_module_t* module, loom_type_t* out_type,
    iree_string_view_t* out_diagnostic_detail) {
  (void)user_data;
  *out_diagnostic_detail = iree_string_view_empty();
  if (result_index >= packet->result_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm result index is out of range");
  }

  const loom_low_descriptor_set_t* descriptor_set =
      (const loom_low_descriptor_set_t*)packet->descriptor_set;
  const loom_low_descriptor_t* descriptor =
      (const loom_low_descriptor_t*)packet->descriptor;
  const loom_low_asm_form_t* asm_form =
      (const loom_low_asm_form_t*)packet->form;

  const uint32_t asm_operand_index =
      asm_form->result_operand_index_start + result_index;
  if (asm_operand_index >= descriptor_set->asm_operand_index_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm result operand index is out of range");
  }
  const uint16_t descriptor_operand_index =
      descriptor_set->asm_operand_indices[asm_operand_index];
  if (descriptor_operand_index >= descriptor->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm result references an operand outside the "
                            "descriptor");
  }
  const loom_low_operand_t* operand =
      &descriptor_set
           ->operands[descriptor->operand_start + descriptor_operand_index];
  if (operand->reg_class_alt_count != 1) {
    *out_diagnostic_detail =
        IREE_SV("result type annotation is required for multi-class result");
    return iree_ok_status();
  }

  const uint32_t alternative_index = operand->reg_class_alt_start;
  if (alternative_index >= descriptor_set->reg_class_alt_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm result register-class alternative is out "
                            "of range");
  }
  const loom_low_reg_class_alt_t* alternative =
      &descriptor_set->reg_class_alts[alternative_index];
  if (alternative->reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      alternative->reg_class_id >= descriptor_set->reg_class_count) {
    *out_diagnostic_detail =
        IREE_SV("result type annotation is required for non-register result");
    return iree_ok_status();
  }

  iree_string_view_t reg_class_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_string(
      descriptor_set,
      descriptor_set->reg_classes[alternative->reg_class_id].name_string_offset,
      &reg_class_name));

  loom_string_id_t reg_class_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, reg_class_name, &reg_class_id));
  loom_type_t type = loom_type_register(reg_class_id, operand->unit_count);
  return loom_module_intern_type(module, type, out_type);
}

static iree_status_t loom_low_descriptor_text_asm_immediate_descriptor(
    void* user_data, const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t immediate_index,
    loom_text_low_asm_immediate_descriptor_t* out_immediate) {
  (void)user_data;
  *out_immediate = (loom_text_low_asm_immediate_descriptor_t){0};
  if (immediate_index >= packet->immediate_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm immediate index is out of range");
  }

  const loom_low_descriptor_set_t* descriptor_set =
      (const loom_low_descriptor_set_t*)packet->descriptor_set;
  const loom_low_descriptor_t* descriptor =
      (const loom_low_descriptor_t*)packet->descriptor;
  const loom_low_asm_form_t* asm_form =
      (const loom_low_asm_form_t*)packet->form;
  const uint32_t asm_immediate_index =
      asm_form->immediate_start + immediate_index;
  if (asm_immediate_index >= descriptor_set->asm_immediate_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm immediate row is out of range");
  }
  const loom_low_asm_immediate_t* asm_immediate =
      &descriptor_set->asm_immediates[asm_immediate_index];
  return loom_low_descriptor_text_asm_immediate_info(
      descriptor_set, descriptor, asm_immediate, out_immediate);
}

static iree_status_t loom_low_descriptor_text_asm_build_packet(
    void* user_data, loom_builder_t* builder,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attributes, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_location_id_t location,
    loom_op_t** out_op) {
  (void)user_data;
  if (operand_count != packet->operand_count ||
      result_count != packet->result_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm packet build shape does not match the "
                            "packet descriptor");
  }
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      builder->module, packet->opcode_key, &opcode_id));
  if (packet->builds_as_const) {
    if (result_count != 1) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm const packet must have one result");
    }
    return loom_low_const_build(builder, opcode_id, attributes, result_types[0],
                                location, out_op);
  }
  return loom_low_op_build(builder, opcode_id, operands, operand_count,
                           attributes, result_types, result_count,
                           /*tied_results=*/NULL,
                           /*tied_result_count=*/0, location, out_op);
}

static iree_status_t loom_low_descriptor_text_asm_build_return(
    void* user_data, loom_builder_t* builder, const loom_value_id_t* values,
    iree_host_size_t value_count, loom_location_id_t location,
    loom_op_t** out_op) {
  (void)user_data;
  return loom_low_return_build(builder, values, value_count, location, out_op);
}

static const loom_text_low_asm_vtable_t kLowDescriptorTextAsmVtable = {
    .lookup_descriptor_set = loom_low_descriptor_text_asm_lookup_descriptor_set,
    .lookup_packet = loom_low_descriptor_text_asm_lookup_packet,
    .infer_result_type = loom_low_descriptor_text_asm_infer_result_type,
    .immediate_descriptor = loom_low_descriptor_text_asm_immediate_descriptor,
    .build_packet = loom_low_descriptor_text_asm_build_packet,
    .build_return = loom_low_descriptor_text_asm_build_return,
};

void loom_low_descriptor_text_asm_environment_initialize(
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_text_low_asm_environment_t* out_environment) {
  *out_environment = (loom_text_low_asm_environment_t){
      .vtable = &kLowDescriptorTextAsmVtable,
      .user_data = (void*)descriptor_registry,
  };
}
