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

static iree_status_t loom_low_descriptor_text_asm_descriptor_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t descriptor_operand_index,
    const loom_low_operand_t** out_operand) {
  *out_operand = NULL;
  if (descriptor_operand_index >= descriptor->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm packet references an operand outside the "
                            "descriptor");
  }
  const uint32_t operand_row =
      descriptor->operand_start + descriptor_operand_index;
  if (operand_row >= descriptor_set->operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm descriptor operand row is out of range");
  }
  *out_operand = &descriptor_set->operands[operand_row];
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_result_operand(
    const loom_text_low_asm_packet_descriptor_t* packet, uint16_t result_index,
    uint16_t* out_descriptor_operand_index,
    const loom_low_operand_t** out_operand) {
  *out_descriptor_operand_index = 0;
  *out_operand = NULL;
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
  const loom_low_operand_t* operand = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_descriptor_operand(
      descriptor_set, descriptor, descriptor_operand_index, &operand));
  if (operand->role != LOOM_LOW_OPERAND_ROLE_RESULT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm result references a non-result operand");
  }
  *out_descriptor_operand_index = descriptor_operand_index;
  *out_operand = operand;
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_find_packet_operand_index(
    const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t descriptor_operand_index, bool* out_found,
    uint16_t* out_packet_operand_index) {
  *out_found = false;
  *out_packet_operand_index = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      (const loom_low_descriptor_set_t*)packet->descriptor_set;
  const loom_low_asm_form_t* asm_form =
      (const loom_low_asm_form_t*)packet->form;
  for (uint16_t i = 0; i < packet->operand_count; ++i) {
    const uint32_t asm_operand_index = asm_form->operand_index_start + i;
    if (asm_operand_index >= descriptor_set->asm_operand_index_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm operand index is out of range");
    }
    if (descriptor_set->asm_operand_indices[asm_operand_index] ==
        descriptor_operand_index) {
      *out_found = true;
      *out_packet_operand_index = i;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_find_packet_result_index(
    const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t descriptor_operand_index, bool* out_found,
    uint16_t* out_packet_result_index) {
  *out_found = false;
  *out_packet_result_index = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      (const loom_low_descriptor_set_t*)packet->descriptor_set;
  const loom_low_asm_form_t* asm_form =
      (const loom_low_asm_form_t*)packet->form;
  for (uint16_t i = 0; i < packet->result_count; ++i) {
    const uint32_t asm_operand_index = asm_form->result_operand_index_start + i;
    if (asm_operand_index >= descriptor_set->asm_operand_index_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm result index is out of range");
    }
    if (descriptor_set->asm_operand_indices[asm_operand_index] ==
        descriptor_operand_index) {
      *out_found = true;
      *out_packet_result_index = i;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool loom_low_descriptor_text_asm_constraint_ties_result_type(
    loom_low_constraint_kind_t kind) {
  return kind == LOOM_LOW_CONSTRAINT_KIND_TIED ||
         kind == LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE;
}

static iree_status_t loom_low_descriptor_text_asm_find_tied_packet_operand(
    const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t result_descriptor_operand_index, bool* out_found,
    uint16_t* out_packet_operand_index) {
  *out_found = false;
  *out_packet_operand_index = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      (const loom_low_descriptor_set_t*)packet->descriptor_set;
  const loom_low_descriptor_t* descriptor =
      (const loom_low_descriptor_t*)packet->descriptor;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const uint32_t constraint_index = descriptor->constraint_start + i;
    if (constraint_index >= descriptor_set->constraint_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm descriptor constraint is out of range");
    }
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[constraint_index];
    if (!loom_low_descriptor_text_asm_constraint_ties_result_type(
            constraint->kind)) {
      continue;
    }
    if (constraint->lhs_operand_index != result_descriptor_operand_index) {
      continue;
    }
    if (constraint->rhs_operand_index == LOOM_LOW_ID_NONE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm result tie has no rhs operand");
    }
    bool found = false;
    uint16_t packet_operand_index = 0;
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_find_packet_operand_index(
        packet, constraint->rhs_operand_index, &found, &packet_operand_index));
    if (!found) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low asm result tie references an operand outside the asm form");
    }
    *out_found = true;
    *out_packet_operand_index = packet_operand_index;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_copy_to_module_arena(
    loom_module_t* module, iree_string_view_t value,
    iree_string_view_t* out_value) {
  if (iree_string_view_is_empty(value)) {
    *out_value = iree_string_view_empty();
    return iree_ok_status();
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&module->arena, value.size, (void**)&storage));
  memcpy(storage, value.data, value.size);
  *out_value = iree_make_string_view(storage, value.size);
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_append_reg_type(
    iree_string_builder_t* builder,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, uint16_t reg_class_id) {
  if (reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm result register class is out of range");
  }
  iree_string_view_t reg_class_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_string(
      descriptor_set,
      descriptor_set->reg_classes[reg_class_id].name_string_offset,
      &reg_class_name));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "reg<"));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, reg_class_name));
  if (operand->unit_count != 1) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " x%u", (unsigned)operand->unit_count));
  }
  return iree_string_builder_append_cstring(builder, ">");
}

static iree_status_t loom_low_descriptor_text_asm_format_expected_result_types(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, iree_string_view_t prefix,
    iree_string_view_t* out_detail) {
  *out_detail = iree_string_view_empty();
  iree_string_builder_t builder;
  iree_string_builder_initialize(module->allocator, &builder);
  iree_status_t status = iree_string_builder_append_string(&builder, prefix);
  uint16_t appended_count = 0;
  for (uint16_t i = 0;
       i < operand->reg_class_alt_count && iree_status_is_ok(status); ++i) {
    const uint32_t alt_index = operand->reg_class_alt_start + i;
    if (alt_index >= descriptor_set->reg_class_alt_count) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "low asm result register-class alternative is "
                                "out of range");
      break;
    }
    const loom_low_reg_class_alt_t* alternative =
        &descriptor_set->reg_class_alts[alt_index];
    if (alternative->reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
        iree_all_bits_set(alternative->flags,
                          LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    if (appended_count > 0) {
      status = iree_string_builder_append_cstring(&builder, " | ");
      if (!iree_status_is_ok(status)) break;
    }
    status = loom_low_descriptor_text_asm_append_reg_type(
        &builder, descriptor_set, operand, alternative->reg_class_id);
    ++appended_count;
  }
  if (iree_status_is_ok(status) && appended_count == 0) {
    status = iree_string_builder_append_cstring(&builder, "<none>");
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_descriptor_text_asm_copy_to_module_arena(
        module, iree_string_builder_view(&builder), out_detail);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t loom_low_descriptor_text_asm_make_register_type(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, uint16_t reg_class_id,
    loom_type_t* out_type) {
  iree_string_view_t reg_class_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_string(
      descriptor_set,
      descriptor_set->reg_classes[reg_class_id].name_string_offset,
      &reg_class_name));
  loom_string_id_t reg_class_string_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, reg_class_name, &reg_class_string_id));
  loom_type_t type =
      loom_type_register(reg_class_string_id, operand->unit_count);
  return loom_module_intern_type(module, type, out_type);
}

static iree_status_t loom_low_descriptor_text_asm_find_single_register_alt(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, bool* out_found, bool* out_ambiguous,
    uint16_t* out_reg_class_id) {
  *out_found = false;
  *out_ambiguous = false;
  *out_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const uint32_t alt_index = operand->reg_class_alt_start + i;
    if (alt_index >= descriptor_set->reg_class_alt_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm result register-class alternative is "
                              "out of range");
    }
    const loom_low_reg_class_alt_t* alternative =
        &descriptor_set->reg_class_alts[alt_index];
    if (alternative->reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
        iree_all_bits_set(alternative->flags,
                          LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    if (*out_found) {
      *out_ambiguous = true;
      return iree_ok_status();
    }
    if (alternative->reg_class_id >= descriptor_set->reg_class_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm result register class is out of range");
    }
    *out_found = true;
    *out_reg_class_id = alternative->reg_class_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_result_accepts_type(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, loom_module_t* module, loom_type_t type,
    bool* out_accepted) {
  *out_accepted = false;
  if (!loom_type_is_register(type)) return iree_ok_status();
  if (loom_type_register_unit_count(type) != operand->unit_count) {
    return iree_ok_status();
  }
  const loom_string_id_t actual_class_id = loom_type_register_class_id(type);
  if (actual_class_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm result type register class is out of "
                            "range");
  }
  const iree_string_view_t actual_class =
      module->strings.entries[actual_class_id];
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    const uint32_t alt_index = operand->reg_class_alt_start + i;
    if (alt_index >= descriptor_set->reg_class_alt_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm result register-class alternative is "
                              "out of range");
    }
    const loom_low_reg_class_alt_t* alternative =
        &descriptor_set->reg_class_alts[alt_index];
    if (alternative->reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
        iree_all_bits_set(alternative->flags,
                          LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
      continue;
    }
    if (alternative->reg_class_id >= descriptor_set->reg_class_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm result register class is out of range");
    }
    iree_string_view_t expected_class = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_string(
        descriptor_set,
        descriptor_set->reg_classes[alternative->reg_class_id]
            .name_string_offset,
        &expected_class));
    if (iree_string_view_equal(actual_class, expected_class)) {
      *out_accepted = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_infer_result_type(
    void* user_data, const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, loom_module_t* module, loom_type_t* out_type,
    iree_string_view_t* out_diagnostic_detail) {
  (void)user_data;
  *out_type = loom_type_none();
  *out_diagnostic_detail = iree_string_view_empty();

  const loom_low_descriptor_set_t* descriptor_set =
      (const loom_low_descriptor_set_t*)packet->descriptor_set;
  uint16_t descriptor_operand_index = 0;
  const loom_low_operand_t* operand = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_result_operand(
      packet, result_index, &descriptor_operand_index, &operand));

  bool has_tied_operand = false;
  uint16_t tied_operand_index = 0;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_find_tied_packet_operand(
      packet, descriptor_operand_index, &has_tied_operand,
      &tied_operand_index));
  if (has_tied_operand) {
    if (tied_operand_index >= operand_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm tied operand index is out of range");
    }
    loom_type_t tied_type =
        loom_module_value_type(module, operands[tied_operand_index]);
    bool accepted = false;
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_result_accepts_type(
        descriptor_set, operand, module, tied_type, &accepted));
    if (accepted) {
      *out_type = tied_type;
      return iree_ok_status();
    }
    return loom_low_descriptor_text_asm_format_expected_result_types(
        module, descriptor_set, operand,
        IREE_SV("tied operand type must be one of: "), out_diagnostic_detail);
  }

  bool found = false;
  bool ambiguous = false;
  uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_find_single_register_alt(
      descriptor_set, operand, &found, &ambiguous, &reg_class_id));
  if (!found || ambiguous) {
    return loom_low_descriptor_text_asm_format_expected_result_types(
        module, descriptor_set, operand,
        IREE_SV("result type annotation is required for one of: "),
        out_diagnostic_detail);
  }

  return loom_low_descriptor_text_asm_make_register_type(
      module, descriptor_set, operand, reg_class_id, out_type);
}

static iree_status_t loom_low_descriptor_text_asm_validate_result_type(
    void* user_data, const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, loom_module_t* module, loom_type_t type,
    iree_string_view_t* out_diagnostic_detail) {
  (void)user_data;
  *out_diagnostic_detail = iree_string_view_empty();
  const loom_low_descriptor_set_t* descriptor_set =
      (const loom_low_descriptor_set_t*)packet->descriptor_set;
  uint16_t descriptor_operand_index = 0;
  const loom_low_operand_t* operand = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_result_operand(
      packet, result_index, &descriptor_operand_index, &operand));

  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_result_accepts_type(
      descriptor_set, operand, module, type, &accepted));
  if (!accepted) {
    return loom_low_descriptor_text_asm_format_expected_result_types(
        module, descriptor_set, operand,
        IREE_SV("result type annotation must be one of: "),
        out_diagnostic_detail);
  }

  bool has_tied_operand = false;
  uint16_t tied_operand_index = 0;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_find_tied_packet_operand(
      packet, descriptor_operand_index, &has_tied_operand,
      &tied_operand_index));
  if (!has_tied_operand) return iree_ok_status();
  if (tied_operand_index >= operand_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm tied operand index is out of range");
  }
  loom_type_t tied_type =
      loom_module_value_type(module, operands[tied_operand_index]);
  if (!loom_type_equal(type, tied_type)) {
    *out_diagnostic_detail =
        IREE_SV("result type annotation must match tied operand type");
    return iree_ok_status();
  }
  return iree_ok_status();
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

static iree_status_t loom_low_descriptor_text_asm_build_tied_results(
    loom_builder_t* builder,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_tied_result_t** out_tied_results,
    iree_host_size_t* out_tied_result_count) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      (const loom_low_descriptor_set_t*)packet->descriptor_set;
  const loom_low_descriptor_t* descriptor =
      (const loom_low_descriptor_t*)packet->descriptor;

  iree_host_size_t tied_result_count = 0;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const uint32_t constraint_index = descriptor->constraint_start + i;
    if (constraint_index >= descriptor_set->constraint_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm descriptor constraint is out of range");
    }
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[constraint_index];
    if (constraint->kind == LOOM_LOW_CONSTRAINT_KIND_TIED) {
      ++tied_result_count;
    }
  }
  if (tied_result_count == 0) return iree_ok_status();

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&builder->module->arena, tied_result_count,
                                sizeof(*tied_results), (void**)&tied_results));

  iree_host_size_t next_tied_result = 0;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED) continue;
    if (constraint->rhs_operand_index == LOOM_LOW_ID_NONE) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm tied result has no rhs operand");
    }
    bool found_result = false;
    uint16_t result_index = 0;
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_find_packet_result_index(
        packet, constraint->lhs_operand_index, &found_result, &result_index));
    if (!found_result) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low asm tied result references a result outside the asm form");
    }

    bool found_operand = false;
    uint16_t operand_index = 0;
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_find_packet_operand_index(
        packet, constraint->rhs_operand_index, &found_operand, &operand_index));
    if (!found_operand) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low asm tied result references an operand outside the asm form");
    }

    tied_results[next_tied_result++] = (loom_tied_result_t){
        .result_index = result_index,
        .operand_index = operand_index,
    };
  }
  *out_tied_results = tied_results;
  *out_tied_result_count = tied_result_count;
  return iree_ok_status();
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
  const loom_tied_result_t* tied_results = NULL;
  iree_host_size_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_build_tied_results(
      builder, packet, &tied_results, &tied_result_count));
  return loom_low_op_build(builder, opcode_id, operands, operand_count,
                           attributes, result_types, result_count, tied_results,
                           tied_result_count, location, out_op);
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
    .validate_result_type = loom_low_descriptor_text_asm_validate_result_type,
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
