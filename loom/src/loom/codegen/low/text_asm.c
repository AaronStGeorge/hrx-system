// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/text_asm.h"

#include <inttypes.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

static const uint8_t kLowAsmResourceIndexName[] =
    LOOM_BSTRING_LITERAL("\x05", "index");
static const uint8_t kLowAsmResourceSemanticTypeName[] =
    LOOM_BSTRING_LITERAL("\x0d", "semantic_type");
static const uint8_t kLowAsmResourceValidByteCountName[] =
    LOOM_BSTRING_LITERAL("\x10", "valid_byte_count");
static const uint8_t kLowAsmResourceCacheSwizzleStrideName[] =
    LOOM_BSTRING_LITERAL("\x14", "cache_swizzle_stride");
static const uint8_t kLowAsmFrameIndexOffsetName[] =
    LOOM_BSTRING_LITERAL("\x06", "offset");

static const loom_attr_descriptor_t kLowAsmResourceIndexAttr = {
    kLowAsmResourceIndexName, LOOM_ATTR_I64, 0, 0, NULL, NULL};
static const loom_attr_descriptor_t kLowAsmResourceSemanticTypeAttr = {
    kLowAsmResourceSemanticTypeName, LOOM_ATTR_TYPE, 0, 0, NULL, NULL};
static const loom_attr_descriptor_t kLowAsmResourceValidByteCountAttr = {
    kLowAsmResourceValidByteCountName,
    LOOM_ATTR_I64,
    LOOM_ATTR_OPTIONAL,
    0,
    NULL,
    NULL};
static const loom_attr_descriptor_t kLowAsmResourceCacheSwizzleStrideAttr = {
    kLowAsmResourceCacheSwizzleStrideName,
    LOOM_ATTR_I64,
    LOOM_ATTR_OPTIONAL,
    0,
    NULL,
    NULL};
static const loom_attr_descriptor_t kLowAsmFrameIndexOffsetAttr = {
    kLowAsmFrameIndexOffsetName, LOOM_ATTR_I64, 0, 0, NULL, NULL};

static const loom_low_descriptor_registry_t*
loom_low_descriptor_text_asm_state_registry(
    const loom_text_low_asm_environment_state_t* state) {
  return (const loom_low_descriptor_registry_t*)state;
}

static const loom_text_low_asm_descriptor_set_t*
loom_low_descriptor_text_asm_descriptor_set_handle(
    const loom_low_descriptor_set_t* descriptor_set) {
  return (const loom_text_low_asm_descriptor_set_t*)descriptor_set;
}

static const loom_low_descriptor_set_t*
loom_low_descriptor_text_asm_descriptor_set(
    const loom_text_low_asm_descriptor_set_t* descriptor_set) {
  return (const loom_low_descriptor_set_t*)descriptor_set;
}

static const loom_text_low_asm_form_t* loom_low_descriptor_text_asm_form_handle(
    const loom_low_asm_form_t* asm_form) {
  return (const loom_text_low_asm_form_t*)asm_form;
}

static const loom_low_asm_form_t* loom_low_descriptor_text_asm_form(
    const loom_text_low_asm_form_t* asm_form) {
  return (const loom_low_asm_form_t*)asm_form;
}

static const loom_text_low_asm_descriptor_handle_t*
loom_low_descriptor_text_asm_descriptor_handle(
    const loom_low_descriptor_t* descriptor) {
  return (const loom_text_low_asm_descriptor_handle_t*)descriptor;
}

static const loom_low_descriptor_t* loom_low_descriptor_text_asm_descriptor(
    const loom_text_low_asm_descriptor_handle_t* descriptor) {
  return (const loom_low_descriptor_t*)descriptor;
}

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
  out_immediate->has_default_value =
      iree_any_bit_set(immediate->flags, LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE);
  out_immediate->default_value = immediate->default_value;
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
    const loom_text_low_asm_environment_state_t* state, iree_string_view_t key,
    const loom_text_low_asm_descriptor_set_t** out_descriptor_set) {
  *out_descriptor_set = NULL;
  const loom_low_descriptor_registry_t* registry =
      loom_low_descriptor_text_asm_state_registry(state);
  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_registry_lookup(registry, key, &descriptor_set));
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "low descriptor set was not found");
  }
  *out_descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set_handle(descriptor_set);
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_make_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_asm_form_t* asm_form,
    const loom_low_descriptor_t* descriptor,
    loom_text_low_asm_packet_descriptor_t* out_packet) {
  iree_string_view_t opcode_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_string(
      descriptor_set, descriptor->key_string_offset, &opcode_key));

  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_string(
      descriptor_set, asm_form->mnemonic_string_offset, &mnemonic));

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
    if (asm_immediate->immediate_index >= descriptor->immediate_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low asm immediate references an invalid descriptor field");
    }
    const uint32_t immediate_index =
        descriptor->immediate_start + asm_immediate->immediate_index;
    if (immediate_index >= descriptor_set->immediate_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm immediate field is out of range");
    }
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[immediate_index];
    if (iree_any_bit_set(immediate->flags,
                         LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
      has_named_immediates = true;
      break;
    }
  }

  const bool builds_as_const =
      asm_form->operand_index_count == 0 &&
      asm_form->result_operand_index_count == 1 &&
      asm_form->immediate_count > 0 &&
      !iree_all_bits_set(descriptor->flags,
                         LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING);
  *out_packet = (loom_text_low_asm_packet_descriptor_t){
      .descriptor_set =
          loom_low_descriptor_text_asm_descriptor_set_handle(descriptor_set),
      .form = loom_low_descriptor_text_asm_form_handle(asm_form),
      .descriptor = loom_low_descriptor_text_asm_descriptor_handle(descriptor),
      .opcode_key = opcode_key,
      .mnemonic = mnemonic,
      .result_count = asm_form->result_operand_index_count,
      .operand_count = asm_form->operand_index_count,
      .immediate_count = asm_form->immediate_count,
      .immediate_attribute_field_index = builds_as_const
                                             ? loom_low_const_attrs_ATTR_INDEX
                                             : loom_low_op_attrs_ATTR_INDEX,
      .has_named_immediates = has_named_immediates,
      .builds_as_const = builds_as_const,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_lookup_packet(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set_handle,
    iree_string_view_t mnemonic,
    loom_text_low_asm_packet_descriptor_t* out_packet) {
  (void)state;
  *out_packet = (loom_text_low_asm_packet_descriptor_t){0};
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(descriptor_set_handle);

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

  return loom_low_descriptor_text_asm_make_packet(descriptor_set, asm_form,
                                                  descriptor, out_packet);
}

static iree_status_t loom_low_descriptor_text_asm_lookup_packet_by_opcode(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t opcode,
    loom_text_low_asm_packet_descriptor_t* out_packet) {
  *out_packet = (loom_text_low_asm_packet_descriptor_t){0};
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, opcode, &descriptor_ordinal));
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  if (descriptor == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor ordinal is out of range");
  }

  uint32_t asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    return iree_ok_status();
  }
  asm_form_ordinal = descriptor->canonical_asm_form_ordinal;
  const loom_low_asm_form_t* selected_form =
      loom_low_descriptor_set_asm_form_at(descriptor_set, asm_form_ordinal);
  if (selected_form == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor '%.*s' canonical asm form ordinal "
                            "is out of range",
                            (int)opcode.size, opcode.data);
  }
  return loom_low_descriptor_text_asm_make_packet(descriptor_set, selected_form,
                                                  descriptor, out_packet);
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
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_text_asm_descriptor(packet->descriptor);
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_text_asm_form(packet->form);

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
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_text_asm_form(packet->form);
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
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_text_asm_form(packet->form);
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
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_text_asm_descriptor(packet->descriptor);
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
    const loom_low_operand_t* operand, const loom_module_t* module,
    loom_type_t type, bool* out_accepted) {
  *out_accepted = false;
  if (!loom_type_is_register(type)) {
    return iree_ok_status();
  }
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
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, loom_module_t* module, loom_type_t* out_type,
    iree_string_view_t* out_diagnostic_detail) {
  (void)state;
  *out_type = loom_type_none();
  *out_diagnostic_detail = iree_string_view_empty();

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
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
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, loom_module_t* module, loom_type_t type,
    iree_string_view_t* out_diagnostic_detail) {
  (void)state;
  *out_diagnostic_detail = iree_string_view_empty();
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
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
  if (!has_tied_operand) {
    return iree_ok_status();
  }
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

static iree_status_t
loom_low_descriptor_text_asm_result_type_annotation_required(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    uint16_t result_index, const loom_module_t* module, loom_type_t type,
    bool* out_required, iree_string_view_t* out_diagnostic_detail) {
  (void)state;
  *out_required = true;
  *out_diagnostic_detail = iree_string_view_empty();
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  uint16_t descriptor_operand_index = 0;
  const loom_low_operand_t* operand = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_result_operand(
      packet, result_index, &descriptor_operand_index, &operand));

  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_result_accepts_type(
      descriptor_set, operand, module, type, &accepted));
  if (!accepted) {
    *out_diagnostic_detail =
        IREE_SV("result type annotation is not accepted by low descriptor");
    return iree_ok_status();
  }

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
    if (!loom_type_equal(type, tied_type)) {
      *out_diagnostic_detail =
          IREE_SV("result type annotation must match tied operand type");
      return iree_ok_status();
    }
    *out_required = false;
    return iree_ok_status();
  }

  bool found = false;
  bool ambiguous = false;
  uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_find_single_register_alt(
      descriptor_set, operand, &found, &ambiguous, &reg_class_id));
  *out_required = !found || ambiguous;
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_immediate_descriptor(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_packet_descriptor_t* packet,
    uint16_t immediate_index,
    loom_text_low_asm_immediate_descriptor_t* out_immediate) {
  (void)state;
  *out_immediate = (loom_text_low_asm_immediate_descriptor_t){0};
  if (immediate_index >= packet->immediate_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm immediate index is out of range");
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_text_asm_descriptor(packet->descriptor);
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_text_asm_form(packet->form);
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

static iree_status_t loom_low_descriptor_text_asm_tied_result_count(
    const loom_text_low_asm_packet_descriptor_t* packet,
    iree_host_size_t* out_tied_result_count) {
  *out_tied_result_count = 0;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_text_asm_descriptor(packet->descriptor);
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const uint32_t constraint_index = descriptor->constraint_start + i;
    if (constraint_index >= descriptor_set->constraint_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm descriptor constraint is out of range");
    }
    if (descriptor_set->constraints[constraint_index].kind ==
        LOOM_LOW_CONSTRAINT_KIND_TIED) {
      ++*out_tied_result_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_tied_result_at(
    const loom_text_low_asm_packet_descriptor_t* packet,
    iree_host_size_t tied_result_ordinal, loom_tied_result_t* out_tied_result) {
  *out_tied_result = (loom_tied_result_t){0};
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(packet->descriptor_set);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_text_asm_descriptor(packet->descriptor);

  iree_host_size_t current_ordinal = 0;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const uint32_t constraint_index = descriptor->constraint_start + i;
    if (constraint_index >= descriptor_set->constraint_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm descriptor constraint is out of range");
    }
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[constraint_index];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_TIED) continue;
    if (current_ordinal++ != tied_result_ordinal) continue;

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

    *out_tied_result = (loom_tied_result_t){
        .result_index = result_index,
        .operand_index = operand_index,
        .has_type_change = false,
    };
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "low asm tied result ordinal is out of range");
}

static iree_status_t loom_low_descriptor_text_asm_build_tied_results(
    loom_builder_t* builder,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_tied_result_t** out_tied_results,
    iree_host_size_t* out_tied_result_count) {
  *out_tied_results = NULL;
  *out_tied_result_count = 0;
  iree_host_size_t tied_result_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_tied_result_count(
      packet, &tied_result_count));
  if (tied_result_count == 0) {
    return iree_ok_status();
  }

  loom_tied_result_t* tied_results = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&builder->module->arena, tied_result_count,
                                sizeof(*tied_results), (void**)&tied_results));

  for (iree_host_size_t i = 0; i < tied_result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_tied_result_at(
        packet, i, &tied_results[i]));
  }
  *out_tied_results = tied_results;
  *out_tied_result_count = tied_result_count;
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_build_packet(
    const loom_text_low_asm_environment_state_t* state, loom_builder_t* builder,
    const loom_text_low_asm_packet_descriptor_t* packet,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attributes, const loom_type_t* result_types,
    iree_host_size_t result_count, loom_location_id_t location,
    loom_op_t** out_op) {
  (void)state;
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
    const loom_text_low_asm_environment_state_t* state, loom_builder_t* builder,
    const loom_value_id_t* values, iree_host_size_t value_count,
    loom_location_id_t location, loom_op_t** out_op) {
  (void)state;
  return loom_low_return_build(builder, values, value_count, location, out_op);
}

static iree_string_view_t loom_low_descriptor_text_asm_op_name(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  return vtable ? loom_op_vtable_name(vtable) : IREE_SV("<unknown>");
}

static iree_status_t loom_low_descriptor_text_asm_opcode_attr(
    const loom_module_t* module, const loom_op_t* op, uint8_t attr_index,
    iree_string_view_t* out_opcode) {
  *out_opcode = iree_string_view_empty();
  if (attr_index >= op->attribute_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm packet opcode attribute is missing");
  }
  const loom_attribute_t attr = loom_op_attrs(op)[attr_index];
  if (attr.kind != LOOM_ATTR_STRING ||
      attr.string_id == LOOM_STRING_ID_INVALID ||
      attr.string_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm packet opcode attribute is invalid");
  }
  *out_opcode = module->strings.entries[attr.string_id];
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_attr_slice(
    const loom_op_t* op, uint8_t attr_index,
    loom_named_attr_slice_t* out_attrs) {
  *out_attrs = loom_make_named_attr_slice(NULL, 0);
  if (attr_index >= op->attribute_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm packet immediate dictionary is missing");
  }
  const loom_attribute_t attr = loom_op_attrs(op)[attr_index];
  if (loom_attr_is_absent(attr)) {
    return iree_ok_status();
  }
  if (attr.kind != LOOM_ATTR_DICT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm packet immediate dictionary is invalid");
  }
  if (attr.count > 0 && attr.dict_entries == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm packet immediate dictionary has NULL entries");
  }
  *out_attrs = loom_attr_as_dict(attr);
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_attr_name(
    const loom_module_t* module, const loom_named_attr_t* attr,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (attr->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm packet immediate name is invalid");
  }
  *out_name = module->strings.entries[attr->name_id];
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t field_name, bool* out_found) {
  *out_found = false;
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    iree_string_view_t attr_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_attr_name(
        module, &attrs.entries[i], &attr_name));
    if (iree_string_view_equal(attr_name, field_name)) {
      *out_found = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_lookup_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t field_name, const loom_named_attr_t** out_attr) {
  *out_attr = NULL;
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    iree_string_view_t attr_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_attr_name(
        module, &attrs.entries[i], &attr_name));
    if (iree_string_view_equal(attr_name, field_name)) {
      *out_attr = &attrs.entries[i];
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_resource_key_to_kind(
    iree_string_view_t key, uint8_t* out_kind) {
  if (iree_string_view_equal(key, IREE_SV("native_pointer"))) {
    *out_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER;
    return iree_ok_status();
  }
  if (iree_string_view_equal(key, IREE_SV("vm_state"))) {
    *out_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_VM_STATE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(key, IREE_SV("vm_import"))) {
    *out_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_VM_IMPORT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(key, IREE_SV("hal_buffer_resource"))) {
    *out_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BUFFER_RESOURCE;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown low asm resource kind '%.*s'", (int)key.size,
                          key.data);
}

static iree_status_t loom_low_descriptor_text_asm_resource_kind_to_key(
    uint8_t kind, iree_string_view_t* out_key) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER:
      *out_key = IREE_SV("native_pointer");
      return iree_ok_status();
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_STATE:
      *out_key = IREE_SV("vm_state");
      return iree_ok_status();
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_IMPORT:
      *out_key = IREE_SV("vm_import");
      return iree_ok_status();
    case LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BUFFER_RESOURCE:
      *out_key = IREE_SV("hal_buffer_resource");
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown low resource import kind %u",
                              (unsigned)kind);
  }
}

static iree_status_t loom_low_descriptor_text_asm_validate_immediates(
    const loom_module_t* module,
    const loom_text_low_asm_packet_descriptor_t* packet,
    loom_named_attr_slice_t attrs) {
  if (attrs.count > packet->immediate_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm packet '%.*s' expects at most %u immediate attributes but op "
        "has "
        "%" PRIhsz,
        (int)packet->opcode_key.size, packet->opcode_key.data,
        packet->immediate_count, attrs.count);
  }

  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    iree_string_view_t attr_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_attr_name(
        module, &attrs.entries[i], &attr_name));
    bool expected = false;
    for (uint16_t j = 0; j < packet->immediate_count; ++j) {
      loom_text_low_asm_immediate_descriptor_t immediate = {0};
      IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_immediate_descriptor(
          NULL, packet, j, &immediate));
      if (iree_string_view_equal(attr_name, immediate.field_name)) {
        expected = true;
        break;
      }
    }
    if (!expected) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm packet '%.*s' has unexpected "
                              "immediate attribute '%.*s'",
                              (int)packet->opcode_key.size,
                              packet->opcode_key.data, (int)attr_name.size,
                              attr_name.data);
    }
  }

  for (uint16_t i = 0; i < packet->immediate_count; ++i) {
    loom_text_low_asm_immediate_descriptor_t immediate = {0};
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_immediate_descriptor(
        NULL, packet, i, &immediate));
    bool found = false;
    IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_find_attr(
        module, attrs, immediate.field_name, &found));
    if (!found && !immediate.has_default_value) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low asm packet '%.*s' is missing immediate "
          "attribute '%.*s'",
          (int)packet->opcode_key.size, packet->opcode_key.data,
          (int)immediate.field_name.size, immediate.field_name.data);
    }
  }
  return iree_ok_status();
}

static bool loom_low_descriptor_text_asm_tied_result_equal(
    loom_tied_result_t lhs, loom_tied_result_t rhs) {
  return lhs.result_index == rhs.result_index &&
         lhs.operand_index == rhs.operand_index &&
         lhs.has_type_change == rhs.has_type_change;
}

static iree_status_t loom_low_descriptor_text_asm_validate_tied_results(
    const loom_text_low_asm_packet_descriptor_t* packet, const loom_op_t* op) {
  iree_host_size_t expected_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_text_asm_tied_result_count(packet, &expected_count));
  if (op->tied_result_count != expected_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm packet '%.*s' expects %" PRIhsz " tied results but op has %u",
        (int)packet->opcode_key.size, packet->opcode_key.data, expected_count,
        op->tied_result_count);
  }

  const loom_tied_result_t* actual_ties = loom_op_tied_results(op);
  for (iree_host_size_t i = 0; i < expected_count; ++i) {
    loom_tied_result_t expected_tie = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_descriptor_text_asm_tied_result_at(packet, i, &expected_tie));
    bool found = false;
    for (uint16_t j = 0; j < op->tied_result_count; ++j) {
      if (loom_low_descriptor_text_asm_tied_result_equal(expected_tie,
                                                         actual_ties[j])) {
        found = true;
        break;
      }
    }
    if (!found) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low asm packet '%.*s' is missing tied result %u -> operand %u",
          (int)packet->opcode_key.size, packet->opcode_key.data,
          expected_tie.result_index, expected_tie.operand_index);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_describe_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_module_t* module, const loom_op_t* op, bool is_const,
    loom_text_low_asm_statement_t* out_statement) {
  const uint8_t opcode_attr_index = is_const ? loom_low_const_opcode_ATTR_INDEX
                                             : loom_low_op_opcode_ATTR_INDEX;
  const uint8_t attrs_attr_index =
      is_const ? loom_low_const_attrs_ATTR_INDEX : loom_low_op_attrs_ATTR_INDEX;

  iree_string_view_t opcode = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_opcode_attr(
      module, op, opcode_attr_index, &opcode));
  loom_text_low_asm_packet_descriptor_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_lookup_packet_by_opcode(
      descriptor_set, opcode, &packet));
  if (packet.descriptor == NULL) {
    return iree_ok_status();
  }
  if (is_const != packet.builds_as_const) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm packet '%.*s' has the wrong canonical operation form",
        (int)opcode.size, opcode.data);
  }

  loom_named_attr_slice_t attrs = loom_make_named_attr_slice(NULL, 0);
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_text_asm_attr_slice(op, attrs_attr_index, &attrs));
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_text_asm_validate_immediates(module, &packet, attrs));

  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  if (is_const) {
    if (op->result_count != 1 || op->operand_count != 0 ||
        op->tied_result_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low const packet '%.*s' has invalid shape",
                              (int)opcode.size, opcode.data);
    }
  } else {
    if (op->result_count != packet.result_count ||
        op->operand_count != packet.operand_count) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_low_descriptor_text_asm_validate_tied_results(&packet, op));
  }

  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_PACKET,
      .op = op,
      .packet = packet,
      .results = results,
      .result_count = op->result_count,
      .operands = operands,
      .operand_count = op->operand_count,
      .attributes = attrs,
      .has_immediate_attribute_field = true,
      .immediate_attribute_field_index = attrs_attr_index,
      .location = op->location,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_structural_attr_descriptor(
    const loom_text_low_asm_environment_state_t* state,
    loom_text_low_asm_structural_kind_t kind, iree_string_view_t attr_name,
    const loom_attr_descriptor_t** out_descriptor) {
  (void)state;
  *out_descriptor = NULL;
  if (kind != LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE) {
    if (kind == LOOM_TEXT_LOW_ASM_STRUCTURAL_FRAME_INDEX &&
        iree_string_view_equal(attr_name, IREE_SV("offset"))) {
      *out_descriptor = &kLowAsmFrameIndexOffsetAttr;
    }
    return iree_ok_status();
  }
  if (iree_string_view_equal(attr_name, IREE_SV("index"))) {
    *out_descriptor = &kLowAsmResourceIndexAttr;
    return iree_ok_status();
  }
  if (iree_string_view_equal(attr_name, IREE_SV("semantic_type"))) {
    *out_descriptor = &kLowAsmResourceSemanticTypeAttr;
    return iree_ok_status();
  }
  if (iree_string_view_equal(attr_name, IREE_SV("valid_byte_count"))) {
    *out_descriptor = &kLowAsmResourceValidByteCountAttr;
    return iree_ok_status();
  }
  if (iree_string_view_equal(attr_name, IREE_SV("cache_swizzle_stride"))) {
    *out_descriptor = &kLowAsmResourceCacheSwizzleStrideAttr;
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_required_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, const loom_named_attr_t** out_attr) {
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_text_asm_lookup_attr(module, attrs, name, out_attr));
  if (*out_attr == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm structural intrinsic is missing "
                            "required attribute '%.*s'",
                            (int)name.size, name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_build_resource(
    loom_builder_t* builder, iree_string_view_t key,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  uint8_t import_kind = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_text_asm_resource_key_to_kind(key, &import_kind));

  const loom_named_attr_t* index_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_required_attr(
      builder->module, attrs, IREE_SV("index"), &index_attr));
  if (index_attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm resource index must be an i64 attr");
  }

  const loom_named_attr_t* semantic_type_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_required_attr(
      builder->module, attrs, IREE_SV("semantic_type"), &semantic_type_attr));
  if (semantic_type_attr->value.kind != LOOM_ATTR_TYPE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low asm resource semantic_type must be a type attr");
  }

  loom_low_resource_build_flags_t flags = 0;
  int64_t valid_byte_count = 0;
  const loom_named_attr_t* valid_byte_count_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_lookup_attr(
      builder->module, attrs, IREE_SV("valid_byte_count"),
      &valid_byte_count_attr));
  if (valid_byte_count_attr != NULL) {
    if (valid_byte_count_attr->value.kind != LOOM_ATTR_I64) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low asm resource valid_byte_count must be an i64 attr");
    }
    flags |= LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_VALID_BYTE_COUNT;
    valid_byte_count = valid_byte_count_attr->value.i64;
  }

  int64_t cache_swizzle_stride = 0;
  const loom_named_attr_t* cache_swizzle_stride_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_lookup_attr(
      builder->module, attrs, IREE_SV("cache_swizzle_stride"),
      &cache_swizzle_stride_attr));
  if (cache_swizzle_stride_attr != NULL) {
    if (cache_swizzle_stride_attr->value.kind != LOOM_ATTR_I64) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low asm resource cache_swizzle_stride must be an i64 attr");
    }
    flags |= LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_CACHE_SWIZZLE_STRIDE;
    cache_swizzle_stride = cache_swizzle_stride_attr->value.i64;
  }

  return loom_low_resource_build(
      builder, flags, import_kind, index_attr->value.i64,
      semantic_type_attr->value.type_id, valid_byte_count, cache_swizzle_stride,
      result_type, location, out_op);
}

static iree_status_t loom_low_descriptor_text_asm_build_frame_index(
    loom_builder_t* builder, loom_named_attr_slice_t attrs,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op) {
  const loom_named_attr_t* slot_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_required_attr(
      builder->module, attrs, IREE_SV("slot"), &slot_attr));
  if (slot_attr->value.kind != LOOM_ATTR_SYMBOL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm frame_index slot must be a symbol attr");
  }

  const loom_named_attr_t* offset_attr = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_required_attr(
      builder->module, attrs, IREE_SV("offset"), &offset_attr));
  if (offset_attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm frame_index offset must be an i64 attr");
  }

  return loom_low_frame_index_build(builder, slot_attr->value.symbol,
                                    offset_attr->value.i64, result_type,
                                    location, out_op);
}

static iree_status_t loom_low_descriptor_text_asm_build_structural(
    const loom_text_low_asm_environment_state_t* state, loom_builder_t* builder,
    loom_text_low_asm_structural_kind_t kind, iree_string_view_t key,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attributes, int64_t offset, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  (void)state;
  switch (kind) {
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE:
      if (operand_count != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm resource takes no operands");
      }
      return loom_low_descriptor_text_asm_build_resource(
          builder, key, attributes, result_type, location, out_op);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_LIVE_IN: {
      if (operand_count != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm live_in takes no operands");
      }
      loom_string_id_t source = LOOM_STRING_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_module_intern_string(builder->module, key, &source));
      return loom_low_live_in_build(builder, source, attributes, result_type,
                                    location, out_op);
    }
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_CONCAT:
      return loom_low_concat_build(builder, operands, operand_count,
                                   result_type, location, out_op);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_SLICE:
      if (operand_count != 1) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm slice takes one operand");
      }
      return loom_low_slice_build(builder, operands[0], offset, result_type,
                                  location, out_op);
    case LOOM_TEXT_LOW_ASM_STRUCTURAL_FRAME_INDEX:
      if (operand_count != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm frame_index takes no operands");
      }
      return loom_low_descriptor_text_asm_build_frame_index(
          builder, attributes, result_type, location, out_op);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown low asm structural kind %u",
                              (unsigned)kind);
  }
}

static iree_status_t loom_low_descriptor_text_asm_set_structural_attr(
    const loom_module_t* module, const loom_op_t* op, uint8_t attr_index,
    iree_string_view_t name, const loom_attr_descriptor_t* descriptor,
    loom_text_low_asm_statement_t* statement) {
  if (attr_index >= op->attribute_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm structural attr is out of range");
  }
  const loom_attribute_t* attr = &loom_op_attrs(op)[attr_index];
  if (loom_attr_is_absent(*attr)) {
    return iree_ok_status();
  }
  if (statement->structural_attribute_count >=
      IREE_ARRAYSIZE(statement->structural_attributes)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low asm structural attr array is full");
  }
  statement->structural_attributes[statement->structural_attribute_count++] =
      (loom_text_low_asm_structural_attribute_t){
          .name = name,
          .value = attr,
          .descriptor = descriptor,
      };
  (void)module;
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_describe_resource(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_resource_kind_to_key(
      loom_low_resource_import_kind(op), &key));
  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL,
      .op = op,
      .structural_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_RESOURCE,
      .structural_key = key,
      .results = loom_op_const_results(op),
      .result_count = 1,
      .location = op->location,
  };
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_resource_index_ATTR_INDEX, IREE_SV("index"),
      &kLowAsmResourceIndexAttr, out_statement));
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_resource_semantic_type_ATTR_INDEX,
      IREE_SV("semantic_type"), &kLowAsmResourceSemanticTypeAttr,
      out_statement));
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_resource_valid_byte_count_ATTR_INDEX,
      IREE_SV("valid_byte_count"), &kLowAsmResourceValidByteCountAttr,
      out_statement));
  return loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_resource_cache_swizzle_stride_ATTR_INDEX,
      IREE_SV("cache_swizzle_stride"), &kLowAsmResourceCacheSwizzleStrideAttr,
      out_statement);
}

static iree_status_t loom_low_descriptor_text_asm_describe_live_in(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  loom_string_id_t source_id = loom_low_live_in_source(op);
  if (source_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low.live_in source string is out of range");
  }
  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL,
      .op = op,
      .structural_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_LIVE_IN,
      .structural_key = module->strings.entries[source_id],
      .results = loom_op_const_results(op),
      .result_count = 1,
      .attributes = loom_low_live_in_attrs(op),
      .location = op->location,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_describe_concat(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  (void)module;
  loom_value_slice_t sources = loom_low_concat_sources(op);
  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL,
      .op = op,
      .structural_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_CONCAT,
      .results = loom_op_const_results(op),
      .result_count = 1,
      .operands = sources.values,
      .operand_count = sources.count,
      .location = op->location,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_describe_slice(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  (void)module;
  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL,
      .op = op,
      .structural_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_SLICE,
      .structural_offset = loom_low_slice_offset(op),
      .results = loom_op_const_results(op),
      .result_count = 1,
      .operands = loom_op_const_operands(op),
      .operand_count = 1,
      .location = op->location,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t ref,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm structural symbol is out of range");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm structural symbol name is out of range");
  }
  *out_name = module->strings.entries[symbol->name_id];
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_text_asm_describe_frame_index(
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  iree_string_view_t slot_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_descriptor_text_asm_symbol_name(
      module, loom_low_frame_index_slot(op), &slot_name));
  *out_statement = (loom_text_low_asm_statement_t){
      .kind = LOOM_TEXT_LOW_ASM_STATEMENT_STRUCTURAL,
      .op = op,
      .structural_kind = LOOM_TEXT_LOW_ASM_STRUCTURAL_FRAME_INDEX,
      .structural_key = slot_name,
      .results = loom_op_const_results(op),
      .result_count = 1,
      .location = op->location,
  };
  return loom_low_descriptor_text_asm_set_structural_attr(
      module, op, loom_low_frame_index_offset_ATTR_INDEX, IREE_SV("offset"),
      &kLowAsmFrameIndexOffsetAttr, out_statement);
}

static iree_status_t loom_low_descriptor_text_asm_describe_operation(
    const loom_text_low_asm_environment_state_t* state,
    const loom_text_low_asm_descriptor_set_t* descriptor_set_handle,
    const loom_module_t* module, const loom_op_t* op,
    loom_text_low_asm_statement_t* out_statement) {
  (void)state;
  *out_statement = (loom_text_low_asm_statement_t){0};
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_text_asm_descriptor_set(descriptor_set_handle);
  if (loom_low_return_isa(op)) {
    *out_statement = (loom_text_low_asm_statement_t){
        .kind = LOOM_TEXT_LOW_ASM_STATEMENT_RETURN,
        .op = op,
        .operands = loom_low_return_values(op).values,
        .operand_count = (uint16_t)loom_low_return_values(op).count,
        .location = op->location,
    };
    return iree_ok_status();
  }
  if (loom_low_const_isa(op)) {
    return loom_low_descriptor_text_asm_describe_packet(
        descriptor_set, module, op, /*is_const=*/true, out_statement);
  }
  if (loom_low_op_isa(op)) {
    return loom_low_descriptor_text_asm_describe_packet(
        descriptor_set, module, op, /*is_const=*/false, out_statement);
  }
  if (loom_low_resource_isa(op)) {
    return loom_low_descriptor_text_asm_describe_resource(module, op,
                                                          out_statement);
  }
  if (loom_low_live_in_isa(op)) {
    return loom_low_descriptor_text_asm_describe_live_in(module, op,
                                                         out_statement);
  }
  if (loom_low_concat_isa(op)) {
    return loom_low_descriptor_text_asm_describe_concat(module, op,
                                                        out_statement);
  }
  if (loom_low_slice_isa(op)) {
    return loom_low_descriptor_text_asm_describe_slice(module, op,
                                                       out_statement);
  }
  if (loom_low_frame_index_isa(op)) {
    return loom_low_descriptor_text_asm_describe_frame_index(module, op,
                                                             out_statement);
  }

  iree_string_view_t op_name = loom_low_descriptor_text_asm_op_name(module, op);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "low asm region contains unsupported op '%.*s'",
                          (int)op_name.size, op_name.data);
}

static const loom_text_low_asm_vtable_t kLowDescriptorTextAsmVtable = {
    .lookup_descriptor_set = loom_low_descriptor_text_asm_lookup_descriptor_set,
    .lookup_packet = loom_low_descriptor_text_asm_lookup_packet,
    .infer_result_type = loom_low_descriptor_text_asm_infer_result_type,
    .validate_result_type = loom_low_descriptor_text_asm_validate_result_type,
    .result_type_annotation_required =
        loom_low_descriptor_text_asm_result_type_annotation_required,
    .immediate_descriptor = loom_low_descriptor_text_asm_immediate_descriptor,
    .build_packet = loom_low_descriptor_text_asm_build_packet,
    .build_return = loom_low_descriptor_text_asm_build_return,
    .structural_attr_descriptor =
        loom_low_descriptor_text_asm_structural_attr_descriptor,
    .build_structural = loom_low_descriptor_text_asm_build_structural,
    .describe_operation = loom_low_descriptor_text_asm_describe_operation,
};

void loom_low_descriptor_text_asm_environment_initialize(
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_text_low_asm_environment_t* out_environment) {
  *out_environment = (loom_text_low_asm_environment_t){
      .vtable = &kLowDescriptorTextAsmVtable,
      .state =
          (const loom_text_low_asm_environment_state_t*)descriptor_registry,
  };
}
