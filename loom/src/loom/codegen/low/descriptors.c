// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptors.h"

#include <inttypes.h>

#include "loom/util/stable_id.h"

static iree_status_t loom_low_verify_pointer_for_count(const void* pointer,
                                                       uint32_t count,
                                                       const char* table_name) {
  if (count != 0 && pointer == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set table '%s' is required when "
                            "its count is non-zero",
                            table_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_span(uint32_t start, uint32_t count,
                                          uint32_t total_count,
                                          const char* table_name) {
  if (start > total_count || count > total_count - start) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor set span [%" PRIu32 ", %" PRIu32
                            ") exceeds table '%s' count %" PRIu32,
                            start, start + count, table_name, total_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_known_flags(uint16_t flags,
                                                 uint16_t known_mask,
                                                 const char* table_name,
                                                 uint32_t table_index) {
  const uint16_t unknown_flags = flags & (uint16_t)~known_mask;
  if (unknown_flags == 0) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "low %s %" PRIu32
                          " has unknown generic flag bits 0x%04x",
                          table_name, table_index, (unsigned)unknown_flags);
}

static iree_status_t loom_low_descriptor_set_string_impl(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, bool allow_none,
    iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    if (allow_none) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "required low descriptor string offset is NONE");
  }
  if (descriptor_set->string_table.data == NULL ||
      descriptor_set->string_table.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor string table is empty");
  }
  if (string_offset >= descriptor_set->string_table.data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor string offset %" PRIu32
                            " exceeds string table length %" PRIu32,
                            string_offset,
                            descriptor_set->string_table.data_length);
  }
  loom_bstring_t bstring = NULL;
  if (!loom_bstring_table_try_get(&descriptor_set->string_table, string_offset,
                                  &bstring)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor string offset %" PRIu32
                            " does not name a complete B-string",
                            string_offset);
  }
  *out_string = loom_bstring_view(bstring);
  return iree_ok_status();
}

static iree_status_t loom_low_verify_required_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, const char* field_name) {
  iree_string_view_t value = iree_string_view_empty();
  iree_status_t status = loom_low_descriptor_set_string_impl(
      descriptor_set, string_offset, /*allow_none=*/false, &value);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(status, "invalid required string field '%s'",
                                  field_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_optional_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, const char* field_name) {
  iree_string_view_t value = iree_string_view_empty();
  iree_status_t status = loom_low_descriptor_set_string_impl(
      descriptor_set, string_offset, /*allow_none=*/true, &value);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(status, "invalid optional string field '%s'",
                                  field_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_non_empty_required_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, const char* field_name,
    iree_string_view_t* out_value) {
  iree_string_view_t value = iree_string_view_empty();
  iree_status_t status = loom_low_descriptor_set_string_impl(
      descriptor_set, string_offset, /*allow_none=*/false, &value);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(status, "invalid required string field '%s'",
                                  field_name);
  }
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "required low descriptor string field '%s' is "
                            "empty",
                            field_name);
  }
  if (out_value != NULL) {
    *out_value = value;
  }
  return iree_ok_status();
}

uint64_t loom_low_descriptor_stable_id_from_key(iree_string_view_t key) {
  return loom_stable_id_from_string(key);
}

static iree_status_t loom_low_verify_tables_present(
    const loom_low_descriptor_set_t* descriptor_set) {
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->descriptors, descriptor_set->descriptor_count,
      "descriptors"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->descriptor_refs, descriptor_set->descriptor_ref_count,
      "descriptor_refs"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->descriptor_id_refs,
      descriptor_set->descriptor_id_ref_count, "descriptor_id_refs"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->asm_forms, descriptor_set->asm_form_count, "asm_forms"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->asm_operand_indices,
      descriptor_set->asm_operand_index_count, "asm_operand_indices"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->asm_immediates, descriptor_set->asm_immediate_count,
      "asm_immediates"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->operands, descriptor_set->operand_count, "operands"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->immediates, descriptor_set->immediate_count,
      "immediates"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->enum_domains, descriptor_set->enum_domain_count,
      "enum_domains"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->enum_values, descriptor_set->enum_value_count,
      "enum_values"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->effects, descriptor_set->effect_count, "effects"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->constraints, descriptor_set->constraint_count,
      "constraints"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->reg_classes, descriptor_set->reg_class_count,
      "reg_classes"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->reg_class_alts, descriptor_set->reg_class_alt_count,
      "reg_class_alts"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->schedule_classes, descriptor_set->schedule_class_count,
      "schedule_classes"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->issue_uses, descriptor_set->issue_use_count,
      "issue_uses"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->resources, descriptor_set->resource_count, "resources"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->hazards, descriptor_set->hazard_count, "hazards"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->pressure_deltas, descriptor_set->pressure_delta_count,
      "pressure_deltas"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->feature_mask_words,
      descriptor_set->feature_mask_word_count, "feature_mask_words"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->encoding_field_values,
      descriptor_set->encoding_field_value_count, "encoding_field_values"));
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_refs(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set->descriptor_ref_count !=
      descriptor_set->descriptor_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor reference map count %" PRIu32
                            " does not match descriptor count %" PRIu32,
                            descriptor_set->descriptor_ref_count,
                            descriptor_set->descriptor_count);
  }
  iree_string_view_t previous_key = iree_string_view_empty();
  for (uint32_t i = 0; i < descriptor_set->descriptor_ref_count; ++i) {
    const loom_low_descriptor_ref_t* descriptor_ref =
        &descriptor_set->descriptor_refs[i];
    iree_string_view_t ref_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set, descriptor_ref->key_string_offset, /*allow_none=*/false,
        &ref_key));
    if (i > 0 && iree_string_view_compare(previous_key, ref_key) >= 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor reference map is not strictly sorted near '%.*s'",
          (int)ref_key.size, ref_key.data);
    }
    if (descriptor_ref->descriptor_ordinal >=
        descriptor_set->descriptor_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low descriptor reference '%.*s' points at descriptor ordinal "
          "%" PRIu32 " but only %" PRIu32 " descriptors exist",
          (int)ref_key.size, ref_key.data, descriptor_ref->descriptor_ordinal,
          descriptor_set->descriptor_count);
    }
    iree_string_view_t descriptor_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set,
        descriptor_set->descriptors[descriptor_ref->descriptor_ordinal]
            .key_string_offset,
        /*allow_none=*/false, &descriptor_key));
    if (!iree_string_view_equal(ref_key, descriptor_key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor reference key '%.*s' does not "
                              "match descriptor %" PRIu32 " key '%.*s'",
                              (int)ref_key.size, ref_key.data,
                              descriptor_ref->descriptor_ordinal,
                              (int)descriptor_key.size, descriptor_key.data);
    }
    previous_key = ref_key;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_id_refs(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set->descriptor_id_ref_count !=
      descriptor_set->descriptor_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor ID reference map count %" PRIu32
                            " does not match descriptor count %" PRIu32,
                            descriptor_set->descriptor_id_ref_count,
                            descriptor_set->descriptor_count);
  }
  uint64_t previous_stable_id = 0;
  for (uint32_t i = 0; i < descriptor_set->descriptor_id_ref_count; ++i) {
    const loom_low_descriptor_id_ref_t* descriptor_id_ref =
        &descriptor_set->descriptor_id_refs[i];
    if (descriptor_id_ref->stable_id == LOOM_LOW_DESCRIPTOR_ID_NONE) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor ID reference map contains absent ID at row %" PRIu32,
          i);
    }
    if (i > 0 && previous_stable_id >= descriptor_id_ref->stable_id) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor ID reference map is not strictly sorted near "
          "0x%016" PRIx64,
          descriptor_id_ref->stable_id);
    }
    if (descriptor_id_ref->descriptor_ordinal >=
        descriptor_set->descriptor_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low descriptor ID reference 0x%016" PRIx64
                              " points at descriptor ordinal %" PRIu32
                              " but only %" PRIu32 " descriptors exist",
                              descriptor_id_ref->stable_id,
                              descriptor_id_ref->descriptor_ordinal,
                              descriptor_set->descriptor_count);
    }
    const loom_low_descriptor_t* descriptor =
        &descriptor_set->descriptors[descriptor_id_ref->descriptor_ordinal];
    if (descriptor->stable_id != descriptor_id_ref->stable_id) {
      iree_string_view_t descriptor_key = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
          descriptor_set, descriptor->key_string_offset, /*allow_none=*/false,
          &descriptor_key));
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor ID reference 0x%016" PRIx64
          " does not match descriptor %" PRIu32 " key '%.*s' ID 0x%016" PRIx64,
          descriptor_id_ref->stable_id, descriptor_id_ref->descriptor_ordinal,
          (int)descriptor_key.size, descriptor_key.data, descriptor->stable_id);
    }
    previous_stable_id = descriptor_id_ref->stable_id;
  }
  return iree_ok_status();
}

static bool loom_low_operand_role_is_packet_operand(
    loom_low_operand_role_t role) {
  return role == LOOM_LOW_OPERAND_ROLE_OPERAND ||
         role == LOOM_LOW_OPERAND_ROLE_PREDICATE ||
         role == LOOM_LOW_OPERAND_ROLE_RESOURCE;
}

static bool loom_low_operand_role_is_valid(loom_low_operand_role_t role) {
  switch (role) {
    case LOOM_LOW_OPERAND_ROLE_RESULT:
    case LOOM_LOW_OPERAND_ROLE_OPERAND:
    case LOOM_LOW_OPERAND_ROLE_OPERAND_RESULT:
    case LOOM_LOW_OPERAND_ROLE_PREDICATE:
    case LOOM_LOW_OPERAND_ROLE_RESOURCE:
    case LOOM_LOW_OPERAND_ROLE_IMPLICIT:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_low_verify_asm_operand_indices(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index,
    uint32_t start, uint16_t count, bool expect_result) {
  for (uint16_t i = 0; i < count; ++i) {
    const uint16_t operand_index =
        descriptor_set->asm_operand_indices[start + i];
    if (operand_index >= descriptor->operand_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low asm form for descriptor %" PRIu32 " references operand %" PRIu16
          " but descriptor has only %" PRIu16 " operands",
          descriptor_index, operand_index, descriptor->operand_count);
    }
    for (uint16_t j = 0; j < i; ++j) {
      if (descriptor_set->asm_operand_indices[start + j] == operand_index) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm form for descriptor %" PRIu32
                                " references operand %" PRIu16
                                " more than once",
                                descriptor_index, operand_index);
      }
    }
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + operand_index];
    if (expect_result) {
      if (operand_index >= descriptor->result_count ||
          operand->role != LOOM_LOW_OPERAND_ROLE_RESULT) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low asm form for descriptor %" PRIu32
                                " result operand %" PRIu16
                                " does not name a descriptor result",
                                descriptor_index, operand_index);
      }
    } else if (!loom_low_operand_role_is_packet_operand(operand->role) ||
               iree_all_bits_set(operand->flags,
                                 LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm form for descriptor %" PRIu32
                              " operand %" PRIu16
                              " does not name an explicit packet operand",
                              descriptor_index, operand_index);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_asm_immediates(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_asm_form_t* asm_form,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  for (uint16_t i = 0; i < asm_form->immediate_count; ++i) {
    const uint32_t row_index = asm_form->immediate_start + i;
    const loom_low_asm_immediate_t* asm_immediate =
        &descriptor_set->asm_immediates[row_index];
    if (asm_immediate->immediate_index >= descriptor->immediate_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low asm form for descriptor %" PRIu32
                              " references immediate %" PRIu16
                              " but descriptor has only %" PRIu16 " immediates",
                              descriptor_index, asm_immediate->immediate_index,
                              descriptor->immediate_count);
    }
    for (uint16_t j = 0; j < i; ++j) {
      const loom_low_asm_immediate_t* previous =
          &descriptor_set->asm_immediates[asm_form->immediate_start + j];
      if (previous->immediate_index == asm_immediate->immediate_index) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low asm form for descriptor %" PRIu32
            " references immediate %" PRIu16 " more than once",
            descriptor_index, asm_immediate->immediate_index);
      }
    }

    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set, asm_immediate->name_string_offset, /*allow_none=*/true,
        &name));
    if (asm_immediate->name_string_offset != LOOM_LOW_STRING_OFFSET_NONE &&
        iree_string_view_is_empty(name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm form for descriptor %" PRIu32
                              " has an empty immediate name",
                              descriptor_index);
    }
    if (!iree_string_view_is_empty(name)) {
      for (uint16_t j = 0; j < i; ++j) {
        const loom_low_asm_immediate_t* previous =
            &descriptor_set->asm_immediates[asm_form->immediate_start + j];
        iree_string_view_t previous_name = iree_string_view_empty();
        IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
            descriptor_set, previous->name_string_offset, /*allow_none=*/true,
            &previous_name));
        if (iree_string_view_equal(previous_name, name)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "low asm form for descriptor %" PRIu32
                                  " uses immediate name '%.*s' more than once",
                                  descriptor_index, (int)name.size, name.data);
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_asm_form(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t asm_form_index,
    iree_string_view_t* out_mnemonic) {
  const loom_low_asm_form_t* asm_form =
      &descriptor_set->asm_forms[asm_form_index];
  iree_string_view_t mnemonic = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_verify_non_empty_required_string(
      descriptor_set, asm_form->mnemonic_string_offset, "asm_form.mnemonic",
      &mnemonic));
  if (asm_form->descriptor_ordinal >= descriptor_set->descriptor_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low asm form %" PRIu32
                            " references descriptor ordinal %" PRIu32
                            " but only %" PRIu32 " descriptors exist",
                            asm_form_index, asm_form->descriptor_ordinal,
                            descriptor_set->descriptor_count);
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[asm_form->descriptor_ordinal];
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      asm_form->result_operand_index_start,
      asm_form->result_operand_index_count,
      descriptor_set->asm_operand_index_count, "asm_operand_indices"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      asm_form->operand_index_start, asm_form->operand_index_count,
      descriptor_set->asm_operand_index_count, "asm_operand_indices"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      asm_form->immediate_start, asm_form->immediate_count,
      descriptor_set->asm_immediate_count, "asm_immediates"));
  IREE_RETURN_IF_ERROR(loom_low_verify_asm_operand_indices(
      descriptor_set, descriptor, asm_form->descriptor_ordinal,
      asm_form->result_operand_index_start,
      asm_form->result_operand_index_count, /*expect_result=*/true));
  IREE_RETURN_IF_ERROR(loom_low_verify_asm_operand_indices(
      descriptor_set, descriptor, asm_form->descriptor_ordinal,
      asm_form->operand_index_start, asm_form->operand_index_count,
      /*expect_result=*/false));
  IREE_RETURN_IF_ERROR(loom_low_verify_asm_immediates(
      descriptor_set, asm_form, descriptor, asm_form->descriptor_ordinal));
  if (out_mnemonic != NULL) *out_mnemonic = mnemonic;
  return iree_ok_status();
}

static iree_status_t loom_low_verify_asm_forms(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set->asm_form_count == 0) {
    if (descriptor_set->asm_operand_index_count != 0 ||
        descriptor_set->asm_immediate_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor set has asm operand or immediate rows but no asm "
          "forms");
    }
    return iree_ok_status();
  }
  iree_string_view_t previous_mnemonic = iree_string_view_empty();
  for (uint32_t i = 0; i < descriptor_set->asm_form_count; ++i) {
    iree_string_view_t mnemonic = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        loom_low_verify_asm_form(descriptor_set, i, &mnemonic));
    if (i > 0 && iree_string_view_compare(previous_mnemonic, mnemonic) >= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low asm forms are not strictly sorted near "
                              "mnemonic '%.*s'",
                              (int)mnemonic.size, mnemonic.data);
    }
    previous_mnemonic = mnemonic;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_binary_constraint(
    const loom_low_constraint_t* constraint, const char* constraint_name) {
  if (constraint->rhs_operand_index == LOOM_LOW_ID_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low %s constraint requires an rhs operand",
                            constraint_name);
  }
  if (constraint->lhs_operand_index == constraint->rhs_operand_index) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low %s constraint cannot reference the same operand twice",
        constraint_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_constraints(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->lhs_operand_index >= descriptor->operand_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low descriptor constraint lhs operand %" PRIu16
                              " exceeds descriptor operand count %" PRIu16,
                              constraint->lhs_operand_index,
                              descriptor->operand_count);
    }
    if (constraint->rhs_operand_index != LOOM_LOW_ID_NONE &&
        constraint->rhs_operand_index >= descriptor->operand_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low descriptor constraint rhs operand %" PRIu16
                              " exceeds descriptor operand count %" PRIu16,
                              constraint->rhs_operand_index,
                              descriptor->operand_count);
    }
    const loom_low_operand_t* lhs =
        &descriptor_set->operands[descriptor->operand_start +
                                  constraint->lhs_operand_index];
    const loom_low_operand_t* rhs = NULL;
    if (constraint->rhs_operand_index != LOOM_LOW_ID_NONE) {
      rhs = &descriptor_set->operands[descriptor->operand_start +
                                      constraint->rhs_operand_index];
    }
    switch (constraint->kind) {
      case LOOM_LOW_CONSTRAINT_KIND_TIED: {
        IREE_RETURN_IF_ERROR(
            loom_low_verify_binary_constraint(constraint, "tied"));
        if (lhs->role != LOOM_LOW_OPERAND_ROLE_RESULT ||
            !loom_low_operand_role_is_packet_operand(rhs->role)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low tied constraint requires a result lhs and packet operand "
              "rhs");
        }
        break;
      }
      case LOOM_LOW_CONSTRAINT_KIND_COMMUTABLE: {
        IREE_RETURN_IF_ERROR(
            loom_low_verify_binary_constraint(constraint, "commutable"));
        if (lhs->role != LOOM_LOW_OPERAND_ROLE_OPERAND ||
            rhs->role != LOOM_LOW_OPERAND_ROLE_OPERAND) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low commutable constraint requires two packet operand rows");
        }
        break;
      }
      case LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE: {
        IREE_RETURN_IF_ERROR(
            loom_low_verify_binary_constraint(constraint, "destructive"));
        if (lhs->role != LOOM_LOW_OPERAND_ROLE_RESULT ||
            !loom_low_operand_role_is_packet_operand(rhs->role)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low destructive constraint requires a result lhs and packet "
              "operand rhs");
        }
        break;
      }
      case LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER:
        if (constraint->rhs_operand_index != LOOM_LOW_ID_NONE ||
            lhs->role != LOOM_LOW_OPERAND_ROLE_RESULT) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low early-clobber constraint requires one result operand");
        }
        break;
      case LOOM_LOW_CONSTRAINT_KIND_REMATERIALIZABLE:
      case LOOM_LOW_CONSTRAINT_KIND_FOLDABLE:
        if (constraint->rhs_operand_index != LOOM_LOW_ID_NONE ||
            lhs->role != LOOM_LOW_OPERAND_ROLE_RESULT) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "low unary descriptor constraint requires one result operand");
        }
        break;
      default:
        break;
    }
  }
  return iree_ok_status();
}

static bool loom_low_immediate_kind_is_valid(loom_low_immediate_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED:
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED:
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL:
    case LOOM_LOW_IMMEDIATE_KIND_ENUM:
      return true;
    default:
      return false;
  }
}

static bool loom_low_effect_kind_is_valid(loom_low_effect_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
    case LOOM_LOW_EFFECT_KIND_WRITE:
    case LOOM_LOW_EFFECT_KIND_CALL:
    case LOOM_LOW_EFFECT_KIND_BARRIER:
    case LOOM_LOW_EFFECT_KIND_COUNTER:
    case LOOM_LOW_EFFECT_KIND_CONVERGENT:
    case LOOM_LOW_EFFECT_KIND_CONTROL:
      return true;
    default:
      return false;
  }
}

static bool loom_low_memory_space_is_valid(loom_low_memory_space_t space) {
  switch (space) {
    case LOOM_LOW_MEMORY_SPACE_NONE:
    case LOOM_LOW_MEMORY_SPACE_GENERIC:
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
    case LOOM_LOW_MEMORY_SPACE_STACK:
    case LOOM_LOW_MEMORY_SPACE_VM_REF:
    case LOOM_LOW_MEMORY_SPACE_WASM_MEMORY:
      return true;
    default:
      return false;
  }
}

bool loom_low_spill_slot_space_is_valid(loom_low_spill_slot_space_t space) {
  switch (space) {
    case LOOM_LOW_SPILL_SLOT_SPACE_STACK:
    case LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH:
    case LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE:
    case LOOM_LOW_SPILL_SLOT_SPACE_LDS:
      return true;
    default:
      return false;
  }
}

static bool loom_low_constraint_kind_is_valid(loom_low_constraint_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_CONSTRAINT_KIND_TIED:
    case LOOM_LOW_CONSTRAINT_KIND_COMMUTABLE:
    case LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE:
    case LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER:
    case LOOM_LOW_CONSTRAINT_KIND_REMATERIALIZABLE:
    case LOOM_LOW_CONSTRAINT_KIND_FOLDABLE:
      return true;
    default:
      return false;
  }
}

static bool loom_low_latency_kind_is_valid(loom_low_latency_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_LATENCY_KIND_EXACT:
    case LOOM_LOW_LATENCY_KIND_ESTIMATE:
    case LOOM_LOW_LATENCY_KIND_VARIABLE:
      return true;
    default:
      return false;
  }
}

static bool loom_low_model_quality_is_valid(loom_low_model_quality_t quality) {
  switch (quality) {
    case LOOM_LOW_MODEL_QUALITY_EXACT:
    case LOOM_LOW_MODEL_QUALITY_CALIBRATED:
    case LOOM_LOW_MODEL_QUALITY_ESTIMATED:
    case LOOM_LOW_MODEL_QUALITY_FALLBACK:
      return true;
    default:
      return false;
  }
}

static bool loom_low_resource_kind_is_valid(loom_low_resource_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_KIND_SCALAR_ALU:
    case LOOM_LOW_RESOURCE_KIND_VECTOR_ALU:
    case LOOM_LOW_RESOURCE_KIND_MATRIX:
    case LOOM_LOW_RESOURCE_KIND_LOAD:
    case LOOM_LOW_RESOURCE_KIND_STORE:
    case LOOM_LOW_RESOURCE_KIND_CONTROL:
    case LOOM_LOW_RESOURCE_KIND_ADDRESS:
      return true;
    default:
      return false;
  }
}

static bool loom_low_hazard_kind_is_valid(loom_low_hazard_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_HAZARD_KIND_MIN_DISTANCE:
    case LOOM_LOW_HAZARD_KIND_WAIT_COUNTER:
    case LOOM_LOW_HAZARD_KIND_BYPASS:
    case LOOM_LOW_HAZARD_KIND_FUSION:
      return true;
    default:
      return false;
  }
}

static bool loom_low_hazard_reference_kind_is_valid(
    loom_low_hazard_reference_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE:
    case LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER:
    case LOOM_LOW_HAZARD_REFERENCE_KIND_TARGET:
      return true;
    default:
      return false;
  }
}

static loom_low_schedule_class_flags_t
loom_low_schedule_flags_required_for_effect(loom_low_effect_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
      return LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD;
    case LOOM_LOW_EFFECT_KIND_WRITE:
      return LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_STORE;
    case LOOM_LOW_EFFECT_KIND_CALL:
      return LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_CALL;
    case LOOM_LOW_EFFECT_KIND_CONTROL:
      return LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL;
    default:
      return 0;
  }
}

static bool loom_low_effect_kind_requires_side_effecting_flag(
    loom_low_effect_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_EFFECT_KIND_WRITE:
    case LOOM_LOW_EFFECT_KIND_CALL:
    case LOOM_LOW_EFFECT_KIND_BARRIER:
    case LOOM_LOW_EFFECT_KIND_COUNTER:
    case LOOM_LOW_EFFECT_KIND_CONTROL:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_low_verify_descriptor_encoding_contract(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  const bool is_pseudo =
      iree_all_bits_set(descriptor->flags, LOOM_LOW_DESCRIPTOR_FLAG_PSEUDO);
  if (is_pseudo && descriptor->encoding_id != LOOM_LOW_ID_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " is pseudo but has target encoding id %" PRIu16,
                            descriptor_index, descriptor->encoding_id);
  }
  if (is_pseudo && descriptor->encoding_format_id != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low descriptor %" PRIu32
        " is pseudo but has target encoding format id %" PRIu16,
        descriptor_index, descriptor->encoding_format_id);
  }
  if (!is_pseudo && descriptor->encoding_id == LOOM_LOW_ID_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " has no target encoding id but is not pseudo",
                            descriptor_index);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      descriptor->encoding_field_value_start,
      descriptor->encoding_field_value_count,
      descriptor_set->encoding_field_value_count, "encoding_field_values"));
  for (uint16_t i = 0; i < descriptor->encoding_field_value_count; ++i) {
    const uint32_t field_value_index =
        descriptor->encoding_field_value_start + i;
    const loom_low_encoding_field_value_t* field_value =
        &descriptor_set->encoding_field_values[field_value_index];
    if (field_value->encoding_field_id == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor %" PRIu32
                              " fixed encoding field value %" PRIu32
                              " has field id zero",
                              descriptor_index, field_value_index);
    }
    for (uint16_t j = 0; j < i; ++j) {
      const loom_low_encoding_field_value_t* previous =
          &descriptor_set
               ->encoding_field_values[descriptor->encoding_field_value_start +
                                       j];
      if (previous->encoding_field_id == field_value->encoding_field_id) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low descriptor %" PRIu32
                                " repeats fixed encoding field id %" PRIu16,
                                descriptor_index,
                                field_value->encoding_field_id);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_effect_contract(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  const bool is_side_effecting = iree_all_bits_set(
      descriptor->flags, LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING);
  const bool is_terminator =
      iree_all_bits_set(descriptor->flags, LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR);
  const bool is_dead_removable = iree_all_bits_set(
      descriptor->flags, LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE);
  if (is_side_effecting && is_dead_removable) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " is both side-effecting and dead-removable",
                            descriptor_index);
  }
  if (is_terminator && !is_side_effecting) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " is a terminator but is not side-effecting",
                            descriptor_index);
  }
  if (descriptor->effect_count == 0 && is_side_effecting) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " is side-effecting but has no effect rows",
                            descriptor_index);
  }
  if (descriptor->effect_count == 0) {
    return iree_ok_status();
  }

  bool has_control_effect = false;
  loom_low_schedule_class_flags_t required_schedule_flags = 0;
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const uint32_t effect_index = descriptor->effect_start + i;
    const loom_low_effect_t* effect = &descriptor_set->effects[effect_index];
    if (effect->kind == LOOM_LOW_EFFECT_KIND_READ ||
        effect->kind == LOOM_LOW_EFFECT_KIND_WRITE) {
      if (effect->memory_space == LOOM_LOW_MEMORY_SPACE_NONE) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low descriptor %" PRIu32 " effect %" PRIu32
                                " reads or writes no memory space",
                                descriptor_index, effect_index);
      }
    }
    if (effect->kind == LOOM_LOW_EFFECT_KIND_CONTROL) {
      has_control_effect = true;
    }
    if (!is_side_effecting &&
        loom_low_effect_kind_requires_side_effecting_flag(effect->kind)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor %" PRIu32 " effect %" PRIu32
                              " requires the side-effecting descriptor flag",
                              descriptor_index, effect_index);
    }
    required_schedule_flags |=
        loom_low_schedule_flags_required_for_effect(effect->kind);
  }

  if (is_terminator && !has_control_effect) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " is a terminator but has no control effect",
                            descriptor_index);
  }
  if (has_control_effect && !is_terminator) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " has a control effect but is not a terminator",
                            descriptor_index);
  }
  if (descriptor->schedule_class_id != LOOM_LOW_SCHEDULE_CLASS_NONE &&
      required_schedule_flags != 0) {
    const loom_low_schedule_class_t* schedule_class =
        &descriptor_set->schedule_classes[descriptor->schedule_class_id];
    const loom_low_schedule_class_flags_t missing_flags =
        required_schedule_flags & ~schedule_class->flags;
    if (missing_flags != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor %" PRIu32
                              " effect rows require schedule flags 0x%x but "
                              "schedule class %" PRIu16 " is missing 0x%x",
                              descriptor_index, required_schedule_flags,
                              descriptor->schedule_class_id, missing_flags);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_canonical_asm_form(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    return iree_ok_status();
  }
  if (descriptor->canonical_asm_form_ordinal >=
      descriptor_set->asm_form_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low descriptor %" PRIu32 " canonical asm form ordinal %" PRIu32
        " exceeds asm form count %" PRIu32,
        descriptor_index, descriptor->canonical_asm_form_ordinal,
        descriptor_set->asm_form_count);
  }
  const loom_low_asm_form_t* asm_form =
      &descriptor_set->asm_forms[descriptor->canonical_asm_form_ordinal];
  if (asm_form->descriptor_ordinal != descriptor_index) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low descriptor %" PRIu32 " canonical asm form ordinal %" PRIu32
        " belongs to descriptor %" PRIu32,
        descriptor_index, descriptor->canonical_asm_form_ordinal,
        asm_form->descriptor_ordinal);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor_operand_roles(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint32_t descriptor_index) {
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const uint32_t operand_index = descriptor->operand_start + i;
    const loom_low_operand_t* operand =
        &descriptor_set->operands[operand_index];
    if (operand->role == LOOM_LOW_OPERAND_ROLE_OPERAND_RESULT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low descriptor %" PRIu32 " operand row %" PRIu16
          " uses OPERAND_RESULT, but SSA low packets require separate result "
          "and operand rows plus an explicit constraint",
          descriptor_index, i);
    }
    if (i < descriptor->result_count) {
      if (operand->role != LOOM_LOW_OPERAND_ROLE_RESULT) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "low descriptor %" PRIu32 " result row %" PRIu16
                                " has non-result role %u",
                                descriptor_index, i, (unsigned)operand->role);
      }
      continue;
    }
    if (operand->role == LOOM_LOW_OPERAND_ROLE_RESULT) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor %" PRIu32 " operand row %" PRIu16
                              " appears after the result prefix but has "
                              "result role",
                              descriptor_index, i);
    }
    if (operand->role == LOOM_LOW_OPERAND_ROLE_IMPLICIT &&
        !iree_all_bits_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low descriptor %" PRIu32
                              " implicit operand row %" PRIu16
                              " must set LOOM_LOW_OPERAND_FLAG_IMPLICIT",
                              descriptor_index, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_index) {
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_index];
  IREE_RETURN_IF_ERROR(
      loom_low_verify_known_flags(descriptor->flags,
                                  LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                                      LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR |
                                      LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE |
                                      LOOM_LOW_DESCRIPTOR_FLAG_PSEUDO,
                                  "descriptor", descriptor_index));
  iree_string_view_t descriptor_key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_low_verify_non_empty_required_string(
      descriptor_set, descriptor->key_string_offset, "descriptor.key",
      &descriptor_key));
  if (descriptor->stable_id == LOOM_LOW_DESCRIPTOR_ID_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " has no stable descriptor ID",
                            descriptor_index);
  }
  const uint64_t expected_stable_id =
      loom_low_descriptor_stable_id_from_key(descriptor_key);
  if (descriptor->stable_id != expected_stable_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " key '%.*s' stable ID "
                            "0x%016" PRIx64 " should be 0x%016" PRIx64,
                            descriptor_index, (int)descriptor_key.size,
                            descriptor_key.data, descriptor->stable_id,
                            expected_stable_id);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_string(
      descriptor_set, descriptor->mnemonic_string_offset,
      "descriptor.mnemonic"));
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_string(
      descriptor_set, descriptor->semantic_tag_string_offset,
      "descriptor.semantic_tag"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      descriptor->feature_mask_word_start, descriptor->feature_mask_word_count,
      descriptor_set->feature_mask_word_count, "feature_mask_words"));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_span(descriptor->operand_start, descriptor->operand_count,
                           descriptor_set->operand_count, "operands"));
  if (descriptor->result_count > descriptor->operand_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low descriptor %" PRIu32 " has result count %" PRIu16
        " greater than operand count %" PRIu16,
        descriptor_index, descriptor->result_count, descriptor->operand_count);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      descriptor->immediate_start, descriptor->immediate_count,
      descriptor_set->immediate_count, "immediates"));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_span(descriptor->effect_start, descriptor->effect_count,
                           descriptor_set->effect_count, "effects"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      descriptor->constraint_start, descriptor->constraint_count,
      descriptor_set->constraint_count, "constraints"));
  if (descriptor->schedule_class_id != LOOM_LOW_SCHEDULE_CLASS_NONE &&
      descriptor->schedule_class_id >= descriptor_set->schedule_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor %" PRIu32
                            " references schedule class %" PRIu16
                            " but only %" PRIu32 " classes exist",
                            descriptor_index, descriptor->schedule_class_id,
                            descriptor_set->schedule_class_count);
  }
  if (descriptor->schedule_class_id == LOOM_LOW_SCHEDULE_CLASS_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor %" PRIu32
                            " has no schedule class; use an explicit fallback "
                            "schedule class for unknown scheduling",
                            descriptor_index);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_operand_roles(
      descriptor_set, descriptor, descriptor_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_encoding_contract(
      descriptor_set, descriptor, descriptor_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_effect_contract(
      descriptor_set, descriptor, descriptor_index));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_descriptor_constraints(descriptor_set, descriptor));
  return loom_low_verify_descriptor_canonical_asm_form(
      descriptor_set, descriptor, descriptor_index);
}

static iree_status_t loom_low_verify_operand(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t operand_index) {
  const loom_low_operand_t* operand = &descriptor_set->operands[operand_index];
  IREE_RETURN_IF_ERROR(loom_low_verify_known_flags(
      operand->flags,
      LOOM_LOW_OPERAND_FLAG_IMPLICIT | LOOM_LOW_OPERAND_FLAG_TIED |
          LOOM_LOW_OPERAND_FLAG_EARLY_CLOBBER | LOOM_LOW_OPERAND_FLAG_OPTIONAL,
      "operand", operand_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, operand->field_name_string_offset, "operand.field_name"));
  if (operand->role == LOOM_LOW_OPERAND_ROLE_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low operand %" PRIu32 " has unknown role",
                            operand_index);
  }
  if (!loom_low_operand_role_is_valid(operand->role)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low operand %" PRIu32 " has invalid role %u",
                            operand_index, (unsigned)operand->role);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      operand->reg_class_alt_start, operand->reg_class_alt_count,
      descriptor_set->reg_class_alt_count, "reg_class_alts"));
  return iree_ok_status();
}

static iree_status_t loom_low_verify_immediate(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t immediate_index) {
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_index];
  IREE_RETURN_IF_ERROR(loom_low_verify_known_flags(
      immediate->flags,
      LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC | LOOM_LOW_IMMEDIATE_FLAG_RELATIVE,
      "immediate", immediate_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, immediate->field_name_string_offset,
      "immediate.field_name"));
  if (immediate->kind == LOOM_LOW_IMMEDIATE_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low immediate %" PRIu32 " has unknown kind",
                            immediate_index);
  }
  if (!loom_low_immediate_kind_is_valid(immediate->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low immediate %" PRIu32 " has invalid kind %u",
                            immediate_index, (unsigned)immediate->kind);
  }
  if (immediate->kind == LOOM_LOW_IMMEDIATE_KIND_ENUM) {
    if (immediate->enum_domain_id == LOOM_LOW_ENUM_DOMAIN_NONE) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low enum immediate %" PRIu32 " has no enum domain", immediate_index);
    }
    if (immediate->enum_domain_id >= descriptor_set->enum_domain_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low enum immediate %" PRIu32
                              " references enum domain %" PRIu16
                              " but only %" PRIu32 " domains exist",
                              immediate_index, immediate->enum_domain_id,
                              descriptor_set->enum_domain_count);
    }
  } else if (immediate->enum_domain_id != LOOM_LOW_ENUM_DOMAIN_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low non-enum immediate %" PRIu32
                            " references enum domain %" PRIu16,
                            immediate_index, immediate->enum_domain_id);
  }
  if (immediate->bit_width > 64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low immediate %" PRIu32
                            " has unsupported bit width %" PRIu16,
                            immediate_index, immediate->bit_width);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_enum_domain(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t enum_domain_index) {
  const loom_low_enum_domain_t* domain =
      &descriptor_set->enum_domains[enum_domain_index];
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, domain->name_string_offset, "enum_domain.name"));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_span(domain->value_start, domain->value_count,
                           descriptor_set->enum_value_count, "enum_values"));
  if (domain->value_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low enum domain %" PRIu32 " has no values",
                            enum_domain_index);
  }
  iree_string_view_t previous_token = iree_string_view_empty();
  for (uint16_t i = 0; i < domain->value_count; ++i) {
    const uint32_t value_index = domain->value_start + i;
    const loom_low_enum_value_t* value =
        &descriptor_set->enum_values[value_index];
    iree_string_view_t token = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
        descriptor_set, value->token_string_offset, &token));
    if (i != 0 && iree_string_view_compare(previous_token, token) >= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "low enum domain %" PRIu32
                              " value tokens are not strictly sorted",
                              enum_domain_index);
    }
    previous_token = token;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_enum_value(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t enum_value_index) {
  const loom_low_enum_value_t* value =
      &descriptor_set->enum_values[enum_value_index];
  return loom_low_verify_required_string(
      descriptor_set, value->token_string_offset, "enum_value.token");
}

static iree_status_t loom_low_verify_reg_class(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t reg_class_index) {
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[reg_class_index];
  IREE_RETURN_IF_ERROR(loom_low_verify_known_flags(
      reg_class->flags,
      LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY | LOOM_LOW_REG_CLASS_FLAG_PHYSICAL |
          LOOM_LOW_REG_CLASS_FLAG_REFERENCE,
      "register class", reg_class_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, reg_class->name_string_offset, "reg_class.name"));
  if (reg_class->alloc_unit_bits == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low register class %" PRIu32
                            " has zero allocation-unit width",
                            reg_class_index);
  }
  if (!loom_low_spill_slot_space_is_valid(
          (loom_low_spill_slot_space_t)reg_class->spill_slot_space)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low register class %" PRIu32 " has unknown spill slot space %u",
        reg_class_index, (unsigned)reg_class->spill_slot_space);
  }
  const bool is_virtual_only =
      iree_all_bits_set(reg_class->flags, LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY);
  const bool is_physical =
      iree_all_bits_set(reg_class->flags, LOOM_LOW_REG_CLASS_FLAG_PHYSICAL);
  if (is_virtual_only == is_physical) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low register class %" PRIu32
        " must name exactly one virtual or physical storage kind",
        reg_class_index);
  }
  if (is_virtual_only && reg_class->physical_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low virtual register class %" PRIu32
                            " has non-zero physical register count %" PRIu16,
                            reg_class_index, reg_class->physical_count);
  }
  if (is_physical && reg_class->physical_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low physical register class %" PRIu32
                            " has zero physical register count",
                            reg_class_index);
  }
  if (reg_class->spill_class_id != LOOM_LOW_REG_CLASS_NONE &&
      reg_class->spill_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low register class %" PRIu32
                            " references spill class %" PRIu16
                            " but only %" PRIu32 " classes exist",
                            reg_class_index, reg_class->spill_class_id,
                            descriptor_set->reg_class_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_reg_class_alt(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t alt_index) {
  const loom_low_reg_class_alt_t* alt =
      &descriptor_set->reg_class_alts[alt_index];
  IREE_RETURN_IF_ERROR(
      loom_low_verify_known_flags(alt->flags,
                                  LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED |
                                      LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE |
                                      LOOM_LOW_REG_CLASS_ALT_FLAG_PHYSICAL_ONLY,
                                  "register-class alternative", alt_index));
  const bool is_immediate =
      (alt->flags & LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE) != 0;
  if (is_immediate && alt->reg_class_id != LOOM_LOW_REG_CLASS_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low register-class alternative %" PRIu32
        " marks an immediate but also references register class %" PRIu16,
        alt_index, alt->reg_class_id);
  }
  if (!is_immediate && alt->reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low register-class alternative %" PRIu32
                            " references register class %" PRIu16
                            " but only %" PRIu32 " classes exist",
                            alt_index, alt->reg_class_id,
                            descriptor_set->reg_class_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_schedule_class(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t schedule_class_index) {
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[schedule_class_index];
  IREE_RETURN_IF_ERROR(
      loom_low_verify_known_flags(schedule_class->flags,
                                  LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD |
                                      LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_STORE |
                                      LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_CALL |
                                      LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL,
                                  "schedule class", schedule_class_index));
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, schedule_class->name_string_offset,
      "schedule_class.name"));
  if (schedule_class->latency_kind == LOOM_LOW_LATENCY_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low schedule class %" PRIu32
                            " has unknown latency kind",
                            schedule_class_index);
  }
  if (!loom_low_latency_kind_is_valid(schedule_class->latency_kind)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low schedule class %" PRIu32 " has invalid latency kind %u",
        schedule_class_index, (unsigned)schedule_class->latency_kind);
  }
  if (schedule_class->model_quality == LOOM_LOW_MODEL_QUALITY_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low schedule class %" PRIu32
                            " has unknown model quality",
                            schedule_class_index);
  }
  if (!loom_low_model_quality_is_valid(schedule_class->model_quality)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low schedule class %" PRIu32 " has invalid model quality %u",
        schedule_class_index, (unsigned)schedule_class->model_quality);
  }
  if (schedule_class->model_quality == LOOM_LOW_MODEL_QUALITY_EXACT &&
      schedule_class->latency_kind != LOOM_LOW_LATENCY_KIND_EXACT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low schedule class %" PRIu32
                            " has exact model quality without exact latency",
                            schedule_class_index);
  }
  if (schedule_class->model_quality == LOOM_LOW_MODEL_QUALITY_FALLBACK &&
      schedule_class->latency_kind != LOOM_LOW_LATENCY_KIND_VARIABLE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low schedule class %" PRIu32
                            " is a fallback model without variable latency",
                            schedule_class_index);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      schedule_class->issue_use_start, schedule_class->issue_use_count,
      descriptor_set->issue_use_count, "issue_uses"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      schedule_class->hazard_start, schedule_class->hazard_count,
      descriptor_set->hazard_count, "hazards"));
  IREE_RETURN_IF_ERROR(loom_low_verify_span(
      schedule_class->pressure_delta_start,
      schedule_class->pressure_delta_count,
      descriptor_set->pressure_delta_count, "pressure_deltas"));
  return iree_ok_status();
}

static iree_status_t loom_low_verify_issue_use(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t issue_use_index) {
  const loom_low_issue_use_t* issue_use =
      &descriptor_set->issue_uses[issue_use_index];
  if (issue_use->resource_id >= descriptor_set->resource_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low issue-use %" PRIu32
                            " references resource %" PRIu16 " but only %" PRIu32
                            " resources exist",
                            issue_use_index, issue_use->resource_id,
                            descriptor_set->resource_count);
  }
  if (issue_use->cycles == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low issue-use %" PRIu32 " has zero occupied cycles", issue_use_index);
  }
  if (issue_use->units == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low issue-use %" PRIu32
                            " consumes zero resource units",
                            issue_use_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_resource(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t resource_index) {
  const loom_low_resource_t* resource =
      &descriptor_set->resources[resource_index];
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, resource->name_string_offset, "resource.name"));
  if (resource->kind == LOOM_LOW_RESOURCE_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low resource %" PRIu32 " has unknown kind",
                            resource_index);
  }
  if (!loom_low_resource_kind_is_valid(resource->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low resource %" PRIu32 " has invalid kind %u",
                            resource_index, (unsigned)resource->kind);
  }
  if (resource->capacity_per_cycle == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low resource %" PRIu32 " has zero capacity per cycle", resource_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_effect(uint32_t effect_index,
                                            const loom_low_effect_t* effect) {
  IREE_RETURN_IF_ERROR(loom_low_verify_known_flags(
      effect->flags,
      LOOM_LOW_EFFECT_FLAG_ORDERED | LOOM_LOW_EFFECT_FLAG_DEPENDENCY, "effect",
      effect_index));
  if (effect->kind == LOOM_LOW_EFFECT_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low effect %" PRIu32 " has unknown kind",
                            effect_index);
  }
  if (!loom_low_effect_kind_is_valid(effect->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low effect %" PRIu32 " has invalid kind %u",
                            effect_index, (unsigned)effect->kind);
  }
  if (!loom_low_memory_space_is_valid(effect->memory_space)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low effect %" PRIu32
                            " has invalid memory space %u",
                            effect_index, (unsigned)effect->memory_space);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_constraint(
    uint32_t constraint_index, const loom_low_constraint_t* constraint) {
  if (constraint->kind == LOOM_LOW_CONSTRAINT_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low constraint %" PRIu32 " has unknown kind",
                            constraint_index);
  }
  if (!loom_low_constraint_kind_is_valid(constraint->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low constraint %" PRIu32 " has invalid kind %u",
                            constraint_index, (unsigned)constraint->kind);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_hazard(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t hazard_index) {
  const loom_low_hazard_t* hazard = &descriptor_set->hazards[hazard_index];
  if (hazard->kind == LOOM_LOW_HAZARD_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low hazard %" PRIu32 " has unknown kind",
                            hazard_index);
  }
  if (!loom_low_hazard_kind_is_valid(hazard->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low hazard %" PRIu32 " has invalid kind %u",
                            hazard_index, (unsigned)hazard->kind);
  }
  if (hazard->reference_kind == LOOM_LOW_HAZARD_REFERENCE_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low hazard %" PRIu32 " has unknown reference kind",
                            hazard_index);
  }
  if (!loom_low_hazard_reference_kind_is_valid(hazard->reference_kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low hazard %" PRIu32
                            " has invalid reference kind %u",
                            hazard_index, (unsigned)hazard->reference_kind);
  }
  if (hazard->reference_kind == LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE &&
      hazard->reference_id >= descriptor_set->resource_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low hazard %" PRIu32 " references resource %" PRIu16
        " but only %" PRIu32 " resources exist",
        hazard_index, hazard->reference_id, descriptor_set->resource_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_pressure_delta(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t pressure_delta_index) {
  const loom_low_pressure_delta_t* pressure_delta =
      &descriptor_set->pressure_deltas[pressure_delta_index];
  if (pressure_delta->reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low pressure delta %" PRIu32
                            " references register class %" PRIu16
                            " but only %" PRIu32 " classes exist",
                            pressure_delta_index, pressure_delta->reg_class_id,
                            descriptor_set->reg_class_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_append_json_string(iree_string_builder_t* builder,
                                                 iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\""));
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const unsigned char c = (unsigned char)value.data[i];
    switch (c) {
      case '"': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\\""));
        break;
      }
      case '\\': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\\\"));
        break;
      }
      case '\n': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\n"));
        break;
      }
      case '\r': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\r"));
        break;
      }
      case '\t': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\t"));
        break;
      }
      default: {
        if (c < 0x20) {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, "\\u%04x", (unsigned)c));
        } else {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
              builder, iree_make_string_view((const char*)&value.data[i], 1)));
        }
        break;
      }
    }
  }
  return iree_string_builder_append_cstring(builder, "\"");
}

static iree_status_t loom_low_descriptor_set_append_string_field(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder, const char* field_name,
    loom_bstring_table_offset_t string_offset) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_set_string(descriptor_set, string_offset, &value));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, "\"%s\":", field_name));
  return loom_low_append_json_string(builder, value);
}

iree_string_view_t loom_low_operand_role_name(loom_low_operand_role_t role) {
  switch (role) {
    case LOOM_LOW_OPERAND_ROLE_RESULT:
      return IREE_SV("result");
    case LOOM_LOW_OPERAND_ROLE_OPERAND:
      return IREE_SV("operand");
    case LOOM_LOW_OPERAND_ROLE_OPERAND_RESULT:
      return IREE_SV("operand_result");
    case LOOM_LOW_OPERAND_ROLE_PREDICATE:
      return IREE_SV("predicate");
    case LOOM_LOW_OPERAND_ROLE_RESOURCE:
      return IREE_SV("resource");
    case LOOM_LOW_OPERAND_ROLE_IMPLICIT:
      return IREE_SV("implicit");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_immediate_kind_name(
    loom_low_immediate_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_IMMEDIATE_KIND_SIGNED:
      return IREE_SV("signed");
    case LOOM_LOW_IMMEDIATE_KIND_UNSIGNED:
      return IREE_SV("unsigned");
    case LOOM_LOW_IMMEDIATE_KIND_ORDINAL:
      return IREE_SV("ordinal");
    case LOOM_LOW_IMMEDIATE_KIND_ENUM:
      return IREE_SV("enum");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_effect_kind_name(loom_low_effect_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
      return IREE_SV("read");
    case LOOM_LOW_EFFECT_KIND_WRITE:
      return IREE_SV("write");
    case LOOM_LOW_EFFECT_KIND_CALL:
      return IREE_SV("call");
    case LOOM_LOW_EFFECT_KIND_BARRIER:
      return IREE_SV("barrier");
    case LOOM_LOW_EFFECT_KIND_COUNTER:
      return IREE_SV("counter");
    case LOOM_LOW_EFFECT_KIND_CONVERGENT:
      return IREE_SV("convergent");
    case LOOM_LOW_EFFECT_KIND_CONTROL:
      return IREE_SV("control");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_memory_space_name(
    loom_low_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_LOW_MEMORY_SPACE_NONE:
      return IREE_SV("none");
    case LOOM_LOW_MEMORY_SPACE_GENERIC:
      return IREE_SV("generic");
    case LOOM_LOW_MEMORY_SPACE_GLOBAL:
      return IREE_SV("global");
    case LOOM_LOW_MEMORY_SPACE_WORKGROUP:
      return IREE_SV("workgroup");
    case LOOM_LOW_MEMORY_SPACE_STACK:
      return IREE_SV("stack");
    case LOOM_LOW_MEMORY_SPACE_VM_REF:
      return IREE_SV("vm_ref");
    case LOOM_LOW_MEMORY_SPACE_WASM_MEMORY:
      return IREE_SV("wasm_memory");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_spill_slot_space_name(
    loom_low_spill_slot_space_t space) {
  switch (space) {
    case LOOM_LOW_SPILL_SLOT_SPACE_STACK:
      return IREE_SV("stack");
    case LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH:
      return IREE_SV("scratch");
    case LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE:
      return IREE_SV("private");
    case LOOM_LOW_SPILL_SLOT_SPACE_LDS:
      return IREE_SV("lds");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_constraint_kind_name(
    loom_low_constraint_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_CONSTRAINT_KIND_TIED:
      return IREE_SV("tied");
    case LOOM_LOW_CONSTRAINT_KIND_COMMUTABLE:
      return IREE_SV("commutable");
    case LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE:
      return IREE_SV("destructive");
    case LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER:
      return IREE_SV("early_clobber");
    case LOOM_LOW_CONSTRAINT_KIND_REMATERIALIZABLE:
      return IREE_SV("rematerializable");
    case LOOM_LOW_CONSTRAINT_KIND_FOLDABLE:
      return IREE_SV("foldable");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_latency_kind_name(loom_low_latency_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_LATENCY_KIND_EXACT:
      return IREE_SV("exact");
    case LOOM_LOW_LATENCY_KIND_ESTIMATE:
      return IREE_SV("estimate");
    case LOOM_LOW_LATENCY_KIND_VARIABLE:
      return IREE_SV("variable");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_model_quality_name(
    loom_low_model_quality_t quality) {
  switch (quality) {
    case LOOM_LOW_MODEL_QUALITY_EXACT:
      return IREE_SV("exact");
    case LOOM_LOW_MODEL_QUALITY_CALIBRATED:
      return IREE_SV("calibrated");
    case LOOM_LOW_MODEL_QUALITY_ESTIMATED:
      return IREE_SV("estimated");
    case LOOM_LOW_MODEL_QUALITY_FALLBACK:
      return IREE_SV("fallback");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_resource_kind_name(loom_low_resource_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_KIND_SCALAR_ALU:
      return IREE_SV("scalar_alu");
    case LOOM_LOW_RESOURCE_KIND_VECTOR_ALU:
      return IREE_SV("vector_alu");
    case LOOM_LOW_RESOURCE_KIND_MATRIX:
      return IREE_SV("matrix");
    case LOOM_LOW_RESOURCE_KIND_LOAD:
      return IREE_SV("load");
    case LOOM_LOW_RESOURCE_KIND_STORE:
      return IREE_SV("store");
    case LOOM_LOW_RESOURCE_KIND_CONTROL:
      return IREE_SV("control");
    case LOOM_LOW_RESOURCE_KIND_ADDRESS:
      return IREE_SV("address");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_hazard_kind_name(loom_low_hazard_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_HAZARD_KIND_MIN_DISTANCE:
      return IREE_SV("min_distance");
    case LOOM_LOW_HAZARD_KIND_WAIT_COUNTER:
      return IREE_SV("wait_counter");
    case LOOM_LOW_HAZARD_KIND_BYPASS:
      return IREE_SV("bypass");
    case LOOM_LOW_HAZARD_KIND_FUSION:
      return IREE_SV("fusion");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_hazard_reference_kind_name(
    loom_low_hazard_reference_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE:
      return IREE_SV("resource");
    case LOOM_LOW_HAZARD_REFERENCE_KIND_COUNTER:
      return IREE_SV("counter");
    case LOOM_LOW_HAZARD_REFERENCE_KIND_TARGET:
      return IREE_SV("target");
    default:
      return IREE_SV("unknown");
  }
}

typedef struct loom_low_manifest_flag_name_t {
  uint16_t bit;
  const char* name;
} loom_low_manifest_flag_name_t;

static const loom_low_manifest_flag_name_t kLoomLowRegClassFlagNames[] = {
    {LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY, "virtual_only"},
    {LOOM_LOW_REG_CLASS_FLAG_PHYSICAL, "physical"},
    {LOOM_LOW_REG_CLASS_FLAG_REFERENCE, "reference"},
};

static const loom_low_manifest_flag_name_t kLoomLowRegClassAltFlagNames[] = {
    {LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED, "preferred"},
    {LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE, "immediate"},
    {LOOM_LOW_REG_CLASS_ALT_FLAG_PHYSICAL_ONLY, "physical_only"},
};

static const loom_low_manifest_flag_name_t kLoomLowOperandFlagNames[] = {
    {LOOM_LOW_OPERAND_FLAG_IMPLICIT, "implicit"},
    {LOOM_LOW_OPERAND_FLAG_TIED, "tied"},
    {LOOM_LOW_OPERAND_FLAG_EARLY_CLOBBER, "early_clobber"},
    {LOOM_LOW_OPERAND_FLAG_OPTIONAL, "optional"},
};

static const loom_low_manifest_flag_name_t kLoomLowImmediateFlagNames[] = {
    {LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC, "symbolic"},
    {LOOM_LOW_IMMEDIATE_FLAG_RELATIVE, "relative"},
};

static const loom_low_manifest_flag_name_t kLoomLowEffectFlagNames[] = {
    {LOOM_LOW_EFFECT_FLAG_ORDERED, "ordered"},
    {LOOM_LOW_EFFECT_FLAG_DEPENDENCY, "dependency"},
};

static const loom_low_manifest_flag_name_t kLoomLowScheduleClassFlagNames[] = {
    {LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD, "may_load"},
    {LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_STORE, "may_store"},
    {LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_CALL, "may_call"},
    {LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL, "control"},
};

static const loom_low_manifest_flag_name_t kLoomLowDescriptorFlagNames[] = {
    {LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, "side_effecting"},
    {LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR, "terminator"},
    {LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE, "dead_removable"},
    {LOOM_LOW_DESCRIPTOR_FLAG_PSEUDO, "pseudo"},
};

static iree_status_t loom_low_append_manifest_flag_names(
    iree_string_builder_t* builder, uint16_t flags,
    const loom_low_manifest_flag_name_t* flag_names,
    iree_host_size_t flag_name_count) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"flag_names\":["));
  bool has_name = false;
  for (iree_host_size_t i = 0; i < flag_name_count; ++i) {
    if (!iree_all_bits_set(flags, flag_names[i].bit)) continue;
    if (has_name) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(loom_low_append_json_string(
        builder, iree_make_cstring_view(flag_names[i].name)));
    has_name = true;
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t loom_low_descriptor_set_append_string_value(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder, loom_bstring_table_offset_t string_offset) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_set_string(descriptor_set, string_offset, &value));
  return loom_low_append_json_string(builder, value);
}

static iree_status_t loom_low_append_named_enum_field(
    iree_string_builder_t* builder, const char* value_field_name,
    const char* name_field_name, uint32_t value, iree_string_view_t name) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\"%s\":%" PRIu32 ",\"%s\":", value_field_name, value,
      name_field_name));
  return loom_low_append_json_string(builder, name);
}

static iree_status_t loom_low_append_resource_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder, uint16_t resource_id) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\"resource\":%" PRIu16 ",\"resource_name\":", resource_id));
  if (resource_id == LOOM_LOW_RESOURCE_NONE ||
      resource_id >= descriptor_set->resource_count) {
    return loom_low_append_json_string(builder, IREE_SV(""));
  }
  return loom_low_descriptor_set_append_string_value(
      descriptor_set, builder,
      descriptor_set->resources[resource_id].name_string_offset);
}

static iree_status_t loom_low_append_hazard_reference(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder, const loom_low_hazard_t* hazard) {
  IREE_RETURN_IF_ERROR(loom_low_append_named_enum_field(
      builder, "reference_kind", "reference_kind_name", hazard->reference_kind,
      loom_low_hazard_reference_kind_name(hazard->reference_kind)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"reference\":%" PRIu16 ",\"reference_name\":", hazard->reference_id));
  if (hazard->reference_kind == LOOM_LOW_HAZARD_REFERENCE_KIND_RESOURCE &&
      hazard->reference_id < descriptor_set->resource_count) {
    return loom_low_descriptor_set_append_string_value(
        descriptor_set, builder,
        descriptor_set->resources[hazard->reference_id].name_string_offset);
  }
  return loom_low_append_json_string(builder, IREE_SV(""));
}

static iree_status_t loom_low_append_reg_class_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder, uint16_t reg_class_id) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\"reg_class\":%" PRIu16 ",\"reg_class_name\":", reg_class_id));
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      reg_class_id >= descriptor_set->reg_class_count) {
    return loom_low_append_json_string(builder, IREE_SV(""));
  }
  return loom_low_descriptor_set_append_string_value(
      descriptor_set, builder,
      descriptor_set->reg_classes[reg_class_id].name_string_offset);
}

static iree_status_t loom_low_append_manifest_reg_classes(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"reg_classes\":["));
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const loom_low_reg_class_t* reg_class = &descriptor_set->reg_classes[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"ordinal\":%" PRIu32 ",\"name\":", i));
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_value(
        descriptor_set, builder, reg_class->name_string_offset));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"target_bank\":%" PRIu16 ",\"flags\":%" PRIu16,
        reg_class->target_bank_id, reg_class->flags));
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_flag_names(
        builder, reg_class->flags, kLoomLowRegClassFlagNames,
        IREE_ARRAYSIZE(kLoomLowRegClassFlagNames)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"alloc_unit_bits\":%" PRIu16 ",\"physical_count\":%" PRIu16
        ",\"alias_set\":%" PRIu16 ",\"spill_class\":%" PRIu16 ",",
        reg_class->alloc_unit_bits, reg_class->physical_count,
        reg_class->alias_set_id, reg_class->spill_class_id));
    IREE_RETURN_IF_ERROR(loom_low_append_named_enum_field(
        builder, "spill_slot_space", "spill_slot_space_name",
        reg_class->spill_slot_space,
        loom_low_spill_slot_space_name(
            (loom_low_spill_slot_space_t)reg_class->spill_slot_space)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t loom_low_append_manifest_resources(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"resources\":["));
  for (uint32_t i = 0; i < descriptor_set->resource_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const loom_low_resource_t* resource = &descriptor_set->resources[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"ordinal\":%" PRIu32 ",\"name\":", i));
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_value(
        descriptor_set, builder, resource->name_string_offset));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"capacity_per_cycle\":%" PRIu16 ",\"flags\":%" PRIu16 ",",
        resource->capacity_per_cycle, resource->flags));
    IREE_RETURN_IF_ERROR(loom_low_append_named_enum_field(
        builder, "kind", "kind_name", resource->kind,
        loom_low_resource_kind_name(resource->kind)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"contention_group\":%" PRIu16 "}",
        resource->contention_group_id));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t loom_low_append_manifest_issue_uses(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder, const loom_low_schedule_class_t* schedule) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"issue_uses\":["));
  for (uint16_t i = 0; i < schedule->issue_use_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const loom_low_issue_use_t* issue_use =
        &descriptor_set->issue_uses[schedule->issue_use_start + i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "{"));
    IREE_RETURN_IF_ERROR(loom_low_append_resource_ref(descriptor_set, builder,
                                                      issue_use->resource_id));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"cycles\":%" PRIu16 ",\"units\":%" PRIu16 ",\"stage\":%" PRIu16 "}",
        issue_use->cycles, issue_use->units, issue_use->stage));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t loom_low_append_manifest_hazard_rows(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder, const loom_low_schedule_class_t* schedule) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"hazard_rows\":["));
  for (uint16_t i = 0; i < schedule->hazard_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const loom_low_hazard_t* hazard =
        &descriptor_set->hazards[schedule->hazard_start + i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"index\":%" PRIu16 ",", i));
    IREE_RETURN_IF_ERROR(loom_low_append_named_enum_field(
        builder, "kind", "kind_name", hazard->kind,
        loom_low_hazard_kind_name(hazard->kind)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    IREE_RETURN_IF_ERROR(
        loom_low_append_hazard_reference(descriptor_set, builder, hazard));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"producer_stage\":%" PRIu16 ",\"consumer_stage\":%" PRIu16
        ",\"distance\":%" PRIu16 ",\"flags\":%" PRIu16 "}",
        hazard->producer_stage, hazard->consumer_stage, hazard->distance,
        hazard->flags));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t loom_low_append_manifest_pressure_delta_rows(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder, const loom_low_schedule_class_t* schedule) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder, ",\"pressure_delta_rows\":["));
  for (uint16_t i = 0; i < schedule->pressure_delta_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const loom_low_pressure_delta_t* pressure_delta =
        &descriptor_set->pressure_deltas[schedule->pressure_delta_start + i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"index\":%" PRIu16 ",", i));
    IREE_RETURN_IF_ERROR(loom_low_append_reg_class_ref(
        descriptor_set, builder, pressure_delta->reg_class_id));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"delta\":%d}", (int)pressure_delta->delta));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t loom_low_append_manifest_schedule_classes(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"schedule_classes\":["));
  for (uint32_t i = 0; i < descriptor_set->schedule_class_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const loom_low_schedule_class_t* schedule =
        &descriptor_set->schedule_classes[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"ordinal\":%" PRIu32 ",\"name\":", i));
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_value(
        descriptor_set, builder, schedule->name_string_offset));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"latency_cycles\":%" PRIu16 ",", schedule->latency_cycles));
    IREE_RETURN_IF_ERROR(loom_low_append_named_enum_field(
        builder, "latency_kind", "latency_kind_name", schedule->latency_kind,
        loom_low_latency_kind_name(schedule->latency_kind)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    IREE_RETURN_IF_ERROR(loom_low_append_named_enum_field(
        builder, "model_quality", "model_quality_name", schedule->model_quality,
        loom_low_model_quality_name(schedule->model_quality)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"flags\":%" PRIu16, schedule->flags));
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_flag_names(
        builder, schedule->flags, kLoomLowScheduleClassFlagNames,
        IREE_ARRAYSIZE(kLoomLowScheduleClassFlagNames)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"hazards\":%" PRIu16 ",\"pressure_deltas\":%" PRIu16,
        schedule->hazard_count, schedule->pressure_delta_count));
    IREE_RETURN_IF_ERROR(
        loom_low_append_manifest_issue_uses(descriptor_set, builder, schedule));
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_hazard_rows(
        descriptor_set, builder, schedule));
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_pressure_delta_rows(
        descriptor_set, builder, schedule));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t loom_low_append_manifest_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder, const loom_low_operand_t* operand,
    uint16_t descriptor_operand_index) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "{\"index\":%" PRIu16 ",\"field\":", descriptor_operand_index));
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_value(
      descriptor_set, builder, operand->field_name_string_offset));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(loom_low_append_named_enum_field(
      builder, "role", "role_name", operand->role,
      loom_low_operand_role_name(operand->role)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ",\"flags\":%" PRIu16, operand->flags));
  IREE_RETURN_IF_ERROR(loom_low_append_manifest_flag_names(
      builder, operand->flags, kLoomLowOperandFlagNames,
      IREE_ARRAYSIZE(kLoomLowOperandFlagNames)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"unit_count\":%" PRIu16 ",\"encoding_field\":%" PRIu16
      ",\"data_format\":%" PRIu16 ",\"read_stage\":%" PRIu16
      ",\"ready_stage\":%" PRIu16 ",\"reg_class_alts\":[",
      operand->unit_count, operand->encoding_field_id, operand->data_format_id,
      operand->read_stage, operand->ready_stage));
  for (uint16_t i = 0; i < operand->reg_class_alt_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[operand->reg_class_alt_start + i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "{"));
    IREE_RETURN_IF_ERROR(loom_low_append_reg_class_ref(descriptor_set, builder,
                                                       alt->reg_class_id));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"flags\":%" PRIu16, alt->flags));
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_flag_names(
        builder, alt->flags, kLoomLowRegClassAltFlagNames,
        IREE_ARRAYSIZE(kLoomLowRegClassAltFlagNames)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  return iree_string_builder_append_cstring(builder, "]}");
}

static iree_status_t loom_low_append_manifest_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder, const loom_low_immediate_t* immediate,
    uint16_t descriptor_immediate_index) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "{\"index\":%" PRIu16 ",\"field\":", descriptor_immediate_index));
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_value(
      descriptor_set, builder, immediate->field_name_string_offset));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(loom_low_append_named_enum_field(
      builder, "kind", "kind_name", immediate->kind,
      loom_low_immediate_kind_name(immediate->kind)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ",\"flags\":%" PRIu16, immediate->flags));
  IREE_RETURN_IF_ERROR(loom_low_append_manifest_flag_names(
      builder, immediate->flags, kLoomLowImmediateFlagNames,
      IREE_ARRAYSIZE(kLoomLowImmediateFlagNames)));
  return iree_string_builder_append_format(
      builder,
      ",\"bit_width\":%" PRIu16 ",\"enum_domain\":%" PRIu16
      ",\"encoding_field\":%" PRIu16 ",\"encoding\":%" PRIu16
      ",\"signed_min\":%" PRId64 ",\"unsigned_max\":%" PRIu64 "}",
      immediate->bit_width, immediate->enum_domain_id,
      immediate->encoding_field_id, immediate->encoding_id,
      immediate->signed_min, immediate->unsigned_max);
}

static iree_status_t loom_low_append_manifest_effect(
    iree_string_builder_t* builder, const loom_low_effect_t* effect,
    uint16_t descriptor_effect_index) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "{\"index\":%" PRIu16 ",", descriptor_effect_index));
  IREE_RETURN_IF_ERROR(loom_low_append_named_enum_field(
      builder, "kind", "kind_name", effect->kind,
      loom_low_effect_kind_name(effect->kind)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(loom_low_append_named_enum_field(
      builder, "memory_space", "memory_space_name", effect->memory_space,
      loom_low_memory_space_name(effect->memory_space)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ",\"scope\":%" PRIu16 ",\"flags\":%" PRIu16, effect->scope_id,
      effect->flags));
  IREE_RETURN_IF_ERROR(loom_low_append_manifest_flag_names(
      builder, effect->flags, kLoomLowEffectFlagNames,
      IREE_ARRAYSIZE(kLoomLowEffectFlagNames)));
  return iree_string_builder_append_format(
      builder, ",\"counter\":%" PRIu16 ",\"width_bits\":%" PRIu16 "}",
      effect->counter_id, effect->width_bits);
}

static iree_status_t loom_low_append_manifest_constraint(
    iree_string_builder_t* builder, const loom_low_constraint_t* constraint,
    uint16_t descriptor_constraint_index) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "{\"index\":%" PRIu16 ",", descriptor_constraint_index));
  IREE_RETURN_IF_ERROR(loom_low_append_named_enum_field(
      builder, "kind", "kind_name", constraint->kind,
      loom_low_constraint_kind_name(constraint->kind)));
  return iree_string_builder_append_format(
      builder,
      ",\"lhs_operand\":%" PRIu16 ",\"rhs_operand\":%" PRIu16
      ",\"flags\":%" PRIu16 "}",
      constraint->lhs_operand_index, constraint->rhs_operand_index,
      constraint->flags);
}

static iree_status_t loom_low_append_manifest_asm_operand_fields(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder, const loom_low_descriptor_t* descriptor,
    uint32_t start, uint16_t count) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "["));
  for (uint16_t i = 0; i < count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const uint16_t operand_index =
        descriptor_set->asm_operand_indices[start + i];
    const loom_low_operand_t* operand =
        &descriptor_set->operands[descriptor->operand_start + operand_index];
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_value(
        descriptor_set, builder, operand->field_name_string_offset));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t loom_low_append_manifest_asm_immediates(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder, const loom_low_asm_form_t* asm_form,
    const loom_low_descriptor_t* descriptor) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "["));
  for (uint16_t i = 0; i < asm_form->immediate_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const loom_low_asm_immediate_t* asm_immediate =
        &descriptor_set->asm_immediates[asm_form->immediate_start + i];
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[descriptor->immediate_start +
                                    asm_immediate->immediate_index];
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "{\"field\":"));
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_value(
        descriptor_set, builder, immediate->field_name_string_offset));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"name\":"));
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_value(
        descriptor_set, builder, asm_immediate->name_string_offset));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t loom_low_append_manifest_asm_forms(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"asm_forms\":["));
  for (uint32_t i = 0; i < descriptor_set->asm_form_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const loom_low_asm_form_t* asm_form = &descriptor_set->asm_forms[i];
    const loom_low_descriptor_t* descriptor =
        &descriptor_set->descriptors[asm_form->descriptor_ordinal];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"ordinal\":%" PRIu32 ",\"mnemonic\":", i));
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_value(
        descriptor_set, builder, asm_form->mnemonic_string_offset));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, ",\"descriptor\":%" PRIu32 ",\"descriptor_key\":",
        asm_form->descriptor_ordinal));
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_value(
        descriptor_set, builder, descriptor->key_string_offset));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"results\":"));
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_asm_operand_fields(
        descriptor_set, builder, descriptor,
        asm_form->result_operand_index_start,
        asm_form->result_operand_index_count));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"operands\":"));
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_asm_operand_fields(
        descriptor_set, builder, descriptor, asm_form->operand_index_start,
        asm_form->operand_index_count));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, ",\"immediates\":"));
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_asm_immediates(
        descriptor_set, builder, asm_form, descriptor));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t loom_low_append_manifest_descriptor_rows(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder, const loom_low_descriptor_t* descriptor) {
  const loom_low_schedule_class_t* schedule =
      &descriptor_set->schedule_classes[descriptor->schedule_class_id];
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ",\"schedule_class\":%" PRIu16 ",\"canonical_asm_form\":",
      descriptor->schedule_class_id));
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "null"));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%" PRIu32, descriptor->canonical_asm_form_ordinal));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"schedule_class_name\":"));
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_value(
      descriptor_set, builder, schedule->name_string_offset));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, ",\"operands\":["));
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_operand(
        descriptor_set, builder,
        &descriptor_set->operands[descriptor->operand_start + i], i));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "],\"results\":%" PRIu16 ",\"immediates\":[",
      descriptor->result_count));
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_immediate(
        descriptor_set, builder,
        &descriptor_set->immediates[descriptor->immediate_start + i], i));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "],\"effects\":["));
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_effect(
        builder, &descriptor_set->effects[descriptor->effect_start + i], i));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "],\"constraints\":["));
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_constraint(
        builder, &descriptor_set->constraints[descriptor->constraint_start + i],
        i));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder, "],\"feature_mask_words\":["));
  for (uint16_t i = 0; i < descriptor->feature_mask_word_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "%" PRIu64,
        descriptor_set
            ->feature_mask_words[descriptor->feature_mask_word_start + i]));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder, "],\"fixed_encoding_fields\":["));
  for (uint16_t i = 0; i < descriptor->encoding_field_value_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const loom_low_encoding_field_value_t* field_value =
        &descriptor_set
             ->encoding_field_values[descriptor->encoding_field_value_start +
                                     i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"encoding_field\":%" PRIu16 ",\"value\":%" PRIu64 "}",
        field_value->encoding_field_id, field_value->value));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

iree_status_t loom_low_descriptor_set_verify(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  if (descriptor_set->abi_version != LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set ABI version %" PRIu32
                            " does not match supported version %u",
                            descriptor_set->abi_version,
                            LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_tables_present(descriptor_set));
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, descriptor_set->key_string_offset, "set.key"));
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_string(
      descriptor_set, descriptor_set->target_key_string_offset, "set.target"));
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_string(
      descriptor_set, descriptor_set->feature_key_string_offset,
      "set.feature"));

  for (uint32_t i = 0; i < descriptor_set->descriptor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_descriptor(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_operand(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->immediate_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_immediate(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->enum_domain_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_enum_domain(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->enum_value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_enum_value(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->effect_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_verify_effect(i, &descriptor_set->effects[i]));
  }
  for (uint32_t i = 0; i < descriptor_set->constraint_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_verify_constraint(i, &descriptor_set->constraints[i]));
  }
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_reg_class(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->reg_class_alt_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_reg_class_alt(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->schedule_class_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_schedule_class(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->issue_use_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_issue_use(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->resource_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_resource(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->hazard_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_hazard(descriptor_set, i));
  }
  for (uint32_t i = 0; i < descriptor_set->pressure_delta_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_pressure_delta(descriptor_set, i));
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_asm_forms(descriptor_set));
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_refs(descriptor_set));
  return loom_low_verify_descriptor_id_refs(descriptor_set);
}

static iree_status_t loom_low_descriptor_set_key(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t* out_key) {
  *out_key = iree_string_view_empty();
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, out_key));
  if (iree_string_view_is_empty(*out_key)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set key is empty");
  }
  return iree_ok_status();
}

iree_host_size_t loom_low_descriptor_registry_descriptor_set_count(
    const loom_low_descriptor_registry_t* registry) {
  if (registry == NULL) return 0;
  return registry->descriptor_set_count +
         registry->descriptor_set_provider_count;
}

const loom_low_descriptor_set_t* loom_low_descriptor_registry_descriptor_set_at(
    const loom_low_descriptor_registry_t* registry, iree_host_size_t index) {
  if (registry == NULL) return NULL;
  if (index < registry->descriptor_set_count) {
    if (registry->descriptor_sets == NULL) return NULL;
    return registry->descriptor_sets[index];
  }
  index -= registry->descriptor_set_count;
  if (index < registry->descriptor_set_provider_count) {
    if (registry->descriptor_set_providers == NULL) return NULL;
    loom_low_descriptor_set_provider_t provider =
        registry->descriptor_set_providers[index];
    return provider ? provider() : NULL;
  }
  return NULL;
}

iree_status_t loom_low_descriptor_registry_verify(
    const loom_low_descriptor_registry_t* registry) {
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry is required");
  }
  if (registry->descriptor_set_count != 0 &&
      registry->descriptor_sets == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry entries are required");
  }
  if (registry->descriptor_set_provider_count != 0 &&
      registry->descriptor_set_providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry providers are required");
  }
  iree_host_size_t descriptor_set_count =
      loom_low_descriptor_registry_descriptor_set_count(registry);
  for (iree_host_size_t i = 0; i < descriptor_set_count; ++i) {
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_descriptor_set_at(registry, i);
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_verify(descriptor_set));
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_key(descriptor_set, &key));
    for (iree_host_size_t j = i + 1; j < descriptor_set_count; ++j) {
      const loom_low_descriptor_set_t* other_descriptor_set =
          loom_low_descriptor_registry_descriptor_set_at(registry, j);
      iree_string_view_t other_key = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          loom_low_descriptor_set_key(other_descriptor_set, &other_key));
      if (iree_string_view_equal(key, other_key)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "low descriptor registry has duplicate descriptor set key '%.*s'",
            (int)key.size, key.data);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_descriptor_registry_lookup(
    const loom_low_descriptor_registry_t* registry, iree_string_view_t key,
    const loom_low_descriptor_set_t** out_descriptor_set) {
  if (out_descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor lookup output is required");
  }
  *out_descriptor_set = NULL;
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry is required");
  }
  if (registry->descriptor_set_count != 0 &&
      registry->descriptor_sets == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry entries are required");
  }
  if (registry->descriptor_set_provider_count != 0 &&
      registry->descriptor_set_providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry providers are required");
  }
  iree_host_size_t descriptor_set_count =
      loom_low_descriptor_registry_descriptor_set_count(registry);
  for (iree_host_size_t i = 0; i < descriptor_set_count; ++i) {
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_descriptor_registry_descriptor_set_at(registry, i);
    iree_string_view_t candidate_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        loom_low_descriptor_set_key(descriptor_set, &candidate_key));
    if (!iree_string_view_equal(candidate_key, key)) continue;
    if (*out_descriptor_set != NULL) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "low descriptor registry has duplicate descriptor set key '%.*s'",
          (int)key.size, key.data);
    }
    *out_descriptor_set = descriptor_set;
  }
  return iree_ok_status();
}

iree_status_t loom_low_descriptor_set_string(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, iree_string_view_t* out_string) {
  if (out_string == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor string output is required");
  }
  *out_string = iree_string_view_empty();
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  return loom_low_descriptor_set_string_impl(descriptor_set, string_offset,
                                             /*allow_none=*/true, out_string);
}

const loom_low_descriptor_t* loom_low_descriptor_set_descriptor_at(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal) {
  if (descriptor_set == NULL ||
      descriptor_ordinal >= descriptor_set->descriptor_count) {
    return NULL;
  }
  return &descriptor_set->descriptors[descriptor_ordinal];
}

const loom_low_asm_form_t* loom_low_descriptor_set_asm_form_at(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t asm_form_ordinal) {
  if (descriptor_set == NULL ||
      asm_form_ordinal >= descriptor_set->asm_form_count) {
    return NULL;
  }
  return &descriptor_set->asm_forms[asm_form_ordinal];
}

iree_status_t loom_low_descriptor_set_lookup_canonical_asm_form(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, uint32_t* out_asm_form_ordinal) {
  if (out_asm_form_ordinal == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low canonical asm form output is required");
  }
  *out_asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  if (descriptor == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor ordinal %" PRIu32
                            " is out of range",
                            descriptor_ordinal);
  }
  if (descriptor->canonical_asm_form_ordinal ==
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    iree_string_view_t key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set, descriptor->key_string_offset, /*allow_none=*/false,
        &key));
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low descriptor '%.*s' has no unique canonical "
                            "asm form",
                            (int)key.size, key.data);
  }
  const loom_low_asm_form_t* asm_form = loom_low_descriptor_set_asm_form_at(
      descriptor_set, descriptor->canonical_asm_form_ordinal);
  if (asm_form == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor canonical asm form ordinal is out "
                            "of range");
  }
  if (asm_form->descriptor_ordinal != descriptor_ordinal) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor canonical asm form references a "
                            "different descriptor");
  }
  *out_asm_form_ordinal = descriptor->canonical_asm_form_ordinal;
  return iree_ok_status();
}

iree_status_t loom_low_descriptor_set_lookup_descriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    uint32_t* out_descriptor_ordinal) {
  if (out_descriptor_ordinal == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor lookup output is required");
  }
  *out_descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  if (descriptor_set->descriptor_ref_count !=
      descriptor_set->descriptor_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor reference map count %" PRIu32
                            " does not match descriptor count %" PRIu32,
                            descriptor_set->descriptor_ref_count,
                            descriptor_set->descriptor_count);
  }
  if (descriptor_set->descriptor_ref_count != 0 &&
      descriptor_set->descriptor_refs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor reference map is required");
  }
  uint32_t low = 0;
  uint32_t high = descriptor_set->descriptor_ref_count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const loom_low_descriptor_ref_t* descriptor_ref =
        &descriptor_set->descriptor_refs[mid];
    iree_string_view_t descriptor_ref_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set, descriptor_ref->key_string_offset, /*allow_none=*/false,
        &descriptor_ref_key));
    const int comparison = iree_string_view_compare(descriptor_ref_key, key);
    if (comparison == 0) {
      if (descriptor_ref->descriptor_ordinal >=
          descriptor_set->descriptor_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low descriptor reference '%.*s' points at descriptor ordinal "
            "%" PRIu32 " but only %" PRIu32 " descriptors exist",
            (int)descriptor_ref_key.size, descriptor_ref_key.data,
            descriptor_ref->descriptor_ordinal,
            descriptor_set->descriptor_count);
      }
      *out_descriptor_ordinal = descriptor_ref->descriptor_ordinal;
      return iree_ok_status();
    }
    if (comparison < 0) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "low descriptor key '%.*s' was not found",
                          (int)key.size, key.data);
}

uint32_t loom_low_descriptor_set_lookup_descriptor_by_id(
    const loom_low_descriptor_set_t* descriptor_set, uint64_t stable_id) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT(stable_id != LOOM_LOW_DESCRIPTOR_ID_NONE);

  uint32_t low = 0;
  uint32_t high = descriptor_set->descriptor_id_ref_count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const loom_low_descriptor_id_ref_t* descriptor_id_ref =
        &descriptor_set->descriptor_id_refs[mid];
    if (descriptor_id_ref->stable_id == stable_id) {
      IREE_ASSERT(descriptor_id_ref->descriptor_ordinal <
                  descriptor_set->descriptor_count);
      return descriptor_id_ref->descriptor_ordinal;
    }
    if (descriptor_id_ref->stable_id < stable_id) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

iree_status_t loom_low_descriptor_set_lookup_asm_form(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_view_t mnemonic, uint32_t* out_asm_form_ordinal) {
  if (out_asm_form_ordinal == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm form lookup output is required");
  }
  *out_asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  if (descriptor_set->asm_form_count != 0 &&
      descriptor_set->asm_forms == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low asm form table is required");
  }
  uint32_t low = 0;
  uint32_t high = descriptor_set->asm_form_count;
  while (low < high) {
    const uint32_t mid = low + (high - low) / 2;
    const loom_low_asm_form_t* asm_form = &descriptor_set->asm_forms[mid];
    iree_string_view_t asm_mnemonic = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set, asm_form->mnemonic_string_offset, /*allow_none=*/false,
        &asm_mnemonic));
    const int comparison = iree_string_view_compare(asm_mnemonic, mnemonic);
    if (comparison == 0) {
      if (asm_form->descriptor_ordinal >= descriptor_set->descriptor_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low asm form '%.*s' points at descriptor "
            "ordinal %" PRIu32 " but only %" PRIu32 " descriptors exist",
            (int)asm_mnemonic.size, asm_mnemonic.data,
            asm_form->descriptor_ordinal, descriptor_set->descriptor_count);
      }
      *out_asm_form_ordinal = mid;
      return iree_ok_status();
    }
    if (comparison < 0) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "low asm mnemonic '%.*s' was not found",
                          (int)mnemonic.size, mnemonic.data);
}

iree_status_t loom_low_descriptor_set_format_manifest_json(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder) {
  if (builder == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor manifest builder is required");
  }
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_verify(descriptor_set));

  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "{"));
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_field(
      descriptor_set, builder, "key", descriptor_set->key_string_offset));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_field(
      descriptor_set, builder, "target",
      descriptor_set->target_key_string_offset));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_field(
      descriptor_set, builder, "feature_namespace",
      descriptor_set->feature_key_string_offset));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ",\"abi_version\":%" PRIu32 ",\"generator_version\":%" PRIu32,
      descriptor_set->abi_version, descriptor_set->generator_version));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"table_counts\":{\"descriptors\":%" PRIu32
      ",\"descriptor_refs\":%" PRIu32 ",\"descriptor_id_refs\":%" PRIu32
      ",\"asm_forms\":%" PRIu32 ",\"asm_operand_indices\":%" PRIu32
      ",\"asm_immediates\":%" PRIu32 ",\"operands\":%" PRIu32
      ",\"immediates\":%" PRIu32 ",\"enum_domains\":%" PRIu32
      ",\"enum_values\":%" PRIu32 ",\"effects\":%" PRIu32
      ",\"constraints\":%" PRIu32 ",\"reg_classes\":%" PRIu32
      ",\"reg_class_alts\":%" PRIu32 ",\"schedule_classes\":%" PRIu32
      ",\"issue_uses\":%" PRIu32 ",\"resources\":%" PRIu32
      ",\"hazards\":%" PRIu32 ",\"pressure_deltas\":%" PRIu32
      ",\"feature_mask_words\":%" PRIu32 ",\"encoding_field_values\":%" PRIu32
      "}",
      descriptor_set->descriptor_count, descriptor_set->descriptor_ref_count,
      descriptor_set->descriptor_id_ref_count, descriptor_set->asm_form_count,
      descriptor_set->asm_operand_index_count,
      descriptor_set->asm_immediate_count, descriptor_set->operand_count,
      descriptor_set->immediate_count, descriptor_set->enum_domain_count,
      descriptor_set->enum_value_count, descriptor_set->effect_count,
      descriptor_set->constraint_count, descriptor_set->reg_class_count,
      descriptor_set->reg_class_alt_count, descriptor_set->schedule_class_count,
      descriptor_set->issue_use_count, descriptor_set->resource_count,
      descriptor_set->hazard_count, descriptor_set->pressure_delta_count,
      descriptor_set->feature_mask_word_count,
      descriptor_set->encoding_field_value_count));
  IREE_RETURN_IF_ERROR(
      loom_low_append_manifest_reg_classes(descriptor_set, builder));
  IREE_RETURN_IF_ERROR(
      loom_low_append_manifest_resources(descriptor_set, builder));
  IREE_RETURN_IF_ERROR(
      loom_low_append_manifest_schedule_classes(descriptor_set, builder));
  IREE_RETURN_IF_ERROR(
      loom_low_append_manifest_asm_forms(descriptor_set, builder));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"descriptors\":["));
  for (uint32_t i = 0; i < descriptor_set->descriptor_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    const loom_low_descriptor_t* descriptor = &descriptor_set->descriptors[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"ordinal\":%" PRIu32 ",", i));
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_field(
        descriptor_set, builder, "key", descriptor->key_string_offset));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_field(
        descriptor_set, builder, "mnemonic",
        descriptor->mnemonic_string_offset));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_append_string_field(
        descriptor_set, builder, "semantic_tag",
        descriptor->semantic_tag_string_offset));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"encoding_format\":%" PRIu16 ",\"encoding\":%" PRIu16
        ",\"flags\":%" PRIu16,
        descriptor->encoding_format_id, descriptor->encoding_id,
        descriptor->flags));
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_flag_names(
        builder, descriptor->flags, kLoomLowDescriptorFlagNames,
        IREE_ARRAYSIZE(kLoomLowDescriptorFlagNames)));
    IREE_RETURN_IF_ERROR(loom_low_append_manifest_descriptor_rows(
        descriptor_set, builder, descriptor));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}"));
  }
  return iree_string_builder_append_cstring(builder, "]}");
}
