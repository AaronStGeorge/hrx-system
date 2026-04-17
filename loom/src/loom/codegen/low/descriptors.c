// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptors.h"

#include <inttypes.h>

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

static iree_status_t loom_low_descriptor_set_string_impl(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_bstring_table_offset_t string_offset, bool allow_none,
    iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    if (allow_none) return iree_ok_status();
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

static iree_status_t loom_low_verify_tables_present(
    const loom_low_descriptor_set_t* descriptor_set) {
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->descriptors, descriptor_set->descriptor_count,
      "descriptors"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->descriptor_refs, descriptor_set->descriptor_ref_count,
      "descriptor_refs"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->operands, descriptor_set->operand_count, "operands"));
  IREE_RETURN_IF_ERROR(loom_low_verify_pointer_for_count(
      descriptor_set->immediates, descriptor_set->immediate_count,
      "immediates"));
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

static bool loom_low_operand_role_is_packet_operand(
    loom_low_operand_role_t role) {
  return role == LOOM_LOW_OPERAND_ROLE_OPERAND ||
         role == LOOM_LOW_OPERAND_ROLE_PREDICATE ||
         role == LOOM_LOW_OPERAND_ROLE_RESOURCE;
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
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, descriptor->key_string_offset, "descriptor.key"));
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
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_operand_roles(
      descriptor_set, descriptor, descriptor_index));
  return loom_low_verify_descriptor_constraints(descriptor_set, descriptor);
}

static iree_status_t loom_low_verify_operand(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t operand_index) {
  const loom_low_operand_t* operand = &descriptor_set->operands[operand_index];
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, operand->field_name_string_offset, "operand.field_name"));
  if (operand->role == LOOM_LOW_OPERAND_ROLE_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low operand %" PRIu32 " has unknown role",
                            operand_index);
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
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, immediate->field_name_string_offset,
      "immediate.field_name"));
  if (immediate->kind == LOOM_LOW_IMMEDIATE_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low immediate %" PRIu32 " has unknown kind",
                            immediate_index);
  }
  if (immediate->bit_width > 64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low immediate %" PRIu32
                            " has unsupported bit width %" PRIu16,
                            immediate_index, immediate->bit_width);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_reg_class(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t reg_class_index) {
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[reg_class_index];
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, reg_class->name_string_offset, "reg_class.name"));
  if (reg_class->alloc_unit_bits == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low register class %" PRIu32
                            " has zero allocation-unit width",
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
  IREE_RETURN_IF_ERROR(loom_low_verify_required_string(
      descriptor_set, schedule_class->name_string_offset,
      "schedule_class.name"));
  if (schedule_class->latency_kind == LOOM_LOW_LATENCY_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low schedule class %" PRIu32
                            " has unknown latency kind",
                            schedule_class_index);
  }
  if (schedule_class->model_quality == LOOM_LOW_MODEL_QUALITY_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low schedule class %" PRIu32
                            " has unknown model quality",
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
  if (resource->capacity_per_cycle == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low resource %" PRIu32 " has zero capacity per cycle", resource_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_effect(uint32_t effect_index,
                                            const loom_low_effect_t* effect) {
  if (effect->kind == LOOM_LOW_EFFECT_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low effect %" PRIu32 " has unknown kind",
                            effect_index);
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
  return iree_ok_status();
}

static iree_status_t loom_low_verify_hazard(uint32_t hazard_index,
                                            const loom_low_hazard_t* hazard) {
  if (hazard->kind == LOOM_LOW_HAZARD_KIND_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low hazard %" PRIu32 " has unknown kind",
                            hazard_index);
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
    IREE_RETURN_IF_ERROR(
        loom_low_verify_hazard(i, &descriptor_set->hazards[i]));
  }
  for (uint32_t i = 0; i < descriptor_set->pressure_delta_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_pressure_delta(descriptor_set, i));
  }
  return loom_low_verify_descriptor_refs(descriptor_set);
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
      ",\"descriptor_refs\":%" PRIu32 ",\"operands\":%" PRIu32
      ",\"immediates\":%" PRIu32 ",\"effects\":%" PRIu32
      ",\"constraints\":%" PRIu32 ",\"reg_classes\":%" PRIu32
      ",\"schedule_classes\":%" PRIu32 "}",
      descriptor_set->descriptor_count, descriptor_set->descriptor_ref_count,
      descriptor_set->operand_count, descriptor_set->immediate_count,
      descriptor_set->effect_count, descriptor_set->constraint_count,
      descriptor_set->reg_class_count, descriptor_set->schedule_class_count));
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
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        ",\"schedule_class\":%" PRIu16 ",\"operands\":%" PRIu16
        ",\"results\":%" PRIu16 ",\"immediates\":%" PRIu16
        ",\"effects\":%" PRIu16 ",\"flags\":%" PRIu16 "}",
        descriptor->schedule_class_id, descriptor->operand_count,
        descriptor->result_count, descriptor->immediate_count,
        descriptor->effect_count, descriptor->flags));
  }
  return iree_string_builder_append_cstring(builder, "]}");
}
