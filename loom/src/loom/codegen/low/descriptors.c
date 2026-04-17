// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptors.h"

#include <inttypes.h>

#define LOOM_LOW_FINGERPRINT_OFFSET UINT64_C(1469598103934665603)
#define LOOM_LOW_FINGERPRINT_PRIME UINT64_C(1099511628211)
#define LOOM_LOW_FINGERPRINT_HIGH_SALT UINT64_C(0x9E3779B97F4A7C15)

typedef struct loom_low_fingerprint_builder_t {
  uint64_t low;
  uint64_t high;
} loom_low_fingerprint_builder_t;

static loom_low_fingerprint_builder_t loom_low_fingerprint_builder_initialize(
    void) {
  return (loom_low_fingerprint_builder_t){
      .low = LOOM_LOW_FINGERPRINT_OFFSET,
      .high = LOOM_LOW_FINGERPRINT_OFFSET ^ LOOM_LOW_FINGERPRINT_HIGH_SALT,
  };
}

static void loom_low_fingerprint_builder_byte(
    loom_low_fingerprint_builder_t* builder, uint8_t value) {
  builder->low ^= value;
  builder->low *= LOOM_LOW_FINGERPRINT_PRIME;
  builder->high ^= (uint8_t)(value + 0xA5u);
  builder->high *= LOOM_LOW_FINGERPRINT_PRIME;
}

static void loom_low_fingerprint_builder_u16(
    loom_low_fingerprint_builder_t* builder, uint16_t value) {
  for (uint32_t i = 0; i < 2; ++i) {
    loom_low_fingerprint_builder_byte(builder, (uint8_t)(value & 0xFFu));
    value >>= 8;
  }
}

static void loom_low_fingerprint_builder_u32(
    loom_low_fingerprint_builder_t* builder, uint32_t value) {
  for (uint32_t i = 0; i < 4; ++i) {
    loom_low_fingerprint_builder_byte(builder, (uint8_t)(value & 0xFFu));
    value >>= 8;
  }
}

static void loom_low_fingerprint_builder_u64(
    loom_low_fingerprint_builder_t* builder, uint64_t value) {
  for (uint32_t i = 0; i < 8; ++i) {
    loom_low_fingerprint_builder_byte(builder, (uint8_t)(value & 0xFFu));
    value >>= 8;
  }
}

static void loom_low_fingerprint_builder_i64(
    loom_low_fingerprint_builder_t* builder, int64_t value) {
  loom_low_fingerprint_builder_u64(builder, (uint64_t)value);
}

static void loom_low_fingerprint_builder_bytes(
    loom_low_fingerprint_builder_t* builder, const char* data,
    uint32_t data_length) {
  loom_low_fingerprint_builder_u32(builder, data_length);
  for (uint32_t i = 0; i < data_length; ++i) {
    loom_low_fingerprint_builder_byte(builder, (uint8_t)data[i]);
  }
}

static loom_low_fingerprint_t loom_low_fingerprint_builder_finalize(
    loom_low_fingerprint_builder_t builder) {
  return (loom_low_fingerprint_t){
      .low = builder.low,
      .high = builder.high,
  };
}

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
    const loom_low_descriptor_set_t* descriptor_set, uint32_t string_offset,
    bool allow_none, iree_string_view_t* out_string) {
  *out_string = iree_string_view_empty();
  if (string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    if (allow_none) return iree_ok_status();
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "required low descriptor string offset is NONE");
  }
  if (descriptor_set->string_data == NULL ||
      descriptor_set->string_data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor string table is empty");
  }
  if (string_offset >= descriptor_set->string_data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low descriptor string offset %" PRIu32
                            " exceeds string table length %" PRIu32,
                            string_offset, descriptor_set->string_data_length);
  }
  const char* data = descriptor_set->string_data + string_offset;
  uint32_t remaining = descriptor_set->string_data_length - string_offset;
  for (uint32_t i = 0; i < remaining; ++i) {
    if (data[i] == '\0') {
      *out_string = iree_make_string_view(data, i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "low descriptor string offset %" PRIu32
                          " has no NUL terminator in string table",
                          string_offset);
}

static iree_status_t loom_low_verify_required_string(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t string_offset,
    const char* field_name) {
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
    const loom_low_descriptor_set_t* descriptor_set, uint32_t string_offset,
    const char* field_name) {
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

static iree_status_t loom_low_verify_descriptor_keys_unique(
    const loom_low_descriptor_set_t* descriptor_set) {
  for (uint32_t i = 0; i < descriptor_set->descriptor_count; ++i) {
    iree_string_view_t lhs = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set, descriptor_set->descriptors[i].key_string_offset,
        /*allow_none=*/false, &lhs));
    for (uint32_t j = i + 1; j < descriptor_set->descriptor_count; ++j) {
      iree_string_view_t rhs = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
          descriptor_set, descriptor_set->descriptors[j].key_string_offset,
          /*allow_none=*/false, &rhs));
      if (iree_string_view_equal(lhs, rhs)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate low descriptor key '%.*s'",
                                (int)lhs.size, lhs.data);
      }
    }
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
      descriptor_set, descriptor->source_string_offset, "descriptor.source"));
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

static void loom_low_fingerprint_reg_class(
    loom_low_fingerprint_builder_t* builder,
    const loom_low_reg_class_t* reg_class) {
  loom_low_fingerprint_builder_u32(builder, reg_class->name_string_offset);
  loom_low_fingerprint_builder_u16(builder, reg_class->target_bank_id);
  loom_low_fingerprint_builder_u16(builder, reg_class->flags);
  loom_low_fingerprint_builder_u16(builder, reg_class->alloc_unit_bits);
  loom_low_fingerprint_builder_u16(builder, reg_class->physical_count);
  loom_low_fingerprint_builder_u16(builder, reg_class->alias_set_id);
  loom_low_fingerprint_builder_u16(builder, reg_class->spill_class_id);
}

static void loom_low_fingerprint_reg_class_alt(
    loom_low_fingerprint_builder_t* builder,
    const loom_low_reg_class_alt_t* alt) {
  loom_low_fingerprint_builder_u16(builder, alt->reg_class_id);
  loom_low_fingerprint_builder_u16(builder, alt->flags);
}

static void loom_low_fingerprint_operand(
    loom_low_fingerprint_builder_t* builder,
    const loom_low_operand_t* operand) {
  loom_low_fingerprint_builder_u32(builder, operand->field_name_string_offset);
  loom_low_fingerprint_builder_u16(builder, (uint16_t)operand->role);
  loom_low_fingerprint_builder_u16(builder, operand->flags);
  loom_low_fingerprint_builder_u16(builder, operand->reg_class_alt_start);
  loom_low_fingerprint_builder_u16(builder, operand->reg_class_alt_count);
  loom_low_fingerprint_builder_u16(builder, operand->unit_count);
  loom_low_fingerprint_builder_u16(builder, operand->data_format_id);
  loom_low_fingerprint_builder_u16(builder, operand->read_stage);
  loom_low_fingerprint_builder_u16(builder, operand->ready_stage);
}

static void loom_low_fingerprint_immediate(
    loom_low_fingerprint_builder_t* builder,
    const loom_low_immediate_t* immediate) {
  loom_low_fingerprint_builder_u32(builder,
                                   immediate->field_name_string_offset);
  loom_low_fingerprint_builder_u16(builder, (uint16_t)immediate->kind);
  loom_low_fingerprint_builder_u16(builder, immediate->flags);
  loom_low_fingerprint_builder_u16(builder, immediate->bit_width);
  loom_low_fingerprint_builder_u16(builder, immediate->encoding_id);
  loom_low_fingerprint_builder_i64(builder, immediate->signed_min);
  loom_low_fingerprint_builder_u64(builder, immediate->unsigned_max);
}

static void loom_low_fingerprint_effect(loom_low_fingerprint_builder_t* builder,
                                        const loom_low_effect_t* effect) {
  loom_low_fingerprint_builder_u16(builder, (uint16_t)effect->kind);
  loom_low_fingerprint_builder_u16(builder, (uint16_t)effect->memory_space);
  loom_low_fingerprint_builder_u16(builder, effect->scope_id);
  loom_low_fingerprint_builder_u16(builder, effect->flags);
  loom_low_fingerprint_builder_u16(builder, effect->counter_id);
  loom_low_fingerprint_builder_u16(builder, effect->width_bits);
}

static void loom_low_fingerprint_constraint(
    loom_low_fingerprint_builder_t* builder,
    const loom_low_constraint_t* constraint) {
  loom_low_fingerprint_builder_u16(builder, (uint16_t)constraint->kind);
  loom_low_fingerprint_builder_u16(builder, constraint->lhs_operand_index);
  loom_low_fingerprint_builder_u16(builder, constraint->rhs_operand_index);
  loom_low_fingerprint_builder_u16(builder, constraint->flags);
}

static void loom_low_fingerprint_issue_use(
    loom_low_fingerprint_builder_t* builder,
    const loom_low_issue_use_t* issue_use) {
  loom_low_fingerprint_builder_u16(builder, issue_use->resource_id);
  loom_low_fingerprint_builder_u16(builder, issue_use->cycles);
  loom_low_fingerprint_builder_u16(builder, issue_use->units);
  loom_low_fingerprint_builder_u16(builder, issue_use->stage);
}

static void loom_low_fingerprint_pressure_delta(
    loom_low_fingerprint_builder_t* builder,
    const loom_low_pressure_delta_t* pressure_delta) {
  loom_low_fingerprint_builder_u16(builder, pressure_delta->reg_class_id);
  loom_low_fingerprint_builder_u16(builder, (uint16_t)pressure_delta->delta);
}

static void loom_low_fingerprint_resource(
    loom_low_fingerprint_builder_t* builder,
    const loom_low_resource_t* resource) {
  loom_low_fingerprint_builder_u32(builder, resource->name_string_offset);
  loom_low_fingerprint_builder_u16(builder, resource->capacity_per_cycle);
  loom_low_fingerprint_builder_u16(builder, resource->flags);
  loom_low_fingerprint_builder_u16(builder, (uint16_t)resource->kind);
  loom_low_fingerprint_builder_u16(builder, resource->contention_group_id);
}

static void loom_low_fingerprint_hazard(loom_low_fingerprint_builder_t* builder,
                                        const loom_low_hazard_t* hazard) {
  loom_low_fingerprint_builder_u16(builder, (uint16_t)hazard->kind);
  loom_low_fingerprint_builder_u16(builder, hazard->resource_or_counter_id);
  loom_low_fingerprint_builder_u16(builder, hazard->producer_stage);
  loom_low_fingerprint_builder_u16(builder, hazard->consumer_stage);
  loom_low_fingerprint_builder_u16(builder, hazard->distance);
  loom_low_fingerprint_builder_u16(builder, hazard->flags);
}

static void loom_low_fingerprint_schedule_class(
    loom_low_fingerprint_builder_t* builder,
    const loom_low_schedule_class_t* schedule_class) {
  loom_low_fingerprint_builder_u32(builder, schedule_class->name_string_offset);
  loom_low_fingerprint_builder_u16(builder, schedule_class->latency_cycles);
  loom_low_fingerprint_builder_u16(builder,
                                   (uint16_t)schedule_class->latency_kind);
  loom_low_fingerprint_builder_u16(builder, schedule_class->issue_use_start);
  loom_low_fingerprint_builder_u16(builder, schedule_class->issue_use_count);
  loom_low_fingerprint_builder_u16(builder, schedule_class->hazard_start);
  loom_low_fingerprint_builder_u16(builder, schedule_class->hazard_count);
  loom_low_fingerprint_builder_u16(builder, schedule_class->flags);
  loom_low_fingerprint_builder_u16(builder,
                                   (uint16_t)schedule_class->model_quality);
  loom_low_fingerprint_builder_u16(builder,
                                   schedule_class->pressure_delta_start);
  loom_low_fingerprint_builder_u16(builder,
                                   schedule_class->pressure_delta_count);
}

static void loom_low_fingerprint_descriptor(
    loom_low_fingerprint_builder_t* builder,
    const loom_low_descriptor_t* descriptor) {
  loom_low_fingerprint_builder_u32(builder, descriptor->key_string_offset);
  loom_low_fingerprint_builder_u32(builder, descriptor->mnemonic_string_offset);
  loom_low_fingerprint_builder_u32(builder, descriptor->source_string_offset);
  loom_low_fingerprint_builder_u32(builder,
                                   descriptor->semantic_tag_string_offset);
  loom_low_fingerprint_builder_u32(builder,
                                   descriptor->feature_mask_word_start);
  loom_low_fingerprint_builder_u16(builder,
                                   descriptor->feature_mask_word_count);
  loom_low_fingerprint_builder_u16(builder, descriptor->encoding_id);
  loom_low_fingerprint_builder_u32(builder, descriptor->operand_start);
  loom_low_fingerprint_builder_u16(builder, descriptor->operand_count);
  loom_low_fingerprint_builder_u16(builder, descriptor->result_count);
  loom_low_fingerprint_builder_u32(builder, descriptor->immediate_start);
  loom_low_fingerprint_builder_u16(builder, descriptor->immediate_count);
  loom_low_fingerprint_builder_u32(builder, descriptor->effect_start);
  loom_low_fingerprint_builder_u16(builder, descriptor->effect_count);
  loom_low_fingerprint_builder_u32(builder, descriptor->constraint_start);
  loom_low_fingerprint_builder_u16(builder, descriptor->constraint_count);
  loom_low_fingerprint_builder_u16(builder, descriptor->schedule_class_id);
  loom_low_fingerprint_builder_u16(builder, descriptor->flags);
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
    uint32_t string_offset) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_set_string(descriptor_set, string_offset, &value));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, "\"%s\":", field_name));
  return loom_low_append_json_string(builder, value);
}

bool loom_low_fingerprint_equal(loom_low_fingerprint_t lhs,
                                loom_low_fingerprint_t rhs) {
  return lhs.low == rhs.low && lhs.high == rhs.high;
}

iree_status_t loom_low_descriptor_set_compute_fingerprint(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_fingerprint_t* out_fingerprint) {
  if (out_fingerprint == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor fingerprint output is required");
  }
  *out_fingerprint = (loom_low_fingerprint_t){0};
  if (descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor set is required");
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_tables_present(descriptor_set));
  if (descriptor_set->string_data_length != 0 &&
      descriptor_set->string_data == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor string table pointer is required "
                            "when its length is non-zero");
  }

  loom_low_fingerprint_builder_t builder =
      loom_low_fingerprint_builder_initialize();
  loom_low_fingerprint_builder_bytes(&builder, "loom.low.descriptor_set.v1",
                                     sizeof("loom.low.descriptor_set.v1") - 1);
  loom_low_fingerprint_builder_u32(&builder, descriptor_set->abi_version);
  loom_low_fingerprint_builder_u32(&builder, descriptor_set->generator_version);
  loom_low_fingerprint_builder_u32(&builder, descriptor_set->key_string_offset);
  loom_low_fingerprint_builder_u32(&builder,
                                   descriptor_set->target_key_string_offset);
  loom_low_fingerprint_builder_u32(&builder,
                                   descriptor_set->feature_key_string_offset);
  loom_low_fingerprint_builder_bytes(&builder, descriptor_set->string_data,
                                     descriptor_set->string_data_length);

  loom_low_fingerprint_builder_u32(&builder, descriptor_set->descriptor_count);
  for (uint32_t i = 0; i < descriptor_set->descriptor_count; ++i) {
    loom_low_fingerprint_descriptor(&builder, &descriptor_set->descriptors[i]);
  }
  loom_low_fingerprint_builder_u32(&builder, descriptor_set->operand_count);
  for (uint32_t i = 0; i < descriptor_set->operand_count; ++i) {
    loom_low_fingerprint_operand(&builder, &descriptor_set->operands[i]);
  }
  loom_low_fingerprint_builder_u32(&builder, descriptor_set->immediate_count);
  for (uint32_t i = 0; i < descriptor_set->immediate_count; ++i) {
    loom_low_fingerprint_immediate(&builder, &descriptor_set->immediates[i]);
  }
  loom_low_fingerprint_builder_u32(&builder, descriptor_set->effect_count);
  for (uint32_t i = 0; i < descriptor_set->effect_count; ++i) {
    loom_low_fingerprint_effect(&builder, &descriptor_set->effects[i]);
  }
  loom_low_fingerprint_builder_u32(&builder, descriptor_set->constraint_count);
  for (uint32_t i = 0; i < descriptor_set->constraint_count; ++i) {
    loom_low_fingerprint_constraint(&builder, &descriptor_set->constraints[i]);
  }
  loom_low_fingerprint_builder_u32(&builder, descriptor_set->reg_class_count);
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    loom_low_fingerprint_reg_class(&builder, &descriptor_set->reg_classes[i]);
  }
  loom_low_fingerprint_builder_u32(&builder,
                                   descriptor_set->reg_class_alt_count);
  for (uint32_t i = 0; i < descriptor_set->reg_class_alt_count; ++i) {
    loom_low_fingerprint_reg_class_alt(&builder,
                                       &descriptor_set->reg_class_alts[i]);
  }
  loom_low_fingerprint_builder_u32(&builder,
                                   descriptor_set->schedule_class_count);
  for (uint32_t i = 0; i < descriptor_set->schedule_class_count; ++i) {
    loom_low_fingerprint_schedule_class(&builder,
                                        &descriptor_set->schedule_classes[i]);
  }
  loom_low_fingerprint_builder_u32(&builder, descriptor_set->issue_use_count);
  for (uint32_t i = 0; i < descriptor_set->issue_use_count; ++i) {
    loom_low_fingerprint_issue_use(&builder, &descriptor_set->issue_uses[i]);
  }
  loom_low_fingerprint_builder_u32(&builder, descriptor_set->resource_count);
  for (uint32_t i = 0; i < descriptor_set->resource_count; ++i) {
    loom_low_fingerprint_resource(&builder, &descriptor_set->resources[i]);
  }
  loom_low_fingerprint_builder_u32(&builder, descriptor_set->hazard_count);
  for (uint32_t i = 0; i < descriptor_set->hazard_count; ++i) {
    loom_low_fingerprint_hazard(&builder, &descriptor_set->hazards[i]);
  }
  loom_low_fingerprint_builder_u32(&builder,
                                   descriptor_set->pressure_delta_count);
  for (uint32_t i = 0; i < descriptor_set->pressure_delta_count; ++i) {
    loom_low_fingerprint_pressure_delta(&builder,
                                        &descriptor_set->pressure_deltas[i]);
  }
  loom_low_fingerprint_builder_u32(&builder,
                                   descriptor_set->feature_mask_word_count);
  for (uint32_t i = 0; i < descriptor_set->feature_mask_word_count; ++i) {
    loom_low_fingerprint_builder_u64(&builder,
                                     descriptor_set->feature_mask_words[i]);
  }

  *out_fingerprint = loom_low_fingerprint_builder_finalize(builder);
  return iree_ok_status();
}

iree_status_t loom_low_descriptor_set_fingerprint_matches(
    const loom_low_descriptor_set_t* descriptor_set, bool* out_matches) {
  if (out_matches == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low descriptor fingerprint match output is required");
  }
  *out_matches = false;
  loom_low_fingerprint_t fingerprint = {0};
  IREE_RETURN_IF_ERROR(loom_low_descriptor_set_compute_fingerprint(
      descriptor_set, &fingerprint));
  *out_matches =
      loom_low_fingerprint_equal(fingerprint, descriptor_set->fingerprint);
  return iree_ok_status();
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
  return loom_low_verify_descriptor_keys_unique(descriptor_set);
}

iree_status_t loom_low_descriptor_set_string(
    const loom_low_descriptor_set_t* descriptor_set, uint32_t string_offset,
    iree_string_view_t* out_string) {
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
  for (uint32_t i = 0; i < descriptor_set->descriptor_count; ++i) {
    iree_string_view_t descriptor_key = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string_impl(
        descriptor_set, descriptor_set->descriptors[i].key_string_offset,
        /*allow_none=*/false, &descriptor_key));
    if (iree_string_view_equal(descriptor_key, key)) {
      *out_descriptor_ordinal = i;
      return iree_ok_status();
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
      builder,
      ",\"abi_version\":%" PRIu32 ",\"generator_version\":%" PRIu32
      ",\"fingerprint\":\"%016" PRIx64 "%016" PRIx64 "\"",
      descriptor_set->abi_version, descriptor_set->generator_version,
      descriptor_set->fingerprint.high, descriptor_set->fingerprint.low));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"table_counts\":{\"descriptors\":%" PRIu32 ",\"operands\":%" PRIu32
      ",\"immediates\":%" PRIu32 ",\"effects\":%" PRIu32
      ",\"constraints\":%" PRIu32 ",\"reg_classes\":%" PRIu32
      ",\"schedule_classes\":%" PRIu32 "}",
      descriptor_set->descriptor_count, descriptor_set->operand_count,
      descriptor_set->immediate_count, descriptor_set->effect_count,
      descriptor_set->constraint_count, descriptor_set->reg_class_count,
      descriptor_set->schedule_class_count));
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
