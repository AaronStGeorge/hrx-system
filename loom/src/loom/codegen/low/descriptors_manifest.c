// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptors_manifest.h"

#include <inttypes.h>

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
  iree_string_view_t value =
      loom_low_descriptor_set_string(descriptor_set, string_offset);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, "\"%s\":", field_name));
  return loom_low_append_json_string(builder, value);
}

typedef struct loom_low_manifest_flag_name_t {
  uint16_t bit;
  const char* name;
} loom_low_manifest_flag_name_t;

static const loom_low_manifest_flag_name_t kLoomLowRegClassFlagNames[] = {
    {LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY, "virtual_only"},
    {LOOM_LOW_REG_CLASS_FLAG_PHYSICAL, "physical"},
    {LOOM_LOW_REG_CLASS_FLAG_REFERENCE, "reference"},
    {LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE, "unspillable"},
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
    {LOOM_LOW_OPERAND_FLAG_STATE_READ, "state_read"},
    {LOOM_LOW_OPERAND_FLAG_STATE_WRITE, "state_write"},
};

static const loom_low_manifest_flag_name_t kLoomLowImmediateFlagNames[] = {
    {LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC, "symbolic"},
    {LOOM_LOW_IMMEDIATE_FLAG_RELATIVE, "relative"},
    {LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE, "default_value"},
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
  iree_string_view_t value =
      loom_low_descriptor_set_string(descriptor_set, string_offset);
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
      ",\"signed_min\":%" PRId64 ",\"unsigned_max\":%" PRIu64
      ",\"default_value\":%" PRId64 "}",
      immediate->bit_width, immediate->enum_domain_id,
      immediate->encoding_field_id, immediate->encoding_id,
      immediate->signed_min, immediate->unsigned_max, immediate->default_value);
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
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "],\"operand_forms\":"));
  return iree_string_builder_append_format(builder, "%" PRIu16,
                                           descriptor->operand_form_count);
}

iree_status_t loom_low_descriptor_set_format_manifest_json(
    const loom_low_descriptor_set_t* descriptor_set,
    iree_string_builder_t* builder) {
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
      ",\"stable_id\":%" PRIu64 ",\"target_stable_id\":%" PRIu64,
      descriptor_set->abi_version, descriptor_set->generator_version,
      descriptor_set->stable_id, descriptor_set->target_stable_id));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"table_counts\":{\"descriptors\":%" PRIu32
      ",\"descriptor_refs\":%" PRIu32 ",\"asm_forms\":%" PRIu32
      ",\"asm_operand_indices\":%" PRIu32 ",\"asm_immediates\":%" PRIu32
      ",\"operands\":%" PRIu32 ",\"immediates\":%" PRIu32
      ",\"enum_domains\":%" PRIu32 ",\"enum_values\":%" PRIu32
      ",\"effects\":%" PRIu32 ",\"constraints\":%" PRIu32
      ",\"reg_classes\":%" PRIu32 ",\"operand_forms\":%" PRIu32
      ",\"operand_form_operand_indices\":%" PRIu32
      ",\"reg_class_alts\":%" PRIu32 ",\"schedule_classes\":%" PRIu32
      ",\"issue_uses\":%" PRIu32 ",\"resources\":%" PRIu32
      ",\"hazards\":%" PRIu32 ",\"pressure_deltas\":%" PRIu32
      ",\"feature_mask_words\":%" PRIu32 ",\"encoding_field_values\":%" PRIu32
      "}",
      descriptor_set->descriptor_count, descriptor_set->descriptor_ref_count,
      descriptor_set->asm_form_count, descriptor_set->asm_operand_index_count,
      descriptor_set->asm_immediate_count, descriptor_set->operand_count,
      descriptor_set->immediate_count, descriptor_set->enum_domain_count,
      descriptor_set->enum_value_count, descriptor_set->effect_count,
      descriptor_set->constraint_count, descriptor_set->reg_class_count,
      descriptor_set->operand_form_count,
      descriptor_set->operand_form_operand_index_count,
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
