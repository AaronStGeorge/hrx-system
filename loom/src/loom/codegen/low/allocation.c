// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/requirements.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

typedef struct loom_low_allocation_build_state_t {
  // Module containing the allocated low function.
  const loom_module_t* module;
  // Caller-provided allocation options.
  const loom_low_allocation_options_t* options;
  // Arena owning all sidecar arrays.
  iree_arena_allocator_t* arena;
  // Body region of the low function.
  loom_region_t* body;
  // Resolved target selected by the low function.
  loom_low_resolved_target_t target;
  // Liveness analysis for |body|.
  loom_liveness_analysis_t liveness;
  // Mutable assignment records being built.
  loom_low_allocation_assignment_t* assignments;
  // Mutable spill materialization plan records being built.
  loom_low_allocation_spill_plan_t* spill_plans;
  // Mutable remark records being built.
  loom_low_allocation_remark_t* remarks;
  // Mutable copy/coalescing decision records being built.
  loom_low_allocation_copy_decision_t* copy_decisions;
  // Number of initialized assignment records.
  iree_host_size_t assignment_count;
  // Number of initialized spill materialization plan records.
  iree_host_size_t spill_plan_count;
  // Number of initialized remark records.
  iree_host_size_t remark_count;
  // Number of initialized copy/coalescing decision records.
  iree_host_size_t copy_decision_count;
  // Number of spill-slot assignments.
  iree_host_size_t spill_count;
  // Number of coalesced copy decisions.
  iree_host_size_t coalesced_copy_count;
  // Number of materialized copy decisions.
  iree_host_size_t materialized_copy_count;
} loom_low_allocation_build_state_t;

typedef struct loom_low_allocation_class_capacity_t {
  // Location kind used for this class.
  loom_low_allocation_location_kind_t location_kind;
  // Maximum allocation units when |is_bounded| is true.
  uint32_t max_units;
  // Number of bits in one allocation unit.
  uint16_t alloc_unit_bits;
  // Storage space used when this class spills.
  loom_low_spill_slot_space_t spill_slot_space;
  // True when |max_units| is a hard allocation budget.
  bool is_bounded;
} loom_low_allocation_class_capacity_t;

static iree_string_view_t loom_low_allocation_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static bool loom_low_allocation_interval_overlaps(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs->start_point < rhs->end_point && rhs->start_point < lhs->end_point;
}

static bool loom_low_allocation_liveness_intervals_overlap(
    const loom_liveness_interval_t* lhs, const loom_liveness_interval_t* rhs) {
  return lhs->start_point < rhs->end_point && rhs->start_point < lhs->end_point;
}

static bool loom_low_allocation_location_is_register_like(
    loom_low_allocation_location_kind_t location_kind) {
  return location_kind == LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
         location_kind == LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;
}

static bool loom_low_allocation_is_power_of_two_u32(uint32_t value) {
  return value != 0 && (value & (value - 1u)) == 0;
}

static uint32_t loom_low_allocation_round_up_to_power_of_two_u32(
    uint32_t value) {
  if (value <= 1) {
    return 1;
  }
  --value;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return value == UINT32_MAX ? 0 : value + 1u;
}

static uint32_t loom_low_allocation_interval_alignment(
    const loom_liveness_interval_t* interval) {
  if (interval->unit_count <= 1 ||
      !loom_low_allocation_is_power_of_two_u32(interval->unit_count)) {
    return 1;
  }
  return interval->unit_count;
}

static bool loom_low_allocation_location_kind_is_known(
    loom_low_allocation_location_kind_t location_kind) {
  switch (location_kind) {
    case LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED:
    case LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER:
    case LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID:
    case LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT:
      return true;
    default:
      return false;
  }
}

static bool loom_low_allocation_remark_kind_is_known(
    loom_low_allocation_remark_kind_t remark_kind) {
  switch (remark_kind) {
    case LOOM_LOW_ALLOCATION_REMARK_UNKNOWN:
    case LOOM_LOW_ALLOCATION_REMARK_SPILL:
      return true;
    default:
      return false;
  }
}

static bool loom_low_allocation_copy_kind_is_known(
    loom_low_allocation_copy_kind_t copy_kind) {
  switch (copy_kind) {
    case LOOM_LOW_ALLOCATION_COPY_UNKNOWN:
    case LOOM_LOW_ALLOCATION_COPY_COALESCED:
    case LOOM_LOW_ALLOCATION_COPY_MATERIALIZED:
      return true;
    default:
      return false;
  }
}

static bool loom_low_allocation_location_ranges_overlap(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  const uint64_t lhs_end = (uint64_t)lhs->location_base + lhs->location_count;
  const uint64_t rhs_end = (uint64_t)rhs->location_base + rhs->location_count;
  return lhs->location_base < rhs_end && rhs->location_base < lhs_end;
}

static bool loom_low_allocation_location_range_overlaps(uint32_t lhs_base,
                                                        uint32_t lhs_count,
                                                        uint32_t rhs_base,
                                                        uint32_t rhs_count) {
  const uint64_t lhs_end = (uint64_t)lhs_base + lhs_count;
  const uint64_t rhs_end = (uint64_t)rhs_base + rhs_count;
  return lhs_base < rhs_end && rhs_base < lhs_end;
}

static bool loom_low_allocation_interval_is_allocatable(
    const loom_liveness_interval_t* interval) {
  return interval->value_class.type_kind == LOOM_TYPE_REGISTER &&
         interval->unit_count > 0;
}

static bool loom_low_allocation_mode_can_synthesize(uint8_t allocation_mode) {
  return allocation_mode == 0 || allocation_mode == LOOM_LOW_ALLOCATION_VIRTUAL;
}

static const char* loom_low_allocation_mode_name(uint8_t allocation_mode) {
  switch (allocation_mode) {
    case 0:
    case LOOM_LOW_ALLOCATION_VIRTUAL:
      return "virtual";
    case LOOM_LOW_ALLOCATION_ASSIGNED:
      return "assigned";
    case LOOM_LOW_ALLOCATION_FIXED:
      return "fixed";
    default:
      return "unknown";
  }
}

static iree_status_t loom_low_allocation_validate_synthesis_mode(
    const loom_op_t* low_func_op) {
  uint8_t allocation_mode = loom_low_func_def_allocation(low_func_op);
  if (loom_low_allocation_mode_can_synthesize(allocation_mode)) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "low allocation synthesis requires allocation(virtual), but function has "
      "allocation(%s)",
      loom_low_allocation_mode_name(allocation_mode));
}

static bool loom_low_allocation_interval_less(
    const loom_liveness_interval_t* lhs, const loom_liveness_interval_t* rhs) {
  if (lhs->start_point != rhs->start_point) {
    return lhs->start_point < rhs->start_point;
  }
  if (lhs->end_point != rhs->end_point) {
    return lhs->end_point < rhs->end_point;
  }
  return lhs->value_id < rhs->value_id;
}

static void loom_low_allocation_sort_intervals(
    const loom_liveness_interval_t** intervals, iree_host_size_t count) {
  for (iree_host_size_t i = 1; i < count; ++i) {
    const loom_liveness_interval_t* value = intervals[i];
    iree_host_size_t j = i;
    while (j > 0 &&
           loom_low_allocation_interval_less(value, intervals[j - 1])) {
      intervals[j] = intervals[j - 1];
      --j;
    }
    intervals[j] = value;
  }
}

static bool loom_low_allocation_budget_matches(
    const loom_module_t* module, loom_liveness_value_class_t value_class,
    iree_string_view_t register_class) {
  if (value_class.type_kind != LOOM_TYPE_REGISTER) {
    return false;
  }
  iree_string_view_t value_register_class =
      loom_low_allocation_module_string(module, value_class.register_class_id);
  return iree_string_view_equal(value_register_class, register_class);
}

static bool loom_low_allocation_register_class_matches(
    const loom_module_t* module, loom_liveness_value_class_t value_class,
    iree_string_view_t register_class) {
  return loom_low_allocation_budget_matches(module, value_class,
                                            register_class);
}

static bool loom_low_allocation_lookup_budget(
    const loom_low_allocation_build_state_t* state,
    loom_liveness_value_class_t value_class, uint32_t* out_max_units) {
  for (iree_host_size_t i = 0; i < state->options->budget_count; ++i) {
    const loom_low_allocation_budget_t* budget = &state->options->budgets[i];
    if (!loom_low_allocation_budget_matches(state->module, value_class,
                                            budget->register_class)) {
      continue;
    }
    *out_max_units = budget->max_units;
    return true;
  }
  return false;
}

static iree_status_t loom_low_allocation_lookup_descriptor_register_class(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t name,
    const loom_low_reg_class_t** out_reg_class, bool* out_found) {
  *out_reg_class = NULL;
  *out_found = false;
  if (!descriptor_set) {
    return iree_ok_status();
  }
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    const loom_low_reg_class_t* reg_class = &descriptor_set->reg_classes[i];
    iree_string_view_t reg_class_name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_low_descriptor_set_string(
        descriptor_set, reg_class->name_string_offset, &reg_class_name));
    if (iree_string_view_equal(reg_class_name, name)) {
      *out_reg_class = reg_class;
      *out_found = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_require_descriptor_register_class(
    const loom_low_allocation_build_state_t* state, iree_string_view_t name,
    const loom_low_reg_class_t** out_reg_class) {
  bool found_reg_class = false;
  IREE_RETURN_IF_ERROR(loom_low_allocation_lookup_descriptor_register_class(
      state->target.descriptor_set, name, out_reg_class, &found_reg_class));
  if (!found_reg_class) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation register class '%.*s' is not defined by descriptor set "
        "'%.*s'",
        (int)name.size, name.data, (int)state->target.descriptor_set_key.size,
        state->target.descriptor_set_key.data);
  }
  return iree_ok_status();
}

static loom_low_allocation_location_kind_t
loom_low_allocation_reg_class_location_kind(
    const loom_low_reg_class_t* reg_class) {
  if (reg_class->physical_count > 0 ||
      iree_any_bit_set(reg_class->flags, LOOM_LOW_REG_CLASS_FLAG_PHYSICAL)) {
    return LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
  }
  return LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;
}

static iree_status_t loom_low_allocation_class_capacity(
    const loom_low_allocation_build_state_t* state,
    loom_liveness_value_class_t value_class,
    loom_low_allocation_class_capacity_t* out_capacity) {
  iree_string_view_t register_class_name = loom_low_allocation_module_string(
      state->module, value_class.register_class_id);
  const loom_low_reg_class_t* reg_class = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_require_descriptor_register_class(
      state, register_class_name, &reg_class));

  uint32_t budget_units = 0;
  if (loom_low_allocation_lookup_budget(state, value_class, &budget_units)) {
    *out_capacity = (loom_low_allocation_class_capacity_t){
        .location_kind = loom_low_allocation_reg_class_location_kind(reg_class),
        .max_units = budget_units,
        .alloc_unit_bits = reg_class->alloc_unit_bits,
        .spill_slot_space =
            (loom_low_spill_slot_space_t)reg_class->spill_slot_space,
        .is_bounded = true,
    };
    return iree_ok_status();
  }

  if (reg_class->physical_count > 0) {
    *out_capacity = (loom_low_allocation_class_capacity_t){
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .max_units = reg_class->physical_count,
        .alloc_unit_bits = reg_class->alloc_unit_bits,
        .spill_slot_space =
            (loom_low_spill_slot_space_t)reg_class->spill_slot_space,
        .is_bounded = true,
    };
    return iree_ok_status();
  }

  *out_capacity = (loom_low_allocation_class_capacity_t){
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
      .max_units = UINT32_MAX,
      .alloc_unit_bits = reg_class->alloc_unit_bits,
      .spill_slot_space =
          (loom_low_spill_slot_space_t)reg_class->spill_slot_space,
      .is_bounded = false,
  };
  return iree_ok_status();
}

static bool loom_low_allocation_location_kind_is_register_owned(
    loom_low_allocation_location_kind_t location_kind) {
  return location_kind == LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
         location_kind == LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;
}

static iree_status_t loom_low_allocation_validate_location_range(
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, iree_string_view_t subject) {
  if (!loom_low_allocation_location_kind_is_register_owned(location_kind)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low allocation %.*s uses unsupported location kind %u",
        (int)subject.size, subject.data, (unsigned)location_kind);
  }
  if (location_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low allocation %.*s has an empty location range",
                            (int)subject.size, subject.data);
  }
  if ((uint64_t)location_base + location_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low allocation %.*s location range exceeds uint32_t",
        (int)subject.size, subject.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_validate_physical_bounds(
    const loom_low_reg_class_t* reg_class,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, iree_string_view_t subject) {
  const loom_low_allocation_location_kind_t expected_location_kind =
      loom_low_allocation_reg_class_location_kind(reg_class);
  if (location_kind != expected_location_kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low allocation %.*s location kind %u does not match register-class "
        "location kind %u",
        (int)subject.size, subject.data, (unsigned)location_kind,
        (unsigned)expected_location_kind);
  }
  if (reg_class->physical_count == 0) {
    return iree_ok_status();
  }
  const uint64_t location_end = (uint64_t)location_base + location_count;
  if (location_end > reg_class->physical_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low allocation %.*s location range exceeds register-class physical "
        "count %" PRIu16,
        (int)subject.size, subject.data, reg_class->physical_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_validate_budgets(
    const loom_low_allocation_build_state_t* state) {
  for (iree_host_size_t i = 0; i < state->options->budget_count; ++i) {
    const loom_low_allocation_budget_t* budget = &state->options->budgets[i];
    const loom_low_reg_class_t* reg_class = NULL;
    bool found_reg_class = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_lookup_descriptor_register_class(
        state->target.descriptor_set, budget->register_class, &reg_class,
        &found_reg_class));
    (void)reg_class;
    if (!found_reg_class) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown low allocation budget register class '%.*s' for descriptor "
          "set '%.*s'",
          (int)budget->register_class.size, budget->register_class.data,
          (int)state->target.descriptor_set_key.size,
          state->target.descriptor_set_key.data);
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(state->options->budgets[j].register_class,
                                 budget->register_class)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "duplicate low allocation budget register class '%.*s'",
            (int)budget->register_class.size, budget->register_class.data);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_validate_option_shapes(
    const loom_low_allocation_options_t* options) {
  if (options->budget_count > 0 && !options->budgets) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low allocation budgets are required when budget_count is non-zero");
  }
  if (options->fixed_value_count > 0 && !options->fixed_values) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low allocation fixed values are required when fixed_value_count is "
        "non-zero");
  }
  if (options->reserved_range_count > 0 && !options->reserved_ranges) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low allocation reserved ranges are required when reserved_range_count "
        "is non-zero");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_validate_reserved_ranges(
    const loom_low_allocation_build_state_t* state) {
  for (iree_host_size_t i = 0; i < state->options->reserved_range_count; ++i) {
    const loom_low_allocation_reserved_range_t* reserved_range =
        &state->options->reserved_ranges[i];
    if (iree_string_view_is_empty(reserved_range->register_class)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low allocation reserved range %zu has no register class", i);
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_validate_location_range(
        reserved_range->location_kind, reserved_range->location_base,
        reserved_range->location_count, IREE_SV("reserved range")));

    const loom_low_reg_class_t* reg_class = NULL;
    bool found_reg_class = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_lookup_descriptor_register_class(
        state->target.descriptor_set, reserved_range->register_class,
        &reg_class, &found_reg_class));
    if (!found_reg_class) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown low allocation reserved range register class '%.*s' for "
          "descriptor set '%.*s'",
          (int)reserved_range->register_class.size,
          reserved_range->register_class.data,
          (int)state->target.descriptor_set_key.size,
          state->target.descriptor_set_key.data);
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_validate_physical_bounds(
        reg_class, reserved_range->location_kind, reserved_range->location_base,
        reserved_range->location_count, IREE_SV("reserved range")));
    for (iree_host_size_t j = 0; j < i; ++j) {
      const loom_low_allocation_reserved_range_t* existing =
          &state->options->reserved_ranges[j];
      if (reserved_range->location_kind != existing->location_kind ||
          !iree_string_view_equal(reserved_range->register_class,
                                  existing->register_class)) {
        continue;
      }
      if (loom_low_allocation_location_range_overlaps(
              reserved_range->location_base, reserved_range->location_count,
              existing->location_base, existing->location_count)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low allocation reserved ranges %zu and %zu overlap", j, i);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_validate_fixed_values(
    const loom_low_allocation_build_state_t* state) {
  for (iree_host_size_t i = 0; i < state->options->fixed_value_count; ++i) {
    const loom_low_allocation_fixed_value_t* fixed_value =
        &state->options->fixed_values[i];
    if (fixed_value->value_id >= state->module->values.count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low allocation fixed value %zu references out-of-range value %u", i,
          (unsigned)fixed_value->value_id);
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_validate_location_range(
        fixed_value->location_kind, fixed_value->location_base,
        fixed_value->location_count, IREE_SV("fixed value")));

    const loom_liveness_interval_t* interval = loom_liveness_interval_for_value(
        &state->liveness, fixed_value->value_id);
    if (interval == NULL ||
        !loom_low_allocation_interval_is_allocatable(interval)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low allocation fixed value %u has no allocatable live interval",
          (unsigned)fixed_value->value_id);
    }
    if (fixed_value->location_count != interval->unit_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low allocation fixed value %u requires %u unit(s) but location has "
          "%u",
          (unsigned)fixed_value->value_id, interval->unit_count,
          fixed_value->location_count);
    }
    const uint32_t alignment = loom_low_allocation_interval_alignment(interval);
    if (fixed_value->location_base % alignment != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low allocation fixed value %u location base %u is not aligned to "
          "%u",
          (unsigned)fixed_value->value_id, fixed_value->location_base,
          alignment);
    }

    iree_string_view_t register_class_name = loom_low_allocation_module_string(
        state->module, interval->value_class.register_class_id);
    const loom_low_reg_class_t* reg_class = NULL;
    IREE_RETURN_IF_ERROR(loom_low_allocation_require_descriptor_register_class(
        state, register_class_name, &reg_class));
    IREE_RETURN_IF_ERROR(loom_low_allocation_validate_physical_bounds(
        reg_class, fixed_value->location_kind, fixed_value->location_base,
        fixed_value->location_count, IREE_SV("fixed value")));
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (state->options->fixed_values[j].value_id == fixed_value->value_id) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate low allocation fixed value %u",
                                (unsigned)fixed_value->value_id);
      }
    }
  }
  return iree_ok_status();
}

static const loom_low_allocation_fixed_value_t*
loom_low_allocation_fixed_value_for_value(
    const loom_low_allocation_build_state_t* state, loom_value_id_t value_id) {
  for (iree_host_size_t i = 0; i < state->options->fixed_value_count; ++i) {
    const loom_low_allocation_fixed_value_t* fixed_value =
        &state->options->fixed_values[i];
    if (fixed_value->value_id == value_id) {
      return fixed_value;
    }
  }
  return NULL;
}

static bool loom_low_allocation_fixed_value_conflicts(
    const loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, loom_value_id_t ignored_value_id) {
  for (iree_host_size_t i = 0; i < state->options->fixed_value_count; ++i) {
    const loom_low_allocation_fixed_value_t* fixed_value =
        &state->options->fixed_values[i];
    if (fixed_value->value_id == ignored_value_id) {
      continue;
    }
    if (fixed_value->location_kind != location_kind) {
      continue;
    }
    if (!loom_low_allocation_location_range_overlaps(
            fixed_value->location_base, fixed_value->location_count,
            location_base, location_count)) {
      continue;
    }
    const loom_liveness_interval_t* fixed_interval =
        loom_liveness_interval_for_value(&state->liveness,
                                         fixed_value->value_id);
    if (!fixed_interval ||
        !loom_liveness_value_class_equal(fixed_interval->value_class,
                                         interval->value_class)) {
      continue;
    }
    if (loom_low_allocation_liveness_intervals_overlap(fixed_interval,
                                                       interval)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_reserved_range_conflicts(
    const loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count) {
  for (iree_host_size_t i = 0; i < state->options->reserved_range_count; ++i) {
    const loom_low_allocation_reserved_range_t* reserved_range =
        &state->options->reserved_ranges[i];
    if (reserved_range->location_kind != location_kind) {
      continue;
    }
    if (!loom_low_allocation_register_class_matches(
            state->module, interval->value_class,
            reserved_range->register_class)) {
      continue;
    }
    if (loom_low_allocation_location_range_overlaps(
            reserved_range->location_base, reserved_range->location_count,
            location_base, location_count)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_candidate_conflicts(
    const loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, loom_value_id_t ignored_value_id) {
  loom_low_allocation_assignment_t candidate = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .start_point = interval->start_point,
      .end_point = interval->end_point,
      .unit_count = interval->unit_count,
      .location_kind = location_kind,
      .location_base = location_base,
      .location_count = location_count,
  };
  for (iree_host_size_t i = 0; i < state->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* existing = &state->assignments[i];
    if (existing->value_id == ignored_value_id) {
      continue;
    }
    if (existing->location_kind != location_kind) {
      continue;
    }
    if (!loom_liveness_value_class_equal(existing->value_class,
                                         interval->value_class)) {
      continue;
    }
    if (!loom_low_allocation_interval_overlaps(existing, &candidate)) {
      continue;
    }
    if (loom_low_allocation_location_ranges_overlap(existing, &candidate)) {
      return true;
    }
  }
  if (loom_low_allocation_fixed_value_conflicts(state, interval, location_kind,
                                                location_base, location_count,
                                                ignored_value_id)) {
    return true;
  }
  if (loom_low_allocation_reserved_range_conflicts(
          state, interval, location_kind, location_base, location_count)) {
    return true;
  }
  return false;
}

static uint32_t loom_low_allocation_unbounded_search_limit(
    const loom_low_allocation_build_state_t* state,
    loom_liveness_value_class_t value_class,
    loom_low_allocation_location_kind_t location_kind) {
  uint32_t max_end = 0;
  for (iree_host_size_t i = 0; i < state->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment = &state->assignments[i];
    if (assignment->location_kind != location_kind) {
      continue;
    }
    if (!loom_liveness_value_class_equal(assignment->value_class,
                                         value_class)) {
      continue;
    }
    uint64_t assignment_end =
        (uint64_t)assignment->location_base + assignment->location_count;
    if (assignment_end > UINT32_MAX) {
      return UINT32_MAX;
    }
    if ((uint32_t)assignment_end > max_end) {
      max_end = (uint32_t)assignment_end;
    }
  }
  for (iree_host_size_t i = 0; i < state->options->fixed_value_count; ++i) {
    const loom_low_allocation_fixed_value_t* fixed_value =
        &state->options->fixed_values[i];
    if (fixed_value->location_kind != location_kind) {
      continue;
    }
    const loom_liveness_interval_t* fixed_interval =
        loom_liveness_interval_for_value(&state->liveness,
                                         fixed_value->value_id);
    if (!fixed_interval || !loom_liveness_value_class_equal(
                               fixed_interval->value_class, value_class)) {
      continue;
    }
    uint64_t fixed_value_end =
        (uint64_t)fixed_value->location_base + fixed_value->location_count;
    if (fixed_value_end > UINT32_MAX) {
      return UINT32_MAX;
    }
    if ((uint32_t)fixed_value_end > max_end) {
      max_end = (uint32_t)fixed_value_end;
    }
  }
  for (iree_host_size_t i = 0; i < state->options->reserved_range_count; ++i) {
    const loom_low_allocation_reserved_range_t* reserved_range =
        &state->options->reserved_ranges[i];
    if (reserved_range->location_kind != location_kind) {
      continue;
    }
    if (!loom_low_allocation_register_class_matches(
            state->module, value_class, reserved_range->register_class)) {
      continue;
    }
    uint64_t reserved_range_end = (uint64_t)reserved_range->location_base +
                                  reserved_range->location_count;
    if (reserved_range_end > UINT32_MAX) {
      return UINT32_MAX;
    }
    if ((uint32_t)reserved_range_end > max_end) {
      max_end = (uint32_t)reserved_range_end;
    }
  }
  return max_end;
}

static bool loom_low_allocation_find_location(
    const loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_low_allocation_class_capacity_t capacity, uint32_t* out_base) {
  if (capacity.is_bounded && interval->unit_count > capacity.max_units) {
    return false;
  }

  uint32_t last_base = 0;
  if (capacity.is_bounded) {
    last_base = capacity.max_units - interval->unit_count;
  } else {
    last_base = loom_low_allocation_unbounded_search_limit(
        state, interval->value_class, capacity.location_kind);
  }

  const uint32_t alignment = loom_low_allocation_interval_alignment(interval);
  for (uint32_t base = 0; base <= last_base;) {
    if (!loom_low_allocation_candidate_conflicts(
            state, interval, capacity.location_kind, base, interval->unit_count,
            LOOM_VALUE_ID_INVALID)) {
      *out_base = base;
      return true;
    }
    if (base > UINT32_MAX - alignment) {
      break;
    }
    base += alignment;
  }
  return false;
}

static void loom_low_allocation_record_spill_remark(
    loom_low_allocation_build_state_t* state, uint32_t assignment_index,
    uint32_t budget_units, uint32_t required_units) {
  state->remarks[state->remark_count++] = (loom_low_allocation_remark_t){
      .kind = LOOM_LOW_ALLOCATION_REMARK_SPILL,
      .assignment_index = assignment_index,
      .budget_units = budget_units,
      .required_units = required_units,
  };
}

static iree_status_t loom_low_allocation_spill_plan_layout(
    const loom_low_allocation_assignment_t* assignment,
    uint16_t alloc_unit_bits, uint32_t* out_byte_size,
    uint32_t* out_byte_alignment) {
  if (alloc_unit_bits == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot plan spill for register class with zero allocation unit bits");
  }
  uint64_t bit_size = (uint64_t)assignment->unit_count * alloc_unit_bits;
  uint64_t byte_size = (bit_size + 7u) / 8u;
  if (byte_size > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "spill slot byte size exceeds uint32_t");
  }
  uint32_t unit_byte_size = ((uint32_t)alloc_unit_bits + 7u) / 8u;
  uint32_t byte_alignment =
      loom_low_allocation_round_up_to_power_of_two_u32(unit_byte_size);
  if (byte_alignment == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "spill slot byte alignment exceeds uint32_t");
  }
  *out_byte_size = (uint32_t)byte_size;
  *out_byte_alignment = byte_alignment;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_record_spill_plan(
    loom_low_allocation_build_state_t* state, uint32_t assignment_index,
    uint16_t alloc_unit_bits, loom_low_spill_slot_space_t spill_slot_space) {
  const loom_low_allocation_assignment_t* assignment =
      &state->assignments[assignment_index];
  uint32_t byte_size = 0;
  uint32_t byte_alignment = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_layout(
      assignment, alloc_unit_bits, &byte_size, &byte_alignment));

  if (assignment->value_id >= state->module->values.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot plan spill for out-of-range value %u",
                            (unsigned)assignment->value_id);
  }
  uint32_t reload_count =
      loom_module_value(state->module, assignment->value_id)->use_count;
  state->spill_plans[state->spill_plan_count++] =
      (loom_low_allocation_spill_plan_t){
          .value_id = assignment->value_id,
          .assignment_index = assignment_index,
          .slot_index = assignment->location_base,
          .slot_space = spill_slot_space,
          .byte_size = byte_size,
          .byte_alignment = byte_alignment,
          .store_count = reload_count > 0 ? 1u : 0u,
          .reload_count = reload_count,
      };
  return iree_ok_status();
}

static bool loom_low_allocation_assignment_locations_equal(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         loom_liveness_value_class_equal(lhs->value_class, rhs->value_class) &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

static bool loom_low_allocation_assignment_is_coalescable(
    const loom_low_allocation_assignment_t* assignment) {
  return loom_low_allocation_location_is_register_like(
      assignment->location_kind);
}

static iree_status_t loom_low_allocation_assignment_index_for_value(
    const loom_low_allocation_build_state_t* state, loom_value_id_t value_id,
    uint32_t* out_assignment_index) {
  for (iree_host_size_t i = 0; i < state->assignment_count; ++i) {
    if (state->assignments[i].value_id == value_id) {
      if (i > UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "allocation assignment index exceeds uint32_t");
      }
      *out_assignment_index = (uint32_t)i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "low allocation references value %u without an "
                          "assignment",
                          (unsigned)value_id);
}

static iree_status_t loom_low_allocation_tied_operand_for_result(
    const loom_region_t* body, loom_value_id_t result_id, bool* out_has_tie,
    loom_value_id_t* out_operand_id) {
  *out_has_tie = false;
  *out_operand_id = LOOM_VALUE_ID_INVALID;
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (op->tied_result_count == 0) {
        continue;
      }
      const loom_value_id_t* results = loom_op_const_results(op);
      const loom_value_id_t* operands = loom_op_const_operands(op);
      const loom_tied_result_t* tied_results = loom_op_tied_results(op);
      for (uint16_t i = 0; i < op->tied_result_count; ++i) {
        const loom_tied_result_t tied = tied_results[i];
        if (tied.result_index >= op->result_count ||
            tied.operand_index >= op->operand_count) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "low allocation saw malformed tied result metadata");
        }
        if (results[tied.result_index] != result_id) {
          continue;
        }
        if (*out_has_tie) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "low allocation result value has multiple tied operands");
        }
        if (tied.has_type_change) {
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "low allocation requires explicit materialization for "
              "type-changing tied results");
        }
        *out_has_tie = true;
        *out_operand_id = operands[tied.operand_index];
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_assign_tied_interval(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  bool has_tie = false;
  loom_value_id_t tied_operand_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_allocation_tied_operand_for_result(
      state->body, interval->value_id, &has_tie, &tied_operand_id));
  if (!has_tie) {
    return iree_ok_status();
  }

  uint32_t operand_assignment_index = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
      state, tied_operand_id, &operand_assignment_index));
  const loom_low_allocation_assignment_t* operand_assignment =
      &state->assignments[operand_assignment_index];
  if (!loom_low_allocation_assignment_is_coalescable(operand_assignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low tied result operand is not assigned to a register-like location");
  }
  if (!loom_liveness_value_class_equal(operand_assignment->value_class,
                                       interval->value_class) ||
      operand_assignment->location_count != interval->unit_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low tied result does not match operand allocation class");
  }
  if (loom_low_allocation_candidate_conflicts(
          state, interval, operand_assignment->location_kind,
          operand_assignment->location_base, operand_assignment->location_count,
          tied_operand_id)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low tied result cannot share the operand location without "
        "overlapping another live interval");
  }

  state->assignments[state->assignment_count++] =
      (loom_low_allocation_assignment_t){
          .value_id = interval->value_id,
          .value_class = interval->value_class,
          .start_point = interval->start_point,
          .end_point = interval->end_point,
          .unit_count = interval->unit_count,
          .location_kind = operand_assignment->location_kind,
          .location_base = operand_assignment->location_base,
          .location_count = operand_assignment->location_count,
      };
  *out_assigned = true;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_assign_fixed_interval(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  const loom_low_allocation_fixed_value_t* fixed_value =
      loom_low_allocation_fixed_value_for_value(state, interval->value_id);
  if (!fixed_value) {
    return iree_ok_status();
  }
  if (state->assignment_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation sidecar exceeds uint32_t range");
  }
  if (loom_low_allocation_candidate_conflicts(
          state, interval, fixed_value->location_kind,
          fixed_value->location_base, fixed_value->location_count,
          interval->value_id)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low fixed value cannot occupy its required location without "
        "overlapping another live interval or reserved range");
  }

  state->assignments[state->assignment_count++] =
      (loom_low_allocation_assignment_t){
          .value_id = interval->value_id,
          .value_class = interval->value_class,
          .start_point = interval->start_point,
          .end_point = interval->end_point,
          .unit_count = interval->unit_count,
          .location_kind = fixed_value->location_kind,
          .location_base = fixed_value->location_base,
          .location_count = fixed_value->location_count,
      };
  *out_assigned = true;
  return iree_ok_status();
}

static iree_host_size_t loom_low_allocation_count_copy_ops(
    const loom_region_t* body) {
  iree_host_size_t copy_count = 0;
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_copy_isa(op)) ++copy_count;
    }
  }
  return copy_count;
}

static iree_status_t loom_low_allocation_record_copy_decision(
    loom_low_allocation_build_state_t* state, const loom_op_t* op) {
  uint32_t source_assignment_index = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
      state, loom_low_copy_source(op), &source_assignment_index));
  uint32_t result_assignment_index = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
      state, loom_low_copy_result(op), &result_assignment_index));

  const loom_low_allocation_assignment_t* source_assignment =
      &state->assignments[source_assignment_index];
  const loom_low_allocation_assignment_t* result_assignment =
      &state->assignments[result_assignment_index];
  const bool coalesced =
      loom_low_allocation_assignment_is_coalescable(source_assignment) &&
      loom_low_allocation_assignment_is_coalescable(result_assignment) &&
      loom_low_allocation_assignment_locations_equal(source_assignment,
                                                     result_assignment);
  state->copy_decisions[state->copy_decision_count++] =
      (loom_low_allocation_copy_decision_t){
          .source_value_id = loom_low_copy_source(op),
          .result_value_id = loom_low_copy_result(op),
          .source_assignment_index = source_assignment_index,
          .result_assignment_index = result_assignment_index,
          .kind = coalesced ? LOOM_LOW_ALLOCATION_COPY_COALESCED
                            : LOOM_LOW_ALLOCATION_COPY_MATERIALIZED,
      };
  if (coalesced) {
    ++state->coalesced_copy_count;
  } else {
    ++state->materialized_copy_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_record_copy_decisions(
    loom_low_allocation_build_state_t* state) {
  iree_host_size_t copy_count = loom_low_allocation_count_copy_ops(state->body);
  if (copy_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, copy_count, sizeof(*state->copy_decisions),
      (void**)&state->copy_decisions));
  memset(state->copy_decisions, 0, copy_count * sizeof(*state->copy_decisions));

  loom_block_t* block = NULL;
  loom_region_for_each_block(state->body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_copy_isa(op)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_allocation_record_copy_decision(state, op));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_assign_intervals(
    loom_low_allocation_build_state_t* state) {
  iree_host_size_t allocatable_count = 0;
  for (iree_host_size_t i = 0; i < state->liveness.interval_count; ++i) {
    if (loom_low_allocation_interval_is_allocatable(
            &state->liveness.intervals[i])) {
      ++allocatable_count;
    }
  }
  if (allocatable_count == 0) {
    return iree_ok_status();
  }

  const loom_liveness_interval_t** intervals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, allocatable_count, sizeof(*intervals), (void**)&intervals));
  state->assignments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, allocatable_count, sizeof(*state->assignments),
      (void**)&state->assignments));
  memset(state->assignments, 0,
         allocatable_count * sizeof(*state->assignments));
  state->spill_plans = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, allocatable_count, sizeof(*state->spill_plans),
      (void**)&state->spill_plans));
  memset(state->spill_plans, 0,
         allocatable_count * sizeof(*state->spill_plans));
  state->remarks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, allocatable_count, sizeof(*state->remarks),
      (void**)&state->remarks));
  memset(state->remarks, 0, allocatable_count * sizeof(*state->remarks));

  iree_host_size_t interval_index = 0;
  for (iree_host_size_t i = 0; i < state->liveness.interval_count; ++i) {
    const loom_liveness_interval_t* interval = &state->liveness.intervals[i];
    if (loom_low_allocation_interval_is_allocatable(interval)) {
      intervals[interval_index++] = interval;
    }
  }
  loom_low_allocation_sort_intervals(intervals, allocatable_count);

  for (iree_host_size_t i = 0; i < allocatable_count; ++i) {
    const loom_liveness_interval_t* interval = intervals[i];
    bool assigned_fixed_interval = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_assign_fixed_interval(
        state, interval, &assigned_fixed_interval));
    if (assigned_fixed_interval) {
      continue;
    }

    bool assigned_tied_interval = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_assign_tied_interval(
        state, interval, &assigned_tied_interval));
    if (assigned_tied_interval) {
      continue;
    }

    loom_low_allocation_class_capacity_t capacity = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_class_capacity(
        state, interval->value_class, &capacity));
    if (state->assignment_count > UINT32_MAX ||
        state->spill_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "allocation sidecar exceeds uint32_t range");
    }

    uint32_t location_base = 0;
    bool assigned = loom_low_allocation_find_location(state, interval, capacity,
                                                      &location_base);

    loom_low_allocation_assignment_t* assignment =
        &state->assignments[state->assignment_count];
    *assignment = (loom_low_allocation_assignment_t){
        .value_id = interval->value_id,
        .value_class = interval->value_class,
        .start_point = interval->start_point,
        .end_point = interval->end_point,
        .unit_count = interval->unit_count,
        .location_kind = assigned ? capacity.location_kind
                                  : LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
        .location_base =
            assigned ? location_base : (uint32_t)state->spill_count,
        .location_count = interval->unit_count,
    };

    uint32_t assignment_index = (uint32_t)state->assignment_count++;
    if (!assigned) {
      ++state->spill_count;
      IREE_RETURN_IF_ERROR(loom_low_allocation_record_spill_plan(
          state, assignment_index, capacity.alloc_unit_bits,
          capacity.spill_slot_space));
      loom_low_allocation_record_spill_remark(
          state, assignment_index,
          capacity.is_bounded ? capacity.max_units : UINT32_MAX,
          interval->unit_count);
    }
  }
  return iree_ok_status();
}

static bool loom_low_allocation_assignment_pair_conflicts(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  if (!loom_low_allocation_location_is_register_like(lhs->location_kind) ||
      !loom_low_allocation_location_is_register_like(rhs->location_kind)) {
    return false;
  }
  if (lhs->location_kind != rhs->location_kind) {
    return false;
  }
  if (!loom_liveness_value_class_equal(lhs->value_class, rhs->value_class)) {
    return false;
  }
  return loom_low_allocation_interval_overlaps(lhs, rhs) &&
         loom_low_allocation_location_ranges_overlap(lhs, rhs);
}

static iree_status_t loom_low_allocation_assignment_index_in_sidecar(
    const loom_low_allocation_sidecar_t* sidecar, loom_value_id_t value_id,
    uint32_t* out_assignment_index) {
  for (iree_host_size_t i = 0; i < sidecar->assignment_count; ++i) {
    if (sidecar->assignments[i].value_id == value_id) {
      if (i > UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "allocation assignment index exceeds uint32_t");
      }
      *out_assignment_index = (uint32_t)i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "allocation sidecar is missing assignment for value "
                          "%u",
                          (unsigned)value_id);
}

static iree_status_t loom_low_allocation_tied_values_for_op(
    const loom_op_t* op, loom_tied_result_t tied,
    loom_value_id_t* out_result_id, loom_value_id_t* out_operand_id) {
  if (tied.result_index >= op->result_count ||
      tied.operand_index >= op->operand_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation sidecar saw malformed tied result "
                            "metadata");
  }
  if (tied.has_type_change) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "allocation sidecar requires explicit materialization for "
        "type-changing tied results");
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  *out_result_id = results[tied.result_index];
  *out_operand_id = operands[tied.operand_index];
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_values_are_tied(
    const loom_region_t* body, loom_value_id_t lhs_value_id,
    loom_value_id_t rhs_value_id, bool* out_are_tied) {
  *out_are_tied = false;
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_tied_result_t* tied_results = loom_op_tied_results(op);
      for (uint16_t i = 0; i < op->tied_result_count; ++i) {
        loom_value_id_t result_id = LOOM_VALUE_ID_INVALID;
        loom_value_id_t operand_id = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_low_allocation_tied_values_for_op(
            op, tied_results[i], &result_id, &operand_id));
        if ((result_id == lhs_value_id && operand_id == rhs_value_id) ||
            (result_id == rhs_value_id && operand_id == lhs_value_id)) {
          *out_are_tied = true;
          return iree_ok_status();
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_tied_result_assignments(
    const loom_low_allocation_sidecar_t* sidecar, const loom_region_t* body) {
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_tied_result_t* tied_results = loom_op_tied_results(op);
      for (uint16_t i = 0; i < op->tied_result_count; ++i) {
        loom_value_id_t result_id = LOOM_VALUE_ID_INVALID;
        loom_value_id_t operand_id = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_low_allocation_tied_values_for_op(
            op, tied_results[i], &result_id, &operand_id));

        uint32_t result_assignment_index = 0;
        IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_in_sidecar(
            sidecar, result_id, &result_assignment_index));
        uint32_t operand_assignment_index = 0;
        IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_in_sidecar(
            sidecar, operand_id, &operand_assignment_index));

        const loom_low_allocation_assignment_t* result_assignment =
            &sidecar->assignments[result_assignment_index];
        const loom_low_allocation_assignment_t* operand_assignment =
            &sidecar->assignments[operand_assignment_index];
        if (!loom_low_allocation_assignment_is_coalescable(result_assignment) ||
            !loom_low_allocation_assignment_is_coalescable(
                operand_assignment) ||
            !loom_low_allocation_assignment_locations_equal(
                result_assignment, operand_assignment)) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "allocation tied result does not share the operand location");
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_assignment(
    const loom_low_allocation_assignment_t* assignment,
    iree_host_size_t assignment_index) {
  if (!loom_low_allocation_location_kind_is_known(assignment->location_kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation assignment %zu has unknown location "
                            "kind %u",
                            assignment_index,
                            (unsigned)assignment->location_kind);
  }
  if (assignment->location_kind != LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED &&
      assignment->location_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation assignment %zu has an empty assigned location",
        assignment_index);
  }
  uint64_t location_end =
      (uint64_t)assignment->location_base + assignment->location_count;
  if (location_end > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation assignment %zu location range exceeds uint32_t",
        assignment_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_remark(
    const loom_low_allocation_remark_t* remark, iree_host_size_t remark_index,
    iree_host_size_t assignment_count) {
  if (!loom_low_allocation_remark_kind_is_known(remark->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation remark %zu has unknown kind %u",
                            remark_index, (unsigned)remark->kind);
  }
  if (remark->kind == LOOM_LOW_ALLOCATION_REMARK_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation remark %zu has no kind", remark_index);
  }
  if (remark->assignment_index >= assignment_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation remark %zu references assignment %u, but sidecar has only "
        "%zu assignments",
        remark_index, remark->assignment_index, assignment_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_spill_plan(
    const loom_low_allocation_sidecar_t* sidecar,
    const loom_low_allocation_spill_plan_t* spill_plan,
    iree_host_size_t spill_plan_index) {
  if (spill_plan->assignment_index >= sidecar->assignment_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation spill plan %zu references assignment %u, but sidecar has "
        "only %zu assignments",
        spill_plan_index, spill_plan->assignment_index,
        sidecar->assignment_count);
  }
  const loom_low_allocation_assignment_t* assignment =
      &sidecar->assignments[spill_plan->assignment_index];
  if (spill_plan->value_id >= sidecar->module->values.count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation spill plan %zu references out-of-range value %u",
        spill_plan_index, (unsigned)spill_plan->value_id);
  }
  if (assignment->location_kind != LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan %zu references a non-spill assignment",
        spill_plan_index);
  }
  if (assignment->value_id != spill_plan->value_id ||
      assignment->location_base != spill_plan->slot_index) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan %zu does not match referenced assignment",
        spill_plan_index);
  }
  if (!loom_low_spill_slot_space_is_valid(spill_plan->slot_space)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation spill plan %zu has unknown spill slot space %u",
        spill_plan_index, (unsigned)spill_plan->slot_space);
  }
  if (spill_plan->byte_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation spill plan %zu has empty byte size",
                            spill_plan_index);
  }
  if (!loom_low_allocation_is_power_of_two_u32(spill_plan->byte_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation spill plan %zu has non-power-of-two byte alignment",
        spill_plan_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_copy_decision(
    const loom_low_allocation_sidecar_t* sidecar,
    const loom_low_allocation_copy_decision_t* copy_decision,
    iree_host_size_t copy_decision_index) {
  if (!loom_low_allocation_copy_kind_is_known(copy_decision->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation copy decision %zu has unknown kind %u",
                            copy_decision_index, (unsigned)copy_decision->kind);
  }
  if (copy_decision->kind == LOOM_LOW_ALLOCATION_COPY_UNKNOWN) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation copy decision %zu has no kind",
                            copy_decision_index);
  }
  if (copy_decision->source_assignment_index >= sidecar->assignment_count ||
      copy_decision->result_assignment_index >= sidecar->assignment_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation copy decision %zu references assignments %u and %u, but "
        "sidecar has only %zu assignments",
        copy_decision_index, copy_decision->source_assignment_index,
        copy_decision->result_assignment_index, sidecar->assignment_count);
  }
  const loom_low_allocation_assignment_t* source_assignment =
      &sidecar->assignments[copy_decision->source_assignment_index];
  const loom_low_allocation_assignment_t* result_assignment =
      &sidecar->assignments[copy_decision->result_assignment_index];
  if (source_assignment->value_id != copy_decision->source_value_id ||
      result_assignment->value_id != copy_decision->result_value_id) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation copy decision %zu value ids do not match referenced "
        "assignments",
        copy_decision_index);
  }
  const bool locations_equal = loom_low_allocation_assignment_locations_equal(
      source_assignment, result_assignment);
  if (copy_decision->kind == LOOM_LOW_ALLOCATION_COPY_COALESCED &&
      (!loom_low_allocation_assignment_is_coalescable(source_assignment) ||
       !loom_low_allocation_assignment_is_coalescable(result_assignment) ||
       !locations_equal)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation copy decision %zu is coalesced but assignments differ",
        copy_decision_index);
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_verify_sidecar(
    const loom_low_allocation_sidecar_t* sidecar) {
  if (!sidecar) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation sidecar is required");
  }
  if (!sidecar->module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation sidecar module is required");
  }
  if (!sidecar->function_op || !loom_low_func_def_isa(sidecar->function_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation sidecar low.func.def is required");
  }
  loom_region_t* body = loom_low_func_def_body(sidecar->function_op);
  if (!body) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation sidecar low.func.def body is required");
  }
  if (sidecar->assignment_count > 0 && !sidecar->assignments) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation sidecar assignments are required");
  }
  if (sidecar->spill_plan_count > 0 && !sidecar->spill_plans) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation sidecar spill plans are required");
  }
  if (sidecar->remark_count > 0 && !sidecar->remarks) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation sidecar remarks are required");
  }
  if (sidecar->copy_decision_count > 0 && !sidecar->copy_decisions) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation sidecar copy decisions are required");
  }
  iree_host_size_t spill_count = 0;
  for (iree_host_size_t i = 0; i < sidecar->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* lhs = &sidecar->assignments[i];
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_assignment(lhs, i));
    if (lhs->location_kind == LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT) {
      ++spill_count;
    }
    for (iree_host_size_t j = i + 1; j < sidecar->assignment_count; ++j) {
      const loom_low_allocation_assignment_t* rhs = &sidecar->assignments[j];
      if (!loom_low_allocation_assignment_pair_conflicts(lhs, rhs)) {
        continue;
      }
      bool pair_is_tied = false;
      IREE_RETURN_IF_ERROR(loom_low_allocation_values_are_tied(
          body, lhs->value_id, rhs->value_id, &pair_is_tied));
      if (pair_is_tied) {
        continue;
      }
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "allocation assigns overlapping live intervals to the same location");
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_verify_tied_result_assignments(sidecar, body));
  for (iree_host_size_t i = 0; i < sidecar->remark_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_remark(
        &sidecar->remarks[i], i, sidecar->assignment_count));
  }
  for (iree_host_size_t i = 0; i < sidecar->spill_plan_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_spill_plan(
        sidecar, &sidecar->spill_plans[i], i));
  }
  iree_host_size_t coalesced_copy_count = 0;
  iree_host_size_t materialized_copy_count = 0;
  for (iree_host_size_t i = 0; i < sidecar->copy_decision_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_copy_decision(
        sidecar, &sidecar->copy_decisions[i], i));
    switch (sidecar->copy_decisions[i].kind) {
      case LOOM_LOW_ALLOCATION_COPY_COALESCED:
        ++coalesced_copy_count;
        break;
      case LOOM_LOW_ALLOCATION_COPY_MATERIALIZED:
        ++materialized_copy_count;
        break;
      default:
        break;
    }
  }
  iree_host_size_t expected_copy_decision_count =
      loom_low_allocation_count_copy_ops(body);
  if (sidecar->copy_decision_count != expected_copy_decision_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation sidecar has %zu copy decisions for %zu low.copy ops",
        sidecar->copy_decision_count, expected_copy_decision_count);
  }
  if (sidecar->spill_count != spill_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation sidecar spill_count is %zu but assignments contain %zu "
        "spills",
        sidecar->spill_count, spill_count);
  }
  if (sidecar->spill_plan_count != spill_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation sidecar has %zu spill plans for %zu spilled assignments",
        sidecar->spill_plan_count, spill_count);
  }
  if (sidecar->coalesced_copy_count != coalesced_copy_count ||
      sidecar->materialized_copy_count != materialized_copy_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation sidecar copy counters do not match copy decisions");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_emit_predicted_spills(
    const loom_low_allocation_sidecar_t* sidecar,
    iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < sidecar->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan =
        &sidecar->spill_plans[i];
    const loom_low_allocation_assignment_t* assignment =
        &sidecar->assignments[spill_plan->assignment_index];
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_diagnostic_target_key(&sidecar->target)),
        loom_param_string(loom_low_diagnostic_export_name(&sidecar->target)),
        loom_param_string(loom_low_diagnostic_config_key(&sidecar->target)),
        loom_param_string(loom_low_diagnostic_function_name(
            sidecar->module, sidecar->function_op)),
        loom_param_string(loom_low_diagnostic_value_name(sidecar->module,
                                                         spill_plan->value_id)),
        loom_param_string(loom_low_diagnostic_value_class_name(
            sidecar->module, assignment->value_class)),
        loom_param_u32(spill_plan->byte_size),
        loom_param_u32(spill_plan->store_count),
        loom_param_u32(spill_plan->reload_count),
        loom_param_string(IREE_SV(
            "no available non-overlapping location within register budget")),
    };
    loom_diagnostic_emission_t emission = {
        .op = loom_low_diagnostic_value_origin_op(
            sidecar->module, spill_plan->value_id, sidecar->function_op),
        .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 8),
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    IREE_RETURN_IF_ERROR(iree_diagnostic_emit(emitter, &emission));
  }
  return iree_ok_status();
}

static iree_string_view_t loom_low_allocation_copy_decision_name(
    loom_low_allocation_copy_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_ALLOCATION_COPY_COALESCED:
      return IREE_SV("accepted");
    case LOOM_LOW_ALLOCATION_COPY_MATERIALIZED:
      return IREE_SV("rejected");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_low_allocation_copy_decision_reason(
    const loom_low_allocation_sidecar_t* sidecar,
    const loom_low_allocation_copy_decision_t* copy_decision) {
  const loom_low_allocation_assignment_t* source_assignment =
      &sidecar->assignments[copy_decision->source_assignment_index];
  const loom_low_allocation_assignment_t* result_assignment =
      &sidecar->assignments[copy_decision->result_assignment_index];
  if (!loom_low_allocation_assignment_is_coalescable(source_assignment)) {
    return IREE_SV("source value is not assigned to a register-like location");
  }
  if (!loom_low_allocation_assignment_is_coalescable(result_assignment)) {
    return IREE_SV("result value is not assigned to a register-like location");
  }
  if (loom_low_allocation_assignment_locations_equal(source_assignment,
                                                     result_assignment)) {
    return IREE_SV("assigned locations match and are register-like");
  }
  return IREE_SV("assigned locations differ");
}

static iree_status_t loom_low_allocation_emit_copy_decisions(
    const loom_low_allocation_sidecar_t* sidecar,
    iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < sidecar->copy_decision_count; ++i) {
    const loom_low_allocation_copy_decision_t* copy_decision =
        &sidecar->copy_decisions[i];
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_diagnostic_target_key(&sidecar->target)),
        loom_param_string(loom_low_diagnostic_export_name(&sidecar->target)),
        loom_param_string(loom_low_diagnostic_config_key(&sidecar->target)),
        loom_param_string(loom_low_diagnostic_function_name(
            sidecar->module, sidecar->function_op)),
        loom_param_string(loom_low_diagnostic_value_name(
            sidecar->module, copy_decision->source_value_id)),
        loom_param_string(loom_low_diagnostic_value_name(
            sidecar->module, copy_decision->result_value_id)),
        loom_param_string(
            loom_low_allocation_copy_decision_name(copy_decision->kind)),
        loom_param_string(
            loom_low_allocation_copy_decision_reason(sidecar, copy_decision)),
    };
    loom_diagnostic_emission_t emission = {
        .op = loom_low_diagnostic_value_origin_op(
            sidecar->module, copy_decision->result_value_id,
            sidecar->function_op),
        .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 6),
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    IREE_RETURN_IF_ERROR(iree_diagnostic_emit(emitter, &emission));
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocate_function(
    const loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_allocation_options_t* options, iree_arena_allocator_t* arena,
    loom_low_allocation_sidecar_t* out_sidecar) {
  if (!module || !low_func_op || !options || !options->descriptor_registry ||
      !arena || !out_sidecar) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, low function op, options with descriptor registry, arena, and "
        "output sidecar are required");
  }
  if (!loom_low_func_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def");
  }
  IREE_RETURN_IF_ERROR(loom_low_allocation_validate_option_shapes(options));
  *out_sidecar = (loom_low_allocation_sidecar_t){0};

  loom_low_allocation_build_state_t state = {
      .module = module,
      .options = options,
      .arena = arena,
      .body = loom_low_func_def_body(low_func_op),
  };
  if (!state.body) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low.func.def body is required");
  }
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_validate_synthesis_mode(low_func_op));

  IREE_RETURN_IF_ERROR(loom_low_descriptor_registry_verify_requirements(
      options->descriptor_registry,
      LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, low_func_op, options->descriptor_registry, options->emitter,
      &state.target));
  if (!state.target.descriptor_set) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low function target did not resolve");
  }
  IREE_RETURN_IF_ERROR(loom_low_allocation_validate_budgets(&state));
  IREE_RETURN_IF_ERROR(loom_low_allocation_validate_reserved_ranges(&state));

  IREE_RETURN_IF_ERROR(
      loom_liveness_analyze_region(module, state.body, arena, &state.liveness));
  IREE_RETURN_IF_ERROR(loom_low_allocation_validate_fixed_values(&state));
  IREE_RETURN_IF_ERROR(loom_low_allocation_assign_intervals(&state));
  IREE_RETURN_IF_ERROR(loom_low_allocation_record_copy_decisions(&state));

  loom_low_allocation_sidecar_t sidecar = {
      .module = module,
      .function_op = low_func_op,
      .target = state.target,
      .liveness = state.liveness,
      .allocation_mode = loom_low_func_def_allocation(low_func_op),
      .assignments = state.assignments,
      .assignment_count = state.assignment_count,
      .spill_plans = state.spill_plans,
      .spill_plan_count = state.spill_plan_count,
      .remarks = state.remarks,
      .remark_count = state.remark_count,
      .copy_decisions = state.copy_decisions,
      .copy_decision_count = state.copy_decision_count,
      .spill_count = state.spill_count,
      .coalesced_copy_count = state.coalesced_copy_count,
      .materialized_copy_count = state.materialized_copy_count,
  };
  loom_target_ir_bundle_storage_rebind(&sidecar.target.bundle_storage);
  IREE_RETURN_IF_ERROR(loom_low_allocation_verify_sidecar(&sidecar));
  if (iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS)) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_emit_predicted_spills(&sidecar, options->emitter));
  }
  if (iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS)) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_emit_copy_decisions(&sidecar, options->emitter));
  }
  *out_sidecar = sidecar;
  return iree_ok_status();
}
