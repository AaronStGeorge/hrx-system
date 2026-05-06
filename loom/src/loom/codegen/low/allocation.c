// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/register_class_map.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

typedef struct loom_low_allocation_build_state_t {
  // Module containing the allocated low function.
  loom_module_t* module;
  // Caller-provided allocation options.
  const loom_low_allocation_options_t* options;
  // Arena owning all table arrays.
  iree_arena_allocator_t* arena;
  // Body region of the low function.
  loom_region_t* body;
  // Low function definition operation being allocated.
  const loom_op_t* function_op;
  // Resolved target selected by the low function.
  loom_low_resolved_target_t target;
  // Descriptor register-class lookup map for module register types.
  loom_low_register_class_map_t register_class_map;
  // Resolved explicit per-class register budgets.
  struct loom_low_allocation_resolved_budget_t* resolved_budgets;
  // Number of entries in |resolved_budgets|.
  iree_host_size_t resolved_budget_count;
  // Resolved fixed SSA value locations.
  struct loom_low_allocation_resolved_fixed_value_t* resolved_fixed_values;
  // Number of entries in |resolved_fixed_values|.
  iree_host_size_t resolved_fixed_value_count;
  // Resolved whole-function target-owned location ranges.
  struct loom_low_allocation_resolved_reserved_range_t*
      resolved_reserved_ranges;
  // Number of entries in |resolved_reserved_ranges|.
  iree_host_size_t resolved_reserved_range_count;
  // Liveness analysis for |body|.
  loom_liveness_analysis_t liveness;
  // Function-local placement relations over |liveness|.
  loom_low_placement_table_t placement;
  // Active function-local value domain shared with liveness/allocation.
  const loom_local_value_domain_t* value_domain;
  // Mutable assignment records being built.
  loom_low_allocation_assignment_t* assignments;
  // Assignment indices by liveness local value ordinal. Missing entries contain
  // UINT32_MAX.
  uint32_t* assignment_indices_by_value_ordinal;
  // Assignment-index window still live at the current interval start.
  struct {
    // Assignment indices sorted by increasing assignment end point.
    uint32_t* assignment_indices;
    // First active entry in |assignment_indices|.
    iree_host_size_t start;
    // Number of active entries in |assignment_indices|.
    iree_host_size_t count;
  } active;
  // Maximum assigned location end indexed by descriptor register class ID.
  uint32_t* max_assigned_location_end_by_reg_class;
  // Mutable spill materialization plan records being built.
  loom_low_allocation_spill_plan_t* spill_plans;
  // Mutable remark records being built.
  loom_low_allocation_remark_t* remarks;
  // Mutable copy/coalescing decision records being built.
  loom_low_allocation_copy_decision_t* copy_decisions;
  // Mutable branch edge-copy records being built.
  loom_low_allocation_edge_copy_t* edge_copies;
  // Mutable branch edge-copy group records being built.
  loom_low_allocation_edge_copy_group_t* edge_copy_groups;
  // Mutable branch edge-copy scratch records being built.
  loom_low_allocation_edge_copy_temporary_t* edge_copy_temporaries;
  // Mutable packet-local move scratch groups being built.
  loom_low_allocation_packet_move_temporary_group_t*
      packet_move_temporary_groups;
  // Mutable packet-local move scratch records being built.
  loom_low_allocation_packet_move_temporary_t* packet_move_temporaries;
  // Number of initialized assignment records.
  iree_host_size_t assignment_count;
  // Number of initialized spill materialization plan records.
  iree_host_size_t spill_plan_count;
  // Number of initialized remark records.
  iree_host_size_t remark_count;
  // Number of initialized copy/coalescing decision records.
  iree_host_size_t copy_decision_count;
  // Number of initialized branch edge-copy records.
  iree_host_size_t edge_copy_count;
  // Number of initialized branch edge-copy group records.
  iree_host_size_t edge_copy_group_count;
  // Number of initialized branch edge-copy scratch records.
  iree_host_size_t edge_copy_temporary_count;
  // Number of initialized packet-local move scratch groups.
  iree_host_size_t packet_move_temporary_group_count;
  // Number of initialized packet-local move scratch records.
  iree_host_size_t packet_move_temporary_count;
  // Number of spill-slot assignments.
  iree_host_size_t spill_count;
  // Number of coalesced copy decisions.
  iree_host_size_t coalesced_copy_count;
  // Number of materialized copy decisions.
  iree_host_size_t materialized_copy_count;
} loom_low_allocation_build_state_t;

typedef struct loom_low_allocation_resolved_budget_t {
  // Descriptor-set-local register class ID.
  uint16_t descriptor_reg_class_id;
  // Maximum allocation units available for the class.
  uint32_t max_units;
} loom_low_allocation_resolved_budget_t;

typedef struct loom_low_allocation_resolved_reserved_range_t {
  // Descriptor-set-local register class ID.
  uint16_t descriptor_reg_class_id;
  // Target-visible reserved location kind.
  loom_low_allocation_location_kind_t location_kind;
  // First physical register or target ID in the reserved range.
  uint32_t location_base;
  // Number of contiguous units reserved at |location_base|.
  uint32_t location_count;
} loom_low_allocation_resolved_reserved_range_t;

typedef struct loom_low_allocation_resolved_fixed_value_t {
  // SSA value forced to the fixed location.
  loom_value_id_t value_id;
  // Descriptor-set-local register class ID for |value_id|.
  uint16_t descriptor_reg_class_id;
  // Live interval for |value_id|.
  const loom_liveness_interval_t* interval;
  // Target-visible fixed location kind.
  loom_low_allocation_location_kind_t location_kind;
  // Base physical register or target ID.
  uint32_t location_base;
  // Number of contiguous units fixed at |location_base|.
  uint32_t location_count;
} loom_low_allocation_resolved_fixed_value_t;

typedef struct loom_low_allocation_class_capacity_t {
  // Descriptor-set-local register class ID.
  uint16_t descriptor_reg_class_id;
  // Location kind used for this class.
  loom_low_allocation_location_kind_t location_kind;
  // Maximum allocation units when |is_bounded| is true.
  uint32_t max_units;
  // Number of bits in one allocation unit.
  uint16_t alloc_unit_bits;
  // Storage space used when this class spills.
  loom_low_spill_slot_space_t spill_slot_space;
  // True when values in this class can be assigned to spill slots.
  bool is_spillable;
  // True when |max_units| is a hard allocation budget.
  bool is_bounded;
} loom_low_allocation_class_capacity_t;

typedef struct loom_low_allocation_unit_location_t {
  // Target-visible storage kind.
  loom_low_allocation_location_kind_t location_kind;
  // Storage class for the unit.
  loom_liveness_value_class_t value_class;
  // Descriptor-set-local register class ID for |value_class|.
  uint16_t descriptor_reg_class_id;
  // Physical register, target ID, or spill slot ordinal.
  uint32_t location;
} loom_low_allocation_unit_location_t;

typedef struct loom_low_allocation_packet_unit_move_t {
  // Unit overwritten by the packet-local move.
  loom_low_allocation_unit_location_t destination;
  // Unit read by the packet-local move.
  loom_low_allocation_unit_location_t source;
} loom_low_allocation_packet_unit_move_t;

static iree_string_view_t loom_low_allocation_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static const loom_low_reg_class_t* loom_low_allocation_reg_class_at(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id) {
  IREE_ASSERT(descriptor_set != NULL);
  IREE_ASSERT(reg_class_id != LOOM_LOW_REG_CLASS_NONE &&
              reg_class_id < descriptor_set->reg_class_count);
  return &descriptor_set->reg_classes[reg_class_id];
}

static bool loom_low_allocation_reg_classes_share_storage(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t lhs_reg_class_id,
    uint16_t rhs_reg_class_id) {
  if (lhs_reg_class_id == rhs_reg_class_id) {
    return true;
  }
  const loom_low_reg_class_t* lhs_reg_class =
      loom_low_allocation_reg_class_at(descriptor_set, lhs_reg_class_id);
  const loom_low_reg_class_t* rhs_reg_class =
      loom_low_allocation_reg_class_at(descriptor_set, rhs_reg_class_id);
  return lhs_reg_class->alias_set_id != 0 &&
         lhs_reg_class->alias_set_id == rhs_reg_class->alias_set_id;
}

static bool loom_low_allocation_assignment_classes_share_storage(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         loom_low_allocation_reg_classes_share_storage(
             descriptor_set, lhs->descriptor_reg_class_id,
             rhs->descriptor_reg_class_id);
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

static bool loom_low_allocation_value_id_is_ignored(
    loom_value_id_t value_id, const loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_count) {
  for (uint16_t i = 0; i < ignored_value_count; ++i) {
    if (ignored_value_ids[i] == value_id) {
      return true;
    }
  }
  return false;
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
  uint8_t allocation_mode = loom_low_function_allocation(low_func_op);
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

static bool loom_low_allocation_lookup_budget(
    const loom_low_allocation_build_state_t* state, uint16_t reg_class_id,
    uint32_t* out_max_units) {
  for (iree_host_size_t i = 0; i < state->resolved_budget_count; ++i) {
    const loom_low_allocation_resolved_budget_t* budget =
        &state->resolved_budgets[i];
    if (!loom_low_allocation_reg_classes_share_storage(
            state->target.descriptor_set, budget->descriptor_reg_class_id,
            reg_class_id)) {
      continue;
    }
    *out_max_units = budget->max_units;
    return true;
  }
  return false;
}

static iree_status_t loom_low_allocation_resolve_descriptor_register_class(
    const loom_low_allocation_build_state_t* state,
    loom_liveness_value_class_t value_class, uint16_t* out_reg_class_id,
    const loom_low_reg_class_t** out_reg_class) {
  if (out_reg_class_id) {
    *out_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  }
  if (out_reg_class) {
    *out_reg_class = NULL;
  }
  if (value_class.type_kind != LOOM_TYPE_REGISTER ||
      value_class.register_class_id == LOOM_STRING_ID_INVALID) {
    iree_string_view_t register_class = loom_low_allocation_module_string(
        state->module, value_class.register_class_id);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation value class '%.*s' is not a descriptor register class",
        (int)register_class.size, register_class.data);
  }
  uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* reg_class = NULL;
  bool found_reg_class = false;
  IREE_RETURN_IF_ERROR(loom_low_register_class_map_try_resolve_string_id(
      &state->register_class_map, value_class.register_class_id, &reg_class_id,
      &reg_class, &found_reg_class));
  if (!found_reg_class) {
    iree_string_view_t register_class = loom_low_allocation_module_string(
        state->module, value_class.register_class_id);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation register class '%.*s' is not defined by descriptor set "
        "'%.*s'",
        (int)register_class.size, register_class.data,
        (int)state->target.descriptor_set_key.size,
        state->target.descriptor_set_key.data);
  }
  if (out_reg_class_id) {
    *out_reg_class_id = reg_class_id;
  }
  if (out_reg_class) {
    *out_reg_class = reg_class;
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

static iree_status_t loom_low_allocation_resolve_budgets(
    loom_low_allocation_build_state_t* state) {
  if (state->options->budget_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->options->budget_count,
      sizeof(*state->resolved_budgets), (void**)&state->resolved_budgets));
  for (iree_host_size_t i = 0; i < state->options->budget_count; ++i) {
    const loom_low_allocation_budget_t* budget = &state->options->budgets[i];
    uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    bool found_reg_class = false;
    IREE_RETURN_IF_ERROR(loom_low_register_class_try_lookup_name(
        state->target.descriptor_set, budget->register_class, &reg_class_id,
        NULL, &found_reg_class));
    if (!found_reg_class) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unknown low allocation budget register class '%.*s' for descriptor "
          "set '%.*s'",
          (int)budget->register_class.size, budget->register_class.data,
          (int)state->target.descriptor_set_key.size,
          state->target.descriptor_set_key.data);
    }
    for (iree_host_size_t j = 0; j < state->resolved_budget_count; ++j) {
      if (!loom_low_allocation_reg_classes_share_storage(
              state->target.descriptor_set,
              state->resolved_budgets[j].descriptor_reg_class_id,
              reg_class_id)) {
        continue;
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate low allocation budget register class '%.*s'",
          (int)budget->register_class.size, budget->register_class.data);
    }
    state->resolved_budgets[state->resolved_budget_count++] =
        (loom_low_allocation_resolved_budget_t){
            .descriptor_reg_class_id = reg_class_id,
            .max_units = budget->max_units,
        };
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_class_capacity(
    const loom_low_allocation_build_state_t* state,
    loom_liveness_value_class_t value_class,
    loom_low_allocation_class_capacity_t* out_capacity) {
  uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  const loom_low_reg_class_t* reg_class = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_resolve_descriptor_register_class(
      state, value_class, &reg_class_id, &reg_class));

  uint32_t budget_units = 0;
  const bool is_spillable =
      !iree_any_bit_set(reg_class->flags, LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);
  if (loom_low_allocation_lookup_budget(state, reg_class_id, &budget_units)) {
    *out_capacity = (loom_low_allocation_class_capacity_t){
        .descriptor_reg_class_id = reg_class_id,
        .location_kind = loom_low_allocation_reg_class_location_kind(reg_class),
        .max_units = budget_units,
        .alloc_unit_bits = reg_class->alloc_unit_bits,
        .spill_slot_space =
            (loom_low_spill_slot_space_t)reg_class->spill_slot_space,
        .is_spillable = is_spillable,
        .is_bounded = true,
    };
    return iree_ok_status();
  }

  if (reg_class->physical_count > 0) {
    *out_capacity = (loom_low_allocation_class_capacity_t){
        .descriptor_reg_class_id = reg_class_id,
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .max_units = reg_class->physical_count,
        .alloc_unit_bits = reg_class->alloc_unit_bits,
        .spill_slot_space =
            (loom_low_spill_slot_space_t)reg_class->spill_slot_space,
        .is_spillable = is_spillable,
        .is_bounded = true,
    };
    return iree_ok_status();
  }

  *out_capacity = (loom_low_allocation_class_capacity_t){
      .descriptor_reg_class_id = reg_class_id,
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
      .max_units = UINT32_MAX,
      .alloc_unit_bits = reg_class->alloc_unit_bits,
      .spill_slot_space =
          (loom_low_spill_slot_space_t)reg_class->spill_slot_space,
      .is_spillable = is_spillable,
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

static iree_status_t loom_low_allocation_resolve_reserved_ranges(
    loom_low_allocation_build_state_t* state) {
  if (state->options->reserved_range_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->options->reserved_range_count,
      sizeof(*state->resolved_reserved_ranges),
      (void**)&state->resolved_reserved_ranges));
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

    uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    const loom_low_reg_class_t* reg_class = NULL;
    bool found_reg_class = false;
    IREE_RETURN_IF_ERROR(loom_low_register_class_try_lookup_name(
        state->target.descriptor_set, reserved_range->register_class,
        &reg_class_id, &reg_class, &found_reg_class));
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
    for (iree_host_size_t j = 0; j < state->resolved_reserved_range_count;
         ++j) {
      const loom_low_allocation_resolved_reserved_range_t* existing =
          &state->resolved_reserved_ranges[j];
      if (reserved_range->location_kind != existing->location_kind ||
          !loom_low_allocation_reg_classes_share_storage(
              state->target.descriptor_set, reg_class_id,
              existing->descriptor_reg_class_id)) {
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
    state->resolved_reserved_ranges[state->resolved_reserved_range_count++] =
        (loom_low_allocation_resolved_reserved_range_t){
            .descriptor_reg_class_id = reg_class_id,
            .location_kind = reserved_range->location_kind,
            .location_base = reserved_range->location_base,
            .location_count = reserved_range->location_count,
        };
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_resolve_fixed_values(
    loom_low_allocation_build_state_t* state) {
  if (state->options->fixed_value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, state->options->fixed_value_count,
                                sizeof(*state->resolved_fixed_values),
                                (void**)&state->resolved_fixed_values));
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

    const loom_value_ordinal_t value_ordinal =
        loom_local_value_domain_try_ordinal(state->value_domain,
                                            fixed_value->value_id);
    const loom_liveness_interval_t* interval =
        value_ordinal == LOOM_VALUE_ORDINAL_INVALID
            ? NULL
            : loom_liveness_interval_for_value_ordinal(&state->liveness,
                                                       value_ordinal);
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

    uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    const loom_low_reg_class_t* reg_class = NULL;
    IREE_RETURN_IF_ERROR(loom_low_allocation_resolve_descriptor_register_class(
        state, interval->value_class, &reg_class_id, &reg_class));
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
    state->resolved_fixed_values[state->resolved_fixed_value_count++] =
        (loom_low_allocation_resolved_fixed_value_t){
            .value_id = fixed_value->value_id,
            .descriptor_reg_class_id = reg_class_id,
            .interval = interval,
            .location_kind = fixed_value->location_kind,
            .location_base = fixed_value->location_base,
            .location_count = fixed_value->location_count,
        };
  }
  return iree_ok_status();
}

static const loom_low_allocation_resolved_fixed_value_t*
loom_low_allocation_fixed_value_for_value(
    const loom_low_allocation_build_state_t* state, loom_value_id_t value_id) {
  for (iree_host_size_t i = 0; i < state->resolved_fixed_value_count; ++i) {
    const loom_low_allocation_resolved_fixed_value_t* fixed_value =
        &state->resolved_fixed_values[i];
    if (fixed_value->value_id == value_id) {
      return fixed_value;
    }
  }
  return NULL;
}

static bool loom_low_allocation_fixed_value_conflicts(
    const loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval, uint16_t reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, const loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_count) {
  for (iree_host_size_t i = 0; i < state->resolved_fixed_value_count; ++i) {
    const loom_low_allocation_resolved_fixed_value_t* fixed_value =
        &state->resolved_fixed_values[i];
    if (loom_low_allocation_value_id_is_ignored(
            fixed_value->value_id, ignored_value_ids, ignored_value_count)) {
      continue;
    }
    if (fixed_value->location_kind != location_kind) {
      continue;
    }
    if (!loom_low_allocation_reg_classes_share_storage(
            state->target.descriptor_set, fixed_value->descriptor_reg_class_id,
            reg_class_id)) {
      continue;
    }
    if (!loom_low_allocation_location_range_overlaps(
            fixed_value->location_base, fixed_value->location_count,
            location_base, location_count)) {
      continue;
    }
    if (loom_low_allocation_liveness_intervals_overlap(fixed_value->interval,
                                                       interval)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_reserved_range_conflicts(
    const loom_low_allocation_build_state_t* state, uint16_t reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count) {
  for (iree_host_size_t i = 0; i < state->resolved_reserved_range_count; ++i) {
    const loom_low_allocation_resolved_reserved_range_t* reserved_range =
        &state->resolved_reserved_ranges[i];
    if (reserved_range->location_kind != location_kind) {
      continue;
    }
    if (!loom_low_allocation_reg_classes_share_storage(
            state->target.descriptor_set,
            reserved_range->descriptor_reg_class_id, reg_class_id)) {
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
    const loom_liveness_interval_t* interval, uint16_t reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, const loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_count) {
  loom_low_allocation_assignment_t candidate = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = reg_class_id,
      .start_point = interval->start_point,
      .end_point = interval->end_point,
      .unit_count = interval->unit_count,
      .location_kind = location_kind,
      .location_base = location_base,
      .location_count = location_count,
  };
  for (iree_host_size_t i = 0; i < state->active.count; ++i) {
    const uint32_t assignment_index =
        state->active.assignment_indices[state->active.start + i];
    IREE_ASSERT_LT(assignment_index, state->assignment_count);
    const loom_low_allocation_assignment_t* existing =
        &state->assignments[assignment_index];
    if (loom_low_allocation_value_id_is_ignored(
            existing->value_id, ignored_value_ids, ignored_value_count)) {
      continue;
    }
    if (existing->location_kind != location_kind) {
      continue;
    }
    if (!loom_low_allocation_reg_classes_share_storage(
            state->target.descriptor_set, existing->descriptor_reg_class_id,
            reg_class_id)) {
      continue;
    }
    if (!loom_low_allocation_interval_overlaps(existing, &candidate)) {
      continue;
    }
    if (loom_low_allocation_location_ranges_overlap(existing, &candidate)) {
      return true;
    }
  }
  if (loom_low_allocation_fixed_value_conflicts(
          state, interval, reg_class_id, location_kind, location_base,
          location_count, ignored_value_ids, ignored_value_count)) {
    return true;
  }
  if (loom_low_allocation_reserved_range_conflicts(
          state, reg_class_id, location_kind, location_base, location_count)) {
    return true;
  }
  return false;
}

static uint32_t loom_low_allocation_assignment_location_end(
    const loom_low_allocation_assignment_t* assignment) {
  const uint64_t location_end =
      (uint64_t)assignment->location_base + assignment->location_count;
  return location_end > UINT32_MAX ? UINT32_MAX : (uint32_t)location_end;
}

static void loom_low_allocation_record_assignment_location_end(
    loom_low_allocation_build_state_t* state,
    const loom_low_allocation_assignment_t* assignment) {
  IREE_ASSERT(state->max_assigned_location_end_by_reg_class != NULL);
  const uint16_t reg_class_id = assignment->descriptor_reg_class_id;
  IREE_ASSERT(reg_class_id != LOOM_LOW_REG_CLASS_NONE &&
              reg_class_id < state->target.descriptor_set->reg_class_count);
  const loom_low_reg_class_t* reg_class = loom_low_allocation_reg_class_at(
      state->target.descriptor_set, reg_class_id);
  if (assignment->location_kind !=
      loom_low_allocation_reg_class_location_kind(reg_class)) {
    return;
  }
  const uint32_t location_end =
      loom_low_allocation_assignment_location_end(assignment);
  if (state->max_assigned_location_end_by_reg_class[reg_class_id] <
      location_end) {
    state->max_assigned_location_end_by_reg_class[reg_class_id] = location_end;
  }
}

static uint32_t loom_low_allocation_assigned_location_search_limit(
    const loom_low_allocation_build_state_t* state, uint16_t reg_class_id,
    loom_low_allocation_location_kind_t location_kind) {
  uint32_t max_end = 0;
  for (uint16_t i = 0; i < state->target.descriptor_set->reg_class_count; ++i) {
    if (!loom_low_allocation_reg_classes_share_storage(
            state->target.descriptor_set, i, reg_class_id)) {
      continue;
    }
    const loom_low_reg_class_t* reg_class =
        loom_low_allocation_reg_class_at(state->target.descriptor_set, i);
    if (loom_low_allocation_reg_class_location_kind(reg_class) !=
        location_kind) {
      continue;
    }
    const uint32_t assignment_end =
        state->max_assigned_location_end_by_reg_class[i];
    if (assignment_end > max_end) {
      max_end = assignment_end;
    }
  }
  for (iree_host_size_t i = 0; i < state->resolved_fixed_value_count; ++i) {
    const loom_low_allocation_resolved_fixed_value_t* fixed_value =
        &state->resolved_fixed_values[i];
    if (fixed_value->location_kind != location_kind) {
      continue;
    }
    if (!loom_low_allocation_reg_classes_share_storage(
            state->target.descriptor_set, fixed_value->descriptor_reg_class_id,
            reg_class_id)) {
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
  for (iree_host_size_t i = 0; i < state->resolved_reserved_range_count; ++i) {
    const loom_low_allocation_resolved_reserved_range_t* reserved_range =
        &state->resolved_reserved_ranges[i];
    if (reserved_range->location_kind != location_kind) {
      continue;
    }
    if (!loom_low_allocation_reg_classes_share_storage(
            state->target.descriptor_set,
            reserved_range->descriptor_reg_class_id, reg_class_id)) {
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
    last_base = loom_low_allocation_assigned_location_search_limit(
        state, capacity.descriptor_reg_class_id, capacity.location_kind);
  }

  const uint32_t alignment = loom_low_allocation_interval_alignment(interval);
  for (uint32_t base = 0; base <= last_base;) {
    if (!loom_low_allocation_candidate_conflicts(
            state, interval, capacity.descriptor_reg_class_id,
            capacity.location_kind, base, interval->unit_count,
            /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0)) {
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

static bool loom_low_allocation_align_up_u32(uint32_t value, uint32_t alignment,
                                             uint32_t* out_value) {
  if (alignment <= 1) {
    *out_value = value;
    return true;
  }
  const uint32_t remainder = value % alignment;
  if (remainder == 0) {
    *out_value = value;
    return true;
  }
  const uint32_t increment = alignment - remainder;
  if (value > UINT32_MAX - increment) {
    return false;
  }
  *out_value = value + increment;
  return true;
}

static bool loom_low_allocation_find_location_span(
    const loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_low_allocation_class_capacity_t capacity, uint32_t location_count,
    uint32_t alignment, uint32_t* out_base) {
  if (location_count == 0 ||
      (capacity.is_bounded && location_count > capacity.max_units)) {
    return false;
  }
  if (alignment == 0) {
    alignment = 1;
  }

  const uint32_t assigned_limit =
      loom_low_allocation_assigned_location_search_limit(
          state, capacity.descriptor_reg_class_id, capacity.location_kind);

  uint32_t last_base = 0;
  if (capacity.is_bounded) {
    last_base = capacity.max_units - location_count;
  } else if (!loom_low_allocation_align_up_u32(assigned_limit, alignment,
                                               &last_base)) {
    return false;
  }

  for (uint32_t base = 0; base <= last_base;) {
    if (!loom_low_allocation_candidate_conflicts(
            state, interval, capacity.descriptor_reg_class_id,
            capacity.location_kind, base, location_count,
            /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0)) {
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
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         lhs->location_base == rhs->location_base &&
         lhs->location_count == rhs->location_count;
}

static bool loom_low_allocation_unit_locations_equal(
    const loom_low_allocation_unit_location_t* lhs,
    const loom_low_allocation_unit_location_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         lhs->location == rhs->location;
}

static bool loom_low_allocation_unit_storage_classes_equal(
    const loom_low_allocation_unit_location_t* lhs,
    const loom_low_allocation_unit_location_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         loom_liveness_value_class_equal(lhs->value_class, rhs->value_class);
}

static iree_status_t loom_low_allocation_assignment_unit_location(
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_index,
    loom_low_allocation_unit_location_t* out_location) {
  if (unit_index >= assignment->location_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation unit index exceeds assignment range");
  }
  if (assignment->location_base > UINT32_MAX - unit_index) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation unit location exceeds uint32_t");
  }
  *out_location = (loom_low_allocation_unit_location_t){
      .location_kind = assignment->location_kind,
      .value_class = assignment->value_class,
      .descriptor_reg_class_id = assignment->descriptor_reg_class_id,
      .location = assignment->location_base + unit_index,
  };
  return iree_ok_status();
}

static bool loom_low_allocation_assignment_is_coalescable(
    const loom_low_allocation_assignment_t* assignment) {
  return loom_low_allocation_location_is_register_like(
      assignment->location_kind);
}

static bool loom_low_allocation_value_ordinal_for_value(
    const loom_low_allocation_build_state_t* state, loom_value_id_t value_id,
    loom_value_ordinal_t* out_value_ordinal) {
  IREE_ASSERT(loom_local_value_domain_is_acquired(state->value_domain));
  const loom_value_ordinal_t value_ordinal =
      loom_module_value_ordinal_scratch_lookup(state->module, value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= state->liveness.value_count) {
    return false;
  }
  IREE_ASSERT_EQ(state->liveness.value_ids[value_ordinal], value_id);
  *out_value_ordinal = value_ordinal;
  return true;
}

static iree_status_t loom_low_allocation_assignment_index_for_value(
    const loom_low_allocation_build_state_t* state, loom_value_id_t value_id,
    uint32_t* out_assignment_index) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (loom_low_allocation_value_ordinal_for_value(state, value_id,
                                                  &value_ordinal)) {
    const uint32_t assignment_index =
        state->assignment_indices_by_value_ordinal[value_ordinal];
    if (assignment_index != UINT32_MAX) {
      IREE_ASSERT(assignment_index < state->assignment_count,
                  "assignment index exceeds initialized assignment count");
      IREE_ASSERT(state->assignments[assignment_index].value_id == value_id,
                  "assignment index table points at the wrong value");
      *out_assignment_index = assignment_index;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "low allocation references value %u without an "
                          "assignment",
                          (unsigned)value_id);
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_current_assignment_for_value_ordinal(
    const loom_low_allocation_build_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  IREE_ASSERT_LT(value_ordinal, state->liveness.value_count);
  IREE_ASSERT(state->assignment_indices_by_value_ordinal != NULL);
  const uint32_t assignment_index =
      state->assignment_indices_by_value_ordinal[value_ordinal];
  if (assignment_index == UINT32_MAX) {
    return NULL;
  }
  IREE_ASSERT_LT(assignment_index, state->assignment_count);
  const loom_low_allocation_assignment_t* assignment =
      &state->assignments[assignment_index];
  IREE_ASSERT_EQ(assignment->value_id,
                 state->liveness.value_ids[value_ordinal]);
  return assignment;
}

static iree_status_t loom_low_allocation_value_ordinal_for_interval(
    const loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_value_ordinal_t* out_value_ordinal) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_value_ordinal_for_value(state, interval->value_id,
                                                   &value_ordinal)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation interval value %u is outside the local value domain",
        (unsigned)interval->value_id);
  }
  *out_value_ordinal = value_ordinal;
  return iree_ok_status();
}

static const loom_low_placement_relation_t*
loom_low_allocation_first_placement_relation(
    const loom_low_allocation_build_state_t* state,
    loom_value_ordinal_t result_ordinal, loom_low_placement_cause_t cause) {
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(&state->placement,
                                                          result_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &state->placement.relations[range.start + i];
    if (relation->cause == cause) {
      return relation;
    }
  }
  return NULL;
}

static void loom_low_allocation_expire_active_assignments(
    loom_low_allocation_build_state_t* state, uint32_t start_point) {
  while (state->active.count > 0) {
    const uint32_t assignment_index =
        state->active.assignment_indices[state->active.start];
    IREE_ASSERT_LT(assignment_index, state->assignment_count);
    const loom_low_allocation_assignment_t* assignment =
        &state->assignments[assignment_index];
    if (assignment->end_point > start_point) {
      break;
    }
    ++state->active.start;
    --state->active.count;
  }
}

static bool loom_low_allocation_active_assignment_less(
    const loom_low_allocation_build_state_t* state, uint32_t lhs_index,
    uint32_t rhs_index) {
  const loom_low_allocation_assignment_t* lhs = &state->assignments[lhs_index];
  const loom_low_allocation_assignment_t* rhs = &state->assignments[rhs_index];
  if (lhs->end_point != rhs->end_point) {
    return lhs->end_point < rhs->end_point;
  }
  return lhs_index < rhs_index;
}

static void loom_low_allocation_activate_assignment(
    loom_low_allocation_build_state_t* state, uint32_t assignment_index) {
  IREE_ASSERT(state->active.assignment_indices != NULL);
  IREE_ASSERT_LT(assignment_index, state->assignment_count);
  iree_host_size_t insert_index = state->active.start + state->active.count;
  while (insert_index > state->active.start) {
    const uint32_t previous_assignment_index =
        state->active.assignment_indices[insert_index - 1];
    if (loom_low_allocation_active_assignment_less(
            state, previous_assignment_index, assignment_index)) {
      break;
    }
    state->active.assignment_indices[insert_index] = previous_assignment_index;
    --insert_index;
  }
  state->active.assignment_indices[insert_index] = assignment_index;
  ++state->active.count;
}

static iree_status_t loom_low_allocation_append_assignment(
    loom_low_allocation_build_state_t* state,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t* out_assignment_index) {
  if (state->assignment_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation table exceeds uint32_t range");
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_value_ordinal_for_value(state, assignment->value_id,
                                                   &value_ordinal)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation saw assignment for value %u outside the analyzed "
        "liveness value range",
        (unsigned)assignment->value_id);
  }
  if (state->assignment_indices_by_value_ordinal[value_ordinal] != UINT32_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low allocation saw duplicate assignment for value "
                            "%u",
                            (unsigned)assignment->value_id);
  }
  const uint32_t assignment_index = (uint32_t)state->assignment_count;
  state->assignments[state->assignment_count++] = *assignment;
  state->assignment_indices_by_value_ordinal[value_ordinal] = assignment_index;
  loom_low_allocation_record_assignment_location_end(state, assignment);
  loom_low_allocation_activate_assignment(state, assignment_index);
  if (out_assignment_index) {
    *out_assignment_index = assignment_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_assign_tied_interval(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  loom_value_ordinal_t result_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_allocation_value_ordinal_for_interval(
      state, interval, &result_ordinal));
  const loom_low_placement_relation_t* relation =
      loom_low_allocation_first_placement_relation(
          state, result_ordinal, LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT);
  if (!relation) {
    return iree_ok_status();
  }

  const loom_value_id_t tied_operand_id =
      loom_low_placement_value_id(&state->placement, relation->source_ordinal);
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
          state, interval, operand_assignment->descriptor_reg_class_id,
          operand_assignment->location_kind, operand_assignment->location_base,
          operand_assignment->location_count, &tied_operand_id,
          /*ignored_value_count=*/1)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low tied result cannot share the operand location without "
        "overlapping another live interval");
  }

  const loom_low_allocation_assignment_t assignment = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = operand_assignment->descriptor_reg_class_id,
      .start_point = interval->start_point,
      .end_point = interval->end_point,
      .unit_count = interval->unit_count,
      .location_kind = operand_assignment->location_kind,
      .location_base = operand_assignment->location_base,
      .location_count = operand_assignment->location_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_append_assignment(state, &assignment, NULL));
  *out_assigned = true;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_assign_fixed_interval(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  const loom_low_allocation_resolved_fixed_value_t* fixed_value =
      loom_low_allocation_fixed_value_for_value(state, interval->value_id);
  if (!fixed_value) {
    return iree_ok_status();
  }
  if (loom_low_allocation_candidate_conflicts(
          state, interval, fixed_value->descriptor_reg_class_id,
          fixed_value->location_kind, fixed_value->location_base,
          fixed_value->location_count, &interval->value_id,
          /*ignored_value_count=*/1)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low fixed value cannot occupy its required location without "
        "overlapping another live interval or reserved range");
  }

  const loom_low_allocation_assignment_t assignment = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = fixed_value->descriptor_reg_class_id,
      .start_point = interval->start_point,
      .end_point = interval->end_point,
      .unit_count = interval->unit_count,
      .location_kind = fixed_value->location_kind,
      .location_base = fixed_value->location_base,
      .location_count = fixed_value->location_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_append_assignment(state, &assignment, NULL));
  *out_assigned = true;
  return iree_ok_status();
}

static bool loom_low_allocation_assignment_unit_span_fits(
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_offset,
    uint32_t unit_count) {
  return unit_offset <= assignment->location_count &&
         unit_count <= assignment->location_count - unit_offset &&
         assignment->location_base <= UINT32_MAX - unit_offset;
}

static iree_status_t loom_low_allocation_append_interval_at_location(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval, uint16_t descriptor_reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t unit_count, const loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_count, bool* out_assigned) {
  *out_assigned = false;
  if (location_base > UINT32_MAX - unit_count) {
    return iree_ok_status();
  }
  const uint32_t alignment = loom_low_allocation_interval_alignment(interval);
  if (location_base % alignment != 0) {
    return iree_ok_status();
  }
  if (loom_low_allocation_candidate_conflicts(
          state, interval, descriptor_reg_class_id, location_kind,
          location_base, unit_count, ignored_value_ids, ignored_value_count)) {
    return iree_ok_status();
  }

  const loom_low_allocation_assignment_t assignment = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = descriptor_reg_class_id,
      .start_point = interval->start_point,
      .end_point = interval->end_point,
      .unit_count = interval->unit_count,
      .location_kind = location_kind,
      .location_base = location_base,
      .location_count = unit_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_append_assignment(state, &assignment, NULL));
  *out_assigned = true;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_append_relation_interval(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    bool* out_assigned) {
  *out_assigned = false;
  if (relation->result_unit_offset > interval->unit_count ||
      relation->unit_count >
          interval->unit_count - relation->result_unit_offset) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation placement relation exceeds result "
                            "interval units");
  }
  uint32_t source_assignment_index = 0;
  const loom_value_id_t source_value_id =
      loom_low_placement_value_id(&state->placement, relation->source_ordinal);
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
      state, source_value_id, &source_assignment_index));
  const loom_low_allocation_assignment_t* source_assignment =
      &state->assignments[source_assignment_index];
  if (!loom_low_allocation_assignment_is_coalescable(source_assignment)) {
    return iree_ok_status();
  }
  if (!loom_low_allocation_assignment_unit_span_fits(
          source_assignment, relation->source_unit_offset,
          relation->unit_count)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation placement relation exceeds source "
                            "assignment units");
  }
  const uint32_t source_unit_location =
      source_assignment->location_base + relation->source_unit_offset;
  if (source_unit_location < relation->result_unit_offset) {
    return iree_ok_status();
  }
  const uint32_t result_location_base =
      source_unit_location - relation->result_unit_offset;
  return loom_low_allocation_append_interval_at_location(
      state, interval, source_assignment->descriptor_reg_class_id,
      source_assignment->location_kind, result_location_base,
      interval->unit_count, ignored_value_ids, ignored_value_count,
      out_assigned);
}

static iree_status_t loom_low_allocation_assign_relation_interval(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation, bool* out_assigned) {
  *out_assigned = false;
  const loom_value_id_t source_value_id =
      loom_low_placement_value_id(&state->placement, relation->source_ordinal);
  return loom_low_allocation_append_relation_interval(
      state, interval, relation, &source_value_id,
      /*ignored_value_count=*/1, out_assigned);
}

static iree_status_t loom_low_allocation_assign_concat_interval(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_range_t* range, bool* out_assigned) {
  *out_assigned = false;
  if (range->count == 0) {
    return iree_ok_status();
  }

  uint16_t ignored_value_count = 0;
  for (uint32_t i = 0; i < range->count; ++i) {
    const loom_low_placement_relation_t* relation =
        &state->placement.relations[range->start + i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
      continue;
    }
    if (ignored_value_count == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low.concat placement source count exceeds "
                              "uint16_t");
    }
    ++ignored_value_count;
  }
  if (ignored_value_count == 0) {
    return iree_ok_status();
  }
  loom_value_id_t inline_ignored_value_ids[8];
  loom_value_id_t* ignored_value_ids = inline_ignored_value_ids;
  if (ignored_value_count > IREE_ARRAYSIZE(inline_ignored_value_ids)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, ignored_value_count, sizeof(*ignored_value_ids),
        (void**)&ignored_value_ids));
  }

  uint32_t result_location_base = 0;
  uint32_t coalesced_unit_count = 0;
  uint16_t ignored_value_index = 0;
  bool has_result_location_base = false;
  for (uint32_t i = 0; i < range->count; ++i) {
    const loom_low_placement_relation_t* relation =
        &state->placement.relations[range->start + i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
      continue;
    }
    uint32_t source_assignment_index = 0;
    const loom_value_id_t source_value_id = loom_low_placement_value_id(
        &state->placement, relation->source_ordinal);
    ignored_value_ids[ignored_value_index++] = source_value_id;
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
        state, source_value_id, &source_assignment_index));
    const loom_low_allocation_assignment_t* source_assignment =
        &state->assignments[source_assignment_index];
    if (!loom_low_allocation_assignment_is_coalescable(source_assignment)) {
      return iree_ok_status();
    }
    if (!loom_low_allocation_assignment_unit_span_fits(
            source_assignment, relation->source_unit_offset,
            relation->unit_count)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "low.concat placement relation exceeds source "
                              "assignment units");
    }
    const uint32_t source_unit_location =
        source_assignment->location_base + relation->source_unit_offset;
    if (source_unit_location < relation->result_unit_offset) {
      return iree_ok_status();
    }
    const uint32_t candidate_base =
        source_unit_location - relation->result_unit_offset;
    if (!has_result_location_base) {
      result_location_base = candidate_base;
      has_result_location_base = true;
    } else if (result_location_base != candidate_base) {
      return iree_ok_status();
    }
    if (relation->unit_count > UINT32_MAX - coalesced_unit_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low.concat coalesced unit count exceeds u32");
    }
    coalesced_unit_count += relation->unit_count;
  }
  if (coalesced_unit_count != interval->unit_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low.concat placement relations do not cover the "
                            "result interval");
  }
  const loom_value_id_t first_source_value_id = ignored_value_ids[0];
  uint32_t first_assignment_index = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
      state, first_source_value_id, &first_assignment_index));
  const loom_low_allocation_assignment_t* first_assignment =
      &state->assignments[first_assignment_index];
  return loom_low_allocation_append_interval_at_location(
      state, interval, first_assignment->descriptor_reg_class_id,
      first_assignment->location_kind, result_location_base,
      interval->unit_count, ignored_value_ids, ignored_value_count,
      out_assigned);
}

static bool loom_low_allocation_relation_source_matches_interval(
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* interval) {
  if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
    return false;
  }
  IREE_ASSERT_EQ(relation->source_unit_offset, 0);
  IREE_ASSERT_EQ(relation->unit_count, interval->unit_count);
  return true;
}

static bool loom_low_allocation_candidate_location_for_concat_source(
    const loom_low_placement_relation_t* relation,
    const loom_low_placement_relation_t* sibling_relation,
    const loom_low_allocation_assignment_t* sibling_assignment,
    uint32_t* out_location_base) {
  if (sibling_assignment->location_base >
      UINT32_MAX - sibling_relation->source_unit_offset) {
    return false;
  }
  const uint32_t sibling_source_location =
      sibling_assignment->location_base + sibling_relation->source_unit_offset;
  if (sibling_source_location < sibling_relation->result_unit_offset) {
    return false;
  }
  const uint32_t result_location_base =
      sibling_source_location - sibling_relation->result_unit_offset;
  if (result_location_base > UINT32_MAX - relation->result_unit_offset) {
    return false;
  }
  const uint32_t source_location =
      result_location_base + relation->result_unit_offset;
  if (source_location < relation->source_unit_offset) {
    return false;
  }
  *out_location_base = source_location - relation->source_unit_offset;
  return true;
}

static iree_status_t loom_low_allocation_assign_concat_source_interval(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  loom_value_ordinal_t source_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_allocation_value_ordinal_for_interval(
      state, interval, &source_ordinal));
  const loom_low_placement_relation_range_t source_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          &state->placement, source_ordinal);
  for (uint32_t source_index = 0; source_index < source_range.count;
       ++source_index) {
    const uint32_t relation_index =
        state->placement.relation_indices_by_source_ordinal[source_range.start +
                                                            source_index];
    IREE_ASSERT_LT(relation_index, state->placement.relation_count);
    const loom_low_placement_relation_t* relation =
        &state->placement.relations[relation_index];
    if (!loom_low_allocation_relation_source_matches_interval(relation,
                                                              interval)) {
      continue;
    }

    const loom_low_placement_relation_range_t result_range =
        loom_low_placement_relation_range_for_value_ordinal(
            &state->placement, relation->result_ordinal);
    for (uint32_t result_index = 0; result_index < result_range.count;
         ++result_index) {
      const loom_low_placement_relation_t* sibling_relation =
          &state->placement.relations[result_range.start + result_index];
      if (sibling_relation == relation ||
          sibling_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
          sibling_relation->source_ordinal == relation->source_ordinal) {
        continue;
      }
      const loom_low_allocation_assignment_t* sibling_assignment =
          loom_low_allocation_current_assignment_for_value_ordinal(
              state, sibling_relation->source_ordinal);
      if (!sibling_assignment ||
          !loom_low_allocation_assignment_is_coalescable(sibling_assignment) ||
          !loom_liveness_value_class_equal(sibling_assignment->value_class,
                                           interval->value_class) ||
          !loom_low_allocation_assignment_unit_span_fits(
              sibling_assignment, sibling_relation->source_unit_offset,
              sibling_relation->unit_count)) {
        continue;
      }

      uint32_t location_base = 0;
      if (!loom_low_allocation_candidate_location_for_concat_source(
              relation, sibling_relation, sibling_assignment, &location_base)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_allocation_append_interval_at_location(
          state, interval, sibling_assignment->descriptor_reg_class_id,
          sibling_assignment->location_kind, location_base,
          interval->unit_count, /*ignored_value_ids=*/NULL,
          /*ignored_value_count=*/0, out_assigned));
      if (*out_assigned) {
        return iree_ok_status();
      }
    }

    const loom_liveness_interval_t* result_interval =
        loom_liveness_interval_for_value_ordinal(&state->liveness,
                                                 relation->result_ordinal);
    if (!result_interval ||
        !loom_liveness_value_class_equal(result_interval->value_class,
                                         interval->value_class)) {
      continue;
    }
    loom_low_allocation_class_capacity_t capacity = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_class_capacity(
        state, interval->value_class, &capacity));
    uint32_t result_location_base = 0;
    if (!loom_low_allocation_find_location_span(
            state, interval, capacity, result_interval->unit_count,
            loom_low_allocation_interval_alignment(result_interval),
            &result_location_base)) {
      continue;
    }
    if (result_location_base > UINT32_MAX - relation->result_unit_offset) {
      continue;
    }
    const uint32_t source_unit_location =
        result_location_base + relation->result_unit_offset;
    if (source_unit_location < relation->source_unit_offset) {
      continue;
    }
    const uint32_t source_location_base =
        source_unit_location - relation->source_unit_offset;
    IREE_RETURN_IF_ERROR(loom_low_allocation_append_interval_at_location(
        state, interval, capacity.descriptor_reg_class_id,
        capacity.location_kind, source_location_base, interval->unit_count,
        /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0, out_assigned));
    if (*out_assigned) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_assign_structural_interval(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  loom_value_ordinal_t result_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_allocation_value_ordinal_for_interval(
      state, interval, &result_ordinal));
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(&state->placement,
                                                          result_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &state->placement.relations[range.start + i];
    switch (relation->cause) {
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY:
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE: {
        IREE_RETURN_IF_ERROR(loom_low_allocation_assign_relation_interval(
            state, interval, relation, out_assigned));
        if (*out_assigned) {
          return iree_ok_status();
        }
        break;
      }
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT: {
        IREE_RETURN_IF_ERROR(loom_low_allocation_assign_concat_interval(
            state, interval, &range, out_assigned));
        return iree_ok_status();
      }
      default:
        break;
    }
  }
  return iree_ok_status();
}

static iree_host_size_t loom_low_allocation_count_copy_ops(
    const loom_region_t* body) {
  iree_host_size_t copy_count = 0;
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_copy_isa(op)) {
        ++copy_count;
      }
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

static iree_status_t loom_low_allocation_count_edge_copy_groups(
    const loom_region_t* body, iree_host_size_t* out_group_count,
    iree_host_size_t* out_copy_count) {
  *out_group_count = 0;
  *out_copy_count = 0;
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_br_isa(op)) {
        continue;
      }
      loom_value_slice_t args = loom_low_br_args(op);
      if (args.count == 0) {
        continue;
      }
      if (*out_group_count == IREE_HOST_SIZE_MAX ||
          args.count > IREE_HOST_SIZE_MAX - *out_copy_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "low.br edge-copy count exceeds host size");
      }
      ++*out_group_count;
      *out_copy_count += args.count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_op_program_point_in_liveness(
    const loom_liveness_analysis_t* liveness, const loom_op_t* op,
    uint32_t* out_program_point) {
  *out_program_point = UINT32_MAX;
  const loom_liveness_block_info_t* block_info =
      loom_liveness_block_info_for_block(liveness, op->parent_block);
  if (!block_info) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation cannot find liveness block for low operation");
  }
  uint32_t program_point = block_info->start_point;
  const loom_op_t* block_op = op->parent_block->first_op;
  while (block_op && block_op != op) {
    ++program_point;
    block_op = block_op->next_op;
  }
  if (block_op != op) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low operation is not linked in its parent block");
  }
  *out_program_point = program_point;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_op_program_point(
    const loom_low_allocation_build_state_t* state, const loom_op_t* op,
    uint32_t* out_program_point) {
  if (!loom_liveness_order_is_empty(state->options->liveness_order)) {
    uint16_t block_index = 0;
    if (!loom_region_try_block_index(state->body, op->parent_block,
                                     &block_index) ||
        block_index >= state->options->liveness_order.block_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low allocation cannot find ordered block for low operation");
    }
    const loom_liveness_block_info_t* block_info =
        loom_liveness_block_info_for_block(&state->liveness, op->parent_block);
    if (!block_info) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low allocation cannot find ordered liveness block for low "
          "operation");
    }
    const loom_liveness_block_order_t* block_order =
        &state->options->liveness_order.blocks[block_index];
    for (iree_host_size_t i = 0; i < block_order->op_count; ++i) {
      if (block_order->ops[i] != op) {
        continue;
      }
      if (i > UINT32_MAX - block_info->start_point) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "ordered low operation program point exceeds uint32_t");
      }
      *out_program_point = block_info->start_point + (uint32_t)i;
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "ordered low operation has no operation order "
                            "entry");
  }
  return loom_low_allocation_op_program_point_in_liveness(&state->liveness, op,
                                                          out_program_point);
}

static iree_status_t loom_low_allocation_record_edge_copy_group(
    loom_low_allocation_build_state_t* state, const loom_op_t* op,
    uint32_t source_ordinal) {
  loom_value_slice_t args = loom_low_br_args(op);
  if (args.count == 0) {
    return iree_ok_status();
  }
  const loom_block_t* dest = loom_low_br_dest(op);
  if (args.count != dest->arg_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.br edge-copy payload count does not match destination block args");
  }
  if (state->edge_copy_count > UINT32_MAX ||
      args.count > UINT32_MAX - state->edge_copy_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low.br edge-copy group exceeds u32 range");
  }
  uint32_t program_point = UINT32_MAX;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_op_program_point(state, op, &program_point));
  loom_low_allocation_edge_copy_group_t* group =
      &state->edge_copy_groups[state->edge_copy_group_count++];
  *group = (loom_low_allocation_edge_copy_group_t){
      .terminator_op = op,
      .source_ordinal = source_ordinal,
      .program_point = program_point,
      .copy_start = (uint32_t)state->edge_copy_count,
      .copy_count = (uint32_t)args.count,
      .temporary_start = 0,
      .temporary_count = 0,
  };
  for (uint16_t i = 0; i < dest->arg_count; ++i) {
    uint32_t source_assignment_index = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
        state, args.values[i], &source_assignment_index));
    uint32_t destination_assignment_index = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
        state, dest->arg_ids[i], &destination_assignment_index));
    state->edge_copies[state->edge_copy_count++] =
        (loom_low_allocation_edge_copy_t){
            .source_value_id = args.values[i],
            .destination_value_id = dest->arg_ids[i],
            .source_assignment_index = source_assignment_index,
            .destination_assignment_index = destination_assignment_index,
        };
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_record_edge_copy_groups(
    loom_low_allocation_build_state_t* state) {
  iree_host_size_t group_count = 0;
  iree_host_size_t copy_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_count_edge_copy_groups(
      state->body, &group_count, &copy_count));
  if (copy_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, copy_count,
                                                 sizeof(*state->edge_copies),
                                                 (void**)&state->edge_copies));
  memset(state->edge_copies, 0, copy_count * sizeof(*state->edge_copies));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, group_count, sizeof(*state->edge_copy_groups),
      (void**)&state->edge_copy_groups));
  memset(state->edge_copy_groups, 0,
         group_count * sizeof(*state->edge_copy_groups));

  uint32_t source_ordinal = 0;
  loom_block_t* block = NULL;
  loom_region_for_each_block(state->body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_br_isa(op)) {
        ++source_ordinal;
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_allocation_record_edge_copy_group(
          state, op, source_ordinal));
      ++source_ordinal;
    }
  }
  return iree_ok_status();
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_edge_copy_source_assignment(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_edge_copy_t* edge_copy) {
  IREE_ASSERT(edge_copy->source_assignment_index < state->assignment_count);
  return &state->assignments[edge_copy->source_assignment_index];
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_edge_copy_destination_assignment(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_edge_copy_t* edge_copy) {
  IREE_ASSERT(edge_copy->destination_assignment_index <
              state->assignment_count);
  return &state->assignments[edge_copy->destination_assignment_index];
}

static iree_status_t loom_low_allocation_edge_copy_unit_locations(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_edge_copy_t* edge_copy, uint32_t unit_index,
    loom_low_allocation_unit_location_t* out_source,
    loom_low_allocation_unit_location_t* out_destination) {
  const loom_low_allocation_assignment_t* source_assignment =
      loom_low_allocation_edge_copy_source_assignment(state, edge_copy);
  const loom_low_allocation_assignment_t* destination_assignment =
      loom_low_allocation_edge_copy_destination_assignment(state, edge_copy);
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_unit_location(
      source_assignment, unit_index, out_source));
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_unit_location(
      destination_assignment, unit_index, out_destination));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_group_unit_count(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_edge_copy_group_t* group,
    uint32_t* out_unit_count) {
  *out_unit_count = 0;
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &state->edge_copies[group->copy_start + i];
    const loom_low_allocation_assignment_t* source_assignment =
        loom_low_allocation_edge_copy_source_assignment(state, edge_copy);
    if (source_assignment->location_count > UINT32_MAX - *out_unit_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low.br edge-copy unit count exceeds u32");
    }
    *out_unit_count += source_assignment->location_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_group_find_destination(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_edge_copy_group_t* group,
    const loom_low_allocation_unit_location_t* destination, bool* out_found,
    loom_low_allocation_unit_location_t* out_source) {
  *out_found = false;
  *out_source = (loom_low_allocation_unit_location_t){0};
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &state->edge_copies[group->copy_start + i];
    const loom_low_allocation_assignment_t* source_assignment =
        loom_low_allocation_edge_copy_source_assignment(state, edge_copy);
    for (uint32_t unit_index = 0;
         unit_index < source_assignment->location_count; ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t candidate_destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          state, edge_copy, unit_index, &source, &candidate_destination));
      if (loom_low_allocation_unit_locations_equal(&source,
                                                   &candidate_destination)) {
        continue;
      }
      if (!loom_low_allocation_unit_storage_classes_equal(
              destination, &candidate_destination)) {
        continue;
      }
      if (!loom_low_allocation_unit_locations_equal(destination,
                                                    &candidate_destination)) {
        continue;
      }
      if (*out_found) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low allocation edge-copy group has duplicate destinations");
      }
      *out_found = true;
      *out_source = source;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_unit_starts_cycle(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_edge_copy_group_t* group,
    const loom_low_allocation_unit_location_t* destination,
    const loom_low_allocation_unit_location_t* source, uint32_t max_unit_count,
    bool* out_has_cycle) {
  *out_has_cycle = false;
  loom_low_allocation_unit_location_t next_destination = *source;
  for (uint32_t step = 0; step < max_unit_count; ++step) {
    if (loom_low_allocation_unit_locations_equal(&next_destination,
                                                 destination)) {
      *out_has_cycle = true;
      return iree_ok_status();
    }
    bool found = false;
    loom_low_allocation_unit_location_t next_source = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_group_find_destination(
        state, group, &next_destination, &found, &next_source));
    if (!found) {
      return iree_ok_status();
    }
    next_destination = next_source;
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "low allocation edge-copy cycle is malformed");
}

static iree_status_t loom_low_allocation_edge_copy_class_seen_before(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_edge_copy_group_t* group,
    const loom_low_allocation_unit_location_t* storage_class,
    uint32_t stop_copy_index, uint32_t stop_unit_index, bool* out_seen) {
  *out_seen = false;
  for (uint32_t copy_index = 0; copy_index <= stop_copy_index; ++copy_index) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &state->edge_copies[group->copy_start + copy_index];
    const loom_low_allocation_assignment_t* source_assignment =
        loom_low_allocation_edge_copy_source_assignment(state, edge_copy);
    uint32_t unit_limit = source_assignment->location_count;
    if (copy_index == stop_copy_index) {
      unit_limit = stop_unit_index;
    }
    for (uint32_t unit_index = 0; unit_index < unit_limit; ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          state, edge_copy, unit_index, &source, &destination));
      if (loom_low_allocation_unit_storage_classes_equal(storage_class,
                                                         &destination)) {
        *out_seen = true;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_class_has_cycle(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_edge_copy_group_t* group,
    const loom_low_allocation_unit_location_t* storage_class,
    bool* out_has_cycle) {
  *out_has_cycle = false;
  uint32_t unit_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_group_unit_count(
      state, group, &unit_count));
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &state->edge_copies[group->copy_start + i];
    const loom_low_allocation_assignment_t* source_assignment =
        loom_low_allocation_edge_copy_source_assignment(state, edge_copy);
    for (uint32_t unit_index = 0;
         unit_index < source_assignment->location_count; ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          state, edge_copy, unit_index, &source, &destination));
      if (!loom_low_allocation_unit_storage_classes_equal(storage_class,
                                                          &destination) ||
          loom_low_allocation_unit_locations_equal(&source, &destination)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_starts_cycle(
          state, group, &destination, &source, unit_count, out_has_cycle));
      if (*out_has_cycle) {
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_count_edge_copy_temporaries_for_group(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_edge_copy_group_t* group,
    iree_host_size_t* inout_temporary_count) {
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &state->edge_copies[group->copy_start + i];
    const loom_low_allocation_assignment_t* source_assignment =
        loom_low_allocation_edge_copy_source_assignment(state, edge_copy);
    for (uint32_t unit_index = 0;
         unit_index < source_assignment->location_count; ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          state, edge_copy, unit_index, &source, &destination));
      if (loom_low_allocation_unit_locations_equal(&source, &destination)) {
        continue;
      }
      bool seen_class = false;
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_class_seen_before(
          state, group, &destination, i, unit_index, &seen_class));
      if (seen_class) {
        continue;
      }
      bool has_cycle = false;
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_class_has_cycle(
          state, group, &destination, &has_cycle));
      if (!has_cycle) {
        continue;
      }
      if (*inout_temporary_count == IREE_HOST_SIZE_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "low.br edge-copy temporary count overflow");
      }
      ++*inout_temporary_count;
    }
  }
  return iree_ok_status();
}

static bool loom_low_allocation_location_is_live_at_point(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_unit_location_t* location, uint32_t point) {
  for (iree_host_size_t i = 0; i < state->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment = &state->assignments[i];
    if (assignment->location_kind != location->location_kind ||
        point < assignment->start_point || point >= assignment->end_point) {
      continue;
    }
    if (!loom_low_allocation_reg_classes_share_storage(
            state->target.descriptor_set, assignment->descriptor_reg_class_id,
            location->descriptor_reg_class_id)) {
      continue;
    }
    const uint64_t assignment_end =
        (uint64_t)assignment->location_base + assignment->location_count;
    if (location->location >= assignment->location_base &&
        location->location < assignment_end) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_allocation_edge_copy_group_uses_location(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_edge_copy_group_t* group,
    const loom_low_allocation_unit_location_t* location, bool* out_uses) {
  *out_uses = false;
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &state->edge_copies[group->copy_start + i];
    const loom_low_allocation_assignment_t* source_assignment =
        loom_low_allocation_edge_copy_source_assignment(state, edge_copy);
    for (uint32_t unit_index = 0;
         unit_index < source_assignment->location_count; ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          state, edge_copy, unit_index, &source, &destination));
      if (loom_low_allocation_unit_locations_equal(location, &source) ||
          loom_low_allocation_unit_locations_equal(location, &destination)) {
        *out_uses = true;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_emit_failure(
    const loom_low_allocation_build_state_t* state, const loom_op_t* op,
    loom_liveness_value_class_t value_class, uint32_t budget_units,
    uint32_t peak_units, iree_string_view_t failure_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&state->target)),
      loom_param_string(loom_low_diagnostic_export_name(&state->target)),
      loom_param_string(loom_low_diagnostic_config_key(&state->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(
          loom_low_diagnostic_value_class_name(state->module, value_class)),
      loom_param_u32(budget_units),
      loom_param_u32(peak_units),
      loom_param_string(failure_kind),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_BACKEND_005,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(state->options->emitter, &emission);
}

static iree_status_t loom_low_allocation_find_edge_copy_temporary(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_edge_copy_group_t* group,
    const loom_low_allocation_unit_location_t* storage_class,
    loom_low_allocation_unit_location_t* out_temporary) {
  *out_temporary = (loom_low_allocation_unit_location_t){0};
  if (!loom_low_allocation_location_is_register_like(
          storage_class->location_kind)) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_emit_failure(
        state, group->terminator_op, storage_class->value_class, 0, 1,
        IREE_SV("edge-copy-non-register-storage")));
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low allocation cannot reserve a branch edge-copy temporary for "
        "non-register storage");
  }

  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_class_capacity(
      state, storage_class->value_class, &capacity));
  if (capacity.location_kind != storage_class->location_kind) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_emit_failure(
        state, group->terminator_op, storage_class->value_class,
        capacity.is_bounded ? capacity.max_units : UINT32_MAX, 1,
        IREE_SV("edge-copy-storage-kind-mismatch")));
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low allocation cannot reserve a branch edge-copy temporary for a "
        "different storage kind");
  }

  uint32_t last_location = 0;
  if (capacity.is_bounded) {
    if (capacity.max_units == 0) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_emit_failure(
          state, group->terminator_op, storage_class->value_class,
          capacity.max_units, 1, IREE_SV("edge-copy-empty-budget")));
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "low allocation cannot reserve a branch "
                              "edge-copy temporary from an empty budget");
    }
    last_location = capacity.max_units - 1u;
  } else {
    last_location = loom_low_allocation_assigned_location_search_limit(
        state, storage_class->descriptor_reg_class_id,
        storage_class->location_kind);
    if (last_location == UINT32_MAX) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_emit_failure(
          state, group->terminator_op, storage_class->value_class, UINT32_MAX,
          1, IREE_SV("edge-copy-location-range-overflow")));
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "low allocation cannot reserve a branch "
                              "edge-copy temporary in uint32 range");
    }
  }

  for (uint32_t location = 0; location <= last_location; ++location) {
    loom_low_allocation_unit_location_t temporary = {
        .location_kind = storage_class->location_kind,
        .value_class = storage_class->value_class,
        .descriptor_reg_class_id = storage_class->descriptor_reg_class_id,
        .location = location,
    };
    bool group_uses_location = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_group_uses_location(
        state, group, &temporary, &group_uses_location));
    if (loom_low_allocation_reserved_range_conflicts(
            state, temporary.descriptor_reg_class_id, temporary.location_kind,
            temporary.location, 1) ||
        loom_low_allocation_location_is_live_at_point(state, &temporary,
                                                      group->program_point) ||
        group_uses_location) {
      if (location == UINT32_MAX) {
        break;
      }
      continue;
    }
    *out_temporary = temporary;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_low_allocation_emit_failure(
      state, group->terminator_op, storage_class->value_class,
      capacity.is_bounded ? capacity.max_units : UINT32_MAX, 1,
      IREE_SV("edge-copy-no-scratch-unit")));
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "low allocation cannot reserve a branch edge-copy "
                          "temporary");
}

static iree_status_t loom_low_allocation_record_edge_copy_temporaries_for_group(
    loom_low_allocation_build_state_t* state,
    loom_low_allocation_edge_copy_group_t* group) {
  if (state->edge_copy_temporary_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low.br edge-copy temporary group exceeds u32 range");
  }
  group->temporary_start = (uint32_t)state->edge_copy_temporary_count;
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &state->edge_copies[group->copy_start + i];
    const loom_low_allocation_assignment_t* source_assignment =
        loom_low_allocation_edge_copy_source_assignment(state, edge_copy);
    for (uint32_t unit_index = 0;
         unit_index < source_assignment->location_count; ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          state, edge_copy, unit_index, &source, &destination));
      if (loom_low_allocation_unit_locations_equal(&source, &destination)) {
        continue;
      }
      bool seen_class = false;
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_class_seen_before(
          state, group, &destination, i, unit_index, &seen_class));
      if (seen_class) {
        continue;
      }
      bool has_cycle = false;
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_class_has_cycle(
          state, group, &destination, &has_cycle));
      if (!has_cycle) {
        continue;
      }
      loom_low_allocation_unit_location_t temporary = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_find_edge_copy_temporary(
          state, group, &destination, &temporary));
      if (state->edge_copy_temporary_count > UINT32_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low.br edge-copy temporary group exceeds u32 range");
      }
      state->edge_copy_temporaries[state->edge_copy_temporary_count++] =
          (loom_low_allocation_edge_copy_temporary_t){
              .value_class = temporary.value_class,
              .descriptor_reg_class_id = temporary.descriptor_reg_class_id,
              .location_kind = temporary.location_kind,
              .location = temporary.location,
          };
      ++group->temporary_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_record_edge_copy_temporaries(
    loom_low_allocation_build_state_t* state) {
  iree_host_size_t temporary_count = 0;
  for (iree_host_size_t i = 0; i < state->edge_copy_group_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_count_edge_copy_temporaries_for_group(
            state, &state->edge_copy_groups[i], &temporary_count));
  }
  if (temporary_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, temporary_count, sizeof(*state->edge_copy_temporaries),
      (void**)&state->edge_copy_temporaries));
  memset(state->edge_copy_temporaries, 0,
         temporary_count * sizeof(*state->edge_copy_temporaries));
  for (iree_host_size_t i = 0; i < state->edge_copy_group_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_record_edge_copy_temporaries_for_group(
            state, &state->edge_copy_groups[i]));
  }
  return iree_ok_status();
}

static bool loom_low_allocation_op_has_packet_moves(const loom_op_t* op) {
  return loom_low_copy_isa(op) || loom_low_slice_isa(op) ||
         loom_low_concat_isa(op);
}

static iree_status_t loom_low_allocation_assignment_for_value(
    const loom_low_allocation_build_state_t* state, loom_value_id_t value_id,
    const loom_low_allocation_assignment_t** out_assignment) {
  uint32_t assignment_index = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
      state, value_id, &assignment_index));
  *out_assignment = &state->assignments[assignment_index];
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_append_packet_unit_move(
    const loom_low_allocation_assignment_t* source_assignment,
    uint32_t source_unit_index,
    const loom_low_allocation_assignment_t* destination_assignment,
    uint32_t destination_unit_index,
    loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_capacity, iree_host_size_t* inout_move_count) {
  if (*inout_move_count >= move_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "packet-local move count exceeds reserved capacity");
  }
  loom_low_allocation_packet_unit_move_t* move =
      moves ? &moves[*inout_move_count] : NULL;
  if (move) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_unit_location(
        source_assignment, source_unit_index, &move->source));
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_unit_location(
        destination_assignment, destination_unit_index, &move->destination));
  }
  ++*inout_move_count;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_moves_for_copy(
    const loom_low_allocation_build_state_t* state, const loom_op_t* op,
    loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_capacity, iree_host_size_t* out_move_count) {
  *out_move_count = 0;
  const loom_low_allocation_assignment_t* source_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_for_value(
      state, loom_low_copy_source(op), &source_assignment));
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_for_value(
      state, loom_low_copy_result(op), &result_assignment));
  if (source_assignment->location_count != result_assignment->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.copy packet-local move requires matching location counts");
  }
  for (uint32_t i = 0; i < source_assignment->location_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_append_packet_unit_move(
        source_assignment, i, result_assignment, i, moves, move_capacity,
        out_move_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_moves_for_slice(
    const loom_low_allocation_build_state_t* state, const loom_op_t* op,
    loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_capacity, iree_host_size_t* out_move_count) {
  *out_move_count = 0;
  const int64_t offset = loom_low_slice_offset(op);
  if (offset < 0 || offset > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low.slice offset is outside uint32_t range");
  }
  const loom_low_allocation_assignment_t* source_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_for_value(
      state, loom_low_slice_source(op), &source_assignment));
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_for_value(
      state, loom_low_slice_result(op), &result_assignment));
  const uint32_t source_offset = (uint32_t)offset;
  if (source_offset > source_assignment->location_count ||
      result_assignment->location_count >
          source_assignment->location_count - source_offset) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.slice packet-local move range exceeds source assignment");
  }
  for (uint32_t i = 0; i < result_assignment->location_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_append_packet_unit_move(
        source_assignment, source_offset + i, result_assignment, i, moves,
        move_capacity, out_move_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_moves_for_concat(
    const loom_low_allocation_build_state_t* state, const loom_op_t* op,
    loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_capacity, iree_host_size_t* out_move_count) {
  *out_move_count = 0;
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_for_value(
      state, loom_low_concat_result(op), &result_assignment));
  uint32_t result_offset = 0;
  loom_value_slice_t sources = loom_low_concat_sources(op);
  for (uint16_t i = 0; i < sources.count; ++i) {
    const loom_low_allocation_assignment_t* source_assignment = NULL;
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_for_value(
        state, sources.values[i], &source_assignment));
    if (result_offset > result_assignment->location_count ||
        source_assignment->location_count >
            result_assignment->location_count - result_offset) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low.concat packet-local move source ranges exceed result "
          "assignment");
    }
    for (uint32_t source_unit = 0;
         source_unit < source_assignment->location_count; ++source_unit) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_append_packet_unit_move(
          source_assignment, source_unit, result_assignment,
          result_offset + source_unit, moves, move_capacity, out_move_count));
    }
    result_offset += source_assignment->location_count;
  }
  if (result_offset != result_assignment->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.concat packet-local move sources do not fill result assignment");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_moves_for_op(
    const loom_low_allocation_build_state_t* state, const loom_op_t* op,
    loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_capacity, iree_host_size_t* out_move_count) {
  if (loom_low_copy_isa(op)) {
    return loom_low_allocation_packet_moves_for_copy(
        state, op, moves, move_capacity, out_move_count);
  }
  if (loom_low_slice_isa(op)) {
    return loom_low_allocation_packet_moves_for_slice(
        state, op, moves, move_capacity, out_move_count);
  }
  if (loom_low_concat_isa(op)) {
    return loom_low_allocation_packet_moves_for_concat(
        state, op, moves, move_capacity, out_move_count);
  }
  *out_move_count = 0;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_count_packet_move_capacity(
    loom_low_allocation_build_state_t* state,
    iree_host_size_t* out_group_capacity,
    iree_host_size_t* out_temporary_capacity) {
  *out_group_capacity = 0;
  *out_temporary_capacity = 0;
  loom_block_t* block = NULL;
  loom_region_for_each_block(state->body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_allocation_op_has_packet_moves(op)) {
        continue;
      }
      iree_host_size_t move_count = 0;
      IREE_RETURN_IF_ERROR(loom_low_allocation_packet_moves_for_op(
          state, op, /*moves=*/NULL, /*move_capacity=*/IREE_HOST_SIZE_MAX,
          &move_count));
      if (move_count == 0) {
        continue;
      }
      if (*out_group_capacity == IREE_HOST_SIZE_MAX ||
          move_count > IREE_HOST_SIZE_MAX - *out_temporary_capacity) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "packet-local move temporary capacity exceeds host size");
      }
      ++*out_group_capacity;
      *out_temporary_capacity += move_count;
    }
  }
  return iree_ok_status();
}

static bool loom_low_allocation_packet_move_uses_location(
    const loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_count,
    const loom_low_allocation_unit_location_t* location) {
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    if (loom_low_allocation_unit_locations_equal(location, &moves[i].source) ||
        loom_low_allocation_unit_locations_equal(location,
                                                 &moves[i].destination)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_allocation_packet_move_find_destination(
    const loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_count,
    const loom_low_allocation_unit_location_t* destination, bool* out_found,
    loom_low_allocation_unit_location_t* out_source) {
  *out_found = false;
  *out_source = (loom_low_allocation_unit_location_t){0};
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    const loom_low_allocation_packet_unit_move_t* move = &moves[i];
    if (loom_low_allocation_unit_locations_equal(&move->source,
                                                 &move->destination)) {
      continue;
    }
    if (!loom_low_allocation_unit_storage_classes_equal(destination,
                                                        &move->destination)) {
      continue;
    }
    if (!loom_low_allocation_unit_locations_equal(destination,
                                                  &move->destination)) {
      continue;
    }
    if (*out_found) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "packet-local move set has duplicate destinations");
    }
    *out_found = true;
    *out_source = move->source;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_packet_move_starts_cycle(
    const loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_count,
    const loom_low_allocation_unit_location_t* destination,
    const loom_low_allocation_unit_location_t* source, bool* out_has_cycle) {
  *out_has_cycle = false;
  loom_low_allocation_unit_location_t next_destination = *source;
  for (iree_host_size_t step = 0; step < move_count; ++step) {
    if (loom_low_allocation_unit_locations_equal(&next_destination,
                                                 destination)) {
      *out_has_cycle = true;
      return iree_ok_status();
    }
    bool found = false;
    loom_low_allocation_unit_location_t next_source = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_find_destination(
        moves, move_count, &next_destination, &found, &next_source));
    if (!found) {
      return iree_ok_status();
    }
    next_destination = next_source;
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "packet-local move cycle is malformed");
}

static bool loom_low_allocation_packet_move_class_seen_before(
    const loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t stop_move_index,
    const loom_low_allocation_unit_location_t* storage_class) {
  for (iree_host_size_t i = 0; i < stop_move_index; ++i) {
    if (loom_low_allocation_unit_storage_classes_equal(storage_class,
                                                       &moves[i].destination)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_allocation_find_packet_move_temporary(
    const loom_low_allocation_build_state_t* state, const loom_op_t* op,
    uint32_t program_point,
    const loom_low_allocation_unit_location_t* storage_class,
    const loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_count,
    loom_low_allocation_unit_location_t* out_temporary) {
  *out_temporary = (loom_low_allocation_unit_location_t){0};
  if (!loom_low_allocation_location_is_register_like(
          storage_class->location_kind)) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_emit_failure(
        state, op, storage_class->value_class, 0, 1,
        IREE_SV("packet-move-non-register-storage")));
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low allocation cannot reserve a packet-local move temporary for "
        "non-register storage");
  }

  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_class_capacity(
      state, storage_class->value_class, &capacity));
  if (capacity.location_kind != storage_class->location_kind) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_emit_failure(
        state, op, storage_class->value_class,
        capacity.is_bounded ? capacity.max_units : UINT32_MAX, 1,
        IREE_SV("packet-move-storage-kind-mismatch")));
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low allocation cannot reserve a packet-local move temporary for a "
        "different storage kind");
  }

  uint32_t last_location = 0;
  if (capacity.is_bounded) {
    if (capacity.max_units == 0) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_emit_failure(
          state, op, storage_class->value_class, capacity.max_units, 1,
          IREE_SV("packet-move-empty-budget")));
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "low allocation cannot reserve a packet-local move temporary from "
          "an empty budget");
    }
    last_location = capacity.max_units - 1u;
  } else {
    last_location = loom_low_allocation_assigned_location_search_limit(
        state, storage_class->descriptor_reg_class_id,
        storage_class->location_kind);
    if (last_location == UINT32_MAX) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_emit_failure(
          state, op, storage_class->value_class, UINT32_MAX, 1,
          IREE_SV("packet-move-location-range-overflow")));
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "low allocation cannot reserve a packet-local move temporary in "
          "uint32 range");
    }
  }

  for (uint32_t location = 0; location <= last_location; ++location) {
    loom_low_allocation_unit_location_t temporary = {
        .location_kind = storage_class->location_kind,
        .value_class = storage_class->value_class,
        .descriptor_reg_class_id = storage_class->descriptor_reg_class_id,
        .location = location,
    };
    if (loom_low_allocation_reserved_range_conflicts(
            state, temporary.descriptor_reg_class_id, temporary.location_kind,
            temporary.location, 1) ||
        loom_low_allocation_location_is_live_at_point(state, &temporary,
                                                      program_point) ||
        loom_low_allocation_packet_move_uses_location(moves, move_count,
                                                      &temporary)) {
      if (location == UINT32_MAX) {
        break;
      }
      continue;
    }
    *out_temporary = temporary;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_low_allocation_emit_failure(
      state, op, storage_class->value_class,
      capacity.is_bounded ? capacity.max_units : UINT32_MAX, 1,
      IREE_SV("packet-move-no-scratch-unit")));
  return iree_make_status(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      "low allocation cannot reserve a packet-local move temporary");
}

static iree_status_t loom_low_allocation_record_packet_move_temporaries_for_op(
    loom_low_allocation_build_state_t* state, const loom_op_t* op,
    uint32_t source_ordinal) {
  iree_host_size_t move_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_packet_moves_for_op(
      state, op, /*moves=*/NULL, IREE_HOST_SIZE_MAX, &move_capacity));
  if (move_capacity == 0) {
    return iree_ok_status();
  }
  loom_low_allocation_packet_unit_move_t* moves = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, move_capacity, sizeof(*moves), (void**)&moves));
  iree_host_size_t move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_packet_moves_for_op(
      state, op, moves, move_capacity, &move_count));

  uint32_t program_point = UINT32_MAX;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_op_program_point(state, op, &program_point));
  const iree_host_size_t temporary_start = state->packet_move_temporary_count;

  for (iree_host_size_t i = 0; i < move_count; ++i) {
    const loom_low_allocation_packet_unit_move_t* move = &moves[i];
    if (loom_low_allocation_unit_locations_equal(&move->source,
                                                 &move->destination) ||
        loom_low_allocation_packet_move_class_seen_before(moves, i,
                                                          &move->destination)) {
      continue;
    }
    bool has_cycle = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_starts_cycle(
        moves, move_count, &move->destination, &move->source, &has_cycle));
    if (!has_cycle) {
      continue;
    }
    if (state->packet_move_temporary_count >= UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "packet-local move temporary table exceeds u32 range");
    }
    loom_low_allocation_unit_location_t temporary = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_find_packet_move_temporary(
        state, op, program_point, &move->destination, moves, move_count,
        &temporary));
    state->packet_move_temporaries[state->packet_move_temporary_count++] =
        (loom_low_allocation_packet_move_temporary_t){
            .value_class = temporary.value_class,
            .descriptor_reg_class_id = temporary.descriptor_reg_class_id,
            .location_kind = temporary.location_kind,
            .location = temporary.location,
        };
  }

  if (state->packet_move_temporary_count == temporary_start) {
    return iree_ok_status();
  }
  if (state->packet_move_temporary_group_count >= UINT32_MAX ||
      temporary_start > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "packet-local move temporary group exceeds u32 range");
  }
  state->packet_move_temporary_groups
      [state->packet_move_temporary_group_count++] =
      (loom_low_allocation_packet_move_temporary_group_t){
          .op = op,
          .source_ordinal = source_ordinal,
          .program_point = program_point,
          .temporary_start = (uint32_t)temporary_start,
          .temporary_count =
              (uint32_t)(state->packet_move_temporary_count - temporary_start),
      };
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_record_packet_move_temporaries(
    loom_low_allocation_build_state_t* state) {
  iree_host_size_t group_capacity = 0;
  iree_host_size_t temporary_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_count_packet_move_capacity(
      state, &group_capacity, &temporary_capacity));
  if (temporary_capacity == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, group_capacity,
                                sizeof(*state->packet_move_temporary_groups),
                                (void**)&state->packet_move_temporary_groups));
  memset(state->packet_move_temporary_groups, 0,
         group_capacity * sizeof(*state->packet_move_temporary_groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, temporary_capacity, sizeof(*state->packet_move_temporaries),
      (void**)&state->packet_move_temporaries));
  memset(state->packet_move_temporaries, 0,
         temporary_capacity * sizeof(*state->packet_move_temporaries));

  uint32_t source_ordinal = 0;
  loom_block_t* block = NULL;
  loom_region_for_each_block(state->body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_allocation_op_has_packet_moves(op)) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_record_packet_move_temporaries_for_op(
                state, op, source_ordinal));
      }
      ++source_ordinal;
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
  state->assignment_indices_by_value_ordinal = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->liveness.value_count,
      sizeof(*state->assignment_indices_by_value_ordinal),
      (void**)&state->assignment_indices_by_value_ordinal));
  for (iree_host_size_t i = 0; i < state->liveness.value_count; ++i) {
    state->assignment_indices_by_value_ordinal[i] = UINT32_MAX;
  }
  state->active.assignment_indices = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, allocatable_count,
                                sizeof(*state->active.assignment_indices),
                                (void**)&state->active.assignment_indices));
  state->max_assigned_location_end_by_reg_class = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->target.descriptor_set->reg_class_count,
      sizeof(*state->max_assigned_location_end_by_reg_class),
      (void**)&state->max_assigned_location_end_by_reg_class));
  memset(state->max_assigned_location_end_by_reg_class, 0,
         state->target.descriptor_set->reg_class_count *
             sizeof(*state->max_assigned_location_end_by_reg_class));
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
    loom_low_allocation_expire_active_assignments(state, interval->start_point);

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

    bool assigned_concat_source_interval = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_assign_concat_source_interval(
        state, interval, &assigned_concat_source_interval));
    if (assigned_concat_source_interval) {
      continue;
    }

    bool assigned_structural_interval = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_assign_structural_interval(
        state, interval, &assigned_structural_interval));
    if (assigned_structural_interval) {
      continue;
    }

    loom_low_allocation_class_capacity_t capacity = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_class_capacity(
        state, interval->value_class, &capacity));
    if (state->spill_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "allocation table exceeds uint32_t range");
    }

    uint32_t location_base = 0;
    bool assigned = loom_low_allocation_find_location(state, interval, capacity,
                                                      &location_base);
    if (!assigned && !capacity.is_spillable) {
      const loom_low_descriptor_set_t* descriptor_set =
          state->target.descriptor_set;
      const loom_low_reg_class_t* reg_class =
          &descriptor_set->reg_classes[capacity.descriptor_reg_class_id];
      iree_string_view_t reg_class_name = loom_low_descriptor_set_string(
          descriptor_set, reg_class->name_string_offset);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "allocation exhausted unspillable register class '%.*s'",
          (int)reg_class_name.size, reg_class_name.data);
    }

    const loom_low_allocation_assignment_t assignment = {
        .value_id = interval->value_id,
        .value_class = interval->value_class,
        .descriptor_reg_class_id = capacity.descriptor_reg_class_id,
        .start_point = interval->start_point,
        .end_point = interval->end_point,
        .unit_count = interval->unit_count,
        .location_kind = assigned ? capacity.location_kind
                                  : LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT,
        .location_base =
            assigned ? location_base : (uint32_t)state->spill_count,
        .location_count = interval->unit_count,
    };

    uint32_t assignment_index = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_append_assignment(
        state, &assignment, &assignment_index));
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
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  if (!loom_low_allocation_location_is_register_like(lhs->location_kind) ||
      !loom_low_allocation_location_is_register_like(rhs->location_kind)) {
    return false;
  }
  if (!loom_low_allocation_assignment_classes_share_storage(descriptor_set, lhs,
                                                            rhs)) {
    return false;
  }
  return loom_low_allocation_interval_overlaps(lhs, rhs) &&
         loom_low_allocation_location_ranges_overlap(lhs, rhs);
}

static iree_status_t loom_low_allocation_map_assignment_index(
    const loom_low_allocation_table_t* table, loom_value_id_t value_id,
    uint32_t* out_assignment_index) {
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_try_map_active_value_assignment(table, value_id,
                                                          out_assignment_index);
  if (assignment) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "allocation table is missing assignment for value "
                          "%u",
                          (unsigned)value_id);
}

static bool loom_low_allocation_relation_cause_can_alias(
    loom_low_placement_cause_t cause) {
  switch (cause) {
    case LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE:
    case LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_low_allocation_assignment_value_ordinal(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment,
    loom_value_ordinal_t* out_value_ordinal) {
  const loom_value_ordinal_t value_ordinal =
      loom_module_value_ordinal_scratch_lookup(table->module,
                                               assignment->value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      value_ordinal >= table->placement.value_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation assignment value %u is outside the placement table",
        (unsigned)assignment->value_id);
  }
  if (loom_low_placement_value_id(&table->placement, value_ordinal) !=
      assignment->value_id) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation placement ordinal does not match assignment value");
  }
  *out_value_ordinal = value_ordinal;
  return iree_ok_status();
}

static bool loom_low_allocation_relation_matches_pair(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs,
    const loom_low_placement_relation_t* relation) {
  const loom_value_id_t result_value_id =
      loom_low_placement_value_id(&table->placement, relation->result_ordinal);
  const loom_value_id_t source_value_id =
      loom_low_placement_value_id(&table->placement, relation->source_ordinal);
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  const loom_low_allocation_assignment_t* source_assignment = NULL;
  if (lhs->value_id == result_value_id && rhs->value_id == source_value_id) {
    result_assignment = lhs;
    source_assignment = rhs;
  } else if (rhs->value_id == result_value_id &&
             lhs->value_id == source_value_id) {
    result_assignment = rhs;
    source_assignment = lhs;
  } else {
    return false;
  }
  return loom_low_allocation_assignment_subranges_match(
      result_assignment, relation->result_unit_offset, source_assignment,
      relation->source_unit_offset, relation->unit_count);
}

static iree_status_t loom_low_allocation_pair_is_placement_alias(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs, bool* out_is_alias) {
  *out_is_alias = false;
  loom_value_ordinal_t lhs_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_value_ordinal(table, lhs, &lhs_ordinal));
  loom_value_ordinal_t rhs_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_value_ordinal(table, rhs, &rhs_ordinal));
  const loom_value_ordinal_t ordinals[] = {lhs_ordinal, rhs_ordinal};
  for (iree_host_size_t ordinal_index = 0;
       ordinal_index < IREE_ARRAYSIZE(ordinals); ++ordinal_index) {
    const loom_low_placement_relation_range_t range =
        loom_low_placement_relation_range_for_value_ordinal(
            &table->placement, ordinals[ordinal_index]);
    for (uint32_t i = 0; i < range.count; ++i) {
      const loom_low_placement_relation_t* relation =
          &table->placement.relations[range.start + i];
      if (!loom_low_allocation_relation_cause_can_alias(relation->cause)) {
        continue;
      }
      if (loom_low_allocation_relation_matches_pair(table, lhs, rhs,
                                                    relation)) {
        *out_is_alias = true;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_tied_result_assignments(
    const loom_low_allocation_table_t* table) {
  for (iree_host_size_t i = 0; i < table->placement.relation_count; ++i) {
    const loom_low_placement_relation_t* relation =
        &table->placement.relations[i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
      continue;
    }
    const loom_value_id_t result_id = loom_low_placement_value_id(
        &table->placement, relation->result_ordinal);
    const loom_value_id_t operand_id = loom_low_placement_value_id(
        &table->placement, relation->source_ordinal);
    uint32_t result_assignment_index = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_map_assignment_index(
        table, result_id, &result_assignment_index));
    uint32_t operand_assignment_index = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_map_assignment_index(
        table, operand_id, &operand_assignment_index));

    const loom_low_allocation_assignment_t* result_assignment =
        &table->assignments[result_assignment_index];
    const loom_low_allocation_assignment_t* operand_assignment =
        &table->assignments[operand_assignment_index];
    if (!loom_low_allocation_assignment_is_coalescable(result_assignment) ||
        !loom_low_allocation_assignment_is_coalescable(operand_assignment) ||
        !loom_low_allocation_assignment_locations_equal(result_assignment,
                                                        operand_assignment)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "allocation tied result does not share the operand location");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_assignment(
    const loom_low_allocation_table_t* table,
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
  if (assignment->descriptor_reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      assignment->descriptor_reg_class_id >=
          table->target.descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation assignment %zu has invalid descriptor register class ID "
        "%" PRIu16,
        assignment_index, assignment->descriptor_reg_class_id);
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
        "allocation remark %zu references assignment %u, but table has only "
        "%zu assignments",
        remark_index, remark->assignment_index, assignment_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_spill_plan(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* spill_plan,
    iree_host_size_t spill_plan_index) {
  if (spill_plan->assignment_index >= table->assignment_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation spill plan %zu references assignment %u, but table has "
        "only %zu assignments",
        spill_plan_index, spill_plan->assignment_index,
        table->assignment_count);
  }
  const loom_low_allocation_assignment_t* assignment =
      &table->assignments[spill_plan->assignment_index];
  if (spill_plan->value_id >= table->module->values.count) {
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
    const loom_low_allocation_table_t* table,
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
  if (copy_decision->source_assignment_index >= table->assignment_count ||
      copy_decision->result_assignment_index >= table->assignment_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation copy decision %zu references assignments %u and %u, but "
        "table has only %zu assignments",
        copy_decision_index, copy_decision->source_assignment_index,
        copy_decision->result_assignment_index, table->assignment_count);
  }
  const loom_low_allocation_assignment_t* source_assignment =
      &table->assignments[copy_decision->source_assignment_index];
  const loom_low_allocation_assignment_t* result_assignment =
      &table->assignments[copy_decision->result_assignment_index];
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

static iree_status_t loom_low_allocation_verify_edge_copy(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_edge_copy_t* edge_copy,
    iree_host_size_t edge_copy_index) {
  if (edge_copy->source_assignment_index >= table->assignment_count ||
      edge_copy->destination_assignment_index >= table->assignment_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy %zu references assignment %u -> %u outside "
        "assignment count %zu",
        edge_copy_index, edge_copy->source_assignment_index,
        edge_copy->destination_assignment_index, table->assignment_count);
  }
  const loom_low_allocation_assignment_t* source_assignment =
      &table->assignments[edge_copy->source_assignment_index];
  const loom_low_allocation_assignment_t* destination_assignment =
      &table->assignments[edge_copy->destination_assignment_index];
  if (source_assignment->value_id != edge_copy->source_value_id ||
      destination_assignment->value_id != edge_copy->destination_value_id) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy %zu assignment indices do not match values",
        edge_copy_index);
  }
  if (source_assignment->descriptor_reg_class_id !=
      destination_assignment->descriptor_reg_class_id) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy %zu crosses descriptor register classes",
        edge_copy_index);
  }
  if (source_assignment->location_count == 0 ||
      source_assignment->location_count !=
          destination_assignment->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy %zu requires non-empty matching location ranges",
        edge_copy_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_edge_copy_group(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_edge_copy_group_t* group,
    iree_host_size_t group_index, uint32_t expected_source_ordinal,
    iree_host_size_t expected_copy_start) {
  if (!group->terminator_op || !loom_low_br_isa(group->terminator_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation edge-copy group %zu is not low.br",
                            group_index);
  }
  if (group->copy_start != expected_copy_start) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy group %zu starts at %u but expected %zu",
        group_index, group->copy_start, expected_copy_start);
  }
  if (group->source_ordinal != expected_source_ordinal) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy group %zu has source ordinal %u but expected %u",
        group_index, group->source_ordinal, expected_source_ordinal);
  }
  uint32_t expected_program_point = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_low_allocation_op_program_point_in_liveness(
      &table->liveness, group->terminator_op, &expected_program_point));
  if (group->program_point != expected_program_point) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy group %zu has program point %u but expected %u",
        group_index, group->program_point, expected_program_point);
  }
  if (group->copy_start > table->edge_copy_count ||
      group->copy_count > table->edge_copy_count - group->copy_start) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy group %zu range is outside edge-copy count %zu",
        group_index, table->edge_copy_count);
  }
  if (group->temporary_start > table->edge_copy_temporary_count ||
      group->temporary_count >
          table->edge_copy_temporary_count - group->temporary_start) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy group %zu temporary range is outside temporary "
        "count %zu",
        group_index, table->edge_copy_temporary_count);
  }
  loom_value_slice_t args = loom_low_br_args(group->terminator_op);
  const loom_block_t* dest = loom_low_br_dest(group->terminator_op);
  if (args.count != dest->arg_count || args.count != group->copy_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy group %zu does not match low.br edge payload",
        group_index);
  }
  for (uint16_t i = 0; i < dest->arg_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &table->edge_copies[group->copy_start + i];
    if (edge_copy->source_value_id != args.values[i] ||
        edge_copy->destination_value_id != dest->arg_ids[i]) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "allocation edge-copy group %zu payload %u does not match low.br",
          group_index, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_edge_copy_temporary(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_edge_copy_temporary_t* temporary,
    iree_host_size_t temporary_index) {
  if (!loom_low_allocation_location_kind_is_known(temporary->location_kind) ||
      !loom_low_allocation_location_is_register_like(
          temporary->location_kind)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy temporary %zu has invalid location kind %u",
        temporary_index, (unsigned)temporary->location_kind);
  }
  if (temporary->descriptor_reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      temporary->descriptor_reg_class_id >=
          table->target.descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy temporary %zu has invalid descriptor register "
        "class ID %" PRIu16,
        temporary_index, temporary->descriptor_reg_class_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_packet_move_temporary_group(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_packet_move_temporary_group_t* group,
    iree_host_size_t group_index, iree_host_size_t expected_temporary_start) {
  if (!group->op || !loom_low_allocation_op_has_packet_moves(group->op)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation packet-local move temporary group %zu does not reference a "
        "packet-local move op",
        group_index);
  }
  if (group->temporary_start != expected_temporary_start) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation packet-local move temporary group %zu starts at %u but "
        "expected %zu",
        group_index, group->temporary_start, expected_temporary_start);
  }
  if (group->temporary_start > table->packet_move_temporary_count ||
      group->temporary_count >
          table->packet_move_temporary_count - group->temporary_start) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation packet-local move temporary group %zu range is outside "
        "temporary count %zu",
        group_index, table->packet_move_temporary_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_packet_move_temporary(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_packet_move_temporary_t* temporary,
    iree_host_size_t temporary_index) {
  if (!loom_low_allocation_location_kind_is_known(temporary->location_kind) ||
      !loom_low_allocation_location_is_register_like(
          temporary->location_kind)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation packet-local move temporary %zu has invalid location kind "
        "%u",
        temporary_index, (unsigned)temporary->location_kind);
  }
  if (temporary->descriptor_reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      temporary->descriptor_reg_class_id >=
          table->target.descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation packet-local move temporary %zu has invalid descriptor "
        "register class ID %" PRIu16,
        temporary_index, temporary->descriptor_reg_class_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_table_contents(
    const loom_low_allocation_table_t* table) {
  loom_region_t* body = loom_low_function_body((loom_op_t*)table->function_op);
  if (table->placement.module != table->module ||
      table->placement.region != body ||
      table->placement.value_ids != table->liveness.value_ids ||
      table->placement.value_count != table->liveness.value_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation table placement analysis must match liveness");
  }
  iree_host_size_t spill_count = 0;
  for (iree_host_size_t i = 0; i < table->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment = &table->assignments[i];
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_verify_assignment(table, assignment, i));
    if (assignment->location_kind == LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT) {
      ++spill_count;
    }
  }
  for (iree_host_size_t i = 0; i < table->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* lhs = &table->assignments[i];
    for (iree_host_size_t j = i + 1; j < table->assignment_count; ++j) {
      const loom_low_allocation_assignment_t* rhs = &table->assignments[j];
      if (!loom_low_allocation_assignment_pair_conflicts(
              table->target.descriptor_set, lhs, rhs)) {
        continue;
      }
      bool pair_is_placement_alias = false;
      IREE_RETURN_IF_ERROR(loom_low_allocation_pair_is_placement_alias(
          table, lhs, rhs, &pair_is_placement_alias));
      if (pair_is_placement_alias) {
        continue;
      }
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "allocation assigns overlapping live intervals to the same location");
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_verify_tied_result_assignments(table));
  for (iree_host_size_t i = 0; i < table->remark_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_remark(
        &table->remarks[i], i, table->assignment_count));
  }
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_spill_plan(
        table, &table->spill_plans[i], i));
  }
  iree_host_size_t coalesced_copy_count = 0;
  iree_host_size_t materialized_copy_count = 0;
  for (iree_host_size_t i = 0; i < table->copy_decision_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_copy_decision(
        table, &table->copy_decisions[i], i));
    switch (table->copy_decisions[i].kind) {
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
  iree_host_size_t edge_copy_group_count = 0;
  iree_host_size_t edge_copy_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_count_edge_copy_groups(
      body, &edge_copy_group_count, &edge_copy_count));
  if (table->edge_copy_group_count != edge_copy_group_count ||
      table->edge_copy_count != edge_copy_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation table has %zu edge-copy groups/%zu edge copies for "
        "%zu/%zu low.br edge payloads",
        table->edge_copy_group_count, table->edge_copy_count,
        edge_copy_group_count, edge_copy_count);
  }
  iree_host_size_t expected_edge_copy_start = 0;
  iree_host_size_t expected_edge_copy_temporary_start = 0;
  iree_host_size_t expected_edge_copy_group_index = 0;
  uint32_t source_ordinal = 0;
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_br_isa(op) || loom_low_br_args(op).count == 0) {
        ++source_ordinal;
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_allocation_verify_edge_copy_group(
          table, &table->edge_copy_groups[expected_edge_copy_group_index],
          expected_edge_copy_group_index, source_ordinal,
          expected_edge_copy_start));
      if (table->edge_copy_groups[expected_edge_copy_group_index]
              .temporary_start != expected_edge_copy_temporary_start) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "allocation edge-copy group %zu temporary range is not "
            "contiguous",
            expected_edge_copy_group_index);
      }
      expected_edge_copy_start +=
          table->edge_copy_groups[expected_edge_copy_group_index].copy_count;
      expected_edge_copy_temporary_start +=
          table->edge_copy_groups[expected_edge_copy_group_index]
              .temporary_count;
      ++expected_edge_copy_group_index;
      ++source_ordinal;
    }
  }
  if (expected_edge_copy_temporary_start != table->edge_copy_temporary_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy temporary ranges cover %zu records but table "
        "has %zu",
        expected_edge_copy_temporary_start, table->edge_copy_temporary_count);
  }
  for (iree_host_size_t i = 0; i < table->edge_copy_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_verify_edge_copy(table, &table->edge_copies[i], i));
  }
  for (iree_host_size_t i = 0; i < table->edge_copy_temporary_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_edge_copy_temporary(
        table, &table->edge_copy_temporaries[i], i));
  }
  iree_host_size_t expected_packet_move_temporary_start = 0;
  uint32_t previous_packet_move_source_ordinal = 0;
  for (iree_host_size_t i = 0; i < table->packet_move_temporary_group_count;
       ++i) {
    const loom_low_allocation_packet_move_temporary_group_t* group =
        &table->packet_move_temporary_groups[i];
    if (i != 0 &&
        group->source_ordinal <= previous_packet_move_source_ordinal) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "allocation packet-local move temporary groups are not sorted by "
          "source ordinal");
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_packet_move_temporary_group(
        table, group, i, expected_packet_move_temporary_start));
    expected_packet_move_temporary_start += group->temporary_count;
    previous_packet_move_source_ordinal = group->source_ordinal;
  }
  if (expected_packet_move_temporary_start !=
      table->packet_move_temporary_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation packet-local move temporary ranges cover %zu records but "
        "table has %zu",
        expected_packet_move_temporary_start,
        table->packet_move_temporary_count);
  }
  for (iree_host_size_t i = 0; i < table->packet_move_temporary_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_packet_move_temporary(
        table, &table->packet_move_temporaries[i], i));
  }
  iree_host_size_t expected_copy_decision_count =
      loom_low_allocation_count_copy_ops(body);
  if (table->copy_decision_count != expected_copy_decision_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation table has %zu copy decisions for %zu low.copy ops",
        table->copy_decision_count, expected_copy_decision_count);
  }
  if (table->spill_count != spill_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation table spill_count is %zu but assignments contain %zu "
        "spills",
        table->spill_count, spill_count);
  }
  if (table->spill_plan_count != spill_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation table has %zu spill plans for %zu spilled assignments",
        table->spill_plan_count, spill_count);
  }
  if (table->coalesced_copy_count != coalesced_copy_count ||
      table->materialized_copy_count != materialized_copy_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation table copy counters do not match copy decisions");
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_verify_table(
    const loom_low_allocation_table_t* table) {
  loom_low_allocation_value_scratch_t scratch = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_acquire_value_scratch(table, &scratch));
  iree_status_t status = loom_low_allocation_verify_table_contents(table);
  loom_low_allocation_release_value_scratch(&scratch);
  return status;
}

iree_status_t loom_low_allocation_acquire_value_scratch(
    const loom_low_allocation_table_t* table,
    loom_low_allocation_value_scratch_t* out_scratch) {
  *out_scratch = (loom_low_allocation_value_scratch_t){
      .module = table->module,
      .table = table,
      .value_ids = table->liveness.value_ids,
      .value_count = table->liveness.value_count,
  };
  if (table->liveness.value_count >= LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation value count exceeds local ordinal "
                            "range");
  }
  loom_module_value_ordinal_scratch_acquire(table->module);
  out_scratch->flags |= LOOM_LOW_ALLOCATION_VALUE_SCRATCH_FLAG_ACQUIRED;
  iree_host_size_t value_ordinal = 0;
  iree_status_t status = iree_ok_status();
  for (; value_ordinal < table->liveness.value_count; ++value_ordinal) {
    const loom_value_id_t value_id = table->liveness.value_ids[value_ordinal];
    if (value_id >= table->module->values.count) {
      status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "allocation liveness value id %u is out of range",
                           (unsigned)value_id);
      break;
    }
    loom_module_value_ordinal_scratch_set(table->module, value_id,
                                          (loom_value_ordinal_t)value_ordinal);
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < value_ordinal; ++i) {
      loom_module_value_ordinal_scratch_clear(table->module,
                                              table->liveness.value_ids[i]);
    }
    loom_module_value_ordinal_scratch_release(table->module);
    out_scratch->flags &= ~LOOM_LOW_ALLOCATION_VALUE_SCRATCH_FLAG_ACQUIRED;
  }
  return status;
}

void loom_low_allocation_release_value_scratch(
    loom_low_allocation_value_scratch_t* scratch) {
  if (!scratch ||
      !iree_any_bit_set(scratch->flags,
                        LOOM_LOW_ALLOCATION_VALUE_SCRATCH_FLAG_ACQUIRED)) {
    return;
  }
  for (iree_host_size_t i = 0; i < scratch->value_count; ++i) {
    loom_module_value_ordinal_scratch_clear(scratch->module,
                                            scratch->value_ids[i]);
  }
  loom_module_value_ordinal_scratch_release(scratch->module);
  scratch->flags &= ~LOOM_LOW_ALLOCATION_VALUE_SCRATCH_FLAG_ACQUIRED;
}

const loom_low_allocation_assignment_t*
loom_low_allocation_assignment_for_value_ordinal(
    const loom_low_allocation_table_t* table,
    loom_value_ordinal_t value_ordinal, uint32_t* out_assignment_index) {
  if (out_assignment_index) {
    *out_assignment_index = UINT32_MAX;
  }
  if (!table->assignment_indices_by_value_ordinal ||
      value_ordinal >= table->liveness.value_count) {
    return NULL;
  }
  const uint32_t assignment_index =
      table->assignment_indices_by_value_ordinal[value_ordinal];
  if (assignment_index == UINT32_MAX) {
    return NULL;
  }
  IREE_ASSERT(assignment_index < table->assignment_count);
  const loom_low_allocation_assignment_t* assignment =
      &table->assignments[assignment_index];
  IREE_ASSERT_EQ(assignment->value_id,
                 table->liveness.value_ids[value_ordinal]);
  if (out_assignment_index) {
    *out_assignment_index = assignment_index;
  }
  return assignment;
}

const loom_low_allocation_assignment_t*
loom_low_allocation_try_map_active_value_assignment(
    const loom_low_allocation_table_t* table, loom_value_id_t value_id,
    uint32_t* out_assignment_index) {
  const loom_value_ordinal_t value_ordinal =
      loom_module_value_ordinal_scratch_lookup(table->module, value_id);
  if (value_ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    if (out_assignment_index) {
      *out_assignment_index = UINT32_MAX;
    }
    return NULL;
  }
  return loom_low_allocation_assignment_for_value_ordinal(table, value_ordinal,
                                                          out_assignment_index);
}

const loom_low_allocation_assignment_t*
loom_low_allocation_map_active_value_assignment(
    const loom_low_allocation_table_t* table, loom_value_id_t value_id,
    uint32_t* out_assignment_index) {
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_try_map_active_value_assignment(table, value_id,
                                                          out_assignment_index);
  IREE_ASSERT(assignment != NULL);
  return assignment;
}

const loom_low_allocation_edge_copy_group_t*
loom_low_allocation_find_edge_copy_group_by_source_ordinal(
    const loom_low_allocation_table_t* table, uint32_t source_ordinal) {
  iree_host_size_t lower = 0;
  iree_host_size_t upper = table->edge_copy_group_count;
  while (lower < upper) {
    iree_host_size_t middle = lower + (upper - lower) / 2;
    const loom_low_allocation_edge_copy_group_t* group =
        &table->edge_copy_groups[middle];
    if (source_ordinal < group->source_ordinal) {
      upper = middle;
    } else if (source_ordinal > group->source_ordinal) {
      lower = middle + 1;
    } else {
      return group;
    }
  }
  return NULL;
}

const loom_low_allocation_packet_move_temporary_group_t*
loom_low_allocation_find_packet_move_temporary_group_by_source_ordinal(
    const loom_low_allocation_table_t* table, uint32_t source_ordinal) {
  iree_host_size_t lower = 0;
  iree_host_size_t upper = table->packet_move_temporary_group_count;
  while (lower < upper) {
    iree_host_size_t middle = lower + (upper - lower) / 2;
    const loom_low_allocation_packet_move_temporary_group_t* group =
        &table->packet_move_temporary_groups[middle];
    if (source_ordinal < group->source_ordinal) {
      upper = middle;
    } else if (source_ordinal > group->source_ordinal) {
      lower = middle + 1;
    } else {
      return group;
    }
  }
  return NULL;
}

iree_status_t loom_low_allocation_assignment_register_class_name(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment,
    iree_string_view_t* out_register_class_name) {
  *out_register_class_name = iree_string_view_empty();
  if (assignment->descriptor_reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      assignment->descriptor_reg_class_id >=
          table->target.descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation assignment %" PRIu32
        " has invalid descriptor register class ID %" PRIu16,
        assignment->value_id, assignment->descriptor_reg_class_id);
  }
  const loom_low_reg_class_t* reg_class =
      &table->target.descriptor_set
           ->reg_classes[assignment->descriptor_reg_class_id];
  *out_register_class_name = loom_low_descriptor_set_string(
      table->target.descriptor_set, reg_class->name_string_offset);
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_emit_predicted_spills(
    const loom_low_allocation_table_t* table,
    iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan = &table->spill_plans[i];
    const loom_low_allocation_assignment_t* assignment =
        &table->assignments[spill_plan->assignment_index];
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_diagnostic_target_key(&table->target)),
        loom_param_string(loom_low_diagnostic_export_name(&table->target)),
        loom_param_string(loom_low_diagnostic_config_key(&table->target)),
        loom_param_string(loom_low_diagnostic_function_name(
            table->module, table->function_op)),
        loom_param_string(loom_low_diagnostic_value_name(table->module,
                                                         spill_plan->value_id)),
        loom_param_string(loom_low_diagnostic_value_class_name(
            table->module, assignment->value_class)),
        loom_param_u32(spill_plan->byte_size),
        loom_param_u32(spill_plan->store_count),
        loom_param_u32(spill_plan->reload_count),
        loom_param_string(IREE_SV("register-budget-conflict")),
    };
    loom_diagnostic_emission_t emission = {
        .op = loom_low_diagnostic_value_origin_op(
            table->module, spill_plan->value_id, table->function_op),
        .error = LOOM_ERR_BACKEND_008,
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

static iree_string_view_t loom_low_allocation_coalescing_constraint(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_copy_decision_t* copy_decision) {
  const loom_low_allocation_assignment_t* source_assignment =
      &table->assignments[copy_decision->source_assignment_index];
  const loom_low_allocation_assignment_t* result_assignment =
      &table->assignments[copy_decision->result_assignment_index];
  if (!loom_low_allocation_assignment_is_coalescable(source_assignment)) {
    return IREE_SV("source-not-register-like");
  }
  if (!loom_low_allocation_assignment_is_coalescable(result_assignment)) {
    return IREE_SV("result-not-register-like");
  }
  if (loom_low_allocation_assignment_locations_equal(source_assignment,
                                                     result_assignment)) {
    return IREE_SV("assigned-locations-match");
  }
  return IREE_SV("assigned-locations-differ");
}

static iree_status_t loom_low_allocation_emit_copy_decisions(
    const loom_low_allocation_table_t* table,
    iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < table->copy_decision_count; ++i) {
    const loom_low_allocation_copy_decision_t* copy_decision =
        &table->copy_decisions[i];
    loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_diagnostic_target_key(&table->target)),
        loom_param_string(loom_low_diagnostic_export_name(&table->target)),
        loom_param_string(loom_low_diagnostic_config_key(&table->target)),
        loom_param_string(loom_low_diagnostic_function_name(
            table->module, table->function_op)),
        loom_param_string(loom_low_diagnostic_value_name(
            table->module, copy_decision->source_value_id)),
        loom_param_string(loom_low_diagnostic_value_name(
            table->module, copy_decision->result_value_id)),
        loom_param_string(
            loom_low_allocation_copy_decision_name(copy_decision->kind)),
        loom_param_string(
            loom_low_allocation_coalescing_constraint(table, copy_decision)),
    };
    loom_diagnostic_emission_t emission = {
        .op = loom_low_diagnostic_value_origin_op(
            table->module, copy_decision->result_value_id, table->function_op),
        .error = LOOM_ERR_BACKEND_006,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    IREE_RETURN_IF_ERROR(iree_diagnostic_emit(emitter, &emission));
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocate_function(
    loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_allocation_options_t* options, iree_arena_allocator_t* arena,
    loom_low_allocation_table_t* out_table) {
  if (!loom_low_function_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def or low.kernel.def");
  }
  *out_table = (loom_low_allocation_table_t){0};

  loom_low_allocation_build_state_t state = {
      .module = module,
      .options = options,
      .arena = arena,
      .body = loom_low_function_body((loom_op_t*)low_func_op),
      .function_op = low_func_op,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_validate_synthesis_mode(low_func_op));

  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, low_func_op, options->descriptor_registry, options->emitter,
      &state.target));
  if (!state.target.descriptor_set) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low function target did not resolve");
  }
  loom_local_value_domain_t value_domain = {0};
  IREE_RETURN_IF_ERROR(loom_low_register_class_map_initialize(
      module, state.target.descriptor_set, arena, &state.register_class_map));
  IREE_RETURN_IF_ERROR(loom_low_allocation_resolve_budgets(&state));
  IREE_RETURN_IF_ERROR(loom_low_allocation_resolve_reserved_ranges(&state));

  iree_status_t status = loom_local_value_domain_acquire_for_region(
      module, state.body, arena, &value_domain);
  if (iree_status_is_ok(status)) {
    state.value_domain = &value_domain;
    status = loom_liveness_analyze_local_value_domain(
        &value_domain, options->liveness_order, arena, &state.liveness);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_placement_analyze_region(module, state.body,
                                               &value_domain, &state.liveness,
                                               arena, &state.placement);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_resolve_fixed_values(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_assign_intervals(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_record_copy_decisions(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_record_edge_copy_groups(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_record_edge_copy_temporaries(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_record_packet_move_temporaries(&state);
  }

  loom_low_allocation_table_t table = {0};
  if (iree_status_is_ok(status)) {
    table = (loom_low_allocation_table_t){
        .module = module,
        .function_op = low_func_op,
        .target = state.target,
        .liveness = state.liveness,
        .placement = state.placement,
        .allocation_mode = loom_low_function_allocation(low_func_op),
        .assignments = state.assignments,
        .assignment_count = state.assignment_count,
        .assignment_indices_by_value_ordinal =
            state.assignment_indices_by_value_ordinal,
        .spill_plans = state.spill_plans,
        .spill_plan_count = state.spill_plan_count,
        .remarks = state.remarks,
        .remark_count = state.remark_count,
        .copy_decisions = state.copy_decisions,
        .copy_decision_count = state.copy_decision_count,
        .edge_copies = state.edge_copies,
        .edge_copy_count = state.edge_copy_count,
        .edge_copy_groups = state.edge_copy_groups,
        .edge_copy_group_count = state.edge_copy_group_count,
        .edge_copy_temporaries = state.edge_copy_temporaries,
        .edge_copy_temporary_count = state.edge_copy_temporary_count,
        .packet_move_temporary_groups = state.packet_move_temporary_groups,
        .packet_move_temporary_group_count =
            state.packet_move_temporary_group_count,
        .packet_move_temporaries = state.packet_move_temporaries,
        .packet_move_temporary_count = state.packet_move_temporary_count,
        .spill_count = state.spill_count,
        .coalesced_copy_count = state.coalesced_copy_count,
        .materialized_copy_count = state.materialized_copy_count,
    };
    loom_target_bundle_storage_rebind(&table.target.bundle_storage);
  }
  loom_local_value_domain_release(&value_domain);
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_verify_table(&table);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_ALLOCATION_DIAGNOSTIC_PREDICTED_SPILLS)) {
    status =
        loom_low_allocation_emit_predicted_spills(&table, options->emitter);
  }
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_ALLOCATION_DIAGNOSTIC_COPY_DECISIONS)) {
    status = loom_low_allocation_emit_copy_decisions(&table, options->emitter);
  }
  if (iree_status_is_ok(status)) {
    *out_table = table;
  }
  return status;
}
