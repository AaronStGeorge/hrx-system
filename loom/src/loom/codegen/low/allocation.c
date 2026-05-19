// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation.h"

#include <inttypes.h>
#include <string.h>

#include "loom/analysis/consumption.h"
#include "loom/codegen/low/allocation/active_unit.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

typedef struct loom_low_allocation_storage_lease_unit_entry_t {
  // Storage-lease instance occupying this leased unit.
  uint32_t storage_lease_index;
  // Next entry in the hashed unit bucket.
  uint32_t next_entry;
  // Target-storage identity key for the register class owning this unit.
  uint32_t storage_key;
  // Target-visible storage kind for this unit.
  loom_low_allocation_location_kind_t location_kind;
  // Physical register or target ID for this unit.
  uint32_t location;
} loom_low_allocation_storage_lease_unit_entry_t;

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
  // Reusable consumed-value query for |body|.
  loom_consumption_region_query_t consumption_query;
  // True once |consumption_query| has been initialized.
  bool consumption_query_initialized;
  // Active function-local value domain shared with liveness/allocation.
  const loom_local_value_domain_t* value_domain;
  // Unit end-point starts indexed by liveness local value ordinal. Values
  // without allocatable unit liveness contain UINT32_MAX.
  uint32_t* unit_end_point_starts_by_value_ordinal;
  // Mutable per-assignment-unit live end points.
  uint32_t* unit_end_points;
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
  // Hash index for active register-like assignment units.
  loom_low_allocation_active_unit_index_t active_units;
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
  // Mutable assignment-backed storage-lease records being built.
  loom_low_allocation_storage_lease_t* storage_lease_instances;
  // Mutable allocator-requested storage release actions being built.
  loom_low_storage_release_action_t* storage_release_actions;
  // Storage-lease record heads indexed by allocation-local value ordinal.
  uint32_t* storage_lease_record_heads_by_value_ordinal;
  // Next storage-lease record index for the same allocation-local value.
  uint32_t* storage_lease_next_record_indices;
  // True when the storage-lease record has a materialized instance.
  uint8_t* storage_lease_instance_written;
  // Hash index for materialized storage-lease units.
  struct {
    // Bucket heads into |entries|. Missing buckets contain UINT32_MAX.
    uint32_t* bucket_heads;
    // Power-of-two number of entries in |bucket_heads|.
    uint32_t bucket_count;
    // Unit entries stored for materialized storage leases.
    loom_low_allocation_storage_lease_unit_entry_t* entries;
    // Maximum number of entries that can be appended to |entries|.
    iree_host_size_t entry_capacity;
    // Number of initialized entries in |entries|.
    iree_host_size_t entry_count;
  } storage_lease_units;
  // Number of initialized assignment records.
  iree_host_size_t assignment_count;
  // Number of initialized unit end-point records.
  iree_host_size_t unit_end_point_count;
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
  // Number of initialized assignment-backed storage-lease records.
  iree_host_size_t storage_lease_instance_count;
  // Number of initialized storage release actions.
  iree_host_size_t storage_release_action_count;
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

typedef enum loom_low_allocation_storage_release_policy_e {
  // Active storage leases are hard conflicts.
  LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN = 0,
  // Active storage leases may be released before the candidate assignment.
  LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED = 1,
} loom_low_allocation_storage_release_policy_t;

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

static const loom_low_reg_class_t* loom_low_allocation_reg_class_at(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id) {
  IREE_ASSERT(descriptor_set != NULL);
  IREE_ASSERT(reg_class_id != LOOM_LOW_REG_CLASS_NONE &&
              reg_class_id < descriptor_set->reg_class_count);
  return &descriptor_set->reg_classes[reg_class_id];
}

// Branch placement may make two values overlap in the linear interval space
// even when no block can observe both values live at once. Only those
// CFG-induced overlaps are safe to ignore during phi-style coalescing.
static bool loom_low_allocation_can_ignore_branch_counterpart_conflict(
    const loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    const loom_low_allocation_assignment_t* counterpart) {
  if (!loom_low_allocation_live_range_assignment_overlaps_interval(counterpart,
                                                                   interval)) {
    return false;
  }
  return !loom_low_allocation_live_range_values_overlap(
      &state->liveness, interval->value_id, interval->start_point,
      interval->end_point, counterpart->value_id, counterpart->start_point,
      counterpart->end_point);
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

static uint32_t loom_low_allocation_unit_end_point_start_for_value_ordinal(
    const loom_low_allocation_build_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  IREE_ASSERT(state->unit_end_point_starts_by_value_ordinal != NULL);
  IREE_ASSERT_LT(value_ordinal, state->liveness.value_count);
  return state->unit_end_point_starts_by_value_ordinal[value_ordinal];
}

static uint32_t loom_low_allocation_unit_end_point_start_for_value(
    const loom_low_allocation_build_state_t* state, loom_value_id_t value_id) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_value_ordinal_for_value(state, value_id,
                                                   &value_ordinal)) {
    return UINT32_MAX;
  }
  return loom_low_allocation_unit_end_point_start_for_value_ordinal(
      state, value_ordinal);
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

// Allocation intervals describe target-visible storage occupancy. A semantic
// dead result still writes a physical destination at its definition boundary,
// so it must occupy storage until the next program boundary.
static uint32_t loom_low_allocation_interval_storage_end_point(
    const loom_liveness_interval_t* interval) {
  if (interval->end_point > interval->start_point) {
    return interval->end_point;
  }
  return interval->start_point == UINT32_MAX ? UINT32_MAX
                                             : interval->start_point + 1u;
}

static uint32_t loom_low_allocation_interval_initial_unit_end_point(
    const loom_liveness_interval_t* interval) {
  if (interval->end_point == interval->start_point) {
    return loom_low_allocation_interval_storage_end_point(interval);
  }
  return interval->start_point;
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
    if (!loom_low_allocation_storage_reg_classes_share(
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
      value_class.register_descriptor_set_stable_id !=
          state->target.descriptor_set->stable_id ||
      value_class.register_class_id >=
          state->target.descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation value class is not defined by descriptor set '%.*s'",
        (int)state->target.descriptor_set_key.size,
        state->target.descriptor_set_key.data);
  }
  uint16_t reg_class_id = value_class.register_class_id;
  const loom_low_reg_class_t* reg_class =
      &state->target.descriptor_set->reg_classes[reg_class_id];
  if (out_reg_class_id) {
    *out_reg_class_id = reg_class_id;
  }
  if (out_reg_class) {
    *out_reg_class = reg_class;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_reg_class_capacity(
    const loom_low_allocation_build_state_t* state, uint16_t reg_class_id,
    loom_low_allocation_class_capacity_t* out_capacity) {
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      reg_class_id >= state->target.descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation cannot resolve invalid register class %" PRIu16,
        reg_class_id);
  }
  const loom_low_reg_class_t* reg_class = loom_low_allocation_reg_class_at(
      state->target.descriptor_set, reg_class_id);

  uint32_t budget_units = 0;
  const bool has_budget =
      loom_low_allocation_lookup_budget(state, reg_class_id, &budget_units);
  const bool has_allocatable_count = reg_class->allocatable_count != 0;
  uint32_t max_units = UINT32_MAX;
  bool is_bounded = false;
  if (has_budget && has_allocatable_count) {
    max_units = budget_units < reg_class->allocatable_count
                    ? budget_units
                    : reg_class->allocatable_count;
    is_bounded = true;
  } else if (has_budget) {
    max_units = budget_units;
    is_bounded = true;
  } else if (has_allocatable_count) {
    max_units = reg_class->allocatable_count;
    is_bounded = true;
  }

  const bool is_spillable =
      !iree_any_bit_set(reg_class->flags, LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE);
  *out_capacity = (loom_low_allocation_class_capacity_t){
      .descriptor_reg_class_id = reg_class_id,
      .location_kind =
          loom_low_allocation_storage_reg_class_location_kind(reg_class),
      .max_units = max_units,
      .alloc_unit_bits = reg_class->alloc_unit_bits,
      .spill_slot_space =
          (loom_low_spill_slot_space_t)reg_class->spill_slot_space,
      .is_spillable = is_spillable,
      .is_bounded = is_bounded,
  };
  return iree_ok_status();
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
    bool found_reg_class = loom_low_descriptor_set_lookup_register_class(
        state->target.descriptor_set, budget->register_class, &reg_class_id,
        NULL);
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
      if (!loom_low_allocation_storage_reg_classes_share(
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_resolve_descriptor_register_class(
      state, value_class, &reg_class_id, NULL));
  return loom_low_allocation_reg_class_capacity(state, reg_class_id,
                                                out_capacity);
}

static iree_status_t loom_low_allocation_validate_location_range(
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, iree_string_view_t subject) {
  if (!loom_low_allocation_location_kind_is_register_like(location_kind)) {
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

static bool loom_low_allocation_location_range_fits_capacity(
    const loom_low_allocation_class_capacity_t* capacity,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count) {
  if (location_kind != capacity->location_kind || location_count == 0 ||
      (uint64_t)location_base + location_count > UINT32_MAX) {
    return false;
  }
  return !capacity->is_bounded ||
         (uint64_t)location_base + location_count <= capacity->max_units;
}

static iree_status_t loom_low_allocation_emit_capacity_failure(
    const loom_low_allocation_build_state_t* state, const loom_op_t* op,
    uint16_t reg_class_id, iree_string_view_t subject, uint32_t location_base,
    uint32_t location_count, uint64_t location_end, uint32_t max_units) {
  const loom_low_reg_class_t* reg_class = loom_low_allocation_reg_class_at(
      state->target.descriptor_set, reg_class_id);
  const iree_string_view_t register_class = loom_low_descriptor_set_string(
      state->target.descriptor_set, reg_class->name_string_offset);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&state->target)),
      loom_param_string(loom_low_diagnostic_export_name(&state->target)),
      loom_param_string(loom_low_diagnostic_config_key(&state->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(state->module, state->function_op)),
      loom_param_string(subject),
      loom_param_string(register_class),
      loom_param_u32(location_base),
      loom_param_u32(location_count),
      loom_param_u64(location_end),
      loom_param_u32(max_units),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op ? op : state->function_op,
      .error = LOOM_ERR_BACKEND_022,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(state->options->emitter, &emission);
}

static iree_status_t loom_low_allocation_validate_register_location_capacity(
    const loom_low_allocation_build_state_t* state, uint16_t reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, iree_string_view_t subject,
    const loom_op_t* diagnostic_op) {
  IREE_RETURN_IF_ERROR(loom_low_allocation_validate_location_range(
      location_kind, location_base, location_count, subject));

  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_reg_class_capacity(state, reg_class_id, &capacity));
  if (location_kind != capacity.location_kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low allocation %.*s location kind %u does not match register-class "
        "location kind %u",
        (int)subject.size, subject.data, (unsigned)location_kind,
        (unsigned)capacity.location_kind);
  }
  const uint64_t location_end = (uint64_t)location_base + location_count;
  if (capacity.is_bounded && location_end > capacity.max_units) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_emit_capacity_failure(
        state, diagnostic_op, reg_class_id, subject, location_base,
        location_count, location_end, capacity.max_units));
    const loom_low_reg_class_t* reg_class = loom_low_allocation_reg_class_at(
        state->target.descriptor_set, reg_class_id);
    const iree_string_view_t register_class = loom_low_descriptor_set_string(
        state->target.descriptor_set, reg_class->name_string_offset);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low allocation %.*s location range [%" PRIu32 ", %" PRIu64
        ") exceeds register-class '%.*s' capacity %" PRIu32,
        (int)subject.size, subject.data, location_base, location_end,
        (int)register_class.size, register_class.data, capacity.max_units);
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
    uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    bool found_reg_class = loom_low_descriptor_set_lookup_register_class(
        state->target.descriptor_set, reserved_range->register_class,
        &reg_class_id, NULL);
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
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_validate_register_location_capacity(
            state, reg_class_id, reserved_range->location_kind,
            reserved_range->location_base, reserved_range->location_count,
            IREE_SV("reserved range"), state->function_op));
    for (iree_host_size_t j = 0; j < state->resolved_reserved_range_count;
         ++j) {
      const loom_low_allocation_resolved_reserved_range_t* existing =
          &state->resolved_reserved_ranges[j];
      if (reserved_range->location_kind != existing->location_kind ||
          !loom_low_allocation_storage_reg_classes_share(
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
    IREE_RETURN_IF_ERROR(loom_low_allocation_resolve_descriptor_register_class(
        state, interval->value_class, &reg_class_id, NULL));
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_validate_register_location_capacity(
            state, reg_class_id, fixed_value->location_kind,
            fixed_value->location_base, fixed_value->location_count,
            IREE_SV("fixed value"),
            loom_low_diagnostic_value_origin_op(
                state->module, fixed_value->value_id, state->function_op)));
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
  const loom_low_allocation_assignment_t candidate = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = reg_class_id,
      .start_point = interval->start_point,
      .end_point = loom_low_allocation_interval_storage_end_point(interval),
      .unit_count = interval->unit_count,
      .location_kind = location_kind,
      .location_base = location_base,
      .location_count = location_count,
      .unit_end_point_start =
          loom_low_allocation_unit_end_point_start_for_value(
              state, interval->value_id),
  };
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
    if (!loom_low_allocation_storage_reg_classes_share(
            state->target.descriptor_set, fixed_value->descriptor_reg_class_id,
            reg_class_id)) {
      continue;
    }
    const loom_low_allocation_assignment_t fixed_assignment = {
        .value_id = fixed_value->value_id,
        .value_class = fixed_value->interval->value_class,
        .descriptor_reg_class_id = fixed_value->descriptor_reg_class_id,
        .start_point = fixed_value->interval->start_point,
        .end_point = loom_low_allocation_interval_storage_end_point(
            fixed_value->interval),
        .unit_count = fixed_value->interval->unit_count,
        .location_kind = fixed_value->location_kind,
        .location_base = fixed_value->location_base,
        .location_count = fixed_value->location_count,
        .unit_end_point_start =
            loom_low_allocation_unit_end_point_start_for_value(
                state, fixed_value->value_id),
    };
    if (loom_low_allocation_live_range_assignments_conflict(
            state->target.descriptor_set, &state->liveness,
            state->unit_end_points, state->unit_end_point_count,
            &fixed_assignment, &candidate)) {
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
    if (!loom_low_allocation_storage_reg_classes_share(
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

static uint32_t loom_low_allocation_unit_location_hash(
    loom_low_allocation_location_kind_t location_kind, uint32_t storage_key,
    uint32_t location) {
  uint32_t hash = location ^ ((uint32_t)location_kind * 0x9E3779B9u);
  hash ^= storage_key * 0x7F4A7C15u;
  hash ^= hash >> 16;
  hash *= 0x85EBCA6Bu;
  hash ^= hash >> 13;
  hash *= 0xC2B2AE35u;
  hash ^= hash >> 16;
  return hash;
}

static bool loom_low_allocation_active_assignment_conflicts(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_assignment_t* existing,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count) {
  if (loom_low_allocation_value_id_is_ignored(
          existing->value_id, ignored_value_ids, ignored_value_count)) {
    return false;
  }
  if (existing->location_kind != candidate->location_kind) {
    return false;
  }
  return loom_low_allocation_live_range_assignments_conflict(
      state->target.descriptor_set, &state->liveness, state->unit_end_points,
      state->unit_end_point_count, existing, candidate);
}

static bool loom_low_allocation_active_scan_conflicts(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_assignment_t* candidate,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    bool unindexed_only) {
  for (iree_host_size_t i = 0; i < state->active.count; ++i) {
    const uint32_t assignment_index =
        state->active.assignment_indices[state->active.start + i];
    IREE_ASSERT_LT(assignment_index, state->assignment_count);
    const loom_low_allocation_assignment_t* existing =
        &state->assignments[assignment_index];
    const bool assignment_is_indexed =
        loom_low_allocation_active_unit_index_contains_assignment(
            &state->active_units, assignment_index);
    if (unindexed_only && assignment_is_indexed) {
      continue;
    }
    if (loom_low_allocation_active_assignment_conflicts(
            state, existing, candidate, ignored_value_ids,
            ignored_value_count)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_storage_lease_unit_index_is_enabled(
    const loom_low_allocation_build_state_t* state) {
  return state->storage_lease_units.bucket_heads != NULL &&
         state->storage_lease_units.bucket_count != 0 &&
         state->storage_lease_units.entries != NULL;
}

static uint32_t loom_low_allocation_storage_lease_unit_bucket_index(
    const loom_low_allocation_build_state_t* state,
    loom_low_allocation_location_kind_t location_kind, uint32_t storage_key,
    uint32_t location) {
  return loom_low_allocation_unit_location_hash(location_kind, storage_key,
                                                location) &
         (state->storage_lease_units.bucket_count - 1u);
}

static bool loom_low_allocation_storage_lease_instance_conflicts(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_storage_lease_t* lease,
    const loom_low_allocation_assignment_t* candidate) {
  if (lease->release_action_index !=
      LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE) {
    return false;
  }
  if (lease->end_point <= candidate->start_point ||
      lease->start_point >= candidate->end_point) {
    return false;
  }
  if (lease->location_kind != candidate->location_kind) {
    return false;
  }
  if (!loom_low_allocation_storage_reg_classes_share(
          state->target.descriptor_set, lease->descriptor_reg_class_id,
          candidate->descriptor_reg_class_id)) {
    return false;
  }
  return loom_low_allocation_location_range_overlaps(
      lease->location_base, lease->location_count, candidate->location_base,
      candidate->location_count);
}

static bool loom_low_allocation_try_packet_at_program_point(
    const loom_low_allocation_build_state_t* state, uint32_t program_point,
    iree_host_size_t* out_packet_index, uint32_t* out_node_index,
    uint32_t* out_block_index, uint32_t* out_scheduled_ordinal) {
  *out_packet_index = LOOM_LOW_STORAGE_LEASE_PACKET_NONE;
  *out_node_index = LOOM_LOW_STORAGE_LEASE_NODE_NONE;
  *out_block_index = UINT32_MAX;
  *out_scheduled_ordinal = LOOM_LOW_STORAGE_LEASE_ORDINAL_NONE;

  const loom_low_schedule_table_t* schedule =
      state->options->storage_leases.schedule;
  for (iree_host_size_t i = 0; i < state->liveness.block_count; ++i) {
    const loom_liveness_block_info_t* block_info = &state->liveness.blocks[i];
    if (program_point < block_info->start_point ||
        program_point >= block_info->end_point) {
      continue;
    }
    const uint32_t scheduled_ordinal = program_point - block_info->start_point;
    if (scheduled_ordinal >= schedule->blocks[i].scheduled_node_count) {
      return false;
    }
    const uint64_t packet_index =
        (uint64_t)schedule->blocks[i].scheduled_node_start + scheduled_ordinal;
    if (packet_index >= schedule->scheduled_node_count ||
        packet_index > IREE_HOST_SIZE_MAX) {
      return false;
    }
    const uint32_t node_index =
        schedule->scheduled_node_indices[(iree_host_size_t)packet_index];
    if (node_index >= schedule->node_count) {
      return false;
    }
    *out_packet_index = (iree_host_size_t)packet_index;
    *out_node_index = node_index;
    *out_block_index = (uint32_t)i;
    *out_scheduled_ordinal = scheduled_ordinal;
    return true;
  }
  return false;
}

static iree_status_t loom_low_allocation_packet_at_program_point(
    const loom_low_allocation_build_state_t* state, uint32_t program_point,
    iree_host_size_t* out_packet_index, uint32_t* out_node_index,
    uint32_t* out_block_index, uint32_t* out_scheduled_ordinal) {
  if (loom_low_allocation_try_packet_at_program_point(
          state, program_point, out_packet_index, out_node_index,
          out_block_index, out_scheduled_ordinal)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "storage release insertion point does not map to a "
                          "scheduled packet");
}

static bool loom_low_allocation_storage_lease_can_release_before(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_storage_lease_t* lease,
    const loom_low_allocation_assignment_t* candidate) {
  const loom_low_storage_lease_record_t* record =
      &state->options->storage_leases.records[lease->lease_record_index];
  if (record->release_scope !=
      LOOM_LOW_STORAGE_LEASE_RELEASE_SCOPE_PROGRESS_CLASS) {
    return false;
  }
  if (candidate->start_point <= lease->start_point) {
    return false;
  }
  iree_host_size_t packet_index = LOOM_LOW_STORAGE_LEASE_PACKET_NONE;
  uint32_t node_index = LOOM_LOW_STORAGE_LEASE_NODE_NONE;
  uint32_t block_index = UINT32_MAX;
  uint32_t scheduled_ordinal = LOOM_LOW_STORAGE_LEASE_ORDINAL_NONE;
  if (!loom_low_allocation_try_packet_at_program_point(
          state, candidate->start_point, &packet_index, &node_index,
          &block_index, &scheduled_ordinal)) {
    return false;
  }
  return packet_index != LOOM_LOW_STORAGE_LEASE_PACKET_NONE &&
         packet_index > record->packet_index;
}

static bool loom_low_allocation_storage_lease_scan_conflicts(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_assignment_t* candidate) {
  const iree_host_size_t record_count =
      state->options->storage_leases.record_count;
  for (iree_host_size_t i = 0; i < record_count; ++i) {
    if (state->storage_lease_instance_written[i] == 0) {
      continue;
    }
    if (loom_low_allocation_storage_lease_instance_conflicts(
            state, &state->storage_lease_instances[i], candidate)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_storage_lease_unit_index_conflicts(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_assignment_t* candidate,
    loom_low_allocation_storage_release_policy_t release_policy) {
  if (release_policy == LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED) {
    const iree_host_size_t record_count =
        state->options->storage_leases.record_count;
    for (iree_host_size_t i = 0; i < record_count; ++i) {
      if (state->storage_lease_instance_written[i] == 0) {
        continue;
      }
      const loom_low_allocation_storage_lease_t* lease =
          &state->storage_lease_instances[i];
      if (!loom_low_allocation_storage_lease_instance_conflicts(state, lease,
                                                                candidate)) {
        continue;
      }
      if (!loom_low_allocation_storage_lease_can_release_before(state, lease,
                                                                candidate)) {
        return true;
      }
    }
    return false;
  }
  if (!loom_low_allocation_location_kind_is_register_like(
          candidate->location_kind)) {
    return false;
  }
  if (!loom_low_allocation_storage_lease_unit_index_is_enabled(state)) {
    return loom_low_allocation_storage_lease_scan_conflicts(state, candidate);
  }
  const uint32_t storage_key = loom_low_reg_class_storage_key(
      state->target.descriptor_set, candidate->descriptor_reg_class_id);
  for (uint32_t unit_offset = 0; unit_offset < candidate->location_count;
       ++unit_offset) {
    if (candidate->location_base > UINT32_MAX - unit_offset) {
      break;
    }
    const uint32_t location = candidate->location_base + unit_offset;
    const uint32_t bucket_index =
        loom_low_allocation_storage_lease_unit_bucket_index(
            state, candidate->location_kind, storage_key, location);
    uint32_t entry_index =
        state->storage_lease_units.bucket_heads[bucket_index];
    while (entry_index != UINT32_MAX) {
      const loom_low_allocation_storage_lease_unit_entry_t* entry =
          &state->storage_lease_units.entries[entry_index];
      const loom_low_allocation_storage_lease_t* lease =
          &state->storage_lease_instances[entry->storage_lease_index];
      if (entry->location_kind == candidate->location_kind &&
          entry->storage_key == storage_key && entry->location == location &&
          lease->release_action_index ==
              LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE &&
          lease->end_point > candidate->start_point &&
          lease->start_point < candidate->end_point) {
        return true;
      }
      entry_index = entry->next_entry;
    }
  }
  return false;
}

static bool loom_low_allocation_candidate_conflicts(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval, uint16_t reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, const loom_value_id_t* ignored_value_ids,
    uint16_t ignored_value_count,
    loom_low_allocation_storage_release_policy_t release_policy) {
  loom_low_allocation_assignment_t candidate = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = reg_class_id,
      .start_point = interval->start_point,
      .end_point = loom_low_allocation_interval_storage_end_point(interval),
      .unit_count = interval->unit_count,
      .location_kind = location_kind,
      .location_base = location_base,
      .location_count = location_count,
      .unit_end_point_start =
          loom_low_allocation_unit_end_point_start_for_value(
              state, interval->value_id),
  };
  if (loom_low_allocation_active_unit_index_is_enabled(&state->active_units)) {
    if (loom_low_allocation_active_unit_index_conflicts(
            &state->active_units, state->target.descriptor_set,
            &state->liveness, state->unit_end_points,
            state->unit_end_point_count, state->assignments,
            state->assignment_count, &candidate, ignored_value_ids,
            ignored_value_count)) {
      return true;
    }
    if (loom_low_allocation_active_unit_index_unindexed_count(
            &state->active_units) != 0 &&
        loom_low_allocation_active_scan_conflicts(
            state, &candidate, ignored_value_ids, ignored_value_count,
            /*unindexed_only=*/true)) {
      return true;
    }
  } else if (loom_low_allocation_active_scan_conflicts(
                 state, &candidate, ignored_value_ids, ignored_value_count,
                 /*unindexed_only=*/false)) {
    return true;
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
  if (loom_low_allocation_storage_lease_unit_index_conflicts(state, &candidate,
                                                             release_policy)) {
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
      loom_low_allocation_storage_reg_class_location_kind(reg_class)) {
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
    if (!loom_low_allocation_storage_reg_classes_share(
            state->target.descriptor_set, i, reg_class_id)) {
      continue;
    }
    const loom_low_reg_class_t* reg_class =
        loom_low_allocation_reg_class_at(state->target.descriptor_set, i);
    if (loom_low_allocation_storage_reg_class_location_kind(reg_class) !=
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
    if (!loom_low_allocation_storage_reg_classes_share(
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
    if (!loom_low_allocation_storage_reg_classes_share(
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

static bool loom_low_allocation_find_location(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_low_allocation_class_capacity_t capacity, uint32_t* out_base) {
  if (capacity.is_bounded && interval->unit_count > capacity.max_units) {
    return false;
  }

  const uint32_t alignment = loom_low_allocation_interval_alignment(interval);
  uint32_t last_base = 0;
  if (capacity.is_bounded) {
    last_base = capacity.max_units - interval->unit_count;
  } else {
    const uint32_t search_limit =
        loom_low_allocation_assigned_location_search_limit(
            state, capacity.descriptor_reg_class_id, capacity.location_kind);
    if (!loom_low_allocation_align_up_u32(search_limit, alignment,
                                          &last_base)) {
      return false;
    }
  }

  for (uint32_t base = 0; base <= last_base;) {
    if (!loom_low_allocation_candidate_conflicts(
            state, interval, capacity.descriptor_reg_class_id,
            capacity.location_kind, base, interval->unit_count,
            /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
      *out_base = base;
      return true;
    }
    if (base > UINT32_MAX - alignment) {
      break;
    }
    base += alignment;
  }
  for (uint32_t base = 0; base <= last_base;) {
    if (!loom_low_allocation_candidate_conflicts(
            state, interval, capacity.descriptor_reg_class_id,
            capacity.location_kind, base, interval->unit_count,
            /*ignored_value_ids=*/NULL, /*ignored_value_count=*/0,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_ALLOWED)) {
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

static bool loom_low_allocation_use_is_removed_block_arg_edge(
    loom_use_t use, const loom_block_t* block, uint16_t arg_index) {
  const loom_op_t* user_op = loom_use_user_op(use);
  return user_op && loom_low_br_isa(user_op) &&
         loom_low_br_dest(user_op) == block &&
         loom_use_operand_index(use) == arg_index;
}

static uint32_t loom_low_allocation_spill_plan_reload_count(
    const loom_value_t* value) {
  if (!loom_value_is_block_arg(value)) {
    return value->use_count;
  }

  const loom_block_t* block = loom_value_def_block(value);
  const uint16_t arg_index = loom_value_def_index(value);
  uint32_t reload_count = 0;
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    if (!loom_low_allocation_use_is_removed_block_arg_edge(uses[i], block,
                                                           arg_index)) {
      ++reload_count;
    }
  }
  return reload_count;
}

static iree_status_t loom_low_allocation_spill_plan_store_count(
    const loom_low_allocation_build_state_t* state, loom_value_id_t value_id,
    const loom_value_t* value, uint32_t reload_count,
    uint32_t* out_store_count) {
  *out_store_count = 0;
  if (reload_count == 0) {
    return iree_ok_status();
  }
  if (!loom_value_is_block_arg(value)) {
    *out_store_count = 1;
    return iree_ok_status();
  }

  const loom_block_t* block = loom_value_def_block(value);
  const uint16_t arg_index = loom_value_def_index(value);
  if (block == loom_region_entry_block(state->body)) {
    *out_store_count = 1;
    return iree_ok_status();
  }

  uint32_t store_count = 0;
  loom_block_t* predecessor_block = NULL;
  loom_region_for_each_block(state->body, predecessor_block) {
    const loom_op_t* terminator = loom_block_const_last_op(predecessor_block);
    if (!terminator || !loom_low_br_isa(terminator) ||
        loom_low_br_dest(terminator) != block) {
      continue;
    }
    if (arg_index >= terminator->operand_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low.br predecessor payload count is stale for spilled block "
          "argument");
    }
    const loom_value_id_t payload =
        loom_op_const_operands(terminator)[arg_index];
    if (payload == LOOM_VALUE_ID_INVALID ||
        payload >= state->module->values.count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low.br predecessor payload is invalid for spilled block argument");
    }
    if (payload == value_id) {
      continue;
    }
    if (store_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "spill store count overflow");
    }
    ++store_count;
  }
  if (store_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "spilled non-entry block argument has reloads but no incoming value "
        "to store");
  }
  *out_store_count = store_count;
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
  const loom_value_t* value =
      loom_module_value(state->module, assignment->value_id);
  uint32_t reload_count = loom_low_allocation_spill_plan_reload_count(value);
  uint32_t store_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_store_count(
      state, assignment->value_id, value, reload_count, &store_count));
  state->spill_plans[state->spill_plan_count++] =
      (loom_low_allocation_spill_plan_t){
          .value_id = assignment->value_id,
          .assignment_index = assignment_index,
          .slot_index = assignment->location_base,
          .slot_space = spill_slot_space,
          .byte_size = byte_size,
          .byte_alignment = byte_alignment,
          .store_count = store_count,
          .reload_count = reload_count,
      };
  return iree_ok_status();
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

static bool loom_low_allocation_value_requires_register_location(
    const loom_low_allocation_build_state_t* state, loom_value_id_t value_id) {
  if (value_id >= state->module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(state->module, value_id);
  const loom_op_t* defining_op = loom_def_op(value->def);
  if (defining_op && loom_low_reload_isa(defining_op)) {
    return true;
  }
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    const loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (user_op && loom_low_spill_isa(user_op) &&
        loom_use_operand_index(uses[i]) == 0) {
      return true;
    }
  }
  return false;
}

static bool loom_low_allocation_interval_requires_register_location(
    const loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval) {
  return loom_low_allocation_value_requires_register_location(
      state, interval->value_id);
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

static iree_status_t loom_low_allocation_consumption_query(
    loom_low_allocation_build_state_t* state,
    loom_consumption_region_query_t** out_query) {
  *out_query = NULL;
  if (!state->consumption_query_initialized) {
    IREE_RETURN_IF_ERROR(loom_consumption_region_query_initialize(
        state->module, state->body, state->arena, &state->consumption_query));
    state->consumption_query_initialized = true;
  }
  *out_query = &state->consumption_query;
  return iree_ok_status();
}

static bool loom_low_allocation_copy_source_for_value(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_id_t* out_source_id) {
  *out_source_id = LOOM_VALUE_ID_INVALID;
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!loom_low_copy_isa(defining_op)) {
    return false;
  }
  *out_source_id = loom_low_copy_source(defining_op);
  return true;
}

static iree_status_t loom_low_allocation_copy_source_used_after_tied_consume(
    loom_low_allocation_build_state_t* state,
    const loom_low_placement_relation_t* tied_relation,
    loom_value_id_t tied_operand_id, loom_value_id_t* out_copy_source_id,
    bool* out_used_after) {
  *out_copy_source_id = LOOM_VALUE_ID_INVALID;
  *out_used_after = false;
  if (!loom_low_allocation_copy_source_for_value(state->module, tied_operand_id,
                                                 out_copy_source_id)) {
    return iree_ok_status();
  }

  loom_consumption_region_query_t* query = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_consumption_query(state, &query));
  loom_consumption_use_t use = {0};
  return loom_consumption_find_use_after(
      query, tied_relation->op, *out_copy_source_id, &use, out_used_after);
}

static iree_status_t
loom_low_allocation_copy_relation_requires_materialized_storage(
    loom_low_allocation_build_state_t* state,
    const loom_low_placement_relation_t* copy_relation,
    bool* out_requires_materialized_storage) {
  *out_requires_materialized_storage = false;
  const loom_value_id_t copy_result_id = loom_low_placement_value_id(
      &state->placement, copy_relation->result_ordinal);
  const loom_low_placement_relation_range_t tied_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          &state->placement, copy_relation->result_ordinal);
  for (uint32_t i = 0; i < tied_range.count; ++i) {
    const uint32_t relation_index =
        state->placement
            .relation_indices_by_source_ordinal[tied_range.start + i];
    const loom_low_placement_relation_t* tied_relation =
        &state->placement.relations[relation_index];
    if (tied_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
      continue;
    }
    loom_value_id_t copy_source_id = LOOM_VALUE_ID_INVALID;
    bool used_after = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_copy_source_used_after_tied_consume(
            state, tied_relation, copy_result_id, &copy_source_id,
            &used_after));
    if (used_after) {
      *out_requires_materialized_storage = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_initialize_storage_lease_unit_index(
    loom_low_allocation_build_state_t* state,
    iree_host_size_t lease_unit_capacity) {
  if (lease_unit_capacity == 0) {
    return iree_ok_status();
  }
  if (lease_unit_capacity > UINT32_MAX / 2u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation storage lease unit count exceeds "
                            "index range");
  }

  const uint32_t bucket_count =
      loom_low_allocation_round_up_to_power_of_two_u32(
          (uint32_t)lease_unit_capacity * 2u);
  if (bucket_count == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation storage lease bucket count exceeds "
                            "u32 range");
  }

  state->storage_lease_units.bucket_count = bucket_count;
  state->storage_lease_units.entry_capacity = lease_unit_capacity;

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, bucket_count,
      sizeof(*state->storage_lease_units.bucket_heads),
      (void**)&state->storage_lease_units.bucket_heads));
  for (uint32_t i = 0; i < bucket_count; ++i) {
    state->storage_lease_units.bucket_heads[i] = UINT32_MAX;
  }

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, lease_unit_capacity,
                                sizeof(*state->storage_lease_units.entries),
                                (void**)&state->storage_lease_units.entries));
  memset(state->storage_lease_units.entries, 0,
         lease_unit_capacity * sizeof(*state->storage_lease_units.entries));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_storage_lease_unit_index_insert(
    loom_low_allocation_build_state_t* state, uint32_t storage_lease_index) {
  if (!loom_low_allocation_storage_lease_unit_index_is_enabled(state)) {
    return iree_ok_status();
  }
  const loom_low_allocation_storage_lease_t* lease =
      &state->storage_lease_instances[storage_lease_index];
  if (!loom_low_allocation_location_kind_is_register_like(
          lease->location_kind)) {
    return iree_ok_status();
  }
  if (lease->location_count > state->storage_lease_units.entry_capacity -
                                  state->storage_lease_units.entry_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation storage lease unit index capacity was exhausted");
  }

  const uint32_t storage_key = loom_low_reg_class_storage_key(
      state->target.descriptor_set, lease->descriptor_reg_class_id);
  for (uint32_t unit_offset = 0; unit_offset < lease->location_count;
       ++unit_offset) {
    if (lease->location_base > UINT32_MAX - unit_offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "allocation storage lease unit location exceeds "
                              "u32 range");
    }
    const uint32_t location = lease->location_base + unit_offset;
    const uint32_t bucket_index =
        loom_low_allocation_storage_lease_unit_bucket_index(
            state, lease->location_kind, storage_key, location);
    const uint32_t entry_index =
        (uint32_t)state->storage_lease_units.entry_count++;
    loom_low_allocation_storage_lease_unit_entry_t* entry =
        &state->storage_lease_units.entries[entry_index];
    *entry = (loom_low_allocation_storage_lease_unit_entry_t){
        .storage_lease_index = storage_lease_index,
        .next_entry = state->storage_lease_units.bucket_heads[bucket_index],
        .storage_key = storage_key,
        .location_kind = lease->location_kind,
        .location = location,
    };
    state->storage_lease_units.bucket_heads[bucket_index] = entry_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_validate_storage_lease_table(
    const loom_low_allocation_build_state_t* state) {
  const loom_low_storage_lease_table_t* lease_table =
      &state->options->storage_leases;
  if (lease_table->record_count == 0) {
    return iree_ok_status();
  }
  if (lease_table->schedule == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation storage leases have records but no schedule");
  }
  if (lease_table->records == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation storage leases have a record count but no records");
  }
  if (lease_table->schedule->module != state->module ||
      lease_table->schedule->function_op != state->function_op) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation storage leases must describe the allocated low function");
  }
  if (lease_table->record_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation storage lease count exceeds u32 "
                            "range");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_storage_lease_value_id(
    const loom_low_storage_lease_table_t* lease_table,
    const loom_low_storage_lease_record_t* record,
    loom_value_id_t* out_value_id) {
  const loom_low_schedule_table_t* schedule = lease_table->schedule;
  if (record->node_index >= schedule->node_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease references schedule node %" PRIu32
                            " outside the schedule",
                            record->node_index);
  }
  const loom_low_schedule_node_t* node = &schedule->nodes[record->node_index];
  if (node->block_index != record->block_index ||
      node->scheduled_ordinal != record->scheduled_ordinal) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "storage lease record no longer matches its schedule node");
  }

  const loom_value_ordinal_t* value_ordinals = NULL;
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (record->attachment == LOOM_LOW_STORAGE_LEASE_ATTACHMENT_OPERAND) {
    if (record->attachment_index >= node->operand_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "storage lease operand index exceeds schedule "
                              "node operand count");
    }
    value_ordinals = loom_low_schedule_node_const_operand_ordinals(node);
    value_ordinal = value_ordinals[record->attachment_index];
  } else if (record->attachment == LOOM_LOW_STORAGE_LEASE_ATTACHMENT_RESULT) {
    if (record->attachment_index >= node->result_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "storage lease result index exceeds schedule "
                              "node result count");
    }
    value_ordinals = loom_low_schedule_node_const_result_ordinals(node);
    value_ordinal = value_ordinals[record->attachment_index];
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "storage lease record has invalid attachment kind %u",
        (unsigned)record->attachment);
  }
  if (value_ordinal >= schedule->liveness.value_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "storage lease references a value outside schedule liveness");
  }
  *out_value_id = schedule->liveness.value_ids[value_ordinal];
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_storage_lease_start_point(
    const loom_liveness_analysis_t* liveness,
    const loom_low_storage_lease_record_t* record, uint32_t* out_start_point) {
  if (record->block_index >= liveness->block_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease block index exceeds allocation "
                            "liveness block count");
  }
  const loom_liveness_block_info_t* block_info =
      &liveness->blocks[record->block_index];
  if (record->scheduled_ordinal > UINT32_MAX - block_info->start_point) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease start point exceeds u32 range");
  }
  const uint32_t start_point =
      block_info->start_point + record->scheduled_ordinal;
  if (start_point >= block_info->end_point) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease scheduled ordinal exceeds "
                            "allocation liveness block extent");
  }
  *out_start_point = start_point;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_storage_lease_end_point(
    const loom_liveness_analysis_t* liveness,
    const loom_low_storage_lease_record_t* record, uint32_t* out_end_point) {
  if (record->block_index >= liveness->block_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease block index exceeds allocation "
                            "liveness block count");
  }
  if (iree_any_bit_set(record->flags,
                       LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_BEFORE_BOUNDARY)) {
    *out_end_point = liveness->blocks[record->block_index].end_point;
    return iree_ok_status();
  }
  uint32_t function_end_point = 0;
  for (iree_host_size_t i = 0; i < liveness->block_count; ++i) {
    if (function_end_point < liveness->blocks[i].end_point) {
      function_end_point = liveness->blocks[i].end_point;
    }
  }
  *out_end_point = function_end_point;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_record_storage_lease_instance(
    loom_low_allocation_build_state_t* state, uint32_t lease_record_index,
    uint32_t assignment_index, const loom_low_storage_lease_record_t* record) {
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_allocation_storage_lease_value_id(
      &state->options->storage_leases, record, &value_id));

  const loom_low_allocation_assignment_t* assignment =
      &state->assignments[assignment_index];
  if (assignment->value_id != value_id) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "storage lease record does not match assigned "
                            "value");
  }
  if (!loom_low_allocation_location_kind_is_register_like(
          assignment->location_kind)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "storage lease references value %u assigned to "
                            "non-register storage",
                            (unsigned)value_id);
  }
  uint32_t lease_assignment_end = 0;
  if (record->unit_offset > UINT32_MAX - record->unit_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease unit range exceeds u32 range");
  }
  lease_assignment_end = record->unit_offset + record->unit_count;
  if (lease_assignment_end > assignment->location_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease unit range exceeds assigned "
                            "physical storage");
  }
  uint32_t location_base = 0;
  if (assignment->location_base > UINT32_MAX - record->unit_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "storage lease physical location exceeds u32 "
                            "range");
  }
  location_base = assignment->location_base + record->unit_offset;
  if (!iree_any_bit_set(record->flags,
                        LOOM_LOW_STORAGE_LEASE_FLAG_STARTS_AT_ISSUE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "storage lease record has no supported start "
                            "condition");
  }
  uint32_t start_point = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_storage_lease_start_point(
      &state->liveness, record, &start_point));
  uint32_t end_point = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_storage_lease_end_point(
      &state->liveness, record, &end_point));
  if (end_point <= start_point) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "storage lease end point must be after its start "
                            "point");
  }

  if (state->storage_lease_instance_written[lease_record_index] != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "storage lease record was materialized twice");
  }
  state->storage_lease_instances[lease_record_index] =
      (loom_low_allocation_storage_lease_t){
          .lease_record_index = lease_record_index,
          .assignment_index = assignment_index,
          .value_id = value_id,
          .start_point = start_point,
          .end_point = end_point,
          .release_action_index = LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE,
          .descriptor_reg_class_id = assignment->descriptor_reg_class_id,
          .location_kind = assignment->location_kind,
          .location_base = location_base,
          .location_count = record->unit_count,
      };
  state->storage_lease_instance_written[lease_record_index] = 1;
  ++state->storage_lease_instance_count;
  IREE_RETURN_IF_ERROR(loom_low_allocation_storage_lease_unit_index_insert(
      state, lease_record_index));
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_record_storage_lease_instances_for_assignment(
    loom_low_allocation_build_state_t* state, uint32_t assignment_index,
    loom_value_ordinal_t value_ordinal) {
  if (state->storage_lease_record_heads_by_value_ordinal == NULL) {
    return iree_ok_status();
  }
  uint32_t lease_record_index =
      state->storage_lease_record_heads_by_value_ordinal[value_ordinal];
  while (lease_record_index != UINT32_MAX) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_record_storage_lease_instance(
        state, lease_record_index, assignment_index,
        &state->options->storage_leases.records[lease_record_index]));
    lease_record_index =
        state->storage_lease_next_record_indices[lease_record_index];
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_append_storage_release_action(
    loom_low_allocation_build_state_t* state, uint32_t lease_record_index,
    const loom_low_allocation_assignment_t* candidate) {
  const loom_low_storage_lease_record_t* record =
      &state->options->storage_leases.records[lease_record_index];
  loom_low_allocation_storage_lease_t* lease =
      &state->storage_lease_instances[lease_record_index];
  if (state->storage_release_action_count >=
      state->options->storage_leases.record_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation storage release action count exceeded lease count");
  }
  if (!loom_low_allocation_storage_lease_can_release_before(state, lease,
                                                            candidate)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "storage lease conflict has no legal release before assignment");
  }

  iree_host_size_t insertion_packet_index = LOOM_LOW_STORAGE_LEASE_PACKET_NONE;
  uint32_t insertion_node_index = LOOM_LOW_STORAGE_LEASE_NODE_NONE;
  uint32_t block_index = UINT32_MAX;
  uint32_t scheduled_ordinal = LOOM_LOW_STORAGE_LEASE_ORDINAL_NONE;
  IREE_RETURN_IF_ERROR(loom_low_allocation_packet_at_program_point(
      state, candidate->start_point, &insertion_packet_index,
      &insertion_node_index, &block_index, &scheduled_ordinal));

  const uint32_t release_action_index =
      (uint32_t)state->storage_release_action_count++;
  state->storage_release_actions[release_action_index] =
      (loom_low_storage_release_action_t){
          .insertion_packet_index = insertion_packet_index,
          .insertion_node_index = insertion_node_index,
          .block_index = block_index,
          .scheduled_ordinal = scheduled_ordinal,
          .release_class_id = record->release_class_id,
          .release_class_name = record->release_class_name,
          .release_action_id = record->release_action_id,
          .release_action_name = record->release_action_name,
          .release_reason_id = record->release_reason_id,
          .release_reason_name = record->release_reason_name,
          .required_progress = 1,
          .lease_record_index = lease_record_index,
      };
  lease->release_action_index = release_action_index;
  lease->end_point = candidate->start_point;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_record_storage_release_actions(
    loom_low_allocation_build_state_t* state,
    const loom_low_allocation_assignment_t* candidate) {
  if (state->storage_lease_instances == NULL) {
    return iree_ok_status();
  }
  const iree_host_size_t record_count =
      state->options->storage_leases.record_count;
  for (iree_host_size_t i = 0; i < record_count; ++i) {
    if (state->storage_lease_instance_written[i] == 0) {
      continue;
    }
    loom_low_allocation_storage_lease_t* lease =
        &state->storage_lease_instances[i];
    if (!loom_low_allocation_storage_lease_instance_conflicts(state, lease,
                                                              candidate)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_append_storage_release_action(
        state, (uint32_t)i, candidate));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_prepare_storage_lease_instances(
    loom_low_allocation_build_state_t* state) {
  IREE_RETURN_IF_ERROR(loom_low_allocation_validate_storage_lease_table(state));
  const loom_low_storage_lease_table_t* lease_table =
      &state->options->storage_leases;
  if (lease_table->record_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, lease_table->record_count,
                                sizeof(*state->storage_lease_instances),
                                (void**)&state->storage_lease_instances));
  memset(state->storage_lease_instances, 0,
         lease_table->record_count * sizeof(*state->storage_lease_instances));
  for (iree_host_size_t i = 0; i < lease_table->record_count; ++i) {
    state->storage_lease_instances[i].release_action_index =
        LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE;
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, lease_table->record_count,
                                sizeof(*state->storage_release_actions),
                                (void**)&state->storage_release_actions));
  memset(state->storage_release_actions, 0,
         lease_table->record_count * sizeof(*state->storage_release_actions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, lease_table->record_count,
      sizeof(*state->storage_lease_instance_written),
      (void**)&state->storage_lease_instance_written));
  memset(state->storage_lease_instance_written, 0,
         lease_table->record_count *
             sizeof(*state->storage_lease_instance_written));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, lease_table->record_count,
      sizeof(*state->storage_lease_next_record_indices),
      (void**)&state->storage_lease_next_record_indices));
  for (iree_host_size_t i = 0; i < lease_table->record_count; ++i) {
    state->storage_lease_next_record_indices[i] = UINT32_MAX;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->liveness.value_count,
      sizeof(*state->storage_lease_record_heads_by_value_ordinal),
      (void**)&state->storage_lease_record_heads_by_value_ordinal));
  for (iree_host_size_t i = 0; i < state->liveness.value_count; ++i) {
    state->storage_lease_record_heads_by_value_ordinal[i] = UINT32_MAX;
  }

  iree_host_size_t lease_unit_capacity = 0;
  for (iree_host_size_t i = 0; i < lease_table->record_count; ++i) {
    const loom_low_storage_lease_record_t* record = &lease_table->records[i];
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_allocation_storage_lease_value_id(
        lease_table, record, &value_id));
    loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    if (!loom_low_allocation_value_ordinal_for_value(state, value_id,
                                                     &value_ordinal)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "storage lease references value outside allocation liveness");
    }
    state->storage_lease_next_record_indices[i] =
        state->storage_lease_record_heads_by_value_ordinal[value_ordinal];
    state->storage_lease_record_heads_by_value_ordinal[value_ordinal] =
        (uint32_t)i;
    if (!iree_host_size_checked_add(lease_unit_capacity, record->unit_count,
                                    &lease_unit_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "storage lease unit count exceeds host size");
    }
  }
  IREE_RETURN_IF_ERROR(loom_low_allocation_initialize_storage_lease_unit_index(
      state, lease_unit_capacity));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_finalize_storage_lease_instances(
    const loom_low_allocation_build_state_t* state) {
  const loom_low_storage_lease_table_t* lease_table =
      &state->options->storage_leases;
  if (lease_table->record_count == 0) {
    return iree_ok_status();
  }
  if (state->storage_lease_instance_count != lease_table->record_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation materialized %" PRIhsz
        " storage lease instance(s) for %" PRIhsz " storage lease record(s)",
        state->storage_lease_instance_count, lease_table->record_count);
  }
  for (iree_host_size_t i = 0; i < lease_table->record_count; ++i) {
    if (state->storage_lease_instance_written[i] == 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "storage lease record %" PRIhsz
                              " has no assigned physical storage",
                              i);
    }
  }
  return iree_ok_status();
}

static bool loom_low_allocation_assignment_can_spill(
    loom_low_allocation_build_state_t* state,
    const loom_low_allocation_assignment_t* assignment,
    loom_low_allocation_class_capacity_t* out_capacity) {
  if (!loom_low_allocation_assignment_is_register_like(assignment)) {
    return false;
  }
  if (loom_low_allocation_fixed_value_for_value(state, assignment->value_id)) {
    return false;
  }
  if (loom_low_allocation_value_requires_register_location(
          state, assignment->value_id)) {
    return false;
  }
  loom_low_allocation_class_capacity_t capacity = {0};
  if (!iree_status_is_ok(loom_low_allocation_reg_class_capacity(
          state, assignment->descriptor_reg_class_id, &capacity))) {
    return false;
  }
  if (!capacity.is_spillable) {
    return false;
  }
  if (out_capacity) {
    *out_capacity = capacity;
  }
  return true;
}

static iree_status_t loom_low_allocation_spill_active_assignment(
    loom_low_allocation_build_state_t* state, uint32_t assignment_index,
    const loom_low_allocation_class_capacity_t* capacity) {
  if (state->spill_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation table exceeds uint32_t range");
  }
  loom_low_allocation_active_unit_index_remove_assignment(
      &state->active_units, state->assignments, state->assignment_count,
      assignment_index);
  loom_low_allocation_assignment_t* assignment =
      &state->assignments[assignment_index];
  assignment->location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT;
  assignment->location_base = (uint32_t)state->spill_count;
  assignment->location_count = assignment->unit_count;
  ++state->spill_count;
  IREE_RETURN_IF_ERROR(loom_low_allocation_record_spill_plan(
      state, assignment_index, capacity->alloc_unit_bits,
      capacity->spill_slot_space));
  loom_low_allocation_record_spill_remark(
      state, assignment_index,
      capacity->is_bounded ? capacity->max_units : UINT32_MAX,
      assignment->unit_count);
  return iree_ok_status();
}

static bool loom_low_allocation_active_spill_victim_set_is_better(
    uint16_t candidate_count, uint32_t candidate_unit_count,
    uint32_t candidate_latest_end_point, uint32_t candidate_location_base,
    uint16_t best_count, uint32_t best_unit_count,
    uint32_t best_latest_end_point, uint32_t best_location_base) {
  if (best_count == 0) {
    return true;
  }
  if (candidate_count != best_count) {
    return candidate_count < best_count;
  }
  if (candidate_unit_count != best_unit_count) {
    return candidate_unit_count < best_unit_count;
  }
  if (candidate_latest_end_point != best_latest_end_point) {
    return candidate_latest_end_point > best_latest_end_point;
  }
  return candidate_location_base < best_location_base;
}

static iree_status_t loom_low_allocation_collect_active_spill_victim_set(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    const loom_low_allocation_class_capacity_t* capacity,
    uint32_t location_base, bool interval_requires_register,
    uint32_t* assignment_indices, loom_value_id_t* ignored_value_ids,
    uint16_t* out_assignment_count, uint32_t* out_unit_count,
    uint32_t* out_latest_end_point, bool* out_blocked) {
  *out_assignment_count = 0;
  *out_unit_count = 0;
  *out_latest_end_point = 0;
  *out_blocked = false;
  const uint32_t interval_end =
      loom_low_allocation_interval_storage_end_point(interval);

  const loom_low_allocation_assignment_t candidate = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = capacity->descriptor_reg_class_id,
      .start_point = interval->start_point,
      .end_point = interval_end,
      .unit_count = interval->unit_count,
      .location_kind = capacity->location_kind,
      .location_base = location_base,
      .location_count = interval->unit_count,
      .unit_end_point_start =
          loom_low_allocation_unit_end_point_start_for_value(
              state, interval->value_id),
  };

  uint16_t assignment_count = 0;
  uint32_t unit_count = 0;
  uint32_t latest_end_point = 0;
  for (iree_host_size_t i = 0; i < state->active.count; ++i) {
    const uint32_t assignment_index =
        state->active.assignment_indices[state->active.start + i];
    IREE_ASSERT_LT(assignment_index, state->assignment_count);
    const loom_low_allocation_assignment_t* assignment =
        &state->assignments[assignment_index];
    if (!loom_low_allocation_active_assignment_conflicts(
            state, assignment, &candidate, /*ignored_value_ids=*/NULL,
            /*ignored_value_count=*/0)) {
      continue;
    }
    if (!loom_low_allocation_assignment_can_spill(state, assignment, NULL)) {
      *out_blocked = true;
      return iree_ok_status();
    }
    if (!interval_requires_register && assignment->end_point <= interval_end) {
      *out_blocked = true;
      return iree_ok_status();
    }
    if (assignment_count == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "active spill victim set exceeds uint16_t");
    }
    if (unit_count > UINT32_MAX - assignment->unit_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "active spill victim unit count overflow");
    }
    assignment_indices[assignment_count] = assignment_index;
    ignored_value_ids[assignment_count] = assignment->value_id;
    ++assignment_count;
    unit_count += assignment->unit_count;
    if (latest_end_point < assignment->end_point) {
      latest_end_point = assignment->end_point;
    }
  }

  if (assignment_count == 0 ||
      loom_low_allocation_candidate_conflicts(
          state, interval, capacity->descriptor_reg_class_id,
          capacity->location_kind, location_base, interval->unit_count,
          ignored_value_ids, assignment_count,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
    *out_blocked = true;
    return iree_ok_status();
  }

  *out_assignment_count = assignment_count;
  *out_unit_count = unit_count;
  *out_latest_end_point = latest_end_point;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_spill_active_assignment_set(
    loom_low_allocation_build_state_t* state,
    const uint32_t* assignment_indices, uint16_t assignment_count) {
  for (uint16_t i = 0; i < assignment_count; ++i) {
    const uint32_t assignment_index = assignment_indices[i];
    IREE_ASSERT_LT(assignment_index, state->assignment_count);
    const loom_low_allocation_assignment_t* assignment =
        &state->assignments[assignment_index];
    loom_low_allocation_class_capacity_t assignment_capacity = {0};
    if (!loom_low_allocation_assignment_can_spill(state, assignment,
                                                  &assignment_capacity)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "active spill victim set became stale while assigning value %u",
          (unsigned)assignment->value_id);
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_spill_active_assignment(
        state, assignment_index, &assignment_capacity));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_find_active_spill_victim_set(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    const loom_low_allocation_class_capacity_t* capacity,
    bool interval_requires_register, uint32_t* out_location_base,
    bool* out_found) {
  *out_location_base = 0;
  *out_found = false;
  if (capacity->is_bounded && interval->unit_count > capacity->max_units) {
    return iree_ok_status();
  }
  if (state->active.count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "active allocation set exceeds uint16_t");
  }

  uint32_t last_base = 0;
  if (capacity->is_bounded) {
    last_base = capacity->max_units - interval->unit_count;
  } else {
    const uint32_t search_limit =
        loom_low_allocation_assigned_location_search_limit(
            state, capacity->descriptor_reg_class_id, capacity->location_kind);
    if (!loom_low_allocation_align_up_u32(
            search_limit, loom_low_allocation_interval_alignment(interval),
            &last_base)) {
      return iree_ok_status();
    }
  }

  uint32_t* candidate_assignment_indices = NULL;
  uint32_t* best_assignment_indices = NULL;
  loom_value_id_t* ignored_value_ids = NULL;
  if (state->active.count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->arena, state->active.count,
                                  sizeof(*candidate_assignment_indices),
                                  (void**)&candidate_assignment_indices));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, state->active.count, sizeof(*best_assignment_indices),
        (void**)&best_assignment_indices));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, state->active.count, sizeof(*ignored_value_ids),
        (void**)&ignored_value_ids));
  }

  const uint32_t alignment = loom_low_allocation_interval_alignment(interval);
  uint16_t best_assignment_count = 0;
  uint32_t best_unit_count = 0;
  uint32_t best_latest_end_point = 0;
  uint32_t best_location_base = 0;
  for (uint32_t base = 0; base <= last_base;) {
    uint16_t candidate_assignment_count = 0;
    uint32_t candidate_unit_count = 0;
    uint32_t candidate_latest_end_point = 0;
    bool blocked = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_collect_active_spill_victim_set(
        state, interval, capacity, base, interval_requires_register,
        candidate_assignment_indices, ignored_value_ids,
        &candidate_assignment_count, &candidate_unit_count,
        &candidate_latest_end_point, &blocked));
    if (!blocked &&
        loom_low_allocation_active_spill_victim_set_is_better(
            candidate_assignment_count, candidate_unit_count,
            candidate_latest_end_point, base, best_assignment_count,
            best_unit_count, best_latest_end_point, best_location_base)) {
      best_assignment_count = candidate_assignment_count;
      best_unit_count = candidate_unit_count;
      best_latest_end_point = candidate_latest_end_point;
      best_location_base = base;
      memcpy(best_assignment_indices, candidate_assignment_indices,
             (iree_host_size_t)candidate_assignment_count *
                 sizeof(*best_assignment_indices));
    }
    if (base > UINT32_MAX - alignment) {
      break;
    }
    base += alignment;
  }

  if (best_assignment_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_active_assignment_set(
      state, best_assignment_indices, best_assignment_count));
  *out_location_base = best_location_base;
  *out_found = true;
  return iree_ok_status();
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
    loom_low_allocation_active_unit_index_remove_assignment(
        &state->active_units, state->assignments, state->assignment_count,
        assignment_index);
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
  loom_low_allocation_active_unit_index_insert_assignment(
      &state->active_units, state->target.descriptor_set, state->assignments,
      state->assignment_count, assignment_index);
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
  if (loom_low_allocation_location_kind_is_register_like(
          assignment->location_kind)) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_validate_register_location_capacity(
            state, assignment->descriptor_reg_class_id,
            assignment->location_kind, assignment->location_base,
            assignment->location_count, IREE_SV("assignment"),
            state->function_op));
  }
  const uint32_t assignment_index = (uint32_t)state->assignment_count;
  loom_low_allocation_assignment_t stored_assignment = *assignment;
  stored_assignment.unit_end_point_start =
      loom_low_allocation_unit_end_point_start_for_value_ordinal(state,
                                                                 value_ordinal);
  stored_assignment.end_point =
      loom_low_allocation_live_range_assignment_max_unit_end_point(
          state->unit_end_points, state->unit_end_point_count,
          &stored_assignment);
  IREE_RETURN_IF_ERROR(loom_low_allocation_record_storage_release_actions(
      state, &stored_assignment));
  state->assignments[state->assignment_count++] = stored_assignment;
  state->assignment_indices_by_value_ordinal[value_ordinal] = assignment_index;
  loom_low_allocation_record_assignment_location_end(state, &stored_assignment);
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_record_storage_lease_instances_for_assignment(
          state, assignment_index, value_ordinal));
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
  if (!loom_low_allocation_assignment_is_register_like(operand_assignment)) {
    return iree_ok_status();
  }
  uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_low_allocation_resolve_descriptor_register_class(
      state, interval->value_class, &interval_reg_class_id, NULL));
  if (!loom_liveness_value_class_equal(operand_assignment->value_class,
                                       interval->value_class) ||
      operand_assignment->location_count != interval->unit_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low tied result does not match operand allocation class");
  }
  loom_value_id_t ignored_value_ids[2] = {tied_operand_id,
                                          LOOM_VALUE_ID_INVALID};
  uint16_t ignored_value_count = 1;
  loom_value_id_t copy_source_id = LOOM_VALUE_ID_INVALID;
  bool copy_source_used_after = false;
  IREE_RETURN_IF_ERROR(loom_low_allocation_copy_source_used_after_tied_consume(
      state, relation, tied_operand_id, &copy_source_id,
      &copy_source_used_after));
  if (copy_source_id != LOOM_VALUE_ID_INVALID && !copy_source_used_after) {
    ignored_value_ids[ignored_value_count++] = copy_source_id;
  }
  if (loom_low_allocation_candidate_conflicts(
          state, interval, interval_reg_class_id,
          operand_assignment->location_kind, operand_assignment->location_base,
          operand_assignment->location_count, ignored_value_ids,
          ignored_value_count, LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low tied result cannot share the operand location without "
        "overlapping another live interval");
  }

  const loom_low_allocation_assignment_t assignment = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = interval_reg_class_id,
      .start_point = interval->start_point,
      .end_point = loom_low_allocation_interval_storage_end_point(interval),
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
          /*ignored_value_count=*/1,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
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
      .end_point = loom_low_allocation_interval_storage_end_point(interval),
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
  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_reg_class_capacity(
      state, descriptor_reg_class_id, &capacity));
  if (!loom_low_allocation_location_range_fits_capacity(
          &capacity, location_kind, location_base, unit_count)) {
    return iree_ok_status();
  }
  const uint32_t alignment = loom_low_allocation_interval_alignment(interval);
  if (location_base % alignment != 0) {
    return iree_ok_status();
  }
  if (loom_low_allocation_candidate_conflicts(
          state, interval, descriptor_reg_class_id, location_kind,
          location_base, unit_count, ignored_value_ids, ignored_value_count,
          LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
    return iree_ok_status();
  }

  const loom_low_allocation_assignment_t assignment = {
      .value_id = interval->value_id,
      .value_class = interval->value_class,
      .descriptor_reg_class_id = descriptor_reg_class_id,
      .start_point = interval->start_point,
      .end_point = loom_low_allocation_interval_storage_end_point(interval),
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
  if (!loom_low_allocation_assignment_is_register_like(source_assignment)) {
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
  uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_low_allocation_resolve_descriptor_register_class(
      state, interval->value_class, &interval_reg_class_id, NULL));
  return loom_low_allocation_append_interval_at_location(
      state, interval, interval_reg_class_id, source_assignment->location_kind,
      result_location_base, interval->unit_count, ignored_value_ids,
      ignored_value_count, out_assigned);
}

static iree_status_t
loom_low_allocation_append_relation_interval_if_source_assigned(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation, bool* out_assigned) {
  *out_assigned = false;
  if (relation->result_unit_offset > interval->unit_count ||
      relation->unit_count >
          interval->unit_count - relation->result_unit_offset) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation placement relation exceeds result "
                            "interval units");
  }
  const loom_low_allocation_assignment_t* source_assignment =
      loom_low_allocation_current_assignment_for_value_ordinal(
          state, relation->source_ordinal);
  if (!source_assignment) {
    return iree_ok_status();
  }
  if (!loom_low_allocation_assignment_is_register_like(source_assignment)) {
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
  const loom_value_id_t* ignored_value_ids = NULL;
  uint16_t ignored_value_count = 0;
  const loom_value_id_t source_value_id = source_assignment->value_id;
  if (loom_low_allocation_can_ignore_branch_counterpart_conflict(
          state, interval, source_assignment)) {
    ignored_value_ids = &source_value_id;
    ignored_value_count = 1;
  }
  uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_low_allocation_resolve_descriptor_register_class(
      state, interval->value_class, &interval_reg_class_id, NULL));
  return loom_low_allocation_append_interval_at_location(
      state, interval, interval_reg_class_id, source_assignment->location_kind,
      result_location_base, interval->unit_count, ignored_value_ids,
      ignored_value_count, out_assigned);
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

static iree_status_t loom_low_allocation_assign_branch_destination_interval(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation, bool* out_assigned) {
  return loom_low_allocation_append_relation_interval_if_source_assigned(
      state, interval, relation, out_assigned);
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
    if (!loom_low_allocation_assignment_is_register_like(source_assignment)) {
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
  uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_low_allocation_resolve_descriptor_register_class(
      state, interval->value_class, &interval_reg_class_id, NULL));
  return loom_low_allocation_append_interval_at_location(
      state, interval, interval_reg_class_id, first_assignment->location_kind,
      result_location_base, interval->unit_count, ignored_value_ids,
      ignored_value_count, out_assigned);
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

static bool loom_low_allocation_source_location_for_concat_result(
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* result_assignment,
    uint32_t* out_location_base) {
  if (!loom_low_allocation_assignment_unit_span_fits(
          result_assignment, relation->result_unit_offset,
          relation->unit_count)) {
    return false;
  }
  if (result_assignment->location_base >
      UINT32_MAX - relation->result_unit_offset) {
    return false;
  }
  const uint32_t source_unit_location =
      result_assignment->location_base + relation->result_unit_offset;
  if (source_unit_location < relation->source_unit_offset) {
    return false;
  }
  *out_location_base = source_unit_location - relation->source_unit_offset;
  return true;
}

static iree_status_t loom_low_allocation_assign_concat_source_from_result(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* relation,
    const loom_low_allocation_assignment_t* result_assignment,
    bool* out_assigned) {
  *out_assigned = false;
  if (!loom_low_allocation_assignment_is_register_like(result_assignment) ||
      !loom_liveness_value_class_equal(result_assignment->value_class,
                                       interval->value_class)) {
    return iree_ok_status();
  }
  uint32_t location_base = 0;
  if (!loom_low_allocation_source_location_for_concat_result(
          relation, result_assignment, &location_base)) {
    return iree_ok_status();
  }
  const loom_value_id_t result_value_id =
      loom_low_placement_value_id(&state->placement, relation->result_ordinal);
  uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  IREE_RETURN_IF_ERROR(loom_low_allocation_resolve_descriptor_register_class(
      state, interval->value_class, &interval_reg_class_id, NULL));
  return loom_low_allocation_append_interval_at_location(
      state, interval, interval_reg_class_id, result_assignment->location_kind,
      location_base, interval->unit_count, &result_value_id,
      /*ignored_value_count=*/1, out_assigned);
}

static iree_status_t loom_low_allocation_concat_ignored_sources(
    loom_low_allocation_build_state_t* state,
    const loom_low_placement_relation_range_t* range,
    loom_value_id_t inline_ignored_value_ids[8],
    iree_host_size_t inline_ignored_value_capacity,
    loom_value_id_t** out_ignored_value_ids,
    uint16_t* out_ignored_value_count) {
  *out_ignored_value_ids = inline_ignored_value_ids;
  *out_ignored_value_count = 0;
  for (uint32_t i = 0; i < range->count; ++i) {
    const loom_low_placement_relation_t* relation =
        &state->placement.relations[range->start + i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
      continue;
    }
    if (*out_ignored_value_count == UINT16_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low.concat placement source count exceeds "
                              "uint16_t");
    }
    ++*out_ignored_value_count;
  }
  if (*out_ignored_value_count > inline_ignored_value_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, *out_ignored_value_count, sizeof(**out_ignored_value_ids),
        (void**)out_ignored_value_ids));
  }

  uint16_t ignored_value_index = 0;
  for (uint32_t i = 0; i < range->count; ++i) {
    const loom_low_placement_relation_t* relation =
        &state->placement.relations[range->start + i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
      continue;
    }
    (*out_ignored_value_ids)[ignored_value_index++] =
        loom_low_placement_value_id(&state->placement,
                                    relation->source_ordinal);
  }
  return iree_ok_status();
}

// Chooses a concat result span that can also accept the current source slice.
// Scheduled allocation may see scalar concat sources long before the concat op,
// so selecting only for the future result interval can reserve a span that the
// current source cannot occupy without a packet-local move.
static bool loom_low_allocation_find_concat_result_location_for_source(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* source_interval,
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* result_interval,
    loom_low_allocation_class_capacity_t capacity,
    const loom_value_id_t* ignored_value_ids, uint16_t ignored_value_count,
    uint32_t* out_result_location_base) {
  if (result_interval->unit_count == 0 ||
      (capacity.is_bounded &&
       result_interval->unit_count > capacity.max_units)) {
    return false;
  }

  const uint32_t result_alignment =
      loom_low_allocation_interval_alignment(result_interval);
  const uint32_t source_alignment =
      loom_low_allocation_interval_alignment(source_interval);
  const uint32_t assigned_limit =
      loom_low_allocation_assigned_location_search_limit(
          state, capacity.descriptor_reg_class_id, capacity.location_kind);

  uint32_t last_base = 0;
  if (capacity.is_bounded) {
    last_base = capacity.max_units - result_interval->unit_count;
  } else if (!loom_low_allocation_align_up_u32(assigned_limit, result_alignment,
                                               &last_base)) {
    return false;
  }

  for (uint32_t base = 0; base <= last_base;) {
    if (!loom_low_allocation_candidate_conflicts(
            state, result_interval, capacity.descriptor_reg_class_id,
            capacity.location_kind, base, result_interval->unit_count,
            ignored_value_ids, ignored_value_count,
            LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
      bool source_location_ok = false;
      uint32_t source_location_base = 0;
      if (base <= UINT32_MAX - relation->result_unit_offset) {
        const uint32_t source_unit_location =
            base + relation->result_unit_offset;
        if (source_unit_location >= relation->source_unit_offset) {
          source_location_base =
              source_unit_location - relation->source_unit_offset;
          source_location_ok =
              source_location_base % source_alignment == 0 &&
              source_location_base <= UINT32_MAX - source_interval->unit_count;
        }
      }
      if (source_location_ok &&
          !(capacity.is_bounded &&
            source_location_base + source_interval->unit_count >
                capacity.max_units) &&
          !loom_low_allocation_candidate_conflicts(
              state, source_interval, capacity.descriptor_reg_class_id,
              capacity.location_kind, source_location_base,
              source_interval->unit_count, /*ignored_value_ids=*/NULL,
              /*ignored_value_count=*/0,
              LOOM_LOW_ALLOCATION_STORAGE_RELEASE_FORBIDDEN)) {
        *out_result_location_base = base;
        return true;
      }
    }
    if (base > UINT32_MAX - result_alignment) {
      break;
    }
    base += result_alignment;
  }
  return false;
}

static iree_status_t loom_low_allocation_assign_concat_result_reservation(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* source_interval,
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* result_interval,
    const loom_low_placement_relation_range_t* result_range,
    bool* out_assigned) {
  *out_assigned = false;
  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_class_capacity(
      state, result_interval->value_class, &capacity));

  loom_value_id_t inline_ignored_value_ids[8];
  loom_value_id_t* ignored_value_ids = inline_ignored_value_ids;
  uint16_t ignored_value_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_concat_ignored_sources(
      state, result_range, inline_ignored_value_ids,
      IREE_ARRAYSIZE(inline_ignored_value_ids), &ignored_value_ids,
      &ignored_value_count));
  if (ignored_value_count == 0) {
    return iree_ok_status();
  }

  uint32_t result_location_base = 0;
  if (!loom_low_allocation_find_concat_result_location_for_source(
          state, source_interval, relation, result_interval, capacity,
          ignored_value_ids, ignored_value_count, &result_location_base)) {
    return iree_ok_status();
  }

  return loom_low_allocation_append_interval_at_location(
      state, result_interval, capacity.descriptor_reg_class_id,
      capacity.location_kind, result_location_base, result_interval->unit_count,
      ignored_value_ids, ignored_value_count, out_assigned);
}

static bool loom_low_allocation_branch_relation_covers_concat_source(
    const loom_low_placement_relation_t* branch_relation,
    const loom_low_placement_relation_t* concat_relation) {
  if (branch_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH ||
      branch_relation->source_ordinal != concat_relation->result_ordinal) {
    return false;
  }
  if (concat_relation->result_unit_offset <
      branch_relation->source_unit_offset) {
    return false;
  }
  if (concat_relation->unit_count >
      UINT32_MAX - concat_relation->result_unit_offset) {
    return false;
  }
  if (branch_relation->unit_count >
      UINT32_MAX - branch_relation->source_unit_offset) {
    return false;
  }
  const uint32_t concat_source_end =
      concat_relation->result_unit_offset + concat_relation->unit_count;
  const uint32_t branch_source_end =
      branch_relation->source_unit_offset + branch_relation->unit_count;
  return concat_source_end <= branch_source_end;
}

static iree_status_t
loom_low_allocation_assign_concat_source_from_branch_destination(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    const loom_low_placement_relation_t* concat_relation, bool* out_assigned) {
  *out_assigned = false;
  const loom_low_placement_relation_range_t source_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          &state->placement, concat_relation->result_ordinal);
  for (uint32_t i = 0; i < source_range.count; ++i) {
    const uint32_t relation_index =
        state->placement
            .relation_indices_by_source_ordinal[source_range.start + i];
    IREE_ASSERT_LT(relation_index, state->placement.relation_count);
    const loom_low_placement_relation_t* branch_relation =
        &state->placement.relations[relation_index];
    if (!loom_low_allocation_branch_relation_covers_concat_source(
            branch_relation, concat_relation)) {
      continue;
    }

    const loom_low_allocation_assignment_t* destination_assignment =
        loom_low_allocation_current_assignment_for_value_ordinal(
            state, branch_relation->result_ordinal);
    if (!destination_assignment ||
        !loom_low_allocation_assignment_is_register_like(
            destination_assignment) ||
        !loom_liveness_value_class_equal(destination_assignment->value_class,
                                         interval->value_class)) {
      continue;
    }

    const uint32_t concat_to_branch_unit_offset =
        concat_relation->result_unit_offset -
        branch_relation->source_unit_offset;
    if (branch_relation->result_unit_offset >
        UINT32_MAX - concat_to_branch_unit_offset) {
      continue;
    }
    const uint32_t destination_unit_offset =
        branch_relation->result_unit_offset + concat_to_branch_unit_offset;
    if (!loom_low_allocation_assignment_unit_span_fits(
            destination_assignment, destination_unit_offset,
            concat_relation->unit_count)) {
      continue;
    }
    const uint32_t destination_unit_location =
        destination_assignment->location_base + destination_unit_offset;
    if (destination_unit_location < concat_relation->source_unit_offset) {
      continue;
    }
    const uint32_t source_location_base =
        destination_unit_location - concat_relation->source_unit_offset;
    const loom_value_id_t* ignored_value_ids = NULL;
    uint16_t ignored_value_count = 0;
    const loom_value_id_t destination_value_id =
        destination_assignment->value_id;
    if (loom_low_allocation_can_ignore_branch_counterpart_conflict(
            state, interval, destination_assignment)) {
      ignored_value_ids = &destination_value_id;
      ignored_value_count = 1;
    }
    uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    IREE_RETURN_IF_ERROR(loom_low_allocation_resolve_descriptor_register_class(
        state, interval->value_class, &interval_reg_class_id, NULL));
    IREE_RETURN_IF_ERROR(loom_low_allocation_append_interval_at_location(
        state, interval, interval_reg_class_id,
        destination_assignment->location_kind, source_location_base,
        interval->unit_count, ignored_value_ids, ignored_value_count,
        out_assigned));
    if (*out_assigned) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool loom_low_allocation_relation_source_matches_branch_interval(
    const loom_low_placement_relation_t* relation,
    const loom_liveness_interval_t* interval) {
  if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH) {
    return false;
  }
  return relation->source_unit_offset <= interval->unit_count &&
         relation->unit_count <=
             interval->unit_count - relation->source_unit_offset;
}

static iree_status_t loom_low_allocation_assign_branch_source_interval(
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
    if (!loom_low_allocation_relation_source_matches_branch_interval(
            relation, interval)) {
      continue;
    }

    const loom_low_allocation_assignment_t* destination_assignment =
        loom_low_allocation_current_assignment_for_value_ordinal(
            state, relation->result_ordinal);
    if (!destination_assignment ||
        !loom_low_allocation_assignment_is_register_like(
            destination_assignment) ||
        !loom_liveness_value_class_equal(destination_assignment->value_class,
                                         interval->value_class) ||
        !loom_low_allocation_assignment_unit_span_fits(
            destination_assignment, relation->result_unit_offset,
            relation->unit_count)) {
      continue;
    }

    const uint32_t destination_unit_location =
        destination_assignment->location_base + relation->result_unit_offset;
    if (destination_unit_location < relation->source_unit_offset) {
      continue;
    }
    const uint32_t source_location_base =
        destination_unit_location - relation->source_unit_offset;
    const loom_value_id_t* ignored_value_ids = NULL;
    uint16_t ignored_value_count = 0;
    const loom_value_id_t destination_value_id =
        destination_assignment->value_id;
    if (loom_low_allocation_can_ignore_branch_counterpart_conflict(
            state, interval, destination_assignment)) {
      ignored_value_ids = &destination_value_id;
      ignored_value_count = 1;
    }
    uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    IREE_RETURN_IF_ERROR(loom_low_allocation_resolve_descriptor_register_class(
        state, interval->value_class, &interval_reg_class_id, NULL));
    IREE_RETURN_IF_ERROR(loom_low_allocation_append_interval_at_location(
        state, interval, interval_reg_class_id,
        destination_assignment->location_kind, source_location_base,
        interval->unit_count, ignored_value_ids, ignored_value_count,
        out_assigned));
    if (*out_assigned) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
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

    const loom_liveness_interval_t* result_interval =
        loom_liveness_interval_for_value_ordinal(&state->liveness,
                                                 relation->result_ordinal);
    if (!result_interval ||
        !loom_liveness_value_class_equal(result_interval->value_class,
                                         interval->value_class)) {
      continue;
    }
    const loom_low_placement_relation_range_t result_range =
        loom_low_placement_relation_range_for_value_ordinal(
            &state->placement, relation->result_ordinal);

    const loom_low_allocation_assignment_t* result_assignment =
        loom_low_allocation_current_assignment_for_value_ordinal(
            state, relation->result_ordinal);
    if (result_assignment) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_assign_concat_source_from_result(
          state, interval, relation, result_assignment, out_assigned));
      if (*out_assigned) {
        return iree_ok_status();
      }
      continue;
    }

    IREE_RETURN_IF_ERROR(
        loom_low_allocation_assign_concat_source_from_branch_destination(
            state, interval, relation, out_assigned));
    if (*out_assigned) {
      return iree_ok_status();
    }

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
          !loom_low_allocation_assignment_is_register_like(
              sibling_assignment) ||
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
      uint16_t interval_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_resolve_descriptor_register_class(
              state, interval->value_class, &interval_reg_class_id, NULL));
      IREE_RETURN_IF_ERROR(loom_low_allocation_append_interval_at_location(
          state, interval, interval_reg_class_id,
          sibling_assignment->location_kind, location_base,
          interval->unit_count, /*ignored_value_ids=*/NULL,
          /*ignored_value_count=*/0, out_assigned));
      if (*out_assigned) {
        return iree_ok_status();
      }
    }

    bool assigned_result_reservation = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_assign_concat_result_reservation(
        state, interval, relation, result_interval, &result_range,
        &assigned_result_reservation));
    if (!assigned_result_reservation) {
      continue;
    }
    result_assignment =
        loom_low_allocation_current_assignment_for_value_ordinal(
            state, relation->result_ordinal);
    IREE_RETURN_IF_ERROR(loom_low_allocation_assign_concat_source_from_result(
        state, interval, relation, result_assignment, out_assigned));
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
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY: {
        bool requires_materialized_storage = false;
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_copy_relation_requires_materialized_storage(
                state, relation, &requires_materialized_storage));
        if (requires_materialized_storage) {
          break;
        }
        IREE_RETURN_IF_ERROR(loom_low_allocation_assign_relation_interval(
            state, interval, relation, out_assigned));
        if (*out_assigned) {
          return iree_ok_status();
        }
        break;
      }
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
      case LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH: {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_assign_branch_destination_interval(
                state, interval, relation, out_assigned));
        if (*out_assigned) {
          return iree_ok_status();
        }
        break;
      }
      default:
        break;
    }
  }
  return iree_ok_status();
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
      loom_low_allocation_assignment_is_register_like(source_assignment) &&
      loom_low_allocation_assignment_is_register_like(result_assignment) &&
      loom_low_allocation_storage_assignment_locations_share(
          state->target.descriptor_set, source_assignment, result_assignment);
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
  iree_host_size_t copy_count =
      loom_low_allocation_move_topology_count_copy_ops(state->body);
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

static iree_status_t loom_low_allocation_record_edge_copy_segment(
    loom_low_allocation_build_state_t* state, uint16_t payload_index,
    loom_value_id_t source_value_id, loom_value_id_t destination_value_id,
    uint32_t source_unit_offset, uint32_t destination_unit_offset,
    uint32_t unit_count) {
  uint32_t source_assignment_index = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
      state, source_value_id, &source_assignment_index));
  uint32_t destination_assignment_index = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
      state, destination_value_id, &destination_assignment_index));
  state->edge_copies[state->edge_copy_count++] =
      (loom_low_allocation_edge_copy_t){
          .payload_index = payload_index,
          .source_value_id = source_value_id,
          .destination_value_id = destination_value_id,
          .source_assignment_index = source_assignment_index,
          .destination_assignment_index = destination_assignment_index,
          .source_unit_offset = source_unit_offset,
          .destination_unit_offset = destination_unit_offset,
          .unit_count = unit_count,
      };
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_record_branch_payload_edge_copies(
    loom_low_allocation_build_state_t* state, uint16_t payload_index,
    loom_value_id_t payload_value_id, loom_value_id_t destination_value_id) {
  const loom_op_t* concat_op =
      loom_low_allocation_move_topology_value_defining_concat(state->module,
                                                              payload_value_id);
  if (!concat_op) {
    uint32_t source_assignment_index = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
        state, payload_value_id, &source_assignment_index));
    const loom_low_allocation_assignment_t* source_assignment =
        &state->assignments[source_assignment_index];
    return loom_low_allocation_record_edge_copy_segment(
        state, payload_index, payload_value_id, destination_value_id,
        /*source_unit_offset=*/0, /*destination_unit_offset=*/0,
        source_assignment->location_count);
  }

  uint32_t destination_assignment_index = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
      state, destination_value_id, &destination_assignment_index));
  const loom_low_allocation_assignment_t* destination_assignment =
      &state->assignments[destination_assignment_index];

  uint32_t destination_unit_offset = 0;
  loom_value_slice_t sources = loom_low_concat_sources(concat_op);
  for (uint16_t i = 0; i < sources.count; ++i) {
    uint32_t source_assignment_index = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_index_for_value(
        state, sources.values[i], &source_assignment_index));
    const loom_low_allocation_assignment_t* source_assignment =
        &state->assignments[source_assignment_index];
    if (source_assignment->location_count >
        UINT32_MAX - destination_unit_offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low.br edge-copy destination offset exceeds "
                              "u32 range");
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_record_edge_copy_segment(
        state, payload_index, sources.values[i], destination_value_id,
        /*source_unit_offset=*/0, destination_unit_offset,
        source_assignment->location_count));
    destination_unit_offset += source_assignment->location_count;
  }
  if (destination_unit_offset != destination_assignment->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.br decomposed low.concat payload does not fill destination");
  }
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
  return loom_low_allocation_live_range_op_program_point(&state->liveness, op,
                                                         out_program_point);
}

static iree_status_t loom_low_allocation_note_unit_use_at_point(
    loom_low_allocation_build_state_t* state, loom_value_id_t value_id,
    uint32_t unit_offset, uint32_t unit_count, uint32_t point) {
  if (unit_count == 0) {
    return iree_ok_status();
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_value_ordinal_for_value(state, value_id,
                                                   &value_ordinal)) {
    return iree_ok_status();
  }
  const uint32_t unit_end_point_start =
      loom_low_allocation_unit_end_point_start_for_value_ordinal(state,
                                                                 value_ordinal);
  if (unit_end_point_start == UINT32_MAX) {
    return iree_ok_status();
  }
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value_ordinal(&state->liveness, value_ordinal);
  if (!interval || unit_offset > interval->unit_count ||
      unit_count > interval->unit_count - unit_offset) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation unit liveness use exceeds value unit count");
  }
  if (point == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low allocation unit use point exceeds u32 range");
  }
  const uint32_t end_point = point + 1u;
  for (uint32_t i = 0; i < unit_count; ++i) {
    uint32_t* unit_end_point =
        &state->unit_end_points[unit_end_point_start + unit_offset + i];
    if (*unit_end_point < end_point) {
      *unit_end_point = end_point;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_note_value_use_at_point(
    loom_low_allocation_build_state_t* state, loom_value_id_t value_id,
    uint32_t point) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_value_ordinal_for_value(state, value_id,
                                                   &value_ordinal)) {
    return iree_ok_status();
  }
  const loom_liveness_interval_t* interval =
      loom_liveness_interval_for_value_ordinal(&state->liveness, value_ordinal);
  if (!interval || !loom_low_allocation_interval_is_allocatable(interval)) {
    return iree_ok_status();
  }
  return loom_low_allocation_note_unit_use_at_point(
      state, value_id, /*unit_offset=*/0, interval->unit_count, point);
}

static iree_status_t loom_low_allocation_note_generic_op_unit_uses(
    loom_low_allocation_build_state_t* state, const loom_op_t* op,
    uint32_t point) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_note_value_use_at_point(state, operands[i], point));
  }
  return iree_ok_status();
}

static bool loom_low_allocation_descriptor_operand_is_explicit_packet_value(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_operand_index) {
  IREE_ASSERT_LT(descriptor_operand_index, descriptor->operand_count);
  const loom_low_operand_t* descriptor_operand =
      &descriptor_set
           ->operands[descriptor->operand_start + descriptor_operand_index];
  return loom_low_operand_role_is_packet_operand(descriptor_operand->role) &&
         !iree_any_bit_set(descriptor_operand->flags,
                           LOOM_LOW_OPERAND_FLAG_IMPLICIT);
}

static uint16_t loom_low_allocation_descriptor_packet_operand_index(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_operand_index) {
  uint16_t packet_operand_index = 0;
  for (uint16_t i = descriptor->result_count; i < descriptor_operand_index;
       ++i) {
    if (loom_low_allocation_descriptor_operand_is_explicit_packet_value(
            descriptor_set, descriptor, i)) {
      ++packet_operand_index;
    }
  }
  return packet_operand_index;
}

static iree_status_t loom_low_allocation_note_early_clobber_operand_uses(
    loom_low_allocation_build_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor,
    uint16_t early_clobber_result_index, uint32_t clobber_point) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = descriptor->result_count; i < descriptor->operand_count;
       ++i) {
    if (!loom_low_allocation_descriptor_operand_is_explicit_packet_value(
            descriptor_set, descriptor, i) ||
        loom_low_descriptor_operands_are_tied(descriptor_set, descriptor,
                                              early_clobber_result_index, i)) {
      continue;
    }
    const uint16_t operand_index =
        loom_low_allocation_descriptor_packet_operand_index(descriptor_set,
                                                            descriptor, i);
    if (operand_index >= op->operand_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low allocation early-clobber operand index exceeds packet operand "
          "count");
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_note_value_use_at_point(
        state, operands[operand_index], clobber_point));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_note_descriptor_unit_uses(
    loom_low_allocation_build_state_t* state, const loom_op_t* op,
    uint32_t point) {
  if (!loom_low_op_isa(op) && !loom_low_const_isa(op)) {
    return iree_ok_status();
  }
  if (point >= UINT32_MAX - 1u) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low allocation descriptor operation point exceeds u32 range");
  }
  const uint32_t clobber_point = point + 1u;
  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
      state->module, &state->target, op, &packet));
  if (packet.descriptor == NULL) {
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  const loom_low_descriptor_t* descriptor = packet.descriptor;
  for (uint16_t i = 0; i < descriptor->constraint_count; ++i) {
    const loom_low_constraint_t* constraint =
        &descriptor_set->constraints[descriptor->constraint_start + i];
    if (constraint->kind != LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_note_early_clobber_operand_uses(
        state, op, descriptor_set, descriptor, constraint->lhs_operand_index,
        clobber_point));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_note_slice_unit_uses(
    loom_low_allocation_build_state_t* state, const loom_op_t* op,
    uint32_t point) {
  const int64_t offset = loom_low_slice_offset(op);
  if (offset < 0 || offset > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation unit liveness saw malformed low.slice offset");
  }
  loom_value_ordinal_t result_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_value_ordinal_for_value(
          state, loom_low_slice_result(op), &result_ordinal)) {
    return iree_ok_status();
  }
  const loom_liveness_interval_t* result_interval =
      loom_liveness_interval_for_value_ordinal(&state->liveness,
                                               result_ordinal);
  if (!result_interval ||
      !loom_low_allocation_interval_is_allocatable(result_interval)) {
    return iree_ok_status();
  }
  return loom_low_allocation_note_unit_use_at_point(
      state, loom_low_slice_source(op), (uint32_t)offset,
      result_interval->unit_count, point);
}

static iree_status_t loom_low_allocation_note_op_unit_uses(
    loom_low_allocation_build_state_t* state, const loom_op_t* op) {
  uint32_t point = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_low_allocation_op_program_point(state, op, &point));
  if (loom_low_slice_isa(op)) {
    return loom_low_allocation_note_slice_unit_uses(state, op, point);
  }
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_note_generic_op_unit_uses(state, op, point));
  return loom_low_allocation_note_descriptor_unit_uses(state, op, point);
}

static iree_status_t loom_low_allocation_note_value_unit_uses_at_point(
    loom_low_allocation_build_state_t* state, const loom_value_id_t* values,
    iree_host_size_t value_count, uint32_t point) {
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_note_value_use_at_point(state, values[i], point));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_note_block_boundary_unit_uses(
    loom_low_allocation_build_state_t* state) {
  for (iree_host_size_t i = 0; i < state->liveness.block_count; ++i) {
    const loom_liveness_block_info_t* block_info = &state->liveness.blocks[i];
    IREE_RETURN_IF_ERROR(loom_low_allocation_note_value_unit_uses_at_point(
        state, block_info->live_in_values, block_info->live_in_count,
        block_info->start_point));
    IREE_RETURN_IF_ERROR(loom_low_allocation_note_value_unit_uses_at_point(
        state, block_info->live_out_values, block_info->live_out_count,
        block_info->end_point));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_initialize_unit_liveness(
    loom_low_allocation_build_state_t* state) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->liveness.value_count,
      sizeof(*state->unit_end_point_starts_by_value_ordinal),
      (void**)&state->unit_end_point_starts_by_value_ordinal));
  for (iree_host_size_t i = 0; i < state->liveness.value_count; ++i) {
    state->unit_end_point_starts_by_value_ordinal[i] = UINT32_MAX;
  }

  iree_host_size_t unit_end_point_count = 0;
  for (iree_host_size_t i = 0; i < state->liveness.interval_count; ++i) {
    const loom_liveness_interval_t* interval = &state->liveness.intervals[i];
    if (!loom_low_allocation_interval_is_allocatable(interval)) {
      continue;
    }
    if (interval->unit_count > IREE_HOST_SIZE_MAX - unit_end_point_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low allocation unit liveness count exceeds host size");
    }
    unit_end_point_count += interval->unit_count;
  }
  if (unit_end_point_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, unit_end_point_count, sizeof(*state->unit_end_points),
      (void**)&state->unit_end_points));
  state->unit_end_point_count = unit_end_point_count;

  iree_host_size_t unit_end_point_start = 0;
  for (iree_host_size_t i = 0; i < state->liveness.interval_count; ++i) {
    const loom_liveness_interval_t* interval = &state->liveness.intervals[i];
    if (!loom_low_allocation_interval_is_allocatable(interval)) {
      continue;
    }
    loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_allocation_value_ordinal_for_interval(
        state, interval, &value_ordinal));
    if (unit_end_point_start > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low allocation unit liveness start exceeds u32 range");
    }
    state->unit_end_point_starts_by_value_ordinal[value_ordinal] =
        (uint32_t)unit_end_point_start;
    for (uint32_t unit_index = 0; unit_index < interval->unit_count;
         ++unit_index) {
      state->unit_end_points[unit_end_point_start + unit_index] =
          loom_low_allocation_interval_initial_unit_end_point(interval);
    }
    unit_end_point_start += interval->unit_count;
  }

  // Unit liveness refines the value-granular analysis inside blocks so
  // operations like low.slice can release dead units independently. CFG
  // boundaries are still value-granular: every unit of a block live-in/out
  // value must stay reserved across the boundary until a per-unit dataflow
  // analysis can prove otherwise.
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_note_block_boundary_unit_uses(state));

  loom_block_t* block = NULL;
  loom_region_for_each_block(state->body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_note_op_unit_uses(state, op));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_extend_unit_liveness_for_tied_results(
    loom_low_allocation_build_state_t* state) {
  if (state->unit_end_points == NULL) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < state->placement.relation_count; ++i) {
    const loom_low_placement_relation_t* relation =
        &state->placement.relations[i];
    if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT) {
      continue;
    }

    const loom_liveness_interval_t* source_interval =
        loom_liveness_interval_for_value_ordinal(&state->liveness,
                                                 relation->source_ordinal);
    const loom_liveness_interval_t* result_interval =
        loom_liveness_interval_for_value_ordinal(&state->liveness,
                                                 relation->result_ordinal);
    if (!source_interval || !result_interval ||
        !loom_low_allocation_interval_is_allocatable(source_interval) ||
        !loom_low_allocation_interval_is_allocatable(result_interval)) {
      continue;
    }
    if (relation->source_unit_offset > source_interval->unit_count ||
        relation->unit_count >
            source_interval->unit_count - relation->source_unit_offset ||
        relation->result_unit_offset > result_interval->unit_count ||
        relation->unit_count >
            result_interval->unit_count - relation->result_unit_offset) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low tied-result placement relation exceeds allocation units");
    }

    const uint32_t source_unit_end_point_start =
        loom_low_allocation_unit_end_point_start_for_value_ordinal(
            state, relation->source_ordinal);
    const uint32_t result_unit_end_point_start =
        loom_low_allocation_unit_end_point_start_for_value_ordinal(
            state, relation->result_ordinal);
    if (source_unit_end_point_start == UINT32_MAX ||
        result_unit_end_point_start == UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low tied-result placement relation references a value without "
          "allocation unit liveness");
    }

    for (uint32_t unit_index = 0; unit_index < relation->unit_count;
         ++unit_index) {
      const uint32_t source_unit_index = source_unit_end_point_start +
                                         relation->source_unit_offset +
                                         unit_index;
      const uint32_t result_unit_index = result_unit_end_point_start +
                                         relation->result_unit_offset +
                                         unit_index;
      IREE_ASSERT_LT(source_unit_index, state->unit_end_point_count);
      IREE_ASSERT_LT(result_unit_index, state->unit_end_point_count);
      uint32_t* source_end_point = &state->unit_end_points[source_unit_index];
      const uint32_t result_end_point =
          state->unit_end_points[result_unit_index];
      if (*source_end_point < result_end_point) {
        *source_end_point = result_end_point;
      }
    }
  }
  return iree_ok_status();
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
      .copy_count = 0,
      .temporary_start = 0,
      .temporary_count = 0,
  };
  const uint32_t copy_start = (uint32_t)state->edge_copy_count;
  for (uint16_t i = 0; i < dest->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_record_branch_payload_edge_copies(
        state, i, args.values[i], dest->arg_ids[i]));
  }
  if (state->edge_copy_count < copy_start ||
      state->edge_copy_count - copy_start > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low.br edge-copy group exceeds u32 range");
  }
  group->copy_count = (uint32_t)(state->edge_copy_count - copy_start);
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_record_edge_copy_groups(
    loom_low_allocation_build_state_t* state) {
  iree_host_size_t group_count = 0;
  iree_host_size_t copy_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_move_topology_count_edge_copy_groups(
      state->module, state->body, &group_count, &copy_count));
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
  if (unit_index >= edge_copy->unit_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low.br edge-copy unit index exceeds copied segment");
  }
  const loom_low_allocation_assignment_t* source_assignment =
      loom_low_allocation_edge_copy_source_assignment(state, edge_copy);
  const loom_low_allocation_assignment_t* destination_assignment =
      loom_low_allocation_edge_copy_destination_assignment(state, edge_copy);
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_unit_location(
      source_assignment, edge_copy->source_unit_offset + unit_index,
      out_source));
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_unit_location(
      destination_assignment, edge_copy->destination_unit_offset + unit_index,
      out_destination));
  return iree_ok_status();
}

static bool loom_low_allocation_unit_locations_form_register_move(
    const loom_low_allocation_unit_location_t* source,
    const loom_low_allocation_unit_location_t* destination) {
  return loom_low_allocation_location_kind_is_register_like(
             source->location_kind) &&
         loom_low_allocation_location_kind_is_register_like(
             destination->location_kind) &&
         !loom_low_allocation_unit_locations_equal(source, destination);
}

static iree_status_t loom_low_allocation_edge_copy_group_unit_count(
    const loom_low_allocation_build_state_t* state,
    const loom_low_allocation_edge_copy_group_t* group,
    uint32_t* out_unit_count) {
  *out_unit_count = 0;
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &state->edge_copies[group->copy_start + i];
    for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
         ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          state, edge_copy, unit_index, &source, &destination));
      if (!loom_low_allocation_unit_locations_form_register_move(
              &source, &destination)) {
        continue;
      }
      if (*out_unit_count == UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "low.br edge-copy unit count exceeds u32");
      }
      ++*out_unit_count;
    }
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
    for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
         ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t candidate_destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          state, edge_copy, unit_index, &source, &candidate_destination));
      if (!loom_low_allocation_unit_locations_form_register_move(
              &source, &candidate_destination)) {
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
    uint32_t unit_limit = edge_copy->unit_count;
    if (copy_index == stop_copy_index) {
      unit_limit = stop_unit_index;
    }
    for (uint32_t unit_index = 0; unit_index < unit_limit; ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          state, edge_copy, unit_index, &source, &destination));
      if (!loom_low_allocation_unit_locations_form_register_move(
              &source, &destination)) {
        continue;
      }
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
    for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
         ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          state, edge_copy, unit_index, &source, &destination));
      if (!loom_low_allocation_unit_storage_classes_equal(storage_class,
                                                          &destination) ||
          !loom_low_allocation_unit_locations_form_register_move(
              &source, &destination)) {
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
    for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
         ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          state, edge_copy, unit_index, &source, &destination));
      if (!loom_low_allocation_unit_locations_form_register_move(
              &source, &destination)) {
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
        point < assignment->start_point) {
      continue;
    }
    if (!loom_low_allocation_storage_reg_classes_share(
            state->target.descriptor_set, assignment->descriptor_reg_class_id,
            location->descriptor_reg_class_id)) {
      continue;
    }
    const uint64_t assignment_end =
        (uint64_t)assignment->location_base + assignment->location_count;
    if (location->location >= assignment->location_base &&
        location->location < assignment_end) {
      const uint32_t unit_offset =
          (uint32_t)(location->location - assignment->location_base);
      if (point < loom_low_allocation_live_range_assignment_unit_end_point(
                      state->unit_end_points, state->unit_end_point_count,
                      assignment, unit_offset)) {
        return true;
      }
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
    for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
         ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          state, edge_copy, unit_index, &source, &destination));
      if (!loom_low_allocation_unit_locations_form_register_move(
              &source, &destination)) {
        continue;
      }
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
      loom_param_string(loom_low_diagnostic_value_class_name(
          state->target.descriptor_set, value_class)),
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
  if (!loom_low_allocation_location_kind_is_register_like(
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
    for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
         ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          state, edge_copy, unit_index, &source, &destination));
      if (!loom_low_allocation_unit_locations_form_register_move(
              &source, &destination)) {
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
  if (!loom_low_allocation_move_topology_concat_requires_packet_materialization_for_module(
          state->module, op)) {
    return iree_ok_status();
  }
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
      if (!loom_low_allocation_move_topology_op_has_packet_moves(op)) {
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
    if (!loom_low_allocation_unit_locations_form_register_move(
            &moves[i].source, &moves[i].destination)) {
      continue;
    }
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
    if (!loom_low_allocation_unit_locations_form_register_move(
            &move->source, &move->destination)) {
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

static iree_status_t loom_low_allocation_packet_move_class_has_cycle(
    const loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t move_count,
    const loom_low_allocation_unit_location_t* storage_class,
    bool* out_has_cycle) {
  *out_has_cycle = false;
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    const loom_low_allocation_packet_unit_move_t* move = &moves[i];
    if (!loom_low_allocation_unit_storage_classes_equal(storage_class,
                                                        &move->destination) ||
        !loom_low_allocation_unit_locations_form_register_move(
            &move->source, &move->destination)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_starts_cycle(
        moves, move_count, &move->destination, &move->source, out_has_cycle));
    if (*out_has_cycle) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool loom_low_allocation_packet_move_class_seen_before(
    const loom_low_allocation_packet_unit_move_t* moves,
    iree_host_size_t stop_move_index,
    const loom_low_allocation_unit_location_t* storage_class) {
  for (iree_host_size_t i = 0; i < stop_move_index; ++i) {
    if (!loom_low_allocation_unit_locations_form_register_move(
            &moves[i].source, &moves[i].destination)) {
      continue;
    }
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
  if (!loom_low_allocation_location_kind_is_register_like(
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
    if (!loom_low_allocation_unit_locations_form_register_move(
            &move->source, &move->destination) ||
        loom_low_allocation_packet_move_class_seen_before(moves, i,
                                                          &move->destination)) {
      continue;
    }
    bool has_cycle = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_packet_move_class_has_cycle(
        moves, move_count, &move->destination, &has_cycle));
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
      if (loom_low_allocation_move_topology_op_has_packet_moves(op)) {
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
  iree_host_size_t allocatable_unit_count = 0;
  for (iree_host_size_t i = 0; i < state->liveness.interval_count; ++i) {
    if (loom_low_allocation_interval_is_allocatable(
            &state->liveness.intervals[i])) {
      ++allocatable_count;
      if (state->liveness.intervals[i].unit_count >
          IREE_HOST_SIZE_MAX - allocatable_unit_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "allocation unit count exceeds host size");
      }
      allocatable_unit_count += state->liveness.intervals[i].unit_count;
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_active_unit_index_initialize(
      allocatable_count, allocatable_unit_count, state->arena,
      &state->active_units));
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

    loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_allocation_value_ordinal_for_interval(
        state, interval, &value_ordinal));
    if (loom_low_allocation_current_assignment_for_value_ordinal(
            state, value_ordinal)) {
      continue;
    }

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

    bool assigned_branch_source_interval = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_assign_branch_source_interval(
        state, interval, &assigned_branch_source_interval));
    if (assigned_branch_source_interval) {
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
    const bool requires_register =
        loom_low_allocation_interval_requires_register_location(state,
                                                                interval);
    if (!assigned && (capacity.is_spillable || requires_register)) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_find_active_spill_victim_set(
          state, interval, &capacity, requires_register, &location_base,
          &assigned));
    }
    if (!assigned && (!capacity.is_spillable || requires_register)) {
      const loom_low_descriptor_set_t* descriptor_set =
          state->target.descriptor_set;
      const loom_low_reg_class_t* reg_class =
          &descriptor_set->reg_classes[capacity.descriptor_reg_class_id];
      iree_string_view_t reg_class_name = loom_low_descriptor_set_string(
          descriptor_set, reg_class->name_string_offset);
      const uint32_t budget_units =
          capacity.is_bounded ? capacity.max_units : UINT32_MAX;
      if (requires_register) {
        iree_string_view_t value_name =
            loom_low_diagnostic_value_name(state->module, interval->value_id);
        IREE_RETURN_IF_ERROR(loom_low_allocation_emit_failure(
            state,
            loom_low_diagnostic_value_origin_op(
                state->module, interval->value_id, state->function_op),
            interval->value_class, budget_units, interval->unit_count,
            IREE_SV("spill-traffic-register-exhausted")));
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "allocation exhausted register class '%.*s' for materialized "
            "spill traffic value '%.*s'",
            (int)reg_class_name.size, reg_class_name.data, (int)value_name.size,
            value_name.data);
      }
      IREE_RETURN_IF_ERROR(loom_low_allocation_emit_failure(
          state, state->function_op, interval->value_class, budget_units,
          interval->unit_count, IREE_SV("unspillable-register-exhausted")));
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
        .end_point = loom_low_allocation_interval_storage_end_point(interval),
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
      module, low_func_op, options->descriptor_registry,
      options->target_selection, options->emitter, &state.target));
  if (!state.target.descriptor_set) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low function target did not resolve");
  }
  loom_local_value_domain_t value_domain = {0};
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
    status = loom_low_allocation_initialize_unit_liveness(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_extend_unit_liveness_for_tied_results(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_resolve_fixed_values(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_prepare_storage_lease_instances(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_assign_intervals(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_finalize_storage_lease_instances(&state);
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
        .unit_end_points = state.unit_end_points,
        .unit_end_point_count = state.unit_end_point_count,
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
        .storage_leases = options->storage_leases,
        .storage_lease_instances = state.storage_lease_instances,
        .storage_lease_instance_count = state.storage_lease_instance_count,
        .storage_release_actions = state.storage_release_actions,
        .storage_release_action_count = state.storage_release_action_count,
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
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_diagnostics_emit(
        &table, options->diagnostic_flags, options->emitter);
  }
  if (iree_status_is_ok(status)) {
    *out_table = table;
  }
  return status;
}
