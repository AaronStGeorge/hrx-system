// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation.h"

#include <string.h>

#include "loom/codegen/low/requirements.h"
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
  // Mutable remark records being built.
  loom_low_allocation_remark_t* remarks;
  // Number of initialized assignment records.
  iree_host_size_t assignment_count;
  // Number of initialized remark records.
  iree_host_size_t remark_count;
  // Number of spill-slot assignments.
  iree_host_size_t spill_count;
} loom_low_allocation_build_state_t;

typedef struct loom_low_allocation_class_capacity_t {
  // Location kind used for this class.
  loom_low_allocation_location_kind_t location_kind;
  // Maximum allocation units when |is_bounded| is true.
  uint32_t max_units;
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

static bool loom_low_allocation_value_class_equal(
    loom_liveness_value_class_t lhs, loom_liveness_value_class_t rhs) {
  return lhs.type_kind == rhs.type_kind &&
         lhs.element_type == rhs.element_type &&
         lhs.register_class_id == rhs.register_class_id;
}

static bool loom_low_allocation_interval_overlaps(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  return lhs->start_point < rhs->end_point && rhs->start_point < lhs->end_point;
}

static bool loom_low_allocation_location_is_register_like(
    loom_low_allocation_location_kind_t location_kind) {
  return location_kind == LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
         location_kind == LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;
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

static bool loom_low_allocation_location_ranges_overlap(
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs) {
  uint64_t lhs_end = (uint64_t)lhs->location_base + lhs->location_count;
  uint64_t rhs_end = (uint64_t)rhs->location_base + rhs->location_count;
  return lhs->location_base < rhs_end && rhs->location_base < lhs_end;
}

static bool loom_low_allocation_interval_is_allocatable(
    const loom_liveness_interval_t* interval) {
  return interval->value_class.type_kind == LOOM_TYPE_REGISTER &&
         interval->unit_count > 0;
}

static bool loom_low_allocation_interval_less(
    const loom_liveness_interval_t* lhs, const loom_liveness_interval_t* rhs) {
  if (lhs->start_point != rhs->start_point) {
    return lhs->start_point < rhs->start_point;
  }
  if (lhs->end_point != rhs->end_point) return lhs->end_point < rhs->end_point;
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
  if (value_class.type_kind != LOOM_TYPE_REGISTER) return false;
  iree_string_view_t value_register_class =
      loom_low_allocation_module_string(module, value_class.register_class_id);
  return iree_string_view_equal(value_register_class, register_class);
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
  if (!descriptor_set) return iree_ok_status();
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

static iree_status_t loom_low_allocation_descriptor_location_kind(
    const loom_low_allocation_build_state_t* state,
    loom_liveness_value_class_t value_class,
    loom_low_allocation_location_kind_t* out_location_kind) {
  iree_string_view_t register_class_name = loom_low_allocation_module_string(
      state->module, value_class.register_class_id);
  const loom_low_reg_class_t* reg_class = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_require_descriptor_register_class(
      state, register_class_name, &reg_class));
  if (reg_class->physical_count > 0 ||
      iree_any_bit_set(reg_class->flags, LOOM_LOW_REG_CLASS_FLAG_PHYSICAL)) {
    *out_location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
    return iree_ok_status();
  }
  *out_location_kind = LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_class_capacity(
    const loom_low_allocation_build_state_t* state,
    loom_liveness_value_class_t value_class,
    loom_low_allocation_class_capacity_t* out_capacity) {
  uint32_t budget_units = 0;
  if (loom_low_allocation_lookup_budget(state, value_class, &budget_units)) {
    loom_low_allocation_location_kind_t location_kind =
        LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED;
    IREE_RETURN_IF_ERROR(loom_low_allocation_descriptor_location_kind(
        state, value_class, &location_kind));
    *out_capacity = (loom_low_allocation_class_capacity_t){
        .location_kind = location_kind,
        .max_units = budget_units,
        .is_bounded = true,
    };
    return iree_ok_status();
  }

  iree_string_view_t register_class_name = loom_low_allocation_module_string(
      state->module, value_class.register_class_id);
  const loom_low_reg_class_t* reg_class = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_require_descriptor_register_class(
      state, register_class_name, &reg_class));
  if (reg_class->physical_count > 0) {
    *out_capacity = (loom_low_allocation_class_capacity_t){
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .max_units = reg_class->physical_count,
        .is_bounded = true,
    };
    return iree_ok_status();
  }

  *out_capacity = (loom_low_allocation_class_capacity_t){
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_TARGET_ID,
      .max_units = UINT32_MAX,
      .is_bounded = false,
  };
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

static bool loom_low_allocation_candidate_conflicts(
    const loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count) {
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
    if (existing->location_kind != location_kind) continue;
    if (!loom_low_allocation_value_class_equal(existing->value_class,
                                               interval->value_class)) {
      continue;
    }
    if (!loom_low_allocation_interval_overlaps(existing, &candidate)) continue;
    if (loom_low_allocation_location_ranges_overlap(existing, &candidate)) {
      return true;
    }
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
    if (assignment->location_kind != location_kind) continue;
    if (!loom_low_allocation_value_class_equal(assignment->value_class,
                                               value_class)) {
      continue;
    }
    uint64_t assignment_end =
        (uint64_t)assignment->location_base + assignment->location_count;
    if (assignment_end > UINT32_MAX) return UINT32_MAX;
    if ((uint32_t)assignment_end > max_end) {
      max_end = (uint32_t)assignment_end;
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

  for (uint32_t base = 0; base <= last_base; ++base) {
    if (!loom_low_allocation_candidate_conflicts(state, interval,
                                                 capacity.location_kind, base,
                                                 interval->unit_count)) {
      *out_base = base;
      return true;
    }
    if (base == UINT32_MAX) break;
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

static iree_status_t loom_low_allocation_assign_intervals(
    loom_low_allocation_build_state_t* state) {
  iree_host_size_t allocatable_count = 0;
  for (iree_host_size_t i = 0; i < state->liveness.interval_count; ++i) {
    if (loom_low_allocation_interval_is_allocatable(
            &state->liveness.intervals[i])) {
      ++allocatable_count;
    }
  }
  if (allocatable_count == 0) return iree_ok_status();

  const loom_liveness_interval_t** intervals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, allocatable_count, sizeof(*intervals), (void**)&intervals));
  state->assignments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, allocatable_count, sizeof(*state->assignments),
      (void**)&state->assignments));
  memset(state->assignments, 0,
         allocatable_count * sizeof(*state->assignments));
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
  if (lhs->location_kind != rhs->location_kind) return false;
  if (!loom_low_allocation_value_class_equal(lhs->value_class,
                                             rhs->value_class)) {
    return false;
  }
  return loom_low_allocation_interval_overlaps(lhs, rhs) &&
         loom_low_allocation_location_ranges_overlap(lhs, rhs);
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

iree_status_t loom_low_allocation_verify_sidecar(
    const loom_low_allocation_sidecar_t* sidecar) {
  if (!sidecar) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation sidecar is required");
  }
  if (sidecar->assignment_count > 0 && !sidecar->assignments) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation sidecar assignments are required");
  }
  if (sidecar->remark_count > 0 && !sidecar->remarks) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation sidecar remarks are required");
  }
  for (iree_host_size_t i = 0; i < sidecar->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* lhs = &sidecar->assignments[i];
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_assignment(lhs, i));
    for (iree_host_size_t j = i + 1; j < sidecar->assignment_count; ++j) {
      const loom_low_allocation_assignment_t* rhs = &sidecar->assignments[j];
      if (!loom_low_allocation_assignment_pair_conflicts(lhs, rhs)) continue;
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "allocation assigns overlapping live intervals to the same location");
    }
  }
  for (iree_host_size_t i = 0; i < sidecar->remark_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_remark(
        &sidecar->remarks[i], i, sidecar->assignment_count));
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
  if (options->budget_count > 0 && !options->budgets) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low allocation budgets are required when budget_count is non-zero");
  }
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

  IREE_RETURN_IF_ERROR(
      loom_liveness_analyze_region(module, state.body, arena, &state.liveness));
  IREE_RETURN_IF_ERROR(loom_low_allocation_assign_intervals(&state));

  loom_low_allocation_sidecar_t sidecar = {
      .module = module,
      .function_op = low_func_op,
      .target = state.target,
      .liveness = state.liveness,
      .allocation_mode = loom_low_func_def_allocation(low_func_op),
      .assignments = state.assignments,
      .assignment_count = state.assignment_count,
      .remarks = state.remarks,
      .remark_count = state.remark_count,
      .spill_count = state.spill_count,
  };
  IREE_RETURN_IF_ERROR(loom_low_allocation_verify_sidecar(&sidecar));
  *out_sidecar = sidecar;
  return iree_ok_status();
}
