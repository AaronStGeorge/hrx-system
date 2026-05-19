// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation.h"

#include <string.h>

#include "loom/analysis/consumption.h"
#include "loom/codegen/low/allocation/active_set.h"
#include "loom/codegen/low/allocation/assignment_map.h"
#include "loom/codegen/low/allocation/coalescing.h"
#include "loom/codegen/low/allocation/copy_decision.h"
#include "loom/codegen/low/allocation/edge_copy.h"
#include "loom/codegen/low/allocation/interval_order.h"
#include "loom/codegen/low/allocation/packet_move.h"
#include "loom/codegen/low/allocation/search.h"
#include "loom/codegen/low/allocation/spill_plan.h"
#include "loom/codegen/low/allocation/spill_traffic.h"
#include "loom/codegen/low/allocation/storage_lease.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/allocation/unit_location.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

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
  // Resolved target storage budgets, fixed values, and reserved ranges.
  loom_low_allocation_target_constraints_t target_constraints;
  // Liveness analysis for |body|.
  loom_liveness_analysis_t liveness;
  // Function-local placement relations over |liveness|.
  loom_low_placement_table_t placement;
  // Reusable consumed-value query for |body|.
  loom_consumption_region_query_t consumption_query;
  // True once |consumption_query| has been initialized.
  bool consumption_query_initialized;
  // Mutable per-allocation-unit live end points.
  loom_low_allocation_unit_liveness_t unit_liveness;
  // Mutable assignment records being built.
  loom_low_allocation_assignment_t* assignments;
  // Assignment indices by liveness local value ordinal. Missing entries contain
  // UINT32_MAX.
  uint32_t* assignment_indices_by_value_ordinal;
  // Lookup table over assignments and liveness-local value ordinals.
  loom_low_allocation_assignment_map_t assignment_map;
  // Assignment-index window still live at the current interval start.
  loom_low_allocation_active_set_t active;
  // Mutable spill materialization plan records being built.
  loom_low_allocation_spill_plan_t* spill_plans;
  // Mutable remark records being built.
  loom_low_allocation_remark_t* remarks;
  // Mutable low.copy decision plan being built.
  loom_low_allocation_copy_decision_plan_t copy_decision_plan;
  // Mutable branch edge-copy plan being built.
  loom_low_allocation_edge_copy_plan_t edge_copy_plan;
  // Mutable packet-local move scratch plan being built.
  loom_low_allocation_packet_move_plan_t packet_move_plan;
  // Mutable assignment-backed storage leases and release actions being built.
  loom_low_allocation_storage_lease_state_t storage_leases;
  // Number of initialized assignment records.
  iree_host_size_t assignment_count;
  // Number of initialized spill materialization plan records.
  iree_host_size_t spill_plan_count;
  // Number of initialized remark records.
  iree_host_size_t remark_count;
  // Number of spill-slot assignments.
  iree_host_size_t spill_count;
} loom_low_allocation_build_state_t;

static bool loom_low_allocation_value_ordinal_for_value(
    const loom_low_allocation_build_state_t* state, loom_value_id_t value_id,
    loom_value_ordinal_t* out_value_ordinal) {
  return loom_low_allocation_assignment_map_value_ordinal_for_value(
      &state->assignment_map, value_id, out_value_ordinal);
}

static uint32_t loom_low_allocation_unit_end_point_start_for_value_ordinal(
    const loom_low_allocation_build_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  return loom_low_allocation_unit_liveness_end_point_start_for_value_ordinal(
      &state->unit_liveness, &state->liveness, value_ordinal);
}

static loom_low_allocation_search_context_t loom_low_allocation_search_context(
    loom_low_allocation_build_state_t* state) {
  return (loom_low_allocation_search_context_t){
      .module = state->module,
      .descriptor_set = state->target.descriptor_set,
      .liveness = &state->liveness,
      .unit_liveness = &state->unit_liveness,
      .target_constraints = &state->target_constraints,
      .assignment_map = &state->assignment_map,
      .active_set = &state->active,
      .storage_leases = &state->storage_leases,
  };
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

static const loom_low_allocation_assignment_t*
loom_low_allocation_current_assignment_for_value_ordinal(
    const loom_low_allocation_build_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  return loom_low_allocation_assignment_map_assignment_for_value_ordinal(
      &state->assignment_map, value_ordinal, NULL);
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

static iree_status_t loom_low_allocation_consumption_query_callback(
    void* user_data, loom_consumption_region_query_t** out_query) {
  return loom_low_allocation_consumption_query(
      (loom_low_allocation_build_state_t*)user_data, out_query);
}

static iree_status_t loom_low_allocation_spill_active_assignment(
    loom_low_allocation_build_state_t* state, uint32_t assignment_index,
    const loom_low_allocation_class_capacity_t* capacity) {
  if (state->spill_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation table exceeds uint32_t range");
  }
  loom_low_allocation_active_set_remove_assignment_units(
      &state->active, state->assignments, state->assignment_count,
      assignment_index);
  loom_low_allocation_assignment_t* assignment =
      &state->assignments[assignment_index];
  assignment->location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT;
  assignment->location_base = (uint32_t)state->spill_count;
  assignment->location_count = assignment->unit_count;
  ++state->spill_count;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_record(
      state->module, state->body, assignment, assignment_index,
      capacity->alloc_unit_bits, capacity->spill_slot_space, state->spill_plans,
      &state->spill_plan_count));
  loom_low_allocation_spill_remark_record(
      state->remarks, &state->remark_count, assignment_index,
      capacity->is_bounded ? capacity->max_units : UINT32_MAX,
      assignment->unit_count);
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_spill_active_assignment_set(
    loom_low_allocation_build_state_t* state,
    const uint32_t* assignment_indices, uint16_t assignment_count) {
  loom_low_allocation_search_context_t search_context =
      loom_low_allocation_search_context(state);
  for (uint16_t i = 0; i < assignment_count; ++i) {
    const uint32_t assignment_index = assignment_indices[i];
    IREE_ASSERT_LT(assignment_index, state->assignment_count);
    const loom_low_allocation_assignment_t* assignment =
        &state->assignments[assignment_index];
    loom_low_allocation_class_capacity_t assignment_capacity = {0};
    bool can_spill = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_search_assignment_spill_capacity(
        &search_context, assignment, &can_spill, &assignment_capacity));
    if (!can_spill) {
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
        loom_low_allocation_target_constraints_validate_register_location_capacity(
            &state->target_constraints, assignment->descriptor_reg_class_id,
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
          state->unit_liveness.end_points, state->unit_liveness.end_point_count,
          &stored_assignment);
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_storage_lease_state_record_release_actions(
          &state->storage_leases, state->target.descriptor_set,
          &state->liveness, &stored_assignment));
  state->assignments[state->assignment_count++] = stored_assignment;
  state->assignment_map.assignment_count = state->assignment_count;
  state->assignment_indices_by_value_ordinal[value_ordinal] = assignment_index;
  loom_low_allocation_target_constraints_record_assignment_location_end(
      &state->target_constraints, &stored_assignment);
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_storage_lease_state_record_assignment(
          &state->storage_leases, state->target.descriptor_set,
          &state->liveness, &stored_assignment, assignment_index,
          value_ordinal));
  loom_low_allocation_active_set_insert(
      &state->active, state->target.descriptor_set, state->assignments,
      state->assignment_count, assignment_index);
  if (out_assignment_index) {
    *out_assignment_index = assignment_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_append_assignment_callback(
    void* user_data, const loom_low_allocation_assignment_t* assignment,
    uint32_t* out_assignment_index) {
  return loom_low_allocation_append_assignment(
      (loom_low_allocation_build_state_t*)user_data, assignment,
      out_assignment_index);
}

static iree_status_t loom_low_allocation_assign_fixed_interval(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  const loom_low_allocation_resolved_fixed_value_t* fixed_value =
      loom_low_allocation_target_constraints_fixed_value_for_value(
          &state->target_constraints, interval->value_id);
  if (!fixed_value) {
    return iree_ok_status();
  }
  loom_low_allocation_search_context_t search_context =
      loom_low_allocation_search_context(state);
  if (loom_low_allocation_search_location_conflicts(
          &search_context, interval, fixed_value->descriptor_reg_class_id,
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
      .end_point =
          loom_low_allocation_live_range_interval_storage_end_point(interval),
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

static iree_status_t loom_low_allocation_assign_intervals(
    loom_low_allocation_build_state_t* state) {
  loom_low_allocation_interval_order_t order = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_interval_order_build(
      &state->liveness, state->arena, &order));
  if (order.interval_count == 0) {
    return iree_ok_status();
  }

  state->assignments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, order.interval_count, sizeof(*state->assignments),
      (void**)&state->assignments));
  memset(state->assignments, 0,
         order.interval_count * sizeof(*state->assignments));
  state->assignment_indices_by_value_ordinal = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->liveness.value_count,
      sizeof(*state->assignment_indices_by_value_ordinal),
      (void**)&state->assignment_indices_by_value_ordinal));
  for (iree_host_size_t i = 0; i < state->liveness.value_count; ++i) {
    state->assignment_indices_by_value_ordinal[i] = UINT32_MAX;
  }
  state->assignment_map = (loom_low_allocation_assignment_map_t){
      .module = state->module,
      .liveness = &state->liveness,
      .assignments = state->assignments,
      .assignment_count = 0,
      .assignment_indices_by_value_ordinal =
          state->assignment_indices_by_value_ordinal,
  };
  IREE_RETURN_IF_ERROR(loom_low_allocation_active_set_initialize(
      order.interval_count, order.unit_count, state->arena, &state->active));
  state->spill_plans = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, order.interval_count, sizeof(*state->spill_plans),
      (void**)&state->spill_plans));
  memset(state->spill_plans, 0,
         order.interval_count * sizeof(*state->spill_plans));
  state->remarks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, order.interval_count, sizeof(*state->remarks),
      (void**)&state->remarks));
  memset(state->remarks, 0, order.interval_count * sizeof(*state->remarks));

  for (iree_host_size_t i = 0; i < order.interval_count; ++i) {
    const loom_liveness_interval_t* interval = order.intervals[i];
    loom_low_allocation_active_set_expire(&state->active, state->assignments,
                                          state->assignment_count,
                                          interval->start_point);

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

    loom_low_allocation_search_context_t search_context =
        loom_low_allocation_search_context(state);
    loom_low_allocation_coalescing_context_t coalescing_context = {
        .module = state->module,
        .arena = state->arena,
        .liveness = &state->liveness,
        .placement = &state->placement,
        .target_constraints = &state->target_constraints,
        .assignment_map = &state->assignment_map,
        .search_context = &search_context,
        .append_assignment = loom_low_allocation_append_assignment_callback,
        .consumption_query = loom_low_allocation_consumption_query_callback,
        .user_data = state,
    };

    bool assigned_tied_interval = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_coalescing_assign_tied_interval(
        &coalescing_context, interval, &assigned_tied_interval));
    if (assigned_tied_interval) {
      continue;
    }

    bool assigned_concat_source_interval = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_assign_concat_source_interval(
            &coalescing_context, interval, &assigned_concat_source_interval));
    if (assigned_concat_source_interval) {
      continue;
    }

    bool assigned_structural_interval = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_assign_structural_interval(
            &coalescing_context, interval, &assigned_structural_interval));
    if (assigned_structural_interval) {
      continue;
    }

    bool assigned_branch_source_interval = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_coalescing_assign_branch_source_interval(
            &coalescing_context, interval, &assigned_branch_source_interval));
    if (assigned_branch_source_interval) {
      continue;
    }

    loom_low_allocation_class_capacity_t capacity = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_class_capacity(
        &state->target_constraints, interval->value_class, &capacity));
    if (state->spill_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "allocation table exceeds uint32_t range");
    }

    uint32_t location_base = 0;
    bool assigned = loom_low_allocation_search_find_free_location(
        &search_context, interval, capacity, &location_base);
    const bool requires_register =
        loom_low_allocation_spill_traffic_interval_requires_register_location(
            state->module, interval);
    if (!assigned && (capacity.is_spillable || requires_register)) {
      loom_low_allocation_search_spill_victim_set_t victim_set = {0};
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_search_find_active_spill_victim_set(
              &search_context, interval, &capacity, requires_register,
              state->arena, &victim_set));
      if (victim_set.found) {
        IREE_RETURN_IF_ERROR(loom_low_allocation_spill_active_assignment_set(
            state, victim_set.assignment_indices, victim_set.assignment_count));
        location_base = victim_set.location_base;
        assigned = true;
      }
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
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_target_constraints_emit_failure(
                &state->target_constraints,
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
      IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_emit_failure(
          &state->target_constraints, state->function_op, interval->value_class,
          budget_units, interval->unit_count,
          IREE_SV("unspillable-register-exhausted")));
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
        .end_point =
            loom_low_allocation_live_range_interval_storage_end_point(interval),
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
      IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_record(
          state->module, state->body, &state->assignments[assignment_index],
          assignment_index, capacity.alloc_unit_bits, capacity.spill_slot_space,
          state->spill_plans, &state->spill_plan_count));
      loom_low_allocation_spill_remark_record(
          state->remarks, &state->remark_count, assignment_index,
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_initialize(
      module, low_func_op, &state.target, options->budgets,
      options->budget_count, options->reserved_ranges,
      options->reserved_range_count, options->emitter, arena,
      &state.target_constraints));

  loom_local_value_domain_t value_domain = {0};
  iree_status_t status = loom_local_value_domain_acquire_for_region(
      module, state.body, arena, &value_domain);
  if (iree_status_is_ok(status)) {
    status = loom_liveness_analyze_local_value_domain(
        &value_domain, options->liveness_order, arena, &state.liveness);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_placement_analyze_region(module, state.body,
                                               &value_domain, &state.liveness,
                                               arena, &state.placement);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_unit_liveness_initialize(
        module, state.body, &state.target, options->liveness_order,
        &state.liveness, arena, &state.unit_liveness);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_unit_liveness_extend_for_tied_results(
        &state.unit_liveness, &state.liveness, &state.placement);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_target_constraints_resolve_fixed_values(
        &state.target_constraints, &state.liveness, &value_domain,
        options->fixed_values, options->fixed_value_count, arena);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_storage_lease_state_initialize(
        &options->storage_leases, module, low_func_op, &state.liveness, arena,
        &state.storage_leases);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_allocation_assign_intervals(&state);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_low_allocation_storage_lease_state_finalize(&state.storage_leases);
  }
  if (iree_status_is_ok(status)) {
    const loom_low_allocation_copy_decision_context_t copy_decision_context = {
        .body = state.body,
        .descriptor_set = state.target.descriptor_set,
        .assignment_map = state.assignment_map,
    };
    status = loom_low_allocation_copy_decision_plan_build(
        &copy_decision_context, arena, &state.copy_decision_plan);
  }
  if (iree_status_is_ok(status)) {
    const loom_low_allocation_edge_copy_context_t edge_copy_context = {
        .module = state.module,
        .body = state.body,
        .descriptor_set = state.target.descriptor_set,
        .liveness_order = options->liveness_order,
        .target_constraints = &state.target_constraints,
        .unit_liveness = &state.unit_liveness,
        .assignment_map = state.assignment_map,
    };
    status = loom_low_allocation_edge_copy_plan_build(&edge_copy_context, arena,
                                                      &state.edge_copy_plan);
  }
  if (iree_status_is_ok(status)) {
    const loom_low_allocation_packet_move_context_t packet_move_context = {
        .module = state.module,
        .body = state.body,
        .descriptor_set = state.target.descriptor_set,
        .liveness_order = options->liveness_order,
        .target_constraints = &state.target_constraints,
        .unit_liveness = &state.unit_liveness,
        .assignment_map = state.assignment_map,
    };
    status = loom_low_allocation_packet_move_plan_build(
        &packet_move_context, arena, &state.packet_move_plan);
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
        .unit_end_points = state.unit_liveness.end_points,
        .unit_end_point_count = state.unit_liveness.end_point_count,
        .spill_plans = state.spill_plans,
        .spill_plan_count = state.spill_plan_count,
        .remarks = state.remarks,
        .remark_count = state.remark_count,
        .copy_decisions = state.copy_decision_plan.decisions,
        .copy_decision_count = state.copy_decision_plan.decision_count,
        .edge_copies = state.edge_copy_plan.copies,
        .edge_copy_count = state.edge_copy_plan.copy_count,
        .edge_copy_groups = state.edge_copy_plan.groups,
        .edge_copy_group_count = state.edge_copy_plan.group_count,
        .edge_copy_temporaries = state.edge_copy_plan.temporaries,
        .edge_copy_temporary_count = state.edge_copy_plan.temporary_count,
        .packet_move_temporary_groups = state.packet_move_plan.groups,
        .packet_move_temporary_group_count = state.packet_move_plan.group_count,
        .packet_move_temporaries = state.packet_move_plan.temporaries,
        .packet_move_temporary_count = state.packet_move_plan.temporary_count,
        .storage_leases = options->storage_leases,
        .storage_lease_instances = state.storage_leases.instances,
        .storage_lease_instance_count = state.storage_leases.instance_count,
        .storage_release_actions = state.storage_leases.release_actions,
        .storage_release_action_count =
            state.storage_leases.release_action_count,
        .spill_count = state.spill_count,
        .coalesced_copy_count = state.copy_decision_plan.coalesced_count,
        .materialized_copy_count = state.copy_decision_plan.materialized_count,
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
