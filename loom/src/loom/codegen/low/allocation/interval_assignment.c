// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/interval_assignment.h"

#include <string.h>

#include "loom/analysis/consumption.h"
#include "loom/codegen/low/allocation/active_set.h"
#include "loom/codegen/low/allocation/coalescing.h"
#include "loom/codegen/low/allocation/interval_order.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/search.h"
#include "loom/codegen/low/allocation/spill_plan.h"
#include "loom/codegen/low/allocation/spill_traffic.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/diagnostics.h"

typedef struct loom_low_allocation_interval_assignment_state_t {
  // Caller-provided facts and mutable owner state.
  const loom_low_allocation_interval_assignment_context_t* context;
  // Reusable consumed-value query for |context->body|.
  loom_consumption_region_query_t consumption_query;
  // True once |consumption_query| has been initialized.
  bool consumption_query_initialized;
  // Assignment-index window still live at the current interval start.
  loom_low_allocation_active_set_t active;
  // Mutable assignment, spill, remark, and lookup state being built.
  loom_low_allocation_interval_assignment_result_t result;
} loom_low_allocation_interval_assignment_state_t;

static bool loom_low_allocation_interval_assignment_value_ordinal_for_value(
    const loom_low_allocation_interval_assignment_state_t* state,
    loom_value_id_t value_id, loom_value_ordinal_t* out_value_ordinal) {
  return loom_low_allocation_assignment_map_value_ordinal_for_value(
      &state->result.assignment_map, value_id, out_value_ordinal);
}

static uint32_t
loom_low_allocation_interval_assignment_unit_end_point_start_for_value_ordinal(
    const loom_low_allocation_interval_assignment_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  return loom_low_allocation_unit_liveness_end_point_start_for_value_ordinal(
      state->context->unit_liveness, state->context->liveness, value_ordinal);
}

static loom_low_allocation_search_context_t
loom_low_allocation_interval_assignment_search_context(
    loom_low_allocation_interval_assignment_state_t* state) {
  return (loom_low_allocation_search_context_t){
      .module = state->context->module,
      .descriptor_set = state->context->target->descriptor_set,
      .liveness = state->context->liveness,
      .unit_liveness = state->context->unit_liveness,
      .target_constraints = state->context->target_constraints,
      .assignment_map = &state->result.assignment_map,
      .active_set = &state->active,
      .storage_leases = state->context->storage_leases,
  };
}

static const loom_low_allocation_assignment_t*
loom_low_allocation_interval_assignment_current_assignment_for_value_ordinal(
    const loom_low_allocation_interval_assignment_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  return loom_low_allocation_assignment_map_assignment_for_value_ordinal(
      &state->result.assignment_map, value_ordinal, NULL);
}

static iree_status_t
loom_low_allocation_interval_assignment_value_ordinal_for_interval(
    const loom_low_allocation_interval_assignment_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_value_ordinal_t* out_value_ordinal) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_interval_assignment_value_ordinal_for_value(
          state, interval->value_id, &value_ordinal)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation interval value %u is outside the local value domain",
        (unsigned)interval->value_id);
  }
  *out_value_ordinal = value_ordinal;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_interval_assignment_consumption_query(
    loom_low_allocation_interval_assignment_state_t* state,
    loom_consumption_region_query_t** out_query) {
  *out_query = NULL;
  if (!state->consumption_query_initialized) {
    IREE_RETURN_IF_ERROR(loom_consumption_region_query_initialize(
        state->context->module, state->context->body, state->context->arena,
        &state->consumption_query));
    state->consumption_query_initialized = true;
  }
  *out_query = &state->consumption_query;
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_interval_assignment_consumption_query_callback(
    void* user_data, loom_consumption_region_query_t** out_query) {
  return loom_low_allocation_interval_assignment_consumption_query(
      (loom_low_allocation_interval_assignment_state_t*)user_data, out_query);
}

static iree_status_t
loom_low_allocation_interval_assignment_spill_active_assignment(
    loom_low_allocation_interval_assignment_state_t* state,
    uint32_t assignment_index,
    const loom_low_allocation_class_capacity_t* capacity) {
  if (state->result.spill_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation table exceeds uint32_t range");
  }
  loom_low_allocation_active_set_remove_assignment_units(
      &state->active, state->result.assignments, state->result.assignment_count,
      assignment_index);
  loom_low_allocation_assignment_t* assignment =
      &state->result.assignments[assignment_index];
  assignment->location_kind = LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT;
  assignment->location_base = (uint32_t)state->result.spill_count;
  assignment->location_count = assignment->unit_count;
  ++state->result.spill_count;
  IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_record(
      state->context->module, state->context->body, assignment,
      assignment_index, capacity->alloc_unit_bits, capacity->spill_slot_space,
      state->result.spill_plans, &state->result.spill_plan_count));
  loom_low_allocation_spill_remark_record(
      state->result.remarks, &state->result.remark_count, assignment_index,
      capacity->is_bounded ? capacity->max_units : UINT32_MAX,
      assignment->unit_count);
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_interval_assignment_spill_active_assignment_set(
    loom_low_allocation_interval_assignment_state_t* state,
    const uint32_t* assignment_indices, uint16_t assignment_count) {
  loom_low_allocation_search_context_t search_context =
      loom_low_allocation_interval_assignment_search_context(state);
  for (uint16_t i = 0; i < assignment_count; ++i) {
    const uint32_t assignment_index = assignment_indices[i];
    const loom_low_allocation_assignment_t* assignment =
        &state->result.assignments[assignment_index];
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
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_interval_assignment_spill_active_assignment(
            state, assignment_index, &assignment_capacity));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_interval_assignment_append_assignment(
    loom_low_allocation_interval_assignment_state_t* state,
    const loom_low_allocation_assignment_t* assignment,
    uint32_t* out_assignment_index) {
  if (state->result.assignment_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation table exceeds uint32_t range");
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (!loom_low_allocation_interval_assignment_value_ordinal_for_value(
          state, assignment->value_id, &value_ordinal)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation saw assignment for value %u outside the analyzed "
        "liveness value range",
        (unsigned)assignment->value_id);
  }
  if (state->result.assignment_indices_by_value_ordinal[value_ordinal] !=
      UINT32_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low allocation saw duplicate assignment for value "
                            "%u",
                            (unsigned)assignment->value_id);
  }
  if (loom_low_allocation_location_kind_is_register_like(
          assignment->location_kind)) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_validate_register_location_capacity(
            state->context->target_constraints,
            assignment->descriptor_reg_class_id, assignment->location_kind,
            assignment->location_base, assignment->location_count,
            IREE_SV("assignment"), state->context->function_op));
  }
  const uint32_t assignment_index = (uint32_t)state->result.assignment_count;
  loom_low_allocation_assignment_t stored_assignment = *assignment;
  stored_assignment.unit_end_point_start =
      loom_low_allocation_interval_assignment_unit_end_point_start_for_value_ordinal(
          state, value_ordinal);
  stored_assignment.end_point =
      loom_low_allocation_live_range_assignment_max_unit_end_point(
          state->context->unit_liveness->end_points,
          state->context->unit_liveness->end_point_count, &stored_assignment);
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_storage_lease_state_record_release_actions(
          state->context->storage_leases,
          state->context->target->descriptor_set, state->context->liveness,
          &stored_assignment));
  state->result.assignments[state->result.assignment_count++] =
      stored_assignment;
  state->result.assignment_map.assignment_count =
      state->result.assignment_count;
  state->result.assignment_indices_by_value_ordinal[value_ordinal] =
      assignment_index;
  loom_low_allocation_target_constraints_record_assignment_location_end(
      state->context->target_constraints, &stored_assignment);
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_storage_lease_state_record_assignment(
          state->context->storage_leases,
          state->context->target->descriptor_set, state->context->liveness,
          &stored_assignment, assignment_index, value_ordinal));
  loom_low_allocation_active_set_insert(
      &state->active, state->context->target->descriptor_set,
      state->result.assignments, state->result.assignment_count,
      assignment_index);
  if (out_assignment_index) {
    *out_assignment_index = assignment_index;
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_interval_assignment_append_assignment_callback(
    void* user_data, const loom_low_allocation_assignment_t* assignment,
    uint32_t* out_assignment_index) {
  return loom_low_allocation_interval_assignment_append_assignment(
      (loom_low_allocation_interval_assignment_state_t*)user_data, assignment,
      out_assignment_index);
}

static iree_status_t
loom_low_allocation_interval_assignment_assign_fixed_interval(
    loom_low_allocation_interval_assignment_state_t* state,
    const loom_liveness_interval_t* interval, bool* out_assigned) {
  *out_assigned = false;
  const loom_low_allocation_resolved_fixed_value_t* fixed_value =
      loom_low_allocation_target_constraints_fixed_value_for_value(
          state->context->target_constraints, interval->value_id);
  if (!fixed_value) {
    return iree_ok_status();
  }
  loom_low_allocation_search_context_t search_context =
      loom_low_allocation_interval_assignment_search_context(state);
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
      loom_low_allocation_interval_assignment_append_assignment(
          state, &assignment, NULL));
  *out_assigned = true;
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_interval_assignment_initialize_result_storage(
    loom_low_allocation_interval_assignment_state_t* state,
    const loom_low_allocation_interval_order_t* order) {
  if (state->context->liveness->value_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->context->arena, state->context->liveness->value_count,
        sizeof(*state->result.assignment_indices_by_value_ordinal),
        (void**)&state->result.assignment_indices_by_value_ordinal));
    for (iree_host_size_t i = 0; i < state->context->liveness->value_count;
         ++i) {
      state->result.assignment_indices_by_value_ordinal[i] = UINT32_MAX;
    }
  }
  if (order->interval_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->context->arena, order->interval_count,
                                  sizeof(*state->result.assignments),
                                  (void**)&state->result.assignments));
    memset(state->result.assignments, 0,
           order->interval_count * sizeof(*state->result.assignments));
    IREE_RETURN_IF_ERROR(loom_low_allocation_active_set_initialize(
        order->interval_count, order->unit_count, state->context->arena,
        &state->active));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->context->arena, order->interval_count,
                                  sizeof(*state->result.spill_plans),
                                  (void**)&state->result.spill_plans));
    memset(state->result.spill_plans, 0,
           order->interval_count * sizeof(*state->result.spill_plans));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->context->arena, order->interval_count,
        sizeof(*state->result.remarks), (void**)&state->result.remarks));
    memset(state->result.remarks, 0,
           order->interval_count * sizeof(*state->result.remarks));
  }

  state->result.assignment_map = (loom_low_allocation_assignment_map_t){
      .module = state->context->module,
      .liveness = state->context->liveness,
      .assignments = state->result.assignments,
      .assignment_count = 0,
      .assignment_indices_by_value_ordinal =
          state->result.assignment_indices_by_value_ordinal,
  };
  return iree_ok_status();
}

iree_status_t loom_low_allocation_interval_assignment_build(
    const loom_low_allocation_interval_assignment_context_t* context,
    loom_low_allocation_interval_assignment_result_t* out_result) {
  *out_result = (loom_low_allocation_interval_assignment_result_t){0};
  loom_low_allocation_interval_assignment_state_t state = {
      .context = context,
  };

  loom_low_allocation_interval_order_t order = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_interval_order_build(
      context->liveness, context->arena, &order));
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_interval_assignment_initialize_result_storage(
          &state, &order));
  if (order.interval_count == 0) {
    *out_result = state.result;
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < order.interval_count; ++i) {
    const loom_liveness_interval_t* interval = order.intervals[i];
    loom_low_allocation_active_set_expire(
        &state.active, state.result.assignments, state.result.assignment_count,
        interval->start_point);

    loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_interval_assignment_value_ordinal_for_interval(
            &state, interval, &value_ordinal));
    if (loom_low_allocation_interval_assignment_current_assignment_for_value_ordinal(
            &state, value_ordinal)) {
      continue;
    }

    bool assigned_fixed_interval = false;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_interval_assignment_assign_fixed_interval(
            &state, interval, &assigned_fixed_interval));
    if (assigned_fixed_interval) {
      continue;
    }

    loom_low_allocation_search_context_t search_context =
        loom_low_allocation_interval_assignment_search_context(&state);
    loom_low_allocation_coalescing_context_t coalescing_context = {
        .module = context->module,
        .arena = context->arena,
        .liveness = context->liveness,
        .placement = context->placement,
        .target_constraints = context->target_constraints,
        .assignment_map = &state.result.assignment_map,
        .search_context = &search_context,
        .append_assignment =
            loom_low_allocation_interval_assignment_append_assignment_callback,
        .consumption_query =
            loom_low_allocation_interval_assignment_consumption_query_callback,
        .user_data = &state,
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
        context->target_constraints, interval->value_class, &capacity));
    if (state.result.spill_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "allocation table exceeds uint32_t range");
    }

    uint32_t location_base = 0;
    bool assigned = loom_low_allocation_search_find_free_location(
        &search_context, interval, capacity, &location_base);
    const bool requires_register =
        loom_low_allocation_spill_traffic_interval_requires_register_location(
            context->module, interval);
    if (!assigned && (capacity.is_spillable || requires_register)) {
      loom_low_allocation_search_spill_victim_set_t victim_set = {0};
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_search_find_active_spill_victim_set(
              &search_context, interval, &capacity, requires_register,
              context->arena, &victim_set));
      if (victim_set.found) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_interval_assignment_spill_active_assignment_set(
                &state, victim_set.assignment_indices,
                victim_set.assignment_count));
        location_base = victim_set.location_base;
        assigned = true;
      }
    }
    if (!assigned && (!capacity.is_spillable || requires_register)) {
      const loom_low_descriptor_set_t* descriptor_set =
          context->target->descriptor_set;
      const loom_low_reg_class_t* reg_class =
          &descriptor_set->reg_classes[capacity.descriptor_reg_class_id];
      iree_string_view_t reg_class_name = loom_low_descriptor_set_string(
          descriptor_set, reg_class->name_string_offset);
      const uint32_t budget_units =
          capacity.is_bounded ? capacity.max_units : UINT32_MAX;
      if (requires_register) {
        iree_string_view_t value_name =
            loom_low_diagnostic_value_name(context->module, interval->value_id);
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_target_constraints_emit_failure(
                context->target_constraints,
                loom_low_diagnostic_value_origin_op(
                    context->module, interval->value_id, context->function_op),
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
          context->target_constraints, context->function_op,
          interval->value_class, budget_units, interval->unit_count,
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
            assigned ? location_base : (uint32_t)state.result.spill_count,
        .location_count = interval->unit_count,
    };

    uint32_t assignment_index = 0;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_interval_assignment_append_assignment(
            &state, &assignment, &assignment_index));
    if (!assigned) {
      ++state.result.spill_count;
      IREE_RETURN_IF_ERROR(loom_low_allocation_spill_plan_record(
          context->module, context->body,
          &state.result.assignments[assignment_index], assignment_index,
          capacity.alloc_unit_bits, capacity.spill_slot_space,
          state.result.spill_plans, &state.result.spill_plan_count));
      loom_low_allocation_spill_remark_record(
          state.result.remarks, &state.result.remark_count, assignment_index,
          capacity.is_bounded ? capacity.max_units : UINT32_MAX,
          interval->unit_count);
    }
  }

  *out_result = state.result;
  return iree_ok_status();
}
