// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation.h"

#include <inttypes.h>
#include <string.h>

#include "loom/analysis/consumption.h"
#include "loom/codegen/low/allocation/active_set.h"
#include "loom/codegen/low/allocation/storage_lease.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/allocation/unit_liveness.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/error/error_catalog.h"
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
  // Active function-local value domain shared with liveness/allocation.
  const loom_local_value_domain_t* value_domain;
  // Mutable per-allocation-unit live end points.
  loom_low_allocation_unit_liveness_t unit_liveness;
  // Mutable assignment records being built.
  loom_low_allocation_assignment_t* assignments;
  // Assignment indices by liveness local value ordinal. Missing entries contain
  // UINT32_MAX.
  uint32_t* assignment_indices_by_value_ordinal;
  // Assignment-index window still live at the current interval start.
  loom_low_allocation_active_set_t active;
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
  // Mutable assignment-backed storage leases and release actions being built.
  loom_low_allocation_storage_lease_state_t storage_leases;
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
  return loom_low_allocation_unit_liveness_end_point_start_for_value_ordinal(
      &state->unit_liveness, &state->liveness, value_ordinal);
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
      .end_point =
          loom_low_allocation_live_range_interval_storage_end_point(interval),
      .unit_count = interval->unit_count,
      .location_kind = location_kind,
      .location_base = location_base,
      .location_count = location_count,
      .unit_end_point_start =
          loom_low_allocation_unit_end_point_start_for_value(
              state, interval->value_id),
  };
  if (loom_low_allocation_active_set_conflicts(
          &state->active, state->target.descriptor_set, &state->liveness,
          state->unit_liveness.end_points, state->unit_liveness.end_point_count,
          state->assignments, state->assignment_count, &candidate,
          ignored_value_ids, ignored_value_count)) {
    return true;
  }
  if (loom_low_allocation_target_constraints_fixed_value_conflicts(
          &state->target_constraints, &state->liveness, &state->unit_liveness,
          &candidate, ignored_value_ids, ignored_value_count)) {
    return true;
  }
  if (loom_low_allocation_target_constraints_reserved_range_conflicts(
          &state->target_constraints, reg_class_id, location_kind,
          location_base, location_count)) {
    return true;
  }
  if (loom_low_allocation_storage_lease_state_conflicts(
          &state->storage_leases, state->target.descriptor_set,
          &state->liveness, &candidate, release_policy)) {
    return true;
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

static bool loom_low_allocation_find_location(
    loom_low_allocation_build_state_t* state,
    const loom_liveness_interval_t* interval,
    loom_low_allocation_class_capacity_t capacity, uint32_t* out_base) {
  if (capacity.is_bounded && interval->unit_count > capacity.max_units) {
    return false;
  }

  const uint32_t alignment =
      loom_low_allocation_live_range_interval_alignment(interval);
  uint32_t last_base = 0;
  if (capacity.is_bounded) {
    last_base = capacity.max_units - interval->unit_count;
  } else {
    const uint32_t search_limit =
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            &state->target_constraints, capacity.descriptor_reg_class_id,
            capacity.location_kind);
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

static bool loom_low_allocation_assignment_can_spill(
    loom_low_allocation_build_state_t* state,
    const loom_low_allocation_assignment_t* assignment,
    loom_low_allocation_class_capacity_t* out_capacity) {
  if (!loom_low_allocation_assignment_is_register_like(assignment)) {
    return false;
  }
  if (loom_low_allocation_target_constraints_fixed_value_for_value(
          &state->target_constraints, assignment->value_id)) {
    return false;
  }
  if (loom_low_allocation_value_requires_register_location(
          state, assignment->value_id)) {
    return false;
  }
  loom_low_allocation_class_capacity_t capacity = {0};
  if (!iree_status_is_ok(
          loom_low_allocation_target_constraints_reg_class_capacity(
              &state->target_constraints, assignment->descriptor_reg_class_id,
              &capacity))) {
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
  loom_low_allocation_active_set_remove_assignment_units(
      &state->active, state->assignments, state->assignment_count,
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
      loom_low_allocation_live_range_interval_storage_end_point(interval);

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
            state->target.descriptor_set, &state->liveness,
            state->unit_liveness.end_points,
            state->unit_liveness.end_point_count, assignment, &candidate,
            /*ignored_value_ids=*/NULL,
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
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            &state->target_constraints, capacity->descriptor_reg_class_id,
            capacity->location_kind);
    if (!loom_low_allocation_align_up_u32(
            search_limit,
            loom_low_allocation_live_range_interval_alignment(interval),
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

  const uint32_t alignment =
      loom_low_allocation_live_range_interval_alignment(interval);
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_resolve_reg_class(
      &state->target_constraints, interval->value_class, &interval_reg_class_id,
      NULL));
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
      .end_point =
          loom_low_allocation_live_range_interval_storage_end_point(interval),
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
      loom_low_allocation_target_constraints_fixed_value_for_value(
          &state->target_constraints, interval->value_id);
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
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_target_constraints_reg_class_capacity(
          &state->target_constraints, descriptor_reg_class_id, &capacity));
  if (!loom_low_allocation_target_constraints_location_range_fits_capacity(
          &capacity, location_kind, location_base, unit_count)) {
    return iree_ok_status();
  }
  const uint32_t alignment =
      loom_low_allocation_live_range_interval_alignment(interval);
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
      .end_point =
          loom_low_allocation_live_range_interval_storage_end_point(interval),
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_resolve_reg_class(
      &state->target_constraints, interval->value_class, &interval_reg_class_id,
      NULL));
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_resolve_reg_class(
      &state->target_constraints, interval->value_class, &interval_reg_class_id,
      NULL));
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_resolve_reg_class(
      &state->target_constraints, interval->value_class, &interval_reg_class_id,
      NULL));
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_resolve_reg_class(
      &state->target_constraints, interval->value_class, &interval_reg_class_id,
      NULL));
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
      loom_low_allocation_live_range_interval_alignment(result_interval);
  const uint32_t source_alignment =
      loom_low_allocation_live_range_interval_alignment(source_interval);
  const uint32_t assigned_limit =
      loom_low_allocation_target_constraints_assigned_location_search_limit(
          &state->target_constraints, capacity.descriptor_reg_class_id,
          capacity.location_kind);

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
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_class_capacity(
      &state->target_constraints, result_interval->value_class, &capacity));

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
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_resolve_reg_class(
            &state->target_constraints, interval->value_class,
            &interval_reg_class_id, NULL));
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
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_target_constraints_resolve_reg_class(
            &state->target_constraints, interval->value_class,
            &interval_reg_class_id, NULL));
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
          loom_low_allocation_target_constraints_resolve_reg_class(
              &state->target_constraints, interval->value_class,
              &interval_reg_class_id, NULL));
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
  return loom_low_allocation_live_range_ordered_op_program_point(
      &state->liveness, state->body, state->options->liveness_order, op,
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
                      state->unit_liveness.end_points,
                      state->unit_liveness.end_point_count, assignment,
                      unit_offset)) {
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_class_capacity(
      &state->target_constraints, storage_class->value_class, &capacity));
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
    last_location =
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            &state->target_constraints, storage_class->descriptor_reg_class_id,
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
    if (loom_low_allocation_target_constraints_reserved_range_conflicts(
            &state->target_constraints, temporary.descriptor_reg_class_id,
            temporary.location_kind, temporary.location, 1) ||
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_class_capacity(
      &state->target_constraints, storage_class->value_class, &capacity));
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
    last_location =
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            &state->target_constraints, storage_class->descriptor_reg_class_id,
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
    if (loom_low_allocation_target_constraints_reserved_range_conflicts(
            &state->target_constraints, temporary.descriptor_reg_class_id,
            temporary.location_kind, temporary.location, 1) ||
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
    if (loom_low_allocation_live_range_interval_is_allocatable(
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_active_set_initialize(
      allocatable_count, allocatable_unit_count, state->arena, &state->active));
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
    if (loom_low_allocation_live_range_interval_is_allocatable(interval)) {
      intervals[interval_index++] = interval;
    }
  }
  loom_low_allocation_sort_intervals(intervals, allocatable_count);

  for (iree_host_size_t i = 0; i < allocatable_count; ++i) {
    const loom_liveness_interval_t* interval = intervals[i];
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
    IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_class_capacity(
        &state->target_constraints, interval->value_class, &capacity));
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_initialize(
      module, low_func_op, &state.target, options->budgets,
      options->budget_count, options->reserved_ranges,
      options->reserved_range_count, options->emitter, arena,
      &state.target_constraints));

  loom_local_value_domain_t value_domain = {0};
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
        .unit_end_points = state.unit_liveness.end_points,
        .unit_end_point_count = state.unit_liveness.end_point_count,
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
        .storage_lease_instances = state.storage_leases.instances,
        .storage_lease_instance_count = state.storage_leases.instance_count,
        .storage_release_actions = state.storage_leases.release_actions,
        .storage_release_action_count =
            state.storage_leases.release_action_count,
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
