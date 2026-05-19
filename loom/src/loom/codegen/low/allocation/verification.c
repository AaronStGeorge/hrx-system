// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/verification.h"

#include <inttypes.h>

#include "iree/base/alignment.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/move_topology.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/placement.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

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

static bool loom_low_allocation_assignment_units_share_location(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs, uint32_t lhs_unit_offset,
    uint32_t rhs_unit_offset) {
  if (lhs_unit_offset >= lhs->location_count ||
      rhs_unit_offset >= rhs->location_count) {
    return false;
  }
  if (!loom_low_allocation_storage_assignment_classes_share(
          table->target.descriptor_set, lhs, rhs)) {
    return false;
  }
  const uint64_t lhs_location = (uint64_t)lhs->location_base + lhs_unit_offset;
  const uint64_t rhs_location = (uint64_t)rhs->location_base + rhs_unit_offset;
  return lhs_location == rhs_location;
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

static iree_status_t loom_low_allocation_pair_is_placement_alias(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs, uint32_t lhs_unit_offset,
    uint32_t rhs_unit_offset, iree_host_size_t remaining_depth,
    bool* out_is_alias) {
  *out_is_alias =
      lhs->value_id == rhs->value_id && lhs_unit_offset == rhs_unit_offset;
  if (*out_is_alias || remaining_depth == 0) {
    return iree_ok_status();
  }

  loom_value_ordinal_t lhs_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_value_ordinal(table, lhs, &lhs_ordinal));

  const loom_low_placement_relation_range_t result_range =
      loom_low_placement_relation_range_for_value_ordinal(&table->placement,
                                                          lhs_ordinal);
  for (uint32_t i = 0; i < result_range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &table->placement.relations[result_range.start + i];
    if (!loom_low_placement_cause_can_alias(relation->cause)) {
      continue;
    }
    const loom_value_id_t source_value_id = loom_low_placement_value_id(
        &table->placement, relation->source_ordinal);

    if (relation->result_unit_offset > lhs_unit_offset) {
      continue;
    }
    const uint32_t result_unit_delta =
        lhs_unit_offset - relation->result_unit_offset;
    if (result_unit_delta >= relation->unit_count ||
        result_unit_delta > UINT32_MAX - relation->source_unit_offset) {
      continue;
    }
    const uint32_t next_unit_offset =
        relation->source_unit_offset + result_unit_delta;

    uint32_t next_assignment_index = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_map_assignment_index(
        table, source_value_id, &next_assignment_index));
    const loom_low_allocation_assignment_t* next_assignment =
        &table->assignments[next_assignment_index];
    if (!loom_low_allocation_assignment_units_share_location(
            table, lhs, next_assignment, lhs_unit_offset, next_unit_offset)) {
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_low_allocation_pair_is_placement_alias(
        table, next_assignment, rhs, next_unit_offset, rhs_unit_offset,
        remaining_depth - 1, out_is_alias));
    if (*out_is_alias) {
      return iree_ok_status();
    }
  }

  const loom_low_placement_relation_range_t source_range =
      loom_low_placement_relation_range_for_source_value_ordinal(
          &table->placement, lhs_ordinal);
  for (uint32_t i = 0; i < source_range.count; ++i) {
    const uint32_t relation_index =
        table->placement
            .relation_indices_by_source_ordinal[source_range.start + i];
    IREE_ASSERT_LT(relation_index, table->placement.relation_count);
    const loom_low_placement_relation_t* relation =
        &table->placement.relations[relation_index];
    if (!loom_low_placement_cause_can_alias(relation->cause)) {
      continue;
    }
    const loom_value_id_t result_value_id = loom_low_placement_value_id(
        &table->placement, relation->result_ordinal);

    if (relation->source_unit_offset > lhs_unit_offset) {
      continue;
    }
    const uint32_t source_unit_delta =
        lhs_unit_offset - relation->source_unit_offset;
    if (source_unit_delta >= relation->unit_count ||
        source_unit_delta > UINT32_MAX - relation->result_unit_offset) {
      continue;
    }
    const uint32_t next_unit_offset =
        relation->result_unit_offset + source_unit_delta;

    uint32_t next_assignment_index = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_map_assignment_index(
        table, result_value_id, &next_assignment_index));
    const loom_low_allocation_assignment_t* next_assignment =
        &table->assignments[next_assignment_index];
    if (!loom_low_allocation_assignment_units_share_location(
            table, lhs, next_assignment, lhs_unit_offset, next_unit_offset)) {
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_low_allocation_pair_is_placement_alias(
        table, next_assignment, rhs, next_unit_offset, rhs_unit_offset,
        remaining_depth - 1, out_is_alias));
    if (*out_is_alias) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_assignments_are_placement_aliases(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* lhs,
    const loom_low_allocation_assignment_t* rhs, bool* out_are_aliases) {
  *out_are_aliases = false;
  const uint64_t lhs_end = (uint64_t)lhs->location_base + lhs->location_count;
  const uint64_t rhs_end = (uint64_t)rhs->location_base + rhs->location_count;
  const uint64_t overlap_begin = lhs->location_base > rhs->location_base
                                     ? lhs->location_base
                                     : rhs->location_base;
  const uint64_t overlap_end = lhs_end < rhs_end ? lhs_end : rhs_end;
  if (overlap_begin >= overlap_end) {
    return iree_ok_status();
  }
  const iree_host_size_t max_depth = table->placement.relation_count;
  for (uint64_t location = overlap_begin; location < overlap_end; ++location) {
    bool unit_is_alias = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_pair_is_placement_alias(
        table, lhs, rhs, (uint32_t)(location - lhs->location_base),
        (uint32_t)(location - rhs->location_base), max_depth, &unit_is_alias));
    if (!unit_is_alias) {
      return iree_ok_status();
    }
  }
  *out_are_aliases = true;
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
    const bool result_spilled =
        loom_low_allocation_assignment_is_spill_slot(result_assignment);
    const bool operand_spilled =
        loom_low_allocation_assignment_is_spill_slot(operand_assignment);
    if (result_spilled || operand_spilled) {
      continue;
    }
    if (!loom_low_allocation_assignment_is_register_like(result_assignment) ||
        !loom_low_allocation_assignment_is_register_like(operand_assignment)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "allocation tied result has a non-register-like non-spill location");
    }
    if (!loom_low_allocation_storage_assignment_locations_share(
            table->target.descriptor_set, result_assignment,
            operand_assignment)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "allocation tied result does not share the operand location");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_register_location_capacity(
    const loom_low_allocation_table_t* table, uint16_t reg_class_id,
    loom_low_allocation_location_kind_t location_kind, uint32_t location_base,
    uint32_t location_count, iree_string_view_t subject,
    iree_host_size_t subject_index) {
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      reg_class_id >= table->target.descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation %.*s %zu has invalid descriptor register class ID %" PRIu16,
        (int)subject.size, subject.data, subject_index, reg_class_id);
  }
  const loom_low_reg_class_t* reg_class =
      &table->target.descriptor_set->reg_classes[reg_class_id];
  const loom_low_allocation_location_kind_t expected_location_kind =
      loom_low_allocation_storage_reg_class_location_kind(reg_class);
  if (location_kind != expected_location_kind) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation %.*s %zu location kind %u does not match register-class "
        "location kind %u",
        (int)subject.size, subject.data, subject_index, (unsigned)location_kind,
        (unsigned)expected_location_kind);
  }
  if (reg_class->allocatable_count == 0) {
    return iree_ok_status();
  }
  if ((uint64_t)location_base + location_count > reg_class->allocatable_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation %.*s %zu location range exceeds register-class "
        "allocatable count %" PRIu16,
        (int)subject.size, subject.data, subject_index,
        reg_class->allocatable_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_assignment(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_assignment_t* assignment,
    iree_host_size_t assignment_index) {
  if (!loom_low_allocation_location_kind_is_known(assignment->location_kind)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation assignment %zu has unknown location "
                            "kind %u",
                            assignment_index,
                            (unsigned)assignment->location_kind);
  }
  if (assignment->location_kind != LOOM_LOW_ALLOCATION_LOCATION_UNASSIGNED &&
      assignment->location_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
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
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation assignment %zu has invalid descriptor register class ID "
        "%" PRIu16,
        assignment_index, assignment->descriptor_reg_class_id);
  }
  if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation assignment %zu has a non-register value class",
        assignment_index);
  }
  if (assignment->value_class.register_descriptor_set_stable_id !=
          table->target.descriptor_set->stable_id ||
      assignment->value_class.register_class_id !=
          assignment->descriptor_reg_class_id) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation assignment %zu descriptor register class %" PRIu16
        " does not match value class descriptor set %" PRIu64
        " register class %" PRIu16,
        assignment_index, assignment->descriptor_reg_class_id,
        assignment->value_class.register_descriptor_set_stable_id,
        assignment->value_class.register_class_id);
  }
  if (loom_low_allocation_location_kind_is_register_like(
          assignment->location_kind)) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_register_location_capacity(
        table, assignment->descriptor_reg_class_id, assignment->location_kind,
        assignment->location_base, assignment->location_count,
        IREE_SV("assignment"), assignment_index));
  }
  if (assignment->unit_count > assignment->location_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation assignment %zu requires more units than it locates",
        assignment_index);
  }
  if (assignment->unit_end_point_start > table->unit_end_point_count ||
      assignment->unit_count >
          table->unit_end_point_count - assignment->unit_end_point_start) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation assignment %zu unit end-point range is outside the table",
        assignment_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_remark(
    const loom_low_allocation_remark_t* remark, iree_host_size_t remark_index,
    iree_host_size_t assignment_count) {
  if (!loom_low_allocation_remark_kind_is_known(remark->kind)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation remark %zu has unknown kind %u",
                            remark_index, (unsigned)remark->kind);
  }
  if (remark->kind == LOOM_LOW_ALLOCATION_REMARK_UNKNOWN) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
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
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan %zu has unknown spill slot space %u",
        spill_plan_index, (unsigned)spill_plan->slot_space);
  }
  if (spill_plan->byte_size == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation spill plan %zu has empty byte size",
                            spill_plan_index);
  }
  if (!iree_is_power_of_two_uint64(spill_plan->byte_alignment)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
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
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation copy decision %zu has unknown kind %u",
                            copy_decision_index, (unsigned)copy_decision->kind);
  }
  if (copy_decision->kind == LOOM_LOW_ALLOCATION_COPY_UNKNOWN) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
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
  const bool locations_equal =
      loom_low_allocation_storage_assignment_locations_share(
          table->target.descriptor_set, source_assignment, result_assignment);
  if (copy_decision->kind == LOOM_LOW_ALLOCATION_COPY_COALESCED &&
      (!loom_low_allocation_assignment_is_register_like(source_assignment) ||
       !loom_low_allocation_assignment_is_register_like(result_assignment) ||
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
  if (edge_copy->unit_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation edge-copy %zu has an empty unit range",
                            edge_copy_index);
  }
  if (edge_copy->source_unit_offset > source_assignment->location_count ||
      edge_copy->unit_count >
          source_assignment->location_count - edge_copy->source_unit_offset) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy %zu source segment is outside its assignment",
        edge_copy_index);
  }
  if (edge_copy->destination_unit_offset >
          destination_assignment->location_count ||
      edge_copy->unit_count > destination_assignment->location_count -
                                  edge_copy->destination_unit_offset) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy %zu destination segment is outside its "
        "assignment",
        edge_copy_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_edge_copy_matches_payload(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_edge_copy_t* edge_copy,
    iree_host_size_t edge_copy_index, uint16_t expected_payload_index,
    loom_value_id_t expected_source_value_id,
    loom_value_id_t expected_destination_value_id,
    uint32_t expected_source_unit_offset,
    uint32_t expected_destination_unit_offset, uint32_t expected_unit_count) {
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_verify_edge_copy(table, edge_copy, edge_copy_index));
  if (edge_copy->payload_index != expected_payload_index ||
      edge_copy->source_value_id != expected_source_value_id ||
      edge_copy->destination_value_id != expected_destination_value_id ||
      edge_copy->source_unit_offset != expected_source_unit_offset ||
      edge_copy->destination_unit_offset != expected_destination_unit_offset ||
      edge_copy->unit_count != expected_unit_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy %zu does not match low.br payload segment",
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_live_range_op_program_point(
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
  if (args.count != dest->arg_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy group %zu does not match low.br edge payload",
        group_index);
  }
  iree_host_size_t edge_copy_index = group->copy_start;
  for (uint16_t payload_index = 0; payload_index < dest->arg_count;
       ++payload_index) {
    const loom_value_id_t payload_value_id = args.values[payload_index];
    const loom_value_id_t destination_value_id = dest->arg_ids[payload_index];
    const loom_op_t* concat_op =
        loom_low_allocation_move_topology_value_defining_concat(
            table->module, payload_value_id);
    if (!concat_op) {
      if (edge_copy_index >= group->copy_start + group->copy_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "allocation edge-copy group %zu ended before low.br payload %u",
            group_index, payload_index);
      }
      const loom_low_allocation_edge_copy_t* edge_copy =
          &table->edge_copies[edge_copy_index];
      uint32_t source_assignment_index = 0;
      IREE_RETURN_IF_ERROR(loom_low_allocation_map_assignment_index(
          table, payload_value_id, &source_assignment_index));
      uint32_t destination_assignment_index = 0;
      IREE_RETURN_IF_ERROR(loom_low_allocation_map_assignment_index(
          table, destination_value_id, &destination_assignment_index));
      const loom_low_allocation_assignment_t* source_assignment =
          &table->assignments[source_assignment_index];
      const loom_low_allocation_assignment_t* destination_assignment =
          &table->assignments[destination_assignment_index];
      IREE_RETURN_IF_ERROR(loom_low_allocation_verify_edge_copy_matches_payload(
          table, edge_copy, edge_copy_index, payload_index, payload_value_id,
          destination_value_id,
          /*expected_source_unit_offset=*/0,
          /*expected_destination_unit_offset=*/0,
          source_assignment->location_count));
      if (source_assignment->location_count !=
          destination_assignment->location_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "allocation edge-copy group %zu payload %u does not fill its "
            "destination",
            group_index, payload_index);
      }
      ++edge_copy_index;
      continue;
    }

    uint32_t destination_unit_offset = 0;
    loom_value_slice_t sources = loom_low_concat_sources(concat_op);
    for (uint16_t source_index = 0; source_index < sources.count;
         ++source_index) {
      if (edge_copy_index >= group->copy_start + group->copy_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "allocation edge-copy group %zu ended inside low.concat payload "
            "%u",
            group_index, payload_index);
      }
      const loom_low_allocation_edge_copy_t* edge_copy =
          &table->edge_copies[edge_copy_index];
      uint32_t source_assignment_index = 0;
      IREE_RETURN_IF_ERROR(loom_low_allocation_map_assignment_index(
          table, sources.values[source_index], &source_assignment_index));
      const loom_low_allocation_assignment_t* source_assignment =
          &table->assignments[source_assignment_index];
      IREE_RETURN_IF_ERROR(loom_low_allocation_verify_edge_copy_matches_payload(
          table, edge_copy, edge_copy_index, payload_index,
          sources.values[source_index], destination_value_id,
          /*expected_source_unit_offset=*/0, destination_unit_offset,
          source_assignment->location_count));
      if (source_assignment->location_count >
          UINT32_MAX - destination_unit_offset) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "allocation edge-copy group %zu destination offset exceeds u32 "
            "range",
            group_index);
      }
      destination_unit_offset += source_assignment->location_count;
      ++edge_copy_index;
    }
    uint32_t destination_assignment_index = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_map_assignment_index(
        table, destination_value_id, &destination_assignment_index));
    const loom_low_allocation_assignment_t* destination_assignment =
        &table->assignments[destination_assignment_index];
    if (destination_unit_offset != destination_assignment->location_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "allocation edge-copy group %zu payload %u does not fill its "
          "destination",
          group_index, payload_index);
    }
  }
  if (edge_copy_index != group->copy_start + group->copy_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation edge-copy group %zu has extra low.br payload segments",
        group_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_edge_copy_temporary(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_edge_copy_temporary_t* temporary,
    iree_host_size_t temporary_index) {
  if (!loom_low_allocation_location_kind_is_known(temporary->location_kind) ||
      !loom_low_allocation_location_kind_is_register_like(
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_verify_register_location_capacity(
      table, temporary->descriptor_reg_class_id, temporary->location_kind,
      temporary->location, 1, IREE_SV("edge-copy temporary"), temporary_index));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_packet_move_temporary_group(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_packet_move_temporary_group_t* group,
    iree_host_size_t group_index, iree_host_size_t expected_temporary_start) {
  if (!group->op ||
      !loom_low_allocation_move_topology_op_has_packet_moves(group->op)) {
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
      !loom_low_allocation_location_kind_is_register_like(
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_verify_register_location_capacity(
      table, temporary->descriptor_reg_class_id, temporary->location_kind,
      temporary->location, 1, IREE_SV("packet-local move temporary"),
      temporary_index));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_storage_lease_instance(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_storage_lease_t* instance,
    iree_host_size_t instance_index) {
  if (instance->lease_record_index >= table->storage_leases.record_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation storage lease instance %zu references lease record %u, but "
        "table has only %zu record(s)",
        instance_index, instance->lease_record_index,
        table->storage_leases.record_count);
  }
  if (instance->assignment_index >= table->assignment_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation storage lease instance %zu references assignment %u, but "
        "table has only %zu assignment(s)",
        instance_index, instance->assignment_index, table->assignment_count);
  }
  if (instance->end_point <= instance->start_point) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation storage lease instance %zu has an empty lifetime",
        instance_index);
  }
  if (instance->release_action_index !=
      LOOM_LOW_STORAGE_RELEASE_ACTION_INDEX_NONE) {
    if (instance->release_action_index >= table->storage_release_action_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "allocation storage lease instance %zu references release action %u "
          "but table has %zu release action(s)",
          instance_index, instance->release_action_index,
          table->storage_release_action_count);
    }
    const loom_low_storage_release_action_t* action =
        &table->storage_release_actions[instance->release_action_index];
    if (action->lease_record_index != instance->lease_record_index) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "allocation storage lease instance %zu release action covers lease "
          "record %u",
          instance_index, action->lease_record_index);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_storage_release_action(
    const loom_low_allocation_table_t* table,
    const loom_low_storage_release_action_t* action,
    iree_host_size_t action_index) {
  if (table->storage_leases.schedule == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation storage release action %zu has no schedule", action_index);
  }
  if (action->lease_record_index >= table->storage_leases.record_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation storage release action %zu references lease record %u but "
        "table has %zu lease record(s)",
        action_index, action->lease_record_index,
        table->storage_leases.record_count);
  }
  if (action->insertion_packet_index >=
      table->storage_leases.schedule->scheduled_node_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "allocation storage release action %zu insertion packet is out of "
        "range",
        action_index);
  }
  if (action->required_progress == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation storage release action %zu has no required progress",
        action_index);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_verify_storage_lease_instances(
    const loom_low_allocation_table_t* table) {
  if (table->storage_leases.record_count == 0) {
    if (table->storage_lease_instance_count != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "allocation table has storage lease instances without storage lease "
          "records");
    }
    if (table->storage_release_action_count != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "allocation table has storage release actions without storage lease "
          "records");
    }
    return iree_ok_status();
  }
  if (table->storage_leases.schedule == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation storage leases have no schedule");
  }
  if (table->storage_leases.records == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation storage leases have no records");
  }
  if (table->storage_leases.schedule->module != table->module ||
      table->storage_leases.schedule->function_op != table->function_op) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation storage leases do not describe the allocated function");
  }
  if (table->storage_lease_instance_count !=
      table->storage_leases.record_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation table has %zu storage lease instances for %zu storage "
        "lease records",
        table->storage_lease_instance_count,
        table->storage_leases.record_count);
  }
  if (table->storage_lease_instances == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation table has a storage lease instance count but no storage "
        "lease instances");
  }
  if (table->storage_release_action_count >
      table->storage_leases.record_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation table has more storage release actions than storage lease "
        "records");
  }
  if (table->storage_release_action_count != 0 &&
      table->storage_release_actions == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation table has a storage release action count but no storage "
        "release actions");
  }
  for (iree_host_size_t i = 0; i < table->storage_release_action_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_storage_release_action(
        table, &table->storage_release_actions[i], i));
  }
  for (iree_host_size_t i = 0; i < table->storage_lease_instance_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_storage_lease_instance(
        table, &table->storage_lease_instances[i], i));
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
        IREE_STATUS_FAILED_PRECONDITION,
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
      if (!loom_low_allocation_live_range_assignments_conflict(
              table->target.descriptor_set, &table->liveness,
              table->unit_end_points, table->unit_end_point_count, lhs, rhs)) {
        continue;
      }
      bool pair_are_placement_aliases = false;
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_assignments_are_placement_aliases(
              table, lhs, rhs, &pair_are_placement_aliases));
      if (pair_are_placement_aliases) {
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
  IREE_RETURN_IF_ERROR(loom_low_allocation_move_topology_count_edge_copy_groups(
      table->module, body, &edge_copy_group_count, &edge_copy_count));
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
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_verify_storage_lease_instances(table));
  iree_host_size_t expected_copy_decision_count =
      loom_low_allocation_move_topology_count_copy_ops(body);
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
