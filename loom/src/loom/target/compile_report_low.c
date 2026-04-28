// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/compile_report_low.h"

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/ops.h"

static iree_string_view_t loom_target_compile_report_module_string(
    const loom_module_t* module, loom_string_id_t string_id,
    iree_string_view_t fallback) {
  if (module == NULL || string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return fallback;
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_target_compile_report_value_name(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (module == NULL || value_id >= module->values.count) {
    return IREE_SV("<unknown>");
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_target_compile_report_module_string(module, value->name_id,
                                                  IREE_SV("<unnamed>"));
}

static iree_string_view_t loom_target_compile_report_block_name(
    const loom_module_t* module, const loom_block_t* block) {
  if (block == NULL) {
    return IREE_SV("<anonymous>");
  }
  return loom_target_compile_report_module_string(module, block->label_id,
                                                  IREE_SV("<anonymous>"));
}

static iree_string_view_t loom_target_compile_report_value_class_name(
    const loom_module_t* module, loom_liveness_value_class_t value_class) {
  return loom_target_compile_report_module_string(
      module, value_class.register_class_id, iree_string_view_empty());
}

static iree_string_view_t loom_target_compile_report_op_name(
    const loom_module_t* module, const loom_op_t* op) {
  if (module == NULL || op == NULL) {
    return IREE_SV("<unknown>");
  }
  return loom_op_name(module, op);
}

static void loom_target_compile_report_record_move_cause_if_nonzero(
    loom_target_compile_report_t* report,
    loom_target_compile_report_move_cause_t cause, uint64_t packet_count,
    uint64_t unit_count) {
  if (packet_count == 0 && unit_count == 0) {
    return;
  }
  loom_target_compile_report_record_move_cause(report, cause, packet_count,
                                               unit_count);
}

static uint64_t loom_target_compile_report_value_register_unit_count(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (module == NULL || value_id >= module->values.count) {
    return 0;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  return loom_type_is_register(type) ? loom_type_register_unit_count(type) : 0;
}

static const loom_low_allocation_assignment_t*
loom_target_compile_report_find_assignment(
    const loom_low_allocation_sidecar_t* allocation, loom_value_id_t value_id) {
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    if (allocation->assignments[i].value_id == value_id) {
      return &allocation->assignments[i];
    }
  }
  return NULL;
}

static bool loom_target_compile_report_slice_move_count(
    const loom_low_allocation_sidecar_t* allocation, const loom_op_t* op,
    uint64_t* out_unit_count) {
  *out_unit_count = 0;
  const int64_t offset = loom_low_slice_offset(op);
  if (offset < 0 || offset > UINT32_MAX) {
    IREE_ASSERT(false, "verified low.slice offset must fit in uint32_t");
    return false;
  }
  const loom_low_allocation_assignment_t* source_assignment =
      loom_target_compile_report_find_assignment(allocation,
                                                 loom_low_slice_source(op));
  const loom_low_allocation_assignment_t* result_assignment =
      loom_target_compile_report_find_assignment(allocation,
                                                 loom_low_slice_result(op));
  if (source_assignment == NULL || result_assignment == NULL) {
    IREE_ASSERT(false, "allocated low.slice values must have assignments");
    return false;
  }
  const uint32_t source_offset = (uint32_t)offset;
  if (source_offset > source_assignment->location_count ||
      result_assignment->location_count >
          source_assignment->location_count - source_offset) {
    IREE_ASSERT(false, "verified low.slice range must fit source assignment");
    return false;
  }
  if (!loom_low_allocation_assignments_share_storage(source_assignment,
                                                     result_assignment)) {
    IREE_ASSERT(false,
                "allocated low.slice values must share one storage class");
    return false;
  }
  for (uint32_t unit_index = 0; unit_index < result_assignment->location_count;
       ++unit_index) {
    if (!loom_low_allocation_assignment_subranges_match(
            result_assignment, unit_index, source_assignment,
            source_offset + unit_index, /*unit_count=*/1)) {
      ++*out_unit_count;
    }
  }
  return true;
}

static bool loom_target_compile_report_concat_move_count(
    const loom_low_allocation_sidecar_t* allocation, const loom_op_t* op,
    uint64_t* out_unit_count) {
  *out_unit_count = 0;
  const loom_low_allocation_assignment_t* result_assignment =
      loom_target_compile_report_find_assignment(allocation,
                                                 loom_low_concat_result(op));
  if (result_assignment == NULL) {
    IREE_ASSERT(false, "allocated low.concat result must have an assignment");
    return false;
  }
  uint32_t result_offset = 0;
  loom_value_slice_t sources = loom_low_concat_sources(op);
  for (uint16_t i = 0; i < sources.count; ++i) {
    const loom_low_allocation_assignment_t* source_assignment =
        loom_target_compile_report_find_assignment(allocation,
                                                   sources.values[i]);
    if (source_assignment == NULL ||
        !loom_low_allocation_assignments_share_storage(result_assignment,
                                                       source_assignment)) {
      IREE_ASSERT(false,
                  "allocated low.concat values must share one storage class");
      return false;
    }
    if (result_offset > result_assignment->location_count ||
        source_assignment->location_count >
            result_assignment->location_count - result_offset) {
      IREE_ASSERT(false, "verified low.concat range must fit result");
      return false;
    }
    for (uint32_t source_unit = 0;
         source_unit < source_assignment->location_count; ++source_unit) {
      if (!loom_low_allocation_assignment_subranges_match(
              result_assignment, result_offset + source_unit, source_assignment,
              source_unit, /*unit_count=*/1)) {
        ++*out_unit_count;
      }
    }
    result_offset += source_assignment->location_count;
  }
  if (result_offset != result_assignment->location_count) {
    IREE_ASSERT(false, "verified low.concat sources must fill result");
    return false;
  }
  return true;
}

static void loom_target_compile_report_record_low_copy_moves(
    loom_target_compile_report_t* report,
    const loom_low_allocation_sidecar_t* allocation) {
  uint64_t packet_count = 0;
  uint64_t unit_count = 0;
  for (iree_host_size_t i = 0; i < allocation->copy_decision_count; ++i) {
    const loom_low_allocation_copy_decision_t* decision =
        &allocation->copy_decisions[i];
    if (decision->kind != LOOM_LOW_ALLOCATION_COPY_MATERIALIZED) {
      continue;
    }
    ++packet_count;
    if (decision->result_assignment_index >= allocation->assignment_count) {
      IREE_ASSERT(false,
                  "verified copy decision result assignment must fit "
                  "allocation sidecar");
      continue;
    }
    unit_count += allocation->assignments[decision->result_assignment_index]
                      .location_count;
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY, packet_count,
      unit_count);
}

static void loom_target_compile_report_record_edge_copy_moves(
    loom_target_compile_report_t* report,
    const loom_low_allocation_sidecar_t* allocation) {
  uint64_t packet_count = 0;
  uint64_t unit_count = 0;
  for (iree_host_size_t i = 0; i < allocation->edge_copy_group_count; ++i) {
    const loom_low_allocation_edge_copy_group_t* group =
        &allocation->edge_copy_groups[i];
    if (group->copy_start > allocation->edge_copy_count ||
        group->copy_count > allocation->edge_copy_count - group->copy_start) {
      IREE_ASSERT(false,
                  "verified edge-copy group range must fit allocation sidecar");
      continue;
    }
    uint64_t group_unit_count = 0;
    for (uint32_t j = 0; j < group->copy_count; ++j) {
      const loom_low_allocation_edge_copy_t* edge_copy =
          &allocation->edge_copies[group->copy_start + j];
      if (edge_copy->source_assignment_index >= allocation->assignment_count ||
          edge_copy->destination_assignment_index >=
              allocation->assignment_count) {
        IREE_ASSERT(false,
                    "verified edge-copy assignments must fit allocation "
                    "sidecar");
        continue;
      }
      const loom_low_allocation_assignment_t* source_assignment =
          &allocation->assignments[edge_copy->source_assignment_index];
      const loom_low_allocation_assignment_t* destination_assignment =
          &allocation->assignments[edge_copy->destination_assignment_index];
      if (source_assignment->location_count !=
          destination_assignment->location_count) {
        IREE_ASSERT(false,
                    "verified edge-copy source and destination counts must "
                    "match");
        continue;
      }
      for (uint32_t unit_index = 0;
           unit_index < source_assignment->location_count; ++unit_index) {
        if (!loom_low_allocation_assignment_subranges_match(
                destination_assignment, unit_index, source_assignment,
                unit_index, /*unit_count=*/1)) {
          ++group_unit_count;
        }
      }
    }
    if (group_unit_count != 0) {
      ++packet_count;
      unit_count += group_unit_count;
    }
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_BRANCH_EDGE, packet_count,
      unit_count);
}

static void loom_target_compile_report_record_structural_packet_moves(
    loom_target_compile_report_t* report,
    const loom_low_packetization_t* packetization) {
  uint64_t constant_packet_count = 0;
  uint64_t constant_unit_count = 0;
  uint64_t slice_packet_count = 0;
  uint64_t slice_unit_count = 0;
  uint64_t concat_packet_count = 0;
  uint64_t concat_unit_count = 0;
  const loom_module_t* module = packetization->schedule.module;
  const loom_low_allocation_sidecar_t* allocation = &packetization->allocation;
  for (iree_host_size_t i = 0; i < packetization->schedule.node_count; ++i) {
    const loom_op_t* op = packetization->schedule.nodes[i].op;
    if (op == NULL) {
      continue;
    }
    if (loom_low_const_isa(op)) {
      ++constant_packet_count;
      constant_unit_count +=
          loom_target_compile_report_value_register_unit_count(
              module, loom_low_const_result(op));
    } else if (loom_low_slice_isa(op)) {
      uint64_t move_count = 0;
      if (loom_target_compile_report_slice_move_count(allocation, op,
                                                      &move_count) &&
          move_count != 0) {
        ++slice_packet_count;
        slice_unit_count += move_count;
      }
    } else if (loom_low_concat_isa(op)) {
      uint64_t move_count = 0;
      if (loom_target_compile_report_concat_move_count(allocation, op,
                                                       &move_count) &&
          move_count != 0) {
        ++concat_packet_count;
        concat_unit_count += move_count;
      }
    }
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_CONSTANT_MATERIALIZATION,
      constant_packet_count, constant_unit_count);
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_SLICE,
      slice_packet_count, slice_unit_count);
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_CONCAT,
      concat_packet_count, concat_unit_count);
}

static void loom_target_compile_report_record_move_causes(
    loom_target_compile_report_t* report,
    const loom_low_packetization_t* packetization) {
  loom_target_compile_report_record_low_copy_moves(report,
                                                   &packetization->allocation);
  loom_target_compile_report_record_edge_copy_moves(report,
                                                    &packetization->allocation);
  loom_target_compile_report_record_structural_packet_moves(report,
                                                            packetization);
}

static loom_target_compile_report_source_low_selection_kind_t
loom_target_compile_report_source_low_selection_kind(
    loom_low_lower_report_selection_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_LOWER_REPORT_SELECTION_RULE:
      return LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE;
    case LOOM_LOW_LOWER_REPORT_SELECTION_PLAN:
      return LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_PLAN;
    case LOOM_LOW_LOWER_REPORT_SELECTION_NONE:
    default:
      return LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_NONE;
  }
}

void loom_target_compile_report_record_low_lowering(
    loom_target_compile_report_t* report,
    const loom_low_lower_result_t* lower_result) {
  IREE_ASSERT_ARGUMENT(report);
  IREE_ASSERT_ARGUMENT(lower_result);
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  report->source_low_selected_op_count +=
      lower_result->selected_source_op_count;
  report->source_low_emitted_op_count += lower_result->emitted_low_op_count;
  for (iree_host_size_t i = 0; i < lower_result->report_row_count; ++i) {
    const loom_low_lower_report_row_t* source_row =
        &lower_result->report_rows[i];
    const loom_target_compile_report_source_low_row_t row = {
        .function_name = source_row->function_name,
        .source_op_name = source_row->source_op_name,
        .source_op_kind = source_row->source_op_kind,
        .selection_kind = loom_target_compile_report_source_low_selection_kind(
            source_row->selection_kind),
        .rule_set_index = source_row->rule_set_index,
        .rule_index = source_row->rule_index,
        .plan_id = source_row->plan_id,
        .descriptor_id = source_row->descriptor_id,
        .emitted_low_op_count = source_row->emitted_low_op_count,
    };
    loom_target_compile_report_record_source_low_row(report, &row);
  }
  if (lower_result->report_row_total_count > lower_result->report_row_count) {
    report->source_low_row_total_count +=
        lower_result->report_row_total_count - lower_result->report_row_count;
  }
}

iree_status_t loom_target_compile_report_allocate_low_lowering_rows(
    const loom_target_compile_report_t* report, iree_arena_allocator_t* arena,
    loom_low_lower_report_storage_t* out_storage) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_storage);
  *out_storage = (loom_low_lower_report_storage_t){0};
  if (report == NULL || report->source_low_rows == NULL ||
      report->source_low_row_count >= report->source_low_row_capacity) {
    return iree_ok_status();
  }
  out_storage->row_capacity =
      report->source_low_row_capacity - report->source_low_row_count;
  return iree_arena_allocate_array(arena, out_storage->row_capacity,
                                   sizeof(*out_storage->rows),
                                   (void**)&out_storage->rows);
}

static void loom_target_compile_report_record_pressure_rows(
    loom_target_compile_report_t* report,
    const loom_liveness_analysis_t* liveness) {
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    const loom_liveness_pressure_summary_t* summary =
        &liveness->pressure_summaries[i];
    const loom_target_compile_report_pressure_row_t row = {
        .register_class = loom_target_compile_report_value_class_name(
            liveness->module, summary->value_class),
        .type_kind = summary->value_class.type_kind,
        .element_type = summary->value_class.element_type,
        .peak_live_units = summary->peak_live_units,
        .peak_live_values = summary->peak_live_values,
        .peak_point = summary->peak_point,
        .peak_block_name = loom_target_compile_report_block_name(
            liveness->module, summary->peak_block),
        .peak_operation_name = summary->peak_op
                                   ? loom_target_compile_report_op_name(
                                         liveness->module, summary->peak_op)
                                   : IREE_SV("<block-boundary>"),
    };
    loom_target_compile_report_record_pressure_row(report, &row);
  }
}

static void loom_target_compile_report_record_spill_rows(
    loom_target_compile_report_t* report,
    const loom_low_allocation_sidecar_t* allocation) {
  for (iree_host_size_t i = 0; i < allocation->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan =
        &allocation->spill_plans[i];
    const loom_low_allocation_assignment_t* assignment = NULL;
    if (spill_plan->assignment_index < allocation->assignment_count) {
      assignment = &allocation->assignments[spill_plan->assignment_index];
    }
    const loom_liveness_value_class_t value_class =
        assignment != NULL ? assignment->value_class
                           : (loom_liveness_value_class_t){0};
    const loom_target_compile_report_spill_row_t row = {
        .value_name = loom_target_compile_report_value_name(
            allocation->module, spill_plan->value_id),
        .register_class = loom_target_compile_report_value_class_name(
            allocation->module, value_class),
        .type_kind = value_class.type_kind,
        .element_type = value_class.element_type,
        .assignment_index = spill_plan->assignment_index,
        .slot_index = spill_plan->slot_index,
        .slot_space = loom_low_spill_slot_space_name(spill_plan->slot_space),
        .byte_size = spill_plan->byte_size,
        .byte_alignment = spill_plan->byte_alignment,
        .store_count = spill_plan->store_count,
        .reload_count = spill_plan->reload_count,
    };
    loom_target_compile_report_record_spill_row(report, &row);
  }
}

void loom_target_compile_report_record_low_packetization(
    loom_target_compile_report_t* report,
    const loom_low_packetization_t* packetization) {
  IREE_ASSERT_ARGUMENT(report);
  IREE_ASSERT_ARGUMENT(packetization);

  const loom_liveness_analysis_t* liveness =
      &packetization->allocation.liveness;
  uint64_t peak_live_units = 0;
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    const uint64_t live_units = liveness->pressure_summaries[i].peak_live_units;
    peak_live_units = iree_max(peak_live_units, live_units);
  }
  loom_target_compile_report_record_schedule(
      report, packetization->schedule.node_count,
      packetization->schedule.scheduled_node_count,
      packetization->schedule.dependency_count,
      packetization->schedule.resource_use_count,
      packetization->schedule.hazard_gap_count,
      packetization->schedule.model_summary_count,
      liveness->pressure_summary_count, peak_live_units);
  loom_target_compile_report_record_allocation(
      report, packetization->allocation.assignment_count,
      packetization->allocation.spill_count,
      packetization->allocation.spill_plan_count,
      packetization->allocation.coalesced_copy_count,
      packetization->allocation.materialized_copy_count);
  loom_target_compile_report_record_move_causes(report, packetization);
  loom_target_compile_report_record_pressure_rows(report, liveness);
  loom_target_compile_report_record_spill_rows(report,
                                               &packetization->allocation);
}
