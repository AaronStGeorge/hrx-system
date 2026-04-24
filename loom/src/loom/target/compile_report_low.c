// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/compile_report_low.h"

#include "loom/ir/context.h"
#include "loom/ir/module.h"

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
  loom_target_compile_report_record_pressure_rows(report, liveness);
  loom_target_compile_report_record_spill_rows(report,
                                               &packetization->allocation);
}
