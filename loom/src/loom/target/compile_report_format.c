// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/compile_report_format.h"

#include <inttypes.h>

#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"

void loom_target_compile_report_format_options_initialize(
    loom_target_compile_report_format_options_t* out_options) {
  *out_options = (loom_target_compile_report_format_options_t){
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE,
  };
}

iree_status_t loom_target_compile_report_format_mode_parse(
    iree_string_view_t value,
    loom_target_compile_report_format_mode_t* out_mode) {
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, IREE_SV("none"))) {
    *out_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("summary"))) {
    *out_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("details"))) {
    *out_mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported compile report mode '%.*s'; expected "
                          "'none', 'summary', or 'details'",
                          (int)value.size, value.data);
}

static iree_string_view_t loom_target_compile_report_artifact_kind_name(
    loom_target_compile_artifact_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_NONE:
      return IREE_SV("none");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE:
      return IREE_SV("vm-archive");
    case LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE:
      return IREE_SV("hal-executable");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_target_compile_report_source_low_selection_name(
    loom_target_compile_report_source_low_selection_kind_t kind) {
  switch (kind) {
    case LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE:
      return IREE_SV("rule");
    case LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_PLAN:
      return IREE_SV("plan");
    case LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_NONE:
    default:
      return IREE_SV("none");
  }
}

typedef struct loom_target_compile_report_move_cause_descriptor_t {
  // Residual move cause used as the report counter index.
  loom_target_compile_report_move_cause_t cause;
  // Stable text name emitted in compile reports.
  iree_string_view_t name;
} loom_target_compile_report_move_cause_descriptor_t;

static const loom_target_compile_report_move_cause_descriptor_t
    loom_target_compile_report_move_cause_descriptors[] = {
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_CONSTANT_MATERIALIZATION,
         IREE_SVL("constant_materialization")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY, IREE_SVL("low_copy")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_SLICE,
         IREE_SVL("low_slice")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_CONCAT,
         IREE_SVL("low_concat")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_BRANCH_EDGE,
         IREE_SVL("branch_edge")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_CONSTRAINT_REPAIR,
         IREE_SVL("operand_constraint_repair")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_ABI_COPY, IREE_SVL("abi_copy")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_SPILL_RELOAD,
         IREE_SVL("spill_reload")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_PARTIAL_REGISTER_REPAIR,
         IREE_SVL("partial_register_repair")},
        {LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_UNKNOWN, IREE_SVL("unknown")},
};
static_assert(
    IREE_ARRAYSIZE(loom_target_compile_report_move_cause_descriptors) ==
        LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT - 1,
    "move cause descriptor table must cover each reportable move cause");

static iree_string_view_t loom_target_compile_report_type_kind_name(
    uint32_t type_kind) {
  switch (type_kind) {
    case LOOM_TYPE_NONE:
      return IREE_SV("none");
    case LOOM_TYPE_SCALAR:
      return IREE_SV("scalar");
    case LOOM_TYPE_TENSOR:
      return IREE_SV("tensor");
    case LOOM_TYPE_TILE:
      return IREE_SV("tile");
    case LOOM_TYPE_VECTOR:
      return IREE_SV("vector");
    case LOOM_TYPE_VIEW:
      return IREE_SV("view");
    case LOOM_TYPE_BUFFER:
      return IREE_SV("buffer");
    case LOOM_TYPE_FUNCTION:
      return IREE_SV("function");
    case LOOM_TYPE_ENCODING:
      return IREE_SV("encoding");
    case LOOM_TYPE_DIALECT:
      return IREE_SV("dialect");
    case LOOM_TYPE_GROUP:
      return IREE_SV("group");
    case LOOM_TYPE_POOL:
      return IREE_SV("pool");
    case LOOM_TYPE_REGISTER:
      return IREE_SV("register");
    case LOOM_TYPE_STORAGE:
      return IREE_SV("storage");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_target_compile_report_scalar_type_name(
    uint32_t element_type) {
  const char* name = loom_scalar_type_name((loom_scalar_type_t)element_type);
  if (name == NULL) {
    return IREE_SV("unknown");
  }
  return iree_make_cstring_view(name);
}

static iree_string_view_t loom_target_compile_report_non_empty(
    iree_string_view_t value) {
  return iree_string_view_is_empty(value) ? IREE_SV("-") : value;
}

static void loom_target_compile_report_accumulate_move_cause(
    const loom_target_compile_report_move_cause_counts_t* counts,
    uint64_t* kind_count, uint64_t* packet_count, uint64_t* unit_count) {
  if (counts->packet_count == 0 && counts->unit_count == 0) {
    return;
  }
  ++*kind_count;
  *packet_count += counts->packet_count;
  *unit_count += counts->unit_count;
}

static void loom_target_compile_report_move_cause_totals(
    const loom_target_compile_report_t* report, uint64_t* out_kind_count,
    uint64_t* out_packet_count, uint64_t* out_unit_count) {
  *out_kind_count = 0;
  *out_packet_count = 0;
  *out_unit_count = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(loom_target_compile_report_move_cause_descriptors);
       ++i) {
    const loom_target_compile_report_move_cause_descriptor_t* descriptor =
        &loom_target_compile_report_move_cause_descriptors[i];
    loom_target_compile_report_accumulate_move_cause(
        &report->move_causes[descriptor->cause], out_kind_count,
        out_packet_count, out_unit_count);
  }
}

static iree_status_t loom_target_compile_report_append_string_field(
    iree_string_builder_t* builder, iree_string_view_t name,
    iree_string_view_t value) {
  value = loom_target_compile_report_non_empty(value);
  return iree_string_builder_append_format(builder, " %.*s=%.*s",
                                           (int)name.size, name.data,
                                           (int)value.size, value.data);
}

static iree_status_t loom_target_compile_report_format_summary(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      builder, IREE_SV("COMPILE-REPORT: summary")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("artifact"),
      loom_target_compile_report_artifact_kind_name(report->artifact_kind)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, " status=%s", iree_status_code_string(report->status_code)));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("backend"), report->backend_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("entry"), report->entry_symbol));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("bundle"), report->target_bundle_name));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_append_string_field(
      builder, IREE_SV("lowered"), report->lowered_symbol));
  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " artifact_bytes=%" PRIu64, report->artifact_size));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("\n")));

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: schedule nodes=%" PRIu64 " scheduled=%" PRIu64
        " deps=%" PRIu64 " resources=%" PRIu64 " hazards=%" PRIu64
        " models=%" PRIu64 " pressure_classes=%" PRIu64
        " peak_live_units=%" PRIu64 "\n",
        report->schedule_node_count, report->scheduled_node_count,
        report->schedule_dependency_count, report->schedule_resource_use_count,
        report->schedule_hazard_gap_count, report->schedule_model_summary_count,
        report->register_pressure_summary_count,
        report->register_pressure_peak_live_units));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: allocation assignments=%" PRIu64 " spills=%" PRIu64
        " spill_plans=%" PRIu64 " coalesced_copies=%" PRIu64
        " materialized_copies=%" PRIu64 "\n",
        report->allocation_assignment_count, report->allocation_spill_count,
        report->allocation_spill_plan_count,
        report->allocation_coalesced_copy_count,
        report->allocation_materialized_copy_count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES)) {
    uint64_t kind_count = 0;
    uint64_t packet_count = 0;
    uint64_t unit_count = 0;
    loom_target_compile_report_move_cause_totals(report, &kind_count,
                                                 &packet_count, &unit_count);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: move_causes kinds=%" PRIu64 " packets=%" PRIu64
        " units=%" PRIu64 "\n",
        kind_count, packet_count, unit_count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: emission instructions=%" PRIu64 " code_bytes=%" PRIu64
        " storage_bytes=%" PRIu64 "\n",
        report->emitted_instruction_count, report->emitted_code_byte_count,
        report->emitted_code_storage_byte_count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: source_low selected_ops=%" PRIu64
        " emitted_ops=%" PRIu64 " rows copied=%" PRIhsz " total=%" PRIhsz "\n",
        report->source_low_selected_op_count,
        report->source_low_emitted_op_count, report->source_low_row_count,
        report->source_low_row_total_count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_MEMORY)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: memory private_bytes=%" PRIu64 " local_bytes=%" PRIu64
        "\n",
        report->private_memory_bytes, report->local_memory_bytes));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: pressure_rows copied=%" PRIhsz " total=%" PRIhsz "\n",
        report->pressure_row_count, report->pressure_row_total_count));
  }

  if (iree_any_bit_set(report->detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: spill_rows copied=%" PRIhsz " total=%" PRIhsz "\n",
        report->spill_row_count, report->spill_row_total_count));
  }

  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_move_cause(
    iree_string_builder_t* builder,
    const loom_target_compile_report_move_cause_descriptor_t* descriptor,
    const loom_target_compile_report_move_cause_counts_t* counts) {
  if (counts->packet_count == 0 && counts->unit_count == 0) {
    return iree_ok_status();
  }
  return iree_string_builder_append_format(
      builder,
      "COMPILE-REPORT: move_cause[%.*s] packets=%" PRIu64 " units=%" PRIu64
      "\n",
      (int)descriptor->name.size, descriptor->name.data, counts->packet_count,
      counts->unit_count);
}

static iree_status_t loom_target_compile_report_format_move_causes(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(loom_target_compile_report_move_cause_descriptors);
       ++i) {
    const loom_target_compile_report_move_cause_descriptor_t* descriptor =
        &loom_target_compile_report_move_cause_descriptors[i];
    IREE_RETURN_IF_ERROR(loom_target_compile_report_format_move_cause(
        builder, descriptor, &report->move_causes[descriptor->cause]));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_pressure_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  for (iree_host_size_t i = 0; i < report->pressure_row_count; ++i) {
    const loom_target_compile_report_pressure_row_t* row =
        &report->pressure_rows[i];
    const iree_string_view_t register_class =
        loom_target_compile_report_non_empty(row->register_class);
    const iree_string_view_t type_kind_name =
        loom_target_compile_report_type_kind_name(row->type_kind);
    const iree_string_view_t element_type_name =
        loom_target_compile_report_scalar_type_name(row->element_type);
    const iree_string_view_t peak_block_name =
        loom_target_compile_report_non_empty(row->peak_block_name);
    const iree_string_view_t peak_operation_name =
        loom_target_compile_report_non_empty(row->peak_operation_name);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: pressure[%" PRIhsz
        "] class=%.*s type=%.*s element=%.*s peak_units=%" PRIu64
        " peak_values=%" PRIu64 " point=%u block=%.*s op=%.*s\n",
        i, (int)register_class.size, register_class.data,
        (int)type_kind_name.size, type_kind_name.data,
        (int)element_type_name.size, element_type_name.data,
        row->peak_live_units, row->peak_live_values, row->peak_point,
        (int)peak_block_name.size, peak_block_name.data,
        (int)peak_operation_name.size, peak_operation_name.data));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_spill_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  for (iree_host_size_t i = 0; i < report->spill_row_count; ++i) {
    const loom_target_compile_report_spill_row_t* row = &report->spill_rows[i];
    const iree_string_view_t value_name =
        loom_target_compile_report_non_empty(row->value_name);
    const iree_string_view_t register_class =
        loom_target_compile_report_non_empty(row->register_class);
    const iree_string_view_t type_kind_name =
        loom_target_compile_report_type_kind_name(row->type_kind);
    const iree_string_view_t element_type_name =
        loom_target_compile_report_scalar_type_name(row->element_type);
    const iree_string_view_t slot_space =
        loom_target_compile_report_non_empty(row->slot_space);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: spill[%" PRIhsz
        "] value=%.*s class=%.*s type=%.*s element=%.*s assignment=%u"
        " slot=%u space=%.*s bytes=%" PRIu64 " align=%" PRIu64
        " stores=%" PRIu64 " reloads=%" PRIu64 "\n",
        i, (int)value_name.size, value_name.data, (int)register_class.size,
        register_class.data, (int)type_kind_name.size, type_kind_name.data,
        (int)element_type_name.size, element_type_name.data,
        row->assignment_index, row->slot_index, (int)slot_space.size,
        slot_space.data, row->byte_size, row->byte_alignment, row->store_count,
        row->reload_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_source_low_rows(
    const loom_target_compile_report_t* report,
    iree_string_builder_t* builder) {
  for (iree_host_size_t i = 0; i < report->source_low_row_count; ++i) {
    const loom_target_compile_report_source_low_row_t* row =
        &report->source_low_rows[i];
    const iree_string_view_t function_name =
        loom_target_compile_report_non_empty(row->function_name);
    const iree_string_view_t source_op_name =
        loom_target_compile_report_non_empty(row->source_op_name);
    const iree_string_view_t selection_name =
        loom_target_compile_report_source_low_selection_name(
            row->selection_kind);
    if (row->selection_kind ==
        LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: source_low[%" PRIhsz
          "] function=%.*s source_op=%.*s selection=%.*s rule_set=%u rule=%u "
          "descriptor=%" PRIu64 " emitted_ops=%u\n",
          i, (int)function_name.size, function_name.data,
          (int)source_op_name.size, source_op_name.data,
          (int)selection_name.size, selection_name.data, row->rule_set_index,
          row->rule_index, row->descriptor_id, row->emitted_low_op_count));
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: source_low[%" PRIhsz
        "] function=%.*s source_op=%.*s selection=%.*s plan=%" PRIu64
        " descriptor=%" PRIu64 " emitted_ops=%u\n",
        i, (int)function_name.size, function_name.data,
        (int)source_op_name.size, source_op_name.data, (int)selection_name.size,
        selection_name.data, row->plan_id, row->descriptor_id,
        row->emitted_low_op_count));
  }
  return iree_ok_status();
}

iree_status_t loom_target_compile_report_format_text(
    const loom_target_compile_report_t* report,
    const loom_target_compile_report_format_options_t* options,
    iree_string_builder_t* builder) {
  if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_NONE) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_summary(report, builder));
  if (options->mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_move_causes(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pressure_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_spill_rows(report, builder));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_source_low_rows(report, builder));
  }
  return iree_ok_status();
}
