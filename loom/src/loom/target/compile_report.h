// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured module compilation reports.

#ifndef LOOM_TARGET_COMPILE_REPORT_H_
#define LOOM_TARGET_COMPILE_REPORT_H_

#include "iree/base/api.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_target_compile_artifact_kind_e {
  // No artifact was produced.
  LOOM_TARGET_COMPILE_ARTIFACT_KIND_NONE = 0,
  // IREE VM bytecode archive artifact.
  LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE = 1,
  // IREE HAL executable container artifact.
  LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE = 2,
} loom_target_compile_artifact_kind_t;

typedef uint32_t loom_target_compile_report_detail_flags_t;
enum {
  // No optional report details are populated.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_NONE = 0u,
  // |artifact_size| is populated.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE = 1u << 0,
  // Schedule node, dependency, pressure, and resource summaries are populated.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE = 1u << 1,
  // Allocation assignment, copy, and spill counts are populated.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION = 1u << 2,
  // Target private/local memory estimates are populated.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_MEMORY = 1u << 3,
  // Target emission instruction and code-size summaries are populated.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION = 1u << 4,
  // Per-pressure-class rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS = 1u << 5,
  // Per-spill-plan rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS = 1u << 6,
  // Source-to-target-low selection rows were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS = 1u << 7,
  // Residual target move causes were recorded or counted.
  LOOM_TARGET_COMPILE_REPORT_DETAIL_MOVE_CAUSES = 1u << 8,
};

typedef enum loom_target_compile_report_move_cause_e {
  // No residual target move cause was recorded.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_NONE = 0,
  // A low constant packet materialized an immediate value into a register.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_CONSTANT_MATERIALIZATION,
  // A low.copy packet survived allocation and must be emitted as a move.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY,
  // A low.slice packet survived allocation and must be emitted as moves.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_SLICE,
  // A low.concat packet survived allocation and must be emitted as moves.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_CONCAT,
  // A control-flow edge payload must be emitted as moves.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_BRANCH_EDGE,
  // A tied, destructive, or fixed operand constraint required repair moves.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_CONSTRAINT_REPAIR,
  // ABI lowering inserted entry, exit, or call-boundary copies.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_ABI_COPY,
  // Spill or reload materialization inserted target moves.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_SPILL_RELOAD,
  // Partial-register lowering inserted target moves.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_PARTIAL_REGISTER_REPAIR,
  // A residual move could not be assigned a more precise structural cause.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_UNKNOWN,
  // Number of residual target move cause values, including NONE.
  LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT,
} loom_target_compile_report_move_cause_t;

typedef enum loom_target_compile_report_source_low_selection_kind_e {
  // No source-low selection was recorded.
  LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_NONE = 0,
  // Selection came from a table-driven lowering rule.
  LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE = 1,
  // Selection came from a target-owned callback plan.
  LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_PLAN = 2,
} loom_target_compile_report_source_low_selection_kind_t;

// Residual move-cause counters for one category.
typedef struct loom_target_compile_report_move_cause_counts_t {
  // Number of target packets attributed to this cause.
  uint64_t packet_count;
  // Number of register-unit moves attributed to this cause.
  uint64_t unit_count;
} loom_target_compile_report_move_cause_counts_t;

// One register-pressure peak row copied into a compile report.
typedef struct loom_target_compile_report_pressure_row_t {
  // Register class name for register values, or an empty string otherwise.
  iree_string_view_t register_class;
  // Numeric Loom type kind for the pressure class.
  uint32_t type_kind;
  // Numeric Loom scalar element type for the pressure class.
  uint32_t element_type;
  // Maximum boundary-live units observed for the class.
  uint64_t peak_live_units;
  // Maximum simultaneously live values observed at the same point.
  uint64_t peak_live_values;
  // Program point associated with the peak.
  uint32_t peak_point;
  // Block label containing the peak, or a fallback diagnostic name.
  iree_string_view_t peak_block_name;
  // Operation name after which the peak was observed, or a boundary marker.
  iree_string_view_t peak_operation_name;
} loom_target_compile_report_pressure_row_t;

// One predicted spill row copied into a compile report.
typedef struct loom_target_compile_report_spill_row_t {
  // SSA value name represented by the spilled assignment.
  iree_string_view_t value_name;
  // Register class name for the spilled value.
  iree_string_view_t register_class;
  // Numeric Loom type kind for the spilled value class.
  uint32_t type_kind;
  // Numeric Loom scalar element type for the spilled value class.
  uint32_t element_type;
  // Allocation assignment index associated with this spill.
  uint32_t assignment_index;
  // Spill slot ordinal assigned to the interval.
  uint32_t slot_index;
  // Spill storage-space name.
  iree_string_view_t slot_space;
  // Slot size in bytes.
  uint64_t byte_size;
  // Required slot alignment in bytes.
  uint64_t byte_alignment;
  // Predicted stores needed by the current synthetic spill plan.
  uint64_t store_count;
  // Predicted operand-use reloads in the current synthetic spill plan.
  uint64_t reload_count;
} loom_target_compile_report_spill_row_t;

// One source-to-target-low selection row copied into a compile report.
typedef struct loom_target_compile_report_source_low_row_t {
  // Source function symbol containing the lowered source operation.
  iree_string_view_t function_name;
  // Source operation mnemonic lowered by this row.
  iree_string_view_t source_op_name;
  // Numeric source operation kind lowered by this row.
  uint32_t source_op_kind;
  // Selection mechanism used for this source operation.
  loom_target_compile_report_source_low_selection_kind_t selection_kind;
  // Policy rule-set ordinal for table-driven rules, or UINT16_MAX otherwise.
  uint16_t rule_set_index;
  // Rule-table ordinal inside |rule_set_index|, or UINT16_MAX otherwise.
  uint16_t rule_index;
  // Target-owned plan id for callback selections, or UINT64_MAX otherwise.
  uint64_t plan_id;
  // First stable low descriptor id emitted by a table rule, or none for plans.
  uint64_t descriptor_id;
  // Number of low operations emitted for this source operation.
  uint32_t emitted_low_op_count;
} loom_target_compile_report_source_low_row_t;

// Caller-owned storage for optional detailed compile report rows.
typedef struct loom_target_compile_report_row_storage_t {
  // Caller-owned pressure row storage.
  loom_target_compile_report_pressure_row_t* pressure_rows;
  // Capacity of |pressure_rows|.
  iree_host_size_t pressure_row_capacity;
  // Caller-owned spill row storage.
  loom_target_compile_report_spill_row_t* spill_rows;
  // Capacity of |spill_rows|.
  iree_host_size_t spill_row_capacity;
  // Caller-owned source-low row storage.
  loom_target_compile_report_source_low_row_t* source_low_rows;
  // Capacity of |source_low_rows|.
  iree_host_size_t source_low_row_capacity;
} loom_target_compile_report_row_storage_t;

// Structured feedback from one module-to-artifact compilation.
//
// Reports are allocation-free and borrow every string view from the compiled
// module, target records, compile options, backend tables, or artifact storage.
// Optional detail rows are copied into caller-owned row storage configured
// before row recording; compile entry points expose this through their options
// because they initialize reports before populating them. Consumers that need a
// report to outlive those owners must copy the strings and rows before
// releasing the module or candidate.
typedef struct loom_target_compile_report_t {
  // Artifact kind requested or produced by compilation.
  loom_target_compile_artifact_kind_t artifact_kind;
  // Terminal status code observed by compilation.
  iree_status_code_t status_code;
  // Optional detail flags indicating which numeric summaries are populated.
  loom_target_compile_report_detail_flags_t detail_flags;
  // VM module name requested for archive emission.
  iree_string_view_t module_name;
  // Function entry symbol requested by the caller, if any.
  iree_string_view_t entry_symbol;
  // Execution or codegen backend name that produced the candidate, if any.
  iree_string_view_t backend_name;
  // Target family name selected by the backend, if any.
  iree_string_view_t target_family_name;
  // Target preset key selected by the backend, if any.
  iree_string_view_t target_preset_key;
  // Resolved target record name selected for compilation, if any.
  iree_string_view_t target_bundle_name;
  // Resolved target snapshot name selected for compilation, if any.
  iree_string_view_t target_snapshot_name;
  // Resolved target export-plan name selected for compilation, if any.
  iree_string_view_t target_export_name;
  // Resolved target config name selected for compilation, if any.
  iree_string_view_t target_config_name;
  // Low function symbol produced or selected after lowering, if any.
  iree_string_view_t lowered_symbol;
  // HAL executable format string, if a HAL artifact was produced.
  iree_string_view_t executable_format;
  // Number of bytes in the produced artifact.
  uint64_t artifact_size;
  // Number of low schedule nodes before target emission.
  uint64_t schedule_node_count;
  // Number of low schedule nodes in scheduled order.
  uint64_t scheduled_node_count;
  // Number of low schedule dependency edges.
  uint64_t schedule_dependency_count;
  // Number of descriptor resource-use records.
  uint64_t schedule_resource_use_count;
  // Number of required schedule hazard gaps.
  uint64_t schedule_hazard_gap_count;
  // Number of schedule model-quality summary records.
  uint64_t schedule_model_summary_count;
  // Number of register-pressure summary records.
  uint64_t register_pressure_summary_count;
  // Maximum boundary-live register units observed across pressure summaries.
  uint64_t register_pressure_peak_live_units;
  // Number of allocation assignments.
  uint64_t allocation_assignment_count;
  // Number of values assigned to spill slots.
  uint64_t allocation_spill_count;
  // Number of synthetic spill plans.
  uint64_t allocation_spill_plan_count;
  // Number of low.copy ops coalesced away by allocation.
  uint64_t allocation_coalesced_copy_count;
  // Number of low.copy ops that must remain materialized.
  uint64_t allocation_materialized_copy_count;
  // Number of target instructions or bytecode opcodes emitted.
  uint64_t emitted_instruction_count;
  // Number of semantic target code bytes before target-local padding.
  uint64_t emitted_code_byte_count;
  // Number of target code storage bytes including target-local padding.
  uint64_t emitted_code_storage_byte_count;
  // Number of source operations selected during source-to-low lowering.
  uint64_t source_low_selected_op_count;
  // Number of low operations emitted during source-to-low lowering.
  uint64_t source_low_emitted_op_count;
  // Residual target move counts indexed by
  // loom_target_compile_report_move_cause_t.
  loom_target_compile_report_move_cause_counts_t
      move_causes[LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_COUNT];
  // Caller-owned pressure row storage.
  loom_target_compile_report_pressure_row_t* pressure_rows;
  // Capacity of |pressure_rows|.
  iree_host_size_t pressure_row_capacity;
  // Number of pressure rows copied into |pressure_rows|.
  iree_host_size_t pressure_row_count;
  // Total number of available pressure rows before capacity truncation.
  iree_host_size_t pressure_row_total_count;
  // Caller-owned spill row storage.
  loom_target_compile_report_spill_row_t* spill_rows;
  // Capacity of |spill_rows|.
  iree_host_size_t spill_row_capacity;
  // Number of spill rows copied into |spill_rows|.
  iree_host_size_t spill_row_count;
  // Total number of available spill rows before capacity truncation.
  iree_host_size_t spill_row_total_count;
  // Caller-owned source-low row storage.
  loom_target_compile_report_source_low_row_t* source_low_rows;
  // Capacity of |source_low_rows|.
  iree_host_size_t source_low_row_capacity;
  // Number of source-low rows copied into |source_low_rows|.
  iree_host_size_t source_low_row_count;
  // Total number of available source-low rows before capacity truncation.
  iree_host_size_t source_low_row_total_count;
  // Estimated target private memory bytes.
  uint64_t private_memory_bytes;
  // Estimated target local/shared memory bytes.
  uint64_t local_memory_bytes;
} loom_target_compile_report_t;

// Initializes an empty compile report.
void loom_target_compile_report_initialize(
    loom_target_compile_report_t* out_report);

// Configures caller-owned row storage for optional detailed report rows.
void loom_target_compile_report_set_row_storage(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_row_storage_t* row_storage);

// Records a terminal status code in |report|.
void loom_target_compile_report_record_status(
    loom_target_compile_report_t* report, iree_status_t status);

// Records the target bundle selected for compilation.
void loom_target_compile_report_record_target_bundle(
    loom_target_compile_report_t* report, const loom_target_bundle_t* bundle);

// Records the produced artifact byte size in |report|.
void loom_target_compile_report_record_artifact_size(
    loom_target_compile_report_t* report, uint64_t artifact_size);

// Records target-low schedule summary counts in |report|.
void loom_target_compile_report_record_schedule(
    loom_target_compile_report_t* report, uint64_t node_count,
    uint64_t scheduled_node_count, uint64_t dependency_count,
    uint64_t resource_use_count, uint64_t hazard_gap_count,
    uint64_t model_summary_count, uint64_t pressure_summary_count,
    uint64_t peak_live_units);

// Records target-low allocation summary counts in |report|.
void loom_target_compile_report_record_allocation(
    loom_target_compile_report_t* report, uint64_t assignment_count,
    uint64_t spill_count, uint64_t spill_plan_count,
    uint64_t coalesced_copy_count, uint64_t materialized_copy_count);

// Records target move materialization attributed to one residual move cause.
void loom_target_compile_report_record_move_cause(
    loom_target_compile_report_t* report,
    loom_target_compile_report_move_cause_t cause, uint64_t packet_count,
    uint64_t unit_count);

// Records target emission instruction and code-size summary counts in |report|.
void loom_target_compile_report_record_emission(
    loom_target_compile_report_t* report, uint64_t instruction_count,
    uint64_t code_byte_count, uint64_t code_storage_byte_count);

// Records target memory estimates in |report|.
void loom_target_compile_report_record_memory(
    loom_target_compile_report_t* report, uint64_t private_memory_bytes,
    uint64_t local_memory_bytes);

// Records one pressure row, truncating only the copied row storage when full.
void loom_target_compile_report_record_pressure_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_pressure_row_t* row);

// Records one spill row, truncating only the copied row storage when full.
void loom_target_compile_report_record_spill_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_spill_row_t* row);

// Records one source-low row, truncating only the copied row storage when full.
void loom_target_compile_report_record_source_low_row(
    loom_target_compile_report_t* report,
    const loom_target_compile_report_source_low_row_t* row);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_COMPILE_REPORT_H_
