// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/compile_report.h"

void loom_target_compile_report_initialize(
    loom_target_compile_report_t* out_report) {
  IREE_ASSERT_ARGUMENT(out_report);
  *out_report = (loom_target_compile_report_t){
      .status_code = IREE_STATUS_OK,
  };
}

void loom_target_compile_report_record_status(
    loom_target_compile_report_t* report, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(report);
  report->status_code = iree_status_code(status);
}

void loom_target_compile_report_record_target_bundle(
    loom_target_compile_report_t* report, const loom_target_bundle_t* bundle) {
  IREE_ASSERT_ARGUMENT(report);
  if (bundle == NULL) {
    return;
  }
  report->target_bundle_name = bundle->name;
  if (bundle->snapshot != NULL) {
    report->target_snapshot_name = bundle->snapshot->name;
  }
  if (bundle->export_plan != NULL) {
    report->target_export_name = bundle->export_plan->name;
  }
  if (bundle->config != NULL) {
    report->target_config_name = bundle->config->name;
  }
}

void loom_target_compile_report_record_artifact_size(
    loom_target_compile_report_t* report, uint64_t artifact_size) {
  IREE_ASSERT_ARGUMENT(report);
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE;
  report->artifact_size = artifact_size;
}

void loom_target_compile_report_record_schedule(
    loom_target_compile_report_t* report, uint64_t node_count,
    uint64_t scheduled_node_count, uint64_t dependency_count,
    uint64_t resource_use_count, uint64_t hazard_gap_count,
    uint64_t model_summary_count, uint64_t pressure_summary_count,
    uint64_t peak_live_units) {
  IREE_ASSERT_ARGUMENT(report);
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE;
  report->schedule_node_count = node_count;
  report->scheduled_node_count = scheduled_node_count;
  report->schedule_dependency_count = dependency_count;
  report->schedule_resource_use_count = resource_use_count;
  report->schedule_hazard_gap_count = hazard_gap_count;
  report->schedule_model_summary_count = model_summary_count;
  report->register_pressure_summary_count = pressure_summary_count;
  report->register_pressure_peak_live_units = peak_live_units;
}

void loom_target_compile_report_record_allocation(
    loom_target_compile_report_t* report, uint64_t assignment_count,
    uint64_t spill_count, uint64_t spill_plan_count,
    uint64_t coalesced_copy_count, uint64_t materialized_copy_count) {
  IREE_ASSERT_ARGUMENT(report);
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION;
  report->allocation_assignment_count = assignment_count;
  report->allocation_spill_count = spill_count;
  report->allocation_spill_plan_count = spill_plan_count;
  report->allocation_coalesced_copy_count = coalesced_copy_count;
  report->allocation_materialized_copy_count = materialized_copy_count;
}
