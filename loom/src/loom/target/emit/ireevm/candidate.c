// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/candidate.h"

#include "loom/target/emit/ireevm/archive_emitter.h"

static void loom_ireevm_run_candidate_publish_compile_report(
    const loom_run_candidate_compile_options_t* options,
    const loom_ireevm_run_candidate_t* candidate) {
  if (options->report == NULL) {
    return;
  }
  *options->report = candidate->compile_report;
}

iree_status_t loom_ireevm_run_candidate_emit(
    loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_ireevm_run_candidate_t* out_candidate) {
  *out_candidate = (loom_ireevm_run_candidate_t){
      .host_allocator = allocator,
  };
  loom_target_compile_report_t* report =
      options->report != NULL ? &out_candidate->compile_report : NULL;
  if (report != NULL) {
    loom_target_compile_report_initialize(report);
    loom_target_compile_report_set_row_storage(report,
                                               &options->report_row_storage);
    report->artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;
    report->module_name = options->module_name;
    report->entry_symbol = options->entry_symbol;
  }

  const loom_ireevm_archive_emit_options_t archive_emit_options = {
      .module_name = options->module_name,
      .entry_symbol = options->entry_symbol,
      .diagnostic_sink = options->diagnostic_sink,
      .source_resolver = options->source_resolver,
      .max_errors = options->max_errors,
      .report = report,
      .report_row_storage = options->report_row_storage,
  };
  iree_status_t status = loom_ireevm_emit_module_archive_from_ir(
      run_module->module, &archive_emit_options, allocator,
      &out_candidate->emitted, &out_candidate->archive);
  if (report != NULL) {
    report->artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;
    report->module_name = options->module_name;
    report->entry_symbol = options->entry_symbol;
  }
  if (iree_status_is_ok(status) && out_candidate->emitted && report != NULL) {
    loom_target_compile_report_record_artifact_size(
        report, out_candidate->archive.data_length);
  }
  if (report != NULL) {
    loom_target_compile_report_record_status(report, status);
  }
  loom_ireevm_run_candidate_publish_compile_report(options, out_candidate);
  if (!iree_status_is_ok(status)) {
    loom_ireevm_run_candidate_deinitialize(out_candidate);
  }
  return status;
}

void loom_ireevm_run_candidate_deinitialize(
    loom_ireevm_run_candidate_t* candidate) {
  if (candidate == NULL) {
    return;
  }
  loom_ireevm_module_archive_deinitialize(&candidate->archive,
                                          candidate->host_allocator);
  *candidate = (loom_ireevm_run_candidate_t){0};
}
