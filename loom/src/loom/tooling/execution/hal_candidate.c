// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal_candidate.h"

static void loom_run_hal_candidate_publish_compile_report(
    const loom_run_candidate_compile_options_t* options,
    const loom_run_hal_candidate_t* candidate) {
  if (options->report == NULL) {
    return;
  }
  *options->report = candidate->compile_report;
}

iree_status_t loom_run_hal_candidate_compile(
    const loom_run_hal_backend_t* backend,
    const loom_run_hal_runtime_t* runtime, loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_run_hal_candidate_t* out_candidate) {
  *out_candidate = (loom_run_hal_candidate_t){
      .host_allocator = allocator,
      .backend = backend,
  };
  loom_target_compile_report_t* report =
      options->report != NULL ? &out_candidate->compile_report : NULL;
  if (report != NULL) {
    loom_target_compile_report_initialize(report);
    loom_target_compile_report_set_row_storage(report,
                                               &options->report_row_storage);
    report->artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE;
    report->entry_symbol = options->entry_symbol;
    report->backend_name = backend->name;
    report->target_family_name = backend->target_family_name;
  }

  iree_status_t status = iree_ok_status();
  if (backend->select_target == NULL || backend->compile == NULL) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL backend '%.*s' is missing required candidate hooks",
        (int)backend->name.size, backend->name.data);
  }

  if (iree_status_is_ok(status)) {
    status = backend->select_target(backend, runtime, allocator,
                                    &out_candidate->target);
  }
  if (iree_status_is_ok(status) && report != NULL) {
    report->target_preset_key = out_candidate->target.preset_key;
  }
  if (iree_status_is_ok(status)) {
    status = backend->compile(
        backend, run_module->module, &out_candidate->target,
        options->entry_symbol, options->diagnostic_sink,
        options->source_resolver, options->max_errors, report, allocator,
        &out_candidate->compiled, &out_candidate->executable);
    if (iree_status_is_ok(status) && out_candidate->compiled &&
        out_candidate->executable.target_bundle == NULL) {
      out_candidate->executable.target_bundle =
          out_candidate->target.target_bundle;
    }
  }
  if (report != NULL) {
    report->artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE;
    report->entry_symbol = options->entry_symbol;
    report->backend_name = backend->name;
    report->target_family_name = backend->target_family_name;
  }
  if (iree_status_is_ok(status) && out_candidate->compiled && report != NULL) {
    report->target_preset_key = out_candidate->target.preset_key;
    report->executable_format = out_candidate->executable.executable_format;
    loom_target_compile_report_record_artifact_size(
        report, out_candidate->executable.executable_data.data_length);
  }
  if (report != NULL) {
    loom_target_compile_report_record_status(report, status);
  }
  loom_run_hal_candidate_publish_compile_report(options, out_candidate);
  if (!iree_status_is_ok(status)) {
    loom_run_hal_candidate_deinitialize(out_candidate);
  }
  return status;
}

void loom_run_hal_candidate_deinitialize(loom_run_hal_candidate_t* candidate) {
  if (candidate == NULL) {
    return;
  }
  if (candidate->backend != NULL &&
      candidate->backend->deinitialize_executable != NULL) {
    candidate->backend->deinitialize_executable(
        candidate->backend, &candidate->executable, candidate->host_allocator);
  }
  *candidate = (loom_run_hal_candidate_t){0};
}
