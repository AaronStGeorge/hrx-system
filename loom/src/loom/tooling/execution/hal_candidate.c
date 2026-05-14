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

static void loom_run_hal_candidate_initialize(
    const loom_run_hal_backend_t* backend,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_run_hal_candidate_t* out_candidate) {
  *out_candidate = (loom_run_hal_candidate_t){
      .host_allocator = allocator,
      .backend = backend,
  };
  loom_target_compile_report_t* report =
      options->report != NULL ? &out_candidate->compile_report : NULL;
  if (report == NULL) {
    return;
  }
  loom_target_compile_report_initialize(report);
  loom_target_compile_report_set_row_storage(report,
                                             &options->report_row_storage);
  report->artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE;
  report->entry_symbol = options->entry_symbol;
  report->backend_name = backend->name;
  report->target_family_name = backend->target_family_name;
}

static void loom_run_hal_candidate_record_report_status(
    const loom_run_candidate_compile_options_t* options,
    loom_run_hal_candidate_t* candidate, iree_status_t status) {
  loom_target_compile_report_t* report =
      options->report != NULL ? &candidate->compile_report : NULL;
  if (report == NULL) {
    return;
  }
  report->artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE;
  report->entry_symbol = options->entry_symbol;
  report->backend_name = candidate->backend->name;
  report->target_family_name = candidate->backend->target_family_name;
  if (candidate->compiled) {
    report->target_key = candidate->target.target_key;
    report->executable_format = candidate->executable.executable_format;
    loom_target_compile_report_record_artifact_size(
        report, candidate->executable.executable_data.data_length);
  }
  loom_target_compile_report_record_status(report, status);
}

static iree_status_t loom_run_hal_candidate_emit_selected_target(
    const loom_run_hal_backend_t* backend, loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_run_hal_candidate_t* candidate) {
  if (backend->emit_executable == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL backend '%.*s' is missing required emit hook",
                            (int)backend->name.size, backend->name.data);
  }

  loom_target_compile_report_t* report =
      options->report != NULL ? &candidate->compile_report : NULL;
  iree_status_t status = backend->emit_executable(
      backend, run_module->module, &candidate->target, options->entry_symbol,
      options->diagnostic_sink, options->source_resolver, options->max_errors,
      options->artifact_flags, report, allocator, &candidate->compiled,
      &candidate->executable);
  if (iree_status_is_ok(status) && candidate->compiled &&
      candidate->executable.target_bundle == NULL) {
    candidate->executable.target_bundle = candidate->target.target_bundle;
  }
  return status;
}

iree_status_t loom_run_hal_candidate_compile(
    const loom_run_hal_backend_t* backend,
    const loom_run_hal_runtime_t* runtime, loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_run_hal_candidate_t* out_candidate) {
  loom_run_hal_candidate_initialize(backend, options, allocator, out_candidate);

  iree_status_t status = iree_ok_status();
  if (backend->select_target == NULL) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL backend '%.*s' is missing required target selection hook",
        (int)backend->name.size, backend->name.data);
  }

  if (iree_status_is_ok(status)) {
    status = backend->select_target(backend, runtime, allocator,
                                    &out_candidate->target);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_candidate_emit_selected_target(
        backend, run_module, options, allocator, out_candidate);
  }
  loom_run_hal_candidate_record_report_status(options, out_candidate, status);
  loom_run_hal_candidate_publish_compile_report(options, out_candidate);
  if (!iree_status_is_ok(status)) {
    loom_run_hal_candidate_deinitialize(out_candidate);
  }
  return status;
}

iree_status_t loom_run_hal_candidate_emit_module_target(
    const loom_run_hal_backend_t* backend, loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_run_hal_candidate_t* out_candidate) {
  loom_run_hal_candidate_initialize(backend, options, allocator, out_candidate);
  iree_status_t status = loom_run_hal_candidate_emit_selected_target(
      backend, run_module, options, allocator, out_candidate);
  loom_run_hal_candidate_record_report_status(options, out_candidate, status);
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
