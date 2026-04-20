// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/candidate.h"

#include "loom/target/emit/ireevm/module_compiler.h"

static void loom_run_candidate_publish_compile_report(
    const loom_run_candidate_compile_options_t* options,
    const loom_run_candidate_t* candidate) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(candidate);
  if (options->report == NULL) {
    return;
  }
  *options->report = candidate->compile_report;
}

iree_status_t loom_run_candidate_compile_vm(
    loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_run_candidate_t* out_candidate) {
  IREE_ASSERT_ARGUMENT(run_module);
  IREE_ASSERT_ARGUMENT(run_module->module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_candidate);
  *out_candidate = (loom_run_candidate_t){
      .kind = LOOM_RUN_CANDIDATE_KIND_VM_ARCHIVE,
      .host_allocator = allocator,
  };
  loom_target_compile_report_initialize(&out_candidate->compile_report);
  loom_target_compile_report_set_row_storage(&out_candidate->compile_report,
                                             &options->report_row_storage);
  out_candidate->compile_report.artifact_kind =
      LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;
  out_candidate->compile_report.module_name = options->module_name;
  out_candidate->compile_report.target_symbol = options->target_symbol;

  const loom_ireevm_module_compile_options_t compile_options = {
      .module_name = options->module_name,
      .target_symbol = options->target_symbol,
      .diagnostic_sink = options->diagnostic_sink,
      .source_resolver = options->source_resolver,
      .max_errors = options->max_errors,
      .report = &out_candidate->compile_report,
      .report_row_storage = options->report_row_storage,
  };
  iree_status_t status =
      loom_ireevm_compile_module_archive(run_module->module, &compile_options,
                                         allocator, &out_candidate->vm_archive);
  out_candidate->compile_report.artifact_kind =
      LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;
  out_candidate->compile_report.module_name = options->module_name;
  out_candidate->compile_report.target_symbol = options->target_symbol;
  if (iree_status_is_ok(status)) {
    loom_target_compile_report_record_artifact_size(
        &out_candidate->compile_report, out_candidate->vm_archive.data_length);
  }
  loom_target_compile_report_record_status(&out_candidate->compile_report,
                                           status);
  loom_run_candidate_publish_compile_report(options, out_candidate);
  if (!iree_status_is_ok(status)) {
    loom_run_candidate_deinitialize(out_candidate);
  }
  return status;
}

iree_status_t loom_run_candidate_compile_hal(
    const loom_run_hal_backend_t* backend,
    const loom_run_hal_runtime_t* runtime, loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_run_candidate_t* out_candidate) {
  IREE_ASSERT_ARGUMENT(backend);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(run_module);
  IREE_ASSERT_ARGUMENT(run_module->module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_candidate);
  *out_candidate = (loom_run_candidate_t){
      .kind = LOOM_RUN_CANDIDATE_KIND_HAL_EXECUTABLE,
      .host_allocator = allocator,
      .hal_backend = backend,
  };
  loom_target_compile_report_initialize(&out_candidate->compile_report);
  loom_target_compile_report_set_row_storage(&out_candidate->compile_report,
                                             &options->report_row_storage);
  out_candidate->compile_report.artifact_kind =
      LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE;
  out_candidate->compile_report.target_symbol = options->target_symbol;
  out_candidate->compile_report.backend_name = backend->name;
  out_candidate->compile_report.target_family_name =
      backend->target_family_name;

  iree_status_t status = iree_ok_status();
  if (backend->select_target == NULL || backend->compile == NULL) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL backend '%.*s' is missing required candidate hooks",
        (int)backend->name.size, backend->name.data);
  }

  if (iree_status_is_ok(status)) {
    status = backend->select_target(backend, runtime, allocator,
                                    &out_candidate->hal_target);
  }
  if (iree_status_is_ok(status)) {
    out_candidate->compile_report.target_preset_key =
        out_candidate->hal_target.preset_key;
  }
  if (iree_status_is_ok(status)) {
    status =
        backend->compile(backend, run_module->module,
                         &out_candidate->hal_target, options->target_symbol,
                         options->diagnostic_sink, options->source_resolver,
                         options->max_errors, &out_candidate->compile_report,
                         allocator, &out_candidate->hal_executable);
  }
  out_candidate->compile_report.artifact_kind =
      LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE;
  out_candidate->compile_report.target_symbol = options->target_symbol;
  out_candidate->compile_report.backend_name = backend->name;
  out_candidate->compile_report.target_family_name =
      backend->target_family_name;
  if (iree_status_is_ok(status)) {
    out_candidate->compile_report.target_preset_key =
        out_candidate->hal_target.preset_key;
  }
  if (iree_status_is_ok(status)) {
    out_candidate->compile_report.executable_format =
        out_candidate->hal_executable.executable_format;
    loom_target_compile_report_record_artifact_size(
        &out_candidate->compile_report,
        out_candidate->hal_executable.executable_data.data_length);
  }
  loom_target_compile_report_record_status(&out_candidate->compile_report,
                                           status);
  loom_run_candidate_publish_compile_report(options, out_candidate);
  if (!iree_status_is_ok(status)) {
    loom_run_candidate_deinitialize(out_candidate);
  }
  return status;
}

void loom_run_candidate_deinitialize(loom_run_candidate_t* candidate) {
  if (candidate == NULL) {
    return;
  }
  switch (candidate->kind) {
    case LOOM_RUN_CANDIDATE_KIND_VM_ARCHIVE:
      loom_ireevm_module_archive_deinitialize(&candidate->vm_archive,
                                              candidate->host_allocator);
      break;
    case LOOM_RUN_CANDIDATE_KIND_HAL_EXECUTABLE:
      if (candidate->hal_backend != NULL &&
          candidate->hal_backend->deinitialize_executable != NULL) {
        candidate->hal_backend->deinitialize_executable(
            candidate->hal_backend, &candidate->hal_executable,
            candidate->host_allocator);
      }
      break;
    case LOOM_RUN_CANDIDATE_KIND_UNINITIALIZED:
      break;
  }
  *candidate = (loom_run_candidate_t){0};
}
