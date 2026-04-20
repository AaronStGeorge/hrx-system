// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/candidate.h"

#include "loom/error/diagnostic.h"
#include "loom/target/emit/ireevm/module_compiler.h"

enum {
  LOOM_RUN_DEFAULT_MAX_COMPILE_ERRORS = 20,
};

void loom_run_candidate_compile_options_initialize(
    loom_run_candidate_compile_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_run_candidate_compile_options_t){
      .module_name = IREE_SVL("loom"),
      .diagnostic_sink = {.fn = loom_diagnostic_stderr_sink},
      .max_errors = LOOM_RUN_DEFAULT_MAX_COMPILE_ERRORS,
  };
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

  const loom_ireevm_module_compile_options_t compile_options = {
      .module_name = options->module_name,
      .target_symbol = options->target_symbol,
      .diagnostic_sink = options->diagnostic_sink,
      .source_resolver = options->source_resolver,
      .max_errors = options->max_errors,
  };
  iree_status_t status =
      loom_ireevm_compile_module_archive(run_module->module, &compile_options,
                                         allocator, &out_candidate->vm_archive);
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
    status = backend->compile(backend, run_module->module,
                              &out_candidate->hal_target,
                              options->target_symbol, options->diagnostic_sink,
                              options->source_resolver, options->max_errors,
                              allocator, &out_candidate->hal_executable);
  }
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
