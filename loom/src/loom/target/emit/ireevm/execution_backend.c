// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/execution_backend.h"

#include "loom/tooling/execution/candidate.h"
#include "loom/tooling/execution/one_shot.h"
#include "loom/tooling/execution/vm_invocation.h"

static iree_status_t loom_ireevm_execution_backend_run_one_shot(
    const loom_run_execution_backend_t* backend,
    const loom_run_one_shot_request_t* request) {
  IREE_ASSERT_ARGUMENT(backend);
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT_ARGUMENT(request->run_module);
  IREE_ASSERT_ARGUMENT(request->compile_options);
  IREE_ASSERT_ARGUMENT(request->options);
  IREE_ASSERT_ARGUMENT(request->result);

  loom_run_candidate_t candidate = {0};
  loom_run_vm_runtime_t runtime = {0};
  loom_run_vm_invocation_result_t invocation_result = {0};
  loom_run_vm_invocation_result_initialize(request->host_allocator,
                                           &invocation_result);

  iree_status_t status = loom_run_candidate_compile_vm(
      request->run_module, request->compile_options, request->host_allocator,
      &candidate);
  if (iree_status_is_ok(status)) {
    status = loom_run_vm_runtime_initialize(request->host_allocator, &runtime);
  }
  if (iree_status_is_ok(status)) {
    loom_run_vm_invocation_request_t invocation_request = {0};
    loom_run_vm_invocation_request_initialize(&invocation_request);
    invocation_request.runtime = &runtime;
    invocation_request.archive = &candidate.vm_archive;
    invocation_request.options = request->options->vm_options;
    status = loom_run_vm_invocation_run(
        &invocation_request, request->host_allocator, &invocation_result);
  }
  if (iree_status_is_ok(status)) {
    request->result->exit_code = invocation_result.exit_code;
    status = iree_string_builder_append_string(
        &request->result->output,
        iree_string_builder_view(&invocation_result.output));
  }

  loom_run_vm_invocation_result_deinitialize(&invocation_result);
  loom_run_vm_runtime_deinitialize(&runtime);
  loom_run_candidate_deinitialize(&candidate);
  return status;
}

const loom_run_execution_backend_t loom_ireevm_execution_backend = {
    .name = IREE_SVL("vm"),
    .flags = LOOM_RUN_EXECUTION_BACKEND_FLAG_VM_OPTIONS,
    .run_one_shot = loom_ireevm_execution_backend_run_one_shot,
};
