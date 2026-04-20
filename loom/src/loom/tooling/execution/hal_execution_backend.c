// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal_execution_backend.h"

#include "loom/tooling/execution/candidate.h"
#include "loom/tooling/execution/hal_invocation.h"
#include "loom/tooling/execution/hal_runtime.h"

static const loom_run_hal_execution_backend_config_t*
loom_run_hal_execution_backend_config(
    const loom_run_execution_backend_t* backend) {
  if (backend == NULL) {
    return NULL;
  }
  return (const loom_run_hal_execution_backend_config_t*)backend->user_data;
}

static const loom_run_hal_backend_t* loom_run_hal_execution_backend_hal_backend(
    const loom_run_execution_backend_t* backend) {
  const loom_run_hal_execution_backend_config_t* config =
      loom_run_hal_execution_backend_config(backend);
  if (config == NULL) {
    return NULL;
  }
  return config->hal_backend;
}

iree_status_t loom_run_hal_execution_backend_probe(
    const loom_run_execution_backend_t* backend,
    const loom_run_one_shot_probe_request_t* request) {
  IREE_ASSERT_ARGUMENT(backend);
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT_ARGUMENT(request->result);

  const loom_run_hal_backend_t* hal_backend =
      loom_run_hal_execution_backend_hal_backend(backend);
  if (hal_backend == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "execution backend '%.*s' has no HAL backend configuration",
        (int)backend->name.size, backend->name.data);
  }
  if (hal_backend->select_target == NULL ||
      hal_backend->format_target == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL backend '%.*s' is missing required probe hooks",
        (int)hal_backend->name.size, hal_backend->name.data);
  }

  loom_run_hal_runtime_t runtime = {0};
  loom_run_hal_selected_target_t target = {0};
  iree_string_builder_t target_id_builder;
  iree_string_builder_initialize(request->host_allocator, &target_id_builder);

  iree_status_t status = loom_run_hal_runtime_initialize(
      hal_backend, request->host_allocator, &runtime);
  if (iree_status_is_ok(status)) {
    status = hal_backend->select_target(hal_backend, &runtime,
                                        request->host_allocator, &target);
  }
  if (iree_status_is_ok(status)) {
    status =
        hal_backend->format_target(hal_backend, &target, &target_id_builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        &request->result->output,
        "backend: %.*s\nhal backend: %.*s\nhal driver: %.*s\nhal target: "
        "%.*s\n",
        (int)backend->name.size, backend->name.data,
        (int)hal_backend->name.size, hal_backend->name.data,
        (int)hal_backend->hal_driver_name.size,
        hal_backend->hal_driver_name.data,
        (int)iree_string_builder_size(&target_id_builder),
        iree_string_builder_buffer(&target_id_builder));
  }
  if (iree_status_is_ok(status) &&
      !iree_string_view_is_empty(target.preset_key)) {
    status = iree_string_builder_append_format(
        &request->result->output, "hal preset: %.*s\n",
        (int)target.preset_key.size, target.preset_key.data);
  }

  iree_string_builder_deinitialize(&target_id_builder);
  loom_run_hal_runtime_deinitialize(&runtime);
  return status;
}

iree_status_t loom_run_hal_execution_backend_run_one_shot(
    const loom_run_execution_backend_t* backend,
    const loom_run_one_shot_request_t* request) {
  IREE_ASSERT_ARGUMENT(backend);
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT_ARGUMENT(request->run_module);
  IREE_ASSERT_ARGUMENT(request->compile_options);
  IREE_ASSERT_ARGUMENT(request->options);
  IREE_ASSERT_ARGUMENT(request->result);

  const loom_run_hal_backend_t* hal_backend =
      loom_run_hal_execution_backend_hal_backend(backend);
  if (hal_backend == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "execution backend '%.*s' has no HAL backend configuration",
        (int)backend->name.size, backend->name.data);
  }

  loom_run_hal_runtime_t runtime = {0};
  loom_run_candidate_t candidate = {0};
  loom_run_hal_invocation_result_t invocation_result = {0};
  loom_run_hal_invocation_result_initialize(request->host_allocator,
                                            &invocation_result);

  iree_status_t status = loom_run_hal_runtime_initialize(
      hal_backend, request->host_allocator, &runtime);
  if (iree_status_is_ok(status)) {
    status = loom_run_candidate_compile_hal(
        hal_backend, &runtime, request->run_module, request->compile_options,
        request->host_allocator, &candidate);
  }
  if (iree_status_is_ok(status)) {
    loom_run_hal_invocation_request_t invocation_request = {0};
    loom_run_hal_invocation_request_initialize(&invocation_request);
    invocation_request.runtime = &runtime;
    invocation_request.executable = &candidate.hal_executable;
    invocation_request.options = request->options->hal_options;
    invocation_request.bindings = request->options->hal_bindings;
    invocation_request.expected_bindings =
        request->options->expected_hal_bindings;
    invocation_request.max_output_element_count =
        request->options->hal_max_output_element_count;
    status = loom_run_hal_invocation_run(
        &invocation_request, request->host_allocator, &invocation_result);
  }
  if (iree_status_is_ok(status)) {
    request->result->exit_code = invocation_result.exit_code;
    status = iree_string_builder_append_string(
        &request->result->output,
        iree_string_builder_view(&invocation_result.output));
  }

  loom_run_hal_invocation_result_deinitialize(&invocation_result);
  loom_run_candidate_deinitialize(&candidate);
  loom_run_hal_runtime_deinitialize(&runtime);
  return status;
}
