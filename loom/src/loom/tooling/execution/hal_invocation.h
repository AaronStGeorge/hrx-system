// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic HAL executable invocation helpers for Loom execution sessions.

#ifndef LOOM_TOOLING_EXECUTION_HAL_INVOCATION_H_
#define LOOM_TOOLING_EXECUTION_HAL_INVOCATION_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "loom/tooling/execution/hal_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Maximum number of HAL dispatch bindings accepted by the generic runner.
  LOOM_RUN_HAL_MAX_BINDING_COUNT = 64,
};

typedef struct loom_run_hal_invocation_options_t {
  // HAL executable entry point ordinal to dispatch.
  uint32_t entry_point;
  // Dispatch workgroup count in x, y, z order.
  uint32_t workgroup_count[3];
} loom_run_hal_invocation_options_t;

typedef struct loom_run_hal_binding_specs_t {
  // Textual binding value specs in HAL binding ordinal order.
  const iree_string_view_t* values;
  // Calling-convention character for each binding value spec.
  const char* conventions;
  // Number of entries in |values| and |conventions|.
  iree_host_size_t count;
} loom_run_hal_binding_specs_t;

typedef struct loom_run_hal_invocation_plan_t {
  // HAL executable entry point and dispatch geometry.
  loom_run_hal_invocation_options_t options;
  // Plan-owned materialized binding values in HAL binding ordinal order.
  iree_vm_list_t* bindings;
  // Plan-owned optional expected binding values compared after dispatch.
  iree_vm_list_t* expected_bindings;
  // Maximum number of output elements to format.
  iree_host_size_t max_output_element_count;
} loom_run_hal_invocation_plan_t;

typedef struct loom_run_hal_invocation_request_t {
  // Initialized HAL runtime that owns the device used for dispatch.
  const loom_run_hal_runtime_t* runtime;
  // Backend-produced executable bytes to prepare and dispatch.
  const loom_run_hal_executable_t* executable;
  // HAL dispatch entry point and workgroup count.
  loom_run_hal_invocation_options_t options;
  // Textual input/output binding specs parsed before dispatch.
  loom_run_hal_binding_specs_t bindings;
  // Optional textual expected binding specs compared after dispatch.
  loom_run_hal_binding_specs_t expected_bindings;
  // Maximum number of output elements to format for human-readable output.
  iree_host_size_t max_output_element_count;
} loom_run_hal_invocation_request_t;

typedef struct loom_run_hal_invocation_result_t {
  // Human-readable execution output or comparison diagnostics.
  iree_string_builder_t output;
  // Process-style exit code: zero for success, non-zero for mismatches.
  int exit_code;
} loom_run_hal_invocation_result_t;

// Initializes invocation options to entry point 0 and a single workgroup.
void loom_run_hal_invocation_options_initialize(
    loom_run_hal_invocation_options_t* out_options);

// Initializes a request to dispatch entry point 0 over one workgroup.
void loom_run_hal_invocation_request_initialize(
    loom_run_hal_invocation_request_t* out_request);

// Initializes an empty invocation plan.
void loom_run_hal_invocation_plan_initialize(
    loom_run_hal_invocation_plan_t* out_plan);

// Releases storage owned by |plan|.
void loom_run_hal_invocation_plan_deinitialize(
    loom_run_hal_invocation_plan_t* plan);

// Initializes an invocation result. Must be paired with
// loom_run_hal_invocation_result_deinitialize().
void loom_run_hal_invocation_result_initialize(
    iree_allocator_t allocator, loom_run_hal_invocation_result_t* out_result);

// Releases storage owned by |result|.
void loom_run_hal_invocation_result_deinitialize(
    loom_run_hal_invocation_result_t* result);

// Prepares a HAL executable object from backend-produced executable bytes.
iree_status_t loom_run_hal_executable_prepare(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_executable_t* executable,
    iree_hal_executable_t** out_hal_executable);

// Dispatches a prepared HAL executable with |binding_list|.
iree_status_t loom_run_hal_dispatch(
    iree_hal_device_t* device, iree_hal_executable_t* executable,
    iree_vm_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options);

// Prepares and dispatches |executable| through |runtime|.
iree_status_t loom_run_hal_invocation_execute(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_executable_t* executable, iree_vm_list_t* binding_list,
    const loom_run_hal_invocation_options_t* options);

// Parses textual binding specs into a reusable typed invocation plan.
iree_status_t loom_run_hal_invocation_plan_prepare_from_specs(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_invocation_options_t* options,
    const loom_run_hal_binding_specs_t* bindings,
    const loom_run_hal_binding_specs_t* expected_bindings,
    iree_host_size_t max_output_element_count, iree_allocator_t allocator,
    loom_run_hal_invocation_plan_t* out_plan);

// Dispatches |executable| using |plan| and records either formatted outputs or
// expected comparison diagnostics in |result|. The plan bindings are cloned
// before dispatch so host-transfer helpers cannot mutate the plan's list
// container.
iree_status_t loom_run_hal_invocation_run_plan(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_executable_t* executable,
    const loom_run_hal_invocation_plan_t* plan, iree_allocator_t allocator,
    loom_run_hal_invocation_result_t* result);

// Transfers dispatch bindings back to host-visible storage for inspection.
iree_status_t loom_run_hal_transfer_bindings_to_host(
    const loom_run_hal_runtime_t* runtime, iree_vm_list_t* binding_list);

// Parses bindings, dispatches |request->executable|, transfers bindings back
// to host-visible storage, and records either formatted outputs or expected
// comparison diagnostics in |result|.
iree_status_t loom_run_hal_invocation_run(
    const loom_run_hal_invocation_request_t* request,
    iree_allocator_t allocator, loom_run_hal_invocation_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_INVOCATION_H_
