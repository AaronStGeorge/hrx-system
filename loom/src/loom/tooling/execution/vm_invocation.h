// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic VM archive invocation helpers for Loom execution sessions.

#ifndef LOOM_TOOLING_EXECUTION_VM_INVOCATION_H_
#define LOOM_TOOLING_EXECUTION_VM_INVOCATION_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "loom/target/emit/ireevm/module_archive.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_vm_runtime_t {
  // VM instance retaining the runtime type registrations used by invocations.
  iree_vm_instance_t* instance;
} loom_run_vm_runtime_t;

typedef struct loom_run_vm_value_specs_t {
  // Textual value specs in function ABI order.
  const iree_string_view_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_run_vm_value_specs_t;

typedef enum loom_run_vm_output_plan_mode_t {
  // Formats every output value to the result builder.
  LOOM_RUN_VM_OUTPUT_PLAN_MODE_FORMAT_ALL = 0,
  // Writes selected outputs using |output_specs|.
  LOOM_RUN_VM_OUTPUT_PLAN_MODE_WRITE_SPECS = 1,
  // Compares outputs against |expected_outputs|.
  LOOM_RUN_VM_OUTPUT_PLAN_MODE_COMPARE_EXPECTED = 2,
} loom_run_vm_output_plan_mode_t;

typedef struct loom_run_vm_invocation_plan_t {
  // Plan-owned materialized input values in VM function ABI order.
  iree_vm_list_t* inputs;
  // Output handling mode selected for this plan.
  loom_run_vm_output_plan_mode_t output_mode;
  // Borrowed textual output sink specs when |output_mode| is WRITE_SPECS.
  loom_run_vm_value_specs_t output_specs;
  // Plan-owned optional materialized expected output values for comparison.
  iree_vm_list_t* expected_outputs;
  // Maximum number of output elements to format.
  iree_host_size_t max_output_element_count;
} loom_run_vm_invocation_plan_t;

typedef struct loom_run_vm_invocation_options_t {
  // Exported function name to invoke. Empty selects the single export.
  iree_string_view_t function_name;
  // Optional default HAL device URI for VM modules that depend on HAL.
  iree_string_view_t default_device_uri;
  // Textual input value specs in VM function ABI order.
  loom_run_vm_value_specs_t inputs;
  // Optional textual output handling specs. Empty formats all outputs.
  loom_run_vm_value_specs_t outputs;
  // Optional textual expected output specs compared after invocation.
  loom_run_vm_value_specs_t expected_outputs;
  // Maximum number of output elements to format for human-readable output.
  iree_host_size_t max_output_element_count;
} loom_run_vm_invocation_options_t;

typedef struct loom_run_vm_invocation_plan_prepare_request_t {
  // Textual options to materialize into a typed invocation plan.
  const loom_run_vm_invocation_options_t* options;
  // VM argument calling convention fragment for |options->inputs|.
  iree_string_view_t arguments_cconv;
  // VM result calling convention fragment for expected outputs.
  iree_string_view_t results_cconv;
  // Optional HAL device used for buffer transfer while parsing values.
  iree_hal_device_t* device;
  // Optional HAL allocator used when textual values require device buffers.
  iree_hal_allocator_t* device_allocator;
} loom_run_vm_invocation_plan_prepare_request_t;

typedef struct loom_run_vm_invocation_request_t {
  // Initialized VM runtime that owns the instance used for invocation.
  const loom_run_vm_runtime_t* runtime;
  // Compiled VM bytecode archive to load and invoke.
  const loom_ireevm_module_archive_t* archive;
  // Function, input, output, and comparison options.
  loom_run_vm_invocation_options_t options;
} loom_run_vm_invocation_request_t;

typedef struct loom_run_vm_invocation_result_t {
  // Human-readable execution output or comparison diagnostics.
  iree_string_builder_t output;
  // Process-style exit code: zero for success, non-zero for mismatches.
  int exit_code;
} loom_run_vm_invocation_result_t;

// Initializes reusable VM runtime state.
iree_status_t loom_run_vm_runtime_initialize(
    iree_allocator_t allocator, loom_run_vm_runtime_t* out_runtime);

// Releases all resources owned by |runtime|.
void loom_run_vm_runtime_deinitialize(loom_run_vm_runtime_t* runtime);

// Initializes invocation options with a small output formatting cap.
void loom_run_vm_invocation_options_initialize(
    loom_run_vm_invocation_options_t* out_options);

// Initializes an empty VM invocation plan.
void loom_run_vm_invocation_plan_initialize(
    loom_run_vm_invocation_plan_t* out_plan);

// Releases storage owned by |plan|.
void loom_run_vm_invocation_plan_deinitialize(
    loom_run_vm_invocation_plan_t* plan);

// Initializes an invocation request.
void loom_run_vm_invocation_request_initialize(
    loom_run_vm_invocation_request_t* out_request);

// Initializes an invocation result. Must be paired with
// loom_run_vm_invocation_result_deinitialize().
void loom_run_vm_invocation_result_initialize(
    iree_allocator_t allocator, loom_run_vm_invocation_result_t* out_result);

// Releases storage owned by |result|.
void loom_run_vm_invocation_result_deinitialize(
    loom_run_vm_invocation_result_t* result);

// Parses textual invocation options into a reusable typed invocation plan.
iree_status_t loom_run_vm_invocation_plan_prepare_from_specs(
    const loom_run_vm_invocation_plan_prepare_request_t* request,
    iree_allocator_t allocator, loom_run_vm_invocation_plan_t* out_plan);

// Invokes |function| with |plan| and records formatted outputs or expected
// comparison diagnostics in |result|. The plan inputs are cloned before
// invocation so async fences and transfer helpers cannot mutate the plan's list
// container.
iree_status_t loom_run_vm_invocation_invoke_plan(
    const loom_run_vm_invocation_plan_t* plan, iree_vm_context_t* context,
    iree_vm_function_t function, iree_hal_device_t* device,
    iree_hal_allocator_t* device_allocator, iree_allocator_t allocator,
    loom_run_vm_invocation_result_t* result);

// Loads |request->archive| into a fresh VM context, invokes the selected
// function, and records formatted outputs or expected comparison diagnostics in
// |result|.
iree_status_t loom_run_vm_invocation_run(
    const loom_run_vm_invocation_request_t* request, iree_allocator_t allocator,
    loom_run_vm_invocation_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_VM_INVOCATION_H_
