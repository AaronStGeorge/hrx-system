// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic VM archive invocation helpers for Loom execution sessions.

#ifndef LOOM_TOOLING_EXECUTION_VM_INVOCATION_H_
#define LOOM_TOOLING_EXECUTION_VM_INVOCATION_H_

#include "iree/base/api.h"
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
  // Value specs in function ABI order.
  const iree_string_view_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_run_vm_value_specs_t;

typedef struct loom_run_vm_invocation_options_t {
  // Exported function name to invoke. Empty selects the single export.
  iree_string_view_t function_name;
  // Optional default HAL device URI for VM modules that depend on HAL.
  iree_string_view_t default_device_uri;
  // Input value specs parsed before invocation.
  loom_run_vm_value_specs_t inputs;
  // Optional output handling specs. Empty formats all outputs.
  loom_run_vm_value_specs_t outputs;
  // Optional expected output specs compared after invocation.
  loom_run_vm_value_specs_t expected_outputs;
  // Maximum number of output elements to format for human-readable output.
  iree_host_size_t max_output_element_count;
} loom_run_vm_invocation_options_t;

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
