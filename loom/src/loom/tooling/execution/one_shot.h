// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One-shot execution request/result adapters for CLI and lit-style tools.

#ifndef LOOM_TOOLING_EXECUTION_ONE_SHOT_H_
#define LOOM_TOOLING_EXECUTION_ONE_SHOT_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/tooling/execution/hal_invocation.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/execution/vm_invocation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_compile_report_capture_t
    loom_run_compile_report_capture_t;

// Invocation options for the current one-shot run front door.
//
// This intentionally preserves the existing textual VM/HAL input specs used by
// CLI and loom-check adapters. Benchmark/tune hot loops should eventually use
// typed invocation plans instead of this structure.
typedef struct loom_run_one_shot_options_t {
  // VM function, input, output, and comparison options.
  loom_run_vm_invocation_options_t vm_options;
  // HAL entry point and dispatch geometry.
  loom_run_hal_invocation_options_t hal_options;
  // HAL binding specs in dispatch binding order.
  loom_run_hal_binding_specs_t hal_bindings;
  // Optional expected HAL binding specs compared after dispatch.
  loom_run_hal_binding_specs_t expected_hal_bindings;
  // Maximum number of HAL output elements to format.
  iree_host_size_t hal_max_output_element_count;
} loom_run_one_shot_options_t;

typedef struct loom_run_one_shot_result_t {
  // Human-readable output or comparison diagnostics.
  iree_string_builder_t output;
  // Process-style exit code: zero for success, non-zero for mismatches.
  int exit_code;
} loom_run_one_shot_result_t;

struct loom_run_one_shot_probe_request_t {
  // Host allocator for transient probe allocations.
  iree_allocator_t host_allocator;
  // Result receiving human-readable probe output.
  loom_run_one_shot_result_t* result;
};

struct loom_run_one_shot_request_t {
  // Parsed module to compile and invoke.
  loom_run_module_t* run_module;
  // Candidate compile options shared by backends.
  const loom_run_candidate_compile_options_t* compile_options;
  // Invocation options selected by the front door.
  const loom_run_one_shot_options_t* options;
  // Optional compile report capture formatted before backend candidate
  // teardown.
  loom_run_compile_report_capture_t* compile_report_capture;
  // Host allocator for transient execution allocations.
  iree_allocator_t host_allocator;
  // Result receiving human-readable output and exit status.
  loom_run_one_shot_result_t* result;
};

// Initializes one-shot options to match the existing iree-run-loom defaults.
void loom_run_one_shot_options_initialize(
    loom_run_one_shot_options_t* out_options);

// Initializes a one-shot result. Must be paired with
// loom_run_one_shot_result_deinitialize().
void loom_run_one_shot_result_initialize(
    iree_allocator_t allocator, loom_run_one_shot_result_t* out_result);

// Releases storage owned by |result|.
void loom_run_one_shot_result_deinitialize(loom_run_one_shot_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_ONE_SHOT_H_
