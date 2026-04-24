// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile report capture helpers for one-shot execution tools.

#ifndef LOOM_TOOLING_EXECUTION_COMPILE_REPORT_CAPTURE_H_
#define LOOM_TOOLING_EXECUTION_COMPILE_REPORT_CAPTURE_H_

#include "iree/base/api.h"
#include "loom/target/compile_report_format.h"
#include "loom/tooling/execution/compile_options.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Default number of detailed pressure/spill rows captured per category.
  LOOM_RUN_COMPILE_REPORT_DEFAULT_ROW_LIMIT = 8,
};

typedef struct loom_run_compile_report_capture_options_t {
  // Formatting mode requested by the caller.
  loom_target_compile_report_format_mode_t mode;
  // Maximum copied pressure rows and maximum copied spill rows in details mode.
  iree_host_size_t row_limit;
} loom_run_compile_report_capture_options_t;

typedef struct loom_run_compile_report_capture_t {
  // Capture options used to configure and format |report|.
  loom_run_compile_report_capture_options_t options;
  // Host allocator used for optional row arrays.
  iree_allocator_t host_allocator;
  // Compile report populated by candidate compilation.
  loom_target_compile_report_t report;
  // Capture-owned pressure row storage passed to candidate compilation.
  loom_target_compile_report_pressure_row_t* pressure_rows;
  // Capture-owned spill row storage passed to candidate compilation.
  loom_target_compile_report_spill_row_t* spill_rows;
  // Capture-owned source-low row storage passed to candidate compilation.
  loom_target_compile_report_source_low_row_t* source_low_rows;
} loom_run_compile_report_capture_t;

// Initializes capture options with report output disabled.
void loom_run_compile_report_capture_options_initialize(
    loom_run_compile_report_capture_options_t* out_options);

// Parses and stores a compile report mode.
iree_status_t loom_run_compile_report_capture_options_parse_mode(
    iree_string_view_t value,
    loom_run_compile_report_capture_options_t* options);

// Parses and stores the per-category row limit for details mode.
iree_status_t loom_run_compile_report_capture_options_parse_row_limit(
    iree_string_view_t value,
    loom_run_compile_report_capture_options_t* options);

// Allocates optional row storage for |options| and initializes |out_capture|.
iree_status_t loom_run_compile_report_capture_initialize(
    const loom_run_compile_report_capture_options_t* options,
    iree_allocator_t host_allocator,
    loom_run_compile_report_capture_t* out_capture);

// Configures |compile_options| to populate |capture| when capture is enabled.
void loom_run_compile_report_capture_configure_compile_options(
    loom_run_compile_report_capture_t* capture,
    loom_run_candidate_compile_options_t* compile_options);

// Appends the captured report to |builder| when capture is enabled.
iree_status_t loom_run_compile_report_capture_append_text(
    const loom_run_compile_report_capture_t* capture,
    iree_string_builder_t* builder);

// Releases storage owned by |capture|.
void loom_run_compile_report_capture_deinitialize(
    loom_run_compile_report_capture_t* capture);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_COMPILE_REPORT_CAPTURE_H_
