// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared candidate compilation options for Loom execution tools.

#ifndef LOOM_TOOLING_EXECUTION_COMPILE_OPTIONS_H_
#define LOOM_TOOLING_EXECUTION_COMPILE_OPTIONS_H_

#include "iree/base/api.h"
#include "loom/target/compile_report.h"
#include "loom/verify/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_candidate_compile_options_t {
  // VM module name stored in VM bytecode archives. Empty uses "loom".
  iree_string_view_t module_name;
  // Optional function symbol to compile. Empty requires one compatible function
  // entry for the selected backend.
  iree_string_view_t entry_symbol;
  // Diagnostic sink used for verification, lowering, scheduling, and
  // allocation diagnostics.
  loom_diagnostic_sink_t diagnostic_sink;
  // Source resolver used to render carets for op-backed diagnostics.
  loom_source_resolver_t source_resolver;
  // Maximum diagnostics to emit before stopping. Zero uses a conservative
  // default.
  uint32_t max_errors;
  // Optional caller-owned structured compile report to populate.
  loom_target_compile_report_t* report;
  // Optional caller-owned row storage for detailed compile report rows.
  loom_target_compile_report_row_storage_t report_row_storage;
} loom_run_candidate_compile_options_t;

// Initializes compile options with stderr diagnostics and a small error cap.
void loom_run_candidate_compile_options_initialize(
    loom_run_candidate_compile_options_t* out_options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_COMPILE_OPTIONS_H_
