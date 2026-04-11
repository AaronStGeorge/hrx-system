// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Execution engine for loom-check test cases.
//
// Dispatches parsed test cases to mode-specific execution functions
// (roundtrip, verify, pass, format), applies XFAIL inversion, and
// produces structured results with diff output and actual printed IR.
//
// The result type carries test-level verdicts: diffs, annotation match
// failures, and forwarded diagnostic messages. These are fundamentally
// different from loom_diagnostic_t (which represents IR-level errors
// like type mismatches and parse failures). The execution engine uses
// the diagnostic infrastructure internally — parser and verifier
// diagnostics flow through loom_diagnostic_sink_t callbacks — but the
// result aggregates them into a test-framework report.
//
// The execution engine does not own the loom_context_t or block pool.
// A convenience function (loom_check_context_initialize) registers all
// known dialects so callers don't duplicate the registration boilerplate.

#ifndef LOOM_TOOLS_LOOM_CHECK_EXECUTE_H_
#define LOOM_TOOLS_LOOM_CHECK_EXECUTE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/context.h"
#include "loom/tools/loom-check/check.h"
#include "loom/tools/loom-check/report.h"
#include "loom/tools/loom-check/update.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// Whether a single test case passed or failed.
typedef enum loom_check_outcome_e {
  LOOM_CHECK_PASS = 0,
  LOOM_CHECK_FAIL = 1,
} loom_check_outcome_t;

// Result of executing a single test case.
typedef struct loom_check_result_t {
  // Outcome before XFAIL inversion.
  loom_check_outcome_t raw_outcome;
  // Outcome after XFAIL inversion.
  loom_check_outcome_t final_outcome;

  // Human-readable detail: diff output for roundtrip mismatches,
  // unmatched annotations for verify failures, formatted diagnostic
  // messages for parse errors. Empty on PASS.
  iree_string_builder_t detail;

  // Structured diff hunk JSON objects for roundtrip/pass mismatches, separated
  // by ",\n" for direct embedding in a JSON array.
  iree_string_builder_t diff_hunk_json;

  // Number of objects in diff_hunk_json.
  iree_host_size_t diff_hunk_count;

  // Printed IR from roundtrip/pass/format modes. Empty for verify
  // mode. Used by --update to rewrite expected sections in test files.
  iree_string_builder_t actual_output;

  // Machine-readable edit for accepting actual_output into the expected
  // section.
  struct {
    // Whether an update edit is available for this result.
    bool present;
    // Edit kind and source byte range. Valid only when present is true.
    loom_check_update_edit_t value;
    // Replacement text to apply at the edit range. Valid only when present is
    // true.
    iree_string_builder_t text;
  } update_edit;

  // Machine-readable edits for accepting actual verify diagnostics into
  // diagnostic annotation comments. Edit ranges are in the original .loom-test
  // source; apply multiple edits atomically or in descending range order.
  struct {
    // Structured annotation edit JSON objects, separated by ",\n" for direct
    // embedding in a JSON array.
    iree_string_builder_t json;
    // Number of objects in json.
    iree_host_size_t count;
  } annotation_edits;

  // Structured diagnostic JSON objects emitted through the shared
  // loom_diagnostic_json_write_object path, separated by ",\n". This is ready
  // to embed in a JSON array while preserving the full parser/verifier
  // diagnostic shape: source ranges, highlights, related locations, params,
  // field refs, rendered message, and fix hints.
  iree_string_builder_t diagnostic_json;

  // Number of objects in diagnostic_json.
  iree_host_size_t diagnostic_count;
} loom_check_result_t;

// Diagnostic capture shared by loom-check execution modes. The sink appends
// human-readable caret diagnostics to |detail| when non-NULL and appends the
// canonical structured diagnostic JSON object to |result->diagnostic_json|
// when |result| is non-NULL.
typedef struct loom_check_diagnostic_capture_t {
  iree_string_builder_t* detail;
  loom_check_result_t* result;
} loom_check_diagnostic_capture_t;

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

// Initializes a result with the given allocator. Must be paired with
// loom_check_result_deinitialize.
void loom_check_result_initialize(iree_allocator_t allocator,
                                  loom_check_result_t* out_result);

// Releases all resources owned by the result.
void loom_check_result_deinitialize(loom_check_result_t* result);

// Captures one diagnostic for loom-check detail and JSON output. Pass a
// loom_check_diagnostic_capture_t* as user_data.
iree_status_t loom_check_diagnostic_capture_sink(
    void* user_data, const loom_diagnostic_t* diagnostic);

// Records a roundtrip/pass mismatch as both human-readable unified diff text
// in |result->detail| and structured hunk objects in |result->diff_hunk_json|.
iree_status_t loom_check_result_record_diff(iree_string_view_t expected,
                                            iree_string_view_t actual,
                                            iree_allocator_t allocator,
                                            loom_check_result_t* result);

// Registers all dialects used by loom-check with the context and finalizes it.
// The context must have been initialized with loom_context_initialize() before
// calling this.
iree_status_t loom_check_context_initialize(loom_context_t* context);

// Executes a single test case: dispatches to the mode-specific function,
// then applies XFAIL inversion to produce the final outcome.
//
// |filename| is passed through to the parser for diagnostic source
// locations. Infrastructure errors (OOM, missing vtables) propagate as
// non-ok status. Test failures (mismatch, unmatched annotations) set
// raw_outcome = FAIL and return iree_ok_status().
iree_status_t loom_check_execute_case(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_check_result_t* result);

// Strips comments from input, parses, prints, and compares against the
// expected section. On mismatch, appends a unified diff to result->detail
// and copies the printed output to result->actual_output for --update.
iree_status_t loom_check_execute_roundtrip(const loom_check_case_t* test_case,
                                           iree_string_view_t filename,
                                           loom_context_t* context,
                                           iree_arena_block_pool_t* block_pool,
                                           iree_allocator_t allocator,
                                           loom_check_result_t* result);

// Strips comments from input, parses (collecting diagnostics), verifies
// (collecting more diagnostics), then matches collected diagnostics against
// the case's annotations. Unmatched annotations and unexpected diagnostics
// are reported in result->detail.
iree_status_t loom_check_execute_verify(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_check_result_t* result);

// Strips comments from input, parses, runs the pass pipeline specified
// in test_case->pipeline, prints the result, and compares against the
// expected section. Same diff/update behavior as roundtrip.
iree_status_t loom_check_execute_pass(const loom_check_case_t* test_case,
                                      iree_string_view_t filename,
                                      loom_context_t* context,
                                      iree_arena_block_pool_t* block_pool,
                                      iree_allocator_t allocator,
                                      loom_check_result_t* result);

// Strips comments from input, parses, converts to the format specified
// in test_case->format_target (e.g. bytecode), converts back to text,
// and compares against the expected section. Same diff/update behavior
// as roundtrip.
iree_status_t loom_check_execute_format(const loom_check_case_t* test_case,
                                        iree_string_view_t filename,
                                        loom_context_t* context,
                                        iree_arena_block_pool_t* block_pool,
                                        iree_allocator_t allocator,
                                        loom_check_result_t* result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_EXECUTE_H_
