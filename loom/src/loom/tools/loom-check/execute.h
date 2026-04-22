// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Execution engine for loom-check test cases.
//
// Dispatches parsed test cases to mode-specific execution functions
// (roundtrip, verify, pass, format, emit), applies XFAIL inversion, and
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
// The execution engine does not own the loom_context_t or block pool. A
// loom_check_environment_t supplies the dialect registration and target-low
// descriptor registry package selected by each test runner binary or embedding.

#ifndef LOOM_TOOLS_LOOM_CHECK_EXECUTE_H_
#define LOOM_TOOLS_LOOM_CHECK_EXECUTE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/context.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/low_legality.h"
#include "loom/target/low_packet_diagnostics.h"
#include "loom/tools/loom-check/check.h"
#include "loom/tools/loom-check/report.h"
#include "loom/tools/loom-check/update.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_policy_registry_t
    loom_low_lower_policy_registry_t;
typedef struct loom_check_diagnostic_collector_t
    loom_check_diagnostic_collector_t;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// Whether a single test case passed, failed, or was skipped by an unavailable
// declared requirement.
typedef enum loom_check_outcome_e {
  LOOM_CHECK_PASS = 0,
  LOOM_CHECK_FAIL = 1,
  LOOM_CHECK_SKIP = 2,
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

  // Printed IR from roundtrip/pass/format/emit modes. Used by --update to
  // rewrite expected sections in test files when has_actual_output is true.
  iree_string_builder_t actual_output;

  // True when actual_output is the comparable output for this case, even when
  // the output is intentionally empty.
  bool has_actual_output;
  // True when a missing expected section compares as empty output instead of
  // the input section.
  bool expected_output_defaults_to_empty;

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

// Registers dialects into |context| before loom-check finalizes it.
typedef iree_status_t (*loom_check_register_context_fn_t)(
    void* user_data, loom_context_t* context);

// Callback for dialect registration. The callback must not finalize the
// context; loom-check finalizes it after registration succeeds.
typedef struct loom_check_register_context_callback_t {
  // Function that registers the dialect surface selected by this environment.
  loom_check_register_context_fn_t fn;
  // Opaque callback state forwarded to |fn|.
  void* user_data;
} loom_check_register_context_callback_t;

// Initializes a linked target-low descriptor registry package.
typedef iree_status_t (*loom_check_initialize_low_descriptor_registry_fn_t)(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry);

// Callback for the descriptor registry package used by low asm parsing,
// target preset expansion, descriptor-local verification, scheduling, and
// allocation sidecar emission.
typedef struct loom_check_initialize_low_descriptor_registry_callback_t {
  // Function that initializes the selected linked target-low registry package.
  loom_check_initialize_low_descriptor_registry_fn_t fn;
  // Opaque callback state forwarded to |fn|.
  void* user_data;
} loom_check_initialize_low_descriptor_registry_callback_t;

// Initializes a linked source-to-target-low lowering policy registry package.
typedef iree_status_t (*loom_check_initialize_low_lower_policy_registry_fn_t)(
    void* user_data, loom_low_lower_policy_registry_t* out_registry);

// Callback for the lowering policy registry package used by source-to-low emit
// modes. The policy package is intentionally separate from descriptor tables:
// tools may want low parsing/scheduling without linking source lowering.
typedef struct loom_check_initialize_low_lower_policy_registry_callback_t {
  // Function that initializes the selected linked lowering policy package.
  loom_check_initialize_low_lower_policy_registry_fn_t fn;
  // Opaque callback state forwarded to |fn|.
  void* user_data;
} loom_check_initialize_low_lower_policy_registry_callback_t;

typedef struct loom_check_environment_t loom_check_environment_t;
typedef struct loom_check_emit_provider_t loom_check_emit_provider_t;
typedef struct loom_check_run_provider_t loom_check_run_provider_t;
typedef struct loom_check_requirement_provider_t
    loom_check_requirement_provider_t;

// Parsed RUN: run arguments after loom-check shell-like quote handling.
typedef struct loom_check_run_arguments_t {
  // Argument values in RUN-line order.
  const iree_string_view_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_check_run_arguments_t;

// Comparable result produced by a linked RUN: run provider.
typedef struct loom_check_run_result_t {
  // Text written to the provider's stdout stream.
  iree_string_builder_t stdout_text;
  // Text written to the provider's stderr stream.
  iree_string_builder_t stderr_text;
  // Process-style exit code: zero for success and non-zero for failure.
  int exit_code;
} loom_check_run_result_t;

// Prepared RUN: run request passed to a linked provider.
typedef struct loom_check_run_provider_request_t {
  // Source filename reported in diagnostics and emitted target modules.
  iree_string_view_t filename;
  // Parsed test case being executed.
  const loom_check_case_t* test_case;
  // Runner environment that selected this provider.
  const loom_check_environment_t* environment;
  // Parsed RUN: run arguments after the "run" verb.
  const loom_check_run_arguments_t* arguments;
  // Host allocator for transient provider allocations.
  iree_allocator_t host_allocator;
  // Result receiving provider stdout/stderr and exit status.
  loom_check_run_result_t* result;
} loom_check_run_provider_request_t;

// Returns true when |provider| can execute |arguments|.
typedef bool (*loom_check_run_provider_match_fn_t)(
    const loom_check_run_provider_t* provider,
    const loom_check_run_arguments_t* arguments);

// Executes a RUN: run case in-process.
typedef iree_status_t (*loom_check_run_provider_execute_fn_t)(
    const loom_check_run_provider_t* provider,
    const loom_check_run_provider_request_t* request);

// Appends provider-owned runner names to a diagnostic list.
typedef iree_status_t (*loom_check_run_provider_append_names_fn_t)(
    const loom_check_run_provider_t* provider, iree_string_builder_t* builder);

// In-process RUN: run provider linked into a loom-check runner.
struct loom_check_run_provider_t {
  // Human-readable provider name used for debugging and diagnostics.
  iree_string_view_t name;
  // Returns true when this provider owns the RUN: run argument set.
  loom_check_run_provider_match_fn_t match;
  // Executes the RUN: run case and fills the provider result.
  loom_check_run_provider_execute_fn_t execute;
  // Appends supported runner names to diagnostic help text.
  loom_check_run_provider_append_names_fn_t append_names;
};

// Registry of optional in-process RUN: run providers linked into a runner.
typedef struct loom_check_run_provider_registry_t {
  // Linked run provider table.
  const loom_check_run_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_check_run_provider_registry_t;

// Matches and consumes an option from parsed RUN: run arguments.
//
// Accepts both "--name=value" and "--name value" forms. When the current
// argument at |*index| is a matching split option this advances |*index| to the
// value argument.
iree_status_t loom_check_run_arguments_take_option_value(
    const loom_check_run_arguments_t* arguments, iree_host_size_t* index,
    iree_string_view_t option_name, iree_string_view_t* out_value,
    bool* out_matched);

// Appends |failure_status| to |result->stderr_text| and consumes the status.
iree_status_t loom_check_run_result_append_status(
    iree_status_t failure_status, loom_check_run_result_t* result);

// Prepared module state passed to a linked emit provider.
typedef struct loom_check_emit_provider_request_t {
  // Full RUN: emit target payload, after the "emit" verb.
  iree_string_view_t emit_target;
  // First token of |emit_target| used to select the provider.
  iree_string_view_t target_name;
  // Remaining target-specific options after |target_name|.
  iree_string_view_t target_options;
  // Source filename reported in diagnostics and emitted target modules.
  iree_string_view_t filename;
  // Parsed test case being executed.
  const loom_check_case_t* test_case;
  // Runner environment that selected this provider.
  const loom_check_environment_t* environment;
  // Parsed module after comment stripping.
  loom_module_t* module;
  // Linked target-low registry visible to this runner.
  const loom_target_low_descriptor_registry_t* low_registry;
  // Diagnostic collector for provider diagnostics.
  loom_check_diagnostic_collector_t* diagnostic_collector;
  // Arena scoped to this emit case for analysis and diagnostics.
  iree_arena_allocator_t* case_arena;
  // Host allocator for transient provider allocations.
  iree_allocator_t host_allocator;
  // Result receiving provider output.
  loom_check_result_t* result;
} loom_check_emit_provider_request_t;

// Returns true when |provider| owns emit targets named |target_name|.
typedef bool (*loom_check_emit_provider_match_fn_t)(
    const loom_check_emit_provider_t* provider, iree_string_view_t target_name);

// Checks provider-specific REQUIRES declarations before execution.
typedef iree_status_t (*loom_check_emit_provider_check_requirements_fn_t)(
    const loom_check_emit_provider_t* provider,
    const loom_check_case_t* test_case, loom_check_result_t* result,
    bool* out_continue_execution);

// Emits the provider-owned comparable output for |request|.
typedef iree_status_t (*loom_check_emit_provider_execute_fn_t)(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request);

// Appends provider-owned emit target names to a diagnostic list.
typedef iree_status_t (*loom_check_emit_provider_append_names_fn_t)(
    const loom_check_emit_provider_t* provider, iree_string_builder_t* builder);

// Emit namespace provider linked into a loom-check runner. Providers own
// target-specific target syntax, requirement preflight, and emission.
struct loom_check_emit_provider_t {
  // Human-readable provider name used for debugging and ownership comments.
  iree_string_view_t name;
  // Returns true when this provider owns an emit target name.
  loom_check_emit_provider_match_fn_t match;
  // Checks provider-specific REQUIRES declarations for an emit case.
  loom_check_emit_provider_check_requirements_fn_t check_requirements;
  // Emits provider-owned comparable output.
  loom_check_emit_provider_execute_fn_t execute;
  // Appends supported emit target names to diagnostic help text.
  loom_check_emit_provider_append_names_fn_t append_names;
};

// Registry of optional emit providers linked into a runner binary.
typedef struct loom_check_emit_provider_registry_t {
  // Linked emit provider table.
  const loom_check_emit_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_check_emit_provider_registry_t;

// Returns true when |provider| owns |requirement|.
typedef bool (*loom_check_requirement_provider_match_fn_t)(
    const loom_check_requirement_provider_t* provider,
    iree_string_view_t requirement);

// Queries whether |requirement| is available in |environment|.
typedef iree_status_t (*loom_check_requirement_provider_query_fn_t)(
    const loom_check_requirement_provider_t* provider,
    const loom_check_environment_t* environment, iree_string_view_t requirement,
    iree_allocator_t allocator);

// Appends provider-owned requirement names to a diagnostic list.
typedef iree_status_t (*loom_check_requirement_provider_append_names_fn_t)(
    const loom_check_requirement_provider_t* provider,
    iree_string_builder_t* builder);

// Requirement namespace provider linked into a loom-check runner. Providers own
// availability probing for external tools, devices, target runners, and
// target-specific feature requirements.
struct loom_check_requirement_provider_t {
  // Human-readable provider name used for debugging and ownership comments.
  iree_string_view_t name;
  // Returns true when this provider owns a requirement name.
  loom_check_requirement_provider_match_fn_t match;
  // Queries availability for a requirement owned by this provider.
  loom_check_requirement_provider_query_fn_t query;
  // Appends supported requirement names to diagnostic help text.
  loom_check_requirement_provider_append_names_fn_t append_names;
};

// Registry of optional requirement providers linked into a runner binary.
typedef struct loom_check_requirement_provider_registry_t {
  // Linked requirement provider table.
  const loom_check_requirement_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_check_requirement_provider_registry_t;

// Execution environment supplied by each loom-check binary or embedding.
struct loom_check_environment_t {
  // Dialect registration callback for the IR surface accepted by this runner.
  loom_check_register_context_callback_t register_context;
  // Target-low registry callback for descriptor-backed low IR operations.
  loom_check_initialize_low_descriptor_registry_callback_t
      initialize_low_descriptor_registry;
  // Source-to-low lowering policy callback for emit modes that lower source IR
  // into descriptor-backed low IR.
  loom_check_initialize_low_lower_policy_registry_callback_t
      initialize_low_lower_policy_registry;
  // Optional target-low source legality providers linked into this runner.
  loom_target_low_legality_provider_list_t low_legality_provider_list;
  // Optional target-low packet diagnostic providers linked into this runner.
  loom_target_low_packet_diagnostic_provider_list_t
      low_packet_diagnostic_provider_list;
  // Optional emit providers linked into this runner.
  loom_check_emit_provider_registry_t emit_providers;
  // Optional in-process RUN: run providers linked into this runner.
  loom_check_run_provider_registry_t run_providers;
  // Optional requirement providers linked into this runner.
  loom_check_requirement_provider_registry_t requirement_providers;
};

// Returns the linked emit provider for |target_name|, or NULL when none owns
// it.
const loom_check_emit_provider_t* loom_check_environment_lookup_emit_provider(
    const loom_check_environment_t* environment,
    iree_string_view_t target_name);

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

// Registers the dialects selected by |environment|, then finalizes the context.
// The context must have been initialized with loom_context_initialize() before
// calling this.
iree_status_t loom_check_context_initialize(
    const loom_check_environment_t* environment, loom_context_t* context);

// Initializes the target-low descriptor registry selected by |environment|.
iree_status_t loom_check_environment_initialize_low_descriptor_registry(
    const loom_check_environment_t* environment,
    loom_target_low_descriptor_registry_t* out_registry);

// Initializes the source-to-target-low lowering policy registry selected by
// |environment|.
iree_status_t loom_check_environment_initialize_low_lower_policy_registry(
    const loom_check_environment_t* environment,
    loom_low_lower_policy_registry_t* out_registry);

// Executes a single test case: checks declared environment requirements,
// dispatches to the mode-specific function, then applies XFAIL inversion to
// IR/test-subject outcomes. Requirement harness failures and skips are final
// outcomes and are not hidden by XFAIL.
//
// |filename| is passed through to the parser for diagnostic source
// locations. Infrastructure errors (OOM, missing vtables) propagate as
// non-ok status. Test failures (mismatch, unmatched annotations) set
// raw_outcome = FAIL and return iree_ok_status().
iree_status_t loom_check_execute_case(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result);

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
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result);

// Strips comments from input, parses, runs the pass pipeline specified
// in test_case->pipeline, verifies the transformed module, prints the result,
// and compares against the expected section. Same diff/update behavior as
// roundtrip.
iree_status_t loom_check_execute_pass(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
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

// Strips comments from input, parses, lowers to the target specified in
// test_case->emit_target, writes a comparable target output form, and compares
// against the expected section. Same diff/update behavior as roundtrip.
iree_status_t loom_check_execute_emit(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result);

// Executes the case input through a linked run provider selected by
// test_case->run_arguments, captures stdout/stderr, and compares the captured
// output against the expected section. Same diff/update behavior as roundtrip.
iree_status_t loom_check_execute_run(
    const loom_check_case_t* test_case, iree_string_view_t filename,
    const loom_check_environment_t* environment, iree_allocator_t allocator,
    loom_check_result_t* result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_EXECUTE_H_
