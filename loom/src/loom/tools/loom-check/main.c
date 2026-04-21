// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared loom-check command-line implementation.

#include "loom/tools/loom-check/main.h"

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "loom/codegen/low/lower.h"
#include "loom/ops/op_registry.h"
#include "loom/tools/loom-check/check.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/json_output.h"
#include "loom/tools/loom-check/update.h"
#include "loom/util/stream.h"

IREE_FLAG(bool, update, false,
          "Rewrite test files with actual output in the expected\n"
          "section (after // ----). Inserts the separator for non-empty\n"
          "output when absent.\n"
          "Cannot be used with stdin or verify mode.");
IREE_FLAG(bool, verbose, false,
          "Print PASS/FAIL/SKIP for every case, not just failures.");

typedef struct loom_check_json_flag_t {
  bool enabled;
  loom_check_json_output_mode_t output_mode;
} loom_check_json_flag_t;

static const char* loom_check_json_output_mode_name(
    loom_check_json_output_mode_t output_mode) {
  switch (output_mode) {
    case LOOM_CHECK_JSON_OUTPUT_FAILURES:
      return "failures";
    case LOOM_CHECK_JSON_OUTPUT_SUMMARY:
      return "summary";
    case LOOM_CHECK_JSON_OUTPUT_ALL:
      return "all";
  }
  return "unknown";
}

static iree_status_t loom_check_parse_json_flag(iree_string_view_t flag_name,
                                                void* storage,
                                                iree_string_view_t value) {
  (void)flag_name;
  loom_check_json_flag_t* flag = (loom_check_json_flag_t*)storage;
  IREE_ASSERT_ARGUMENT(flag);

  flag->enabled = true;
  if (iree_string_view_is_empty(value) ||
      iree_string_view_equal(value, iree_make_cstring_view("failures"))) {
    flag->output_mode = LOOM_CHECK_JSON_OUTPUT_FAILURES;
  } else if (iree_string_view_equal(value, iree_make_cstring_view("summary"))) {
    flag->output_mode = LOOM_CHECK_JSON_OUTPUT_SUMMARY;
  } else if (iree_string_view_equal(value, iree_make_cstring_view("all"))) {
    flag->output_mode = LOOM_CHECK_JSON_OUTPUT_ALL;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid --json mode '%.*s'; expected failures, summary, or all",
        (int)value.size, value.data);
  }

  return iree_ok_status();
}

static void loom_check_print_json_flag(iree_string_view_t flag_name,
                                       void* storage, FILE* file) {
  const loom_check_json_flag_t* flag = (const loom_check_json_flag_t*)storage;
  IREE_ASSERT_ARGUMENT(flag);
  if (!flag->enabled) {
    fprintf(file, "# --%.*s[=failures|summary|all]\n", (int)flag_name.size,
            flag_name.data);
    return;
  }
  fprintf(file, "--%.*s=%s\n", (int)flag_name.size, flag_name.data,
          loom_check_json_output_mode_name(flag->output_mode));
}

static loom_check_json_flag_t FLAG_json = {
    .enabled = false,
    .output_mode = LOOM_CHECK_JSON_OUTPUT_FAILURES,
};
IREE_FLAG_CALLBACK(loom_check_parse_json_flag, loom_check_print_json_flag,
                   &FLAG_json, json,
                   "Structured JSON output to stdout. Bare --json is the same\n"
                   "as --json=failures. Modes: failures, summary, all.");

iree_status_t loom_check_register_production_context(void* user_data,
                                                     loom_context_t* context) {
  (void)user_data;
  return loom_op_registry_register_all_dialects(context);
}

//===----------------------------------------------------------------------===//
// Outcome formatting
//===----------------------------------------------------------------------===//

// ANSI color codes for terminal output. Disabled when stderr is not a tty.
static const char* loom_check_color_pass(void) {
  return isatty(fileno(stderr)) ? "\033[32m" : "";
}
static const char* loom_check_color_fail(void) {
  return isatty(fileno(stderr)) ? "\033[31m" : "";
}
static const char* loom_check_color_skip(void) {
  return isatty(fileno(stderr)) ? "\033[33m" : "";
}
static const char* loom_check_color_reset(void) {
  return isatty(fileno(stderr)) ? "\033[0m" : "";
}

static const char* loom_check_outcome_label(loom_check_outcome_t outcome,
                                            bool xfail) {
  if (outcome == LOOM_CHECK_SKIP) {
    return "SKIP";
  }
  if (outcome == LOOM_CHECK_PASS) {
    return xfail ? "XFAIL" : "PASS";
  }
  return xfail ? "XPASS" : "FAIL";
}

// Prints the case header: filename :: case N [mode] OUTCOME.
static void loom_check_print_case_header(iree_string_view_t filename,
                                         iree_host_size_t case_index,
                                         const loom_check_case_t* test_case,
                                         const loom_check_result_t* result) {
  const char* color = loom_check_color_fail();
  if (result->final_outcome == LOOM_CHECK_PASS) {
    color = loom_check_color_pass();
  } else if (result->final_outcome == LOOM_CHECK_SKIP) {
    color = loom_check_color_skip();
  }
  const char* label =
      loom_check_outcome_label(result->final_outcome, test_case->xfail);

  fprintf(stderr, "%s%s%s ", color, label, loom_check_color_reset());
  fprintf(stderr, "%.*s :: case %zu", (int)filename.size, filename.data,
          case_index + 1);
  fprintf(stderr, " [%s", loom_check_mode_name(test_case->mode));
  if (test_case->mode == LOOM_CHECK_MODE_PASS) {
    fprintf(stderr, " %.*s", (int)test_case->pipeline.size,
            test_case->pipeline.data);
  } else if (test_case->mode == LOOM_CHECK_MODE_FORMAT) {
    fprintf(stderr, " %.*s", (int)test_case->format_target.size,
            test_case->format_target.data);
  } else if (test_case->mode == LOOM_CHECK_MODE_EMIT) {
    fprintf(stderr, " %.*s", (int)test_case->emit_target.size,
            test_case->emit_target.data);
  } else if (test_case->mode == LOOM_CHECK_MODE_RUN) {
    fprintf(stderr, " %.*s", (int)test_case->run_arguments.size,
            test_case->run_arguments.data);
  }
  fprintf(stderr, "]\n");
}

// File writing for --update mode (the reconstruction logic is in update.h).
static iree_status_t loom_check_write_updates(
    iree_string_view_t path, iree_string_view_t original_source,
    const loom_check_file_t* file, const loom_check_case_update_t* updates,
    iree_allocator_t allocator) {
  iree_string_builder_t new_source;
  iree_string_builder_initialize(allocator, &new_source);

  iree_host_size_t update_count = 0;
  iree_status_t status = loom_check_apply_updates(
      original_source, file, updates, &new_source, &update_count);

  if (iree_status_is_ok(status) && update_count > 0) {
    FILE* fp = fopen(path.data, "wb");
    if (!fp) {
      status = iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                                "failed to open '%.*s' for writing",
                                (int)path.size, path.data);
    } else {
      fwrite(iree_string_builder_buffer(&new_source), 1,
             iree_string_builder_size(&new_source), fp);
      fclose(fp);
      fprintf(stderr, "updated %zu case%s in %.*s\n", update_count,
              update_count == 1 ? "" : "s", (int)path.size, path.data);
    }
  }

  iree_string_builder_deinitialize(&new_source);
  return status;
}

//===----------------------------------------------------------------------===//
// File processing
//===----------------------------------------------------------------------===//

// Processes a single source buffer: parses test cases, executes each one,
// and reports results. Optionally applies --update to rewrite expected
// sections.
static iree_status_t loom_check_process_file(
    iree_string_view_t filename, iree_string_view_t source, bool is_stdin,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    iree_host_size_t* pass_count, iree_host_size_t* fail_count,
    iree_host_size_t* skip_count) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);

  loom_check_file_t file = {0};
  iree_status_t status = loom_check_parse(source, &arena, &file);

  loom_check_file_report_t report = {0};
  if (iree_status_is_ok(status)) {
    status = loom_check_file_report_initialize(&file, &arena, &report);
  }

  // Allocate update tracking if --update is requested.
  loom_check_case_update_t* updates = NULL;
  if (iree_status_is_ok(status) && FLAG_update) {
    if (is_stdin) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--update cannot be used with stdin");
    } else {
      updates =
          (loom_check_case_update_t*)calloc(file.case_count, sizeof(*updates));
      if (!updates) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "failed to allocate update tracking");
      }
    }
  }

  // Execute each case. Results are allocated per-case and cleaned up
  // after reporting (or after --update).
  loom_check_result_t* results = NULL;
  if (iree_status_is_ok(status) && file.case_count > 0) {
    results = (loom_check_result_t*)calloc(file.case_count, sizeof(*results));
    if (!results) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to allocate result array");
    }
  }

  iree_host_size_t initialized_result_count = 0;
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < file.case_count;
       ++i) {
    const loom_check_case_t* test_case = &file.cases[i];

    loom_check_result_initialize(allocator, &results[i]);
    ++initialized_result_count;
    status =
        loom_check_execute_case(test_case, i, &report, filename, environment,
                                context, block_pool, allocator, &results[i]);
    if (!iree_status_is_ok(status)) {
      // Infrastructure failure — bail.
      break;
    }

    // Report outcome.
    if (FLAG_verbose || results[i].final_outcome == LOOM_CHECK_FAIL) {
      loom_check_print_case_header(filename, i, test_case, &results[i]);
    }
    if ((results[i].final_outcome == LOOM_CHECK_FAIL ||
         (FLAG_verbose && results[i].final_outcome == LOOM_CHECK_SKIP)) &&
        results[i].detail.size > 0) {
      fprintf(stderr, "%.*s", (int)results[i].detail.size,
              results[i].detail.buffer);
    }

    if (results[i].final_outcome == LOOM_CHECK_PASS) {
      ++(*pass_count);
    } else if (results[i].final_outcome == LOOM_CHECK_SKIP) {
      ++(*skip_count);
    } else {
      ++(*fail_count);
    }

    // Track update info for JSON code actions and --update.
    bool wants_json_case =
        FLAG_json.enabled &&
        (FLAG_json.output_mode == LOOM_CHECK_JSON_OUTPUT_ALL ||
         (FLAG_json.output_mode == LOOM_CHECK_JSON_OUTPUT_FAILURES &&
          results[i].final_outcome == LOOM_CHECK_FAIL));
    if ((wants_json_case || updates) && results[i].has_actual_output) {
      iree_string_view_t comparable_expected = test_case->expected;
      if (results[i].expected_output_defaults_to_empty &&
          !test_case->has_expected_section) {
        comparable_expected = iree_string_view_empty();
      }
      iree_string_view_t stripped_expected_trimmed =
          iree_string_view_trim(comparable_expected);
      iree_string_view_t actual_output =
          iree_string_builder_view(&results[i].actual_output);
      iree_string_view_t actual_trimmed = iree_string_view_trim(actual_output);
      bool delete_expected_section =
          results[i].expected_output_defaults_to_empty &&
          test_case->has_expected_section &&
          iree_string_view_is_empty(actual_trimmed);
      if (delete_expected_section ||
          !iree_string_view_equal(stripped_expected_trimmed, actual_trimmed)) {
        if (wants_json_case) {
          status = loom_check_build_update_edit(
              source, test_case, actual_output,
              results[i].expected_output_defaults_to_empty,
              &results[i].update_edit.text, &results[i].update_edit.value);
          if (iree_status_is_ok(status)) {
            results[i].update_edit.present = true;
          }
        }
        if (iree_status_is_ok(status) && updates) {
          updates[i].needs_update = true;
          updates[i].actual_output = actual_output;
          updates[i].empty_output_omits_expected_section =
              results[i].expected_output_defaults_to_empty;
          updates[i].input_end = test_case->input.data + test_case->input.size;
          if (test_case->has_expected_section) {
            updates[i].expected_start = test_case->expected.data;
            updates[i].expected_end =
                test_case->expected.data + test_case->expected.size;
          }
        }
      }
    }
  }

  // Apply --update if any cases need it.
  if (iree_status_is_ok(status) && updates) {
    bool any_updates = false;
    for (iree_host_size_t i = 0; i < file.case_count; ++i) {
      if (updates[i].needs_update) {
        any_updates = true;
        break;
      }
    }
    if (any_updates) {
      status =
          loom_check_write_updates(filename, source, &file, updates, allocator);
    }
  }

  // Emit --json output to stdout (after execution, before cleanup).
  if (iree_status_is_ok(status) && FLAG_json.enabled) {
    loom_output_stream_t stdout_stream;
    loom_output_stream_for_file(stdout, &stdout_stream);
    status = loom_check_json_write_file_result(
        filename, &file, &report, results, *pass_count, *fail_count,
        *skip_count, FLAG_json.output_mode, &stdout_stream);
  }

  // Clean up results.
  if (results) {
    for (iree_host_size_t i = 0; i < initialized_result_count; ++i) {
      loom_check_result_deinitialize(&results[i]);
    }
    free(results);
  }
  free(updates);
  iree_arena_deinitialize(&arena);
  return status;
}

//===----------------------------------------------------------------------===//
// File reading
//===----------------------------------------------------------------------===//

// Reads a source from stdin or a file path, then processes it.
// |path| is "-" or empty for stdin, otherwise a filesystem path.
static iree_status_t loom_check_read_and_process(
    iree_string_view_t path, const loom_check_environment_t* environment,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator, iree_host_size_t* pass_count,
    iree_host_size_t* fail_count, iree_host_size_t* skip_count) {
  bool is_stdin = iree_string_view_is_empty(path) ||
                  iree_string_view_equal(path, iree_make_cstring_view("-"));

  iree_io_file_contents_t* contents = NULL;
  iree_status_t status;
  if (is_stdin) {
    status = iree_io_file_contents_read_stdin(host_allocator, &contents);
  } else {
    status = iree_io_file_contents_read(path, host_allocator, &contents);
  }

  if (iree_status_is_ok(status)) {
    iree_string_view_t source = {
        .data = (const char*)contents->const_buffer.data,
        .size = contents->const_buffer.data_length,
    };
    iree_string_view_t filename =
        is_stdin ? iree_make_cstring_view("<stdin>") : path;
    status = loom_check_process_file(filename, source, is_stdin, environment,
                                     context, block_pool, host_allocator,
                                     pass_count, fail_count, skip_count);
  }

  iree_io_file_contents_free(contents);
  return status;
}

//===----------------------------------------------------------------------===//
// Entry points
//===----------------------------------------------------------------------===//

int loom_check_main(int argc, char** argv,
                    const loom_check_environment_t* base_environment) {
  if (!base_environment) {
    fprintf(stderr, "loom-check environment is required\n");
    return 1;
  }

  iree_flags_set_usage(
      "loom-check",
      "Test runner for .loom-test check files.\n"
      "\n"
      "Parses .loom-test files into cases, executes each case according to "
      "its\n"
      "mode directive, and reports pass/fail/skip results with diffs or\n"
      "diagnostic details on failure. Use .loom for ordinary Loom IR files.\n"
      "\n"
      "Usage:\n"
      "  loom-check [flags] [file]\n"
      "  cat test.loom-test | loom-check\n"
      "\n"
      "Modes (set via // RUN: directive, default is roundtrip):\n"
      "  roundtrip   Parse, print, compare against expected output.\n"
      "  verify      Parse, verify, match diagnostics against annotations.\n"
      "  pass <p>    Parse, run pass pipeline <p>, print, compare.\n"
      "  format <f>  Parse, convert to format <f>, convert back, compare.\n"
      "  emit <t>    Parse, lower to target output <t>, print, compare.\n"
      "              Core targets include liveness-json, low-schedule-json,\n"
      "              low-allocation-json, low-packet-json,\n"
      "              low-descriptor-manifest, target-low-registry-manifest,\n"
      "              target-coverage-manifest, and source-low. source-low\n"
      "              accepts output=module|low|none and\n"
      "              diagnostics=none|memory|all. Linked providers may add\n"
      "              more.\n"
      "  run <args>  Execute input with a linked run provider and compare "
      "output.\n"
      "\n"
      "File format:\n"
      "  A .loom-test file contains one or more cases separated by // ====.\n"
      "  Each case has directives at the top, then input IR, and\n"
      "  optionally a // ---- separator followed by expected output.\n"
      "  When // ---- is absent, the expected output equals the input\n"
      "  (round-trip identity test).\n"
      "\n"
      "  A // RUN: directive before the first // ==== sets the file-level\n"
      "  default mode. Cases without their own // RUN: inherit from it.\n"
      "\n"
      "  Directives:\n"
      "    // RUN: <mode> [args]    Set the test mode (one per case).\n"
      "    // REQUIRES: <name>[, ...] Skip when requirements are unavailable.\n"
      "    // XFAIL: <reason>       Mark as expected failure.\n"
      "    Known REQUIRES names: loom-check-test-unavailable and names from "
      "providers linked\n"
      "    into this runner.\n"
      "\n"
      "  Annotations (verify mode):\n"
      "    // ERROR: DOMAIN/CODE \"substring\"\n"
      "    // ERROR@+1: PARSE/006\n"
      "    // WARNING@-2: \"some message\"\n"
      "    // REMARK: TYPE\n"
      "    Domain and code are optional (omit to match any).\n"
      "    @+N/@-N targets a line relative to the annotation.\n"
      "\n"
      "Examples:\n"
      "  # Round-trip: print output must match input exactly.\n"
      "  echo '// RUN: roundtrip\n"
      "  func.def @f() {\n"
      "  }' | loom-check\n"
      "\n"
      "  # Verify: parse error must match the annotation.\n"
      "  echo '// RUN: verify\n"
      "  // ERROR@+1: PARSE/006\n"
      "  bogus.nonexistent' | loom-check\n"
      "\n"
      "Exit code is 0 when all cases pass, 1 if any fail.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, host_allocator, &block_pool);

  // Initialize context with the dialects selected by this loom-check binary.
  loom_context_t context;
  loom_context_initialize(host_allocator, &context);
  iree_status_t status = iree_ok_status();
  if (argc > 2) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "loom-check accepts at most one input file or '-' for stdin; got %d "
        "inputs",
        argc - 1);
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_context_initialize(base_environment, &context);
  }

  iree_host_size_t pass_count = 0;
  iree_host_size_t fail_count = 0;
  iree_host_size_t skip_count = 0;

  if (iree_status_is_ok(status)) {
    loom_check_environment_t environment = *base_environment;
    if (argc < 2) {
      // No positional args: read from stdin.
      status = loom_check_read_and_process(
          iree_string_view_empty(), &environment, &context, &block_pool,
          host_allocator, &pass_count, &fail_count, &skip_count);
    } else {
      status = loom_check_read_and_process(
          iree_make_cstring_view(argv[1]), &environment, &context, &block_pool,
          host_allocator, &pass_count, &fail_count, &skip_count);
    }
  }

  // Print summary.
  iree_host_size_t total = pass_count + fail_count + skip_count;
  if (total > 0 && iree_status_is_ok(status)) {
    fprintf(stderr, "\n%s%zu passed%s", loom_check_color_pass(), pass_count,
            loom_check_color_reset());
    if (fail_count > 0) {
      fprintf(stderr, ", %s%zu failed%s", loom_check_color_fail(), fail_count,
              loom_check_color_reset());
    }
    if (skip_count > 0) {
      fprintf(stderr, ", %s%zu skipped%s", loom_check_color_skip(), skip_count,
              loom_check_color_reset());
    }
    fprintf(stderr, " (%zu total)\n", total);
  }

  bool had_error = !iree_status_is_ok(status);
  if (had_error) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }

  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);

  if (had_error || fail_count > 0) {
    return 1;
  }
  return 0;
}

enum {
  LOOM_CHECK_PROVIDER_DESCRIPTOR_SET_PROVIDER_CAPACITY = 256,
  LOOM_CHECK_PROVIDER_TARGET_BUNDLE_CAPACITY = 256,
  LOOM_CHECK_PROVIDER_LOW_LOWER_POLICY_CAPACITY = 128,
  LOOM_CHECK_PROVIDER_LOW_LEGALITY_PROVIDER_CAPACITY = 64,
  LOOM_CHECK_PROVIDER_EMIT_PROVIDER_CAPACITY = 64,
  LOOM_CHECK_PROVIDER_RUN_PROVIDER_CAPACITY = 64,
  LOOM_CHECK_PROVIDER_REQUIREMENT_PROVIDER_CAPACITY = 64,
  LOOM_CHECK_PROVIDER_COVERAGE_PROVIDER_CAPACITY = 64,
};

typedef struct loom_check_provider_environment_state_t {
  // Provider table selected by the linked binary or embedding.
  const loom_check_provider_set_t* provider_set;
  // Descriptor-set provider scratch table assembled on demand.
  loom_low_descriptor_set_provider_t descriptor_set_providers
      [LOOM_CHECK_PROVIDER_DESCRIPTOR_SET_PROVIDER_CAPACITY];
  // Number of entries in |descriptor_set_providers|.
  iree_host_size_t descriptor_set_provider_count;
  // Target bundle scratch table assembled on demand.
  const loom_target_bundle_t*
      target_bundles[LOOM_CHECK_PROVIDER_TARGET_BUNDLE_CAPACITY];
  // Number of entries in |target_bundles|.
  iree_host_size_t target_bundle_count;
  // Source-to-low policy scratch table assembled on demand.
  loom_low_lower_policy_registry_entry_t
      low_lower_policy_entries[LOOM_CHECK_PROVIDER_LOW_LOWER_POLICY_CAPACITY];
  // Number of entries in |low_lower_policy_entries|.
  iree_host_size_t low_lower_policy_entry_count;
  // Target-low source legality provider table assembled once for the
  // environment.
  const loom_target_low_legality_provider_t* low_legality_providers
      [LOOM_CHECK_PROVIDER_LOW_LEGALITY_PROVIDER_CAPACITY];
  // Number of entries in |low_legality_providers|.
  iree_host_size_t low_legality_provider_count;
  // Emit provider table assembled once for the environment.
  const loom_check_emit_provider_t*
      emit_providers[LOOM_CHECK_PROVIDER_EMIT_PROVIDER_CAPACITY];
  // Number of entries in |emit_providers|.
  iree_host_size_t emit_provider_count;
  // RUN: run provider table assembled once for the environment.
  const loom_check_run_provider_t*
      run_providers[LOOM_CHECK_PROVIDER_RUN_PROVIDER_CAPACITY];
  // Number of entries in |run_providers|.
  iree_host_size_t run_provider_count;
  // Requirement provider table assembled once for the environment.
  const loom_check_requirement_provider_t*
      requirement_providers[LOOM_CHECK_PROVIDER_REQUIREMENT_PROVIDER_CAPACITY];
  // Number of entries in |requirement_providers|.
  iree_host_size_t requirement_provider_count;
  // Coverage provider table assembled once for the environment.
  const loom_target_coverage_provider_t*
      coverage_providers[LOOM_CHECK_PROVIDER_COVERAGE_PROVIDER_CAPACITY];
  // Number of entries in |coverage_providers|.
  iree_host_size_t coverage_provider_count;
} loom_check_provider_environment_state_t;

static iree_status_t loom_check_provider_validate(
    const loom_check_provider_t* provider, iree_host_size_t provider_index) {
  if (provider == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check provider %" PRIhsz " is null",
                            provider_index);
  }
  if (iree_string_view_is_empty(iree_string_view_trim(provider->name))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check provider %" PRIhsz " has no name",
                            provider_index);
  }
  if (provider->emit_provider_count != 0 && provider->emit_providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check provider '%.*s' has no emit provider "
                            "table",
                            (int)provider->name.size, provider->name.data);
  }
  if (provider->run_provider_count != 0 && provider->run_providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check provider '%.*s' has no run provider "
                            "table",
                            (int)provider->name.size, provider->name.data);
  }
  if (provider->requirement_provider_count != 0 &&
      provider->requirement_providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check provider '%.*s' has no requirement "
                            "provider table",
                            (int)provider->name.size, provider->name.data);
  }
  if (provider->coverage_provider_count != 0 &&
      provider->coverage_providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check provider '%.*s' has no coverage "
                            "provider table",
                            (int)provider->name.size, provider->name.data);
  }
  IREE_RETURN_IF_ERROR(loom_target_low_legality_provider_list_verify(
      provider->low_legality_provider_list));
  return iree_ok_status();
}

static iree_status_t loom_check_provider_append_low_legality_providers(
    loom_check_provider_environment_state_t* state,
    const loom_check_provider_t* provider) {
  if (state->low_legality_provider_count +
          provider->low_legality_provider_list.count >
      IREE_ARRAYSIZE(state->low_legality_providers)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "loom-check low legality provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->low_legality_provider_list.count;
       ++i) {
    state->low_legality_providers[state->low_legality_provider_count++] =
        provider->low_legality_provider_list.values[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_check_provider_append_emit_providers(
    loom_check_provider_environment_state_t* state,
    const loom_check_provider_t* provider) {
  if (state->emit_provider_count + provider->emit_provider_count >
      IREE_ARRAYSIZE(state->emit_providers)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "loom-check emit provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->emit_provider_count; ++i) {
    state->emit_providers[state->emit_provider_count++] =
        provider->emit_providers[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_check_provider_append_run_providers(
    loom_check_provider_environment_state_t* state,
    const loom_check_provider_t* provider) {
  if (state->run_provider_count + provider->run_provider_count >
      IREE_ARRAYSIZE(state->run_providers)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "loom-check run provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->run_provider_count; ++i) {
    state->run_providers[state->run_provider_count++] =
        provider->run_providers[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_check_provider_append_requirement_providers(
    loom_check_provider_environment_state_t* state,
    const loom_check_provider_t* provider) {
  if (state->requirement_provider_count + provider->requirement_provider_count >
      IREE_ARRAYSIZE(state->requirement_providers)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "loom-check requirement provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->requirement_provider_count; ++i) {
    state->requirement_providers[state->requirement_provider_count++] =
        provider->requirement_providers[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_check_provider_append_coverage_providers(
    loom_check_provider_environment_state_t* state,
    const loom_check_provider_t* provider) {
  if (state->coverage_provider_count + provider->coverage_provider_count >
      IREE_ARRAYSIZE(state->coverage_providers)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "loom-check coverage provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->coverage_provider_count; ++i) {
    state->coverage_providers[state->coverage_provider_count++] =
        provider->coverage_providers[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_check_provider_environment_state_initialize(
    const loom_check_provider_set_t* provider_set,
    loom_check_provider_environment_state_t* out_state) {
  IREE_ASSERT_ARGUMENT(out_state);
  *out_state = (loom_check_provider_environment_state_t){
      .provider_set = provider_set,
  };
  if (provider_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check provider set is required");
  }
  if (provider_set->provider_count != 0 && provider_set->providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "loom-check provider table is required");
  }

  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_check_provider_t* provider = provider_set->providers[i];
    IREE_RETURN_IF_ERROR(loom_check_provider_validate(provider, i));
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_low_legality_providers(out_state, provider));
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_emit_providers(out_state, provider));
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_run_providers(out_state, provider));
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_requirement_providers(out_state, provider));
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_coverage_providers(out_state, provider));
  }
  const loom_target_coverage_provider_set_t coverage_provider_set = {
      .providers = out_state->coverage_providers,
      .provider_count = out_state->coverage_provider_count,
  };
  return loom_target_coverage_provider_set_verify(&coverage_provider_set);
}

static iree_status_t loom_check_provider_initialize_low_descriptor_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(out_registry);
  loom_check_provider_environment_state_t* state =
      (loom_check_provider_environment_state_t*)user_data;
  IREE_ASSERT_ARGUMENT(state);
  state->descriptor_set_provider_count = 0;
  state->target_bundle_count = 0;

  const loom_check_provider_set_t* provider_set = state->provider_set;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_check_provider_t* provider = provider_set->providers[i];
    if (provider->initialize_low_descriptor_registry == NULL) {
      continue;
    }
    loom_target_low_descriptor_registry_t provider_registry = {0};
    provider->initialize_low_descriptor_registry(&provider_registry);
    IREE_RETURN_IF_ERROR(loom_target_low_descriptor_registry_append_to_tables(
        &provider_registry, state->descriptor_set_providers,
        IREE_ARRAYSIZE(state->descriptor_set_providers),
        &state->descriptor_set_provider_count, state->target_bundles,
        IREE_ARRAYSIZE(state->target_bundles), &state->target_bundle_count));
  }

  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, state->descriptor_set_providers,
      state->descriptor_set_provider_count, state->target_bundles,
      state->target_bundle_count);
  return iree_ok_status();
}

static iree_status_t loom_check_provider_initialize_low_lower_policy_registry(
    void* user_data, loom_low_lower_policy_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(out_registry);
  loom_check_provider_environment_state_t* state =
      (loom_check_provider_environment_state_t*)user_data;
  IREE_ASSERT_ARGUMENT(state);
  state->low_lower_policy_entry_count = 0;

  const loom_check_provider_set_t* provider_set = state->provider_set;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_check_provider_t* provider = provider_set->providers[i];
    if (provider->initialize_low_lower_policy_registry == NULL) {
      continue;
    }
    loom_low_lower_policy_registry_t provider_registry = {0};
    provider->initialize_low_lower_policy_registry(&provider_registry);
    if (provider_registry.entry_count != 0 &&
        provider_registry.entries == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "loom-check provider '%.*s' has no lower policy "
                              "entry table",
                              (int)provider->name.size, provider->name.data);
    }
    if (state->low_lower_policy_entry_count + provider_registry.entry_count >
        IREE_ARRAYSIZE(state->low_lower_policy_entries)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "loom-check source-to-low policy capacity exceeded");
    }
    for (iree_host_size_t j = 0; j < provider_registry.entry_count; ++j) {
      state->low_lower_policy_entries[state->low_lower_policy_entry_count++] =
          provider_registry.entries[j];
    }
  }

  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, state->low_lower_policy_entries,
      state->low_lower_policy_entry_count);
  return iree_ok_status();
}

int loom_check_provider_main(int argc, char** argv,
                             const loom_check_provider_set_t* provider_set) {
  loom_check_provider_environment_state_t state;
  iree_status_t status =
      loom_check_provider_environment_state_initialize(provider_set, &state);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }

  const loom_check_environment_t environment = {
      .register_context =
          {
              .fn = loom_check_register_production_context,
              .user_data = NULL,
          },
      .initialize_low_descriptor_registry =
          {
              .fn = loom_check_provider_initialize_low_descriptor_registry,
              .user_data = &state,
          },
      .initialize_low_lower_policy_registry =
          {
              .fn = loom_check_provider_initialize_low_lower_policy_registry,
              .user_data = &state,
          },
      .low_legality_provider_list = loom_target_low_legality_provider_list_make(
          state.low_legality_providers, state.low_legality_provider_count),
      .emit_providers =
          {
              .providers = state.emit_providers,
              .provider_count = state.emit_provider_count,
          },
      .run_providers =
          {
              .providers = state.run_providers,
              .provider_count = state.run_provider_count,
          },
      .requirement_providers =
          {
              .providers = state.requirement_providers,
              .provider_count = state.requirement_provider_count,
          },
      .coverage_providers =
          {
              .providers = state.coverage_providers,
              .provider_count = state.coverage_provider_count,
          },
  };
  return loom_check_main(argc, argv, &environment);
}
