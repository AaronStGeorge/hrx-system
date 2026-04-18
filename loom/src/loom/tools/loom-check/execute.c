// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/execute.h"

#include "loom/codegen/low/allocation_pass.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/diagnostic.h"
#include "loom/error/json_sink.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/testing/diff.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/requirements.h"
#include "loom/transforms/branch_fusion.h"
#include "loom/transforms/branch_sink.h"
#include "loom/transforms/canonicalize.h"
#include "loom/transforms/cfg_simplify.h"
#include "loom/transforms/cse.h"
#include "loom/transforms/dce.h"
#include "loom/transforms/kernel_async_legality.h"
#include "loom/transforms/kernel_resources.h"
#include "loom/transforms/licm.h"
#include "loom/transforms/loop_fusion.h"
#include "loom/transforms/pass.h"
#include "loom/transforms/refine_boundaries.h"
#include "loom/transforms/scf_to_cfg.h"
#include "loom/transforms/strip_hints.h"
#include "loom/transforms/symbol_dce.h"
#include "loom/transforms/vector_memory_footprint.h"
#include "loom/transforms/vector_to_scalar.h"
#include "loom/transforms/verify.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

void loom_check_result_initialize(iree_allocator_t allocator,
                                  loom_check_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  iree_string_builder_initialize(allocator, &out_result->detail);
  iree_string_builder_initialize(allocator, &out_result->diff_hunk_json);
  iree_string_builder_initialize(allocator, &out_result->actual_output);
  iree_string_builder_initialize(allocator, &out_result->update_edit.text);
  iree_string_builder_initialize(allocator, &out_result->annotation_edits.json);
  iree_string_builder_initialize(allocator, &out_result->diagnostic_json);
}

void loom_check_result_deinitialize(loom_check_result_t* result) {
  iree_string_builder_deinitialize(&result->diagnostic_json);
  iree_string_builder_deinitialize(&result->annotation_edits.json);
  iree_string_builder_deinitialize(&result->update_edit.text);
  iree_string_builder_deinitialize(&result->actual_output);
  iree_string_builder_deinitialize(&result->diff_hunk_json);
  iree_string_builder_deinitialize(&result->detail);
}

static iree_status_t loom_check_write_diff_hunk_json(
    const loom_diff_result_t* diff, const loom_diff_hunk_t* hunk,
    loom_output_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(diff);
  IREE_ASSERT_ARGUMENT(hunk);
  IREE_ASSERT_ARGUMENT(stream);

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream,
      "\"expected_start_line\": %zu, \"expected_line_count\": %zu, "
      "\"actual_start_line\": %zu, \"actual_line_count\": %zu, \"lines\": [",
      hunk->expected_start_line, hunk->expected_line_count,
      hunk->actual_start_line, hunk->actual_line_count));
  for (iree_host_size_t i = 0; i < hunk->line_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ", "));
    }
    const loom_diff_hunk_line_t* line = &diff->lines[hunk->line_offset + i];
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "{\"kind\": "));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
        stream, loom_diff_hunk_line_kind_name(line->kind)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ", \"text\": "));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, line->text));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]}"));
  return iree_ok_status();
}

static iree_status_t loom_check_write_diff_json_hunks(
    const loom_diff_result_t* diff, loom_check_result_t* result) {
  IREE_ASSERT_ARGUMENT(diff);
  IREE_ASSERT_ARGUMENT(result);

  loom_output_stream_t stream;
  loom_output_stream_for_builder(&result->diff_hunk_json, &stream);
  for (iree_host_size_t i = 0; i < diff->hunk_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\n"));
    }
    IREE_RETURN_IF_ERROR(
        loom_check_write_diff_hunk_json(diff, &diff->hunks[i], &stream));
  }
  result->diff_hunk_count = diff->hunk_count;
  return iree_ok_status();
}

iree_status_t loom_check_result_record_diff(iree_string_view_t expected,
                                            iree_string_view_t actual,
                                            iree_allocator_t allocator,
                                            loom_check_result_t* result) {
  IREE_ASSERT_ARGUMENT(result);

  loom_diff_result_t diff = {0};
  iree_status_t status = loom_diff_compute(
      expected, actual, LOOM_DIFF_DEFAULT_CONTEXT, allocator, &diff);
  if (iree_status_is_ok(status)) {
    status = loom_diff_format_result(&diff, &result->detail);
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_write_diff_json_hunks(&diff, result);
  }
  loom_diff_result_deinitialize(allocator, &diff);
  return status;
}

static iree_status_t loom_check_append_diagnostic_json(
    loom_check_result_t* result, const loom_diagnostic_t* diagnostic) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&result->diagnostic_json, &stream);
  if (result->diagnostic_count > 0) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(&stream, ",\n"));
  }
  IREE_RETURN_IF_ERROR(loom_diagnostic_json_write_object(
      &stream, diagnostic,
      (loom_type_formatter_t){loom_type_format_minimal, NULL}));
  ++result->diagnostic_count;
  return iree_ok_status();
}

iree_status_t loom_check_diagnostic_capture_sink(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_check_diagnostic_capture_t* capture =
      (loom_check_diagnostic_capture_t*)user_data;
  if (capture->detail) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(capture->detail, &stream);
    IREE_RETURN_IF_ERROR(loom_diagnostic_format(diagnostic, &stream));
  }
  if (capture->result) {
    IREE_RETURN_IF_ERROR(
        loom_check_append_diagnostic_json(capture->result, diagnostic));
  }
  return iree_ok_status();
}

iree_status_t loom_check_context_initialize(loom_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_op_registry_register_all_dialects(context));
  return loom_context_finalize(context);
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

iree_status_t loom_check_execute_case(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_check_result_t* result) {
  IREE_ASSERT_ARGUMENT(test_case);
  IREE_ASSERT_ARGUMENT(report);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(result);

  bool continue_execution = true;
  IREE_RETURN_IF_ERROR(loom_check_preflight_requirements(
      test_case, allocator, result, &continue_execution));
  if (!continue_execution) {
    return iree_ok_status();
  }

  switch (test_case->mode) {
    case LOOM_CHECK_MODE_ROUNDTRIP: {
      IREE_RETURN_IF_ERROR(loom_check_execute_roundtrip(
          test_case, filename, context, block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_VERIFY: {
      IREE_RETURN_IF_ERROR(
          loom_check_execute_verify(test_case, case_index, report, filename,
                                    context, block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_PASS: {
      IREE_RETURN_IF_ERROR(
          loom_check_execute_pass(test_case, case_index, report, filename,
                                  context, block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_FORMAT: {
      IREE_RETURN_IF_ERROR(loom_check_execute_format(
          test_case, filename, context, block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_EMIT: {
      IREE_RETURN_IF_ERROR(
          loom_check_execute_emit(test_case, case_index, report, filename,
                                  context, block_pool, allocator, result));
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown check mode: %d", (int)test_case->mode);
  }

  // XFAIL inversion: expected failure that fails is a pass, expected
  // failure that passes is a failure.
  if (test_case->xfail) {
    result->final_outcome = (result->raw_outcome == LOOM_CHECK_FAIL)
                                ? LOOM_CHECK_PASS
                                : LOOM_CHECK_FAIL;
  } else {
    result->final_outcome = result->raw_outcome;
  }

  return iree_ok_status();
}

// Runs a comma-separated pass pipeline on a module.
static iree_status_t loom_check_run_pipeline(
    iree_string_view_t pipeline, iree_arena_block_pool_t* block_pool,
    iree_diagnostic_emitter_t diagnostic_emitter, loom_module_t* module) {
  loom_pass_manager_t manager;
  IREE_RETURN_IF_ERROR(loom_pass_manager_initialize(
      block_pool, 0, iree_allocator_system(), &manager));
  manager.diagnostic_emitter = diagnostic_emitter;

  // Parse comma-separated pass names and add each to the pipeline.
  iree_string_view_t remaining = pipeline;
  while (true) {
    loom_pass_pipeline_entry_spec_t entry = {0};
    bool has_entry = false;
    iree_status_t status =
        loom_pass_pipeline_consume_entry(&remaining, &entry, &has_entry);
    if (!iree_status_is_ok(status)) {
      loom_pass_manager_deinitialize(&manager);
      return status;
    }
    if (!has_entry) break;

    const loom_pass_info_t* info = NULL;
    loom_module_pass_fn_t module_run = NULL;
    loom_function_pass_fn_t function_run = NULL;
    loom_pass_create_fn_t create = NULL;
    loom_pass_destroy_fn_t destroy = NULL;
    if (iree_string_view_equal(entry.name, IREE_SV("canonicalize"))) {
      info = loom_canonicalize_pass_info();
      function_run = loom_canonicalize_run;
      create = loom_canonicalize_create;
    } else if (iree_string_view_equal(entry.name, IREE_SV("dce"))) {
      info = loom_dce_pass_info();
      function_run = loom_dce_run;
    } else if (iree_string_view_equal(entry.name,
                                      IREE_SV("kernel-async-legality"))) {
      info = loom_kernel_async_legality_pass_info();
      function_run = loom_kernel_async_legality_run;
    } else if (iree_string_view_equal(entry.name,
                                      IREE_SV("low-materialize-allocation"))) {
      info = loom_low_materialize_allocation_pass_info();
      function_run = loom_low_materialize_allocation_run;
      create = loom_low_materialize_allocation_create;
    } else if (iree_string_view_equal(entry.name,
                                      IREE_SV("normalize-kernel-resources"))) {
      info = loom_normalize_kernel_resources_pass_info();
      function_run = loom_normalize_kernel_resources_run;
    } else if (iree_string_view_equal(entry.name, IREE_SV("cse"))) {
      info = loom_cse_pass_info();
      function_run = loom_cse_run;
    } else if (iree_string_view_equal(entry.name, IREE_SV("branch-sink"))) {
      info = loom_branch_sink_pass_info();
      function_run = loom_branch_sink_run;
    } else if (iree_string_view_equal(entry.name, IREE_SV("branch-fusion"))) {
      info = loom_branch_fusion_pass_info();
      function_run = loom_branch_fusion_run;
    } else if (iree_string_view_equal(entry.name, IREE_SV("cfg-simplify"))) {
      info = loom_cfg_simplify_pass_info();
      function_run = loom_cfg_simplify_run;
    } else if (iree_string_view_equal(entry.name, IREE_SV("licm"))) {
      info = loom_licm_pass_info();
      function_run = loom_licm_run;
    } else if (iree_string_view_equal(entry.name, IREE_SV("loop-fusion"))) {
      info = loom_loop_fusion_pass_info();
      function_run = loom_loop_fusion_run;
    } else if (iree_string_view_equal(entry.name,
                                      IREE_SV("refine-boundaries"))) {
      info = loom_refine_boundaries_pass_info();
      module_run = loom_refine_boundaries_run;
      create = loom_refine_boundaries_create;
    } else if (iree_string_view_equal(entry.name, IREE_SV("scf-to-cfg"))) {
      info = loom_scf_to_cfg_pass_info();
      function_run = loom_scf_to_cfg_run;
    } else if (iree_string_view_equal(entry.name, IREE_SV("strip-hints"))) {
      info = loom_strip_hints_pass_info();
      function_run = loom_strip_hints_run;
    } else if (iree_string_view_equal(entry.name, IREE_SV("symbol-dce"))) {
      info = loom_symbol_dce_pass_info();
      module_run = loom_symbol_dce_run;
    } else if (iree_string_view_equal(entry.name,
                                      IREE_SV("vector-to-scalar"))) {
      info = loom_vector_to_scalar_pass_info();
      function_run = loom_vector_to_scalar_run;
    } else if (iree_string_view_equal(entry.name,
                                      IREE_SV("vector-memory-footprint"))) {
      info = loom_vector_memory_footprint_pass_info();
      function_run = loom_vector_memory_footprint_run;
    } else {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unknown pass: '%.*s'",
                           (int)entry.name.size, entry.name.data);
    }
    if (iree_status_is_ok(status) && info && create == NULL &&
        !iree_string_view_is_empty(entry.options)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "pass '%.*s' does not accept options, got '{%.*s}'",
                           (int)entry.name.size, entry.name.data,
                           (int)entry.options.size, entry.options.data);
    }
    if (iree_status_is_ok(status) && info) {
      if (info->kind == LOOM_PASS_MODULE) {
        status = loom_pass_manager_add_module_pass(
            &manager, info, module_run, create, destroy, entry.options);
      } else {
        status = loom_pass_manager_add_function_pass(
            &manager, info, function_run, create, destroy, entry.options);
      }
    }
    if (!iree_status_is_ok(status)) {
      loom_pass_manager_deinitialize(&manager);
      return status;
    }
  }

  iree_status_t status = loom_pass_manager_run(&manager, module);
  loom_pass_manager_deinitialize(&manager);
  return status;
}

static iree_status_t loom_check_verify_pass_output(
    const loom_check_case_t* test_case, iree_string_view_t filename,
    loom_check_diagnostic_collector_t* diagnostic_collector,
    loom_module_t* module, bool* out_failed_verification) {
  *out_failed_verification = false;
  loom_source_entry_t source_entry = {0};
  loom_source_table_resolver_t resolver_data = {0};
  IREE_RETURN_IF_ERROR(loom_check_source_resolver_for_case(
      module->context, filename, test_case->input, &source_entry,
      &resolver_data));
  loom_verify_options_t verify_options = {
      .sink = {.fn = loom_check_diagnostic_collector_sink,
               .user_data = diagnostic_collector},
      .max_errors = 100,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &resolver_data},
  };
  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(
      loom_verify_module(module, &verify_options, &verify_result));
  *out_failed_verification = verify_result.error_count > 0;
  if (*out_failed_verification) return iree_ok_status();

  loom_target_low_descriptor_registry_t low_registry;
  loom_target_low_descriptor_registry_initialize(&low_registry);
  loom_check_diagnostic_emitter_capture_t low_diagnostic_capture = {
      .diagnostic_collector = diagnostic_collector,
      .module = module,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &resolver_data},
      .emitter = LOOM_EMITTER_VERIFIER,
  };
  loom_low_verify_options_t low_verify_options = {
      .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
      .descriptor_registry = &low_registry.registry,
      .emitter =
          {
              .fn = loom_check_diagnostic_emitter_capture_emit,
              .user_data = &low_diagnostic_capture,
          },
      .max_errors = 100,
  };
  loom_low_verify_result_t low_verify_result = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_verify_module(module, &low_verify_options, &low_verify_result));
  *out_failed_verification = low_verify_result.error_count > 0;
  return iree_ok_status();
}

static iree_status_t loom_check_finish_diagnostics_if_needed(
    loom_check_diagnostic_collector_t* diagnostic_collector,
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_allocator_t allocator,
    loom_check_result_t* result, bool* out_failed) {
  *out_failed = false;
  if (diagnostic_collector->count == 0 && test_case->annotation_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_check_diagnostic_collector_finish(
      diagnostic_collector, test_case, case_index, report, allocator, result));
  *out_failed = result->raw_outcome == LOOM_CHECK_FAIL;
  return iree_ok_status();
}

static bool loom_check_case_has_expected_output(
    const loom_check_case_t* test_case) {
  return !iree_string_view_is_empty(iree_string_view_trim(test_case->expected));
}

iree_status_t loom_check_execute_pass(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_check_result_t* result) {
  loom_module_t* module = NULL;
  iree_arena_allocator_t diagnostic_arena;
  iree_arena_initialize(block_pool, &diagnostic_arena);
  loom_check_diagnostic_collector_t diagnostic_collector = {
      .arena = &diagnostic_arena,
      .host_allocator = allocator,
      .result = result,
  };
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                          .user_data = &diagnostic_collector},
      .max_errors = 20,
  };
  iree_status_t status = loom_text_parse(test_case->input, filename, context,
                                         block_pool, &parse_options, &module);
  diagnostic_collector.module = module;
  if (!module) {
    if (iree_status_is_ok(status)) {
      status = loom_check_diagnostic_collector_finish(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
    }
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  if (iree_status_is_ok(status) && diagnostic_collector.count > 0) {
    status = loom_check_diagnostic_collector_finish(&diagnostic_collector,
                                                    test_case, case_index,
                                                    report, allocator, result);
  }
  if (!iree_status_is_ok(status) || diagnostic_collector.count > 0) {
    loom_module_free(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  // Build and run the pass pipeline.
  loom_source_entry_t source_entry = {0};
  loom_source_table_resolver_t resolver_data = {0};
  status = loom_check_source_resolver_for_case(
      context, filename, test_case->input, &source_entry, &resolver_data);
  loom_check_diagnostic_emitter_capture_t pass_diagnostic_capture = {
      .diagnostic_collector = &diagnostic_collector,
      .module = module,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &resolver_data},
      .emitter = LOOM_EMITTER_PASS,
  };
  iree_diagnostic_emitter_t pass_diagnostic_emitter = {
      .fn = loom_check_diagnostic_emitter_capture_emit,
      .user_data = &pass_diagnostic_capture,
  };
  if (iree_status_is_ok(status)) {
    status = loom_check_run_pipeline(test_case->pipeline, block_pool,
                                     pass_diagnostic_emitter, module);
  }
  if (!iree_status_is_ok(status)) {
    if (pass_diagnostic_capture.emission_count > 0 ||
        diagnostic_collector.count > 0 || test_case->annotation_count > 0) {
      iree_status_ignore(status);
      status = loom_check_diagnostic_collector_finish(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
    }
    loom_module_free(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  bool failed_verification = false;
  status = loom_check_verify_pass_output(
      test_case, filename, &diagnostic_collector, module, &failed_verification);
  bool diagnostics_failed = false;
  if (iree_status_is_ok(status)) {
    status = loom_check_finish_diagnostics_if_needed(
        &diagnostic_collector, test_case, case_index, report, allocator, result,
        &diagnostics_failed);
  }
  bool has_expected_output = loom_check_case_has_expected_output(test_case);
  if (!iree_status_is_ok(status) || diagnostics_failed || failed_verification ||
      (diagnostic_collector.count > 0 && !has_expected_output)) {
    loom_module_free(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  // Print the result (stripping comments for comparison).
  status = loom_text_print_module_to_builder(module, &result->actual_output,
                                             LOOM_TEXT_PRINT_DEFAULT);
  loom_module_free(module);
  if (iree_status_is_ok(status)) {
    result->has_actual_output = true;
  }

  // Strip comments from the expected section.
  iree_string_builder_t stripped_expected;
  iree_string_builder_initialize(allocator, &stripped_expected);
  if (iree_status_is_ok(status)) {
    status = loom_check_strip_comments(test_case->expected, &stripped_expected);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t actual_trimmed =
        iree_string_view_trim(iree_string_builder_view(&result->actual_output));
    iree_string_view_t expected_trimmed =
        iree_string_view_trim(iree_string_builder_view(&stripped_expected));
    if (iree_string_view_equal(actual_trimmed, expected_trimmed)) {
      result->raw_outcome = LOOM_CHECK_PASS;
    } else {
      result->raw_outcome = LOOM_CHECK_FAIL;
      status = loom_check_result_record_diff(expected_trimmed, actual_trimmed,
                                             allocator, result);
    }
  }

  iree_string_builder_deinitialize(&stripped_expected);
  iree_arena_deinitialize(&diagnostic_arena);
  return status;
}

iree_status_t loom_check_execute_format(const loom_check_case_t* test_case,
                                        iree_string_view_t filename,
                                        loom_context_t* context,
                                        iree_arena_block_pool_t* block_pool,
                                        iree_allocator_t allocator,
                                        loom_check_result_t* result) {
  (void)test_case;
  (void)filename;
  (void)context;
  (void)block_pool;
  (void)allocator;
  (void)result;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "format mode execution not yet implemented");
}
