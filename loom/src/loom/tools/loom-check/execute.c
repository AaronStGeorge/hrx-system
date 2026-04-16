// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/execute.h"

#include "loom/error/diagnostic.h"
#include "loom/error/json_sink.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/ops/pool/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/testing/diff.h"
#include "loom/transforms/branch_fusion.h"
#include "loom/transforms/branch_sink.h"
#include "loom/transforms/canonicalize.h"
#include "loom/transforms/cse.h"
#include "loom/transforms/dce.h"
#include "loom/transforms/kernel_async_legality.h"
#include "loom/transforms/kernel_resources.h"
#include "loom/transforms/licm.h"
#include "loom/transforms/loop_fusion.h"
#include "loom/transforms/pass.h"
#include "loom/transforms/refine_boundaries.h"
#include "loom/transforms/strip_hints.h"
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

typedef struct loom_check_pass_diagnostic_capture_t {
  // Diagnostic sink shared with parser, verifier, and pass-mode reporting.
  loom_check_diagnostic_capture_t* diagnostic_capture;

  // Number of diagnostics emitted by the pass pipeline in this execution.
  iree_host_size_t emission_count;
} loom_check_pass_diagnostic_capture_t;

static iree_status_t loom_check_pass_diagnostic_emitter(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loom_check_pass_diagnostic_capture_t* capture =
      (loom_check_pass_diagnostic_capture_t*)user_data;
  loom_diagnostic_t diagnostic = {
      .severity = emission->error->severity,
      .error = emission->error,
      .params = emission->params,
      .param_count = emission->param_count,
      .emitter = LOOM_EMITTER_PASS,
      .origin = {.provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE},
      .source_location = {.provenance =
                              LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE},
  };
  IREE_RETURN_IF_ERROR(loom_check_diagnostic_capture_sink(
      capture->diagnostic_capture, &diagnostic));
  ++capture->emission_count;
  return iree_ok_status();
}

// Registers a single dialect's vtable array with the context.
static iree_status_t loom_check_register_dialect(
    loom_context_t* context, uint8_t dialect_id,
    const loom_op_vtable_t* const* (*vtable_fn)(iree_host_size_t*)) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = vtable_fn(&count);
  return loom_context_register_dialect(context, dialect_id, vtables,
                                       (uint16_t)count);
}

iree_status_t loom_check_context_initialize(loom_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(context, LOOM_DIALECT_TEST,
                                                   loom_test_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(context, LOOM_DIALECT_FUNC,
                                                   loom_func_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(
      context, LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(
      context, LOOM_DIALECT_ENCODING, loom_encoding_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(context, LOOM_DIALECT_POOL,
                                                   loom_pool_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(
      context, LOOM_DIALECT_GLOBAL, loom_global_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(context, LOOM_DIALECT_INDEX,
                                                   loom_index_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(
      context, LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(
      context, LOOM_DIALECT_LLVMIR, loom_llvmir_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(context, LOOM_DIALECT_SCF,
                                                   loom_scf_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(
      context, LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(context, LOOM_DIALECT_VIEW,
                                                   loom_view_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(
      context, LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables));
  IREE_RETURN_IF_ERROR(loom_context_register_builtin_encoding_vtables(context));
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
      IREE_RETURN_IF_ERROR(loom_check_execute_pass(
          test_case, filename, context, block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_FORMAT: {
      IREE_RETURN_IF_ERROR(loom_check_execute_format(
          test_case, filename, context, block_pool, allocator, result));
      break;
    }
    case LOOM_CHECK_MODE_EMIT: {
      IREE_RETURN_IF_ERROR(loom_check_execute_emit(
          test_case, filename, context, block_pool, allocator, result));
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
  while (remaining.size > 0) {
    iree_string_view_t pass_name = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &pass_name, &remaining);
    pass_name = iree_string_view_trim(pass_name);
    if (pass_name.size == 0) continue;

    iree_status_t status = iree_ok_status();
    if (iree_string_view_equal(pass_name, IREE_SV("canonicalize"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, loom_canonicalize_pass_info(), loom_canonicalize_run, NULL,
          NULL, iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name, IREE_SV("dce"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, loom_dce_pass_info(), loom_dce_run, NULL, NULL,
          iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name,
                                      IREE_SV("kernel-async-legality"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, loom_kernel_async_legality_pass_info(),
          loom_kernel_async_legality_run, NULL, NULL, iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name,
                                      IREE_SV("normalize-kernel-resources"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, loom_normalize_kernel_resources_pass_info(),
          loom_normalize_kernel_resources_run, NULL, NULL,
          iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name, IREE_SV("cse"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, loom_cse_pass_info(), loom_cse_run, NULL, NULL,
          iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name, IREE_SV("branch-sink"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, loom_branch_sink_pass_info(), loom_branch_sink_run, NULL,
          NULL, iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name, IREE_SV("branch-fusion"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, loom_branch_fusion_pass_info(), loom_branch_fusion_run,
          NULL, NULL, iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name, IREE_SV("licm"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, loom_licm_pass_info(), loom_licm_run, NULL, NULL,
          iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name, IREE_SV("loop-fusion"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, loom_loop_fusion_pass_info(), loom_loop_fusion_run, NULL,
          NULL, iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name,
                                      IREE_SV("refine-boundaries"))) {
      status = loom_pass_manager_add_module_pass(
          &manager, loom_refine_boundaries_pass_info(),
          loom_refine_boundaries_run, NULL, NULL, iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name, IREE_SV("strip-hints"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, loom_strip_hints_pass_info(), loom_strip_hints_run, NULL,
          NULL, iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name, IREE_SV("vector-to-scalar"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, loom_vector_to_scalar_pass_info(),
          loom_vector_to_scalar_run, NULL, NULL, iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name,
                                      IREE_SV("vector-memory-footprint"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, loom_vector_memory_footprint_pass_info(),
          loom_vector_memory_footprint_run, NULL, NULL,
          iree_string_view_empty());
    } else {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unknown pass: '%.*s'",
                           (int)pass_name.size, pass_name.data);
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
    loom_check_diagnostic_capture_t* diagnostic_capture, loom_module_t* module,
    bool* out_failed_verification) {
  *out_failed_verification = false;
  loom_source_entry_t source_entry = {
      .source_id = 0,
      .source = test_case->input,
      .filename = filename,
  };
  loom_source_table_resolver_t resolver_data = {
      .entries = &source_entry,
      .count = 1,
  };
  loom_verify_options_t verify_options = {
      .sink = {.fn = loom_check_diagnostic_capture_sink,
               .user_data = diagnostic_capture},
      .max_errors = 100,
      .source_resolver = {.fn = loom_source_table_resolve,
                          .user_data = &resolver_data},
  };
  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(
      loom_verify_module(module, &verify_options, &verify_result));
  *out_failed_verification = verify_result.error_count > 0;
  return iree_ok_status();
}

iree_status_t loom_check_execute_pass(const loom_check_case_t* test_case,
                                      iree_string_view_t filename,
                                      loom_context_t* context,
                                      iree_arena_block_pool_t* block_pool,
                                      iree_allocator_t allocator,
                                      loom_check_result_t* result) {
  // Parse the input.
  loom_module_t* module = NULL;
  loom_check_diagnostic_capture_t diagnostic_capture = {
      .detail = &result->detail,
      .result = result,
  };
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_capture_sink,
                          .user_data = &diagnostic_capture},
      .max_errors = 20,
  };
  iree_status_t parse_status = loom_text_parse(
      test_case->input, filename, context, block_pool, &parse_options, &module);
  if (!iree_status_is_ok(parse_status)) return parse_status;
  if (!module) {
    result->raw_outcome = LOOM_CHECK_FAIL;
    return iree_ok_status();
  }

  // Build and run the pass pipeline.
  loom_check_pass_diagnostic_capture_t pass_diagnostic_capture = {
      .diagnostic_capture = &diagnostic_capture,
  };
  iree_diagnostic_emitter_t pass_diagnostic_emitter = {
      .fn = loom_check_pass_diagnostic_emitter,
      .user_data = &pass_diagnostic_capture,
  };
  iree_status_t pipeline_status = loom_check_run_pipeline(
      test_case->pipeline, block_pool, pass_diagnostic_emitter, module);
  if (!iree_status_is_ok(pipeline_status)) {
    loom_module_free(module);
    if (pass_diagnostic_capture.emission_count > 0) {
      result->raw_outcome = LOOM_CHECK_FAIL;
      return iree_status_ignore(pipeline_status);
    }
    return pipeline_status;
  }

  bool failed_verification = false;
  iree_status_t verify_status = loom_check_verify_pass_output(
      test_case, filename, &diagnostic_capture, module, &failed_verification);
  if (!iree_status_is_ok(verify_status)) {
    loom_module_free(module);
    return verify_status;
  }
  if (failed_verification) {
    loom_module_free(module);
    result->raw_outcome = LOOM_CHECK_FAIL;
    return iree_ok_status();
  }

  // Print the result (stripping comments for comparison).
  iree_status_t print_status = loom_text_print_module_to_builder(
      module, &result->actual_output, LOOM_TEXT_PRINT_DEFAULT);
  loom_module_free(module);
  IREE_RETURN_IF_ERROR(print_status);

  // Strip comments from the expected section.
  iree_string_builder_t stripped_expected;
  iree_string_builder_initialize(allocator, &stripped_expected);
  IREE_RETURN_IF_ERROR(
      loom_check_strip_comments(test_case->expected, &stripped_expected));

  // Compare.
  iree_string_view_t actual_trimmed =
      iree_string_view_trim(iree_string_builder_view(&result->actual_output));
  iree_string_view_t expected_trimmed =
      iree_string_view_trim(iree_string_builder_view(&stripped_expected));

  iree_status_t status = iree_ok_status();
  if (iree_string_view_equal(actual_trimmed, expected_trimmed)) {
    result->raw_outcome = LOOM_CHECK_PASS;
  } else {
    result->raw_outcome = LOOM_CHECK_FAIL;
    status = loom_check_result_record_diff(expected_trimmed, actual_trimmed,
                                           allocator, result);
  }

  iree_string_builder_deinitialize(&stripped_expected);
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
