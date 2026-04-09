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
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/pool/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/diff.h"
#include "loom/transforms/canonicalize.h"
#include "loom/transforms/cse.h"
#include "loom/transforms/dce.h"
#include "loom/transforms/pass.h"

void loom_check_result_initialize(iree_allocator_t allocator,
                                  loom_check_result_t* out_result) {
  memset(out_result, 0, sizeof(*out_result));
  iree_string_builder_initialize(allocator, &out_result->detail);
  iree_string_builder_initialize(allocator, &out_result->actual_output);
  iree_string_builder_initialize(allocator, &out_result->diagnostic_json);
}

void loom_check_result_deinitialize(loom_check_result_t* result) {
  iree_string_builder_deinitialize(&result->diagnostic_json);
  iree_string_builder_deinitialize(&result->actual_output);
  iree_string_builder_deinitialize(&result->detail);
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
  IREE_RETURN_IF_ERROR(loom_check_register_dialect(context, LOOM_DIALECT_SCF,
                                                   loom_scf_dialect_vtables));
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
    loom_module_t* module) {
  loom_pass_manager_t manager;
  IREE_RETURN_IF_ERROR(loom_pass_manager_initialize(
      block_pool, 0, iree_allocator_system(), &manager));

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
          &manager, &loom_canonicalize_pass_info, loom_canonicalize_run, NULL,
          NULL, iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name, IREE_SV("dce"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, &loom_dce_pass_info, loom_dce_run, NULL, NULL,
          iree_string_view_empty());
    } else if (iree_string_view_equal(pass_name, IREE_SV("cse"))) {
      status = loom_pass_manager_add_function_pass(
          &manager, &loom_cse_pass_info, loom_cse_run, NULL, NULL,
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
  iree_status_t pipeline_status =
      loom_check_run_pipeline(test_case->pipeline, block_pool, module);
  if (!iree_status_is_ok(pipeline_status)) {
    loom_module_free(module);
    return pipeline_status;
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
    status = loom_diff(expected_trimmed, actual_trimmed,
                       LOOM_DIFF_DEFAULT_CONTEXT, allocator, &result->detail);
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
