// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser.h"
#include "loom/ir/module.h"
#include "loom/target/llvmir/lower.h"
#include "loom/target/llvmir/text_writer.h"
#include "loom/target/llvmir/verify.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/util/stream.h"

static iree_status_t loom_check_emit_status_failure(
    iree_status_t failure_status, loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_FAIL;
  iree_status_t append_status =
      iree_string_builder_append_status(&result->detail, failure_status);
  iree_status_ignore(failure_status);
  IREE_RETURN_IF_ERROR(append_status);
  return iree_string_builder_append_cstring(&result->detail, "\n");
}

static iree_status_t loom_check_emit_select_llvmir_profile(
    iree_string_view_t emit_target,
    const loom_llvmir_target_profile_t** out_profile) {
  emit_target = iree_string_view_trim(emit_target);
  iree_string_view_t target_name = iree_string_view_empty();
  iree_string_view_t profile_name = iree_string_view_empty();
  iree_string_view_split(emit_target, ' ', &target_name, &profile_name);
  target_name = iree_string_view_trim(target_name);
  profile_name = iree_string_view_trim(profile_name);

  if (!iree_string_view_equal(target_name, IREE_SV("llvmir"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown emit target '%.*s'", (int)target_name.size,
                            target_name.data);
  }
  if (iree_string_view_is_empty(profile_name) ||
      iree_string_view_equal(profile_name, IREE_SV("x86_64-object"))) {
    *out_profile = loom_llvmir_target_profile_x86_64_object();
    return iree_ok_status();
  }
  if (iree_string_view_equal(profile_name, IREE_SV("amdgpu-hal"))) {
    *out_profile = loom_llvmir_target_profile_amdgpu_hal();
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown LLVMIR target profile '%.*s'",
                          (int)profile_name.size, profile_name.data);
}

static iree_status_t loom_check_emit_compare_output(
    const loom_check_case_t* test_case, iree_allocator_t allocator,
    loom_check_result_t* result) {
  iree_string_builder_t stripped_expected;
  iree_string_builder_initialize(allocator, &stripped_expected);

  iree_status_t status =
      loom_check_strip_comments(test_case->expected, &stripped_expected);
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
  return status;
}

iree_status_t loom_check_execute_emit(const loom_check_case_t* test_case,
                                      iree_string_view_t filename,
                                      loom_context_t* context,
                                      iree_arena_block_pool_t* block_pool,
                                      iree_allocator_t allocator,
                                      loom_check_result_t* result) {
  const loom_llvmir_target_profile_t* profile = NULL;
  iree_status_t status =
      loom_check_emit_select_llvmir_profile(test_case->emit_target, &profile);
  if (!iree_status_is_ok(status)) {
    return loom_check_emit_status_failure(status, result);
  }

  iree_string_builder_t stripped_input;
  iree_string_builder_initialize(allocator, &stripped_input);
  status = loom_check_strip_comments(test_case->input, &stripped_input);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&stripped_input);
    return status;
  }

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
  status = loom_text_parse(iree_string_builder_view(&stripped_input), filename,
                           context, block_pool, &parse_options, &module);
  iree_string_builder_deinitialize(&stripped_input);
  IREE_RETURN_IF_ERROR(status);
  if (!module) {
    result->raw_outcome = LOOM_CHECK_FAIL;
    return iree_ok_status();
  }

  loom_llvmir_lowering_options_t options = {
      .target_profile = profile,
      .source_name = filename,
  };
  loom_llvmir_module_t* lowered_module = NULL;
  status =
      loom_llvmir_lower_module(module, &options, allocator, &lowered_module);
  loom_module_free(module);
  if (!iree_status_is_ok(status)) {
    return loom_check_emit_status_failure(status, result);
  }

  status = loom_llvmir_verify_module(lowered_module);
  if (iree_status_is_ok(status)) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&result->actual_output, &stream);
    status = loom_llvmir_text_write_module(lowered_module, &stream);
  }
  loom_llvmir_module_free(lowered_module);
  if (!iree_status_is_ok(status)) {
    return loom_check_emit_status_failure(status, result);
  }

  return loom_check_emit_compare_output(test_case, allocator, result);
}
