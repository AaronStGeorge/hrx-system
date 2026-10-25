// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/target/provider.h"
#include "loom/tooling/io/module_format.h"
#include "loom/tools/loom-check/comparison.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/compile_diagnostics.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/input.h"
#include "loomc/iree.h"

iree_status_t loom_check_validate_printed_ir(
    iree_string_view_t source, loom_context_t* context,
    iree_arena_block_pool_t* block_pool,
    const loom_text_print_options_t* print_options, loom_check_result_t* result,
    bool* out_valid) {
  *out_valid = false;
  loom_check_diagnostic_capture_t diagnostic_capture = {
      .detail = &result->detail,
      .result = result,
  };
  const loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_capture_sink,
                          .user_data = &diagnostic_capture},
      .max_errors = 20,
      .low_asm_environment = print_options->low_asm_environment,
  };
  loom_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(loom_text_parse(source, IREE_SV("<printed IR>"), context,
                                       block_pool, &parse_options, &module));
  if (!module) {
    return iree_string_builder_append_cstring(&result->detail,
                                              "printed IR failed to parse\n");
  }

  iree_string_builder_t reprinted;
  iree_string_builder_initialize(result->detail.allocator, &reprinted);
  iree_status_t status = loom_text_print_module_to_builder_with_options(
      module, &reprinted, print_options);
  loom_module_free(module);
  if (iree_status_is_ok(status)) {
    *out_valid =
        iree_string_view_equal(source, iree_string_builder_view(&reprinted));
    if (!*out_valid) {
      status = iree_string_builder_append_cstring(
          &result->detail, "printed IR changed after reparsing\n");
      if (iree_status_is_ok(status)) {
        status = loom_check_result_record_diff(
            source, iree_string_builder_view(&reprinted),
            result->detail.allocator, result);
      }
    }
  }
  iree_string_builder_deinitialize(&reprinted);
  return status;
}

static loom_text_print_flags_t loom_check_roundtrip_print_flags(
    const loom_test_case_t* test_case) {
  loom_text_print_flags_t flags =
      LOOM_TEXT_PRINT_DEFAULT | LOOM_TEXT_PRINT_PREFER_LOW_ASM;
  if (iree_all_bits_set(test_case->output_flags, LOOM_TEST_OUTPUT_LOCATIONS)) {
    flags |= LOOM_TEXT_PRINT_LOCATIONS;
  }
  return flags;
}

iree_status_t loom_check_execute_roundtrip(
    const loom_test_case_t* test_case, iree_string_view_t filename,
    const loom_input_request_t* input_request,
    const loom_check_environment_t* environment, loom_context_t* context,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  loom_input_module_t input = {0};
  loom_check_diagnostic_capture_t diagnostic_capture = {
      .detail = &result->detail,
      .result = result,
  };
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_capture_sink,
                          .user_data = &diagnostic_capture},
      .max_errors = 20,
  };
  const loom_target_low_descriptor_registry_t low_registry =
      loom_target_environment_low_descriptor_registry(
          environment->target_environment);
  loom_low_descriptor_text_asm_environment_storage_t low_asm_storage = {0};
  loom_low_descriptor_text_asm_environment_initialize_with_diagnostics(
      &low_registry.registry,
      loom_target_environment_low_asm_diagnostic_provider_list(
          environment->target_environment),
      &low_asm_storage, &parse_options.low_asm_environment);
  iree_status_t parse_status =
      loom_check_load_input(test_case, input_request, environment, context,
                            block_pool, &parse_options, allocator, &input);
  if (!iree_status_is_ok(parse_status) || !input.module) {
    result->raw_outcome = LOOM_CHECK_FAIL;
    loom_input_module_deinitialize(&input);
    return parse_status;
  }

  // Print the parsed module to canonical text (directly into the result's
  // actual_output so --update can use it) and free the module.
  loom_text_low_asm_environment_t low_asm_environment = {0};
  loom_low_descriptor_text_asm_environment_initialize(&low_registry.registry,
                                                      &low_asm_environment);
  const loom_text_print_options_t print_options = {
      .flags = loom_check_roundtrip_print_flags(test_case),
      .low_asm_environment = low_asm_environment,
  };
  iree_status_t print_status = loom_text_print_module_to_builder_with_options(
      input.module, &result->actual_output, &print_options);
  loom_input_module_deinitialize(&input);
  IREE_RETURN_IF_ERROR(print_status);
  bool valid_output = false;
  IREE_RETURN_IF_ERROR(loom_check_validate_printed_ir(
      iree_string_builder_view(&result->actual_output), context, block_pool,
      &print_options, result, &valid_output));
  if (!valid_output) {
    result->raw_outcome = LOOM_CHECK_FAIL;
    return iree_ok_status();
  }
  result->has_actual_output = true;

  return loom_check_compare_output(test_case, allocator, result);
}

static iree_status_t loom_check_format_verify_module(
    loomc_module_t* module, loom_check_diagnostic_collector_t* collector,
    uint32_t max_errors, iree_allocator_t allocator) {
  const loomc_module_verify_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS,
      .structure_size = sizeof(options),
      .max_errors = max_errors,
      .scope = LOOMC_MODULE_VERIFY_SCOPE_STRUCTURAL,
  };
  loomc_result_t* verification = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_module_verify(
      module, &options, loomc_allocator_from_iree(allocator), &verification));
  if (iree_status_is_ok(status)) {
    status =
        loom_check_compile_append_result_diagnostics(collector, verification);
  }
  if (iree_status_is_ok(status) && !loomc_result_succeeded(verification)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "format module failed verification");
  }
  loomc_result_release(verification);
  return status;
}

iree_status_t loom_check_execute_format(
    const loom_test_case_t* test_case, iree_string_view_t filename,
    const loom_input_request_t* input_request,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  loom_module_format_t format = LOOM_MODULE_FORMAT_AUTO;
  IREE_RETURN_IF_ERROR(loom_module_format_parse(test_case->format_target,
                                                /*allow_auto=*/false, &format));
  iree_arena_allocator_t diagnostic_arena;
  iree_arena_initialize(block_pool, &diagnostic_arena);
  loom_check_diagnostic_collector_t collector = {
      .arena = &diagnostic_arena,
      .host_allocator = allocator,
      .filename = filename,
      .result = result,
  };
  loom_check_compile_session_t* session = environment->compile_session;
  loomc_module_t* module = NULL;
  iree_status_t status = loom_check_compile_admit_module(
      test_case, filename, input_request, session, environment,
      /*max_errors=*/0, &collector, block_pool, allocator, &module);
  if (iree_status_is_ok(status) && module == NULL) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "input failed to produce a module");
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_format_verify_module(module, &collector,
                                             /*max_errors=*/0, allocator);
  }

  const loomc_allocator_t public_allocator =
      loomc_allocator_from_iree(allocator);
  loomc_module_serialize_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      .structure_size = sizeof(options),
      .format = format == LOOM_MODULE_FORMAT_TEXT
                    ? LOOMC_SOURCE_FORMAT_TEXT
                    : LOOMC_SOURCE_FORMAT_BYTECODE,
      .identifier = loomc_string_view_from_iree(filename),
      .text_presentation = LOOMC_MODULE_TEXT_PRESENTATION_PRESERVE,
  };
  loomc_source_t* serialized = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_serialize_to_source(
        module, &options, public_allocator, &serialized));
  }
  loomc_module_release(module);
  module = NULL;

  loomc_result_t* decoding = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_deserialize_from_source(
        session->context, session->workspace, serialized, /*options=*/NULL,
        public_allocator, &module, &decoding));
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result_diagnostics(&collector, decoding);
  }
  if (iree_status_is_ok(status) && !loomc_result_succeeded(decoding)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "formatted output failed to produce a module");
  }
  loomc_result_release(decoding);
  loomc_source_release(serialized);
  if (iree_status_is_ok(status)) {
    status = loom_check_format_verify_module(module, &collector,
                                             /*max_errors=*/100, allocator);
  }

  loomc_source_t* text = NULL;
  if (iree_status_is_ok(status)) {
    options.format = LOOMC_SOURCE_FORMAT_TEXT;
    status = iree_status_from_loomc(loomc_module_serialize_to_source(
        module, &options, public_allocator, &text));
  }
  loomc_module_release(module);
  if (iree_status_is_ok(status)) {
    const loomc_byte_span_t contents = loomc_source_contents(text);
    status = iree_string_builder_append_string(
        &result->actual_output,
        iree_make_string_view((const char*)contents.data,
                              contents.data_length));
  }
  loomc_source_release(text);

  iree_status_t diagnostic_status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(diagnostic_status) && i < collector.count; ++i) {
    diagnostic_status = iree_string_builder_append_string(
        &result->detail, collector.diagnostics[i].formatted_diagnostic);
  }
  iree_arena_deinitialize(&diagnostic_arena);
  IREE_RETURN_IF_ERROR(iree_status_join(status, diagnostic_status));
  result->has_actual_output = true;
  return loom_check_compare_output(test_case, allocator, result);
}
