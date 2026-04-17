// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/vec_stream.h"
#include "loom/format/text/parser.h"
#include "loom/ir/module.h"
#include "loom/target/emit/ireevm/descriptors.h"
#include "loom/target/emit/llvmir/bitcode_writer.h"
#include "loom/target/emit/llvmir/legality.h"
#include "loom/target/emit/llvmir/lower.h"
#include "loom/target/emit/llvmir/text_writer.h"
#include "loom/target/emit/llvmir/tool.h"
#include "loom/target/emit/llvmir/verify.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/llvmir_targets.h"
#include "loom/util/stream.h"

typedef enum loom_check_emit_format_e {
  LOOM_CHECK_EMIT_LLVMIR_TEXT = 0,
  LOOM_CHECK_EMIT_LLVMIR_BODY_TEXT = 1,
  LOOM_CHECK_EMIT_LLVMIR_BITCODE_DISASSEMBLY = 2,
  LOOM_CHECK_EMIT_LLVMIR_OBJECT = 3,
  LOOM_CHECK_EMIT_LLVMIR_ASSEMBLY_MNEMONICS = 4,
  LOOM_CHECK_EMIT_LOW_DESCRIPTOR_MANIFEST = 5,
} loom_check_emit_format_t;

typedef struct loom_check_emit_request_t {
  // Serialized target form to produce before comparison.
  loom_check_emit_format_t format;
  // Generic target bundle used by legality and LLVMIR profile derivation.
  const loom_target_bundle_t* target_bundle;
  // Low descriptor set used by descriptor-manifest dumps.
  const loom_low_descriptor_set_t* low_descriptor_set;
} loom_check_emit_request_t;

typedef struct loom_check_emit_byte_buffer_t {
  // Allocated byte storage owned by this buffer.
  uint8_t* data;
  // Number of valid bytes in |data|.
  iree_host_size_t length;
} loom_check_emit_byte_buffer_t;

static iree_status_t loom_check_emit_parse_request(
    iree_string_view_t emit_target, loom_check_emit_request_t* out_request) {
  *out_request = (loom_check_emit_request_t){
      .format = LOOM_CHECK_EMIT_LLVMIR_TEXT,
      .target_bundle = NULL,
      .low_descriptor_set = NULL,
  };
  emit_target = iree_string_view_trim(emit_target);
  iree_string_view_t target_name = iree_string_view_empty();
  iree_string_view_t profile_name = iree_string_view_empty();
  iree_string_view_split(emit_target, ' ', &target_name, &profile_name);
  target_name = iree_string_view_trim(target_name);
  profile_name = iree_string_view_trim(profile_name);

  if (iree_string_view_equal(target_name, IREE_SV("llvmir")) ||
      iree_string_view_equal(target_name, IREE_SV("llvmir-text"))) {
    out_request->format = LOOM_CHECK_EMIT_LLVMIR_TEXT;
  } else if (iree_string_view_equal(target_name, IREE_SV("llvmir-body")) ||
             iree_string_view_equal(target_name, IREE_SV("llvmir-text-body"))) {
    out_request->format = LOOM_CHECK_EMIT_LLVMIR_BODY_TEXT;
  } else if (iree_string_view_equal(target_name, IREE_SV("llvmir-bitcode"))) {
    out_request->format = LOOM_CHECK_EMIT_LLVMIR_BITCODE_DISASSEMBLY;
  } else if (iree_string_view_equal(target_name, IREE_SV("llvmir-object"))) {
    out_request->format = LOOM_CHECK_EMIT_LLVMIR_OBJECT;
  } else if (iree_string_view_equal(target_name,
                                    IREE_SV("llvmir-assembly-mnemonics")) ||
             iree_string_view_equal(target_name,
                                    IREE_SV("llvmir-asm-mnemonics"))) {
    out_request->format = LOOM_CHECK_EMIT_LLVMIR_ASSEMBLY_MNEMONICS;
  } else if (iree_string_view_equal(target_name,
                                    IREE_SV("low-descriptor-manifest")) ||
             iree_string_view_equal(target_name,
                                    IREE_SV("low-descriptor-json"))) {
    if (iree_string_view_equal(profile_name, IREE_SV("iree.vm.core"))) {
      out_request->format = LOOM_CHECK_EMIT_LOW_DESCRIPTOR_MANIFEST;
      out_request->low_descriptor_set = loom_ireevm_core_descriptor_set();
      return iree_ok_status();
    }
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "unknown low descriptor set '%.*s'; expected 'iree.vm.core'",
        (int)profile_name.size, profile_name.data);
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown emit target '%.*s'", (int)target_name.size,
                            target_name.data);
  }
  return loom_check_llvmir_target_bundle_lookup(profile_name,
                                                &out_request->target_bundle);
}

static iree_string_view_t loom_check_emit_consume_line(
    iree_string_view_t* remaining) {
  iree_host_size_t newline_pos =
      iree_string_view_find(*remaining, IREE_SV("\n"), 0);
  if (newline_pos == IREE_STRING_VIEW_NPOS) {
    iree_string_view_t line = *remaining;
    *remaining = iree_string_view_empty();
    return line;
  }
  iree_string_view_t line = iree_string_view_substr(*remaining, 0, newline_pos);
  *remaining =
      iree_string_view_substr(*remaining, newline_pos + 1, IREE_HOST_SIZE_MAX);
  return line;
}

static bool loom_check_emit_line_has_ir(iree_string_view_t line) {
  line = iree_string_view_trim(line);
  return !iree_string_view_is_empty(line) &&
         !iree_string_view_starts_with(line, IREE_SV("//"));
}

static loom_source_range_t loom_check_emit_first_ir_source_range(
    const loom_check_case_t* test_case, iree_string_view_t filename) {
  if (iree_string_view_is_empty(test_case->input)) {
    return (loom_source_range_t){
        .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
    };
  }

  iree_string_view_t remaining = test_case->input;
  uint32_t line_number = 1;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_check_emit_consume_line(&remaining);
    if (loom_check_emit_line_has_ir(line)) {
      iree_host_size_t indentation_length = 0;
      while (indentation_length < line.size &&
             (line.data[indentation_length] == ' ' ||
              line.data[indentation_length] == '\t')) {
        ++indentation_length;
      }
      iree_host_size_t start =
          (iree_host_size_t)(line.data - test_case->input.data) +
          indentation_length;
      return (loom_source_range_t){
          .provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
          .filename = filename,
          .source = test_case->input,
          .start = start,
          .end = start + 1,
          .start_line = line_number,
          .start_column = (uint32_t)indentation_length + 1,
          .end_line = line_number,
          .end_column = (uint32_t)indentation_length + 2,
      };
    }
    ++line_number;
  }

  return (loom_source_range_t){
      .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
  };
}

static iree_status_t loom_check_emit_collect_status_diagnostic(
    loom_check_diagnostic_collector_t* collector,
    const loom_check_case_t* test_case, iree_string_view_t filename,
    iree_status_t failure_status, iree_allocator_t allocator) {
  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 1);
  if (!error) {
    iree_status_ignore(failure_status);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "LOWERING/001 diagnostic is not registered");
  }

  char* status_buffer = NULL;
  iree_host_size_t status_length = 0;
  if (!iree_status_to_string(failure_status, &allocator, &status_buffer,
                             &status_length)) {
    iree_status_ignore(failure_status);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to render lowering failure status");
  }

  iree_string_view_t status_text =
      iree_make_string_view(status_buffer, status_length);
  loom_diagnostic_param_t params[3] = {
      loom_param_string(IREE_SV("module")),
      loom_param_string(IREE_SV("llvmir")),
      loom_param_string(status_text),
  };
  loom_source_range_t source_range =
      loom_check_emit_first_ir_source_range(test_case, filename);
  loom_diagnostic_t diagnostic = {
      .severity = error->severity,
      .error = error,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .emitter = LOOM_EMITTER_PASS,
      .origin = source_range,
      .source_location = source_range,
  };
  iree_status_t status =
      loom_check_diagnostic_collector_sink(collector, &diagnostic);

  iree_allocator_free(allocator, status_buffer);
  iree_status_ignore(failure_status);
  return status;
}

static iree_status_t loom_check_emit_finish_status_failure(
    iree_status_t failure_status, loom_check_diagnostic_collector_t* collector,
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    iree_allocator_t allocator, loom_check_result_t* result) {
  IREE_RETURN_IF_ERROR(loom_check_emit_collect_status_diagnostic(
      collector, test_case, filename, failure_status, allocator));
  return loom_check_diagnostic_collector_finish(
      collector, test_case, case_index, report, allocator, result);
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

static iree_status_t loom_check_emit_strip_llvmir_comments(
    iree_string_view_t input, iree_string_builder_t* output) {
  iree_string_view_t remaining = input;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_check_emit_consume_line(&remaining);
    iree_string_view_t trimmed = iree_string_view_trim(line);
    if (iree_string_view_starts_with(trimmed, IREE_SV(";"))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, line));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
  }
  return iree_ok_status();
}

static bool loom_check_emit_is_llvmir_module_header_line(
    iree_string_view_t line) {
  line = iree_string_view_trim(line);
  return iree_string_view_starts_with(line, IREE_SV("source_filename =")) ||
         iree_string_view_starts_with(line, IREE_SV("target datalayout =")) ||
         iree_string_view_starts_with(line, IREE_SV("target triple ="));
}

static iree_status_t loom_check_emit_strip_llvmir_module_header(
    iree_string_view_t input, iree_string_builder_t* output) {
  bool stripped_header = false;
  bool emitted_body_line = false;
  iree_string_view_t remaining = input;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_check_emit_consume_line(&remaining);
    if (loom_check_emit_is_llvmir_module_header_line(line)) {
      stripped_header = true;
      continue;
    }
    if (stripped_header && !emitted_body_line &&
        iree_string_view_is_empty(iree_string_view_trim(line))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, line));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
    emitted_body_line = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_check_emit_write_llvmir_text(
    const loom_llvmir_module_t* lowered_module, loom_check_result_t* result) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&result->actual_output, &stream);
  return loom_llvmir_text_write_module(lowered_module, &stream);
}

static iree_status_t loom_check_emit_write_llvmir_body_text(
    const loom_llvmir_module_t* lowered_module, iree_allocator_t allocator,
    loom_check_result_t* result) {
  iree_string_builder_t module_text;
  iree_string_builder_initialize(allocator, &module_text);

  loom_output_stream_t stream;
  loom_output_stream_for_builder(&module_text, &stream);
  iree_status_t status = loom_llvmir_text_write_module(lowered_module, &stream);
  if (iree_status_is_ok(status)) {
    status = loom_check_emit_strip_llvmir_module_header(
        iree_string_builder_view(&module_text), &result->actual_output);
  }

  iree_string_builder_deinitialize(&module_text);
  return status;
}

static void loom_check_emit_byte_buffer_deinitialize(
    loom_check_emit_byte_buffer_t* buffer, iree_allocator_t allocator) {
  iree_allocator_free(allocator, buffer->data);
  *buffer = (loom_check_emit_byte_buffer_t){0};
}

static iree_status_t loom_check_emit_write_llvmir_bitcode_bytes(
    const loom_llvmir_module_t* lowered_module, iree_allocator_t allocator,
    loom_check_emit_byte_buffer_t* out_bitcode) {
  *out_bitcode = (loom_check_emit_byte_buffer_t){0};
  iree_io_stream_t* bitcode_stream = NULL;
  iree_status_t status = iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE,
      4096, allocator, &bitcode_stream);

  uint8_t* bitcode_data = NULL;
  iree_host_size_t bitcode_length = 0;
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_module(lowered_module, bitcode_stream);
  }
  if (iree_status_is_ok(status)) {
    iree_io_stream_pos_t stream_length = iree_io_stream_length(bitcode_stream);
    if (stream_length <= 0 ||
        (uint64_t)stream_length > (uint64_t)IREE_HOST_SIZE_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "LLVM bitcode output length is invalid");
    } else {
      bitcode_length = (iree_host_size_t)stream_length;
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(allocator, bitcode_length, (void**)&bitcode_data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_seek(bitcode_stream, IREE_IO_STREAM_SEEK_SET, 0);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_read(bitcode_stream, bitcode_length, bitcode_data, NULL);
  }
  if (iree_status_is_ok(status)) {
    *out_bitcode = (loom_check_emit_byte_buffer_t){
        .data = bitcode_data,
        .length = bitcode_length,
    };
    bitcode_data = NULL;
  }

  iree_allocator_free(allocator, bitcode_data);
  iree_io_stream_release(bitcode_stream);
  return status;
}

static iree_status_t loom_check_emit_write_llvmir_bitcode_disassembly(
    const loom_llvmir_module_t* lowered_module, iree_allocator_t allocator,
    loom_check_result_t* result) {
  loom_check_emit_byte_buffer_t bitcode = {0};
  iree_status_t status = loom_check_emit_write_llvmir_bitcode_bytes(
      lowered_module, allocator, &bitcode);

  loom_llvmir_tool_output_t disassembly = {0};
  if (iree_status_is_ok(status)) {
    loom_llvmir_toolchain_t toolchain;
    loom_llvmir_toolchain_initialize_from_environment(&toolchain);
    status = loom_llvmir_tool_disassemble_bitcode(
        &toolchain, iree_make_const_byte_span(bitcode.data, bitcode.length),
        allocator, &disassembly);
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_emit_strip_llvmir_comments(
        iree_make_string_view(disassembly.data, disassembly.length),
        &result->actual_output);
  }

  loom_llvmir_tool_output_deinitialize(&disassembly, allocator);
  loom_check_emit_byte_buffer_deinitialize(&bitcode, allocator);
  return status;
}

static iree_status_t loom_check_emit_write_llvmir_object(
    const loom_llvmir_module_t* lowered_module,
    const loom_llvmir_target_profile_t* profile, iree_allocator_t allocator,
    loom_check_result_t* result) {
  loom_check_emit_byte_buffer_t bitcode = {0};
  iree_status_t status = loom_check_emit_write_llvmir_bitcode_bytes(
      lowered_module, allocator, &bitcode);

  loom_llvmir_tool_output_t object = {0};
  if (iree_status_is_ok(status)) {
    loom_llvmir_toolchain_t toolchain;
    loom_llvmir_toolchain_initialize_from_environment(&toolchain);
    loom_llvmir_target_profile_llc_arguments_t llc_arguments = {0};
    status = loom_llvmir_target_profile_llc_arguments(profile, &llc_arguments);
    if (iree_status_is_ok(status)) {
      status = loom_llvmir_tool_compile_object(
          &toolchain, iree_make_const_byte_span(bitcode.data, bitcode.length),
          llc_arguments.values, llc_arguments.count, allocator, &object);
    }
  }
  if (iree_status_is_ok(status) && object.length == 0) {
    status =
        iree_make_status(IREE_STATUS_DATA_LOSS, "LLVM object output is empty");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        &result->actual_output, "object emitted: %.*s\n",
        (int)profile->name.size, profile->name.data);
  }

  loom_llvmir_tool_output_deinitialize(&object, allocator);
  loom_check_emit_byte_buffer_deinitialize(&bitcode, allocator);
  return status;
}

static bool loom_check_emit_assembly_line_is_label(iree_string_view_t line) {
  return line.size > 0 && line.data[line.size - 1] == ':';
}

static iree_string_view_t loom_check_emit_strip_assembly_comment(
    iree_string_view_t line) {
  iree_host_size_t hash_position = iree_string_view_find(line, IREE_SV("#"), 0);
  iree_host_size_t semicolon_position =
      iree_string_view_find(line, IREE_SV(";"), 0);
  iree_host_size_t comment_position = IREE_STRING_VIEW_NPOS;
  if (hash_position != IREE_STRING_VIEW_NPOS) {
    comment_position = hash_position;
  }
  if (semicolon_position != IREE_STRING_VIEW_NPOS &&
      (comment_position == IREE_STRING_VIEW_NPOS ||
       semicolon_position < comment_position)) {
    comment_position = semicolon_position;
  }
  if (comment_position == IREE_STRING_VIEW_NPOS) {
    return line;
  }
  return iree_string_view_substr(line, 0, comment_position);
}

static iree_status_t loom_check_emit_write_assembly_mnemonics(
    iree_string_view_t assembly, iree_string_builder_t* output) {
  iree_string_view_t remaining = assembly;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_check_emit_strip_assembly_comment(
        loom_check_emit_consume_line(&remaining));
    line = iree_string_view_trim(line);
    if (iree_string_view_is_empty(line) ||
        iree_string_view_starts_with_char(line, '.') ||
        iree_string_view_starts_with_char(line, '#') ||
        iree_string_view_starts_with_char(line, ';') ||
        loom_check_emit_assembly_line_is_label(line)) {
      continue;
    }
    while (iree_string_view_starts_with_char(line, '{')) {
      iree_host_size_t closing_brace =
          iree_string_view_find(line, IREE_SV("}"), 0);
      if (closing_brace == IREE_STRING_VIEW_NPOS) {
        break;
      }
      line = iree_string_view_trim(
          iree_string_view_substr(line, closing_brace + 1, IREE_HOST_SIZE_MAX));
    }
    if (iree_string_view_is_empty(line)) {
      continue;
    }
    iree_host_size_t mnemonic_length = 0;
    while (mnemonic_length < line.size && line.data[mnemonic_length] != ' ' &&
           line.data[mnemonic_length] != '\t') {
      ++mnemonic_length;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        output, iree_string_view_substr(line, 0, mnemonic_length)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_emit_write_llvmir_assembly_mnemonics(
    const loom_llvmir_module_t* lowered_module,
    const loom_llvmir_target_profile_t* profile, iree_allocator_t allocator,
    loom_check_result_t* result) {
  loom_check_emit_byte_buffer_t bitcode = {0};
  iree_status_t status = loom_check_emit_write_llvmir_bitcode_bytes(
      lowered_module, allocator, &bitcode);

  loom_llvmir_tool_output_t assembly = {0};
  if (iree_status_is_ok(status)) {
    loom_llvmir_toolchain_t toolchain;
    loom_llvmir_toolchain_initialize_from_environment(&toolchain);
    loom_llvmir_target_profile_llc_arguments_t llc_arguments = {0};
    status = loom_llvmir_target_profile_llc_arguments(profile, &llc_arguments);
    if (iree_status_is_ok(status)) {
      status = loom_llvmir_tool_compile_assembly(
          &toolchain, iree_make_const_byte_span(bitcode.data, bitcode.length),
          llc_arguments.values, llc_arguments.count, allocator, &assembly);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_check_emit_write_assembly_mnemonics(
        iree_make_string_view(assembly.data, assembly.length),
        &result->actual_output);
  }

  loom_llvmir_tool_output_deinitialize(&assembly, allocator);
  loom_check_emit_byte_buffer_deinitialize(&bitcode, allocator);
  return status;
}

iree_status_t loom_check_execute_emit(
    const loom_check_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_check_result_t* result) {
  iree_arena_allocator_t diagnostic_arena;
  iree_arena_initialize(block_pool, &diagnostic_arena);
  loom_check_diagnostic_collector_t diagnostic_collector = {
      .arena = &diagnostic_arena,
      .host_allocator = allocator,
      .result = result,
  };

  loom_check_emit_request_t request;
  iree_status_t status =
      loom_check_emit_parse_request(test_case->emit_target, &request);
  if (!iree_status_is_ok(status)) {
    status = loom_check_emit_finish_status_failure(
        status, &diagnostic_collector, test_case, case_index, report, filename,
        allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  if (request.format == LOOM_CHECK_EMIT_LOW_DESCRIPTOR_MANIFEST) {
    status = loom_low_descriptor_set_format_manifest_json(
        request.low_descriptor_set, &result->actual_output);
    if (!iree_status_is_ok(status)) {
      status = loom_check_emit_finish_status_failure(
          status, &diagnostic_collector, test_case, case_index, report,
          filename, allocator, result);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    if (test_case->annotation_count > 0) {
      status = loom_check_diagnostic_collector_finish(
          &diagnostic_collector, test_case, case_index, report, allocator,
          result);
      iree_arena_deinitialize(&diagnostic_arena);
      return status;
    }
    status = loom_check_emit_compare_output(test_case, allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  iree_string_builder_t stripped_input;
  iree_string_builder_initialize(allocator, &stripped_input);
  status = loom_check_strip_comments(test_case->input, &stripped_input);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_deinitialize(&stripped_input);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  loom_module_t* module = NULL;
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink = {.fn = loom_check_diagnostic_collector_sink,
                          .user_data = &diagnostic_collector},
      .max_errors = 20,
  };
  status = loom_text_parse(iree_string_builder_view(&stripped_input), filename,
                           context, block_pool, &parse_options, &module);
  diagnostic_collector.module = module;
  iree_string_builder_deinitialize(&stripped_input);
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  if (!module || diagnostic_collector.count > 0) {
    status = loom_check_diagnostic_collector_finish(&diagnostic_collector,
                                                    test_case, case_index,
                                                    report, allocator, result);
    loom_module_free(module);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  loom_check_llvmir_legality_providers_t legality_providers;
  loom_check_llvmir_legality_providers_initialize(request.target_bundle,
                                                  &legality_providers);
  loom_llvmir_target_legality_options_t legality_options = {
      .snapshot = request.target_bundle->snapshot,
      .export_plan = request.target_bundle->export_plan,
      .config = request.target_bundle->config,
      .providers = legality_providers.providers,
      .provider_count = legality_providers.provider_count,
  };
  status = loom_llvmir_verify_target_legality(module, &legality_options, NULL);
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    status = loom_check_emit_finish_status_failure(
        status, &diagnostic_collector, test_case, case_index, report, filename,
        allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  loom_llvmir_target_profile_storage_t profile_storage;
  status = loom_llvmir_target_profile_storage_initialize_from_bundle(
      request.target_bundle, &profile_storage);
  if (!iree_status_is_ok(status)) {
    loom_module_free(module);
    status = loom_check_emit_finish_status_failure(
        status, &diagnostic_collector, test_case, case_index, report, filename,
        allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }
  const loom_llvmir_target_profile_t* profile = &profile_storage.profile;
  loom_check_llvmir_lowering_providers_t lowering_providers;
  loom_check_llvmir_lowering_providers_initialize(profile, &lowering_providers);
  loom_llvmir_lowering_options_t options = {
      .target_profile = profile,
      .source_name = filename,
      .providers = lowering_providers.providers,
      .provider_count = lowering_providers.provider_count,
  };
  loom_llvmir_module_t* lowered_module = NULL;
  status =
      loom_llvmir_lower_module(module, &options, allocator, &lowered_module);
  loom_module_free(module);
  diagnostic_collector.module = NULL;
  if (!iree_status_is_ok(status)) {
    loom_llvmir_module_free(lowered_module);
    status = loom_check_emit_finish_status_failure(
        status, &diagnostic_collector, test_case, case_index, report, filename,
        allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  status = loom_llvmir_verify_module(lowered_module);
  if (iree_status_is_ok(status)) {
    switch (request.format) {
      case LOOM_CHECK_EMIT_LLVMIR_TEXT:
        status = loom_check_emit_write_llvmir_text(lowered_module, result);
        break;
      case LOOM_CHECK_EMIT_LLVMIR_BODY_TEXT:
        status = loom_check_emit_write_llvmir_body_text(lowered_module,
                                                        allocator, result);
        break;
      case LOOM_CHECK_EMIT_LLVMIR_BITCODE_DISASSEMBLY:
        status = loom_check_emit_write_llvmir_bitcode_disassembly(
            lowered_module, allocator, result);
        break;
      case LOOM_CHECK_EMIT_LLVMIR_OBJECT:
        status = loom_check_emit_write_llvmir_object(lowered_module, profile,
                                                     allocator, result);
        break;
      case LOOM_CHECK_EMIT_LLVMIR_ASSEMBLY_MNEMONICS:
        status = loom_check_emit_write_llvmir_assembly_mnemonics(
            lowered_module, profile, allocator, result);
        break;
      case LOOM_CHECK_EMIT_LOW_DESCRIPTOR_MANIFEST:
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "low descriptor manifest emit should bypass LLVMIR lowering");
        break;
    }
  }
  loom_llvmir_module_free(lowered_module);
  if (!iree_status_is_ok(status)) {
    status = loom_check_emit_finish_status_failure(
        status, &diagnostic_collector, test_case, case_index, report, filename,
        allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  if (test_case->annotation_count > 0) {
    status = loom_check_diagnostic_collector_finish(&diagnostic_collector,
                                                    test_case, case_index,
                                                    report, allocator, result);
    iree_arena_deinitialize(&diagnostic_arena);
    return status;
  }

  status = loom_check_emit_compare_output(test_case, allocator, result);
  iree_arena_deinitialize(&diagnostic_arena);
  return status;
}
