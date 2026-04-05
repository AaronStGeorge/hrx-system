// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/json_sink.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"

namespace loom {
namespace {

// Helper: emit a diagnostic through the JSON sink and return the output.
std::string EmitJson(const loom_diagnostic_t* diagnostic,
                     loom_type_formatter_t type_formatter = {nullptr,
                                                             nullptr}) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loom_json_sink_options_t options = {
      .stream = &stream,
      .type_formatter = type_formatter,
  };
  iree_status_t status = loom_diagnostic_json_sink(&options, diagnostic);
  std::string result;
  if (iree_status_is_ok(status)) {
    result = std::string(iree_string_builder_buffer(&builder),
                         iree_string_builder_size(&builder));
  }
  iree_status_ignore(status);
  iree_string_builder_deinitialize(&builder);
  return result;
}

//===----------------------------------------------------------------------===//
// Basic structured diagnostics
//===----------------------------------------------------------------------===//

TEST(JsonSink, SimpleStructuredError) {
  // ERR_PARSE_001: "undefined SSA value '%{value_name}'"
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("x")),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = &loom_err_parse_001;
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.emitter = LOOM_EMITTER_PARSER;

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("\"severity\":\"error\""), std::string::npos);
  EXPECT_NE(json.find("\"domain\":\"PARSE\""), std::string::npos);
  EXPECT_NE(json.find("\"code\":1"), std::string::npos);
  EXPECT_NE(json.find("\"emitter\":\"parser\""), std::string::npos);
  EXPECT_NE(json.find("\"message\":\"undefined SSA value '%x'\""),
            std::string::npos);
  // PARSE_001 has no fix hint.
  EXPECT_EQ(json.find("\"fix_hint\""), std::string::npos);
  // Params object.
  EXPECT_NE(json.find("\"params\":{"), std::string::npos);
  EXPECT_NE(json.find("\"value_name\":\"x\""), std::string::npos);
  // Ends with newline.
  EXPECT_EQ(json.back(), '\n');
}

TEST(JsonSink, WarningFormat) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("y")),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_WARNING;
  diagnostic.error = &loom_err_parse_001;
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("\"severity\":\"warning\""), std::string::npos);
}

//===----------------------------------------------------------------------===//
// Structured diagnostics with type params
//===----------------------------------------------------------------------===//

TEST(JsonSink, StructuredSameType) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_diagnostic_param_t params[4] = {
      loom_param_string(IREE_SV("lhs")),
      loom_param_type(i32_type),
      loom_param_string(IREE_SV("rhs")),
      loom_param_type(f32_type),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = &loom_err_type_001;
  diagnostic.params = params;
  diagnostic.param_count = 4;
  diagnostic.emitter = LOOM_EMITTER_VERIFIER;

  std::string json = EmitJson(&diagnostic, {loom_type_format_minimal, nullptr});

  // Check structured fields.
  EXPECT_NE(json.find("\"domain\":\"TYPE\""), std::string::npos);
  EXPECT_NE(json.find("\"code\":1"), std::string::npos);
  EXPECT_NE(json.find("\"emitter\":\"verifier\""), std::string::npos);

  // Check rendered message.
  EXPECT_NE(json.find("'lhs' type i32 does not match 'rhs' type f32"),
            std::string::npos);

  // Check fix_hint.
  EXPECT_NE(json.find("\"fix_hint\":"), std::string::npos);
  EXPECT_NE(json.find("Ensure 'lhs' and 'rhs' have the same type"),
            std::string::npos);

  // Check params object.
  EXPECT_NE(json.find("\"params\":{"), std::string::npos);
  EXPECT_NE(json.find("\"field_a\":\"lhs\""), std::string::npos);
  EXPECT_NE(json.find("\"type_a\":\"i32\""), std::string::npos);
  EXPECT_NE(json.find("\"field_b\":\"rhs\""), std::string::npos);
  EXPECT_NE(json.find("\"type_b\":\"f32\""), std::string::npos);
}

TEST(JsonSink, StructuredStructureError) {
  loom_diagnostic_param_t params[3] = {
      loom_param_string(IREE_SV("test.addi")),
      loom_param_u32(3),
      loom_param_u32(2),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = &loom_err_structure_001;
  diagnostic.params = params;
  diagnostic.param_count = 3;
  diagnostic.emitter = LOOM_EMITTER_VERIFIER;

  std::string json = EmitJson(&diagnostic);

  EXPECT_NE(json.find("\"domain\":\"STRUCTURE\""), std::string::npos);
  EXPECT_NE(json.find("\"code\":1"), std::string::npos);
  // STRUCTURE_001 has no fix_hint.
  EXPECT_EQ(json.find("\"fix_hint\""), std::string::npos);
  // U32 params render as numbers.
  EXPECT_NE(json.find("\"actual_count\":3"), std::string::npos);
  EXPECT_NE(json.find("\"expected_count\":2"), std::string::npos);
}

//===----------------------------------------------------------------------===//
// JSON escaping
//===----------------------------------------------------------------------===//

TEST(JsonSink, EscapesSpecialCharacters) {
  // ERR_PARSE_001 with a name containing special characters.
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("line1\nline2\ttab\"quote\\backslash")),
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = &loom_err_parse_001;
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(json.find("line1\\nline2\\ttab\\\"quote\\\\backslash"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Source locations and highlights
//===----------------------------------------------------------------------===//

TEST(JsonSink, SerializesSourceRangesAndHighlights) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("x")),
  };
  const char source[] = "%x = test.constant 0 : i32";
  loom_highlight_range_t highlights[] = {
      {0, 2},
      {5, 18},
  };

  loom_diagnostic_t diagnostic = {};
  diagnostic.severity = LOOM_DIAGNOSTIC_ERROR;
  diagnostic.error = &loom_err_parse_001;
  diagnostic.params = params;
  diagnostic.param_count = IREE_ARRAYSIZE(params);
  diagnostic.emitter = LOOM_EMITTER_VERIFIER;
  diagnostic.origin.filename = IREE_SV("<verifier>");
  diagnostic.origin.source = iree_make_cstring_view(source);
  diagnostic.origin.start = 0;
  diagnostic.origin.end = 26;
  diagnostic.origin.start_line = 1;
  diagnostic.origin.start_column = 1;
  diagnostic.origin.end_line = 1;
  diagnostic.origin.end_column = 27;
  diagnostic.source_location.filename = IREE_SV("model.loom");
  diagnostic.source_location.source = iree_make_cstring_view(source);
  diagnostic.source_location.start = 64;
  diagnostic.source_location.end = 90;
  diagnostic.source_location.start_line = 7;
  diagnostic.source_location.start_column = 3;
  diagnostic.source_location.end_line = 7;
  diagnostic.source_location.end_column = 29;
  diagnostic.highlights = highlights;
  diagnostic.highlight_count = IREE_ARRAYSIZE(highlights);

  std::string json = EmitJson(&diagnostic);
  EXPECT_NE(
      json.find("\"origin\":{\"filename\":\"<verifier>\",\"start_line\":1,"
                "\"start_column\":1,\"end_line\":1,\"end_column\":27,"
                "\"start_byte\":0,\"end_byte\":26}"),
      std::string::npos);
  EXPECT_NE(json.find("\"source_location\":{\"filename\":\"model.loom\","
                      "\"start_line\":7,\"start_column\":3,\"end_line\":7,"
                      "\"end_column\":29,\"start_byte\":64,\"end_byte\":90}"),
            std::string::npos);
  EXPECT_NE(json.find("\"highlights\":[{\"start_byte\":0,\"end_byte\":2},"
                      "{\"start_byte\":5,\"end_byte\":18}]"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Multiple diagnostics (JSONL)
//===----------------------------------------------------------------------===//

TEST(JsonSink, MultipleDiagnostics) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loom_json_sink_options_t options = {
      .stream = &stream,
      .type_formatter = {nullptr, nullptr},
  };

  loom_diagnostic_param_t params1[] = {
      loom_param_string(IREE_SV("first")),
  };
  loom_diagnostic_t d1 = {};
  d1.severity = LOOM_DIAGNOSTIC_ERROR;
  d1.error = &loom_err_parse_001;
  d1.params = params1;
  d1.param_count = IREE_ARRAYSIZE(params1);
  IREE_ASSERT_OK(loom_diagnostic_json_sink(&options, &d1));

  loom_diagnostic_param_t params2[] = {
      loom_param_string(IREE_SV("second")),
  };
  loom_diagnostic_t d2 = {};
  d2.severity = LOOM_DIAGNOSTIC_WARNING;
  d2.error = &loom_err_parse_001;
  d2.params = params2;
  d2.param_count = IREE_ARRAYSIZE(params2);
  IREE_ASSERT_OK(loom_diagnostic_json_sink(&options, &d2));

  std::string output(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  // Two newline-terminated lines.
  size_t first_newline = output.find('\n');
  ASSERT_NE(first_newline, std::string::npos);
  size_t second_newline = output.find('\n', first_newline + 1);
  ASSERT_NE(second_newline, std::string::npos);

  std::string line1 = output.substr(0, first_newline);
  std::string line2 =
      output.substr(first_newline + 1, second_newline - first_newline - 1);

  EXPECT_NE(line1.find("%first"), std::string::npos);
  EXPECT_NE(line2.find("%second"), std::string::npos);
}

}  // namespace
}  // namespace loom
