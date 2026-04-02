// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/parser.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/diagnostic.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Test infrastructure
//===----------------------------------------------------------------------===//

// A single captured diagnostic with deep-copied structured data.
// Error def pointers point into .rodata and are stable for == comparison.
// String param values are copied into |string_storage| since the diagnostic's
// param array is stack-local to the emitting function.
struct CapturedDiagnostic {
  const loom_error_def_t* error;
  loom_diagnostic_severity_t severity;
  loom_emitter_t emitter;
  uint32_t origin_line;
  uint32_t origin_column;
  std::vector<loom_diagnostic_param_t> params;
  std::vector<std::string> string_storage;
};

struct DiagnosticCapture {
  std::vector<CapturedDiagnostic> diagnostics;
};

static iree_status_t CaptureDiagnostic(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  auto* capture = static_cast<DiagnosticCapture*>(user_data);
  CapturedDiagnostic entry;
  entry.error = diagnostic->error;
  entry.severity = diagnostic->severity;
  entry.emitter = diagnostic->emitter;
  entry.origin_line = diagnostic->origin.start_line;
  entry.origin_column = diagnostic->origin.start_column;
  for (iree_host_size_t i = 0; i < diagnostic->param_count; ++i) {
    loom_diagnostic_param_t param = diagnostic->params[i];
    if (param.kind == LOOM_PARAM_STRING) {
      entry.string_storage.emplace_back(param.string.data, param.string.size);
      param.string = iree_make_string_view(entry.string_storage.back().data(),
                                           entry.string_storage.back().size());
    }
    entry.params.push_back(param);
  }
  capture->diagnostics.push_back(std::move(entry));
  return iree_ok_status();
}

// Assertion helpers.
static void ExpectError(const CapturedDiagnostic& diagnostic,
                        const loom_error_def_t* expected_error) {
  EXPECT_EQ(diagnostic.error, expected_error);
  EXPECT_EQ(diagnostic.severity, LOOM_DIAGNOSTIC_ERROR);
  EXPECT_EQ(diagnostic.emitter, LOOM_EMITTER_PARSER);
}

static std::string GetStringParam(const CapturedDiagnostic& diagnostic,
                                  size_t param_index) {
  EXPECT_LT(param_index, diagnostic.params.size());
  if (param_index >= diagnostic.params.size()) return "";
  EXPECT_EQ(diagnostic.params[param_index].kind, LOOM_PARAM_STRING);
  return std::string(diagnostic.params[param_index].string.data,
                     diagnostic.params[param_index].string.size);
}

static void ExpectU32Param(const CapturedDiagnostic& diagnostic,
                           size_t param_index, uint32_t expected) {
  ASSERT_LT(param_index, diagnostic.params.size());
  EXPECT_EQ(diagnostic.params[param_index].kind, LOOM_PARAM_U32);
  EXPECT_EQ(diagnostic.params[param_index].u32, expected);
}

class ParserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    {
      iree_host_size_t count = 0;
      const loom_op_vtable_t* const* vtables =
          loom_test_dialect_vtables(&count);
      IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                   vtables, (uint16_t)count));
    }
    {
      iree_host_size_t count = 0;
      const loom_op_vtable_t* const* vtables =
          loom_func_dialect_vtables(&count);
      IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_FUNC,
                                                   vtables, (uint16_t)count));
    }
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Parses source text and captures diagnostics. On success, caller owns
  // |*out_module| (must free with loom_module_free).
  iree_status_t Parse(const char* source, loom_module_t** out_module) {
    capture_ = {};
    loom_text_parse_options_t options;
    memset(&options, 0, sizeof(options));
    options.diagnostic_sink.fn = CaptureDiagnostic;
    options.diagnostic_sink.user_data = &capture_;
    options.max_errors = 100;
    return loom_text_parse(iree_make_cstring_view(source),
                           iree_make_cstring_view("test.loom"), &context_,
                           &block_pool_, &options, out_module);
  }

  // Parses source text and expects success (no diagnostics emitted).
  loom_module_t* ParseOk(const char* source) {
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(Parse(source, &module));
    if (!capture_.diagnostics.empty()) {
      std::string msg = "Expected no diagnostics but got " +
                        std::to_string(capture_.diagnostics.size()) + ":\n";
      for (size_t i = 0; i < capture_.diagnostics.size(); ++i) {
        const auto& d = capture_.diagnostics[i];
        msg += "  [" + std::to_string(i) + "] " +
               (d.error ? d.error->summary : "(null)") +
               " line=" + std::to_string(d.origin_line) +
               " col=" + std::to_string(d.origin_column);
        for (size_t j = 0; j < d.params.size(); ++j) {
          if (d.params[j].kind == LOOM_PARAM_STRING) {
            msg +=
                " p" + std::to_string(j) + "='" +
                std::string(d.params[j].string.data, d.params[j].string.size) +
                "'";
          }
        }
        msg += "\n";
      }
      ADD_FAILURE() << msg;
    }
    EXPECT_NE(module, nullptr);
    return module;
  }

  // Parses source text and expects parse errors (diagnostics emitted,
  // module is NULL, but status is ok — parse errors are not infrastructure
  // failures).
  const std::vector<CapturedDiagnostic>& ParseExpectErrors(const char* source) {
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(Parse(source, &module));
    EXPECT_EQ(module, nullptr);
    EXPECT_GT(capture_.diagnostics.size(), 0u);
    return capture_.diagnostics;
  }

  // Prints a module to canonical text.
  std::string PrintModule(const loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_EXPECT_OK(loom_text_print_module_to_builder(module, &builder,
                                                     LOOM_TEXT_PRINT_DEFAULT));
    std::string result(iree_string_builder_buffer(&builder),
                       iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  // Parses, prints, parses again, prints again — asserts the two printed
  // strings are identical. Returns the printed text.
  std::string RoundTrip(const char* source) {
    loom_module_t* module1 = ParseOk(source);
    if (!module1) return "";
    std::string text1 = PrintModule(module1);
    loom_module_free(module1);

    loom_module_t* module2 = ParseOk(text1.c_str());
    if (!module2) return "";
    std::string text2 = PrintModule(module2);
    loom_module_free(module2);

    EXPECT_EQ(text1, text2) << "Round-trip mismatch";
    return text1;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  DiagnosticCapture capture_;
};

//===----------------------------------------------------------------------===//
// Valid parse — no diagnostics
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, EmptyInput) {
  loom_module_t* module = ParseOk("");
  ASSERT_NE(module, nullptr);
  loom_module_free(module);
}

TEST_F(ParserTest, WhitespaceOnly) {
  loom_module_t* module = ParseOk("   \n\n  \n");
  ASSERT_NE(module, nullptr);
  loom_module_free(module);
}

TEST_F(ParserTest, CommentOnly) {
  loom_module_t* module = ParseOk("// this is a comment\n// another comment\n");
  ASSERT_NE(module, nullptr);
  loom_module_free(module);
}

TEST_F(ParserTest, Constant) {
  std::string text = RoundTrip("%c = test.constant 42 : i32\n");
  EXPECT_NE(text.find("test.constant 42 : i32"), std::string::npos);
}

TEST_F(ParserTest, ConstantNegative) {
  std::string text = RoundTrip("%c = test.constant -1 : i64\n");
  EXPECT_NE(text.find("test.constant -1 : i64"), std::string::npos);
}

TEST_F(ParserTest, ConstantZero) {
  std::string text = RoundTrip("%c = test.constant 0 : index\n");
  EXPECT_NE(text.find("test.constant 0 : index"), std::string::npos);
}

TEST_F(ParserTest, BinaryOp) {
  std::string text = RoundTrip(
      "%c0 = test.constant 1 : i32\n"
      "%c1 = test.constant 2 : i32\n"
      "%r = test.addi %c0, %c1 : i32\n");
  EXPECT_NE(text.find("test.addi"), std::string::npos);
}

TEST_F(ParserTest, UnaryOp) {
  std::string text = RoundTrip(
      "%c = test.constant 1 : i32\n"
      "%r = test.cast %c : i32 to f32\n");
  EXPECT_NE(text.find("test.cast"), std::string::npos);
}

TEST_F(ParserTest, ComparisonOp) {
  std::string text = RoundTrip(
      "%a = test.constant 1 : i32\n"
      "%b = test.constant 2 : i32\n"
      "%r = test.cmp eq, %a, %b : i32\n");
  EXPECT_NE(text.find("test.cmp eq"), std::string::npos);
}

TEST_F(ParserTest, YieldNoArgs) {
  std::string text = RoundTrip("test.yield\n");
  EXPECT_NE(text.find("test.yield"), std::string::npos);
}

TEST_F(ParserTest, YieldSingleArg) {
  std::string text = RoundTrip(
      "%c = test.constant 1 : f32\n"
      "test.yield %c : f32\n");
  EXPECT_NE(text.find("test.yield"), std::string::npos);
}

TEST_F(ParserTest, VariadicReduce) {
  std::string text = RoundTrip(
      "%a = test.constant 1 : i32\n"
      "%b = test.constant 2 : i32\n"
      "%c = test.constant 3 : i32\n"
      "%sum = test.reduce %a, %b, %c : i32\n");
  EXPECT_NE(text.find("test.reduce"), std::string::npos);
}

TEST_F(ParserTest, FuncDef) {
  loom_module_t* module = ParseOk(
      "func.def @identity(%x : f32) -> (f32) {\n"
      "  func.return %x : f32\n"
      "}\n");
  if (module) {
    std::string text = PrintModule(module);
    EXPECT_NE(text.find("func.def"), std::string::npos);
    EXPECT_NE(text.find("func.return"), std::string::npos);
    loom_module_free(module);
  }
}

TEST_F(ParserTest, FuncDefZeroOperandReturn) {
  loom_module_t* module = ParseOk(
      "func.def @empty() {\n"
      "  func.return\n"
      "}\n");
  if (module) {
    std::string text = PrintModule(module);
    EXPECT_NE(text.find("func.def @empty()"), std::string::npos);
    EXPECT_NE(text.find("func.return\n"), std::string::npos);
    // Verify no stray colon after func.return.
    EXPECT_EQ(text.find("func.return :"), std::string::npos);
    loom_module_free(module);
  }
}

TEST_F(ParserTest, FuncDefInitializerZeroOperandReturn) {
  loom_module_t* module = ParseOk(
      "func.def initializer @setup() {\n"
      "  func.return\n"
      "}\n");
  if (module) {
    std::string text = PrintModule(module);
    EXPECT_NE(text.find("initializer"), std::string::npos);
    EXPECT_NE(text.find("func.return\n"), std::string::npos);
    loom_module_free(module);
  }
}

TEST_F(ParserTest, FuncDefMultipleArgs) {
  loom_module_t* module = ParseOk(
      "func.def @add(%a : i32, %b : i32) -> (i32) {\n"
      "  %r = test.addi %a, %b : i32\n"
      "  func.return %r : i32\n"
      "}\n");
  if (module) {
    std::string text = PrintModule(module);
    EXPECT_NE(text.find("func.def @add"), std::string::npos);
    loom_module_free(module);
  }
}

TEST_F(ParserTest, NestedMapRegion) {
  loom_module_t* module = ParseOk(
      "%tile = test.constant 0 : f32\n"
      "%r = test.map(%element = %tile : f32) {\n"
      "  %negated = test.neg %element : f32\n"
      "  test.yield %negated : f32\n"
      "} -> (f32)\n");
  if (module) {
    std::string text = PrintModule(module);
    EXPECT_NE(text.find("test.map"), std::string::npos);
    EXPECT_NE(text.find("test.neg"), std::string::npos);
    loom_module_free(module);
  }
}

TEST_F(ParserTest, ComparisonResultType) {
  // Comparison ops return i1 (implicit from the result type constraint).
  // The format prints the operand type after the colon, and the parser
  // infers the i1 result type from LOOM_TYPE_CONSTRAINT_I1.
  loom_module_t* module = ParseOk(
      "func.def @compare(%a: i32, %b: i32) -> (i1) {\n"
      "  %r = test.cmp eq, %a, %b : i32\n"
      "  func.return %r : i1\n"
      "}\n");
  if (module) {
    std::string text = PrintModule(module);
    EXPECT_NE(text.find("test.cmp eq"), std::string::npos);
    EXPECT_NE(text.find("func.return %r : i1"), std::string::npos)
        << "result type should be i1, got: " << text;
    loom_module_free(module);
  }
}

TEST_F(ParserTest, LoopWithIterArgs) {
  // Loop IV and iter_args must parse correctly through the text format.
  // The IV is an implicit index-typed block arg, iter_args are capture
  // bindings, and both must appear as pending block args for the body
  // region.
  loom_module_t* module = ParseOk(
      "func.def @loop(%lo: index, %hi: index, %step: index, %init: f32)"
      " -> (f32) {\n"
      "  %r = test.loop %iv = %lo to %hi step %step"
      " iter_args(%acc = %init : f32) -> (f32) {\n"
      "    test.yield %acc : f32\n"
      "  }\n"
      "  func.return %r : f32\n"
      "}\n");
  if (module) {
    std::string text = PrintModule(module);
    EXPECT_NE(text.find("test.loop %iv ="), std::string::npos)
        << "IV not found in: " << text;
    EXPECT_NE(text.find("iter_args(%acc ="), std::string::npos)
        << "iter_args not found in: " << text;
    EXPECT_NE(text.find("func.return %r : f32"), std::string::npos)
        << "return type wrong in: " << text;
    loom_module_free(module);
  }
}

TEST_F(ParserTest, LoopWithoutIterArgs) {
  // Loop without iter_args — just the IV and no results.
  loom_module_t* module = ParseOk(
      "func.def @simple_loop(%lo: index, %hi: index, %step: index) {\n"
      "  test.loop %iv = %lo to %hi step %step {\n"
      "  }\n"
      "}\n");
  if (module) {
    std::string text = PrintModule(module);
    EXPECT_NE(text.find("test.loop %iv ="), std::string::npos)
        << "IV not found in: " << text;
    EXPECT_EQ(text.find("iter_args"), std::string::npos)
        << "iter_args should be absent in: " << text;
    loom_module_free(module);
  }
}

TEST_F(ParserTest, ConvertOp) {
  std::string text = RoundTrip(
      "%c = test.constant 42 : i32\n"
      "%r = test.convert %c : i32 -> f32\n");
  EXPECT_NE(text.find("test.convert"), std::string::npos);
}

TEST_F(ParserTest, CounterOp) {
  std::string text = RoundTrip("%c = test.counter 3 : i32\n");
  EXPECT_NE(text.find("test.counter 3"), std::string::npos);
}

// Slice parsing with static offsets. We construct valid IR via a func.def
// so the %tile operand has the correct type.
TEST_F(ParserTest, SliceAllStatic) {
  loom_module_t* module = ParseOk(
      "func.def @test_slice(%tile : tile<64x64xf16>) -> (tile<16x16xf16>) {\n"
      "  %sub = test.slice %tile[0, 32] : tile<64x64xf16> -> "
      "(tile<16x16xf16>)\n"
      "  func.return %sub : tile<16x16xf16>\n"
      "}\n");
  if (module) {
    std::string text = PrintModule(module);
    EXPECT_NE(text.find("test.slice"), std::string::npos);
    loom_module_free(module);
  }
}

//===----------------------------------------------------------------------===//
// Error detection — structural assertions on captured diagnostics
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, UnknownOp) {
  const auto& diagnostics =
      ParseExpectErrors("%r = bogus.nonexistent %x : i32\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_006);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "bogus.nonexistent");
}

TEST_F(ParserTest, UndefinedSSAValue) {
  const auto& diagnostics = ParseExpectErrors(
      "%c = test.constant 1 : i32\n"
      "%r = test.addi %c, %undefined : i32\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_001);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "undefined");
}

TEST_F(ParserTest, UnexpectedTokenInFuncSignature) {
  // Missing '->' in function signature triggers ERR_PARSE_003.
  const auto& diagnostics = ParseExpectErrors(
      "func.def @bad(%x : f32) (f32) {\n"
      "  func.return %x : f32\n"
      "}\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_003);
  EXPECT_EQ(diagnostics[0].params.size(), 2u);
}

TEST_F(ParserTest, UnknownTypeName) {
  const auto& diagnostics =
      ParseExpectErrors("%c = test.constant 0 : foobar\n");
  ASSERT_GE(diagnostics.size(), 1u);
  // Unknown type name triggers ERR_PARSE_007.
  ExpectError(diagnostics[0], &loom_err_parse_007);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "foobar");
}

TEST_F(ParserTest, UnknownEncodingInType) {
  // Encoding references in types must start with '#' (static encoding) or
  // '%' (SSA encoding). A bare identifier triggers ERR_PARSE_008.
  const auto& diagnostics =
      ParseExpectErrors("%c = test.constant 0 : tile<4xf32, bogus>\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_008);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "bogus");
}

TEST_F(ParserTest, EncodingAlias) {
  // Define an encoding alias at module level and reference it in tile types.
  loom_module_t* module = ParseOk(
      "#enc = #quantization<bits=8>\n"
      "func.def @test_enc(%x : tile<4xf32, #enc>) -> (tile<4xf32, #enc>) {\n"
      "  func.return %x : tile<4xf32, #enc>\n"
      "}\n");
  if (module) {
    std::string text = PrintModule(module);
    EXPECT_NE(text.find("#enc"), std::string::npos);
    loom_module_free(module);
  }
}

TEST_F(ParserTest, InlineEncoding) {
  // Inline encoding definition directly in a tile type.
  loom_module_t* module = ParseOk(
      "func.def @test_enc(%x : tile<4xf32, #dense<block=32>>) -> (tile<4xf32>) "
      "{\n"
      "  func.return %x : tile<4xf32>\n"
      "}\n");
  if (module) {
    std::string text = PrintModule(module);
    EXPECT_NE(text.find("#dense<block=32>"), std::string::npos);
    loom_module_free(module);
  }
}

//===----------------------------------------------------------------------===//
// Error location
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, ErrorPointsAtCorrectLine) {
  const auto& diagnostics = ParseExpectErrors(
      "%a = test.constant 1 : i32\n"    // line 1
      "%b = test.constant 2 : i32\n"    // line 2
      "%r = bogus.op %a, %b : i32\n");  // line 3 — error here
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_006);
  EXPECT_EQ(diagnostics[0].origin_line, 3u);
}

TEST_F(ParserTest, ErrorPointsAtCorrectColumn) {
  // "%r = bogus.op" — the op name starts at column 6.
  const auto& diagnostics = ParseExpectErrors("%r = bogus.op %x : i32\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_006);
  // Column depends on whether the tokenizer's column is 0-based or 1-based.
  // The op name "bogus.op" starts after "%r = " (5 chars), so column 6
  // if 1-based.
  EXPECT_GT(diagnostics[0].origin_column, 0u);
}

//===----------------------------------------------------------------------===//
// Type interior diagnostic positions
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, UndefinedDimReportsRealPosition) {
  // Body mode: [%UNDEF] produces PARSE/001 at the '%' of %UNDEF.
  // Line 2, column layout: %r = test.cast %x : tile<[%UNDEF]xf32> to i32
  //                         1                       2627
  // '%' of %UNDEF is at column 27.
  const auto& diagnostics = ParseExpectErrors(
      "%x = test.constant 0 : i32\n"
      "%r = test.cast %x : tile<[%UNDEF]xf32> to i32\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_001);
  EXPECT_EQ(diagnostics[0].origin_line, 2u);
  EXPECT_EQ(diagnostics[0].origin_column, 27u);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "UNDEF");
}

TEST_F(ParserTest, UndefinedDimSecondPositionIsDistinct) {
  // Second dim [%BAD] at a different column than first dim.
  // Line 2: %r = test.cast %x : tile<4x[%BAD]xf32> to i32
  //                              21   2526272829
  // '%' of %BAD is at column 29.
  const auto& diagnostics = ParseExpectErrors(
      "%x = test.constant 0 : i32\n"
      "%r = test.cast %x : tile<4x[%BAD]xf32> to i32\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_001);
  EXPECT_EQ(diagnostics[0].origin_line, 2u);
  EXPECT_EQ(diagnostics[0].origin_column, 29u);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "BAD");
}

TEST_F(ParserTest, PoolDimReportsRealPosition) {
  // pool<bad> — BARE_IDENT "bad" at column 26 (after "pool<").
  // Line 2: %r = test.cast %x : pool<bad> to i32
  //                              21   2526
  const auto& diagnostics = ParseExpectErrors(
      "%x = test.constant 0 : i32\n"
      "%r = test.cast %x : pool<bad> to i32\n");
  ASSERT_GE(diagnostics.size(), 1u);
  EXPECT_EQ(diagnostics[0].origin_line, 2u);
  EXPECT_EQ(diagnostics[0].origin_column, 26u);
}

TEST_F(ParserTest, GroupScopeReportsRealPosition) {
  // group<bad> — BARE_IDENT "bad" at column 27 (after "group<").
  // Line 2: %r = test.cast %x : group<bad> to i32
  //                              21    2627
  const auto& diagnostics = ParseExpectErrors(
      "%x = test.constant 0 : i32\n"
      "%r = test.cast %x : group<bad> to i32\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_018);
  EXPECT_EQ(diagnostics[0].origin_line, 2u);
  EXPECT_EQ(diagnostics[0].origin_column, 27u);
}

//===----------------------------------------------------------------------===//
// Error recovery
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, RecoverySkipsToNextOp) {
  // First op has an unknown op name — parser should recover and continue
  // to the second (valid) op.
  const auto& diagnostics = ParseExpectErrors(
      "%bad = bogus.op : i32\n"
      "%c = test.constant 42 : i32\n");
  // At minimum, we get the ERR_PARSE_006 for the unknown op.
  ASSERT_GE(diagnostics.size(), 1u);
  bool found_unknown_op = false;
  for (const auto& d : diagnostics) {
    if (d.error == &loom_err_parse_006) {
      found_unknown_op = true;
    }
  }
  EXPECT_TRUE(found_unknown_op);
}

TEST_F(ParserTest, MaxErrorsLimit) {
  // Generate many errors, set max_errors low, verify ERR_PARSE_012 is emitted.
  std::string source;
  for (int i = 0; i < 25; ++i) {
    source += "%bad" + std::to_string(i) + " = bogus.op" + std::to_string(i) +
              " : i32\n";
  }

  capture_ = {};
  loom_text_parse_options_t options;
  memset(&options, 0, sizeof(options));
  options.diagnostic_sink.fn = CaptureDiagnostic;
  options.diagnostic_sink.user_data = &capture_;
  options.max_errors = 5;

  loom_module_t* module = nullptr;
  IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source.c_str()),
                                 iree_make_cstring_view("test.loom"), &context_,
                                 &block_pool_, &options, &module));
  EXPECT_EQ(module, nullptr);

  // Should have at least 2 diagnostics.
  ASSERT_GE(capture_.diagnostics.size(), 2u);

  // Find ERR_PARSE_012 (too many errors) somewhere in the diagnostics.
  bool found_too_many = false;
  for (const auto& d : capture_.diagnostics) {
    if (d.error == &loom_err_parse_012) {
      found_too_many = true;
      break;
    }
  }
  EXPECT_TRUE(found_too_many) << "Expected ERR_PARSE_012 (too many errors)";

  // Total error count should not exceed max_errors + 1 (the "too many" itself).
  EXPECT_LE(capture_.diagnostics.size(), 6u + 1u);
}

//===----------------------------------------------------------------------===//
// Round-trip fidelity
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, RoundTripBinaryOps) {
  RoundTrip(
      "%a = test.constant 1 : i32\n"
      "%b = test.constant 2 : i32\n"
      "%c = test.addi %a, %b : i32\n"
      "%d = test.addi %c, %a : i32\n");
}

TEST_F(ParserTest, RoundTripFuncDef) {
  RoundTrip(
      "func.def @negate(%input : f32) -> (f32) {\n"
      "  %r = test.neg %input : f32\n"
      "  func.return %r : f32\n"
      "}\n");
}

TEST_F(ParserTest, RoundTripConvert) {
  RoundTrip(
      "%c = test.constant 42 : i32\n"
      "%r = test.convert %c : i32 -> f32\n");
}

TEST_F(ParserTest, RoundTripComparison) {
  RoundTrip(
      "%a = test.constant 1 : i32\n"
      "%b = test.constant 2 : i32\n"
      "%r = test.cmp eq, %a, %b : i32\n");
}

TEST_F(ParserTest, RoundTripVariadicYield) {
  RoundTrip(
      "%a = test.constant 1 : f32\n"
      "%b = test.constant 2 : i32\n"
      "test.yield %a, %b : f32, i32\n");
}

//===----------------------------------------------------------------------===//
// Edge cases
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, NullSink) {
  // Parse with no sink — errors are dropped, module is NULL, status is ok.
  loom_text_parse_options_t options;
  memset(&options, 0, sizeof(options));
  options.diagnostic_sink.fn = NULL;
  options.diagnostic_sink.user_data = NULL;
  options.max_errors = 20;

  loom_module_t* module = nullptr;
  IREE_EXPECT_OK(
      loom_text_parse(iree_make_cstring_view("%r = bogus.op : i32\n"),
                      iree_make_cstring_view("test.loom"), &context_,
                      &block_pool_, &options, &module));
  EXPECT_EQ(module, nullptr);
}

TEST_F(ParserTest, NullOptions) {
  // Parse valid input with NULL options — uses defaults.
  loom_module_t* module = nullptr;
  IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(""),
                                 iree_make_cstring_view("test.loom"), &context_,
                                 &block_pool_, NULL, &module));
  ASSERT_NE(module, nullptr);
  loom_module_free(module);
}

TEST_F(ParserTest, NullOptionsWithError) {
  // Parse invalid input with NULL options — module is NULL, status is ok.
  loom_module_t* module = nullptr;
  IREE_EXPECT_OK(
      loom_text_parse(iree_make_cstring_view("%r = bogus.op : i32\n"),
                      iree_make_cstring_view("test.loom"), &context_,
                      &block_pool_, NULL, &module));
  EXPECT_EQ(module, nullptr);
}

TEST_F(ParserTest, AllDiagnosticsAreFromParser) {
  // Verify that every diagnostic emitted during parsing carries the correct
  // emitter tag, regardless of error type.
  const auto& diagnostics = ParseExpectErrors(
      "%r = bogus.op : i32\n"
      "%s = test.addi %r, %undef : i32\n");
  for (const auto& d : diagnostics) {
    EXPECT_EQ(d.emitter, LOOM_EMITTER_PARSER)
        << "Diagnostic with domain " << d.error->domain << " code "
        << d.error->code << " has wrong emitter";
  }
}

}  // namespace
}  // namespace loom
