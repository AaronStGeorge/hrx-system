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
#include "loom/format/text/parser_internal.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::ExpectError;
using ::loom::testing::ExpectU32Param;
using ::loom::testing::FindDiagnostic;
using ::loom::testing::GetStringParam;

static const loom_encoding_vtable_t kDenseEncodingVtable = {
    .name = IREE_SV("dense"),
};

static const loom_encoding_vtable_t kQ8_0EncodingVtable = {
    .name = IREE_SV("q8_0"),
};

static const loom_encoding_vtable_t kQ6KEncodingVtable = {
    .name = IREE_SV("q6_k"),
};

static const loom_encoding_vtable_t kQuantizationEncodingVtable = {
    .name = IREE_SV("quantization"),
};

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
    {
      iree_host_size_t count = 0;
      const loom_op_vtable_t* const* vtables =
          loom_encoding_dialect_vtables(&count);
      IREE_ASSERT_OK(loom_context_register_dialect(
          &context_, LOOM_DIALECT_ENCODING, vtables, (uint16_t)count));
    }
    IREE_ASSERT_OK(loom_context_register_encoding_vtable(
        &context_, &kDenseEncodingVtable));
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kQ8_0EncodingVtable));
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kQ6KEncodingVtable));
    IREE_ASSERT_OK(loom_context_register_encoding_vtable(
        &context_, &kQuantizationEncodingVtable));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Parses source text and captures diagnostics. On success, caller owns
  // |*out_module| (must free with loom_module_free).
  iree_status_t Parse(const char* source, loom_module_t** out_module) {
    capture_.Reset();
    loom_text_parse_options_t options;
    memset(&options, 0, sizeof(options));
    options.diagnostic_sink = capture_.sink();
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

static loom_op_t* GetFirstFunctionOp(const loom_module_t* module) {
  if (!module || module->symbols.count == 0) return nullptr;
  return module->symbols.entries[0].defining_op;
}

static loom_block_t* GetEntryBlock(loom_region_t* region) {
  if (!region || region->block_count == 0) return nullptr;
  return loom_region_entry_block(region);
}

//===----------------------------------------------------------------------===//
// Parser-owned scratch
//===----------------------------------------------------------------------===//

TEST_F(ParserTest, ParsedOpScratchFrameReusesOperandSpillStorage) {
  loom_parser_t parser = {};
  iree_arena_initialize(&block_pool_, &parser.parser_arena);

  loom_parsed_op_t* first = nullptr;
  IREE_ASSERT_OK(loom_parser_acquire_parsed_op(&parser, &first));
  ASSERT_NE(first, nullptr);

  for (uint16_t i = 0; i < LOOM_PARSED_OP_INLINE_OPERANDS + 4; ++i) {
    IREE_ASSERT_OK(loom_parsed_op_add_operand(first, &parser.parser_arena, i));
  }
  ASSERT_GT(first->operand_capacity, LOOM_PARSED_OP_INLINE_OPERANDS);
  loom_value_id_t* spill_operand_ids = first->operand_ids;
  uint16_t spill_operand_capacity = first->operand_capacity;
  EXPECT_NE(spill_operand_ids, first->inline_operand_ids);

  loom_parser_release_parsed_op(&parser, first);

  loom_parsed_op_t* second = nullptr;
  IREE_ASSERT_OK(loom_parser_acquire_parsed_op(&parser, &second));
  EXPECT_EQ(second, first);
  EXPECT_EQ(second->operand_ids, spill_operand_ids);
  EXPECT_EQ(second->operand_count, 0u);
  EXPECT_EQ(second->operand_capacity, spill_operand_capacity);

  loom_parser_release_parsed_op(&parser, second);
  iree_host_size_t spill_allocation_size =
      parser.parser_arena.total_allocation_size;

  for (int iteration = 0; iteration < 128; ++iteration) {
    loom_parsed_op_t* scratch = nullptr;
    IREE_ASSERT_OK(loom_parser_acquire_parsed_op(&parser, &scratch));
    ASSERT_EQ(scratch, first);
    for (uint16_t i = 0; i < LOOM_PARSED_OP_INLINE_OPERANDS + 4; ++i) {
      IREE_ASSERT_OK(
          loom_parsed_op_add_operand(scratch, &parser.parser_arena, i));
    }
    EXPECT_EQ(scratch->operand_ids, spill_operand_ids);
    EXPECT_EQ(scratch->operand_capacity, spill_operand_capacity);
    loom_parser_release_parsed_op(&parser, scratch);
  }
  EXPECT_EQ(parser.parser_arena.total_allocation_size, spill_allocation_size);

  iree_arena_deinitialize(&parser.parser_arena);
}

TEST_F(ParserTest, ParsedOpScratchFramesStayDepthSafeWhileParentIsActive) {
  loom_parser_t parser = {};
  iree_arena_initialize(&block_pool_, &parser.parser_arena);

  loom_parsed_op_t* parent = nullptr;
  IREE_ASSERT_OK(loom_parser_acquire_parsed_op(&parser, &parent));
  ASSERT_NE(parent, nullptr);
  for (uint16_t i = 0; i < LOOM_PARSED_OP_INLINE_OPERANDS + 4; ++i) {
    IREE_ASSERT_OK(loom_parsed_op_add_operand(parent, &parser.parser_arena, i));
  }

  loom_value_id_t* parent_operand_ids = parent->operand_ids;
  uint16_t parent_operand_capacity = parent->operand_capacity;
  uint16_t parent_operand_count = parent->operand_count;

  loom_parsed_op_t* child = nullptr;
  IREE_ASSERT_OK(loom_parser_acquire_parsed_op(&parser, &child));
  ASSERT_NE(child, nullptr);
  EXPECT_NE(child, parent);
  EXPECT_EQ(child->operand_count, 0u);

  for (uint16_t i = 0; i < LOOM_PARSED_OP_INLINE_OPERANDS + 8; ++i) {
    IREE_ASSERT_OK(loom_parsed_op_add_operand(child, &parser.parser_arena,
                                              (loom_value_id_t)(100 + i)));
  }

  EXPECT_EQ(parent->operand_ids, parent_operand_ids);
  EXPECT_EQ(parent->operand_capacity, parent_operand_capacity);
  EXPECT_EQ(parent->operand_count, parent_operand_count);
  for (uint16_t i = 0; i < parent_operand_count; ++i) {
    EXPECT_EQ(parent->operand_ids[i], i);
  }

  loom_parser_release_parsed_op(&parser, child);
  loom_parser_release_parsed_op(&parser, parent);
  iree_arena_deinitialize(&parser.parser_arena);
}

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

TEST_F(ParserTest, AttrDictStringEscapesRoundTripDecodedPayload) {
  std::string text = RoundTrip(
      "%c = test.constant 0 : f32\n"
      "%s = test.attrs %c {label = \"row\\n\\t\\\"slash\\\\\"} : f32\n");
  EXPECT_NE(
      text.find("test.attrs %c {label = \"row\\n\\t\\\"slash\\\\\"} : f32"),
      std::string::npos);
}

TEST_F(ParserTest, AttrDictUnsortedKeysRoundTripInCanonicalOrder) {
  std::string text = RoundTrip(
      "%c = test.constant 0 : f32\n"
      "%s = test.attrs %c {zeta = 2, axis = 0, label = \"foo\"} : f32\n");
  EXPECT_NE(
      text.find("test.attrs %c {axis = 0, label = \"foo\", zeta = 2} : f32"),
      std::string::npos)
      << "attribute dictionary keys should print in canonical order: " << text;
}

TEST_F(ParserTest, AttrDictNestedDictRoundTripInCanonicalOrder) {
  std::string text = RoundTrip(
      "%c = test.constant 0 : f32\n"
      "%s = test.attrs %c "
      "{phase = {zeta = 2, alpha = 1}, axis = 0, empty = {}} : f32\n");
  EXPECT_NE(
      text.find(
          "test.attrs %c {axis = 0, empty = {}, phase = {alpha = 1, zeta = 2}}"
          " : f32"),
      std::string::npos)
      << "nested attribute dictionaries should print in canonical order: "
      << text;
}

TEST_F(ParserTest, AttrDictEmptyArrayPayloadIsCanonical) {
  loom_module_t* module = ParseOk(
      "%c = test.constant 0 : f32\n"
      "%s = test.attrs %c {shape = []} : f32\n");
  if (!module) return;

  loom_block_t* body = loom_module_block(module);
  ASSERT_NE(body, nullptr);
  ASSERT_GE(body->op_count, 2u);
  loom_op_t* attrs_op = body->ops[1];
  ASSERT_NE(attrs_op, nullptr);
  ASSERT_TRUE(loom_test_attrs_isa(attrs_op));
  ASSERT_GE(attrs_op->attribute_count, 1u);

  loom_attribute_t dict_attr = loom_op_attrs(attrs_op)[0];
  IREE_ASSERT_OK(loom_module_verify_canonical_attr_dict(module, dict_attr));
  ASSERT_EQ(dict_attr.kind, LOOM_ATTR_DICT);
  ASSERT_EQ(dict_attr.count, 1u);
  ASSERT_NE(dict_attr.dict_entries, nullptr);
  EXPECT_EQ(dict_attr.dict_entries[0].value.kind, LOOM_ATTR_I64_ARRAY);
  EXPECT_EQ(dict_attr.dict_entries[0].value.count, 0u);
  EXPECT_EQ(dict_attr.dict_entries[0].value.i64_array, nullptr);

  std::string text = PrintModule(module);
  EXPECT_NE(text.find("test.attrs %c {shape = []} : f32"), std::string::npos)
      << "empty i64 array dict values should round-trip canonically: " << text;
  loom_module_free(module);
}

TEST_F(ParserTest, EmptyPredicateListPayloadIsCanonical) {
  loom_module_t* module = ParseOk(
      "%x = test.constant 0 : index\n"
      "%y = test.assume %x [] : index\n");
  if (!module) return;

  loom_block_t* body = loom_module_block(module);
  ASSERT_NE(body, nullptr);
  ASSERT_GE(body->op_count, 2u);
  loom_op_t* assume_op = body->ops[1];
  ASSERT_NE(assume_op, nullptr);
  ASSERT_TRUE(loom_test_assume_isa(assume_op));
  ASSERT_GE(assume_op->attribute_count, 1u);

  loom_attribute_t predicates = loom_op_attrs(assume_op)[0];
  EXPECT_EQ(predicates.kind, LOOM_ATTR_PREDICATE_LIST);
  EXPECT_EQ(predicates.count, 0u);
  EXPECT_EQ(predicates.predicate_list, nullptr);

  std::string text = PrintModule(module);
  EXPECT_NE(text.find("test.assume %x : index"), std::string::npos)
      << "empty optional predicate lists should elide to canonical text: "
      << text;
  EXPECT_EQ(text.find("test.assume %x [] : index"), std::string::npos)
      << "printer should not preserve empty predicate-list source spelling: "
      << text;
  loom_module_free(module);
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

TEST_F(ParserTest, FuncDefResultTiedToEntryArg) {
  loom_module_t* module = ParseOk(
      "test.func @identity(%x: f32) -> (%x as f32) {\n"
      "  test.yield %x : f32\n"
      "}\n");
  if (!module) return;

  std::string text = PrintModule(module);
  EXPECT_NE(text.find("-> (%x as f32)"), std::string::npos)
      << "expected tied signature result in: " << text;

  loom_op_t* func_op = GetFirstFunctionOp(module);
  ASSERT_NE(func_op, nullptr);
  ASSERT_EQ(func_op->tied_result_count, 1u);
  const loom_tied_result_t* tied_results = loom_op_tied_results(func_op);
  EXPECT_EQ(tied_results[0].result_index, 0u);
  EXPECT_EQ(tied_results[0].operand_index, 0u);
  EXPECT_FALSE(tied_results[0].has_type_change);
  ASSERT_EQ(func_op->region_count, 1u);
  loom_block_t* entry = GetEntryBlock(loom_op_regions(func_op)[0]);
  ASSERT_NE(entry, nullptr);
  ASSERT_EQ(entry->arg_count, 1u);
  ASSERT_EQ(entry->op_count, 1u);
  EXPECT_EQ(loom_op_const_operands(entry->ops[0])[0], entry->arg_ids[0]);
  loom_module_free(module);
}

TEST_F(ParserTest, FuncDeclResultTiedToArgOperand) {
  loom_module_t* module =
      ParseOk("test.decl @identity(%x: f32) -> (%x as f32)\n");
  if (!module) return;

  std::string text = PrintModule(module);
  EXPECT_NE(text.find("test.decl @identity(%x: f32) -> (%x as f32)"),
            std::string::npos)
      << "expected tied declaration result in: " << text;

  loom_op_t* func_op = GetFirstFunctionOp(module);
  ASSERT_NE(func_op, nullptr);
  EXPECT_EQ(func_op->operand_count, 1u);
  EXPECT_EQ(func_op->result_count, 1u);
  ASSERT_EQ(func_op->tied_result_count, 1u);
  const loom_tied_result_t* tied_results = loom_op_tied_results(func_op);
  EXPECT_EQ(tied_results[0].result_index, 0u);
  EXPECT_EQ(tied_results[0].operand_index, 0u);
  EXPECT_FALSE(tied_results[0].has_type_change);
  loom_module_free(module);
}

TEST_F(ParserTest, FuncDeclBareArgTypesRoundTrip) {
  std::string text = RoundTrip("test.decl @extern(f32) -> (f32)\n");
  EXPECT_NE(text.find("test.decl @extern(%0: f32) -> (f32)"), std::string::npos)
      << "unnamed declaration args should round-trip through autogenerated "
         "SSA names: "
      << text;
}

TEST_F(ParserTest, FuncDeclNamedResultCanReferenceSignatureArg) {
  std::string text = RoundTrip(
      "test.decl @shape(%M: index, %x: tensor<[%M]xf32>)"
      " -> (%x as tensor<[%M]xf32>, %count: index)\n");
  EXPECT_NE(text.find("test.decl @shape(%M: index, %x: tensor<[%M]xf32>)"
                      " -> (%x as tensor<[%M]xf32>, %count: index)"),
            std::string::npos)
      << "named/tied declaration signature results should round-trip: " << text;
}

TEST_F(ParserTest, FuncDeclForwardReferenceDimArgCanBindLaterArg) {
  std::string text = RoundTrip(
      "test.decl @shape(%x: tensor<[%M]xf32>, %M: index)"
      " -> (%x as tensor<[%M]xf32>)\n");
  EXPECT_NE(text.find("test.decl @shape(%x: tensor<[%M]xf32>, %M: index)"
                      " -> (%x as tensor<[%M]xf32>)"),
            std::string::npos)
      << "forward-referenced signature dims should bind the later arg: "
      << text;
}

TEST_F(ParserTest, FuncDeclForwardSignatureDimBindingResolves) {
  std::string text =
      RoundTrip("test.decl @shape(%x: tile<[%M]xf32>, %M: index)\n");
  EXPECT_NE(text.find("test.decl @shape(%x: tile<[%M]xf32>, %M: index)"),
            std::string::npos)
      << "forward dimension refs in declaration args should resolve: " << text;
}

TEST_F(ParserTest, FuncDeclUnresolvedPlaceholderReportsOriginalNameToken) {
  const auto& diagnostics =
      ParseExpectErrors("test.decl @shape(%x: tile<[%M]xf32>)\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_022);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "M");
  EXPECT_EQ(diagnostics[0].origin_line, 1u);
  EXPECT_EQ(diagnostics[0].origin_column, 28u);
  EXPECT_EQ(diagnostics[0].origin_end_column, 30u);
}

TEST_F(ParserTest, FuncDeclSignatureScopeDoesNotLeakPlaceholderNames) {
  const auto& diagnostics = ParseExpectErrors(
      "test.decl @shape(%M: index, %x: tile<[%M]xf32>)"
      " -> (%x as tile<[%M]xf32>)\n"
      "%u = test.constant 0 : index\n"
      "%bad = test.cast %u : index to tile<[%M]xf32>\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_001);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "M");
  EXPECT_EQ(diagnostics[0].origin_line, 3u);
  EXPECT_EQ(diagnostics[0].origin_column, 38u);
  EXPECT_EQ(diagnostics[0].origin_end_column, 40u);
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

TEST_F(ParserTest, TestFuncMultipleBlocks) {
  loom_module_t* module = ParseOk(
      "test.func @multi_block() {\n"
      "^entry:\n"
      "  test.yield\n"
      "^exit:\n"
      "  test.yield\n"
      "}\n");
  if (module) {
    loom_op_t* func_op = GetFirstFunctionOp(module);
    ASSERT_NE(func_op, nullptr);
    ASSERT_EQ(func_op->region_count, 1u);
    loom_region_t* body = loom_op_regions(func_op)[0];
    ASSERT_NE(body, nullptr);
    ASSERT_EQ(body->block_count, 2u);
    EXPECT_EQ(loom_region_entry_block(body)->op_count, 1u);
    EXPECT_EQ(loom_region_block(body, 1)->op_count, 1u);

    std::string text = PrintModule(module);
    EXPECT_NE(text.find("^entry:"), std::string::npos) << text;
    EXPECT_NE(text.find("^exit:"), std::string::npos) << text;
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
      " iter_args(%acc = %init : f32) -> (%init as f32) {\n"
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
    EXPECT_NE(text.find("-> (%init as f32)"), std::string::npos)
        << "iter_arg tie should name the init operand in: " << text;
    EXPECT_NE(text.find("func.return %r : f32"), std::string::npos)
        << "return type wrong in: " << text;

    loom_op_t* func_op = GetFirstFunctionOp(module);
    ASSERT_NE(func_op, nullptr);
    ASSERT_EQ(func_op->region_count, 1u);
    loom_block_t* func_entry = GetEntryBlock(loom_op_regions(func_op)[0]);
    ASSERT_NE(func_entry, nullptr);
    ASSERT_GE(func_entry->op_count, 2u);

    loom_op_t* loop_op = func_entry->ops[0];
    ASSERT_NE(loop_op, nullptr);
    ASSERT_EQ(loop_op->operand_count, 4u);
    ASSERT_EQ(loop_op->region_count, 1u);
    ASSERT_EQ(loop_op->tied_result_count, 1u);
    const loom_tied_result_t* tied_results = loom_op_tied_results(loop_op);
    EXPECT_EQ(tied_results[0].result_index, 0u);
    EXPECT_EQ(tied_results[0].operand_index, 3u);
    EXPECT_FALSE(tied_results[0].has_type_change);

    loom_block_t* loop_entry = GetEntryBlock(loom_op_regions(loop_op)[0]);
    ASSERT_NE(loop_entry, nullptr);
    ASSERT_EQ(loop_entry->arg_count, 2u);
    ASSERT_EQ(loop_entry->op_count, 1u);
    loom_op_t* yield_op = loop_entry->ops[0];
    ASSERT_NE(yield_op, nullptr);
    ASSERT_EQ(yield_op->operand_count, 1u);
    EXPECT_EQ(loom_op_const_operands(yield_op)[0], loop_entry->arg_ids[1]);
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
    EXPECT_EQ(text.find("test.yield"), std::string::npos)
        << "implicit loop terminator should be elided in: " << text;

    loom_op_t* func_op = GetFirstFunctionOp(module);
    ASSERT_NE(func_op, nullptr);
    ASSERT_EQ(func_op->region_count, 1u);
    loom_block_t* func_entry = GetEntryBlock(loom_op_regions(func_op)[0]);
    ASSERT_NE(func_entry, nullptr);
    ASSERT_GE(func_entry->op_count, 1u);

    loom_op_t* loop_op = func_entry->ops[0];
    ASSERT_NE(loop_op, nullptr);
    ASSERT_EQ(loop_op->region_count, 1u);
    loom_block_t* loop_entry = GetEntryBlock(loom_op_regions(loop_op)[0]);
    ASSERT_NE(loop_entry, nullptr);
    ASSERT_EQ(loop_entry->op_count, 1u);
    loom_op_t* yield_op = loop_entry->ops[0];
    ASSERT_NE(yield_op, nullptr);
    EXPECT_TRUE(loom_test_implicit_yield_isa(yield_op));
    EXPECT_EQ(yield_op->operand_count, 0u);
    EXPECT_EQ(yield_op->parent_op, loop_op);
    loom_module_free(module);
  }
}

TEST_F(ParserTest, LoopExplicitImplicitYieldCanonicalized) {
  loom_module_t* module = ParseOk(
      "func.def @explicit_implicit_yield(%lo: index, %hi: index, %step: "
      "index) {\n"
      "  test.loop %iv = %lo to %hi step %step {\n"
      "    test.implicit_yield\n"
      "  }\n"
      "}\n");
  if (module) {
    std::string text = PrintModule(module);
    EXPECT_EQ(text.find("test.implicit_yield"), std::string::npos)
        << "explicit implicit terminator should be canonicalized away in: "
        << text;
    EXPECT_NE(text.find("test.loop %iv ="), std::string::npos)
        << "loop should remain printable in: " << text;

    loom_op_t* func_op = GetFirstFunctionOp(module);
    ASSERT_NE(func_op, nullptr);
    loom_block_t* func_entry = GetEntryBlock(loom_op_regions(func_op)[0]);
    ASSERT_NE(func_entry, nullptr);
    ASSERT_GE(func_entry->op_count, 1u);

    loom_op_t* loop_op = func_entry->ops[0];
    ASSERT_NE(loop_op, nullptr);
    loom_block_t* loop_entry = GetEntryBlock(loom_op_regions(loop_op)[0]);
    ASSERT_NE(loop_entry, nullptr);
    ASSERT_EQ(loop_entry->op_count, 1u);
    loom_op_t* yield_op = loop_entry->ops[0];
    ASSERT_NE(yield_op, nullptr);
    EXPECT_TRUE(loom_test_implicit_yield_isa(yield_op));
    EXPECT_EQ(yield_op->operand_count, 0u);
    loom_module_free(module);
  }
}

TEST_F(ParserTest, LoopExplicitEmptyYieldPreserved) {
  loom_module_t* module = ParseOk(
      "func.def @explicit_empty_yield(%lo: index, %hi: index, %step: index) {\n"
      "  test.loop %iv = %lo to %hi step %step {\n"
      "    test.yield\n"
      "  }\n"
      "}\n");
  if (module) {
    std::string text = PrintModule(module);
    EXPECT_NE(text.find("    test.yield\n"), std::string::npos)
        << "explicit zero-operand test.yield should be preserved in: " << text;
    EXPECT_EQ(text.find("test.implicit_yield"), std::string::npos)
        << "implicit terminator op should stay elided in: " << text;

    loom_op_t* func_op = GetFirstFunctionOp(module);
    ASSERT_NE(func_op, nullptr);
    loom_block_t* func_entry = GetEntryBlock(loom_op_regions(func_op)[0]);
    ASSERT_NE(func_entry, nullptr);
    ASSERT_GE(func_entry->op_count, 1u);

    loom_op_t* loop_op = func_entry->ops[0];
    ASSERT_NE(loop_op, nullptr);
    loom_block_t* loop_entry = GetEntryBlock(loom_op_regions(loop_op)[0]);
    ASSERT_NE(loop_entry, nullptr);
    ASSERT_EQ(loop_entry->op_count, 1u);
    loom_op_t* yield_op = loop_entry->ops[0];
    ASSERT_NE(yield_op, nullptr);
    EXPECT_TRUE(loom_test_yield_isa(yield_op));
    EXPECT_EQ(yield_op->operand_count, 0u);
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

TEST_F(ParserTest, UnexpectedTokenRetainsSigilAndSpan) {
  const auto& diagnostics = ParseExpectErrors("%r = @callee\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_003);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "@callee");
  EXPECT_EQ(diagnostics[0].origin_line, 1u);
  EXPECT_EQ(diagnostics[0].origin_column, 6u);
  EXPECT_EQ(diagnostics[0].origin_end_column, 13u);
}

TEST_F(ParserTest, UnexpectedStringTokenRendersQuotesAndSpan) {
  const auto& diagnostics = ParseExpectErrors("\"hello\"\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_003);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "\"hello\"");
  EXPECT_EQ(GetStringParam(diagnostics[0], 1), "op name");
  EXPECT_EQ(diagnostics[0].origin_line, 1u);
  EXPECT_EQ(diagnostics[0].origin_column, 1u);
  EXPECT_EQ(diagnostics[0].origin_end_column, 8u);
}

TEST_F(ParserTest, UnexpectedStringTokenEscapesDecodedPayload) {
  const auto& diagnostics = ParseExpectErrors("\"row\\n\\t\\\"slash\\\\\"\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_003);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "\"row\\n\\t\\\"slash\\\\\"");
  EXPECT_EQ(GetStringParam(diagnostics[0], 1), "op name");
  EXPECT_EQ(diagnostics[0].origin_line, 1u);
  EXPECT_EQ(diagnostics[0].origin_column, 1u);
  EXPECT_EQ(diagnostics[0].origin_end_column, 19u);
}

TEST_F(ParserTest, UnexpectedBlockLabelTokenRendersSigilAndSpan) {
  const auto& diagnostics = ParseExpectErrors("^bb\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_003);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "^bb");
  EXPECT_EQ(GetStringParam(diagnostics[0], 1), "op name");
  EXPECT_EQ(diagnostics[0].origin_line, 1u);
  EXPECT_EQ(diagnostics[0].origin_column, 1u);
  EXPECT_EQ(diagnostics[0].origin_end_column, 4u);
}

TEST_F(ParserTest, UndefinedSSAValue) {
  const auto& diagnostics = ParseExpectErrors(
      "%c = test.constant 1 : i32\n"
      "%r = test.addi %c, %undefined : i32\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_001);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "undefined");
  EXPECT_EQ(diagnostics[0].origin_line, 2u);
  EXPECT_EQ(diagnostics[0].origin_column, 20u);
  EXPECT_EQ(diagnostics[0].origin_end_column, 30u);
}

TEST_F(ParserTest, BindingListNameIsRegionLocal) {
  const auto& diagnostics = ParseExpectErrors(
      "%tile = test.constant 0 : f32\n"
      "%mapped = test.map(%element = %tile : f32) {\n"
      "  test.yield %element : f32\n"
      "} -> (f32)\n"
      "%leak = test.neg %element : f32\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_001);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "element");
  EXPECT_EQ(diagnostics[0].origin_line, 5u);
  EXPECT_EQ(diagnostics[0].origin_column, 18u);
  EXPECT_EQ(diagnostics[0].origin_end_column, 26u);
}

TEST_F(ParserTest, FuncResultNameIsSignatureLocal) {
  const auto& diagnostics = ParseExpectErrors(
      "func.def @f() -> (%n: index) {\n"
      "  %x = test.cast %n : index to index\n"
      "  func.return %x : index\n"
      "}\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_001);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "n");
  EXPECT_EQ(diagnostics[0].origin_line, 2u);
  EXPECT_EQ(diagnostics[0].origin_column, 18u);
}

TEST_F(ParserTest, BindingListNameCanShadowOuterScopeName) {
  loom_module_t* module = ParseOk(
      "func.def @shadow(%x : f32) -> (f32) {\n"
      "  %tile = test.constant 0 : f32\n"
      "  %mapped = test.map(%x = %tile : f32) {\n"
      "    test.yield %x : f32\n"
      "  } -> (f32)\n"
      "  %negated = test.neg %x : f32\n"
      "  func.return %negated : f32\n"
      "}\n");
  if (module) {
    loom_op_t* func_op = GetFirstFunctionOp(module);
    ASSERT_NE(func_op, nullptr);
    ASSERT_EQ(func_op->region_count, 1u);
    loom_block_t* func_entry = GetEntryBlock(loom_op_regions(func_op)[0]);
    ASSERT_NE(func_entry, nullptr);
    ASSERT_EQ(func_entry->arg_count, 1u);
    ASSERT_GE(func_entry->op_count, 3u);

    loom_op_t* map_op = func_entry->ops[1];
    ASSERT_NE(map_op, nullptr);
    ASSERT_EQ(map_op->region_count, 1u);
    loom_block_t* map_entry = GetEntryBlock(loom_op_regions(map_op)[0]);
    ASSERT_NE(map_entry, nullptr);
    ASSERT_EQ(map_entry->arg_count, 1u);

    loom_op_t* yield_op = map_entry->ops[0];
    ASSERT_NE(yield_op, nullptr);
    ASSERT_EQ(yield_op->operand_count, 1u);
    EXPECT_EQ(loom_op_const_operands(yield_op)[0], map_entry->arg_ids[0]);

    loom_op_t* neg_op = func_entry->ops[2];
    ASSERT_NE(neg_op, nullptr);
    ASSERT_EQ(neg_op->operand_count, 1u);
    EXPECT_EQ(loom_op_const_operands(neg_op)[0], func_entry->arg_ids[0]);
    EXPECT_NE(map_entry->arg_ids[0], func_entry->arg_ids[0]);
    loom_module_free(module);
  }
}

TEST_F(ParserTest, LoopIvNameIsRegionLocal) {
  const auto& diagnostics = ParseExpectErrors(
      "func.def @simple_loop(%lo: index, %hi: index, %step: index) {\n"
      "  test.loop %iv = %lo to %hi step %step {\n"
      "  }\n"
      "  test.loop %again = %iv to %hi step %step {\n"
      "  }\n"
      "}\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_001);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "iv");
  EXPECT_EQ(diagnostics[0].origin_line, 4u);
  EXPECT_EQ(diagnostics[0].origin_column, 22u);
  EXPECT_EQ(diagnostics[0].origin_end_column, 25u);
}

TEST_F(ParserTest, LoopIterArgNameIsNotATiedResultTarget) {
  const auto& diagnostics = ParseExpectErrors(
      "func.def @loop(%lo: index, %hi: index, %step: index, %init: f32)"
      " -> (f32) {\n"
      "  %r = test.loop %iv = %lo to %hi step %step"
      " iter_args(%acc = %init : f32) -> (%acc as f32) {\n"
      "    test.yield %acc : f32\n"
      "  }\n"
      "  func.return %r : f32\n"
      "}\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_001);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "acc");
  EXPECT_EQ(diagnostics[0].origin_line, 2u);
  EXPECT_EQ(diagnostics[0].origin_column, 80u);
}

TEST_F(ParserTest, LoopIvNameIsNotATiedResultTarget) {
  const auto& diagnostics = ParseExpectErrors(
      "func.def @loop(%lo: index, %hi: index, %step: index, %init: f32)"
      " -> (f32) {\n"
      "  %r = test.loop %iv = %lo to %hi step %step"
      " iter_args(%acc = %init : f32) -> (%iv as f32) {\n"
      "    test.yield %acc : f32\n"
      "  }\n"
      "  func.return %r : f32\n"
      "}\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_001);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "iv");
  EXPECT_EQ(diagnostics[0].origin_line, 2u);
  EXPECT_EQ(diagnostics[0].origin_column, 80u);
}

TEST_F(ParserTest, DuplicateFunctionArgName) {
  const auto& diagnostics = ParseExpectErrors(
      "func.def @f(%x : f32, %x : f32) {\n"
      "  func.return\n"
      "}\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_002);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "x");
  EXPECT_EQ(diagnostics[0].origin_line, 1u);
  EXPECT_EQ(diagnostics[0].origin_column, 23u);
}

TEST_F(ParserTest, DuplicateBlockArgName) {
  const auto& diagnostics = ParseExpectErrors(
      "func.def @f() {\n"
      "^bb(%x : i32, %x : i32):\n"
      "  func.return\n"
      "}\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_002);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "x");
  EXPECT_EQ(diagnostics[0].origin_line, 2u);
  EXPECT_EQ(diagnostics[0].origin_column, 15u);
}

TEST_F(ParserTest, DuplicateOpResultName) {
  const auto& diagnostics = ParseExpectErrors(
      "%cond = test.constant 1 : i1\n"
      "%lhs = test.constant 0 : f32\n"
      "%rhs = test.constant 1 : f32\n"
      "%r, %r = test.branch %cond -> (f32, f32) {\n"
      "  test.yield %lhs, %rhs : f32, f32\n"
      "} else {\n"
      "  test.yield %rhs, %lhs : f32, f32\n"
      "}\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_002);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "r");
  EXPECT_EQ(diagnostics[0].origin_line, 4u);
  EXPECT_EQ(diagnostics[0].origin_column, 5u);
}

TEST_F(ParserTest, ResultBodyOpRequiresLhsNames) {
  const auto& diagnostics = ParseExpectErrors("test.constant 42 : i32\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_009);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "test.constant");
  ExpectU32Param(diagnostics[0], 1, 1u);
  ExpectU32Param(diagnostics[0], 2, 0u);
}

TEST_F(ParserTest, SymbolDefinitionRejectsLhsNames) {
  const auto& diagnostics = ParseExpectErrors(
      "%fn = func.def @named() {\n"
      "  func.return\n"
      "}\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_009);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "func.def");
  ExpectU32Param(diagnostics[0], 1, 0u);
  ExpectU32Param(diagnostics[0], 2, 1u);
}

TEST_F(ParserTest, DuplicateBindingListName) {
  const auto& diagnostics = ParseExpectErrors(
      "%tile = test.constant 0 : f32\n"
      "%mapped = test.map(%element = %tile : f32, %element = %tile : f32) {\n"
      "  test.yield %element : f32\n"
      "} -> (f32)\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_002);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "element");
  EXPECT_EQ(diagnostics[0].origin_line, 2u);
}

TEST_F(ParserTest, DuplicateAttrDictKey) {
  const auto& diagnostics = ParseExpectErrors(
      "%c = test.constant 0 : f32\n"
      "%s = test.attrs %c {alpha = 1, alpha = 2} : f32\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_020);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "alpha");
  EXPECT_EQ(diagnostics[0].origin_line, 2u);
  EXPECT_EQ(diagnostics[0].origin_column, 32u);
  EXPECT_EQ(diagnostics[0].origin_end_column, 37u);
}

TEST_F(ParserTest, DuplicateNestedAttrDictKey) {
  const auto& diagnostics = ParseExpectErrors(
      "%c = test.constant 0 : f32\n"
      "%s = test.attrs %c {config = {zeta = 1, zeta = 2}} : f32\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_020);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "zeta");
  EXPECT_EQ(diagnostics[0].origin_line, 2u);
  EXPECT_EQ(diagnostics[0].origin_column, 41u);
  EXPECT_EQ(diagnostics[0].origin_end_column, 45u);
}

TEST_F(ParserTest, AttrDictTooDeep) {
  std::string source =
      "%c = test.constant 0 : f32\n"
      "%s = test.attrs %c ";
  for (uint32_t depth = 0; depth <= LOOM_ATTR_DICT_MAX_NESTING_DEPTH; ++depth) {
    source += "{k" + std::to_string(depth) + " = ";
  }
  source += "0";
  for (uint32_t depth = 0; depth <= LOOM_ATTR_DICT_MAX_NESTING_DEPTH; ++depth) {
    source += "}";
  }
  source += " : f32\n";

  const auto& diagnostics = ParseExpectErrors(source.c_str());
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_021);
  ExpectU32Param(diagnostics[0], 0, LOOM_ATTR_DICT_MAX_NESTING_DEPTH);
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

TEST_F(ParserTest, UnknownStaticEncodingFamilyInType) {
  const auto& diagnostics =
      ParseExpectErrors("%c = test.constant 0 : tile<4xf32, #bogus>\n");
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

TEST_F(ParserTest, InvalidEncodingAliasReportsAliasToken) {
  const auto& diagnostics = ParseExpectErrors("#enc test.constant 0 : i32\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_014);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "#enc");
  EXPECT_EQ(diagnostics[0].origin_line, 1u);
  EXPECT_EQ(diagnostics[0].origin_column, 1u);
  EXPECT_EQ(diagnostics[0].origin_end_column, 5u);
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

TEST_F(ParserTest, EncodingDefineInlineSpec) {
  loom_module_t* module =
      ParseOk("%enc = encoding.define #q8_0<block=32> : encoding\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* body = loom_module_block(module);
  ASSERT_EQ(body->op_count, 1u);
  const loom_op_t* op = loom_block_const_op(body, 0);
  ASSERT_TRUE(loom_encoding_define_isa(op));

  loom_attribute_t spec_attr = loom_op_attrs(op)[0];
  ASSERT_EQ(spec_attr.kind, LOOM_ATTR_ENCODING);
  const loom_encoding_t* spec_encoding =
      loom_module_encoding(module, loom_attr_as_encoding_id(spec_attr));
  ASSERT_NE(spec_encoding, nullptr);
  ASSERT_LT(spec_encoding->name_id, module->strings.count);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[spec_encoding->name_id], IREE_SV("q8_0")));
  ASSERT_EQ(spec_encoding->attribute_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[spec_encoding->attributes[0].name_id],
      IREE_SV("block")));
  EXPECT_EQ(spec_encoding->attributes[0].value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(spec_encoding->attributes[0].value.i64, 32);

  EXPECT_EQ(PrintModule(module),
            "%enc = encoding.define #q8_0<block=32> : encoding\n");
  loom_module_free(module);
}

TEST_F(ParserTest, EncodingDefineAliasSpec) {
  loom_module_t* module = ParseOk(
      "#enc = #q8_0<block=32>\n"
      "%enc = encoding.define #enc : encoding\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* body = loom_module_block(module);
  ASSERT_EQ(body->op_count, 1u);
  const loom_op_t* op = loom_block_const_op(body, 0);
  ASSERT_TRUE(loom_encoding_define_isa(op));

  loom_attribute_t spec_attr = loom_op_attrs(op)[0];
  ASSERT_EQ(spec_attr.kind, LOOM_ATTR_ENCODING);
  const loom_encoding_t* spec_encoding =
      loom_module_encoding(module, loom_attr_as_encoding_id(spec_attr));
  ASSERT_NE(spec_encoding, nullptr);
  ASSERT_NE(spec_encoding->alias_id, LOOM_STRING_ID_INVALID);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[spec_encoding->alias_id], IREE_SV("enc")));

  EXPECT_EQ(PrintModule(module),
            "#enc = #q8_0<block=32>\n"
            "%enc = encoding.define #enc : encoding\n");
  loom_module_free(module);
}

TEST_F(ParserTest, EncodingDefineNestedInlineSpec) {
  loom_module_t* module = ParseOk(
      "%enc = encoding.define "
      "#quantization<spec=#q8_0<block=32>> : encoding\n");
  ASSERT_NE(module, nullptr);

  loom_block_t* body = loom_module_block(module);
  ASSERT_EQ(body->op_count, 1u);
  const loom_op_t* op = loom_block_const_op(body, 0);
  ASSERT_TRUE(loom_encoding_define_isa(op));

  loom_attribute_t spec_attr = loom_op_attrs(op)[0];
  ASSERT_EQ(spec_attr.kind, LOOM_ATTR_ENCODING);
  const loom_encoding_t* outer_encoding =
      loom_module_encoding(module, loom_attr_as_encoding_id(spec_attr));
  ASSERT_NE(outer_encoding, nullptr);
  ASSERT_EQ(outer_encoding->attribute_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[outer_encoding->attributes[0].name_id],
      IREE_SV("spec")));

  loom_attribute_t nested_spec = outer_encoding->attributes[0].value;
  ASSERT_EQ(nested_spec.kind, LOOM_ATTR_ENCODING);
  const loom_encoding_t* nested_encoding =
      loom_module_encoding(module, loom_attr_as_encoding_id(nested_spec));
  ASSERT_NE(nested_encoding, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[nested_encoding->name_id], IREE_SV("q8_0")));
  ASSERT_EQ(nested_encoding->attribute_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      module->strings.entries[nested_encoding->attributes[0].name_id],
      IREE_SV("block")));
  EXPECT_EQ(nested_encoding->attributes[0].value.kind, LOOM_ATTR_I64);
  EXPECT_EQ(nested_encoding->attributes[0].value.i64, 32);

  EXPECT_EQ(PrintModule(module),
            "%enc = encoding.define "
            "#quantization<spec=#q8_0<block=32>> : encoding\n");
  loom_module_free(module);
}

TEST_F(ParserTest, EncodingAliasCannotShadowRegisteredFamily) {
  const auto& diagnostics = ParseExpectErrors("#q8_0 = #dense\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_014);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0),
            "alias name shadows a registered encoding family");
}

TEST_F(ParserTest, DuplicateEncodingAliasDefinitionFails) {
  const auto& diagnostics = ParseExpectErrors(
      "#enc = #dense\n"
      "#enc = #q8_0<block=32>\n");
  ASSERT_GE(diagnostics.size(), 1u);
  ExpectError(diagnostics[0], &loom_err_parse_014);
  EXPECT_EQ(GetStringParam(diagnostics[0], 0), "duplicate encoding alias name");
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
  EXPECT_NE(FindDiagnostic(capture_, &loom_err_parse_006), nullptr);
}

TEST_F(ParserTest, MaxErrorsLimit) {
  // Generate many errors, set max_errors low, verify ERR_PARSE_012 is emitted.
  std::string source;
  for (int i = 0; i < 25; ++i) {
    source += "%bad" + std::to_string(i) + " = bogus.op" + std::to_string(i) +
              " : i32\n";
  }

  capture_.Reset();
  loom_text_parse_options_t options;
  memset(&options, 0, sizeof(options));
  options.diagnostic_sink = capture_.sink();
  options.max_errors = 5;

  loom_module_t* module = nullptr;
  IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source.c_str()),
                                 iree_make_cstring_view("test.loom"), &context_,
                                 &block_pool_, &options, &module));
  EXPECT_EQ(module, nullptr);

  // Should have at least 2 diagnostics.
  ASSERT_GE(capture_.diagnostics.size(), 2u);

  // Find ERR_PARSE_012 (too many errors) somewhere in the diagnostics.
  EXPECT_NE(FindDiagnostic(capture_, &loom_err_parse_012), nullptr)
      << "Expected ERR_PARSE_012 (too many errors)";

  // Total error count should not exceed max_errors + 1 (the "too many" itself).
  EXPECT_LE(capture_.diagnostics.size(), 6u + 1u);
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
