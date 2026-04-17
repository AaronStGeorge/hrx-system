// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/verify.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::ExpectError;
using ::loom::testing::ExpectFieldRefParam;
using ::loom::testing::FindDiagnostic;
using ::loom::testing::GetStringParam;

class LowVerifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_op_registry_initialize_context(iree_allocator_system(),
                                                       &context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_verify_result_t VerifySource(const char* source,
                                    DiagnosticCapture* verify_capture) {
    const char* filename = "low_verify_test.loom";
    DiagnosticCapture parse_capture;
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink = parse_capture.sink();
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   iree_make_cstring_view(filename), &context_,
                                   &block_pool_, &parse_options, &module));
    if (!parse_capture.diagnostics.empty()) {
      ADD_FAILURE() << "expected parser success, got "
                    << parse_capture.diagnostics.size() << " diagnostic(s)";
      if (module) loom_module_free(module);
      return {};
    }
    EXPECT_NE(module, nullptr);
    if (!module) return {};

    loom_source_entry_t source_entries[] = {{
        .source_id = FindContextSourceId(filename),
        .source = iree_make_cstring_view(source),
        .filename = iree_make_cstring_view(filename),
    }};
    EXPECT_NE(source_entries[0].source_id, LOOM_SOURCE_ID_INVALID);
    loom_source_table_resolver_t resolver_data = {
        .entries = source_entries,
        .count = IREE_ARRAYSIZE(source_entries),
    };

    verify_capture->Reset();
    loom_verify_options_t verify_options = {};
    verify_options.sink = verify_capture->sink();
    verify_options.max_errors = 20;
    verify_options.source_resolver = {loom_source_table_resolve,
                                      &resolver_data};
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module, &verify_options, &result));
    loom_module_free(module);
    return result;
  }

  loom_source_id_t FindContextSourceId(const char* filename) const {
    iree_string_view_t source_name = iree_make_cstring_view(filename);
    for (iree_host_size_t i = 0; i < context_.sources.count; ++i) {
      if (iree_string_view_equal(context_.sources.entries[i], source_name)) {
        return (loom_source_id_t)i;
      }
    }
    ADD_FAILURE() << "expected context source table to contain " << filename;
    return LOOM_SOURCE_ID_INVALID;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(LowVerifyTest, DescriptorKeysPassWithQualifiedSegments) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.def target(@gfx1100) @add(%lhs : reg<amdgpu.vgpr x1>, "
      "%rhs : reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.op<amdgpu.v_add_u32>(%lhs, %rhs) : "
      "(reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> "
      "reg<amdgpu.vgpr x1>\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, InvokeMatchesDirectLowFunctionSignature) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs : "
      "reg<amdgpu.vgpr x1>, %rhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "low.func.def target(@gfx1100) @caller(%lhs : reg<amdgpu.vgpr x1>, "
      "%rhs : reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.invoke @extern_add(%lhs, %rhs) : "
      "(reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, InvokeRejectsNonLowCallee) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "func.decl @semantic(%arg : i32) -> (i32)\n"
      "test.record @gfx1100 {}\n"
      "low.func.def target(@gfx1100) @caller(%lhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.invoke @semantic(%lhs) : (reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_invoke_callee_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "semantic");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "function");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "low function");
  ASSERT_EQ(diagnostic->related_locations.size(), 1u);
  EXPECT_EQ(diagnostic->related_locations[0].label, "defined here");
}

TEST_F(LowVerifyTest, InvokeRejectsOperandCountMismatch) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs : "
      "reg<amdgpu.vgpr x1>, %rhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "low.func.def target(@gfx1100) @caller(%lhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.invoke @extern_add(%lhs) : (reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 13);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "operand");
  ExpectU32Param(*diagnostic, 2, 1);
  ExpectU32Param(*diagnostic, 3, 2);
  ASSERT_EQ(diagnostic->related_locations.size(), 1u);
  EXPECT_EQ(diagnostic->related_locations[0].label, "defined here");
}

TEST_F(LowVerifyTest, InvokeRejectsOperandTypeMismatch) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs : "
      "reg<amdgpu.vgpr x1>, %rhs : reg<amdgpu.sgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "low.func.def target(@gfx1100) @caller(%lhs : reg<amdgpu.vgpr x1>, "
      "%rhs : reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.invoke @extern_add(%lhs, %rhs) : "
      "(reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 14);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "operand");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_OPERAND, 1);
  ExpectU32Param(*diagnostic, 2, 1);
  ASSERT_EQ(diagnostic->params[3].kind, LOOM_PARAM_TYPE);
  loom_type_t actual_type = diagnostic->params[3].type;
  EXPECT_TRUE(loom_type_is_register(actual_type));
  EXPECT_EQ(loom_type_register_unit_count(actual_type), 1u);
  EXPECT_EQ(GetStringParam(*diagnostic, 4), "argument");
  ASSERT_EQ(diagnostic->params[5].kind, LOOM_PARAM_TYPE);
  loom_type_t expected_type = diagnostic->params[5].type;
  EXPECT_TRUE(loom_type_is_register(expected_type));
  EXPECT_EQ(loom_type_register_unit_count(expected_type), 1u);
  EXPECT_NE(loom_type_register_class_id(actual_type),
            loom_type_register_class_id(expected_type));
  ASSERT_EQ(diagnostic->related_locations.size(), 1u);
  EXPECT_EQ(diagnostic->related_locations[0].label, "defined here");
}

TEST_F(LowVerifyTest, InvokeRejectsResultTypeMismatch) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs : "
      "reg<amdgpu.vgpr x1>) -> (reg<amdgpu.sgpr x1>)\n"
      "low.func.def target(@gfx1100) @caller(%lhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.invoke @extern_add(%lhs) : (reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 14);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "result");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_RESULT, 0);
  ExpectU32Param(*diagnostic, 2, 0);
  ASSERT_EQ(diagnostic->params[3].kind, LOOM_PARAM_TYPE);
  loom_type_t actual_type = diagnostic->params[3].type;
  EXPECT_TRUE(loom_type_is_register(actual_type));
  EXPECT_EQ(loom_type_register_unit_count(actual_type), 1u);
  EXPECT_EQ(GetStringParam(*diagnostic, 4), "result");
  ASSERT_EQ(diagnostic->params[5].kind, LOOM_PARAM_TYPE);
  loom_type_t expected_type = diagnostic->params[5].type;
  EXPECT_TRUE(loom_type_is_register(expected_type));
  EXPECT_EQ(loom_type_register_unit_count(expected_type), 1u);
  EXPECT_NE(loom_type_register_class_id(actual_type),
            loom_type_register_class_id(expected_type));
  ASSERT_EQ(diagnostic->related_locations.size(), 1u);
  EXPECT_EQ(diagnostic->related_locations[0].label, "defined here");
}

TEST_F(LowVerifyTest, DescriptorKeyRejectsEmptySegment) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.def target(@gfx1100) @const() -> (reg<amdgpu.sgpr x1>) {\n"
      "  %c0 = low.const<amdgpu.> {imm = 0} : reg<amdgpu.sgpr x1>\n"
      "  low.return %c0 : reg<amdgpu.sgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 27);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_const_opcode_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "opcode");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "amdgpu.");
}

TEST_F(LowVerifyTest, DescriptorKeyRejectsInvalidCharacter) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.def target(@gfx1100) @add(%lhs : reg<amdgpu.vgpr x1>, "
      "%rhs : reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.op<amdgpu.v$add_u32>(%lhs, %rhs) : "
      "(reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> "
      "reg<amdgpu.vgpr x1>\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 27);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_op_opcode_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "opcode");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "amdgpu.v$add_u32");
}

}  // namespace
}  // namespace loom
