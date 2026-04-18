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
using ::loom::testing::ExpectI64Param;
using ::loom::testing::ExpectU32Param;
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

TEST_F(LowVerifyTest, InvokeMatchesDirectAbiAdapterSignature) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs : "
      "reg<amdgpu.vgpr x1>, %rhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "low.abi.adapter @extern_add_direct {callee = @extern_add, conversion = "
      "direct, operand_count = 2, result_count = 1}\n"
      "func.def @caller(%lhs : reg<amdgpu.vgpr x1>, %rhs : "
      "reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.invoke @extern_add(%lhs, %rhs) {adapter = "
      "@extern_add_direct} : "
      "(reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "  func.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, InvokeMatchesMappedAbiAdapterSignature) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs : reg<vm.i32>, "
      "%rhs : reg<vm.i32>) -> (reg<vm.i32>)\n"
      "low.abi.adapter @extern_add_i32 {callee = @extern_add, conversion = "
      "mapped, operand_count = 2, result_count = 1}\n"
      "low.abi.operand @extern_add_i32_lhs {adapter = @extern_add_i32, index "
      "= 0, conversion = scalar_to_register, semantic_type = i32, abi_type = "
      "reg<vm.i32>}\n"
      "low.abi.operand @extern_add_i32_rhs {adapter = @extern_add_i32, index "
      "= 1, conversion = scalar_to_register, semantic_type = i32, abi_type = "
      "reg<vm.i32>}\n"
      "low.abi.result @extern_add_i32_result {adapter = @extern_add_i32, "
      "index = 0, conversion = register_to_scalar, semantic_type = i32, "
      "abi_type = reg<vm.i32>}\n"
      "func.def @caller(%lhs : i32, %rhs : i32) -> (i32) {\n"
      "  %sum = low.invoke @extern_add(%lhs, %rhs) {adapter = "
      "@extern_add_i32} : (i32, i32) -> (i32)\n"
      "  func.return %sum : i32\n"
      "}\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, MappedAbiAdapterRejectsMissingEntry) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs : reg<vm.i32>, "
      "%rhs : reg<vm.i32>) -> (reg<vm.i32>)\n"
      "low.abi.adapter @extern_add_i32 {callee = @extern_add, conversion = "
      "mapped, operand_count = 2, result_count = 1}\n"
      "low.abi.operand @extern_add_i32_lhs {adapter = @extern_add_i32, index "
      "= 0, conversion = scalar_to_register, semantic_type = i32, abi_type = "
      "reg<vm.i32>}\n"
      "low.abi.result @extern_add_i32_result {adapter = @extern_add_i32, "
      "index = 0, conversion = register_to_scalar, semantic_type = i32, "
      "abi_type = reg<vm.i32>}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 18);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add_i32");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "<missing>");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "operand");
  ExpectI64Param(*diagnostic, 3, 1);
  EXPECT_EQ(GetStringParam(*diagnostic, 4),
            "mapped adapter is missing this entry");
}

TEST_F(LowVerifyTest, MappedAbiAdapterRejectsDuplicateEntry) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>)\n"
      "low.abi.adapter @extern_add_i32 {callee = @extern_add, conversion = "
      "mapped, operand_count = 1, result_count = 1}\n"
      "low.abi.operand @extern_add_i32_lhs {adapter = @extern_add_i32, index "
      "= 0, conversion = scalar_to_register, semantic_type = i32, abi_type = "
      "reg<vm.i32>}\n"
      "low.abi.operand @extern_add_i32_lhs_copy {adapter = @extern_add_i32, "
      "index = 0, conversion = scalar_to_register, semantic_type = i32, "
      "abi_type = reg<vm.i32>}\n"
      "low.abi.result @extern_add_i32_result {adapter = @extern_add_i32, "
      "index = 0, conversion = register_to_scalar, semantic_type = i32, "
      "abi_type = reg<vm.i32>}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 18);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add_i32");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "extern_add_i32_lhs_copy");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "operand");
  ExpectI64Param(*diagnostic, 3, 0);
  EXPECT_EQ(GetStringParam(*diagnostic, 4),
            "mapped adapter has more than one entry for this index");
}

TEST_F(LowVerifyTest, MappedAbiAdapterRejectsWrongEntryDirection) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>)\n"
      "low.abi.adapter @extern_add_i32 {callee = @extern_add, conversion = "
      "mapped, operand_count = 1, result_count = 1}\n"
      "low.abi.operand @extern_add_i32_lhs {adapter = @extern_add_i32, index "
      "= 0, conversion = register_to_scalar, semantic_type = i32, abi_type = "
      "reg<vm.i32>}\n"
      "low.abi.result @extern_add_i32_result {adapter = @extern_add_i32, "
      "index = 0, conversion = register_to_scalar, semantic_type = i32, "
      "abi_type = reg<vm.i32>}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 18);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add_i32");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "extern_add_i32_lhs");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "operand");
  ExpectI64Param(*diagnostic, 3, 0);
  EXPECT_EQ(GetStringParam(*diagnostic, 4),
            "register_to_scalar is only valid for result entries");
}

TEST_F(LowVerifyTest, InvokeRejectsMappedAdapterSemanticTypeMismatch) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>)\n"
      "low.abi.adapter @extern_add_i32 {callee = @extern_add, conversion = "
      "mapped, operand_count = 1, result_count = 1}\n"
      "low.abi.operand @extern_add_i32_lhs {adapter = @extern_add_i32, index "
      "= 0, conversion = scalar_to_register, semantic_type = f32, abi_type = "
      "reg<vm.i32>}\n"
      "low.abi.result @extern_add_i32_result {adapter = @extern_add_i32, "
      "index = 0, conversion = register_to_scalar, semantic_type = i32, "
      "abi_type = reg<vm.i32>}\n"
      "func.def @caller(%lhs : i32) -> (i32) {\n"
      "  %sum = low.invoke @extern_add(%lhs) {adapter = @extern_add_i32} : "
      "(i32) -> (i32)\n"
      "  func.return %sum : i32\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 19);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add_i32");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "extern_add_i32_lhs");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "scalar_to_register");
  EXPECT_EQ(GetStringParam(*diagnostic, 3), "operand");
  ExpectU32Param(*diagnostic, 4, 0);
  ASSERT_EQ(diagnostic->params[5].kind, LOOM_PARAM_TYPE);
  EXPECT_EQ(GetStringParam(*diagnostic, 6), "adapter semantic");
  ASSERT_EQ(diagnostic->params[7].kind, LOOM_PARAM_TYPE);
}

TEST_F(LowVerifyTest, MappedAbiAdapterRejectsAbiTypeMismatch) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>)\n"
      "low.abi.adapter @extern_add_i32 {callee = @extern_add, conversion = "
      "mapped, operand_count = 1, result_count = 1}\n"
      "low.abi.operand @extern_add_i32_lhs {adapter = @extern_add_i32, index "
      "= 0, conversion = scalar_to_register, semantic_type = i32, abi_type = "
      "reg<vm.i64>}\n"
      "low.abi.result @extern_add_i32_result {adapter = @extern_add_i32, "
      "index = 0, conversion = register_to_scalar, semantic_type = i32, "
      "abi_type = reg<vm.i32>}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 19);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add_i32");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "extern_add_i32_lhs");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "scalar_to_register");
  EXPECT_EQ(GetStringParam(*diagnostic, 3), "operand");
  ExpectU32Param(*diagnostic, 4, 0);
  ASSERT_EQ(diagnostic->params[5].kind, LOOM_PARAM_TYPE);
  EXPECT_EQ(GetStringParam(*diagnostic, 6), "callee argument");
  ASSERT_EQ(diagnostic->params[7].kind, LOOM_PARAM_TYPE);
}

TEST_F(LowVerifyTest, DirectAbiAdapterRejectsMappingEntries) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>)\n"
      "low.abi.adapter @extern_add_direct {callee = @extern_add, conversion = "
      "direct, operand_count = 1, result_count = 1}\n"
      "low.abi.operand @extern_add_i32_lhs {adapter = @extern_add_direct, "
      "index = 0, conversion = scalar_to_register, semantic_type = i32, "
      "abi_type = reg<vm.i32>}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 18);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add_direct");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "extern_add_i32_lhs");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "operand");
  ExpectI64Param(*diagnostic, 3, 0);
  EXPECT_EQ(GetStringParam(*diagnostic, 4),
            "only mapped adapters accept operand/result mapping entries");
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

TEST_F(LowVerifyTest, AbiAdapterRejectsNonLowCallee) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "func.decl @semantic(%arg : i32) -> (i32)\n"
      "low.abi.adapter @bad {callee = @semantic, conversion = direct, "
      "operand_count = 1, result_count = 1}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_abi_adapter_callee_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "semantic");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "function");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "low function");
  ASSERT_EQ(diagnostic->related_locations.size(), 1u);
  EXPECT_EQ(diagnostic->related_locations[0].label, "defined here");
}

TEST_F(LowVerifyTest, AbiAdapterRejectsCalleeArityMismatch) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs : "
      "reg<amdgpu.vgpr x1>, %rhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "low.abi.adapter @bad {callee = @extern_add, conversion = direct, "
      "operand_count = 1, result_count = 1}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 16);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "bad");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "adapter operand");
  ExpectI64Param(*diagnostic, 3, 1);
  ExpectI64Param(*diagnostic, 4, 2);
  EXPECT_EQ(GetStringParam(*diagnostic, 5), "direct");
  ASSERT_GE(diagnostic->related_locations.size(), 1u);
  EXPECT_EQ(diagnostic->related_locations[0].label, "callee defined here");
}

TEST_F(LowVerifyTest, InvokeRejectsNonAdapterRecord) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "test.record @not_adapter {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs : "
      "reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)\n"
      "low.func.def target(@gfx1100) @caller(%lhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.invoke @extern_add(%lhs) {adapter = @not_adapter} : "
      "(reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)\n"
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
                      loom_low_invoke_adapter_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "not_adapter");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "record");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "low ABI adapter");
}

TEST_F(LowVerifyTest, InvokeRejectsAdapterBoundToDifferentCallee) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs : "
      "reg<amdgpu.vgpr x1>, %rhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "low.func.decl target(@gfx1100) @extern_mul(%lhs : "
      "reg<amdgpu.vgpr x1>, %rhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "low.abi.adapter @mul_direct {callee = @extern_mul, conversion = direct, "
      "operand_count = 2, result_count = 1}\n"
      "low.func.def target(@gfx1100) @caller(%lhs : reg<amdgpu.vgpr x1>, "
      "%rhs : reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.invoke @extern_add(%lhs, %rhs) {adapter = @mul_direct} : "
      "(reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 15);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_invoke_callee_ATTR_INDEX);
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_invoke_adapter_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "mul_direct");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "extern_mul");
  ASSERT_EQ(diagnostic->related_locations.size(), 2u);
  EXPECT_EQ(diagnostic->related_locations[0].label, "callee defined here");
  EXPECT_EQ(diagnostic->related_locations[1].label, "adapter defined here");
}

TEST_F(LowVerifyTest, InvokeRejectsAdapterOperandCountMismatch) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs : "
      "reg<amdgpu.vgpr x1>, %rhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "low.abi.adapter @extern_add_direct {callee = @extern_add, conversion = "
      "direct, operand_count = 2, result_count = 1}\n"
      "low.func.def target(@gfx1100) @caller(%lhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.invoke @extern_add(%lhs) {adapter = @extern_add_direct} : "
      "(reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 16);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "extern_add_direct");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "invoke operand");
  ExpectI64Param(*diagnostic, 3, 1);
  ExpectI64Param(*diagnostic, 4, 2);
  EXPECT_EQ(GetStringParam(*diagnostic, 5), "direct");
}

TEST_F(LowVerifyTest, InvokeRejectsAdapterDirectConversionTypeMismatch) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs : "
      "reg<amdgpu.vgpr x1>, %rhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "low.abi.adapter @extern_add_direct {callee = @extern_add, conversion = "
      "direct, operand_count = 2, result_count = 1}\n"
      "func.def @caller(%lhs : i32, %rhs : i32) -> (i32) {\n"
      "  %sum = low.invoke @extern_add(%lhs, %rhs) {adapter = "
      "@extern_add_direct} : (i32, i32) -> (i32)\n"
      "  func.return %sum : i32\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 17);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "extern_add_direct");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "direct");
  EXPECT_EQ(GetStringParam(*diagnostic, 3), "operand");
  ExpectFieldRefParam(*diagnostic, 3, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0);
  ExpectU32Param(*diagnostic, 4, 0);
  ASSERT_EQ(diagnostic->params[5].kind, LOOM_PARAM_TYPE);
  ASSERT_EQ(diagnostic->params[7].kind, LOOM_PARAM_TYPE);
  EXPECT_EQ(GetStringParam(*diagnostic, 6), "argument");
  ASSERT_EQ(diagnostic->related_locations.size(), 2u);
  EXPECT_EQ(diagnostic->related_locations[0].label, "callee defined here");
  EXPECT_EQ(diagnostic->related_locations[1].label, "adapter defined here");
}

TEST_F(LowVerifyTest, InvokeWithoutAdapterRequiresDirectTypes) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs : "
      "reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)\n"
      "func.def @caller(%lhs : i32) -> (i32) {\n"
      "  %sum = low.invoke @extern_add(%lhs) : (i32) -> (i32)\n"
      "  func.return %sum : i32\n"
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
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0);
  ExpectU32Param(*diagnostic, 2, 0);
  ASSERT_EQ(diagnostic->params[3].kind, LOOM_PARAM_TYPE);
  EXPECT_EQ(GetStringParam(*diagnostic, 4), "argument");
  ASSERT_EQ(diagnostic->params[5].kind, LOOM_PARAM_TYPE);
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
