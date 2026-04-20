// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/testing/context.h"
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
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_verify_result_t VerifySource(const char* source,
                                    DiagnosticCapture* verify_capture) {
    loom_module_t* module = ParseSource(source);
    if (!module) return {};
    loom_verify_result_t result =
        VerifyParsedModule(source, module, verify_capture);
    loom_module_free(module);
    return result;
  }

  loom_module_t* ParseSource(const char* source) {
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
      return nullptr;
    }
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_verify_result_t VerifyParsedModule(const char* source,
                                          loom_module_t* module,
                                          DiagnosticCapture* verify_capture) {
    const char* filename = "low_verify_test.loom";
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
    return result;
  }

  loom_op_t* FindFirstLowFuncDecl(loom_module_t* module) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_module_block(module), op) {
      if (loom_low_func_decl_isa(op)) return op;
    }
    ADD_FAILURE() << "expected module to contain low.func.decl";
    return nullptr;
  }

  loom_op_t* FindFirstLowFuncDef(loom_module_t* module) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_module_block(module), op) {
      if (loom_low_func_def_isa(op)) return op;
    }
    ADD_FAILURE() << "expected module to contain low.func.def";
    return nullptr;
  }

  loom_op_t* FindFirstLowSlot(loom_module_t* module) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_module_block(module), op) {
      if (loom_low_slot_isa(op)) return op;
    }
    ADD_FAILURE() << "expected module to contain low.slot";
    return nullptr;
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

TEST_F(LowVerifyTest, InvokeMatchesPureDirectLowFunctionSignature) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl pure target(@vm_target) @extern_add(%lhs : "
      "reg<vm.i32>) -> (reg<vm.i32>)\n"
      "low.func.def target(@vm_target) @caller(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>) {\n"
      "  %sum = low.invoke pure @extern_add(%lhs) : (reg<vm.i32>) -> "
      "(reg<vm.i32>)\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, ImportedDeclMatchesMappedSemanticCaller) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl allocation(fixed) schedule(locked) import(vm, "
      "\"iree.vm.core.add_i32\") target(@vm_target) @extern_add(%lhs : "
      "reg<vm.i32>, %rhs : reg<vm.i32>) -> (reg<vm.i32>)\n"
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

TEST_F(LowVerifyTest, ImportedDeclRejectsMissingCodeSymbol) {
  static const char* kSource =
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>)\n";
  loom_module_t* module = ParseSource(kSource);
  ASSERT_NE(module, nullptr);
  loom_op_t* decl = FindFirstLowFuncDecl(module);
  ASSERT_NE(decl, nullptr);
  loom_op_attrs(decl)[loom_low_func_decl_import_kind_ATTR_INDEX] =
      loom_attr_enum(LOOM_LOW_IMPORT_KIND_VM);

  DiagnosticCapture capture;
  loom_verify_result_t result = VerifyParsedModule(kSource, module, &capture);
  loom_module_free(module);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 23);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "import");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_func_decl_import_kind_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "import kind requires a code symbol string");
}

TEST_F(LowVerifyTest, ImportedDeclRejectsMissingImportKind) {
  static const char* kSource =
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>)\n";
  loom_module_t* module = ParseSource(kSource);
  ASSERT_NE(module, nullptr);
  loom_op_t* decl = FindFirstLowFuncDecl(module);
  ASSERT_NE(decl, nullptr);
  loom_string_id_t code_symbol_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(
      module, IREE_SV("iree.vm.core.add_i32"), &code_symbol_id));
  loom_op_attrs(decl)[loom_low_func_decl_code_symbol_ATTR_INDEX] =
      loom_attr_string(code_symbol_id);

  DiagnosticCapture capture;
  loom_verify_result_t result = VerifyParsedModule(kSource, module, &capture);
  loom_module_free(module);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 23);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "import");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_func_decl_code_symbol_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "code symbol requires an import kind");
}

TEST_F(LowVerifyTest, ImportedDeclRejectsEmptyCodeSymbol) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl import(vm, \"\") target(@vm_target) @extern_add(%lhs : "
      "reg<vm.i32>) -> (reg<vm.i32>)\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 23);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "import");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_func_decl_code_symbol_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "code symbol must not be empty");
}

TEST_F(LowVerifyTest, FuncDefRejectsImportedCodeContract) {
  static const char* kSource =
      "test.record @vm_target {}\n"
      "low.func.def target(@vm_target) @inline_body() {\n"
      "  low.return\n"
      "}\n";
  loom_module_t* module = ParseSource(kSource);
  ASSERT_NE(module, nullptr);
  loom_op_t* def = FindFirstLowFuncDef(module);
  ASSERT_NE(def, nullptr);
  loom_op_attrs(def)[loom_low_func_def_import_kind_ATTR_INDEX] =
      loom_attr_enum(LOOM_LOW_IMPORT_KIND_VM);

  DiagnosticCapture capture;
  loom_verify_result_t result = VerifyParsedModule(kSource, module, &capture);
  loom_module_free(module);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 23);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "inline_body");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "import");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_func_def_import_kind_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "imported code belongs on low.func.decl; low.func.def owns an "
            "inline body");
}

TEST_F(LowVerifyTest, FuncDefRejectsNamelessAllocationMode) {
  static const char* kSource =
      "test.record @vm_target {}\n"
      "low.func.def target(@vm_target) @inline_body() {\n"
      "  low.return\n"
      "}\n";
  loom_module_t* module = ParseSource(kSource);
  ASSERT_NE(module, nullptr);
  loom_op_t* def = FindFirstLowFuncDef(module);
  ASSERT_NE(def, nullptr);
  loom_op_attrs(def)[loom_low_func_def_allocation_ATTR_INDEX] =
      loom_attr_enum(0);

  DiagnosticCapture capture;
  loom_verify_result_t result = VerifyParsedModule(kSource, module, &capture);
  loom_module_free(module);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 23);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "inline_body");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "allocation");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_func_def_allocation_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "explicit allocation mode must name virtual, assigned, or fixed");
}

TEST_F(LowVerifyTest, ImportedDeclRejectsNamelessImportKind) {
  static const char* kSource =
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>)\n";
  loom_module_t* module = ParseSource(kSource);
  ASSERT_NE(module, nullptr);
  loom_op_t* decl = FindFirstLowFuncDecl(module);
  ASSERT_NE(decl, nullptr);
  loom_string_id_t code_symbol_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(
      module, IREE_SV("iree.vm.core.add_i32"), &code_symbol_id));
  loom_op_attrs(decl)[loom_low_func_decl_import_kind_ATTR_INDEX] =
      loom_attr_enum(0);
  loom_op_attrs(decl)[loom_low_func_decl_code_symbol_ATTR_INDEX] =
      loom_attr_string(code_symbol_id);

  DiagnosticCapture capture;
  loom_verify_result_t result = VerifyParsedModule(kSource, module, &capture);
  loom_module_free(module);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 23);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "import");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_func_decl_import_kind_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "import kind must name vm, native, rocasm, or object");
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

TEST_F(LowVerifyTest, InvokeMatchesPureMappedAbiAdapterWithoutEffects) {
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
      "  %sum = low.invoke pure @extern_add(%lhs, %rhs) {adapter = "
      "@extern_add_i32} : (i32, i32) -> (i32)\n"
      "  func.return %sum : i32\n"
      "}\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, InvokeAcceptsEffectfulMappedAbiAdapterWhenNotPure) {
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
      "low.abi.effect @extern_add_i32_call {adapter = @extern_add_i32, kind = "
      "call, resource = \"vm.import\"}\n"
      "low.abi.clobber @extern_add_i32_state {adapter = @extern_add_i32, "
      "resource = \"vm.state\"}\n"
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

TEST_F(LowVerifyTest, SemanticInvokeRejectsMissingAdapter) {
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
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 20);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "semantic function");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "semantic function bodies require an explicit ABI adapter");
}

TEST_F(LowVerifyTest, InvokeRejectsTopLevelContext) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_bar()\n"
      "low.invoke @extern_bar() : ()\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 20);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_bar");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "module");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "low.invoke must be nested under a function-like body");
}

TEST_F(LowVerifyTest, LowFunctionRejectsMappedAbiAdapter) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>)\n"
      "low.abi.adapter @extern_add_i32 {callee = @extern_add, conversion = "
      "mapped, operand_count = 1, result_count = 1}\n"
      "low.abi.operand @extern_add_i32_lhs {adapter = @extern_add_i32, index "
      "= 0, conversion = direct, semantic_type = reg<vm.i32>, abi_type = "
      "reg<vm.i32>}\n"
      "low.abi.result @extern_add_i32_result {adapter = @extern_add_i32, "
      "index = 0, conversion = direct, semantic_type = reg<vm.i32>, "
      "abi_type = reg<vm.i32>}\n"
      "low.func.def target(@vm_target) @caller(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>) {\n"
      "  %sum = low.invoke @extern_add(%lhs) {adapter = @extern_add_i32} : "
      "(reg<vm.i32>) -> (reg<vm.i32>)\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 20);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "low function");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "mapped ABI adapters cross the semantic-to-low boundary and are "
            "not valid in low function bodies");
}

TEST_F(LowVerifyTest, LowFunctionRejectsCrossTargetInvoke) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "test.record @gfx1200 {}\n"
      "low.func.decl target(@gfx1200) @extern_add(%lhs : "
      "reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)\n"
      "low.func.def target(@gfx1100) @caller(%lhs : reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.invoke @extern_add(%lhs) : (reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 20);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "low function");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "callee target must match enclosing low function target");
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

TEST_F(LowVerifyTest, AbiEffectRejectsMissingResourceForRead) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_load(%address : reg<vm.i64>) "
      "-> (reg<vm.i32>)\n"
      "low.abi.adapter @extern_load_i32 {callee = @extern_load, conversion = "
      "direct, operand_count = 1, result_count = 1}\n"
      "low.abi.effect @extern_load_read {adapter = @extern_load_i32, kind = "
      "read}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 21);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_load_i32");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "effect");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "extern_load_read");
  EXPECT_EQ(GetStringParam(*diagnostic, 3),
            "read/write/readwrite effects require a resource key");
}

TEST_F(LowVerifyTest, AbiEffectRejectsUnqualifiedResource) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_load(%address : reg<vm.i64>) "
      "-> (reg<vm.i32>)\n"
      "low.abi.adapter @extern_load_i32 {callee = @extern_load, conversion = "
      "direct, operand_count = 1, result_count = 1}\n"
      "low.abi.effect @extern_load_read {adapter = @extern_load_i32, kind = "
      "read, resource = \"state\"}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 27);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_abi_effect_resource_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "resource");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "state");
}

TEST_F(LowVerifyTest, AbiClobberRejectsUnqualifiedResource) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_call()\n"
      "low.abi.adapter @extern_call_i32 {callee = @extern_call, conversion = "
      "direct, operand_count = 0, result_count = 0}\n"
      "low.abi.clobber @extern_call_state {adapter = @extern_call_i32, "
      "resource = \"state\"}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 27);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_abi_clobber_resource_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "resource");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "state");
}

TEST_F(LowVerifyTest, InvokeRejectsPureAdapterWithEffects) {
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
      "low.abi.result @extern_add_i32_result {adapter = @extern_add_i32, "
      "index = 0, conversion = register_to_scalar, semantic_type = i32, "
      "abi_type = reg<vm.i32>}\n"
      "low.abi.effect @extern_add_i32_call {adapter = @extern_add_i32, kind = "
      "call, resource = \"vm.import\"}\n"
      "low.abi.clobber @extern_add_i32_state {adapter = @extern_add_i32, "
      "resource = \"vm.state\"}\n"
      "func.def @caller(%lhs : i32) -> (i32) {\n"
      "  %sum = low.invoke pure @extern_add(%lhs) {adapter = "
      "@extern_add_i32} : (i32) -> (i32)\n"
      "  func.return %sum : i32\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 22);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "extern_add_i32");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "ABI adapter declares observable effects or clobbers");
  ExpectU32Param(*diagnostic, 3, 1);
  ExpectU32Param(*diagnostic, 4, 1);
}

TEST_F(LowVerifyTest, InvokeRejectsPureDirectImpureCallee) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>)\n"
      "low.func.def target(@vm_target) @caller(%lhs : reg<vm.i32>) -> "
      "(reg<vm.i32>) {\n"
      "  %sum = low.invoke pure @extern_add(%lhs) : (reg<vm.i32>) -> "
      "(reg<vm.i32>)\n"
      "  low.return %sum : reg<vm.i32>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 22);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "<direct>");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "direct callee has no pure contract");
  ExpectU32Param(*diagnostic, 3, 0);
  ExpectU32Param(*diagnostic, 4, 0);
}

TEST_F(LowVerifyTest, SlotTrafficPassesWithOwnedSlot) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.def target(@vm_target) @slot_roundtrip(%input : reg<vm.i32>) "
      "-> (reg<vm.i32>) {\n"
      "  low.spill %input, @slot_roundtrip_spill {offset = 0} : reg<vm.i32>\n"
      "  %reload = low.reload @slot_roundtrip_spill {offset = 0} : "
      "reg<vm.i32>\n"
      "  %addr = low.frame_index @slot_roundtrip_spill {offset = 0} : "
      "reg<vm.ptr>\n"
      "  low.return %reload : reg<vm.i32>\n"
      "}\n"
      "low.slot @slot_roundtrip_spill {function = @slot_roundtrip, space = "
      "scratch, size = 4, align = 4}\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, ResourceImportsPassForVmNativeAndHalAbiShapes) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "target.snapshot @vm_snapshot {codegen_format = vm, target_triple = "
      "\"iree-vm\", data_layout = \"\", artifact_format = vm_bytecode, "
      "target_cpu = \"\", target_features = \"\", "
      "default_pointer_bitwidth = 64, index_bitwidth = 64, "
      "offset_bitwidth = 64, memory_space_generic = 0, "
      "memory_space_global = 0, memory_space_workgroup = 0, "
      "memory_space_constant = 0, memory_space_private = 0, "
      "memory_space_host = 0, memory_space_descriptor = 0}\n"
      "target.export @vm_export {source = @vm_resource, export_symbol = "
      "\"vm_resource\", abi = vm_module_function, linkage = default, "
      "hal_binding_alignment = 0, hal_workgroup_size_x = 0, "
      "hal_workgroup_size_y = 0, hal_workgroup_size_z = 0, "
      "hal_flat_workgroup_size_min = 0, hal_flat_workgroup_size_max = 0, "
      "hal_buffer_resource_flags = 0}\n"
      "target.config @vm_config {contract_set_key = \"test.low.core\", "
      "contract_feature_bits = 0}\n"
      "target.bundle @vm_target {snapshot = @vm_snapshot, export_plan = "
      "@vm_export, config = @vm_config}\n"
      "low.func.def target(@vm_target) @vm_resource() -> (reg<vm.i32>) {\n"
      "  %state = low.resource @vm_state : reg<vm.i32>\n"
      "  low.return %state : reg<vm.i32>\n"
      "}\n"
      "low.abi.resource @vm_state {function = @vm_resource, kind = vm_state, "
      "index = 0, semantic_type = vm.state, abi_type = reg<vm.i32>}\n"
      "target.snapshot @native_snapshot {codegen_format = low_native, "
      "target_triple = \"x86_64-unknown-linux-gnu\", data_layout = \"\", "
      "artifact_format = elf, target_cpu = \"\", target_features = \"\", "
      "default_pointer_bitwidth = 64, index_bitwidth = 64, "
      "offset_bitwidth = 64, memory_space_generic = 0, "
      "memory_space_global = 0, memory_space_workgroup = 4294967295, "
      "memory_space_constant = 0, memory_space_private = 0, "
      "memory_space_host = 0, memory_space_descriptor = 4294967295}\n"
      "target.export @native_export {source = @native_resource, "
      "export_symbol = \"native_resource\", abi = object_function, linkage = "
      "dso_local, hal_binding_alignment = 0, hal_workgroup_size_x = 0, "
      "hal_workgroup_size_y = 0, hal_workgroup_size_z = 0, "
      "hal_flat_workgroup_size_min = 0, hal_flat_workgroup_size_max = 0, "
      "hal_buffer_resource_flags = 0}\n"
      "target.config @native_config {contract_set_key = \"test.low.core\", "
      "contract_feature_bits = 0}\n"
      "target.bundle @native_target {snapshot = @native_snapshot, "
      "export_plan = @native_export, config = @native_config}\n"
      "low.func.def target(@native_target) @native_resource() -> "
      "(reg<native.ptr>) {\n"
      "  %ptr = low.resource @native_arg0 : reg<native.ptr>\n"
      "  low.return %ptr : reg<native.ptr>\n"
      "}\n"
      "low.abi.resource @native_arg0 {function = @native_resource, kind = "
      "native_pointer, index = 0, semantic_type = buffer, abi_type = "
      "reg<native.ptr>}\n"
      "target.snapshot @hal_snapshot {codegen_format = low_native, "
      "target_triple = \"amdgcn-amd-amdhsa\", data_layout = \"\", "
      "artifact_format = elf, target_cpu = \"gfx1100\", target_features = "
      "\"\", default_pointer_bitwidth = 64, index_bitwidth = 64, "
      "offset_bitwidth = 64, memory_space_generic = 0, "
      "memory_space_global = 1, memory_space_workgroup = 3, "
      "memory_space_constant = 4, memory_space_private = 5, "
      "memory_space_host = 4294967295, memory_space_descriptor = "
      "4294967295}\n"
      "target.export @hal_export {source = @hal_kernel, export_symbol = "
      "\"hal_kernel\", abi = hal_kernel, linkage = default, "
      "hal_binding_alignment = 16, hal_workgroup_size_x = 1, "
      "hal_workgroup_size_y = 1, hal_workgroup_size_z = 1, "
      "hal_flat_workgroup_size_min = 1, hal_flat_workgroup_size_max = 1, "
      "hal_buffer_resource_flags = 0}\n"
      "target.config @hal_config {contract_set_key = \"test.low.core\", "
      "contract_feature_bits = 0}\n"
      "target.bundle @hal_target {snapshot = @hal_snapshot, export_plan = "
      "@hal_export, config = @hal_config}\n"
      "low.func.def target(@hal_target) @hal_kernel() -> "
      "(reg<amdgpu.sgpr x4>) {\n"
      "  %binding = low.resource @binding0 : reg<amdgpu.sgpr x4>\n"
      "  low.return %binding : reg<amdgpu.sgpr x4>\n"
      "}\n"
      "low.abi.resource @binding0 {function = @hal_kernel, kind = "
      "hal_buffer_resource, index = 0, semantic_type = hal.buffer, abi_type = "
      "reg<amdgpu.sgpr x4>}\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, ResourceRejectsLowFunctionDeclarationOwner) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_owner()\n"
      "low.abi.resource @bad_resource {function = @extern_owner, kind = "
      "vm_state, index = 0, semantic_type = vm.state, abi_type = "
      "reg<vm.i32>}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_abi_resource_function_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_owner");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "low function definition");
}

TEST_F(LowVerifyTest, ResourceRejectsWrongExportAbi) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "target.snapshot @native_snapshot {codegen_format = low_native, "
      "target_triple = \"x86_64-unknown-linux-gnu\", data_layout = \"\", "
      "artifact_format = elf, target_cpu = \"\", target_features = \"\", "
      "default_pointer_bitwidth = 64, index_bitwidth = 64, "
      "offset_bitwidth = 64, memory_space_generic = 0, "
      "memory_space_global = 0, memory_space_workgroup = 4294967295, "
      "memory_space_constant = 0, memory_space_private = 0, "
      "memory_space_host = 0, memory_space_descriptor = 4294967295}\n"
      "target.export @native_export {source = @vm_resource, export_symbol = "
      "\"vm_resource\", abi = object_function, linkage = dso_local, "
      "hal_binding_alignment = 0, hal_workgroup_size_x = 0, "
      "hal_workgroup_size_y = 0, hal_workgroup_size_z = 0, "
      "hal_flat_workgroup_size_min = 0, hal_flat_workgroup_size_max = 0, "
      "hal_buffer_resource_flags = 0}\n"
      "target.config @native_config {contract_set_key = \"test.low.core\", "
      "contract_feature_bits = 0}\n"
      "target.bundle @native_target {snapshot = @native_snapshot, "
      "export_plan = @native_export, config = @native_config}\n"
      "low.func.def target(@native_target) @vm_resource() -> (reg<vm.i32>) {\n"
      "  %state = low.resource @vm_state : reg<vm.i32>\n"
      "  low.return %state : reg<vm.i32>\n"
      "}\n"
      "low.abi.resource @vm_state {function = @vm_resource, kind = vm_state, "
      "index = 0, semantic_type = vm.state, abi_type = reg<vm.i32>}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 24);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.abi.resource");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_abi_resource_kind_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "kind");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "VM resources require vm_module_function export ABI");
}

TEST_F(LowVerifyTest, ResourceImportRejectsDifferentOwnerFunction) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "target.snapshot @vm_snapshot {codegen_format = vm, target_triple = "
      "\"iree-vm\", data_layout = \"\", artifact_format = vm_bytecode, "
      "target_cpu = \"\", target_features = \"\", "
      "default_pointer_bitwidth = 64, index_bitwidth = 64, "
      "offset_bitwidth = 64, memory_space_generic = 0, "
      "memory_space_global = 0, memory_space_workgroup = 0, "
      "memory_space_constant = 0, memory_space_private = 0, "
      "memory_space_host = 0, memory_space_descriptor = 0}\n"
      "target.export @vm_export {source = @resource_owner, export_symbol = "
      "\"resource_owner\", abi = vm_module_function, linkage = default, "
      "hal_binding_alignment = 0, hal_workgroup_size_x = 0, "
      "hal_workgroup_size_y = 0, hal_workgroup_size_z = 0, "
      "hal_flat_workgroup_size_min = 0, hal_flat_workgroup_size_max = 0, "
      "hal_buffer_resource_flags = 0}\n"
      "target.config @vm_config {contract_set_key = \"test.low.core\", "
      "contract_feature_bits = 0}\n"
      "target.bundle @vm_target {snapshot = @vm_snapshot, export_plan = "
      "@vm_export, config = @vm_config}\n"
      "low.func.def target(@vm_target) @resource_owner() {\n"
      "  low.return\n"
      "}\n"
      "low.func.def target(@vm_target) @resource_user() -> (reg<vm.i32>) {\n"
      "  %state = low.resource @vm_state : reg<vm.i32>\n"
      "  low.return %state : reg<vm.i32>\n"
      "}\n"
      "low.abi.resource @vm_state {function = @resource_owner, kind = "
      "vm_state, index = 0, semantic_type = vm.state, abi_type = "
      "reg<vm.i32>}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 24);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.resource");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_resource_resource_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "resource");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "resource owner must match the enclosing low function");
}

TEST_F(LowVerifyTest, ResourceImportRejectsResultTypeMismatch) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "target.snapshot @vm_snapshot {codegen_format = vm, target_triple = "
      "\"iree-vm\", data_layout = \"\", artifact_format = vm_bytecode, "
      "target_cpu = \"\", target_features = \"\", "
      "default_pointer_bitwidth = 64, index_bitwidth = 64, "
      "offset_bitwidth = 64, memory_space_generic = 0, "
      "memory_space_global = 0, memory_space_workgroup = 0, "
      "memory_space_constant = 0, memory_space_private = 0, "
      "memory_space_host = 0, memory_space_descriptor = 0}\n"
      "target.export @vm_export {source = @vm_resource, export_symbol = "
      "\"vm_resource\", abi = vm_module_function, linkage = default, "
      "hal_binding_alignment = 0, hal_workgroup_size_x = 0, "
      "hal_workgroup_size_y = 0, hal_workgroup_size_z = 0, "
      "hal_flat_workgroup_size_min = 0, hal_flat_workgroup_size_max = 0, "
      "hal_buffer_resource_flags = 0}\n"
      "target.config @vm_config {contract_set_key = \"test.low.core\", "
      "contract_feature_bits = 0}\n"
      "target.bundle @vm_target {snapshot = @vm_snapshot, export_plan = "
      "@vm_export, config = @vm_config}\n"
      "low.func.def target(@vm_target) @vm_resource() -> (reg<vm.i64>) {\n"
      "  %state = low.resource @vm_state : reg<vm.i64>\n"
      "  low.return %state : reg<vm.i64>\n"
      "}\n"
      "low.abi.resource @vm_state {function = @vm_resource, kind = vm_state, "
      "index = 0, semantic_type = vm.state, abi_type = reg<vm.i32>}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 24);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.resource");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_RESULT, 0);
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "result");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "result type must match the resource ABI type");
}

TEST_F(LowVerifyTest, SlotRejectsSemanticFunctionOwner) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "func.def @semantic_owner() {\n"
      "  func.return\n"
      "}\n"
      "low.slot @bad_slot {function = @semantic_owner, space = scratch, size "
      "= 4, align = 4}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_slot_function_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "semantic_owner");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "low function definition");
}

TEST_F(LowVerifyTest, SlotRejectsLowFunctionDeclarationOwner) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_owner()\n"
      "low.slot @bad_slot {function = @extern_owner, space = scratch, size = "
      "4, align = 4}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_slot_function_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_owner");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "low function definition");
}

TEST_F(LowVerifyTest, SlotRejectsUnnamedSpace) {
  static const char* kSource =
      "test.record @vm_target {}\n"
      "low.func.def target(@vm_target) @slot_owner() {\n"
      "  low.return\n"
      "}\n"
      "low.slot @bad_slot {function = @slot_owner, space = scratch, size = 4, "
      "align = 4}\n";
  loom_module_t* module = ParseSource(kSource);
  ASSERT_NE(module, nullptr);
  loom_op_t* slot = FindFirstLowSlot(module);
  ASSERT_NE(slot, nullptr);
  loom_op_attrs(slot)[loom_low_slot_space_ATTR_INDEX] = loom_attr_enum(0);

  DiagnosticCapture capture;
  loom_verify_result_t result = VerifyParsedModule(kSource, module, &capture);
  loom_module_free(module);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 24);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.slot");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_slot_space_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "space");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "space must name a supported low slot space");
}

TEST_F(LowVerifyTest, SlotRejectsNonPositiveSize) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.def target(@vm_target) @slot_owner() {\n"
      "  low.return\n"
      "}\n"
      "low.slot @bad_slot {function = @slot_owner, space = scratch, size = 0, "
      "align = 4}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 24);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.slot");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_slot_size_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "size");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "size must be positive");
}

TEST_F(LowVerifyTest, SlotRejectsNonPowerOfTwoAlignment) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.def target(@vm_target) @slot_owner() {\n"
      "  low.return\n"
      "}\n"
      "low.slot @bad_slot {function = @slot_owner, space = scratch, size = 4, "
      "align = 3}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 24);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.slot");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_slot_align_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "align");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "alignment must be a positive power of two");
}

TEST_F(LowVerifyTest, SpillRejectsSlotOwnedByDifferentLowFunction) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.def target(@vm_target) @slot_owner() {\n"
      "  low.return\n"
      "}\n"
      "low.func.def target(@vm_target) @slot_user(%input : reg<vm.i32>) {\n"
      "  low.spill %input, @owner_slot {offset = 0} : reg<vm.i32>\n"
      "  low.return\n"
      "}\n"
      "low.slot @owner_slot {function = @slot_owner, space = scratch, size = "
      "4, align = 4}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 24);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.spill");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_spill_slot_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "slot");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "slot owner must match the enclosing low function");
}

TEST_F(LowVerifyTest, ReloadRejectsOffsetOutsideSlot) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.def target(@vm_target) @slot_user() -> (reg<vm.i32>) {\n"
      "  %reload = low.reload @small_slot {offset = 4} : reg<vm.i32>\n"
      "  low.return %reload : reg<vm.i32>\n"
      "}\n"
      "low.slot @small_slot {function = @slot_user, space = scratch, size = "
      "4, align = 4}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 24);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.reload");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_reload_offset_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "offset");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "offset must address a byte inside the referenced slot");
}

TEST_F(LowVerifyTest, FrameIndexRejectsNonSlotRecord) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.def target(@vm_target) @slot_user() -> (reg<vm.ptr>) {\n"
      "  %addr = low.frame_index @vm_target {offset = 0} : reg<vm.ptr>\n"
      "  low.return %addr : reg<vm.ptr>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_frame_index_slot_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "vm_target");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "low slot");
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
