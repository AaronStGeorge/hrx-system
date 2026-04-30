// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::ExpectError;
using ::loom::testing::ExpectFieldRefParam;
using ::loom::testing::ExpectU32Param;
using ::loom::testing::FindDiagnostic;
using ::loom::testing::GetStringParam;

class LowVerifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_verify_result_t VerifySource(const char* source,
                                    DiagnosticCapture* verify_capture) {
    loom_module_t* module = ParseSource(source);
    if (!module) {
      return {};
    }
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
      if (module) {
        loom_module_free(module);
      }
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
      if (loom_low_func_decl_isa(op)) {
        return op;
      }
    }
    ADD_FAILURE() << "expected module to contain low.func.decl";
    return nullptr;
  }

  loom_op_t* FindFirstLowFuncDef(loom_module_t* module) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_module_block(module), op) {
      if (loom_low_func_def_isa(op)) {
        return op;
      }
    }
    ADD_FAILURE() << "expected module to contain low.func.def";
    return nullptr;
  }

  loom_op_t* FindFirstLowOpInBody(loom_op_t* func_op) {
    loom_region_t* body = loom_low_func_def_body(func_op);
    if (!body || body->block_count == 0) {
      ADD_FAILURE() << "expected low.func.def to have a body";
      return nullptr;
    }
    loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_region_entry_block(body), op) {
      if (loom_low_op_isa(op)) {
        return op;
      }
    }
    ADD_FAILURE() << "expected low.func.def body to contain low.op";
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

static constexpr const char* kVmTargetBundle =
    "target.profile @vm_target preset(\"test-low\")\n";

static constexpr const char* kSemanticVmTargetBundle =
    "target.profile @vm_target preset(\"test-low\")\n";

static constexpr const char* kNativeTargetBundle =
    "target.profile @native_target preset(\"test-low\")\n";

static constexpr const char* kHalTargetBundle =
    "target.profile @hal_target preset(\"test-low\")\n";

TEST_F(LowVerifyTest, DescriptorKeysPassWithQualifiedSegments) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.def target(@gfx1100) @add(%lhs: reg<amdgpu.vgpr x1>, "
      "%rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.op<amdgpu.v_add_u32>(%lhs, %rhs) : "
      "(reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> "
      "reg<amdgpu.vgpr x1>\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, DescriptorIdMustMatchDescriptorKey) {
  static const char* kSource =
      "test.record @gfx1100 {}\n"
      "low.func.def target(@gfx1100) @add(%lhs: reg<amdgpu.vgpr x1>, "
      "%rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.op<amdgpu.v_add_u32>(%lhs, %rhs) : "
      "(reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> "
      "reg<amdgpu.vgpr x1>\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n";
  loom_module_t* module = ParseSource(kSource);
  ASSERT_NE(module, nullptr);
  loom_op_t* func_op = FindFirstLowFuncDef(module);
  ASSERT_NE(func_op, nullptr);
  loom_op_t* packet_op = FindFirstLowOpInBody(func_op);
  ASSERT_NE(packet_op, nullptr);
  loom_op_attrs(packet_op)[loom_low_op_descriptor_id_ATTR_INDEX] =
      loom_attr_i64(0);

  DiagnosticCapture capture;
  loom_verify_result_t result = VerifyParsedModule(kSource, module, &capture);
  loom_module_free(module);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 17);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.op");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "descriptor_id");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_op_descriptor_id_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "descriptor ID must match the stable ID derived from opcode");
}

TEST_F(LowVerifyTest, LiveInAndResourceAcceptEntryPreamble) {
  std::string source = kVmTargetBundle;
  source.append(
      "low.func.def target(@vm_target) @vm_resource() -> (reg<vm.i32>) {\n"
      "  %state = low.resource<vm_state> {index = 0, semantic_type = "
      "vm.state} : reg<vm.i32>\n"
      "  %arg0 = low.live_in<test.arg0> : reg<vm.i32>\n"
      "  %copy = low.copy %state : reg<vm.i32> -> reg<vm.i32>\n"
      "  low.return %copy : reg<vm.i32>\n"
      "}\n");
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(source.c_str(), &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, LiveInRejectsAfterEntryPreamble) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @target {}\n"
      "low.func.def target(@target) @kernel() -> (reg<test.i32>) {\n"
      "  %arg0 = low.live_in<test.arg0> : reg<test.i32>\n"
      "  %copy = low.copy %arg0 : reg<test.i32> -> reg<test.i32>\n"
      "  %late = low.live_in<test.arg1> : reg<test.i32>\n"
      "  low.return %late : reg<test.i32>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 17);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.live_in");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "position");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "live-ins and resources must form an entry-block prefix before "
            "ordinary low packets");
}

TEST_F(LowVerifyTest, ResourceRejectsAfterEntryPreamble) {
  std::string source = kVmTargetBundle;
  source.append(
      "low.func.def target(@vm_target) @vm_resource() -> (reg<vm.i32>) {\n"
      "  %arg0 = low.live_in<test.arg0> : reg<vm.i32>\n"
      "  %copy = low.copy %arg0 : reg<vm.i32> -> reg<vm.i32>\n"
      "  %late = low.resource<vm_state> {index = 0, semantic_type = "
      "vm.state} : reg<vm.i32>\n"
      "  low.return %late : reg<vm.i32>\n"
      "}\n");
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(source.c_str(), &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 17);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.resource");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "position");
}

TEST_F(LowVerifyTest, LiveInRejectsNonEntryBlock) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @target {}\n"
      "low.func.def target(@target) @kernel() -> (reg<test.i32>) {\n"
      "  %arg0 = low.live_in<test.arg0> : reg<test.i32>\n"
      "  low.br ^exit\n"
      "^exit:\n"
      "  %late = low.live_in<test.arg1> : reg<test.i32>\n"
      "  low.return %late : reg<test.i32>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 17);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.live_in");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "position");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "live-ins and resources must appear in the low entry block");
}

TEST_F(LowVerifyTest, ImportedDeclContractsAreLocalToDecl) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl import(vm, \"iree.vm.core.add_i32\") target(@vm_target) "
      "@extern_add(%lhs: reg<vm.i32>) -> (reg<vm.i32>)\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, ImportedDeclRejectsMissingCodeSymbol) {
  static const char* kSource =
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs: reg<vm.i32>) -> "
      "(reg<vm.i32>)\n";
  loom_module_t* module = ParseSource(kSource);
  ASSERT_NE(module, nullptr);
  loom_op_t* decl = FindFirstLowFuncDecl(module);
  ASSERT_NE(decl, nullptr);
  loom_op_attrs(decl)[loom_low_func_decl_import_kind_ATTR_INDEX] =
      loom_attr_enum(LOOM_LOW_FUNC_DECL_IMPORT_KIND_VM);

  DiagnosticCapture capture;
  loom_verify_result_t result = VerifyParsedModule(kSource, module, &capture);
  loom_module_free(module);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 16);
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
      "low.func.decl target(@vm_target) @extern_add(%lhs: reg<vm.i32>) -> "
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
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 16);
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

TEST_F(LowVerifyTest, ImportedDeclRejectsNamelessImportKind) {
  static const char* kSource =
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs: reg<vm.i32>) -> "
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
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 16);
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

TEST_F(LowVerifyTest, InvokeAcceptsExplicitTranslatedLowFunction) {
  std::string source = kSemanticVmTargetBundle;
  source.append(
      "func.def @semantic(%value: i32, %state: vm.state) -> (i32) {\n"
      "  func.return %value : i32\n"
      "}\n"
      "low.func.def target(@vm_target) @semantic_low(%value: reg<vm.i32>) "
      "-> (reg<vm.i32>) {\n"
      "  %state = low.resource<vm_state> {index = 0, semantic_type = "
      "vm.state} : reg<vm.i32>\n"
      "  low.return %value : reg<vm.i32>\n"
      "}\n"
      "func.def @caller(%value: i32, %state: vm.state) -> (i32) {\n"
      "  %result = low.invoke @semantic_low(%value, %state) : "
      "(i32, vm.state) -> (i32)\n"
      "  func.return %result : i32\n"
      "}\n");
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(source.c_str(), &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, InvokeDoesNotDependOnSourceSignature) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.def target(@vm_target) @semantic_low(%value: reg<vm.i32>) "
      "-> (reg<vm.i32>) {\n"
      "  low.return %value : reg<vm.i32>\n"
      "}\n"
      "func.def @caller(%value: f32) -> (i32) {\n"
      "  %result = low.invoke @semantic_low(%value) : (f32) -> (i32)\n"
      "  func.return %result : i32\n"
      "}\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, InvokeRejectsNonLowCallee) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "func.decl @semantic(%arg: i32) -> (i32)\n"
      "func.def @caller(%lhs: i32) -> (i32) {\n"
      "  %sum = low.invoke @semantic(%lhs) : (i32) -> (i32)\n"
      "  func.return %sum : i32\n"
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
}

TEST_F(LowVerifyTest, FuncCallMatchesDirectLowFunctionSignature) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs: "
      "reg<amdgpu.vgpr x1>, %rhs: reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "low.func.def target(@gfx1100) @caller(%lhs: reg<amdgpu.vgpr x1>, "
      "%rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.func.call @extern_add(%lhs, %rhs) : "
      "(reg<amdgpu.vgpr x1>, reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, FuncCallRequiresLowFunctionBody) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add()\n"
      "func.def @caller() {\n"
      "  low.func.call @extern_add() : ()\n"
      "  func.return\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 17);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.func.call");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "enclosing low entry");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "low.func.call must be nested under a low function or low kernel "
            "body");
}

TEST_F(LowVerifyTest, InvokeRejectsLowFunctionBody) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @semantic_low()\n"
      "low.func.def target(@vm_target) @caller() {\n"
      "  low.invoke @semantic_low() : ()\n"
      "  low.return\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 29);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.invoke");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "forbidden");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "low.func.def");
  EXPECT_EQ(GetStringParam(*diagnostic, 3), "low.func.def");
}

TEST_F(LowVerifyTest, FuncCallRejectsNonLowCallee) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "func.decl @semantic(%arg: i32) -> (i32)\n"
      "test.record @gfx1100 {}\n"
      "low.func.def target(@gfx1100) @caller(%lhs: reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.func.call @semantic(%lhs) : (reg<amdgpu.vgpr x1>) -> "
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
                      loom_low_func_call_callee_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "semantic");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "function");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "low function");
}

TEST_F(LowVerifyTest, FuncCallRejectsCrossTargetCallee) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "test.record @gfx1200 {}\n"
      "low.func.decl target(@gfx1200) @extern_add(%lhs: "
      "reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>)\n"
      "low.func.def target(@gfx1100) @caller(%lhs: reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.func.call @extern_add(%lhs) : (reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "  low.return %sum : reg<amdgpu.vgpr x1>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 17);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.func.call");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "callee");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "callee target must match enclosing low entry target");
}

TEST_F(LowVerifyTest, FuncCallRejectsOperandCountMismatch) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs: "
      "reg<amdgpu.vgpr x1>, %rhs: reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "low.func.def target(@gfx1100) @caller(%lhs: reg<amdgpu.vgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.func.call @extern_add(%lhs) : (reg<amdgpu.vgpr x1>) -> "
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
}

TEST_F(LowVerifyTest, FuncCallRejectsOperandTypeMismatch) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @gfx1100 {}\n"
      "low.func.decl target(@gfx1100) @extern_add(%lhs: "
      "reg<amdgpu.vgpr x1>, %rhs: reg<amdgpu.sgpr x1>) -> "
      "(reg<amdgpu.vgpr x1>)\n"
      "low.func.def target(@gfx1100) @caller(%lhs: reg<amdgpu.vgpr x1>, "
      "%rhs: reg<amdgpu.vgpr x1>) -> (reg<amdgpu.vgpr x1>) {\n"
      "  %sum = low.func.call @extern_add(%lhs, %rhs) : "
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
  EXPECT_EQ(GetStringParam(*diagnostic, 4), "argument");
}

TEST_F(LowVerifyTest, InvokeRejectsPureImpureCallee) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.decl target(@vm_target) @extern_add(%lhs: reg<vm.i32>) -> "
      "(reg<vm.i32>)\n"
      "func.def @caller(%lhs: i32) -> (i32) {\n"
      "  %sum = low.invoke pure @extern_add(%lhs) : (i32) -> (i32)\n"
      "  func.return %sum : i32\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 15);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "extern_add");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "callee has no pure contract");
}

TEST_F(LowVerifyTest, StorageTrafficPassesWithOwnedStorage) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "test.record @vm_target {}\n"
      "low.func.def target(@vm_target) @slot_roundtrip(%input: reg<vm.i32>) "
      "-> (reg<vm.i32>) {\n"
      "  %storage = low.storage.reserve {byte_alignment = 4, byte_length = 4} "
      ": low.storage<scratch>\n"
      "  low.spill %input, %storage {offset = 0} : reg<vm.i32>, "
      "low.storage<scratch>\n"
      "  %reload = low.reload %storage {offset = 0} : low.storage<scratch> "
      "-> reg<vm.i32>\n"
      "  %addr = low.storage.address %storage : "
      "low.storage<scratch> -> reg<vm.ptr>\n"
      "  low.return %reload : reg<vm.i32>\n"
      "}\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, ResourceImportsPassForVmNativeAndHalAbiShapes) {
  DiagnosticCapture capture;
  std::string source = kVmTargetBundle;
  source +=
      "low.func.def target(@vm_target) abi(vm_module_function) @vm_resource() "
      "-> (reg<vm.i32>) {\n"
      "  %state = low.resource<vm_state> {index = 0, semantic_type = "
      "vm.state} : reg<vm.i32>\n"
      "  low.return %state : reg<vm.i32>\n"
      "}\n";
  source += kNativeTargetBundle;
  source +=
      "low.func.def target(@native_target) abi(object_function) "
      "@native_resource() -> "
      "(reg<native.ptr>) {\n"
      "  %ptr = low.resource<native_pointer> {index = 0, semantic_type = "
      "buffer} : reg<native.ptr>\n"
      "  low.return %ptr : reg<native.ptr>\n"
      "}\n";
  source += kHalTargetBundle;
  source +=
      "low.kernel.def target(@hal_target) workgroup_size(64, 1, 1) "
      "@hal_kernel() {\n"
      "  %binding = low.resource<hal_buffer_resource> {index = 0, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  low.return\n"
      "}\n";
  loom_verify_result_t result = VerifySource(source.c_str(), &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(LowVerifyTest, ResourceRejectsNegativeValidByteCount) {
  DiagnosticCapture capture;
  std::string source = kHalTargetBundle;
  source +=
      "low.func.def target(@hal_target) @hal_kernel() -> "
      "(reg<amdgpu.sgpr x4>) {\n"
      "  %binding = low.resource<hal_buffer_resource> {index = 0, "
      "semantic_type = hal.buffer, valid_byte_count = -1} : "
      "reg<amdgpu.sgpr x4>\n"
      "  low.return %binding : reg<amdgpu.sgpr x4>\n"
      "}\n";
  loom_verify_result_t result = VerifySource(source.c_str(), &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 17);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.resource");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_resource_valid_byte_count_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "valid_byte_count");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "valid_byte_count must be non-negative");
}

TEST_F(LowVerifyTest, ResourceRejectsInvalidCacheSwizzleStride) {
  DiagnosticCapture capture;
  std::string source = kHalTargetBundle;
  source +=
      "low.func.def target(@hal_target) @hal_kernel() -> "
      "(reg<amdgpu.sgpr x4>) {\n"
      "  %binding = low.resource<hal_buffer_resource> {index = 0, "
      "semantic_type = hal.buffer, cache_swizzle_stride = 16384} : "
      "reg<amdgpu.sgpr x4>\n"
      "  low.return %binding : reg<amdgpu.sgpr x4>\n"
      "}\n";
  loom_verify_result_t result = VerifySource(source.c_str(), &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 17);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.resource");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_resource_cache_swizzle_stride_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "cache_swizzle_stride");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "cache_swizzle_stride must fit a 14-bit byte stride");
}

TEST_F(LowVerifyTest, ResourceRejectsWrongExportAbi) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "target.profile @native_target preset(\"test-low\")\n"
      "low.func.def target(@native_target) abi(object_function) @vm_resource() "
      "-> (reg<vm.i32>) {\n"
      "  %state = low.resource<vm_state> {index = 0, semantic_type = "
      "vm.state} : reg<vm.i32>\n"
      "  low.return %state : reg<vm.i32>\n"
      "}\n",
      &capture);
  EXPECT_GT(result.error_count, 0u);

  const loom_error_def_t* error =
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 17);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(capture, error);
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, error, LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "low.resource");
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                      loom_low_resource_import_kind_ATTR_INDEX);
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "import_kind");
  EXPECT_EQ(GetStringParam(*diagnostic, 2),
            "VM resources require vm_module_function export ABI");
}

}  // namespace
}  // namespace loom
