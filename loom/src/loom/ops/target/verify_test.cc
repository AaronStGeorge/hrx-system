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
#include "loom/ops/target/ops.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::ExpectError;
using ::loom::testing::ExpectI64Param;
using ::loom::testing::FindDiagnostic;
using ::loom::testing::GetStringParam;

class TargetVerifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
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
    const char* filename = "target_verify_test.loom";
    loom_module_t* module = ParseSource(source, filename);
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

  loom_module_t* ParseSource(const char* source, const char* filename) {
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

  loom_op_t* FindFirstMutableOp(loom_module_t* module, loom_op_kind_t kind) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_module_block(module), op) {
      if (op->kind == kind) return op;
    }
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

static const char* kValidTargetRecords =
    "target.profile @vm preset(\"iree-vm\")\n"
    "target.profile @gfx11 preset(\"amdgpu-gfx11\") {target_cpu = "
    "\"gfx1100\", contract_feature_bits = 1}\n"
    "target.profile @wasm preset(\"wasm-simd128\")\n"
    "target.artifact @vm_module target(@vm) {artifact_format = vm_bytecode, "
    "abi = vm_module}\n"
    "target.artifact @gfx_hal target(@gfx11) {artifact_format = elf, "
    "abi = hal_executable}\n"
    "target.artifact @wasm_module target(@wasm) {artifact_format = "
    "wasm_binary, abi = wasm_module}\n";

TEST_F(TargetVerifyTest, ProfilesAndArtifactsVerify) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(kValidTargetRecords, &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(TargetVerifyTest, FutureTargetEnumOrdinalsVerifyAsOpenEnums) {
  loom_module_t* module =
      ParseSource(kValidTargetRecords, "target_verify_test.loom");
  ASSERT_NE(module, nullptr);
  loom_op_t* artifact = FindFirstMutableOp(module, LOOM_OP_TARGET_ARTIFACT);
  ASSERT_NE(artifact, nullptr);
  loom_op_attrs(artifact)[loom_target_artifact_artifact_format_ATTR_INDEX] =
      loom_attr_enum(250);
  loom_op_attrs(artifact)[loom_target_artifact_abi_ATTR_INDEX] =
      loom_attr_enum(251);

  DiagnosticCapture capture;
  loom_verify_options_t verify_options = {};
  verify_options.sink = capture.sink();
  verify_options.max_errors = 20;
  loom_verify_result_t result = {};
  IREE_ASSERT_OK(loom_verify_module(module, &verify_options, &result));

  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
  loom_module_free(module);
}

TEST_F(TargetVerifyTest, ProfileRejectsEmptyPresetKey) {
  DiagnosticCapture capture;
  VerifySource("target.profile @bad preset(\"\")\n", &capture);

  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "preset");
  ExpectI64Param(*diagnostic, 1, 0);
}

TEST_F(TargetVerifyTest, ArtifactRejectsWrongRecordClass) {
  loom_module_t* module = ParseSource(
      "target.profile @ok preset(\"test-low\")\n"
      "target.artifact @not_profile target(@ok)\n"
      "target.artifact @bad target(@ok)\n",
      "target_verify_test.loom");
  ASSERT_NE(module, nullptr);

  const loom_string_id_t not_profile_name =
      loom_module_lookup_string(module, IREE_SV("not_profile"));
  ASSERT_NE(not_profile_name, LOOM_STRING_ID_INVALID);
  const uint16_t not_profile_symbol =
      loom_module_find_symbol(module, not_profile_name);
  ASSERT_NE(not_profile_symbol, LOOM_SYMBOL_ID_INVALID);
  loom_op_t* artifact = FindFirstMutableOp(module, LOOM_OP_TARGET_ARTIFACT);
  ASSERT_NE(artifact, nullptr);
  loom_op_attrs(artifact)[loom_target_artifact_target_ATTR_INDEX] =
      loom_attr_symbol((loom_symbol_ref_t){
          .module_id = 0,
          .symbol_id = not_profile_symbol,
      });

  DiagnosticCapture capture;
  loom_verify_options_t verify_options = {};
  verify_options.sink = capture.sink();
  verify_options.max_errors = 20;
  loom_verify_result_t result = {};
  IREE_ASSERT_OK(loom_verify_module(module, &verify_options, &result));
  EXPECT_EQ(result.error_count, 1u);

  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3));
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3),
              LOOM_EMITTER_VERIFIER);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "not_profile");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "target artifact");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "target profile");

  loom_module_free(module);
}

TEST_F(TargetVerifyTest, RejectsDuplicateTargetRecordDefinition) {
  DiagnosticCapture capture;
  VerifySource(
      "target.profile @duplicate preset(\"first\")\n"
      "target.profile @duplicate preset(\"second\")\n",
      &capture);

  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 5));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "duplicate");
  ExpectError(*diagnostic, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 5),
              LOOM_EMITTER_VERIFIER);
  ASSERT_EQ(diagnostic->related_locations.size(), 1u);
  EXPECT_EQ(diagnostic->related_locations[0].label, "first definition here");
  EXPECT_TRUE(diagnostic->related_locations[0].has_source_range);
}

}  // namespace
}  // namespace loom
