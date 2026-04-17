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
    DiagnosticCapture parse_capture;
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink = parse_capture.sink();
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("low_verify_test.loom"), &context_,
                                   &block_pool_, &parse_options, &module));
    if (!parse_capture.diagnostics.empty()) {
      ADD_FAILURE() << "expected parser success, got "
                    << parse_capture.diagnostics.size() << " diagnostic(s)";
      if (module) loom_module_free(module);
      return {};
    }
    EXPECT_NE(module, nullptr);
    if (!module) return {};

    verify_capture->Reset();
    loom_verify_options_t verify_options = {};
    verify_options.sink = verify_capture->sink();
    verify_options.max_errors = 20;
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module, &verify_options, &result));
    loom_module_free(module);
    return result;
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
