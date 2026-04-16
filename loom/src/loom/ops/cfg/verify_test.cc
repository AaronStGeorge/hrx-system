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
#include "loom/ir/types.h"
#include "loom/ops/op_registry.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::ExpectError;
using ::loom::testing::ExpectFieldRefParam;
using ::loom::testing::ExpectTypeParam;
using ::loom::testing::ExpectU32Param;
using ::loom::testing::FindDiagnostic;

class CfgVerifyTest : public ::testing::Test {
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
                                   IREE_SV("cfg_verify_test.loom"), &context_,
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

TEST_F(CfgVerifyTest, BranchCanTerminateFunctionEntryBlock) {
  DiagnosticCapture capture;
  loom_verify_result_t result = VerifySource(
      "func.def @forward(%arg : i32) -> (i32) {\n"
      "  cfg.br ^exit(%arg : i32)\n"
      "^exit(%value : i32):\n"
      "  func.return %value : i32\n"
      "}\n",
      &capture);
  EXPECT_EQ(result.error_count, 0u);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(CfgVerifyTest, BranchArgumentCountMustMatchDestinationBlock) {
  DiagnosticCapture capture;
  VerifySource(
      "func.def @count_mismatch(%arg : i32) {\n"
      "^entry:\n"
      "  cfg.br ^exit\n"
      "^exit(%value : i32):\n"
      "  func.return\n"
      "}\n",
      &capture);

  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 25));
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic,
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 25),
              LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_SUCCESSOR, 0);
  ExpectU32Param(*diagnostic, 1, 0);
  ExpectU32Param(*diagnostic, 2, 0);
  ExpectU32Param(*diagnostic, 3, 1);
}

TEST_F(CfgVerifyTest, BranchArgumentTypesMustMatchDestinationBlock) {
  DiagnosticCapture capture;
  VerifySource(
      "func.def @type_mismatch(%arg : f32) {\n"
      "  cfg.br ^exit(%arg : f32)\n"
      "^exit(%value : i32):\n"
      "  func.return\n"
      "}\n",
      &capture);

  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 26));
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic,
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 26),
              LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_SUCCESSOR, 0);
  ExpectU32Param(*diagnostic, 1, 0);
  ExpectU32Param(*diagnostic, 2, 0);
  ExpectTypeParam(*diagnostic, 3, loom_type_scalar(LOOM_SCALAR_TYPE_F32));
  ExpectTypeParam(*diagnostic, 4, loom_type_scalar(LOOM_SCALAR_TYPE_I32));
}

TEST_F(CfgVerifyTest, ConditionalBranchRequiresArgumentFreeDestinations) {
  DiagnosticCapture capture;
  VerifySource(
      "func.def @cond_payload(%condition : i1) {\n"
      "  cfg.cond_br %condition, ^then, ^else : i1\n"
      "^then:\n"
      "  func.return\n"
      "^else(%value : i32):\n"
      "  func.return\n"
      "}\n",
      &capture);

  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 25));
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic,
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 25),
              LOOM_EMITTER_VERIFIER);
  ExpectFieldRefParam(*diagnostic, 1, LOOM_DIAGNOSTIC_FIELD_SUCCESSOR, 1);
  ExpectU32Param(*diagnostic, 1, 1);
  ExpectU32Param(*diagnostic, 2, 0);
  ExpectU32Param(*diagnostic, 3, 1);
}

TEST_F(CfgVerifyTest, CfgRegionRequiresExplicitTerminators) {
  DiagnosticCapture capture;
  VerifySource(
      "func.def @missing_terminator(%condition : i1) {\n"
      "  cfg.cond_br %condition, ^then, ^else : i1\n"
      "^then:\n"
      "  func.return\n"
      "^else:\n"
      "}\n",
      &capture);

  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 5));
  ASSERT_NE(diagnostic, nullptr);
  ExpectError(*diagnostic,
              loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 5),
              LOOM_EMITTER_VERIFIER);
}

}  // namespace
}  // namespace loom
