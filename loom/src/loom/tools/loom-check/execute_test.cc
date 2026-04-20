// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/execute.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/testing/context.h"
#include "loom/tools/loom-check/check.h"

namespace loom {
namespace {

iree_status_t RegisterTestContext(void* user_data, loom_context_t* context) {
  (void)user_data;
  return loom_testing_context_register_all_dialects(context);
}

iree_status_t InitializeTestLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_target_core_test_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

bool TestRequirementProviderMatches(
    const loom_check_requirement_provider_t* provider,
    iree_string_view_t requirement) {
  (void)provider;
  return iree_string_view_equal(requirement, IREE_SV("fake-target"));
}

iree_status_t TestRequirementProviderQuery(
    const loom_check_requirement_provider_t* provider,
    const loom_check_environment_t* environment, iree_string_view_t requirement,
    iree_allocator_t allocator) {
  (void)provider;
  (void)environment;
  (void)allocator;
  if (iree_string_view_equal(requirement, IREE_SV("fake-target"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "fake target is intentionally unavailable");
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown fake requirement '%.*s'",
                          (int)requirement.size, requirement.data);
}

iree_status_t TestRequirementProviderAppendNames(
    const loom_check_requirement_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "fake-target");
}

const loom_check_requirement_provider_t kTestRequirementProvider = {
    .name = IREE_SVL("test"),
    .match = TestRequirementProviderMatches,
    .query = TestRequirementProviderQuery,
    .append_names = TestRequirementProviderAppendNames,
};

const loom_check_requirement_provider_t* const kTestRequirementProviders[] = {
    &kTestRequirementProvider,
};

bool TestEmitProviderMatches(const loom_check_emit_provider_t* provider,
                             iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("fake-emit"));
}

iree_status_t TestEmitProviderExecute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  return iree_string_builder_append_cstring(&request->result->actual_output,
                                            "fake emit\n");
}

iree_status_t TestEmitProviderAppendNames(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "fake-emit");
}

const loom_check_emit_provider_t kTestEmitProvider = {
    .name = IREE_SVL("test"),
    .match = TestEmitProviderMatches,
    .check_requirements = nullptr,
    .execute = TestEmitProviderExecute,
    .append_names = TestEmitProviderAppendNames,
};

const loom_check_emit_provider_t* const kTestEmitProviders[] = {
    &kTestEmitProvider,
};

const loom_check_environment_t kExecuteTestEnvironment = {
    .register_context =
        {
            .fn = RegisterTestContext,
            .user_data = nullptr,
        },
    .initialize_low_descriptor_registry =
        {
            .fn = InitializeTestLowDescriptorRegistry,
            .user_data = nullptr,
        },
};

const loom_check_environment_t kExecuteTestProviderEnvironment = {
    .register_context =
        {
            .fn = RegisterTestContext,
            .user_data = nullptr,
        },
    .initialize_low_descriptor_registry =
        {
            .fn = InitializeTestLowDescriptorRegistry,
            .user_data = nullptr,
        },
    .emit_providers =
        {
            .providers = kTestEmitProviders,
            .provider_count = IREE_ARRAYSIZE(kTestEmitProviders),
        },
    .requirement_providers =
        {
            .providers = kTestRequirementProviders,
            .provider_count = IREE_ARRAYSIZE(kTestRequirementProviders),
        },
};

class ExecuteTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(
        loom_check_context_initialize(&kExecuteTestEnvironment, &context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Parses source into a check file, executes case[0], returns the result.
  // On success the caller must deinitialize the result. On error the
  // result is already cleaned up.
  iree_status_t ExecuteFirst(const char* source,
                             loom_check_result_t* out_result) {
    return ExecuteFirstWithEnvironment(source, &kExecuteTestEnvironment,
                                       out_result);
  }

  iree_status_t ExecuteFirstWithEnvironment(
      const char* source, const loom_check_environment_t* environment,
      loom_check_result_t* out_result) {
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);
    loom_check_file_t file = {0};
    iree_status_t status =
        loom_check_parse(iree_make_cstring_view(source), &arena, &file);

    loom_check_file_report_t report = {};
    if (iree_status_is_ok(status)) {
      status = loom_check_file_report_initialize(&file, &arena, &report);
    }

    bool result_initialized = false;
    if (iree_status_is_ok(status) && file.case_count == 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no test cases");
    }
    if (iree_status_is_ok(status)) {
      loom_check_result_initialize(iree_allocator_system(), out_result);
      result_initialized = true;
      status = loom_check_execute_case(&file.cases[0], 0, &report,
                                       iree_make_cstring_view("test.loom-test"),
                                       environment, &context_, &block_pool_,
                                       iree_allocator_system(), out_result);
    }
    iree_arena_deinitialize(&arena);
    if (!iree_status_is_ok(status) && result_initialized) {
      loom_check_result_deinitialize(out_result);
    }
    return status;
  }

  std::string DetailString(const loom_check_result_t& result) {
    return std::string(result.detail.buffer, result.detail.size);
  }

  std::string DiffJsonString(const loom_check_result_t& result) {
    return std::string(result.diff_hunk_json.buffer,
                       result.diff_hunk_json.size);
  }

  std::string ActualOutputString(const loom_check_result_t& result) {
    return std::string(result.actual_output.buffer, result.actual_output.size);
  }

  std::string AnnotationEditJsonString(const loom_check_result_t& result) {
    return std::string(result.annotation_edits.json.buffer,
                       result.annotation_edits.json.size);
  }

  std::string DiagnosticJsonString(const loom_check_result_t& result) {
    return std::string(result.diagnostic_json.buffer,
                       result.diagnostic_json.size);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

//===----------------------------------------------------------------------===//
// Roundtrip tests
//===----------------------------------------------------------------------===//

TEST_F(ExecuteTest, RoundtripIdentity) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: roundtrip\n"
                   "func.def @f() {\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, RoundtripWithExpectedSection) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: roundtrip\n"
                   "func.def @f() {\n"
                   "}\n"
                   "// ----\n"
                   "func.def @f() {\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, RoundtripMismatch) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: roundtrip\n"
                   "func.def @f() {\n"
                   "}\n"
                   "// ----\n"
                   "func.def @different() {\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  EXPECT_FALSE(DetailString(result).empty()) << "expected diff output";
  EXPECT_GT(result.diff_hunk_count, 0u);
  EXPECT_NE(DiffJsonString(result).find("\"kind\": \"delete\""),
            std::string::npos);
  EXPECT_NE(DiffJsonString(result).find("\"kind\": \"insert\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, RoundtripParseError) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: roundtrip\n"
                   "%bogus\n",
                   &result));
  // Parse errors are content failures (FAIL), not infrastructure errors.
  // Diagnostics are collected in the detail string.
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_FALSE(DetailString(result).empty()) << "expected parse diagnostics";
  EXPECT_GT(result.diagnostic_count, 0u);
  EXPECT_NE(DiagnosticJsonString(result).find("\"error_id\":\"ERR_PARSE_"),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, RoundtripXfailFail) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: roundtrip\n"
                   "// XFAIL: testing\n"
                   "func.def @f() {\n"
                   "}\n"
                   "// ----\n"
                   "func.def @different() {\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, RoundtripXfailUnexpectedPass) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: roundtrip\n"
                   "// XFAIL: should fail but doesn't\n"
                   "func.def @f() {\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_PASS);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, RoundtripCommentsStripped) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: roundtrip\n"
                   "// This is a comment.\n"
                   "func.def @f() {\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, RoundtripActualOutputPopulated) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: roundtrip\n"
                   "func.def @f() {\n"
                   "}\n",
                   &result));
  EXPECT_FALSE(ActualOutputString(result).empty());
  loom_check_result_deinitialize(&result);
}

//===----------------------------------------------------------------------===//
// Verify tests
//===----------------------------------------------------------------------===//

TEST_F(ExecuteTest, VerifyCleanIR) {
  // Valid IR with no annotations — no diagnostics expected.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "func.def @f() {\n"
                   "  func.return\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS);
  loom_check_result_deinitialize(&result);
}

static const char kVerifyTestLowPreset[] =
    "target.preset @test_target {key = \"test-low\", source = @constant}\n";

TEST_F(ExecuteTest, VerifyRunsLowDescriptorVerifier) {
  loom_check_result_t result;
  std::string source =
      std::string("// RUN: verify\n") + kVerifyTestLowPreset +
      "low.func.def target(@test_target) @constant() -> (reg<test.i32>) {\n"
      "  %c0 = low.const<test.const.i32> : reg<test.i32>\n"
      "  low.return %c0 : reg<test.i32>\n"
      "}\n";
  IREE_ASSERT_OK(ExecuteFirst(source.c_str(), &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL)
      << "detail: " << DetailString(result);
  EXPECT_EQ(result.diagnostic_count, 1u);
  const std::string diagnostic_json = DiagnosticJsonString(result);
  EXPECT_NE(diagnostic_json.find("\"domain\":\"LOWERING\""), std::string::npos);
  EXPECT_NE(diagnostic_json.find("\"error_id\":\"ERR_LOWERING_007\""),
            std::string::npos);
  EXPECT_NE(diagnostic_json.find("\"emitter\":\"verifier\""),
            std::string::npos);
  EXPECT_NE(diagnostic_json.find("\"opcode\":\"test.const.i32\""),
            std::string::npos);
  EXPECT_NE(diagnostic_json.find("\"immediate_name\":\"i32_value\""),
            std::string::npos);
  EXPECT_NE(diagnostic_json.find("\"param_fields\""), std::string::npos);
  EXPECT_NE(diagnostic_json.find("\"kind\":\"attribute\""), std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyMatchedParseError) {
  // Parse error (unknown op) matched by an annotation. The annotation
  // is on input line 1, @+1 targets input line 2. After comment
  // stripping, line 2 is still the op line.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// ERROR@+1: PARSE/006\n"
                   "bogus.nonexistent\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyUnmatchedAnnotation) {
  // Annotation with no corresponding diagnostic — should fail.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// ERROR@+1: PARSE/006\n"
                   "func.def @f() {\n"
                   "  func.return\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  EXPECT_TRUE(DetailString(result).find("unmatched annotation") !=
              std::string::npos)
      << "detail: " << DetailString(result);
  EXPECT_EQ(result.annotation_edits.count, 1u);
  EXPECT_NE(AnnotationEditJsonString(result).find(
                "\"kind\": \"delete_diagnostic_annotation\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyAnnotationDeleteEditConsumesCrlf) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\r\n"
                   "// ERROR@+1: PARSE/006\r\n"
                   "func.def @f() {\r\n"
                   "  func.return\r\n"
                   "}\r\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL)
      << "detail: " << DetailString(result);
  EXPECT_EQ(result.annotation_edits.count, 1u);
  EXPECT_NE(AnnotationEditJsonString(result).find(
                "\"range\": {\"start_byte\": 16, \"end_byte\": 40}"),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyUnexpectedDiagnostic) {
  // Parse error with no annotation — should fail.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "bogus.nonexistent\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  EXPECT_TRUE(DetailString(result).find("unexpected") != std::string::npos)
      << "detail: " << DetailString(result);
  EXPECT_EQ(result.diagnostic_count, 1u);
  EXPECT_NE(DiagnosticJsonString(result).find("\"domain\":\"PARSE\""),
            std::string::npos);
  EXPECT_NE(DiagnosticJsonString(result).find("\"error_id\":\"ERR_PARSE_006\""),
            std::string::npos);
  EXPECT_EQ(result.annotation_edits.count, 1u);
  EXPECT_NE(AnnotationEditJsonString(result).find(
                "\"kind\": \"insert_diagnostic_annotations\""),
            std::string::npos);
  EXPECT_NE(AnnotationEditJsonString(result).find(
                "\"text\": \"// ERROR@+1: PARSE/006\\n\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyWildcardDomain) {
  // Annotation with no domain (wildcard) matches any domain.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// ERROR@+1:\n"
                   "bogus.nonexistent\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyWildcardCode) {
  // Annotation with domain but code 0 (wildcard) matches any code.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// ERROR@+1: PARSE\n"
                   "bogus.nonexistent\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifySubstringMatch) {
  // Annotation with a substring that must appear in the message.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// ERROR@+1: PARSE/006 \"bogus.nonexistent\"\n"
                   "bogus.nonexistent\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifySubstringMismatch) {
  // Annotation with a substring that does NOT appear in the message.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// ERROR@+1: PARSE/006 \"totally_wrong\"\n"
                   "bogus.nonexistent\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyXfailMatchedIsPass) {
  // XFAIL + all diagnostics matched → raw PASS → XFAIL inverts to FAIL
  // (unexpected pass).
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// XFAIL: testing\n"
                   "// ERROR@+1: PARSE/006\n"
                   "bogus.nonexistent\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_PASS);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyXfailUnmatchedIsPass) {
  // XFAIL + unmatched annotation → raw FAIL → XFAIL inverts to PASS.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// XFAIL: testing\n"
                   "// ERROR@+1: PARSE/006\n"
                   "func.def @f() {\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyNegativeOffset) {
  // Annotation with @-1 targets the line above.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "func.def @f() {\n"
                   "  bogus.nonexistent\n"
                   "  // ERROR@-1: PARSE/006\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyMultipleAnnotations) {
  // Two annotations on different lines matching two separate diagnostics.
  // Uses top-level bogus ops (not inside a function) because the parser's
  // error recovery sync within indented blocks consumes subsequent ops.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// ERROR@+1: PARSE/006\n"
                   "bogus.first\n"
                   "// ERROR@+1: PARSE/006\n"
                   "bogus.second\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifySeverityMismatch) {
  // Annotation says WARNING but the diagnostic is ERROR → FAIL.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// WARNING@+1: PARSE/006\n"
                   "bogus.nonexistent\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyDomainMismatch) {
  // Annotation says TYPE but the diagnostic is PARSE → FAIL.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// ERROR@+1: TYPE/003\n"
                   "bogus.nonexistent\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyCodeMismatch) {
  // Annotation says PARSE/001 but the diagnostic is PARSE/006 → FAIL.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// ERROR@+1: PARSE/001\n"
                   "bogus.nonexistent\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyLineMismatch) {
  // Annotation targets the wrong line → unmatched annotation + unexpected
  // diagnostic.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "func.def @f() {\n"
                   "  // ERROR@+2: PARSE/006\n"
                   "  bogus.nonexistent\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL)
      << "detail: " << DetailString(result);
  EXPECT_EQ(result.annotation_edits.count, 2u);
  EXPECT_NE(AnnotationEditJsonString(result).find(
                "\"kind\": \"delete_diagnostic_annotation\""),
            std::string::npos);
  EXPECT_NE(AnnotationEditJsonString(result).find(
                "\"kind\": \"insert_diagnostic_annotations\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyTypeError) {
  // Verifier-emitted TYPE errors: test.addi requires INTEGER operands and
  // result but receives f32. The parser accepts this (it doesn't check type
  // constraints), the verifier catches all three named fields:
  //   TYPE/003 for lhs, TYPE/003 for rhs, TYPE/004 for result.
  // All three annotations target the same line (the op), so each needs a
  // different offset to reach it from its own position.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "func.def @f(%a: f32, %b: f32) -> (f32) {\n"
                   "  // ERROR@+3: TYPE/003 \"lhs\"\n"
                   "  // ERROR@+2: TYPE/003 \"rhs\"\n"
                   "  // ERROR@+1: TYPE/004 \"result\"\n"
                   "  %r = test.addi %a, %b : f32\n"
                   "  func.return %r : f32\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyTypeErrorAnnotationEditOffsets) {
  // Three unexpected diagnostics on one target line should be grouped into one
  // insertion edit whose generated offsets all still point at the op line after
  // the edit is applied.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "func.def @f(%a: f32, %b: f32) -> (f32) {\n"
                   "  %r = test.addi %a, %b : f32\n"
                   "  func.return %r : f32\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL)
      << "detail: " << DetailString(result);
  EXPECT_EQ(result.annotation_edits.count, 1u);
  std::string edits = AnnotationEditJsonString(result);
  EXPECT_NE(edits.find("\"text\": \"  // ERROR@+3: TYPE/003\\n  // ERROR@+2: "
                       "TYPE/003\\n  // ERROR@+1: TYPE/004\\n\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyTypeErrorWithSubstring) {
  // Type constraint violations with substring matching on the constraint
  // name. All three annotations use "integer" substring to match the
  // INTEGER constraint name in the diagnostic message.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "func.def @f(%a: f32, %b: f32) -> (f32) {\n"
                   "  // ERROR@+3: TYPE/003 \"integer\"\n"
                   "  // ERROR@+2: TYPE/003 \"integer\"\n"
                   "  // ERROR@+1: TYPE/004 \"integer\"\n"
                   "  %r = test.addi %a, %b : f32\n"
                   "  func.return %r : f32\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyTypeErrorWildcardDomain) {
  // Wildcard annotations (no domain) match verifier-emitted TYPE errors.
  // All three diagnostics matched by wildcards.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "func.def @f(%a: f32, %b: f32) -> (f32) {\n"
                   "  // ERROR@+3:\n"
                   "  // ERROR@+2:\n"
                   "  // ERROR@+1:\n"
                   "  %r = test.addi %a, %b : f32\n"
                   "  func.return %r : f32\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyUnmatchedAnnotationDetail) {
  // Verify the detail output format for unmatched annotations.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// ERROR@+1: PARSE/006\n"
                   "func.def @f() {\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  std::string detail = DetailString(result);
  EXPECT_NE(detail.find("unmatched annotation"), std::string::npos)
      << "detail: " << detail;
  EXPECT_NE(detail.find("PARSE/006"), std::string::npos)
      << "detail: " << detail;
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyUnexpectedDiagnosticDetail) {
  // Verify the detail output format for unexpected diagnostics.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "bogus.nonexistent\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  std::string detail = DetailString(result);
  EXPECT_NE(detail.find("unexpected"), std::string::npos)
      << "detail: " << detail;
  EXPECT_NE(detail.find("PARSE/006"), std::string::npos)
      << "detail: " << detail;
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, VerifyAnnotationOnLastLine) {
  // Annotation targeting an op on the last line of input.
  // Uses @+1 because the annotation line becomes blank after comment
  // stripping — the op is always on the next line.
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: verify\n"
                   "// ERROR@+1: PARSE/006\n"
                   "bogus.nonexistent\n"
                   "// ERROR@+1: PARSE/006\n"
                   "bogus.also_nonexistent\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

//===----------------------------------------------------------------------===//
// Stub tests
//===----------------------------------------------------------------------===//

TEST_F(ExecuteTest, RoundtripMapRegion) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: roundtrip\n"
                   "func.def @map_test(%tile: f32) -> (f32) {\n"
                   "  %r = test.map(%element = %tile : f32) {\n"
                   "    %negated = test.neg %element : f32\n"
                   "    test.yield %negated : f32\n"
                   "  } -> (f32)\n"
                   "  func.return %r : f32\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, PassModeDce) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: pass dce\n"
                   "func.def @f() {\n"
                   "  func.return\n"
                   "}\n"
                   "// ----\n"
                   "func.def @f() {\n"
                   "  func.return\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, PassModeVerifiesTransformedModule) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: pass dce\n"
                   "func.def @f(%a: f32, %b: f32) -> (f32) {\n"
                   "  %r = test.addi %a, %b : f32\n"
                   "  func.return %r : f32\n"
                   "}\n"
                   "// ----\n"
                   "func.def @f(%a: f32, %b: f32) -> (f32) {\n"
                   "  %r = test.addi %a, %b : f32\n"
                   "  func.return %r : f32\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  EXPECT_GT(result.diagnostic_count, 0u);
  EXPECT_NE(DiagnosticJsonString(result).find("\"emitter\":\"verifier\""),
            std::string::npos);
  EXPECT_NE(DetailString(result).find("TYPE/"), std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, PassModeCapturesPassDiagnostic) {
  loom_source_id_t unrelated_source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(loom_context_register_source(
      &context_, IREE_SV("unrelated-before-parse.loom"), &unrelated_source_id));

  loom_check_result_t result;
  IREE_ASSERT_OK(ExecuteFirst(
      "// RUN: pass vector-memory-footprint\n"
      "func.def @f(%buffer: buffer, %base: offset) {\n"
      "  %layout = encoding.layout.dense : encoding<layout>\n"
      "  %view = buffer.view %buffer[%base] : buffer -> view<8xf32, %layout>\n"
      "  %loaded = vector.load %view[5] : view<8xf32, %layout> -> "
      "vector<4xf32>\n"
      "  func.return\n"
      "}\n",
      &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  EXPECT_GT(result.diagnostic_count, 0u);
  EXPECT_NE(DiagnosticJsonString(result).find("\"emitter\":\"pass\""),
            std::string::npos);
  EXPECT_NE(
      DiagnosticJsonString(result).find("\"error_id\":\"ERR_SUBRANGE_005\""),
      std::string::npos);
  EXPECT_NE(DiagnosticJsonString(result).find(
                "\"source_location\":{\"provenance\":\"exact_source\","
                "\"filename\":\"test.loom-test\""),
            std::string::npos);
  EXPECT_NE(DetailString(result).find("SUBRANGE/005"), std::string::npos);
  EXPECT_NE(DetailString(result).find("test.loom-test:"), std::string::npos);
  EXPECT_NE(DetailString(result).find("%loaded = vector.load"),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, PassModePassesOptionsToPassCreate) {
  loom_check_result_t result;
  IREE_ASSERT_OK(ExecuteFirst(
      "// RUN: pass refine-boundaries{max-iterations=1}\n"
      "func.def @identity(%value: index) -> (index) {\n"
      "  func.return %value : index\n"
      "}\n"
      "\n"
      "func.def public @caller() -> (index) {\n"
      "  %zero = index.constant 0 : index\n"
      "  %result = func.call @identity(%zero) : (index) -> (index)\n"
      "  func.return %result : index\n"
      "}\n",
      &result));
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  EXPECT_NE(DiagnosticJsonString(result).find("\"emitter\":\"pass\""),
            std::string::npos);
  EXPECT_NE(
      DiagnosticJsonString(result).find("\"error_id\":\"ERR_LOWERING_002\""),
      std::string::npos);
  EXPECT_NE(DiagnosticJsonString(result).find(
                "\"origin\":{\"provenance\":\"unavailable_source\""),
            std::string::npos);
  EXPECT_NE(DetailString(result).find("did not converge"), std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, PassModeRejectsOptionsForUnsupportedPass) {
  loom_check_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ExecuteFirst("// RUN: pass dce{max-iterations=1}\n"
                                     "func.def @f() {\n"
                                     "  func.return\n"
                                     "}\n",
                                     &result));
}

TEST_F(ExecuteTest, PassModeRejectsMalformedOptions) {
  loom_check_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        ExecuteFirst("// RUN: pass canonicalize{bogus=1}\n"
                                     "func.def @f() {\n"
                                     "  func.return\n"
                                     "}\n",
                                     &result));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      ExecuteFirst("// RUN: pass canonicalize{max-iterations=1\n"
                   "func.def @f() {\n"
                   "  func.return\n"
                   "}\n",
                   &result));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      ExecuteFirst(
          "// RUN: pass canonicalize{max-iterations=1,max-iterations=2}\n"
          "func.def @f() {\n"
          "  func.return\n"
          "}\n",
          &result));
}

TEST_F(ExecuteTest, RequiresUnavailableSkipsCaseBeforeParsingIr) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: roundtrip\n"
                   "// REQUIRES: loom-check-test-unavailable\n"
                   "this.is.not.valid.ir\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_SKIP);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_SKIP);
  EXPECT_NE(DetailString(result).find("loom-check-test-unavailable"),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, UnknownRequiresRequirementFailsLoudly) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: roundtrip\n"
                   "// REQUIRES: definitely-not-real\n"
                   "func.def @f() {}\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  EXPECT_NE(DetailString(result).find("unknown REQUIRES requirement"),
            std::string::npos);
  EXPECT_NE(DetailString(result).find("iree-run-loom"), std::string::npos);
  EXPECT_NE(DetailString(result).find("loom-check-test-unavailable"),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, RequirementProviderCanOwnAvailabilityQuery) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirstWithEnvironment("// RUN: roundtrip\n"
                                  "// REQUIRES: fake-target\n"
                                  "this.is.not.valid.ir\n",
                                  &kExecuteTestProviderEnvironment, &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_SKIP);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_SKIP);
  EXPECT_NE(DetailString(result).find("fake-target"), std::string::npos);
  EXPECT_NE(
      DetailString(result).find("fake target is intentionally unavailable"),
      std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, UnknownRequiresListsProviderRequirements) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirstWithEnvironment("// RUN: roundtrip\n"
                                  "// REQUIRES: definitely-not-real\n"
                                  "func.def @f() {}\n",
                                  &kExecuteTestProviderEnvironment, &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  EXPECT_NE(DetailString(result).find("unknown REQUIRES requirement"),
            std::string::npos);
  EXPECT_NE(DetailString(result).find("fake-target"), std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, EmitProviderCanOwnTarget) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirstWithEnvironment("// RUN: emit fake-emit\n"
                                  "func.def @f() {\n"
                                  "}\n"
                                  "// ----\n"
                                  "fake emit\n",
                                  &kExecuteTestProviderEnvironment, &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_PASS);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_PASS);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, UnknownEmitTargetListsProviderTargets) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirstWithEnvironment("// RUN: emit definitely-not-real\n"
                                  "this.is.not.valid.ir\n",
                                  &kExecuteTestProviderEnvironment, &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  EXPECT_NE(DetailString(result).find("unknown emit target"),
            std::string::npos);
  EXPECT_NE(DetailString(result).find("low-schedule-json"), std::string::npos);
  EXPECT_NE(DetailString(result).find("fake-emit"), std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, XfailDoesNotHideRequiresHarnessFailure) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: roundtrip\n"
                   "// XFAIL: this is a harness error, not an IR failure\n"
                   "// REQUIRES: definitely-not-real\n"
                   "func.def @f() {}\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, EmitLowDescriptorManifestProducesOutputForKnownSet) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: emit low-descriptor-manifest test.low.core\n"
                   "func.def @unused() {\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  const std::string actual_output = ActualOutputString(result);
  EXPECT_FALSE(actual_output.empty());
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, EmitLowDescriptorManifestReportsUnknownSet) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: emit low-descriptor-manifest definitely.missing\n"
                   "func.def @unused() {\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  EXPECT_NE(DetailString(result).find("NOT_FOUND"), std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, EmitTargetLowRegistryManifestUsesRegistryPackage) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: emit target-low-registry-manifest\n"
                   "func.def @unused() {\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  const std::string actual_output = ActualOutputString(result);
  EXPECT_FALSE(actual_output.empty());
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, EmitLowScheduleJsonAnchorsLiveInPreamble) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: emit low-schedule-json @livein\n"
                   "target.preset @test_target {key = \"test-low\", source = "
                   "@livein}\n"
                   "low.func.def target(@test_target) @livein() -> "
                   "(reg<test.i32>) {\n"
                   "  %arg0 = low.live_in<test.arg0> : reg<test.i32>\n"
                   "  %copy = low.copy %arg0 : reg<test.i32> -> "
                   "reg<test.i32>\n"
                   "  low.return %copy : reg<test.i32>\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  const std::string actual_output = ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"format\":\"loom.low.schedule.v0\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"op\":\"low.live_in\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"kind\":\"anchor\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"from\":0,\"to\":1,\"kind\":\"anchor\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"from\":0,\"to\":2,\"kind\":\"anchor\""),
            std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, EmitLivenessJsonReportsPressureSummary) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: emit liveness-json @pressure\n"
                   "func.def @pressure(%a: reg<test.i32>, "
                   "%b: reg<test.i32>, %c: reg<test.i32>) -> "
                   "(reg<test.i32>) {\n"
                   "  %ab = low.copy %a : reg<test.i32> -> reg<test.i32>\n"
                   "  %bc = low.copy %b : reg<test.i32> -> reg<test.i32>\n"
                   "  %cc = low.copy %c : reg<test.i32> -> reg<test.i32>\n"
                   "  func.return %ab : reg<test.i32>\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  const std::string actual_output = ActualOutputString(result);
  EXPECT_NE(actual_output.find("\"format\":\"loom.liveness.v0\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"intervals\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"pressure_summaries\""), std::string::npos);
  EXPECT_NE(actual_output.find("\"register_class\":\"test.i32\""),
            std::string::npos);
  EXPECT_NE(actual_output.find("\"peak_live_units\":3"), std::string::npos);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, EmitLivenessJsonReportsUnknownFunction) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: emit liveness-json @missing\n"
                   "func.def @present() {\n"
                   "  func.return\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  EXPECT_NE(DetailString(result).find("liveness-json"), std::string::npos);
  EXPECT_NE(DetailString(result).find("NOT_FOUND"), std::string::npos)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, RunModeRequiresDeclaredRunner) {
  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirst("// RUN: run --function=main\n"
                   "func.def @main() {\n"
                   "  func.return\n"
                   "}\n",
                   &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_FAIL);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_FAIL);
  EXPECT_NE(DetailString(result).find("REQUIRES: iree-run-loom"),
            std::string::npos)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, RunModeSkipsUnavailableRunner) {
  loom_check_environment_t environment = kExecuteTestEnvironment;
  environment.iree_run_loom_path =
      iree_make_cstring_view("/definitely/not/iree-run-loom");

  loom_check_result_t result;
  IREE_ASSERT_OK(
      ExecuteFirstWithEnvironment("// RUN: run --function=main\n"
                                  "// REQUIRES: iree-run-loom\n"
                                  "func.def @main() {\n"
                                  "  func.return\n"
                                  "}\n",
                                  &environment, &result));
  EXPECT_EQ(result.raw_outcome, LOOM_CHECK_SKIP);
  EXPECT_EQ(result.final_outcome, LOOM_CHECK_SKIP);
  EXPECT_NE(DetailString(result).find("iree-run-loom"), std::string::npos)
      << "detail: " << DetailString(result);
  loom_check_result_deinitialize(&result);
}

TEST_F(ExecuteTest, FormatModeUnimplemented) {
  loom_check_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        ExecuteFirst("// RUN: format bytecode\n"
                                     "func.def @f() {\n"
                                     "}\n",
                                     &result));
}

}  // namespace
}  // namespace loom
