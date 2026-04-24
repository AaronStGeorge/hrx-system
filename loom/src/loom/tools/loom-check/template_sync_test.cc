// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/template_sync.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/op_registry.h"
#include "loom/tools/loom-check/check.h"

namespace loom {
namespace {

class TemplateSyncTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t Build(const char* target_source, const char* template_source,
                      std::string* out_source, bool* out_changed) {
    iree_arena_allocator_t arena;
    iree_arena_initialize(&block_pool_, &arena);

    loom_check_file_t target_file = {};
    iree_status_t status = loom_check_parse(
        iree_make_cstring_view(target_source), &arena, &target_file);

    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    if (iree_status_is_ok(status)) {
      status = loom_check_template_sync_build_source(
          iree_make_cstring_view(target_source), &target_file,
          IREE_SV("target.loom-test"), iree_make_cstring_view(template_source),
          IREE_SV("template.loom-test"), &context_, &block_pool_, &arena,
          iree_allocator_system(), &builder, out_changed);
    }
    if (iree_status_is_ok(status)) {
      *out_source = std::string(iree_string_builder_buffer(&builder),
                                iree_string_builder_size(&builder));
    }

    iree_string_builder_deinitialize(&builder);
    iree_arena_deinitialize(&arena);
    return status;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(TemplateSyncTest, UsesTemplateCasesAndTargetPreamble) {
  std::string result;
  bool changed = false;
  IREE_ASSERT_OK(Build(
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low target-preset=test-low output=module\n"
      "\n",
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "ignored expected\n"
      "\n"
      "// ====\n"
      "\n"
      "// RUN: verify\n"
      "func.def @beta() {\n"
      "}\n",
      &result, &changed));

  EXPECT_TRUE(changed);
  EXPECT_NE(result.find("// TEMPLATE: loom/src/loom/test/corpus/vector/"
                        "arithmetic.loom-test\n"),
            std::string::npos);
  EXPECT_NE(result.find("// RUN: emit source-low target-preset=test-low "
                        "output=module\n"),
            std::string::npos);
  EXPECT_NE(result.find("func.def @alpha()"), std::string::npos);
  EXPECT_NE(result.find("func.def @beta()"), std::string::npos);
  EXPECT_EQ(result.find("ignored expected"), std::string::npos);
  EXPECT_EQ(result.find("// RUN: verify"), std::string::npos);
}

TEST_F(TemplateSyncTest, PreservesMatchingTargetEvidenceAndDirectives) {
  std::string result;
  bool changed = false;
  IREE_ASSERT_OK(Build(
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low target-preset=test-low output=module\n"
      "\n"
      "// REQUIRES: fake-target\n"
      "// XFAIL: pending target support\n"
      "// ERROR@+1: \"unsupported\"\n"
      "func.def @alpha() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "old target evidence\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @stale() {\n"
      "}\n"
      "\n"
      "// ----\n"
      "stale target evidence\n",
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @beta() {\n"
      "}\n",
      &result, &changed));

  EXPECT_TRUE(changed);
  EXPECT_NE(result.find("// REQUIRES: fake-target\n"), std::string::npos);
  EXPECT_NE(result.find("// XFAIL: pending target support\n"),
            std::string::npos);
  EXPECT_NE(result.find("// ERROR@+1: \"unsupported\"\n"), std::string::npos);
  EXPECT_NE(result.find("old target evidence\n"), std::string::npos);
  EXPECT_NE(result.find("func.def @beta()"), std::string::npos);
  EXPECT_EQ(result.find("func.def @stale()"), std::string::npos);
  EXPECT_EQ(result.find("stale target evidence"), std::string::npos);
}

TEST_F(TemplateSyncTest, PreservesTargetAnnotationsAtAnchoredInputLines) {
  std::string result;
  bool changed = false;
  IREE_ASSERT_OK(Build(
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low target-preset=test-low output=module\n"
      "\n"
      "func.def @alpha(%value: i32) -> (i32) {\n"
      "  // ERROR@+1: \"unsupported\"\n"
      "  %zero = scalar.constant 0 : i32\n"
      "  %result = scalar.addi %value, %zero : i32\n"
      "  func.return %result : i32\n"
      "}\n",
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha(%value: i32) -> (i32) {\n"
      "  %zero = scalar.constant 0 : i32\n"
      "  %result = scalar.addi %value, %zero : i32\n"
      "  func.return %result : i32\n"
      "}\n",
      &result, &changed));

  EXPECT_FALSE(changed);
  EXPECT_NE(result.find("  // ERROR@+1: \"unsupported\"\n"
                        "  %zero = scalar.constant 0 : i32\n"),
            std::string::npos);
}

TEST_F(TemplateSyncTest, PreservesAnnotationLineOccurrence) {
  const char* target_source =
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low target-preset=test-low output=module\n"
      "\n"
      "func.def @alpha() {\n"
      "  func.return\n"
      "  // ERROR@+1: \"second return\"\n"
      "  func.return\n"
      "}\n";
  const char* template_source =
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "  func.return\n"
      "  func.return\n"
      "}\n";

  std::string result;
  bool changed = true;
  IREE_ASSERT_OK(Build(target_source, template_source, &result, &changed));

  EXPECT_FALSE(changed);
  EXPECT_EQ(result, target_source);
}

TEST_F(TemplateSyncTest, ReportsUnchangedWhenConcreteFileIsCurrent) {
  const char* target_source =
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low target-preset=test-low output=module\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n";
  const char* template_source =
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n";

  std::string result;
  bool changed = true;
  IREE_ASSERT_OK(Build(target_source, template_source, &result, &changed));

  EXPECT_FALSE(changed);
  EXPECT_EQ(result, target_source);
}

TEST_F(TemplateSyncTest, RejectsTargetCaseRunDirectives) {
  std::string result;
  bool changed = false;
  iree_status_t status = Build(
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low target-preset=test-low output=module\n"
      "\n"
      "// RUN: verify\n"
      "func.def @alpha() {\n"
      "}\n",
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n",
      &result, &changed);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(TemplateSyncTest, RejectsEmptyTemplate) {
  std::string result;
  bool changed = false;
  iree_status_t status = Build(
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low target-preset=test-low output=module\n"
      "\n",
      "// RUN: roundtrip\n", &result, &changed);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(TemplateSyncTest, RejectsDuplicateTemplateFunctions) {
  std::string result;
  bool changed = false;
  iree_status_t status = Build(
      "// TEMPLATE: loom/src/loom/test/corpus/vector/arithmetic.loom-test\n"
      "// RUN: emit source-low target-preset=test-low output=module\n"
      "\n",
      "// RUN: roundtrip\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n"
      "\n"
      "// ====\n"
      "\n"
      "func.def @alpha() {\n"
      "}\n",
      &result, &changed);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

}  // namespace
}  // namespace loom
