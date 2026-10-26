// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/test_diagnostic_loomc.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TestDiagnosticLoomCTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(TestDiagnosticLoomCTest, PreservesPublicCodeStorage) {
  std::string code = "SPIRV/PROFILE";
  const loomc_diagnostic_t source = {
      .severity = LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      .code = loomc_make_string_view(code.data(), code.size()),
  };

  loom_test_diagnostic_t diagnostic = {};
  IREE_ASSERT_OK(
      loom_test_diagnostic_materialize_loomc(&source, &arena_, &diagnostic));
  code.assign(code.size(), '?');

  EXPECT_TRUE(
      iree_string_view_equal(diagnostic.public_code, IREE_SV("SPIRV/PROFILE")));
  EXPECT_EQ(diagnostic.domain, LOOM_ERROR_DOMAIN_COUNT_);
  EXPECT_EQ(diagnostic.code, 0);
}

TEST_F(TestDiagnosticLoomCTest, RetainsGeneratedIdentity) {
  const loomc_diagnostic_t source = {
      .severity = LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      .code = loomc_make_cstring_view("TYPE/003"),
  };

  loom_test_diagnostic_t diagnostic = {};
  IREE_ASSERT_OK(
      loom_test_diagnostic_materialize_loomc(&source, &arena_, &diagnostic));

  EXPECT_TRUE(
      iree_string_view_equal(diagnostic.public_code, IREE_SV("TYPE/003")));
  EXPECT_EQ(diagnostic.domain, LOOM_ERROR_DOMAIN_TYPE);
  EXPECT_EQ(diagnostic.code, 3);
}

}  // namespace
