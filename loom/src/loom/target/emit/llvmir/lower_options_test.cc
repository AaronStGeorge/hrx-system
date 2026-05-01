// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/target/emit/llvmir/lower.h"
#include "loom/target/emit/llvmir/test_target.h"

namespace loom {
namespace {

class LlvmIrLowerOptionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("lower_options"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = NULL;
};

TEST_F(LlvmIrLowerOptionsTest, LowersEmptyModule) {
  loom_llvmir_lowering_options_t options = {};
  options.target_profile = loom_llvmir_target_profile_test_object();

  loom_llvmir_module_t* lowered = NULL;
  IREE_ASSERT_OK(loom_llvmir_lower_module(module_, &options,
                                          iree_allocator_system(), &lowered));
  ASSERT_NE(lowered, nullptr);
  loom_llvmir_module_free(lowered);
}

}  // namespace
}  // namespace loom
