// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/rewrite/rewriter.h"

namespace loom {
namespace {

class RewriterWorklistTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &module_pool_);
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &worklist_pool_);
    iree_arena_initialize(&worklist_pool_, &worklist_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("worklist"),
                                        &module_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_rewriter_initialize(&rewriter_, module_, &worklist_arena_);
    loom_builder_set_block(&rewriter_.builder, loom_module_block(module_));
  }

  void TearDown() override {
    loom_rewriter_deinitialize(&rewriter_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&worklist_arena_);
    iree_arena_block_pool_deinitialize(&worklist_pool_);
    iree_arena_block_pool_deinitialize(&module_pool_);
  }

  std::vector<loom_op_t*> BuildOperations(iree_host_size_t count) {
    std::vector<loom_op_t*> operations;
    operations.reserve(count);
    const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    for (iree_host_size_t i = 0; i < count; ++i) {
      loom_op_t* operation = nullptr;
      IREE_CHECK_OK(loom_test_constant_build(
          &rewriter_.builder, loom_attr_i64((int64_t)i), type,
          LOOM_LOCATION_UNKNOWN, &operation));
      operations.push_back(operation);
    }
    return operations;
  }

  // Stable storage for the fixture module and its operations.
  iree_arena_block_pool_t module_pool_;
  // Small blocks force a large worklist to exercise several pooled chunks.
  iree_arena_block_pool_t worklist_pool_;
  // Arena owning worklist chunks.
  iree_arena_allocator_t worklist_arena_;
  // Registered vocabulary for fixture operations.
  loom_context_t context_;
  // Module containing every queued operation.
  loom_module_t* module_ = nullptr;
  // Worklist owner under test.
  loom_rewriter_t rewriter_ = {};
};

TEST_F(RewriterWorklistTest, ChunksPreserveLifoOrderAcrossReuse) {
  constexpr iree_host_size_t kOperationCount = 1025;
  constexpr iree_host_size_t kPopCount = 700;
  const std::vector<loom_op_t*> operations = BuildOperations(kOperationCount);

  IREE_ASSERT_OK(loom_rewriter_enable_worklist(&rewriter_));
  for (loom_op_t* operation : operations) {
    IREE_ASSERT_OK(loom_rewriter_add_to_worklist(&rewriter_, operation));
  }

  std::vector<loom_op_t*> popped;
  popped.reserve(kPopCount);
  for (iree_host_size_t i = 0; i < kPopCount; ++i) {
    loom_op_t* operation = loom_rewriter_pop(&rewriter_);
    ASSERT_NE(operation, nullptr);
    EXPECT_EQ(operation, operations[kOperationCount - i - 1]);
    popped.push_back(operation);
  }
  for (loom_op_t* operation : popped) {
    IREE_ASSERT_OK(loom_rewriter_add_to_worklist(&rewriter_, operation));
  }

  const iree_host_size_t retained_count = kOperationCount - kPopCount;
  for (iree_host_size_t i = retained_count; i < kOperationCount; ++i) {
    EXPECT_EQ(loom_rewriter_pop(&rewriter_), operations[i]);
  }
  for (iree_host_size_t i = retained_count; i > 0; --i) {
    EXPECT_EQ(loom_rewriter_pop(&rewriter_), operations[i - 1]);
  }
  EXPECT_EQ(loom_rewriter_pop(&rewriter_), nullptr);

  iree_arena_block_pool_statistics_t statistics;
  iree_arena_block_pool_query_statistics(&worklist_pool_, &statistics);
  EXPECT_EQ(statistics.oversized_allocation_count, 0u);
}

TEST_F(RewriterWorklistTest, DeinitializeClearsPendingMembershipAcrossChunks) {
  const std::vector<loom_op_t*> operations = BuildOperations(1025);
  IREE_ASSERT_OK(loom_rewriter_enable_worklist(&rewriter_));
  for (loom_op_t* operation : operations) {
    IREE_ASSERT_OK(loom_rewriter_add_to_worklist(&rewriter_, operation));
  }
  EXPECT_EQ(loom_rewriter_pop(&rewriter_), operations.back());

  loom_rewriter_deinitialize(&rewriter_);
  for (loom_op_t* operation : operations) {
    EXPECT_FALSE(iree_any_bit_set(operation->flags, LOOM_OP_FLAG_ON_WORKLIST));
  }
  loom_rewriter_initialize(&rewriter_, module_, &worklist_arena_);
}

}  // namespace
}  // namespace loom
