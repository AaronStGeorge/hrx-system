// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/loop_execution.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/cfg_loop.h"

namespace loom {
namespace {

class LoopExecutionPathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(intervals_); ++i) {
      facts_by_block_[intervals_[i].header_index] = &facts_[i];
    }
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_low_loop_execution_path_index_t Build() {
    const loom_low_loop_execution_index_t facts = {
        .facts_by_block = facts_by_block_,
        .block_count = IREE_ARRAYSIZE(facts_by_block_),
    };
    loom_low_loop_execution_path_index_t index = {};
    IREE_CHECK_OK(loom_low_loop_execution_path_index_build(&forest_, &facts,
                                                           &arena_, &index));
    return index;
  }

  // Storage backing the execution index's scratch lifetime.
  iree_arena_block_pool_t block_pool_ = {};
  // Reclaimed after each independent arithmetic scenario.
  iree_arena_allocator_t arena_ = {};
  // Three nested loops followed by an unrelated sibling loop.
  loom_cfg_loop_interval_t intervals_[4] = {
      {
          .header_index = 1,
          .latch_index = 8,
          .entry_predecessor_index = 0,
          .parent_loop_index = LOOM_CFG_LOOP_NONE,
          .is_canonical = true,
      },
      {
          .header_index = 3,
          .latch_index = 7,
          .entry_predecessor_index = 2,
          .parent_loop_index = 0,
          .is_canonical = true,
      },
      {
          .header_index = 5,
          .latch_index = 6,
          .entry_predecessor_index = 4,
          .parent_loop_index = 1,
          .is_canonical = true,
      },
      {
          .header_index = 9,
          .latch_index = 10,
          .entry_predecessor_index = 0,
          .parent_loop_index = LOOM_CFG_LOOP_NONE,
          .is_canonical = true,
      },
  };
  // Current block order, including outside entry and exit blocks.
  uint32_t innermost_loops_[12] = {
      LOOM_CFG_LOOP_NONE, 0, 0, 1, 1, 2, 2, 1, 0, 3, 3, LOOM_CFG_LOOP_NONE,
  };
  // Immutable structural input paired with the execution-count block order.
  loom_cfg_loop_forest_t forest_ = {
      .intervals = intervals_,
      .innermost_loop_indices = innermost_loops_,
      .interval_count = IREE_ARRAYSIZE(intervals_),
      .reachable_backward_edge_count = IREE_ARRAYSIZE(intervals_),
  };
  // Body counts; exact and bounded counts share the same upper-bound
  // arithmetic.
  loom_low_loop_execution_facts_t facts_[4] = {
      {.trip_count = 2, .count_kind = LOOM_LOW_LOOP_EXECUTION_COUNT_BOUNDED},
      {.trip_count = 3, .count_kind = LOOM_LOW_LOOP_EXECUTION_COUNT_EXACT},
      {.trip_count = 5, .count_kind = LOOM_LOW_LOOP_EXECUTION_COUNT_EXACT},
      {.trip_count = 11, .count_kind = LOOM_LOW_LOOP_EXECUTION_COUNT_BOUNDED},
  };
  // Null entries denote blocks without a retained loop-header count.
  const loom_low_loop_execution_facts_t* facts_by_block_[12] = {};
};

TEST_F(LoopExecutionPathTest, MultipliesOnlyCrossedLoopCounts) {
  const auto index = Build();
  uint64_t multiplier = 0;
  ASSERT_TRUE(
      loom_low_loop_execution_path_multiplier(&index, 0, 6, &multiplier));
  EXPECT_EQ(multiplier, 30u);
  ASSERT_TRUE(
      loom_low_loop_execution_path_multiplier(&index, 2, 6, &multiplier));
  EXPECT_EQ(multiplier, 15u);
  ASSERT_TRUE(
      loom_low_loop_execution_path_multiplier(&index, 4, 6, &multiplier));
  EXPECT_EQ(multiplier, 5u);
  EXPECT_FALSE(
      loom_low_loop_execution_path_multiplier(&index, 2, 8, &multiplier));
  EXPECT_FALSE(
      loom_low_loop_execution_path_multiplier(&index, 6, 2, &multiplier));
  EXPECT_FALSE(
      loom_low_loop_execution_path_multiplier(&index, 2, 10, &multiplier));
  EXPECT_FALSE(
      loom_low_loop_execution_path_multiplier(&index, 0, 11, &multiplier));
}

TEST_F(LoopExecutionPathTest, AddsFinalHeaderVisitBeforeMultiplying) {
  const auto index = Build();
  uint64_t multiplier = 0;
  ASSERT_TRUE(
      loom_low_loop_execution_path_multiplier(&index, 0, 1, &multiplier));
  EXPECT_EQ(multiplier, 3u);
  ASSERT_TRUE(
      loom_low_loop_execution_path_multiplier(&index, 0, 3, &multiplier));
  EXPECT_EQ(multiplier, 8u);
  ASSERT_TRUE(
      loom_low_loop_execution_path_multiplier(&index, 0, 5, &multiplier));
  EXPECT_EQ(multiplier, 36u);
}

TEST_F(LoopExecutionPathTest, PreservesZeroUnknownAndOverflowArithmetic) {
  struct Case {
    // Whether the enclosing loop has a proven execution bound.
    bool has_outer_count;
    // Enclosing body execution count when present.
    uint64_t outer_count;
    // Nested body execution count.
    uint64_t inner_count;
    // Whether the complete crossed path has a representable bound.
    bool known;
    // Expected path product when known.
    uint64_t product;
  };
  const Case cases[] = {
      {false, 0, 4, false, 0},         {false, 0, 0, true, 0},
      {true, 2, 4, true, 8},           {true, 3, 4, true, 12},
      {true, UINT64_MAX, 0, true, 0},  {true, UINT64_MAX, 1, true, UINT64_MAX},
      {true, UINT64_MAX, 2, false, 0},
  };
  for (const auto& scenario : cases) {
    SCOPED_TRACE(::testing::Message()
                 << scenario.has_outer_count << ":" << scenario.outer_count
                 << " * " << scenario.inner_count);
    facts_by_block_[1] = scenario.has_outer_count ? &facts_[0] : nullptr;
    facts_[0].trip_count = scenario.outer_count;
    facts_[1].trip_count = scenario.inner_count;
    const auto index = Build();
    uint64_t multiplier = 0;
    EXPECT_EQ(
        loom_low_loop_execution_path_multiplier(&index, 0, 4, &multiplier),
        scenario.known);
    if (scenario.known) {
      EXPECT_EQ(multiplier, scenario.product);
    }
  }
}

TEST_F(LoopExecutionPathTest, UnknownCountsArePathLocal) {
  facts_by_block_[1] = nullptr;
  const auto index = Build();
  uint64_t multiplier = 0;
  EXPECT_FALSE(
      loom_low_loop_execution_path_multiplier(&index, 0, 6, &multiplier));
  ASSERT_TRUE(
      loom_low_loop_execution_path_multiplier(&index, 2, 6, &multiplier));
  EXPECT_EQ(multiplier, 15u);
  ASSERT_TRUE(
      loom_low_loop_execution_path_multiplier(&index, 0, 10, &multiplier));
  EXPECT_EQ(multiplier, 11u);
}

TEST_F(LoopExecutionPathTest, ZeroAbsorbsOverflowButNotNoncanonicalStructure) {
  facts_[0].trip_count = 0;
  facts_[1].trip_count = UINT64_MAX;
  facts_[2].trip_count = 2;
  auto index = Build();
  uint64_t multiplier = 0;
  ASSERT_TRUE(
      loom_low_loop_execution_path_multiplier(&index, 0, 6, &multiplier));
  EXPECT_EQ(multiplier, 0u);
  EXPECT_FALSE(
      loom_low_loop_execution_path_multiplier(&index, 2, 6, &multiplier));
  ASSERT_TRUE(
      loom_low_loop_execution_path_multiplier(&index, 0, 3, &multiplier));
  EXPECT_EQ(multiplier, 0u);
  EXPECT_FALSE(
      loom_low_loop_execution_path_multiplier(&index, 2, 3, &multiplier));

  intervals_[1].is_canonical = false;
  index = Build();
  EXPECT_FALSE(
      loom_low_loop_execution_path_multiplier(&index, 0, 6, &multiplier));
  ASSERT_TRUE(
      loom_low_loop_execution_path_multiplier(&index, 4, 6, &multiplier));
  EXPECT_EQ(multiplier, 2u);
}

TEST_F(LoopExecutionPathTest, EmptyForestNeedsNoStorage) {
  forest_ = {};
  const auto index = Build();
  EXPECT_EQ(index.level_count, 0u);
  EXPECT_EQ(index.depths, nullptr);
  EXPECT_EQ(index.ancestors, nullptr);
  EXPECT_EQ(index.products, nullptr);
  EXPECT_EQ(index.factor_flags, nullptr);
  uint64_t multiplier = 0;
  EXPECT_FALSE(
      loom_low_loop_execution_path_multiplier(&index, 0, 0, &multiplier));
}

}  // namespace
}  // namespace loom
