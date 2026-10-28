// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/spill_storage.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class LowSpillStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Reusable storage for each coloring attempt.
  iree_arena_block_pool_t pool_;
  // Scoped scratch and result storage, reset between independent attempts.
  iree_arena_allocator_t arena_;
};

TEST_F(LowSpillStorageTest, SeparatesLayoutsAndSelectsEarliestIncludedPlan) {
  const auto layout = [](loom_low_spill_slot_space_t space, uint32_t size,
                         uint32_t alignment) {
    return loom_low_allocation_spill_plan_t{
        .slot_space = space,
        .byte_size = size,
        .byte_alignment = alignment,
    };
  };
  const loom_low_allocation_spill_plan_t plans[] = {
      layout(LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE, 16, 16),
      layout(LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE, 8, 4),
      layout(LOOM_LOW_SPILL_SLOT_SPACE_STACK, 8, 4),
      layout(LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE, 8, 4),
      layout(LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE, 8, 8),
      layout(LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE, 8, 4),
      layout(LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE, 8, 4),
      layout(LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE, 8, 4),
  };
  // Eight i8 units and two i32 units have the same size but different natural
  // alignments. Plans 3 and 6 have no traffic. The shared group starts at plan
  // 5 in block order, but its reservation must be created by plan 1 in plan
  // order.
  loom_low_spill_storage_interval_t intervals[] = {
      {&plans[7], 7, 4, 4}, {&plans[0], 0, 10, 10}, {&plans[4], 4, 14, 14},
      {&plans[5], 5, 0, 0}, {&plans[1], 1, 2, 2},   {&plans[2], 2, 12, 12},
  };
  const uint32_t* representatives = nullptr;
  IREE_ASSERT_OK(loom_low_allocation_spill_storage_plan(
      intervals, IREE_ARRAYSIZE(intervals), IREE_ARRAYSIZE(plans), &arena_,
      &representatives));
  ASSERT_NE(representatives, nullptr);
  const uint32_t expected[] = {0, 1, 2, 3, 4, 1, 6, 1};
  for (size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    EXPECT_EQ(representatives[i], expected[i]) << "plan " << i;
  }
}

TEST_F(LowSpillStorageTest, KeepsFunctionLifetimeStorageExclusive) {
  const loom_low_allocation_spill_plan_t layout = {
      .slot_space = LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE,
      .byte_size = 4,
      .byte_alignment = 4,
  };
  loom_low_spill_storage_interval_t intervals[] = {
      {&layout, 0, 0, UINT16_MAX},
      {&layout, 1, 0, 0},
      {&layout, 2, 1, 2},
      {&layout, 3, 3, 4},
      {&layout, 4, UINT16_MAX, UINT16_MAX},
  };
  const uint32_t* representatives = nullptr;
  IREE_ASSERT_OK(loom_low_allocation_spill_storage_plan(
      intervals, IREE_ARRAYSIZE(intervals), IREE_ARRAYSIZE(intervals), &arena_,
      &representatives));
  ASSERT_NE(representatives, nullptr);
  const uint32_t expected[] = {0, 1, 1, 1, 1};
  for (size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    EXPECT_EQ(representatives[i], expected[i]) << "plan " << i;
  }
}

TEST_F(LowSpillStorageTest, MatchesPeakOverlapForAllSmallIntervalSets) {
  const loom_low_allocation_spill_plan_t layout = {
      .slot_space = LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE,
      .byte_size = 4,
      .byte_alignment = 4,
  };
  const uint16_t ranges[][2] = {
      {0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2},
  };
  // Four intervals chosen independently from the six ranges cover every
  // overlap, shared endpoint, and input ordering on three blocks.
  constexpr uint32_t kCaseCount = 6 * 6 * 6 * 6;
  for (uint32_t case_index = 0; case_index < kCaseCount; ++case_index) {
    SCOPED_TRACE(case_index);
    iree_arena_reset(&arena_);
    loom_low_spill_storage_interval_t intervals[4] = {};
    uint32_t remaining = case_index;
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(intervals); ++i) {
      const auto& range = ranges[remaining % IREE_ARRAYSIZE(ranges)];
      intervals[i] = {&layout, i, range[0], range[1]};
      remaining /= IREE_ARRAYSIZE(ranges);
    }
    const uint32_t* representatives = nullptr;
    IREE_ASSERT_OK(loom_low_allocation_spill_storage_plan(
        intervals, IREE_ARRAYSIZE(intervals), IREE_ARRAYSIZE(intervals),
        &arena_, &representatives));
    uint32_t slot_count = 0;
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(intervals); ++i) {
      const uint32_t representative = representatives ? representatives[i] : i;
      ASSERT_LE(representative, i);
      EXPECT_EQ(representatives ? representatives[representative] : i,
                representative);
      slot_count += representative == i;
    }
    for (size_t i = 0; i < IREE_ARRAYSIZE(intervals); ++i) {
      for (size_t j = i + 1; j < IREE_ARRAYSIZE(intervals); ++j) {
        if (intervals[i].first_block <= intervals[j].last_block &&
            intervals[j].first_block <= intervals[i].last_block) {
          const uint32_t lhs = intervals[i].plan_index;
          const uint32_t rhs = intervals[j].plan_index;
          EXPECT_NE(representatives ? representatives[lhs] : lhs,
                    representatives ? representatives[rhs] : rhs);
        }
      }
    }
    uint32_t peak_overlap = 0;
    for (uint16_t block = 0; block < 3; ++block) {
      uint32_t overlap = 0;
      for (const auto& interval : intervals) {
        overlap +=
            interval.first_block <= block && block <= interval.last_block;
      }
      peak_overlap = iree_max(peak_overlap, overlap);
    }
    EXPECT_EQ(slot_count, peak_overlap);
    EXPECT_EQ(representatives == nullptr,
              slot_count == IREE_ARRAYSIZE(intervals));
  }
}

}  // namespace
}  // namespace loom
