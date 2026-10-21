// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_range.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class FactRangeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(sizeof(iree_arena_block_t),
                                     {this, Allocate}, &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<FactRangeTest*>(self);
    if (test->fail_allocations_ && command != IREE_ALLOCATOR_COMMAND_FREE) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected fact range allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }
  iree_status_t Grow(loom_value_id_t value_id) {
    return loom_value_fact_range_grow(&arena_, value_id, sizeof(*entries_),
                                      &empty_, &range_, (void**)&entries_);
  }
  uint32_t& Entry(loom_value_id_t value_id) {
    return entries_[loom_value_fact_range_offset(range_, value_id)];
  }
  // Each range growth reaches the fallible backing allocator.
  iree_arena_block_pool_t pool_ = {};
  // Retains old and current entry arrays until the test ends.
  iree_arena_allocator_t arena_ = {};
  // Typed entry range under test.
  loom_value_fact_range_t range_ = {};
  // Direct payload whose publications must survive growth in either direction.
  uint32_t* entries_ = nullptr;
  // Distinct absent-entry sentinel used by origin maps.
  const uint32_t empty_ = UINT32_MAX;
  // Rejects backing allocations while still permitting arena cleanup.
  bool fail_allocations_ = false;
};

TEST_F(FactRangeTest, OutOfOrderGrowthPreservesPublicationsAndFillsGaps) {
  constexpr loom_value_id_t first = 1u << 20;
  for (uint32_t offset : {32u, 35u, 18u, 91u, 0u}) {
    IREE_ASSERT_OK(Grow(first + offset));
    EXPECT_EQ(Entry(first + offset), empty_);
    Entry(first + offset) = offset;
  }
  for (uint32_t offset = 0; offset < 92; ++offset) {
    const bool published = offset == 32 || offset == 35 || offset == 18 ||
                           offset == 91 || offset == 0;
    EXPECT_EQ(Entry(first + offset), published ? offset : empty_);
  }
  EXPECT_LT(range_.capacity, 256u);
  EXPECT_FALSE(loom_value_fact_range_contains(range_, 0));
}

TEST_F(FactRangeTest, DescendingGrowthKeepsHeadroomAtTheGrowingEdge) {
  constexpr loom_value_id_t end = 1u << 20;
  for (uint32_t count = 1; count <= 64; ++count) {
    IREE_ASSERT_OK(Grow(end - count));
    ASSERT_LE(range_.capacity, 2u * count);
    Entry(end - count) = count;
  }
  for (uint32_t count = 1; count <= 64; ++count) {
    EXPECT_EQ(Entry(end - count), count);
  }
}

TEST_F(FactRangeTest, FailedGrowthPreservesRangeAndPublishedEntries) {
  constexpr loom_value_id_t value = 1u << 20;
  IREE_ASSERT_OK(Grow(value));
  Entry(value) = 42;
  const auto* entries = entries_;
  const auto range = range_;
  fail_allocations_ = true;
  IREE_ASSERT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Grow(value - 32));
  EXPECT_EQ(entries_, entries);
  EXPECT_EQ(range_.first_value_id, range.first_value_id);
  EXPECT_EQ(range_.capacity, range.capacity);
  EXPECT_EQ(Entry(value), 42u);
  fail_allocations_ = false;
  IREE_ASSERT_OK(Grow(value - 32));
  EXPECT_EQ(Entry(value), 42u);
  EXPECT_EQ(Entry(value - 32), empty_);
}

TEST_F(FactRangeTest, HighValueIdsDoNotWrapLookupOrGrowth) {
  constexpr loom_value_id_t value = UINT32_MAX - 2;
  IREE_ASSERT_OK(Grow(value));
  Entry(value) = 42;
  IREE_ASSERT_OK(Grow(value + 1));
  IREE_ASSERT_OK(Grow(value - 8));
  EXPECT_EQ(Entry(value), 42u);
  EXPECT_EQ(Entry(value + 1), empty_);
  EXPECT_EQ(Entry(value - 8), empty_);
  EXPECT_FALSE(loom_value_fact_range_contains(range_, 0));
  EXPECT_FALSE(loom_value_fact_range_contains(range_, value - 32));
}

}  // namespace
}  // namespace loom
