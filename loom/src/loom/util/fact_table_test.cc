// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_table.h"

#include <cstdint>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class FactTableTest : public ::testing::Test {
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

//===----------------------------------------------------------------------===//
// Initialize and lookup
//===----------------------------------------------------------------------===//

TEST_F(FactTableTest, ZeroInitIsValid) {
  loom_value_fact_table_t table = {0};
  // Lookup on empty table returns unknown.
  loom_value_facts_t facts = loom_value_fact_table_lookup(&table, 0);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts));
  facts = loom_value_fact_table_lookup(&table, 42);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts));
}

TEST_F(FactTableTest, InitializePreallocates) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 100));
  EXPECT_GE(table.capacity, (iree_host_size_t)100);
  // All entries are undefined (unknown).
  loom_value_facts_t facts = loom_value_fact_table_lookup(&table, 0);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts));
  facts = loom_value_fact_table_lookup(&table, 99);
  EXPECT_TRUE(loom_value_facts_is_unknown(facts));
}

//===----------------------------------------------------------------------===//
// Define and lookup
//===----------------------------------------------------------------------===//

TEST_F(FactTableTest, DefineAndLookup) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 100));

  loom_value_facts_t exact_42 = loom_value_facts_exact_i64(42);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 5, exact_42));

  loom_value_facts_t result = loom_value_fact_table_lookup(&table, 5);
  EXPECT_TRUE(loom_value_facts_is_exact(result));
  EXPECT_EQ(result.range_lo, 42);
  EXPECT_EQ(result.range_hi, 42);
}

TEST_F(FactTableTest, UndefinedEntriesAreUnknown) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 100));

  // Define entry 5, check that entry 3 is still unknown.
  loom_value_facts_t exact_42 = loom_value_facts_exact_i64(42);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 5, exact_42));

  loom_value_facts_t result = loom_value_fact_table_lookup(&table, 3);
  EXPECT_TRUE(loom_value_facts_is_unknown(result));
}

TEST_F(FactTableTest, DefineUpdatesExisting) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 100));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 5, loom_value_facts_exact_i64(42)));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 5, loom_value_facts_exact_i64(99)));

  loom_value_facts_t result = loom_value_fact_table_lookup(&table, 5);
  EXPECT_TRUE(loom_value_facts_is_exact(result));
  EXPECT_EQ(result.range_lo, 99);
}

//===----------------------------------------------------------------------===//
// Growth
//===----------------------------------------------------------------------===//

TEST_F(FactTableTest, LazyGrowth) {
  // Initialize with zero capacity — no entries pre-allocated.
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  EXPECT_EQ(table.capacity, (iree_host_size_t)0);

  // Define a value — should trigger allocation.
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 10, loom_value_facts_exact_i64(7)));

  EXPECT_GT(table.capacity, (iree_host_size_t)10);
  loom_value_facts_t result = loom_value_fact_table_lookup(&table, 10);
  EXPECT_TRUE(loom_value_facts_is_exact(result));
  EXPECT_EQ(result.range_lo, 7);
}

TEST_F(FactTableTest, GrowthPreservesExistingEntries) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 4));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 0, loom_value_facts_exact_i64(10)));
  IREE_ASSERT_OK(
      loom_value_fact_table_define(&table, 1, loom_value_facts_exact_i64(20)));

  // Define beyond capacity to trigger growth.
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 100,
                                              loom_value_facts_exact_i64(30)));

  // Original entries are preserved.
  loom_value_facts_t r0 = loom_value_fact_table_lookup(&table, 0);
  EXPECT_EQ(r0.range_lo, 10);
  loom_value_facts_t r1 = loom_value_fact_table_lookup(&table, 1);
  EXPECT_EQ(r1.range_lo, 20);
  loom_value_facts_t r100 = loom_value_fact_table_lookup(&table, 100);
  EXPECT_EQ(r100.range_lo, 30);
}

//===----------------------------------------------------------------------===//
// Range facts
//===----------------------------------------------------------------------===//

TEST_F(FactTableTest, RangeFacts) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 100));

  loom_value_facts_t range = loom_value_facts_make(0, 1024, 64);
  IREE_ASSERT_OK(loom_value_fact_table_define(&table, 7, range));

  loom_value_facts_t result = loom_value_fact_table_lookup(&table, 7);
  EXPECT_FALSE(loom_value_facts_is_exact(result));
  EXPECT_TRUE(loom_value_facts_is_non_negative(result));
  EXPECT_EQ(result.range_lo, 0);
  EXPECT_EQ(result.range_hi, 1024);
  EXPECT_EQ(result.known_divisor, 64);
  EXPECT_TRUE(loom_value_facts_divisible_by(result, 16));
}

//===----------------------------------------------------------------------===//
// Out-of-range lookup
//===----------------------------------------------------------------------===//

TEST_F(FactTableTest, OutOfRangeIsUnknown) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 10));

  // Value ID beyond count is unknown.
  loom_value_facts_t result = loom_value_fact_table_lookup(&table, 999);
  EXPECT_TRUE(loom_value_facts_is_unknown(result));
}

}  // namespace
}  // namespace loom
