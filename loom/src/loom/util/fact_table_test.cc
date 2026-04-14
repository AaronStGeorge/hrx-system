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

//===----------------------------------------------------------------------===//
// Extensions
//===----------------------------------------------------------------------===//

TEST_F(FactTableTest, UniformElementExtensionRoundTrips) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(42), &facts));

  EXPECT_NE(facts.extension_id, LOOM_VALUE_FACT_EXTENSION_ID_NONE);
  EXPECT_FALSE(loom_value_facts_is_unknown(facts));

  loom_value_fact_uniform_element_t uniform = {0};
  EXPECT_TRUE(
      loom_value_facts_query_uniform_element(&table.context, facts, &uniform));
  EXPECT_TRUE(loom_value_facts_is_exact(uniform.element));
  EXPECT_EQ(uniform.element.range_lo, 42);
}

TEST_F(FactTableTest, IdenticalExtensionsInternToSameId) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t lhs = loom_value_facts_unknown();
  loom_value_facts_t rhs = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(7), &lhs));
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(7), &rhs));

  EXPECT_EQ(lhs.extension_id, rhs.extension_id);
  EXPECT_TRUE(loom_value_facts_equal(lhs, rhs));
}

TEST_F(FactTableTest, DifferentExtensionsDoNotAlias) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t lhs = loom_value_facts_unknown();
  loom_value_facts_t rhs = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(7), &lhs));
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &table.context, loom_value_facts_exact_i64(8), &rhs));

  EXPECT_NE(lhs.extension_id, rhs.extension_id);
  EXPECT_FALSE(loom_value_facts_equal(lhs, rhs));
}

TEST_F(FactTableTest, SmallStaticLanesExtensionRoundTrips) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t lanes[] = {
      loom_value_facts_exact_i64(1),
      loom_value_facts_make(2, 4, 2),
      loom_value_facts_unknown(),
  };
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = IREE_ARRAYSIZE(lanes),
  };
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(&table.context,
                                                          lane_slice, &facts));

  loom_value_fact_small_static_lanes_t result = {};
  EXPECT_TRUE(loom_value_facts_query_small_static_lanes(&table.context, facts,
                                                        &result));
  EXPECT_EQ(result.count, IREE_ARRAYSIZE(lanes));
  EXPECT_NE(result.lanes, lanes);
  EXPECT_EQ(result.lanes[0].range_lo, 1);
  EXPECT_EQ(result.lanes[1].range_hi, 4);
  EXPECT_TRUE(loom_value_facts_is_unknown(result.lanes[2]));
}

TEST_F(FactTableTest, OversizedSmallStaticLanesDegradesToUnknown) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT + 1] = {};
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = IREE_ARRAYSIZE(lanes),
  };
  loom_value_facts_t facts = loom_value_facts_exact_i64(123);
  IREE_ASSERT_OK(loom_value_facts_make_small_static_lanes(&table.context,
                                                          lane_slice, &facts));

  EXPECT_TRUE(loom_value_facts_is_unknown(facts));
  EXPECT_FALSE(
      loom_value_facts_query_small_static_lanes(&table.context, facts, NULL));
}

TEST_F(FactTableTest, VectorIotaExtensionRoundTrips) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_fact_vector_iota_t iota = {
      .base = loom_value_facts_exact_i64(2),
      .step = loom_value_facts_exact_i64(3),
  };
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(
      loom_value_facts_make_vector_iota(&table.context, iota, &facts));

  EXPECT_FALSE(
      loom_value_facts_query_uniform_element(&table.context, facts, NULL));
  loom_value_fact_vector_iota_t result = {};
  EXPECT_TRUE(
      loom_value_facts_query_vector_iota(&table.context, facts, &result));
  EXPECT_EQ(result.base.range_lo, 2);
  EXPECT_EQ(result.step.range_lo, 3);
}

TEST_F(FactTableTest, VectorPrefixMaskExtensionRoundTrips) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  loom_value_fact_vector_prefix_mask_t mask = {
      .lower_bound = loom_value_facts_exact_i64(0),
      .upper_bound = loom_value_facts_make(0, 16, 1),
      .step = loom_value_facts_exact_i64(1),
  };
  loom_value_facts_t facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(
      loom_value_facts_make_vector_prefix_mask(&table.context, mask, &facts));

  loom_value_fact_vector_prefix_mask_t result = {};
  EXPECT_TRUE(loom_value_facts_query_vector_prefix_mask(&table.context, facts,
                                                        &result));
  EXPECT_EQ(result.lower_bound.range_lo, 0);
  EXPECT_EQ(result.upper_bound.range_hi, 16);
  EXPECT_EQ(result.step.range_lo, 1);
}

}  // namespace
}  // namespace loom
