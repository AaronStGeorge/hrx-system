// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/concat_affinity.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class LowConcatAffinityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  iree_status_t Build(std::vector<loom_low_placement_relation_t> relations,
                      const std::vector<loom_value_ordinal_t>& order,
                      loom_low_allocation_concat_affinity_t** out_affinity) {
    std::vector<loom_low_placement_relation_range_t> ranges(order.size());
    std::vector<uint32_t> indices(relations.size());
    uint32_t edge_count = 0;
    loom_low_placement_storage_flags_t storage_flags = 0;
    for (const auto& relation : relations) {
      ++ranges[relation.source_ordinal].count;
      edge_count += loom_low_placement_cause_is_edge(relation.cause);
      if (relation.cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
        storage_flags |= LOOM_LOW_PLACEMENT_STORAGE_FLAG_CONCAT;
      }
    }
    uint32_t position = 0;
    for (auto& range : ranges) {
      range.start = position;
      position += range.count;
    }
    std::vector<uint32_t> cursors(order.size());
    for (uint32_t i = 0; i < relations.size(); ++i) {
      const auto source = relations[i].source_ordinal;
      indices[ranges[source].start + cursors[source]++] = i;
    }
    const loom_low_placement_table_t placement = {
        .value_count = static_cast<loom_value_ordinal_t>(order.size()),
        .relations = relations.data(),
        .relation_count = relations.size(),
        .edge_relation_count = edge_count,
        .storage = {.flags = storage_flags},
        .relation_indices_by_source_ordinal = indices.data(),
        .ranges_by_source_ordinal = ranges.data(),
        .storage_value_order = order.data(),
        .storage_value_order_count =
            static_cast<loom_value_ordinal_t>(order.size()),
    };
    const loom_low_descriptor_set_t descriptor_set = {};
    const loom_liveness_analysis_t empty_liveness = {};
    const loom_low_allocation_unit_liveness_t empty_unit_liveness = {};
    return loom_low_allocation_concat_affinity_build(
        &descriptor_set, &placement, &empty_liveness, &empty_unit_liveness,
        &arena_, out_affinity);
  }

  static loom_low_placement_relation_t Concat(loom_value_ordinal_t source,
                                              loom_value_ordinal_t result,
                                              uint32_t source_offset,
                                              uint32_t result_offset,
                                              uint32_t count) {
    return {
        .result_ordinal = result,
        .source_ordinal = source,
        .result_unit_offset = result_offset,
        .source_unit_offset = source_offset,
        .unit_count = count,
        .cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT,
        .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED |
                 LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE,
    };
  }

  static loom_low_placement_relation_t Edge(loom_value_ordinal_t source,
                                            loom_value_ordinal_t result) {
    return {
        .result_ordinal = result,
        .source_ordinal = source,
        .unit_count = 16,
        .cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH,
        .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_PREFERRED |
                 LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE,
    };
  }

  // Reusable backing blocks for the allocation-local index.
  iree_arena_block_pool_t pool_ = {};
  // Decision-arena lifetime containing every index built by a test.
  iree_arena_allocator_t arena_ = {};
};

TEST_F(LowConcatAffinityTest, ComposesClippedSpansInRetainedOrder) {
  // Numeric ordinals deliberately disagree with the retained storage order.
  const auto leaf = Concat(0, 3, 3, 2, 4);
  const auto middle = Concat(3, 1, 2, 4, 4);
  const auto outer = Concat(1, 2, 5, 8, 2);
  loom_low_allocation_concat_affinity_t* affinity = nullptr;
  IREE_ASSERT_OK(
      Build({leaf, middle, outer, Edge(2, 4)}, {4, 2, 1, 3, 0}, &affinity));
  auto projected = leaf;
  ASSERT_TRUE(
      loom_low_allocation_concat_affinity_project(affinity, &projected));
  EXPECT_EQ(projected.source_ordinal, 0u);
  EXPECT_EQ(projected.result_ordinal, 2u);
  EXPECT_EQ(projected.source_unit_offset, 4u);
  EXPECT_EQ(projected.result_unit_offset, 8u);
  EXPECT_EQ(projected.unit_count, 2u);
  EXPECT_EQ(projected.flags, leaf.flags);
}

TEST_F(LowConcatAffinityTest, TerminatesAtNonAliasableOrCapturedParts) {
  const auto leaf = Concat(0, 1, 0, 0, 2);
  for (const bool captured : {false, true}) {
    SCOPED_TRACE(captured);
    auto parent = Concat(1, 2, 0, 4, 2);
    if (captured) {
      parent.flags |= LOOM_LOW_PLACEMENT_RELATION_FLAG_CAPTURED_PART;
    } else {
      parent.flags &= ~LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE;
    }
    loom_low_allocation_concat_affinity_t* affinity = nullptr;
    IREE_ASSERT_OK(Build({leaf, parent, Edge(2, 3)}, {3, 2, 1, 0}, &affinity));
    auto projected = leaf;
    EXPECT_FALSE(
        loom_low_allocation_concat_affinity_project(affinity, &projected));
    EXPECT_EQ(projected.result_ordinal, leaf.result_ordinal);
    EXPECT_EQ(projected.result_unit_offset, leaf.result_unit_offset);
  }
}

TEST_F(LowConcatAffinityTest, RejectsAmbiguousUsers) {
  const auto leaf = Concat(0, 1, 0, 0, 2);
  const auto parent = Concat(1, 2, 0, 4, 2);
  loom_low_allocation_concat_affinity_t* affinity = nullptr;
  IREE_ASSERT_OK(
      Build({leaf, parent, Concat(1, 3, 0, 0, 2)}, {3, 2, 1, 0}, &affinity));
  auto projected = leaf;
  EXPECT_FALSE(
      loom_low_allocation_concat_affinity_project(affinity, &projected));
  EXPECT_EQ(projected.result_ordinal, leaf.result_ordinal);
  EXPECT_EQ(projected.result_unit_offset, leaf.result_unit_offset);
  loom_low_placement_relation_t relation;
  EXPECT_FALSE(loom_low_allocation_concat_affinity_relation_for_source(
      affinity, parent.source_ordinal, &relation));
}

TEST_F(LowConcatAffinityTest, DisjointProjectionLeavesInputUnchanged) {
  const auto leaf = Concat(0, 1, 0, 0, 2);
  loom_low_allocation_concat_affinity_t* affinity = nullptr;
  IREE_ASSERT_OK(Build({leaf, Concat(1, 2, 2, 0, 2), Edge(2, 3)}, {3, 2, 1, 0},
                       &affinity));
  auto projected = leaf;
  EXPECT_FALSE(
      loom_low_allocation_concat_affinity_project(affinity, &projected));
  EXPECT_EQ(projected.result_ordinal, leaf.result_ordinal);
  EXPECT_EQ(projected.unit_count, leaf.unit_count);
}

TEST_F(LowConcatAffinityTest, RetainsDirectConcatWithoutTransfer) {
  const auto leaf = Concat(0, 1, 0, 0, 2);
  loom_low_allocation_concat_affinity_t* affinity = nullptr;
  IREE_ASSERT_OK(Build({leaf}, {1, 0}, &affinity));
  ASSERT_NE(affinity, nullptr);
  loom_low_placement_relation_t relation;
  ASSERT_TRUE(loom_low_allocation_concat_affinity_relation_for_source(
      affinity, leaf.source_ordinal, &relation));
  EXPECT_EQ(relation.source_ordinal, leaf.source_ordinal);
  EXPECT_EQ(relation.result_ordinal, leaf.result_ordinal);
  EXPECT_EQ(relation.source_unit_offset, leaf.source_unit_offset);
  EXPECT_EQ(relation.result_unit_offset, leaf.result_unit_offset);
  EXPECT_EQ(relation.unit_count, leaf.unit_count);
  EXPECT_TRUE(loom_low_placement_relation_can_alias(&relation));
  IREE_ASSERT_OK(Build({Edge(0, 1)}, {1, 0}, &affinity));
  EXPECT_EQ(affinity, nullptr);
}

TEST_F(LowConcatAffinityTest, DeepChainsUsePooledSegments) {
  constexpr uint32_t kConcatCount = 16 * 1024 + 1;
  std::vector<loom_low_placement_relation_t> relations;
  std::vector<loom_value_ordinal_t> order;
  for (uint32_t i = 0; i < kConcatCount; ++i) {
    relations.push_back(Concat(i, i + 1, 0, 1, i + 1));
  }
  auto edge = Edge(kConcatCount, kConcatCount + 1);
  edge.unit_count = kConcatCount + 1;
  relations.push_back(edge);
  for (uint32_t i = kConcatCount + 2; i > 0; --i) {
    order.push_back(i - 1);
  }
  loom_low_allocation_concat_affinity_t* affinity = nullptr;
  IREE_ASSERT_OK(Build(relations, order, &affinity));
  for (uint32_t i = 0; i + 1 < kConcatCount; ++i) {
    auto projected = relations[i];
    ASSERT_TRUE(
        loom_low_allocation_concat_affinity_project(affinity, &projected));
    EXPECT_EQ(projected.result_ordinal, kConcatCount);
    EXPECT_EQ(projected.result_unit_offset, kConcatCount - i);
    EXPECT_EQ(projected.unit_count, i + 1);
  }
  iree_arena_block_pool_statistics_t statistics = {};
  iree_arena_block_pool_query_statistics(&pool_, &statistics);
  EXPECT_EQ(statistics.oversized_allocation_count, 0u);
}

}  // namespace
}  // namespace loom
