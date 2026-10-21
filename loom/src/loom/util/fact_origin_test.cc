// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_origin.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/fact_table.h"

namespace loom {
namespace {

class FactOriginTest : public ::testing::Test {
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

  // Backing pool shared by independently owned fact tables.
  iree_arena_block_pool_t block_pool_;
  // Persistent storage used for table entries and standalone clones.
  iree_arena_allocator_t arena_;
};

// Inspects the stored record independently of module/type validation in
// queries.
template <typename Origins>
static const auto& OriginEntry(const Origins& origins,
                               loom_value_id_t value_id) {
  return origins.entries[loom_value_fact_range_offset(origins.range, value_id)];
}

static loom_value_fact_uniform_scale_origin_t UniformScaleOrigin(
    loom_value_id_t source_value_id, loom_value_id_t scale_value_id) {
  loom_value_fact_uniform_scale_origin_t origin = {
      .source_value_id = source_value_id,
      .scale_value_id = scale_value_id,
  };
  return origin;
}

static loom_value_fact_exact_lane_origin_t ExactLaneOrigin(
    loom_value_id_t source_value_id, uint32_t source_lane_offset = 0,
    uint32_t source_lane_stride = 1) {
  return {
      .source_value_id = source_value_id,
      .source_lane_offset = source_lane_offset,
      .source_lane_stride = source_lane_stride,
  };
}

static loom_value_fact_contextual_query_origin_t ContextualQueryOrigin(
    loom_parameterized_attr_kind_t family_kind, loom_attribute_t key) {
  loom_value_fact_contextual_query_origin_t origin = {
      .family_kind = family_kind,
      .key = key,
  };
  return origin;
}

TEST_F(FactOriginTest, UniformElementOriginsClearOnlyTouchedEntries) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  IREE_ASSERT_OK(
      loom_value_fact_table_define_uniform_element_origin(&table, 5, 2, 1));
  IREE_ASSERT_OK(
      loom_value_fact_table_define_uniform_element_origin(&table, 8, 3, 3));
  IREE_ASSERT_OK(
      loom_value_fact_table_define_uniform_element_origin(&table, 8, 4, 2));

  EXPECT_TRUE(
      loom_value_fact_range_contains(table.uniform_element_origins.range, 8));
  EXPECT_EQ(table.uniform_element_origins.touched_count, 2u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 5).scalar_value_id, 2u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 5).exact_scalar_value_id,
            1u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 8).scalar_value_id, 4u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 8).exact_scalar_value_id,
            2u);

  loom_value_fact_uniform_element_origin_t* const entries =
      table.uniform_element_origins.entries;
  loom_value_id_t* const touched_values =
      table.uniform_element_origins.touched_values;
  loom_value_fact_table_clear_scope(&table);

  EXPECT_EQ(table.uniform_element_origins.entries, entries);
  EXPECT_EQ(table.uniform_element_origins.touched_values, touched_values);
  EXPECT_EQ(table.uniform_element_origins.touched_count, 0u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 5).scalar_value_id,
            LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 5).exact_scalar_value_id,
            LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 8).scalar_value_id,
            LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 8).exact_scalar_value_id,
            LOOM_VALUE_ID_INVALID);

  IREE_ASSERT_OK(
      loom_value_fact_table_define_uniform_element_origin(&table, 5, 7, 6));
  EXPECT_EQ(table.uniform_element_origins.touched_count, 1u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 5).scalar_value_id, 7u);
  EXPECT_EQ(OriginEntry(table.uniform_element_origins, 5).exact_scalar_value_id,
            6u);
}

TEST_F(FactOriginTest, UniformScaleOriginsClearOnlyTouchedEntries) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));

  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scale_origin(
      &table, 5,
      UniformScaleOrigin(/*source_value_id=*/2,
                         /*scale_value_id=*/3)));
  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scale_origin(
      &table, 8,
      UniformScaleOrigin(/*source_value_id=*/4,
                         /*scale_value_id=*/5)));
  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scale_origin(
      &table, 8,
      UniformScaleOrigin(/*source_value_id=*/6,
                         /*scale_value_id=*/7)));

  EXPECT_TRUE(
      loom_value_fact_range_contains(table.uniform_scale_origins.range, 8));
  EXPECT_EQ(table.uniform_scale_origins.touched_count, 2u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 5).source_value_id, 2u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 5).scale_value_id, 3u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 8).source_value_id, 6u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 8).scale_value_id, 7u);

  loom_value_fact_uniform_scale_origin_t* const entries =
      table.uniform_scale_origins.entries;
  loom_value_id_t* const touched_values =
      table.uniform_scale_origins.touched_values;
  loom_value_fact_table_clear_scope(&table);

  EXPECT_EQ(table.uniform_scale_origins.entries, entries);
  EXPECT_EQ(table.uniform_scale_origins.touched_values, touched_values);
  EXPECT_EQ(table.uniform_scale_origins.touched_count, 0u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 5).source_value_id,
            LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 5).scale_value_id,
            LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 8).source_value_id,
            LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 8).scale_value_id,
            LOOM_VALUE_ID_INVALID);

  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scale_origin(
      &table, 5,
      UniformScaleOrigin(/*source_value_id=*/9,
                         /*scale_value_id=*/10)));
  EXPECT_EQ(table.uniform_scale_origins.touched_count, 1u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 5).source_value_id, 9u);
  EXPECT_EQ(OriginEntry(table.uniform_scale_origins, 5).scale_value_id, 10u);
}

TEST_F(FactOriginTest, ExactLaneOriginsAreSparseAndScopeLocal) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 1024));

  EXPECT_EQ(table.exact_lane_origins.entries, nullptr);
  EXPECT_EQ(table.exact_lane_origins.count, 0u);
  IREE_ASSERT_OK(loom_value_fact_table_define_exact_lane_origin(
      &table, 900,
      ExactLaneOrigin(/*source_value_id=*/12,
                      /*source_lane_offset=*/3,
                      /*source_lane_stride=*/2)));
  IREE_ASSERT_OK(loom_value_fact_table_define_exact_lane_origin(
      &table, 900, ExactLaneOrigin(/*source_value_id=*/20)));
  IREE_ASSERT_OK(loom_value_fact_table_define_exact_lane_origin(
      &table, 7, ExactLaneOrigin(/*source_value_id=*/3)));

  EXPECT_EQ(table.exact_lane_origins.count, 2u);
  EXPECT_LT(table.exact_lane_origins.capacity, (iree_host_size_t)900);

  loom_value_fact_exact_lane_origin_entry_t* const entries =
      table.exact_lane_origins.entries;
  const iree_host_size_t capacity = table.exact_lane_origins.capacity;
  loom_value_fact_table_clear_scope(&table);

  EXPECT_EQ(table.exact_lane_origins.entries, entries);
  EXPECT_EQ(table.exact_lane_origins.capacity, capacity);
  EXPECT_EQ(table.exact_lane_origins.count, 0u);

  IREE_ASSERT_OK(loom_value_fact_table_define_exact_lane_origin(
      &table, 5, ExactLaneOrigin(/*source_value_id=*/2)));
  EXPECT_EQ(table.exact_lane_origins.entries, entries);
  EXPECT_EQ(table.exact_lane_origins.count, 1u);
}

TEST_F(FactOriginTest, ContextualQueryOriginsAreSparseAndScopeLocal) {
  loom_value_fact_table_t table = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 1024));

  EXPECT_EQ(table.contextual_query_origins.entries, nullptr);
  EXPECT_EQ(table.contextual_query_origins.origins, nullptr);
  const loom_parameterized_attr_kind_t family_kind =
      LOOM_PARAMETERIZED_ATTR_KIND(/*dialect_id=*/3, /*family_index=*/7);
  IREE_ASSERT_OK(loom_value_fact_table_define_contextual_query_origin(
      &table, 900, ContextualQueryOrigin(family_kind, loom_attr_enum(4))));

  ASSERT_TRUE(loom_value_fact_range_contains(
      table.contextual_query_origins.range, 900));
  EXPECT_EQ(table.contextual_query_origins.touched_count, 1u);
  EXPECT_EQ(table.contextual_query_origins.origin_count, 1u);
  const uint32_t origin_id = OriginEntry(table.contextual_query_origins, 900);
  ASSERT_NE(origin_id, 0u);
  const loom_value_fact_contextual_query_origin_t& origin =
      table.contextual_query_origins.origins[origin_id - 1];
  EXPECT_EQ(origin.family_kind, family_kind);
  EXPECT_EQ(loom_attr_as_enum(origin.key), 4u);

  uint32_t* const entries = table.contextual_query_origins.entries;
  loom_value_fact_contextual_query_origin_t* const origins =
      table.contextual_query_origins.origins;
  loom_value_fact_table_clear_scope(&table);

  EXPECT_EQ(table.contextual_query_origins.entries, entries);
  EXPECT_EQ(table.contextual_query_origins.origins, origins);
  EXPECT_EQ(OriginEntry(table.contextual_query_origins, 900), 0u);
  EXPECT_EQ(table.contextual_query_origins.touched_count, 0u);
  EXPECT_EQ(table.contextual_query_origins.origin_count, 0u);

  IREE_ASSERT_OK(loom_value_fact_table_define_contextual_query_origin(
      &table, 5, ContextualQueryOrigin(family_kind, loom_attr_enum(9))));
  EXPECT_EQ(table.contextual_query_origins.touched_count, 1u);
  EXPECT_EQ(table.contextual_query_origins.origin_count, 1u);
  const uint32_t reused_origin_id =
      OriginEntry(table.contextual_query_origins, 5);
  ASSERT_NE(reused_origin_id, 0u);
  EXPECT_EQ(
      loom_attr_as_enum(
          table.contextual_query_origins.origins[reused_origin_id - 1].key),
      9u);
}

TEST_F(FactOriginTest, OriginRangesPreserveClonesAndReuseClearedStorage) {
  constexpr loom_value_id_t first = 1u << 16;
  const loom_value_id_t values[] = {first + 32, first + 7, first + 40, first};
  const auto family =
      LOOM_PARAMETERIZED_ATTR_KIND(/*dialect_id=*/3, /*family_index=*/7);
  loom_value_fact_table_t table = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&table, &arena_, 0));
  for (loom_value_id_t value : values) {
    IREE_ASSERT_OK(loom_value_fact_table_define(&table, value,
                                                loom_value_facts_unknown()));
    IREE_ASSERT_OK(loom_value_fact_table_define_uniform_element_origin(
        &table, value, first + 1, first + 2));
    IREE_ASSERT_OK(loom_value_fact_table_define_static_lane_origin(
        &table, value, {first + 3, value - first, 2}));
    IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scale_origin(
        &table, value, UniformScaleOrigin(first + 4, first + 5)));
    IREE_ASSERT_OK(loom_value_fact_table_define_contextual_query_origin(
        &table, value,
        ContextualQueryOrigin(
            family, loom_attr_enum(static_cast<uint8_t>(value - first)))));
  }
  const auto* uniform_entries = table.uniform_element_origins.entries;
  const auto* lane_entries = table.static_lane_origins.entries;
  const auto* scale_entries = table.uniform_scale_origins.entries;
  const auto* query_entries = table.contextual_query_origins.entries;
  loom_value_fact_table_t clone = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&clone, &arena_, 0));
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &clone, {&table, values, IREE_ARRAYSIZE(values)}, nullptr));

  auto expect_origins = [&](const loom_value_fact_table_t& current) {
    for (auto range : {current.uniform_element_origins.range,
                       current.static_lane_origins.range,
                       current.uniform_scale_origins.range,
                       current.contextual_query_origins.range}) {
      EXPECT_LT(range.capacity, 128u);
    }
    for (loom_value_id_t value : values) {
      const auto uniform = OriginEntry(current.uniform_element_origins, value);
      EXPECT_EQ(uniform.scalar_value_id, first + 1);
      EXPECT_EQ(uniform.exact_scalar_value_id, first + 2);
      const auto lane = OriginEntry(current.static_lane_origins, value);
      EXPECT_EQ(lane.source_value_id, first + 3);
      EXPECT_EQ(lane.source_lane_offset, value - first);
      EXPECT_EQ(lane.source_lane_stride, 2u);
      const auto scale = OriginEntry(current.uniform_scale_origins, value);
      EXPECT_EQ(scale.source_value_id, first + 4);
      EXPECT_EQ(scale.scale_value_id, first + 5);
      const auto query = OriginEntry(current.contextual_query_origins, value);
      ASSERT_NE(query, 0u);
      const auto origin = current.contextual_query_origins.origins[query - 1];
      EXPECT_EQ(origin.family_kind, family);
      EXPECT_EQ(loom_attr_as_enum(origin.key), value - first);
    }
  };
  expect_origins(table);
  expect_origins(clone);
  loom_value_fact_table_clear_scope(&table);
  expect_origins(clone);
  for (loom_value_id_t value : values) {
    EXPECT_EQ(OriginEntry(table.uniform_element_origins, value).scalar_value_id,
              LOOM_VALUE_ID_INVALID);
    EXPECT_EQ(OriginEntry(table.static_lane_origins, value).source_value_id,
              LOOM_VALUE_ID_INVALID);
    EXPECT_EQ(OriginEntry(table.uniform_scale_origins, value).source_value_id,
              LOOM_VALUE_ID_INVALID);
    EXPECT_EQ(OriginEntry(table.contextual_query_origins, value), 0u);
  }
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &table, {&clone, values, IREE_ARRAYSIZE(values)}, nullptr));
  expect_origins(table);
  EXPECT_EQ(table.uniform_element_origins.entries, uniform_entries);
  EXPECT_EQ(table.static_lane_origins.entries, lane_entries);
  EXPECT_EQ(table.uniform_scale_origins.entries, scale_entries);
  EXPECT_EQ(table.contextual_query_origins.entries, query_entries);
}

TEST_F(FactOriginTest, CloneValuesCopiesUniformElementOrigins) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 8));

  loom_value_facts_t source_facts = loom_value_facts_unknown();
  IREE_ASSERT_OK(loom_value_facts_make_uniform_element(
      &source.context, loom_value_facts_exact_i64(42), &source_facts));
  IREE_ASSERT_OK(loom_value_fact_table_define(&source, 7, source_facts));
  IREE_ASSERT_OK(
      loom_value_fact_table_define_uniform_element_origin(&source, 7, 2, 1));

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(&block_pool_, &target_arena);
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &target_arena, 8));
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &target, {&source, source.touched_values, source.touched_count},
      nullptr));

  EXPECT_TRUE(loom_value_facts_query_uniform_element(
      &target.context, loom_value_fact_table_lookup(&target, 7), nullptr));
  ASSERT_TRUE(
      loom_value_fact_range_contains(target.uniform_element_origins.range, 7));
  EXPECT_EQ(OriginEntry(target.uniform_element_origins, 7).scalar_value_id, 2u);
  EXPECT_EQ(
      OriginEntry(target.uniform_element_origins, 7).exact_scalar_value_id, 1u);

  iree_arena_deinitialize(&target_arena);
}

TEST_F(FactOriginTest, CloneValuesCopiesUniformScaleOrigins) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 8));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&source, 7, loom_value_facts_unknown()));
  IREE_ASSERT_OK(loom_value_fact_table_define_uniform_scale_origin(
      &source, 7,
      UniformScaleOrigin(/*source_value_id=*/2,
                         /*scale_value_id=*/3)));

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(&block_pool_, &target_arena);
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &target_arena, 8));
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &target, {&source, source.touched_values, source.touched_count},
      nullptr));

  ASSERT_TRUE(
      loom_value_fact_range_contains(target.uniform_scale_origins.range, 7));
  EXPECT_EQ(OriginEntry(target.uniform_scale_origins, 7).source_value_id, 2u);
  EXPECT_EQ(OriginEntry(target.uniform_scale_origins, 7).scale_value_id, 3u);

  iree_arena_deinitialize(&target_arena);
}

TEST_F(FactOriginTest, CloneValuesCopiesExactLaneOrigins) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 8));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&source, 7, loom_value_facts_unknown()));
  IREE_ASSERT_OK(loom_value_fact_table_define_exact_lane_origin(
      &source, 7,
      ExactLaneOrigin(/*source_value_id=*/2,
                      /*source_lane_offset=*/1,
                      /*source_lane_stride=*/2)));

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(&block_pool_, &target_arena);
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &target_arena, 8));
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &target, {&source, source.touched_values, source.touched_count},
      nullptr));

  EXPECT_EQ(target.exact_lane_origins.count, 1u);
  EXPECT_LT(target.exact_lane_origins.capacity, (iree_host_size_t)8);

  iree_arena_deinitialize(&target_arena);
}

TEST_F(FactOriginTest, CloneValuesCopiesContextualQueryOrigins) {
  loom_value_fact_table_t source = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source, &arena_, 8));

  IREE_ASSERT_OK(
      loom_value_fact_table_define(&source, 7, loom_value_facts_unknown()));
  const loom_parameterized_attr_kind_t family_kind =
      LOOM_PARAMETERIZED_ATTR_KIND(/*dialect_id=*/3, /*family_index=*/7);
  IREE_ASSERT_OK(loom_value_fact_table_define_contextual_query_origin(
      &source, 7, ContextualQueryOrigin(family_kind, loom_attr_i64(32))));

  iree_arena_allocator_t target_arena;
  iree_arena_initialize(&block_pool_, &target_arena);
  loom_value_fact_table_t target = {0};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&target, &target_arena, 8));
  IREE_ASSERT_OK(loom_value_fact_table_clone_values(
      &target, {&source, source.touched_values, source.touched_count},
      nullptr));

  ASSERT_TRUE(
      loom_value_fact_range_contains(target.contextual_query_origins.range, 7));
  const uint32_t origin_id = OriginEntry(target.contextual_query_origins, 7);
  ASSERT_NE(origin_id, 0u);
  const loom_value_fact_contextual_query_origin_t& origin =
      target.contextual_query_origins.origins[origin_id - 1];
  EXPECT_EQ(origin.family_kind, family_kind);
  EXPECT_EQ(loom_attr_as_i64(origin.key), 32);

  iree_arena_deinitialize(&target_arena);
}

}  // namespace
}  // namespace loom
