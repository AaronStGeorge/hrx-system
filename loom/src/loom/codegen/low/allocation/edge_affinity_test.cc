// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/edge_affinity.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class LowEdgeAffinityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_low_edge_affinity_list_t Build(
      loom_low_edge_affinity_input_t* inputs, size_t input_count,
      loom_low_edge_affinity_constraint_t* constraints,
      size_t constraint_count) {
    loom_low_edge_affinity_list_t affinities = {};
    IREE_EXPECT_OK(loom_low_edge_affinity_build(inputs, input_count,
                                                constraints, constraint_count,
                                                16, &arena_, &affinities));
    return affinities;
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(LowEdgeAffinityTest, CountsDistinctLogicalEdges) {
  loom_op_t ops[2] = {};
  loom_low_edge_affinity_input_t inputs[] = {
      {10, 1, 0, 5, &ops[0]}, {11, 2, 0, 6, &ops[0]},  {10, 1, 0, 7, &ops[0]},
      {11, 2, 0, 8, &ops[0]}, {10, 1, 1, 10, &ops[1]}, {11, 2, 1, 11, &ops[1]},
  };
  loom_low_edge_affinity_constraint_t constraints[] = {
      {10, 11},
      {11, 10},
  };

  const auto affinities = Build(inputs, IREE_ARRAYSIZE(inputs), constraints,
                                IREE_ARRAYSIZE(constraints));

  ASSERT_EQ(affinities.count, 1u);
  EXPECT_EQ(affinities.values[0].first, 1u);
  EXPECT_EQ(affinities.values[0].second, 2u);
  EXPECT_EQ(affinities.values[0].priority, 2u);
  EXPECT_EQ(affinities.values[0].op, &ops[0]);
}

TEST_F(LowEdgeAffinityTest, DoesNotPairAcrossLogicalEdges) {
  loom_low_edge_affinity_input_t inputs[] = {
      {10, 1, 0, 0, nullptr},
      {11, 2, 1, 1, nullptr},
  };
  loom_low_edge_affinity_constraint_t constraints[] = {{10, 11}};

  const auto affinities = Build(inputs, IREE_ARRAYSIZE(inputs), constraints,
                                IREE_ARRAYSIZE(constraints));

  EXPECT_EQ(affinities.count, 0u);
}

TEST_F(LowEdgeAffinityTest, OmitsAlreadyHardSourcePairs) {
  loom_low_edge_affinity_input_t inputs[] = {
      {10, 1, 0, 0, nullptr},
      {11, 2, 0, 1, nullptr},
  };
  loom_low_edge_affinity_constraint_t constraints[] = {
      {10, 11},
      {1, 2},
  };

  const auto affinities = Build(inputs, IREE_ARRAYSIZE(inputs), constraints,
                                IREE_ARRAYSIZE(constraints));

  EXPECT_EQ(affinities.count, 0u);
}

}  // namespace
}  // namespace loom
