// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/setup_order.h"

#include <algorithm>
#include <initializer_list>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/schedule/context.h"

namespace loom {
namespace {

class SetupOrderTest : public ::testing::Test {
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

  void Initialize(std::initializer_list<uint32_t> block_sizes) {
    nodes_.clear();
    blocks_.clear();
    for (uint32_t size : block_sizes) {
      const uint32_t start = static_cast<uint32_t>(nodes_.size());
      nodes_.resize(start + size);
      for (uint32_t i = start; i < nodes_.size(); ++i) {
        nodes_[i].block_index = static_cast<uint32_t>(blocks_.size());
      }
      blocks_.push_back({.node_start = start, .node_count = size});
    }
    options_ = {};
    state_ = {
        .options = &options_,
        .scratch_arena = &arena_,
        .blocks = blocks_.data(),
        .nodes = nodes_.data(),
        .dependency_arena = &arena_,
    };
    loom_low_schedule_dependency_graph_initialize(&state_.dependencies);
    IREE_ASSERT_OK(loom_low_schedule_setup_order_initialize(
        static_cast<uint32_t>(nodes_.size()), &arena_, &state_.setup_order));
  }

  void Append(uint32_t producer, uint32_t consumer,
              loom_low_schedule_dependency_kind_t kind =
                  LOOM_LOW_SCHEDULE_DEPENDENCY_SSA) {
    const loom_low_schedule_dependency_t edge = {
        .producer_node = producer,
        .consumer_node = consumer,
        .producer_attachment_index = LOOM_LOW_ID_NONE,
        .consumer_attachment_index = LOOM_LOW_ID_NONE,
        .producer_event_id = LOOM_LOW_TIMING_EVENT_NONE,
        .consumer_event_id = LOOM_LOW_TIMING_EVENT_NONE,
        .value_operand_index = LOOM_LOW_ID_NONE,
        .kind = kind,
    };
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_append(
        &state_.dependencies, edge, &arena_));
    loom_low_schedule_setup_order_record_dependency(
        &state_.setup_order, producer, consumer,
        kind == LOOM_LOW_SCHEDULE_DEPENDENCY_SSA);
  }

  void ExpectAddedOrder(
      std::initializer_list<std::pair<uint32_t, uint32_t>> expected) {
    const auto original_count = state_.dependencies.count;
    IREE_ASSERT_OK(loom_low_schedule_setup_order_finish(
        &state_, static_cast<uint32_t>(nodes_.size()),
        static_cast<uint16_t>(blocks_.size())));
    std::vector<std::pair<uint32_t, uint32_t>> actual;
    for (uint32_t i = original_count; i < state_.dependencies.count; ++i) {
      const auto* edge =
          loom_low_schedule_dependency_graph_at(&state_.dependencies, i);
      EXPECT_EQ(edge->kind, LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER);
      actual.emplace_back(edge->producer_node, edge->consumer_node);
    }
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(actual, (std::vector<std::pair<uint32_t, uint32_t>>(expected)));
  }

  // Backing storage for the native graph and temporary closure proof.
  iree_arena_block_pool_t pool_;
  // Graph construction and closure scratch share the invocation lifetime.
  iree_arena_allocator_t arena_;
  // Native source-order node metadata; no operations or module are needed.
  std::vector<loom_low_schedule_node_t> nodes_;
  // Disjoint source-order ranges covering every node exactly once.
  std::vector<loom_low_schedule_block_t> blocks_;
  // Production policy controlling implied-successor repair.
  loom_low_schedule_options_t options_ = {};
  // Native closure inputs and outputs, borrowing the arrays above.
  loom_low_schedule_build_state_t state_ = {};
};

TEST_F(SetupOrderTest, OrdersSharedCompletionAfterItsPrerequisites) {
  Initialize({4});
  nodes_[1].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
  nodes_[2].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
  Append(0, 3);
  Append(1, 3);
  Append(2, 3);

  // The prerequisite precedes both setup members, which retain source order.
  ExpectAddedOrder({{0, 1}, {1, 2}});
  EXPECT_EQ(state_.setup_order.completion_nodes[0], 0u);
  EXPECT_EQ(state_.setup_order.completion_nodes[1], 3u);
  EXPECT_EQ(state_.setup_order.completion_nodes[2], 3u);
  EXPECT_EQ(state_.setup_order.entry_nodes[3], 1u);
}

TEST_F(SetupOrderTest, NonValueSuccessorsDoNotCompleteSetup) {
  for (auto kind :
       {LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
        LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE,
        LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER}) {
    SCOPED_TRACE(kind);
    Initialize({3});
    nodes_[0].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
    Append(0, 2, kind);
    Append(1, 2);

    ExpectAddedOrder({});
    EXPECT_EQ(state_.setup_order.completion_nodes[0], 0u);
    EXPECT_EQ(state_.setup_order.entry_nodes[2], 2u);
  }
}

TEST_F(SetupOrderTest, DistinctSuccessorBreaksSetupRegardlessOfSourcePosition) {
  for (bool side_effect_first : {false, true}) {
    for (bool side_edge_first : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "side effect first=" << side_effect_first
                   << " side edge first=" << side_edge_first);
      Initialize({4});
      nodes_[0].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
      const uint32_t consumer = side_effect_first ? 3 : 2;
      const uint32_t prerequisite = side_effect_first ? 2 : 1;
      const uint32_t side_effect = side_effect_first ? 1 : 3;
      if (side_edge_first) {
        Append(0, side_effect, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE);
      }
      Append(0, consumer);
      Append(prerequisite, consumer);
      if (!side_edge_first) {
        Append(0, side_effect, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE);
      }

      ExpectAddedOrder({});
      EXPECT_EQ(state_.setup_order.completion_nodes[0], 0u);
    }
  }
}

TEST_F(SetupOrderTest, FanoutEndsOnlyTheEarlierPartOfASetupChain) {
  for (auto kind :
       {LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
        LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE}) {
    SCOPED_TRACE(kind);
    Initialize({5});
    nodes_[0].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
    nodes_[1].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
    Append(0, 1);
    Append(1, 4);
    Append(0, 2, kind);
    Append(2, 3, kind);
    Append(3, 4);

    ExpectAddedOrder({{3, 1}});
    EXPECT_EQ(state_.setup_order.completion_nodes[0], 0u);
    EXPECT_EQ(state_.setup_order.completion_nodes[1], 4u);
  }
}

TEST_F(SetupOrderTest, RepairRequiresASoleValueConsumerAndAnImpliedSideEdge) {
  for (bool repair : {false, true}) {
    for (bool implied : {false, true}) {
      for (auto side_kind : {LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
                             LOOM_LOW_SCHEDULE_DEPENDENCY_SSA}) {
        SCOPED_TRACE(::testing::Message()
                     << "repair=" << repair << " implied=" << implied
                     << " side kind=" << side_kind);
        Initialize({4});
        options_.flags =
            repair ? LOOM_LOW_SCHEDULE_FLAG_REPAIR_PRIVATE_SETUP : 0;
        nodes_[1].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
        Append(0, 2);
        Append(1, 2);
        Append(1, 3, side_kind);
        if (implied) {
          Append(2, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER);
        }

        if (repair && implied &&
            side_kind != LOOM_LOW_SCHEDULE_DEPENDENCY_SSA) {
          ExpectAddedOrder({{0, 1}});
        } else {
          ExpectAddedOrder({});
        }
      }
    }
  }
}

TEST_F(SetupOrderTest, SourceBoundaryDoesNotAcquireAnEarlierSetupChain) {
  Initialize({4});
  nodes_[1].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
  nodes_[2].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY;
  Append(0, 2);
  Append(1, 2);
  Append(2, 3, LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER);

  ExpectAddedOrder({});
  EXPECT_EQ(state_.setup_order.completion_nodes[1], 1u);
}

TEST_F(SetupOrderTest, ClosesIndependentChainsInEveryBlock) {
  Initialize({3, 3});
  nodes_[0].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
  nodes_[3].flags = LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
  Append(0, 2);
  Append(1, 2);
  Append(3, 5);
  Append(4, 5);

  ExpectAddedOrder({{1, 0}, {4, 3}});
  EXPECT_EQ(state_.setup_order.entry_nodes[2], 0u);
  EXPECT_EQ(state_.setup_order.entry_nodes[5], 3u);
}

}  // namespace
}  // namespace loom
