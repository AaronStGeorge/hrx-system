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

TEST(SetupMembershipTest, StructuralSetupRequiresPolicyOrRetainedRegister) {
  const loom_low_reg_class_t reg_class = {
      .alloc_unit_bits = 32,
      .allocatable_count = 8,
  };
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  loom_value_id_t value_ids[] = {7, 19};
  loom_local_value_domain_t domain = {
      .value_ids = value_ids,
      .value_count = IREE_ARRAYSIZE(value_ids),
  };
  const struct {
    // Allocation retained private placement for the result.
    bool retained;
    // Allocation requires the result to remain in a register.
    bool required;
    // The caller requested ordering for all structural setup.
    bool order_structural;
    // Whether this node belongs to an ordered setup chain.
    bool ordered;
  } cases[] = {
      {false, false, false, false}, {true, false, false, false},
      {false, true, false, false},  {true, true, false, true},
      {false, false, true, true},   {true, false, true, true},
      {false, true, true, true},    {true, true, true, true},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "retained=" << test_case.retained
                 << " required=" << test_case.required
                 << " structural=" << test_case.order_structural);
    uint64_t placement_word = 0;
    uint64_t required_word = 0;
    loom_low_schedule_options_t options = {
        .required_register_values = {64, &required_word},
        .per_user_placement_values = {64, &placement_word},
    };
    if (test_case.retained) {
      iree_bitmap_set(options.per_user_placement_values, value_ids[1]);
    }
    if (test_case.required) {
      iree_bitmap_set(options.required_register_values, value_ids[1]);
    }
    loom_low_schedule_value_record_t values[] = {
        {.unit_count = 1, .register_class_id = 0},
        {.unit_count = 1, .register_class_id = 0},
    };
    if (test_case.required) {
      values[1].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_REQUIRES_REGISTER;
    }
    loom_low_schedule_build_state_t state = {
        .options = &options,
        .target = {.descriptor_set = &descriptors},
        .value_domain = &domain,
        .values = values,
    };
    loom_low_schedule_node_t node = {
        .kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL,
        .operand_count = 1,
        .result_count = 1,
        .flags = LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP,
        .value_ordinals = {.inline_value_ordinals = {0, 1}},
    };
    loom_low_schedule_setup_order_classify_node(&state, &node,
                                                test_case.order_structural);
    EXPECT_EQ(
        iree_any_bit_set(node.flags, LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP),
        test_case.ordered);
    EXPECT_EQ(state.setup_order.has_members, test_case.ordered);
  }
}

TEST(SetupMembershipTest, SingletonRequiredStorageNeedsExclusionOrdering) {
  const loom_low_reg_class_t reg_class = {
      .alloc_unit_bits = 32,
      .allocatable_count = 8,
  };
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = &reg_class,
      .reg_class_count = 1,
  };
  const loom_low_schedule_options_t options = {};
  const struct {
    // Available physical units in the completion domain, or zero if unbounded.
    uint32_t capacity;
    // Physical units owned by the result.
    uint32_t result_units;
    // The result cannot spill or rematerialize.
    bool required;
    // The structural operation constructs storage for its result.
    bool storage_setup;
    // Whether the singleton storage requires exclusion ordering.
    bool ordered;
  } cases[] = {
      {1, 1, true, true, true},   {1, 1, false, true, false},
      {1, 1, true, false, false}, {2, 1, true, true, false},
      {2, 2, true, true, false},  {0, 1, true, true, false},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "capacity=" << test_case.capacity << " units="
                 << test_case.result_units << " required=" << test_case.required
                 << " storage setup=" << test_case.storage_setup);
    loom_low_schedule_completion_domain_t completion = {
        .capacity = test_case.capacity,
        .reg_class_id = 0,
    };
    uint16_t domain_id = test_case.capacity == 0 ? UINT16_MAX : 0;
    loom_low_schedule_value_record_t values[] = {
        {.unit_count = 1, .register_class_id = 0},
        {.unit_count = test_case.result_units, .register_class_id = 0},
    };
    if (test_case.required) {
      values[1].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_REQUIRES_REGISTER;
    }
    loom_low_schedule_build_state_t state = {
        .options = &options,
        .target = {.descriptor_set = &descriptors},
        .values = values,
        .pressure_limits =
            {
                .unspillable_completion_domains = &completion,
                .unspillable_completion_domain_ids_by_reg_class = &domain_id,
                .unspillable_completion_domain_count =
                    test_case.capacity == 0 ? uint16_t{0} : uint16_t{1},
            },
    };
    loom_low_schedule_node_t node = {
        .kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL,
        .operand_count = 1,
        .result_count = 1,
        .value_ordinals = {.inline_value_ordinals = {0, 1}},
    };
    if (test_case.storage_setup) {
      node.flags = LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP;
    }
    loom_low_schedule_setup_order_classify_node(
        &state, &node, /*order_structural_setup=*/false);
    EXPECT_EQ(
        iree_any_bit_set(node.flags, LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP),
        test_case.ordered);
    EXPECT_EQ(state.setup_order.has_members, test_case.ordered);
  }
}

TEST(SetupMembershipTest, CapturedRequiredStorageMustPreserveItsFootprint) {
  const loom_low_reg_class_t classes[] = {
      {.alloc_unit_bits = 32, .allocatable_count = 8, .alias_set_id = 1},
      {.alloc_unit_bits = 32, .allocatable_count = 8, .alias_set_id = 1},
      {.alloc_unit_bits = 32, .allocatable_count = 8},
      {.alloc_unit_bits = 32, .allocatable_count = 8, .alias_set_id = 2},
  };
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = classes,
      .reg_class_count = IREE_ARRAYSIZE(classes),
  };
  const loom_low_schedule_options_t options = {};
  const struct {
    // Human-readable invariant represented by this row.
    const char* name;
    // Units captured from the first operand in class zero.
    uint32_t first_units;
    // Units captured from a second operand, or zero for a unary conversion.
    uint32_t second_units;
    // Units owned by the result.
    uint32_t result_units;
    // Result class in the same bank, a shared bank, or a disjoint bank.
    uint16_t result_class;
    // Whether the captured operands require physical registers.
    bool inputs_required;
    // Whether the result requires physical registers.
    bool result_required;
    // Whether retaining the inputs preserves required physical storage.
    bool ordered;
  } cases[] = {
      {"same class", 4, 0, 4, 0, true, true, true},
      {"shared bank", 1, 0, 1, 1, true, true, true},
      {"different bank", 1, 0, 1, 2, true, true, false},
      {"different shared bank", 1, 0, 1, 3, true, true, false},
      {"unary contraction", 4, 0, 1, 0, true, true, false},
      {"combined contraction", 4, 4, 4, 0, true, true, false},
      {"expansion", 1, 0, 4, 0, true, true, true},
      {"composition", 1, 1, 2, 0, true, true, true},
      {"spillable result", 4, 0, 4, 0, true, false, false},
      {"spillable inputs", 4, 0, 1, 2, false, true, true},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    loom_low_schedule_value_record_t values[] = {
        {.unit_count = test_case.first_units, .register_class_id = 0},
        {.unit_count = test_case.second_units, .register_class_id = 0},
        {
            .unit_count = test_case.result_units,
            .register_class_id = test_case.result_class,
        },
    };
    if (test_case.inputs_required) {
      values[0].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_REQUIRES_REGISTER;
      if (test_case.second_units != 0) {
        values[1].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_REQUIRES_REGISTER;
      }
    }
    if (test_case.result_required) {
      values[2].flags = LOOM_LOW_SCHEDULE_VALUE_FLAG_REQUIRES_REGISTER;
    }
    loom_low_schedule_build_state_t state = {
        .options = &options,
        .target = {.descriptor_set = &descriptors},
        .values = values,
    };
    loom_low_schedule_node_t node = {
        .kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL,
        .operand_count =
            test_case.second_units == 0 ? uint16_t{1} : uint16_t{2},
        .result_count = 1,
        .flags = LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP,
        .value_ordinals = {.inline_value_ordinals = {0, 1}},
    };
    loom_low_schedule_node_result_ordinals(&node)[0] = 2;
    loom_low_schedule_setup_order_classify_node(
        &state, &node, /*order_structural_setup=*/true);
    EXPECT_EQ(
        iree_any_bit_set(node.flags, LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP),
        test_case.ordered);
    EXPECT_EQ(state.setup_order.has_members, test_case.ordered);
  }
}

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
