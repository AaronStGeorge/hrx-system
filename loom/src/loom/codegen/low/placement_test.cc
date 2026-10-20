// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/placement.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(LowPlacementTest, CombinesDescriptorOperandConstraints) {
  const loom_low_reg_class_alt_t alternatives[] = {
      {.reg_class_id = 3, .unit_alignment_log2 = 1},
      {.reg_class_id = 7, .unit_alignment_log2 = 6},
      {.reg_class_id = 3, .unit_alignment_log2 = 4},
  };
  const loom_low_descriptor_set_t descriptor_set = {
      .reg_class_alts = alternatives,
      .reg_class_alt_count = IREE_ARRAYSIZE(alternatives),
  };
  loom_low_operand_t low_operand = {
      .reg_class_alt_count = IREE_ARRAYSIZE(alternatives),
      .address_map_kind = LOOM_LOW_OPERAND_ADDRESS_MAP_LOW_SUBSET,
      .addressable_unit_count = 32,
  };
  const loom_low_placement_operand_constraints_t low_constraints =
      loom_low_placement_operand_constraints_for_descriptor_operand(
          &descriptor_set, &low_operand, 3);
  EXPECT_EQ(low_constraints.unit_alignment_log2, 4);
  EXPECT_EQ(low_constraints.addressable_unit_count, 32);
  EXPECT_FALSE(low_constraints.has_target_address_state);

  low_operand.address_map_kind = LOOM_LOW_OPERAND_ADDRESS_MAP_TARGET_STATE;
  const loom_low_placement_operand_constraints_t state_constraints =
      loom_low_placement_operand_constraints_for_descriptor_operand(
          &descriptor_set, &low_operand, 7);
  EXPECT_EQ(state_constraints.unit_alignment_log2, 6);
  EXPECT_EQ(state_constraints.addressable_unit_count, 0);
  EXPECT_TRUE(state_constraints.has_target_address_state);

  loom_low_placement_operand_constraints_t combined = {
      .addressable_unit_count = 16,
      .unit_alignment_log2 = 5,
  };
  loom_low_placement_operand_constraints_merge(&combined, &low_constraints);
  loom_low_placement_operand_constraints_merge(&combined, &state_constraints);
  EXPECT_EQ(combined.unit_alignment_log2, 6);
  EXPECT_EQ(combined.addressable_unit_count, 16);
  EXPECT_TRUE(combined.has_target_address_state);
}

TEST(LowPlacementTest, ClassifiesEdgeCauses) {
  EXPECT_FALSE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_UNKNOWN));
  EXPECT_FALSE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT));
  EXPECT_FALSE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY));
  EXPECT_FALSE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_MOVE));
  EXPECT_FALSE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_SLICE));
  EXPECT_FALSE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT));
  EXPECT_TRUE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH));
  EXPECT_TRUE(loom_low_placement_cause_is_edge(
      LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_LOOP_ENTRY));
  EXPECT_TRUE(
      loom_low_placement_cause_is_edge(LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_YIELD));
  EXPECT_TRUE(loom_low_placement_cause_is_edge(
      LOOM_LOW_PLACEMENT_CAUSE_LOW_SCF_CONDITION));
  EXPECT_FALSE(loom_low_placement_cause_is_edge(
      LOOM_LOW_PLACEMENT_CAUSE_SCHEDULE_PAIR_AFFINITY));
  EXPECT_FALSE(loom_low_placement_cause_is_edge(
      LOOM_LOW_PLACEMENT_CAUSE_EDGE_PAIR_AFFINITY));
}

TEST(LowPlacementTest, ComposedRelationHasNoDirectSourceOperand) {
  loom_low_placement_relation_t source_to_intermediate = {
      .result_ordinal = 1,
      .source_ordinal = 0,
      .result_unit_offset = 2,
      .source_unit_offset = 4,
      .unit_count = 3,
      .source_operand_index = 1,
  };
  loom_low_placement_relation_t intermediate_to_result = {
      .result_ordinal = 2,
      .source_ordinal = 1,
      .result_unit_offset = 8,
      .source_unit_offset = 3,
      .unit_count = 3,
      .source_operand_index = 2,
  };

  loom_low_placement_relation_t composed = {};
  ASSERT_TRUE(loom_low_placement_relation_compose(
      &source_to_intermediate, &intermediate_to_result, &composed));
  EXPECT_EQ(composed.source_ordinal, 0u);
  EXPECT_EQ(composed.result_ordinal, 2u);
  EXPECT_EQ(composed.source_unit_offset, 5u);
  EXPECT_EQ(composed.result_unit_offset, 8u);
  EXPECT_EQ(composed.unit_count, 2u);
  EXPECT_EQ(composed.source_operand_index,
            LOOM_LOW_PLACEMENT_SOURCE_OPERAND_NONE);
}

TEST(LowPlacementTest, StorageCompositionDoesNotImplyBitIdentity) {
  loom_low_placement_relation_t write = {
      .result_ordinal = 1,
      .source_ordinal = 0,
      .unit_count = 2,
      .cause = LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT,
      .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_HARD |
               LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE |
               LOOM_LOW_PLACEMENT_RELATION_FLAG_WRITES_STORAGE,
  };
  loom_low_placement_relation_t edge = {
      .result_ordinal = 2,
      .source_ordinal = 1,
      .unit_count = 2,
      .cause = LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH,
      .flags = LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE |
               LOOM_LOW_PLACEMENT_RELATION_FLAG_IDENTITY_EDGE,
  };

  loom_low_placement_relation_t composed = {};
  ASSERT_TRUE(loom_low_placement_relation_compose(&write, &edge, &composed));
  EXPECT_EQ(composed.source_ordinal, 0u);
  EXPECT_EQ(composed.result_ordinal, 2u);
  EXPECT_EQ(composed.flags, LOOM_LOW_PLACEMENT_RELATION_FLAG_CAN_ALIAS_STORAGE);
}

}  // namespace
}  // namespace loom
