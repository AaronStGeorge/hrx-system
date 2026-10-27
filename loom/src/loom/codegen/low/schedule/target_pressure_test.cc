// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/target_pressure.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

class RegisterPackingPressureTest : public ::testing::Test {
 protected:
  loom_low_schedule_candidate_score_t Score() {
    packed_units_[0] = 0;
    for (size_t i = 0; i < IREE_ARRAYSIZE(members_); ++i) {
      packed_units_[0] += loom_low_schedule_register_packing_contribution(
          live_[i], &members_[i]);
    }
    loom_low_schedule_candidate_score_t score = {
        .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
        .units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
    };
    loom_low_schedule_score_candidate_register_packing_resources(
        &state_, &pressure_, LOOM_LOW_SCHEDULE_NODE_NONE, &score);
    return score;
  }

  // A general allocation domain and a singleton view share physical storage.
  loom_low_reg_class_t classes_[2] = {
      {.alloc_unit_bits = 64, .allocatable_count = 4, .alias_set_id = 1},
      {.alloc_unit_bits = 64, .allocatable_count = 1, .alias_set_id = 1},
  };
  // Two allocation domains draw from the same four-location physical bank.
  loom_low_register_packing_resource_member_t members_[2] = {
      {.reg_class_id = 0, .register_unit_count = 1, .resource_unit_count = 1},
      {.reg_class_id = 1, .register_unit_count = 1, .resource_unit_count = 1},
  };
  // Descriptor-owned shared capacity and member slice.
  loom_low_register_packing_resource_t resource_ = {
      .capacity = 4,
      .member_count = IREE_ARRAYSIZE(members_),
  };
  // The descriptor projection consumed by native resource scoring.
  loom_low_descriptor_set_t descriptors_ = {
      .reg_classes = classes_,
      .reg_class_count = IREE_ARRAYSIZE(classes_),
      .register_packing_resources = &resource_,
      .register_packing_resource_count = 1,
      .register_packing_resource_members = members_,
      .register_packing_resource_member_count = IREE_ARRAYSIZE(members_),
  };
  // Function-local capacity after reserved locations have been accounted for.
  uint32_t capacities_[1] = {4};
  // Native resource inputs; no IR or node completion query participates.
  loom_low_schedule_build_state_t state_ = {
      .target = {.descriptor_set = &descriptors_},
      .pressure_limits = {.register_packing_resource_capacities = capacities_},
  };
  // Current register units in the general and fixed-view domains.
  uint64_t live_[2] = {3, 1};
  // Result-minus-input changes for the candidate.
  int64_t delta_[2] = {};
  // Units allocated before ordinary reads complete.
  uint64_t early_[2] = {};
  // Dying units retained through result writes.
  uint64_t late_[2] = {};
  // Both class projections are present, including zero deltas.
  uint8_t touched_[2] = {1, 1};
  // Retained rounded sum of live member contributions.
  uint64_t packed_units_[1] = {};
  // No downstream activation is needed for these immediate phase decisions.
  uint32_t activation_[1] = {};
  // Mutable pressure projection over the native arrays above.
  loom_low_schedule_pressure_state_t pressure_ = {
      .current_live_units_by_reg_class = live_,
      .candidate_delta_units_by_reg_class = delta_,
      .candidate_lifetime =
          {
              .early_added_units = early_,
              .late_released_units = late_,
          },
      .candidate_register_packing_activation_units = activation_,
      .current_register_packing_units = packed_units_,
      .candidate_delta_touched_flags = touched_,
  };
};

TEST_F(RegisterPackingPressureTest, FixedViewConsumesGeneralBankCapacity) {
  // An independent load grows the general class from three to four. It fits
  // that class alone but overlaps a live fixed register in the same bank.
  delta_[0] = 1;
  const auto load = Score();
  EXPECT_EQ(load.pressure_cliff_penalty, 1u);
  EXPECT_EQ(load.persistent_pressure_cliff_penalty, 1u);
  EXPECT_EQ(load.pressure_cliff_source_kind,
            LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_PACKING_RESOURCE);
  EXPECT_EQ(load.pressure_cliff_units, 5u);
  EXPECT_TRUE(iree_any_bit_set(
      load.flags, LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_GROWS_PACKING_RESOURCE));

  // Completing the fixed-register transaction first leaves room for the load.
  delta_[0] = 0;
  delta_[1] = -1;
  const auto completion = Score();
  EXPECT_EQ(completion.pressure_cliff_penalty, 0u);
  EXPECT_EQ(completion.persistent_pressure_cliff_penalty, 0u);
  EXPECT_EQ(completion.flags, 0u);
}

TEST_F(RegisterPackingPressureTest, ReuseDoesNotSumSeparateLifetimePhases) {
  delta_[0] = 1;
  delta_[1] = -1;
  EXPECT_EQ(Score().pressure_cliff_penalty, 0u);

  // Early writes overlap the old fixed value; late reads overlap the new one.
  early_[0] = 1;
  const auto early_write = Score();
  EXPECT_EQ(early_write.pressure_cliff_penalty, 1u);
  EXPECT_EQ(early_write.persistent_pressure_cliff_penalty, 0u);
  early_[0] = 0;
  late_[1] = 1;
  const auto late_read = Score();
  EXPECT_EQ(late_read.pressure_cliff_penalty, 1u);
  EXPECT_EQ(late_read.persistent_pressure_cliff_penalty, 0u);
}

TEST_F(RegisterPackingPressureTest, RoundedMembersReleaseOnlyCompleteGroups) {
  members_[0].register_unit_count = 2;
  resource_.flags =
      LOOM_LOW_REGISTER_PACKING_RESOURCE_FLAG_HAS_AGGREGATE_MEMBER;
  capacities_[0] = 2;
  delta_[0] = -1;
  // Three narrow units round to two groups. Consuming one releases a group;
  // consuming one of four leaves both occupied beside the other member.
  for (uint64_t units : {3u, 4u}) {
    SCOPED_TRACE(units);
    live_[0] = units;
    const auto score = Score();
    // Read-time pressure still includes the consumed unit in either case.
    EXPECT_EQ(score.pressure_cliff_penalty, 1u);
    EXPECT_EQ(score.persistent_pressure_cliff_penalty, units == 3 ? 0u : 1u);
  }
}

}  // namespace
}  // namespace loom
