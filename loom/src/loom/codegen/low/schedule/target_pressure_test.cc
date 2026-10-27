// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/target_pressure.h"

#include "iree/testing/gtest.h"
#include "loom/target/residency.h"

namespace loom {
namespace {

TEST(TargetPressureTest, ResidencyHeadroomBelongsToGrowingResources) {
  const loom_low_reg_class_t classes[] = {
      {.alloc_unit_bits = 64},
      {.alloc_unit_bits = 32},
  };
  const loom_low_descriptor_set_t descriptors = {
      .reg_classes = classes,
      .reg_class_count = IREE_ARRAYSIZE(classes),
  };
  const iree_string_view_t resource_names[] = {IREE_SV("pointer"),
                                               IREE_SV("scalar")};
  const loom_target_residency_cliff_t cliffs[] = {{0, 8, 15, 14},
                                                  {0, 12, 14, 13}};
  const loom_target_residency_cliff_range_t ranges[] = {
      {.start = 0, .count = IREE_ARRAYSIZE(cliffs)},
      {.start = IREE_ARRAYSIZE(cliffs), .count = 0},
  };
  const loom_target_residency_model_t residency = {
      .best_tier = 15,
      .direct_resources =
          {
              .names = resource_names,
              .cliffs = cliffs,
              .cliff_count = IREE_ARRAYSIZE(cliffs),
              .cliff_ranges = ranges,
              .resource_count = IREE_ARRAYSIZE(ranges),
          },
  };
  const loom_low_schedule_options_t options = {
      .residency = loom_target_residency_view(&residency, UINT32_MAX),
  };
  const loom_low_schedule_build_state_t state = {
      .options = &options,
      .pressure_cliffs = &residency.direct_resources,
      .target = {.descriptor_set = &descriptors},
  };
  const uint32_t no_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE;
  const struct {
    // Human-readable pressure transition for assertion failures.
    const char* name;
    // Whether the candidate's pointer delta is current rather than stale.
    uint8_t pointer_touched;
    // Change to the five currently live pointer units.
    int64_t pointer_delta;
    // First cliff not already crossed by the source-order baseline.
    uint32_t first_cliff;
    // Residency tiers lost by the candidate.
    uint32_t penalty;
    // Additional growing-resource units before losing another tier.
    uint32_t units_until_cliff;
  } cases[] = {
      {"untouched pointer bank", 0, 99, 0, 0, no_cliff},
      {"unchanged pointer bank", 1, 0, 0, 0, no_cliff},
      {"shrinking pointer bank", 1, -1, 0, 0, no_cliff},
      {"growing below cliff", 1, 1, 0, 0, 2},
      {"one unit below cliff", 1, 2, 0, 0, 1},
      {"crossing cliff", 1, 3, 0, 1, no_cliff},
      {"first cliff already crossed by source", 1, 3, 1, 0, 4},
      {"next cliff remains actionable", 1, 7, 1, 1, no_cliff},
      {"both cliffs crossed", 1, 7, 0, 2, no_cliff},
      {"all cliffs already crossed by source", 1, 7, 2, 0, no_cliff},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    uint64_t live[] = {5, 0};
    int64_t delta[] = {test_case.pointer_delta, 1};
    uint8_t delta_touched[] = {test_case.pointer_touched, 1};
    uint16_t delta_classes[] = {1, 0};
    uint16_t block_classes[] = {0};
    uint8_t block_touched[] = {1, 0};
    uint32_t first_cliffs[] = {test_case.first_cliff, IREE_ARRAYSIZE(cliffs)};
    loom_low_schedule_pressure_state_t pressure = {
        .current_live_units_by_reg_class = live,
        .first_actionable_pressure_cliff_indices = first_cliffs,
        .block_reg_class_ids = block_classes,
        .block_reg_class_touched_flags = block_touched,
        .candidate_delta_units_by_reg_class = delta,
        .candidate_delta_touched_flags = delta_touched,
        .candidate_delta_touched_reg_class_ids = delta_classes,
        .candidate_delta_touched_count = 1u + test_case.pointer_touched,
        .block_reg_class_count = IREE_ARRAYSIZE(block_classes),
    };
    loom_low_schedule_candidate_score_t score = {
        .pressure_cliff_units = no_cliff,
        .units_until_pressure_cliff = no_cliff,
    };
    loom_low_schedule_target_pressure_score_candidate(
        &state, &pressure, LOOM_LOW_SCHEDULE_NODE_NONE, &score);
    EXPECT_EQ(score.pressure_cliff_penalty, test_case.penalty);
    EXPECT_EQ(score.persistent_pressure_cliff_penalty, test_case.penalty);
    EXPECT_EQ(score.units_until_pressure_cliff, test_case.units_until_cliff);
  }
}

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
