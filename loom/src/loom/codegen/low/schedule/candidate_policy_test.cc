// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/candidate_policy.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

static loom_low_schedule_candidate_score_t MakeCandidate(
    uint32_t effective_stall_cycles, uint32_t source_ordinal) {
  return {
      .data_ready_stall_cycles = effective_stall_cycles,
      .effective_stall_cycles = effective_stall_cycles,
      .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
      .units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
      .active_unspillable_completion_capacity = UINT32_MAX,
      .active_unspillable_transaction_final_capacity = UINT32_MAX,
      .opened_unspillable_completion_capacity = UINT32_MAX,
      .active_register_packing_completion_capacity = UINT32_MAX,
      .source_ordinal = source_ordinal,
      .selected_descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE,
      .active_unspillable_completion_domain_id = UINT16_MAX,
      .bottleneck_resource_id = LOOM_LOW_RESOURCE_NONE,
  };
}

static loom_low_schedule_candidate_score_t MakeConstrainedCandidate(
    uint32_t effective_stall_cycles, uint32_t source_ordinal) {
  auto score = MakeCandidate(effective_stall_cycles, source_ordinal);
  score.projected_live_units = 4;
  score.killed_live_units = 1;
  score.produced_live_units = 1;
  score.killed_live_value_count = 1;
  score.produced_live_value_count = 1;
  score.active_unspillable_completion_capacity = 4;
  score.active_unspillable_completion_domain_id = 1;
  score.pressure_progress_kind =
      LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_CONSTRAINED_COMPLETION;
  score.flags =
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_HAS_UNSPILLABLE_ACTIVATION |
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_CONSTRAINED_COMPLETION;
  return score;
}

class CandidatePolicyTest : public ::testing::Test {
 protected:
  void ExpectPreferred(loom_low_schedule_candidate_compare_mode_t mode,
                       const loom_low_schedule_candidate_score_t& preferred,
                       const loom_low_schedule_candidate_score_t& other,
                       iree_host_size_t register_debt_word_count = 0) {
    EXPECT_TRUE(loom_low_schedule_candidate_score_less(
        &options_, mode, &preferred, &other, register_debt_word_count));
    EXPECT_FALSE(loom_low_schedule_candidate_score_less(
        &options_, mode, &other, &preferred, register_debt_word_count));
    EXPECT_FALSE(loom_low_schedule_candidate_score_less(
        &options_, mode, &preferred, &preferred, register_debt_word_count));
  }

  loom_low_schedule_options_t options_ = {
      .strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
  };
};

TEST_F(CandidatePolicyTest, PressureProgressUsesPhysicalAndFrontierFacts) {
  auto score = MakeCandidate(0, 0);
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_progress(&score),
            LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NONE);

  score.killed_live_units = 8;
  score.produced_live_units = 4;
  score.killed_live_value_count = 2;
  score.produced_live_value_count = 1;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_progress(&score),
            LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_REDUCTION);

  // An aggregate reduction is not relief if a shared physical bank grows.
  score.flags = LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_GROWS_PACKING_RESOURCE;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_progress(&score),
            LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NONE);

  score.flags = 0;
  score.killed_live_units = 4;
  score.killed_live_value_count = 1;
  score.storage_relation_count = 1;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_progress(&score),
            LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_STORAGE);

  score.storage_relation_count = 0;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_progress(&score),
            LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NON_GROWING);

  score.produced_live_units = 8;
  score.produced_live_value_count = 2;
  score.flags = LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_progress(&score),
            LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NONE);

  // A growing candidate can still expose a non-growing descriptor frontier.
  score.flags |=
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_NON_GROWING_DESCRIPTOR;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_progress(&score),
            LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_FRONTIER);

  score.active_unspillable_completion_capacity = 8;
  score.active_unspillable_completion_domain_id = 1;
  score.flags |=
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_CONSTRAINED_COMPLETION;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_progress(&score),
            LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_CONSTRAINED_COMPLETION);
}

TEST_F(CandidatePolicyTest, PressureRiskProtectsBoundedDemandWithoutOldDebt) {
  auto score = MakeCandidate(0, 0);
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_risk(&score, 0),
            LOOM_LOW_SCHEDULE_PRESSURE_RISK_NONE);

  score.pressure_cliff_penalty = 1;
  score.pressure_cliff_units = 8;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_risk(&score, 0),
            LOOM_LOW_SCHEDULE_PRESSURE_RISK_DEBT);
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_risk(&score, 1),
            LOOM_LOW_SCHEDULE_PRESSURE_RISK_NONE);

  score.pressure_cliff_penalty = 0;
  score.pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE;
  score.units_until_pressure_cliff = 4;
  score.pressure_demand_units = 3;
  score.activation_reserve_units = 3;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_risk(&score, 0),
            LOOM_LOW_SCHEDULE_PRESSURE_RISK_NONE);

  score.pressure_demand_units = 4;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_risk(&score, 0),
            LOOM_LOW_SCHEDULE_PRESSURE_RISK_NEAR_CLIFF);
  score.pressure_demand_units = 3;
  score.activation_reserve_units = 4;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_risk(&score, 0),
            LOOM_LOW_SCHEDULE_PRESSURE_RISK_NEAR_CLIFF);

  // An absent bounded cliff is not a horizon, even for saturated demand.
  score.units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE;
  score.pressure_demand_units = UINT32_MAX;
  score.activation_reserve_units = UINT32_MAX;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_risk(&score, 0),
            LOOM_LOW_SCHEDULE_PRESSURE_RISK_NONE);

  score.flags = LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_NEEDS_COMPLETION_RECOVERY;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_risk(&score, 0),
            LOOM_LOW_SCHEDULE_PRESSURE_RISK_NEAR_CLIFF);
  score.pressure_cliff_penalty = 1;
  score.pressure_cliff_units = 8;
  EXPECT_EQ(loom_low_schedule_classify_candidate_pressure_risk(&score, 0),
            LOOM_LOW_SCHEDULE_PRESSURE_RISK_DEBT);
}

TEST_F(CandidatePolicyTest, CompareModeAccountsForEveryNomineeAndCurrentDebt) {
  loom_low_schedule_candidate_score_t scores[] = {
      MakeCandidate(0, 0), MakeCandidate(0, 1), MakeCandidate(0, 2)};
  EXPECT_EQ(loom_low_schedule_choose_candidate_compare_mode(scores, 3, 0),
            LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT);
  EXPECT_EQ(loom_low_schedule_choose_candidate_compare_mode(scores, 3, 1),
            LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF);

  scores[0].pressure_cliff_penalty = 1;
  scores[0].pressure_cliff_units = 8;
  scores[0].pressure_risk =
      loom_low_schedule_classify_candidate_pressure_risk(&scores[0], 0);
  EXPECT_EQ(loom_low_schedule_choose_candidate_compare_mode(scores, 3, 0),
            LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF);

  // A later packing nominee controls the entire comparison, even after an
  // earlier nominee already established a general pressure-recovery need.
  scores[2].flags = LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_NEEDS_COMPLETION_RECOVERY;
  scores[2].pressure_risk =
      loom_low_schedule_classify_candidate_pressure_risk(&scores[2], 0);
  EXPECT_EQ(loom_low_schedule_choose_candidate_compare_mode(scores, 3, 1),
            LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION);
}

TEST_F(CandidatePolicyTest, PackingRecoveryPreservesStorageHandoff) {
  auto storage_handoff = MakeConstrainedCandidate(3, 0);
  storage_handoff.flags |= LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_STORAGE;
  const auto ready_peer = MakeConstrainedCandidate(1, 1);

  // The handoff retains the selected aggregate completion. Opening the peer
  // first can fragment the full bank before that completion reaches its use.
  ExpectPreferred(LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION,
                  storage_handoff, ready_peer);
}

TEST_F(CandidatePolicyTest, HardCapacityPrecedesReadinessAndAggregateRelief) {
  const auto fitting = MakeCandidate(4, 1);
  auto overflowing = MakeCandidate(0, 0);
  overflowing.killed_live_units = 8;
  overflowing.produced_live_units = 1;
  overflowing.killed_live_value_count = 2;
  overflowing.produced_live_value_count = 1;
  overflowing.pressure_progress_kind =
      LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_REDUCTION;
  overflowing.flags =
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXCEEDS_UNSPILLABLE_CAPACITY;
  for (const auto mode :
       {LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT,
        LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF,
        LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION}) {
    SCOPED_TRACE(mode);
    ExpectPreferred(mode, fitting, overflowing);
  }
}

TEST_F(CandidatePolicyTest, PressureReliefSchedulesReadyPeer) {
  auto storage_handoff = MakeConstrainedCandidate(3, 0);
  storage_handoff.flags |= LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_STORAGE;
  const auto ready_peer = MakeConstrainedCandidate(1, 1);

  ExpectPreferred(LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF,
                  ready_peer, storage_handoff);
}

TEST_F(CandidatePolicyTest, PackingRecoverySchedulesReadyPeerWithoutHandoff) {
  const auto stalled_peer = MakeConstrainedCandidate(3, 0);
  const auto ready_peer = MakeConstrainedCandidate(1, 1);

  ExpectPreferred(LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION,
                  ready_peer, stalled_peer);

  // Materialized spill helpers must retain their demand order: their setup
  // cannot run ahead of the transfer that closes the current lifetime.
  options_.flags |= LOOM_LOW_SCHEDULE_FLAG_HAS_LOWERED_SPILL_TRAFFIC;
  ExpectPreferred(LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION,
                  stalled_peer, ready_peer);
}

TEST_F(CandidatePolicyTest, ReadyProgressRequiresTheSameCompletionDomain) {
  const auto stalled_peer = MakeConstrainedCandidate(3, 0);
  auto ready_peer = MakeConstrainedCandidate(0, 1);
  for (const auto mode :
       {LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF,
        LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION}) {
    SCOPED_TRACE(mode);
    ready_peer.active_unspillable_completion_domain_id =
        stalled_peer.active_unspillable_completion_domain_id;
    ExpectPreferred(mode, ready_peer, stalled_peer);

    // Equal capacity does not make independent banks interchangeable. The
    // ready completion cannot displace an older transaction in another bank.
    ready_peer.active_unspillable_completion_domain_id = 2;
    ExpectPreferred(mode, stalled_peer, ready_peer);
  }
}

TEST_F(CandidatePolicyTest, PhysicalCliffPrecedesAggregateReliefAndReadiness) {
  auto safe = MakeCandidate(4, 1);
  safe.projected_live_units = 25;
  safe.produced_live_units = 1;
  safe.produced_live_value_count = 1;

  // This candidate reduces more total units, but exchanges space in one bank
  // for a crossed cliff in another. Its ready issue slot cannot repay that.
  auto crossing = safe;
  crossing.data_ready_stall_cycles = 0;
  crossing.effective_stall_cycles = 0;
  crossing.source_ordinal = 0;
  crossing.projected_live_units = 16;
  crossing.killed_live_units = 12;
  crossing.produced_live_units = 4;
  crossing.killed_live_value_count = 3;
  crossing.pressure_cliff_penalty = 1;
  crossing.pressure_cliff_units = 8;
  crossing.pressure_risk = LOOM_LOW_SCHEDULE_PRESSURE_RISK_DEBT;
  crossing.pressure_progress_kind =
      LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_REDUCTION;

  ExpectPreferred(LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF, safe,
                  crossing);
}

TEST_F(CandidatePolicyTest, RetainedSetupRequiresBoundedProgress) {
  auto setup = MakeCandidate(0, 0);
  setup.projected_live_units = 4;
  setup.killed_live_units = 1;
  setup.produced_live_units = 1;
  setup.killed_live_value_count = 1;
  setup.produced_live_value_count = 1;
  setup.pressure_progress_kind =
      LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NON_GROWING;
  setup.flags = LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_STORAGE_SETUP |
                LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_HAS_UNSPILLABLE_ACTIVATION |
                LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_RETAINS_PRE_SPILL_PLACEMENT;
  auto independent = MakeCandidate(0, 1);
  independent.projected_live_units = 5;
  independent.produced_live_units = 1;
  independent.produced_live_value_count = 1;

  for (const auto mode :
       {LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT,
        LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF,
        LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION}) {
    SCOPED_TRACE(mode);
    ExpectPreferred(mode, setup, independent);
    for (const uint16_t flag :
         {LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_HAS_UNSPILLABLE_ACTIVATION,
          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_RETAINS_PRE_SPILL_PLACEMENT}) {
      auto deferred = setup;
      deferred.flags &= ~flag;
      ExpectPreferred(mode, independent, deferred);
    }

    // Replayable leaves capture no input lifetime to advance. Retention alone
    // does not justify opening them before independent work.
    auto leaf = setup;
    leaf.projected_live_units = 5;
    leaf.killed_live_units = 0;
    leaf.killed_live_value_count = 0;
    leaf.pressure_progress_kind = LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NONE;
    leaf.flags |= LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_REMATERIALIZABLE_LEAF;
    ExpectPreferred(mode, independent, leaf);

    // Several leaves may be needed before a pinned transaction can finish.
    // Its retained terminal path admits them only while their storage fits.
    leaf.active_unspillable_completion_capacity = 1;
    leaf.active_unspillable_completion_domain_id = 0;
    leaf.active_unspillable_transaction_final_capacity = 1;
    leaf.flags |=
        LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_CONSTRAINED_COMPLETION;
    leaf.pressure_progress_kind =
        LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_CONSTRAINED_COMPLETION;
    ExpectPreferred(mode, leaf, independent);
    for (const uint16_t flag :
         {LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REGISTER_DEBT,
          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXCEEDS_UNSPILLABLE_CAPACITY}) {
      auto infeasible = leaf;
      infeasible.flags |= flag;
      infeasible.pressure_risk = LOOM_LOW_SCHEDULE_PRESSURE_RISK_DEBT;
      infeasible.pressure_cliff_penalty = 1;
      infeasible.pressure_cliff_units = 4;
      ExpectPreferred(mode, independent, infeasible);
    }

    auto constrained = setup;
    constrained.pressure_risk = LOOM_LOW_SCHEDULE_PRESSURE_RISK_NEAR_CLIFF;
    constrained.units_until_pressure_cliff = 1;
    constrained.activation_reserve_units = 1;
    ExpectPreferred(mode, independent, constrained);
    constrained.pressure_cliff_penalty = 1;
    constrained.pressure_cliff_units = 4;
    constrained.pressure_risk = LOOM_LOW_SCHEDULE_PRESSURE_RISK_DEBT;
    ExpectPreferred(mode, independent, constrained);
  }
}

TEST_F(CandidatePolicyTest, NearCliffRecoveryDefersReadyGrowth) {
  auto recovery = MakeCandidate(4, 1);
  recovery.projected_live_units = 12;
  recovery.killed_live_units = 4;
  recovery.produced_live_units = 4;
  recovery.killed_live_value_count = 1;
  recovery.produced_live_value_count = 1;
  recovery.pressure_progress_kind =
      LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_NON_GROWING;

  auto growth = MakeCandidate(0, 0);
  growth.projected_live_units = 16;
  growth.produced_live_units = 4;
  growth.produced_live_value_count = 1;
  growth.pressure_demand_units = 4;
  growth.units_until_pressure_cliff = 1;
  growth.pressure_risk = LOOM_LOW_SCHEDULE_PRESSURE_RISK_NEAR_CLIFF;

  ExpectPreferred(LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF, recovery,
                  growth);
}

TEST_F(CandidatePolicyTest, PressureRecoveryFillsSafeIssueGaps) {
  auto recovery = MakeCandidate(4, 0);
  recovery.projected_live_units = 20;
  recovery.killed_live_units = 8;
  recovery.produced_live_units = 4;
  recovery.killed_live_value_count = 2;
  recovery.produced_live_value_count = 1;
  recovery.pressure_progress_kind =
      LOOM_LOW_SCHEDULE_PRESSURE_PROGRESS_REDUCTION;

  auto independent = MakeCandidate(0, 1);
  independent.projected_live_units = 25;
  independent.produced_live_units = 1;
  independent.produced_live_value_count = 1;

  // Pressure in another candidate's bank selects recovery mode, but these two
  // candidates both remain safely below their own bounded pressure horizons.
  ExpectPreferred(LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF,
                  independent, recovery);
}

TEST_F(CandidatePolicyTest, RepairableDebtMustStayWithinRequiredBanks) {
  for (uint32_t bank : {0u, 63u, 64u, 127u}) {
    SCOPED_TRACE(bank);
    loom_low_schedule_register_debt_word_t required_debt[2] = {};
    loom_low_schedule_register_debt_word_t alternative_debt[2] = {};
    const uint32_t word = bank / 64;
    const uint64_t bit = UINT64_C(1) << (bank % 64);
    required_debt[word] = {.banks = bit, .required_banks = bit};
    alternative_debt[word].banks = bit;

    auto required = MakeCandidate(0, 0);
    required.register_debt = required_debt;
    required.projected_live_units = 10;
    required.produced_live_units = 1;
    required.produced_live_value_count = 1;
    required.pressure_cliff_penalty = 1;
    required.persistent_pressure_cliff_penalty = 1;
    required.pressure_cliff_units = 8;
    required.pressure_risk = LOOM_LOW_SCHEDULE_PRESSURE_RISK_DEBT;
    required.flags =
        LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REGISTER_DEBT |
        LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REQUIRED_REGISTER_DEBT;

    auto alternative = required;
    alternative.register_debt = alternative_debt;
    alternative.data_ready_stall_cycles = 2;
    alternative.effective_stall_cycles = 2;
    alternative.pressure_cliff_penalty = 2;
    alternative.persistent_pressure_cliff_penalty = 2;
    alternative.flags = LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REGISTER_DEBT;
    alternative.source_ordinal = 1;

    ExpectPreferred(LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF,
                    alternative, required, 2);

    // Debt in another bank is not relief. The ordinary physical cost wins
    // instead of repeatedly postponing required reloads behind such setup.
    alternative_debt[1 - word].banks = 1;
    ExpectPreferred(LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF,
                    required, alternative, 2);

    // A fitting alternative opens no debt and leaves the required bank alone.
    alternative_debt[0] = {};
    alternative_debt[1] = {};
    alternative.flags = 0;
    alternative.pressure_cliff_penalty = 0;
    alternative.persistent_pressure_cliff_penalty = 0;
    alternative.pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE;
    alternative.pressure_risk = LOOM_LOW_SCHEDULE_PRESSURE_RISK_NONE;
    SCOPED_TRACE("fitting alternative");
    ExpectPreferred(LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF,
                    alternative, required, 2);
  }
}

TEST_F(CandidatePolicyTest, ReadyPackingCompletionPrecedesOtherBankStall) {
  auto ready_completion = MakeConstrainedCandidate(0, 1);
  ready_completion.flags |=
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXACT_PACKING_COMPLETION;
  auto stalled_handoff = MakeConstrainedCandidate(1, 0);
  stalled_handoff.flags |= LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_STORAGE;
  stalled_handoff.active_unspillable_completion_domain_id = 2;

  EXPECT_TRUE(loom_low_schedule_candidate_policy_uses_ready_progress_order(
      &options_, LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION,
      &ready_completion, &stalled_handoff));
  EXPECT_TRUE(loom_low_schedule_candidate_policy_uses_ready_progress_order(
      &options_, LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION,
      &stalled_handoff, &ready_completion));

  // A handoff in the same bank still owns its packing constraint.
  stalled_handoff.active_unspillable_completion_domain_id =
      ready_completion.active_unspillable_completion_domain_id;
  EXPECT_FALSE(loom_low_schedule_candidate_policy_uses_ready_progress_order(
      &options_, LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION,
      &ready_completion, &stalled_handoff));
}

TEST_F(CandidatePolicyTest, StalledPackingIdentityDoesNotOverrideOtherBank) {
  auto stalled_completion = MakeConstrainedCandidate(3, 0);
  stalled_completion.flags |=
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXACT_PACKING_COMPLETION;
  auto ready_handoff = MakeConstrainedCandidate(0, 1);
  ready_handoff.flags |= LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_STORAGE;
  ready_handoff.active_unspillable_completion_domain_id = 2;

  EXPECT_FALSE(loom_low_schedule_candidate_policy_uses_ready_progress_order(
      &options_, LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION,
      &stalled_completion, &ready_handoff));
  EXPECT_FALSE(loom_low_schedule_candidate_policy_uses_ready_progress_order(
      &options_, LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION,
      &ready_handoff, &stalled_completion));
}

}  // namespace
}  // namespace loom
