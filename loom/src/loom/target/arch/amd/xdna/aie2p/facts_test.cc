// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/facts.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/profile.h"
#include "loom/target/arch/amd/xdna/aie2p/records/target_records.h"
#include "loom/target/facts_builder.h"

namespace {

constexpr const char* kDeviceKeys[] = {"amd.xdna.strix.17f0_10",
                                       "amd.xdna.strix_halo.17f0_11",
                                       "amd.xdna.krackan.17f0_20"};

struct Contract {
  // Authored target's input representation.
  loom_aie2p_target_kind_t kind;
  // Default facts for that representation.
  const loom_target_bundle_t* bundle;
};
const Contract kContracts[] = {
    {LOOM_AIE2P_TARGET_KIND_CORE, &loom_aie2p_core_target_bundle},
    {LOOM_AIE2P_TARGET_KIND_ARRAY, &loom_aie2p_array_target_bundle},
    {LOOM_AIE2P_TARGET_KIND_CONFIGURATION,
     &loom_aie2p_configuration_target_bundle}};

class Aie2pFactsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }
  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }
  void Project(const char* key, loom_target_facts_t** facts) {
    const loom_aie2p_target_profile_t* profile = nullptr;
    IREE_ASSERT_OK(loom_aie2p_target_profile_select(iree_make_cstring_view(key),
                                                    &profile));
    IREE_ASSERT_OK(
        loom_target_profile_project_facts(&profile->base, &arena_, facts));
  }
  void Requirement(const Contract& contract,
                   const loom_xdna_device_profile_t* device,
                   loom_aie2p_target_facts_t* facts) {
    *facts = {};
    loom_target_facts_builder_initialize(&loom_aie2p_target_fact_type,
                                         contract.bundle, &facts->base);
    facts->base.selector = contract.kind;
    facts->device_profile = device;
  }
  // Shared backing storage for projected and selected facts.
  iree_arena_block_pool_t pool_;
  // Per-test immutable fact lifetime.
  iree_arena_allocator_t arena_;
};

TEST_F(Aie2pFactsTest, DeviceProfilesSelectOnlyMatchingWholeArrayContracts) {
  for (const char* key : kDeviceKeys) {
    SCOPED_TRACE(key);
    loom_target_facts_t* available = nullptr;
    Project(key, &available);
    ASSERT_NE(available, nullptr);
    const auto* device =
        loom_aie2p_target_facts_cast(available)->device_profile;
    ASSERT_TRUE(
        loom_aie2p_target_facts_cast(available)->array_contract_unselected);
    const loom_xdna_device_profile_t* requirements[] = {
        nullptr,
        loom_xdna_device_profile_lookup(iree_make_cstring_view(kDeviceKeys[0])),
        loom_xdna_device_profile_lookup(iree_make_cstring_view(kDeviceKeys[1])),
        loom_xdna_device_profile_lookup(
            iree_make_cstring_view(kDeviceKeys[2]))};
    for (const auto& contract : kContracts) {
      SCOPED_TRACE(contract.kind);
      for (const auto* required_device : requirements) {
        SCOPED_TRACE(required_device ? required_device->key : "generic");
        loom_aie2p_target_facts_t requirement;
        Requirement(contract, required_device, &requirement);
        const bool accepted = contract.kind != LOOM_AIE2P_TARGET_KIND_CORE &&
                              (!required_device || required_device == device);
        EXPECT_EQ(loom_target_facts_satisfy_specialization_requirement(
                      available, &requirement.base),
                  accepted);
        if (!accepted) {
          continue;
        }
        loom_target_facts_t* selected = nullptr;
        IREE_ASSERT_OK(
            loom_target_facts_builder_clone(available, &arena_, &selected));
        loom_target_facts_builder_apply_requirement(&requirement.base,
                                                    selected);
        EXPECT_EQ(selected->selector, contract.kind);
        EXPECT_EQ(selected->storage.config.contract_set.stable_id,
                  contract.bundle->config->contract_set.stable_id);
        EXPECT_EQ(selected->storage.bundle.config, &selected->storage.config);
        EXPECT_EQ(loom_aie2p_target_facts_cast(selected)->device_profile,
                  device);
        EXPECT_FALSE(
            loom_aie2p_target_facts_cast(selected)->array_contract_unselected);
        EXPECT_FALSE(loom_target_facts_are_equivalent(available, selected));
        const loom_target_facts_t* chosen = nullptr;
        IREE_ASSERT_OK(loom_target_facts_builder_select_execution(
            selected, &arena_, &chosen));
        EXPECT_EQ(chosen, selected);
        for (const auto& callee_contract : kContracts) {
          loom_aie2p_target_facts_t callee;
          Requirement(callee_contract, device, &callee);
          EXPECT_EQ(loom_target_facts_satisfy_specialization_requirement(
                        selected, &callee.base),
                    callee_contract.kind == contract.kind);
        }
      }
    }
  }
}

TEST_F(Aie2pFactsTest, DefaultSelectionAndWorkerProjectionCloseContractChoice) {
  for (const char* key : kDeviceKeys) {
    loom_target_facts_t* available = nullptr;
    Project(key, &available);
    ASSERT_NE(available, nullptr);
    const loom_target_facts_t* selected = nullptr;
    IREE_ASSERT_OK(loom_target_facts_builder_select_execution(
        available, &arena_, &selected));
    EXPECT_EQ(selected->selector, LOOM_AIE2P_TARGET_KIND_ARRAY);
    EXPECT_FALSE(
        loom_aie2p_target_facts_cast(selected)->array_contract_unselected);
    EXPECT_FALSE(loom_target_facts_are_equivalent(available, selected));
    const loom_target_facts_t* worker = nullptr;
    IREE_ASSERT_OK(
        loom_target_facts_builder_project_worker(available, &arena_, &worker));
    EXPECT_EQ(worker->selector, LOOM_AIE2P_TARGET_KIND_CORE);
    EXPECT_FALSE(
        loom_aie2p_target_facts_cast(worker)->array_contract_unselected);
    EXPECT_EQ(loom_aie2p_target_facts_cast(worker)->device_profile,
              loom_aie2p_target_facts_cast(available)->device_profile);
    const loom_target_facts_t* same_worker = nullptr;
    IREE_ASSERT_OK(loom_target_facts_builder_project_worker(worker, &arena_,
                                                            &same_worker));
    EXPECT_EQ(same_worker, worker);
  }
  loom_target_facts_t* core = nullptr;
  Project("core", &core);
  ASSERT_NE(core, nullptr);
  const loom_target_facts_t* selected_core = nullptr;
  IREE_ASSERT_OK(loom_target_facts_builder_select_execution(core, &arena_,
                                                            &selected_core));
  EXPECT_EQ(selected_core, core);
  for (const auto& contract : kContracts) {
    loom_aie2p_target_facts_t requirement;
    Requirement(contract, nullptr, &requirement);
    EXPECT_EQ(loom_target_facts_satisfy_specialization_requirement(
                  core, &requirement.base),
              contract.kind == LOOM_AIE2P_TARGET_KIND_CORE);
  }
}

TEST_F(Aie2pFactsTest,
       ConfigurationSelectionPreservesPhysicalAndFeatureRequirements) {
  loom_target_facts_t* available = nullptr;
  Project(kDeviceKeys[0], &available);
  ASSERT_NE(available, nullptr);
  loom_aie2p_target_facts_t requirement;
  Requirement(kContracts[2], nullptr, &requirement);
  requirement.base.storage.snapshot.offset_bitwidth = 64;
  EXPECT_FALSE(loom_target_facts_satisfy_specialization_requirement(
      available, &requirement.base));
  Requirement(kContracts[2], nullptr, &requirement);
  requirement.base.storage.config.contract_feature_bits = 1;
  EXPECT_FALSE(loom_target_facts_satisfy_specialization_requirement(
      available, &requirement.base));
  Requirement(kContracts[2], nullptr, &requirement);
  ++requirement.base.storage.config.contract_set.stable_id;
  EXPECT_FALSE(loom_target_facts_satisfy_specialization_requirement(
      available, &requirement.base));
}

}  // namespace
