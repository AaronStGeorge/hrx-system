// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower.h"
#include "loom/target/test/lower.h"

namespace loom {
namespace {

static loom_low_lower_policy_registry_t MakeTestPolicyRegistry() {
  loom_low_lower_policy_registry_t registry = {};
  loom_test_low_lower_policy_registry_initialize(&registry);
  return registry;
}

TEST(LowLowerPolicyRegistryTest, LooksUpPolicyByContractKey) {
  loom_low_lower_policy_registry_t registry = MakeTestPolicyRegistry();
  IREE_ASSERT_OK(loom_low_lower_policy_registry_verify(&registry));

  const loom_low_lower_policy_t* policy = nullptr;
  IREE_ASSERT_OK(loom_low_lower_policy_registry_lookup(
      &registry, IREE_SV("test.low.core"), &policy));
  EXPECT_EQ(policy, loom_test_low_lower_policy());
}

TEST(LowLowerPolicyRegistryTest, LooksUpPolicyForTargetBundle) {
  loom_low_lower_policy_registry_t registry = MakeTestPolicyRegistry();
  loom_target_config_t config = {};
  config.contract_set_key = IREE_SV("test.low.core");
  loom_target_bundle_t bundle = {};
  bundle.name = IREE_SV("test-low");
  bundle.config = &config;

  const loom_low_lower_policy_t* policy = nullptr;
  IREE_ASSERT_OK(loom_low_lower_policy_registry_lookup_for_bundle(
      &registry, &bundle, &policy));
  EXPECT_EQ(policy, loom_test_low_lower_policy());
}

TEST(LowLowerPolicyRegistryTest, RejectsMissingContractKey) {
  loom_low_lower_policy_registry_t registry = MakeTestPolicyRegistry();

  const loom_low_lower_policy_t* policy = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_low_lower_policy_registry_lookup(
                            &registry, IREE_SV("missing-target"), &policy));
  EXPECT_EQ(policy, nullptr);
}

TEST(LowLowerPolicyRegistryTest, RejectsMalformedRegistries) {
  loom_low_lower_policy_registry_t missing_entries = {};
  missing_entries.entry_count = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_low_lower_policy_registry_verify(&missing_entries));

  const loom_low_lower_policy_registry_entry_t empty_key_entries[] = {
      {
          .contract_set_key = IREE_SVL(""),
          .policy = loom_test_low_lower_policy(),
      },
  };
  loom_low_lower_policy_registry_t empty_key_registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &empty_key_registry, empty_key_entries,
      IREE_ARRAYSIZE(empty_key_entries));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_low_lower_policy_registry_verify(&empty_key_registry));

  const loom_low_lower_policy_t incomplete_policy = {
      .name = IREE_SVL("incomplete"),
  };
  const loom_low_lower_policy_registry_entry_t incomplete_entries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = &incomplete_policy,
      },
  };
  loom_low_lower_policy_registry_t incomplete_registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &incomplete_registry, incomplete_entries,
      IREE_ARRAYSIZE(incomplete_entries));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_low_lower_policy_registry_verify(&incomplete_registry));
}

TEST(LowLowerPolicyRegistryTest, RejectsDuplicateContractKeys) {
  const loom_low_lower_policy_registry_entry_t entries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = loom_test_low_lower_policy(),
      },
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = loom_test_low_lower_policy(),
      },
  };
  loom_low_lower_policy_registry_t registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &registry, entries, IREE_ARRAYSIZE(entries));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        loom_low_lower_policy_registry_verify(&registry));

  const loom_low_lower_policy_t* policy = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        loom_low_lower_policy_registry_lookup(
                            &registry, IREE_SV("test.low.core"), &policy));
  EXPECT_EQ(policy, nullptr);
}

}  // namespace
}  // namespace loom
