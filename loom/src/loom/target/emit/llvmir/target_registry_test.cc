// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/target_registry.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

bool StringViewEqual(iree_string_view_t lhs, const char* rhs) {
  return iree_string_view_equal(lhs, iree_make_cstring_view(rhs));
}

bool HasLegalityProvider(
    const loom_llvmir_target_legality_provider_list_t& providers,
    const char* name) {
  for (iree_host_size_t i = 0; i < providers.provider_count; ++i) {
    if (StringViewEqual(providers.providers[i]->name, name)) return true;
  }
  return false;
}

bool HasLoweringProvider(const loom_llvmir_lowering_provider_list_t& providers,
                         const char* name) {
  for (iree_host_size_t i = 0; i < providers.provider_count; ++i) {
    if (StringViewEqual(providers.providers[i]->name, name)) return true;
  }
  return false;
}

TEST(LlvmIrTargetRegistryTest, LooksUpDefaultAndNativeBundles) {
  loom_llvmir_target_registry_t registry = {};
  loom_llvmir_target_registry_initialize(&registry);

  const char* names[] = {
      "test-object",
      "x86_64-object",
      "x86_64-packed-dot-object",
      "amdgpu-hal",
  };
  for (const char* name : names) {
    const loom_target_bundle_t* bundle = nullptr;
    ASSERT_TRUE(loom_llvmir_target_registry_lookup_bundle(
        &registry, iree_make_cstring_view(name), &bundle));
    ASSERT_NE(bundle, nullptr) << name;
    EXPECT_TRUE(StringViewEqual(bundle->name, name)) << name;
  }

  const loom_target_bundle_t* default_bundle = nullptr;
  ASSERT_TRUE(loom_llvmir_target_registry_lookup_bundle(
      &registry, iree_string_view_empty(), &default_bundle));
  ASSERT_NE(default_bundle, nullptr);
  EXPECT_TRUE(StringViewEqual(default_bundle->name, "test-object"));
}

TEST(LlvmIrTargetRegistryTest, LooksUpNativeProfilesAndOwners) {
  loom_llvmir_target_registry_t registry = {};
  loom_llvmir_target_registry_initialize(&registry);

  const loom_llvmir_target_profile_t* profile = nullptr;
  const loom_llvmir_target_profile_provider_t* provider = nullptr;
  ASSERT_TRUE(loom_llvmir_target_registry_lookup_profile_provider(
      &registry, IREE_SV("x86_64-packed-dot-object"), &profile, &provider));
  ASSERT_NE(profile, nullptr);
  ASSERT_NE(provider, nullptr);
  EXPECT_TRUE(StringViewEqual(profile->name, "x86_64-packed-dot-object"));
  EXPECT_TRUE(StringViewEqual(provider->name, "x86"));

  ASSERT_TRUE(loom_llvmir_target_registry_lookup_profile_provider(
      &registry, IREE_SV("amdgpu-hal"), &profile, &provider));
  ASSERT_NE(profile, nullptr);
  ASSERT_NE(provider, nullptr);
  EXPECT_TRUE(StringViewEqual(profile->name, "amdgpu-hal"));
  EXPECT_TRUE(StringViewEqual(provider->name, "amdgpu"));
}

TEST(LlvmIrTargetRegistryTest, SelectsLegalityAndLoweringProvidersByTarget) {
  loom_llvmir_target_registry_t registry = {};
  loom_llvmir_target_registry_initialize(&registry);

  const loom_target_bundle_t* x86_bundle = nullptr;
  ASSERT_TRUE(loom_llvmir_target_registry_lookup_bundle(
      &registry, IREE_SV("x86_64-packed-dot-object"), &x86_bundle));
  const loom_llvmir_target_profile_t* profile = nullptr;
  const loom_llvmir_target_profile_provider_t* provider = nullptr;
  EXPECT_TRUE(loom_llvmir_target_registry_project_bundle(&registry, x86_bundle,
                                                         &profile, &provider));
  loom_llvmir_target_legality_provider_list_t legality_providers = {};
  loom_llvmir_target_legality_provider_list_select(provider,
                                                   &legality_providers);
  EXPECT_TRUE(HasLegalityProvider(legality_providers, "x86"));
  EXPECT_FALSE(HasLegalityProvider(legality_providers, "amdgpu"));

  loom_llvmir_lowering_provider_list_t lowering_providers = {};
  loom_llvmir_lowering_provider_list_select(provider, &lowering_providers);
  EXPECT_TRUE(HasLoweringProvider(lowering_providers, "x86"));
  EXPECT_FALSE(HasLoweringProvider(lowering_providers, "amdgpu"));

  const loom_target_bundle_t* amdgpu_bundle = nullptr;
  ASSERT_TRUE(loom_llvmir_target_registry_lookup_bundle(
      &registry, IREE_SV("amdgpu-hal"), &amdgpu_bundle));
  EXPECT_TRUE(loom_llvmir_target_registry_project_bundle(
      &registry, amdgpu_bundle, &profile, &provider));
  loom_llvmir_target_legality_provider_list_select(provider,
                                                   &legality_providers);
  EXPECT_TRUE(HasLegalityProvider(legality_providers, "amdgpu"));
  EXPECT_FALSE(HasLegalityProvider(legality_providers, "x86"));
}

TEST(LlvmIrTargetRegistryTest, MatchesLlcRequirements) {
  loom_llvmir_target_registry_t registry = {};
  loom_llvmir_target_registry_initialize(&registry);

  const loom_llvmir_target_profile_provider_t* provider = nullptr;
  EXPECT_TRUE(loom_llvmir_target_registry_llc_requirement_provider(
      &registry, IREE_SV("llc-x86"), &provider));
  ASSERT_NE(provider, nullptr);
  EXPECT_TRUE(StringViewEqual(provider->name, "x86"));

  EXPECT_TRUE(loom_llvmir_target_registry_llc_requirement_provider(
      &registry, IREE_SV("llc-amdgpu"), &provider));
  ASSERT_NE(provider, nullptr);
  EXPECT_TRUE(StringViewEqual(provider->name, "amdgpu"));

  EXPECT_FALSE(loom_llvmir_target_registry_llc_requirement_provider(
      &registry, IREE_SV("llc-missing"), &provider));
  EXPECT_EQ(provider, nullptr);
  EXPECT_FALSE(loom_llvmir_target_registry_llc_requirement_provider(
      &registry, IREE_SV("llc"), &provider));
  EXPECT_EQ(provider, nullptr);
}

}  // namespace
}  // namespace loom
