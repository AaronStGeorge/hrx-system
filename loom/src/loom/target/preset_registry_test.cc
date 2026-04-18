// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/preset_registry.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(TargetPresetRegistryTest, LooksUpPresetBundleByExactKey) {
  loom_target_bundle_t vm_bundle = {};
  vm_bundle.name = IREE_SV("iree-vm");
  const loom_target_bundle_t* bundles[] = {&vm_bundle};
  loom_target_preset_registry_t registry = {};
  registry.target_bundles = bundles;
  registry.target_bundle_count = 1;

  const loom_target_bundle_t* bundle = nullptr;
  IREE_ASSERT_OK(loom_target_preset_registry_lookup_bundle(
      &registry, IREE_SV("iree-vm"), &bundle));
  EXPECT_EQ(bundle, &vm_bundle);
}

TEST(TargetPresetRegistryTest, RejectsEmptyKey) {
  loom_target_preset_registry_t registry = {};

  const loom_target_bundle_t* bundle = nullptr;
  iree_status_t status = loom_target_preset_registry_lookup_bundle(
      &registry, IREE_SV(""), &bundle);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(bundle, nullptr);
}

TEST(TargetPresetRegistryTest, RejectsUnknownKey) {
  loom_target_bundle_t vm_bundle = {};
  vm_bundle.name = IREE_SV("iree-vm");
  const loom_target_bundle_t* bundles[] = {&vm_bundle};
  loom_target_preset_registry_t registry = {};
  registry.target_bundles = bundles;
  registry.target_bundle_count = 1;

  const loom_target_bundle_t* bundle = nullptr;
  iree_status_t status = loom_target_preset_registry_lookup_bundle(
      &registry, IREE_SV("missing-target"), &bundle);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(bundle, nullptr);
}

TEST(TargetPresetRegistryTest, RejectsNullBundleTable) {
  loom_target_preset_registry_t registry = {};
  registry.target_bundle_count = 1;

  const loom_target_bundle_t* bundle = nullptr;
  iree_status_t status = loom_target_preset_registry_lookup_bundle(
      &registry, IREE_SV("iree-vm"), &bundle);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(bundle, nullptr);
}

TEST(TargetPresetRegistryTest, RejectsNullBundleRow) {
  const loom_target_bundle_t* bundles[] = {nullptr};
  loom_target_preset_registry_t registry = {};
  registry.target_bundles = bundles;
  registry.target_bundle_count = 1;

  const loom_target_bundle_t* bundle = nullptr;
  iree_status_t status = loom_target_preset_registry_lookup_bundle(
      &registry, IREE_SV("iree-vm"), &bundle);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(bundle, nullptr);
}

}  // namespace
}  // namespace loom
