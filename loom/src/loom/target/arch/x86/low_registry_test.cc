// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/low_registry.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/requirements.h"

namespace loom {
namespace {

void ExpectBundleSelectsDescriptorSet(
    const loom_target_low_descriptor_registry_t* registry,
    iree_string_view_t bundle_key,
    const loom_target_bundle_t* expected_bundle) {
  const loom_target_bundle_t* bundle = nullptr;
  IREE_ASSERT_OK(loom_target_low_descriptor_registry_lookup_bundle(
      registry, bundle_key, &bundle));
  ASSERT_EQ(bundle, expected_bundle);
  ASSERT_NE(bundle->snapshot, nullptr);
  ASSERT_NE(bundle->export_plan, nullptr);
  ASSERT_NE(bundle->config, nullptr);
  EXPECT_EQ(bundle->snapshot->codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE);
  EXPECT_EQ(bundle->snapshot->artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(bundle->export_plan->abi_kind, LOOM_TARGET_ABI_OBJECT_FUNCTION);

  const loom_low_descriptor_set_t* descriptor_set = nullptr;
  IREE_ASSERT_OK(loom_target_low_descriptor_set_select_for_bundle(
      &registry->registry, bundle,
      LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION, &descriptor_set));
  ASSERT_NE(descriptor_set, nullptr);
}

TEST(X86LowRegistryTest, VerifiesLinkedRegistryPackage) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_x86_low_descriptor_registry_initialize(&registry);

  EXPECT_EQ(registry.registry.descriptor_set_provider_count, 2u);
  EXPECT_EQ(registry.target_bundle_count, 2u);
  IREE_ASSERT_OK(loom_target_low_descriptor_registry_verify(
      &registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));

  ExpectBundleSelectsDescriptorSet(&registry, IREE_SV("x86-avx512"),
                                   &loom_x86_low_target_bundle_avx512_core);
  ExpectBundleSelectsDescriptorSet(&registry, IREE_SV("x86-packed-dot"),
                                   &loom_x86_low_target_bundle_packed_dot_core);
}

}  // namespace
}  // namespace loom
