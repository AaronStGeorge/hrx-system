// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/all/low_registry.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/requirements.h"

namespace loom {
namespace {

TEST(AllLowRegistryTest, VerifiesEveryLinkedTargetPackage) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_all_low_descriptor_registry_initialize(&registry);

  EXPECT_EQ(registry.registry.descriptor_set_provider_count, 8u);
  EXPECT_EQ(registry.target_bundle_count, 8u);
  IREE_ASSERT_OK(loom_target_low_descriptor_registry_verify(
      &registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));

  const char* expected_bundle_keys[] = {
      "iree-vm",       "wasm-simd128", "x86-avx512",   "x86-packed-dot",
      "amdgpu-gfx950", "amdgpu-gfx11", "amdgpu-gfx12", "amdgpu-gfx1250",
  };
  for (const char* expected_bundle_key : expected_bundle_keys) {
    const loom_target_bundle_t* bundle = nullptr;
    IREE_ASSERT_OK(loom_target_low_descriptor_registry_lookup_bundle(
        &registry, iree_make_cstring_view(expected_bundle_key), &bundle))
        << expected_bundle_key;
    ASSERT_NE(bundle, nullptr) << expected_bundle_key;

    const loom_low_descriptor_set_t* descriptor_set = nullptr;
    IREE_ASSERT_OK(loom_target_low_descriptor_set_select_for_bundle(
        &registry.registry, bundle,
        LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION, &descriptor_set))
        << expected_bundle_key;
    ASSERT_NE(descriptor_set, nullptr) << expected_bundle_key;
  }

  const loom_target_bundle_t* test_bundle = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_target_low_descriptor_registry_lookup_bundle(
          &registry, iree_make_cstring_view("test-low"), &test_bundle));
}

TEST(AllLowRegistryTest, ManifestNamesRepresentativeTargets) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_all_low_descriptor_registry_initialize(&registry);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_target_low_descriptor_registry_format_manifest_json(
      &registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
      &builder));
  std::string json(iree_string_builder_buffer(&builder),
                   iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_NE(json.find("\"key\":\"iree.vm.core\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"wasm.core.simd128\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"x86.avx512.core\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.gfx11.core\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.gfx1250.core\""), std::string::npos);
}

}  // namespace
}  // namespace loom
