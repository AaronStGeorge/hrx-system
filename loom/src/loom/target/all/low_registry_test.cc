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
#include "loom/target/low_descriptor_registry_manifest.h"
#include "loom/target/testing/low_descriptor_registry_verify.h"

namespace loom {
namespace {

TEST(AllLowRegistryTest, VerifiesEveryLinkedTargetPackage) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_all_low_descriptor_registry_initialize(&registry);

  EXPECT_EQ(registry.registry.descriptor_set_provider_count, 8u);
  EXPECT_EQ(registry.target_bundle_count, 8u);
  IREE_ASSERT_OK(loom_target_low_descriptor_registry_verify(
      &registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));

  struct ExpectedBundle {
    const char* bundle_key;
    const char* descriptor_set_key;
  };
  const ExpectedBundle expected_bundles[] = {
      {"iree-vm", "iree.vm.core"},
      {"wasm-simd128", "wasm.core.simd128"},
      {"x86-avx512", "x86.avx512.core"},
      {"x86-packed-dot", "x86.packed_dot.core"},
      {"amdgpu-gfx950", "amdgpu.gfx950.core"},
      {"amdgpu-gfx11", "amdgpu.gfx11.core"},
      {"amdgpu-gfx12", "amdgpu.gfx12.core"},
      {"amdgpu-gfx1250", "amdgpu.gfx1250.core"},
  };
  for (const ExpectedBundle& expected : expected_bundles) {
    const loom_target_bundle_t* bundle = nullptr;
    IREE_ASSERT_OK(loom_target_low_descriptor_registry_lookup_bundle(
        &registry, iree_make_cstring_view(expected.bundle_key), &bundle))
        << expected.bundle_key;
    ASSERT_NE(bundle, nullptr) << expected.bundle_key;

    const loom_low_descriptor_set_t* descriptor_set = nullptr;
    IREE_ASSERT_OK(loom_target_low_descriptor_set_select_for_bundle(
        &registry.registry, bundle, &descriptor_set))
        << expected.bundle_key;
    ASSERT_NE(descriptor_set, nullptr) << expected.bundle_key;
    iree_string_view_t descriptor_set_key = loom_low_descriptor_set_string(
        descriptor_set, descriptor_set->key_string_offset);
    EXPECT_TRUE(iree_string_view_equal(
        descriptor_set_key,
        iree_make_cstring_view(expected.descriptor_set_key)))
        << expected.bundle_key;
  }

  const loom_target_bundle_t* test_bundle = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_target_low_descriptor_registry_lookup_bundle(
          &registry, iree_make_cstring_view("test-low"), &test_bundle));
}

TEST(AllLowRegistryTest, FormatsRegistryManifest) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_all_low_descriptor_registry_initialize(&registry);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_target_low_descriptor_registry_format_manifest_json(
      &registry, &builder));
  EXPECT_NE(iree_string_builder_size(&builder), 0u);
  iree_string_builder_deinitialize(&builder);
}

}  // namespace
}  // namespace loom
