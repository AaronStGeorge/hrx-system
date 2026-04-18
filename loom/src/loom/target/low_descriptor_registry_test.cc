// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/low_descriptor_registry.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(LowDescriptorRegistryTest, RegistryVerifiesSelectedTargetPackages) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_low_descriptor_registry_initialize(&registry);

  EXPECT_NE(registry.registry.descriptor_sets, nullptr);
  EXPECT_GT(registry.registry.descriptor_set_count, 0u);
  IREE_EXPECT_OK(loom_low_descriptor_registry_verify(&registry.registry));
}

TEST(LowDescriptorRegistryTest, LooksUpRepresentativeDescriptorSets) {
  const char* keys[] = {
      "iree.vm.core",        "wasm.core.simd128",   "x86.avx512.core",
      "x86.packed_dot.core", "amdgpu.gfx950.core",  "amdgpu.gfx11.core",
      "amdgpu.gfx12.core",   "amdgpu.gfx1250.core",
  };

  for (const char* key : keys) {
    const loom_low_descriptor_set_t* descriptor_set = nullptr;
    IREE_ASSERT_OK(loom_target_low_descriptor_set_lookup(
        iree_make_cstring_view(key), &descriptor_set));
    ASSERT_NE(descriptor_set, nullptr) << key;

    iree_string_view_t set_key = iree_string_view_empty();
    IREE_ASSERT_OK(loom_low_descriptor_set_string(
        descriptor_set, descriptor_set->key_string_offset, &set_key));
    EXPECT_TRUE(iree_string_view_equal(set_key, iree_make_cstring_view(key)))
        << key;
  }
}

TEST(LowDescriptorRegistryTest, MissingKeyReturnsNullDescriptorSet) {
  const loom_low_descriptor_set_t* descriptor_set = nullptr;
  IREE_ASSERT_OK(loom_target_low_descriptor_set_lookup(
      IREE_SV("target.missing"), &descriptor_set));
  EXPECT_EQ(descriptor_set, nullptr);
}

}  // namespace
}  // namespace loom
