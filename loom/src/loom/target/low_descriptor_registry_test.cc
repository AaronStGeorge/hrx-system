// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/low_descriptor_registry.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/requirements.h"

namespace loom {
namespace {

TEST(LowDescriptorRegistryTest, RegistryVerifiesSelectedTargetPackages) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_low_descriptor_registry_initialize(&registry);

  EXPECT_NE(registry.registry.descriptor_sets, nullptr);
  EXPECT_GT(registry.registry.descriptor_set_count, 0u);
  IREE_EXPECT_OK(loom_low_descriptor_registry_verify(&registry.registry));
}

TEST(LowDescriptorRegistryTest, RegistrySatisfiesTargetLowFoundation) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_low_descriptor_registry_initialize(&registry);

  IREE_ASSERT_OK(loom_low_descriptor_registry_verify_requirements(
      &registry.registry,
      LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
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

TEST(LowDescriptorRegistryTest, LooksUpRepresentativeDescriptors) {
  struct ExpectedDescriptor {
    // Descriptor-set key owning the representative descriptor.
    const char* set_key;
    // Descriptor key that must remain addressable through the selected set.
    const char* descriptor_key;
  };
  const ExpectedDescriptor expected_descriptors[] = {
      {"iree.vm.core", "iree.vm.add.i32"},
      {"iree.vm.core", "iree.vm.cond_br.i32"},
      {"wasm.core.simd128", "wasm.i32x4.add"},
      {"wasm.core.simd128", "wasm.v128.store"},
      {"x86.avx512.core", "x86.avx512.vpdpbusd.zmm"},
      {"x86.avx512.core", "x86.avx512.jmp"},
      {"x86.packed_dot.core", "x86.avx512-vnni.vpdpbusd.512"},
      {"x86.packed_dot.core", "x86.avx10.2.vdpphps.512"},
      {"amdgpu.gfx950.core", "amdgpu.v_add_u32"},
      {"amdgpu.gfx950.core", "amdgpu.v_mfma_f32_16x16x16_f16"},
      {"amdgpu.gfx11.core", "amdgpu.v_add_u32"},
      {"amdgpu.gfx11.core", "amdgpu.s_waitcnt_depctr"},
      {"amdgpu.gfx12.core", "amdgpu.s_wait_loadcnt"},
      {"amdgpu.gfx12.core", "amdgpu.v_wmma_f32_16x16x16_f16"},
      {"amdgpu.gfx1250.core", "amdgpu.s_wait_storecnt"},
      {"amdgpu.gfx1250.core", "amdgpu.v_wmma_scale_f32_16x16x128_f8f6f4_f8_f8"},
  };

  for (const ExpectedDescriptor& expected : expected_descriptors) {
    const loom_low_descriptor_set_t* descriptor_set = nullptr;
    IREE_ASSERT_OK(loom_target_low_descriptor_set_lookup(
        iree_make_cstring_view(expected.set_key), &descriptor_set));
    ASSERT_NE(descriptor_set, nullptr) << expected.set_key;

    uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
    IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
        descriptor_set, iree_make_cstring_view(expected.descriptor_key),
        &descriptor_ordinal))
        << expected.set_key << " :: " << expected.descriptor_key;
    EXPECT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE)
        << expected.set_key << " :: " << expected.descriptor_key;
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
