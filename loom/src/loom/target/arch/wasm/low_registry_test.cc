// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/low_registry.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/requirements.h"
#include "loom/target/testing/low_descriptor_registry_verify.h"

namespace loom {
namespace {

TEST(WasmLowRegistryTest, VerifiesLinkedRegistryPackage) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_wasm_low_descriptor_registry_initialize(&registry);

  EXPECT_EQ(registry.registry.descriptor_set_provider_count, 1u);
  EXPECT_EQ(registry.target_bundle_count, 1u);
  IREE_ASSERT_OK(loom_target_low_descriptor_registry_verify(
      &registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));

  const loom_target_bundle_t* bundle = nullptr;
  IREE_ASSERT_OK(loom_target_low_descriptor_registry_lookup_bundle(
      &registry, IREE_SV("wasm-simd128"), &bundle));
  ASSERT_EQ(bundle, &loom_wasm_low_target_bundle_core_simd128);
  ASSERT_NE(bundle->snapshot, nullptr);
  ASSERT_NE(bundle->export_plan, nullptr);
  ASSERT_NE(bundle->config, nullptr);
  EXPECT_EQ(bundle->snapshot->codegen_format, LOOM_TARGET_CODEGEN_FORMAT_WASM);
  EXPECT_EQ(bundle->snapshot->artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY);
  EXPECT_EQ(bundle->export_plan->abi_kind, LOOM_TARGET_ABI_WASM_FUNCTION);

  const loom_low_descriptor_set_t* descriptor_set = nullptr;
  IREE_ASSERT_OK(loom_target_low_descriptor_set_select_for_bundle(
      &registry.registry, bundle, &descriptor_set));
  ASSERT_NE(descriptor_set, nullptr);
  iree_string_view_t descriptor_set_key = loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset);
  EXPECT_TRUE(
      iree_string_view_equal(descriptor_set_key, IREE_SV("wasm.core.simd128")));
}

}  // namespace
}  // namespace loom
