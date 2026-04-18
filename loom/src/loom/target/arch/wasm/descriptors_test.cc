// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/wasm/descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(WasmDescriptorsTest, CoreSimd128DescriptorSetVerifies) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_wasm_core_simd128_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(descriptor_set));

  iree_string_view_t set_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &set_key));
  EXPECT_TRUE(iree_string_view_equal(set_key, IREE_SV("wasm.core.simd128")));

  EXPECT_GE(descriptor_set->descriptor_count, 12u);
  EXPECT_EQ(descriptor_set->descriptor_ref_count,
            descriptor_set->descriptor_count);
  EXPECT_GE(descriptor_set->reg_class_count, 5u);
  EXPECT_GE(descriptor_set->schedule_class_count, 6u);
}

TEST(WasmDescriptorsTest, CoreSimd128DescriptorLookupUsesStableKeys) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_wasm_core_simd128_descriptor_set();

  uint32_t add_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("wasm.i32x4.add"), &add_ordinal));
  EXPECT_NE(add_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  const loom_low_descriptor_t* add_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, add_ordinal);
  ASSERT_NE(add_descriptor, nullptr);
  iree_string_view_t add_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, add_descriptor->key_string_offset, &add_key));
  EXPECT_TRUE(iree_string_view_equal(add_key, IREE_SV("wasm.i32x4.add")));
  EXPECT_EQ(add_descriptor->operand_count, 3u);
  EXPECT_EQ(add_descriptor->result_count, 1u);

  uint32_t store_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("wasm.v128.store"), &store_ordinal));
  const loom_low_descriptor_t* store_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, store_ordinal);
  ASSERT_NE(store_descriptor, nullptr);
  EXPECT_EQ(store_descriptor->operand_count, 2u);
  EXPECT_EQ(store_descriptor->result_count, 0u);
  EXPECT_EQ(store_descriptor->effect_count, 1u);
  EXPECT_NE(store_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
            0u);
}

TEST(WasmDescriptorsTest, ManifestNamesSimdAndMemoryPackets) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_wasm_core_simd128_descriptor_set();

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      loom_low_descriptor_set_format_manifest_json(descriptor_set, &builder));
  std::string json(iree_string_builder_buffer(&builder),
                   iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_NE(json.find("\"key\":\"wasm.core.simd128\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"wasm.i32x4.add\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"wasm.v128.load\""), std::string::npos);
  EXPECT_NE(json.find("\"descriptor_refs\""), std::string::npos);
}

}  // namespace
}  // namespace loom
