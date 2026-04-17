// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

TEST(IreeVmDescriptorsTest, CoreDescriptorSetVerifies) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_ireevm_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(descriptor_set));

  iree_string_view_t set_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &set_key));
  EXPECT_TRUE(iree_string_view_equal(set_key, IREE_SV("iree.vm.core")));

  EXPECT_EQ(descriptor_set->descriptor_count, 9u);
  EXPECT_EQ(descriptor_set->reg_class_count, 6u);
  EXPECT_EQ(descriptor_set->schedule_class_count, 4u);
}

TEST(IreeVmDescriptorsTest, CoreDescriptorLookupUsesStableKeys) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_ireevm_core_descriptor_set();

  uint32_t add_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("iree.vm.add.i32"), &add_ordinal));
  EXPECT_EQ(add_ordinal, 1u);
  const loom_low_descriptor_t* add_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, add_ordinal);
  ASSERT_NE(add_descriptor, nullptr);
  EXPECT_EQ(add_descriptor->operand_count, 3u);
  EXPECT_EQ(add_descriptor->result_count, 1u);
  EXPECT_EQ(add_descriptor->schedule_class_id, 1u);

  uint32_t branch_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("iree.vm.cond_br.i32"), &branch_ordinal));
  const loom_low_descriptor_t* branch_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, branch_ordinal);
  ASSERT_NE(branch_descriptor, nullptr);
  EXPECT_EQ(branch_descriptor->immediate_count, 2u);
  EXPECT_EQ(branch_descriptor->effect_count, 1u);
  EXPECT_NE(branch_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR, 0u);
}

TEST(IreeVmDescriptorsTest, CoreDescriptorFingerprintMatchesEmbeddedValue) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_ireevm_core_descriptor_set();

  loom_low_fingerprint_t fingerprint = {};
  IREE_ASSERT_OK(loom_low_descriptor_set_compute_fingerprint(descriptor_set,
                                                             &fingerprint));
  EXPECT_NE(fingerprint.low, UINT64_C(0));
  EXPECT_NE(fingerprint.high, UINT64_C(0));

  bool matches = false;
  IREE_ASSERT_OK(
      loom_low_descriptor_set_fingerprint_matches(descriptor_set, &matches));
  EXPECT_TRUE(matches);
}

TEST(IreeVmDescriptorsTest, ManifestNamesCallAndControlPackets) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_ireevm_core_descriptor_set();

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      loom_low_descriptor_set_format_manifest_json(descriptor_set, &builder));
  std::string json(iree_string_builder_buffer(&builder),
                   iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_NE(json.find("\"key\":\"iree.vm.core\""), std::string::npos);
  EXPECT_NE(json.find("\"abi_version\":2"), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"iree.vm.call.import.i32\""),
            std::string::npos);
  EXPECT_NE(json.find("\"key\":\"iree.vm.cond_br.i32\""), std::string::npos);
  EXPECT_NE(json.find("\"schedule_class\":3"), std::string::npos);
}

}  // namespace
}  // namespace loom
