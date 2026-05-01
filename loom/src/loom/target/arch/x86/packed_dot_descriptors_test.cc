// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/packed_dot_descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/testing/descriptors_verify.h"
#include "loom/target/arch/x86/packed_dot_contract.h"

namespace loom {
namespace {

const loom_low_descriptor_t* LookupDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  uint32_t ordinal =
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, key);
  EXPECT_NE(ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  return loom_low_descriptor_set_descriptor_at(descriptor_set, ordinal);
}

std::string DescriptorString(const loom_low_descriptor_set_t* descriptor_set,
                             loom_bstring_table_offset_t string_offset) {
  iree_string_view_t value =
      loom_low_descriptor_set_string(descriptor_set, string_offset);
  return std::string(value.data, value.size);
}

TEST(X86PackedDotDescriptorsTest, DescriptorSetVerifies) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_packed_dot_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(descriptor_set));

  EXPECT_EQ(DescriptorString(descriptor_set, descriptor_set->key_string_offset),
            "x86.packed_dot.core");
  EXPECT_EQ(descriptor_set->descriptor_count,
            loom_x86_packed_dot_descriptor_count());
  EXPECT_EQ(descriptor_set->descriptor_ref_count,
            descriptor_set->descriptor_count);
  EXPECT_EQ(descriptor_set->reg_class_count, 3u);
  EXPECT_EQ(descriptor_set->resource_count, 1u);
  EXPECT_EQ(descriptor_set->schedule_class_count, 3u);
}

TEST(X86PackedDotDescriptorsTest, LowDescriptorsMirrorContractKeys) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_packed_dot_core_descriptor_set();

  for (iree_host_size_t i = 0; i < loom_x86_packed_dot_descriptor_count();
       ++i) {
    const loom_x86_packed_dot_descriptor_t* contract_descriptor =
        loom_x86_packed_dot_descriptor_at(i);
    ASSERT_NE(contract_descriptor, nullptr);

    const loom_low_descriptor_t* low_descriptor =
        LookupDescriptor(descriptor_set, contract_descriptor->name);
    ASSERT_NE(low_descriptor, nullptr);
    EXPECT_EQ(low_descriptor->stable_id, contract_descriptor->stable_id);
    EXPECT_NE(loom_low_descriptor_set_lookup_descriptor_by_id(
                  descriptor_set, contract_descriptor->stable_id),
              LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
    EXPECT_EQ(DescriptorString(descriptor_set,
                               low_descriptor->mnemonic_string_offset),
              std::string(contract_descriptor->instruction_mnemonic.data,
                          contract_descriptor->instruction_mnemonic.size));
    ASSERT_EQ(low_descriptor->feature_mask_word_count, 1u);
    EXPECT_EQ(descriptor_set
                  ->feature_mask_words[low_descriptor->feature_mask_word_start],
              contract_descriptor->required_feature_bits);
    EXPECT_EQ(low_descriptor->operand_count, 4u);
    EXPECT_EQ(low_descriptor->result_count, 1u);
    EXPECT_EQ(low_descriptor->constraint_count, 2u);
  }
}

TEST(X86PackedDotDescriptorsTest, RepresentativeContractsCarryLowMetadata) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_packed_dot_core_descriptor_set();

  const loom_low_descriptor_t* avx512_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512_vnni.vpdpbusd.zmm"));
  ASSERT_NE(avx512_descriptor, nullptr);
  EXPECT_EQ(DescriptorString(descriptor_set,
                             avx512_descriptor->semantic_tag_string_offset),
            "dot.u8s8.i32x16");
  EXPECT_EQ(
      descriptor_set
          ->feature_mask_words[avx512_descriptor->feature_mask_word_start],
      LOOM_X86_FEATURE_AVX512_VNNI);
  const loom_low_schedule_class_t* avx512_schedule =
      &descriptor_set->schedule_classes[avx512_descriptor->schedule_class_id];
  ASSERT_EQ(avx512_schedule->issue_use_count, 1u);
  const loom_low_issue_use_t* avx512_issue_use =
      &descriptor_set->issue_uses[avx512_schedule->issue_use_start];
  EXPECT_EQ(avx512_issue_use->units, 4u);

  const loom_low_descriptor_t* saturating_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("x86.avx512_vnni.vpdpbusds.zmm"));
  ASSERT_NE(saturating_descriptor, nullptr);
  EXPECT_EQ(DescriptorString(descriptor_set,
                             saturating_descriptor->semantic_tag_string_offset),
            "dot.u8s8.i32x16.sat");

  const loom_low_descriptor_t* int8_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("x86.avx_vnni_int8.vpdpbssd.ymm"));
  ASSERT_NE(int8_descriptor, nullptr);
  EXPECT_EQ(descriptor_set
                ->feature_mask_words[int8_descriptor->feature_mask_word_start],
            LOOM_X86_FEATURE_AVX_VNNI_INT8);
  const loom_low_schedule_class_t* int8_schedule =
      &descriptor_set->schedule_classes[int8_descriptor->schedule_class_id];
  ASSERT_EQ(int8_schedule->issue_use_count, 1u);
  const loom_low_issue_use_t* int8_issue_use =
      &descriptor_set->issue_uses[int8_schedule->issue_use_start];
  EXPECT_EQ(int8_issue_use->units, 2u);

  const loom_low_descriptor_t* avx10_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx10_2.vdpphps.zmm"));
  ASSERT_NE(avx10_descriptor, nullptr);
  EXPECT_EQ(DescriptorString(descriptor_set,
                             avx10_descriptor->semantic_tag_string_offset),
            "dot.f16f16.f32x16");
  EXPECT_EQ(descriptor_set
                ->feature_mask_words[avx10_descriptor->feature_mask_word_start],
            LOOM_X86_FEATURE_AVX10_2);

  const loom_low_descriptor_t* bf16_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("x86.avx512_bf16.vdpbf16ps.zmm"));
  ASSERT_NE(bf16_descriptor, nullptr);
  EXPECT_EQ(DescriptorString(descriptor_set,
                             bf16_descriptor->semantic_tag_string_offset),
            "dot.bf16bf16.f32x16");
  EXPECT_EQ(descriptor_set
                ->feature_mask_words[bf16_descriptor->feature_mask_word_start],
            LOOM_X86_FEATURE_AVX512_BF16);
}

}  // namespace
}  // namespace loom
