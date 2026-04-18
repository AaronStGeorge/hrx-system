// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/avx512_descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

const loom_low_descriptor_t* LookupDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  uint32_t ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_EXPECT_OK(
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, key, &ordinal));
  return loom_low_descriptor_set_descriptor_at(descriptor_set, ordinal);
}

TEST(X86DescriptorsTest, Avx512CoreDescriptorSetVerifies) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_avx512_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(descriptor_set));

  iree_string_view_t set_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &set_key));
  EXPECT_TRUE(iree_string_view_equal(set_key, IREE_SV("x86.avx512.core")));

  EXPECT_GE(descriptor_set->descriptor_count, 7u);
  EXPECT_EQ(descriptor_set->descriptor_ref_count,
            descriptor_set->descriptor_count);
  EXPECT_EQ(descriptor_set->reg_class_count, 3u);
  EXPECT_GE(descriptor_set->schedule_class_count, 6u);
  EXPECT_GE(descriptor_set->resource_count, 7u);
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    EXPECT_NE(
        descriptor_set->reg_classes[i].flags & LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
        0u);
    EXPECT_GT(descriptor_set->reg_classes[i].physical_count, 0u);
  }
}

TEST(X86DescriptorsTest, Avx512CoreDescriptorLookupUsesStableKeys) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_avx512_core_descriptor_set();

  const loom_low_descriptor_t* add_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.vpaddd.zmm"));
  ASSERT_NE(add_descriptor, nullptr);
  iree_string_view_t add_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, add_descriptor->key_string_offset, &add_key));
  EXPECT_TRUE(
      iree_string_view_equal(add_key, IREE_SV("x86.avx512.vpaddd.zmm")));
  EXPECT_EQ(add_descriptor->operand_count, 3u);
  EXPECT_EQ(add_descriptor->result_count, 1u);

  const loom_low_descriptor_t* mask_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.kandq"));
  ASSERT_NE(mask_descriptor, nullptr);
  EXPECT_EQ(mask_descriptor->operand_count, 3u);
  EXPECT_EQ(mask_descriptor->result_count, 1u);
  const loom_low_operand_t* mask_operands =
      &descriptor_set->operands[mask_descriptor->operand_start];
  EXPECT_EQ(mask_operands[0].role, LOOM_LOW_OPERAND_ROLE_RESULT);
  EXPECT_EQ(mask_operands[1].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
  EXPECT_EQ(mask_operands[2].role, LOOM_LOW_OPERAND_ROLE_OPERAND);

  const loom_low_descriptor_t* branch_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.jmp"));
  ASSERT_NE(branch_descriptor, nullptr);
  EXPECT_EQ(branch_descriptor->operand_count, 0u);
  EXPECT_EQ(branch_descriptor->immediate_count, 1u);
  EXPECT_EQ(branch_descriptor->effect_count, 1u);
  EXPECT_NE(branch_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
            0u);
  EXPECT_NE(branch_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR, 0u);
  const loom_low_immediate_t* target_block =
      &descriptor_set->immediates[branch_descriptor->immediate_start];
  EXPECT_EQ(target_block->kind, LOOM_LOW_IMMEDIATE_KIND_ORDINAL);
  EXPECT_NE(target_block->flags & LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC, 0u);
  EXPECT_NE(target_block->flags & LOOM_LOW_IMMEDIATE_FLAG_RELATIVE, 0u);
  const loom_low_effect_t* branch_effect =
      &descriptor_set->effects[branch_descriptor->effect_start];
  EXPECT_EQ(branch_effect->kind, LOOM_LOW_EFFECT_KIND_CONTROL);
  EXPECT_NE(branch_effect->flags & LOOM_LOW_EFFECT_FLAG_ORDERED, 0u);
  const loom_low_schedule_class_t* branch_schedule =
      &descriptor_set->schedule_classes[branch_descriptor->schedule_class_id];
  EXPECT_NE(branch_schedule->flags & LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL, 0u);
}

TEST(X86DescriptorsTest, Avx512MemoryPacketsModelAddressAndDataResources) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_avx512_core_descriptor_set();

  const loom_low_descriptor_t* load_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("x86.avx512.vmovdqu32.load.zmm"));
  ASSERT_NE(load_descriptor, nullptr);
  EXPECT_EQ(load_descriptor->operand_count, 2u);
  EXPECT_EQ(load_descriptor->result_count, 1u);
  EXPECT_EQ(load_descriptor->immediate_count, 1u);
  EXPECT_EQ(load_descriptor->effect_count, 1u);
  EXPECT_NE(load_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
            0u);

  const loom_low_immediate_t* displacement =
      &descriptor_set->immediates[load_descriptor->immediate_start];
  EXPECT_EQ(displacement->kind, LOOM_LOW_IMMEDIATE_KIND_SIGNED);
  EXPECT_EQ(displacement->bit_width, 32u);
  EXPECT_EQ(displacement->signed_min, -(INT64_C(1) << 31));
  EXPECT_EQ(displacement->unsigned_max, (UINT64_C(1) << 31) - 1);

  const loom_low_effect_t* load_effect =
      &descriptor_set->effects[load_descriptor->effect_start];
  EXPECT_EQ(load_effect->kind, LOOM_LOW_EFFECT_KIND_READ);
  EXPECT_EQ(load_effect->memory_space, LOOM_LOW_MEMORY_SPACE_GENERIC);
  EXPECT_EQ(load_effect->width_bits, 512u);
  EXPECT_NE(load_effect->flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);

  const loom_low_schedule_class_t* load_schedule =
      &descriptor_set->schedule_classes[load_descriptor->schedule_class_id];
  EXPECT_NE(load_schedule->flags & LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD, 0u);
  ASSERT_EQ(load_schedule->issue_use_count, 2u);
  const loom_low_issue_use_t* load_issue_uses =
      &descriptor_set->issue_uses[load_schedule->issue_use_start];
  EXPECT_EQ(descriptor_set->resources[load_issue_uses[0].resource_id].kind,
            LOOM_LOW_RESOURCE_KIND_ADDRESS);
  EXPECT_EQ(descriptor_set->resources[load_issue_uses[1].resource_id].kind,
            LOOM_LOW_RESOURCE_KIND_LOAD);

  const loom_low_descriptor_t* store_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("x86.avx512.vmovdqu32.store.zmm"));
  ASSERT_NE(store_descriptor, nullptr);
  EXPECT_EQ(store_descriptor->result_count, 0u);
  EXPECT_EQ(store_descriptor->effect_count, 1u);
  const loom_low_effect_t* store_effect =
      &descriptor_set->effects[store_descriptor->effect_start];
  EXPECT_EQ(store_effect->kind, LOOM_LOW_EFFECT_KIND_WRITE);
}

TEST(X86DescriptorsTest, Avx512DotPacketsExposeDestructiveAccumulator) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_avx512_core_descriptor_set();

  const loom_low_descriptor_t* dot_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.vpdpbusd.zmm"));
  ASSERT_NE(dot_descriptor, nullptr);
  EXPECT_EQ(dot_descriptor->operand_count, 4u);
  EXPECT_EQ(dot_descriptor->result_count, 1u);
  EXPECT_EQ(dot_descriptor->constraint_count, 2u);
  EXPECT_NE(dot_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE,
            0u);

  const loom_low_operand_t* dot_operands =
      &descriptor_set->operands[dot_descriptor->operand_start];
  EXPECT_EQ(dot_operands[0].unit_count, 1u);
  EXPECT_EQ(dot_operands[1].unit_count, 1u);
  EXPECT_EQ(dot_operands[2].unit_count, 1u);
  EXPECT_EQ(dot_operands[3].unit_count, 1u);

  const loom_low_constraint_t* dot_constraints =
      &descriptor_set->constraints[dot_descriptor->constraint_start];
  EXPECT_EQ(dot_constraints[0].kind, LOOM_LOW_CONSTRAINT_KIND_TIED);
  EXPECT_EQ(dot_constraints[0].lhs_operand_index, 0u);
  EXPECT_EQ(dot_constraints[0].rhs_operand_index, 1u);
  EXPECT_EQ(dot_constraints[1].kind, LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE);
  EXPECT_EQ(dot_constraints[1].lhs_operand_index, 0u);
  EXPECT_EQ(dot_constraints[1].rhs_operand_index, 1u);

  const loom_low_descriptor_t* bf16_dot_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.vdpbf16ps.zmm"));
  ASSERT_NE(bf16_dot_descriptor, nullptr);
  EXPECT_EQ(bf16_dot_descriptor->constraint_count, 2u);
}

TEST(X86DescriptorsTest, ManifestNamesVectorMemoryAndDotPackets) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_avx512_core_descriptor_set();

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      loom_low_descriptor_set_format_manifest_json(descriptor_set, &builder));
  std::string json(iree_string_builder_buffer(&builder),
                   iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_NE(json.find("\"key\":\"x86.avx512.core\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"x86.avx512.vpaddd.zmm\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"x86.avx512.vmovdqu32.load.zmm\""),
            std::string::npos);
  EXPECT_NE(json.find("\"key\":\"x86.avx512.vpdpbusd.zmm\""),
            std::string::npos);
  EXPECT_NE(json.find("\"key\":\"x86.avx512.vdpbf16ps.zmm\""),
            std::string::npos);
  EXPECT_NE(json.find("\"key\":\"x86.avx512.kandq\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"x86.avx512.jmp\""), std::string::npos);
  EXPECT_NE(json.find("\"schedule_class_name\":\"x86.vector.dot\""),
            std::string::npos);
  EXPECT_NE(json.find("\"resource_name\":\"x86.vector.dot\""),
            std::string::npos);
  EXPECT_NE(json.find("\"field\":\"acc\",\"role\":2,\"role_name\":\"operand\""),
            std::string::npos);
  EXPECT_NE(json.find("\"kind_name\":\"destructive\""), std::string::npos);
  EXPECT_NE(json.find("\"descriptor_refs\""), std::string::npos);
}

}  // namespace
}  // namespace loom
