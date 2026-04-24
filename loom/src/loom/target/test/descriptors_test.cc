// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/test/descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/testing/text_asm_roundtrip_test_util.h"
#include "loom/target/test/alt_descriptors.h"

namespace loom {
namespace {

using ::loom::testing::LowTextAsmRoundTripHarness;

std::string ToString(const loom_low_descriptor_set_t* descriptor_set,
                     loom_bstring_table_offset_t string_offset) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(
      loom_low_descriptor_set_string(descriptor_set, string_offset, &value));
  return std::string(value.data, value.size);
}

const loom_low_descriptor_t* LookupDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_EXPECT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, key, &descriptor_ordinal));
  EXPECT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  return loom_low_descriptor_set_descriptor_at(descriptor_set,
                                               descriptor_ordinal);
}

TEST(TestLowDescriptorsTest, CoreDescriptorSetVerifies) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(descriptor_set));

  EXPECT_EQ(ToString(descriptor_set, descriptor_set->key_string_offset),
            "test.low.core");
  EXPECT_EQ(descriptor_set->descriptor_ref_count,
            descriptor_set->descriptor_count);
  EXPECT_GE(descriptor_set->descriptor_count, 12u);
  EXPECT_GE(descriptor_set->reg_class_count, 4u);
  EXPECT_GE(descriptor_set->schedule_class_count, 7u);
  EXPECT_GE(descriptor_set->resource_count, 7u);
}

TEST(TestLowDescriptorsTest, AltDescriptorSetVerifies) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_alt_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(descriptor_set));

  EXPECT_EQ(ToString(descriptor_set, descriptor_set->key_string_offset),
            "test.low.alt");
  EXPECT_EQ(descriptor_set->descriptor_ref_count,
            descriptor_set->descriptor_count);
  EXPECT_GE(descriptor_set->descriptor_count, 2u);
  EXPECT_GE(descriptor_set->reg_class_count, 1u);
  EXPECT_GE(descriptor_set->schedule_class_count, 2u);
  EXPECT_GE(descriptor_set->resource_count, 1u);
}

TEST(TestLowDescriptorsTest, StableKeysCoverGenericLowBehaviors) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();

  struct ExpectedDescriptor {
    iree_string_view_t key;
    uint16_t operand_count;
    uint16_t result_count;
    uint16_t effect_count;
  };
  const ExpectedDescriptor expected_descriptors[] = {
      {IREE_SV("test.add.i32"), 3, 1, 0},
      {IREE_SV("test.ambiguous"), 1, 1, 0},
      {IREE_SV("test.add.v4i32"), 3, 1, 0},
      {IREE_SV("test.add.phys"), 3, 1, 0},
      {IREE_SV("test.add.special"), 3, 1, 0},
      {IREE_SV("test.tied.any"), 2, 1, 0},
      {IREE_SV("test.load.v4i32"), 2, 1, 1},
      {IREE_SV("test.store.v4i32"), 2, 0, 1},
      {IREE_SV("test.call.i32"), 2, 1, 1},
  };

  for (const ExpectedDescriptor& expected : expected_descriptors) {
    const loom_low_descriptor_t* descriptor =
        LookupDescriptor(descriptor_set, expected.key);
    ASSERT_NE(descriptor, nullptr);
    EXPECT_EQ(descriptor->operand_count, expected.operand_count)
        << expected.key.data;
    EXPECT_EQ(descriptor->result_count, expected.result_count)
        << expected.key.data;
    EXPECT_EQ(descriptor->effect_count, expected.effect_count)
        << expected.key.data;
  }
}

TEST(TestLowDescriptorsTest, LoadUsesAddressAndLoadResources) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  const loom_low_descriptor_t* descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("test.load.v4i32"));
  ASSERT_NE(descriptor, nullptr);

  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[descriptor->schedule_class_id];
  EXPECT_EQ(ToString(descriptor_set, schedule_class->name_string_offset),
            "test.load");
  ASSERT_EQ(schedule_class->issue_use_count, 2u);

  const loom_low_issue_use_t* address_use =
      &descriptor_set->issue_uses[schedule_class->issue_use_start];
  const loom_low_issue_use_t* load_use = address_use + 1;
  ASSERT_LT(address_use->resource_id, descriptor_set->resource_count);
  ASSERT_LT(load_use->resource_id, descriptor_set->resource_count);

  const loom_low_resource_t* address_resource =
      &descriptor_set->resources[address_use->resource_id];
  const loom_low_resource_t* load_resource =
      &descriptor_set->resources[load_use->resource_id];
  EXPECT_EQ(ToString(descriptor_set, address_resource->name_string_offset),
            "test.address");
  EXPECT_EQ(address_resource->kind, LOOM_LOW_RESOURCE_KIND_ADDRESS);
  EXPECT_EQ(address_use->stage, 0u);
  EXPECT_EQ(ToString(descriptor_set, load_resource->name_string_offset),
            "test.load");
  EXPECT_EQ(load_resource->kind, LOOM_LOW_RESOURCE_KIND_LOAD);
  EXPECT_EQ(load_use->stage, 1u);
}

TEST(TestLowDescriptorsTest, PhysicalClassHasTargetVisiblePressureBudget) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();

  bool found_physical_class = false;
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    const loom_low_reg_class_t* reg_class = &descriptor_set->reg_classes[i];
    if (ToString(descriptor_set, reg_class->name_string_offset) !=
        "test.phys") {
      continue;
    }
    found_physical_class = true;
    EXPECT_EQ(reg_class->alloc_unit_bits, 512u);
    EXPECT_EQ(reg_class->physical_count, 32u);
    EXPECT_NE(reg_class->flags & LOOM_LOW_REG_CLASS_FLAG_PHYSICAL, 0u);
    EXPECT_EQ(reg_class->spill_slot_space, LOOM_LOW_SPILL_SLOT_SPACE_STACK);
  }
  EXPECT_TRUE(found_physical_class);
}

TEST(TestLowDescriptorsTest, SpecialPhysicalClassIsUnspillable) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();

  bool found_special_class = false;
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    const loom_low_reg_class_t* reg_class = &descriptor_set->reg_classes[i];
    if (ToString(descriptor_set, reg_class->name_string_offset) !=
        "test.special") {
      continue;
    }
    found_special_class = true;
    EXPECT_EQ(reg_class->alloc_unit_bits, 32u);
    EXPECT_EQ(reg_class->physical_count, 1u);
    EXPECT_NE(reg_class->flags & LOOM_LOW_REG_CLASS_FLAG_PHYSICAL, 0u);
    EXPECT_NE(reg_class->flags & LOOM_LOW_REG_CLASS_FLAG_UNSPILLABLE, 0u);
    EXPECT_EQ(reg_class->spill_slot_space, LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE);
  }
  EXPECT_TRUE(found_special_class);
}

TEST(TestLowDescriptorsTest, AsmFormsExposeGenericAndSpirvLikePackets) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_test_low_core_descriptor_set();
  ASSERT_GE(descriptor_set->asm_form_count, 15u);

  uint32_t asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_asm_form(
      descriptor_set, IREE_SV("OpIAdd"), &asm_form_ordinal));
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_set_asm_form_at(descriptor_set, asm_form_ordinal);
  ASSERT_NE(asm_form, nullptr);
  EXPECT_EQ(asm_form->result_operand_index_count, 1u);
  EXPECT_EQ(asm_form->operand_index_count, 2u);

  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set,
                                            asm_form->descriptor_ordinal);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor_set, descriptor->key_string_offset),
            "test.spv.op_iadd.i32");
}

TEST(TestLowDescriptorsTest, SpirvLikeLowAsmRegionRoundTrips) {
  LowTextAsmRoundTripHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_test_low_core_descriptor_set));

  const char* source =
      "test.low_asm_region asm<test.low.core> {\n"
      "  %c0 = test.const.i32 7\n"
      "  %sum = test.add.i32 %c0, %c0\n"
      "  %spv = OpIAdd %sum, %c0\n"
      "  %call = test.call.i32 %spv {callee = 4}\n"
      "  return %call\n"
      "}\n";
  std::string printed;
  IREE_ASSERT_OK(
      harness.RoundTrip(IREE_SV(source), IREE_SV("test.low.core"), &printed));
  EXPECT_EQ(printed, source);
}

}  // namespace
}  // namespace loom
