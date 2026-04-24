// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/testing/text_asm_roundtrip_test_util.h"
#include "loom/codegen/low/testing/text_asm_test_util.h"

namespace loom {
namespace {

using ::loom::testing::LowTextAsmRoundTripHarness;
using ::loom::testing::LowTextAsmTypeInferenceHarness;

constexpr uint16_t kVmOpcodeAddI32 = 0x22;
constexpr uint16_t kVmOpcodeCondBranch = 0x57;

TEST(IreeVmDescriptorsTest, CoreDescriptorSetVerifies) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_ireevm_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(descriptor_set));

  iree_string_view_t set_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &set_key));
  EXPECT_TRUE(iree_string_view_equal(set_key, IREE_SV("iree.vm.core")));

  EXPECT_GE(descriptor_set->descriptor_count, 9u);
  EXPECT_GE(descriptor_set->reg_class_count, 6u);
  EXPECT_GE(descriptor_set->schedule_class_count, 4u);
}

TEST(IreeVmDescriptorsTest, CoreDescriptorLookupUsesStableKeys) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_ireevm_core_descriptor_set();

  uint32_t add_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("iree.vm.add.i32"), &add_ordinal));
  EXPECT_NE(add_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  const loom_low_descriptor_t* add_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, add_ordinal);
  ASSERT_NE(add_descriptor, nullptr);
  iree_string_view_t add_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, add_descriptor->key_string_offset, &add_key));
  EXPECT_TRUE(iree_string_view_equal(add_key, IREE_SV("iree.vm.add.i32")));
  EXPECT_EQ(add_descriptor->encoding_id, kVmOpcodeAddI32);
  EXPECT_EQ(add_descriptor->operand_count, 3u);
  EXPECT_EQ(add_descriptor->result_count, 1u);
  ASSERT_NE(add_descriptor->schedule_class_id, LOOM_LOW_SCHEDULE_CLASS_NONE);
  ASSERT_LT(add_descriptor->schedule_class_id,
            descriptor_set->schedule_class_count);
  iree_string_view_t add_schedule_name = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set,
      descriptor_set->schedule_classes[add_descriptor->schedule_class_id]
          .name_string_offset,
      &add_schedule_name));
  EXPECT_TRUE(iree_string_view_equal(add_schedule_name, IREE_SV("vm.alu.i32")));

  uint32_t branch_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      descriptor_set, IREE_SV("iree.vm.cond_br.i32"), &branch_ordinal));
  const loom_low_descriptor_t* branch_descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, branch_ordinal);
  ASSERT_NE(branch_descriptor, nullptr);
  EXPECT_EQ(branch_descriptor->encoding_id, kVmOpcodeCondBranch);
  EXPECT_EQ(branch_descriptor->immediate_count, 2u);
  EXPECT_EQ(branch_descriptor->effect_count, 1u);
  EXPECT_NE(branch_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR, 0u);
}

TEST(IreeVmDescriptorsTest, LowAsmInfersScalarResultType) {
  LowTextAsmTypeInferenceHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_ireevm_core_descriptor_set));

  loom_text_low_asm_packet_descriptor_t packet = {};
  IREE_ASSERT_OK(harness.LookupPacket(IREE_SV("iree.vm.core"),
                                      IREE_SV("vm.add.i32"), &packet));

  loom_type_t result_type = loom_type_none();
  iree_string_view_t diagnostic_detail = iree_string_view_empty();
  IREE_ASSERT_OK(harness.InferResultType(
      &packet, /*operands=*/nullptr, /*operand_count=*/0, /*result_index=*/0,
      &result_type, &diagnostic_detail));
  EXPECT_TRUE(iree_string_view_is_empty(diagnostic_detail));
  EXPECT_TRUE(harness.RegisterTypeEquals(result_type, IREE_SV("vm.i32"), 1));
}

TEST(IreeVmDescriptorsTest, LowAsmRegionRoundTrips) {
  LowTextAsmRoundTripHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_ireevm_core_descriptor_set));

  const char* source =
      "test.low_asm_region asm<iree.vm.core> {\n"
      "  %c0 = vm.const.i32 7\n"
      "  %sum = vm.add.i32 %c0, %c0\n"
      "  return %sum\n"
      "}\n";
  std::string printed;
  IREE_ASSERT_OK(
      harness.RoundTrip(IREE_SV(source), IREE_SV("iree.vm.core"), &printed));
  EXPECT_EQ(printed, source);
}

}  // namespace
}  // namespace loom
