// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/gfx950_descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm_roundtrip_test_util.h"
#include "loom/codegen/low/text_asm_test_util.h"

namespace loom {
namespace {

using ::loom::testing::LowTextAsmRoundTripHarness;
using ::loom::testing::LowTextAsmTypeInferenceHarness;

const loom_low_descriptor_t* LookupDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  uint32_t ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_EXPECT_OK(
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, key, &ordinal));
  return loom_low_descriptor_set_descriptor_at(descriptor_set, ordinal);
}

TEST(AmdgpuDescriptorsTest, Gfx950CoreDescriptorSetVerifies) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx950_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(descriptor_set));

  iree_string_view_t set_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &set_key));
  EXPECT_TRUE(iree_string_view_equal(set_key, IREE_SV("amdgpu.gfx950.core")));

  EXPECT_GE(descriptor_set->descriptor_count, 8u);
  EXPECT_EQ(descriptor_set->descriptor_ref_count,
            descriptor_set->descriptor_count);
  EXPECT_EQ(descriptor_set->reg_class_count, 3u);
  EXPECT_GE(descriptor_set->schedule_class_count, 7u);
  EXPECT_GE(descriptor_set->resource_count, 7u);
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    EXPECT_NE(
        descriptor_set->reg_classes[i].flags & LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
        0u);
    EXPECT_GT(descriptor_set->reg_classes[i].physical_count, 0u);
  }
}

TEST(AmdgpuDescriptorsTest, Gfx950CoreDescriptorLookupUsesStableKeys) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx950_core_descriptor_set();

  const loom_low_descriptor_t* add_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.v_add_u32"));
  ASSERT_NE(add_descriptor, nullptr);
  iree_string_view_t add_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, add_descriptor->key_string_offset, &add_key));
  EXPECT_TRUE(iree_string_view_equal(add_key, IREE_SV("amdgpu.v_add_u32")));
  EXPECT_EQ(add_descriptor->operand_count, 3u);
  EXPECT_EQ(add_descriptor->result_count, 1u);
  EXPECT_EQ(add_descriptor->encoding_id, 52u);

  const loom_low_descriptor_t* scalar_subtract_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_sub_u32"));
  ASSERT_NE(scalar_subtract_descriptor, nullptr);
  EXPECT_EQ(scalar_subtract_descriptor->operand_count, 3u);
  EXPECT_EQ(scalar_subtract_descriptor->result_count, 1u);

  const loom_low_descriptor_t* vector_subtract_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.v_sub_u32"));
  ASSERT_NE(vector_subtract_descriptor, nullptr);
  EXPECT_EQ(vector_subtract_descriptor->operand_count, 3u);
  EXPECT_EQ(vector_subtract_descriptor->result_count, 1u);

  const loom_low_descriptor_t* multiply_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.v_mul_lo_u32"));
  ASSERT_NE(multiply_descriptor, nullptr);
  EXPECT_EQ(multiply_descriptor->operand_count, 3u);
  EXPECT_EQ(multiply_descriptor->result_count, 1u);
  EXPECT_EQ(multiply_descriptor->encoding_id, 645u);

  const loom_low_descriptor_t* f32_add_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.v_add_f32"));
  ASSERT_NE(f32_add_descriptor, nullptr);
  EXPECT_EQ(f32_add_descriptor->operand_count, 3u);
  EXPECT_EQ(f32_add_descriptor->result_count, 1u);

  const loom_low_descriptor_t* f32_multiply_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.v_mul_f32"));
  ASSERT_NE(f32_multiply_descriptor, nullptr);
  EXPECT_EQ(f32_multiply_descriptor->operand_count, 3u);
  EXPECT_EQ(f32_multiply_descriptor->result_count, 1u);

  const loom_low_descriptor_t* load_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.buffer_load_dword"));
  ASSERT_NE(load_descriptor, nullptr);
  EXPECT_EQ(load_descriptor->operand_count, 4u);
  EXPECT_EQ(load_descriptor->result_count, 1u);
  EXPECT_EQ(load_descriptor->effect_count, 1u);
  EXPECT_EQ(load_descriptor->encoding_id, 20u);
  EXPECT_NE(load_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
            0u);

  const loom_low_descriptor_t* load_128_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.buffer_load_dwordx4"));
  ASSERT_NE(load_128_descriptor, nullptr);
  EXPECT_EQ(load_128_descriptor->operand_count, 4u);
  EXPECT_EQ(load_128_descriptor->result_count, 1u);
  EXPECT_EQ(load_128_descriptor->effect_count, 1u);
  const loom_low_operand_t* load_128_operands =
      &descriptor_set->operands[load_128_descriptor->operand_start];
  EXPECT_EQ(load_128_operands[0].unit_count, 4u);
  EXPECT_EQ(load_128_operands[1].unit_count, 4u);
  const loom_low_effect_t* load_128_effect =
      &descriptor_set->effects[load_128_descriptor->effect_start];
  EXPECT_EQ(load_128_effect->kind, LOOM_LOW_EFFECT_KIND_READ);
  EXPECT_EQ(load_128_effect->width_bits, 128u);

  const loom_low_descriptor_t* store_128_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.buffer_store_dwordx4"));
  ASSERT_NE(store_128_descriptor, nullptr);
  EXPECT_EQ(store_128_descriptor->operand_count, 4u);
  EXPECT_EQ(store_128_descriptor->result_count, 0u);
  EXPECT_EQ(store_128_descriptor->effect_count, 1u);
  const loom_low_operand_t* store_128_operands =
      &descriptor_set->operands[store_128_descriptor->operand_start];
  EXPECT_EQ(store_128_operands[0].unit_count, 4u);
  EXPECT_EQ(store_128_operands[1].unit_count, 4u);
  const loom_low_effect_t* store_128_effect =
      &descriptor_set->effects[store_128_descriptor->effect_start];
  EXPECT_EQ(store_128_effect->kind, LOOM_LOW_EFFECT_KIND_WRITE);
  EXPECT_EQ(store_128_effect->width_bits, 128u);

  const loom_low_descriptor_t* wait_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_waitcnt"));
  ASSERT_NE(wait_descriptor, nullptr);
  EXPECT_EQ(wait_descriptor->operand_count, 0u);
  EXPECT_EQ(wait_descriptor->immediate_count, 2u);
  EXPECT_EQ(wait_descriptor->effect_count, 1u);
  EXPECT_EQ(wait_descriptor->encoding_id, 12u);
  EXPECT_NE(wait_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
            0u);
}

TEST(AmdgpuDescriptorsTest, Gfx950MfmaPacketMatchesCdnaRegisterShape) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx950_core_descriptor_set();

  const loom_low_descriptor_t* mfma_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("amdgpu.v_mfma_f32_16x16x16_f16"));
  ASSERT_NE(mfma_descriptor, nullptr);
  EXPECT_EQ(mfma_descriptor->operand_count, 4u);
  EXPECT_EQ(mfma_descriptor->result_count, 1u);
  EXPECT_EQ(mfma_descriptor->encoding_id, 77u);

  const loom_low_operand_t* mfma_operands =
      &descriptor_set->operands[mfma_descriptor->operand_start];
  EXPECT_EQ(mfma_operands[0].unit_count, 4u);
  EXPECT_EQ(mfma_operands[1].unit_count, 2u);
  EXPECT_EQ(mfma_operands[2].unit_count, 2u);
  EXPECT_EQ(mfma_operands[3].unit_count, 4u);
  EXPECT_EQ(mfma_operands[0].reg_class_alt_count, 2u);
  EXPECT_EQ(mfma_operands[1].reg_class_alt_count, 2u);
  EXPECT_EQ(mfma_operands[2].reg_class_alt_count, 2u);
  EXPECT_EQ(mfma_operands[3].reg_class_alt_count, 3u);

  const loom_low_reg_class_alt_t* result_alts =
      &descriptor_set->reg_class_alts[mfma_operands[0].reg_class_alt_start];
  EXPECT_NE(result_alts[0].reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_NE(result_alts[1].reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  const loom_low_reg_class_alt_t* accumulator_alts =
      &descriptor_set->reg_class_alts[mfma_operands[3].reg_class_alt_start];
  EXPECT_EQ(accumulator_alts[0].reg_class_id, result_alts[0].reg_class_id);
  EXPECT_EQ(accumulator_alts[1].reg_class_id, result_alts[1].reg_class_id);
  EXPECT_EQ(accumulator_alts[2].reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_NE(accumulator_alts[2].flags & LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE,
            0u);
}

TEST(AmdgpuDescriptorsTest, Gfx950LowAsmRequiresExplicitMfmaResultType) {
  LowTextAsmTypeInferenceHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_amdgpu_gfx950_core_descriptor_set));

  loom_text_low_asm_packet_descriptor_t packet = {};
  IREE_ASSERT_OK(harness.LookupPacket(IREE_SV("amdgpu.gfx950.core"),
                                      IREE_SV("v_mfma_f32_16x16x16_f16"),
                                      &packet));

  loom_type_t result_type = loom_type_none();
  iree_string_view_t diagnostic_detail = iree_string_view_empty();
  IREE_ASSERT_OK(harness.InferResultType(
      &packet, /*operands=*/nullptr, /*operand_count=*/0, /*result_index=*/0,
      &result_type, &diagnostic_detail));
  std::string detail(diagnostic_detail.data, diagnostic_detail.size);
  EXPECT_NE(detail.find("result type annotation"), std::string::npos);
  EXPECT_NE(detail.find("reg<amdgpu.vgpr x4>"), std::string::npos);
  EXPECT_NE(detail.find("reg<amdgpu.agpr x4>"), std::string::npos);

  loom_type_t agpr_type = loom_type_none();
  IREE_ASSERT_OK(
      harness.MakeRegisterType(IREE_SV("amdgpu.agpr"), 4, &agpr_type));
  diagnostic_detail = iree_string_view_empty();
  IREE_ASSERT_OK(harness.ValidateResultType(
      &packet, /*operands=*/nullptr, /*operand_count=*/0, /*result_index=*/0,
      agpr_type, &diagnostic_detail));
  EXPECT_TRUE(iree_string_view_is_empty(diagnostic_detail));
}

TEST(AmdgpuDescriptorsTest, Gfx950LowAsmRegionRoundTrips) {
  LowTextAsmRoundTripHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_amdgpu_gfx950_core_descriptor_set));

  const char* source =
      "test.low_asm_region asm<amdgpu.gfx950.core> {\n"
      "  %c0 = s_mov_b32 7\n"
      "  %c1 = s_mov_b32 5\n"
      "  %sum = s_add_u32 %c0, %c1\n"
      "  %diff = s_sub_u32 %sum, %c1\n"
      "  s_waitcnt {vmcnt = 0, lgkmcnt = 0}\n"
      "  return %diff\n"
      "}\n";
  std::string printed;
  IREE_ASSERT_OK(harness.RoundTrip(IREE_SV(source),
                                   IREE_SV("amdgpu.gfx950.core"), &printed));
  EXPECT_EQ(printed, source);
}

TEST(AmdgpuDescriptorsTest, ManifestNamesScalarVectorMemoryAndMatrixPackets) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx950_core_descriptor_set();

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      loom_low_descriptor_set_format_manifest_json(descriptor_set, &builder));
  std::string json(iree_string_builder_buffer(&builder),
                   iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_NE(json.find("\"key\":\"amdgpu.gfx950.core\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.s_add_u32\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.v_add_u32\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.v_add_f32\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.v_mul_f32\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.v_mul_lo_u32\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.buffer_load_dword\""),
            std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.buffer_load_dwordx4\""),
            std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.buffer_store_dwordx4\""),
            std::string::npos);
  EXPECT_NE(json.find("\"key\":\"amdgpu.v_mfma_f32_16x16x16_f16\""),
            std::string::npos);
  EXPECT_NE(json.find("\"descriptor_refs\""), std::string::npos);
}

}  // namespace
}  // namespace loom
