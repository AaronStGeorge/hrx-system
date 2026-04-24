// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/gfx12_descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/testing/text_asm_roundtrip_test_util.h"
#include "loom/target/arch/amdgpu/descriptor_test_util.h"
#include "loom/target/arch/amdgpu/encoding.h"

namespace loom {
namespace {

using ::loom::testing::ExpectAmdgpuCacheControlDescriptor;
using ::loom::testing::ExpectAmdgpuDs2AddrMemoryDescriptors;
using ::loom::testing::ExpectAmdgpuDsAddtidMemoryDescriptors;
using ::loom::testing::ExpectAmdgpuDsCrosslaneDescriptors;
using ::loom::testing::ExpectAmdgpuDsMemoryDescriptor;
using ::loom::testing::ExpectAmdgpuGlobalMemoryDescriptors;
using ::loom::testing::ExpectAmdgpuGlobalSaddrMemoryDescriptors;
using ::loom::testing::ExpectAmdgpuPrefetchDescriptor;
using ::loom::testing::ExpectAmdgpuWmmaDescriptorForTest;
using ::loom::testing::LowTextAsmRoundTripHarness;

const loom_low_descriptor_t* LookupDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  uint32_t ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_EXPECT_OK(
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, key, &ordinal));
  return loom_low_descriptor_set_descriptor_at(descriptor_set, ordinal);
}

TEST(AmdgpuDescriptorsTest, Gfx12CoreDescriptorSetVerifies) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx12_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(descriptor_set));

  iree_string_view_t set_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &set_key));
  EXPECT_TRUE(iree_string_view_equal(set_key, IREE_SV("amdgpu.gfx12.core")));

  EXPECT_GE(descriptor_set->descriptor_count, 10u);
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

TEST(AmdgpuDescriptorsTest, Gfx12CoreDescriptorLookupUsesStableKeys) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx12_core_descriptor_set();

  const loom_low_descriptor_t* add_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.v_add_u32"));
  ASSERT_NE(add_descriptor, nullptr);
  iree_string_view_t add_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, add_descriptor->key_string_offset, &add_key));
  EXPECT_TRUE(iree_string_view_equal(add_key, IREE_SV("amdgpu.v_add_u32")));
  EXPECT_EQ(add_descriptor->operand_count, 3u);
  EXPECT_EQ(add_descriptor->result_count, 1u);
  EXPECT_EQ(add_descriptor->encoding_id, 37u);

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
  EXPECT_EQ(multiply_descriptor->encoding_id, 812u);

  const loom_low_descriptor_t* f32_add_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.v_add_f32"));
  ASSERT_NE(f32_add_descriptor, nullptr);
  EXPECT_EQ(f32_add_descriptor->operand_count, 3u);
  EXPECT_EQ(f32_add_descriptor->result_count, 1u);

  const loom_low_descriptor_t* f32_subtract_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.v_sub_f32"));
  ASSERT_NE(f32_subtract_descriptor, nullptr);
  EXPECT_EQ(f32_subtract_descriptor->operand_count, 3u);
  EXPECT_EQ(f32_subtract_descriptor->result_count, 1u);

  const loom_low_descriptor_t* f32_multiply_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.v_mul_f32"));
  ASSERT_NE(f32_multiply_descriptor, nullptr);
  EXPECT_EQ(f32_multiply_descriptor->operand_count, 3u);
  EXPECT_EQ(f32_multiply_descriptor->result_count, 1u);

  const loom_low_descriptor_t* f32_fma_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.v_fma_f32"));
  ASSERT_NE(f32_fma_descriptor, nullptr);
  EXPECT_EQ(f32_fma_descriptor->operand_count, 4u);
  EXPECT_EQ(f32_fma_descriptor->result_count, 1u);

  const loom_low_descriptor_t* load_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.buffer_load_dword"));
  ASSERT_NE(load_descriptor, nullptr);
  EXPECT_EQ(load_descriptor->operand_count, 4u);
  EXPECT_EQ(load_descriptor->result_count, 1u);
  EXPECT_EQ(load_descriptor->effect_count, 1u);
  EXPECT_EQ(load_descriptor->encoding_id, 20u);
  EXPECT_NE(load_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
            0u);

  const loom_low_descriptor_t* load_64_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.buffer_load_b64"));
  ASSERT_NE(load_64_descriptor, nullptr);
  EXPECT_EQ(load_64_descriptor->operand_count, 4u);
  EXPECT_EQ(load_64_descriptor->result_count, 1u);
  EXPECT_EQ(load_64_descriptor->effect_count, 1u);
  const loom_low_operand_t* load_64_operands =
      &descriptor_set->operands[load_64_descriptor->operand_start];
  EXPECT_EQ(load_64_operands[0].unit_count, 2u);
  const loom_low_effect_t* load_64_effect =
      &descriptor_set->effects[load_64_descriptor->effect_start];
  EXPECT_EQ(load_64_effect->width_bits, 64u);

  const loom_low_descriptor_t* load_128_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.buffer_load_b128"));
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
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.buffer_store_b128"));
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

  const loom_low_descriptor_t* store_64_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.buffer_store_b64"));
  ASSERT_NE(store_64_descriptor, nullptr);
  EXPECT_EQ(store_64_descriptor->operand_count, 4u);
  EXPECT_EQ(store_64_descriptor->result_count, 0u);
  EXPECT_EQ(store_64_descriptor->effect_count, 1u);
  const loom_low_operand_t* store_64_operands =
      &descriptor_set->operands[store_64_descriptor->operand_start];
  EXPECT_EQ(store_64_operands[0].unit_count, 2u);
  const loom_low_effect_t* store_64_effect =
      &descriptor_set->effects[store_64_descriptor->effect_start];
  EXPECT_EQ(store_64_effect->width_bits, 64u);

  ExpectAmdgpuGlobalMemoryDescriptors(descriptor_set,
                                      LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL, 24u,
                                      /*expected_implicit_m0=*/false);
  ExpectAmdgpuGlobalSaddrMemoryDescriptors(
      descriptor_set, LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL, 24u,
      /*expected_implicit_m0=*/false);
  ExpectAmdgpuDsMemoryDescriptor(descriptor_set, IREE_SV("amdgpu.ds_read_b64"),
                                 LOOM_LOW_EFFECT_KIND_READ, 2u, 64u,
                                 LOOM_AMDGPU_ENCODING_FORMAT_VDS);
  ExpectAmdgpuDsMemoryDescriptor(descriptor_set, IREE_SV("amdgpu.ds_read_b96"),
                                 LOOM_LOW_EFFECT_KIND_READ, 3u, 96u,
                                 LOOM_AMDGPU_ENCODING_FORMAT_VDS);
  ExpectAmdgpuDsMemoryDescriptor(descriptor_set, IREE_SV("amdgpu.ds_read_b128"),
                                 LOOM_LOW_EFFECT_KIND_READ, 4u, 128u,
                                 LOOM_AMDGPU_ENCODING_FORMAT_VDS);
  ExpectAmdgpuDsMemoryDescriptor(descriptor_set, IREE_SV("amdgpu.ds_write_b64"),
                                 LOOM_LOW_EFFECT_KIND_WRITE, 2u, 64u,
                                 LOOM_AMDGPU_ENCODING_FORMAT_VDS);
  ExpectAmdgpuDsMemoryDescriptor(descriptor_set, IREE_SV("amdgpu.ds_write_b96"),
                                 LOOM_LOW_EFFECT_KIND_WRITE, 3u, 96u,
                                 LOOM_AMDGPU_ENCODING_FORMAT_VDS);
  ExpectAmdgpuDsMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_write_b128"),
      LOOM_LOW_EFFECT_KIND_WRITE, 4u, 128u, LOOM_AMDGPU_ENCODING_FORMAT_VDS);
  ExpectAmdgpuDs2AddrMemoryDescriptors(descriptor_set,
                                       LOOM_AMDGPU_ENCODING_FORMAT_VDS);
  ExpectAmdgpuDsAddtidMemoryDescriptors(descriptor_set,
                                        LOOM_AMDGPU_ENCODING_FORMAT_VDS);
  ExpectAmdgpuDsCrosslaneDescriptors(descriptor_set,
                                     LOOM_AMDGPU_ENCODING_FORMAT_VDS,
                                     /*expect_fetch_invalid=*/true);

  ExpectAmdgpuCacheControlDescriptor(descriptor_set,
                                     IREE_SV("amdgpu.global_inv"),
                                     LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL, 43u);
  ExpectAmdgpuCacheControlDescriptor(descriptor_set,
                                     IREE_SV("amdgpu.global_wb"),
                                     LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL, 44u);
  ExpectAmdgpuCacheControlDescriptor(descriptor_set,
                                     IREE_SV("amdgpu.global_wbinv"),
                                     LOOM_AMDGPU_ENCODING_FORMAT_VGLOBAL, 79u);
  ExpectAmdgpuCacheControlDescriptor(descriptor_set,
                                     IREE_SV("amdgpu.s_dcache_inv"),
                                     LOOM_AMDGPU_ENCODING_FORMAT_SMEM, 33u);
  ExpectAmdgpuCacheControlDescriptor(descriptor_set,
                                     IREE_SV("amdgpu.s_icache_inv"),
                                     LOOM_AMDGPU_ENCODING_FORMAT_SOPP, 60u);
  ExpectAmdgpuPrefetchDescriptor(
      descriptor_set, IREE_SV("amdgpu.s_prefetch_inst"), 36u, 2u, 2u,
      LOOM_LOW_OPERAND_ROLE_OPERAND, LOOM_LOW_MEMORY_SPACE_GENERIC);
  ExpectAmdgpuPrefetchDescriptor(
      descriptor_set, IREE_SV("amdgpu.s_prefetch_inst_pc_rel"), 37u, 1u, 0u,
      LOOM_LOW_OPERAND_ROLE_UNKNOWN, LOOM_LOW_MEMORY_SPACE_GENERIC);
  ExpectAmdgpuPrefetchDescriptor(
      descriptor_set, IREE_SV("amdgpu.s_prefetch_data"), 38u, 2u, 2u,
      LOOM_LOW_OPERAND_ROLE_OPERAND, LOOM_LOW_MEMORY_SPACE_GLOBAL);
  ExpectAmdgpuPrefetchDescriptor(
      descriptor_set, IREE_SV("amdgpu.s_buffer_prefetch_data"), 39u, 2u, 4u,
      LOOM_LOW_OPERAND_ROLE_RESOURCE, LOOM_LOW_MEMORY_SPACE_GLOBAL);
  ExpectAmdgpuPrefetchDescriptor(
      descriptor_set, IREE_SV("amdgpu.s_prefetch_data_pc_rel"), 40u, 1u, 0u,
      LOOM_LOW_OPERAND_ROLE_UNKNOWN, LOOM_LOW_MEMORY_SPACE_GLOBAL);

  const loom_low_descriptor_t* load_wait_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_wait_loadcnt"));
  ASSERT_NE(load_wait_descriptor, nullptr);
  EXPECT_EQ(load_wait_descriptor->operand_count, 0u);
  EXPECT_EQ(load_wait_descriptor->immediate_count, 1u);
  EXPECT_EQ(load_wait_descriptor->effect_count, 1u);
  EXPECT_EQ(load_wait_descriptor->encoding_id, 64u);
  EXPECT_NE(
      load_wait_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
      0u);
  const loom_low_immediate_t* load_wait_immediate =
      &descriptor_set->immediates[load_wait_descriptor->immediate_start];
  EXPECT_EQ(load_wait_immediate->kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(load_wait_immediate->bit_width, 6u);
  EXPECT_EQ(load_wait_immediate->unsigned_max, 63u);
  const loom_low_effect_t* load_wait_effect =
      &descriptor_set->effects[load_wait_descriptor->effect_start];
  EXPECT_EQ(load_wait_effect->kind, LOOM_LOW_EFFECT_KIND_COUNTER);
  EXPECT_NE(load_wait_effect->flags & LOOM_LOW_EFFECT_FLAG_ORDERED, 0u);

  const loom_low_descriptor_t* lds_wait_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_wait_dscnt"));
  ASSERT_NE(lds_wait_descriptor, nullptr);
  EXPECT_EQ(lds_wait_descriptor->operand_count, 0u);
  EXPECT_EQ(lds_wait_descriptor->immediate_count, 1u);
  EXPECT_EQ(lds_wait_descriptor->effect_count, 1u);
  EXPECT_EQ(lds_wait_descriptor->encoding_format_id,
            LOOM_AMDGPU_ENCODING_FORMAT_SOPP);

  const loom_low_descriptor_t* smem_wait_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_wait_kmcnt"));
  ASSERT_NE(smem_wait_descriptor, nullptr);
  EXPECT_EQ(smem_wait_descriptor->operand_count, 0u);
  EXPECT_EQ(smem_wait_descriptor->immediate_count, 1u);
  EXPECT_EQ(smem_wait_descriptor->effect_count, 1u);
  EXPECT_EQ(smem_wait_descriptor->encoding_format_id,
            LOOM_AMDGPU_ENCODING_FORMAT_SOPP);
}

TEST(AmdgpuDescriptorsTest, Gfx12WmmaPacketMatchesRdnaRegisterShape) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx12_core_descriptor_set();

  ExpectAmdgpuWmmaDescriptorForTest(
      descriptor_set, IREE_SV("amdgpu.v_wmma_f32_16x16x16_f16"),
      /*expected_encoding_id=*/64u, /*expected_lhs_units=*/4u,
      /*expected_rhs_units=*/4u, /*expected_accumulator_units=*/8u);
  ExpectAmdgpuWmmaDescriptorForTest(
      descriptor_set, IREE_SV("amdgpu.v_wmma_i32_16x16x16_iu8"),
      /*expected_encoding_id=*/68u, /*expected_lhs_units=*/2u,
      /*expected_rhs_units=*/2u, /*expected_accumulator_units=*/8u);
  ExpectAmdgpuWmmaDescriptorForTest(
      descriptor_set, IREE_SV("amdgpu.v_wmma_i32_16x16x16_iu4"),
      /*expected_encoding_id=*/69u, /*expected_lhs_units=*/1u,
      /*expected_rhs_units=*/1u, /*expected_accumulator_units=*/8u);
}

TEST(AmdgpuDescriptorsTest, Gfx12LowAsmRegionRoundTripsPrefetch) {
  LowTextAsmRoundTripHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_amdgpu_gfx12_core_descriptor_set));

  const char* source =
      "test.low_asm_region asm<amdgpu.gfx12.core> {\n"
      "  %soffset = s_mov_b32 0\n"
      "  s_prefetch_data_pc_rel %soffset {offset = 64, count = 1}\n"
      "  s_prefetch_inst_pc_rel %soffset {offset = 128, count = 2}\n"
      "  return %soffset\n"
      "}\n";
  std::string printed;
  IREE_ASSERT_OK(harness.RoundTrip(IREE_SV(source),
                                   IREE_SV("amdgpu.gfx12.core"), &printed));
  EXPECT_EQ(printed, source);
}

}  // namespace
}  // namespace loom
