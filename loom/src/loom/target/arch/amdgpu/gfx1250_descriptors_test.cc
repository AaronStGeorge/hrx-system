// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/gfx1250_descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/encoding.h"

namespace loom {
namespace {

const loom_low_descriptor_t* LookupDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  uint32_t ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_EXPECT_OK(
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, key, &ordinal));
  return loom_low_descriptor_set_descriptor_at(descriptor_set, ordinal);
}

void ExpectPseudoMatrixDescriptor(const loom_low_descriptor_t* descriptor) {
  EXPECT_EQ(descriptor->encoding_id, LOOM_LOW_ID_NONE);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_PSEUDO, 0u);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE, 0u);
}

TEST(AmdgpuDescriptorsTest, Gfx1250CoreDescriptorSetVerifies) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx1250_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(descriptor_set));

  iree_string_view_t set_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &set_key));
  EXPECT_TRUE(iree_string_view_equal(set_key, IREE_SV("amdgpu.gfx1250.core")));

  EXPECT_GE(descriptor_set->descriptor_count, 14u);
  EXPECT_EQ(descriptor_set->descriptor_ref_count,
            descriptor_set->descriptor_count);
  EXPECT_EQ(descriptor_set->reg_class_count, 2u);
  EXPECT_GE(descriptor_set->schedule_class_count, 9u);
  EXPECT_GE(descriptor_set->resource_count, 8u);
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    EXPECT_NE(
        descriptor_set->reg_classes[i].flags & LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
        0u);
    EXPECT_GT(descriptor_set->reg_classes[i].physical_count, 0u);
  }
}

TEST(AmdgpuDescriptorsTest, Gfx1250BaselinePacketsMatchGfx12Shape) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx1250_core_descriptor_set();

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

  const loom_low_descriptor_t* ds_read_128_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.ds_read_b128"));
  ASSERT_NE(ds_read_128_descriptor, nullptr);
  EXPECT_EQ(ds_read_128_descriptor->operand_count, 2u);
  EXPECT_EQ(ds_read_128_descriptor->result_count, 1u);
  EXPECT_EQ(ds_read_128_descriptor->effect_count, 1u);
  EXPECT_EQ(ds_read_128_descriptor->encoding_format_id,
            LOOM_AMDGPU_ENCODING_FORMAT_VDS);
  const loom_low_operand_t* ds_read_128_operands =
      &descriptor_set->operands[ds_read_128_descriptor->operand_start];
  EXPECT_EQ(ds_read_128_operands[0].unit_count, 4u);
  EXPECT_EQ(ds_read_128_operands[1].unit_count, 1u);
  const loom_low_effect_t* ds_read_128_effect =
      &descriptor_set->effects[ds_read_128_descriptor->effect_start];
  EXPECT_EQ(ds_read_128_effect->kind, LOOM_LOW_EFFECT_KIND_READ);
  EXPECT_EQ(ds_read_128_effect->memory_space, LOOM_LOW_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(ds_read_128_effect->width_bits, 128u);

  const loom_low_descriptor_t* ds_write_128_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.ds_write_b128"));
  ASSERT_NE(ds_write_128_descriptor, nullptr);
  EXPECT_EQ(ds_write_128_descriptor->operand_count, 2u);
  EXPECT_EQ(ds_write_128_descriptor->result_count, 0u);
  EXPECT_EQ(ds_write_128_descriptor->effect_count, 1u);
  EXPECT_EQ(ds_write_128_descriptor->encoding_format_id,
            LOOM_AMDGPU_ENCODING_FORMAT_VDS);
  const loom_low_operand_t* ds_write_128_operands =
      &descriptor_set->operands[ds_write_128_descriptor->operand_start];
  EXPECT_EQ(ds_write_128_operands[0].unit_count, 1u);
  EXPECT_EQ(ds_write_128_operands[1].unit_count, 4u);
  const loom_low_effect_t* ds_write_128_effect =
      &descriptor_set->effects[ds_write_128_descriptor->effect_start];
  EXPECT_EQ(ds_write_128_effect->kind, LOOM_LOW_EFFECT_KIND_WRITE);
  EXPECT_EQ(ds_write_128_effect->memory_space, LOOM_LOW_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(ds_write_128_effect->width_bits, 128u);

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

  const loom_low_descriptor_t* store_wait_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_wait_storecnt"));
  ASSERT_NE(store_wait_descriptor, nullptr);
  EXPECT_EQ(store_wait_descriptor->encoding_id, 65u);

  const loom_low_descriptor_t* alu_wait_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_wait_alu"));
  ASSERT_NE(alu_wait_descriptor, nullptr);
  EXPECT_EQ(alu_wait_descriptor->encoding_id, 8u);

  const loom_low_descriptor_t* idle_wait_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_wait_idle"));
  ASSERT_NE(idle_wait_descriptor, nullptr);
  EXPECT_EQ(idle_wait_descriptor->encoding_id, 10u);
}

TEST(AmdgpuDescriptorsTest, Gfx1250WmmaPseudoModelsRegisterShape) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx1250_core_descriptor_set();

  const loom_low_descriptor_t* wmma_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("amdgpu.v_wmma_f32_16x16x32_f16"));
  ASSERT_NE(wmma_descriptor, nullptr);
  ExpectPseudoMatrixDescriptor(wmma_descriptor);
  EXPECT_EQ(wmma_descriptor->operand_count, 4u);
  EXPECT_EQ(wmma_descriptor->result_count, 1u);

  const loom_low_operand_t* wmma_operands =
      &descriptor_set->operands[wmma_descriptor->operand_start];
  EXPECT_EQ(wmma_operands[0].unit_count, 8u);
  EXPECT_EQ(wmma_operands[1].unit_count, 8u);
  EXPECT_EQ(wmma_operands[2].unit_count, 8u);
  EXPECT_EQ(wmma_operands[3].unit_count, 8u);
  EXPECT_EQ(wmma_operands[0].reg_class_alt_count, 1u);
  EXPECT_EQ(wmma_operands[1].reg_class_alt_count, 1u);
  EXPECT_EQ(wmma_operands[2].reg_class_alt_count, 1u);
  EXPECT_EQ(wmma_operands[3].reg_class_alt_count, 2u);

  const uint16_t vgpr_class_id =
      descriptor_set->reg_class_alts[wmma_operands[0].reg_class_alt_start]
          .reg_class_id;
  EXPECT_NE(vgpr_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_EQ(descriptor_set->reg_class_alts[wmma_operands[1].reg_class_alt_start]
                .reg_class_id,
            vgpr_class_id);
  EXPECT_EQ(descriptor_set->reg_class_alts[wmma_operands[2].reg_class_alt_start]
                .reg_class_id,
            vgpr_class_id);
  const loom_low_reg_class_alt_t* accumulator_alts =
      &descriptor_set->reg_class_alts[wmma_operands[3].reg_class_alt_start];
  EXPECT_EQ(accumulator_alts[0].reg_class_id, vgpr_class_id);
  EXPECT_EQ(accumulator_alts[1].reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_NE(accumulator_alts[1].flags & LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE,
            0u);
}

TEST(AmdgpuDescriptorsTest, Gfx1250ScaledWmmaModelsScaleOperandsAndImmediates) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx1250_core_descriptor_set();

  const loom_low_descriptor_t* scale_descriptor = LookupDescriptor(
      descriptor_set,
      IREE_SV("amdgpu.v_wmma_scale_f32_16x16x128_f8f6f4_f8_f8"));
  ASSERT_NE(scale_descriptor, nullptr);
  ExpectPseudoMatrixDescriptor(scale_descriptor);
  EXPECT_EQ(scale_descriptor->operand_count, 6u);
  EXPECT_EQ(scale_descriptor->result_count, 1u);
  ASSERT_EQ(scale_descriptor->immediate_count, 8u);

  const loom_low_operand_t* scale_operands =
      &descriptor_set->operands[scale_descriptor->operand_start];
  EXPECT_EQ(scale_operands[0].unit_count, 8u);
  EXPECT_EQ(scale_operands[1].unit_count, 16u);
  EXPECT_EQ(scale_operands[2].unit_count, 16u);
  EXPECT_EQ(scale_operands[3].unit_count, 8u);
  EXPECT_EQ(scale_operands[4].unit_count, 1u);
  EXPECT_EQ(scale_operands[5].unit_count, 1u);

  const loom_low_immediate_t* scale_immediates =
      &descriptor_set->immediates[scale_descriptor->immediate_start];
  for (uint32_t i = 0; i < 6; ++i) {
    EXPECT_EQ(scale_immediates[i].kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
    EXPECT_EQ(scale_immediates[i].bit_width, 8u);
    EXPECT_EQ(scale_immediates[i].unsigned_max, 255u);
  }
  EXPECT_EQ(scale_immediates[6].kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(scale_immediates[6].bit_width, 1u);
  EXPECT_EQ(scale_immediates[6].unsigned_max, 1u);
  EXPECT_EQ(scale_immediates[7].kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(scale_immediates[7].bit_width, 1u);
  EXPECT_EQ(scale_immediates[7].unsigned_max, 1u);

  const loom_low_descriptor_t* scale16_descriptor = LookupDescriptor(
      descriptor_set,
      IREE_SV("amdgpu.v_wmma_scale16_f32_16x16x128_f8f6f4_f8_f8"));
  ASSERT_NE(scale16_descriptor, nullptr);
  ExpectPseudoMatrixDescriptor(scale16_descriptor);
  EXPECT_EQ(scale16_descriptor->operand_count, 6u);
  ASSERT_EQ(scale16_descriptor->immediate_count, 8u);
  const loom_low_operand_t* scale16_operands =
      &descriptor_set->operands[scale16_descriptor->operand_start];
  EXPECT_EQ(scale16_operands[4].unit_count, 2u);
  EXPECT_EQ(scale16_operands[5].unit_count, 2u);
}

TEST(AmdgpuDescriptorsTest,
     Gfx1250SwmmacModelsSparseIndexAndDestructiveAccumulator) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx1250_core_descriptor_set();

  const loom_low_descriptor_t* swmmac_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("amdgpu.v_swmmac_f32_16x16x64_f16"));
  ASSERT_NE(swmmac_descriptor, nullptr);
  ExpectPseudoMatrixDescriptor(swmmac_descriptor);
  EXPECT_EQ(swmmac_descriptor->operand_count, 5u);
  EXPECT_EQ(swmmac_descriptor->result_count, 1u);
  ASSERT_EQ(swmmac_descriptor->immediate_count, 1u);
  ASSERT_EQ(swmmac_descriptor->constraint_count, 3u);

  const loom_low_operand_t* swmmac_operands =
      &descriptor_set->operands[swmmac_descriptor->operand_start];
  EXPECT_EQ(swmmac_operands[0].unit_count, 8u);
  EXPECT_EQ(swmmac_operands[1].unit_count, 8u);
  EXPECT_EQ(swmmac_operands[2].unit_count, 8u);
  EXPECT_EQ(swmmac_operands[3].unit_count, 16u);
  EXPECT_EQ(swmmac_operands[4].unit_count, 1u);

  const loom_low_immediate_t* index_key =
      &descriptor_set->immediates[swmmac_descriptor->immediate_start];
  EXPECT_EQ(index_key->kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(index_key->bit_width, 32u);
  EXPECT_EQ(index_key->unsigned_max, UINT64_C(4294967295));

  const loom_low_constraint_t* constraints =
      &descriptor_set->constraints[swmmac_descriptor->constraint_start];
  EXPECT_EQ(constraints[0].kind, LOOM_LOW_CONSTRAINT_KIND_TIED);
  EXPECT_EQ(constraints[0].lhs_operand_index, 0u);
  EXPECT_EQ(constraints[0].rhs_operand_index, 1u);
  EXPECT_EQ(constraints[1].kind, LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE);
  EXPECT_EQ(constraints[1].lhs_operand_index, 0u);
  EXPECT_EQ(constraints[1].rhs_operand_index, 1u);
  EXPECT_EQ(constraints[2].kind, LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER);
  EXPECT_EQ(constraints[2].lhs_operand_index, 0u);
  EXPECT_EQ(constraints[2].rhs_operand_index, LOOM_LOW_ID_NONE);
}

}  // namespace
}  // namespace loom
