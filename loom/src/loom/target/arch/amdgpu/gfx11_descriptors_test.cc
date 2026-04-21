// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/gfx11_descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm_roundtrip_test_util.h"
#include "loom/codegen/low/text_asm_test_util.h"
#include "loom/target/arch/amdgpu/descriptor_test_util.h"
#include "loom/target/arch/amdgpu/encoding.h"

namespace loom {
namespace {

using ::loom::testing::ExpectAmdgpuDs2AddrMemoryDescriptors;
using ::loom::testing::ExpectAmdgpuDsAddtidMemoryDescriptors;
using ::loom::testing::ExpectAmdgpuDsMemoryDescriptor;
using ::loom::testing::ExpectAmdgpuGlobalMemoryDescriptors;
using ::loom::testing::ExpectAmdgpuGlobalSaddrMemoryDescriptors;
using ::loom::testing::LowFuncAsmRoundTripHarness;
using ::loom::testing::LowTextAsmRoundTripHarness;
using ::loom::testing::LowTextAsmTypeInferenceHarness;

std::string ToString(const loom_low_descriptor_set_t* descriptor_set,
                     loom_bstring_table_offset_t string_offset) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(
      loom_low_descriptor_set_string(descriptor_set, string_offset, &value));
  return std::string(value.data, value.size);
}

const loom_low_descriptor_t* LookupDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  uint32_t ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_EXPECT_OK(
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, key, &ordinal));
  return loom_low_descriptor_set_descriptor_at(descriptor_set, ordinal);
}

TEST(AmdgpuDescriptorsTest, Gfx11CoreDescriptorSetVerifies) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx11_core_descriptor_set();
  ASSERT_NE(descriptor_set, nullptr);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(descriptor_set));

  iree_string_view_t set_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, descriptor_set->key_string_offset, &set_key));
  EXPECT_TRUE(iree_string_view_equal(set_key, IREE_SV("amdgpu.gfx11.core")));

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

TEST(AmdgpuDescriptorsTest, Gfx11CoreDescriptorLookupUsesStableKeys) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx11_core_descriptor_set();

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
  iree_string_view_t vector_subtract_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, vector_subtract_descriptor->key_string_offset,
      &vector_subtract_key));
  EXPECT_TRUE(
      iree_string_view_equal(vector_subtract_key, IREE_SV("amdgpu.v_sub_u32")));
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

  const loom_low_descriptor_t* vector_move_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.v_mov_b32"));
  ASSERT_NE(vector_move_descriptor, nullptr);
  EXPECT_EQ(vector_move_descriptor->operand_count, 1u);
  EXPECT_EQ(vector_move_descriptor->result_count, 1u);
  EXPECT_EQ(vector_move_descriptor->immediate_count, 1u);
  EXPECT_EQ(vector_move_descriptor->encoding_format_id,
            LOOM_AMDGPU_ENCODING_FORMAT_VOP1_LITERAL);
  EXPECT_EQ(vector_move_descriptor->encoding_id, 1u);

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
  EXPECT_EQ(load_64_operands[1].unit_count, 4u);
  const loom_low_effect_t* load_64_effect =
      &descriptor_set->effects[load_64_descriptor->effect_start];
  EXPECT_EQ(load_64_effect->kind, LOOM_LOW_EFFECT_KIND_READ);
  EXPECT_EQ(load_64_effect->memory_space, LOOM_LOW_MEMORY_SPACE_GLOBAL);
  EXPECT_EQ(load_64_effect->width_bits, 64u);
  EXPECT_NE(load_64_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
            0u);

  const loom_low_descriptor_t* load_128_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.buffer_load_b128"));
  ASSERT_NE(load_128_descriptor, nullptr);
  EXPECT_EQ(load_128_descriptor->operand_count, 4u);
  EXPECT_EQ(load_128_descriptor->result_count, 1u);
  EXPECT_EQ(load_128_descriptor->effect_count, 1u);
  EXPECT_EQ(load_128_descriptor->encoding_id, 23u);
  const loom_low_operand_t* load_128_operands =
      &descriptor_set->operands[load_128_descriptor->operand_start];
  EXPECT_EQ(load_128_operands[0].unit_count, 4u);
  EXPECT_EQ(load_128_operands[1].unit_count, 4u);
  EXPECT_EQ(load_128_operands[2].unit_count, 1u);
  EXPECT_EQ(load_128_operands[3].unit_count, 1u);
  const loom_low_effect_t* load_128_effect =
      &descriptor_set->effects[load_128_descriptor->effect_start];
  EXPECT_EQ(load_128_effect->kind, LOOM_LOW_EFFECT_KIND_READ);
  EXPECT_EQ(load_128_effect->memory_space, LOOM_LOW_MEMORY_SPACE_GLOBAL);
  EXPECT_EQ(load_128_effect->width_bits, 128u);
  EXPECT_NE(
      load_128_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);

  const loom_low_descriptor_t* store_128_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.buffer_store_b128"));
  ASSERT_NE(store_128_descriptor, nullptr);
  EXPECT_EQ(store_128_descriptor->operand_count, 4u);
  EXPECT_EQ(store_128_descriptor->result_count, 0u);
  EXPECT_EQ(store_128_descriptor->effect_count, 1u);
  EXPECT_EQ(store_128_descriptor->encoding_id, 29u);
  const loom_low_operand_t* store_128_operands =
      &descriptor_set->operands[store_128_descriptor->operand_start];
  EXPECT_EQ(store_128_operands[0].unit_count, 4u);
  EXPECT_EQ(store_128_operands[1].unit_count, 4u);
  EXPECT_EQ(store_128_operands[2].unit_count, 1u);
  EXPECT_EQ(store_128_operands[3].unit_count, 1u);
  const loom_low_effect_t* store_128_effect =
      &descriptor_set->effects[store_128_descriptor->effect_start];
  EXPECT_EQ(store_128_effect->kind, LOOM_LOW_EFFECT_KIND_WRITE);
  EXPECT_EQ(store_128_effect->memory_space, LOOM_LOW_MEMORY_SPACE_GLOBAL);
  EXPECT_EQ(store_128_effect->width_bits, 128u);
  EXPECT_NE(
      store_128_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
      0u);

  const loom_low_descriptor_t* store_64_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.buffer_store_b64"));
  ASSERT_NE(store_64_descriptor, nullptr);
  EXPECT_EQ(store_64_descriptor->operand_count, 4u);
  EXPECT_EQ(store_64_descriptor->result_count, 0u);
  EXPECT_EQ(store_64_descriptor->effect_count, 1u);
  const loom_low_operand_t* store_64_operands =
      &descriptor_set->operands[store_64_descriptor->operand_start];
  EXPECT_EQ(store_64_operands[0].unit_count, 2u);
  EXPECT_EQ(store_64_operands[1].unit_count, 4u);
  const loom_low_effect_t* store_64_effect =
      &descriptor_set->effects[store_64_descriptor->effect_start];
  EXPECT_EQ(store_64_effect->kind, LOOM_LOW_EFFECT_KIND_WRITE);
  EXPECT_EQ(store_64_effect->memory_space, LOOM_LOW_MEMORY_SPACE_GLOBAL);
  EXPECT_EQ(store_64_effect->width_bits, 64u);
  EXPECT_NE(
      store_64_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);

  ExpectAmdgpuGlobalMemoryDescriptors(
      descriptor_set, LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLOBAL, 13u,
      /*expected_implicit_m0=*/false);
  ExpectAmdgpuGlobalSaddrMemoryDescriptors(
      descriptor_set, LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLOBAL, 13u,
      /*expected_implicit_m0=*/false);
  ExpectAmdgpuDsMemoryDescriptor(descriptor_set, IREE_SV("amdgpu.ds_read_b64"),
                                 LOOM_LOW_EFFECT_KIND_READ, 2u, 64u,
                                 LOOM_AMDGPU_ENCODING_FORMAT_DS);
  ExpectAmdgpuDsMemoryDescriptor(descriptor_set, IREE_SV("amdgpu.ds_read_b96"),
                                 LOOM_LOW_EFFECT_KIND_READ, 3u, 96u,
                                 LOOM_AMDGPU_ENCODING_FORMAT_DS);
  ExpectAmdgpuDsMemoryDescriptor(descriptor_set, IREE_SV("amdgpu.ds_read_b128"),
                                 LOOM_LOW_EFFECT_KIND_READ, 4u, 128u,
                                 LOOM_AMDGPU_ENCODING_FORMAT_DS);
  ExpectAmdgpuDsMemoryDescriptor(descriptor_set, IREE_SV("amdgpu.ds_write_b64"),
                                 LOOM_LOW_EFFECT_KIND_WRITE, 2u, 64u,
                                 LOOM_AMDGPU_ENCODING_FORMAT_DS);
  ExpectAmdgpuDsMemoryDescriptor(descriptor_set, IREE_SV("amdgpu.ds_write_b96"),
                                 LOOM_LOW_EFFECT_KIND_WRITE, 3u, 96u,
                                 LOOM_AMDGPU_ENCODING_FORMAT_DS);
  ExpectAmdgpuDsMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_write_b128"),
      LOOM_LOW_EFFECT_KIND_WRITE, 4u, 128u, LOOM_AMDGPU_ENCODING_FORMAT_DS);
  ExpectAmdgpuDs2AddrMemoryDescriptors(descriptor_set,
                                       LOOM_AMDGPU_ENCODING_FORMAT_DS);
  ExpectAmdgpuDsAddtidMemoryDescriptors(descriptor_set,
                                        LOOM_AMDGPU_ENCODING_FORMAT_DS);

  const loom_low_descriptor_t* barrier_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_barrier"));
  ASSERT_NE(barrier_descriptor, nullptr);
  EXPECT_EQ(barrier_descriptor->operand_count, 0u);
  EXPECT_EQ(barrier_descriptor->immediate_count, 0u);
  EXPECT_EQ(barrier_descriptor->effect_count, 2u);
  EXPECT_EQ(barrier_descriptor->encoding_format_id,
            LOOM_AMDGPU_ENCODING_FORMAT_SOPP);
  EXPECT_NE(barrier_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
            0u);

  const loom_low_descriptor_t* scalar_load_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_load_dwordx2"));
  ASSERT_NE(scalar_load_descriptor, nullptr);
  EXPECT_EQ(scalar_load_descriptor->operand_count, 3u);
  EXPECT_EQ(scalar_load_descriptor->result_count, 1u);
  EXPECT_EQ(scalar_load_descriptor->immediate_count, 1u);
  EXPECT_EQ(scalar_load_descriptor->effect_count, 1u);
  EXPECT_EQ(scalar_load_descriptor->encoding_format_id,
            LOOM_AMDGPU_ENCODING_FORMAT_SMEM);
  EXPECT_NE(
      scalar_load_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
      0u);

  const loom_low_descriptor_t* scalar_buffer_load_64_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_buffer_load_b64"));
  ASSERT_NE(scalar_buffer_load_64_descriptor, nullptr);
  EXPECT_EQ(scalar_buffer_load_64_descriptor->operand_count, 3u);
  EXPECT_EQ(scalar_buffer_load_64_descriptor->result_count, 1u);
  EXPECT_EQ(scalar_buffer_load_64_descriptor->effect_count, 1u);
  const loom_low_operand_t* scalar_buffer_load_64_operands =
      &descriptor_set
           ->operands[scalar_buffer_load_64_descriptor->operand_start];
  EXPECT_EQ(scalar_buffer_load_64_operands[0].unit_count, 2u);
  EXPECT_EQ(scalar_buffer_load_64_operands[1].unit_count, 4u);
  const loom_low_effect_t* scalar_buffer_load_64_effect =
      &descriptor_set->effects[scalar_buffer_load_64_descriptor->effect_start];
  EXPECT_EQ(scalar_buffer_load_64_effect->kind, LOOM_LOW_EFFECT_KIND_READ);
  EXPECT_EQ(scalar_buffer_load_64_effect->memory_space,
            LOOM_LOW_MEMORY_SPACE_GLOBAL);
  EXPECT_EQ(scalar_buffer_load_64_effect->width_bits, 64u);

  const loom_low_descriptor_t* wait_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_waitcnt"));
  ASSERT_NE(wait_descriptor, nullptr);
  EXPECT_EQ(wait_descriptor->operand_count, 0u);
  EXPECT_EQ(wait_descriptor->immediate_count, 2u);
  EXPECT_EQ(wait_descriptor->effect_count, 1u);
  EXPECT_EQ(wait_descriptor->encoding_id, 9u);
  EXPECT_NE(wait_descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
            0u);

  const loom_low_descriptor_t* depctr_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_waitcnt_depctr"));
  ASSERT_NE(depctr_descriptor, nullptr);
  EXPECT_EQ(depctr_descriptor->operand_count, 0u);
  EXPECT_EQ(depctr_descriptor->immediate_count, 1u);
  EXPECT_EQ(depctr_descriptor->effect_count, 1u);
  EXPECT_EQ(depctr_descriptor->encoding_id, 8u);

  const loom_low_descriptor_t* idle_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("amdgpu.s_wait_idle"));
  ASSERT_NE(idle_descriptor, nullptr);
  EXPECT_EQ(idle_descriptor->operand_count, 0u);
  EXPECT_EQ(idle_descriptor->immediate_count, 0u);
  EXPECT_EQ(idle_descriptor->effect_count, 1u);
  EXPECT_EQ(idle_descriptor->encoding_id, 10u);
}

TEST(AmdgpuDescriptorsTest, Gfx11WmmaPacketMatchesRdnaRegisterShape) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx11_core_descriptor_set();

  const loom_low_descriptor_t* wmma_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("amdgpu.v_wmma_f32_16x16x16_f16"));
  ASSERT_NE(wmma_descriptor, nullptr);
  EXPECT_EQ(wmma_descriptor->operand_count, 4u);
  EXPECT_EQ(wmma_descriptor->result_count, 1u);
  EXPECT_EQ(wmma_descriptor->encoding_id, 64u);

  const loom_low_operand_t* wmma_operands =
      &descriptor_set->operands[wmma_descriptor->operand_start];
  EXPECT_EQ(wmma_operands[0].unit_count, 8u);
  EXPECT_EQ(wmma_operands[1].unit_count, 4u);
  EXPECT_EQ(wmma_operands[2].unit_count, 4u);
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

TEST(AmdgpuDescriptorsTest, Gfx11AsmFormsExposeNamedWaitcntImmediates) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_amdgpu_gfx11_core_descriptor_set();
  ASSERT_GE(descriptor_set->asm_form_count, 10u);

  uint32_t asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_asm_form(
      descriptor_set, IREE_SV("s_waitcnt"), &asm_form_ordinal));
  const loom_low_asm_form_t* asm_form =
      loom_low_descriptor_set_asm_form_at(descriptor_set, asm_form_ordinal);
  ASSERT_NE(asm_form, nullptr);
  EXPECT_EQ(asm_form->result_operand_index_count, 0u);
  EXPECT_EQ(asm_form->operand_index_count, 0u);
  ASSERT_EQ(asm_form->immediate_count, 2u);

  const loom_low_asm_immediate_t* first_immediate =
      &descriptor_set->asm_immediates[asm_form->immediate_start];
  const loom_low_asm_immediate_t* second_immediate = first_immediate + 1;
  EXPECT_EQ(ToString(descriptor_set, first_immediate->name_string_offset),
            "vmcnt");
  EXPECT_EQ(ToString(descriptor_set, second_immediate->name_string_offset),
            "lgkmcnt");

  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_asm_form(
      descriptor_set, IREE_SV("v_sub_nc_u32"), &asm_form_ordinal));
  asm_form =
      loom_low_descriptor_set_asm_form_at(descriptor_set, asm_form_ordinal);
  ASSERT_NE(asm_form, nullptr);
  EXPECT_EQ(asm_form->result_operand_index_count, 1u);
  EXPECT_EQ(asm_form->operand_index_count, 2u);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set,
                                            asm_form->descriptor_ordinal);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor_set, descriptor->key_string_offset),
            "amdgpu.v_sub_u32");
}

TEST(AmdgpuDescriptorsTest, Gfx11LowAsmInfersForcedVgprWmmaResultType) {
  LowTextAsmTypeInferenceHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_amdgpu_gfx11_core_descriptor_set));

  loom_text_low_asm_packet_descriptor_t packet = {};
  IREE_ASSERT_OK(harness.LookupPacket(IREE_SV("amdgpu.gfx11.core"),
                                      IREE_SV("v_wmma_f32_16x16x16_f16"),
                                      &packet));

  loom_type_t result_type = loom_type_none();
  iree_string_view_t diagnostic_detail = iree_string_view_empty();
  IREE_ASSERT_OK(harness.InferResultType(
      &packet, /*operands=*/nullptr, /*operand_count=*/0, /*result_index=*/0,
      &result_type, &diagnostic_detail));
  EXPECT_TRUE(iree_string_view_is_empty(diagnostic_detail));
  EXPECT_TRUE(
      harness.RegisterTypeEquals(result_type, IREE_SV("amdgpu.vgpr"), 8));
}

TEST(AmdgpuDescriptorsTest, Gfx11LowAsmRegionRoundTrips) {
  LowTextAsmRoundTripHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_amdgpu_gfx11_core_descriptor_set));

  const char* source =
      "test.low_asm_region asm<amdgpu.gfx11.core> {\n"
      "  %c0 = s_mov_b32 7\n"
      "  %c1 = s_mov_b32 5\n"
      "  %sum = s_add_u32 %c0, %c1\n"
      "  %v = v_mov_b32 42\n"
      "  s_waitcnt {vmcnt = 0, lgkmcnt = 0}\n"
      "  return %sum\n"
      "}\n";
  std::string printed;
  IREE_ASSERT_OK(harness.RoundTrip(IREE_SV(source),
                                   IREE_SV("amdgpu.gfx11.core"), &printed));
  EXPECT_EQ(printed, source);
}

TEST(AmdgpuDescriptorsTest, Gfx11LowFuncAsmRoundTripsVectorMemoryAndMatrix) {
  LowFuncAsmRoundTripHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_amdgpu_gfx11_core_descriptor_set));

  const char* source =
      "target.snapshot @gfx1100 {codegen_format = low_native, target_triple = "
      "\"amdgcn-amd-amdhsa\", data_layout = \"amdgpu-layout\", "
      "artifact_format = elf, target_cpu = \"gfx1100\", target_features = "
      "\"+wavefrontsize32\", default_pointer_bitwidth = 64, index_bitwidth = "
      "32, offset_bitwidth = 64, memory_space_generic = 0, "
      "memory_space_global = 1, memory_space_workgroup = 3, "
      "memory_space_constant = 4, memory_space_private = 5, "
      "memory_space_host = 4294967295, memory_space_descriptor = 7}\n"
      "target.export @hal_export {export_symbol = \"kernel\", abi = "
      "hal_kernel, linkage = default, hal_binding_alignment = 16, "
      "hal_workgroup_size_x = 64, hal_workgroup_size_y = 1, "
      "hal_workgroup_size_z = 1, hal_flat_workgroup_size_min = 64, "
      "hal_flat_workgroup_size_max = 64, hal_buffer_resource_flags = 0}\n"
      "target.config @gfx_config {contract_set_key = "
      "\"amdgpu.gfx11.core\", contract_feature_bits = 0}\n"
      "target.bundle @gfx_target {snapshot = @gfx1100, export_plan = "
      "@hal_export, config = @gfx_config}\n"
      "low.func.def target(@gfx_target) @vector_memory_matrix(%lhs : "
      "reg<amdgpu.vgpr>, %rhs : reg<amdgpu.vgpr>, %a : reg<amdgpu.vgpr x4>, "
      "%b : reg<amdgpu.vgpr x4>, %acc : reg<amdgpu.vgpr x8>, "
      "%resource : reg<amdgpu.sgpr x4>, %vaddr : reg<amdgpu.vgpr>, "
      "%soffset : reg<amdgpu.sgpr>) -> (reg<amdgpu.vgpr>, "
      "reg<amdgpu.vgpr x4>, reg<amdgpu.vgpr x8>) asm<amdgpu.gfx11.core> {\n"
      "  %sum = v_add_u32 %lhs, %rhs\n"
      "  %product = v_mul_lo_u32 %sum, %rhs\n"
      "  %loaded = buffer_load_b128 %resource, %vaddr, %soffset {offset = 16}\n"
      "  buffer_store_b128 %loaded, %resource, %vaddr, %soffset {offset = 32}\n"
      "  %matrix = v_wmma_f32_16x16x16_f16 %a, %b, %acc\n"
      "  return %product, %loaded, %matrix\n"
      "}\n";
  std::string printed;
  IREE_ASSERT_OK(harness.RoundTripAndVerify(
      IREE_SV(source), IREE_SV("amdgpu.gfx11.core"), &printed));
  EXPECT_NE(printed.find("v_add_u32 %lhs, %rhs"), std::string::npos);
  EXPECT_NE(printed.find("v_mul_lo_u32 %sum, %rhs"), std::string::npos);
  EXPECT_NE(printed.find("buffer_load_b128 %resource, %vaddr, %soffset"),
            std::string::npos);
  EXPECT_NE(printed.find("buffer_store_b128 %loaded, %resource, %vaddr, "
                         "%soffset"),
            std::string::npos);
  EXPECT_NE(printed.find("v_wmma_f32_16x16x16_f16 %a, %b, %acc"),
            std::string::npos);
}

}  // namespace
}  // namespace loom
