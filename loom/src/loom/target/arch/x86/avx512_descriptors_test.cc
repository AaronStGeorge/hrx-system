// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/avx512_descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm_roundtrip_test_util.h"
#include "loom/codegen/low/text_asm_test_util.h"
#include "loom/target/arch/x86/low_registry.h"

namespace loom {
namespace {

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
  EXPECT_GE(descriptor_set->reg_class_count, 5u);
  EXPECT_GE(descriptor_set->schedule_class_count, 6u);
  EXPECT_GE(descriptor_set->resource_count, 7u);
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    EXPECT_NE(
        descriptor_set->reg_classes[i].flags & LOOM_LOW_REG_CLASS_FLAG_PHYSICAL,
        0u);
    EXPECT_GT(descriptor_set->reg_classes[i].physical_count, 0u);
  }
  EXPECT_EQ(descriptor_set->reg_classes[X86_AVX512_CORE_REG_CLASS_ID_X86_GPR32]
                .alias_set_id,
            descriptor_set->reg_classes[X86_AVX512_CORE_REG_CLASS_ID_X86_GPR64]
                .alias_set_id);
  EXPECT_NE(descriptor_set->reg_classes[X86_AVX512_CORE_REG_CLASS_ID_X86_GPR32]
                .alias_set_id,
            0u);
  EXPECT_EQ(descriptor_set->reg_classes[X86_AVX512_CORE_REG_CLASS_ID_X86_XMM]
                .alias_set_id,
            descriptor_set->reg_classes[X86_AVX512_CORE_REG_CLASS_ID_X86_ZMM]
                .alias_set_id);
  EXPECT_NE(descriptor_set->reg_classes[X86_AVX512_CORE_REG_CLASS_ID_X86_XMM]
                .alias_set_id,
            0u);
}

TEST(X86DescriptorsTest, Avx512CoreDescriptorLookupUsesStableKeys) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_avx512_core_descriptor_set();

  const loom_low_descriptor_t* splat_i32_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.vpbroadcastd.zmm"));
  ASSERT_NE(splat_i32_descriptor, nullptr);
  EXPECT_EQ(splat_i32_descriptor->operand_count, 2u);
  EXPECT_EQ(splat_i32_descriptor->result_count, 1u);

  const loom_low_descriptor_t* splat_f32_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.vbroadcastss.zmm"));
  ASSERT_NE(splat_f32_descriptor, nullptr);
  EXPECT_EQ(splat_f32_descriptor->operand_count, 2u);
  EXPECT_EQ(splat_f32_descriptor->result_count, 1u);

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

  const loom_low_descriptor_t* subtract_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.vpsubd.zmm"));
  ASSERT_NE(subtract_descriptor, nullptr);
  iree_string_view_t subtract_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, subtract_descriptor->key_string_offset, &subtract_key));
  EXPECT_TRUE(
      iree_string_view_equal(subtract_key, IREE_SV("x86.avx512.vpsubd.zmm")));
  EXPECT_EQ(subtract_descriptor->operand_count, 3u);
  EXPECT_EQ(subtract_descriptor->result_count, 1u);

  const loom_low_descriptor_t* multiply_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.vpmulld.zmm"));
  ASSERT_NE(multiply_descriptor, nullptr);
  iree_string_view_t multiply_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, multiply_descriptor->key_string_offset, &multiply_key));
  EXPECT_TRUE(
      iree_string_view_equal(multiply_key, IREE_SV("x86.avx512.vpmulld.zmm")));
  EXPECT_EQ(multiply_descriptor->operand_count, 3u);
  EXPECT_EQ(multiply_descriptor->result_count, 1u);

  const loom_low_descriptor_t* f32_add_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.vaddps.zmm"));
  ASSERT_NE(f32_add_descriptor, nullptr);
  iree_string_view_t f32_add_key = iree_string_view_empty();
  IREE_ASSERT_OK(loom_low_descriptor_set_string(
      descriptor_set, f32_add_descriptor->key_string_offset, &f32_add_key));
  EXPECT_TRUE(
      iree_string_view_equal(f32_add_key, IREE_SV("x86.avx512.vaddps.zmm")));
  EXPECT_EQ(f32_add_descriptor->operand_count, 3u);
  EXPECT_EQ(f32_add_descriptor->result_count, 1u);

  const loom_low_descriptor_t* f32_subtract_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.vsubps.zmm"));
  ASSERT_NE(f32_subtract_descriptor, nullptr);
  EXPECT_EQ(f32_subtract_descriptor->operand_count, 3u);
  EXPECT_EQ(f32_subtract_descriptor->result_count, 1u);

  const loom_low_descriptor_t* f32_multiply_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.vmulps.zmm"));
  ASSERT_NE(f32_multiply_descriptor, nullptr);
  EXPECT_EQ(f32_multiply_descriptor->operand_count, 3u);
  EXPECT_EQ(f32_multiply_descriptor->result_count, 1u);

  const loom_low_descriptor_t* f32_fma_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.vfmadd231ps.zmm"));
  ASSERT_NE(f32_fma_descriptor, nullptr);
  EXPECT_EQ(f32_fma_descriptor->operand_count, 4u);
  EXPECT_EQ(f32_fma_descriptor->result_count, 1u);
  EXPECT_EQ(f32_fma_descriptor->constraint_count, 2u);

  const loom_low_descriptor_t* indexed_load_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("x86.avx512.vmovdqu32.load.indexed.zmm"));
  ASSERT_NE(indexed_load_descriptor, nullptr);
  EXPECT_EQ(indexed_load_descriptor->operand_count, 3u);
  EXPECT_EQ(indexed_load_descriptor->result_count, 1u);
  EXPECT_EQ(indexed_load_descriptor->immediate_count, 2u);
  const loom_low_immediate_t* indexed_load_immediates =
      &descriptor_set->immediates[indexed_load_descriptor->immediate_start];
  EXPECT_EQ(indexed_load_immediates[0].kind, LOOM_LOW_IMMEDIATE_KIND_SIGNED);
  EXPECT_EQ(indexed_load_immediates[1].kind, LOOM_LOW_IMMEDIATE_KIND_ENUM);

  const loom_low_descriptor_t* indexed_store_descriptor = LookupDescriptor(
      descriptor_set, IREE_SV("x86.avx512.vmovdqu32.store.indexed.zmm"));
  ASSERT_NE(indexed_store_descriptor, nullptr);
  EXPECT_EQ(indexed_store_descriptor->operand_count, 3u);
  EXPECT_EQ(indexed_store_descriptor->result_count, 0u);
  EXPECT_EQ(indexed_store_descriptor->immediate_count, 2u);

  const loom_low_descriptor_t* lea_add_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.lea.add.gpr64"));
  ASSERT_NE(lea_add_descriptor, nullptr);
  EXPECT_EQ(lea_add_descriptor->operand_count, 3u);
  EXPECT_EQ(lea_add_descriptor->result_count, 1u);
  EXPECT_EQ(lea_add_descriptor->immediate_count, 0u);

  const loom_low_descriptor_t* imul_descriptor =
      LookupDescriptor(descriptor_set, IREE_SV("x86.avx512.imul.gpr64"));
  ASSERT_NE(imul_descriptor, nullptr);
  EXPECT_EQ(imul_descriptor->operand_count, 3u);
  EXPECT_EQ(imul_descriptor->result_count, 1u);
  EXPECT_EQ(imul_descriptor->immediate_count, 0u);
  EXPECT_EQ(imul_descriptor->constraint_count, 2u);

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
  EXPECT_EQ(load_issue_uses[0].units, 1u);
  EXPECT_EQ(load_issue_uses[1].units, 4u);
  EXPECT_EQ(descriptor_set->resources[load_issue_uses[1].resource_id]
                .capacity_per_cycle,
            4u);

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

  const loom_low_schedule_class_t* dot_schedule =
      &descriptor_set->schedule_classes[dot_descriptor->schedule_class_id];
  ASSERT_EQ(dot_schedule->issue_use_count, 1u);
  const loom_low_issue_use_t* dot_issue_use =
      &descriptor_set->issue_uses[dot_schedule->issue_use_start];
  EXPECT_EQ(dot_issue_use->units, 4u);
  EXPECT_EQ(
      descriptor_set->resources[dot_issue_use->resource_id].capacity_per_cycle,
      4u);
}

TEST(X86DescriptorsTest, Avx512AsmFormsNameUnambiguousPackets) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_x86_avx512_core_descriptor_set();
  ASSERT_GE(descriptor_set->asm_form_count, 6u);

  uint32_t asm_form_ordinal = LOOM_LOW_ASM_FORM_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_asm_form(
      descriptor_set, IREE_SV("vpaddd"), &asm_form_ordinal));
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
            "x86.avx512.vpaddd.zmm");

  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_asm_form(
      descriptor_set, IREE_SV("vpaddd.xmm"), &asm_form_ordinal));
  asm_form =
      loom_low_descriptor_set_asm_form_at(descriptor_set, asm_form_ordinal);
  ASSERT_NE(asm_form, nullptr);
  EXPECT_EQ(asm_form->result_operand_index_count, 1u);
  EXPECT_EQ(asm_form->operand_index_count, 2u);

  descriptor = loom_low_descriptor_set_descriptor_at(
      descriptor_set, asm_form->descriptor_ordinal);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor_set, descriptor->key_string_offset),
            "x86.avx512.vpaddd.xmm");

  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_asm_form(
      descriptor_set, IREE_SV("vpsubd"), &asm_form_ordinal));
  asm_form =
      loom_low_descriptor_set_asm_form_at(descriptor_set, asm_form_ordinal);
  ASSERT_NE(asm_form, nullptr);
  EXPECT_EQ(asm_form->result_operand_index_count, 1u);
  EXPECT_EQ(asm_form->operand_index_count, 2u);

  descriptor = loom_low_descriptor_set_descriptor_at(
      descriptor_set, asm_form->descriptor_ordinal);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor_set, descriptor->key_string_offset),
            "x86.avx512.vpsubd.zmm");

  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_asm_form(
      descriptor_set, IREE_SV("vpmulld"), &asm_form_ordinal));
  asm_form =
      loom_low_descriptor_set_asm_form_at(descriptor_set, asm_form_ordinal);
  ASSERT_NE(asm_form, nullptr);
  EXPECT_EQ(asm_form->result_operand_index_count, 1u);
  EXPECT_EQ(asm_form->operand_index_count, 2u);

  descriptor = loom_low_descriptor_set_descriptor_at(
      descriptor_set, asm_form->descriptor_ordinal);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor_set, descriptor->key_string_offset),
            "x86.avx512.vpmulld.zmm");

  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_asm_form(
      descriptor_set, IREE_SV("vaddps"), &asm_form_ordinal));
  asm_form =
      loom_low_descriptor_set_asm_form_at(descriptor_set, asm_form_ordinal);
  ASSERT_NE(asm_form, nullptr);
  EXPECT_EQ(asm_form->result_operand_index_count, 1u);
  EXPECT_EQ(asm_form->operand_index_count, 2u);

  descriptor = loom_low_descriptor_set_descriptor_at(
      descriptor_set, asm_form->descriptor_ordinal);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(ToString(descriptor_set, descriptor->key_string_offset),
            "x86.avx512.vaddps.zmm");
}

TEST(X86DescriptorsTest, Avx512LowAsmInfersTiedAccumulatorResultType) {
  LowTextAsmTypeInferenceHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_x86_avx512_core_descriptor_set));

  loom_text_low_asm_packet_descriptor_t packet = {};
  IREE_ASSERT_OK(harness.LookupPacket(IREE_SV("x86.avx512.core"),
                                      IREE_SV("vpdpbusd"), &packet));

  loom_value_id_t operands[3] = {};
  IREE_ASSERT_OK(
      harness.DefineRegisterValue(IREE_SV("x86.zmm"), 1, &operands[0]));
  IREE_ASSERT_OK(
      harness.DefineRegisterValue(IREE_SV("x86.zmm"), 1, &operands[1]));
  IREE_ASSERT_OK(
      harness.DefineRegisterValue(IREE_SV("x86.zmm"), 1, &operands[2]));

  loom_type_t result_type = loom_type_none();
  iree_string_view_t diagnostic_detail = iree_string_view_empty();
  IREE_ASSERT_OK(harness.InferResultType(
      &packet, operands, IREE_ARRAYSIZE(operands), /*result_index=*/0,
      &result_type, &diagnostic_detail));
  EXPECT_TRUE(iree_string_view_is_empty(diagnostic_detail));
  EXPECT_TRUE(harness.RegisterTypeEquals(result_type, IREE_SV("x86.zmm"), 1));

  loom_value_id_t invalid_operands[3] = {};
  IREE_ASSERT_OK(
      harness.DefineRegisterValue(IREE_SV("x86.k"), 1, &invalid_operands[0]));
  IREE_ASSERT_OK(
      harness.DefineRegisterValue(IREE_SV("x86.zmm"), 1, &invalid_operands[1]));
  IREE_ASSERT_OK(
      harness.DefineRegisterValue(IREE_SV("x86.zmm"), 1, &invalid_operands[2]));

  result_type = loom_type_none();
  diagnostic_detail = iree_string_view_empty();
  IREE_ASSERT_OK(harness.InferResultType(
      &packet, invalid_operands, IREE_ARRAYSIZE(invalid_operands),
      /*result_index=*/0, &result_type, &diagnostic_detail));
  std::string detail(diagnostic_detail.data, diagnostic_detail.size);
  EXPECT_NE(detail.find("tied operand type"), std::string::npos);
  EXPECT_NE(detail.find("reg<x86.zmm>"), std::string::npos);
}

TEST(X86DescriptorsTest, Avx512LowAsmRegionRoundTrips) {
  LowTextAsmRoundTripHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_x86_avx512_core_descriptor_set));

  const char* source =
      "test.low_asm_region asm<x86.avx512.core> {\n"
      "  jmp 7\n"
      "}\n";
  std::string printed;
  IREE_ASSERT_OK(
      harness.RoundTrip(IREE_SV(source), IREE_SV("x86.avx512.core"), &printed));
  EXPECT_EQ(printed, source);
}

TEST(X86DescriptorsTest, Avx512LowFuncAsmRoundTripsTiedDotArguments) {
  LowFuncAsmRoundTripHarness harness;
  IREE_ASSERT_OK(harness.Initialize(loom_x86_avx512_core_descriptor_set,
                                    &loom_x86_low_target_bundle_avx512_core));

  const char* source =
      "target.profile @x86_target preset(\"x86-avx512\")\n\n"
      "low.func.def target(@x86_target) @dot(%acc: reg<x86.zmm>, %lhs: "
      "reg<x86.zmm>, %rhs: reg<x86.zmm>) -> (reg<x86.zmm>) "
      "asm<x86.avx512.core> {\n"
      "  %out = vpdpbusd %acc, %lhs, %rhs\n"
      "  return %out\n"
      "}\n";
  std::string printed;
  IREE_ASSERT_OK(harness.RoundTripAndVerify(
      IREE_SV(source), IREE_SV("x86.avx512.core"), &printed));
  EXPECT_EQ(printed, source);
}

}  // namespace
}  // namespace loom
