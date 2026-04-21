// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMDGPU_DESCRIPTOR_TEST_UTIL_H_
#define LOOM_TARGET_ARCH_AMDGPU_DESCRIPTOR_TEST_UTIL_H_

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/descriptors.h"

namespace loom::testing {

inline const loom_low_descriptor_t* LookupAmdgpuDescriptorForTest(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  uint32_t ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_EXPECT_OK(
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, key, &ordinal));
  return loom_low_descriptor_set_descriptor_at(descriptor_set, ordinal);
}

inline void ExpectAmdgpuOperandRegisterClassForTest(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand, iree_string_view_t expected_name) {
  ASSERT_EQ(operand->reg_class_alt_count, 1u);
  const uint32_t alt_index = operand->reg_class_alt_start;
  ASSERT_LT(alt_index, descriptor_set->reg_class_alt_count);
  const loom_low_reg_class_alt_t* alt =
      &descriptor_set->reg_class_alts[alt_index];
  ASSERT_NE(alt->reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  ASSERT_LT(alt->reg_class_id, descriptor_set->reg_class_count);
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[alt->reg_class_id];
  iree_string_view_t actual_name = iree_string_view_empty();
  IREE_EXPECT_OK(loom_low_descriptor_set_string(
      descriptor_set, reg_class->name_string_offset, &actual_name));
  EXPECT_TRUE(iree_string_view_equal(actual_name, expected_name));
}

inline void ExpectAmdgpuDsMemoryDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    loom_low_effect_kind_t expected_effect_kind, uint16_t expected_data_units,
    uint32_t expected_width_bits, uint16_t expected_encoding_format_id) {
  const bool is_read = expected_effect_kind == LOOM_LOW_EFFECT_KIND_READ;
  ASSERT_TRUE(is_read || expected_effect_kind == LOOM_LOW_EFFECT_KIND_WRITE);
  const loom_low_descriptor_t* descriptor =
      LookupAmdgpuDescriptorForTest(descriptor_set, key);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->operand_count, 2u);
  EXPECT_EQ(descriptor->result_count, is_read ? 1u : 0u);
  EXPECT_EQ(descriptor->immediate_count, 1u);
  EXPECT_EQ(descriptor->effect_count, 1u);
  EXPECT_EQ(descriptor->encoding_format_id, expected_encoding_format_id);

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  if (is_read) {
    EXPECT_EQ(operands[0].unit_count, expected_data_units);
    EXPECT_EQ(operands[1].unit_count, 1u);
  } else {
    EXPECT_EQ(operands[0].unit_count, 1u);
    EXPECT_EQ(operands[1].unit_count, expected_data_units);
  }

  const loom_low_effect_t* effect =
      &descriptor_set->effects[descriptor->effect_start];
  EXPECT_EQ(effect->kind, expected_effect_kind);
  EXPECT_EQ(effect->memory_space, LOOM_LOW_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(effect->width_bits, expected_width_bits);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);
}

inline void ExpectAmdgpuDs2AddrMemoryDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    loom_low_effect_kind_t expected_effect_kind, uint16_t expected_value_units,
    uint32_t expected_width_bits, uint16_t expected_encoding_format_id) {
  const bool is_read = expected_effect_kind == LOOM_LOW_EFFECT_KIND_READ;
  ASSERT_TRUE(is_read || expected_effect_kind == LOOM_LOW_EFFECT_KIND_WRITE);
  const loom_low_descriptor_t* descriptor =
      LookupAmdgpuDescriptorForTest(descriptor_set, key);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->operand_count, is_read ? 2u : 3u);
  EXPECT_EQ(descriptor->result_count, is_read ? 1u : 0u);
  EXPECT_EQ(descriptor->immediate_count, 2u);
  EXPECT_EQ(descriptor->effect_count, 1u);
  EXPECT_EQ(descriptor->encoding_format_id, expected_encoding_format_id);

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  if (is_read) {
    EXPECT_EQ(operands[0].unit_count, expected_value_units * 2u);
    EXPECT_EQ(operands[1].unit_count, 1u);
  } else {
    EXPECT_EQ(operands[0].unit_count, 1u);
    EXPECT_EQ(operands[1].unit_count, expected_value_units);
    EXPECT_EQ(operands[2].unit_count, expected_value_units);
  }

  const loom_low_immediate_t* immediates =
      &descriptor_set->immediates[descriptor->immediate_start];
  EXPECT_EQ(immediates[0].kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(immediates[0].bit_width, 8u);
  EXPECT_EQ(immediates[0].unsigned_max, 255u);
  EXPECT_EQ(immediates[1].kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(immediates[1].bit_width, 8u);
  EXPECT_EQ(immediates[1].unsigned_max, 255u);

  const loom_low_effect_t* effect =
      &descriptor_set->effects[descriptor->effect_start];
  EXPECT_EQ(effect->kind, expected_effect_kind);
  EXPECT_EQ(effect->memory_space, LOOM_LOW_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(effect->width_bits, expected_width_bits);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);
}

inline void ExpectAmdgpuDs2AddrMemoryDescriptors(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t expected_encoding_format_id) {
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_read2_b32"), LOOM_LOW_EFFECT_KIND_READ,
      1u, 64u, expected_encoding_format_id);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_read2st64_b32"),
      LOOM_LOW_EFFECT_KIND_READ, 1u, 64u, expected_encoding_format_id);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_read2_b64"), LOOM_LOW_EFFECT_KIND_READ,
      2u, 128u, expected_encoding_format_id);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_read2st64_b64"),
      LOOM_LOW_EFFECT_KIND_READ, 2u, 128u, expected_encoding_format_id);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_write2_b32"),
      LOOM_LOW_EFFECT_KIND_WRITE, 1u, 64u, expected_encoding_format_id);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_write2st64_b32"),
      LOOM_LOW_EFFECT_KIND_WRITE, 1u, 64u, expected_encoding_format_id);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_write2_b64"),
      LOOM_LOW_EFFECT_KIND_WRITE, 2u, 128u, expected_encoding_format_id);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_write2st64_b64"),
      LOOM_LOW_EFFECT_KIND_WRITE, 2u, 128u, expected_encoding_format_id);
}

inline void ExpectAmdgpuDsAddtidMemoryDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    loom_low_effect_kind_t expected_effect_kind,
    uint16_t expected_encoding_format_id) {
  const bool is_read = expected_effect_kind == LOOM_LOW_EFFECT_KIND_READ;
  ASSERT_TRUE(is_read || expected_effect_kind == LOOM_LOW_EFFECT_KIND_WRITE);
  const loom_low_descriptor_t* descriptor =
      LookupAmdgpuDescriptorForTest(descriptor_set, key);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->operand_count, 2u);
  EXPECT_EQ(descriptor->result_count, is_read ? 1u : 0u);
  EXPECT_EQ(descriptor->immediate_count, 1u);
  EXPECT_EQ(descriptor->effect_count, 1u);
  EXPECT_EQ(descriptor->encoding_format_id, expected_encoding_format_id);

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  if (is_read) {
    EXPECT_EQ(operands[0].unit_count, 1u);
    EXPECT_EQ(operands[0].role, LOOM_LOW_OPERAND_ROLE_RESULT);
  } else {
    EXPECT_EQ(operands[0].unit_count, 1u);
    EXPECT_EQ(operands[0].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
  }
  EXPECT_EQ(operands[1].unit_count, 1u);
  EXPECT_EQ(operands[1].role, LOOM_LOW_OPERAND_ROLE_RESOURCE);
  EXPECT_EQ(operands[1].encoding_field_id, 0u);
  EXPECT_NE(operands[1].flags & LOOM_LOW_OPERAND_FLAG_IMPLICIT, 0u);
  ExpectAmdgpuOperandRegisterClassForTest(descriptor_set, &operands[1],
                                          IREE_SV("amdgpu.m0"));

  ASSERT_NE(descriptor->canonical_asm_form_ordinal,
            LOOM_LOW_ASM_FORM_ORDINAL_NONE);
  ASSERT_LT(descriptor->canonical_asm_form_ordinal,
            descriptor_set->asm_form_count);
  const loom_low_asm_form_t* asm_form =
      &descriptor_set->asm_forms[descriptor->canonical_asm_form_ordinal];
  EXPECT_EQ(asm_form->result_operand_index_count, is_read ? 1u : 0u);
  EXPECT_EQ(asm_form->operand_index_count, is_read ? 0u : 1u);
  EXPECT_EQ(asm_form->immediate_count, 1u);

  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  EXPECT_EQ(immediate->kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(immediate->bit_width, 8u);
  EXPECT_EQ(immediate->unsigned_max, 255u);

  const loom_low_effect_t* effect =
      &descriptor_set->effects[descriptor->effect_start];
  EXPECT_EQ(effect->kind, expected_effect_kind);
  EXPECT_EQ(effect->memory_space, LOOM_LOW_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(effect->width_bits, 32u);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);
}

inline void ExpectAmdgpuDsAddtidMemoryDescriptors(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t expected_encoding_format_id) {
  ExpectAmdgpuDsAddtidMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_read_addtid_b32"),
      LOOM_LOW_EFFECT_KIND_READ, expected_encoding_format_id);
  ExpectAmdgpuDsAddtidMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_write_addtid_b32"),
      LOOM_LOW_EFFECT_KIND_WRITE, expected_encoding_format_id);
}

}  // namespace loom::testing

#endif  // LOOM_TARGET_ARCH_AMDGPU_DESCRIPTOR_TEST_UTIL_H_
