// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMDGPU_DESCRIPTOR_TEST_UTIL_H_
#define LOOM_TARGET_ARCH_AMDGPU_DESCRIPTOR_TEST_UTIL_H_

#include <stdint.h>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/encoding.h"
#include "loom/target/arch/amdgpu/wait_plan.h"

namespace loom::testing {

inline const loom_low_descriptor_t* LookupAmdgpuDescriptorForTest(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  uint32_t ordinal =
      loom_low_descriptor_set_lookup_descriptor(descriptor_set, key);
  EXPECT_NE(ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  return loom_low_descriptor_set_descriptor_at(descriptor_set, ordinal);
}

inline void ExpectAmdgpuRegisterClassForTest(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t reg_class_id,
    iree_string_view_t expected_name) {
  ASSERT_LT(reg_class_id, descriptor_set->reg_class_count);
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[reg_class_id];
  iree_string_view_t actual_name = loom_low_descriptor_set_string(
      descriptor_set, reg_class->name_string_offset);
  EXPECT_TRUE(iree_string_view_equal(actual_name, expected_name));
  EXPECT_NE(reg_class->flags & LOOM_LOW_REG_CLASS_FLAG_PHYSICAL, 0u);
  EXPECT_GT(reg_class->physical_count, 0u);
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
  iree_string_view_t actual_name = loom_low_descriptor_set_string(
      descriptor_set, reg_class->name_string_offset);
  EXPECT_TRUE(iree_string_view_equal(actual_name, expected_name));
}

inline void ExpectAmdgpuWmmaDescriptorForTest(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    uint16_t expected_lhs_units, uint16_t expected_rhs_units,
    uint16_t expected_accumulator_units) {
  const loom_low_descriptor_t* descriptor =
      LookupAmdgpuDescriptorForTest(descriptor_set, key);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->operand_count, 4u);
  EXPECT_EQ(descriptor->result_count, 1u);
  EXPECT_EQ(descriptor->immediate_count, 0u);
  EXPECT_EQ(descriptor->encoding_format_id, LOOM_AMDGPU_ENCODING_FORMAT_VOP3P);
  ASSERT_EQ(descriptor->constraint_count, 1u);
  const loom_low_constraint_t* constraint =
      &descriptor_set->constraints[descriptor->constraint_start];
  EXPECT_EQ(constraint->kind, LOOM_LOW_CONSTRAINT_KIND_TIED);
  EXPECT_EQ(constraint->lhs_operand_index, 0u);
  EXPECT_EQ(constraint->rhs_operand_index, 3u);

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  EXPECT_EQ(operands[0].unit_count, expected_accumulator_units);
  EXPECT_EQ(operands[1].unit_count, expected_lhs_units);
  EXPECT_EQ(operands[2].unit_count, expected_rhs_units);
  EXPECT_EQ(operands[3].unit_count, expected_accumulator_units);
  EXPECT_EQ(operands[0].reg_class_alt_count, 1u);
  EXPECT_EQ(operands[1].reg_class_alt_count, 1u);
  EXPECT_EQ(operands[2].reg_class_alt_count, 1u);
  EXPECT_EQ(operands[3].reg_class_alt_count, 2u);

  const uint16_t vgpr_class_id =
      descriptor_set->reg_class_alts[operands[0].reg_class_alt_start]
          .reg_class_id;
  EXPECT_NE(vgpr_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_EQ(descriptor_set->reg_class_alts[operands[1].reg_class_alt_start]
                .reg_class_id,
            vgpr_class_id);
  EXPECT_EQ(descriptor_set->reg_class_alts[operands[2].reg_class_alt_start]
                .reg_class_id,
            vgpr_class_id);
  const loom_low_reg_class_alt_t* accumulator_alts =
      &descriptor_set->reg_class_alts[operands[3].reg_class_alt_start];
  EXPECT_EQ(accumulator_alts[0].reg_class_id, vgpr_class_id);
  EXPECT_EQ(accumulator_alts[1].reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  EXPECT_NE(accumulator_alts[1].flags & LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE,
            0u);
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

  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  EXPECT_EQ(immediate->kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(immediate->encoding_id,
            LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_BYTE);

  const loom_low_effect_t* effect =
      &descriptor_set->effects[descriptor->effect_start];
  EXPECT_EQ(effect->kind, expected_effect_kind);
  EXPECT_EQ(effect->memory_space, LOOM_LOW_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(effect->width_bits, expected_width_bits);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);
}

inline void ExpectAmdgpuGlobalMemoryDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    loom_low_effect_kind_t expected_effect_kind, uint16_t expected_value_units,
    uint32_t expected_width_bits, uint16_t expected_encoding_format_id,
    uint16_t expected_offset_bit_width, uint16_t expected_address_units,
    bool expected_saddr_operand, bool expected_implicit_m0) {
  const bool is_read = expected_effect_kind == LOOM_LOW_EFFECT_KIND_READ;
  ASSERT_TRUE(is_read || expected_effect_kind == LOOM_LOW_EFFECT_KIND_WRITE);
  const loom_low_descriptor_t* descriptor =
      LookupAmdgpuDescriptorForTest(descriptor_set, key);
  ASSERT_NE(descriptor, nullptr);
  const uint16_t expected_operand_count = 2u +
                                          (expected_saddr_operand ? 1u : 0u) +
                                          (expected_implicit_m0 ? 1u : 0u);
  const uint16_t expected_cache_immediate_count = 3u;
  EXPECT_EQ(descriptor->operand_count, expected_operand_count);
  EXPECT_EQ(descriptor->result_count, is_read ? 1u : 0u);
  EXPECT_EQ(descriptor->immediate_count, 1u + expected_cache_immediate_count);
  EXPECT_EQ(descriptor->effect_count, 1u);
  EXPECT_EQ(descriptor->encoding_format_id, expected_encoding_format_id);

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  if (is_read) {
    EXPECT_EQ(operands[0].unit_count, expected_value_units);
    EXPECT_EQ(operands[0].role, LOOM_LOW_OPERAND_ROLE_RESULT);
    EXPECT_EQ(operands[1].unit_count, expected_address_units);
    EXPECT_EQ(operands[1].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
  } else {
    EXPECT_EQ(operands[0].unit_count, expected_address_units);
    EXPECT_EQ(operands[0].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
    EXPECT_EQ(operands[1].unit_count, expected_value_units);
    EXPECT_EQ(operands[1].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
  }
  uint16_t next_operand = 2;
  if (expected_saddr_operand) {
    EXPECT_EQ(operands[next_operand].unit_count, 2u);
    EXPECT_EQ(operands[next_operand].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
    ExpectAmdgpuOperandRegisterClassForTest(
        descriptor_set, &operands[next_operand], IREE_SV("amdgpu.sgpr"));
    ++next_operand;
  }
  if (expected_implicit_m0) {
    EXPECT_EQ(operands[next_operand].unit_count, 1u);
    EXPECT_EQ(operands[next_operand].role, LOOM_LOW_OPERAND_ROLE_RESOURCE);
    EXPECT_EQ(operands[next_operand].encoding_field_id, 0u);
    EXPECT_NE(operands[next_operand].flags & LOOM_LOW_OPERAND_FLAG_IMPLICIT,
              0u);
    ExpectAmdgpuOperandRegisterClassForTest(
        descriptor_set, &operands[next_operand], IREE_SV("amdgpu.m0"));
  }

  if (descriptor->canonical_asm_form_ordinal !=
      LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
    ASSERT_LT(descriptor->canonical_asm_form_ordinal,
              descriptor_set->asm_form_count);
    const loom_low_asm_form_t* asm_form =
        &descriptor_set->asm_forms[descriptor->canonical_asm_form_ordinal];
    EXPECT_EQ(asm_form->descriptor_ordinal,
              static_cast<uint32_t>(descriptor - descriptor_set->descriptors));
    EXPECT_EQ(asm_form->result_operand_index_count, is_read ? 1u : 0u);
    EXPECT_EQ(asm_form->operand_index_count,
              descriptor->operand_count - descriptor->result_count -
                  (expected_implicit_m0 ? 1u : 0u));
    EXPECT_EQ(asm_form->immediate_count, descriptor->immediate_count);
  }

  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  EXPECT_EQ(immediate->kind, LOOM_LOW_IMMEDIATE_KIND_SIGNED);
  EXPECT_EQ(immediate->bit_width, expected_offset_bit_width);
  EXPECT_EQ(immediate->encoding_id,
            LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_BYTE);
  EXPECT_EQ(immediate->signed_min,
            -(INT64_C(1) << (expected_offset_bit_width - 1)));
  EXPECT_EQ(immediate->unsigned_max,
            (UINT64_C(1) << (expected_offset_bit_width - 1)) - 1);
  for (uint16_t i = 1; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* cache_immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    EXPECT_EQ(cache_immediate->kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
    EXPECT_GT(cache_immediate->bit_width, 0u);
    ASSERT_LT(cache_immediate->bit_width, 64u);
    EXPECT_EQ(cache_immediate->unsigned_max,
              (UINT64_C(1) << cache_immediate->bit_width) - 1);
    EXPECT_EQ(cache_immediate->default_value, 0);
    EXPECT_TRUE(iree_any_bit_set(cache_immediate->flags,
                                 LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE));
  }

  const loom_low_effect_t* effect =
      &descriptor_set->effects[descriptor->effect_start];
  EXPECT_EQ(effect->kind, expected_effect_kind);
  EXPECT_EQ(effect->memory_space, LOOM_LOW_MEMORY_SPACE_GLOBAL);
  EXPECT_EQ(effect->width_bits, expected_width_bits);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);
}

inline void ExpectAmdgpuGlobalMemoryDescriptors(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t expected_encoding_format_id, uint16_t expected_offset_bit_width,
    bool expected_implicit_m0) {
  ExpectAmdgpuGlobalMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_load_b32"),
      LOOM_LOW_EFFECT_KIND_READ, 1u, 32u, expected_encoding_format_id,
      expected_offset_bit_width, 2u, /*expected_saddr_operand=*/false,
      expected_implicit_m0);
  ExpectAmdgpuGlobalMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_load_b64"),
      LOOM_LOW_EFFECT_KIND_READ, 2u, 64u, expected_encoding_format_id,
      expected_offset_bit_width, 2u, /*expected_saddr_operand=*/false,
      expected_implicit_m0);
  ExpectAmdgpuGlobalMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_load_b128"),
      LOOM_LOW_EFFECT_KIND_READ, 4u, 128u, expected_encoding_format_id,
      expected_offset_bit_width, 2u, /*expected_saddr_operand=*/false,
      expected_implicit_m0);
  ExpectAmdgpuGlobalMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_store_b32"),
      LOOM_LOW_EFFECT_KIND_WRITE, 1u, 32u, expected_encoding_format_id,
      expected_offset_bit_width, 2u, /*expected_saddr_operand=*/false,
      expected_implicit_m0);
  ExpectAmdgpuGlobalMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_store_b64"),
      LOOM_LOW_EFFECT_KIND_WRITE, 2u, 64u, expected_encoding_format_id,
      expected_offset_bit_width, 2u, /*expected_saddr_operand=*/false,
      expected_implicit_m0);
  ExpectAmdgpuGlobalMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_store_b128"),
      LOOM_LOW_EFFECT_KIND_WRITE, 4u, 128u, expected_encoding_format_id,
      expected_offset_bit_width, 2u, /*expected_saddr_operand=*/false,
      expected_implicit_m0);
}

inline void ExpectAmdgpuGlobalSaddrMemoryDescriptors(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t expected_encoding_format_id, uint16_t expected_offset_bit_width,
    bool expected_implicit_m0) {
  ExpectAmdgpuGlobalMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_load_b32_saddr"),
      LOOM_LOW_EFFECT_KIND_READ, 1u, 32u, expected_encoding_format_id,
      expected_offset_bit_width, 1u, /*expected_saddr_operand=*/true,
      expected_implicit_m0);
  ExpectAmdgpuGlobalMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_load_b64_saddr"),
      LOOM_LOW_EFFECT_KIND_READ, 2u, 64u, expected_encoding_format_id,
      expected_offset_bit_width, 1u, /*expected_saddr_operand=*/true,
      expected_implicit_m0);
  ExpectAmdgpuGlobalMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_load_b128_saddr"),
      LOOM_LOW_EFFECT_KIND_READ, 4u, 128u, expected_encoding_format_id,
      expected_offset_bit_width, 1u, /*expected_saddr_operand=*/true,
      expected_implicit_m0);
  ExpectAmdgpuGlobalMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_store_b32_saddr"),
      LOOM_LOW_EFFECT_KIND_WRITE, 1u, 32u, expected_encoding_format_id,
      expected_offset_bit_width, 1u, /*expected_saddr_operand=*/true,
      expected_implicit_m0);
  ExpectAmdgpuGlobalMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_store_b64_saddr"),
      LOOM_LOW_EFFECT_KIND_WRITE, 2u, 64u, expected_encoding_format_id,
      expected_offset_bit_width, 1u, /*expected_saddr_operand=*/true,
      expected_implicit_m0);
  ExpectAmdgpuGlobalMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_store_b128_saddr"),
      LOOM_LOW_EFFECT_KIND_WRITE, 4u, 128u, expected_encoding_format_id,
      expected_offset_bit_width, 1u, /*expected_saddr_operand=*/true,
      expected_implicit_m0);
}

inline void ExpectAmdgpuGlobalLoadLdsDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    uint16_t expected_address_units, uint16_t expected_width_bits,
    bool expected_saddr_operand) {
  const loom_low_descriptor_t* descriptor =
      LookupAmdgpuDescriptorForTest(descriptor_set, key);
  ASSERT_NE(descriptor, nullptr);
  const uint16_t expected_operand_count =
      2u + (expected_saddr_operand ? 1u : 0u);
  const uint16_t expected_cache_immediate_count = 3u;
  EXPECT_EQ(descriptor->operand_count, expected_operand_count);
  EXPECT_EQ(descriptor->result_count, 0u);
  EXPECT_EQ(descriptor->immediate_count, 1u + expected_cache_immediate_count);
  EXPECT_EQ(descriptor->effect_count, 2u);
  EXPECT_EQ(descriptor->encoding_format_id,
            LOOM_AMDGPU_ENCODING_FORMAT_FLAT_GLBL);
  EXPECT_EQ(descriptor->canonical_asm_form_ordinal,
            LOOM_LOW_ASM_FORM_ORDINAL_NONE);

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  EXPECT_EQ(operands[0].unit_count, expected_address_units);
  EXPECT_EQ(operands[0].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
  ExpectAmdgpuOperandRegisterClassForTest(descriptor_set, &operands[0],
                                          IREE_SV("amdgpu.vgpr"));
  uint16_t next_operand = 1;
  if (expected_saddr_operand) {
    EXPECT_EQ(operands[next_operand].unit_count, 2u);
    EXPECT_EQ(operands[next_operand].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
    ExpectAmdgpuOperandRegisterClassForTest(
        descriptor_set, &operands[next_operand], IREE_SV("amdgpu.sgpr"));
    ++next_operand;
  }
  EXPECT_EQ(operands[next_operand].unit_count, 1u);
  EXPECT_EQ(operands[next_operand].role, LOOM_LOW_OPERAND_ROLE_RESOURCE);
  EXPECT_EQ(operands[next_operand].encoding_field_id, 0u);
  EXPECT_NE(operands[next_operand].flags & LOOM_LOW_OPERAND_FLAG_IMPLICIT, 0u);
  ExpectAmdgpuOperandRegisterClassForTest(
      descriptor_set, &operands[next_operand], IREE_SV("amdgpu.m0"));

  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  EXPECT_EQ(immediate->kind, LOOM_LOW_IMMEDIATE_KIND_SIGNED);
  EXPECT_EQ(immediate->bit_width, 13u);
  EXPECT_EQ(immediate->encoding_id,
            LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_BYTE);

  const loom_low_effect_t* effects =
      &descriptor_set->effects[descriptor->effect_start];
  EXPECT_EQ(effects[0].kind, LOOM_LOW_EFFECT_KIND_READ);
  EXPECT_EQ(effects[0].memory_space, LOOM_LOW_MEMORY_SPACE_GLOBAL);
  EXPECT_EQ(effects[0].counter_id, LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD);
  EXPECT_EQ(effects[0].width_bits, expected_width_bits);
  EXPECT_NE(effects[0].flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);
  EXPECT_EQ(effects[1].kind, LOOM_LOW_EFFECT_KIND_WRITE);
  EXPECT_EQ(effects[1].memory_space, LOOM_LOW_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(effects[1].counter_id, LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD);
  EXPECT_EQ(effects[1].width_bits, expected_width_bits);
  EXPECT_NE(effects[1].flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);

  ASSERT_NE(descriptor->schedule_class_id, LOOM_LOW_SCHEDULE_CLASS_NONE);
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[descriptor->schedule_class_id];
  EXPECT_TRUE(iree_any_bit_set(schedule_class->flags,
                               LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD));
  EXPECT_TRUE(iree_any_bit_set(schedule_class->flags,
                               LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_STORE));
}

inline void ExpectAmdgpuGlobalLoadLdsDescriptors(
    const loom_low_descriptor_set_t* descriptor_set) {
  ExpectAmdgpuGlobalLoadLdsDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_load_lds_dword"), 2u, 32u,
      /*expected_saddr_operand=*/false);
  ExpectAmdgpuGlobalLoadLdsDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_load_lds_dwordx3"), 2u, 96u,
      /*expected_saddr_operand=*/false);
  ExpectAmdgpuGlobalLoadLdsDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_load_lds_dwordx4"), 2u, 128u,
      /*expected_saddr_operand=*/false);
  ExpectAmdgpuGlobalLoadLdsDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_load_lds_dword_saddr"), 1u, 32u,
      /*expected_saddr_operand=*/true);
  ExpectAmdgpuGlobalLoadLdsDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_load_lds_dwordx3_saddr"), 1u, 96u,
      /*expected_saddr_operand=*/true);
  ExpectAmdgpuGlobalLoadLdsDescriptor(
      descriptor_set, IREE_SV("amdgpu.global_load_lds_dwordx4_saddr"), 1u, 128u,
      /*expected_saddr_operand=*/true);
}

inline void ExpectAmdgpuFlatAtomicDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    bool expected_returns_old_value, uint16_t expected_value_units,
    uint16_t expected_encoding_format_id, uint16_t expected_offset_bit_width,
    bool expected_offset_signed, uint16_t expected_cache_immediate_count,
    bool expected_implicit_m0) {
  const loom_low_descriptor_t* descriptor =
      LookupAmdgpuDescriptorForTest(descriptor_set, key);
  ASSERT_NE(descriptor, nullptr);
  const uint16_t expected_operand_count =
      (expected_returns_old_value ? 3u : 2u) + (expected_implicit_m0 ? 1u : 0u);
  ASSERT_EQ(descriptor->operand_count, expected_operand_count);
  EXPECT_EQ(descriptor->result_count, expected_returns_old_value ? 1u : 0u);
  ASSERT_EQ(descriptor->immediate_count, 1u + expected_cache_immediate_count);
  ASSERT_EQ(descriptor->effect_count, 2u);
  EXPECT_EQ(descriptor->encoding_format_id, expected_encoding_format_id);
  EXPECT_EQ(descriptor->canonical_asm_form_ordinal,
            LOOM_LOW_ASM_FORM_ORDINAL_NONE);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  uint16_t operand_index = 0;
  if (expected_returns_old_value) {
    EXPECT_EQ(operands[operand_index].role, LOOM_LOW_OPERAND_ROLE_RESULT);
    EXPECT_EQ(operands[operand_index].unit_count, 1u);
    ++operand_index;
  }
  EXPECT_EQ(operands[operand_index].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
  EXPECT_EQ(operands[operand_index].unit_count, 2u);
  ++operand_index;
  EXPECT_EQ(operands[operand_index].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
  EXPECT_EQ(operands[operand_index].unit_count, expected_value_units);
  ++operand_index;
  if (expected_implicit_m0) {
    EXPECT_EQ(operands[operand_index].unit_count, 1u);
    EXPECT_EQ(operands[operand_index].role, LOOM_LOW_OPERAND_ROLE_RESOURCE);
    EXPECT_EQ(operands[operand_index].encoding_field_id, 0u);
    EXPECT_NE(operands[operand_index].flags & LOOM_LOW_OPERAND_FLAG_IMPLICIT,
              0u);
    ExpectAmdgpuOperandRegisterClassForTest(
        descriptor_set, &operands[operand_index], IREE_SV("amdgpu.m0"));
    ++operand_index;
  }
  EXPECT_EQ(operand_index, descriptor->operand_count);

  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  EXPECT_EQ(immediate->kind, expected_offset_signed
                                 ? LOOM_LOW_IMMEDIATE_KIND_SIGNED
                                 : LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(immediate->bit_width, expected_offset_bit_width);
  EXPECT_EQ(immediate->encoding_id,
            LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_BYTE);
  for (uint16_t i = 1; i < descriptor->immediate_count; ++i) {
    const loom_low_immediate_t* cache_immediate =
        &descriptor_set->immediates[descriptor->immediate_start + i];
    EXPECT_EQ(cache_immediate->kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
    EXPECT_TRUE(iree_any_bit_set(cache_immediate->flags,
                                 LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE));
  }

  const uint32_t expected_counter_id =
      expected_returns_old_value ? LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD
                                 : LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE;
  const loom_low_effect_t* effects =
      &descriptor_set->effects[descriptor->effect_start];
  EXPECT_EQ(effects[0].kind, LOOM_LOW_EFFECT_KIND_READ);
  EXPECT_EQ(effects[0].memory_space, LOOM_LOW_MEMORY_SPACE_GENERIC);
  EXPECT_EQ(effects[0].counter_id, expected_counter_id);
  EXPECT_EQ(effects[0].width_bits, 32u);
  EXPECT_NE(effects[0].flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);
  EXPECT_EQ(effects[1].kind, LOOM_LOW_EFFECT_KIND_WRITE);
  EXPECT_EQ(effects[1].memory_space, LOOM_LOW_MEMORY_SPACE_GENERIC);
  EXPECT_EQ(effects[1].counter_id, expected_counter_id);
  EXPECT_EQ(effects[1].width_bits, 32u);
  EXPECT_NE(effects[1].flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);

  ASSERT_NE(descriptor->schedule_class_id, LOOM_LOW_SCHEDULE_CLASS_NONE);
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[descriptor->schedule_class_id];
  EXPECT_TRUE(iree_any_bit_set(schedule_class->flags,
                               LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD));
  EXPECT_TRUE(iree_any_bit_set(schedule_class->flags,
                               LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_STORE));
}

inline void ExpectAmdgpuFlatAtomicDescriptors(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t expected_encoding_format_id, uint16_t expected_offset_bit_width,
    bool expected_offset_signed, uint16_t expected_cache_immediate_count,
    bool expected_implicit_m0, bool expected_f32_minmax) {
  ExpectAmdgpuFlatAtomicDescriptor(
      descriptor_set, IREE_SV("amdgpu.flat_atomic_add_u32"),
      /*expected_returns_old_value=*/false, 1u, expected_encoding_format_id,
      expected_offset_bit_width, expected_offset_signed,
      expected_cache_immediate_count, expected_implicit_m0);
  ExpectAmdgpuFlatAtomicDescriptor(
      descriptor_set, IREE_SV("amdgpu.flat_atomic_add_u32_rtn"),
      /*expected_returns_old_value=*/true, 1u, expected_encoding_format_id,
      expected_offset_bit_width, expected_offset_signed,
      expected_cache_immediate_count, expected_implicit_m0);
  ExpectAmdgpuFlatAtomicDescriptor(
      descriptor_set, IREE_SV("amdgpu.flat_atomic_cmpswap_b32_rtn"),
      /*expected_returns_old_value=*/true, 2u, expected_encoding_format_id,
      expected_offset_bit_width, expected_offset_signed,
      expected_cache_immediate_count, expected_implicit_m0);
  ExpectAmdgpuFlatAtomicDescriptor(
      descriptor_set, IREE_SV("amdgpu.flat_atomic_min_i32"),
      /*expected_returns_old_value=*/false, 1u, expected_encoding_format_id,
      expected_offset_bit_width, expected_offset_signed,
      expected_cache_immediate_count, expected_implicit_m0);
  ExpectAmdgpuFlatAtomicDescriptor(
      descriptor_set, IREE_SV("amdgpu.flat_atomic_max_u32_rtn"),
      /*expected_returns_old_value=*/true, 1u, expected_encoding_format_id,
      expected_offset_bit_width, expected_offset_signed,
      expected_cache_immediate_count, expected_implicit_m0);
  ExpectAmdgpuFlatAtomicDescriptor(
      descriptor_set, IREE_SV("amdgpu.flat_atomic_add_f32"),
      /*expected_returns_old_value=*/false, 1u, expected_encoding_format_id,
      expected_offset_bit_width, expected_offset_signed,
      expected_cache_immediate_count, expected_implicit_m0);
  if (expected_f32_minmax) {
    ExpectAmdgpuFlatAtomicDescriptor(
        descriptor_set, IREE_SV("amdgpu.flat_atomic_min_f32"),
        /*expected_returns_old_value=*/false, 1u, expected_encoding_format_id,
        expected_offset_bit_width, expected_offset_signed,
        expected_cache_immediate_count, expected_implicit_m0);
    ExpectAmdgpuFlatAtomicDescriptor(
        descriptor_set, IREE_SV("amdgpu.flat_atomic_max_f32_rtn"),
        /*expected_returns_old_value=*/true, 1u, expected_encoding_format_id,
        expected_offset_bit_width, expected_offset_signed,
        expected_cache_immediate_count, expected_implicit_m0);
  }
}

inline void ExpectAmdgpuCacheControlDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    uint16_t expected_encoding_format_id,
    uint32_t expected_immediate_count = 0) {
  const loom_low_descriptor_t* descriptor =
      LookupAmdgpuDescriptorForTest(descriptor_set, key);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->operand_count, 0u);
  EXPECT_EQ(descriptor->result_count, 0u);
  EXPECT_EQ(descriptor->immediate_count, expected_immediate_count);
  EXPECT_EQ(descriptor->effect_count, 1u);
  EXPECT_EQ(descriptor->encoding_format_id, expected_encoding_format_id);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);
  ASSERT_NE(descriptor->schedule_class_id, LOOM_LOW_SCHEDULE_CLASS_NONE);
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[descriptor->schedule_class_id];
  EXPECT_NE(schedule_class->flags & LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL, 0u);

  const loom_low_effect_t* effect =
      &descriptor_set->effects[descriptor->effect_start];
  EXPECT_EQ(effect->kind, LOOM_LOW_EFFECT_KIND_BARRIER);
  EXPECT_EQ(effect->memory_space, LOOM_LOW_MEMORY_SPACE_GENERIC);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_ORDERED, 0u);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);
}

inline void ExpectAmdgpuDcacheDiscardDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key) {
  const loom_low_descriptor_t* descriptor =
      LookupAmdgpuDescriptorForTest(descriptor_set, key);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->operand_count, 2u);
  EXPECT_EQ(descriptor->result_count, 0u);
  EXPECT_EQ(descriptor->immediate_count, 0u);
  EXPECT_EQ(descriptor->effect_count, 1u);
  EXPECT_EQ(descriptor->encoding_format_id, LOOM_AMDGPU_ENCODING_FORMAT_SMEM);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  EXPECT_EQ(operands[0].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
  EXPECT_EQ(operands[0].unit_count, 2u);
  EXPECT_EQ(operands[1].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
  EXPECT_EQ(operands[1].unit_count, 1u);

  const loom_low_effect_t* effect =
      &descriptor_set->effects[descriptor->effect_start];
  EXPECT_EQ(effect->kind, LOOM_LOW_EFFECT_KIND_BARRIER);
  EXPECT_EQ(effect->memory_space, LOOM_LOW_MEMORY_SPACE_GENERIC);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_ORDERED, 0u);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);
}

inline void ExpectAmdgpuInstructionPrefetchDistanceDescriptor(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_low_descriptor_t* descriptor = LookupAmdgpuDescriptorForTest(
      descriptor_set, IREE_SV("amdgpu.s_set_inst_prefetch_distance"));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->operand_count, 0u);
  EXPECT_EQ(descriptor->result_count, 0u);
  EXPECT_EQ(descriptor->immediate_count, 1u);
  EXPECT_EQ(descriptor->effect_count, 1u);
  EXPECT_EQ(descriptor->encoding_format_id, LOOM_AMDGPU_ENCODING_FORMAT_SOPP);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);

  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  EXPECT_EQ(immediate->kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(immediate->bit_width, 16u);
  EXPECT_EQ(immediate->unsigned_max, 65535u);

  const loom_low_effect_t* effect =
      &descriptor_set->effects[descriptor->effect_start];
  EXPECT_EQ(effect->kind, LOOM_LOW_EFFECT_KIND_BARRIER);
  EXPECT_EQ(effect->memory_space, LOOM_LOW_MEMORY_SPACE_GENERIC);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_ORDERED, 0u);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);
}

inline void ExpectAmdgpuPrefetchDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    uint16_t expected_operand_count, uint16_t expected_base_units,
    loom_low_operand_role_t expected_base_role,
    loom_low_memory_space_t expected_memory_space) {
  ASSERT_TRUE(expected_operand_count == 1u || expected_operand_count == 2u);
  const loom_low_descriptor_t* descriptor =
      LookupAmdgpuDescriptorForTest(descriptor_set, key);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->operand_count, expected_operand_count);
  EXPECT_EQ(descriptor->result_count, 0u);
  EXPECT_EQ(descriptor->immediate_count, 2u);
  EXPECT_EQ(descriptor->effect_count, 1u);
  EXPECT_EQ(descriptor->encoding_format_id, LOOM_AMDGPU_ENCODING_FORMAT_SMEM);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);
  ASSERT_NE(descriptor->schedule_class_id, LOOM_LOW_SCHEDULE_CLASS_NONE);
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[descriptor->schedule_class_id];
  EXPECT_NE(schedule_class->flags & LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD, 0u);

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  if (expected_operand_count == 2u) {
    EXPECT_EQ(operands[0].role, expected_base_role);
    EXPECT_EQ(operands[0].unit_count, expected_base_units);
    EXPECT_EQ(operands[1].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
    EXPECT_EQ(operands[1].unit_count, 1u);
  } else {
    EXPECT_EQ(operands[0].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
    EXPECT_EQ(operands[0].unit_count, 1u);
  }

  const loom_low_immediate_t* immediates =
      &descriptor_set->immediates[descriptor->immediate_start];
  EXPECT_EQ(immediates[0].kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(immediates[0].bit_width, 24u);
  EXPECT_EQ(immediates[0].encoding_id,
            LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_BYTE);
  EXPECT_EQ(immediates[1].kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(immediates[1].bit_width, 5u);
  EXPECT_EQ(immediates[1].unsigned_max, 31u);

  const loom_low_effect_t* effect =
      &descriptor_set->effects[descriptor->effect_start];
  EXPECT_EQ(effect->kind, LOOM_LOW_EFFECT_KIND_READ);
  EXPECT_EQ(effect->memory_space, expected_memory_space);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);
}

inline void ExpectAmdgpuDs2AddrMemoryDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    loom_low_effect_kind_t expected_effect_kind, uint16_t expected_value_units,
    uint32_t expected_width_bits, uint16_t expected_encoding_format_id,
    uint16_t expected_immediate_encoding_id) {
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
  EXPECT_EQ(immediates[0].encoding_id, expected_immediate_encoding_id);
  EXPECT_EQ(immediates[1].kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(immediates[1].bit_width, 8u);
  EXPECT_EQ(immediates[1].unsigned_max, 255u);
  EXPECT_EQ(immediates[1].encoding_id, expected_immediate_encoding_id);

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
      1u, 64u, expected_encoding_format_id,
      LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_read2st64_b32"),
      LOOM_LOW_EFFECT_KIND_READ, 1u, 64u, expected_encoding_format_id,
      LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD_STRIDE64);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_read2_b64"), LOOM_LOW_EFFECT_KIND_READ,
      2u, 128u, expected_encoding_format_id,
      LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_QWORD);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_read2st64_b64"),
      LOOM_LOW_EFFECT_KIND_READ, 2u, 128u, expected_encoding_format_id,
      LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_QWORD_STRIDE64);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_write2_b32"),
      LOOM_LOW_EFFECT_KIND_WRITE, 1u, 64u, expected_encoding_format_id,
      LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_write2st64_b32"),
      LOOM_LOW_EFFECT_KIND_WRITE, 1u, 64u, expected_encoding_format_id,
      LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD_STRIDE64);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_write2_b64"),
      LOOM_LOW_EFFECT_KIND_WRITE, 2u, 128u, expected_encoding_format_id,
      LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_QWORD);
  ExpectAmdgpuDs2AddrMemoryDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_write2st64_b64"),
      LOOM_LOW_EFFECT_KIND_WRITE, 2u, 128u, expected_encoding_format_id,
      LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_QWORD_STRIDE64);
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
  EXPECT_EQ(immediate->encoding_id,
            LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_BYTE);

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

inline void ExpectAmdgpuDsCrosslaneDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    uint16_t expected_operand_count, uint16_t expected_encoding_format_id) {
  const loom_low_descriptor_t* descriptor =
      LookupAmdgpuDescriptorForTest(descriptor_set, key);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->operand_count, expected_operand_count);
  EXPECT_EQ(descriptor->result_count, 1u);
  EXPECT_EQ(descriptor->immediate_count, 1u);
  EXPECT_EQ(descriptor->effect_count, 1u);
  EXPECT_EQ(descriptor->encoding_format_id, expected_encoding_format_id);
  EXPECT_EQ(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE, 0u);
  EXPECT_EQ(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  EXPECT_EQ(operands[0].unit_count, 1u);
  EXPECT_EQ(operands[0].role, LOOM_LOW_OPERAND_ROLE_RESULT);
  EXPECT_EQ(operands[1].unit_count, 1u);
  EXPECT_EQ(operands[1].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
  if (expected_operand_count == 3u) {
    EXPECT_EQ(operands[2].unit_count, 1u);
    EXPECT_EQ(operands[2].role, LOOM_LOW_OPERAND_ROLE_OPERAND);
  }

  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  EXPECT_EQ(immediate->kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(immediate->bit_width, 16u);
  EXPECT_EQ(immediate->unsigned_max, 65535u);
  EXPECT_EQ(immediate->encoding_field_id, 0u);
  EXPECT_EQ(immediate->encoding_id,
            LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DS16);

  const loom_low_effect_t* effect =
      &descriptor_set->effects[descriptor->effect_start];
  EXPECT_EQ(effect->kind, LOOM_LOW_EFFECT_KIND_CONVERGENT);
  EXPECT_EQ(effect->memory_space, LOOM_LOW_MEMORY_SPACE_NONE);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_ORDERED, 0u);
}

inline void ExpectAmdgpuDsCrosslaneDescriptors(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t expected_encoding_format_id, bool expect_fetch_invalid) {
  ExpectAmdgpuDsCrosslaneDescriptor(descriptor_set,
                                    IREE_SV("amdgpu.ds_swizzle_b32"), 2u,
                                    expected_encoding_format_id);
  ExpectAmdgpuDsCrosslaneDescriptor(descriptor_set,
                                    IREE_SV("amdgpu.ds_permute_b32"), 3u,
                                    expected_encoding_format_id);
  ExpectAmdgpuDsCrosslaneDescriptor(descriptor_set,
                                    IREE_SV("amdgpu.ds_bpermute_b32"), 3u,
                                    expected_encoding_format_id);
  if (expect_fetch_invalid) {
    ExpectAmdgpuDsCrosslaneDescriptor(descriptor_set,
                                      IREE_SV("amdgpu.ds_bpermute_fi_b32"), 3u,
                                      expected_encoding_format_id);
  }
}

inline void ExpectAmdgpuDsTransposeReadDescriptor(
    const loom_low_descriptor_set_t* descriptor_set, iree_string_view_t key,
    uint16_t expected_value_units, uint32_t expected_width_bits,
    uint16_t expected_encoding_format_id) {
  const loom_low_descriptor_t* descriptor =
      LookupAmdgpuDescriptorForTest(descriptor_set, key);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->operand_count, 2u);
  EXPECT_EQ(descriptor->result_count, 1u);
  EXPECT_EQ(descriptor->immediate_count, 1u);
  EXPECT_EQ(descriptor->effect_count, 1u);
  EXPECT_EQ(descriptor->encoding_format_id, expected_encoding_format_id);

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  EXPECT_EQ(operands[0].unit_count, expected_value_units);
  EXPECT_EQ(operands[0].role, LOOM_LOW_OPERAND_ROLE_RESULT);
  EXPECT_EQ(operands[1].unit_count, 1u);
  EXPECT_EQ(operands[1].role, LOOM_LOW_OPERAND_ROLE_OPERAND);

  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];
  EXPECT_EQ(immediate->kind, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED);
  EXPECT_EQ(immediate->bit_width, 16u);
  EXPECT_EQ(immediate->unsigned_max, 65535u);
  EXPECT_EQ(immediate->encoding_field_id, 0u);
  EXPECT_EQ(immediate->encoding_id,
            LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DS16);

  const loom_low_effect_t* effect =
      &descriptor_set->effects[descriptor->effect_start];
  EXPECT_EQ(effect->kind, LOOM_LOW_EFFECT_KIND_READ);
  EXPECT_EQ(effect->memory_space, LOOM_LOW_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(effect->width_bits, expected_width_bits);
  EXPECT_NE(effect->flags & LOOM_LOW_EFFECT_FLAG_DEPENDENCY, 0u);
  EXPECT_NE(descriptor->flags & LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING, 0u);
}

inline void ExpectAmdgpuGfx950DsTransposeReadDescriptors(
    const loom_low_descriptor_set_t* descriptor_set) {
  ExpectAmdgpuDsTransposeReadDescriptor(descriptor_set,
                                        IREE_SV("amdgpu.ds_read_b64_tr_b4"), 2u,
                                        64u, LOOM_AMDGPU_ENCODING_FORMAT_DS);
  ExpectAmdgpuDsTransposeReadDescriptor(descriptor_set,
                                        IREE_SV("amdgpu.ds_read_b96_tr_b6"), 3u,
                                        96u, LOOM_AMDGPU_ENCODING_FORMAT_DS);
  ExpectAmdgpuDsTransposeReadDescriptor(descriptor_set,
                                        IREE_SV("amdgpu.ds_read_b64_tr_b8"), 2u,
                                        64u, LOOM_AMDGPU_ENCODING_FORMAT_DS);
  ExpectAmdgpuDsTransposeReadDescriptor(
      descriptor_set, IREE_SV("amdgpu.ds_read_b64_tr_b16"), 2u, 64u,
      LOOM_AMDGPU_ENCODING_FORMAT_DS);
}

}  // namespace loom::testing

#endif  // LOOM_TARGET_ARCH_AMDGPU_DESCRIPTOR_TEST_UTIL_H_
