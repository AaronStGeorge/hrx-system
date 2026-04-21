// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/descriptor_ids.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amdgpu/gfx11_descriptors.h"
#include "loom/target/arch/amdgpu/gfx1250_descriptors.h"
#include "loom/target/arch/amdgpu/gfx12_descriptors.h"
#include "loom/target/arch/amdgpu/gfx950_descriptors.h"

namespace {

void ExpectHasDescriptor(const loom_low_descriptor_set_t* descriptor_set,
                         uint64_t descriptor_id) {
  EXPECT_NE(loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                            descriptor_id),
            LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
}

void ExpectMissingDescriptor(const loom_low_descriptor_set_t* descriptor_set,
                             uint64_t descriptor_id) {
  EXPECT_EQ(loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                            descriptor_id),
            LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
}

void ExpectImmediateEncoding(const loom_low_descriptor_set_t* descriptor_set,
                             uint64_t descriptor_id,
                             uint16_t expected_encoding_id) {
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                      descriptor_id);
  ASSERT_NE(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  ASSERT_NE(descriptor, nullptr);
  ASSERT_GT(descriptor->immediate_count, 0u);
  const loom_low_immediate_t* immediates =
      &descriptor_set->immediates[descriptor->immediate_start];
  for (uint16_t i = 0; i < descriptor->immediate_count; ++i) {
    EXPECT_EQ(immediates[i].encoding_id, expected_encoding_id);
  }
}

TEST(AmdgpuDescriptorIdsTest, CommonLoweringIdsResolveAcrossDescriptorSets) {
  const loom_low_descriptor_set_t* gfx11 =
      loom_amdgpu_gfx11_core_descriptor_set();
  const loom_low_descriptor_set_t* gfx12 =
      loom_amdgpu_gfx12_core_descriptor_set();
  const loom_low_descriptor_set_t* gfx1250 =
      loom_amdgpu_gfx1250_core_descriptor_set();
  const loom_low_descriptor_set_t* gfx950 =
      loom_amdgpu_gfx950_core_descriptor_set();

  for (const loom_low_descriptor_set_t* descriptor_set :
       {gfx11, gfx12, gfx1250, gfx950}) {
    ExpectHasDescriptor(descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32);
    ExpectHasDescriptor(descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32);
    ExpectHasDescriptor(descriptor_set,
                        LOOM_AMDGPU_DESCRIPTOR_ID_S_LOAD_DWORDX2);
    ExpectHasDescriptor(descriptor_set,
                        LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORD);
    ExpectHasDescriptor(descriptor_set,
                        LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORD);
    ExpectHasDescriptor(descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ_B128);
    ExpectHasDescriptor(descriptor_set,
                        LOOM_AMDGPU_DESCRIPTOR_ID_DS_WRITE_B128);
    ExpectHasDescriptor(descriptor_set,
                        LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_LOAD_B128);
    ExpectHasDescriptor(descriptor_set,
                        LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_STORE_B128);
    ExpectHasDescriptor(descriptor_set,
                        LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_LOAD_B128_SADDR);
    ExpectHasDescriptor(descriptor_set,
                        LOOM_AMDGPU_DESCRIPTOR_ID_GLOBAL_STORE_B128_SADDR);
  }

  ExpectHasDescriptor(gfx11, LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_B128);
  ExpectHasDescriptor(gfx12, LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_B128);
  ExpectHasDescriptor(gfx1250, LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_B128);
  ExpectMissingDescriptor(gfx950, LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_B128);

  ExpectMissingDescriptor(gfx11, LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORDX4);
  ExpectMissingDescriptor(gfx12, LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORDX4);
  ExpectMissingDescriptor(gfx1250,
                          LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORDX4);
  ExpectHasDescriptor(gfx950, LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORDX4);

  ExpectHasDescriptor(gfx11,
                      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORD_OFF_ZERO);
  ExpectHasDescriptor(gfx11,
                      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORD_OFF_ZERO);
  ExpectHasDescriptor(gfx950,
                      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORD_OFF_ZERO);
  ExpectHasDescriptor(gfx950,
                      LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORD_OFF_ZERO);
  ExpectMissingDescriptor(gfx12,
                          LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORD_OFF_ZERO);
  ExpectMissingDescriptor(
      gfx12, LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORD_OFF_ZERO);
  ExpectMissingDescriptor(gfx1250,
                          LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORD_OFF_ZERO);
  ExpectMissingDescriptor(
      gfx1250, LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORD_OFF_ZERO);
}

TEST(AmdgpuDescriptorIdsTest, AddressImmediateEncodingIdsMatchDescriptorSets) {
  for (const loom_low_descriptor_set_t* descriptor_set :
       {loom_amdgpu_gfx11_core_descriptor_set(),
        loom_amdgpu_gfx12_core_descriptor_set(),
        loom_amdgpu_gfx1250_core_descriptor_set(),
        loom_amdgpu_gfx950_core_descriptor_set()}) {
    ExpectImmediateEncoding(
        descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_LOAD_DWORD,
        LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_BYTE);
    ExpectImmediateEncoding(
        descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_BUFFER_STORE_DWORD,
        LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_BYTE);
    ExpectImmediateEncoding(
        descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ_B32,
        LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_BYTE);
    ExpectImmediateEncoding(
        descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ2_B32,
        LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD);
    ExpectImmediateEncoding(
        descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_DS_READ2ST64_B32,
        LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD_STRIDE64);
  }
}

TEST(AmdgpuDescriptorIdsTest, CommonRegClassIdsMatchDescriptorSetContract) {
  EXPECT_EQ(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 0u);
  EXPECT_EQ(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1u);

  for (const loom_low_descriptor_set_t* descriptor_set :
       {loom_amdgpu_gfx11_core_descriptor_set(),
        loom_amdgpu_gfx12_core_descriptor_set(),
        loom_amdgpu_gfx1250_core_descriptor_set(),
        loom_amdgpu_gfx950_core_descriptor_set()}) {
    ASSERT_GT(descriptor_set->reg_class_count, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
    ASSERT_GT(descriptor_set->reg_class_count, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
    iree_string_view_t sgpr_name = iree_string_view_empty();
    iree_string_view_t vgpr_name = iree_string_view_empty();
    IREE_ASSERT_OK(loom_low_descriptor_set_string(
        descriptor_set,
        descriptor_set->reg_classes[LOOM_AMDGPU_REG_CLASS_ID_SGPR]
            .name_string_offset,
        &sgpr_name));
    IREE_ASSERT_OK(loom_low_descriptor_set_string(
        descriptor_set,
        descriptor_set->reg_classes[LOOM_AMDGPU_REG_CLASS_ID_VGPR]
            .name_string_offset,
        &vgpr_name));
    EXPECT_TRUE(iree_string_view_equal(sgpr_name, IREE_SV("amdgpu.sgpr")));
    EXPECT_TRUE(iree_string_view_equal(vgpr_name, IREE_SV("amdgpu.vgpr")));
  }
}

}  // namespace
