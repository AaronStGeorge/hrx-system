// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/encoding.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/target_info.h"

namespace {

const loom_amdgpu_encoding_table_t* Rdna3EncodingTable() {
  const loom_amdgpu_encoding_table_t* table =
      loom_amdgpu_encoding_table_for_descriptor_set_ordinal(
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3);
  EXPECT_NE(table, nullptr);
  return table;
}

TEST(AmdgpuEncodingTest, VMovB32UsesInlineSourceForSmallU32) {
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_v_mov_b32_u32(
      Rdna3EncodingTable(), /*vdst=*/1, /*imm32=*/2, &packet));
  EXPECT_EQ(packet.word_count, 1u);
  EXPECT_EQ(packet.bit_count, 32u);
}

TEST(AmdgpuEncodingTest, VMovB32UsesLiteralForLargeU32) {
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_v_mov_b32_u32(
      Rdna3EncodingTable(), /*vdst=*/1, /*imm32=*/65536, &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
}

TEST(AmdgpuEncodingTest, Vop2U32VgprUsesInlineSourceForSmallU32) {
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_vop2_u32_vgpr(
      Rdna3EncodingTable(), /*opcode=*/0, /*vdst=*/1, /*imm32=*/8,
      /*vsrc1=*/2, &packet));
  EXPECT_EQ(packet.word_count, 1u);
  EXPECT_EQ(packet.bit_count, 32u);
}

TEST(AmdgpuEncodingTest, Vop2U32VgprUsesLiteralForLargeU32) {
  loom_amdgpu_encoding_packet_t packet = {};
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_vop2_u32_vgpr(
      Rdna3EncodingTable(), /*opcode=*/0, /*vdst=*/1, /*imm32=*/65536,
      /*vsrc1=*/2, &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
}

}  // namespace
