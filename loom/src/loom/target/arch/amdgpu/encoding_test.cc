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

TEST(AmdgpuEncodingTest, NamesVopdFormats) {
  EXPECT_TRUE(iree_string_view_equal(
      loom_amdgpu_encoding_format_name(LOOM_AMDGPU_ENCODING_FORMAT_VOPDXY),
      IREE_SV("vopdxy")));
  EXPECT_TRUE(
      iree_string_view_equal(loom_amdgpu_encoding_format_name(
                                 LOOM_AMDGPU_ENCODING_FORMAT_VOPDXY_LITERAL),
                             IREE_SV("vopdxy_literal")));
}

TEST(AmdgpuEncodingTest, PacksVopdxyDualFmacPair) {
  loom_amdgpu_encoding_packet_t packet = {};
  loom_amdgpu_encoding_vopdxy_fields_t fields = {};
  fields.op_x = 0;
  fields.op_y = 0;
  fields.src0_x = 0x104;
  fields.vsrc1_x = 2;
  fields.vdst_x = 255;
  fields.src0_y = 0x101;
  fields.vsrc1_y = 3;
  fields.vdst_y = 6;
  IREE_ASSERT_OK(loom_amdgpu_encoding_pack_vopdxy(&fields, &packet));
  EXPECT_EQ(packet.word_count, 2u);
  EXPECT_EQ(packet.bit_count, 64u);
  EXPECT_EQ(packet.words[0], UINT32_C(0xc8000504));
  EXPECT_EQ(packet.words[1], UINT32_C(0xff060701));
}

TEST(AmdgpuEncodingTest, RejectsOddVopdxyYDestination) {
  loom_amdgpu_encoding_packet_t packet = {};
  loom_amdgpu_encoding_vopdxy_fields_t fields = {};
  fields.op_x = 0;
  fields.op_y = 0;
  fields.src0_x = 0x104;
  fields.vsrc1_x = 2;
  fields.vdst_x = 255;
  fields.src0_y = 0x101;
  fields.vsrc1_y = 3;
  fields.vdst_y = 7;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_amdgpu_encoding_pack_vopdxy(&fields, &packet));
}

TEST(AmdgpuEncodingTest, RejectsOutOfRangeVopdxyOp) {
  loom_amdgpu_encoding_packet_t packet = {};
  loom_amdgpu_encoding_vopdxy_fields_t fields = {};
  fields.op_x = 16;
  fields.op_y = 0;
  fields.src0_x = 0x104;
  fields.vsrc1_x = 2;
  fields.vdst_x = 255;
  fields.src0_y = 0x101;
  fields.vsrc1_y = 3;
  fields.vdst_y = 6;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_amdgpu_encoding_pack_vopdxy(&fields, &packet));
}

TEST(AmdgpuEncodingTest, InlineF32SourceMapsBitPatternToSourceSelector) {
  uint16_t source = 0;
  EXPECT_TRUE(loom_amdgpu_encoding_inline_f32_source(
      Rdna3EncodingTable(), UINT32_C(0x3f800000), &source));
  EXPECT_EQ(source, 242u);
}

TEST(AmdgpuEncodingTest, InlineF32SourceRejectsUnsupportedBitPattern) {
  uint16_t source = 1;
  EXPECT_FALSE(loom_amdgpu_encoding_inline_f32_source(
      Rdna3EncodingTable(), UINT32_C(0x40400000), &source));
  EXPECT_EQ(source, 0u);
}

}  // namespace
