// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/logarithm.h"

#include "iree/math/elementary_test_util.h"

namespace iree::math::testing {
namespace {

TEST_F(ElementaryTest, Log2F32ApproxExceptionalAndFlushBoundaries) {
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x807FFFFF),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0xBF800000),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0xFF800000),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x7F800000),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x7F800001),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0xFFC01234),
                   UINT32_C(0x7FC00000));
}

TEST_F(ElementaryTest, Log2F32ApproxFrozenFiniteMapping) {
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x00800000),
                   UINT32_C(0xC2FC0000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x3F000000),
                   UINT32_C(0xBF800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x3F800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x40000000),
                   UINT32_C(0x3F800000));
  // These inputs distinguish the selected mapping from correct rounding and
  // direct AMDGPU primitives, respectively.
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x70006067),
                   UINT32_C(0x42C2022B));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x3F7FFFFF),
                   UINT32_C(0xB3B8AA3B));
}

TEST_F(ElementaryTest, LogarithmApproxMappingIgnoresHostDenormalFlush) {
  EnableFlushDenormalsToZero();
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x70006067),
                   UINT32_C(0x42C2022B));
}

// MPFR 4.2.2 directed 256-bit intervals, independently rounded to F32.
// Covers domains, subnormal normalization, reduction boundaries, and
// cancellation.
TEST_F(ElementaryTest, LogarithmsF32Boundaries) {
  const F32Function functions[] = {iree_math_log_f32, iree_math_log2_f32,
                                   iree_math_log10_f32};
  const struct {
    // Exact input payload shared by the functions in this row.
    uint32_t input;
    // Independently correctly rounded results in function order.
    uint32_t expected[3];
  } cases[] = {
      {UINT32_C(0x00000000),
       {UINT32_C(0xFF800000), UINT32_C(0xFF800000), UINT32_C(0xFF800000)}},
      {UINT32_C(0x80000000),
       {UINT32_C(0xFF800000), UINT32_C(0xFF800000), UINT32_C(0xFF800000)}},
      {UINT32_C(0x00000001),
       {UINT32_C(0xC2CE8ED0), UINT32_C(0xC3150000), UINT32_C(0xC23369F4)}},
      {UINT32_C(0x80000001),
       {UINT32_C(0x7FC00000), UINT32_C(0x7FC00000), UINT32_C(0x7FC00000)}},
      {UINT32_C(0x007FFFFF),
       {UINT32_C(0xC2AEAC50), UINT32_C(0xC2FC0000), UINT32_C(0xC217B818)}},
      {UINT32_C(0x00800000),
       {UINT32_C(0xC2AEAC50), UINT32_C(0xC2FC0000), UINT32_C(0xC217B818)}},
      {UINT32_C(0x00800001),
       {UINT32_C(0xC2AEAC50), UINT32_C(0xC2FC0000), UINT32_C(0xC217B818)}},
      {UINT32_C(0x7F7FFFFF),
       {UINT32_C(0x42B17218), UINT32_C(0x43000000), UINT32_C(0x421A209B)}},
      {UINT32_C(0x7F800000),
       {UINT32_C(0x7F800000), UINT32_C(0x7F800000), UINT32_C(0x7F800000)}},
      {UINT32_C(0xFF800000),
       {UINT32_C(0x7FC00000), UINT32_C(0x7FC00000), UINT32_C(0x7FC00000)}},
      {UINT32_C(0x7F800001),
       {UINT32_C(0x7FC00000), UINT32_C(0x7FC00000), UINT32_C(0x7FC00000)}},
      {UINT32_C(0x7FC00011),
       {UINT32_C(0x7FC00000), UINT32_C(0x7FC00000), UINT32_C(0x7FC00000)}},
      {UINT32_C(0xFF800001),
       {UINT32_C(0x7FC00000), UINT32_C(0x7FC00000), UINT32_C(0x7FC00000)}},
      {UINT32_C(0xBF800000),
       {UINT32_C(0x7FC00000), UINT32_C(0x7FC00000), UINT32_C(0x7FC00000)}},
      {UINT32_C(0x3F000000),
       {UINT32_C(0xBF317218), UINT32_C(0xBF800000), UINT32_C(0xBE9A209B)}},
      {UINT32_C(0x40000000),
       {UINT32_C(0x3F317218), UINT32_C(0x3F800000), UINT32_C(0x3E9A209B)}},
      {UINT32_C(0x41200000),
       {UINT32_C(0x40135D8E), UINT32_C(0x40549A78), UINT32_C(0x3F800000)}},
      {UINT32_C(0x42C80000),
       {UINT32_C(0x40935D8E), UINT32_C(0x40D49A78), UINT32_C(0x40000000)}},
      {UINT32_C(0x2EDBE6FF),
       {UINT32_C(0xC1B834F1), UINT32_C(0xC204E08B), UINT32_C(0xC1200000)}},
      {UINT32_C(0x3F7FFFFF),
       {UINT32_C(0xB3800000), UINT32_C(0xB3B8AA3C), UINT32_C(0xB2DE5BD9)}},
      {UINT32_C(0x3F800000),
       {UINT32_C(0x00000000), UINT32_C(0x00000000), UINT32_C(0x00000000)}},
      {UINT32_C(0x3F800001),
       {UINT32_C(0x33FFFFFF), UINT32_C(0x3438AA3A), UINT32_C(0x335E5BD8)}},
      {UINT32_C(0x3F32FFFF),
       {UINT32_C(0xBEB7307A), UINT32_C(0xBF0424A8), UINT32_C(0xBE1F1DC3)}},
      {UINT32_C(0x3F330000),
       {UINT32_C(0xBEB73077), UINT32_C(0xBF0424A6), UINT32_C(0xBE1F1DC1)}},
      {UINT32_C(0x3F330001),
       {UINT32_C(0xBEB73074), UINT32_C(0xBF0424A4), UINT32_C(0xBE1F1DBE)}},
      {UINT32_C(0x3F3AFFFF),
       {UINT32_C(0xBEA0CDA4), UINT32_C(0xBEE7FD77), UINT32_C(0xBE0BAC05)}},
      {UINT32_C(0x3F3B0000),
       {UINT32_C(0xBEA0CDA1), UINT32_C(0xBEE7FD73), UINT32_C(0xBE0BAC03)}},
      {UINT32_C(0x3F3B0001),
       {UINT32_C(0xBEA0CD9E), UINT32_C(0xBEE7FD6F), UINT32_C(0xBE0BAC01)}},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << "input_bits=" << std::hex << c.input);
    for (unsigned index = 0; index < 3; ++index) {
      SCOPED_TRACE(index);
      const uint32_t actual = F32ToBits(functions[index](F32FromBits(c.input)));
      const uint32_t expected = c.expected[index];
      const uint32_t magnitude = expected & UINT32_C(0x7FFFFFFF);
      if (magnitude == 0 || magnitude >= UINT32_C(0x7F800000)) {
        EXPECT_EQ(actual, expected);
      } else {
        EXPECT_LE(actual > expected ? actual - expected : expected - actual,
                  UINT32_C(1));
      }
    }
  }
}

// MPFR 4.2.2 directed 256-bit intervals, independently rounded to F32.
// Covers domains, subnormal normalization, reduction boundaries, and
// cancellation.
TEST_F(ElementaryTest, Log1pF32Boundaries) {
  const struct {
    // Exact input payload.
    uint32_t input;
    // Independently correctly rounded result.
    uint32_t expected;
  } cases[] = {
      {UINT32_C(0x00000000), UINT32_C(0x00000000)},
      {UINT32_C(0x80000000), UINT32_C(0x80000000)},
      {UINT32_C(0x00000001), UINT32_C(0x00000001)},
      {UINT32_C(0x80000001), UINT32_C(0x80000001)},
      {UINT32_C(0x007FFFFF), UINT32_C(0x007FFFFF)},
      {UINT32_C(0x00800000), UINT32_C(0x00800000)},
      {UINT32_C(0x00800001), UINT32_C(0x00800001)},
      {UINT32_C(0x7F7FFFFF), UINT32_C(0x42B17218)},
      {UINT32_C(0x7F800000), UINT32_C(0x7F800000)},
      {UINT32_C(0xFF800000), UINT32_C(0x7FC00000)},
      {UINT32_C(0x7F800001), UINT32_C(0x7FC00000)},
      {UINT32_C(0x7FC00011), UINT32_C(0x7FC00000)},
      {UINT32_C(0xFF800001), UINT32_C(0x7FC00000)},
      {UINT32_C(0xBF800000), UINT32_C(0xFF800000)},
      {UINT32_C(0xBF000000), UINT32_C(0xBF317218)},
      {UINT32_C(0x3F000000), UINT32_C(0x3ECF991F)},
      {UINT32_C(0x40400000), UINT32_C(0x3FB17218)},
      {UINT32_C(0x41F80000), UINT32_C(0x405DCE9E)},
      {UINT32_C(0xBF7FBE77), UINT32_C(0xC0DD0C70)},
      {UINT32_C(0x3E000000), UINT32_C(0x3DF1383B)},
      {UINT32_C(0xBF7FFFFF), UINT32_C(0xC1851592)},
      {UINT32_C(0xBF800001), UINT32_C(0x7FC00000)},
      {UINT32_C(0x337FFFFF), UINT32_C(0x337FFFFF)},
      {UINT32_C(0x33800000), UINT32_C(0x33800000)},
      {UINT32_C(0x33800001), UINT32_C(0x33800001)},
      {UINT32_C(0xB37FFFFF), UINT32_C(0xB37FFFFF)},
      {UINT32_C(0xB3800000), UINT32_C(0xB3800000)},
      {UINT32_C(0xB3800001), UINT32_C(0xB3800001)},
      {UINT32_C(0x3ED413CF), UINT32_C(0x3EB1721A)},
      {UINT32_C(0x3ED413D0), UINT32_C(0x3EB1721A)},
      {UINT32_C(0x3ED413D1), UINT32_C(0x3EB1721B)},
      {UINT32_C(0xBE95F618), UINT32_C(0xBEB17216)},
      {UINT32_C(0xBE95F619), UINT32_C(0xBEB17217)},
      {UINT32_C(0xBE95F61A), UINT32_C(0xBEB17219)},
      {UINT32_C(0x4BB504F2), UINT32_C(0x4187DB5A)},
      {UINT32_C(0x4BB504F3), UINT32_C(0x4187DB5A)},
      {UINT32_C(0x4BB504F4), UINT32_C(0x4187DB5A)},
      {UINT32_C(0x3F7FFFFF), UINT32_C(0x3F317217)},
      {UINT32_C(0x3F800000), UINT32_C(0x3F317218)},
      {UINT32_C(0x3F800001), UINT32_C(0x3F317219)},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << "input_bits=" << std::hex << c.input);
    const uint32_t actual =
        F32ToBits(iree_math_log1p_f32(F32FromBits(c.input)));
    const uint32_t expected = c.expected;
    const uint32_t magnitude = expected & UINT32_C(0x7FFFFFFF);
    if (magnitude == 0 || magnitude >= UINT32_C(0x7F800000)) {
      EXPECT_EQ(actual, expected);
    } else {
      EXPECT_LE(actual > expected ? actual - expected : expected - actual,
                UINT32_C(1));
    }
  }
}

// MPFR 4.2.2 directed 256-bit intervals, independently rounded to F64.
// Covers domains, subnormal normalization, reduction boundaries, and
// cancellation.
TEST_F(ElementaryTest, LogarithmsF64Boundaries) {
  const F64Function functions[] = {iree_math_log_f64, iree_math_log2_f64,
                                   iree_math_log10_f64};
  const struct {
    // Exact input payload shared by the functions in this row.
    uint64_t input;
    // Independently correctly rounded results in function order.
    uint64_t expected[3];
  } cases[] = {
      {UINT64_C(0x0000000000000000),
       {UINT64_C(0xFFF0000000000000), UINT64_C(0xFFF0000000000000),
        UINT64_C(0xFFF0000000000000)}},
      {UINT64_C(0x8000000000000000),
       {UINT64_C(0xFFF0000000000000), UINT64_C(0xFFF0000000000000),
        UINT64_C(0xFFF0000000000000)}},
      {UINT64_C(0x0000000000000001),
       {UINT64_C(0xC0874385446D71C3), UINT64_C(0xC090C80000000000),
        UINT64_C(0xC07434E6420F4374)}},
      {UINT64_C(0x8000000000000001),
       {UINT64_C(0x7FF8000000000000), UINT64_C(0x7FF8000000000000),
        UINT64_C(0x7FF8000000000000)}},
      {UINT64_C(0x000FFFFFFFFFFFFF),
       {UINT64_C(0xC086232BDD7ABCD2), UINT64_C(0xC08FF00000000000),
        UINT64_C(0xC0733A7146F72A42)}},
      {UINT64_C(0x0010000000000000),
       {UINT64_C(0xC086232BDD7ABCD2), UINT64_C(0xC08FF00000000000),
        UINT64_C(0xC0733A7146F72A42)}},
      {UINT64_C(0x0010000000000001),
       {UINT64_C(0xC086232BDD7ABCD2), UINT64_C(0xC08FF00000000000),
        UINT64_C(0xC0733A7146F72A42)}},
      {UINT64_C(0x7FEFFFFFFFFFFFFF),
       {UINT64_C(0x40862E42FEFA39EF), UINT64_C(0x4090000000000000),
        UINT64_C(0x40734413509F79FF)}},
      {UINT64_C(0x7FF0000000000000),
       {UINT64_C(0x7FF0000000000000), UINT64_C(0x7FF0000000000000),
        UINT64_C(0x7FF0000000000000)}},
      {UINT64_C(0xFFF0000000000000),
       {UINT64_C(0x7FF8000000000000), UINT64_C(0x7FF8000000000000),
        UINT64_C(0x7FF8000000000000)}},
      {UINT64_C(0x7FF0000000000001),
       {UINT64_C(0x7FF8000000000000), UINT64_C(0x7FF8000000000000),
        UINT64_C(0x7FF8000000000000)}},
      {UINT64_C(0x7FF8000000000011),
       {UINT64_C(0x7FF8000000000000), UINT64_C(0x7FF8000000000000),
        UINT64_C(0x7FF8000000000000)}},
      {UINT64_C(0xFFF0000000000001),
       {UINT64_C(0x7FF8000000000000), UINT64_C(0x7FF8000000000000),
        UINT64_C(0x7FF8000000000000)}},
      {UINT64_C(0xBFF0000000000000),
       {UINT64_C(0x7FF8000000000000), UINT64_C(0x7FF8000000000000),
        UINT64_C(0x7FF8000000000000)}},
      {UINT64_C(0x3FE0000000000000),
       {UINT64_C(0xBFE62E42FEFA39EF), UINT64_C(0xBFF0000000000000),
        UINT64_C(0xBFD34413509F79FF)}},
      {UINT64_C(0x4000000000000000),
       {UINT64_C(0x3FE62E42FEFA39EF), UINT64_C(0x3FF0000000000000),
        UINT64_C(0x3FD34413509F79FF)}},
      {UINT64_C(0x4024000000000000),
       {UINT64_C(0x40026BB1BBB55516), UINT64_C(0x400A934F0979A371),
        UINT64_C(0x3FF0000000000000)}},
      {UINT64_C(0x4059000000000000),
       {UINT64_C(0x40126BB1BBB55516), UINT64_C(0x401A934F0979A371),
        UINT64_C(0x4000000000000000)}},
      {UINT64_C(0x3DDB7CDFD9D7BDBB),
       {UINT64_C(0xC037069E2AA2AA5B), UINT64_C(0xC0409C1165EC0627),
        UINT64_C(0xC024000000000000)}},
      {UINT64_C(0x3FEFFFFFFFFFFFFF),
       {UINT64_C(0xBCA0000000000000), UINT64_C(0xBCA71547652B82FE),
        UINT64_C(0xBC8BCB7B1526E50F)}},
      {UINT64_C(0x3FF0000000000000),
       {UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000),
        UINT64_C(0x0000000000000000)}},
      {UINT64_C(0x3FF0000000000001),
       {UINT64_C(0x3CAFFFFFFFFFFFFF), UINT64_C(0x3CB71547652B82FD),
        UINT64_C(0x3C9BCB7B1526E50D)}},
      {UINT64_C(0x3FE5FFFFFFFFFFFF),
       {UINT64_C(0xBFD7FAFA3BD8151F), UINT64_C(0xBFE14C560FE68AFB),
        UINT64_C(0xBFC4D43F8275A486)}},
      {UINT64_C(0x3FE6000000000000),
       {UINT64_C(0xBFD7FAFA3BD8151C), UINT64_C(0xBFE14C560FE68AF9),
        UINT64_C(0xBFC4D43F8275A483)}},
      {UINT64_C(0x3FE6000000000001),
       {UINT64_C(0xBFD7FAFA3BD81519), UINT64_C(0xBFE14C560FE68AF6),
        UINT64_C(0xBFC4D43F8275A481)}},
      {UINT64_C(0x3FE61FFFFFFFFFFF),
       {UINT64_C(0xBFD79E26687CFB40), UINT64_C(0xBFE1096015DEE4DC),
        UINT64_C(0xBFC4839E83506C89)}},
      {UINT64_C(0x3FE6200000000000),
       {UINT64_C(0xBFD79E26687CFB3E), UINT64_C(0xBFE1096015DEE4DA),
        UINT64_C(0xBFC4839E83506C87)}},
      {UINT64_C(0x3FE6200000000001),
       {UINT64_C(0xBFD79E26687CFB3B), UINT64_C(0xBFE1096015DEE4D8),
        UINT64_C(0xBFC4839E83506C84)}},
      {UINT64_C(0x3FEDFFFFFFFFFFFF),
       {UINT64_C(0xBFB08598B59E3A0F), UINT64_C(0xBFB7D60496CFBB59),
        UINT64_C(0xBF9CB38FCCD8BFEA)}},
      {UINT64_C(0x3FEE000000000000),
       {UINT64_C(0xBFB08598B59E3A07), UINT64_C(0xBFB7D60496CFBB4C),
        UINT64_C(0xBF9CB38FCCD8BFDB)}},
      {UINT64_C(0x3FEE000000000001),
       {UINT64_C(0xBFB08598B59E39FE), UINT64_C(0xBFB7D60496CFBB40),
        UINT64_C(0xBF9CB38FCCD8BFCD)}},
      {UINT64_C(0x3FF108FFFFFFFFFF),
       {UINT64_C(0x3FB00C7C14184EB6), UINT64_C(0x3FB7274A5C9E5E69),
        UINT64_C(0x3F9BE12B3D5939F6)}},
      {UINT64_C(0x3FF1090000000000),
       {UINT64_C(0x3FB00C7C14184EC5), UINT64_C(0x3FB7274A5C9E5E7F),
        UINT64_C(0x3F9BE12B3D593A10)}},
      {UINT64_C(0x3FF1090000000001),
       {UINT64_C(0x3FB00C7C14184ED4), UINT64_C(0x3FB7274A5C9E5E94),
        UINT64_C(0x3F9BE12B3D593A2A)}},
      {UINT64_C(0x3FEEA4AEFFFFFFFF),
       {UINT64_C(0xBFA62E4D9C79EBFC), UINT64_C(0xBFB00007A842D2C9),
        UINT64_C(0xBF93441C89032EB2)}},
      {UINT64_C(0x3FEEA4AF00000000),
       {UINT64_C(0xBFA62E4D9C79EBEB), UINT64_C(0xBFB00007A842D2BD),
        UINT64_C(0xBF93441C89032EA3)}},
      {UINT64_C(0x3FEEA4AF00000001),
       {UINT64_C(0xBFA62E4D9C79EBDA), UINT64_C(0xBFB00007A842D2B0),
        UINT64_C(0xBF93441C89032E95)}},
      {UINT64_C(0x3FF0B558FFFFFFFF),
       {UINT64_C(0x3FA62E549850DF9B), UINT64_C(0x3FB0000CB1EA2BCC),
        UINT64_C(0x3F93442299E9047A)}},
      {UINT64_C(0x3FF0B55900000000),
       {UINT64_C(0x3FA62E549850DFB9), UINT64_C(0x3FB0000CB1EA2BE2),
        UINT64_C(0x3F93442299E90494)}},
      {UINT64_C(0x3FF0B55900000001),
       {UINT64_C(0x3FA62E549850DFD8), UINT64_C(0x3FB0000CB1EA2BF8),
        UINT64_C(0x3F93442299E904AF)}},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << "input_bits=" << std::hex << c.input);
    for (unsigned index = 0; index < 3; ++index) {
      SCOPED_TRACE(index);
      const uint64_t actual = F64ToBits(functions[index](F64FromBits(c.input)));
      const uint64_t expected = c.expected[index];
      const uint64_t magnitude = expected & UINT64_C(0x7FFFFFFFFFFFFFFF);
      if (magnitude == 0 || magnitude >= UINT64_C(0x7FF0000000000000)) {
        EXPECT_EQ(actual, expected);
      } else {
        EXPECT_LE(actual > expected ? actual - expected : expected - actual,
                  UINT64_C(1));
      }
    }
  }
}

// MPFR 4.2.2 directed 256-bit intervals, independently rounded to F64.
// Covers domains, subnormal normalization, reduction boundaries, and
// cancellation.
TEST_F(ElementaryTest, Log1pF64Boundaries) {
  const struct {
    // Exact input payload.
    uint64_t input;
    // Independently correctly rounded result.
    uint64_t expected;
  } cases[] = {
      {UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0x8000000000000000), UINT64_C(0x8000000000000000)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x0000000000000001)},
      {UINT64_C(0x8000000000000001), UINT64_C(0x8000000000000001)},
      {UINT64_C(0x000FFFFFFFFFFFFF), UINT64_C(0x000FFFFFFFFFFFFF)},
      {UINT64_C(0x0010000000000000), UINT64_C(0x0010000000000000)},
      {UINT64_C(0x0010000000000001), UINT64_C(0x0010000000000001)},
      {UINT64_C(0x7FEFFFFFFFFFFFFF), UINT64_C(0x40862E42FEFA39EF)},
      {UINT64_C(0x7FF0000000000000), UINT64_C(0x7FF0000000000000)},
      {UINT64_C(0xFFF0000000000000), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x7FF0000000000001), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x7FF8000000000011), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0xFFF0000000000001), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0xBFF0000000000000), UINT64_C(0xFFF0000000000000)},
      {UINT64_C(0xBFE0000000000000), UINT64_C(0xBFE62E42FEFA39EF)},
      {UINT64_C(0x3FE0000000000000), UINT64_C(0x3FD9F323ECBF984C)},
      {UINT64_C(0x4008000000000000), UINT64_C(0x3FF62E42FEFA39EF)},
      {UINT64_C(0x403F000000000000), UINT64_C(0x400BB9D3BEB8C86B)},
      {UINT64_C(0xBFEFF7CED916872B), UINT64_C(0xC01BA18A998FFF9F)},
      {UINT64_C(0x3FC0000000000000), UINT64_C(0x3FBE27076E2AF2E6)},
      {UINT64_C(0xBFEFFFFFFFFFFFFF), UINT64_C(0xC0425E4F7B2737FA)},
      {UINT64_C(0xBFF0000000000001), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x3C9FFFFFFFFFFFFF), UINT64_C(0x3C9FFFFFFFFFFFFF)},
      {UINT64_C(0x3CA0000000000000), UINT64_C(0x3CA0000000000000)},
      {UINT64_C(0x3CA0000000000001), UINT64_C(0x3CA0000000000001)},
      {UINT64_C(0xBC9FFFFFFFFFFFFF), UINT64_C(0xBC9FFFFFFFFFFFFF)},
      {UINT64_C(0xBCA0000000000000), UINT64_C(0xBCA0000000000000)},
      {UINT64_C(0xBCA0000000000001), UINT64_C(0xBCA0000000000001)},
      {UINT64_C(0x3FDA8279FFFFFFFF), UINT64_C(0x3FD62E43471C5D9D)},
      {UINT64_C(0x3FDA827A00000000), UINT64_C(0x3FD62E43471C5D9E)},
      {UINT64_C(0x3FDA827A00000001), UINT64_C(0x3FD62E43471C5D9F)},
      {UINT64_C(0xBFD2BEC4FFFFFFFF), UINT64_C(0xBFD62E458AEBEFDF)},
      {UINT64_C(0xBFD2BEC500000000), UINT64_C(0xBFD62E458AEBEFE0)},
      {UINT64_C(0xBFD2BEC500000001), UINT64_C(0xBFD62E458AEBEFE2)},
      {UINT64_C(0x4346A09DFFFFFFFF), UINT64_C(0x40428AABFEE15CE4)},
      {UINT64_C(0x4346A09E00000000), UINT64_C(0x40428AABFEE15CE4)},
      {UINT64_C(0x4346A09E00000001), UINT64_C(0x40428AABFEE15CE4)},
      {UINT64_C(0x3FEFFFFFFFFFFFFF), UINT64_C(0x3FE62E42FEFA39EF)},
      {UINT64_C(0x3FF0000000000000), UINT64_C(0x3FE62E42FEFA39EF)},
      {UINT64_C(0x3FF0000000000001), UINT64_C(0x3FE62E42FEFA39F0)},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << "input_bits=" << std::hex << c.input);
    const uint64_t actual =
        F64ToBits(iree_math_log1p_f64(F64FromBits(c.input)));
    const uint64_t expected = c.expected;
    const uint64_t magnitude = expected & UINT64_C(0x7FFFFFFFFFFFFFFF);
    if (magnitude == 0 || magnitude >= UINT64_C(0x7FF0000000000000)) {
      EXPECT_EQ(actual, expected);
    } else {
      EXPECT_LE(actual > expected ? actual - expected : expected - actual,
                UINT64_C(1));
    }
  }
}

}  // namespace
}  // namespace iree::math::testing
