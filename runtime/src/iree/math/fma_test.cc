// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/fma.h"

#include <cmath>

#include "iree/base/internal/math.h"
#include "iree/math/elementary_test_util.h"
#include "iree/math/fma_software.h"

namespace iree::math::testing {
namespace {

// MPFR 4.2.2 fused results, with directed intervals refined from 256 to 8192
// bits until both endpoints round to the same payload. Exact-zero signs use
// nearest-even. These boundaries cover cancellation, overflow, normal and
// subnormal ties, long exponent gaps, and exceptional operands.
TEST_F(ElementaryTest, FmaF32Boundaries) {
  const struct {
    // First multiplicand payload.
    uint32_t a;
    // Second multiplicand payload.
    uint32_t b;
    // Addend payload.
    uint32_t c;
    // Independently rounded result, with canonical NaNs.
    uint32_t expected;
  } cases[] = {
      // Signed-zero products and sums.
      {UINT32_C(0x00000000), UINT32_C(0x3F800000), UINT32_C(0x00000000),
       UINT32_C(0x00000000)},
      {UINT32_C(0x00000000), UINT32_C(0x3F800000), UINT32_C(0x80000000),
       UINT32_C(0x00000000)},
      {UINT32_C(0x00000000), UINT32_C(0xBF800000), UINT32_C(0x00000000),
       UINT32_C(0x00000000)},
      {UINT32_C(0x00000000), UINT32_C(0xBF800000), UINT32_C(0x80000000),
       UINT32_C(0x80000000)},
      {UINT32_C(0x80000000), UINT32_C(0x3F800000), UINT32_C(0x00000000),
       UINT32_C(0x00000000)},
      {UINT32_C(0x80000000), UINT32_C(0x3F800000), UINT32_C(0x80000000),
       UINT32_C(0x80000000)},
      {UINT32_C(0x80000000), UINT32_C(0xBF800000), UINT32_C(0x00000000),
       UINT32_C(0x00000000)},
      {UINT32_C(0x80000000), UINT32_C(0xBF800000), UINT32_C(0x80000000),
       UINT32_C(0x00000000)},
      // Overflowing products can cancel to a finite final result.
      {UINT32_C(0x7F7FFFFF), UINT32_C(0x40000000), UINT32_C(0xFF7FFFFF),
       UINT32_C(0x7F7FFFFF)},
      {UINT32_C(0xFF7FFFFF), UINT32_C(0x40000000), UINT32_C(0x7F7FFFFF),
       UINT32_C(0xFF7FFFFF)},
      {UINT32_C(0x7F7FFFFF), UINT32_C(0x3F800000), UINT32_C(0x7F7FFFFF),
       UINT32_C(0x7F800000)},
      // Gradual underflow, including ties on either side of zero.
      {UINT32_C(0x00800000), UINT32_C(0x3F000000), UINT32_C(0x00000001),
       UINT32_C(0x00400001)},
      {UINT32_C(0x00800000), UINT32_C(0x3F7FFFFF), UINT32_C(0x80800000),
       UINT32_C(0x80000000)},
      {UINT32_C(0x00000001), UINT32_C(0x3F000000), UINT32_C(0x00000000),
       UINT32_C(0x00000000)},
      {UINT32_C(0x00000001), UINT32_C(0x3F000000), UINT32_C(0x00000001),
       UINT32_C(0x00000002)},
      {UINT32_C(0x00000001), UINT32_C(0x3F000000), UINT32_C(0x80000001),
       UINT32_C(0x80000000)},
      {UINT32_C(0x00000001), UINT32_C(0x3FC00000), UINT32_C(0x00000001),
       UINT32_C(0x00000002)},
      {UINT32_C(0x00000001), UINT32_C(0x00000001), UINT32_C(0x00000000),
       UINT32_C(0x00000000)},
      {UINT32_C(0x00000001), UINT32_C(0x00000001), UINT32_C(0x00000001),
       UINT32_C(0x00000001)},
      {UINT32_C(0x00000001), UINT32_C(0x00000001), UINT32_C(0x80000001),
       UINT32_C(0x80000001)},
      // Cancellation exposes product bits lost by separate multiplication.
      {UINT32_C(0x3F800001), UINT32_C(0x3F800001), UINT32_C(0xBF800002),
       UINT32_C(0x28800000)},
      {UINT32_C(0x3F800001), UINT32_C(0x3F7FFFFF), UINT32_C(0xBF800000),
       UINT32_C(0x337FFFFE)},
      {UINT32_C(0x7F7FFFFF), UINT32_C(0x7F7FFFFF), UINT32_C(0xFF7FFFFF),
       UINT32_C(0x7F800000)},
      // Nonfinite operands follow fused arithmetic semantics.
      {UINT32_C(0x7F800000), UINT32_C(0x00000000), UINT32_C(0x3F800000),
       UINT32_C(0x7FC00000)},
      {UINT32_C(0xFF800000), UINT32_C(0x3F800000), UINT32_C(0x7F800000),
       UINT32_C(0x7FC00000)},
      {UINT32_C(0x3F800000), UINT32_C(0x40000000), UINT32_C(0x7F800000),
       UINT32_C(0x7F800000)},
      {UINT32_C(0x7FC00000), UINT32_C(0x00000000), UINT32_C(0x3F800000),
       UINT32_C(0x7FC00000)},
      {UINT32_C(0x00000000), UINT32_C(0x3F800000), UINT32_C(0x7FC00000),
       UINT32_C(0x7FC00000)},
      // Long exponent gaps retain the side of a rounding midpoint.
      {UINT32_C(0x3FC00000), UINT32_C(0x3F800001), UINT32_C(0x33800000),
       UINT32_C(0x3FC00002)},
      {UINT32_C(0x33800000), UINT32_C(0x3FC00000), UINT32_C(0x3F800000),
       UINT32_C(0x3F800001)},
      {UINT32_C(0x3FC00000), UINT32_C(0x3F800001), UINT32_C(0xB3800000),
       UINT32_C(0x3FC00001)},
      {UINT32_C(0xB3800000), UINT32_C(0x3FC00000), UINT32_C(0x3F800000),
       UINT32_C(0x3F7FFFFE)},
      {UINT32_C(0x3FC00000), UINT32_C(0x3F800001), UINT32_C(0x00800000),
       UINT32_C(0x3FC00002)},
      {UINT32_C(0x00800000), UINT32_C(0x3FC00000), UINT32_C(0x3F800000),
       UINT32_C(0x3F800000)},
      {UINT32_C(0x3FC00000), UINT32_C(0x3F800001), UINT32_C(0x80800000),
       UINT32_C(0x3FC00001)},
      {UINT32_C(0x80800000), UINT32_C(0x3FC00000), UINT32_C(0x3F800000),
       UINT32_C(0x3F800000)},
      {UINT32_C(0x3FC00000), UINT32_C(0x3F800001), UINT32_C(0x00000001),
       UINT32_C(0x3FC00002)},
      {UINT32_C(0x00000001), UINT32_C(0x3FC00000), UINT32_C(0x3F800000),
       UINT32_C(0x3F800000)},
      {UINT32_C(0x3FC00000), UINT32_C(0x3F800001), UINT32_C(0x80000001),
       UINT32_C(0x3FC00001)},
      {UINT32_C(0x80000001), UINT32_C(0x3FC00000), UINT32_C(0x3F800000),
       UINT32_C(0x3F800000)},
      // F64 rounding hides a residue above an F32 subnormal midpoint.
      {UINT32_C(0x1A008040), UINT32_C(0x19FF0080), UINT32_C(0x00080000),
       UINT32_C(0x00080001)},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message()
                 << std::hex << test.a << " * " << test.b << " + " << test.c);
    const auto a = F32FromBits(test.a);
    const auto b = F32FromBits(test.b);
    const auto c = F32FromBits(test.c);
    EXPECT_EQ(F32ToBits(iree_math_fma_f32(a, b, c)), test.expected);
    EXPECT_EQ(F32ToBits(iree_math_fma_f32(b, a, c)), test.expected);
    // Exercise the fallback even on hosts built with native FMA enabled.
    const auto software = iree_math_fma_f32_software(a, b, c);
    if (std::isnan(F32FromBits(test.expected))) {
      EXPECT_TRUE(std::isnan(software));
    } else {
      EXPECT_EQ(F32ToBits(software), test.expected);
    }
  }
}

TEST_F(ElementaryTest, FmaF64Boundaries) {
  const struct {
    // First multiplicand payload.
    uint64_t a;
    // Second multiplicand payload.
    uint64_t b;
    // Addend payload.
    uint64_t c;
    // Independently rounded result, with canonical NaNs.
    uint64_t expected;
  } cases[] = {
      // Signed-zero products and sums.
      {UINT64_C(0x0000000000000000), UINT64_C(0x3FF0000000000000),
       UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0x0000000000000000), UINT64_C(0x3FF0000000000000),
       UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0x0000000000000000), UINT64_C(0xBFF0000000000000),
       UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0x0000000000000000), UINT64_C(0xBFF0000000000000),
       UINT64_C(0x8000000000000000), UINT64_C(0x8000000000000000)},
      {UINT64_C(0x8000000000000000), UINT64_C(0x3FF0000000000000),
       UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0x8000000000000000), UINT64_C(0x3FF0000000000000),
       UINT64_C(0x8000000000000000), UINT64_C(0x8000000000000000)},
      {UINT64_C(0x8000000000000000), UINT64_C(0xBFF0000000000000),
       UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0x8000000000000000), UINT64_C(0xBFF0000000000000),
       UINT64_C(0x8000000000000000), UINT64_C(0x0000000000000000)},
      // Overflowing products can cancel to a finite final result.
      {UINT64_C(0x7FEFFFFFFFFFFFFF), UINT64_C(0x4000000000000000),
       UINT64_C(0xFFEFFFFFFFFFFFFF), UINT64_C(0x7FEFFFFFFFFFFFFF)},
      {UINT64_C(0xFFEFFFFFFFFFFFFF), UINT64_C(0x4000000000000000),
       UINT64_C(0x7FEFFFFFFFFFFFFF), UINT64_C(0xFFEFFFFFFFFFFFFF)},
      {UINT64_C(0x7FEFFFFFFFFFFFFF), UINT64_C(0x3FF0000000000000),
       UINT64_C(0x7FEFFFFFFFFFFFFF), UINT64_C(0x7FF0000000000000)},
      // Gradual underflow, including ties on either side of zero.
      {UINT64_C(0x0010000000000000), UINT64_C(0x3FE0000000000000),
       UINT64_C(0x0000000000000001), UINT64_C(0x0008000000000001)},
      {UINT64_C(0x0010000000000000), UINT64_C(0x3FEFFFFFFFFFFFFF),
       UINT64_C(0x8010000000000000), UINT64_C(0x8000000000000000)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x3FE0000000000000),
       UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x3FE0000000000000),
       UINT64_C(0x0000000000000001), UINT64_C(0x0000000000000002)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x3FE0000000000000),
       UINT64_C(0x8000000000000001), UINT64_C(0x8000000000000000)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x3FF8000000000000),
       UINT64_C(0x0000000000000001), UINT64_C(0x0000000000000002)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x0000000000000001),
       UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x0000000000000001),
       UINT64_C(0x0000000000000001), UINT64_C(0x0000000000000001)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x0000000000000001),
       UINT64_C(0x8000000000000001), UINT64_C(0x8000000000000001)},
      // Cancellation exposes product bits lost by separate multiplication.
      {UINT64_C(0x3FF0000000000001), UINT64_C(0x3FF0000000000001),
       UINT64_C(0xBFF0000000000002), UINT64_C(0x3970000000000000)},
      {UINT64_C(0x3FF0000000000001), UINT64_C(0x3FEFFFFFFFFFFFFF),
       UINT64_C(0xBFF0000000000000), UINT64_C(0x3C9FFFFFFFFFFFFE)},
      {UINT64_C(0x7FEFFFFFFFFFFFFF), UINT64_C(0x7FEFFFFFFFFFFFFF),
       UINT64_C(0xFFEFFFFFFFFFFFFF), UINT64_C(0x7FF0000000000000)},
      // Nonfinite operands follow fused arithmetic semantics.
      {UINT64_C(0x7FF0000000000000), UINT64_C(0x0000000000000000),
       UINT64_C(0x3FF0000000000000), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0xFFF0000000000000), UINT64_C(0x3FF0000000000000),
       UINT64_C(0x7FF0000000000000), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x3FF0000000000000), UINT64_C(0x4000000000000000),
       UINT64_C(0x7FF0000000000000), UINT64_C(0x7FF0000000000000)},
      {UINT64_C(0x7FF8000000000000), UINT64_C(0x0000000000000000),
       UINT64_C(0x3FF0000000000000), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x0000000000000000), UINT64_C(0x3FF0000000000000),
       UINT64_C(0x7FF8000000000000), UINT64_C(0x7FF8000000000000)},
      // Long exponent gaps retain the side of a rounding midpoint.
      {UINT64_C(0x3FF8000000000000), UINT64_C(0x3FF0000000000001),
       UINT64_C(0x3CA0000000000000), UINT64_C(0x3FF8000000000002)},
      {UINT64_C(0x3CA0000000000000), UINT64_C(0x3FF8000000000000),
       UINT64_C(0x3FF0000000000000), UINT64_C(0x3FF0000000000001)},
      {UINT64_C(0x3FF8000000000000), UINT64_C(0x3FF0000000000001),
       UINT64_C(0xBCA0000000000000), UINT64_C(0x3FF8000000000001)},
      {UINT64_C(0xBCA0000000000000), UINT64_C(0x3FF8000000000000),
       UINT64_C(0x3FF0000000000000), UINT64_C(0x3FEFFFFFFFFFFFFE)},
      {UINT64_C(0x3FF8000000000000), UINT64_C(0x3FF0000000000001),
       UINT64_C(0x0010000000000000), UINT64_C(0x3FF8000000000002)},
      {UINT64_C(0x0010000000000000), UINT64_C(0x3FF8000000000000),
       UINT64_C(0x3FF0000000000000), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x3FF8000000000000), UINT64_C(0x3FF0000000000001),
       UINT64_C(0x8010000000000000), UINT64_C(0x3FF8000000000001)},
      {UINT64_C(0x8010000000000000), UINT64_C(0x3FF8000000000000),
       UINT64_C(0x3FF0000000000000), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x3FF8000000000000), UINT64_C(0x3FF0000000000001),
       UINT64_C(0x0000000000000001), UINT64_C(0x3FF8000000000002)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x3FF8000000000000),
       UINT64_C(0x3FF0000000000000), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x3FF8000000000000), UINT64_C(0x3FF0000000000001),
       UINT64_C(0x8000000000000001), UINT64_C(0x3FF8000000000001)},
      {UINT64_C(0x8000000000000001), UINT64_C(0x3FF8000000000000),
       UINT64_C(0x3FF0000000000000), UINT64_C(0x3FF0000000000000)},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message()
                 << std::hex << test.a << " * " << test.b << " + " << test.c);
    const auto a = F64FromBits(test.a);
    const auto b = F64FromBits(test.b);
    const auto c = F64FromBits(test.c);
    EXPECT_EQ(F64ToBits(iree_math_fma_f64(a, b, c)), test.expected);
    EXPECT_EQ(F64ToBits(iree_math_fma_f64(b, a, c)), test.expected);
    // Exercise the fallback even on hosts built with native FMA enabled.
    const auto software = iree_math_fma_f64_software(a, b, c);
    if (std::isnan(F64FromBits(test.expected))) {
      EXPECT_TRUE(std::isnan(software));
    } else {
      EXPECT_EQ(F64ToBits(software), test.expected);
    }
  }
}

TEST_F(ElementaryTest, FmaCanonicalizesEveryNaNOperand) {
  for (uint32_t bits : {UINT32_C(0x7F800001), UINT32_C(0xFFC01234)}) {
    const float nan = F32FromBits(bits);
    EXPECT_EQ(F32ToBits(iree_math_fma_f32(nan, 1.0f, 1.0f)), 0x7FC00000u);
    EXPECT_EQ(F32ToBits(iree_math_fma_f32(1.0f, nan, 1.0f)), 0x7FC00000u);
    EXPECT_EQ(F32ToBits(iree_math_fma_f32(1.0f, 1.0f, nan)), 0x7FC00000u);
  }
  for (uint64_t bits :
       {UINT64_C(0x7FF0000000000001), UINT64_C(0xFFF8123456789ABC)}) {
    const double nan = F64FromBits(bits);
    const uint64_t expected = UINT64_C(0x7FF8000000000000);
    EXPECT_EQ(F64ToBits(iree_math_fma_f64(nan, 1.0, 1.0)), expected);
    EXPECT_EQ(F64ToBits(iree_math_fma_f64(1.0, nan, 1.0)), expected);
    EXPECT_EQ(F64ToBits(iree_math_fma_f64(1.0, 1.0, nan)), expected);
  }
}

static void ExpectNarrowFma(F32Function round, double a, double b, double c,
                            double expected) {
  SCOPED_TRACE(::testing::Message()
               << std::hexfloat << a << " * " << b << " + " << c);
  const float actual = round(iree_math_fma_narrow(a, b, c));
  EXPECT_EQ(F32ToBits(actual), F32ToBits(static_cast<float>(expected)));
}

TEST_F(ElementaryTest, NarrowFmaRoundsOnceAtMidpoints) {
  // 129/128 * 3/2 is halfway between two BF16 values. A tiny signed addend
  // selects the side even when it disappears in both F32 and F64 arithmetic.
  for (const double sign : {1.0, -1.0}) {
    ExpectNarrowFma(iree_math_round_to_nearest_bf16, sign * 0x1.02p0, 1.5,
                    -sign * 0x1p-30, sign * 0x1.82p0);
    ExpectNarrowFma(iree_math_round_to_nearest_bf16, sign * 0x1.02p0, 1.5,
                    -sign * 0x1p-133, sign * 0x1.82p0);
    ExpectNarrowFma(iree_math_round_to_nearest_bf16, sign * 0x1.06p0, 1.5,
                    sign * 0x1p-133, sign * 0x1.8ap0);
    ExpectNarrowFma(iree_math_round_to_nearest_f16, sign * 0x1.004p0, 1.5,
                    -sign * 0x1p-24, sign * 0x1.804p0);
    ExpectNarrowFma(iree_math_round_to_nearest_f8e5m2, -sign * 0x1.4p7, 0x1.8p6,
                    sign * 0x1.4p-14, -sign * 0x1.cp13);
  }
  // Exact midpoints retain ties-to-even rather than always rounding away.
  ExpectNarrowFma(iree_math_round_to_nearest_bf16, 0x1.02p0, 1.5, 0.0,
                  0x1.84p0);
  ExpectNarrowFma(iree_math_round_to_nearest_bf16, 0x1.06p0, 1.5, 0.0,
                  0x1.88p0);
  ExpectNarrowFma(iree_math_round_to_nearest_f8e4m3fn, 1.125, 1.5, 0.0, 1.75);
}

TEST_F(ElementaryTest, NarrowFmaPreservesSubnormalAndOverflowBoundaries) {
  // Products may exceed the narrow range before cancellation by the addend.
  ExpectNarrowFma(iree_math_round_to_nearest_bf16, 0x1.fep127, 2.0, -0x1.fep127,
                  0x1.fep127);
  ExpectNarrowFma(iree_math_round_to_nearest_f16, 65504.0, 2.0, -65504.0,
                  65504.0);
  ExpectNarrowFma(iree_math_round_to_nearest_f8e5m2, 57344.0, 2.0, -57344.0,
                  57344.0);
  ExpectNarrowFma(iree_math_round_to_nearest_f8e4m3fn, 448.0, 2.0, -448.0,
                  448.0);
  ExpectNarrowFma(iree_math_round_to_nearest_f8e4m3fn, 448.0, 2.0, 0.0, 448.0);
  ExpectNarrowFma(iree_math_round_to_nearest_f16, 65504.0, 2.0, 0.0, INFINITY);

  // Subnormal result rounding, including a tie to zero and a tie to two units.
  ExpectNarrowFma(iree_math_round_to_nearest_bf16, 0x1p-133, 0.5, 0.0, 0.0);
  ExpectNarrowFma(iree_math_round_to_nearest_bf16, -0x1p-133, 0.5, -0.0, -0.0);
  ExpectNarrowFma(iree_math_round_to_nearest_bf16, 0x1p-133, 0.5, 0x1p-133,
                  0x1p-132);
  ExpectNarrowFma(iree_math_round_to_nearest_f16, 0x1p-24, 0.5, 0x1p-24,
                  0x1p-23);
  ExpectNarrowFma(iree_math_round_to_nearest_f8e5m2, 0x1p-16, 0.5, 0x1p-16,
                  0x1p-15);
  ExpectNarrowFma(iree_math_round_to_nearest_f8e4m3fn, 0x1p-9, 0.5, 0x1p-9,
                  0x1p-8);
  ExpectNarrowFma(iree_math_round_to_nearest_bf16, 0x1p-133, 0x1p-133, 1.0,
                  1.0);
}

TEST_F(ElementaryTest, NarrowFmaPreservesSignedZeroAndNaN) {
  EXPECT_EQ(F32ToBits(iree_math_fma_narrow(-0.0, 1.0, -0.0)), 0x80000000u);
  EXPECT_EQ(F32ToBits(iree_math_fma_narrow(-0.0, 1.0, 0.0)), 0u);
  EXPECT_EQ(F32ToBits(iree_math_fma_narrow(2.0, 3.0, -6.0)), 0u);
  EXPECT_EQ(F32ToBits(iree_math_fma_narrow(INFINITY, 0.0, 1.0)), 0x7FC00000u);
  EXPECT_EQ(F32ToBits(iree_math_fma_narrow(1.0, 2.0, -INFINITY)), 0xFF800000u);
}

}  // namespace
}  // namespace iree::math::testing
