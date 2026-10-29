// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/exponential.h"

#include "iree/math/elementary_test_util.h"

namespace iree::math::testing {
namespace {

TEST_F(ElementaryTest, Exp2F32ApproxExceptionalAndFlushBoundaries) {
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x80000001),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x7F800000),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0xFF800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x7F800001),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0xFFC01234),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0xC2FC0000),
                   UINT32_C(0x00800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0xC2FC0001),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x43000000),
                   UINT32_C(0x7F800000));
}

TEST_F(ElementaryTest, Exp2F32ApproxFrozenFiniteMapping) {
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x3F000000),
                   UINT32_C(0x3FB504F3));
  // These inputs distinguish the selected mapping from correct rounding.
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x3F800B8B),
                   UINT32_C(0x40000801));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0xC0004654),
                   UINT32_C(0x3E7F3D4D));
}

TEST_F(ElementaryTest, ExponentialApproxMappingsIgnoreHostDenormalFlush) {
  EnableFlushDenormalsToZero();
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0xC2FC0000),
                   UINT32_C(0x00800000));
}

// MPFR 4.2.2, directed 256-bit intervals whose endpoints round to the same
// binary32 payload. Covers domain, subnormal, overflow, and rounding
// boundaries.
TEST_F(ElementaryTest, ExpF32OrdinaryBoundaries) {
  const struct {
    // Exact input payload.
    uint32_t input;
    // Independently correctly rounded result (NaNs canonicalized).
    uint32_t expected;
  } cases[] = {
      {UINT32_C(0x00000000), UINT32_C(0x3F800000)},
      {UINT32_C(0x80000000), UINT32_C(0x3F800000)},
      {UINT32_C(0x00000001), UINT32_C(0x3F800000)},
      {UINT32_C(0x007FFFFF), UINT32_C(0x3F800000)},
      {UINT32_C(0x00800000), UINT32_C(0x3F800000)},
      {UINT32_C(0x3F7FFFFF), UINT32_C(0x402DF854)},
      {UINT32_C(0x3F800000), UINT32_C(0x402DF854)},
      {UINT32_C(0x3F800001), UINT32_C(0x402DF856)},
      {UINT32_C(0xBF800000), UINT32_C(0x3EBC5AB2)},
      {UINT32_C(0x7F800000), UINT32_C(0x7F800000)},
      {UINT32_C(0xFF800000), UINT32_C(0x00000000)},
      {UINT32_C(0x7FC00000), UINT32_C(0x7FC00000)},
      {UINT32_C(0x42B17217), UINT32_C(0x7F7FFF84)},
      {UINT32_C(0x42B17218), UINT32_C(0x7F800000)},
      {UINT32_C(0x42B17219), UINT32_C(0x7F800000)},
      {UINT32_C(0xC2CFF1B4), UINT32_C(0x00000001)},
      {UINT32_C(0x33800000), UINT32_C(0x3F800001)},
      {UINT32_C(0xB3800000), UINT32_C(0x3F7FFFFF)},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << "input_bits=" << std::hex << c.input);
    const uint32_t actual = F32ToBits(iree_math_exp_f32(F32FromBits(c.input)));
    if (c.expected == 0 || c.expected >= UINT32_C(0x7F800000)) {
      EXPECT_EQ(actual, c.expected);
    } else {
      EXPECT_LE(actual > c.expected ? actual - c.expected : c.expected - actual,
                UINT32_C(1));
    }
  }
}

// MPFR 4.2.2, directed 256-bit intervals whose endpoints round to the same
// binary32 payload. Covers domain, subnormal, overflow, and rounding
// boundaries.
TEST_F(ElementaryTest, Exp2F32OrdinaryBoundaries) {
  const struct {
    // Exact input payload.
    uint32_t input;
    // Independently correctly rounded result (NaNs canonicalized).
    uint32_t expected;
  } cases[] = {
      {UINT32_C(0x00000000), UINT32_C(0x3F800000)},
      {UINT32_C(0x80000000), UINT32_C(0x3F800000)},
      {UINT32_C(0x00000001), UINT32_C(0x3F800000)},
      {UINT32_C(0x007FFFFF), UINT32_C(0x3F800000)},
      {UINT32_C(0x00800000), UINT32_C(0x3F800000)},
      {UINT32_C(0x3F7FFFFF), UINT32_C(0x3FFFFFFF)},
      {UINT32_C(0x3F800000), UINT32_C(0x40000000)},
      {UINT32_C(0x3F800001), UINT32_C(0x40000001)},
      {UINT32_C(0xBF800000), UINT32_C(0x3F000000)},
      {UINT32_C(0x7F800000), UINT32_C(0x7F800000)},
      {UINT32_C(0xFF800000), UINT32_C(0x00000000)},
      {UINT32_C(0x7FC00000), UINT32_C(0x7FC00000)},
      {UINT32_C(0x42FE0000), UINT32_C(0x7F000000)},
      {UINT32_C(0x43000000), UINT32_C(0x7F800000)},
      {UINT32_C(0xC3150000), UINT32_C(0x00000001)},
      {UINT32_C(0xC3160000), UINT32_C(0x00000000)},
      {UINT32_C(0x33800000), UINT32_C(0x3F800000)},
      {UINT32_C(0xB3800000), UINT32_C(0x3F7FFFFF)},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << "input_bits=" << std::hex << c.input);
    const uint32_t actual = F32ToBits(iree_math_exp2_f32(F32FromBits(c.input)));
    if (c.expected == 0 || c.expected >= UINT32_C(0x7F800000)) {
      EXPECT_EQ(actual, c.expected);
    } else {
      EXPECT_LE(actual > c.expected ? actual - c.expected : c.expected - actual,
                UINT32_C(1));
    }
  }
}

// MPFR 4.2.2, directed 256-bit intervals whose endpoints round to the same
// binary64 payload. Covers domain, subnormal, overflow, and rounding
// boundaries.
TEST_F(ElementaryTest, ExpF64OrdinaryBoundaries) {
  const struct {
    // Exact input payload.
    uint64_t input;
    // Independently correctly rounded result (NaNs canonicalized).
    uint64_t expected;
  } cases[] = {
      {UINT64_C(0x0000000000000000), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x8000000000000000), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x000FFFFFFFFFFFFF), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x0010000000000000), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x3FEFFFFFFFFFFFFF), UINT64_C(0x4005BF0A8B145769)},
      {UINT64_C(0x3FF0000000000000), UINT64_C(0x4005BF0A8B145769)},
      {UINT64_C(0x3FF0000000000001), UINT64_C(0x4005BF0A8B14576B)},
      {UINT64_C(0xBFF0000000000000), UINT64_C(0x3FD78B56362CEF38)},
      {UINT64_C(0x7FF0000000000000), UINT64_C(0x7FF0000000000000)},
      {UINT64_C(0xFFF0000000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0x7FF8000000000000), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x40862E42FEFA39EE), UINT64_C(0x7FEFFFFFFFFFFB2A)},
      {UINT64_C(0x40862E42FEFA39EF), UINT64_C(0x7FEFFFFFFFFFFF2A)},
      {UINT64_C(0x40862E42FEFA39F0), UINT64_C(0x7FF0000000000000)},
      {UINT64_C(0xC0874910D52D3051), UINT64_C(0x0000000000000001)},
      {UINT64_C(0x3CA0000000000000), UINT64_C(0x3FF0000000000001)},
      {UINT64_C(0xBCA0000000000000), UINT64_C(0x3FEFFFFFFFFFFFFF)},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << "input_bits=" << std::hex << c.input);
    const uint64_t actual = F64ToBits(iree_math_exp_f64(F64FromBits(c.input)));
    if (c.expected == 0 || c.expected >= UINT64_C(0x7FF0000000000000)) {
      EXPECT_EQ(actual, c.expected);
    } else {
      EXPECT_LE(actual > c.expected ? actual - c.expected : c.expected - actual,
                UINT64_C(1));
    }
  }
}

// MPFR 4.2.2, directed 256-bit intervals whose endpoints round to the same
// binary64 payload. Covers domain, subnormal, overflow, and rounding
// boundaries.
TEST_F(ElementaryTest, Exp2F64OrdinaryBoundaries) {
  const struct {
    // Exact input payload.
    uint64_t input;
    // Independently correctly rounded result (NaNs canonicalized).
    uint64_t expected;
  } cases[] = {
      {UINT64_C(0x0000000000000000), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x8000000000000000), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x000FFFFFFFFFFFFF), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x0010000000000000), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x3FEFFFFFFFFFFFFF), UINT64_C(0x3FFFFFFFFFFFFFFF)},
      {UINT64_C(0x3FF0000000000000), UINT64_C(0x4000000000000000)},
      {UINT64_C(0x3FF0000000000001), UINT64_C(0x4000000000000001)},
      {UINT64_C(0xBFF0000000000000), UINT64_C(0x3FE0000000000000)},
      {UINT64_C(0x7FF0000000000000), UINT64_C(0x7FF0000000000000)},
      {UINT64_C(0xFFF0000000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0x7FF8000000000000), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x408FF80000000000), UINT64_C(0x7FE0000000000000)},
      {UINT64_C(0x4090000000000000), UINT64_C(0x7FF0000000000000)},
      {UINT64_C(0xC090C80000000000), UINT64_C(0x0000000000000001)},
      {UINT64_C(0xC090CC0000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0x3CA0000000000000), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0xBCA0000000000000), UINT64_C(0x3FEFFFFFFFFFFFFF)},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << "input_bits=" << std::hex << c.input);
    const uint64_t actual = F64ToBits(iree_math_exp2_f64(F64FromBits(c.input)));
    if (c.expected == 0 || c.expected >= UINT64_C(0x7FF0000000000000)) {
      EXPECT_EQ(actual, c.expected);
    } else {
      EXPECT_LE(actual > c.expected ? actual - c.expected : c.expected - actual,
                UINT64_C(1));
    }
  }
}

// MPFR 4.2.2 directed 256-bit intervals, independently rounded to F32.
// Covers cancellation, reduction/reconstruction boundaries, saturation, and
// overflow.
TEST_F(ElementaryTest, Expm1F32Boundaries) {
  const struct {
    // Exact input payload.
    uint32_t input;
    // Independently correctly rounded result (NaNs canonicalized).
    uint32_t expected;
  } cases[] = {
      {UINT32_C(0x00000000), UINT32_C(0x00000000)},
      {UINT32_C(0x80000000), UINT32_C(0x80000000)},
      {UINT32_C(0x00000001), UINT32_C(0x00000001)},
      {UINT32_C(0x80000001), UINT32_C(0x80000001)},
      {UINT32_C(0x007FFFFF), UINT32_C(0x007FFFFF)},
      {UINT32_C(0x00800000), UINT32_C(0x00800000)},
      {UINT32_C(0x00800001), UINT32_C(0x00800001)},
      {UINT32_C(0x7F7FFFFF), UINT32_C(0x7F800000)},
      {UINT32_C(0xFF7FFFFF), UINT32_C(0xBF800000)},
      {UINT32_C(0x7F800000), UINT32_C(0x7F800000)},
      {UINT32_C(0xFF800000), UINT32_C(0xBF800000)},
      {UINT32_C(0x7F800001), UINT32_C(0x7FC00000)},
      {UINT32_C(0x7FC00011), UINT32_C(0x7FC00000)},
      {UINT32_C(0xFF800001), UINT32_C(0x7FC00000)},
      {UINT32_C(0x32FFFFFF), UINT32_C(0x32FFFFFF)},
      {UINT32_C(0x33000000), UINT32_C(0x33000000)},
      {UINT32_C(0x33000001), UINT32_C(0x33000001)},
      {UINT32_C(0xB2FFFFFF), UINT32_C(0xB2FFFFFF)},
      {UINT32_C(0xB3000000), UINT32_C(0xB3000000)},
      {UINT32_C(0xB3000001), UINT32_C(0xB3000001)},
      {UINT32_C(0x3EB17217), UINT32_C(0x3ED413CB)},
      {UINT32_C(0x3EB17218), UINT32_C(0x3ED413CD)},
      {UINT32_C(0x3EB17219), UINT32_C(0x3ED413CE)},
      {UINT32_C(0xBEB17217), UINT32_C(0xBE95F619)},
      {UINT32_C(0xBEB17218), UINT32_C(0xBE95F61A)},
      {UINT32_C(0xBEB17219), UINT32_C(0xBE95F61A)},
      {UINT32_C(0x3F851591), UINT32_C(0x3FEA09E4)},
      {UINT32_C(0x3F851592), UINT32_C(0x3FEA09E6)},
      {UINT32_C(0x3F851593), UINT32_C(0x3FEA09E9)},
      {UINT32_C(0xBF851591), UINT32_C(0xBF257D86)},
      {UINT32_C(0xBF851592), UINT32_C(0xBF257D86)},
      {UINT32_C(0xBF851593), UINT32_C(0xBF257D87)},
      {UINT32_C(0x4195B843), UINT32_C(0x4CFFFFD9)},
      {UINT32_C(0x4195B844), UINT32_C(0x4CFFFFF9)},
      {UINT32_C(0x4195B845), UINT32_C(0x4D00000C)},
      {UINT32_C(0xC195B843), UINT32_C(0xBF800000)},
      {UINT32_C(0xC195B844), UINT32_C(0xBF800000)},
      {UINT32_C(0xC195B845), UINT32_C(0xBF800000)},
      {UINT32_C(0x42B17217), UINT32_C(0x7F7FFF84)},
      {UINT32_C(0x42B17218), UINT32_C(0x7F800000)},
      {UINT32_C(0x42B17219), UINT32_C(0x7F800000)},
      {UINT32_C(0xC2B17217), UINT32_C(0xBF800000)},
      {UINT32_C(0xC2B17218), UINT32_C(0xBF800000)},
      {UINT32_C(0xC2B17219), UINT32_C(0xBF800000)},
      {UINT32_C(0xBFB17218), UINT32_C(0xBF400000)},
      {UINT32_C(0xBF317218), UINT32_C(0xBF000000)},
      {UINT32_C(0x3F317218), UINT32_C(0x3F800000)},
      {UINT32_C(0x4152B77C), UINT32_C(0x48FFFFD9)},
      {UINT32_C(0x415DCE9E), UINT32_C(0x497FFFF1)},
      {UINT32_C(0x4173FCE1), UINT32_C(0x4A7FFFFD)},
      {UINT32_C(0x417F1402), UINT32_C(0x4AFFFFF7)},
      {UINT32_C(0x421B43D5), UINT32_C(0x5B800001)},
      {UINT32_C(0x421E099D), UINT32_C(0x5BFFFFEA)},
      {UINT32_C(0x42B00F34), UINT32_C(0x7F00000E)},
      {UINT32_C(0xC18AA122), UINT32_C(0xBF7FFFFF)},
      {UINT32_C(0xC18AA123), UINT32_C(0xBF800000)},
      {UINT32_C(0xC18AA124), UINT32_C(0xBF800000)},
      {UINT32_C(0x3EE2E42F), UINT32_C(0x3F0EBEF9)},
      {UINT32_C(0x3EE2E430), UINT32_C(0x3F0EBEFA)},
      {UINT32_C(0x3EE2E431), UINT32_C(0x3F0EBEFB)},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << "input_bits=" << std::hex << c.input);
    const uint32_t actual =
        F32ToBits(iree_math_expm1_f32(F32FromBits(c.input)));
    const uint32_t magnitude = c.expected & UINT32_C(0x7FFFFFFF);
    if (magnitude == 0 || magnitude >= UINT32_C(0x7F800000)) {
      EXPECT_EQ(actual, c.expected);
    } else {
      EXPECT_LE(actual > c.expected ? actual - c.expected : c.expected - actual,
                UINT32_C(1));
    }
  }
}

// MPFR 4.2.2 directed 256-bit intervals, independently rounded to F64.
// Covers cancellation, reduction/reconstruction boundaries, saturation, and
// overflow.
TEST_F(ElementaryTest, Expm1F64Boundaries) {
  const struct {
    // Exact input payload.
    uint64_t input;
    // Independently correctly rounded result (NaNs canonicalized).
    uint64_t expected;
  } cases[] = {
      {UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0x8000000000000000), UINT64_C(0x8000000000000000)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x0000000000000001)},
      {UINT64_C(0x8000000000000001), UINT64_C(0x8000000000000001)},
      {UINT64_C(0x000FFFFFFFFFFFFF), UINT64_C(0x000FFFFFFFFFFFFF)},
      {UINT64_C(0x0010000000000000), UINT64_C(0x0010000000000000)},
      {UINT64_C(0x0010000000000001), UINT64_C(0x0010000000000001)},
      {UINT64_C(0x7FEFFFFFFFFFFFFF), UINT64_C(0x7FF0000000000000)},
      {UINT64_C(0xFFEFFFFFFFFFFFFF), UINT64_C(0xBFF0000000000000)},
      {UINT64_C(0x7FF0000000000000), UINT64_C(0x7FF0000000000000)},
      {UINT64_C(0xFFF0000000000000), UINT64_C(0xBFF0000000000000)},
      {UINT64_C(0x7FF0000000000001), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x7FF8000000000011), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0xFFF0000000000001), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x3C8FFFFFFFFFFFFF), UINT64_C(0x3C8FFFFFFFFFFFFF)},
      {UINT64_C(0x3C90000000000000), UINT64_C(0x3C90000000000000)},
      {UINT64_C(0x3C90000000000001), UINT64_C(0x3C90000000000001)},
      {UINT64_C(0xBC8FFFFFFFFFFFFF), UINT64_C(0xBC8FFFFFFFFFFFFF)},
      {UINT64_C(0xBC90000000000000), UINT64_C(0xBC90000000000000)},
      {UINT64_C(0xBC90000000000001), UINT64_C(0xBC90000000000001)},
      {UINT64_C(0x3FD62E42FFFFFFFF), UINT64_C(0x3FDA82799B6F2364)},
      {UINT64_C(0x3FD62E4300000000), UINT64_C(0x3FDA82799B6F2366)},
      {UINT64_C(0x3FD62E4300000001), UINT64_C(0x3FDA82799B6F2367)},
      {UINT64_C(0xBFD62E42FFFFFFFF), UINT64_C(0xBFD2BEC333BAA280)},
      {UINT64_C(0xBFD62E4300000000), UINT64_C(0xBFD2BEC333BAA281)},
      {UINT64_C(0xBFD62E4300000001), UINT64_C(0xBFD2BEC333BAA281)},
      {UINT64_C(0x3FF0A2B1FFFFFFFF), UINT64_C(0x3FFD413C1A24D411)},
      {UINT64_C(0x3FF0A2B200000000), UINT64_C(0x3FFD413C1A24D414)},
      {UINT64_C(0x3FF0A2B200000001), UINT64_C(0x3FFD413C1A24D416)},
      {UINT64_C(0xBFF0A2B1FFFFFFFF), UINT64_C(0xBFE4AFB0A009F887)},
      {UINT64_C(0xBFF0A2B200000000), UINT64_C(0xBFE4AFB0A009F888)},
      {UINT64_C(0xBFF0A2B200000001), UINT64_C(0xBFE4AFB0A009F888)},
      {UINT64_C(0x40436879FFFFFFFF), UINT64_C(0x436FFFD8395C0C01)},
      {UINT64_C(0x4043687A00000000), UINT64_C(0x436FFFD8395C0C41)},
      {UINT64_C(0x4043687A00000001), UINT64_C(0x436FFFD8395C0C81)},
      {UINT64_C(0xC0436879FFFFFFFF), UINT64_C(0xBFF0000000000000)},
      {UINT64_C(0xC043687A00000000), UINT64_C(0xBFF0000000000000)},
      {UINT64_C(0xC043687A00000001), UINT64_C(0xBFF0000000000000)},
      {UINT64_C(0x40862E42FEFA39EE), UINT64_C(0x7FEFFFFFFFFFFB2A)},
      {UINT64_C(0x40862E42FEFA39EF), UINT64_C(0x7FEFFFFFFFFFFF2A)},
      {UINT64_C(0x40862E42FEFA39F0), UINT64_C(0x7FF0000000000000)},
      {UINT64_C(0xC0862E42FEFA39EE), UINT64_C(0xBFF0000000000000)},
      {UINT64_C(0xC0862E42FEFA39EF), UINT64_C(0xBFF0000000000000)},
      {UINT64_C(0xC0862E42FEFA39F0), UINT64_C(0xBFF0000000000000)},
      {UINT64_C(0xBFF62E42FEFA39EF), UINT64_C(0xBFE8000000000000)},
      {UINT64_C(0xBFE62E42FEFA39EF), UINT64_C(0xBFE0000000000000)},
      {UINT64_C(0x3FE62E42FEFA39EF), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x402A56EF8EC924CC), UINT64_C(0x411FFFFBFFFFFFFF)},
      {UINT64_C(0x402BB9D3BEB8C86B), UINT64_C(0x412FFFFE00000000)},
      {UINT64_C(0x402E7F9C1E980FA9), UINT64_C(0x414FFFFF80000001)},
      {UINT64_C(0x402FE2804E87B348), UINT64_C(0x415FFFFFC0000002)},
      {UINT64_C(0x4043687A9F1AF2B1), UINT64_C(0x436FFFFFFFFFFFEC)},
      {UINT64_C(0x4043C133AB16DB99), UINT64_C(0x437FFFFFFFFFFFFD)},
      {UINT64_C(0x408628B76E3A7B61), UINT64_C(0x7FE000000000008D)},
      {UINT64_C(0xC042B708872320E1), UINT64_C(0xBFEFFFFFFFFFFFFF)},
      {UINT64_C(0xC042B708872320E2), UINT64_C(0xBFF0000000000000)},
      {UINT64_C(0xC042B708872320E3), UINT64_C(0xBFF0000000000000)},
      {UINT64_C(0x3FDC5C85FDF473DD), UINT64_C(0x3FE1D7DF3D590415)},
      {UINT64_C(0x3FDC5C85FDF473DE), UINT64_C(0x3FE1D7DF3D590415)},
      {UINT64_C(0x3FDC5C85FDF473DF), UINT64_C(0x3FE1D7DF3D590416)},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << "input_bits=" << std::hex << c.input);
    const uint64_t actual =
        F64ToBits(iree_math_expm1_f64(F64FromBits(c.input)));
    const uint64_t magnitude = c.expected & UINT64_C(0x7FFFFFFFFFFFFFFF);
    if (magnitude == 0 || magnitude >= UINT64_C(0x7FF0000000000000)) {
      EXPECT_EQ(actual, c.expected);
    } else {
      EXPECT_LE(actual > c.expected ? actual - c.expected : c.expected - actual,
                UINT64_C(1));
    }
  }
}

TEST_F(ElementaryTest, Expm1FrozenFiniteMapping) {
  // These reduction-boundary results distinguish the selected mapping from
  // correctly rounded expm1 while remaining within one adjacent value.
  ExpectF32Mapping(iree_math_expm1_f32, UINT32_C(0x3EB17217),
                   UINT32_C(0x3ED413CC));
  ExpectF64Mapping(iree_math_expm1_f64, UINT64_C(0xBFD62E4300000000),
                   UINT64_C(0xBFD2BEC333BAA280));
}

}  // namespace
}  // namespace iree::math::testing
