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

TEST_F(ElementaryTest, ExponentialApproxMappingsIgnoreHostDenormalFlush) {
  EnableFlushDenormalsToZero();
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x3F800000));
  ExpectF32Mapping(iree_math_exp2_f32_approx, UINT32_C(0xC2FC0000),
                   UINT32_C(0x00800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_log2_f32_approx, UINT32_C(0x70006067),
                   UINT32_C(0x42C2022B));
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

}  // namespace
}  // namespace iree::math::testing
