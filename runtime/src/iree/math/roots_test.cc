// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/roots.h"

#include "iree/math/elementary_test_util.h"
#include "iree/math/roots_integer.h"

namespace iree::math::testing {
namespace {

TEST_F(ElementaryTest, ReciprocalF32ApproxExceptionalAndFlushBoundaries) {
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x80000001),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x7F800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0xFF800000),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x7F800001),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0xFFC01234),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x7F7FFFFF),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0xFF7FFFFF),
                   UINT32_C(0x80000000));
}

TEST_F(ElementaryTest, ReciprocalF32ApproxFrozenFiniteMapping) {
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x00800000),
                   UINT32_C(0x7E800000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x40800000),
                   UINT32_C(0x3E800000));
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x0135F782),
                   UINT32_C(0x7DB413A8));
}

TEST_F(ElementaryTest, RsqrtF32ApproxExceptionalAndFlushBoundaries) {
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x80000001),
                   UINT32_C(0xFF800000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0xBF800000),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x7F800000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0xFF800000),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x7F800001),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0xFFC01234),
                   UINT32_C(0x7FC00000));
}

TEST_F(ElementaryTest, RsqrtF32ApproxFrozenFiniteMapping) {
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x40800000),
                   UINT32_C(0x3F000000));
  // This input distinguishes the staged-f32 mapping from correct rounding.
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x70000002),
                   UINT32_C(0x273504F1));
}

TEST_F(ElementaryTest, SqrtF32ApproxExceptionalAndFlushBoundaries) {
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x00000000),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x80000000),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x80000001),
                   UINT32_C(0x80000000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0xBF800000),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x7F800000),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0xFF800000),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x7F800001),
                   UINT32_C(0x7FC00000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0xFFC01234),
                   UINT32_C(0x7FC00000));
}

TEST_F(ElementaryTest, SqrtF32ApproxFrozenFiniteMapping) {
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x40800000),
                   UINT32_C(0x40000000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x6144BCA4),
                   UINT32_C(0x50606BB2));
}

TEST_F(ElementaryTest, RootApproxMappingsIgnoreHostDenormalFlush) {
  EnableFlushDenormalsToZero();
  ExpectF32Mapping(iree_math_reciprocal_f32_approx, UINT32_C(0x7F7FFFFF),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x7F800000));
  ExpectF32Mapping(iree_math_sqrt_f32_approx, UINT32_C(0x00000001),
                   UINT32_C(0x00000000));
  ExpectF32Mapping(iree_math_rsqrt_f32_approx, UINT32_C(0x70000002),
                   UINT32_C(0x273504F1));
}

// MPFR 4.2.2, directed 256-bit intervals whose endpoints round to the same
// binary32 payload. Covers domain, subnormal, overflow, and rounding
// boundaries.
TEST_F(ElementaryTest, SqrtF32OrdinaryBoundaries) {
  const struct {
    // Exact input payload.
    uint32_t input;
    // Independently correctly rounded result (NaNs canonicalized).
    uint32_t expected;
  } cases[] = {
      {UINT32_C(0x00000000), UINT32_C(0x00000000)},
      {UINT32_C(0x80000000), UINT32_C(0x80000000)},
      {UINT32_C(0x00000001), UINT32_C(0x1A3504F3)},
      {UINT32_C(0x007FFFFF), UINT32_C(0x1FFFFFFF)},
      {UINT32_C(0x00800000), UINT32_C(0x20000000)},
      {UINT32_C(0x3F7FFFFF), UINT32_C(0x3F7FFFFF)},
      {UINT32_C(0x3F800000), UINT32_C(0x3F800000)},
      {UINT32_C(0x3F800001), UINT32_C(0x3F800000)},
      {UINT32_C(0xBF800000), UINT32_C(0x7FC00000)},
      {UINT32_C(0x7F800000), UINT32_C(0x7F800000)},
      {UINT32_C(0xFF800000), UINT32_C(0x7FC00000)},
      {UINT32_C(0x7FC00000), UINT32_C(0x7FC00000)},
      {UINT32_C(0x00000002), UINT32_C(0x1A800000)},
      {UINT32_C(0x00400000), UINT32_C(0x1FB504F3)},
      {UINT32_C(0x00800001), UINT32_C(0x20000000)},
      {UINT32_C(0x7F7FFFFF), UINT32_C(0x5F7FFFFF)},
      {UINT32_C(0x40000000), UINT32_C(0x3FB504F3)},
      {UINT32_C(0x40800000), UINT32_C(0x40000000)},
  };
  for (const auto& c : cases) {
    ExpectF32Mapping(iree_math_sqrt_f32, c.input, c.expected);
    if (c.input != 0 && c.input < UINT32_C(0x7F800000)) {
      EXPECT_EQ(iree_math_sqrt_integer(c.input, 23, 127), c.expected);
    }
  }
}

// MPFR 4.2.2, directed 256-bit intervals whose endpoints round to the same
// binary64 payload. Covers domain, subnormal, overflow, and rounding
// boundaries.
TEST_F(ElementaryTest, SqrtF64OrdinaryBoundaries) {
  const struct {
    // Exact input payload.
    uint64_t input;
    // Independently correctly rounded result (NaNs canonicalized).
    uint64_t expected;
  } cases[] = {
      {UINT64_C(0x0000000000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0x8000000000000000), UINT64_C(0x8000000000000000)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x1E60000000000000)},
      {UINT64_C(0x000FFFFFFFFFFFFF), UINT64_C(0x1FFFFFFFFFFFFFFF)},
      {UINT64_C(0x0010000000000000), UINT64_C(0x2000000000000000)},
      {UINT64_C(0x3FEFFFFFFFFFFFFF), UINT64_C(0x3FEFFFFFFFFFFFFF)},
      {UINT64_C(0x3FF0000000000000), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0x3FF0000000000001), UINT64_C(0x3FF0000000000000)},
      {UINT64_C(0xBFF0000000000000), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x7FF0000000000000), UINT64_C(0x7FF0000000000000)},
      {UINT64_C(0xFFF0000000000000), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x7FF8000000000000), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x0000000080000000), UINT64_C(0x1F56A09E667F3BCD)},
      {UINT64_C(0x0000000080000001), UINT64_C(0x1F56A09E6695DC6B)},
      {UINT64_C(0x0000000100000000), UINT64_C(0x1F60000000000000)},
      {UINT64_C(0x7FEFFFFFFFFFFFFF), UINT64_C(0x5FEFFFFFFFFFFFFF)},
      {UINT64_C(0x4000000000000000), UINT64_C(0x3FF6A09E667F3BCD)},
      {UINT64_C(0x4010000000000000), UINT64_C(0x4000000000000000)},
  };
  for (const auto& c : cases) {
    ExpectF64Mapping(iree_math_sqrt_f64, c.input, c.expected);
    if (c.input != 0 && c.input < UINT64_C(0x7FF0000000000000)) {
      EXPECT_EQ(iree_math_sqrt_integer(c.input, 52, 1023), c.expected);
    }
  }
}

// MPFR rounds sqrt to the source width before independently rounding 1/root.
// The finite witnesses distinguish this staged mapping from correctly rounded
// reciprocal square root and exercise both ends of the subnormal interval.
TEST_F(ElementaryTest, RsqrtF32OrdinaryBoundaries) {
  const struct {
    // Exact input payload.
    uint32_t input;
    // Independently rounded staged result (NaNs canonicalized).
    uint32_t expected;
  } cases[] = {
      {UINT32_C(0x00000000), UINT32_C(0x7F800000)},
      {UINT32_C(0x80000000), UINT32_C(0xFF800000)},
      {UINT32_C(0x00000001), UINT32_C(0x64B504F3)},
      {UINT32_C(0x007FFFFF), UINT32_C(0x5F000001)},
      {UINT32_C(0x00800000), UINT32_C(0x5F000000)},
      {UINT32_C(0x70000002), UINT32_C(0x273504F1)},
      {UINT32_C(0x7F7FFFFF), UINT32_C(0x1F800001)},
      {UINT32_C(0xBF800000), UINT32_C(0x7FC00000)},
      {UINT32_C(0x7F800000), UINT32_C(0x00000000)},
      {UINT32_C(0xFF800000), UINT32_C(0x7FC00000)},
      {UINT32_C(0x7F800001), UINT32_C(0x7FC00000)},
      {UINT32_C(0xFFC01234), UINT32_C(0x7FC00000)},
  };
  for (const auto& c : cases) {
    ExpectF32Mapping(iree_math_rsqrt_f32, c.input, c.expected);
  }
}

TEST_F(ElementaryTest, RsqrtF64OrdinaryBoundaries) {
  const struct {
    // Exact input payload.
    uint64_t input;
    // Independently rounded staged result (NaNs canonicalized).
    uint64_t expected;
  } cases[] = {
      {UINT64_C(0x0000000000000000), UINT64_C(0x7FF0000000000000)},
      {UINT64_C(0x8000000000000000), UINT64_C(0xFFF0000000000000)},
      {UINT64_C(0x0000000000000001), UINT64_C(0x6180000000000000)},
      {UINT64_C(0x000FFFFFFFFFFFFF), UINT64_C(0x5FE0000000000001)},
      {UINT64_C(0x0010000000000000), UINT64_C(0x5FE0000000000000)},
      {UINT64_C(0x4000000000000000), UINT64_C(0x3FE6A09E667F3BCC)},
      {UINT64_C(0x7FEFFFFFFFFFFFFF), UINT64_C(0x1FF0000000000001)},
      {UINT64_C(0xBFF0000000000000), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x7FF0000000000000), UINT64_C(0x0000000000000000)},
      {UINT64_C(0xFFF0000000000000), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0x7FF0000000000001), UINT64_C(0x7FF8000000000000)},
      {UINT64_C(0xFFF8000000001234), UINT64_C(0x7FF8000000000000)},
  };
  for (const auto& c : cases) {
    ExpectF64Mapping(iree_math_rsqrt_f64, c.input, c.expected);
  }
}

}  // namespace
}  // namespace iree::math::testing
