// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/rounding.h"

#include <cfenv>
#include <cmath>

#include "iree/math/elementary_test_util.h"

namespace iree::math::testing {
namespace {

template <typename Float>
struct RoundingBoundary {
  // Positive input at an integer, halfway, exponent, or subnormal boundary.
  Float input;
  // Exact rational results: ceil, floor, trunc, nearest-even, nearest-away.
  Float expected[5];
};

class RoundingTest : public ElementaryTest {
 protected:
  void SetUp() override {
    rounding_mode_ = std::fegetround();
    ElementaryTest::SetUp();
  }
  void TearDown() override {
    EXPECT_EQ(std::fesetround(rounding_mode_), 0);
    ElementaryTest::TearDown();
  }

 private:
  // Saved language rounding mode, including any non-SIMD control register.
  int rounding_mode_ = FE_TONEAREST;
};

template <typename Float, size_t Count>
void CheckRounding(const RoundingBoundary<Float> (&cases)[Count],
                   Float (*const (&functions)[5])(Float)) {
  for (int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    for (const auto& test : cases) {
      for (size_t operation = 0; operation < 5; ++operation) {
        for (bool negative : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << "mode=" << mode << " operation=" << operation
                       << " negative=" << negative << " input=" << std::hexfloat
                       << test.input);
          // Negation exchanges the two directed modes. Truncation and both
          // nearest modes commute with negation, including a signed zero.
          const size_t expected_operation =
              negative && operation < 2 ? 1 - operation : operation;
          const Float expected = negative ? -test.expected[expected_operation]
                                          : test.expected[expected_operation];
          const Float actual =
              functions[operation](negative ? -test.input : test.input);
          EXPECT_EQ(actual, expected);
          EXPECT_EQ(std::signbit(actual), std::signbit(expected));
        }
      }
    }
  }
}

// Independent exact numerator/denominator division, including halfway parity.
// The nearest encodings around each branch boundary distinguish bias/mask
// errors.

TEST_F(RoundingTest, F32Boundaries) {
  const RoundingBoundary<float> cases[] = {
      {0x0p+0f, {0x0p+0f, 0x0p+0f, 0x0p+0f, 0x0p+0f, 0x0p+0f}},
      {0x1p-149f, {0x1p+0f, 0x0p+0f, 0x0p+0f, 0x0p+0f, 0x0p+0f}},
      {0x1.fffffcp-127f, {0x1p+0f, 0x0p+0f, 0x0p+0f, 0x0p+0f, 0x0p+0f}},
      {0x1p-126f, {0x1p+0f, 0x0p+0f, 0x0p+0f, 0x0p+0f, 0x0p+0f}},
      {0x1.fffffep+127f,
       {0x1.fffffep+127f, 0x1.fffffep+127f, 0x1.fffffep+127f, 0x1.fffffep+127f,
        0x1.fffffep+127f}},
      {0x1.fffffep-2f, {0x1p+0f, 0x0p+0f, 0x0p+0f, 0x0p+0f, 0x0p+0f}},
      {0x1p-1f, {0x1p+0f, 0x0p+0f, 0x0p+0f, 0x0p+0f, 0x1p+0f}},
      {0x1.000002p-1f, {0x1p+0f, 0x0p+0f, 0x0p+0f, 0x1p+0f, 0x1p+0f}},
      {0x1.fffffep-1f, {0x1p+0f, 0x0p+0f, 0x0p+0f, 0x1p+0f, 0x1p+0f}},
      {0x1p+0f, {0x1p+0f, 0x1p+0f, 0x1p+0f, 0x1p+0f, 0x1p+0f}},
      {0x1.000002p+0f, {0x1p+1f, 0x1p+0f, 0x1p+0f, 0x1p+0f, 0x1p+0f}},
      {0x1.7ffffep+0f, {0x1p+1f, 0x1p+0f, 0x1p+0f, 0x1p+0f, 0x1p+0f}},
      {0x1.8p+0f, {0x1p+1f, 0x1p+0f, 0x1p+0f, 0x1p+1f, 0x1p+1f}},
      {0x1.800002p+0f, {0x1p+1f, 0x1p+0f, 0x1p+0f, 0x1p+1f, 0x1p+1f}},
      {0x1.fffffep+0f, {0x1p+1f, 0x1p+0f, 0x1p+0f, 0x1p+1f, 0x1p+1f}},
      {0x1p+1f, {0x1p+1f, 0x1p+1f, 0x1p+1f, 0x1p+1f, 0x1p+1f}},
      {0x1.000002p+1f, {0x1.8p+1f, 0x1p+1f, 0x1p+1f, 0x1p+1f, 0x1p+1f}},
      {0x1.3ffffep+1f, {0x1.8p+1f, 0x1p+1f, 0x1p+1f, 0x1p+1f, 0x1p+1f}},
      {0x1.4p+1f, {0x1.8p+1f, 0x1p+1f, 0x1p+1f, 0x1p+1f, 0x1.8p+1f}},
      {0x1.400002p+1f, {0x1.8p+1f, 0x1p+1f, 0x1p+1f, 0x1.8p+1f, 0x1.8p+1f}},
      {0x1.bffffep+1f, {0x1p+2f, 0x1.8p+1f, 0x1.8p+1f, 0x1.8p+1f, 0x1.8p+1f}},
      {0x1.cp+1f, {0x1p+2f, 0x1.8p+1f, 0x1.8p+1f, 0x1p+2f, 0x1p+2f}},
      {0x1.c00002p+1f, {0x1p+2f, 0x1.8p+1f, 0x1.8p+1f, 0x1p+2f, 0x1p+2f}},
      {0x1.fffffep+1f, {0x1p+2f, 0x1.8p+1f, 0x1.8p+1f, 0x1p+2f, 0x1p+2f}},
      {0x1p+2f, {0x1p+2f, 0x1p+2f, 0x1p+2f, 0x1p+2f, 0x1p+2f}},
      {0x1.000002p+2f, {0x1.4p+2f, 0x1p+2f, 0x1p+2f, 0x1p+2f, 0x1p+2f}},
      {0x1.fdfffep+6f,
       {0x1p+7f, 0x1.fcp+6f, 0x1.fcp+6f, 0x1.fcp+6f, 0x1.fcp+6f}},
      {0x1.fep+6f, {0x1p+7f, 0x1.fcp+6f, 0x1.fcp+6f, 0x1p+7f, 0x1p+7f}},
      {0x1.fe0002p+6f, {0x1p+7f, 0x1.fcp+6f, 0x1.fcp+6f, 0x1p+7f, 0x1p+7f}},
      {0x1.fffffep+7f, {0x1p+8f, 0x1.fep+7f, 0x1.fep+7f, 0x1p+8f, 0x1p+8f}},
      {0x1p+8f, {0x1p+8f, 0x1p+8f, 0x1p+8f, 0x1p+8f, 0x1p+8f}},
      {0x1.000002p+8f, {0x1.01p+8f, 0x1p+8f, 0x1p+8f, 0x1p+8f, 0x1p+8f}},
      {0x1.fffffep+21f,
       {0x1p+22f, 0x1.fffff8p+21f, 0x1.fffff8p+21f, 0x1p+22f, 0x1p+22f}},
      {0x1p+22f, {0x1p+22f, 0x1p+22f, 0x1p+22f, 0x1p+22f, 0x1p+22f}},
      {0x1.000002p+22f,
       {0x1.000004p+22f, 0x1p+22f, 0x1p+22f, 0x1p+22f, 0x1.000004p+22f}},
      {0x1.fffffep+22f,
       {0x1p+23f, 0x1.fffffcp+22f, 0x1.fffffcp+22f, 0x1p+23f, 0x1p+23f}},
      {0x1p+23f, {0x1p+23f, 0x1p+23f, 0x1p+23f, 0x1p+23f, 0x1p+23f}},
      {0x1.000002p+23f,
       {0x1.000002p+23f, 0x1.000002p+23f, 0x1.000002p+23f, 0x1.000002p+23f,
        0x1.000002p+23f}},
  };
  float (*const functions[5])(float) = {
      iree_math_ceil_f32,       iree_math_floor_f32,      iree_math_trunc_f32,
      iree_math_round_even_f32, iree_math_round_away_f32,
  };
  CheckRounding(cases, functions);
}

TEST_F(RoundingTest, F64Boundaries) {
  const RoundingBoundary<double> cases[] = {
      {0x0p+0, {0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0}},
      {0x0.0000000000001p-1022, {0x1p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0}},
      {0x0.fffffffffffffp-1022, {0x1p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0}},
      {0x1p-1022, {0x1p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0}},
      {0x1.fffffffffffffp+1023,
       {0x1.fffffffffffffp+1023, 0x1.fffffffffffffp+1023,
        0x1.fffffffffffffp+1023, 0x1.fffffffffffffp+1023,
        0x1.fffffffffffffp+1023}},
      {0x1.fffffffffffffp-2, {0x1p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x0p+0}},
      {0x1p-1, {0x1p+0, 0x0p+0, 0x0p+0, 0x0p+0, 0x1p+0}},
      {0x1.0000000000001p-1, {0x1p+0, 0x0p+0, 0x0p+0, 0x1p+0, 0x1p+0}},
      {0x1.fffffffffffffp-1, {0x1p+0, 0x0p+0, 0x0p+0, 0x1p+0, 0x1p+0}},
      {0x1p+0, {0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0}},
      {0x1.0000000000001p+0, {0x1p+1, 0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0}},
      {0x1.7ffffffffffffp+0, {0x1p+1, 0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0}},
      {0x1.8p+0, {0x1p+1, 0x1p+0, 0x1p+0, 0x1p+1, 0x1p+1}},
      {0x1.8000000000001p+0, {0x1p+1, 0x1p+0, 0x1p+0, 0x1p+1, 0x1p+1}},
      {0x1.fffffffffffffp+0, {0x1p+1, 0x1p+0, 0x1p+0, 0x1p+1, 0x1p+1}},
      {0x1p+1, {0x1p+1, 0x1p+1, 0x1p+1, 0x1p+1, 0x1p+1}},
      {0x1.0000000000001p+1, {0x1.8p+1, 0x1p+1, 0x1p+1, 0x1p+1, 0x1p+1}},
      {0x1.3ffffffffffffp+1, {0x1.8p+1, 0x1p+1, 0x1p+1, 0x1p+1, 0x1p+1}},
      {0x1.4p+1, {0x1.8p+1, 0x1p+1, 0x1p+1, 0x1p+1, 0x1.8p+1}},
      {0x1.4000000000001p+1, {0x1.8p+1, 0x1p+1, 0x1p+1, 0x1.8p+1, 0x1.8p+1}},
      {0x1.bffffffffffffp+1, {0x1p+2, 0x1.8p+1, 0x1.8p+1, 0x1.8p+1, 0x1.8p+1}},
      {0x1.cp+1, {0x1p+2, 0x1.8p+1, 0x1.8p+1, 0x1p+2, 0x1p+2}},
      {0x1.c000000000001p+1, {0x1p+2, 0x1.8p+1, 0x1.8p+1, 0x1p+2, 0x1p+2}},
      {0x1.fffffffffffffp+1, {0x1p+2, 0x1.8p+1, 0x1.8p+1, 0x1p+2, 0x1p+2}},
      {0x1p+2, {0x1p+2, 0x1p+2, 0x1p+2, 0x1p+2, 0x1p+2}},
      {0x1.0000000000001p+2, {0x1.4p+2, 0x1p+2, 0x1p+2, 0x1p+2, 0x1p+2}},
      {0x1.fdfffffffffffp+6,
       {0x1p+7, 0x1.fcp+6, 0x1.fcp+6, 0x1.fcp+6, 0x1.fcp+6}},
      {0x1.fep+6, {0x1p+7, 0x1.fcp+6, 0x1.fcp+6, 0x1p+7, 0x1p+7}},
      {0x1.fe00000000001p+6, {0x1p+7, 0x1.fcp+6, 0x1.fcp+6, 0x1p+7, 0x1p+7}},
      {0x1.fffffffffffffp+7, {0x1p+8, 0x1.fep+7, 0x1.fep+7, 0x1p+8, 0x1p+8}},
      {0x1p+8, {0x1p+8, 0x1p+8, 0x1p+8, 0x1p+8, 0x1p+8}},
      {0x1.0000000000001p+8, {0x1.01p+8, 0x1p+8, 0x1p+8, 0x1p+8, 0x1p+8}},
      {0x1.fffffffffffffp+50,
       {0x1p+51, 0x1.ffffffffffffcp+50, 0x1.ffffffffffffcp+50, 0x1p+51,
        0x1p+51}},
      {0x1p+51, {0x1p+51, 0x1p+51, 0x1p+51, 0x1p+51, 0x1p+51}},
      {0x1.0000000000001p+51,
       {0x1.0000000000002p+51, 0x1p+51, 0x1p+51, 0x1p+51,
        0x1.0000000000002p+51}},
      {0x1.fffffffffffffp+51,
       {0x1p+52, 0x1.ffffffffffffep+51, 0x1.ffffffffffffep+51, 0x1p+52,
        0x1p+52}},
      {0x1p+52, {0x1p+52, 0x1p+52, 0x1p+52, 0x1p+52, 0x1p+52}},
      {0x1.0000000000001p+52,
       {0x1.0000000000001p+52, 0x1.0000000000001p+52, 0x1.0000000000001p+52,
        0x1.0000000000001p+52, 0x1.0000000000001p+52}},
  };
  double (*const functions[5])(double) = {
      iree_math_ceil_f64,       iree_math_floor_f64,      iree_math_trunc_f64,
      iree_math_round_even_f64, iree_math_round_away_f64,
  };
  CheckRounding(cases, functions);
}

TEST_F(RoundingTest, CanonicalNaNsAndInfinities) {
  const F32Function single_functions[] = {
      iree_math_ceil_f32,       iree_math_floor_f32,      iree_math_trunc_f32,
      iree_math_round_even_f32, iree_math_round_away_f32,
  };
  const F64Function double_functions[] = {
      iree_math_ceil_f64,       iree_math_floor_f64,      iree_math_trunc_f64,
      iree_math_round_even_f64, iree_math_round_away_f64,
  };
  for (int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    for (auto function : single_functions) {
      for (uint32_t payload :
           {0x7F800001u, 0x7FC12345u, 0xFF800001u, 0xFFC12345u}) {
        ExpectF32Mapping(function, payload, 0x7FC00000u);
      }
      ExpectF32Mapping(function, 0x7F800000u, 0x7F800000u);
      ExpectF32Mapping(function, 0xFF800000u, 0xFF800000u);
    }
    for (auto function : double_functions) {
      for (uint64_t payload :
           {UINT64_C(0x7FF0000000000001), UINT64_C(0x7FF8000000012345),
            UINT64_C(0xFFF0000000000001), UINT64_C(0xFFF8000000012345)}) {
        ExpectF64Mapping(function, payload, UINT64_C(0x7FF8000000000000));
      }
      ExpectF64Mapping(function, UINT64_C(0x7FF0000000000000),
                       UINT64_C(0x7FF0000000000000));
      ExpectF64Mapping(function, UINT64_C(0xFFF0000000000000),
                       UINT64_C(0xFFF0000000000000));
    }
  }
}

}  // namespace
}  // namespace iree::math::testing
