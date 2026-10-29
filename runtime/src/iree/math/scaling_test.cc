// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/scaling.h"

#include <cmath>
#include <limits>

#include "iree/math/elementary_test_util.h"

namespace iree::math {
namespace {
using testing::ElementaryTest;
using testing::F32FromBits;
using testing::F32ToBits;
using testing::F64FromBits;
using testing::F64ToBits;

template <typename T>
struct DecompositionCase {
  // Nonnegative source value; both signs are checked.
  T input;
  // Exact normalized fraction before applying the input sign.
  T fraction;
  // Exact binary exponent, including zero for special values.
  int32_t exponent;
};

template <typename T, size_t N>
void CheckDecomposition(const DecompositionCase<T> (&cases)[N],
                        T (*function)(T, int32_t*)) {
  for (const auto& row : cases) {
    for (bool negative : {false, true}) {
      const T input = negative ? -row.input : row.input;
      const T expected = negative ? -row.fraction : row.fraction;
      SCOPED_TRACE(::testing::Message() << "input=" << input);
      int32_t exponent = INT32_MIN;
      const T actual = function(input, &exponent);
      EXPECT_EQ(expected, actual);
      EXPECT_EQ(std::signbit(expected), std::signbit(actual));
      EXPECT_EQ(row.exponent, exponent);
    }
  }
}

TEST_F(ElementaryTest, FrexpF32Boundaries) {
  const DecompositionCase<float> cases[] = {
      {0.0f, 0.0f, 0},
      {0x1p-149f, 0x1p-1f, -148},
      {0x1.8p-148f, 0x1.8p-1f, -147},
      {0x1.fffffcp-127f, 0x1.fffffcp-1f, -126},
      {0x1p-126f, 0x1p-1f, -125},
      {0x1.000002p-126f, 0x1.000002p-1f, -125},
      {0x1p-1f, 0x1p-1f, 0},
      {0x1.fffffep-1f, 0x1.fffffep-1f, 0},
      {0x1p0f, 0x1p-1f, 1},
      {0x1.8p0f, 0x1.8p-1f, 1},
      {0x1.fffffep127f, 0x1.fffffep-1f, 128},
      {std::numeric_limits<float>::infinity(),
       std::numeric_limits<float>::infinity(), 0},
  };
  CheckDecomposition(cases, iree_math_frexp_f32);
}

TEST_F(ElementaryTest, FrexpF64Boundaries) {
  const DecompositionCase<double> cases[] = {
      {0.0, 0.0, 0},
      {0x1p-1074, 0x1p-1, -1073},
      {0x1.8p-1073, 0x1.8p-1, -1072},
      {0x1.ffffffffffffep-1023, 0x1.ffffffffffffep-1, -1022},
      {0x1p-1022, 0x1p-1, -1021},
      {0x1.0000000000001p-1022, 0x1.0000000000001p-1, -1021},
      {0x1p-1, 0x1p-1, 0},
      {0x1.fffffffffffffp-1, 0x1.fffffffffffffp-1, 0},
      {0x1p0, 0x1p-1, 1},
      {0x1.8p0, 0x1.8p-1, 1},
      {0x1.fffffffffffffp1023, 0x1.fffffffffffffp-1, 1024},
      {std::numeric_limits<double>::infinity(),
       std::numeric_limits<double>::infinity(), 0},
  };
  CheckDecomposition(cases, iree_math_frexp_f64);
}

template <typename T>
struct ScalingCase {
  // Nonnegative source value; both signs are checked.
  T input;
  // Signed power of two applied to the source.
  int32_t exponent;
  // Independently rounded positive result.
  T expected;
};

template <typename T, size_t N>
void CheckScaling(const ScalingCase<T> (&cases)[N], T (*function)(T, int32_t)) {
  for (const auto& row : cases) {
    for (bool negative : {false, true}) {
      const T input = negative ? -row.input : row.input;
      const T expected = negative ? -row.expected : row.expected;
      SCOPED_TRACE(::testing::Message()
                   << "input=" << input << " exponent=" << row.exponent);
      const T actual = function(input, row.exponent);
      EXPECT_EQ(expected, actual);
      EXPECT_EQ(std::signbit(expected), std::signbit(actual));
    }
  }
}

TEST_F(ElementaryTest, LdexpF32Boundaries) {
  const float infinity = std::numeric_limits<float>::infinity();
  const ScalingCase<float> cases[] = {
      {0.0f, INT32_MIN, 0.0f},          {0.0f, INT32_MAX, 0.0f},
      {1.0f, INT32_MIN, 0.0f},          {1.0f, INT32_MAX, infinity},
      {infinity, INT32_MIN, infinity},  {infinity, INT32_MAX, infinity},
      {0x1p-149f, 149, 1.0f},           {0x1p-149f, 277, infinity},
      {0x1p0f, -149, 0x1p-149f},        {0x1p0f, -150, 0.0f},
      {0x1.000002p0f, -150, 0x1p-149f}, {0x1.8p0f, -149, 0x1p-148f},
      {0x1.4p1f, -149, 0x1p-148f},      {0x1.fffffcp0f, -127, 0x1.fffffcp-127f},
      {0x1.fffffep0f, -127, 0x1p-126f}, {0x1.fffffep127f, -1, 0x1.fffffep126f},
      {0x1.fffffep127f, 1, infinity},
  };
  CheckScaling(cases, iree_math_ldexp_f32);
}

TEST_F(ElementaryTest, LdexpF64Boundaries) {
  const double infinity = std::numeric_limits<double>::infinity();
  const ScalingCase<double> cases[] = {
      {0.0, INT32_MIN, 0.0},
      {0.0, INT32_MAX, 0.0},
      {1.0, INT32_MIN, 0.0},
      {1.0, INT32_MAX, infinity},
      {infinity, INT32_MIN, infinity},
      {infinity, INT32_MAX, infinity},
      {0x1p-1074, 1074, 1.0},
      {0x1p-1074, 2098, infinity},
      {0x1p0, -1074, 0x1p-1074},
      {0x1p0, -1075, 0.0},
      {0x1.0000000000001p0, -1075, 0x1p-1074},
      {0x1.8p0, -1074, 0x1p-1073},
      {0x1.4p1, -1074, 0x1p-1073},
      {0x1.ffffffffffffep0, -1023, 0x1.ffffffffffffep-1023},
      {0x1.fffffffffffffp0, -1023, 0x1p-1022},
      {0x1.fffffffffffffp1023, -1, 0x1.fffffffffffffp1022},
      {0x1.fffffffffffffp1023, 1, infinity},
  };
  CheckScaling(cases, iree_math_ldexp_f64);
}

TEST_F(ElementaryTest, ScalingCanonicalNaNs) {
  for (uint32_t bits : {0x7f800001u, 0xff800001u, 0x7fc12345u, 0xffc12345u}) {
    int32_t exponent = INT32_MIN;
    EXPECT_EQ(0x7fc00000u,
              F32ToBits(iree_math_frexp_f32(F32FromBits(bits), &exponent)));
    EXPECT_EQ(0, exponent);
    for (int32_t scale : {INT32_MIN, 0, INT32_MAX}) {
      EXPECT_EQ(0x7fc00000u,
                F32ToBits(iree_math_ldexp_f32(F32FromBits(bits), scale)));
    }
  }
  for (uint64_t bits :
       {UINT64_C(0x7ff0000000000001), UINT64_C(0xfff0000000000001),
        UINT64_C(0x7ff8123456789abc), UINT64_C(0xfff8123456789abc)}) {
    int32_t exponent = INT32_MIN;
    EXPECT_EQ(UINT64_C(0x7ff8000000000000),
              F64ToBits(iree_math_frexp_f64(F64FromBits(bits), &exponent)));
    EXPECT_EQ(0, exponent);
    for (int32_t scale : {INT32_MIN, 0, INT32_MAX}) {
      EXPECT_EQ(UINT64_C(0x7ff8000000000000),
                F64ToBits(iree_math_ldexp_f64(F64FromBits(bits), scale)));
    }
  }
}
}  // namespace
}  // namespace iree::math
