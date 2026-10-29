// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/activation.h"

#include <cmath>
#include <limits>

#include "iree/math/elementary_test_util.h"

namespace iree::math::testing {
namespace {

enum class Activation {
  kLogistic,
  kSilu,
  kSoftplus,
  kGeluErf,
  kGeluTanh,
  kGeluLogistic
};

template <typename Float>
Float Evaluate(Activation operation, Float input, Float scale);

template <>
float Evaluate(Activation operation, float input, float scale) {
  const F32Function unary[] = {iree_math_logistic_f32, iree_math_silu_f32,
                               iree_math_softplus_f32, iree_math_gelu_erf_f32,
                               iree_math_gelu_tanh_f32};
  return operation == Activation::kGeluLogistic
             ? iree_math_gelu_logistic_f32(input, scale)
             : unary[static_cast<unsigned>(operation)](input);
}

template <>
double Evaluate(Activation operation, double input, double scale) {
  const F64Function unary[] = {iree_math_logistic_f64, iree_math_silu_f64,
                               iree_math_softplus_f64, iree_math_gelu_erf_f64,
                               iree_math_gelu_tanh_f64};
  return operation == Activation::kGeluLogistic
             ? iree_math_gelu_logistic_f64(input, scale)
             : unary[static_cast<unsigned>(operation)](input);
}

template <typename Float>
struct ActivationBoundary {
  // Authored activation formula.
  Activation operation;
  // Input, including adjacent range-transition values.
  Float input;
  // Selected-width scale for logistic GELU; ignored by unary formulas.
  Float scale;
  // Independently rounded MPFR value of the real formula.
  Float expected;
};

template <typename Float, size_t Count>
void CheckBoundaries(const ActivationBoundary<Float> (&cases)[Count]) {
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message() << static_cast<unsigned>(test.operation)
                                      << " input=" << std::hexfloat
                                      << test.input << " scale=" << test.scale);
    const Float actual = Evaluate(test.operation, test.input, test.scale);
    if (test.expected == 0 || std::isinf(test.expected)) {
      EXPECT_EQ(actual, test.expected);
      EXPECT_EQ(std::signbit(actual), std::signbit(test.expected));
      continue;
    }
    const double input = test.input;
    double condition = 1.0;
    if (test.operation == Activation::kGeluErf) {
      condition = 1.0 + input * input;
    }
    if (test.operation == Activation::kGeluTanh) {
      condition =
          2.0 * (1.0 + std::abs(2.0 * 0.79788456080286535588 *
                                (input + 0.044715 * input * input * input)));
    }
    if (test.operation == Activation::kGeluLogistic) {
      condition = 1.0 + std::abs(static_cast<double>(test.scale) * input);
    }
    const double allowance = 4.0 * std::numeric_limits<Float>::epsilon() *
                                 condition *
                                 std::abs(static_cast<double>(test.expected)) +
                             std::numeric_limits<Float>::denorm_min();
    EXPECT_TRUE(std::isfinite(actual));
    EXPECT_LE(std::abs(static_cast<double>(actual) - test.expected), allowance);
  }
}

// MPFR 4.2.2 directed intervals (256-4096 bits), rounded independently.
// These boundary fixtures cover cancellation, scaled underflow, and the range
// transitions. Accuracy uses the declared formula conditioning, not a tolerance
// derived from agreement with another floating-point implementation.

TEST_F(ElementaryTest, ActivationF32Boundaries) {
  const ActivationBoundary<float> cases[] = {
      {Activation::kLogistic, -0x1p-1f, 0x1p+0f, 0x1.829a06p-2f},
      {Activation::kLogistic, 0x1p-1f, 0x1p+0f, 0x1.3eb2fep-1f},
      {Activation::kLogistic, -0x1p+0f, 0x1p+0f, 0x1.136562p-2f},
      {Activation::kLogistic, 0x1p+0f, 0x1p+0f, 0x1.764d5p-1f},
      {Activation::kLogistic, 0x1p-149f, 0x1p+0f, 0x1p-1f},
      {Activation::kLogistic, -0x1p-149f, 0x1p+0f, 0x1p-1f},
      {Activation::kLogistic, 0x1.fffffcp-127f, 0x1p+0f, 0x1p-1f},
      {Activation::kLogistic, -0x1p-126f, 0x1p+0f, 0x1p-1f},
      {Activation::kLogistic, -0x1.4p+6f, 0x1p+0f, 0x1.7fd974p-116f},
      {Activation::kLogistic, -0x1.9p+6f, 0x1p+0f, 0x1.bp-145f},
      {Activation::kLogistic, -0x1.ap+6f, 0x1p+0f, 0x0p+0f},
      {Activation::kLogistic, -0x1.bp+6f, 0x1p+0f, 0x0p+0f},
      {Activation::kSilu, -0x1p-1f, 0x1p+0f, -0x1.829a06p-3f},
      {Activation::kSilu, 0x1p-1f, 0x1p+0f, 0x1.3eb2fep-2f},
      {Activation::kSilu, -0x1p+0f, 0x1p+0f, -0x1.136562p-2f},
      {Activation::kSilu, 0x1p+0f, 0x1p+0f, 0x1.764d5p-1f},
      {Activation::kSilu, 0x1p-149f, 0x1p+0f, 0x1p-149f},
      {Activation::kSilu, -0x1p-149f, 0x1p+0f, -0x0p+0f},
      {Activation::kSilu, 0x1.fffffcp-127f, 0x1p+0f, 0x1p-127f},
      {Activation::kSilu, -0x1p-126f, 0x1p+0f, -0x1p-127f},
      {Activation::kSilu, -0x1.4p+6f, 0x1p+0f, -0x1.dfcfd2p-110f},
      {Activation::kSilu, -0x1.9p+6f, 0x1p+0f, -0x1.4bep-138f},
      {Activation::kSilu, -0x1.ap+6f, 0x1p+0f, -0x1.98p-144f},
      {Activation::kSilu, -0x1.bp+6f, 0x1p+0f, -0x1p-149f},
      {Activation::kSoftplus, -0x1p-1f, 0x1p+0f, 0x1.e5747p-2f},
      {Activation::kSoftplus, 0x1p-1f, 0x1p+0f, 0x1.f2ba38p-1f},
      {Activation::kSoftplus, -0x1p+0f, 0x1p+0f, 0x1.40c7acp-2f},
      {Activation::kSoftplus, 0x1p+0f, 0x1p+0f, 0x1.5031eap+0f},
      {Activation::kSoftplus, 0x1p-149f, 0x1p+0f, 0x1.62e43p-1f},
      {Activation::kSoftplus, -0x1p-149f, 0x1p+0f, 0x1.62e43p-1f},
      {Activation::kSoftplus, 0x1.fffffcp-127f, 0x1p+0f, 0x1.62e43p-1f},
      {Activation::kSoftplus, -0x1p-126f, 0x1p+0f, 0x1.62e43p-1f},
      {Activation::kSoftplus, -0x1.4p+6f, 0x1p+0f, 0x1.7fd974p-116f},
      {Activation::kSoftplus, -0x1.9p+6f, 0x1p+0f, 0x1.bp-145f},
      {Activation::kSoftplus, -0x1.ap+6f, 0x1p+0f, 0x0p+0f},
      {Activation::kSoftplus, -0x1.bp+6f, 0x1p+0f, 0x0p+0f},
      {Activation::kGeluErf, -0x1p-1f, 0x1p+0f, -0x1.3bf144p-3f},
      {Activation::kGeluErf, 0x1p-1f, 0x1p+0f, 0x1.62075ep-2f},
      {Activation::kGeluErf, -0x1p+0f, 0x1p+0f, -0x1.44ed0cp-3f},
      {Activation::kGeluErf, 0x1p+0f, 0x1p+0f, 0x1.aec4bep-1f},
      {Activation::kGeluErf, 0x1p-149f, 0x1p+0f, 0x1p-149f},
      {Activation::kGeluErf, -0x1p-149f, 0x1p+0f, -0x0p+0f},
      {Activation::kGeluErf, 0x1.fffffcp-127f, 0x1p+0f, 0x1p-127f},
      {Activation::kGeluErf, -0x1p-126f, 0x1p+0f, -0x1p-127f},
      {Activation::kGeluErf, -0x1.fffffep+2f, 0x1p+0f, -0x1.669d86p-48f},
      {Activation::kGeluErf, -0x1p+3f, 0x1p+0f, -0x1.669d2cp-48f},
      {Activation::kGeluErf, -0x1.000002p+3f, 0x1p+0f, -0x1.669c7ap-48f},
      {Activation::kGeluErf, -0x1.fffffep+3f, 0x1p+0f, -0x0p+0f},
      {Activation::kGeluErf, -0x1p+4f, 0x1p+0f, -0x0p+0f},
      {Activation::kGeluErf, -0x1.000002p+4f, 0x1p+0f, -0x0p+0f},
      {Activation::kGeluErf, 0x1.7ffffep+2f, 0x1p+0f, 0x1.7ffffep+2f},
      {Activation::kGeluErf, 0x1.8p+2f, 0x1p+0f, 0x1.8p+2f},
      {Activation::kGeluErf, 0x1.800002p+2f, 0x1p+0f, 0x1.800002p+2f},
      {Activation::kGeluErf, -0x1.8p+2f, 0x1p+0f, -0x1.96c966p-28f},
      {Activation::kGeluErf, -0x1.8p+3f, 0x1p+0f, -0x1.bac0cap-106f},
      {Activation::kGeluTanh, -0x1p-1f, 0x1p+0f, -0x1.3bfa4cp-3f},
      {Activation::kGeluTanh, 0x1p-1f, 0x1p+0f, 0x1.6202dap-2f},
      {Activation::kGeluTanh, -0x1p+0f, 0x1p+0f, -0x1.453d22p-3f},
      {Activation::kGeluTanh, 0x1p+0f, 0x1p+0f, 0x1.aeb0b8p-1f},
      {Activation::kGeluTanh, 0x1p-149f, 0x1p+0f, 0x1p-149f},
      {Activation::kGeluTanh, -0x1p-149f, 0x1p+0f, -0x0p+0f},
      {Activation::kGeluTanh, 0x1.fffffcp-127f, 0x1p+0f, 0x1p-127f},
      {Activation::kGeluTanh, -0x1p-126f, 0x1p+0f, -0x1p-127f},
      {Activation::kGeluTanh, -0x1p+2f, 0x1p+0f, -0x1.26a204p-14f},
      {Activation::kGeluTanh, -0x1p+3f, 0x1p+0f, -0x1.d5a288p-69f},
      {Activation::kGeluTanh, -0x1.4p+3f, 0x1p+0f, -0x1.47c8fep-123f},
      {Activation::kGeluTanh, -0x1.8p+3f, 0x1p+0f, -0x0p+0f},
      {Activation::kGeluLogistic, -0x1p-1f, 0x1.b3b646p+0f, -0x1.32678cp-3f},
      {Activation::kGeluLogistic, 0x1p-1f, 0x1.b3b646p+0f, 0x1.66cc3ap-2f},
      {Activation::kGeluLogistic, -0x1p+0f, 0x1.b3b646p+0f, -0x1.3bcf6ep-3f},
      {Activation::kGeluLogistic, 0x1p+0f, 0x1.b3b646p+0f, 0x1.b10c24p-1f},
      {Activation::kGeluLogistic, 0x1p-149f, 0x1.b3b646p+0f, 0x1p-149f},
      {Activation::kGeluLogistic, -0x1p-149f, 0x1.b3b646p+0f, -0x0p+0f},
      {Activation::kGeluLogistic, 0x1.fffffcp-127f, 0x1.b3b646p+0f, 0x1p-127f},
      {Activation::kGeluLogistic, -0x1p-126f, 0x1.b3b646p+0f, -0x1p-127f},
      {Activation::kGeluLogistic, -0x1p+3f, 0x1.b3b646p+0f, -0x1.47b4f8p-17f},
      {Activation::kGeluLogistic, -0x1.9p+6f, 0x1.b3b646p+0f, -0x0p+0f},
      {Activation::kGeluLogistic, -0x1p+3f, 0x0p+0f, -0x1p+2f},
      {Activation::kGeluLogistic, -0x1p+3f, 0x1p-3f, -0x1.136562p+1f},
      {Activation::kGeluLogistic, -0x1p+3f, -0x1p+1f, -0x1.fffffcp+2f},
      {Activation::kGeluLogistic, 0x1.8p-1f, -0x1p+1f, 0x1.1834a2p-3f},
      {Activation::kGeluLogistic, -0x1.fffffep+127f, 0x1p-126f,
       -0x1.26afa6p+122f},
  };
  CheckBoundaries(cases);
}

TEST_F(ElementaryTest, ActivationF64Boundaries) {
  const ActivationBoundary<double> cases[] = {
      {Activation::kLogistic, -0x1p-1, 0x1p+0, 0x1.829a0565978dfp-2},
      {Activation::kLogistic, 0x1p-1, 0x1p+0, 0x1.3eb2fd4d34391p-1},
      {Activation::kLogistic, -0x1p+0, 0x1p+0, 0x1.136561454ba86p-2},
      {Activation::kLogistic, 0x1p+0, 0x1p+0, 0x1.764d4f5d5a2bdp-1},
      {Activation::kLogistic, 0x0.0000000000001p-1022, 0x1p+0, 0x1p-1},
      {Activation::kLogistic, -0x0.0000000000001p-1022, 0x1p+0, 0x1p-1},
      {Activation::kLogistic, 0x0.fffffffffffffp-1022, 0x1p+0, 0x1p-1},
      {Activation::kLogistic, -0x1p-1022, 0x1p+0, 0x1p-1},
      {Activation::kLogistic, -0x1.5ep+9, 0x1p+0, 0x1.14f2b0fb9307fp-1010},
      {Activation::kLogistic, -0x1.74p+9, 0x1p+0, 0x0.0000000000002p-1022},
      {Activation::kLogistic, -0x1.748p+9, 0x1p+0, 0x0.0000000000001p-1022},
      {Activation::kLogistic, -0x1.77p+9, 0x1p+0, 0x0p+0},
      {Activation::kLogistic, -0x1.79p+9, 0x1p+0, 0x0p+0},
      {Activation::kSilu, -0x1p-1, 0x1p+0, -0x1.829a0565978dfp-3},
      {Activation::kSilu, 0x1p-1, 0x1p+0, 0x1.3eb2fd4d34391p-2},
      {Activation::kSilu, -0x1p+0, 0x1p+0, -0x1.136561454ba86p-2},
      {Activation::kSilu, 0x1p+0, 0x1p+0, 0x1.764d4f5d5a2bdp-1},
      {Activation::kSilu, 0x0.0000000000001p-1022, 0x1p+0,
       0x0.0000000000001p-1022},
      {Activation::kSilu, -0x0.0000000000001p-1022, 0x1p+0, -0x0p+0},
      {Activation::kSilu, 0x0.fffffffffffffp-1022, 0x1p+0, 0x0.8p-1022},
      {Activation::kSilu, -0x1p-1022, 0x1p+0, -0x0.8p-1022},
      {Activation::kSilu, -0x1.5ep+9, 0x1p+0, -0x1.7aa3cdf7f304ep-1001},
      {Activation::kSilu, -0x1.74p+9, 0x1p+0, -0x0.0000000000483p-1022},
      {Activation::kSilu, -0x1.748p+9, 0x1p+0, -0x0.00000000001aap-1022},
      {Activation::kSilu, -0x1.77p+9, 0x1p+0, -0x0.0000000000003p-1022},
      {Activation::kSilu, -0x1.79p+9, 0x1p+0, -0x0p+0},
      {Activation::kSoftplus, -0x1p-1, 0x1p+0, 0x1.e5746fdb5c064p-2},
      {Activation::kSoftplus, 0x1p-1, 0x1p+0, 0x1.f2ba37edae032p-1},
      {Activation::kSoftplus, -0x1p+0, 0x1p+0, 0x1.40c7abfbec124p-2},
      {Activation::kSoftplus, 0x1p+0, 0x1p+0, 0x1.5031eafefb049p+0},
      {Activation::kSoftplus, 0x0.0000000000001p-1022, 0x1p+0,
       0x1.62e42fefa39efp-1},
      {Activation::kSoftplus, -0x0.0000000000001p-1022, 0x1p+0,
       0x1.62e42fefa39efp-1},
      {Activation::kSoftplus, 0x0.fffffffffffffp-1022, 0x1p+0,
       0x1.62e42fefa39efp-1},
      {Activation::kSoftplus, -0x1p-1022, 0x1p+0, 0x1.62e42fefa39efp-1},
      {Activation::kSoftplus, -0x1.5ep+9, 0x1p+0, 0x1.14f2b0fb9307fp-1010},
      {Activation::kSoftplus, -0x1.74p+9, 0x1p+0, 0x0.0000000000002p-1022},
      {Activation::kSoftplus, -0x1.748p+9, 0x1p+0, 0x0.0000000000001p-1022},
      {Activation::kSoftplus, -0x1.77p+9, 0x1p+0, 0x0p+0},
      {Activation::kSoftplus, -0x1.79p+9, 0x1p+0, 0x0p+0},
      {Activation::kGeluErf, -0x1p-1, 0x1p+0, -0x1.3bf143b9aa712p-3},
      {Activation::kGeluErf, 0x1p-1, 0x1p+0, 0x1.62075e232ac77p-2},
      {Activation::kGeluErf, -0x1p+0, 0x1p+0, -0x1.44ed0bb7cb20bp-3},
      {Activation::kGeluErf, 0x1p+0, 0x1p+0, 0x1.aec4bd120d37dp-1},
      {Activation::kGeluErf, 0x0.0000000000001p-1022, 0x1p+0,
       0x0.0000000000001p-1022},
      {Activation::kGeluErf, -0x0.0000000000001p-1022, 0x1p+0, -0x0p+0},
      {Activation::kGeluErf, 0x0.fffffffffffffp-1022, 0x1p+0, 0x0.8p-1022},
      {Activation::kGeluErf, -0x1p-1022, 0x1p+0, -0x0.8p-1022},
      {Activation::kGeluErf, -0x1.1ffffffffffffp+5, 0x1p+0,
       -0x1.bfd0452322a43p-937},
      {Activation::kGeluErf, -0x1.2p+5, 0x1p+0, -0x1.bfd0452322264p-937},
      {Activation::kGeluErf, -0x1.2000000000001p+5, 0x1p+0,
       -0x1.bfd0452321a85p-937},
      {Activation::kGeluErf, -0x1.3ffffffffffffp+5, 0x1p+0, -0x0p+0},
      {Activation::kGeluErf, -0x1.4p+5, 0x1p+0, -0x0p+0},
      {Activation::kGeluErf, -0x1.4000000000001p+5, 0x1p+0, -0x0p+0},
      {Activation::kGeluErf, 0x1.1ffffffffffffp+3, 0x1p+0,
       0x1.1ffffffffffffp+3},
      {Activation::kGeluErf, 0x1.2p+3, 0x1p+0, 0x1.2p+3},
      {Activation::kGeluErf, 0x1.2000000000001p+3, 0x1p+0,
       0x1.2000000000001p+3},
      {Activation::kGeluErf, -0x1.8p+2, 0x1p+0, -0x1.96c96680dc3ecp-28},
      {Activation::kGeluErf, -0x1.3p+5, 0x1p+0, -0x0.0000084474d57p-1022},
      {Activation::kGeluTanh, -0x1p-1, 0x1p+0, -0x1.3bfa4b104180fp-3},
      {Activation::kGeluTanh, 0x1p-1, 0x1p+0, 0x1.6202da77df3f9p-2},
      {Activation::kGeluTanh, -0x1p+0, 0x1p+0, -0x1.453d22357147ap-3},
      {Activation::kGeluTanh, 0x1p+0, 0x1p+0, 0x1.aeb0b772a3ae1p-1},
      {Activation::kGeluTanh, 0x0.0000000000001p-1022, 0x1p+0,
       0x0.0000000000001p-1022},
      {Activation::kGeluTanh, -0x0.0000000000001p-1022, 0x1p+0, -0x0p+0},
      {Activation::kGeluTanh, 0x0.fffffffffffffp-1022, 0x1p+0, 0x0.8p-1022},
      {Activation::kGeluTanh, -0x1p-1022, 0x1p+0, -0x0.8p-1022},
      {Activation::kGeluTanh, -0x1p+2, 0x1p+0, -0x1.26a20335ddd3cp-14},
      {Activation::kGeluTanh, -0x1.4p+4, 0x1p+0, -0x1.a927dc6157cbp-866},
      {Activation::kGeluTanh, -0x1p+5, 0x1p+0, -0x0p+0},
      {Activation::kGeluLogistic, -0x1p-1, 0x1.b3b645a1cac08p+0,
       -0x1.32678c1f083ffp-3},
      {Activation::kGeluLogistic, 0x1p-1, 0x1.b3b645a1cac08p+0,
       0x1.66cc39f07be01p-2},
      {Activation::kGeluLogistic, -0x1p+0, 0x1.b3b645a1cac08p+0,
       -0x1.3bcf6df1c93d6p-3},
      {Activation::kGeluLogistic, 0x1p+0, 0x1.b3b645a1cac08p+0,
       0x1.b10c24838db0ap-1},
      {Activation::kGeluLogistic, 0x0.0000000000001p-1022, 0x1.b3b645a1cac08p+0,
       0x0.0000000000001p-1022},
      {Activation::kGeluLogistic, -0x0.0000000000001p-1022,
       0x1.b3b645a1cac08p+0, -0x0p+0},
      {Activation::kGeluLogistic, 0x0.fffffffffffffp-1022, 0x1.b3b645a1cac08p+0,
       0x0.8p-1022},
      {Activation::kGeluLogistic, -0x1p-1022, 0x1.b3b645a1cac08p+0,
       -0x0.8p-1022},
      {Activation::kGeluLogistic, -0x1p+3, 0x1.b3b645a1cac08p+0,
       -0x1.47b4fc330e33ap-17},
      {Activation::kGeluLogistic, -0x1.74p+9, 0x1.b3b645a1cac08p+0, -0x0p+0},
      {Activation::kGeluLogistic, -0x1p+3, 0x0p+0, -0x1p+2},
      {Activation::kGeluLogistic, -0x1p+3, 0x1p-3, -0x1.136561454ba86p+1},
      {Activation::kGeluLogistic, -0x1p+3, -0x1p+1, -0x1.fffffc39548fcp+2},
      {Activation::kGeluLogistic, 0x1.8p-1, -0x1p+1, 0x1.1834a280394c4p-3},
      {Activation::kGeluLogistic, -0x1.fffffffffffffp+1023, 0x1p-1022,
       -0x1.26afa1e43c2c4p+1018},
  };
  CheckBoundaries(cases);
}

TEST_F(ElementaryTest, ActivationF32ExceptionalMappings) {
  const float infinity = std::numeric_limits<float>::infinity();
  for (Activation operation :
       {Activation::kLogistic, Activation::kSilu, Activation::kSoftplus,
        Activation::kGeluErf, Activation::kGeluTanh,
        Activation::kGeluLogistic}) {
    SCOPED_TRACE(static_cast<unsigned>(operation));
    for (uint32_t nan : {UINT32_C(0x7F800001), UINT32_C(0xFFC00019)}) {
      EXPECT_EQ(F32ToBits(Evaluate(operation, F32FromBits(nan),
                                   0x1.b3b645a1cac08p+0f)),
                UINT32_C(0x7FC00000));
    }
    EXPECT_EQ(Evaluate(operation, infinity, 0x1.b3b645a1cac08p+0f),
              operation == Activation::kLogistic ? 0x1p+0f : infinity);
    const float negative_limit =
        Evaluate(operation, -infinity, 0x1.b3b645a1cac08p+0f);
    EXPECT_EQ(negative_limit, 0x0p+0f);
    EXPECT_EQ(std::signbit(negative_limit),
              operation != Activation::kLogistic &&
                  operation != Activation::kSoftplus);
    if (operation != Activation::kLogistic &&
        operation != Activation::kSoftplus) {
      EXPECT_EQ(F32ToBits(Evaluate(operation, -0x0p+0f, 0x1.b3b645a1cac08p+0f)),
                UINT32_C(0x80000000));
    }
  }
  for (float zero : {0x0p+0f, -0x0p+0f}) {
    EXPECT_EQ(F32ToBits(iree_math_gelu_logistic_f32(zero, infinity)),
              UINT32_C(0x7FC00000));
    EXPECT_EQ(F32ToBits(iree_math_gelu_logistic_f32(infinity, zero)),
              UINT32_C(0x7FC00000));
  }
  EXPECT_EQ(iree_math_gelu_logistic_f32(-infinity, -0x1p+1f), -infinity);
  EXPECT_EQ(iree_math_gelu_logistic_f32(-0x1p+3f, -infinity), -0x1p+3f);
  for (uint32_t nan : {UINT32_C(0x7F800001), UINT32_C(0xFFC00019)}) {
    EXPECT_EQ(F32ToBits(iree_math_gelu_logistic_f32(-1.0f, F32FromBits(nan))),
              UINT32_C(0x7FC00000));
  }
}

TEST_F(ElementaryTest, ActivationF64ExceptionalMappings) {
  const double infinity = std::numeric_limits<double>::infinity();
  for (Activation operation :
       {Activation::kLogistic, Activation::kSilu, Activation::kSoftplus,
        Activation::kGeluErf, Activation::kGeluTanh,
        Activation::kGeluLogistic}) {
    SCOPED_TRACE(static_cast<unsigned>(operation));
    for (uint64_t nan :
         {UINT64_C(0x7FF0000000000001), UINT64_C(0xFFF8000000000019)}) {
      EXPECT_EQ(F64ToBits(Evaluate(operation, F64FromBits(nan),
                                   0x1.b3b645a1cac08p+0)),
                UINT64_C(0x7FF8000000000000));
    }
    EXPECT_EQ(Evaluate(operation, infinity, 0x1.b3b645a1cac08p+0),
              operation == Activation::kLogistic ? 0x1p+0 : infinity);
    const double negative_limit =
        Evaluate(operation, -infinity, 0x1.b3b645a1cac08p+0);
    EXPECT_EQ(negative_limit, 0x0p+0);
    EXPECT_EQ(std::signbit(negative_limit),
              operation != Activation::kLogistic &&
                  operation != Activation::kSoftplus);
    if (operation != Activation::kLogistic &&
        operation != Activation::kSoftplus) {
      EXPECT_EQ(F64ToBits(Evaluate(operation, -0x0p+0, 0x1.b3b645a1cac08p+0)),
                UINT64_C(0x8000000000000000));
    }
  }
  for (double zero : {0x0p+0, -0x0p+0}) {
    EXPECT_EQ(F64ToBits(iree_math_gelu_logistic_f64(zero, infinity)),
              UINT64_C(0x7FF8000000000000));
    EXPECT_EQ(F64ToBits(iree_math_gelu_logistic_f64(infinity, zero)),
              UINT64_C(0x7FF8000000000000));
  }
  EXPECT_EQ(iree_math_gelu_logistic_f64(-infinity, -0x1p+1), -infinity);
  EXPECT_EQ(iree_math_gelu_logistic_f64(-0x1p+3, -infinity), -0x1p+3);
  for (uint64_t nan :
       {UINT64_C(0x7FF0000000000001), UINT64_C(0xFFF8000000000019)}) {
    EXPECT_EQ(F64ToBits(iree_math_gelu_logistic_f64(-1.0, F64FromBits(nan))),
              UINT64_C(0x7FF8000000000000));
  }
}

}  // namespace
}  // namespace iree::math::testing
