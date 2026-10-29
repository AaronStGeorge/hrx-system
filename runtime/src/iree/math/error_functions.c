// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/error_functions.h"

#include <stdbool.h>

#include "iree/math/erfc_tail.h"
#include "iree/math/exponential.h"
#include "iree/math/exponential_reduction.h"
#include "iree/math/float_bits.h"

// Adapted from OpenLibm revision 5fe399749f9276eaa0b8403e507470da05cbbb3f:
// s_erf.c and s_erff.c. F32 conversion by Ian Lance Taylor.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

// Rational approximations retain the upstream coefficients and rounding order.
// F32 erf evaluates its central regions in F32. F32 erfc retains F64
// intermediates there: reconstructing a small complement otherwise magnifies
// the rounded erf correction. Both share the F64 evaluation of the tail.
static float iree_math_erf_small_f32(float square) {
  const float numerator =
      1.28379166e-01f + square * (-3.36030394e-01f + square * -1.86260219e-03f);
  const float denominator =
      1.0f + square * (3.12324286e-01f +
                       square * (2.16070302e-02f + square * -1.98859419e-03f));
  return numerator / denominator;
}

static float iree_math_erf_near_one_f32(float residual) {
  const float numerator =
      3.64939137e-06f +
      residual * (4.15109694e-01f +
                  residual * (-1.65179938e-01f + residual * 1.10914491e-01f));
  const float denominator =
      1.0f + residual * (6.02074385e-01f +
                         residual * (5.35934687e-01f +
                                     residual * (1.68576106e-01f +
                                                 residual * 5.62181212e-02f)));
  return numerator / denominator;
}

static double iree_math_erf_small_f32_widened(double square) {
  const double numerator =
      1.28379166e-01f + square * (-3.36030394e-01f + square * -1.86260219e-03f);
  const double denominator =
      1.0f + square * (3.12324286e-01f +
                       square * (2.16070302e-02f + square * -1.98859419e-03f));
  return numerator / denominator;
}

static double iree_math_erf_near_one_f32_widened(double residual) {
  const double numerator =
      3.64939137e-06f +
      residual * (4.15109694e-01f +
                  residual * (-1.65179938e-01f + residual * 1.10914491e-01f));
  const double denominator =
      1.0f + residual * (6.02074385e-01f +
                         residual * (5.35934687e-01f +
                                     residual * (1.68576106e-01f +
                                                 residual * 5.62181212e-02f)));
  return numerator / denominator;
}

static double iree_math_erf_small_f64(double square) {
  const double numerator =
      1.28379167095512558561e-01 +
      square * (-3.25042107247001499370e-01 +
                square * (-2.84817495755985104766e-02 +
                          square * (-5.77027029648944159157e-03 +
                                    square * -2.37630166566501626084e-05)));
  const double denominator =
      1.0 +
      square *
          (3.97917223959155352819e-01 +
           square *
               (6.50222499887672944485e-02 +
                square * (5.08130628187576562776e-03 +
                          square * (1.32494738004321644526e-04 +
                                    square * -3.96022827877536812320e-06))));
  return numerator / denominator;
}

// Positive complementary tail for 1.25 <= value < 11.
// The two rational regions approximate log(value*erfc(value)) + value^2 + 9/16.
static double iree_math_erfc_tail_f32(double value) {
  const double reciprocal_square = 1.0 / (value * value);
  double numerator;
  double denominator;
  if (value < 0x1.6db6dap+1) {
    numerator = -1.43268085e+00f;
    numerator = -2.17589188e+00f + reciprocal_square * numerator;
    numerator = -5.53605914e-01f + reciprocal_square * numerator;
    numerator = -9.87132732e-03f + reciprocal_square * numerator;
    denominator = -5.77397496e-02f;
    denominator = 1.43113089e+00f + reciprocal_square * denominator;
    denominator = 6.69798088e+00f + reciprocal_square * denominator;
    denominator = 5.45995426e+00f + reciprocal_square * denominator;
    denominator = 1.0 + reciprocal_square * denominator;
  } else {
    numerator = -9.53764343e+00f;
    numerator = -1.66696873e+01f + reciprocal_square * numerator;
    numerator = -6.16498327e+00f + reciprocal_square * numerator;
    numerator = -6.25171244e-01f + reciprocal_square * numerator;
    numerator = -9.86494310e-03f + reciprocal_square * numerator;
    denominator = 8.93033314e+00f;
    denominator = 4.72810211e+01f + reciprocal_square * denominator;
    denominator = 4.51839523e+01f + reciprocal_square * denominator;
    denominator = 1.26884899e+01f + reciprocal_square * denominator;
    denominator = 1.0 + reciprocal_square * denominator;
  }
  // A binary32 input's square is exact in binary64. One exponential and one
  // final binary32 rounding avoid the original two-exponential composition.
  return iree_math_exp_f64(-value * value - 0.5625 + numerator / denominator) /
         value;
}

typedef struct iree_math_erfc_exponent_t {
  // Exactly formed high part of log(value*erfc(value)).
  double high;
  // Square residual and rational tail correction.
  double low;
} iree_math_erfc_exponent_t;

// The two rational regions approximate log(value*erfc(value)) + value^2 + 9/16
// for value >= 1.25. Shared reduction keeps ordinary and scaled tails on the
// same coefficients without rounding their exponential reconstruction early.
static iree_math_erfc_exponent_t iree_math_erfc_tail_reduce_f64(double value) {
  const double reciprocal_square = 1.0 / (value * value);
  double numerator;
  double denominator;
  if (value < 0x1.6db6d00000000p+1) {
    numerator = -9.81432934416914548592e+00;
    numerator = -8.12874355063065934246e+01 + reciprocal_square * numerator;
    numerator = -1.84605092906711035994e+02 + reciprocal_square * numerator;
    numerator = -1.62396669462573470355e+02 + reciprocal_square * numerator;
    numerator = -6.23753324503260060396e+01 + reciprocal_square * numerator;
    numerator = -1.05586262253232909814e+01 + reciprocal_square * numerator;
    numerator = -6.93858572707181764372e-01 + reciprocal_square * numerator;
    numerator = -9.86494403484714822705e-03 + reciprocal_square * numerator;
    denominator = -6.04244152148580987438e-02;
    denominator = 6.57024977031928170135e+00 + reciprocal_square * denominator;
    denominator = 1.08635005541779435134e+02 + reciprocal_square * denominator;
    denominator = 4.29008140027567833386e+02 + reciprocal_square * denominator;
    denominator = 6.45387271733267880336e+02 + reciprocal_square * denominator;
    denominator = 4.34565877475229228821e+02 + reciprocal_square * denominator;
    denominator = 1.37657754143519042600e+02 + reciprocal_square * denominator;
    denominator = 1.96512716674392571292e+01 + reciprocal_square * denominator;
    denominator = 1.0 + reciprocal_square * denominator;
  } else {
    numerator = -4.83519191608651397019e+02;
    numerator = -1.02509513161107724954e+03 + reciprocal_square * numerator;
    numerator = -6.37566443368389627722e+02 + reciprocal_square * numerator;
    numerator = -1.60636384855821916062e+02 + reciprocal_square * numerator;
    numerator = -1.77579549177547519889e+01 + reciprocal_square * numerator;
    numerator = -7.99283237680523006574e-01 + reciprocal_square * numerator;
    numerator = -9.86494292470009928597e-03 + reciprocal_square * numerator;
    denominator = -2.24409524465858183362e+01;
    denominator = 4.74528541206955367215e+02 + reciprocal_square * denominator;
    denominator = 2.55305040643316442583e+03 + reciprocal_square * denominator;
    denominator = 3.19985821950859553908e+03 + reciprocal_square * denominator;
    denominator = 1.53672958608443695994e+03 + reciprocal_square * denominator;
    denominator = 3.25792512996573918826e+02 + reciprocal_square * denominator;
    denominator = 3.03380607434824582924e+01 + reciprocal_square * denominator;
    denominator = 1.0 + reciprocal_square * denominator;
  }
  // Truncating the low word makes the high square exact. Its product
  // residual and the rational correction survive exponential reduction.
  const double head = iree_math_f64_from_bits(iree_math_f64_to_bits(value) &
                                              UINT64_C(0xFFFFFFFF00000000));
  return (iree_math_erfc_exponent_t){
      .high = -head * head - 0.5625,
      .low = (head - value) * (head + value) + numerator / denominator,
  };
}

// Positive complementary tail below the F64 zero threshold.
static double iree_math_erfc_tail_f64(double value) {
  const iree_math_erfc_exponent_t exponent =
      iree_math_erfc_tail_reduce_f64(value);
  return iree_math_exp_f64_compensated(exponent.high, exponent.low) / value;
}

double iree_math_erfc_f64_scaled_tail(double value, double factor) {
  const iree_math_erfc_exponent_t exponent =
      iree_math_erfc_tail_reduce_f64(value);
  const double half_exponential =
      iree_math_exp_f64_compensated(0.5 * exponent.high, 0.5 * exponent.low);
  return ((factor / value) * half_exponential) * half_exponential;
}

/* Interval coefficients adapted from CORE-MATH erf.c/erfc.c
Revision c7100a07428584639defdd64409e3621884c362b

Copyright (c) 2022-2026 Alexei Sibidanov, Paul Zimmermann, Tom Hubrecht and
Claude-Pierre Jeannerod.

This file is part of the CORE-MATH project
(https://core-math.gitlabpages.inria.fr/).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Degree-ten approximations on twelve dyadic cells covering [0.5, 1.25).
// Keeping low center/linear coefficients avoids cancellation in erfc. These
// are the unchanged CORE-MATH rows for i=8..19, with unfused evaluation.
typedef struct iree_math_erf_interval_t {
  // Exact dyadic center; avoids converting the table index back to F64.
  double center;
  // High part of erf(center).
  double value_high;
  // Low part of erf(center), retained until final reconstruction.
  double value_low;
  // High part of the polynomial's linear coefficient.
  double linear_high;
  // Low part of the polynomial's linear coefficient.
  double linear_low;
  // Coefficients of degrees two through ten, in ascending order.
  double polynomial[9];
} iree_math_erf_interval_t;

static const iree_math_erf_interval_t iree_math_erf_intervals[12] = {
    {0x1.1000000000000p-1,
     0x1.1855a5fd3dd5p-1,
     0x1.8f6964e67d61ap-55,
     0x1.b3aafcc27502ep-1,
     -0x1.a9dd26edea8a2p-56,
     {-0x1.cee5ac8e9c531p-2, -0x1.fa02983c853d1p-4, 0x1.77cd75ec731p-3,
      -0x1.fa6f82f9333b7p-10, -0x1.8e0db5528e559p-5, 0x1.00bf7062212bcp-7,
      0x1.3319e670adc9fp-7, -0x1.58833e091aa36p-9, -0x1.8f99b6e81e8f5p-10}},
    {0x1.3000000000000p-1,
     0x1.32a54cb8db67bp-1,
     -0x1.96221f7e18978p-57,
     0x1.96164fafd8de3p-1,
     0x1.0887f82841accp-56,
     {-0x1.e23a7ea0d187ep-2, -0x1.3f5ee1564be49p-4, 0x1.70e469de06907p-3,
      -0x1.3da6878ae6fd8p-6, -0x1.6a0d076468415p-5, 0x1.8cf081f1fc304p-7,
      0x1.f6d62866525e6p-8, -0x1.b93149d5701a4p-9, -0x1.1a6c1a9f7ea73p-10}},
    {0x1.5000000000000p-1,
     0x1.4b13713ad3513p-1,
     0x1.e944ee1b212e4p-57,
     0x1.7791b886e7403p-1,
     -0x1.da43cb53d911cp-57,
     {-0x1.ecef42310f844p-2, -0x1.15c3c5ce705dfp-5, 0x1.5f6890affa468p-3,
      -0x1.1da642fabd4dap-5, -0x1.385991202c7ebp-5, 0x1.fa4f37fc7c6d4p-7,
      0x1.7156b4e430998p-8, -0x1.f546a4377d648p-9, -0x1.32e4e5abb1e1ap-11}},
    {0x1.7000000000000p-1,
     0x1.61955607dd15dp-1,
     0x1.98ff39319ab83p-55,
     0x1.58a445da7c74cp-1,
     0x1.08ec8e156809bp-55,
     {-0x1.ef6c246a12e7ep-2, 0x1.e83e0da03048p-8, 0x1.44cc65df8bfc7p-3,
      -0x1.87d3c8dd62c82p-5, -0x1.f9271a8a1d4e2p-6, 0x1.225234c1c0a0ep-6,
      0x1.c0b0e055a0c48p-9, -0x1.0585251f84919p-8, -0x1.85bfb02436e0fp-13}},
    {0x1.9000000000000p-1,
     0x1.762870f720c6fp-1,
     0x1.118b1ba6da9a7p-55,
     0x1.39ccc1b136d5ap-1,
     0x1.faa9371c0dd8p-58,
     {-0x1.ea4feea4e5addp-2, 0x1.715e595343353p-5, 0x1.22cdbdb4cdd0cp-3,
      -0x1.da50ae547e69ep-5, -0x1.75578f87f217dp-6, 0x1.353319c65f251p-6,
      0x1.39db53a2d03d5p-10, -0x1.fc0364ce1787p-9, 0x1.272bc18b0f2cep-12}},
    {0x1.b000000000000p-1,
     0x1.88d1cd474a2ep-1,
     0x1.6f571ada77d52p-55,
     0x1.1b7e98fe26217p-1,
     0x1.952bd607eb12ep-56,
     {-0x1.de65a22ce0587p-2, 0x1.40686a3f3dc2bp-4, 0x1.f6b0cb6926c42p-4,
      -0x1.09c7caecd317dp-4, -0x1.da668f759eaeap-7, 0x1.364e72035e80ap-6,
      -0x1.d421975736447p-11, -0x1.cc98454e96141p-9, 0x1.a8860fdf17259p-11}},
    {0x1.d000000000000p-1,
     0x1.999d4192a5715p-1,
     -0x1.c888a5759a92cp-55,
     0x1.fc3ee5d1524bp-2,
     -0x1.27e60faac0278p-58,
     {-0x1.cc990045b293fp-2, 0x1.b37338e6ac814p-4, 0x1.a0d11fe9ba61ap-4,
      -0x1.19bb2ca3816bap-4, -0x1.a0b7d94791f03p-8, 0x1.274a59774d5e6p-6,
      -0x1.64adea7b36f57p-9, -0x1.83684bd8ef173p-9, 0x1.38905afd229ffp-10}},
    {0x1.f000000000000p-1,
     0x1.a89c850b7d54dp-1,
     -0x1.e2752ebf0cd02p-55,
     0x1.c40b0729ed548p-2,
     -0x1.c4c1c4927306dp-56,
     {-0x1.b5eaaef09de9dp-2, 0x1.0847c7dad86afp-3, 0x1.47de0a4f796cap-4,
      -0x1.1d9de8b54a3ecp-4, 0x1.33252fb810c7cp-10, 0x1.0ab3e329ded2fp-6,
      -0x1.12d82076274edp-8, -0x1.287bb4a78d728p-9, 0x1.57d31bd574dap-10}},
    {0x1.0800000000000p+0,
     0x1.b5e62fce16095p-1,
     0x1.bc3cff4400364p-56,
     0x1.8eed36b886d93p-2,
     0x1.ea7e17b96436dp-56,
     {-0x1.9b64a06e4b1p-2, 0x1.2bb6e2c74d4fep-3, 0x1.dee322c062364p-5,
      -0x1.169960d5a983dp-4, 0x1.feab4ad0bfc14p-8, 0x1.c76eb94b07a5fp-7,
      -0x1.584474ae8f994p-8, -0x1.88df75be9251fp-10, 0x1.4edef5031709p-10}},
    {0x1.1800000000000p+0,
     0x1.c194b1d49a184p-1,
     -0x1.6770a58b27668p-57,
     0x1.5d4fd33729015p-2,
     -0x1.6db7d76e9e97bp-56,
     {-0x1.7e0f4f0454d97p-2, 0x1.444bc66c35bc4p-3, 0x1.356dbb543255p-5,
      -0x1.0643de6e8c574p-4, 0x1.b2e1f789415e4p-7, 0x1.6ba6d9f4af32fp-7,
      -0x1.8138bf4573a6ap-8, -0x1.7e6e52a583322p-11, 0x1.0f87322fa18a3p-10}},
    {0x1.2800000000000p+0,
     0x1.cbc54b476248dp-1,
     0x1.1a5083b01ec0dp-55,
     0x1.2f7cc3fe6f423p-2,
     0x1.9fbb4b774e85dp-56,
     {-0x1.5ee8429e30a49p-2, 0x1.52a8395f9627p-3, 0x1.313759f199499p-6,
      -0x1.dcf844d90282cp-5, 0x1.1e45f25ab54a1p-6, 0x1.091cb68a58665p-7,
      -0x1.8ea40b0ac8b7bp-8, -0x1.6b91b1bf985f2p-17, 0x1.158d9c0e1c327p-10}},
    {0x1.3800000000000p+0,
     0x1.d4970f9ce00d9p-1,
     -0x1.56704209fca7p-56,
     0x1.059f59af7a906p-2,
     -0x1.0ce27da57f153p-56,
     {-0x1.3eda354ddd5ffp-2, 0x1.57b85ad436067p-3, 0x1.8e90c2a157e8dp-10,
      -0x1.a2893b28f4033p-5, 0x1.4d6af4484a1cbp-6, 0x1.4ccee8c8b1f57p-8,
      -0x1.83304b9e2e312p-8, 0x1.40cb679d0a832p-11, 0x1.d6b5f4bdef24bp-11}},
};

static double iree_math_erf_interval_f64(double value, bool complement) {
  const uint32_t index = (uint32_t)(16.0 * value);
  const iree_math_erf_interval_t* entry = &iree_math_erf_intervals[index - 8];
  const double residual = value - entry->center;
  const double square = residual * residual;
  const double fourth = square * square;
  const double* polynomial = entry->polynomial;
  const double degree_nine = polynomial[7] + residual * polynomial[8];
  const double degree_seven = polynomial[5] + residual * polynomial[6];
  const double degree_five = polynomial[3] + residual * polynomial[4];
  const double degree_three = polynomial[1] + residual * polynomial[2];
  const double upper = degree_seven + square * degree_nine;
  const double middle = degree_three + (square * degree_five + fourth * upper);
  const double quadratic = polynomial[0] + residual * middle;
  const double linear = entry->linear_high + residual * quadratic;
  const double correction =
      (residual * linear + residual * entry->linear_low) + entry->value_low;
  // 1-value_high is exact by Sterbenz throughout these intervals.
  return complement ? (1.0 - entry->value_high) - correction
                    : entry->value_high + correction;
}

IREE_API_EXPORT float iree_math_erf_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  const bool negative = (bits & IREE_MATH_F32_SIGN_BIT) != 0;
  if (IREE_UNLIKELY(magnitude > IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }
  if (magnitude < UINT32_C(0x3F580000)) {
    if (magnitude < UINT32_C(0x38800000)) {
      if (magnitude < UINT32_C(0x04000000)) {
        // Keep tiny linear terms normal until their final rounding.
        return 0.125f * (8.0f * value + 1.0270333290e+00f * value);
      }
      return value + 1.2837916613e-01f * value;
    }
    return value + value * iree_math_erf_small_f32(value * value);
  }
  const float absolute = iree_math_f32_from_bits(magnitude);
  if (magnitude < UINT32_C(0x3FA00000)) {
    const float result =
        8.42697144e-01f + iree_math_erf_near_one_f32(absolute - 1.0f);
    return negative ? -result : result;
  }
  if (magnitude >= UINT32_C(0x40800000)) {
    return negative ? -1.0f : 1.0f;
  }
  const float result = (float)(1.0 - iree_math_erfc_tail_f32(absolute));
  return negative ? -result : result;
}

IREE_API_EXPORT float iree_math_erfc_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  const bool negative = (bits & IREE_MATH_F32_SIGN_BIT) != 0;
  if (IREE_UNLIKELY(magnitude > IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }
  if (magnitude < UINT32_C(0x3F580000)) {
    if (magnitude < UINT32_C(0x33800000)) {
      return 1.0f - value;
    }
    const double correction =
        iree_math_erf_small_f32_widened((double)value * value);
    if (value < 0.25f) {
      return (float)(1.0 - (value + value * correction));
    }
    return (float)(0.5 - (value * correction + (value - 0.5)));
  }
  const float absolute = iree_math_f32_from_bits(magnitude);
  if (magnitude < UINT32_C(0x3FA00000)) {
    const double correction =
        iree_math_erf_near_one_f32_widened((double)absolute - 1.0);
    return (float)(negative ? 1.0 + (8.42697144e-01f + correction)
                            : (1.0 - 8.42697144e-01f) - correction);
  }
  if (magnitude >= UINT32_C(0x41300000)) {
    return negative ? 2.0f : 0.0f;
  }
  if (negative && magnitude >= UINT32_C(0x40A00000)) {
    return 2.0f;
  }
  const double tail = iree_math_erfc_tail_f32(absolute);
  return (float)(negative ? 2.0 - tail : tail);
}

IREE_API_EXPORT double iree_math_erf_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  const bool negative = (bits & IREE_MATH_F64_SIGN_BIT) != 0;
  if (IREE_UNLIKELY(magnitude > IREE_MATH_F64_INFINITY)) {
    return iree_math_f64_canonical_nan();
  }
  if (magnitude < UINT64_C(0x3FE0000000000000)) {
    if (magnitude < UINT64_C(0x3E30000000000000)) {
      if (magnitude < UINT64_C(0x0080000000000000)) {
        return 0.125 * (8.0 * value + 1.02703333676410069053 * value);
      }
      return value + 1.28379167095512586316e-01 * value;
    }
    return value + value * iree_math_erf_small_f64(value * value);
  }
  const double absolute = iree_math_f64_from_bits(magnitude);
  if (magnitude < UINT64_C(0x3FF4000000000000)) {
    const double result = iree_math_erf_interval_f64(absolute, false);
    return negative ? -result : result;
  }
  if (magnitude >= UINT64_C(0x4018000000000000)) {
    return negative ? -1.0 : 1.0;
  }
  const double result = 1.0 - iree_math_erfc_tail_f64(absolute);
  return negative ? -result : result;
}

IREE_API_EXPORT double iree_math_erfc_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  const bool negative = (bits & IREE_MATH_F64_SIGN_BIT) != 0;
  if (IREE_UNLIKELY(magnitude > IREE_MATH_F64_INFINITY)) {
    return iree_math_f64_canonical_nan();
  }
  if (magnitude < UINT64_C(0x3FE0000000000000)) {
    if (magnitude < UINT64_C(0x3C70000000000000)) {
      return 1.0 - value;
    }
    const double correction = iree_math_erf_small_f64(value * value);
    if (value < 0.25) {
      return 1.0 - (value + value * correction);
    }
    return 0.5 - (value * correction + (value - 0.5));
  }
  const double absolute = iree_math_f64_from_bits(magnitude);
  if (magnitude < UINT64_C(0x3FF4000000000000)) {
    const double result = iree_math_erf_interval_f64(absolute, true);
    return negative ? 2.0 - result : result;
  }
  // These are rounded-result boundaries, including both infinities. The
  // positive threshold avoids rounding exp(-x*x) before the final division
  // turning an exact zero result into the least subnormal value.
  if (negative && magnitude >= UINT64_C(0x4018000000000000)) {
    return 2.0;
  }
  if (absolute >= 0x1.b39dc41e48bfdp+4) {
    return 0.0;
  }
  const double tail = iree_math_erfc_tail_f64(absolute);
  return negative ? 2.0 - tail : tail;
}
