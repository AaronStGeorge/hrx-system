// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/hyperbolic.h"

#include "iree/math/exponential.h"
#include "iree/math/float_bits.h"
#include "iree/math/logarithm.h"
#include "iree/math/roots.h"

// Large-exponential reconstruction adapted from OpenLibm revision
// 5fe399749f9276eaa0b8403e507470da05cbbb3f, k_exp.c and k_expf.c.
// Copyright (c) 2011 David Schultz <das@FreeBSD.ORG>. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Computes exp(value)/2 in the narrow interval between exp overflow and
// hyperbolic overflow. The reduction constants approximate k*ln(2) especially
// closely. Subtraction is exact here, and the small exponential stays normal;
// reconstructing a maximum-exponent significand postpones overflow until the
// final multiply. The caller has already bounded the input interval.
static float iree_math_hyperbolic_half_exp_f32(float value) {
  const float reduced = iree_math_exp_f32(value - 162.88958740f);  // k = 235
  const uint32_t bits = iree_math_f32_to_bits(reduced);
  const float significand = iree_math_f32_from_bits(
      (bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x7F000000));
  const float scale = iree_math_f32_from_bits(((bits >> 23) + 107u) << 23);
  return significand * scale;
}

static double iree_math_hyperbolic_half_exp_f64(double value) {
  const double reduced =
      iree_math_exp_f64(value - 1246.97177782734161156);  // k = 1799
  const uint64_t bits = iree_math_f64_to_bits(reduced);
  const double significand = iree_math_f64_from_bits(
      (bits & UINT64_C(0x000FFFFFFFFFFFFF)) | UINT64_C(0x7FE0000000000000));
  const double scale = iree_math_f64_from_bits(((bits >> 52) + 775u) << 52);
  return significand * scale;
}

// Interval formulas adapted from the same OpenLibm revision, e_sinh[f].c,
// e_cosh[f].c, s_asinh[f].c, e_acosh[f].c, and e_atanh[f].c.
// F32 conversion by Ian Lance Taylor.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

IREE_API_EXPORT float iree_math_sinh_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F32_INFINITY)) {
    return magnitude == IREE_MATH_F32_INFINITY ? value
                                               : iree_math_f32_canonical_nan();
  }
  if (magnitude < UINT32_C(0x39800000)) {
    return value;
  }
  const float absolute = iree_math_f32_from_bits(magnitude);
  const float half = (bits & IREE_MATH_F32_SIGN_BIT) ? -0.5f : 0.5f;
  if (magnitude < UINT32_C(0x41100000)) {
    const float difference = iree_math_expm1_f32(absolute);
    if (magnitude < UINT32_C(0x3F800000)) {
      return half * (2.0f * difference -
                     difference * difference / (difference + 1.0f));
    }
    return half * (difference + difference / (difference + 1.0f));
  }
  if (magnitude < UINT32_C(0x42B17217)) {
    return half * iree_math_exp_f32(absolute);
  }
  if (magnitude <= UINT32_C(0x42B2D4FC)) {
    return (half * 2.0f) * iree_math_hyperbolic_half_exp_f32(absolute);
  }
  return iree_math_f32_from_bits((bits & IREE_MATH_F32_SIGN_BIT) |
                                 IREE_MATH_F32_INFINITY);
}

IREE_API_EXPORT double iree_math_sinh_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F64_INFINITY)) {
    return magnitude == IREE_MATH_F64_INFINITY ? value
                                               : iree_math_f64_canonical_nan();
  }
  const uint32_t high = (uint32_t)(magnitude >> 32);
  if (high < UINT32_C(0x3E300000)) {
    return value;
  }
  const double absolute = iree_math_f64_from_bits(magnitude);
  const double half = (bits & IREE_MATH_F64_SIGN_BIT) ? -0.5 : 0.5;
  if (high < UINT32_C(0x40360000)) {
    const double difference = iree_math_expm1_f64(absolute);
    if (high < UINT32_C(0x3FF00000)) {
      return half *
             (2.0 * difference - difference * difference / (difference + 1.0));
    }
    return half * (difference + difference / (difference + 1.0));
  }
  if (high < UINT32_C(0x40862E42)) {
    return half * iree_math_exp_f64(absolute);
  }
  if (high <= UINT32_C(0x408633CE)) {
    return (half * 2.0) * iree_math_hyperbolic_half_exp_f64(absolute);
  }
  return iree_math_f64_from_bits((bits & IREE_MATH_F64_SIGN_BIT) |
                                 IREE_MATH_F64_INFINITY);
}

IREE_API_EXPORT float iree_math_cosh_f32(float value) {
  const uint32_t magnitude =
      iree_math_f32_to_bits(value) & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F32_INFINITY)) {
    return magnitude == IREE_MATH_F32_INFINITY
               ? iree_math_f32_from_bits(magnitude)
               : iree_math_f32_canonical_nan();
  }
  if (magnitude < UINT32_C(0x39800000)) {
    return 1.0f;
  }
  const float absolute = iree_math_f32_from_bits(magnitude);
  if (magnitude < UINT32_C(0x3EB17218)) {
    const float difference = iree_math_expm1_f32(absolute);
    const float exponential = 1.0f + difference;
    return 1.0f + (difference * difference) / (exponential + exponential);
  }
  if (magnitude < UINT32_C(0x41100000)) {
    const float exponential = iree_math_exp_f32(absolute);
    return 0.5f * exponential + 0.5f / exponential;
  }
  if (magnitude < UINT32_C(0x42B17217)) {
    return 0.5f * iree_math_exp_f32(absolute);
  }
  if (magnitude <= UINT32_C(0x42B2D4FC)) {
    return iree_math_hyperbolic_half_exp_f32(absolute);
  }
  return iree_math_f32_from_bits(IREE_MATH_F32_INFINITY);
}

IREE_API_EXPORT double iree_math_cosh_f64(double value) {
  const uint64_t magnitude =
      iree_math_f64_to_bits(value) & IREE_MATH_F64_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F64_INFINITY)) {
    return magnitude == IREE_MATH_F64_INFINITY
               ? iree_math_f64_from_bits(magnitude)
               : iree_math_f64_canonical_nan();
  }
  const uint32_t high = (uint32_t)(magnitude >> 32);
  if (high < UINT32_C(0x3C800000)) {
    return 1.0;
  }
  const double absolute = iree_math_f64_from_bits(magnitude);
  if (high < UINT32_C(0x3FD62E43)) {
    const double difference = iree_math_expm1_f64(absolute);
    const double exponential = 1.0 + difference;
    return 1.0 + (difference * difference) / (exponential + exponential);
  }
  if (high < UINT32_C(0x40360000)) {
    const double exponential = iree_math_exp_f64(absolute);
    return 0.5 * exponential + 0.5 / exponential;
  }
  if (high < UINT32_C(0x40862E42)) {
    return 0.5 * iree_math_exp_f64(absolute);
  }
  if (high <= UINT32_C(0x408633CE)) {
    return iree_math_hyperbolic_half_exp_f64(absolute);
  }
  return iree_math_f64_from_bits(IREE_MATH_F64_INFINITY);
}

IREE_API_EXPORT float iree_math_asinh_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F32_INFINITY)) {
    return magnitude == IREE_MATH_F32_INFINITY ? value
                                               : iree_math_f32_canonical_nan();
  }
  if (magnitude < UINT32_C(0x31800000)) {
    return value;
  }
  const float absolute = iree_math_f32_from_bits(magnitude);
  float result;
  if (magnitude > UINT32_C(0x4D800000)) {
    result = iree_math_log_f32(absolute) + 0x1.62e43p-1f;
  } else if (magnitude > UINT32_C(0x40000000)) {
    result = iree_math_log_f32(
        2.0f * absolute +
        1.0f / (iree_math_sqrt_f32(value * value + 1.0f) + absolute));
  } else {
    const float square = value * value;
    result = iree_math_log1p_f32(
        absolute + square / (1.0f + iree_math_sqrt_f32(1.0f + square)));
  }
  return (bits & IREE_MATH_F32_SIGN_BIT) ? -result : result;
}

IREE_API_EXPORT double iree_math_asinh_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= IREE_MATH_F64_INFINITY)) {
    return magnitude == IREE_MATH_F64_INFINITY ? value
                                               : iree_math_f64_canonical_nan();
  }
  const uint32_t high = (uint32_t)(magnitude >> 32);
  if (high < UINT32_C(0x3E300000)) {
    return value;
  }
  const double absolute = iree_math_f64_from_bits(magnitude);
  double result;
  if (high > UINT32_C(0x41B00000)) {
    result = iree_math_log_f64(absolute) + 0x1.62e42fefa39efp-1;
  } else if (high > UINT32_C(0x40000000)) {
    result = iree_math_log_f64(
        2.0 * absolute +
        1.0 / (iree_math_sqrt_f64(value * value + 1.0) + absolute));
  } else {
    const double square = value * value;
    result = iree_math_log1p_f64(
        absolute + square / (1.0 + iree_math_sqrt_f64(1.0 + square)));
  }
  return (bits & IREE_MATH_F64_SIGN_BIT) ? -result : result;
}

IREE_API_EXPORT float iree_math_acosh_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  if (IREE_UNLIKELY(bits < UINT32_C(0x3F800000) ||
                    bits > IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }
  if (bits >= UINT32_C(0x4D800000)) {
    return bits == IREE_MATH_F32_INFINITY
               ? value
               : iree_math_log_f32(value) + 0x1.62e43p-1f;
  }
  if (bits == UINT32_C(0x3F800000)) {
    return 0.0f;
  }
  if (bits > UINT32_C(0x40000000)) {
    return iree_math_log_f32(
        2.0f * value -
        1.0f / (value + iree_math_sqrt_f32(value * value - 1.0f)));
  }
  const float difference = value - 1.0f;
  return iree_math_log1p_f32(
      difference +
      iree_math_sqrt_f32(2.0f * difference + difference * difference));
}

IREE_API_EXPORT double iree_math_acosh_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  if (IREE_UNLIKELY(bits < UINT64_C(0x3FF0000000000000) ||
                    bits > IREE_MATH_F64_INFINITY)) {
    return iree_math_f64_canonical_nan();
  }
  const uint32_t high = (uint32_t)(bits >> 32);
  if (high >= UINT32_C(0x41B00000)) {
    return bits == IREE_MATH_F64_INFINITY
               ? value
               : iree_math_log_f64(value) + 0x1.62e42fefa39efp-1;
  }
  if (bits == UINT64_C(0x3FF0000000000000)) {
    return 0.0;
  }
  if (high > UINT32_C(0x40000000)) {
    return iree_math_log_f64(
        2.0 * value - 1.0 / (value + iree_math_sqrt_f64(value * value - 1.0)));
  }
  const double difference = value - 1.0;
  return iree_math_log1p_f64(
      difference +
      iree_math_sqrt_f64(2.0 * difference + difference * difference));
}

IREE_API_EXPORT float iree_math_atanh_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= UINT32_C(0x3F800000))) {
    return magnitude == UINT32_C(0x3F800000)
               ? iree_math_f32_from_bits((bits & IREE_MATH_F32_SIGN_BIT) |
                                         IREE_MATH_F32_INFINITY)
               : iree_math_f32_canonical_nan();
  }
  if (magnitude < UINT32_C(0x31800000)) {
    return value;
  }
  const float absolute = iree_math_f32_from_bits(magnitude);
  const float twice = absolute + absolute;
  const float argument = magnitude < UINT32_C(0x3F000000)
                             ? twice + twice * absolute / (1.0f - absolute)
                             : twice / (1.0f - absolute);
  const float result = 0.5f * iree_math_log1p_f32(argument);
  return (bits & IREE_MATH_F32_SIGN_BIT) ? -result : result;
}

IREE_API_EXPORT double iree_math_atanh_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= UINT64_C(0x3FF0000000000000))) {
    return magnitude == UINT64_C(0x3FF0000000000000)
               ? iree_math_f64_from_bits((bits & IREE_MATH_F64_SIGN_BIT) |
                                         IREE_MATH_F64_INFINITY)
               : iree_math_f64_canonical_nan();
  }
  if (magnitude < UINT64_C(0x3E30000000000000)) {
    return value;
  }
  const double absolute = iree_math_f64_from_bits(magnitude);
  const double twice = absolute + absolute;
  const double argument = magnitude < UINT64_C(0x3FE0000000000000)
                              ? twice + twice * absolute / (1.0 - absolute)
                              : twice / (1.0 - absolute);
  const double result = 0.5 * iree_math_log1p_f64(argument);
  return (bits & IREE_MATH_F64_SIGN_BIT) ? -result : result;
}

/* F32 rational tangent adapted from CORE-MATH revision
c7100a07428584639defdd64409e3621884c362b, src/binary32/tanh/tanhf.c.

Copyright (c) 2022-2025 Alexei Sibidanov.

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

IREE_API_EXPORT float iree_math_tanh_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude > IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }
  // |tanh(x)-x| < |x|^3/3 < half an F32 ulp throughout this tiny interval.
  if (magnitude < UINT32_C(0x39800000)) {
    return value;
  }
  if (magnitude > UINT32_C(0x41102CB3)) {
    return iree_math_f32_from_bits((bits & IREE_MATH_F32_SIGN_BIT) |
                                   UINT32_C(0x3F800000));
  }
  const double wide = value;
  const double square = wide * wide;
  const double fourth = square * square;
  const double eighth = fourth * fourth;
  static const double numerator_coefficients[] = {0x1p+0,
                                                  0x1.30877b8b72d33p-3,
                                                  0x1.694aa09ae9e5ep-8,
                                                  0x1.4101377abb729p-14,
                                                  0x1.e0392b1db0018p-22,
                                                  0x1.2533756e546f7p-30,
                                                  0x1.d62e5abe6ae8ap-41,
                                                  0x1.b06be534182dep-54};
  static const double denominator_coefficients[] = {0x1p+0,
                                                    0x1.ed99131b0ebeap-2,
                                                    0x1.0d27ed6c95a69p-5,
                                                    0x1.7cbdaca0e9fccp-11,
                                                    0x1.b4e60b892578ep-18,
                                                    0x1.a6f707c5c71abp-26,
                                                    0x1.35a8b6e2cd94cp-35,
                                                    0x1.ca8230677aa01p-47};
  double numerator_low =
      numerator_coefficients[0] + square * numerator_coefficients[1];
  double numerator_middle =
      numerator_coefficients[2] + square * numerator_coefficients[3];
  double numerator_high =
      numerator_coefficients[4] + square * numerator_coefficients[5];
  double numerator_top =
      numerator_coefficients[6] + square * numerator_coefficients[7];
  numerator_low += fourth * numerator_middle;
  numerator_high += fourth * numerator_top;
  numerator_low += eighth * numerator_high;
  double denominator_low =
      denominator_coefficients[0] + square * denominator_coefficients[1];
  double denominator_middle =
      denominator_coefficients[2] + square * denominator_coefficients[3];
  double denominator_high =
      denominator_coefficients[4] + square * denominator_coefficients[5];
  double denominator_top =
      denominator_coefficients[6] + square * denominator_coefficients[7];
  denominator_low += fourth * denominator_middle;
  denominator_high += fourth * denominator_top;
  denominator_low += eighth * denominator_high;
  const double result = wide * numerator_low / denominator_low;
  return (float)result;
}

/*
 * Copyright (C) 2008-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

// AOCL-LibM 29fd054f383e6c5e2dec2fce781d5220059f1836,
// src/optimized/tanh.c. Two [3,3] rational intervals avoid cancellation near
// zero; above one, the shared exponential supplies the saturated
// reconstruction.
IREE_API_EXPORT double iree_math_tanh_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (magnitude < UINT64_C(0x3E30000000000000)) {
    return value;
  }
  if (IREE_UNLIKELY(magnitude > IREE_MATH_F64_INFINITY)) {
    return iree_math_f64_canonical_nan();
  }
  const double absolute = iree_math_f64_from_bits(magnitude);
  double result;
  if (absolute > 20.0) {
    result = 1.0;
  } else if (absolute <= 1.0) {
    const double square = absolute * absolute;
    const double cube = square * absolute;
    const double fourth = square * square;
    const double fifth = cube * square;
    const double sixth = fourth * square;
    const double seventh = fifth * square;
    const double ninth = seventh * square;
    double numerator;
    double denominator;
    if (absolute < 0.9) {
      numerator = -0.274030424656179760118928e0 * cube +
                  -0.176016349003044679402273e-1 * fifth;
      numerator += -0.200047621071909498730453e-3 * seventh -
                   0.142077926378834722618091e-7 * ninth;
      denominator =
          0.822091273968539282568011e0 + 0.381641414288328849317962e0 * square;
      denominator += 0.201562166026937652780575e-1 * fourth +
                     0.2091140262529164482568557e-3 * sixth;
    } else {
      numerator = -0.227793870659088295252442e0 * cube +
                  -0.146173047288731678404066e-1 * fifth;
      numerator += -0.165597043903549960486816e-3 * seventh -
                   0.115475878996143396378318e-7 * ninth;
      denominator = 0.683381611977295894959554e0 +
                    (0.317204558977294374244770e0 +
                     (0.167358775461896562588695e-1 +
                      0.173076050126225961768710e-3 * square) *
                         square) *
                        square;
    }
    result = absolute + numerator / denominator;
  } else {
    result = 1.0 - 2.0 / (iree_math_exp_f64(2.0 * absolute) + 1.0);
  }
  return (bits & IREE_MATH_F64_SIGN_BIT) ? -result : result;
}
