// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/inverse_trigonometry.h"

#include <stdbool.h>

#include "iree/math/float_bits.h"
#include "iree/math/roots.h"

// Adapted from OpenLibm revision 5fe399749f9276eaa0b8403e507470da05cbbb3f:
// e_asin[f].c, e_acos[f].c, s_atan[f].c, and e_atan2[f].c.
// F32 conversion by Ian Lance Taylor.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

// asin(x) = x + x*R(x*x) on [-0.5, 0.5]. Both asin and acos reduce their
// endpoint intervals into this domain and share the same rounded rational
// evaluation. Keep the numerator/denominator grouping fixed.
static float iree_math_asin_correction_f32(float square) {
  const float numerator =
      square * (1.6666586697e-01f +
                square * (-4.2743422091e-02f + square * -8.6563630030e-03f));
  const float denominator = 1.0f + square * -7.0662963390e-01f;
  return numerator / denominator;
}

static double iree_math_asin_correction_f64(double square) {
  const double numerator =
      square *
      (1.66666666666666657415e-01 +
       square *
           (-3.25565818622400915405e-01 +
            square *
                (2.01212532134862925881e-01 +
                 square * (-4.00555345006794114027e-02 +
                           square * (7.91534994289814532176e-04 +
                                     square * 3.47933107596021167570e-05)))));
  const double denominator =
      1.0 +
      square * (-2.40339491173441421878e+00 +
                square * (2.02094576023350569471e+00 +
                          square * (-6.88283971605453293030e-01 +
                                    square * 7.70381505559019352791e-02)));
  return numerator / denominator;
}

IREE_API_EXPORT float iree_math_asin_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= UINT32_C(0x3F800000))) {
    return magnitude == UINT32_C(0x3F800000)
               ? iree_math_f32_from_bits((bits & IREE_MATH_F32_SIGN_BIT) |
                                         UINT32_C(0x3FC90FDB))
               : iree_math_f32_canonical_nan();
  }
  if (magnitude < UINT32_C(0x39800000)) {
    return value;
  }
  if (magnitude < UINT32_C(0x3F000000)) {
    return value + value * iree_math_asin_correction_f32(value * value);
  }
  const float square = (1.0f - iree_math_f32_from_bits(magnitude)) * 0.5f;
  const float correction = iree_math_asin_correction_f32(square);
  // F64 root/reconstruction prevents loss at the final F32 endpoint rounding.
  const double root = iree_math_sqrt_f64(square);
  const float result =
      (float)(0x1.921fb54442d18p+0 - 2.0 * (root + root * correction));
  return (bits & IREE_MATH_F32_SIGN_BIT) ? -result : result;
}

IREE_API_EXPORT double iree_math_asin_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= UINT64_C(0x3FF0000000000000))) {
    return magnitude == UINT64_C(0x3FF0000000000000)
               ? iree_math_f64_from_bits((bits & IREE_MATH_F64_SIGN_BIT) |
                                         UINT64_C(0x3FF921FB54442D18))
               : iree_math_f64_canonical_nan();
  }
  if (magnitude < UINT64_C(0x3E50000000000000)) {
    return value;
  }
  if (magnitude < UINT64_C(0x3FE0000000000000)) {
    return value + value * iree_math_asin_correction_f64(value * value);
  }
  const double square = (1.0 - iree_math_f64_from_bits(magnitude)) * 0.5;
  const double correction = iree_math_asin_correction_f64(square);
  const double root = iree_math_sqrt_f64(square);
  const double half_pi = 0x1.921fb54442d18p+0;
  const double half_pi_tail = 0x1.1a62633145c07p-54;
  double result;
  if (magnitude >= UINT64_C(0x3FEF333300000000)) {
    result = half_pi - (2.0 * (root + root * correction) - half_pi_tail);
  } else {
    // Split the root so reconstruction near pi/4 keeps its low contribution.
    const double head = iree_math_f64_from_bits(iree_math_f64_to_bits(root) &
                                                UINT64_C(0xFFFFFFFF00000000));
    const double tail = (square - head * head) / (root + head);
    const double low = 2.0 * root * correction - (half_pi_tail - 2.0 * tail);
    const double high = 0x1.921fb54442d18p-1 - 2.0 * head;
    result = 0x1.921fb54442d18p-1 - (low - high);
  }
  return (bits & IREE_MATH_F64_SIGN_BIT) ? -result : result;
}

IREE_API_EXPORT float iree_math_acos_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= UINT32_C(0x3F800000))) {
    if (magnitude > UINT32_C(0x3F800000)) {
      return iree_math_f32_canonical_nan();
    }
    return (bits & IREE_MATH_F32_SIGN_BIT) ? 0x1.921fb6p+1f : 0.0f;
  }
  if (magnitude <= UINT32_C(0x32800000)) {
    return 0x1.921fb6p+0f;
  }
  const float half_pi = 0x1.921fb4p+0f;
  const float half_pi_tail = 0x1.4442dp-24f;
  if (magnitude < UINT32_C(0x3F000000)) {
    const float correction = iree_math_asin_correction_f32(value * value);
    return half_pi - (value - (half_pi_tail - value * correction));
  }
  if (bits & IREE_MATH_F32_SIGN_BIT) {
    const float square = (1.0f + value) * 0.5f;
    const float root = iree_math_sqrt_f32(square);
    const float correction = iree_math_asin_correction_f32(square);
    return 0x1.921fb4p+1f - 2.0f * (root + (correction * root - half_pi_tail));
  }
  const float square = (1.0f - value) * 0.5f;
  const float root = iree_math_sqrt_f32(square);
  const float head = iree_math_f32_from_bits(iree_math_f32_to_bits(root) &
                                             UINT32_C(0xFFFFF000));
  const float tail = (square - head * head) / (root + head);
  const float correction = iree_math_asin_correction_f32(square);
  return 2.0f * (head + (correction * root + tail));
}

IREE_API_EXPORT double iree_math_acos_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= UINT64_C(0x3FF0000000000000))) {
    if (magnitude > UINT64_C(0x3FF0000000000000)) {
      return iree_math_f64_canonical_nan();
    }
    return (bits & IREE_MATH_F64_SIGN_BIT) ? 0x1.921fb54442d18p+1 : 0.0;
  }
  if (magnitude < UINT64_C(0x3C60000100000000)) {
    return 0x1.921fb54442d18p+0;
  }
  const double half_pi = 0x1.921fb54442d18p+0;
  const double half_pi_tail = 0x1.1a62633145c07p-54;
  if (magnitude < UINT64_C(0x3FE0000000000000)) {
    const double correction = iree_math_asin_correction_f64(value * value);
    return half_pi - (value - (half_pi_tail - value * correction));
  }
  if (bits & IREE_MATH_F64_SIGN_BIT) {
    const double square = (1.0 + value) * 0.5;
    const double root = iree_math_sqrt_f64(square);
    const double correction = iree_math_asin_correction_f64(square);
    return 0x1.921fb54442d18p+1 -
           2.0 * (root + (correction * root - half_pi_tail));
  }
  const double square = (1.0 - value) * 0.5;
  const double root = iree_math_sqrt_f64(square);
  const double head = iree_math_f64_from_bits(iree_math_f64_to_bits(root) &
                                              UINT64_C(0xFFFFFFFF00000000));
  const double tail = (square - head * head) / (root + head);
  const double correction = iree_math_asin_correction_f64(square);
  return 2.0 * (head + (correction * root + tail));
}

IREE_API_EXPORT float iree_math_atan_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (magnitude >= UINT32_C(0x4C800000)) {
    return magnitude > IREE_MATH_F32_INFINITY
               ? iree_math_f32_canonical_nan()
               : iree_math_f32_from_bits((bits & IREE_MATH_F32_SIGN_BIT) |
                                         UINT32_C(0x3FC90FDB));
  }
  if (magnitude < UINT32_C(0x39800000)) {
    return value;
  }
  int interval = -1;
  if (magnitude >= UINT32_C(0x3EE00000)) {
    value = iree_math_f32_from_bits(magnitude);
    if (magnitude < UINT32_C(0x3F300000)) {
      interval = 0;
      value = (2.0f * value - 1.0f) / (2.0f + value);
    } else if (magnitude < UINT32_C(0x3F980000)) {
      interval = 1;
      value = (value - 1.0f) / (value + 1.0f);
    } else if (magnitude < UINT32_C(0x401C0000)) {
      interval = 2;
      value = (value - 1.5f) / (1.0f + 1.5f * value);
    } else {
      interval = 3;
      value = -1.0f / value;
    }
  }
  const float square = value * value;
  const float fourth = square * square;
  const float odd =
      square * (3.3333328366e-01f +
                fourth * (1.4253635705e-01f + fourth * 6.1687607318e-02f));
  const float even =
      fourth * (-1.9999158382e-01f + fourth * -1.0648017377e-01f);
  if (interval < 0) {
    return value - value * (odd + even);
  }
  static const float heads[] = {4.6364760399e-01f, 7.8539812565e-01f,
                                9.8279368877e-01f, 1.5707962513e+00f};
  static const float tails[] = {5.0121582440e-09f, 3.7748947079e-08f,
                                3.4473217170e-08f, 7.5497894159e-08f};
  const float result =
      heads[interval] - ((value * (odd + even) - tails[interval]) - value);
  return (bits & IREE_MATH_F32_SIGN_BIT) ? -result : result;
}

IREE_API_EXPORT double iree_math_atan_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (magnitude >= UINT64_C(0x4410000000000000)) {
    return magnitude > IREE_MATH_F64_INFINITY
               ? iree_math_f64_canonical_nan()
               : iree_math_f64_from_bits((bits & IREE_MATH_F64_SIGN_BIT) |
                                         UINT64_C(0x3FF921FB54442D18));
  }
  if (magnitude < UINT64_C(0x3E40000000000000)) {
    return value;
  }
  int interval = -1;
  if (magnitude >= UINT64_C(0x3FDC000000000000)) {
    value = iree_math_f64_from_bits(magnitude);
    if (magnitude < UINT64_C(0x3FE6000000000000)) {
      interval = 0;
      value = (2.0 * value - 1.0) / (2.0 + value);
    } else if (magnitude < UINT64_C(0x3FF3000000000000)) {
      interval = 1;
      value = (value - 1.0) / (value + 1.0);
    } else if (magnitude < UINT64_C(0x4003800000000000)) {
      interval = 2;
      value = (value - 1.5) / (1.0 + 1.5 * value);
    } else {
      interval = 3;
      value = -1.0 / value;
    }
  }
  const double square = value * value;
  const double fourth = square * square;
  const double odd =
      square *
      (3.33333333333329318027e-01 +
       fourth *
           (1.42857142725034663711e-01 +
            fourth *
                (9.09088713343650656196e-02 +
                 fourth * (6.66107313738753120669e-02 +
                           fourth * (4.97687799461593236017e-02 +
                                     fourth * 1.62858201153657823623e-02)))));
  const double even =
      fourth *
      (-1.99999999998764832476e-01 +
       fourth * (-1.11111104054623557880e-01 +
                 fourth * (-7.69187620504482999495e-02 +
                           fourth * (-5.83357013379057348645e-02 +
                                     fourth * -3.65315727442169155270e-02))));
  if (interval < 0) {
    return value - value * (odd + even);
  }
  static const double heads[] = {
      4.63647609000806093515e-01, 7.85398163397448278999e-01,
      9.82793723247329054082e-01, 1.57079632679489655800e+00};
  static const double tails[] = {
      2.26987774529616870924e-17, 3.06161699786838301793e-17,
      1.39033110312309984516e-17, 6.12323399573676603587e-17};
  const double result =
      heads[interval] - ((value * (odd + even) - tails[interval]) - value);
  return (bits & IREE_MATH_F64_SIGN_BIT) ? -result : result;
}

IREE_API_EXPORT float iree_math_atan2_f32(float y, float x) {
  const uint32_t x_bits = iree_math_f32_to_bits(x);
  const uint32_t y_bits = iree_math_f32_to_bits(y);
  const uint32_t x_magnitude = x_bits & IREE_MATH_F32_MAGNITUDE_MASK;
  const uint32_t y_magnitude = y_bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(x_magnitude > IREE_MATH_F32_INFINITY ||
                    y_magnitude > IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }
  if (x_bits == UINT32_C(0x3F800000)) {
    return iree_math_atan_f32(y);
  }
  const bool negative_x = (x_bits & IREE_MATH_F32_SIGN_BIT) != 0;
  float result;
  if (y_magnitude == 0) {
    result = negative_x ? 0x1.921fb6p+1f : 0.0f;
  } else if (x_magnitude == 0) {
    result = 0x1.921fb6p+0f;
  } else if (x_magnitude == IREE_MATH_F32_INFINITY) {
    if (y_magnitude == IREE_MATH_F32_INFINITY) {
      result = negative_x ? 0x1.2d97c8p+1f : 0x1.921fb6p-1f;
    } else {
      result = negative_x ? 0x1.921fb6p+1f : 0.0f;
    }
  } else if (y_magnitude == IREE_MATH_F32_INFINITY) {
    result = 0x1.921fb6p+0f;
  } else {
    const int32_t difference = (int32_t)y_magnitude - (int32_t)x_magnitude;
    // Compare biased exponent differences directly, without a signed shift.
    if (difference >= INT32_C(0x0D800000)) {
      result = 0x1.921fb6p+0f;
    } else if (negative_x && difference < -INT32_C(0x0D000000)) {
      result = 0x1.921fb6p+1f;
    } else {
      const float ratio = iree_math_f32_from_bits(iree_math_f32_to_bits(y / x) &
                                                  IREE_MATH_F32_MAGNITUDE_MASK);
      result = iree_math_atan_f32(ratio);
      if (negative_x) {
        const float pi_tail = -0x1.777a5cp-24f;
        result = 0x1.921fb6p+1f - (result - pi_tail);
      }
    }
  }
  return (y_bits & IREE_MATH_F32_SIGN_BIT) ? -result : result;
}

IREE_API_EXPORT double iree_math_atan2_f64(double y, double x) {
  const uint64_t x_bits = iree_math_f64_to_bits(x);
  const uint64_t y_bits = iree_math_f64_to_bits(y);
  const uint64_t x_magnitude = x_bits & IREE_MATH_F64_MAGNITUDE_MASK;
  const uint64_t y_magnitude = y_bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(x_magnitude > IREE_MATH_F64_INFINITY ||
                    y_magnitude > IREE_MATH_F64_INFINITY)) {
    return iree_math_f64_canonical_nan();
  }
  if (x_bits == UINT64_C(0x3FF0000000000000)) {
    return iree_math_atan_f64(y);
  }
  const bool negative_x = (x_bits & IREE_MATH_F64_SIGN_BIT) != 0;
  double result;
  if (y_magnitude == 0) {
    result = negative_x ? 0x1.921fb54442d18p+1 : 0.0;
  } else if (x_magnitude == 0) {
    result = 0x1.921fb54442d18p+0;
  } else if (x_magnitude == IREE_MATH_F64_INFINITY) {
    if (y_magnitude == IREE_MATH_F64_INFINITY) {
      result = negative_x ? 0x1.2d97c7f3321d2p+1 : 0x1.921fb54442d18p-1;
    } else {
      result = negative_x ? 0x1.921fb54442d18p+1 : 0.0;
    }
  } else if (y_magnitude == IREE_MATH_F64_INFINITY) {
    result = 0x1.921fb54442d18p+0;
  } else {
    const int32_t difference =
        (int32_t)(y_magnitude >> 32) - (int32_t)(x_magnitude >> 32);
    if (difference >= INT32_C(0x03D00000)) {
      result = 0x1.921fb54442d18p+0;
    } else if (negative_x && difference < -INT32_C(0x03C00000)) {
      result = 0x1.921fb54442d18p+1;
    } else {
      const double ratio = iree_math_f64_from_bits(
          iree_math_f64_to_bits(y / x) & IREE_MATH_F64_MAGNITUDE_MASK);
      result = iree_math_atan_f64(ratio);
      if (negative_x) {
        result = 0x1.921fb54442d18p+1 - (result - 0x1.1a62633145c07p-53);
      }
    }
  }
  return (y_bits & IREE_MATH_F64_SIGN_BIT) ? -result : result;
}
