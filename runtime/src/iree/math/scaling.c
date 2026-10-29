// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exponent decomposition follows Sun's frexp, with integer normalization.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
//
// Binary scaling adapted from musl scalbn.c, revision
// 7813e9119eb5fe681b6750b8a033a0490bca0941.
// Copyright © 2005-2020 Rich Felker, et al.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "iree/math/scaling.h"

#include "iree/base/internal/math.h"
#include "iree/math/float_bits.h"

// The typed entry points specialize the representation constants. Integer
// normalization handles subnormal inputs without floating-point arithmetic.
static inline uint64_t iree_math_frexp_bits(uint64_t bits,
                                            unsigned fraction_bits,
                                            unsigned exponent_bias,
                                            unsigned width,
                                            int32_t* out_exponent) {
  const uint64_t sign = bits & (UINT64_C(1) << (width - 1));
  const uint64_t magnitude = bits ^ sign;
  const uint64_t fraction_mask = (UINT64_C(1) << fraction_bits) - 1;
  const uint64_t infinity = (uint64_t)(2 * exponent_bias + 1) << fraction_bits;
  int32_t exponent = (int32_t)(magnitude >> fraction_bits);
  *out_exponent = 0;
  if (magnitude >= infinity) {
    return magnitude == infinity
               ? bits
               : infinity | (UINT64_C(1) << (fraction_bits - 1));
  }
  if (exponent == 0) {
    if (magnitude == 0) {
      return bits;
    }
    const int32_t normalization =
        iree_math_count_leading_zeros_u64(magnitude) - (63 - fraction_bits);
    bits = sign | (magnitude << normalization);
    exponent = 1 - normalization;
  }
  *out_exponent = exponent - (int32_t)(exponent_bias - 1);
  return sign | (bits & fraction_mask) |
         ((uint64_t)(exponent_bias - 1) << fraction_bits);
}

IREE_API_EXPORT float iree_math_frexp_f32(float value, int32_t* out_exponent) {
  return iree_math_f32_from_bits((uint32_t)iree_math_frexp_bits(
      iree_math_f32_to_bits(value), 23, 127, 32, out_exponent));
}

IREE_API_EXPORT double iree_math_frexp_f64(double value,
                                           int32_t* out_exponent) {
  return iree_math_f64_from_bits(iree_math_frexp_bits(
      iree_math_f64_to_bits(value), 52, 1023, 64, out_exponent));
}

// Each typed multiply rounds at its declared width. Large exponents are
// consumed in at most two bounded chunks before constructing a normal power
// of two for the final multiply. Negative chunks leave more than one full
// significand of scaling for the last step, preventing double rounding when
// the final result is subnormal. Exponent adjustments cannot overflow i32.
#define IREE_MATH_DEFINE_LDEXP(width, type, bits_type, fraction_bits, bias)   \
  IREE_API_EXPORT type iree_math_ldexp_f##width(type value,                   \
                                                int32_t exponent) {           \
    const bits_type magnitude = iree_math_f##width##_to_bits(value) &         \
                                (((bits_type)1 << (width - 1)) - 1);          \
    const bits_type infinity = (bits_type)(2 * bias + 1) << fraction_bits;    \
    if (magnitude > infinity) {                                               \
      return iree_math_f##width##_canonical_nan();                            \
    }                                                                         \
    type result = value;                                                      \
    if (exponent > bias) {                                                    \
      result *= iree_math_f##width##_from_bits((bits_type)(2 * bias)          \
                                               << fraction_bits);             \
      exponent -= bias;                                                       \
      if (exponent > bias) {                                                  \
        result *= iree_math_f##width##_from_bits((bits_type)(2 * bias)        \
                                                 << fraction_bits);           \
        exponent -= bias;                                                     \
        if (exponent > bias) exponent = bias;                                 \
      }                                                                       \
    } else if (exponent < 1 - bias) {                                         \
      result *= iree_math_f##width##_from_bits((bits_type)(fraction_bits + 2) \
                                               << fraction_bits);             \
      exponent += bias - fraction_bits - 2;                                   \
      if (exponent < 1 - bias) {                                              \
        result *= iree_math_f##width##_from_bits(                             \
            (bits_type)(fraction_bits + 2) << fraction_bits);                 \
        exponent += bias - fraction_bits - 2;                                 \
        if (exponent < 1 - bias) exponent = 1 - bias;                         \
      }                                                                       \
    }                                                                         \
    return result * iree_math_f##width##_from_bits(                           \
                        (bits_type)(bias + exponent) << fraction_bits);       \
  }
IREE_MATH_DEFINE_LDEXP(32, float, uint32_t, 23, 127)
IREE_MATH_DEFINE_LDEXP(64, double, uint64_t, 52, 1023)
#undef IREE_MATH_DEFINE_LDEXP
