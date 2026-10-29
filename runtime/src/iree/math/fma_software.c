// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fused arithmetic adapted from musl, revision
// 7813e9119eb5fe681b6750b8a033a0490bca0941 (src/math/fma.c and fmaf.c).
// Floating exception flags are unobservable; only nearest-even result mapping
// and gradual underflow are retained.
//
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

#include "iree/math/fma_software.h"

#include "iree/base/internal/math.h"
#include "iree/math/float_bits.h"
#include "iree/math/scaling.h"

float iree_math_fma_f32_software(float a, float b, float c) {
  double product = (double)a * b;
  double sum = product + c;
  uint64_t bits = iree_math_f64_to_bits(sum);
  int exponent = (int)((bits >> 52) & 0x7ff);
  // F64-to-F32 double rounding is possible at a halfway normal result.
  bool halfway = (bits & 0x1fffffff) == 0x10000000;
  // Subnormal F32 results have a moving rounding boundary.
  bool tiny = exponent <= 0x3ff - 126 && exponent >= 0x3ff - 149;
  if (!halfway && !tiny) {
    // The F64 result is unambiguous at F32 precision.
    return (float)sum;
  }
  if (exponent != 0x7ff) {
    // FastTwoSum recovers the exact residual in the required nearest mode.
    bool negative = (bits >> 63) != 0;
    double error =
        negative == (product < c) ? product - sum + c : c - sum + product;
    if (error != 0.0) {
      // Round to odd before the final F32 conversion: retain the exact
      // side of a normal or subnormal F32 rounding midpoint.
      bits -= negative ^ (error < 0);
      bits |= 1;
      sum = iree_math_f64_from_bits(bits);
    }
  }
  return (float)sum;
}

typedef struct iree_math_fma_operand_t {
  // Normalized 53-bit significand followed by one zero low bit.
  uint64_t significand;
  // Binary exponent of the low significand bit; zero/nonfinite use sentinels.
  int exponent;
  // Sign bit, independent of the normalized significand.
  bool negative;
} iree_math_fma_operand_t;

static iree_math_fma_operand_t iree_math_fma_normalize_f64(double a) {
  uint64_t bits = iree_math_f64_to_bits(a);
  int exponent = (int)(bits >> 52);
  bool negative = (exponent & 0x800) != 0;
  exponent &= 0x7ff;
  if (exponent == 0) {
    bits = iree_math_f64_to_bits(a * 0x1p63);
    exponent = (int)((bits >> 52) & 0x7ff);
    exponent = exponent ? exponent - 63 : 0x800;
  }
  bits &= (UINT64_C(1) << 52) - 1;
  bits |= UINT64_C(1) << 52;
  bits <<= 1;
  exponent -= 0x3ff + 52 + 1;
  return (iree_math_fma_operand_t){bits, exponent, negative};
}

double iree_math_fma_f64_software(double a, double b, double c) {
  // Both top ten bits and the low bit of each significand are zero.
  const iree_math_fma_operand_t a_parts = iree_math_fma_normalize_f64(a);
  const iree_math_fma_operand_t b_parts = iree_math_fma_normalize_f64(b);
  const iree_math_fma_operand_t c_parts = iree_math_fma_normalize_f64(c);
  // Normalization reserves the largest two exponents for nonfinite and zero.
  const int nonfinite_exponent = 0x7ff - 0x3ff - 52 - 1;

  if (a_parts.exponent >= nonfinite_exponent ||
      b_parts.exponent >= nonfinite_exponent) {
    return a * b + c;
  }
  if (c_parts.exponent >= nonfinite_exponent) {
    if (c_parts.exponent > nonfinite_exponent) {  // Zero addend.
      return a * b;
    }
    return c;
  }

  // Multiply exactly into two integer limbs before adding the third operand.
  uint64_t result_high, result_low;
  iree_math_mul_u64_to_u128(a_parts.significand, b_parts.significand,
                            &result_high, &result_low);
  // The product has at most 108 bits, including two zero low bits.

  int exponent = a_parts.exponent + b_parts.exponent;
  int shift = c_parts.exponent - exponent;
  // Align the addend and product, retaining every discarded nonzero bit
  // in a sticky low bit. Their exponent gap may exceed both limbs.
  uint64_t addend_high, addend_low;
  if (shift > 0) {
    if (shift < 64) {
      addend_low = c_parts.significand << shift;
      addend_high = c_parts.significand >> (64 - shift);
    } else {
      addend_low = 0;
      addend_high = c_parts.significand;
      exponent = c_parts.exponent - 64;
      shift -= 64;
      if (shift > 0 && shift < 64) {
        result_low = result_high << (64 - shift) | result_low >> shift |
                     !!(result_low << (64 - shift));
        result_high = result_high >> shift;
      } else if (shift >= 64) {
        result_low = 1;
        result_high = 0;
      }
    }
  } else {
    addend_high = 0;
    shift = -shift;
    if (shift == 0) {
      addend_low = c_parts.significand;
    } else if (shift < 64) {
      addend_low = c_parts.significand >> shift |
                   !!(c_parts.significand << (64 - shift));
    } else {
      addend_low = 1;
    }
  }

  bool negative = a_parts.negative != b_parts.negative;
  bool same_sign = negative == c_parts.negative;
  bool high_nonzero = true;
  if (same_sign) {
    result_low += addend_low;
    result_high += addend_high + (result_low < addend_low);
  } else {
    uint64_t previous_low = result_low;
    result_low -= addend_low;
    result_high = result_high - addend_high - (previous_low < result_low);
    if (result_high >> 63) {
      result_low = -result_low;
      result_high = -result_high - !!result_low;
      negative = !negative;
    }
    high_nonzero = !!result_high;
  }

  // Normalize to 63 significant bits, with a sticky low bit. Signed
  // conversion rounds the normal result; scaling handles subnormal rounding.
  if (high_nonzero) {
    exponent += 64;
    shift = iree_math_count_leading_zeros_u64(result_high) - 1;
    // The aligned high limb has fewer than 63 significant bits.
    result_high = result_high << shift | result_low >> (64 - shift) |
                  !!(result_low << shift);
  } else if (result_low) {
    shift = iree_math_count_leading_zeros_u64(result_low) - 1;
    if (shift < 0) {
      result_high = result_low >> 1 | (result_low & 1);
    } else {
      result_high = result_low << shift;
    }
  } else {
    // Exact cancellation has the same zero sign as ordinary addition.
    return a * b + c;
  }
  exponent -= shift;

  // Magnitude is in [2^62, 2^63), so signed conversion is representable.
  int64_t integer = (int64_t)result_high;
  if (negative) {
    integer = -integer;
  }
  double result = (double)integer;

  if (exponent < -1022 - 62) {
    // The scaled result is subnormal before rounding.
    if (exponent == -1022 - 63) {
      double bias = 0x1p63;
      if (negative) {
        bias = -bias;
      }
      if (result == bias) {
        return negative ? -0x1p-1022 : 0x1p-1022;
      }
      // One bit is lost during scaling. Add a top bit so an inexact
      // conversion rounds only once, then remove that bit exactly.
      if (result_high << 53) {
        integer =
            (int64_t)(result_high >> 1 | (result_high & 1) | UINT64_C(1) << 62);
        if (negative) {
          integer = -integer;
        }
        result = (double)integer;
        result = 2 * result - bias;
      }
    } else {
      // Keep at most 53 significant bits so only the final scale rounds.
      shift = 10;
      integer =
          (int64_t)((result_high >> shift | !!(result_high << (64 - shift)))
                    << shift);
      if (negative) {
        integer = -integer;
      }
      result = (double)integer;
    }
  }
  return iree_math_ldexp_f64(result, exponent);
}
