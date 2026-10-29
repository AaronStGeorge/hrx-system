// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exponent/mantissa masking adapted from Sun's ceil, floor, and trunc routines.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

#include "iree/math/rounding.h"

#include <stdbool.h>

#if defined(__SSE4_1__) || (defined(IREE_COMPILER_MSVC) && defined(__AVX__))
#include <smmintrin.h>
#define IREE_MATH_ROUNDING_SSE4_1 1
#else
#define IREE_MATH_ROUNDING_SSE4_1 0
#endif

#include "iree/math/float_bits.h"

typedef enum iree_math_rounding_mode_e {
  IREE_MATH_ROUNDING_DOWN,
  IREE_MATH_ROUNDING_UP,
  IREE_MATH_ROUNDING_ZERO,
  IREE_MATH_ROUNDING_EVEN,
  IREE_MATH_ROUNDING_AWAY,
} iree_math_rounding_mode_t;

// Width and mode are constants at each typed call site. Integer arithmetic
// avoids ambient rounding and intermediate floating-point rounding entirely.
static inline uint64_t iree_math_rounding_bits(uint64_t bits,
                                               unsigned fraction_bits,
                                               unsigned exponent_bias,
                                               unsigned width,
                                               iree_math_rounding_mode_t mode) {
  const uint64_t sign = bits & (UINT64_C(1) << (width - 1));
  const uint64_t magnitude = bits ^ sign;
  const unsigned exponent = (unsigned)(magnitude >> fraction_bits);
  const uint64_t one = (uint64_t)exponent_bias << fraction_bits;
  if (exponent >= exponent_bias + fraction_bits) {
    // Every finite value here is integral. Canonicalize either kind of NaN.
    const uint64_t infinity = (uint64_t)(exponent_bias * 2 + 1)
                              << fraction_bits;
    return magnitude > infinity
               ? infinity | (UINT64_C(1) << (fraction_bits - 1))
               : bits;
  }
  if (exponent < exponent_bias) {
    // Values below one round to signed zero or signed one. At exactly half,
    // even chooses zero and away chooses one.
    const uint64_t half = (uint64_t)(exponent_bias - 1) << fraction_bits;
    bool increment = false;
    switch (mode) {
      case IREE_MATH_ROUNDING_DOWN:
        increment = sign != 0 && magnitude != 0;
        break;
      case IREE_MATH_ROUNDING_UP:
        increment = sign == 0 && magnitude != 0;
        break;
      case IREE_MATH_ROUNDING_EVEN:
        increment = magnitude > half;
        break;
      case IREE_MATH_ROUNDING_AWAY:
        increment = magnitude >= half;
        break;
      case IREE_MATH_ROUNDING_ZERO:
        break;
    }
    return sign | (increment ? one : 0);
  }

  // The low 'shift' bits encode the fractional part. A directed or half-unit
  // bias carries into the integer part before those bits are discarded. The
  // nearest-even bias excludes an exact half unless the retained bit is odd.
  const unsigned shift = exponent_bias + fraction_bits - exponent;
  const uint64_t unit = UINT64_C(1) << shift;
  const uint64_t mask = unit - 1;
  uint64_t rounding_bias = 0;
  switch (mode) {
    case IREE_MATH_ROUNDING_DOWN:
      rounding_bias = sign ? mask : 0;
      break;
    case IREE_MATH_ROUNDING_UP:
      rounding_bias = sign ? 0 : mask;
      break;
    case IREE_MATH_ROUNDING_EVEN:
      rounding_bias = (unit >> 1) - 1 + ((magnitude >> shift) & 1);
      break;
    case IREE_MATH_ROUNDING_AWAY:
      rounding_bias = unit >> 1;
      break;
    case IREE_MATH_ROUNDING_ZERO:
      break;
  }
  return (bits + rounding_bias) & ~mask;
}

#define IREE_MATH_DEFINE_INTEGER_ROUNDING(name, mode)                 \
  IREE_API_EXPORT float iree_math_##name##_f32(float value) {         \
    return iree_math_f32_from_bits((uint32_t)iree_math_rounding_bits( \
        iree_math_f32_to_bits(value), 23, 127, 32, mode));            \
  }                                                                   \
  IREE_API_EXPORT double iree_math_##name##_f64(double value) {       \
    return iree_math_f64_from_bits(iree_math_rounding_bits(           \
        iree_math_f64_to_bits(value), 52, 1023, 64, mode));           \
  }

#if IREE_MATH_ROUNDING_SSE4_1
// The instruction immediate selects a fixed direction, never MXCSR's mode.
// Keep NaN canonicalization off the ordinary result's dependency chain.
#define IREE_MATH_DEFINE_NATIVE_ROUNDING(name, mode)                         \
  IREE_API_EXPORT float iree_math_##name##_f32(float value) {                \
    if (IREE_UNLIKELY(value != value)) return iree_math_f32_canonical_nan(); \
    return _mm_cvtss_f32(_mm_round_ss(_mm_setzero_ps(), _mm_set_ss(value),   \
                                      mode | _MM_FROUND_NO_EXC));            \
  }                                                                          \
  IREE_API_EXPORT double iree_math_##name##_f64(double value) {              \
    if (IREE_UNLIKELY(value != value)) return iree_math_f64_canonical_nan(); \
    return _mm_cvtsd_f64(_mm_round_sd(_mm_setzero_pd(), _mm_set_sd(value),   \
                                      mode | _MM_FROUND_NO_EXC));            \
  }
IREE_MATH_DEFINE_NATIVE_ROUNDING(ceil, _MM_FROUND_TO_POS_INF)
IREE_MATH_DEFINE_NATIVE_ROUNDING(floor, _MM_FROUND_TO_NEG_INF)
IREE_MATH_DEFINE_NATIVE_ROUNDING(trunc, _MM_FROUND_TO_ZERO)
IREE_MATH_DEFINE_NATIVE_ROUNDING(round_even, _MM_FROUND_TO_NEAREST_INT)
#undef IREE_MATH_DEFINE_NATIVE_ROUNDING
#else
IREE_MATH_DEFINE_INTEGER_ROUNDING(ceil, IREE_MATH_ROUNDING_UP)
IREE_MATH_DEFINE_INTEGER_ROUNDING(floor, IREE_MATH_ROUNDING_DOWN)
IREE_MATH_DEFINE_INTEGER_ROUNDING(trunc, IREE_MATH_ROUNDING_ZERO)
IREE_MATH_DEFINE_INTEGER_ROUNDING(round_even, IREE_MATH_ROUNDING_EVEN)
#endif

// SSE4.1 has no ties-away rounding instruction.
IREE_MATH_DEFINE_INTEGER_ROUNDING(round_away, IREE_MATH_ROUNDING_AWAY)
#undef IREE_MATH_DEFINE_INTEGER_ROUNDING
