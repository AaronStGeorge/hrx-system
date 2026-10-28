// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Copyright 2026 The IREE Authors
//
// Developed at SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

#ifndef IREE_MATH_ROOTS_INTEGER_H_
#define IREE_MATH_ROOTS_INTEGER_H_

#include <stdint.h>

// Correctly rounded square root of a positive, finite, nonzero IEEE payload.
// The restoring algorithm is derived from Sun fdlibm e_sqrt.c/e_sqrtf.c (via
// OpenLibm 5fe399749f9276eaa0b8403e507470da05cbbb3f). A 64-bit accumulator
// holds both formats' significands, replacing the original double-word carry
// logic. Only hosts without hardware square root use this path. Its rounding is
// entirely integer and does not depend on the host floating environment.
static inline uint64_t iree_math_sqrt_integer(uint64_t bits,
                                              uint32_t fraction_bits,
                                              int32_t exponent_bias) {
  const uint64_t leading_bit = UINT64_C(1) << fraction_bits;
  int32_t exponent = (int32_t)(bits >> fraction_bits);
  uint64_t remainder = bits & (leading_bit - 1);
  if (exponent == 0) {
    exponent = 1;
    while ((remainder & leading_bit) == 0) {
      remainder <<= 1;
      --exponent;
    }
  } else {
    remainder |= leading_bit;
  }
  exponent -= exponent_bias;
  // Normalize to [1, 4) with an even exponent. Avoid right-shifting a negative
  // signed value when halving the exponent.
  const int32_t odd_exponent = exponent & 1;
  remainder <<= odd_exponent;
  exponent = (exponent - odd_exponent) / 2;
  remainder <<= 1;

  uint64_t root = 0;
  uint64_t doubled_root = 0;
  for (uint64_t bit = leading_bit << 1; bit != 0; bit >>= 1) {
    const uint64_t trial = doubled_root + bit;
    if (trial <= remainder) {
      doubled_root = trial + bit;
      remainder -= trial;
      root += bit;
    }
    remainder <<= 1;
  }
  // The last root bit is a guard bit. A nonzero remainder resolves rounding;
  // the square root of an IEEE value cannot be exactly halfway between two
  // representable values. Exact roots have no discarded bit.
  if (remainder != 0) {
    root += root & 1;
  }
  return (root >> 1) +
         ((uint64_t)(exponent + exponent_bias - 1) << fraction_bits);
}

#endif  // IREE_MATH_ROOTS_INTEGER_H_
