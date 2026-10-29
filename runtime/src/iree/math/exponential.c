// Copyright (c) 2017-2026, Arm Limited.
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/exponential.h"

#include <stdbool.h>

#include "iree/math/exponential_reduction.h"
#include "iree/math/float_bits.h"
#include "iree/math/logarithm_reduction.h"

//===----------------------------------------------------------------------===//
// Cancellation-preserving natural exponential
//===----------------------------------------------------------------------===//

// Adapted from Sun/OpenLibm revision 5fe399749f9276eaa0b8403e507470da05cbbb3f
// (s_expm1.c and s_expm1f.c), with Naohiko Shimizu's parallel F64 polynomial
// evaluation from glibc revision ef321e23c20eebc6d6fb4044425c00e6df27b05f.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// F32 conversion by Ian Lance Taylor; parallel evaluation by Naohiko Shimizu.

IREE_API_EXPORT float iree_math_expm1_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  const bool negative = (bits & IREE_MATH_F32_SIGN_BIT) != 0;
  if (IREE_UNLIKELY(magnitude >= UINT32_C(0x4195B844))) {
    if (magnitude > IREE_MATH_F32_INFINITY) {
      return iree_math_f32_canonical_nan();
    }
    if (negative) {
      return -1.0f;
    }
    if (magnitude >= UINT32_C(0x42B17218)) {
      return iree_math_f32_from_bits(IREE_MATH_F32_INFINITY);
    }
  }
  // value = exponent*ln(2) + residual + correction. Splitting ln(2) keeps
  // its high-part product exact throughout the finite reconstruction range.
  const float ln2_high = 0x1.62e3p-1f;
  const float ln2_low = 0x1.2fefa2p-17f;
  int exponent = 0;
  float correction = 0.0f;
  if (magnitude > UINT32_C(0x3EB17218)) {
    float high;
    float low;
    if (magnitude < UINT32_C(0x3F851592)) {
      if (negative) {
        exponent = -1;
        high = value + ln2_high;
        low = -ln2_low;
      } else {
        exponent = 1;
        high = value - ln2_high;
        low = ln2_low;
      }
    } else {
      exponent = (int)(0x1.715476p0f * value + (negative ? -0.5f : 0.5f));
      high = value - (float)exponent * ln2_high;
      low = (float)exponent * ln2_low;
    }
    value = high - low;
    correction = (high - value) - low;
  } else if (magnitude < UINT32_C(0x33000000)) {
    return value;
  }

  // Rational approximation on |residual| <= ln(2)/2, retaining the small
  // expm1 result instead of subtracting one from a rounded exponential.
  const float half = 0.5f * value;
  const float squared_half = value * half;
  const float polynomial =
      1.0f + squared_half * (-0x1.1110dp-5f + squared_half * 0x1.9e602p-10f);
  const float denominator = 3.0f - polynomial * half;
  float error = squared_half *
                ((polynomial - denominator) / (6.0f - value * denominator));
  if (exponent == 0) {
    return value - (value * error - squared_half);
  }
  error = value * (error - correction) - correction;
  error -= squared_half;
  if (exponent == -1) {
    return 0.5f * (value - error) - 0.5f;
  }
  if (exponent == 1) {
    return value < -0.25f ? -2.0f * (error - (value + 0.5f))
                          : 1.0f + 2.0f * (value - error);
  }

  const float scale = iree_math_f32_from_bits((uint32_t)(127 + exponent) << 23);
  if (exponent <= -2 || exponent > 56) {
    const float result = 1.0f - (error - value);
    // Split the largest scale so a finite result need not form 2^128.
    return (exponent == 128 ? result * 2.0f * 0x1p127f : result * scale) - 1.0f;
  }
  if (exponent < 23) {
    const float offset = iree_math_f32_from_bits(
        UINT32_C(0x3F800000) - (UINT32_C(0x01000000) >> exponent));
    return (offset - (error - value)) * scale;
  }
  const float inverse_scale =
      iree_math_f32_from_bits((uint32_t)(127 - exponent) << 23);
  return ((value - (error + inverse_scale)) + 1.0f) * scale;
}

IREE_API_EXPORT double iree_math_expm1_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  const uint32_t magnitude_high = (uint32_t)(magnitude >> 32);
  const bool negative = (bits & IREE_MATH_F64_SIGN_BIT) != 0;
  if (IREE_UNLIKELY(magnitude_high >= UINT32_C(0x4043687A))) {
    if (magnitude > IREE_MATH_F64_INFINITY) {
      return iree_math_f64_canonical_nan();
    }
    if (negative) {
      return -1.0;
    }
    if (value > 0x1.62e42fefa39efp9) {
      return iree_math_f64_from_bits(IREE_MATH_F64_INFINITY);
    }
  }
  const double ln2_high = 0x1.62e42feep-1;
  const double ln2_low = 0x1.a39ef35793c76p-33;
  int exponent = 0;
  double correction = 0.0;
  if (magnitude_high > UINT32_C(0x3FD62E42)) {
    double high;
    double low;
    if (magnitude_high < UINT32_C(0x3FF0A2B2)) {
      if (negative) {
        exponent = -1;
        high = value + ln2_high;
        low = -ln2_low;
      } else {
        exponent = 1;
        high = value - ln2_high;
        low = ln2_low;
      }
    } else {
      exponent = (int)(0x1.71547652b82fep0 * value + (negative ? -0.5 : 0.5));
      high = value - (double)exponent * ln2_high;
      low = (double)exponent * ln2_low;
    }
    value = high - low;
    correction = (high - value) - low;
  } else if (magnitude_high < UINT32_C(0x3C900000)) {
    return value;
  }

  const double half = 0.5 * value;
  const double squared_half = value * half;
  // Independent polynomial chains shorten the dependency path while keeping
  // the original approximation coefficients and explicit rounding points.
  const double fourth = squared_half * squared_half;
  const double eighth = fourth * fourth;
  const double first = 1.0 + squared_half * -0x1.11111111110f4p-5;
  const double second =
      0x1.a01a019fe5585p-10 + squared_half * -0x1.4ce199eaadbb7p-14;
  const double third =
      0x1.0cfca86e65239p-18 + squared_half * -0x1.afdb76e09c32dp-23;
  const double polynomial = first + fourth * second + eighth * third;
  const double denominator = 3.0 - polynomial * half;
  double error =
      squared_half * ((polynomial - denominator) / (6.0 - value * denominator));
  if (exponent == 0) {
    return value - (value * error - squared_half);
  }
  error = value * (error - correction) - correction;
  error -= squared_half;
  if (exponent == -1) {
    return 0.5 * (value - error) - 0.5;
  }
  if (exponent == 1) {
    return value < -0.25 ? -2.0 * (error - (value + 0.5))
                         : 1.0 + 2.0 * (value - error);
  }

  const double scale =
      iree_math_f64_from_bits((uint64_t)(1023 + exponent) << 52);
  if (exponent <= -2 || exponent > 56) {
    const double result = 1.0 - (error - value);
    return (exponent == 1024 ? result * 2.0 * 0x1p1023 : result * scale) - 1.0;
  }
  if (exponent < 20) {
    const double offset = iree_math_f64_from_bits(
        (uint64_t)(UINT32_C(0x3FF00000) - (UINT32_C(0x00200000) >> exponent))
        << 32);
    return (offset - (error - value)) * scale;
  }
  const double inverse_scale =
      iree_math_f64_from_bits((uint64_t)(1023 - exponent) << 52);
  return ((value - (error + inverse_scale)) + 1.0) * scale;
}

// The finite F32 kernels derive from Arm Optimized Routines revision
// 67126040cf80f956676fbf473c2d9bebdb475283. IREE supplies explicit exceptional
// shells and freezes the resulting F32 mappings as its own named operations.

//===----------------------------------------------------------------------===//
// Base-two exponential
//===----------------------------------------------------------------------===//

// The low five bits select 2^(i/32). Each entry has i*2^(52-5) subtracted
// from its binary64 representation so adding the complete reduction integer
// simultaneously restores the table value and installs its power-of-two
// exponent.
static const uint64_t iree_math_exp2_f32_table[32] = {
    UINT64_C(0x3FF0000000000000), UINT64_C(0x3FEFD9B0D3158574),
    UINT64_C(0x3FEFB5586CF9890F), UINT64_C(0x3FEF9301D0125B51),
    UINT64_C(0x3FEF72B83C7D517B), UINT64_C(0x3FEF54873168B9AA),
    UINT64_C(0x3FEF387A6E756238), UINT64_C(0x3FEF1E9DF51FDEE1),
    UINT64_C(0x3FEF06FE0A31B715), UINT64_C(0x3FEEF1A7373AA9CB),
    UINT64_C(0x3FEEDEA64C123422), UINT64_C(0x3FEECE086061892D),
    UINT64_C(0x3FEEBFDAD5362A27), UINT64_C(0x3FEEB42B569D4F82),
    UINT64_C(0x3FEEAB07DD485429), UINT64_C(0x3FEEA47EB03A5585),
    UINT64_C(0x3FEEA09E667F3BCD), UINT64_C(0x3FEE9F75E8EC5F74),
    UINT64_C(0x3FEEA11473EB0187), UINT64_C(0x3FEEA589994CCE13),
    UINT64_C(0x3FEEACE5422AA0DB), UINT64_C(0x3FEEB737B0CDC5E5),
    UINT64_C(0x3FEEC49182A3F090), UINT64_C(0x3FEED503B23E255D),
    UINT64_C(0x3FEEE89F995AD3AD), UINT64_C(0x3FEEFF76F2FB5E47),
    UINT64_C(0x3FEF199BDD85529C), UINT64_C(0x3FEF3720DCEF9069),
    UINT64_C(0x3FEF5818DCFBA487), UINT64_C(0x3FEF7C97337B9B5F),
    UINT64_C(0x3FEFA4AFA2A490DA), UINT64_C(0x3FEFD0765B6E4540),
};

static bool iree_math_exp2_f32_handle_special(float value, float* out_result) {
  const uint32_t value_bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude_bits = value_bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude_bits >= UINT32_C(0x42FC0000))) {
    if (magnitude_bits >= IREE_MATH_F32_INFINITY) {
      if (magnitude_bits > IREE_MATH_F32_INFINITY) {
        *out_result = iree_math_f32_canonical_nan();
      } else {
        *out_result = (value_bits & IREE_MATH_F32_SIGN_BIT)
                          ? 0.0f
                          : iree_math_f32_from_bits(IREE_MATH_F32_INFINITY);
      }
      return true;
    }
    // -126 is the smallest input whose result is a normal f32. Inputs below
    // it flush to zero; inputs at least 128 overflow to infinity.
    if (value_bits & IREE_MATH_F32_SIGN_BIT) {
      if (magnitude_bits > UINT32_C(0x42FC0000)) {
        *out_result = 0.0f;
        return true;
      }
    } else if (magnitude_bits >= UINT32_C(0x43000000)) {
      *out_result = iree_math_f32_from_bits(IREE_MATH_F32_INFINITY);
      return true;
    }
  }
  // No f32 value this close to zero can make 2^x round away from one.
  if (IREE_UNLIKELY(magnitude_bits <= UINT32_C(0x33000000))) {
    *out_result = 1.0f;
    return true;
  }
  return false;
}

// Reconstruct 2^(k/32 + residual) after reduction. Natural and base-two
// exponentials use the same table and polynomial with exact power-of-two
// scaling of the residual.
static float iree_math_exp2_f32_reconstruct(double residual,
                                            uint64_t encoded_rounded) {
  uint64_t scale_bits =
      iree_math_exp2_f32_table[encoded_rounded & UINT64_C(31)];
  scale_bits += encoded_rounded << 47;
  const double scale = iree_math_f64_from_bits(scale_bits);

  // Approximate 2^residual over [-1/64, 1/64] and apply the table scale.
  const double residual_squared = residual * residual;
  const double upper_polynomial =
      0x1.c6af84b912394p-5 * residual + 0x1.ebfce50fac4f3p-3;
  const double lower_polynomial = 0x1.62e42ff0c52d6p-1 * residual + 1.0;
  const double polynomial =
      upper_polynomial * residual_squared + lower_polynomial;
  return (float)(polynomial * scale);
}

static float iree_math_exp2_f32_finite(float value) {
  const double input = (double)value;
  const double shift = 0x1.8p47;
  double rounded = input + shift;
  const uint64_t encoded_rounded = iree_math_f64_to_bits(rounded);
  rounded -= shift;
  return iree_math_exp2_f32_reconstruct(input - rounded, encoded_rounded);
}

IREE_API_EXPORT float iree_math_exp2_f32_approx(float value) {
  float result = 0.0f;
  if (iree_math_exp2_f32_handle_special(value, &result)) {
    return result;
  }
  return iree_math_exp2_f32_finite(value);
}

IREE_API_EXPORT float iree_math_exp2_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= UINT32_C(0x43000000))) {
    if (magnitude > IREE_MATH_F32_INFINITY) {
      return iree_math_f32_canonical_nan();
    }
    if (!(bits & IREE_MATH_F32_SIGN_BIT)) {
      return iree_math_f32_from_bits(IREE_MATH_F32_INFINITY);
    }
    if (value <= -150.0f) {
      return 0.0f;
    }
  }
  return iree_math_exp2_f32_finite(value);
}

IREE_API_EXPORT float iree_math_exp_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude >= UINT32_C(0x42B00000))) {
    if (magnitude > IREE_MATH_F32_INFINITY) {
      return iree_math_f32_canonical_nan();
    }
    if (value > 0x1.62e42ep6f) {
      return iree_math_f32_from_bits(IREE_MATH_F32_INFINITY);
    }
    if (value < -0x1.9fe368p6f) {
      return 0.0f;
    }
  }
  // x*32/ln(2) = k + r. Scaling r by 1/32 is exact in this range.
  const double reduced = (0x1.71547652b82fep0 * 32.0) * (double)value;
  double rounded = reduced + 0x1.8p52;
  const uint64_t encoded_rounded = iree_math_f64_to_bits(rounded);
  rounded -= 0x1.8p52;
  return iree_math_exp2_f32_reconstruct((reduced - rounded) * 0x1p-5,
                                        encoded_rounded);
}

//===----------------------------------------------------------------------===//
// Binary64 exponentials
//===----------------------------------------------------------------------===//

// Arm Optimized Routines 2ee4b0d45680a9880412c1d186224b08652c35f6, N=128,
// order-five polynomials, nearest-even shift reduction, and no contraction.
// Upstream bounds are 0.511 ulp without FMA for both functions. Sharing the
// reconstruction table and subnormal correction preserves the same rounding
// mechanism across exp and exp2.
typedef struct iree_math_exp_f64_table_entry_t {
  // Encoded small relative correction to the rounded power of two.
  uint64_t tail_bits;
  // Encoded rounded 2^(i/128), minus i*2^(52-7).
  uint64_t scale_bits;
} iree_math_exp_f64_table_entry_t;

static const iree_math_exp_f64_table_entry_t iree_math_exp_f64_table[128] = {
    {UINT64_C(0x0), UINT64_C(0x3ff0000000000000)},
    {UINT64_C(0x3c9b3b4f1a88bf6e), UINT64_C(0x3feff63da9fb3335)},
    {UINT64_C(0xbc7160139cd8dc5d), UINT64_C(0x3fefec9a3e778061)},
    {UINT64_C(0xbc905e7a108766d1), UINT64_C(0x3fefe315e86e7f85)},
    {UINT64_C(0x3c8cd2523567f613), UINT64_C(0x3fefd9b0d3158574)},
    {UINT64_C(0xbc8bce8023f98efa), UINT64_C(0x3fefd06b29ddf6de)},
    {UINT64_C(0x3c60f74e61e6c861), UINT64_C(0x3fefc74518759bc8)},
    {UINT64_C(0x3c90a3e45b33d399), UINT64_C(0x3fefbe3ecac6f383)},
    {UINT64_C(0x3c979aa65d837b6d), UINT64_C(0x3fefb5586cf9890f)},
    {UINT64_C(0x3c8eb51a92fdeffc), UINT64_C(0x3fefac922b7247f7)},
    {UINT64_C(0x3c3ebe3d702f9cd1), UINT64_C(0x3fefa3ec32d3d1a2)},
    {UINT64_C(0xbc6a033489906e0b), UINT64_C(0x3fef9b66affed31b)},
    {UINT64_C(0xbc9556522a2fbd0e), UINT64_C(0x3fef9301d0125b51)},
    {UINT64_C(0xbc5080ef8c4eea55), UINT64_C(0x3fef8abdc06c31cc)},
    {UINT64_C(0xbc91c923b9d5f416), UINT64_C(0x3fef829aaea92de0)},
    {UINT64_C(0x3c80d3e3e95c55af), UINT64_C(0x3fef7a98c8a58e51)},
    {UINT64_C(0xbc801b15eaa59348), UINT64_C(0x3fef72b83c7d517b)},
    {UINT64_C(0xbc8f1ff055de323d), UINT64_C(0x3fef6af9388c8dea)},
    {UINT64_C(0x3c8b898c3f1353bf), UINT64_C(0x3fef635beb6fcb75)},
    {UINT64_C(0xbc96d99c7611eb26), UINT64_C(0x3fef5be084045cd4)},
    {UINT64_C(0x3c9aecf73e3a2f60), UINT64_C(0x3fef54873168b9aa)},
    {UINT64_C(0xbc8fe782cb86389d), UINT64_C(0x3fef4d5022fcd91d)},
    {UINT64_C(0x3c8a6f4144a6c38d), UINT64_C(0x3fef463b88628cd6)},
    {UINT64_C(0x3c807a05b0e4047d), UINT64_C(0x3fef3f49917ddc96)},
    {UINT64_C(0x3c968efde3a8a894), UINT64_C(0x3fef387a6e756238)},
    {UINT64_C(0x3c875e18f274487d), UINT64_C(0x3fef31ce4fb2a63f)},
    {UINT64_C(0x3c80472b981fe7f2), UINT64_C(0x3fef2b4565e27cdd)},
    {UINT64_C(0xbc96b87b3f71085e), UINT64_C(0x3fef24dfe1f56381)},
    {UINT64_C(0x3c82f7e16d09ab31), UINT64_C(0x3fef1e9df51fdee1)},
    {UINT64_C(0xbc3d219b1a6fbffa), UINT64_C(0x3fef187fd0dad990)},
    {UINT64_C(0x3c8b3782720c0ab4), UINT64_C(0x3fef1285a6e4030b)},
    {UINT64_C(0x3c6e149289cecb8f), UINT64_C(0x3fef0cafa93e2f56)},
    {UINT64_C(0x3c834d754db0abb6), UINT64_C(0x3fef06fe0a31b715)},
    {UINT64_C(0x3c864201e2ac744c), UINT64_C(0x3fef0170fc4cd831)},
    {UINT64_C(0x3c8fdd395dd3f84a), UINT64_C(0x3feefc08b26416ff)},
    {UINT64_C(0xbc86a3803b8e5b04), UINT64_C(0x3feef6c55f929ff1)},
    {UINT64_C(0xbc924aedcc4b5068), UINT64_C(0x3feef1a7373aa9cb)},
    {UINT64_C(0xbc9907f81b512d8e), UINT64_C(0x3feeecae6d05d866)},
    {UINT64_C(0xbc71d1e83e9436d2), UINT64_C(0x3feee7db34e59ff7)},
    {UINT64_C(0xbc991919b3ce1b15), UINT64_C(0x3feee32dc313a8e5)},
    {UINT64_C(0x3c859f48a72a4c6d), UINT64_C(0x3feedea64c123422)},
    {UINT64_C(0xbc9312607a28698a), UINT64_C(0x3feeda4504ac801c)},
    {UINT64_C(0xbc58a78f4817895b), UINT64_C(0x3feed60a21f72e2a)},
    {UINT64_C(0xbc7c2c9b67499a1b), UINT64_C(0x3feed1f5d950a897)},
    {UINT64_C(0x3c4363ed60c2ac11), UINT64_C(0x3feece086061892d)},
    {UINT64_C(0x3c9666093b0664ef), UINT64_C(0x3feeca41ed1d0057)},
    {UINT64_C(0x3c6ecce1daa10379), UINT64_C(0x3feec6a2b5c13cd0)},
    {UINT64_C(0x3c93ff8e3f0f1230), UINT64_C(0x3feec32af0d7d3de)},
    {UINT64_C(0x3c7690cebb7aafb0), UINT64_C(0x3feebfdad5362a27)},
    {UINT64_C(0x3c931dbdeb54e077), UINT64_C(0x3feebcb299fddd0d)},
    {UINT64_C(0xbc8f94340071a38e), UINT64_C(0x3feeb9b2769d2ca7)},
    {UINT64_C(0xbc87deccdc93a349), UINT64_C(0x3feeb6daa2cf6642)},
    {UINT64_C(0xbc78dec6bd0f385f), UINT64_C(0x3feeb42b569d4f82)},
    {UINT64_C(0xbc861246ec7b5cf6), UINT64_C(0x3feeb1a4ca5d920f)},
    {UINT64_C(0x3c93350518fdd78e), UINT64_C(0x3feeaf4736b527da)},
    {UINT64_C(0x3c7b98b72f8a9b05), UINT64_C(0x3feead12d497c7fd)},
    {UINT64_C(0x3c9063e1e21c5409), UINT64_C(0x3feeab07dd485429)},
    {UINT64_C(0x3c34c7855019c6ea), UINT64_C(0x3feea9268a5946b7)},
    {UINT64_C(0x3c9432e62b64c035), UINT64_C(0x3feea76f15ad2148)},
    {UINT64_C(0xbc8ce44a6199769f), UINT64_C(0x3feea5e1b976dc09)},
    {UINT64_C(0xbc8c33c53bef4da8), UINT64_C(0x3feea47eb03a5585)},
    {UINT64_C(0xbc845378892be9ae), UINT64_C(0x3feea34634ccc320)},
    {UINT64_C(0xbc93cedd78565858), UINT64_C(0x3feea23882552225)},
    {UINT64_C(0x3c5710aa807e1964), UINT64_C(0x3feea155d44ca973)},
    {UINT64_C(0xbc93b3efbf5e2228), UINT64_C(0x3feea09e667f3bcd)},
    {UINT64_C(0xbc6a12ad8734b982), UINT64_C(0x3feea012750bdabf)},
    {UINT64_C(0xbc6367efb86da9ee), UINT64_C(0x3fee9fb23c651a2f)},
    {UINT64_C(0xbc80dc3d54e08851), UINT64_C(0x3fee9f7df9519484)},
    {UINT64_C(0xbc781f647e5a3ecf), UINT64_C(0x3fee9f75e8ec5f74)},
    {UINT64_C(0xbc86ee4ac08b7db0), UINT64_C(0x3fee9f9a48a58174)},
    {UINT64_C(0xbc8619321e55e68a), UINT64_C(0x3fee9feb564267c9)},
    {UINT64_C(0x3c909ccb5e09d4d3), UINT64_C(0x3feea0694fde5d3f)},
    {UINT64_C(0xbc7b32dcb94da51d), UINT64_C(0x3feea11473eb0187)},
    {UINT64_C(0x3c94ecfd5467c06b), UINT64_C(0x3feea1ed0130c132)},
    {UINT64_C(0x3c65ebe1abd66c55), UINT64_C(0x3feea2f336cf4e62)},
    {UINT64_C(0xbc88a1c52fb3cf42), UINT64_C(0x3feea427543e1a12)},
    {UINT64_C(0xbc9369b6f13b3734), UINT64_C(0x3feea589994cce13)},
    {UINT64_C(0xbc805e843a19ff1e), UINT64_C(0x3feea71a4623c7ad)},
    {UINT64_C(0xbc94d450d872576e), UINT64_C(0x3feea8d99b4492ed)},
    {UINT64_C(0x3c90ad675b0e8a00), UINT64_C(0x3feeaac7d98a6699)},
    {UINT64_C(0x3c8db72fc1f0eab4), UINT64_C(0x3feeace5422aa0db)},
    {UINT64_C(0xbc65b6609cc5e7ff), UINT64_C(0x3feeaf3216b5448c)},
    {UINT64_C(0x3c7bf68359f35f44), UINT64_C(0x3feeb1ae99157736)},
    {UINT64_C(0xbc93091fa71e3d83), UINT64_C(0x3feeb45b0b91ffc6)},
    {UINT64_C(0xbc5da9b88b6c1e29), UINT64_C(0x3feeb737b0cdc5e5)},
    {UINT64_C(0xbc6c23f97c90b959), UINT64_C(0x3feeba44cbc8520f)},
    {UINT64_C(0xbc92434322f4f9aa), UINT64_C(0x3feebd829fde4e50)},
    {UINT64_C(0xbc85ca6cd7668e4b), UINT64_C(0x3feec0f170ca07ba)},
    {UINT64_C(0x3c71affc2b91ce27), UINT64_C(0x3feec49182a3f090)},
    {UINT64_C(0x3c6dd235e10a73bb), UINT64_C(0x3feec86319e32323)},
    {UINT64_C(0xbc87c50422622263), UINT64_C(0x3feecc667b5de565)},
    {UINT64_C(0x3c8b1c86e3e231d5), UINT64_C(0x3feed09bec4a2d33)},
    {UINT64_C(0xbc91bbd1d3bcbb15), UINT64_C(0x3feed503b23e255d)},
    {UINT64_C(0x3c90cc319cee31d2), UINT64_C(0x3feed99e1330b358)},
    {UINT64_C(0x3c8469846e735ab3), UINT64_C(0x3feede6b5579fdbf)},
    {UINT64_C(0xbc82dfcd978e9db4), UINT64_C(0x3feee36bbfd3f37a)},
    {UINT64_C(0x3c8c1a7792cb3387), UINT64_C(0x3feee89f995ad3ad)},
    {UINT64_C(0xbc907b8f4ad1d9fa), UINT64_C(0x3feeee07298db666)},
    {UINT64_C(0xbc55c3d956dcaeba), UINT64_C(0x3feef3a2b84f15fb)},
    {UINT64_C(0xbc90a40e3da6f640), UINT64_C(0x3feef9728de5593a)},
    {UINT64_C(0xbc68d6f438ad9334), UINT64_C(0x3feeff76f2fb5e47)},
    {UINT64_C(0xbc91eee26b588a35), UINT64_C(0x3fef05b030a1064a)},
    {UINT64_C(0x3c74ffd70a5fddcd), UINT64_C(0x3fef0c1e904bc1d2)},
    {UINT64_C(0xbc91bdfbfa9298ac), UINT64_C(0x3fef12c25bd71e09)},
    {UINT64_C(0x3c736eae30af0cb3), UINT64_C(0x3fef199bdd85529c)},
    {UINT64_C(0x3c8ee3325c9ffd94), UINT64_C(0x3fef20ab5fffd07a)},
    {UINT64_C(0x3c84e08fd10959ac), UINT64_C(0x3fef27f12e57d14b)},
    {UINT64_C(0x3c63cdaf384e1a67), UINT64_C(0x3fef2f6d9406e7b5)},
    {UINT64_C(0x3c676b2c6c921968), UINT64_C(0x3fef3720dcef9069)},
    {UINT64_C(0xbc808a1883ccb5d2), UINT64_C(0x3fef3f0b555dc3fa)},
    {UINT64_C(0xbc8fad5d3ffffa6f), UINT64_C(0x3fef472d4a07897c)},
    {UINT64_C(0xbc900dae3875a949), UINT64_C(0x3fef4f87080d89f2)},
    {UINT64_C(0x3c74a385a63d07a7), UINT64_C(0x3fef5818dcfba487)},
    {UINT64_C(0xbc82919e2040220f), UINT64_C(0x3fef60e316c98398)},
    {UINT64_C(0x3c8e5a50d5c192ac), UINT64_C(0x3fef69e603db3285)},
    {UINT64_C(0x3c843a59ac016b4b), UINT64_C(0x3fef7321f301b460)},
    {UINT64_C(0xbc82d52107b43e1f), UINT64_C(0x3fef7c97337b9b5f)},
    {UINT64_C(0xbc892ab93b470dc9), UINT64_C(0x3fef864614f5a129)},
    {UINT64_C(0x3c74b604603a88d3), UINT64_C(0x3fef902ee78b3ff6)},
    {UINT64_C(0x3c83c5ec519d7271), UINT64_C(0x3fef9a51fbc74c83)},
    {UINT64_C(0xbc8ff7128fd391f0), UINT64_C(0x3fefa4afa2a490da)},
    {UINT64_C(0xbc8dae98e223747d), UINT64_C(0x3fefaf482d8e67f1)},
    {UINT64_C(0x3c8ec3bc41aa2008), UINT64_C(0x3fefba1bee615a27)},
    {UINT64_C(0x3c842b94c3a9eb32), UINT64_C(0x3fefc52b376bba97)},
    {UINT64_C(0x3c8a64a931d185ee), UINT64_C(0x3fefd0765b6e4540)},
    {UINT64_C(0xbc8e37bae43be3ed), UINT64_C(0x3fefdbfdad9cbe14)},
    {UINT64_C(0x3c77893b4d91cd9d), UINT64_C(0x3fefe7c1819e90d8)},
    {UINT64_C(0x3c5305c14160cc89), UINT64_C(0x3feff3c22b8f71f1)},
};

// Scale may have an exponent outside the normal binary64 range. Adjust it
// before constructing a float, then multiply by an exact power of two. The
// positive path's 1009 exponent adjustment covers both exp and exp2 reductions.
static double iree_math_exp_f64_scale(double correction, uint64_t scale_bits,
                                      uint64_t encoded_rounded) {
  if ((encoded_rounded & UINT64_C(0x80000000)) == 0) {
    scale_bits -= UINT64_C(1009) << 52;
    const double scale = iree_math_f64_from_bits(scale_bits);
    return 0x1p1009 * (scale + scale * correction);
  }
  scale_bits += UINT64_C(1022) << 52;
  const double scale = iree_math_f64_from_bits(scale_bits);
  double result = scale + scale * correction;
  if ((iree_math_f64_to_bits(result) & IREE_MATH_F64_MAGNITUDE_MASK) <
      UINT64_C(0x3FF0000000000000)) {
    // Round at the final subnormal precision before scaling, avoiding a second
    // rounding at the normal/subnormal boundary. Keep these additions unfused.
    double low = scale - result + scale * correction;
    const double one = iree_math_f64_from_bits(
        (scale_bits & IREE_MATH_F64_SIGN_BIT) | UINT64_C(0x3FF0000000000000));
    const double high = one + result;
    low = one - high + result + low;
    result = (high + low) - one;
    if (result == 0.0) {
      result = iree_math_f64_from_bits(scale_bits & IREE_MATH_F64_SIGN_BIT);
    }
  }
  return 0x1p-1022 * result;
}

// Ordinary exp and power retain different reduction precision, then share
// the same polynomial and table reconstruction. Sign is encoded separately
// from the reduction integer so the overflow direction remains available.
static double iree_math_exp_f64_reconstruct(double residual,
                                            uint64_t encoded_rounded,
                                            uint64_t sign,
                                            bool extended_scale) {
  const iree_math_exp_f64_table_entry_t entry =
      iree_math_exp_f64_table[encoded_rounded & UINT64_C(127)];
  const uint64_t scale_bits = entry.scale_bits + (encoded_rounded << 45) + sign;
  const double tail = iree_math_f64_from_bits(entry.tail_bits);
  const double squared = residual * residual;
  const double correction =
      tail + residual +
      squared * (0x1.ffffffffffdbdp-2 + residual * 0x1.555555555543cp-3) +
      squared * squared *
          (0x1.55555cf172b91p-5 + residual * 0x1.1111167a4d017p-7);
  if (IREE_UNLIKELY(extended_scale)) {
    return iree_math_exp_f64_scale(correction, scale_bits, encoded_rounded);
  }
  const double scale = iree_math_f64_from_bits(scale_bits);
  return scale + scale * correction;
}

double iree_math_exp_f64_compensated(double high, double low) {
  const double value = high + low;
  const double tail = (high - value) + low;
  const double reduced = (0x1.71547652b82fep0 * 128.0) * value;
  double rounded = reduced + 0x1.8p52;
  const uint64_t encoded_rounded = iree_math_f64_to_bits(rounded);
  rounded -= 0x1.8p52;
  const double residual = value + rounded * -0x1.62e42fefa0000p-8 +
                          rounded * -0x1.cf79abc9e3b3ap-47 + tail;
  return iree_math_exp_f64_reconstruct(residual, encoded_rounded, 0,
                                       value <= -512.0);
}

IREE_API_EXPORT double iree_math_exp_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint32_t exponent = (uint32_t)(bits >> 52) & UINT32_C(0x7FF);
  if (IREE_UNLIKELY(exponent - UINT32_C(0x3C9) >=
                    UINT32_C(0x408) - UINT32_C(0x3C9))) {
    if (exponent < UINT32_C(0x3C9)) {
      return 1.0;
    }
    if (exponent >= UINT32_C(0x409)) {
      if ((bits & IREE_MATH_F64_MAGNITUDE_MASK) > IREE_MATH_F64_INFINITY) {
        return iree_math_f64_canonical_nan();
      }
      return bits & IREE_MATH_F64_SIGN_BIT
                 ? 0.0
                 : iree_math_f64_from_bits(IREE_MATH_F64_INFINITY);
    }
  }
  const double reduced = (0x1.71547652b82fep0 * 128.0) * value;
  double rounded = reduced + 0x1.8p52;
  const uint64_t encoded_rounded = iree_math_f64_to_bits(rounded);
  rounded -= 0x1.8p52;
  const double residual = value + rounded * -0x1.62e42fefa0000p-8 +
                          rounded * -0x1.cf79abc9e3b3ap-47;
  return iree_math_exp_f64_reconstruct(residual, encoded_rounded, 0,
                                       exponent >= UINT32_C(0x408));
}

IREE_API_EXPORT double iree_math_exp2_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude < UINT64_C(0x3C90000000000000))) {
    return 1.0;
  }
  if (IREE_UNLIKELY(magnitude >= UINT64_C(0x4090000000000000))) {
    if (magnitude > IREE_MATH_F64_INFINITY) {
      return iree_math_f64_canonical_nan();
    }
    if (!(bits & IREE_MATH_F64_SIGN_BIT)) {
      return iree_math_f64_from_bits(IREE_MATH_F64_INFINITY);
    }
    if (value <= -1075.0) {
      return 0.0;
    }
  }
  double rounded = value + 0x1.8p45;
  const uint64_t encoded_rounded = iree_math_f64_to_bits(rounded);
  rounded -= 0x1.8p45;
  const double residual = value - rounded;
  const iree_math_exp_f64_table_entry_t entry =
      iree_math_exp_f64_table[encoded_rounded & UINT64_C(127)];
  const uint64_t scale_bits = entry.scale_bits + (encoded_rounded << 45);
  const double tail = iree_math_f64_from_bits(entry.tail_bits);
  const double squared = residual * residual;
  const double correction =
      tail + residual * 0x1.62e42fefa39efp-1 +
      squared * (0x1.ebfbdff82c424p-3 + residual * 0x1.c6b08d70cf4b5p-5) +
      squared * squared *
          (0x1.3b2abd24650ccp-7 + residual * 0x1.5d7e09b4e3a84p-10);
  if (IREE_UNLIKELY(magnitude > UINT64_C(0x408D000000000000))) {
    return iree_math_exp_f64_scale(correction, scale_bits, encoded_rounded);
  }
  const double scale = iree_math_f64_from_bits(scale_bits);
  return scale + scale * correction;
}

//===----------------------------------------------------------------------===//
// General real powers
//===----------------------------------------------------------------------===//

// Arm Optimized Routines 2ee4b0d45680a9880412c1d186224b08652c35f6,
// powf/pow with unfused evaluation. F32 keeps the logarithm in F64 until
// exponential reconstruction; F64 retains a compensated high part and tail.
// The upstream error bounds are 0.82 ulp and 0.54 ulp, respectively.

typedef enum iree_math_pow_integer_class_e {
  IREE_MATH_POW_FRACTIONAL = 0,
  IREE_MATH_POW_ODD = 1,
  IREE_MATH_POW_EVEN = 2,
} iree_math_pow_integer_class_t;

// Classifies a finite nonzero exponent from its magnitude bits. The cold
// exceptional path uses this to determine negative-base validity and sign.
static iree_math_pow_integer_class_t iree_math_pow_integer_class(
    uint64_t magnitude, uint32_t fraction_bits, uint32_t bias) {
  const uint32_t exponent = (uint32_t)(magnitude >> fraction_bits);
  if (exponent < bias) {
    return IREE_MATH_POW_FRACTIONAL;
  }
  if (exponent > bias + fraction_bits) {
    return IREE_MATH_POW_EVEN;
  }
  const uint32_t shift = bias + fraction_bits - exponent;
  const uint64_t integer_bit = UINT64_C(1) << shift;
  if (magnitude & (integer_bit - 1)) {
    return IREE_MATH_POW_FRACTIONAL;
  }
  return magnitude & integer_bit ? IREE_MATH_POW_ODD : IREE_MATH_POW_EVEN;
}

static double iree_math_pow_f32_log2(uint32_t bits) {
  const uint32_t adjusted = bits - UINT32_C(0x3F330000);
  const uint32_t index = (adjusted >> 19) & UINT32_C(15);
  const uint32_t exponent_code = adjusted >> 23;
  const int32_t exponent = exponent_code >= UINT32_C(256)
                               ? (int32_t)exponent_code - 512
                               : (int32_t)exponent_code;
  const uint32_t normalized_bits = bits - (adjusted & UINT32_C(0xFF800000));
  const double normalized = (double)iree_math_f32_from_bits(normalized_bits);
  const iree_math_log2_f32_table_entry_t entry =
      iree_math_log2_f32_table[index];
  const double residual = normalized * entry.inverse_center - 1.0;
  const double logarithm = entry.log2_center + (double)exponent;
  const double squared = residual * residual;
  const double fourth = squared * squared;
  const double upper = 0x1.27616c9496e0bp-2 * residual - 0x1.71969a075c67ap-2;
  const double middle = 0x1.ec70a6ca7baddp-2 * residual - 0x1.7154748bef6c8p-1;
  const double lower = 0x1.71547652ab82bp0 * residual + logarithm;
  return upper * fourth + (middle * squared + lower);
}

IREE_API_EXPORT float iree_math_pow_f32(float base, float exponent) {
  uint32_t base_bits = iree_math_f32_to_bits(base);
  const uint32_t exponent_bits = iree_math_f32_to_bits(exponent);
  const uint32_t exponent_magnitude =
      exponent_bits & IREE_MATH_F32_MAGNITUDE_MASK;
  uint32_t sign = 0;
  if (IREE_UNLIKELY(base_bits - IREE_MATH_F32_MIN_NORMAL >=
                        IREE_MATH_F32_INFINITY - IREE_MATH_F32_MIN_NORMAL ||
                    exponent_magnitude == 0 ||
                    exponent_magnitude >= IREE_MATH_F32_INFINITY)) {
    const uint32_t magnitude = base_bits & IREE_MATH_F32_MAGNITUDE_MASK;
    if (exponent_magnitude == 0 || base_bits == UINT32_C(0x3F800000)) {
      return 1.0f;
    }
    if (magnitude > IREE_MATH_F32_INFINITY ||
        exponent_magnitude > IREE_MATH_F32_INFINITY) {
      return iree_math_f32_canonical_nan();
    }
    if (exponent_magnitude == IREE_MATH_F32_INFINITY) {
      if (magnitude == UINT32_C(0x3F800000)) {
        return 1.0f;
      }
      return (magnitude < UINT32_C(0x3F800000)) ==
                     !(exponent_bits & IREE_MATH_F32_SIGN_BIT)
                 ? 0.0f
                 : iree_math_f32_from_bits(IREE_MATH_F32_INFINITY);
    }
    const iree_math_pow_integer_class_t integer_class =
        iree_math_pow_integer_class(exponent_magnitude, 23, 127);
    if (base_bits & IREE_MATH_F32_SIGN_BIT) {
      if (integer_class == IREE_MATH_POW_ODD) {
        sign = IREE_MATH_F32_SIGN_BIT;
      }
      base_bits = magnitude;
    }
    if (magnitude == 0 || magnitude == IREE_MATH_F32_INFINITY) {
      const bool infinite = (magnitude == IREE_MATH_F32_INFINITY) !=
                            ((exponent_bits & IREE_MATH_F32_SIGN_BIT) != 0);
      return iree_math_f32_from_bits(sign |
                                     (infinite ? IREE_MATH_F32_INFINITY : 0));
    }
    if ((iree_math_f32_to_bits(base) & IREE_MATH_F32_SIGN_BIT) &&
        integer_class == IREE_MATH_POW_FRACTIONAL) {
      return iree_math_f32_canonical_nan();
    }
    if (base_bits < IREE_MATH_F32_MIN_NORMAL) {
      base_bits =
          iree_math_f32_to_bits(base * 0x1p23f) & IREE_MATH_F32_MAGNITUDE_MASK;
      base_bits -= UINT32_C(23) << 23;
    }
  }
  const double product = (double)exponent * iree_math_pow_f32_log2(base_bits);
  if (IREE_UNLIKELY((iree_math_f64_to_bits(product) >> 47 & UINT64_C(0xFFFF)) >=
                    (UINT64_C(0x405F800000000000) >> 47))) {
    if (product <= -150.0) {
      return iree_math_f32_from_bits(sign);
    }
    if (product > 0x1.fffffffd1d571p6) {
      return iree_math_f32_from_bits(sign | IREE_MATH_F32_INFINITY);
    }
  }
  double rounded = product + 0x1.8p47;
  uint64_t encoded_rounded = iree_math_f64_to_bits(rounded);
  rounded -= 0x1.8p47;
  // This bias becomes the F64 sign bit during the shared scale reconstruction.
  encoded_rounded += (uint64_t)sign >> 15;
  return iree_math_exp2_f32_reconstruct(product - rounded, encoded_rounded);
}

typedef struct iree_math_pow_log_f64_entry_t {
  // Quantized reciprocal of the reduction center, with at most nine bits.
  double inverse_center;
  // Center logarithm rounded to a multiple of 2^-43.
  double logarithm;
  // Remainder of the center logarithm after rounding the high part.
  double tail;
} iree_math_pow_log_f64_entry_t;

// Quantized centers make the split residual exact without FMA. Keeping the
// logarithm tail retains about fifteen bits beyond ordinary F64 precision.
static const iree_math_pow_log_f64_entry_t iree_math_pow_log_f64_table[128] = {
    {0x1.6a00000000000p+0, -0x1.62c82f2b9c800p-2, 0x1.ab42428375680p-48},
    {0x1.6800000000000p+0, -0x1.5d1bdbf580800p-2, -0x1.ca508d8e0f720p-46},
    {0x1.6600000000000p+0, -0x1.5767717455800p-2, -0x1.362a4d5b6506dp-45},
    {0x1.6400000000000p+0, -0x1.51aad872df800p-2, -0x1.684e49eb067d5p-49},
    {0x1.6200000000000p+0, -0x1.4be5f95777800p-2, -0x1.41b6993293ee0p-47},
    {0x1.6000000000000p+0, -0x1.4618bc21c6000p-2, 0x1.3d82f484c84ccp-46},
    {0x1.5e00000000000p+0, -0x1.404308686a800p-2, 0x1.c42f3ed820b3ap-50},
    {0x1.5c00000000000p+0, -0x1.3a64c55694800p-2, 0x1.0b1c686519460p-45},
    {0x1.5a00000000000p+0, -0x1.347dd9a988000p-2, 0x1.5594dd4c58092p-45},
    {0x1.5800000000000p+0, -0x1.2e8e2bae12000p-2, 0x1.67b1e99b72bd8p-45},
    {0x1.5600000000000p+0, -0x1.2895a13de8800p-2, 0x1.5ca14b6cfb03fp-46},
    {0x1.5600000000000p+0, -0x1.2895a13de8800p-2, 0x1.5ca14b6cfb03fp-46},
    {0x1.5400000000000p+0, -0x1.22941fbcf7800p-2, -0x1.65a242853da76p-46},
    {0x1.5200000000000p+0, -0x1.1c898c1699800p-2, -0x1.fafbc68e75404p-46},
    {0x1.5000000000000p+0, -0x1.1675cababa800p-2, 0x1.f1fc63382a8f0p-46},
    {0x1.4e00000000000p+0, -0x1.1058bf9ae4800p-2, -0x1.6a8c4fd055a66p-45},
    {0x1.4c00000000000p+0, -0x1.0a324e2739000p-2, -0x1.c6bee7ef4030ep-47},
    {0x1.4a00000000000p+0, -0x1.0402594b4d000p-2, -0x1.036b89ef42d7fp-48},
    {0x1.4a00000000000p+0, -0x1.0402594b4d000p-2, -0x1.036b89ef42d7fp-48},
    {0x1.4800000000000p+0, -0x1.fb9186d5e4000p-3, 0x1.d572aab993c87p-47},
    {0x1.4600000000000p+0, -0x1.ef0adcbdc6000p-3, 0x1.b26b79c86af24p-45},
    {0x1.4400000000000p+0, -0x1.e27076e2af000p-3, -0x1.72f4f543fff10p-46},
    {0x1.4200000000000p+0, -0x1.d5c216b4fc000p-3, 0x1.1ba91bbca681bp-45},
    {0x1.4000000000000p+0, -0x1.c8ff7c79aa000p-3, 0x1.7794f689f8434p-45},
    {0x1.4000000000000p+0, -0x1.c8ff7c79aa000p-3, 0x1.7794f689f8434p-45},
    {0x1.3e00000000000p+0, -0x1.bc286742d9000p-3, 0x1.94eb0318bb78fp-46},
    {0x1.3c00000000000p+0, -0x1.af3c94e80c000p-3, 0x1.a4e633fcd9066p-52},
    {0x1.3a00000000000p+0, -0x1.a23bc1fe2b000p-3, -0x1.58c64dc46c1eap-45},
    {0x1.3a00000000000p+0, -0x1.a23bc1fe2b000p-3, -0x1.58c64dc46c1eap-45},
    {0x1.3800000000000p+0, -0x1.9525a9cf45000p-3, -0x1.ad1d904c1d4e3p-45},
    {0x1.3600000000000p+0, -0x1.87fa06520d000p-3, 0x1.bbdbf7fdbfa09p-45},
    {0x1.3400000000000p+0, -0x1.7ab890210e000p-3, 0x1.bdb9072534a58p-45},
    {0x1.3400000000000p+0, -0x1.7ab890210e000p-3, 0x1.bdb9072534a58p-45},
    {0x1.3200000000000p+0, -0x1.6d60fe719d000p-3, -0x1.0e46aa3b2e266p-46},
    {0x1.3000000000000p+0, -0x1.5ff3070a79000p-3, -0x1.e9e439f105039p-46},
    {0x1.3000000000000p+0, -0x1.5ff3070a79000p-3, -0x1.e9e439f105039p-46},
    {0x1.2e00000000000p+0, -0x1.526e5e3a1b000p-3, -0x1.0de8b90075b8fp-45},
    {0x1.2c00000000000p+0, -0x1.44d2b6ccb8000p-3, 0x1.70cc16135783cp-46},
    {0x1.2c00000000000p+0, -0x1.44d2b6ccb8000p-3, 0x1.70cc16135783cp-46},
    {0x1.2a00000000000p+0, -0x1.371fc201e9000p-3, 0x1.178864d27543ap-48},
    {0x1.2800000000000p+0, -0x1.29552f81ff000p-3, -0x1.48d301771c408p-45},
    {0x1.2600000000000p+0, -0x1.1b72ad52f6000p-3, -0x1.e80a41811a396p-45},
    {0x1.2600000000000p+0, -0x1.1b72ad52f6000p-3, -0x1.e80a41811a396p-45},
    {0x1.2400000000000p+0, -0x1.0d77e7cd09000p-3, 0x1.a699688e85bf4p-47},
    {0x1.2400000000000p+0, -0x1.0d77e7cd09000p-3, 0x1.a699688e85bf4p-47},
    {0x1.2200000000000p+0, -0x1.fec9131dbe000p-4, -0x1.575545ca333f2p-45},
    {0x1.2000000000000p+0, -0x1.e27076e2b0000p-4, 0x1.a342c2af0003cp-45},
    {0x1.2000000000000p+0, -0x1.e27076e2b0000p-4, 0x1.a342c2af0003cp-45},
    {0x1.1e00000000000p+0, -0x1.c5e548f5bc000p-4, -0x1.d0c57585fbe06p-46},
    {0x1.1c00000000000p+0, -0x1.a926d3a4ae000p-4, 0x1.53935e85baac8p-45},
    {0x1.1c00000000000p+0, -0x1.a926d3a4ae000p-4, 0x1.53935e85baac8p-45},
    {0x1.1a00000000000p+0, -0x1.8c345d631a000p-4, 0x1.37c294d2f5668p-46},
    {0x1.1a00000000000p+0, -0x1.8c345d631a000p-4, 0x1.37c294d2f5668p-46},
    {0x1.1800000000000p+0, -0x1.6f0d28ae56000p-4, -0x1.69737c93373dap-45},
    {0x1.1600000000000p+0, -0x1.51b073f062000p-4, 0x1.f025b61c65e57p-46},
    {0x1.1600000000000p+0, -0x1.51b073f062000p-4, 0x1.f025b61c65e57p-46},
    {0x1.1400000000000p+0, -0x1.341d7961be000p-4, 0x1.c5edaccf913dfp-45},
    {0x1.1400000000000p+0, -0x1.341d7961be000p-4, 0x1.c5edaccf913dfp-45},
    {0x1.1200000000000p+0, -0x1.16536eea38000p-4, 0x1.47c5e768fa309p-46},
    {0x1.1000000000000p+0, -0x1.f0a30c0118000p-5, 0x1.d599e83368e91p-45},
    {0x1.1000000000000p+0, -0x1.f0a30c0118000p-5, 0x1.d599e83368e91p-45},
    {0x1.0e00000000000p+0, -0x1.b42dd71198000p-5, 0x1.c827ae5d6704cp-46},
    {0x1.0e00000000000p+0, -0x1.b42dd71198000p-5, 0x1.c827ae5d6704cp-46},
    {0x1.0c00000000000p+0, -0x1.77458f632c000p-5, -0x1.cfc4634f2a1eep-45},
    {0x1.0c00000000000p+0, -0x1.77458f632c000p-5, -0x1.cfc4634f2a1eep-45},
    {0x1.0a00000000000p+0, -0x1.39e87b9fec000p-5, 0x1.502b7f526feaap-48},
    {0x1.0a00000000000p+0, -0x1.39e87b9fec000p-5, 0x1.502b7f526feaap-48},
    {0x1.0800000000000p+0, -0x1.f829b0e780000p-6, -0x1.980267c7e09e4p-45},
    {0x1.0800000000000p+0, -0x1.f829b0e780000p-6, -0x1.980267c7e09e4p-45},
    {0x1.0600000000000p+0, -0x1.7b91b07d58000p-6, -0x1.88d5493faa639p-45},
    {0x1.0400000000000p+0, -0x1.fc0a8b0fc0000p-7, -0x1.f1e7cf6d3a69cp-50},
    {0x1.0400000000000p+0, -0x1.fc0a8b0fc0000p-7, -0x1.f1e7cf6d3a69cp-50},
    {0x1.0200000000000p+0, -0x1.fe02a6b100000p-8, -0x1.9e23f0dda40e4p-46},
    {0x1.0200000000000p+0, -0x1.fe02a6b100000p-8, -0x1.9e23f0dda40e4p-46},
    {0x1.0000000000000p+0, 0x0.0000000000000p+0, 0x0.0000000000000p+0},
    {0x1.0000000000000p+0, 0x0.0000000000000p+0, 0x0.0000000000000p+0},
    {0x1.fc00000000000p-1, 0x1.0101575890000p-7, -0x1.0c76b999d2be8p-46},
    {0x1.f800000000000p-1, 0x1.0205658938000p-6, -0x1.3dc5b06e2f7d2p-45},
    {0x1.f400000000000p-1, 0x1.8492528c90000p-6, -0x1.aa0ba325a0c34p-45},
    {0x1.f000000000000p-1, 0x1.0415d89e74000p-5, 0x1.111c05cf1d753p-47},
    {0x1.ec00000000000p-1, 0x1.466aed42e0000p-5, -0x1.c167375bdfd28p-45},
    {0x1.e800000000000p-1, 0x1.894aa149fc000p-5, -0x1.97995d05a267dp-46},
    {0x1.e400000000000p-1, 0x1.ccb73cdddc000p-5, -0x1.a68f247d82807p-46},
    {0x1.e200000000000p-1, 0x1.eea31c006c000p-5, -0x1.e113e4fc93b7bp-47},
    {0x1.de00000000000p-1, 0x1.1973bd1466000p-4, -0x1.5325d560d9e9bp-45},
    {0x1.da00000000000p-1, 0x1.3bdf5a7d1e000p-4, 0x1.cc85ea5db4ed7p-45},
    {0x1.d600000000000p-1, 0x1.5e95a4d97a000p-4, -0x1.c69063c5d1d1ep-45},
    {0x1.d400000000000p-1, 0x1.700d30aeac000p-4, 0x1.c1e8da99ded32p-49},
    {0x1.d000000000000p-1, 0x1.9335e5d594000p-4, 0x1.3115c3abd47dap-45},
    {0x1.cc00000000000p-1, 0x1.b6ac88dad6000p-4, -0x1.390802bf768e5p-46},
    {0x1.ca00000000000p-1, 0x1.c885801bc4000p-4, 0x1.646d1c65aacd3p-45},
    {0x1.c600000000000p-1, 0x1.ec739830a2000p-4, -0x1.dc068afe645e0p-45},
    {0x1.c400000000000p-1, 0x1.fe89139dbe000p-4, -0x1.534d64fa10afdp-45},
    {0x1.c000000000000p-1, 0x1.1178e8227e000p-3, 0x1.1ef78ce2d07f2p-45},
    {0x1.be00000000000p-1, 0x1.1aa2b7e23f000p-3, 0x1.ca78e44389934p-45},
    {0x1.ba00000000000p-1, 0x1.2d1610c868000p-3, 0x1.39d6ccb81b4a1p-47},
    {0x1.b800000000000p-1, 0x1.365fcb0159000p-3, 0x1.62fa8234b7289p-51},
    {0x1.b400000000000p-1, 0x1.4913d8333b000p-3, 0x1.5837954fdb678p-45},
    {0x1.b200000000000p-1, 0x1.527e5e4a1b000p-3, 0x1.633e8e5697dc7p-45},
    {0x1.ae00000000000p-1, 0x1.6574ebe8c1000p-3, 0x1.9cf8b2c3c2e78p-46},
    {0x1.ac00000000000p-1, 0x1.6f0128b757000p-3, -0x1.5118de59c21e1p-45},
    {0x1.aa00000000000p-1, 0x1.7898d85445000p-3, -0x1.c661070914305p-46},
    {0x1.a600000000000p-1, 0x1.8beafeb390000p-3, -0x1.73d54aae92cd1p-47},
    {0x1.a400000000000p-1, 0x1.95a5adcf70000p-3, 0x1.7f22858a0ff6fp-47},
    {0x1.a000000000000p-1, 0x1.a93ed3c8ae000p-3, -0x1.8724350562169p-45},
    {0x1.9e00000000000p-1, 0x1.b31d8575bd000p-3, -0x1.c358d4eace1aap-47},
    {0x1.9c00000000000p-1, 0x1.bd087383be000p-3, -0x1.d4bc4595412b6p-45},
    {0x1.9a00000000000p-1, 0x1.c6ffbc6f01000p-3, -0x1.1ec72c5962bd2p-48},
    {0x1.9600000000000p-1, 0x1.db13db0d49000p-3, -0x1.aff2af715b035p-45},
    {0x1.9400000000000p-1, 0x1.e530effe71000p-3, 0x1.212276041f430p-51},
    {0x1.9200000000000p-1, 0x1.ef5ade4dd0000p-3, -0x1.a211565bb8e11p-51},
    {0x1.9000000000000p-1, 0x1.f991c6cb3b000p-3, 0x1.bcbecca0cdf30p-46},
    {0x1.8c00000000000p-1, 0x1.07138604d5800p-2, 0x1.89cdb16ed4e91p-48},
    {0x1.8a00000000000p-1, 0x1.0c42d67616000p-2, 0x1.7188b163ceae9p-45},
    {0x1.8800000000000p-1, 0x1.1178e8227e800p-2, -0x1.c210e63a5f01cp-45},
    {0x1.8600000000000p-1, 0x1.16b5ccbacf800p-2, 0x1.b9acdf7a51681p-45},
    {0x1.8400000000000p-1, 0x1.1bf99635a6800p-2, 0x1.ca6ed5147bdb7p-45},
    {0x1.8200000000000p-1, 0x1.214456d0eb800p-2, 0x1.a87deba46baeap-47},
    {0x1.7e00000000000p-1, 0x1.2bef07cdc9000p-2, 0x1.a9cfa4a5004f4p-45},
    {0x1.7c00000000000p-1, 0x1.314f1e1d36000p-2, -0x1.8e27ad3213cb8p-45},
    {0x1.7a00000000000p-1, 0x1.36b6776be1000p-2, 0x1.16ecdb0f177c8p-46},
    {0x1.7800000000000p-1, 0x1.3c25277333000p-2, 0x1.83b54b606bd5cp-46},
    {0x1.7600000000000p-1, 0x1.419b423d5e800p-2, 0x1.8e436ec90e09dp-47},
    {0x1.7400000000000p-1, 0x1.4718dc271c800p-2, -0x1.f27ce0967d675p-45},
    {0x1.7200000000000p-1, 0x1.4c9e09e173000p-2, -0x1.e20891b0ad8a4p-45},
    {0x1.7000000000000p-1, 0x1.522ae0738a000p-2, 0x1.ebe708164c759p-45},
    {0x1.6e00000000000p-1, 0x1.57bf753c8d000p-2, 0x1.fadedee5d40efp-46},
    {0x1.6c00000000000p-1, 0x1.5d5bddf596000p-2, -0x1.a0b2a08a465dcp-47},
};

static double iree_math_pow_f64_log(uint64_t bits, double* out_tail) {
  const uint64_t adjusted = bits - UINT64_C(0x3FE6955500000000);
  const uint32_t index = (uint32_t)(adjusted >> 45) & UINT32_C(127);
  const uint32_t exponent_code = (uint32_t)(adjusted >> 52);
  const int32_t exponent = exponent_code >= UINT32_C(2048)
                               ? (int32_t)exponent_code - 4096
                               : (int32_t)exponent_code;
  const uint64_t normalized_bits =
      bits - (adjusted & UINT64_C(0xFFF0000000000000));
  const double normalized = iree_math_f64_from_bits(normalized_bits);
  const iree_math_pow_log_f64_entry_t entry =
      iree_math_pow_log_f64_table[index];
  const double high_normalized = iree_math_f64_from_bits(
      (normalized_bits + UINT64_C(0x80000000)) & UINT64_C(0xFFFFFFFF00000000));
  const double low_normalized = normalized - high_normalized;
  const double high_residual = high_normalized * entry.inverse_center - 1.0;
  const double low_residual = low_normalized * entry.inverse_center;
  const double residual = high_residual + low_residual;
  const double logarithm =
      (double)exponent * 0x1.62e42fefa3800p-1 + entry.logarithm;
  const double linear = logarithm + residual;
  const double low_logarithm =
      (double)exponent * 0x1.ef35793c76730p-45 + entry.tail;
  const double low_linear = logarithm - linear + residual;
  const double negative_half = -0.5 * residual;
  const double quadratic = residual * negative_half;
  const double cubic = residual * quadratic;
  const double high_half = -0.5 * high_residual;
  const double high_quadratic = high_residual * high_half;
  const double high = linear + high_quadratic;
  const double low_quadratic = low_residual * (negative_half + high_half);
  const double low_high = linear - high + high_quadratic;
  const double polynomial =
      cubic *
      ((-2.0 * 0x1.555555555556p-2) +
       residual * (-2.0 * -0x1.0000000000006p-2) +
       quadratic * ((4.0 * 0x1.999999959554ep-3) +
                    residual * (4.0 * -0x1.555555529a47ap-3) +
                    quadratic * ((-8.0 * 0x1.2495b9b4845e9p-3) +
                                 residual * (-8.0 * -0x1.0002b8b263fc3p-3))));
  const double low =
      low_logarithm + low_linear + low_quadratic + low_high + polynomial;
  const double result = high + low;
  *out_tail = high - result + low;
  return result;
}

IREE_API_EXPORT double iree_math_pow_f64(double base, double exponent) {
  uint64_t base_bits = iree_math_f64_to_bits(base);
  const uint64_t exponent_bits = iree_math_f64_to_bits(exponent);
  const uint64_t exponent_magnitude =
      exponent_bits & IREE_MATH_F64_MAGNITUDE_MASK;
  uint32_t base_top = (uint32_t)(base_bits >> 52);
  const uint32_t exponent_top = (uint32_t)(exponent_magnitude >> 52);
  uint64_t sign = 0;
  if (IREE_UNLIKELY(base_top - UINT32_C(1) >= UINT32_C(0x7FE) ||
                    exponent_top - UINT32_C(0x3BE) >=
                        UINT32_C(0x43E) - UINT32_C(0x3BE))) {
    const uint64_t magnitude = base_bits & IREE_MATH_F64_MAGNITUDE_MASK;
    if (exponent_magnitude == 0 || base_bits == UINT64_C(0x3FF0000000000000)) {
      return 1.0;
    }
    if (magnitude > IREE_MATH_F64_INFINITY ||
        exponent_magnitude > IREE_MATH_F64_INFINITY) {
      return iree_math_f64_canonical_nan();
    }
    if (exponent_magnitude == IREE_MATH_F64_INFINITY) {
      if (magnitude == UINT64_C(0x3FF0000000000000)) {
        return 1.0;
      }
      return (magnitude < UINT64_C(0x3FF0000000000000)) ==
                     !(exponent_bits & IREE_MATH_F64_SIGN_BIT)
                 ? 0.0
                 : iree_math_f64_from_bits(IREE_MATH_F64_INFINITY);
    }
    const iree_math_pow_integer_class_t integer_class =
        iree_math_pow_integer_class(exponent_magnitude, 52, 1023);
    if (base_bits & IREE_MATH_F64_SIGN_BIT) {
      if (integer_class == IREE_MATH_POW_ODD) {
        sign = IREE_MATH_F64_SIGN_BIT;
      }
      base_bits = magnitude;
      base_top &= UINT32_C(0x7FF);
    }
    if (magnitude == 0 || magnitude == IREE_MATH_F64_INFINITY) {
      const bool infinite = (magnitude == IREE_MATH_F64_INFINITY) !=
                            ((exponent_bits & IREE_MATH_F64_SIGN_BIT) != 0);
      return iree_math_f64_from_bits(sign |
                                     (infinite ? IREE_MATH_F64_INFINITY : 0));
    }
    if ((iree_math_f64_to_bits(base) & IREE_MATH_F64_SIGN_BIT) &&
        integer_class == IREE_MATH_POW_FRACTIONAL) {
      return iree_math_f64_canonical_nan();
    }
    if (exponent_top - UINT32_C(0x3BE) >= UINT32_C(0x43E) - UINT32_C(0x3BE)) {
      if (magnitude == UINT64_C(0x3FF0000000000000) ||
          exponent_top < UINT32_C(0x3BE)) {
        return 1.0;
      }
      return (magnitude > UINT64_C(0x3FF0000000000000)) ==
                     !(exponent_bits & IREE_MATH_F64_SIGN_BIT)
                 ? iree_math_f64_from_bits(IREE_MATH_F64_INFINITY)
                 : 0.0;
    }
    if (base_top == 0) {
      base_bits =
          iree_math_f64_to_bits(base * 0x1p52) & IREE_MATH_F64_MAGNITUDE_MASK;
      base_bits -= UINT64_C(52) << 52;
    }
  }
  double logarithm_tail;
  const double logarithm = iree_math_pow_f64_log(base_bits, &logarithm_tail);
  // Each high part has 27 low bits cleared, so their product is exact. The
  // remaining products retain the low part before exponential reconstruction.
  const double high_exponent =
      iree_math_f64_from_bits(exponent_bits & UINT64_C(0xFFFFFFFFF8000000));
  const double low_exponent = exponent - high_exponent;
  const double high_logarithm = iree_math_f64_from_bits(
      iree_math_f64_to_bits(logarithm) & UINT64_C(0xFFFFFFFFF8000000));
  const double low_logarithm = logarithm - high_logarithm + logarithm_tail;
  const double high = high_exponent * high_logarithm;
  const double low = low_exponent * high_logarithm + exponent * low_logarithm;
  const uint64_t high_bits = iree_math_f64_to_bits(high);
  const uint32_t high_top = (uint32_t)(high_bits >> 52) & UINT32_C(0x7FF);
  if (IREE_UNLIKELY(high_top < UINT32_C(0x3C9))) {
    return iree_math_f64_from_bits(sign | UINT64_C(0x3FF0000000000000));
  }
  if (IREE_UNLIKELY(high_top >= UINT32_C(0x409))) {
    return iree_math_f64_from_bits(
        sign |
        ((high_bits & IREE_MATH_F64_SIGN_BIT) ? 0 : IREE_MATH_F64_INFINITY));
  }
  const double reduced = (0x1.71547652b82fep0 * 128.0) * high;
  double rounded = reduced + 0x1.8p52;
  const uint64_t encoded_rounded = iree_math_f64_to_bits(rounded);
  rounded -= 0x1.8p52;
  const double residual = high + rounded * -0x1.62e42fefa0000p-8 +
                          rounded * -0x1.cf79abc9e3b3ap-47 + low;
  return iree_math_exp_f64_reconstruct(residual, encoded_rounded, sign,
                                       high_top >= UINT32_C(0x408));
}
