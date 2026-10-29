// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_FLOAT_BITS_H_
#define IREE_MATH_FLOAT_BITS_H_

#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/attributes.h"

// Algorithm error bounds and frozen mappings depend on rounding each arithmetic
// operation at its source width. Rounding only casts and assignments still
// permits double rounding and invalidates compensated range reduction.
static_assert(FLT_EVAL_METHOD == 0,
              "iree/math requires source-width floating-point evaluation");
#if (defined(__i386__) || defined(__x86_64__)) && !defined(__SSE2__)
#error "iree/math requires SSE2 on x86"
#elif defined(_M_IX86) && (!defined(_M_IX86_FP) || _M_IX86_FP < 2)
#error "iree/math requires SSE2 on x86"
#endif

static_assert(FLT_RADIX == 2 && sizeof(float) == sizeof(uint32_t) &&
                  FLT_MANT_DIG == 24 && FLT_MIN_EXP == -125 &&
                  FLT_MAX_EXP == 128,
              "iree/math requires IEEE-754 binary32");
static_assert(FLT_RADIX == 2 && sizeof(double) == sizeof(uint64_t) &&
                  DBL_MANT_DIG == 53 && DBL_MIN_EXP == -1021 &&
                  DBL_MAX_EXP == 1024,
              "iree/math requires IEEE-754 binary64");

#define IREE_MATH_F32_SIGN_BIT UINT32_C(0x80000000)
#define IREE_MATH_F32_MAGNITUDE_MASK UINT32_C(0x7FFFFFFF)
#define IREE_MATH_F32_MIN_NORMAL UINT32_C(0x00800000)
#define IREE_MATH_F32_INFINITY UINT32_C(0x7F800000)
#define IREE_MATH_F32_CANONICAL_NAN UINT32_C(0x7FC00000)

#define IREE_MATH_F64_SIGN_BIT UINT64_C(0x8000000000000000)
#define IREE_MATH_F64_MAGNITUDE_MASK UINT64_C(0x7FFFFFFFFFFFFFFF)
#define IREE_MATH_F64_MIN_NORMAL UINT64_C(0x0010000000000000)
#define IREE_MATH_F64_INFINITY UINT64_C(0x7FF0000000000000)
#define IREE_MATH_F64_CANONICAL_NAN UINT64_C(0x7FF8000000000000)

// MSVC's precise profile permits explicit contraction pragmas. The numerical
// kernels specify their rounding points; a multiply and add must stay separate
// unless the source explicitly requests a fused operation.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma fp_contract(off)
#endif

static inline uint32_t iree_math_f32_to_bits(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static inline float iree_math_f32_from_bits(uint32_t bits) {
  float value = 0.0f;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static inline uint64_t iree_math_f64_to_bits(double value) {
  uint64_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static inline double iree_math_f64_from_bits(uint64_t bits) {
  double value = 0.0;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static inline float iree_math_f32_canonical_nan(void) {
  return iree_math_f32_from_bits(IREE_MATH_F32_CANONICAL_NAN);
}

static inline double iree_math_f64_canonical_nan(void) {
  return iree_math_f64_from_bits(IREE_MATH_F64_CANONICAL_NAN);
}

#endif  // IREE_MATH_FLOAT_BITS_H_
