// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/roots.h"

#include <stdbool.h>

#if defined(IREE_COMPILER_MSVC)
#include <math.h>
#if defined(_M_X64) || defined(_M_ARM64)
#pragma intrinsic(sqrt, sqrtf)
#endif
#endif  // IREE_COMPILER_MSVC

#include "iree/math/float_bits.h"
#include "iree/math/roots_integer.h"

static float iree_math_sqrt_f32_native(float value) {
#if defined(IREE_COMPILER_MSVC) && (defined(_M_X64) || defined(_M_ARM64))
  return sqrtf(value);
#elif defined(__SSE__) || defined(__aarch64__) || defined(__wasm__) || \
    (defined(__ARM_FP) && (__ARM_FP & 4)) ||                           \
    (defined(__riscv_flen) && __riscv_flen >= 32)
  return __builtin_sqrtf(value);
#else
  return iree_math_f32_from_bits(
      (uint32_t)iree_math_sqrt_integer(iree_math_f32_to_bits(value), 23, 127));
#endif
}

static double iree_math_sqrt_f64_native(double value) {
#if defined(IREE_COMPILER_MSVC) && (defined(_M_X64) || defined(_M_ARM64))
  return sqrt(value);
#elif defined(__SSE2__) || defined(__aarch64__) || defined(__wasm__) || \
    (defined(__ARM_FP) && (__ARM_FP & 8)) ||                            \
    (defined(__riscv_flen) && __riscv_flen >= 64)
  return __builtin_sqrt(value);
#else
  return iree_math_f64_from_bits(
      iree_math_sqrt_integer(iree_math_f64_to_bits(value), 52, 1023));
#endif
}

IREE_API_EXPORT float iree_math_sqrt_f32(float value) {
  const uint32_t bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude == 0)) {
    return value;
  }
  if (IREE_UNLIKELY(bits > IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }
  if (IREE_UNLIKELY(bits == IREE_MATH_F32_INFINITY)) {
    return value;
  }
  return iree_math_sqrt_f32_native(value);
}

IREE_API_EXPORT double iree_math_sqrt_f64(double value) {
  const uint64_t bits = iree_math_f64_to_bits(value);
  const uint64_t magnitude = bits & IREE_MATH_F64_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude == 0)) {
    return value;
  }
  if (IREE_UNLIKELY(bits > IREE_MATH_F64_INFINITY)) {
    return iree_math_f64_canonical_nan();
  }
  if (IREE_UNLIKELY(bits == IREE_MATH_F64_INFINITY)) {
    return value;
  }
  return iree_math_sqrt_f64_native(value);
}

IREE_API_EXPORT float iree_math_rsqrt_f32(float value) {
  return 1.0f / iree_math_sqrt_f32(value);
}

IREE_API_EXPORT double iree_math_rsqrt_f64(double value) {
  return 1.0 / iree_math_sqrt_f64(value);
}

IREE_API_EXPORT float iree_math_reciprocal_f32_approx(float value) {
  const uint32_t value_bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude_bits = value_bits & IREE_MATH_F32_MAGNITUDE_MASK;
  const uint32_t sign_bits = value_bits & IREE_MATH_F32_SIGN_BIT;
  if (IREE_UNLIKELY(magnitude_bits > IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }
  if (IREE_UNLIKELY(magnitude_bits < IREE_MATH_F32_MIN_NORMAL)) {
    return iree_math_f32_from_bits(sign_bits | IREE_MATH_F32_INFINITY);
  }
  if (IREE_UNLIKELY(magnitude_bits == IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_from_bits(sign_bits);
  }

  const float result = 1.0f / value;
  const uint32_t result_bits = iree_math_f32_to_bits(result);
  if (IREE_UNLIKELY((result_bits & IREE_MATH_F32_MAGNITUDE_MASK) <
                    IREE_MATH_F32_MIN_NORMAL)) {
    return iree_math_f32_from_bits(result_bits & IREE_MATH_F32_SIGN_BIT);
  }
  return result;
}

static bool iree_math_rsqrt_f32_handle_special(float value, float* out_result) {
  const uint32_t value_bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude_bits = value_bits & IREE_MATH_F32_MAGNITUDE_MASK;
  const uint32_t sign_bits = value_bits & IREE_MATH_F32_SIGN_BIT;
  if (IREE_UNLIKELY(magnitude_bits > IREE_MATH_F32_INFINITY)) {
    *out_result = iree_math_f32_canonical_nan();
    return true;
  }
  if (IREE_UNLIKELY(magnitude_bits < IREE_MATH_F32_MIN_NORMAL)) {
    *out_result = iree_math_f32_from_bits(sign_bits | IREE_MATH_F32_INFINITY);
    return true;
  }
  if (IREE_UNLIKELY(sign_bits != 0)) {
    *out_result = iree_math_f32_canonical_nan();
    return true;
  }
  if (IREE_UNLIKELY(magnitude_bits == IREE_MATH_F32_INFINITY)) {
    *out_result = 0.0f;
    return true;
  }
  return false;
}

IREE_API_EXPORT float iree_math_rsqrt_f32_approx(float value) {
  float result = 0.0f;
  if (iree_math_rsqrt_f32_handle_special(value, &result)) {
    return result;
  }
  return 1.0f / iree_math_sqrt_f32_native(value);
}

IREE_API_EXPORT float iree_math_sqrt_f32_approx(float value) {
  const uint32_t value_bits = iree_math_f32_to_bits(value);
  const uint32_t magnitude_bits = value_bits & IREE_MATH_F32_MAGNITUDE_MASK;
  const uint32_t sign_bits = value_bits & IREE_MATH_F32_SIGN_BIT;
  if (IREE_UNLIKELY(magnitude_bits > IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }
  if (IREE_UNLIKELY(magnitude_bits < IREE_MATH_F32_MIN_NORMAL)) {
    return iree_math_f32_from_bits(sign_bits);
  }
  if (IREE_UNLIKELY(sign_bits != 0)) {
    return iree_math_f32_canonical_nan();
  }
  if (IREE_UNLIKELY(magnitude_bits == IREE_MATH_F32_INFINITY)) {
    return value;
  }
  return iree_math_sqrt_f32_native(value);
}
