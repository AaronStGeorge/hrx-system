// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/fma.h"

#if defined(IREE_COMPILER_MSVC) && defined(__AVX2__)
#include <immintrin.h>
#elif defined(IREE_COMPILER_MSVC) && defined(_M_ARM64)
#include <arm64_neon.h>
#endif

#include "iree/math/float_bits.h"
#include "iree/math/fma_software.h"

IREE_API_EXPORT float iree_math_fma_f32(float a, float b, float c) {
  float result;
#if defined(IREE_COMPILER_MSVC) && defined(__AVX2__)
  result =
      _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
#elif defined(IREE_COMPILER_MSVC) && defined(_M_ARM64)
  result =
      vget_lane_f32(vfma_f32(vdup_n_f32(c), vdup_n_f32(a), vdup_n_f32(b)), 0);
#elif defined(__FMA__) || defined(__aarch64__) ||            \
    defined(__ARM_FEATURE_FMA) || defined(__FP_FAST_FMAF) || \
    (defined(__riscv_flen) && __riscv_flen >= 32)
  result = __builtin_fmaf(a, b, c);
#else
  result = iree_math_fma_f32_software(a, b, c);
#endif
  // Keep NaN selection in the floating-point register domain after native FMA.
  if (IREE_UNLIKELY(result != result)) {
    return iree_math_f32_canonical_nan();
  }
  return result;
}

IREE_API_EXPORT double iree_math_fma_f64(double a, double b, double c) {
  double result;
#if defined(IREE_COMPILER_MSVC) && defined(__AVX2__)
  result =
      _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)));
#elif defined(IREE_COMPILER_MSVC) && defined(_M_ARM64)
  result =
      vget_lane_f64(vfma_f64(vdup_n_f64(c), vdup_n_f64(a), vdup_n_f64(b)), 0);
#elif defined(__FMA__) || defined(__aarch64__) || defined(__FP_FAST_FMA) || \
    (defined(__ARM_FEATURE_FMA) && defined(__ARM_FP) && (__ARM_FP & 8)) ||  \
    (defined(__riscv_flen) && __riscv_flen >= 64)
  result = __builtin_fma(a, b, c);
#else
  result = iree_math_fma_f64_software(a, b, c);
#endif
  if (IREE_UNLIKELY(result != result)) {
    return iree_math_f64_canonical_nan();
  }
  return result;
}

IREE_API_EXPORT float iree_math_fma_narrow(double a, double b, double c) {
  // A narrow product has at most 22 significant bits and is exact in F64.
  const double product = a * b;
  const double sum = product + c;
  float rounded = (float)sum;
  uint32_t bits = iree_math_f32_to_bits(rounded);
  const uint32_t magnitude = bits & IREE_MATH_F32_MAGNITUDE_MASK;
  if (IREE_UNLIKELY(magnitude > IREE_MATH_F32_INFINITY)) {
    return iree_math_f32_canonical_nan();
  }

  // F16 has the widest narrow significand. Every narrow rounding midpoint,
  // including subnormal and overflow boundaries, has twelve zero low F32 bits.
  if ((bits & UINT32_C(0xFFF)) != 0 || magnitude == IREE_MATH_F32_INFINITY) {
    return rounded;
  }

  double error = sum - (double)rounded;
  if (error == 0.0) {
    // FastTwoSum recovers the exact residual even when the addend is too small
    // to survive the F64 addition. Neither product nor residual can exceed
    // F64's exponent range for the supported narrow operand formats.
    const uint64_t product_magnitude =
        iree_math_f64_to_bits(product) & IREE_MATH_F64_MAGNITUDE_MASK;
    const uint64_t addend_magnitude =
        iree_math_f64_to_bits(c) & IREE_MATH_F64_MAGNITUDE_MASK;
    error = product_magnitude >= addend_magnitude ? c - (sum - product)
                                                  : product - (sum - c);
  }
  if (error != 0.0) {
    // Nudge a midpoint toward the exact result before narrowing. The filter
    // also admits non-midpoints, where one F32 ULP cannot change narrow
    // rounding.
    bits += ((error > 0.0) != ((bits >> 31) != 0)) ? 1u : UINT32_MAX;
    rounded = iree_math_f32_from_bits(bits);
  }
  return rounded;
}
