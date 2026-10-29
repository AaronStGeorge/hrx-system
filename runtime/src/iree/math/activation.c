// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/math/activation.h"

#include <math.h>

#include "iree/math/erfc_tail.h"
#include "iree/math/error_functions.h"
#include "iree/math/exponential.h"
#include "iree/math/float_bits.h"
#include "iree/math/logarithm.h"

IREE_API_EXPORT float iree_math_logistic_f32(float value) {
  const float exponential = iree_math_exp_f32(value >= 0.0f ? -value : value);
  return value >= 0.0f ? 1.0f / (1.0f + exponential)
                       : exponential / (1.0f + exponential);
}

IREE_API_EXPORT double iree_math_logistic_f64(double value) {
  const double exponential = iree_math_exp_f64(value >= 0.0 ? -value : value);
  return value >= 0.0 ? 1.0 / (1.0 + exponential)
                      : exponential / (1.0 + exponential);
}

// Shared x*sigmoid(argument) reconstruction. In the deep negative tail the
// denominator rounds to one. Applying x between two half exponentials avoids
// rounding sigmoid to zero while its scaled result is still representable.
static float iree_math_scaled_logistic_f32(float value, float argument) {
  if (isnan(value) || isnan(argument)) {
    return iree_math_f32_canonical_nan();
  }
  if (isinf(value) || isinf(argument)) {
    return argument < 0.0f ? copysignf(0.0f, value) : value;
  }
  if (argument < -80.0f) {
    const float half_exponential = iree_math_exp_f32(argument * 0.5f);
    return (value * half_exponential) * half_exponential;
  }
  return value * iree_math_logistic_f32(argument);
}

static double iree_math_scaled_logistic_f64(double value, double argument) {
  if (isnan(value) || isnan(argument)) {
    return iree_math_f64_canonical_nan();
  }
  if (isinf(value) || isinf(argument)) {
    return argument < 0.0 ? copysign(0.0, value) : value;
  }
  if (argument < -700.0) {
    const double half_exponential = iree_math_exp_f64(argument * 0.5);
    return (value * half_exponential) * half_exponential;
  }
  return value * iree_math_logistic_f64(argument);
}

IREE_API_EXPORT float iree_math_silu_f32(float value) {
  return iree_math_scaled_logistic_f32(value, value);
}

IREE_API_EXPORT double iree_math_silu_f64(double value) {
  return iree_math_scaled_logistic_f64(value, value);
}

IREE_API_EXPORT float iree_math_softplus_f32(float value) {
  return iree_math_log1p_f32(iree_math_exp_f32(-fabsf(value))) +
         (value > 0.0f ? value : 0.0f);
}

IREE_API_EXPORT double iree_math_softplus_f64(double value) {
  return iree_math_log1p_f64(iree_math_exp_f64(-fabs(value))) +
         (value > 0.0 ? value : 0.0);
}

IREE_API_EXPORT float iree_math_gelu_erf_f32(float value) {
  if (isnan(value)) {
    return iree_math_f32_canonical_nan();
  }
  if (value >= 6.0f) {
    return value;
  }
  if (value <= -16.0f) {
    return -0.0f;
  }
  if (value < -8.0f) {
    // Keep erfc's tail wide until the final scaled F32 result is rounded.
    return (float)((0.5 * (double)value) *
                   iree_math_erfc_f64(-(double)value * 0.70710678118654752440));
  }
  return (0.5f * value) * iree_math_erfc_f32(-value * 0.70710678118654752440f);
}

IREE_API_EXPORT double iree_math_gelu_erf_f64(double value) {
  if (isnan(value)) {
    return iree_math_f64_canonical_nan();
  }
  if (value >= 9.0) {
    return value;
  }
  if (value <= -40.0) {
    return -0.0;
  }
  if (value < -36.0) {
    return iree_math_erfc_f64_scaled_tail(-value * 0.70710678118654752440,
                                          0.5 * value);
  }
  return (0.5 * value) * iree_math_erfc_f64(-value * 0.70710678118654752440);
}

IREE_API_EXPORT float iree_math_gelu_tanh_f32(float value) {
  // 1+tanh(y) = 2*sigmoid(2*y), without subtracting its negative tail from one.
  const float argument = 2.0f * (0.79788456080286535588f *
                                 (value + 0.044715f * value * value * value));
  return iree_math_scaled_logistic_f32(value, argument);
}

IREE_API_EXPORT double iree_math_gelu_tanh_f64(double value) {
  const double argument = 2.0 * (0.79788456080286535588 *
                                 (value + 0.044715 * value * value * value));
  return iree_math_scaled_logistic_f64(value, argument);
}

IREE_API_EXPORT float iree_math_gelu_logistic_f32(float value, float scale) {
  return iree_math_scaled_logistic_f32(value, scale * value);
}

IREE_API_EXPORT double iree_math_gelu_logistic_f64(double value, double scale) {
  return iree_math_scaled_logistic_f64(value, scale * value);
}
