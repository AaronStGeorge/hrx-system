// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_HYPERBOLIC_H_
#define IREE_MATH_HYPERBOLIC_H_

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Stable hyperbolic mappings with fixed, unfused rounding points and gradual
// underflow. Finite results differ from correct rounding by at most one
// adjacent value. Every NaN becomes the canonical positive quiet NaN. The
// caller establishes non-trapping round-to-nearest-even arithmetic with gradual
// underflow; errno and floating-point exception flags are not specified.
//
// sinh preserves signed zero and infinity. cosh maps either zero to one and
// either infinity to positive infinity. Both retain the finite result range
// above exp's overflow threshold. tanh preserves signed zero and approaches
// signed one, including for infinite inputs.
IREE_API_EXPORT float iree_math_sinh_f32(float value);
IREE_API_EXPORT double iree_math_sinh_f64(double value);
IREE_API_EXPORT float iree_math_cosh_f32(float value);
IREE_API_EXPORT double iree_math_cosh_f64(double value);
IREE_API_EXPORT float iree_math_tanh_f32(float value);
IREE_API_EXPORT double iree_math_tanh_f64(double value);

// Inverse hyperbolic mappings with the same accuracy and execution profile.
// asinh preserves signed zero and infinity. acosh maps one to positive zero,
// preserves positive infinity, and returns canonical NaN below one. atanh
// preserves signed zero, maps +/-1 to signed infinity, and returns canonical
// NaN outside [-1, 1]. Cancellation-safe formulas retain results near zero and
// the acosh domain boundary without overflowing intermediate squares.
IREE_API_EXPORT float iree_math_asinh_f32(float value);
IREE_API_EXPORT double iree_math_asinh_f64(double value);
IREE_API_EXPORT float iree_math_acosh_f32(float value);
IREE_API_EXPORT double iree_math_acosh_f64(double value);
IREE_API_EXPORT float iree_math_atanh_f32(float value);
IREE_API_EXPORT double iree_math_atanh_f64(double value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_HYPERBOLIC_H_
