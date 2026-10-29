// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_TRIGONOMETRY_H_
#define IREE_MATH_TRIGONOMETRY_H_

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Ordinary radian sine, cosine, and tangent. Each mapping has at most one
// adjacent representable value of error from correct rounding over the full
// finite domain, with fixed unfused intermediates. Sine and tangent preserve
// signed zero; cosine returns one at either zero. Nonfinite inputs return the
// canonical positive quiet NaN. The caller establishes non-trapping nearest-
// even arithmetic with input and output subnormals preserved.
IREE_API_EXPORT float iree_math_sin_f32(float value);
IREE_API_EXPORT double iree_math_sin_f64(double value);
IREE_API_EXPORT float iree_math_cos_f32(float value);
IREE_API_EXPORT double iree_math_cos_f64(double value);
IREE_API_EXPORT float iree_math_tan_f32(float value);
IREE_API_EXPORT double iree_math_tan_f64(double value);

// Binary64 sin(2*pi*turns) and cos(2*pi*turns), with at most one adjacent value
// of error from correct rounding. Cardinal values are structural: sine at
// half-integer turns returns zero with the input sign, cosine at odd quarter-
// turns returns positive zero, and other cardinals return exact signed one.
// Nonfinite inputs return canonical positive quiet NaN. The execution profile
// is the same nearest-even, gradual-underflow profile as the radian functions.
IREE_API_EXPORT double iree_math_sin_turns_f64(double turns);
IREE_API_EXPORT double iree_math_cos_turns_f64(double turns);

// Returns the correctly rounded sin(2*pi*turns) result for every f32 payload.
// Cardinal values are structural: signed zero is preserved, positive and
// negative half-integers return their matching signed zero, and odd
// quarter-turns return exact signed one. Every nonfinite input returns the
// canonical positive quiet NaN. The caller must establish non-trapping
// round-to-nearest-even arithmetic with input and output subnormals preserved.
IREE_API_EXPORT float iree_math_sin_turns_f32_approx(float turns);

// Returns the correctly rounded cos(2*pi*turns) result for every f32 payload.
// Cardinal values are structural: integers return one, half-integers return
// exact signed one, and odd quarter-turns return positive zero. Every
// nonfinite input returns the canonical positive quiet NaN. The caller must
// establish non-trapping round-to-nearest-even arithmetic with input and
// output subnormals preserved.
IREE_API_EXPORT float iree_math_cos_turns_f32_approx(float turns);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_TRIGONOMETRY_H_
