// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_INVERSE_TRIGONOMETRY_H_
#define IREE_MATH_INVERSE_TRIGONOMETRY_H_

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Stable inverse trigonometric mappings in radians with fixed, unfused rounding
// points and gradual underflow. Finite results differ from correct rounding by
// at most one adjacent value. Every NaN becomes the canonical positive quiet
// NaN. The caller establishes non-trapping round-to-nearest-even arithmetic
// with gradual underflow; errno and exception flags are not specified.
//
// asin preserves signed zero and returns canonical NaN outside [-1, 1]. acos
// returns positive zero at one, pi at negative one, and canonical NaN outside
// [-1, 1]. Endpoint reconstruction retains the small result of acos near one.
IREE_API_EXPORT float iree_math_asin_f32(float value);
IREE_API_EXPORT double iree_math_asin_f64(double value);
IREE_API_EXPORT float iree_math_acos_f32(float value);
IREE_API_EXPORT double iree_math_acos_f64(double value);

// atan preserves signed zero and maps infinite inputs to signed pi/2.
IREE_API_EXPORT float iree_math_atan_f32(float value);
IREE_API_EXPORT double iree_math_atan_f64(double value);

// atan2(y, x) returns the argument of x+i*y in [-pi, pi]. The sign of y is
// retained, including for zero results and the negative-real-axis branch cut.
// Signed zero x selects the same quadrant as a nonzero x of that sign. Infinite
// pairs return signed pi/4 or 3*pi/4; a single infinite operand returns the
// corresponding signed axis angle. Exponent guards avoid overflowing y/x.
IREE_API_EXPORT float iree_math_atan2_f32(float y, float x);
IREE_API_EXPORT double iree_math_atan2_f64(double y, double x);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_INVERSE_TRIGONOMETRY_H_
