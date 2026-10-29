// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_ROUNDING_H_
#define IREE_MATH_ROUNDING_H_

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Exact integral rounding independent of the caller's rounding direction.
// Signed zero and infinity are preserved; NaNs become the canonical positive
// quiet NaN. Callers mask floating-point exceptions and retain subnormal
// inputs.

// Rounds toward positive infinity.
IREE_API_EXPORT float iree_math_ceil_f32(float value);
IREE_API_EXPORT double iree_math_ceil_f64(double value);

// Rounds toward negative infinity.
IREE_API_EXPORT float iree_math_floor_f32(float value);
IREE_API_EXPORT double iree_math_floor_f64(double value);

// Rounds toward zero.
IREE_API_EXPORT float iree_math_trunc_f32(float value);
IREE_API_EXPORT double iree_math_trunc_f64(double value);

// Rounds to nearest, choosing the even integer at an exact halfway point.
IREE_API_EXPORT float iree_math_round_even_f32(float value);
IREE_API_EXPORT double iree_math_round_even_f64(double value);

// Rounds to nearest, choosing the integer away from zero at a halfway point.
IREE_API_EXPORT float iree_math_round_away_f32(float value);
IREE_API_EXPORT double iree_math_round_away_f64(double value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_ROUNDING_H_
