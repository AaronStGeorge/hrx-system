// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_EXPONENTIAL_H_
#define IREE_MATH_EXPONENTIAL_H_

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the stable natural or base-two exponential, with gradual underflow.
// Finite results differ from correct rounding by at most one adjacent value.
// Either zero returns one; negative infinity returns positive zero; positive
// infinity is preserved; every NaN returns the canonical positive quiet NaN.
// The caller establishes non-trapping round-to-nearest-even arithmetic with
// gradual underflow. The implementation uses fixed, unfused rounding points.
IREE_API_EXPORT float iree_math_exp_f32(float value);
IREE_API_EXPORT double iree_math_exp_f64(double value);
IREE_API_EXPORT float iree_math_exp2_f32(float value);
IREE_API_EXPORT double iree_math_exp2_f64(double value);

// Returns one frozen machine-like approximation of 2^value.
// Finite normal-path results differ from the correctly rounded result by at
// most one adjacent f32 value. Values below -126 flush to positive zero and
// values at least 128 overflow to positive infinity. Either zero returns one,
// infinities map to their limiting results, and every NaN returns the canonical
// positive quiet NaN. The caller must establish non-trapping
// round-to-nearest-even arithmetic.
IREE_API_EXPORT float iree_math_exp2_f32_approx(float value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_EXPONENTIAL_H_
