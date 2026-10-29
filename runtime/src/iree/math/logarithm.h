// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_LOGARITHM_H_
#define IREE_MATH_LOGARITHM_H_

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the stable natural, base-two, or base-ten logarithm.
// Finite results differ from correct rounding by at most one adjacent value.
// Either zero returns negative infinity; negative values and every NaN return
// the canonical positive quiet NaN; positive infinity is preserved.
// The caller establishes non-trapping round-to-nearest-even arithmetic with
// gradual underflow. The implementation uses fixed, unfused rounding points.
IREE_API_EXPORT float iree_math_log_f32(float value);
IREE_API_EXPORT double iree_math_log_f64(double value);
IREE_API_EXPORT float iree_math_log2_f32(float value);
IREE_API_EXPORT double iree_math_log2_f64(double value);
IREE_API_EXPORT float iree_math_log10_f32(float value);
IREE_API_EXPORT double iree_math_log10_f64(double value);

// Returns log(1+value), preserving information lost by rounding 1+value first.
// Signed zero and positive infinity are preserved; -1 returns negative
// infinity; values below -1 and every NaN return the canonical positive quiet
// NaN. Accuracy and execution profile are the same as the logarithms above.
IREE_API_EXPORT float iree_math_log1p_f32(float value);
IREE_API_EXPORT double iree_math_log1p_f64(double value);

// Returns one frozen machine-like approximation of log2(value).
// Finite normal-path results differ from the correctly rounded result by at
// most one adjacent f32 value. Zeros and subnormals flush to zero and return
// negative infinity. Negative normal values and every NaN return the canonical
// positive quiet NaN, and positive infinity is preserved. The caller must
// establish non-trapping round-to-nearest-even arithmetic.
IREE_API_EXPORT float iree_math_log2_f32_approx(float value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_LOGARITHM_H_
