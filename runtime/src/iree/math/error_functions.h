// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_ERROR_FUNCTIONS_H_
#define IREE_MATH_ERROR_FUNCTIONS_H_

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns the stable error function with at most one adjacent result of error
// from correct rounding. Signed zero is preserved and infinities return signed
// one. Every NaN returns the canonical positive quiet NaN. The caller
// establishes non-trapping round-to-nearest-even arithmetic with gradual
// underflow; the implementation uses fixed, unfused rounding points.
IREE_API_EXPORT float iree_math_erf_f32(float value);
IREE_API_EXPORT double iree_math_erf_f64(double value);

// Returns the stable complementary error function, retaining its small positive
// tail with gradual underflow and at most one adjacent result of error from
// correct rounding. Either zero returns one; positive infinity returns positive
// zero; negative infinity returns two; every NaN returns the canonical positive
// quiet NaN. The caller establishes the same arithmetic environment as erf.
IREE_API_EXPORT float iree_math_erfc_f32(float value);
IREE_API_EXPORT double iree_math_erfc_f64(double value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_ERROR_FUNCTIONS_H_
