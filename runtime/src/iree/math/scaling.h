// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_SCALING_H_
#define IREE_MATH_SCALING_H_

#include <stdint.h>

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Exact binary decomposition and scaling. Signed zeros and infinities are
// preserved; NaNs become the canonical positive quiet NaN. Callers establish
// masked exceptions, nearest-even rounding, and gradual underflow once around
// their evaluation scope. These functions do not change the environment or
// report floating exception flags or errno.

// Returns the signed fraction and writes the binary exponent such that
// value = fraction * 2^exponent. Finite nonzero fractions have magnitude in
// [0.5, 1). Subnormals are normalized exactly. Zero, infinity, and NaN inputs
// write exponent zero. |out_exponent| must point to writable storage.
IREE_API_EXPORT float iree_math_frexp_f32(float value, int32_t* out_exponent);
IREE_API_EXPORT double iree_math_frexp_f64(double value, int32_t* out_exponent);

// Returns value * 2^exponent with one final nearest-even rounding, including
// subnormal results. Every signed i32 exponent is valid; overflow produces
// signed infinity and underflow retains the input sign.
IREE_API_EXPORT float iree_math_ldexp_f32(float value, int32_t exponent);
IREE_API_EXPORT double iree_math_ldexp_f64(double value, int32_t exponent);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_SCALING_H_
