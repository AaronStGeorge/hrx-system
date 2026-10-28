// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_FMA_H_
#define IREE_MATH_FMA_H_

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns a*b+c rounded once to the result width, with gradual underflow.
// Exact cancellation returns positive zero; a negative zero product plus a
// negative zero addend returns negative zero. Every NaN result is the canonical
// positive quiet NaN. The caller establishes non-trapping round-to-nearest-even
// arithmetic with gradual underflow.
IREE_API_EXPORT float iree_math_fma_f32(float a, float b, float c);
IREE_API_EXPORT double iree_math_fma_f64(double a, double b, double c);

// Produces an F32 intermediate that rounds to the exact F8/F16/BF16 FMA result.
// Each operand must be exactly representable in the same narrow format, with
// at most eleven significand bits and an exponent range contained in BF16 or
// F16. The caller rounds the returned intermediate to that format, applying
// its overflow policy. This preserves the side of a narrow rounding midpoint
// even when a tiny addend disappears in ordinary F64 addition; it is not the
// F32 fused result. The same floating environment as above is required.
IREE_API_EXPORT float iree_math_fma_narrow(double a, double b, double c);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_FMA_H_
