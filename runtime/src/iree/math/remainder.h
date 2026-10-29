// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_REMAINDER_H_
#define IREE_MATH_REMAINDER_H_

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Computes x - trunc(x / y) * y exactly, without rounding the quotient or
// intermediate product. The result has x's sign, including zero. A NaN input,
// infinite x, or zero y produces the canonical positive quiet NaN. Finite x
// with infinite y returns x. Integer reduction preserves subnormals and is
// independent of the caller's rounding direction and flush-to-zero mode.
IREE_API_EXPORT float iree_math_fmod_f32(float x, float y);
IREE_API_EXPORT double iree_math_fmod_f64(double x, double y);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_REMAINDER_H_
