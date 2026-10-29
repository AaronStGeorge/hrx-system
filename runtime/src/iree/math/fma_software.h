// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_FMA_SOFTWARE_H_
#define IREE_MATH_FMA_SOFTWARE_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Exact one-rounding FMA for hosts without a native fused instruction. The
// caller establishes nearest-even arithmetic with gradual underflow. NaNs are
// canonicalized by the public numerical entry points.
float iree_math_fma_f32_software(float a, float b, float c);
double iree_math_fma_f64_software(double a, double b, double c);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_FMA_SOFTWARE_H_
