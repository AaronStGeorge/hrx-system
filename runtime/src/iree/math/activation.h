// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_ACTIVATION_H_
#define IREE_MATH_ACTIVATION_H_

#include "iree/base/attributes.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Stable activation mappings with fixed, unfused rounding points and gradual
// underflow. Callers establish non-trapping round-to-nearest-even arithmetic.
// Every NaN input returns the canonical positive quiet NaN.
//
// The composed formulas have error bounded by 4*epsilon*condition*abs(result)
// plus one least subnormal, relative to the correctly rounded real formula.
// epsilon is 2^-23 for F32 and 2^-52 for F64. The condition is one for
// logistic, SiLU, and softplus; GELU's argument conditioning is documented
// below. These bounds describe accuracy; each entry point still names one fixed
// mapping.

// Logistic sigmoid 1/(1+exp(-x)), using sign-split exponential evaluation.
// Either zero returns 1/2; negative/positive infinity returns zero/one.
IREE_API_EXPORT float iree_math_logistic_f32(float value);
IREE_API_EXPORT double iree_math_logistic_f64(double value);

// SiLU x/(1+exp(-x)), retaining scaled negative tails before final underflow.
// Signed zero is preserved; negative infinity returns -0 and positive infinity
// returns +infinity.
IREE_API_EXPORT float iree_math_silu_f32(float value);
IREE_API_EXPORT double iree_math_silu_f64(double value);

// Softplus log(1+exp(x)), without positive overflow or negative cancellation.
// Negative infinity returns +0 and positive infinity returns +infinity.
IREE_API_EXPORT float iree_math_softplus_f32(float value);
IREE_API_EXPORT double iree_math_softplus_f64(double value);

// GELU x/2*(1+erf(x/sqrt(2))), evaluated through erfc to retain negative tails.
// The error condition is 1+x*x. Signed zero is preserved; negative infinity
// returns -0 and positive infinity returns +infinity.
IREE_API_EXPORT float iree_math_gelu_erf_f32(float value);
IREE_API_EXPORT double iree_math_gelu_erf_f64(double value);

// GELU x/2*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))). This authored approximation
// family remains distinct from erf GELU. Its error condition is
// 2*(1+abs(2*sqrt(2/pi)*(x+0.044715*x^3))). Signed zero and infinity follow
// the erf variant, and the small negative tail is retained directly.
IREE_API_EXPORT float iree_math_gelu_tanh_f32(float value);
IREE_API_EXPORT double iree_math_gelu_tanh_f64(double value);

// Scaled-logistic GELU x/(1+exp(-scale*x)). Scale is a selected-width input;
// negative, zero, and infinite scales are valid. The error condition is
// 1+abs(scale*x). Signed zero is preserved unless scale is infinite; a zero
// multiplied by infinity returns canonical NaN. Otherwise infinite arguments
// return signed zero when scale*x is negative, or x when it is positive.
IREE_API_EXPORT float iree_math_gelu_logistic_f32(float value, float scale);
IREE_API_EXPORT double iree_math_gelu_logistic_f64(double value, double scale);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MATH_ACTIVATION_H_
