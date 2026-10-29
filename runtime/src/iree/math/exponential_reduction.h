// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_EXPONENTIAL_REDUCTION_H_
#define IREE_MATH_EXPONENTIAL_REDUCTION_H_

// Shared finite exponential reconstruction for error-function tails. The
// producer supplies |high| >= |low| and high + low in [-746, 0]. Keeping the
// residual through range reduction avoids composing two rounded exponentials.
// The caller establishes nearest-even arithmetic with gradual underflow.
double iree_math_exp_f64_compensated(double high, double low);

#endif  // IREE_MATH_EXPONENTIAL_REDUCTION_H_
