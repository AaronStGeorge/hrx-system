// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_ERFC_TAIL_H_
#define IREE_MATH_ERFC_TAIL_H_

// Returns factor*erfc(value), retaining the factor before final underflow.
// The caller supplies value in [1.25, 40/sqrt(2)] and a finite factor in
// [-20,20] under nearest-even arithmetic with gradual underflow. This shares
// erfc's rational tail reduction and exponent residual without rounding erfc
// first.
double iree_math_erfc_f64_scaled_tail(double value, double factor);

#endif  // IREE_MATH_ERFC_TAIL_H_
