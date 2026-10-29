// Copyright (c) 2017-2025, Arm Limited.
// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MATH_LOGARITHM_REDUCTION_H_
#define IREE_MATH_LOGARITHM_REDUCTION_H_

// The ordinary F32 logarithm and power's extended-precision logarithm use
// identical reduction centers, but distinct approximation polynomials. This
// private table preserves one data definition without exposing a rounded log2
// call that would discard the extra precision required by power.
typedef struct iree_math_log2_f32_table_entry_t {
  // Reciprocal of the subinterval center used to form a small residual.
  double inverse_center;
  // Base-two logarithm of the subinterval center.
  double log2_center;
} iree_math_log2_f32_table_entry_t;

extern const iree_math_log2_f32_table_entry_t iree_math_log2_f32_table[16];

#endif  // IREE_MATH_LOGARITHM_REDUCTION_H_
