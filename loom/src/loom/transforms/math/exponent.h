// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_MATH_EXPONENT_H_
#define LOOM_TRANSFORMS_MATH_EXPONENT_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Exact scalar/vector exponent decomposition and scaling after native
// contract rejection. Narrow floating formats share the F32 realization.
iree_status_t loom_math_legalize_exponent(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_MATH_EXPONENT_H_
