// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_WASM_LEGALIZATION_FLOAT_H_
#define LOOM_TARGET_ARCH_WASM_LEGALIZATION_FLOAT_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Widens packed narrow floating comparisons to an exact F32 carrier while
// preserving the predicate for the complete Wasm predicate contract family.
iree_status_t loom_wasm_legalize_vector_cmpf(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

// Lowers floating extrema and clamps on supported Wasm vector carriers through
// exact comparisons, bit operations, and selects. Narrow comparisons widen to
// F32 while every result retains its original payload representation.
iree_status_t loom_wasm_legalize_vector_float_extrema(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_WASM_LEGALIZATION_FLOAT_H_
