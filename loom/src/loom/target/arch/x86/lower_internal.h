// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private x86 source-to-target-low lowering helpers.
//
// This header is target-local glue between lower.c and focused x86 leaf
// lowerers. Keep declarations narrow: a new declaration should mean two x86
// lowering invariant clusters genuinely share one contract.

#ifndef LOOM_TARGET_ARCH_X86_LOWER_INTERNAL_H_
#define LOOM_TARGET_ARCH_X86_LOWER_INTERNAL_H_

#include "loom/target/arch/x86/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Target-local rule table for regular AVX512 source ops.
extern const loom_low_lower_rule_set_t loom_x86_avx512_rule_set;

// Builds a one-unit AVX512 ZMM register type in the current lowering context.
iree_status_t loom_x86_make_zmm_register_type(loom_low_lower_context_t* context,
                                              loom_type_t* out_type);

// Selects a target-owned plan for x86 AVX512 source ops.
iree_status_t loom_x86_select_avx512_op(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_low_lower_plan_t* out_plan);

// Emits a previously selected x86 AVX512 source op plan.
iree_status_t loom_x86_emit_avx512_op(void* user_data,
                                      loom_low_lower_context_t* context,
                                      const loom_op_t* source_op,
                                      loom_low_lower_plan_t plan);

// Maps a packed-dot source value type to the selected x86 register class.
iree_status_t loom_x86_map_packed_dot_type(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_type_t source_type,
                                           loom_type_t* out_low_type);

// Selects a target-owned plan for x86 packed-dot source ops.
iree_status_t loom_x86_select_packed_dot_op(void* user_data,
                                            loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t* out_plan);

// Emits a previously selected x86 packed-dot source op plan.
iree_status_t loom_x86_emit_packed_dot_op(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan);

// Returns true for source vector-dot ops handled by the x86 packed-dot lowerer.
bool loom_x86_op_is_vector_dot(loom_op_kind_t kind);

// Verifies source vector-dot legality for x86 packed-dot target-low selection.
iree_status_t loom_x86_low_legality_verify_packed_dot(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_LOWER_INTERNAL_H_
