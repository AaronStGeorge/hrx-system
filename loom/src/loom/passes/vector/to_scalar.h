// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vector-to-scalar reference lowering.
//
// This pass exposes vector lane semantics using scalar ops and scf.for loops
// while preserving function ABI. Vector arguments/results/calls/returns remain
// vector-typed; vector.extract/vector.insert/vector.from_elements are the
// aggregate boundary ops used to move between vector values and scalar lane
// programs.

#ifndef LOOM_PASSES_VECTOR_TO_SCALAR_H_
#define LOOM_PASSES_VECTOR_TO_SCALAR_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"
#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_vector_to_scalar_pass_info(void);

iree_status_t loom_vector_to_scalar_run(loom_pass_t* pass,
                                        loom_module_t* module,
                                        loom_func_like_t function);

const loom_pass_info_t* loom_vector_reduce_axes_to_scalar_pass_info(void);

iree_status_t loom_vector_reduce_axes_to_scalar_run(loom_pass_t* pass,
                                                    loom_module_t* module,
                                                    loom_func_like_t function);

// Rewrites one vector.reduce.axes op using the same scalar reference lowering
// as the standalone pass. Statistics for lane and loop materialization are
// recorded in the current pass when provided; the caller owns op-rewrite
// accounting.
iree_status_t loom_vector_reduce_axes_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_rewritten);

// Rewrites one vector.mma op using scalar reference semantics when the result
// can be represented as a dense full-logical matrix fragment. Target-shaped
// results are instead drained through fragment movement boundaries.
iree_status_t loom_vector_mma_to_scalar_rewrite_op(loom_pass_t* pass,
                                                   loom_rewriter_t* rewriter,
                                                   loom_op_t* op,
                                                   bool* out_rewritten);

// Rewrites one vector.store into scalar view.store loops. The source vector is
// consumed lane-by-lane so supported producer trees can disappear through DCE
// without first materializing a dynamic vector aggregate.
iree_status_t loom_vector_store_to_scalar_rewrite_op(loom_pass_t* pass,
                                                     loom_rewriter_t* rewriter,
                                                     loom_op_t* op,
                                                     bool* out_rewritten);

// Rewrites one vector.fragment.store into scalar view.store loops over the
// fragment's logical matrix shape. The source fragment is consumed
// lane-by-lane so target-shaped producers can be erased without first
// materializing a dense vector aggregate.
iree_status_t loom_vector_fragment_store_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_rewritten);

// Rewrites one scalar-result vector.extract when its lane can be rematerialized
// from the source producer tree.
iree_status_t loom_vector_extract_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_rewritten);

const loom_pass_info_t* loom_vector_gather_to_scalar_pass_info(void);

iree_status_t loom_vector_gather_to_scalar_run(loom_pass_t* pass,
                                               loom_module_t* module,
                                               loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_PASSES_VECTOR_TO_SCALAR_H_
