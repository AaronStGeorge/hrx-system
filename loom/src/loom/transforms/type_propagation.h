// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transactional type propagation through generated op constraints.
//
// This utility interprets table-driven semantic constraints as a monotonic
// refinement problem. Candidate type narrowings are collected in scratch state,
// expanded across the connected constraint/use closure, and committed through
// the rewriter only if the whole transaction is consistent. Malformed or
// contradictory IR leaves the module unchanged; the verifier remains
// responsible for structured user diagnostics.

#ifndef LOOM_TRANSFORMS_TYPE_PROPAGATION_H_
#define LOOM_TRANSFORMS_TYPE_PROPAGATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/transforms/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_type_propagator_t loom_type_propagator_t;

// Allocates a type propagator from |arena|. The propagator owns dense scratch
// indexed by value id and may be reused across many op transactions in the same
// pass/function run.
iree_status_t loom_type_propagator_allocate(
    loom_module_t* module, iree_arena_allocator_t* arena,
    loom_type_propagator_t** out_propagator);

// Refreshes function-local block-argument ownership metadata. Call once before
// processing a function and again if a pass creates new region-owning ops
// during the same run. Calling more often is safe; metadata is
// generation-cleared.
iree_status_t loom_type_propagator_prepare_function(
    loom_type_propagator_t* propagator, loom_func_like_t function);

// Applies one transactional propagation seeded at |op|. The transaction uses
// generated constraints on |op| and every affected user/defining op reached by
// candidate type changes. If the closure is consistent, all narrowed value
// types are committed through |rewriter| and |*out_changed| is set. If any
// constraint conflicts, no IR is mutated and |*out_changed| is false.
iree_status_t loom_type_propagator_apply_op(loom_type_propagator_t* propagator,
                                            loom_rewriter_t* rewriter,
                                            loom_op_t* op, bool* out_changed);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_TYPE_PROPAGATION_H_
