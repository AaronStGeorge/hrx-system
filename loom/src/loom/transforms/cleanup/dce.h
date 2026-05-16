// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_DCE_H_
#define LOOM_TRANSFORMS_DCE_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef iree_status_t (*loom_dce_deadness_query_fn_t)(
    void* user_data, const loom_module_t* module, const loom_op_t* op,
    bool* out_is_dead);

typedef struct loom_dce_deadness_query_callback_t {
  // Callback invoked to decide whether one operation may be erased.
  loom_dce_deadness_query_fn_t fn;
  // Opaque caller data passed to |fn|.
  void* user_data;
} loom_dce_deadness_query_callback_t;

// Returns immutable metadata for the dead code elimination pass.
const loom_pass_info_t* loom_dce_pass_info(void);

// Dead code elimination pass.
//
// Removes ops whose results are unused and that have no side effects. The pass
// seeds a deduplicated worklist from all blocks, including nested regions, and
// rechecks producers when an erased subtree drops operand or type-reference
// uses.
iree_status_t loom_dce_run(loom_pass_t* pass, loom_module_t* module,
                           loom_func_like_t function);

// Runs DCE with a caller-provided deadness predicate.
//
// The DCE engine still owns cascading use-def invalidation, type-ref provider
// rechecks, and region traversal. |deadness_query| only answers whether the
// current op may be erased when its results are unused under the caller's
// semantic model.
iree_status_t loom_dce_run_with_deadness_query(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    loom_dce_deadness_query_callback_t deadness_query);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_DCE_H_
