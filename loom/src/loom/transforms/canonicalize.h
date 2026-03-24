// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_CANONICALIZE_H_
#define LOOM_TRANSFORMS_CANONICALIZE_H_

#include "loom/transforms/pass.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_pass_info_t loom_canonicalize_pass_info;

// Canonicalize pass.
//
// Iterates all ops in a function, calling each op's vtable canonicalize
// function (if non-NULL) through the rewriter. The rewriter's worklist
// tracks what needs revisiting after each transformation. Iterates
// until fixed point or max iterations.
//
// Each op kind defines its own canonicalization patterns (e.g.,
// addi(x, 0) → x, neg(neg(x)) → x) as a single C function on the
// vtable. The canonicalize pass is the driver — it doesn't know what
// transformations exist, only how to invoke them.
iree_status_t loom_canonicalize_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_CANONICALIZE_H_
