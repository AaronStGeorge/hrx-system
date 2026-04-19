// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_DCE_H_
#define LOOM_TRANSFORMS_DCE_H_

#include "loom/pass/manager.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns immutable metadata for the dead code elimination pass.
const loom_pass_info_t* loom_dce_pass_info(void);

// Dead code elimination pass.
//
// Removes ops whose results are unused and that have no side effects.
// Walks all blocks in the function including nested regions, using an
// iterative block collector (no recursion, bounded stack). Within each
// block, ops are processed in reverse order so that erasing a dead op
// may expose its operands as newly dead in the same scan. A fixed-point
// loop handles cascading across blocks.
iree_status_t loom_dce_run(loom_pass_t* pass, loom_module_t* module,
                           loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_DCE_H_
