// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Callable materialization adapters.
//
// These helpers adapt generic IR remapping/materialization to function-like
// operations. They do not decide inline profitability or import policy; callers
// choose a call site and these helpers perform the checked mutation.

#ifndef LOOM_TRANSFORMS_CALLABLE_H_
#define LOOM_TRANSFORMS_CALLABLE_H_

#include "iree/base/api.h"
#include "loom/ops/op_defs.h"
#include "loom/transforms/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves the direct symbol target of func.call or func.apply.
iree_status_t loom_callable_resolve_direct_callee(const loom_module_t* module,
                                                  const loom_op_t* call_op,
                                                  loom_func_like_t* out_callee);

// Inlines |callee| at |call_op| using clone materialization.
//
// The callee must be a same-module function-like op with a single-block body
// ending in func.return. Function entry block arguments are bound to call
// operands, body ops are cloned before the call, func.return operands become
// the call result replacements, and the original call-like op is erased through
// the rewriter.
iree_status_t loom_callable_inline_call(loom_rewriter_t* rewriter,
                                        loom_op_t* call_op,
                                        loom_func_like_t callee);

// Resolves |call_op|'s direct callee and then inlines it.
iree_status_t loom_callable_inline_direct_call(loom_rewriter_t* rewriter,
                                               loom_op_t* call_op);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_CALLABLE_H_
