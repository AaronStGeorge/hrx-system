// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Programmatic construction helpers for pass IR.
//
// These helpers are the C-side equivalent of writing a pass.pipeline in Loom
// text. They centralize the small amount of symbol/string plumbing needed by
// tools and production pipeline contributors without inventing a second pass
// representation. Execution still goes through pass.program compilation and the
// interpreter.

#ifndef LOOM_PASS_BUILDER_H_
#define LOOM_PASS_BUILDER_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/pass/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef iree_status_t (*loom_pass_ir_body_build_fn_t)(loom_builder_t* builder,
                                                      void* user_data);

// Builds a named pass.pipeline in |module| and invokes |build_body| while the
// builder insertion point is inside the pipeline body. The helper appends the
// required pass.yield terminator after the body callback succeeds.
iree_status_t loom_pass_ir_build_pipeline(
    loom_module_t* module, iree_string_view_t name, loom_pass_anchor_t anchor,
    loom_pass_ir_body_build_fn_t build_body, void* user_data,
    loom_op_t** out_pipeline_op);

// Appends a pass.run with the canonical pass |key| and typed option attrs.
// |options| must already be owned by |builder->module|.
iree_status_t loom_pass_ir_build_run(loom_builder_t* builder,
                                     iree_string_view_t key,
                                     loom_named_attr_slice_t options,
                                     loom_op_t** out_run_op);

// Appends a pass.yield terminator at the current insertion point.
iree_status_t loom_pass_ir_build_yield(loom_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_BUILDER_H_
