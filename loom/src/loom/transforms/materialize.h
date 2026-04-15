// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic IR clone materialization.
//
// This layer clones operations and regions using an explicit loom_ir_remap_t.
// It does not decide whether cloning is profitable, whether a callable should
// inline, or how symbols should be imported. Callers bind source values in the
// remap and provide symbol policy before asking this helper to materialize IR.

#ifndef LOOM_TRANSFORMS_MATERIALIZE_H_
#define LOOM_TRANSFORMS_MATERIALIZE_H_

#include "iree/base/api.h"
#include "loom/ops/op_defs.h"
#include "loom/transforms/remap.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options for cloning a source block's operation list into the current builder
// insertion block.
typedef struct loom_ir_clone_block_options_t {
  // Skips source ops whose vtable declares LOOM_TRAIT_TERMINATOR. This is used
  // by callable inlining wrappers that interpret the source terminator
  // separately instead of cloning it into the destination block.
  bool omit_terminators;
} loom_ir_clone_block_options_t;

// Clones |source_op| at the builder insertion point.
//
// Source operands, result types, attributes, nested region payloads, locations,
// strings, encodings, and symbols are remapped through |remap|. Source result
// values are mapped to their cloned target result values before result types
// are remapped, so co-result dynamic type references are preserved.
iree_status_t loom_ir_clone_op(loom_builder_t* builder,
                               const loom_op_t* source_op,
                               loom_ir_remap_t* remap,
                               loom_op_t** out_cloned_op);

// Clones every live op in |source_block| into the builder insertion block.
iree_status_t loom_ir_clone_block_ops(
    loom_builder_t* builder, const loom_block_t* source_block,
    loom_ir_remap_t* remap, const loom_ir_clone_block_options_t* options);

// Clones |source_region| and returns a new target-module-owned region.
iree_status_t loom_ir_clone_region(loom_builder_t* builder,
                                   const loom_region_t* source_region,
                                   loom_ir_remap_t* remap,
                                   loom_region_t** out_target_region);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_MATERIALIZE_H_
