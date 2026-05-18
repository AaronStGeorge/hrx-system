// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_SPIRV_STRUCTURED_CONTROL_FLOW_H_
#define LOOM_TARGET_ARCH_SPIRV_STRUCTURED_CONTROL_FLOW_H_

#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Syntactic low CFG loop shape accepted by the SPIR-V verifier/emitter.
typedef struct loom_spirv_low_loop_shape_t {
  // Header block containing the controlling low.cond_br.
  const loom_block_t* header_block;
  // Forward body block entered when the loop condition is true.
  const loom_block_t* body_block;
  // Forward merge block entered when the loop condition is false.
  const loom_block_t* merge_block;
  // Body terminator that branches back to header_block.
  const loom_op_t* backedge_op;
} loom_spirv_low_loop_shape_t;

// Recognizes a low.cond_br shape that can be emitted as a SPIR-V selection.
//
// This is a syntactic target-low check over region block order. It accepts
// selection shapes emitted in one forward block walk: a shared merge, a guard
// arm that branches to the merge, or a two-arm diamond whose arms branch to the
// same merge block. The merge block must appear after the selected arm blocks
// because emission preserves region order.
bool loom_spirv_low_select_merge_block(const loom_op_t* op,
                                       const loom_block_t** out_merge_block);

// Recognizes a low.cond_br shape that can be emitted as a SPIR-V loop.
//
// The accepted form matches the low CFG produced for simple scf.for/scf.while
// loops: the header has a conditional branch to a forward body block and a
// forward merge block, and the body block terminates in the single backedge to
// the header. The emitter synthesizes the SPIR-V continue block for this
// backedge; the low IR remains ordinary CFG.
bool loom_spirv_low_loop_shape(const loom_op_t* op,
                               loom_spirv_low_loop_shape_t* out_loop);

// Returns true when |op| is the backedge of a recognized SPIR-V low loop.
bool loom_spirv_low_br_is_loop_backedge(const loom_op_t* op);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_STRUCTURED_CONTROL_FLOW_H_
