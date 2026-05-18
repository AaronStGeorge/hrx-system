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

// Recognizes a low.cond_br shape that can be emitted as a SPIR-V selection.
//
// This is a syntactic target-low check over region block order. It accepts
// selection shapes emitted in one forward block walk: a shared merge, a guard
// arm that branches to the merge, or a two-arm diamond whose arms branch to the
// same merge block. The merge block must appear after the selected arm blocks
// because emission preserves region order.
bool loom_spirv_low_select_merge_block(const loom_op_t* op,
                                       const loom_block_t** out_merge_block);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_STRUCTURED_CONTROL_FLOW_H_
