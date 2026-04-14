// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vector memory footprint legality gate.
//
// This pass is a pre-lowering proof pass for view-based vector memory ops. It
// keeps cheap structural checks in the ordinary verifier and proves dynamic
// footprint inequalities with value facts, symbolic expressions, and view
// region summaries immediately before lowering out of Loom IR.

#ifndef LOOM_TRANSFORMS_VECTOR_MEMORY_FOOTPRINT_H_
#define LOOM_TRANSFORMS_VECTOR_MEMORY_FOOTPRINT_H_

#include "iree/base/api.h"
#include "loom/transforms/pass.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_vector_memory_footprint_pass_info(void);

iree_status_t loom_vector_memory_footprint_run(loom_pass_t* pass,
                                               loom_module_t* module,
                                               loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_MEMORY_FOOTPRINT_H_
