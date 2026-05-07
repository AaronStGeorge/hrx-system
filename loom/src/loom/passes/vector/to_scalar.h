// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vector-to-scalar reference lowering.
//
// This pass exposes vector lane semantics using scalar ops and scf.for loops
// while preserving function ABI. Vector arguments/results/calls/returns remain
// vector-typed; vector.extract/vector.insert/vector.from_elements are the
// aggregate boundary ops used to move between vector values and scalar lane
// programs.

#ifndef LOOM_PASSES_VECTOR_TO_SCALAR_H_
#define LOOM_PASSES_VECTOR_TO_SCALAR_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_vector_to_scalar_pass_info(void);

iree_status_t loom_vector_to_scalar_run(loom_pass_t* pass,
                                        loom_module_t* module,
                                        loom_func_like_t function);

const loom_pass_info_t* loom_vector_reduce_axes_to_scalar_pass_info(void);

iree_status_t loom_vector_reduce_axes_to_scalar_run(loom_pass_t* pass,
                                                    loom_module_t* module,
                                                    loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_PASSES_VECTOR_TO_SCALAR_H_
