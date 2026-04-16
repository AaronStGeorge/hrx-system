// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Whole-program boundary fact refinement.
//
// This module pass propagates stable value facts across direct function
// boundaries and drives the reusable canonicalizer until those boundary facts
// stop changing. It is intentionally a boundary driver, not a local pattern
// collection: local simplification still happens through canonicalize.

#ifndef LOOM_TRANSFORMS_REFINE_BOUNDARIES_H_
#define LOOM_TRANSFORMS_REFINE_BOUNDARIES_H_

#include "loom/transforms/pass.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns immutable metadata for the refine-boundaries pass.
const loom_pass_info_t* loom_refine_boundaries_pass_info(void);

// Propagates direct-call boundary facts through the module.
iree_status_t loom_refine_boundaries_run(loom_pass_t* pass,
                                         loom_module_t* module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_REFINE_BOUNDARIES_H_
