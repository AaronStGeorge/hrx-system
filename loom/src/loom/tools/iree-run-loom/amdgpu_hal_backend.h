// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL backend provider for iree-run-loom.

#ifndef LOOM_TOOLS_IREE_RUN_LOOM_AMDGPU_HAL_BACKEND_H_
#define LOOM_TOOLS_IREE_RUN_LOOM_AMDGPU_HAL_BACKEND_H_

#include "loom/tools/iree-run-loom/hal_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const iree_amdgpu_hal_backend_t iree_run_loom_amdgpu_hal_backend;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_RUN_LOOM_AMDGPU_HAL_BACKEND_H_
