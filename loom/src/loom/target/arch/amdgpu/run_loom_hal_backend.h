// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU HAL backend provider for Loom execution tools.

#ifndef LOOM_TARGET_ARCH_AMDGPU_RUN_LOOM_HAL_BACKEND_H_
#define LOOM_TARGET_ARCH_AMDGPU_RUN_LOOM_HAL_BACKEND_H_

#include "loom/tooling/execution/hal_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_run_hal_backend_t iree_run_loom_amdgpu_hal_backend;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_RUN_LOOM_HAL_BACKEND_H_
