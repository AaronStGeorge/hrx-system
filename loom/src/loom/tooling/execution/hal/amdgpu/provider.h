// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU execution-tool provider composed from the shared HAL runner.

#ifndef LOOM_TOOLING_EXECUTION_HAL_AMDGPU_PROVIDER_H_
#define LOOM_TOOLING_EXECUTION_HAL_AMDGPU_PROVIDER_H_

#include "loom/tooling/execution/execution_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Provider contribution for tools that compile and run AMDGPU HAL artifacts.
extern const loom_run_execution_provider_t loom_amdgpu_hal_execution_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_AMDGPU_PROVIDER_H_
