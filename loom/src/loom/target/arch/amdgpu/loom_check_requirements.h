// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU requirement probes for target-owned loom-check runners.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOOM_CHECK_REQUIREMENTS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOOM_CHECK_REQUIREMENTS_H_

#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Requirement provider for AMDGPU HAL availability and feature checks.
extern const loom_check_requirement_provider_t
    loom_amdgpu_loom_check_requirement_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOOM_CHECK_REQUIREMENTS_H_
