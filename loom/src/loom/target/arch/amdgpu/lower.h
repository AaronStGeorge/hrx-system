// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-to-target-low lowering policy.
//
// This policy owns the architecture-specific mapping from target-legal source
// Loom ops to AMDGPU descriptor-backed low ops. It deliberately stays below
// native assembly and LLVMIR emission so the same low representation can feed
// every later AMDGPU backend path.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_H_

#include "loom/codegen/low/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the AMDGPU scalar-register lowering policy.
//
// The initial policy is intentionally narrow: it maps i32 scalar values to
// SGPRs and lowers scalar.constant/scalar.addi to s_mov_b32/s_add_u32. Memory,
// resource, and per-lane VGPR lowering require low.resource and lane-uniformity
// facts and are not guessed here.
const loom_low_lower_policy_t* loom_amdgpu_low_lower_policy(void);

// Initializes a target-owned registry mapping AMDGPU target-contract keys to
// their source-to-low lowering policies.
void loom_amdgpu_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_H_
