// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 source-to-target-low lowering policy.
//
// The x86 arch package owns descriptor-level lowering decisions that should be
// shared by native assembly, LLVMIR, and future direct object emitters.

#ifndef LOOM_TARGET_ARCH_X86_LOWER_H_
#define LOOM_TARGET_ARCH_X86_LOWER_H_

#include "loom/codegen/low/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the AVX512 register lowering policy.
//
// The initial policy maps vector<16xi32> values to ZMM registers and lowers
// vector.addi to x86.avx512.vpaddd.zmm. Memory operands and object-function ABI
// pinning are intentionally owned by later low.resource/ABI work.
const loom_low_lower_policy_t* loom_x86_low_lower_policy(void);

// Initializes a target-owned registry mapping x86 target-contract keys to
// their source-to-low lowering policies.
void loom_x86_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_LOWER_H_
