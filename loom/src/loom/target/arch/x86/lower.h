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
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the x86 register lowering policy.
//
// The policy is selected by contract set. The AVX512 core contract maps
// vector<16xi32> values to ZMM registers and lowers vector.addi/vector.muli to
// vpaddd/vpmulld packets. The packed-dot contract maps supported static dot
// vectors to XMM/YMM/ZMM registers and lowers target-legal vector.dot2f and
// vector.dot4i ops through descriptor-selected packed-dot packets. Memory
// operands and object-function ABI pinning are intentionally owned by later
// low.resource/ABI work.
const loom_low_lower_policy_t* loom_x86_low_lower_policy(void);

// Returns the x86 source legality provider for target-low lowering.
//
// The provider accepts vector dot ops only when the selected target bundle uses
// the x86 packed-dot descriptor contract and a descriptor is available for the
// source shape, payload interpretation, and target feature bits.
const loom_target_low_legality_provider_t* loom_x86_low_legality_provider(void);

// Static x86 source legality provider used by linked provider tables.
extern const loom_target_low_legality_provider_t
    loom_x86_low_legality_provider_storage;

// Initializes a target-owned registry mapping x86 target-contract keys to
// their source-to-low lowering policies.
void loom_x86_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_LOWER_H_
