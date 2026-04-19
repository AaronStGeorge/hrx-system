// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Wasm source-to-target-low lowering policy.
//
// This package-local policy plugs the generic source-to-low lowerer into the
// wasm.core.simd128 descriptor set. Keeping the policy here prevents Wasm
// descriptor knowledge from leaking into the target-independent codegen/low
// layer.

#ifndef LOOM_TARGET_EMIT_WASM_LOWER_H_
#define LOOM_TARGET_EMIT_WASM_LOWER_H_

#include "loom/codegen/low/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the package-local lowering policy for the Wasm scalar and SIMD
// subset.
//
// The policy maps i32 source values to reg<wasm.i32>, vector<4xi32> source
// values to reg<wasm.v128>, and currently lowers scalar.constant,
// scalar.addi/subi, and vector.addi/muli over those target-legal values.
// Unsupported source ops are rejected through the generic backend diagnostic
// sink instead of producing partial low IR.
const loom_low_lower_policy_t* loom_wasm_low_lower_policy(void);

// Initializes a target-owned registry mapping Wasm target-contract keys to
// their source-to-low lowering policies.
void loom_wasm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_LOWER_H_
