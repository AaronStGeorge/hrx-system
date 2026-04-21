// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Wasm target-low coverage provider.

#ifndef LOOM_TARGET_ARCH_WASM_COVERAGE_H_
#define LOOM_TARGET_ARCH_WASM_COVERAGE_H_

#include "loom/target/coverage.h"

#ifdef __cplusplus
extern "C" {
#endif

// Wasm target-low execution coverage rows.
extern const loom_target_coverage_provider_t loom_wasm_target_coverage_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_WASM_COVERAGE_H_
