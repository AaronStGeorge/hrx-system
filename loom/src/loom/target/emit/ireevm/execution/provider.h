// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Execution provider for IREE VM archive emission.

#ifndef LOOM_TARGET_EMIT_IREEVM_EXECUTION_PROVIDER_H_
#define LOOM_TARGET_EMIT_IREEVM_EXECUTION_PROVIDER_H_

#include "loom/tooling/execution/execution_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// IREE VM target-low descriptor package for run/benchmark/tune tools.
extern const loom_run_execution_provider_t loom_ireevm_execution_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_IREEVM_EXECUTION_PROVIDER_H_
