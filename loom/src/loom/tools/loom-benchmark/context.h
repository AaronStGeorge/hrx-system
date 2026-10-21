// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmark HAL context lifecycle.

#ifndef LOOM_TOOLS_LOOM_BENCHMARK_CONTEXT_H_
#define LOOM_TOOLS_LOOM_BENCHMARK_CONTEXT_H_

#include "iree/base/api.h"
#include "loom/tools/loom-benchmark/configuration.h"
#include "loom/tools/loom-benchmark/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes shared HAL state used by benchmark compilation and dispatch.
void loom_benchmark_hal_context_initialize(
    const loom_benchmark_configuration_t* configuration,
    iree_allocator_t host_allocator, loom_benchmark_hal_context_t* out_context);

// Releases HAL state owned by |context|.
void loom_benchmark_hal_context_deinitialize(
    loom_benchmark_hal_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_BENCHMARK_CONTEXT_H_
