// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared loom-benchmark command-line implementation.

#ifndef LOOM_TOOLS_LOOM_BENCHMARK_MAIN_H_
#define LOOM_TOOLS_LOOM_BENCHMARK_MAIN_H_

#include "iree/base/api.h"
#include "loom/tools/loom-benchmark/configuration.h"

#ifdef __cplusplus
extern "C" {
#endif

// Runs the configured loom-benchmark command-line tool.
int loom_benchmark_main(int argc, char** argv,
                        const loom_benchmark_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_BENCHMARK_MAIN_H_
