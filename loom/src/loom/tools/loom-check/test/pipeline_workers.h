// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLS_LOOM_CHECK_TEST_PIPELINE_WORKERS_H_
#define LOOM_TOOLS_LOOM_CHECK_TEST_PIPELINE_WORKERS_H_

#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Inspects shared worker transport for an outlined pipeline. The first formal
// is an abstract workgroup pool with 1024 bytes, independent of any target.
extern const loom_check_emit_provider_t
    loom_check_test_pipeline_workers_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_TEST_PIPELINE_WORKERS_H_
