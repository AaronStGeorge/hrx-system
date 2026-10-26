// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Adapts public LoomC diagnostics to loom-check annotation and JSON results.

#ifndef LOOM_TOOLS_LOOM_CHECK_COMPILE_DIAGNOSTICS_H_
#define LOOM_TOOLS_LOOM_CHECK_COMPILE_DIAGNOSTICS_H_

#include "loom/tools/loom-check/diagnostics.h"
#include "loomc/result.h"

#ifdef __cplusplus
extern "C" {
#endif

// Copies result diagnostics and reports compiler success in |out_succeeded|.
// Compiler rejection with an error diagnostic returns OK for annotation
// matching; rejection without an error diagnostic fails the check.
iree_status_t loom_check_compile_append_result(
    loom_check_diagnostic_collector_t* collector, const loomc_result_t* result,
    bool* out_succeeded);

// Copies every diagnostic in |source_result| into |collector|. Source
// identifiers already carry their logical identity from source admission.
iree_status_t loom_check_compile_append_result_diagnostics(
    loom_check_diagnostic_collector_t* collector,
    const loomc_result_t* source_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_COMPILE_DIAGNOSTICS_H_
