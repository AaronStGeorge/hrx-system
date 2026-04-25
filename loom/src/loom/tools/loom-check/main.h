// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared loom-check command-line entry point.
//
// Tool binaries provide a loom_check_environment_t that selects the dialects,
// target-low descriptor package, and source-to-low lowering policies linked
// into that binary. The shared entry point owns flag parsing, file IO, test
// update handling, JSON output, and result reporting.

#ifndef LOOM_TOOLS_LOOM_CHECK_MAIN_H_
#define LOOM_TOOLS_LOOM_CHECK_MAIN_H_

#include "iree/base/api.h"
#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the production Loom dialect surface without the synthetic test
// dialect. This is suitable for backend-owned .loom-test runners.
iree_status_t loom_check_register_production_context(void* user_data,
                                                     loom_context_t* context);

// Runs loom-check using |environment| as the linked tool environment.
int loom_check_main(int argc, char** argv,
                    const loom_check_environment_t* environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_MAIN_H_
