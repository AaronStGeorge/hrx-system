// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL executable candidates produced by Loom execution tooling.

#ifndef LOOM_TOOLING_EXECUTION_HAL_CANDIDATE_H_
#define LOOM_TOOLING_EXECUTION_HAL_CANDIDATE_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/tooling/execution/hal_backend.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_candidate_t {
  // Host allocator used for owned candidate storage.
  iree_allocator_t host_allocator;
  // Structured report for this candidate's compilation.
  loom_target_compile_report_t compile_report;
  // HAL backend that produced |executable|.
  const loom_run_hal_backend_t* backend;
  // HAL target selected during candidate compilation.
  loom_run_hal_selected_target_t target;
  // HAL executable bytes produced by |backend|.
  loom_run_hal_executable_t executable;
} loom_run_hal_candidate_t;

// Selects a HAL target through |backend| and compiles |run_module| to a HAL
// executable candidate. Compilation may mutate the parsed module by expanding
// target records and adding lowered IR.
iree_status_t loom_run_hal_candidate_compile(
    const loom_run_hal_backend_t* backend,
    const loom_run_hal_runtime_t* runtime, loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_run_hal_candidate_t* out_candidate);

// Releases all artifact storage owned by |candidate|.
void loom_run_hal_candidate_deinitialize(loom_run_hal_candidate_t* candidate);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_CANDIDATE_H_
