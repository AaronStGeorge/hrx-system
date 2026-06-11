// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target assignment helpers for HAL artifact compilation.

#ifndef LOOM_TOOLING_EXECUTION_HAL_TARGET_ASSIGNMENT_H_
#define LOOM_TOOLING_EXECUTION_HAL_TARGET_ASSIGNMENT_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/tooling/execution/hal/artifact.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_targetless_kernel_assignment_result_t {
  // Target record assigned to targetless kernels, or null when unchanged.
  loom_symbol_ref_t target_ref;
  // Number of top-level kernel.def ops that had no target attr.
  uint32_t targetless_kernel_count;
  // Number of top-level kernel.def ops updated by the assignment.
  uint32_t assigned_kernel_count;
  // True when at least one kernel.def target attr was assigned.
  bool changed;
} loom_run_hal_targetless_kernel_assignment_result_t;

// Counts top-level kernel.def ops that do not have a target attr.
iree_status_t loom_run_hal_count_targetless_kernels(loom_module_t* module,
                                                    uint32_t* out_count);

// Assigns |device_target| to every targetless top-level kernel.def in |module|.
//
// The artifact provider is called only when at least one targetless kernel is
// present. Explicit kernel targets are preserved. The resolved target ref must
// name a valid target-like op in |module|.
iree_status_t loom_run_hal_assign_targetless_kernel_targets(
    const loom_run_hal_artifact_provider_t* artifact_provider,
    const loom_run_hal_device_target_t* device_target, loom_module_t* module,
    loom_run_hal_targetless_kernel_assignment_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_TARGET_ASSIGNMENT_H_
