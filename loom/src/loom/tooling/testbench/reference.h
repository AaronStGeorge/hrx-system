// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Host reference oracles for check.oracle.call.

#ifndef LOOM_TOOLING_TESTBENCH_REFERENCE_H_
#define LOOM_TOOLING_TESTBENCH_REFERENCE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/tooling/testbench/invocation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_testbench_reference_matmul_oracle_options_t {
  // Optional HAL device used for result-buffer generation transfers.
  iree_hal_device_t* device;
  // HAL allocator used to allocate the result buffer view.
  iree_hal_allocator_t* device_allocator;
  // Buffer placement used for the result buffer view.
  iree_hal_buffer_params_t result_buffer_params;
  // Host allocator used for transient accumulation storage.
  iree_allocator_t host_allocator;
} loom_testbench_reference_matmul_oracle_options_t;

// Initializes the reference.matmul oracle provider.
//
// The provider computes dense row-major C = A * B + C_init. Inputs must be
// three rank-2 buffer views: lhs [M, K], rhs [K, N], and init [M, N]. The
// returned result is a rank-2 f32 buffer view [M, N]. |options| is borrowed by
// the provider and must outlive any invocation using |out_provider|.
void loom_testbench_reference_matmul_oracle_provider_initialize(
    const loom_testbench_reference_matmul_oracle_options_t* options,
    loom_testbench_oracle_provider_t* out_provider);

// Initializes the reference.tiled_matmul oracle provider.
//
// The provider computes tile-packed C = A * B + C_init. Inputs must be three
// rank-4 buffer views: lhs [Mtile, Ktile, M, K], rhs [Ktile, Ntile, K, N], and
// init [Mtile, Ntile, M, N]. The returned result is a rank-4 f32 buffer view
// [Mtile, Ntile, M, N]. |options| is borrowed by the provider and must outlive
// any invocation using |out_provider|.
void loom_testbench_reference_tiled_matmul_oracle_provider_initialize(
    const loom_testbench_reference_matmul_oracle_options_t* options,
    loom_testbench_oracle_provider_t* out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_REFERENCE_H_
