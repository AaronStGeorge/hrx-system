// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/context.h"

void loom_benchmark_hal_context_initialize(
    const loom_benchmark_configuration_t* configuration,
    iree_allocator_t host_allocator,
    loom_benchmark_hal_context_t* out_context) {
  IREE_ASSERT_ARGUMENT(configuration);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = (loom_benchmark_hal_context_t){
      .configuration = configuration,
  };
  loom_run_hal_testbench_context_initialize(
      configuration->target_environment, configuration->hal_target_routes,
      configuration->hal_target_route_count, host_allocator,
      &out_context->execution);
}

void loom_benchmark_hal_context_deinitialize(
    loom_benchmark_hal_context_t* context) {
  if (context == NULL) {
    return;
  }
  loom_run_hal_testbench_context_deinitialize(&context->execution);
  *context = (loom_benchmark_hal_context_t){0};
}
