// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal_runtime.h"

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"

iree_status_t loom_run_hal_runtime_initialize(
    const loom_run_hal_backend_t* backend, iree_allocator_t allocator,
    loom_run_hal_runtime_t* out_runtime) {
  IREE_ASSERT_ARGUMENT(backend);
  IREE_ASSERT_ARGUMENT(out_runtime);
  *out_runtime = (loom_run_hal_runtime_t){0};

  iree_async_proactor_pool_t* proactor_pool = NULL;
  iree_status_t status =
      iree_tooling_create_instance(allocator, &out_runtime->instance);
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), allocator, &proactor_pool);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_create_device_from_flags(
        iree_hal_available_driver_registry(), backend->hal_driver_name,
        &create_params, allocator, &out_runtime->device);
  }
  iree_async_proactor_pool_release(proactor_pool);
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_cache_create(
        out_runtime->device, IREE_SV("loom"), &out_runtime->executable_cache);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_runtime_deinitialize(out_runtime);
  }
  return status;
}

void loom_run_hal_runtime_deinitialize(loom_run_hal_runtime_t* runtime) {
  if (runtime == NULL) {
    return;
  }
  iree_hal_executable_cache_release(runtime->executable_cache);
  iree_hal_device_release(runtime->device);
  iree_vm_instance_release(runtime->instance);
  *runtime = (loom_run_hal_runtime_t){0};
}
