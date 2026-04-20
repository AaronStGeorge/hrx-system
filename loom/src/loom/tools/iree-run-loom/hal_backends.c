// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-run-loom/amdgpu_hal_backend.h"
#include "loom/tools/iree-run-loom/hal_backend.h"

static const iree_amdgpu_hal_backend_t* const kIreeRunLoomHalBackends[] = {
    &iree_run_loom_amdgpu_hal_backend,
};

void iree_amdgpu_hal_backend_registry_initialize(
    iree_amdgpu_hal_backend_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(out_registry);
  *out_registry = (iree_amdgpu_hal_backend_registry_t){
      .backends = kIreeRunLoomHalBackends,
      .backend_count = IREE_ARRAYSIZE(kIreeRunLoomHalBackends),
  };
}

const iree_amdgpu_hal_backend_t* iree_amdgpu_hal_backend_registry_lookup(
    const iree_amdgpu_hal_backend_registry_t* registry,
    iree_string_view_t name) {
  if (!registry) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < registry->backend_count; ++i) {
    const iree_amdgpu_hal_backend_t* backend = registry->backends[i];
    if (backend && iree_string_view_equal(backend->name, name)) {
      return backend;
    }
  }
  return NULL;
}

iree_status_t iree_amdgpu_hal_backend_registry_format_names(
    const iree_amdgpu_hal_backend_registry_t* registry,
    iree_string_builder_t* output) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(output);
  iree_host_size_t appended_count = 0;
  for (iree_host_size_t i = 0; i < registry->backend_count; ++i) {
    const iree_amdgpu_hal_backend_t* backend = registry->backends[i];
    if (!backend) {
      continue;
    }
    if (appended_count > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, ", "));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_string(output, backend->name));
    ++appended_count;
  }
  return iree_ok_status();
}
