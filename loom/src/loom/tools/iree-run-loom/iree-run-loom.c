// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-run-loom binary with build-selected execution providers.

#include "loom/target/emit/ireevm/low_registry.h"
#include "loom/tools/iree-run-loom/main.h"

#ifndef IREE_RUN_LOOM_HAVE_AMDGPU
#define IREE_RUN_LOOM_HAVE_AMDGPU 0
#endif  // IREE_RUN_LOOM_HAVE_AMDGPU

#if IREE_RUN_LOOM_HAVE_AMDGPU
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/amdgpu_hal_backend.h"
#endif  // IREE_RUN_LOOM_HAVE_AMDGPU

enum {
  IREE_RUN_LOOM_LOW_DESCRIPTOR_SET_PROVIDER_CAPACITY = 64,
  IREE_RUN_LOOM_LOW_TARGET_BUNDLE_CAPACITY = 64,
};

typedef struct iree_run_loom_low_registry_tables_t {
  // Linked descriptor-set provider table assembled for this runner.
  loom_low_descriptor_set_provider_t descriptor_set_providers
      [IREE_RUN_LOOM_LOW_DESCRIPTOR_SET_PROVIDER_CAPACITY];
  // Number of valid entries in |descriptor_set_providers|.
  iree_host_size_t descriptor_set_provider_count;
  // Linked target-bundle table assembled for this runner.
  const loom_target_bundle_t*
      target_bundles[IREE_RUN_LOOM_LOW_TARGET_BUNDLE_CAPACITY];
  // Number of valid entries in |target_bundles|.
  iree_host_size_t target_bundle_count;
} iree_run_loom_low_registry_tables_t;

static iree_status_t iree_run_loom_append_low_registry_package(
    const loom_target_low_descriptor_registry_t* package_registry,
    iree_run_loom_low_registry_tables_t* tables) {
  IREE_ASSERT_ARGUMENT(tables);
  return loom_target_low_descriptor_registry_append_to_tables(
      package_registry, tables->descriptor_set_providers,
      IREE_ARRAYSIZE(tables->descriptor_set_providers),
      &tables->descriptor_set_provider_count, tables->target_bundles,
      IREE_ARRAYSIZE(tables->target_bundles), &tables->target_bundle_count);
}

static iree_status_t iree_run_loom_initialize_low_descriptor_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(user_data);
  IREE_ASSERT_ARGUMENT(out_registry);
  iree_run_loom_low_registry_tables_t* tables =
      (iree_run_loom_low_registry_tables_t*)user_data;
  *tables = (iree_run_loom_low_registry_tables_t){0};

  loom_target_low_descriptor_registry_t ireevm_registry = {};
  loom_ireevm_low_descriptor_registry_initialize(&ireevm_registry);
  IREE_RETURN_IF_ERROR(
      iree_run_loom_append_low_registry_package(&ireevm_registry, tables));

#if IREE_RUN_LOOM_HAVE_AMDGPU
  loom_target_low_descriptor_registry_t amdgpu_registry = {};
  loom_amdgpu_low_descriptor_registry_initialize(&amdgpu_registry);
  IREE_RETURN_IF_ERROR(
      iree_run_loom_append_low_registry_package(&amdgpu_registry, tables));
#endif  // IREE_RUN_LOOM_HAVE_AMDGPU

  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, tables->descriptor_set_providers,
      tables->descriptor_set_provider_count, tables->target_bundles,
      tables->target_bundle_count);
  return iree_ok_status();
}

int main(int argc, char** argv) {
#if IREE_RUN_LOOM_HAVE_AMDGPU
  static const loom_run_hal_backend_t* const kHalBackends[] = {
      &iree_run_loom_amdgpu_hal_backend,
  };
#endif  // IREE_RUN_LOOM_HAVE_AMDGPU
  iree_run_loom_low_registry_tables_t low_registry_tables = {0};

  loom_run_hal_backend_registry_t hal_backend_registry;
#if IREE_RUN_LOOM_HAVE_AMDGPU
  loom_run_hal_backend_registry_initialize_from_entries(
      kHalBackends, IREE_ARRAYSIZE(kHalBackends), &hal_backend_registry);
#else
  loom_run_hal_backend_registry_initialize_from_entries(NULL, 0,
                                                        &hal_backend_registry);
#endif  // IREE_RUN_LOOM_HAVE_AMDGPU

  const iree_run_loom_configuration_t configuration = {
      .tool_name = "iree-run-loom",
      .initialize_low_descriptor_registry =
          {
              .fn = iree_run_loom_initialize_low_descriptor_registry,
              .user_data = &low_registry_tables,
          },
      .hal_backend_registry = hal_backend_registry,
  };
  return iree_run_loom_main(argc, argv, &configuration);
}
