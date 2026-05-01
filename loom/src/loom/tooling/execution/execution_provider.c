// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/execution_provider.h"

static iree_status_t loom_run_execution_environment_append_hal_backends(
    loom_run_execution_environment_t* environment,
    const loom_run_execution_provider_t* provider) {
  if (environment->hal_backend_count + provider->hal_backend_count >
      IREE_ARRAYSIZE(environment->hal_backends)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "loom execution HAL backend capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->hal_backend_count; ++i) {
    environment->hal_backends[environment->hal_backend_count++] =
        provider->hal_backends[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_run_execution_environment_append_execution_backends(
    loom_run_execution_environment_t* environment,
    const loom_run_execution_provider_t* provider) {
  if (environment->execution_backend_count + provider->execution_backend_count >
      IREE_ARRAYSIZE(environment->execution_backends)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "loom execution backend capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->execution_backend_count; ++i) {
    environment->execution_backends[environment->execution_backend_count++] =
        provider->execution_backends[i];
  }
  return iree_ok_status();
}

iree_status_t loom_run_execution_environment_initialize(
    const loom_run_execution_provider_set_t* provider_set,
    loom_run_execution_environment_t* out_environment) {
  *out_environment = (loom_run_execution_environment_t){
      .provider_set = provider_set,
  };

  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_run_execution_provider_t* provider = provider_set->providers[i];
    IREE_RETURN_IF_ERROR(loom_run_execution_environment_append_hal_backends(
        out_environment, provider));
    IREE_RETURN_IF_ERROR(
        loom_run_execution_environment_append_execution_backends(
            out_environment, provider));
  }
  loom_run_hal_backend_registry_initialize_from_entries(
      out_environment->hal_backends, out_environment->hal_backend_count,
      &out_environment->hal_backend_registry);
  loom_run_execution_backend_registry_initialize_from_entries(
      out_environment->execution_backends,
      out_environment->execution_backend_count,
      &out_environment->execution_backend_registry);
  return iree_ok_status();
}

void loom_run_execution_environment_deinitialize(
    loom_run_execution_environment_t* environment) {
  if (environment == NULL) {
    return;
  }
  *environment = (loom_run_execution_environment_t){0};
}

static iree_status_t
loom_run_execution_environment_initialize_low_descriptor_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  loom_run_execution_environment_t* environment =
      (loom_run_execution_environment_t*)user_data;
  environment->descriptor_set_provider_count = 0;
  environment->target_bundle_count = 0;

  const loom_run_execution_provider_set_t* provider_set =
      environment->provider_set;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_run_execution_provider_t* provider = provider_set->providers[i];
    if (provider->initialize_low_descriptor_registry == NULL) {
      continue;
    }
    loom_target_low_descriptor_registry_t provider_registry = {0};
    provider->initialize_low_descriptor_registry(&provider_registry);
    IREE_RETURN_IF_ERROR(loom_target_low_descriptor_registry_append_to_tables(
        &provider_registry, environment->descriptor_set_providers,
        IREE_ARRAYSIZE(environment->descriptor_set_providers),
        &environment->descriptor_set_provider_count,
        environment->target_bundles,
        IREE_ARRAYSIZE(environment->target_bundles),
        &environment->target_bundle_count));
  }

  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, environment->descriptor_set_providers,
      environment->descriptor_set_provider_count, environment->target_bundles,
      environment->target_bundle_count);
  return iree_ok_status();
}

loom_run_initialize_low_descriptor_registry_callback_t
loom_run_execution_environment_low_descriptor_registry_callback(
    loom_run_execution_environment_t* environment) {
  return (loom_run_initialize_low_descriptor_registry_callback_t){
      .fn = loom_run_execution_environment_initialize_low_descriptor_registry,
      .user_data = environment,
  };
}

const loom_run_hal_backend_registry_t*
loom_run_execution_environment_hal_backend_registry(
    const loom_run_execution_environment_t* environment) {
  return &environment->hal_backend_registry;
}

const loom_run_execution_backend_registry_t*
loom_run_execution_environment_execution_backend_registry(
    const loom_run_execution_environment_t* environment) {
  return &environment->execution_backend_registry;
}
