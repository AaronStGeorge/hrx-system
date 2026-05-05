// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/provider.h"

static iree_status_t loom_target_environment_append_low_descriptor_registry(
    loom_target_environment_t* environment,
    const loom_target_provider_t* provider) {
  if (provider->initialize_low_descriptor_registry == NULL) {
    return iree_ok_status();
  }
  loom_target_low_descriptor_registry_t provider_registry = {0};
  provider->initialize_low_descriptor_registry(&provider_registry);
  return loom_target_low_descriptor_registry_append_to_tables(
      &provider_registry, environment->descriptor_set_providers,
      IREE_ARRAYSIZE(environment->descriptor_set_providers),
      &environment->descriptor_set_provider_count);
}

static iree_status_t loom_target_environment_append_low_lower_policy_registry(
    loom_target_environment_t* environment,
    const loom_target_provider_t* provider) {
  if (provider->initialize_low_lower_policy_registry == NULL) {
    return iree_ok_status();
  }
  loom_low_lower_policy_registry_t provider_registry = {0};
  provider->initialize_low_lower_policy_registry(&provider_registry);
  if (environment->low_lower_policy_entry_count +
          provider_registry.entry_count >
      IREE_ARRAYSIZE(environment->low_lower_policy_entries)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target source-to-low policy capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider_registry.entry_count; ++i) {
    environment->low_lower_policy_entries
        [environment->low_lower_policy_entry_count++] =
        provider_registry.entries[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_target_environment_append_low_legality_providers(
    loom_target_environment_t* environment,
    const loom_target_provider_t* provider) {
  if (environment->low_legality_provider_count +
          provider->low_legality_provider_list.count >
      IREE_ARRAYSIZE(environment->low_legality_providers)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target low legality provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->low_legality_provider_list.count;
       ++i) {
    environment
        ->low_legality_providers[environment->low_legality_provider_count++] =
        provider->low_legality_provider_list.values[i];
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_environment_append_low_packet_diagnostic_providers(
    loom_target_environment_t* environment,
    const loom_target_provider_t* provider) {
  if (environment->low_packet_diagnostic_provider_count +
          provider->low_packet_diagnostic_provider_list.count >
      IREE_ARRAYSIZE(environment->low_packet_diagnostic_providers)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "target low packet diagnostic provider capacity exceeded");
  }
  for (iree_host_size_t i = 0;
       i < provider->low_packet_diagnostic_provider_list.count; ++i) {
    environment->low_packet_diagnostic_providers
        [environment->low_packet_diagnostic_provider_count++] =
        provider->low_packet_diagnostic_provider_list.values[i];
  }
  return iree_ok_status();
}

iree_status_t loom_target_environment_initialize(
    const loom_target_provider_set_t* provider_set,
    loom_target_environment_t* out_environment) {
  IREE_ASSERT_ARGUMENT(provider_set);
  IREE_ASSERT_ARGUMENT(out_environment);
  *out_environment = (loom_target_environment_t){
      .provider_set = provider_set,
  };

  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_provider_t* provider = provider_set->providers[i];
    IREE_RETURN_IF_ERROR(loom_target_environment_append_low_descriptor_registry(
        out_environment, provider));
    IREE_RETURN_IF_ERROR(
        loom_target_environment_append_low_lower_policy_registry(
            out_environment, provider));
    IREE_RETURN_IF_ERROR(loom_target_environment_append_low_legality_providers(
        out_environment, provider));
    IREE_RETURN_IF_ERROR(
        loom_target_environment_append_low_packet_diagnostic_providers(
            out_environment, provider));
  }
  return iree_ok_status();
}

void loom_target_environment_deinitialize(
    loom_target_environment_t* environment) {
  if (environment == NULL) {
    return;
  }
  *environment = (loom_target_environment_t){0};
}

iree_status_t loom_target_environment_register_context(
    const loom_target_environment_t* environment, loom_context_t* context) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(context);
  const loom_target_provider_set_t* provider_set = environment->provider_set;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_provider_t* provider = provider_set->providers[i];
    if (provider->register_context == NULL) {
      continue;
    }
    IREE_RETURN_IF_ERROR(provider->register_context(context));
  }
  return iree_ok_status();
}

iree_status_t loom_target_environment_initialize_low_descriptor_registry(
    const loom_target_environment_t* environment,
    loom_target_low_descriptor_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(out_registry);
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, environment->descriptor_set_providers,
      environment->descriptor_set_provider_count);
  return iree_ok_status();
}

iree_status_t loom_target_environment_initialize_low_lower_policy_registry(
    const loom_target_environment_t* environment,
    loom_low_lower_policy_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(out_registry);
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, environment->low_lower_policy_entries,
      environment->low_lower_policy_entry_count);
  return iree_ok_status();
}

loom_target_low_legality_provider_list_t
loom_target_environment_low_legality_provider_list(
    const loom_target_environment_t* environment) {
  IREE_ASSERT_ARGUMENT(environment);
  return loom_target_low_legality_provider_list_make(
      environment->low_legality_providers,
      environment->low_legality_provider_count);
}

loom_target_low_packet_diagnostic_provider_list_t
loom_target_environment_low_packet_diagnostic_provider_list(
    const loom_target_environment_t* environment) {
  IREE_ASSERT_ARGUMENT(environment);
  return loom_target_low_packet_diagnostic_provider_list_make(
      environment->low_packet_diagnostic_providers,
      environment->low_packet_diagnostic_provider_count);
}
