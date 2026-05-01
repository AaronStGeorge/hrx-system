// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/provider.h"

#include <stdio.h>

#include "loom/codegen/low/lower.h"
#include "loom/tools/loom-check/main.h"

enum {
  LOOM_CHECK_PROVIDER_DESCRIPTOR_SET_PROVIDER_CAPACITY = 256,
  LOOM_CHECK_PROVIDER_TARGET_BUNDLE_CAPACITY = 256,
  LOOM_CHECK_PROVIDER_LOW_LOWER_POLICY_CAPACITY = 128,
  LOOM_CHECK_PROVIDER_LOW_LEGALITY_PROVIDER_CAPACITY = 64,
  LOOM_CHECK_PROVIDER_LOW_PACKET_DIAGNOSTIC_PROVIDER_CAPACITY = 64,
  LOOM_CHECK_PROVIDER_EMIT_PROVIDER_CAPACITY = 64,
  LOOM_CHECK_PROVIDER_REQUIREMENT_PROVIDER_CAPACITY = 64,
};

typedef struct loom_check_provider_environment_state_t {
  // Provider table selected by the linked binary or embedding.
  const loom_check_provider_set_t* provider_set;
  // Descriptor-set provider scratch table assembled on demand.
  loom_low_descriptor_set_provider_t descriptor_set_providers
      [LOOM_CHECK_PROVIDER_DESCRIPTOR_SET_PROVIDER_CAPACITY];
  // Number of entries in |descriptor_set_providers|.
  iree_host_size_t descriptor_set_provider_count;
  // Target bundle scratch table assembled on demand.
  const loom_target_bundle_t*
      target_bundles[LOOM_CHECK_PROVIDER_TARGET_BUNDLE_CAPACITY];
  // Number of entries in |target_bundles|.
  iree_host_size_t target_bundle_count;
  // Source-to-low policy scratch table assembled on demand.
  loom_low_lower_policy_registry_entry_t
      low_lower_policy_entries[LOOM_CHECK_PROVIDER_LOW_LOWER_POLICY_CAPACITY];
  // Number of entries in |low_lower_policy_entries|.
  iree_host_size_t low_lower_policy_entry_count;
  // Target-low source legality provider table assembled once for the
  // environment.
  const loom_target_low_legality_provider_t* low_legality_providers
      [LOOM_CHECK_PROVIDER_LOW_LEGALITY_PROVIDER_CAPACITY];
  // Number of entries in |low_legality_providers|.
  iree_host_size_t low_legality_provider_count;
  // Target-low packet diagnostic provider table assembled once for the
  // environment.
  const loom_target_low_packet_diagnostic_provider_t*
      low_packet_diagnostic_providers
          [LOOM_CHECK_PROVIDER_LOW_PACKET_DIAGNOSTIC_PROVIDER_CAPACITY];
  // Number of entries in |low_packet_diagnostic_providers|.
  iree_host_size_t low_packet_diagnostic_provider_count;
  // Emit provider table assembled once for the environment.
  const loom_check_emit_provider_t*
      emit_providers[LOOM_CHECK_PROVIDER_EMIT_PROVIDER_CAPACITY];
  // Number of entries in |emit_providers|.
  iree_host_size_t emit_provider_count;
  // Requirement provider table assembled once for the environment.
  const loom_check_requirement_provider_t*
      requirement_providers[LOOM_CHECK_PROVIDER_REQUIREMENT_PROVIDER_CAPACITY];
  // Number of entries in |requirement_providers|.
  iree_host_size_t requirement_provider_count;
} loom_check_provider_environment_state_t;

static iree_status_t loom_check_provider_append_low_legality_providers(
    loom_check_provider_environment_state_t* state,
    const loom_check_provider_t* provider) {
  if (state->low_legality_provider_count +
          provider->low_legality_provider_list.count >
      IREE_ARRAYSIZE(state->low_legality_providers)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "loom-check low legality provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->low_legality_provider_list.count;
       ++i) {
    state->low_legality_providers[state->low_legality_provider_count++] =
        provider->low_legality_provider_list.values[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_check_provider_append_low_packet_diagnostic_providers(
    loom_check_provider_environment_state_t* state,
    const loom_check_provider_t* provider) {
  if (state->low_packet_diagnostic_provider_count +
          provider->low_packet_diagnostic_provider_list.count >
      IREE_ARRAYSIZE(state->low_packet_diagnostic_providers)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "loom-check low packet diagnostic provider capacity exceeded");
  }
  for (iree_host_size_t i = 0;
       i < provider->low_packet_diagnostic_provider_list.count; ++i) {
    const iree_host_size_t output_index =
        state->low_packet_diagnostic_provider_count++;
    state->low_packet_diagnostic_providers[output_index] =
        provider->low_packet_diagnostic_provider_list.values[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_check_provider_append_emit_providers(
    loom_check_provider_environment_state_t* state,
    const loom_check_provider_t* provider) {
  if (state->emit_provider_count + provider->emit_provider_count >
      IREE_ARRAYSIZE(state->emit_providers)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "loom-check emit provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->emit_provider_count; ++i) {
    state->emit_providers[state->emit_provider_count++] =
        provider->emit_providers[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_check_provider_append_requirement_providers(
    loom_check_provider_environment_state_t* state,
    const loom_check_provider_t* provider) {
  if (state->requirement_provider_count + provider->requirement_provider_count >
      IREE_ARRAYSIZE(state->requirement_providers)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "loom-check requirement provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->requirement_provider_count; ++i) {
    state->requirement_providers[state->requirement_provider_count++] =
        provider->requirement_providers[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_check_provider_environment_state_initialize(
    const loom_check_provider_set_t* provider_set,
    loom_check_provider_environment_state_t* out_state) {
  *out_state = (loom_check_provider_environment_state_t){
      .provider_set = provider_set,
  };

  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_check_provider_t* provider = provider_set->providers[i];
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_low_legality_providers(out_state, provider));
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_low_packet_diagnostic_providers(out_state,
                                                                   provider));
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_emit_providers(out_state, provider));
    IREE_RETURN_IF_ERROR(
        loom_check_provider_append_requirement_providers(out_state, provider));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_provider_initialize_low_descriptor_registry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  loom_check_provider_environment_state_t* state =
      (loom_check_provider_environment_state_t*)user_data;
  state->descriptor_set_provider_count = 0;
  state->target_bundle_count = 0;

  const loom_check_provider_set_t* provider_set = state->provider_set;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_check_provider_t* provider = provider_set->providers[i];
    if (provider->initialize_low_descriptor_registry == NULL) {
      continue;
    }
    loom_target_low_descriptor_registry_t provider_registry = {0};
    provider->initialize_low_descriptor_registry(&provider_registry);
    IREE_RETURN_IF_ERROR(loom_target_low_descriptor_registry_append_to_tables(
        &provider_registry, state->descriptor_set_providers,
        IREE_ARRAYSIZE(state->descriptor_set_providers),
        &state->descriptor_set_provider_count, state->target_bundles,
        IREE_ARRAYSIZE(state->target_bundles), &state->target_bundle_count));
  }

  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, state->descriptor_set_providers,
      state->descriptor_set_provider_count, state->target_bundles,
      state->target_bundle_count);
  return iree_ok_status();
}

static iree_status_t loom_check_provider_initialize_low_lower_policy_registry(
    void* user_data, loom_low_lower_policy_registry_t* out_registry) {
  loom_check_provider_environment_state_t* state =
      (loom_check_provider_environment_state_t*)user_data;
  state->low_lower_policy_entry_count = 0;

  const loom_check_provider_set_t* provider_set = state->provider_set;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_check_provider_t* provider = provider_set->providers[i];
    if (provider->initialize_low_lower_policy_registry == NULL) {
      continue;
    }
    loom_low_lower_policy_registry_t provider_registry = {0};
    provider->initialize_low_lower_policy_registry(&provider_registry);
    if (state->low_lower_policy_entry_count + provider_registry.entry_count >
        IREE_ARRAYSIZE(state->low_lower_policy_entries)) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "loom-check source-to-low policy capacity exceeded");
    }
    for (iree_host_size_t j = 0; j < provider_registry.entry_count; ++j) {
      state->low_lower_policy_entries[state->low_lower_policy_entry_count++] =
          provider_registry.entries[j];
    }
  }

  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, state->low_lower_policy_entries,
      state->low_lower_policy_entry_count);
  return iree_ok_status();
}

int loom_check_provider_main(int argc, char** argv,
                             const loom_check_provider_set_t* provider_set) {
  loom_check_provider_environment_state_t state;
  iree_status_t status =
      loom_check_provider_environment_state_initialize(provider_set, &state);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }

  const loom_check_environment_t environment = {
      .register_context =
          {
              .fn = loom_check_register_production_context,
              .user_data = NULL,
          },
      .initialize_low_descriptor_registry =
          {
              .fn = loom_check_provider_initialize_low_descriptor_registry,
              .user_data = &state,
          },
      .initialize_low_lower_policy_registry =
          {
              .fn = loom_check_provider_initialize_low_lower_policy_registry,
              .user_data = &state,
          },
      .low_legality_provider_list = loom_target_low_legality_provider_list_make(
          state.low_legality_providers, state.low_legality_provider_count),
      .low_packet_diagnostic_provider_list =
          loom_target_low_packet_diagnostic_provider_list_make(
              state.low_packet_diagnostic_providers,
              state.low_packet_diagnostic_provider_count),
      .emit_providers =
          {
              .providers = state.emit_providers,
              .provider_count = state.emit_provider_count,
          },
      .requirement_providers =
          {
              .providers = state.requirement_providers,
              .provider_count = state.requirement_provider_count,
          },
  };
  return loom_check_main(argc, argv, &environment);
}
