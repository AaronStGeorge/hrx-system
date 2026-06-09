// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/target_registry.h"

#include "loom/target/emit/llvmir/amdgpu/legality.h"
#include "loom/target/emit/llvmir/amdgpu/lower.h"
#include "loom/target/emit/llvmir/amdgpu/target_env.h"
#include "loom/target/emit/llvmir/test_target.h"
#include "loom/target/emit/llvmir/x86/legality.h"
#include "loom/target/emit/llvmir/x86/lower.h"
#include "loom/target/emit/llvmir/x86/target_env.h"

void loom_llvmir_target_registry_initialize(
    loom_llvmir_target_registry_t* out_registry) {
  *out_registry = (loom_llvmir_target_registry_t){0};
  out_registry->profile_providers[0] =
      loom_llvmir_x86_target_profile_provider();
  out_registry->profile_providers[1] =
      loom_llvmir_amdgpu_target_profile_provider();
  out_registry->bundles[0] = loom_llvmir_target_bundle_test_object();
  out_registry->bundles[1] = loom_llvmir_target_bundle_x86_64_object();
  out_registry->bundles[2] =
      loom_llvmir_target_bundle_x86_64_packed_dot_object();
  out_registry->bundles[3] = loom_llvmir_target_bundle_amdgpu_hal();
  out_registry->profile_registry = (loom_llvmir_target_profile_registry_t){
      .default_profile = loom_llvmir_target_profile_test_object(),
      .providers = out_registry->profile_providers,
      .provider_count = IREE_ARRAYSIZE(out_registry->profile_providers),
  };
  out_registry->bundle_count = IREE_ARRAYSIZE(out_registry->bundles);
}

bool loom_llvmir_target_registry_project_bundle(
    const loom_llvmir_target_registry_t* registry,
    const loom_target_bundle_t* bundle,
    const loom_llvmir_target_profile_t** out_profile,
    const loom_llvmir_target_profile_provider_t** out_provider) {
  *out_profile = NULL;
  if (out_provider != NULL) {
    *out_provider = NULL;
  }
  if (registry->profile_registry.default_profile != NULL &&
      iree_string_view_equal(
          bundle->name, registry->profile_registry.default_profile->name)) {
    *out_profile = registry->profile_registry.default_profile;
    return true;
  }
  for (iree_host_size_t provider_ordinal = 0;
       provider_ordinal < registry->profile_registry.provider_count;
       ++provider_ordinal) {
    const loom_llvmir_target_profile_provider_t* provider =
        registry->profile_registry.providers[provider_ordinal];
    const loom_llvmir_target_profile_t* profile = NULL;
    if (provider->project_bundle(bundle, &profile)) {
      *out_profile = profile;
      if (out_provider != NULL) {
        *out_provider = provider;
      }
      return true;
    }
  }
  return false;
}

bool loom_llvmir_target_registry_lookup_bundle(
    const loom_llvmir_target_registry_t* registry, iree_string_view_t name,
    const loom_target_bundle_t** out_bundle) {
  *out_bundle = NULL;

  name = iree_string_view_trim(name);
  if (iree_string_view_is_empty(name)) {
    *out_bundle = registry->bundles[0];
    return *out_bundle != NULL;
  }
  for (iree_host_size_t i = 0; i < registry->bundle_count; ++i) {
    const loom_target_bundle_t* bundle = registry->bundles[i];
    if (iree_string_view_equal(name, bundle->name)) {
      *out_bundle = bundle;
      return true;
    }
  }
  return false;
}

void loom_llvmir_target_legality_provider_list_select(
    const loom_llvmir_target_profile_provider_t* profile_provider,
    loom_llvmir_target_legality_provider_list_t* out_providers) {
  *out_providers = (loom_llvmir_target_legality_provider_list_t){0};
  if (profile_provider == NULL) {
    return;
  }
  if (profile_provider == loom_llvmir_x86_target_profile_provider()) {
    out_providers->providers[out_providers->provider_count++] =
        loom_llvmir_x86_legality_provider();
  }
  if (profile_provider == loom_llvmir_amdgpu_target_profile_provider()) {
    out_providers->providers[out_providers->provider_count++] =
        loom_llvmir_amdgpu_legality_provider();
  }
}

void loom_llvmir_lowering_provider_list_select(
    const loom_llvmir_target_profile_provider_t* profile_provider,
    loom_llvmir_lowering_provider_list_t* out_providers) {
  *out_providers = (loom_llvmir_lowering_provider_list_t){0};
  if (profile_provider == NULL) {
    return;
  }
  if (profile_provider == loom_llvmir_x86_target_profile_provider()) {
    out_providers->providers[out_providers->provider_count++] =
        loom_llvmir_x86_lowering_provider();
  }
  if (profile_provider == loom_llvmir_amdgpu_target_profile_provider()) {
    out_providers->providers[out_providers->provider_count++] =
        loom_llvmir_amdgpu_lowering_provider();
  }
}

bool loom_llvmir_target_registry_lookup_profile(
    const loom_llvmir_target_registry_t* registry,
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile) {
  return loom_llvmir_target_profile_registry_lookup(&registry->profile_registry,
                                                    profile_name, out_profile);
}

bool loom_llvmir_target_registry_lookup_profile_provider(
    const loom_llvmir_target_registry_t* registry,
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile,
    const loom_llvmir_target_profile_provider_t** out_provider) {
  if (out_profile != NULL) {
    *out_profile = NULL;
  }
  *out_provider = NULL;

  const loom_llvmir_target_profile_t* profile = NULL;
  if (!loom_llvmir_target_registry_lookup_profile(registry, profile_name,
                                                  &profile)) {
    return false;
  }

  for (iree_host_size_t provider_ordinal = 0;
       provider_ordinal < registry->profile_registry.provider_count;
       ++provider_ordinal) {
    const loom_llvmir_target_profile_provider_t* provider =
        registry->profile_registry.providers[provider_ordinal];
    for (iree_host_size_t profile_ordinal = 0;
         profile_ordinal < provider->profile_count; ++profile_ordinal) {
      const loom_llvmir_target_profile_t* provider_profile =
          provider->profiles[profile_ordinal];
      if (provider_profile == profile ||
          iree_string_view_equal(provider_profile->name, profile->name)) {
        if (out_profile != NULL) {
          *out_profile = profile;
        }
        *out_provider = provider;
        return true;
      }
    }
  }
  return false;
}

bool loom_llvmir_target_registry_llc_requirement_provider(
    const loom_llvmir_target_registry_t* registry,
    iree_string_view_t requirement,
    const loom_llvmir_target_profile_provider_t** out_provider) {
  if (out_provider != NULL) {
    *out_provider = NULL;
  }
  requirement = iree_string_view_trim(requirement);
  if (!iree_string_view_consume_prefix(&requirement, IREE_SV("llc-"))) {
    return false;
  }

  for (iree_host_size_t i = 0; i < registry->profile_registry.provider_count;
       ++i) {
    const loom_llvmir_target_profile_provider_t* provider =
        registry->profile_registry.providers[i];
    if (iree_string_view_is_empty(provider->llc_target_name)) {
      continue;
    }
    if (iree_string_view_equal(requirement, provider->name)) {
      if (out_provider != NULL) {
        *out_provider = provider;
      }
      return true;
    }
  }
  return false;
}
