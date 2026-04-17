// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/llvmir_targets.h"

#include "loom/target/emit/llvmir/amdgpu/target_env.h"
#include "loom/target/emit/llvmir/x86/lower.h"
#include "loom/target/emit/llvmir/x86/target_env.h"

void loom_check_llvmir_target_profile_registry_initialize(
    loom_check_llvmir_target_profile_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(out_registry);
  out_registry->providers[0] = loom_llvmir_x86_target_profile_provider();
  out_registry->providers[1] = loom_llvmir_amdgpu_target_profile_provider();
  out_registry->registry = (loom_llvmir_target_profile_registry_t){
      .default_profile = loom_llvmir_target_profile_x86_64_object(),
      .providers = out_registry->providers,
      .provider_count = IREE_ARRAYSIZE(out_registry->providers),
  };
}

void loom_check_llvmir_lowering_providers_initialize(
    loom_check_llvmir_lowering_providers_t* out_providers) {
  IREE_ASSERT_ARGUMENT(out_providers);
  out_providers->providers[0] = loom_llvmir_x86_lowering_provider();
}

iree_status_t loom_check_llvmir_target_profile_lookup(
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile) {
  loom_check_llvmir_target_profile_registry_t registry;
  loom_check_llvmir_target_profile_registry_initialize(&registry);
  return loom_llvmir_target_profile_registry_lookup(&registry.registry,
                                                    profile_name, out_profile);
}

iree_status_t loom_check_llvmir_target_profile_provider_lookup(
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile,
    const loom_llvmir_target_profile_provider_t** out_provider) {
  if (out_provider == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVMIR target profile provider output is required");
  }
  if (out_profile != NULL) {
    *out_profile = NULL;
  }
  *out_provider = NULL;

  loom_check_llvmir_target_profile_registry_t registry;
  loom_check_llvmir_target_profile_registry_initialize(&registry);
  const loom_llvmir_target_profile_t* profile = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_registry_lookup(
      &registry.registry, profile_name, &profile));

  for (iree_host_size_t provider_ordinal = 0;
       provider_ordinal < registry.registry.provider_count;
       ++provider_ordinal) {
    const loom_llvmir_target_profile_provider_t* provider =
        registry.registry.providers[provider_ordinal];
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
        return iree_ok_status();
      }
    }
  }

  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "LLVMIR target profile '%.*s' is not owned by a registered provider",
      (int)profile->name.size, profile->name.data);
}

bool loom_check_llvmir_llc_requirement_provider(
    iree_string_view_t requirement,
    const loom_llvmir_target_profile_provider_t** out_provider) {
  if (out_provider != NULL) {
    *out_provider = NULL;
  }
  requirement = iree_string_view_trim(requirement);
  if (!iree_string_view_consume_prefix(&requirement, IREE_SV("llc-"))) {
    return false;
  }

  loom_check_llvmir_target_profile_registry_t registry;
  loom_check_llvmir_target_profile_registry_initialize(&registry);
  for (iree_host_size_t i = 0; i < registry.registry.provider_count; ++i) {
    const loom_llvmir_target_profile_provider_t* provider =
        registry.registry.providers[i];
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
