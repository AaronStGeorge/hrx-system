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
  IREE_ASSERT_ARGUMENT(out_registry);
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

static bool loom_llvmir_target_registry_profile_has_triple(
    const loom_llvmir_target_profile_t* profile,
    iree_string_view_t target_triple) {
  return profile != NULL && profile->target_env != NULL &&
         iree_string_view_equal(profile->target_env->target_triple,
                                target_triple);
}

static bool loom_llvmir_target_registry_bundle_has_triple(
    const loom_target_bundle_t* bundle, iree_string_view_t target_triple) {
  return bundle != NULL && bundle->snapshot != NULL &&
         iree_string_view_equal(bundle->snapshot->target_triple, target_triple);
}

static iree_status_t loom_llvmir_target_registry_has_profile_provider(
    const loom_llvmir_target_registry_t* registry, iree_string_view_t name,
    bool* out_has_provider) {
  *out_has_provider = false;
  if (registry->profile_registry.provider_count > 0 &&
      registry->profile_registry.providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVMIR target registry has no providers");
  }
  for (iree_host_size_t i = 0; i < registry->profile_registry.provider_count;
       ++i) {
    const loom_llvmir_target_profile_provider_t* provider =
        registry->profile_registry.providers[i];
    if (provider == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVMIR target profile provider is null");
    }
    if (iree_string_view_equal(provider->name, name)) {
      *out_has_provider = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t loom_llvmir_target_registry_lookup_bundle(
    const loom_llvmir_target_registry_t* registry, iree_string_view_t name,
    const loom_target_bundle_t** out_bundle) {
  if (out_bundle == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVMIR target bundle output is required");
  }
  *out_bundle = NULL;
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVMIR target registry is required");
  }
  if (registry->bundle_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVMIR target registry has no bundles");
  }

  name = iree_string_view_trim(name);
  if (iree_string_view_is_empty(name)) {
    *out_bundle = registry->bundles[0];
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < registry->bundle_count; ++i) {
    const loom_target_bundle_t* bundle = registry->bundles[i];
    if (bundle == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVMIR target registry bundle is null");
    }
    if (iree_string_view_equal(name, bundle->name)) {
      *out_bundle = bundle;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown LLVMIR target profile '%.*s'",
                          (int)name.size, name.data);
}

iree_status_t loom_llvmir_target_registry_select_legality_providers(
    const loom_llvmir_target_registry_t* registry,
    const loom_target_bundle_t* bundle,
    loom_llvmir_target_legality_provider_list_t* out_providers) {
  if (out_providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVMIR legality provider list is required");
  }
  *out_providers = (loom_llvmir_target_legality_provider_list_t){0};
  if (registry == NULL || bundle == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVMIR target registry and bundle are required");
  }

  bool has_x86_provider = false;
  IREE_RETURN_IF_ERROR(loom_llvmir_target_registry_has_profile_provider(
      registry, IREE_SV("x86"), &has_x86_provider));
  if (has_x86_provider && loom_llvmir_target_registry_bundle_has_triple(
                              bundle, IREE_SV("x86_64-unknown-linux-gnu"))) {
    out_providers->providers[out_providers->provider_count++] =
        loom_llvmir_x86_legality_provider();
  }
  bool has_amdgpu_provider = false;
  IREE_RETURN_IF_ERROR(loom_llvmir_target_registry_has_profile_provider(
      registry, IREE_SV("amdgpu"), &has_amdgpu_provider));
  if (has_amdgpu_provider && loom_llvmir_target_registry_bundle_has_triple(
                                 bundle, IREE_SV("amdgcn-amd-amdhsa"))) {
    out_providers->providers[out_providers->provider_count++] =
        loom_llvmir_amdgpu_legality_provider();
  }
  return iree_ok_status();
}

iree_status_t loom_llvmir_target_registry_select_lowering_providers(
    const loom_llvmir_target_registry_t* registry,
    const loom_llvmir_target_profile_t* profile,
    loom_llvmir_lowering_provider_list_t* out_providers) {
  if (out_providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVMIR lowering provider list is required");
  }
  *out_providers = (loom_llvmir_lowering_provider_list_t){0};
  if (registry == NULL || profile == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVMIR target registry and profile are required");
  }

  bool has_x86_provider = false;
  IREE_RETURN_IF_ERROR(loom_llvmir_target_registry_has_profile_provider(
      registry, IREE_SV("x86"), &has_x86_provider));
  if (has_x86_provider && loom_llvmir_target_registry_profile_has_triple(
                              profile, IREE_SV("x86_64-unknown-linux-gnu"))) {
    out_providers->providers[out_providers->provider_count++] =
        loom_llvmir_x86_lowering_provider();
  }
  bool has_amdgpu_provider = false;
  IREE_RETURN_IF_ERROR(loom_llvmir_target_registry_has_profile_provider(
      registry, IREE_SV("amdgpu"), &has_amdgpu_provider));
  if (has_amdgpu_provider && loom_llvmir_target_registry_profile_has_triple(
                                 profile, IREE_SV("amdgcn-amd-amdhsa"))) {
    out_providers->providers[out_providers->provider_count++] =
        loom_llvmir_amdgpu_lowering_provider();
  }
  return iree_ok_status();
}

iree_status_t loom_llvmir_target_registry_lookup_profile(
    const loom_llvmir_target_registry_t* registry,
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile) {
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVMIR target registry is required");
  }
  return loom_llvmir_target_profile_registry_lookup(&registry->profile_registry,
                                                    profile_name, out_profile);
}

iree_status_t loom_llvmir_target_registry_lookup_profile_provider(
    const loom_llvmir_target_registry_t* registry,
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
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVMIR target registry is required");
  }

  const loom_llvmir_target_profile_t* profile = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_target_registry_lookup_profile(
      registry, profile_name, &profile));

  for (iree_host_size_t provider_ordinal = 0;
       provider_ordinal < registry->profile_registry.provider_count;
       ++provider_ordinal) {
    const loom_llvmir_target_profile_provider_t* provider =
        registry->profile_registry.providers[provider_ordinal];
    if (provider == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVMIR target profile provider is null");
    }
    if (provider->profile_count > 0 && provider->profiles == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVMIR target profile provider has no rows");
    }
    for (iree_host_size_t profile_ordinal = 0;
         profile_ordinal < provider->profile_count; ++profile_ordinal) {
      const loom_llvmir_target_profile_t* provider_profile =
          provider->profiles[profile_ordinal];
      if (provider_profile == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVMIR target profile provider row is null");
      }
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

bool loom_llvmir_target_registry_llc_requirement_provider(
    const loom_llvmir_target_registry_t* registry,
    iree_string_view_t requirement,
    const loom_llvmir_target_profile_provider_t** out_provider) {
  if (out_provider != NULL) {
    *out_provider = NULL;
  }
  if (registry == NULL) {
    return false;
  }
  requirement = iree_string_view_trim(requirement);
  if (!iree_string_view_consume_prefix(&requirement, IREE_SV("llc-"))) {
    return false;
  }

  for (iree_host_size_t i = 0; i < registry->profile_registry.provider_count;
       ++i) {
    const loom_llvmir_target_profile_provider_t* provider =
        registry->profile_registry.providers[i];
    if (provider == NULL) {
      continue;
    }
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
