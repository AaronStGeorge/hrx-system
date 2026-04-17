// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/target_presets.h"

iree_status_t loom_llvmir_target_profile_registry_lookup(
    const loom_llvmir_target_profile_registry_t* registry,
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile) {
  if (out_profile == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile output is required");
  }
  *out_profile = NULL;
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile registry is required");
  }

  profile_name = iree_string_view_trim(profile_name);
  if (iree_string_view_is_empty(profile_name)) {
    if (registry->default_profile == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM target profile registry has no default");
    }
    *out_profile = registry->default_profile;
    return iree_ok_status();
  }

  if (registry->provider_count > 0 && registry->providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile registry has no providers");
  }

  for (iree_host_size_t provider_ordinal = 0;
       provider_ordinal < registry->provider_count; ++provider_ordinal) {
    const loom_llvmir_target_profile_provider_t* provider =
        registry->providers[provider_ordinal];
    if (provider == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM target profile provider is null");
    }
    if (provider->profile_count > 0 && provider->profiles == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LLVM target profile provider has no rows");
    }
    for (iree_host_size_t profile_ordinal = 0;
         profile_ordinal < provider->profile_count; ++profile_ordinal) {
      const loom_llvmir_target_profile_t* profile =
          provider->profiles[profile_ordinal];
      if (profile == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "LLVM target profile provider row is null");
      }
      if (iree_string_view_equal(profile_name, profile->name)) {
        *out_profile = profile;
        return iree_ok_status();
      }
    }
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown LLVMIR target profile '%.*s'",
                          (int)profile_name.size, profile_name.data);
}
