// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLS_LOOM_CHECK_LLVMIR_TARGETS_H_
#define LOOM_TOOLS_LOOM_CHECK_LLVMIR_TARGETS_H_

#include "iree/base/api.h"
#include "loom/target/emit/llvmir/legality.h"
#include "loom/target/emit/llvmir/lower.h"
#include "loom/target/emit/llvmir/target_presets.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum {
  LOOM_CHECK_LLVMIR_TARGET_PROFILE_PROVIDER_COUNT = 2,
  LOOM_CHECK_LLVMIR_LEGALITY_PROVIDER_COUNT = 2,
  LOOM_CHECK_LLVMIR_LOWERING_PROVIDER_COUNT = 2,
};

typedef struct loom_check_llvmir_target_profile_registry_t {
  // Provider storage backing registry.providers.
  const loom_llvmir_target_profile_provider_t*
      providers[LOOM_CHECK_LLVMIR_TARGET_PROFILE_PROVIDER_COUNT];
  // Registry view passed to generic LLVMIR preset lookup.
  loom_llvmir_target_profile_registry_t registry;
} loom_check_llvmir_target_profile_registry_t;

typedef struct loom_check_llvmir_lowering_providers_t {
  // Target-specific lowering providers exposed to loom-check emit tests.
  const loom_llvmir_lowering_provider_t*
      providers[LOOM_CHECK_LLVMIR_LOWERING_PROVIDER_COUNT];
  // Number of providers selected for one target profile.
  iree_host_size_t provider_count;
} loom_check_llvmir_lowering_providers_t;

typedef struct loom_check_llvmir_legality_providers_t {
  // Target-specific legality providers exposed to loom-check emit tests.
  const loom_llvmir_target_legality_provider_t*
      providers[LOOM_CHECK_LLVMIR_LEGALITY_PROVIDER_COUNT];
  // Number of providers selected for one target bundle.
  iree_host_size_t provider_count;
} loom_check_llvmir_legality_providers_t;

// Initializes |out_registry| with the LLVMIR target providers that loom-check
// intentionally exposes to developer tests.
void loom_check_llvmir_target_profile_registry_initialize(
    loom_check_llvmir_target_profile_registry_t* out_registry);

// Looks up a generic target bundle by the same short profile names accepted by
// loom-check emit commands. Empty names resolve to the target-neutral default.
iree_status_t loom_check_llvmir_target_bundle_lookup(
    iree_string_view_t profile_name, const loom_target_bundle_t** out_bundle);

// Initializes |out_providers| with the LLVMIR target legality providers
// required by |bundle|. Bundles without target-specific legality contracts
// select no providers.
void loom_check_llvmir_legality_providers_initialize(
    const loom_target_bundle_t* bundle,
    loom_check_llvmir_legality_providers_t* out_providers);

// Initializes |out_providers| with the LLVMIR lowering providers required by
// |profile|. Profiles without target-specific lowering contracts select no
// providers.
void loom_check_llvmir_lowering_providers_initialize(
    const loom_llvmir_target_profile_t* profile,
    loom_check_llvmir_lowering_providers_t* out_providers);

// Looks up a profile in the loom-check LLVMIR target provider registry.
iree_status_t loom_check_llvmir_target_profile_lookup(
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile);

// Looks up the provider that owns |profile_name| in the loom-check registry.
// The profile output is optional.
iree_status_t loom_check_llvmir_target_profile_provider_lookup(
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile,
    const loom_llvmir_target_profile_provider_t** out_provider);

// Returns true when |requirement| names an llc backend requirement supplied by
// one of the registered LLVMIR target providers.
bool loom_check_llvmir_llc_requirement_provider(
    iree_string_view_t requirement,
    const loom_llvmir_target_profile_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TOOLS_LOOM_CHECK_LLVMIR_TARGETS_H_
