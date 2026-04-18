// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Linked LLVMIR target registry for selected Loom target packages.
//
// This package is the build-level join point for LLVMIR target profiles,
// generic target bundles, legality providers, and lowering providers that a
// compiler binary elects to link. Generic LLVMIR lowering stays target-family
// agnostic; callers that want the bundled registry depend on this library
// instead of recreating provider arrays locally.

#ifndef LOOM_TARGET_LLVMIR_TARGET_REGISTRY_H_
#define LOOM_TARGET_LLVMIR_TARGET_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/target/emit/llvmir/legality.h"
#include "loom/target/emit/llvmir/lower.h"
#include "loom/target/emit/llvmir/target_presets.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_LLVMIR_TARGET_REGISTRY_MAX_PROFILE_PROVIDERS 2
#define LOOM_LLVMIR_TARGET_REGISTRY_MAX_BUNDLES 4
#define LOOM_LLVMIR_TARGET_REGISTRY_MAX_LEGALITY_PROVIDERS 2
#define LOOM_LLVMIR_TARGET_REGISTRY_MAX_LOWERING_PROVIDERS 2

typedef struct loom_llvmir_target_registry_t {
  // Profile-provider storage backing profile_registry.providers.
  const loom_llvmir_target_profile_provider_t*
      profile_providers[LOOM_LLVMIR_TARGET_REGISTRY_MAX_PROFILE_PROVIDERS];
  // Generic target bundles linked into this registry package.
  const loom_target_bundle_t* bundles[LOOM_LLVMIR_TARGET_REGISTRY_MAX_BUNDLES];
  // Profile-registry view over profile_providers.
  loom_llvmir_target_profile_registry_t profile_registry;
  // Number of valid bundle pointers in |bundles|.
  iree_host_size_t bundle_count;
} loom_llvmir_target_registry_t;

typedef struct loom_llvmir_target_legality_provider_list_t {
  // Target-specific legality providers selected for one target bundle.
  const loom_llvmir_target_legality_provider_t*
      providers[LOOM_LLVMIR_TARGET_REGISTRY_MAX_LEGALITY_PROVIDERS];
  // Number of provider pointers in |providers|.
  iree_host_size_t provider_count;
} loom_llvmir_target_legality_provider_list_t;

typedef struct loom_llvmir_lowering_provider_list_t {
  // Target-specific lowering providers selected for one target profile.
  const loom_llvmir_lowering_provider_t*
      providers[LOOM_LLVMIR_TARGET_REGISTRY_MAX_LOWERING_PROVIDERS];
  // Number of provider pointers in |providers|.
  iree_host_size_t provider_count;
} loom_llvmir_lowering_provider_list_t;

// Initializes the LLVMIR target registry selected by the current target package
// build.
void loom_llvmir_target_registry_initialize(
    loom_llvmir_target_registry_t* out_registry);

// Looks up a generic target bundle by profile/bundle name. Empty names resolve
// to the registry default bundle.
iree_status_t loom_llvmir_target_registry_lookup_bundle(
    const loom_llvmir_target_registry_t* registry, iree_string_view_t name,
    const loom_target_bundle_t** out_bundle);

// Selects target-specific legality providers required by |bundle|.
iree_status_t loom_llvmir_target_registry_select_legality_providers(
    const loom_llvmir_target_registry_t* registry,
    const loom_target_bundle_t* bundle,
    loom_llvmir_target_legality_provider_list_t* out_providers);

// Selects target-specific lowering providers required by |profile|.
iree_status_t loom_llvmir_target_registry_select_lowering_providers(
    const loom_llvmir_target_registry_t* registry,
    const loom_llvmir_target_profile_t* profile,
    loom_llvmir_lowering_provider_list_t* out_providers);

// Looks up a target profile in |registry|.
iree_status_t loom_llvmir_target_registry_lookup_profile(
    const loom_llvmir_target_registry_t* registry,
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile);

// Looks up the provider that owns |profile_name| in |registry|. The profile
// output is optional.
iree_status_t loom_llvmir_target_registry_lookup_profile_provider(
    const loom_llvmir_target_registry_t* registry,
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile,
    const loom_llvmir_target_profile_provider_t** out_provider);

// Returns true when |requirement| names an llc backend requirement supplied by
// one of the registered LLVMIR target providers.
bool loom_llvmir_target_registry_llc_requirement_provider(
    const loom_llvmir_target_registry_t* registry,
    iree_string_view_t requirement,
    const loom_llvmir_target_profile_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_LLVMIR_TARGET_REGISTRY_H_
