// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target provider composition shared by tools and compile drivers.
//
// A target provider owns the target dialects, descriptor registries, lowering
// policies, and diagnostic providers linked into a binary. Tool-specific layers
// may add execution, checking, or artifact-emission providers around this core
// target contribution, but those layers should not duplicate target registry
// aggregation.

#ifndef LOOM_TARGET_PROVIDER_H_
#define LOOM_TARGET_PROVIDER_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower.h"
#include "loom/ir/context.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/low_legality.h"
#include "loom/target/low_packet_diagnostics.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers target-owned dialects and encoding families.
typedef iree_status_t (*loom_target_provider_context_registration_fn_t)(
    loom_context_t* context);

// Initializes a target-low descriptor registry package.
typedef void (*loom_target_low_descriptor_registry_initializer_t)(
    loom_target_low_descriptor_registry_t* out_registry);

// Initializes a source-to-target-low lowering policy registry package.
typedef void (*loom_target_low_lower_policy_registry_initializer_t)(
    loom_low_lower_policy_registry_t* out_registry);

// Target-owned compiler capability contribution linked into a tool or driver.
typedef struct loom_target_provider_t {
  // Optional function that registers target-owned dialects.
  loom_target_provider_context_registration_fn_t register_context;
  // Optional function that initializes target-low descriptor-set providers.
  loom_target_low_descriptor_registry_initializer_t
      initialize_low_descriptor_registry;
  // Optional function that initializes source-to-low lowering policies.
  loom_target_low_lower_policy_registry_initializer_t
      initialize_low_lower_policy_registry;
  // Optional source legality providers contributed by this target.
  loom_target_low_legality_provider_list_t low_legality_provider_list;
  // Optional low-packet diagnostic providers contributed by this target.
  loom_target_low_packet_diagnostic_provider_list_t
      low_packet_diagnostic_provider_list;
} loom_target_provider_t;

// Static target provider table linked into a binary or embedding.
typedef struct loom_target_provider_set_t {
  // Target provider contribution table.
  const loom_target_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_target_provider_set_t;

enum {
  LOOM_TARGET_PROVIDER_DESCRIPTOR_SET_PROVIDER_CAPACITY = 256,
  LOOM_TARGET_PROVIDER_LOW_LOWER_POLICY_CAPACITY = 128,
  LOOM_TARGET_PROVIDER_LOW_LEGALITY_PROVIDER_CAPACITY = 64,
  LOOM_TARGET_PROVIDER_LOW_PACKET_DIAGNOSTIC_PROVIDER_CAPACITY = 64,
};

// Composed target environment derived from a target provider set.
typedef struct loom_target_environment_t {
  // Borrowed provider table selected by the linked binary or embedding.
  const loom_target_provider_set_t* provider_set;
  // Descriptor-set provider table assembled once.
  loom_low_descriptor_set_provider_t descriptor_set_providers
      [LOOM_TARGET_PROVIDER_DESCRIPTOR_SET_PROVIDER_CAPACITY];
  // Number of entries in |descriptor_set_providers|.
  iree_host_size_t descriptor_set_provider_count;
  // Source-to-low policy table assembled once.
  loom_low_lower_policy_registry_entry_t
      low_lower_policy_entries[LOOM_TARGET_PROVIDER_LOW_LOWER_POLICY_CAPACITY];
  // Number of entries in |low_lower_policy_entries|.
  iree_host_size_t low_lower_policy_entry_count;
  // Target-low source legality provider table assembled once.
  const loom_target_low_legality_provider_t* low_legality_providers
      [LOOM_TARGET_PROVIDER_LOW_LEGALITY_PROVIDER_CAPACITY];
  // Number of entries in |low_legality_providers|.
  iree_host_size_t low_legality_provider_count;
  // Target-low packet diagnostic provider table assembled once.
  const loom_target_low_packet_diagnostic_provider_t*
      low_packet_diagnostic_providers
          [LOOM_TARGET_PROVIDER_LOW_PACKET_DIAGNOSTIC_PROVIDER_CAPACITY];
  // Number of entries in |low_packet_diagnostic_providers|.
  iree_host_size_t low_packet_diagnostic_provider_count;
} loom_target_environment_t;

// Creates a borrowed provider set view.
static inline loom_target_provider_set_t loom_target_provider_set_make(
    const loom_target_provider_t* const* providers,
    iree_host_size_t provider_count) {
  return (loom_target_provider_set_t){
      .providers = providers,
      .provider_count = provider_count,
  };
}

// Initializes |out_environment| from |provider_set|. The environment borrows
// |provider_set| until deinitialized.
iree_status_t loom_target_environment_initialize(
    const loom_target_provider_set_t* provider_set,
    loom_target_environment_t* out_environment);

// Resets |environment| to an empty state. No provider-owned storage is freed.
void loom_target_environment_deinitialize(
    loom_target_environment_t* environment);

// Registers every target dialect contributed by |environment|.
iree_status_t loom_target_environment_register_context(
    const loom_target_environment_t* environment, loom_context_t* context);

// Initializes a composed target-low descriptor registry package.
iree_status_t loom_target_environment_initialize_low_descriptor_registry(
    const loom_target_environment_t* environment,
    loom_target_low_descriptor_registry_t* out_registry);

// Initializes a composed source-to-target-low lowering policy registry package.
iree_status_t loom_target_environment_initialize_low_lower_policy_registry(
    const loom_target_environment_t* environment,
    loom_low_lower_policy_registry_t* out_registry);

// Returns target-low source legality providers linked into |environment|.
loom_target_low_legality_provider_list_t
loom_target_environment_low_legality_provider_list(
    const loom_target_environment_t* environment);

// Returns target-low packet diagnostic providers linked into |environment|.
loom_target_low_packet_diagnostic_provider_list_t
loom_target_environment_low_packet_diagnostic_provider_list(
    const loom_target_environment_t* environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PROVIDER_H_
