// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Linked low descriptor registry package support.
//
// These target-neutral support functions operate on caller-provided
// descriptor-set and target-bundle tables. Concrete registry packages live with
// the target families that own the linked descriptor providers and target
// bundles.

#ifndef LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_H_
#define LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/requirements.h"
#include "loom/target/preset_registry.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_low_descriptor_registry_t {
  // Borrowed descriptor-set provider table linked into this target package.
  const loom_low_descriptor_set_provider_t* descriptor_set_providers;
  // Number of provider functions in |descriptor_set_providers|.
  iree_host_size_t descriptor_set_provider_count;
  // Borrowed target-bundle preset pointers linked into this registry package.
  const loom_target_bundle_t* const* target_bundles;
  // Number of valid target-bundle preset pointers in |target_bundles|.
  iree_host_size_t target_bundle_count;
  // Registry view over the package descriptor-set providers.
  loom_low_descriptor_registry_t registry;
} loom_target_low_descriptor_registry_t;

// Initializes |out_registry| from linked package tables. The provider and
// bundle tables are borrowed and must outlive |out_registry|.
void loom_target_low_descriptor_registry_initialize_from_tables(
    loom_target_low_descriptor_registry_t* out_registry,
    const loom_low_descriptor_set_provider_t* descriptor_set_providers,
    iree_host_size_t descriptor_set_provider_count,
    const loom_target_bundle_t* const* target_bundles,
    iree_host_size_t target_bundle_count);

// Returns a generic preset-registry view over |registry|'s bundle table.
static inline loom_target_preset_registry_t
loom_target_low_descriptor_registry_presets(
    const loom_target_low_descriptor_registry_t* registry) {
  return (loom_target_preset_registry_t){
      .target_bundles = registry ? registry->target_bundles : NULL,
      .target_bundle_count = registry ? registry->target_bundle_count : 0,
  };
}

// Verifies that the linked target-low registry package is internally
// consistent. This checks the descriptor registry, bundle table, unique bundle
// keys, required target records, and that each bundle selects a linked
// descriptor set satisfying |requirements|.
iree_status_t loom_target_low_descriptor_registry_verify(
    const loom_target_low_descriptor_registry_t* registry,
    loom_low_descriptor_requirement_flags_t requirements);

// Appends a compact JSON manifest for the linked target-low registry package to
// |builder|. The manifest is a diagnostic and test format that lists linked
// descriptor-set summaries and target bundle selections without embedding full
// instruction descriptor rows.
iree_status_t loom_target_low_descriptor_registry_format_manifest_json(
    const loom_target_low_descriptor_registry_t* registry,
    loom_low_descriptor_requirement_flags_t requirements,
    iree_string_builder_t* builder);

// Looks up a target-low preset bundle by key in |registry|. Empty keys are not
// defaulted because low target selection must be explicit and reproducible.
iree_status_t loom_target_low_descriptor_registry_lookup_bundle(
    const loom_target_low_descriptor_registry_t* registry,
    iree_string_view_t key, const loom_target_bundle_t** out_bundle);

// Selects the descriptor set named by |bundle->config->contract_set_key| from
// |registry| and verifies any requested payload requirements. This is an exact
// key lookup; target triples and CPUs are descriptive facts, not fallback
// descriptor-selection rules.
iree_status_t loom_target_low_descriptor_set_select_for_bundle(
    const loom_low_descriptor_registry_t* registry,
    const loom_target_bundle_t* bundle,
    loom_low_descriptor_requirement_flags_t requirements,
    const loom_low_descriptor_set_t** out_descriptor_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_LOW_DESCRIPTOR_REGISTRY_H_
