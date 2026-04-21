// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target dialect symbol facts.
//
// These facts resolve compact target.profile records into dense target-neutral
// structs. Backend-specific facts remain in backend packages; this layer only
// owns the shared profile shape and the provider-injected preset registry.

#ifndef LOOM_OPS_TARGET_FACTS_H_
#define LOOM_OPS_TARGET_FACTS_H_

#include "iree/base/api.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/ir/ir.h"
#include "loom/target/preset_registry.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_profile_symbol_facts_t
    loom_target_profile_symbol_facts_t;

// Resolved target.profile payload.
typedef struct loom_target_profile_symbol_facts_t {
  // Common symbol-fact header.
  loom_symbol_facts_base_t base;

  // Module-local symbol reference for the profile definition.
  loom_symbol_ref_t symbol;

  // Borrowed profile symbol name from the module string table.
  iree_string_view_t name;

  // Borrowed preset key from the target.profile op.
  iree_string_view_t preset_key;

  // Borrowed provider preset bundle used as the base before overrides.
  const loom_target_bundle_t* preset_bundle;

  // Effective target snapshot after sparse overrides are applied.
  loom_target_snapshot_t snapshot;

  // Effective export and ABI plan after sparse overrides are applied.
  loom_target_export_plan_t export_plan;

  // Effective legalization/configuration record after sparse overrides apply.
  loom_target_config_t config;

  // Effective bundle view pointing at the copied snapshot/export/config fields.
  loom_target_bundle_t bundle;
} loom_target_profile_symbol_facts_t;

// Symbol fact domain used by generated target.profile descriptors.
extern const loom_symbol_fact_domain_t loom_target_profile_symbol_fact_domain;

// Resource key for provider-injected loom_target_preset_registry_t payloads.
extern const uint8_t loom_target_profile_preset_registry_resource_key;

// Returns a symbol fact resource binding a target preset registry for
// target.profile fact computation.
static inline loom_symbol_fact_resource_t
loom_target_profile_preset_registry_resource(
    const loom_target_preset_registry_t* registry) {
  return (loom_symbol_fact_resource_t){
      .key = &loom_target_profile_preset_registry_resource_key,
      .value = registry,
  };
}

// Looks up the provider-injected preset registry.
iree_status_t loom_target_profile_symbol_fact_context_lookup_preset_registry(
    loom_symbol_fact_context_t* context,
    const loom_target_preset_registry_t** out_registry);

// Casts generic symbol facts to target.profile facts when domains match.
const loom_target_profile_symbol_facts_t* loom_target_profile_symbol_facts_cast(
    const loom_symbol_facts_base_t* facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_TARGET_FACTS_H_
