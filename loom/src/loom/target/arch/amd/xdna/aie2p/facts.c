// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/facts.h"

#include "loom/target/arch/amd/xdna/aie2p/records/target_records.h"
#include "loom/target/facts_builder.h"

static bool loom_aie2p_target_facts_satisfy_device_requirement(
    const loom_aie2p_target_facts_t* effective,
    const loom_aie2p_target_facts_t* requirement) {
  return requirement->device_profile == NULL ||
         effective->device_profile == requirement->device_profile;
}

static bool loom_aie2p_target_facts_satisfy_identity_requirement(
    const loom_target_facts_t* base_effective,
    const loom_target_facts_t* base_requirement) {
  const loom_aie2p_target_facts_t* effective =
      (const loom_aie2p_target_facts_t*)base_effective;
  const loom_aie2p_target_facts_t* requirement =
      (const loom_aie2p_target_facts_t*)base_requirement;
  return effective->base.selector == requirement->base.selector &&
         loom_aie2p_target_facts_satisfy_device_requirement(effective,
                                                            requirement);
}

static bool loom_aie2p_target_facts_satisfy_specialization_requirement(
    const loom_target_facts_t* base_effective,
    const loom_target_facts_t* base_requirement) {
  const loom_aie2p_target_facts_t* effective =
      (const loom_aie2p_target_facts_t*)base_effective;
  const loom_aie2p_target_facts_t* requirement =
      (const loom_aie2p_target_facts_t*)base_requirement;
  if (effective->array_contract_unselected &&
      base_requirement->selector == LOOM_AIE2P_TARGET_KIND_CONFIGURATION) {
    // These input forms share the physical snapshot and whole-array ABI.
    // Selecting the configuration form does not alter deployment constraints.
    return loom_target_snapshot_satisfies_specialization_requirement(
               &base_effective->storage.snapshot,
               &base_requirement->storage.snapshot) &&
           base_requirement->storage.config.contract_set.stable_id ==
               loom_aie2p_configuration_target_bundle.config->contract_set
                   .stable_id &&
           iree_all_bits_set(
               base_effective->storage.config.contract_feature_bits,
               base_requirement->storage.config.contract_feature_bits) &&
           loom_aie2p_target_facts_satisfy_device_requirement(effective,
                                                              requirement);
  }
  return loom_target_facts_structural_satisfy_specialization_requirement(
             base_effective, base_requirement) &&
         loom_aie2p_target_facts_satisfy_device_requirement(effective,
                                                            requirement);
}

static void loom_aie2p_target_facts_apply_specialization_requirement(
    const loom_target_facts_t* requirement, loom_target_facts_t* effective) {
  loom_aie2p_target_facts_t* facts = (loom_aie2p_target_facts_t*)effective;
  if (!facts->array_contract_unselected) {
    return;
  }
  if (requirement->selector == LOOM_AIE2P_TARGET_KIND_CONFIGURATION) {
    effective->selector = requirement->selector;
    effective->storage.config = requirement->storage.config;
    effective->storage.bundle.name =
        loom_aie2p_configuration_target_bundle.name;
  }
  facts->array_contract_unselected = false;
}

static bool loom_aie2p_target_facts_family_state_equivalent(
    const loom_target_facts_t* lhs, const loom_target_facts_t* rhs) {
  return ((const loom_aie2p_target_facts_t*)lhs)->array_contract_unselected ==
         ((const loom_aie2p_target_facts_t*)rhs)->array_contract_unselected;
}

static iree_status_t loom_aie2p_target_facts_select_execution(
    const loom_target_facts_t* source, iree_arena_allocator_t* arena,
    const loom_target_facts_t** out_facts) {
  *out_facts = source;
  if (!((const loom_aie2p_target_facts_t*)source)->array_contract_unselected) {
    return iree_ok_status();
  }
  loom_target_facts_t* selected = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_facts_builder_clone(source, arena, &selected));
  ((loom_aie2p_target_facts_t*)selected)->array_contract_unselected = false;
  *out_facts = selected;
  return iree_ok_status();
}

static iree_string_view_t loom_aie2p_target_facts_identity_name(
    const loom_target_facts_t* base_facts) {
  const loom_aie2p_target_facts_t* facts =
      (const loom_aie2p_target_facts_t*)base_facts;
  return facts->device_profile != NULL
             ? iree_make_cstring_view(facts->device_profile->key)
             : facts->base.storage.bundle.name;
}

static iree_status_t loom_aie2p_target_facts_project_worker(
    const loom_target_facts_t* facts, iree_arena_allocator_t* arena,
    const loom_target_facts_t** out_facts) {
  *out_facts = facts;
  if (facts->selector == LOOM_AIE2P_TARGET_KIND_CORE) {
    return iree_ok_status();
  }
  loom_target_facts_t* worker = NULL;
  IREE_RETURN_IF_ERROR(loom_target_facts_builder_clone(facts, arena, &worker));
  loom_target_facts_builder_set_worker_contract(
      LOOM_AIE2P_TARGET_KIND_CORE, &loom_aie2p_core_target_bundle, worker);
  ((loom_aie2p_target_facts_t*)worker)->array_contract_unselected = false;
  *out_facts = worker;
  return iree_ok_status();
}

const loom_target_fact_type_t loom_aie2p_target_fact_type = {
    .name = IREE_SVL("amd.xdna.aie2p"),
    .storage_size = sizeof(loom_aie2p_target_facts_t),
    .satisfies_identity_requirement =
        loom_aie2p_target_facts_satisfy_identity_requirement,
    .satisfies_specialization_requirement =
        loom_aie2p_target_facts_satisfy_specialization_requirement,
    .family_state_equivalent = loom_aie2p_target_facts_family_state_equivalent,
    .apply_specialization_requirement =
        loom_aie2p_target_facts_apply_specialization_requirement,
    .identity_name = loom_aie2p_target_facts_identity_name,
    .select_execution = loom_aie2p_target_facts_select_execution,
    .project_worker = loom_aie2p_target_facts_project_worker,
};
