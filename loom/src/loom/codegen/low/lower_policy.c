// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/lower_rules.h"

void loom_low_lower_policy_registry_initialize_from_entries(
    loom_low_lower_policy_registry_t* out_registry,
    const loom_low_lower_policy_registry_entry_t* entries,
    iree_host_size_t entry_count) {
  IREE_ASSERT_ARGUMENT(out_registry);
  *out_registry = (loom_low_lower_policy_registry_t){
      .entries = entries,
      .entry_count = entry_count,
  };
}

iree_status_t loom_low_lower_policy_registry_lookup(
    const loom_low_lower_policy_registry_t* registry,
    iree_string_view_t contract_set_key,
    const loom_low_lower_policy_t** out_policy) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(out_policy);
  *out_policy = NULL;
  contract_set_key = iree_string_view_trim(contract_set_key);
  if (iree_string_view_is_empty(contract_set_key)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low lowering policy lookup contract key is required");
  }

  for (iree_host_size_t i = 0; i < registry->entry_count; ++i) {
    const loom_low_lower_policy_registry_entry_t* entry = &registry->entries[i];
    if (!iree_string_view_equal(entry->contract_set_key, contract_set_key)) {
      continue;
    }
    *out_policy = entry->policy;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "target-low lowering policy registry has no contract key '%.*s'",
      (int)contract_set_key.size, contract_set_key.data);
}

iree_status_t loom_low_lower_policy_registry_lookup_for_bundle(
    const loom_low_lower_policy_registry_t* registry,
    const loom_target_bundle_t* bundle,
    const loom_low_lower_policy_t** out_policy) {
  IREE_ASSERT_ARGUMENT(bundle);
  return loom_low_lower_policy_registry_lookup(
      registry, bundle->config->contract_set_key, out_policy);
}

bool loom_low_lower_policy_registry_has_bundle(
    const loom_low_lower_policy_registry_t* registry,
    const loom_target_bundle_t* bundle) {
  if (!registry || !bundle || !bundle->config) {
    return false;
  }
  iree_string_view_t contract_set_key = bundle->config->contract_set_key;
  if (iree_string_view_is_empty(iree_string_view_trim(contract_set_key))) {
    return false;
  }
  for (iree_host_size_t i = 0; i < registry->entry_count; ++i) {
    if (iree_string_view_equal(registry->entries[i].contract_set_key,
                               contract_set_key)) {
      return true;
    }
  }
  return false;
}
