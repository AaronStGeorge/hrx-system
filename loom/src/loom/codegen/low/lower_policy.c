// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower.h"

static iree_status_t loom_low_lower_policy_registry_verify_tables(
    const loom_low_lower_policy_registry_t* registry) {
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low lowering policy registry is required");
  }
  if (registry->entry_count != 0 && registry->entries == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low lowering policy registry entries are required");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_policy_registry_verify_entry(
    const loom_low_lower_policy_registry_entry_t* entry) {
  if (iree_string_view_is_empty(
          iree_string_view_trim(entry->contract_set_key))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low lowering policy contract key is "
                            "required");
  }
  return loom_low_lower_policy_verify(entry->policy);
}

iree_status_t loom_low_lower_policy_verify(
    const loom_low_lower_policy_t* policy) {
  if (policy == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low lowering policy is required");
  }
  if (iree_string_view_is_empty(iree_string_view_trim(policy->name))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low lowering policy name is required");
  }
  if (policy->rule_sets.count != 0 && policy->rule_sets.values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low lowering policy rule set list is required");
  }
  for (iree_host_size_t i = 0; i < policy->rule_sets.count; ++i) {
    if (policy->rule_sets.values[i] == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target-low lowering policy rule set entries are required");
    }
  }
  const bool has_rule_sets =
      !loom_low_lower_rule_set_list_is_empty(policy->rule_sets);
  const bool has_select_op = policy->select_op.fn != NULL;
  const bool has_emit_op = policy->emit_op.fn != NULL;
  if (has_select_op != has_emit_op) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low lowering policy must provide both "
                            "select_op and emit_op callbacks or neither");
  }
  const bool has_callback_lowering = has_select_op && has_emit_op;
  if (policy->map_type.fn == NULL ||
      (!has_rule_sets && !has_callback_lowering)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "complete target-low lowering policy is required");
  }
  return iree_ok_status();
}

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

iree_status_t loom_low_lower_policy_registry_verify(
    const loom_low_lower_policy_registry_t* registry) {
  IREE_RETURN_IF_ERROR(loom_low_lower_policy_registry_verify_tables(registry));
  for (iree_host_size_t i = 0; i < registry->entry_count; ++i) {
    const loom_low_lower_policy_registry_entry_t* entry = &registry->entries[i];
    IREE_RETURN_IF_ERROR(loom_low_lower_policy_registry_verify_entry(entry));
    for (iree_host_size_t j = i + 1; j < registry->entry_count; ++j) {
      const loom_low_lower_policy_registry_entry_t* other_entry =
          &registry->entries[j];
      IREE_RETURN_IF_ERROR(
          loom_low_lower_policy_registry_verify_entry(other_entry));
      if (iree_string_view_equal(entry->contract_set_key,
                                 other_entry->contract_set_key)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "target-low lowering policy registry has duplicate contract key "
            "'%.*s'",
            (int)entry->contract_set_key.size, entry->contract_set_key.data);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_policy_registry_lookup(
    const loom_low_lower_policy_registry_t* registry,
    iree_string_view_t contract_set_key,
    const loom_low_lower_policy_t** out_policy) {
  IREE_ASSERT_ARGUMENT(out_policy);
  *out_policy = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_policy_registry_verify_tables(registry));
  if (iree_string_view_is_empty(iree_string_view_trim(contract_set_key))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low lowering policy lookup contract key is required");
  }

  const loom_low_lower_policy_t* match = NULL;
  for (iree_host_size_t i = 0; i < registry->entry_count; ++i) {
    const loom_low_lower_policy_registry_entry_t* entry = &registry->entries[i];
    IREE_RETURN_IF_ERROR(loom_low_lower_policy_registry_verify_entry(entry));
    if (!iree_string_view_equal(entry->contract_set_key, contract_set_key)) {
      continue;
    }
    if (match != NULL) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "target-low lowering policy registry has duplicate contract key "
          "'%.*s'",
          (int)contract_set_key.size, contract_set_key.data);
    }
    match = entry->policy;
  }
  if (match == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "target-low lowering policy registry has no contract key '%.*s'",
        (int)contract_set_key.size, contract_set_key.data);
  }
  *out_policy = match;
  return iree_ok_status();
}

iree_status_t loom_low_lower_policy_registry_lookup_for_bundle(
    const loom_low_lower_policy_registry_t* registry,
    const loom_target_bundle_t* bundle,
    const loom_low_lower_policy_t** out_policy) {
  if (bundle == NULL) {
    if (out_policy != NULL) {
      *out_policy = NULL;
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target bundle is required");
  }
  if (bundle->config == NULL) {
    if (out_policy != NULL) {
      *out_policy = NULL;
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target bundle config is required");
  }
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
