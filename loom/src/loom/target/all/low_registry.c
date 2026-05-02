// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/all/low_registry.h"

#include "iree/base/threading/call_once.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/lower.h"
#include "loom/target/arch/wasm/low_registry.h"
#include "loom/target/arch/x86/low_registry.h"
#include "loom/target/arch/x86/lower.h"
#include "loom/target/emit/ireevm/low_registry.h"
#include "loom/target/emit/ireevm/lower.h"
#include "loom/target/emit/wasm/lower.h"

enum {
  LOOM_ALL_LOW_DESCRIPTOR_SET_PROVIDER_CAPACITY = 8,
  LOOM_ALL_LOW_LOWER_POLICY_ENTRY_CAPACITY = 16,
};

typedef struct loom_all_low_registry_tables_t {
  loom_low_descriptor_set_provider_t
      descriptor_set_providers[LOOM_ALL_LOW_DESCRIPTOR_SET_PROVIDER_CAPACITY];
  iree_host_size_t descriptor_set_provider_count;
  loom_low_lower_policy_registry_entry_t
      low_lower_policy_entries[LOOM_ALL_LOW_LOWER_POLICY_ENTRY_CAPACITY];
  iree_host_size_t low_lower_policy_entry_count;
} loom_all_low_registry_tables_t;

static loom_all_low_registry_tables_t kLowRegistryTables;
static iree_once_flag kLowRegistryTablesOnce = IREE_ONCE_FLAG_INIT;

static const loom_target_low_legality_provider_t* const
    kLowLegalityProviders[] = {
        &loom_amdgpu_low_legality_provider_storage,
};

static void loom_all_low_registry_append(
    const loom_target_low_descriptor_registry_t* source) {
  IREE_ASSERT(kLowRegistryTables.descriptor_set_provider_count +
                  source->descriptor_set_provider_count <=
              IREE_ARRAYSIZE(kLowRegistryTables.descriptor_set_providers));
  for (iree_host_size_t i = 0; i < source->descriptor_set_provider_count; ++i) {
    kLowRegistryTables.descriptor_set_providers
        [kLowRegistryTables.descriptor_set_provider_count++] =
        source->descriptor_set_providers[i];
  }
}

static void loom_all_low_registry_append_lower_policies(
    const loom_low_lower_policy_registry_t* source) {
  IREE_ASSERT(kLowRegistryTables.low_lower_policy_entry_count +
                  source->entry_count <=
              IREE_ARRAYSIZE(kLowRegistryTables.low_lower_policy_entries));
  for (iree_host_size_t i = 0; i < source->entry_count; ++i) {
    kLowRegistryTables.low_lower_policy_entries
        [kLowRegistryTables.low_lower_policy_entry_count++] =
        source->entries[i];
  }
}

static void loom_all_low_registry_initialize_tables(void) {
  loom_target_low_descriptor_registry_t descriptor_registry = {0};
  loom_ireevm_low_descriptor_registry_initialize(&descriptor_registry);
  loom_all_low_registry_append(&descriptor_registry);
  loom_wasm_low_descriptor_registry_initialize(&descriptor_registry);
  loom_all_low_registry_append(&descriptor_registry);
  loom_x86_low_descriptor_registry_initialize(&descriptor_registry);
  loom_all_low_registry_append(&descriptor_registry);
  loom_amdgpu_low_descriptor_registry_initialize(&descriptor_registry);
  loom_all_low_registry_append(&descriptor_registry);

  loom_low_lower_policy_registry_t policy_registry = {0};
  loom_ireevm_low_lower_policy_registry_initialize(&policy_registry);
  loom_all_low_registry_append_lower_policies(&policy_registry);
  loom_wasm_low_lower_policy_registry_initialize(&policy_registry);
  loom_all_low_registry_append_lower_policies(&policy_registry);
  loom_x86_low_lower_policy_registry_initialize(&policy_registry);
  loom_all_low_registry_append_lower_policies(&policy_registry);
  loom_amdgpu_low_lower_policy_registry_initialize(&policy_registry);
  loom_all_low_registry_append_lower_policies(&policy_registry);
}

void loom_all_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  iree_call_once(&kLowRegistryTablesOnce,
                 loom_all_low_registry_initialize_tables);
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kLowRegistryTables.descriptor_set_providers,
      kLowRegistryTables.descriptor_set_provider_count);
}

void loom_all_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  iree_call_once(&kLowRegistryTablesOnce,
                 loom_all_low_registry_initialize_tables);
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kLowRegistryTables.low_lower_policy_entries,
      kLowRegistryTables.low_lower_policy_entry_count);
}

loom_target_low_legality_provider_list_t loom_all_low_legality_provider_list(
    void) {
  return loom_target_low_legality_provider_list_make(
      kLowLegalityProviders, IREE_ARRAYSIZE(kLowLegalityProviders));
}
