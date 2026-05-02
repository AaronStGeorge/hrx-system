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
};

typedef struct loom_all_low_registry_tables_t {
  loom_low_descriptor_set_provider_t
      descriptor_set_providers[LOOM_ALL_LOW_DESCRIPTOR_SET_PROVIDER_CAPACITY];
  iree_host_size_t descriptor_set_provider_count;
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

static void loom_all_low_registry_initialize_tables(void) {
  loom_target_low_descriptor_registry_t registry = {0};
  loom_ireevm_low_descriptor_registry_initialize(&registry);
  loom_all_low_registry_append(&registry);
  loom_wasm_low_descriptor_registry_initialize(&registry);
  loom_all_low_registry_append(&registry);
  loom_x86_low_descriptor_registry_initialize(&registry);
  loom_all_low_registry_append(&registry);
  loom_amdgpu_low_descriptor_registry_initialize(&registry);
  loom_all_low_registry_append(&registry);
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
  static loom_low_lower_policy_registry_entry_t kLowLowerPolicyEntries[] = {
      {.contract_set_key = IREE_SVL("iree.vm.core")},
      {.contract_set_key = IREE_SVL("wasm.core.simd128")},
      {.contract_set_key = IREE_SVL("x86.avx512.core")},
      {.contract_set_key = IREE_SVL("x86.packed_dot.core")},
      {.contract_set_key = IREE_SVL("amdgpu.gfx950.core")},
      {.contract_set_key = IREE_SVL("amdgpu.gfx11.core")},
      {.contract_set_key = IREE_SVL("amdgpu.gfx12.core")},
      {.contract_set_key = IREE_SVL("amdgpu.gfx1250.core")},
  };
  kLowLowerPolicyEntries[0].policy = loom_ireevm_low_lower_policy();
  kLowLowerPolicyEntries[1].policy = loom_wasm_low_lower_policy();
  kLowLowerPolicyEntries[2].policy = loom_x86_avx512_low_lower_policy();
  kLowLowerPolicyEntries[3].policy = loom_x86_packed_dot_low_lower_policy();
  kLowLowerPolicyEntries[4].policy = loom_amdgpu_low_lower_policy();
  kLowLowerPolicyEntries[5].policy = loom_amdgpu_low_lower_policy();
  kLowLowerPolicyEntries[6].policy = loom_amdgpu_low_lower_policy();
  kLowLowerPolicyEntries[7].policy = loom_amdgpu_low_lower_policy();
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kLowLowerPolicyEntries,
      IREE_ARRAYSIZE(kLowLowerPolicyEntries));
}

loom_target_low_legality_provider_list_t loom_all_low_legality_provider_list(
    void) {
  return loom_target_low_legality_provider_list_make(
      kLowLegalityProviders, IREE_ARRAYSIZE(kLowLegalityProviders));
}
