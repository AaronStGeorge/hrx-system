// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/all/low_registry.h"

#include "iree/base/threading/call_once.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/arch/wasm/provider.h"
#include "loom/target/arch/x86/provider.h"
#include "loom/target/emit/ireevm/provider.h"

static const loom_target_provider_t* const kAllTargetProviders[] = {
    &loom_ireevm_target_provider,
    &loom_wasm_target_provider,
    &loom_x86_target_provider,
    &loom_amdgpu_target_provider,
};

static loom_target_environment_t kAllTargetEnvironment;
static iree_once_flag kAllTargetEnvironmentOnce = IREE_ONCE_FLAG_INIT;

static void loom_all_low_registry_initialize_environment(void) {
  static const loom_target_provider_set_t kProviderSet = {
      .providers = kAllTargetProviders,
      .provider_count = IREE_ARRAYSIZE(kAllTargetProviders),
  };
  IREE_CHECK_OK(loom_target_environment_initialize(&kProviderSet,
                                                   &kAllTargetEnvironment));
}

void loom_all_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  iree_call_once(&kAllTargetEnvironmentOnce,
                 loom_all_low_registry_initialize_environment);
  IREE_CHECK_OK(loom_target_environment_initialize_low_descriptor_registry(
      &kAllTargetEnvironment, out_registry));
}

void loom_all_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  iree_call_once(&kAllTargetEnvironmentOnce,
                 loom_all_low_registry_initialize_environment);
  IREE_CHECK_OK(loom_target_environment_initialize_low_lower_policy_registry(
      &kAllTargetEnvironment, out_registry));
}

loom_target_low_legality_provider_list_t loom_all_low_legality_provider_list(
    void) {
  iree_call_once(&kAllTargetEnvironmentOnce,
                 loom_all_low_registry_initialize_environment);
  return loom_target_environment_low_legality_provider_list(
      &kAllTargetEnvironment);
}
