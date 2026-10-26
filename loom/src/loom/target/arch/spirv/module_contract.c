// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/module_contract.h"

loom_spirv_module_contract_t loom_spirv_module_contract_make(
    iree_string_view_t target_name, const loom_target_bundle_t* target_bundle,
    uint64_t descriptor_set_stable_id, uint64_t contract_feature_bits) {
  return (loom_spirv_module_contract_t){
      .target_name = target_name,
      .codegen_format = target_bundle->snapshot->codegen_format,
      .artifact_format = target_bundle->snapshot->artifact_format,
      .abi_kind = target_bundle->export_plan->abi_kind,
      .descriptor_set_stable_id = descriptor_set_stable_id,
      .contract_set_stable_id = target_bundle->config->contract_set.stable_id,
      .contract_feature_bits = contract_feature_bits,
  };
}

bool loom_spirv_module_contract_merge(
    loom_spirv_module_contract_t* accumulator,
    const loom_spirv_module_contract_t* candidate) {
  if (accumulator->codegen_format != candidate->codegen_format ||
      accumulator->artifact_format != candidate->artifact_format ||
      accumulator->descriptor_set_stable_id !=
          candidate->descriptor_set_stable_id ||
      accumulator->contract_set_stable_id !=
          candidate->contract_set_stable_id ||
      accumulator->contract_feature_bits != candidate->contract_feature_bits) {
    return false;
  }
  if (accumulator->abi_kind == LOOM_TARGET_ABI_UNKNOWN) {
    accumulator->abi_kind = candidate->abi_kind;
  } else if (candidate->abi_kind != LOOM_TARGET_ABI_UNKNOWN &&
             accumulator->abi_kind != candidate->abi_kind) {
    return false;
  }
  return true;
}
