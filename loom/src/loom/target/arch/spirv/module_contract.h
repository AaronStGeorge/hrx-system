// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Module-level SPIR-V target contract.
//
// SPIR-V binary emission produces one physical module at a time. Every function
// in that module must agree on the artifact format, ABI, descriptor set, and
// feature contract so module-level declarations and capabilities have one
// meaning. Subgroup widths and dispatch limits belong to individual entries.

#ifndef LOOM_TARGET_ARCH_SPIRV_MODULE_CONTRACT_H_
#define LOOM_TARGET_ARCH_SPIRV_MODULE_CONTRACT_H_

#include "iree/base/api.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_spirv_module_contract_t {
  // Borrowed target-record symbol name selected by the function.
  iree_string_view_t target_name;
  // Target codegen format selected by the function.
  loom_target_codegen_format_t codegen_format;
  // Target artifact format selected by the function.
  loom_target_artifact_format_t artifact_format;
  // Export ABI kind selected by the function.
  loom_target_abi_kind_t abi_kind;
  // Descriptor set stable ID selected by the function.
  uint64_t descriptor_set_stable_id;
  // Stable target-contract descriptor-set identity.
  uint64_t contract_set_stable_id;
  // Target-contract feature bits selected by the function.
  uint64_t contract_feature_bits;
} loom_spirv_module_contract_t;

// Captures the SPIR-V module contract implied by a prepared function binding.
loom_spirv_module_contract_t loom_spirv_module_contract_make(
    iree_string_view_t target_name, const loom_target_bundle_t* target_bundle,
    uint64_t descriptor_set_stable_id, uint64_t contract_feature_bits);

// Merges |candidate| into |accumulator| when both can be emitted into one
// SPIR-V module. Internal functions carry an unknown export ABI; this acts as
// the bottom value and adopts the module's concrete entry ABI without making
// source order part of compatibility.
bool loom_spirv_module_contract_merge(
    loom_spirv_module_contract_t* accumulator,
    const loom_spirv_module_contract_t* candidate);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_MODULE_CONTRACT_H_
