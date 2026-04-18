// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/test/low_registry.h"

#include <stdint.h>

#include "loom/target/test/descriptors.h"

static const loom_target_snapshot_t kTestLowSnapshot = {
    .name = IREE_SVL("test-low"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    .target_triple = IREE_SVL("test-low"),
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .target_cpu = IREE_SVL("test-low"),
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
    .offset_bitwidth = 64,
    .memory_spaces =
        {
            .generic = 0,
            .global = 0,
            .workgroup = UINT32_MAX,
            .constant = 0,
            .private_memory = 0,
            .host = UINT32_MAX,
            .descriptor = UINT32_MAX,
        },
};

static const loom_target_export_plan_t kTestLowExportPlan = {
    .name = IREE_SVL("test-low-function"),
    .abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};

static const loom_target_config_t kTestLowConfig = {
    .name = IREE_SVL("test.low.core"),
    .contract_set_key = IREE_SVL("test.low.core"),
};

const loom_target_bundle_t loom_test_low_target_bundle_core = {
    .name = IREE_SVL("test-low"),
    .snapshot = &kTestLowSnapshot,
    .export_plan = &kTestLowExportPlan,
    .config = &kTestLowConfig,
};

static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
    loom_test_low_core_descriptor_set,
};

static const loom_target_bundle_t* const kLowTargetBundles[] = {
    &loom_test_low_target_bundle_core,
};

void loom_test_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kLowDescriptorSetProviders,
      IREE_ARRAYSIZE(kLowDescriptorSetProviders), kLowTargetBundles,
      IREE_ARRAYSIZE(kLowTargetBundles));
}
