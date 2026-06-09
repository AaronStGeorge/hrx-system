// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/llvmir/target_records.h"

#include <stdint.h>

static const loom_target_snapshot_t kLlvmirObjectProjectionSnapshot = {
    .name = IREE_SVL("llvmir-object-projection"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LLVMIR,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
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
            .host = 0,
            .descriptor = UINT32_MAX,
        },
};

static const loom_target_export_plan_t kLlvmirObjectProjectionExportPlan = {
    .name = IREE_SVL("llvmir-object-projection"),
    .abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};

static const loom_target_config_t kLlvmirObjectProjectionConfig = {
    .name = IREE_SVL("llvmir-object-projection"),
};

static const loom_target_bundle_t kLlvmirObjectProjectionBundle = {
    .name = IREE_SVL("llvmir-object-projection"),
    .snapshot = &kLlvmirObjectProjectionSnapshot,
    .export_plan = &kLlvmirObjectProjectionExportPlan,
    .config = &kLlvmirObjectProjectionConfig,
};

static const loom_target_bundle_t* const kProjectionBundleValues[] = {
    NULL,
    &kLlvmirObjectProjectionBundle,
};

const loom_target_bundle_table_t loom_llvmir_projection_target_bundles = {
    .values = kProjectionBundleValues,
    .count = IREE_ARRAYSIZE(kProjectionBundleValues),
};
