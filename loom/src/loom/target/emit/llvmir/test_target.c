// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/test_target.h"

#include <stdint.h>

static const loom_llvmir_target_env_t kTestObjectEnv = {
    .name = IREE_SVL("test-object"),
    .target_triple = IREE_SVL("loom-test64-unknown-none"),
    .data_layout = IREE_SVL("e-p:64:64-i64:64-n8:16:32:64-S128"),
    .object_format = LOOM_LLVMIR_OBJECT_FORMAT_ELF,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
    .offset_bitwidth = 64,
    .address_spaces =
        {
            .generic = 0,
            .global = 0,
            .local = 0,
            .constant = 0,
            .private_memory = 0,
            .buffer_resource = UINT32_MAX,
        },
};

static const loom_llvmir_target_profile_t kTestObjectProfile = {
    .name = IREE_SVL("test-object"),
    .target_env = &kTestObjectEnv,
    .kind = LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT,
    .exported_linkage = LOOM_LLVMIR_LINKAGE_DSO_LOCAL,
};

static const loom_target_snapshot_t kTestObjectSnapshot = {
    .name = IREE_SVL("test-object"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LLVMIR,
    .target_triple = IREE_SVL("loom-test64-unknown-none"),
    .data_layout = IREE_SVL("e-p:64:64-i64:64-n8:16:32:64-S128"),
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 64,
    .offset_bitwidth = 64,
    .memory_spaces =
        {
            .generic = 0,
            .global = 0,
            .workgroup = 0,
            .constant = 0,
            .private_memory = 0,
            .host = 0,
            .descriptor = UINT32_MAX,
        },
};

static const loom_target_export_plan_t kTestObjectExportPlan = {
    .name = IREE_SVL("test-object"),
    .abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};

static const loom_target_config_t kTestObjectConfig = {
    .name = IREE_SVL("test-object"),
};

static const loom_target_bundle_t kTestObjectBundle = {
    .name = IREE_SVL("test-object"),
    .snapshot = &kTestObjectSnapshot,
    .export_plan = &kTestObjectExportPlan,
    .config = &kTestObjectConfig,
};

const loom_llvmir_target_env_t* loom_llvmir_target_env_test_object(void) {
  return &kTestObjectEnv;
}

const loom_llvmir_target_profile_t* loom_llvmir_target_profile_test_object(
    void) {
  return &kTestObjectProfile;
}

const loom_target_bundle_t* loom_llvmir_target_bundle_test_object(void) {
  return &kTestObjectBundle;
}
