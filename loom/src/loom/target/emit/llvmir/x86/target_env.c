// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/x86/target_env.h"

#include "loom/target/arch/x86/feature_bits.h"

#define LOOM_LLVMIR_X86_64_TARGET_TRIPLE IREE_SVL("x86_64-unknown-linux-gnu")
#define LOOM_LLVMIR_X86_64_DATA_LAYOUT              \
  IREE_SVL(                                         \
      "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:" \
      "64-i128:128-f80:128-n8:16:32:64-S128")
#define LOOM_LLVMIR_X86_64_PACKED_DOT_FEATURES \
  IREE_SVL("+avx512bf16,+avx512vl,+avxvnni,+avxvnniint8")
#define LOOM_LLVMIR_X86_64_PACKED_DOT_CONTRACT_SET \
  IREE_SVL("x86.packed_dot.avx512bf16-avx512vl-avxvnni-avxvnniint8")

static const loom_llvmir_target_env_t kX86_64UnknownLinuxGnuTargetEnv = {
    .name = LOOM_LLVMIR_X86_64_TARGET_TRIPLE,
    .target_triple = LOOM_LLVMIR_X86_64_TARGET_TRIPLE,
    .data_layout = LOOM_LLVMIR_X86_64_DATA_LAYOUT,
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

static const loom_target_snapshot_t kX86_64ObjectSnapshot = {
    .name = LOOM_LLVMIR_X86_64_TARGET_TRIPLE,
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LLVMIR,
    .target_triple = LOOM_LLVMIR_X86_64_TARGET_TRIPLE,
    .data_layout = LOOM_LLVMIR_X86_64_DATA_LAYOUT,
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

static const loom_target_snapshot_t kX86_64PackedDotObjectSnapshot = {
    .name = IREE_SVL("x86_64-unknown-linux-gnu-packed-dot"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LLVMIR,
    .target_triple = LOOM_LLVMIR_X86_64_TARGET_TRIPLE,
    .data_layout = LOOM_LLVMIR_X86_64_DATA_LAYOUT,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .target_features = LOOM_LLVMIR_X86_64_PACKED_DOT_FEATURES,
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

static const loom_target_export_plan_t kX86_64ObjectExportPlan = {
    .name = IREE_SVL("x86_64-object"),
    .abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};

static const loom_target_config_t kX86_64ObjectConfig = {
    .name = IREE_SVL("default"),
};

static const loom_target_config_t kX86_64PackedDotObjectConfig = {
    .name = IREE_SVL("packed-dot"),
    .contract_set_key = LOOM_LLVMIR_X86_64_PACKED_DOT_CONTRACT_SET,
    .contract_feature_bits =
        LOOM_X86_FEATURE_AVX512_BF16 | LOOM_X86_FEATURE_AVX512_VL |
        LOOM_X86_FEATURE_AVX_VNNI | LOOM_X86_FEATURE_AVX_VNNI_INT8,
};

static const loom_target_bundle_t kX86_64ObjectBundle = {
    .name = IREE_SVL("x86_64-object"),
    .snapshot = &kX86_64ObjectSnapshot,
    .export_plan = &kX86_64ObjectExportPlan,
    .config = &kX86_64ObjectConfig,
};

static const loom_target_bundle_t kX86_64PackedDotObjectBundle = {
    .name = IREE_SVL("x86_64-packed-dot-object"),
    .snapshot = &kX86_64PackedDotObjectSnapshot,
    .export_plan = &kX86_64ObjectExportPlan,
    .config = &kX86_64PackedDotObjectConfig,
};

// These built-in profiles are fixture/default provider conveniences. Production
// lowering should prefer derived profiles from the generic target bundles
// above.
static const loom_llvmir_target_profile_t kX86_64ObjectProfile = {
    .name = IREE_SVL("x86_64-object"),
    .target_env = &kX86_64UnknownLinuxGnuTargetEnv,
    .kind = LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT,
    .exported_linkage = LOOM_LLVMIR_LINKAGE_DSO_LOCAL,
    .kernel_calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
};

static const loom_llvmir_target_profile_t kX86_64PackedDotObjectProfile = {
    .name = IREE_SVL("x86_64-packed-dot-object"),
    .target_env = &kX86_64UnknownLinuxGnuTargetEnv,
    .kind = LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT,
    .target_features = LOOM_LLVMIR_X86_64_PACKED_DOT_FEATURES,
    .exported_linkage = LOOM_LLVMIR_LINKAGE_DSO_LOCAL,
    .kernel_calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
    .x86_packed_dot_feature_bits =
        LOOM_X86_FEATURE_AVX512_BF16 | LOOM_X86_FEATURE_AVX512_VL |
        LOOM_X86_FEATURE_AVX_VNNI | LOOM_X86_FEATURE_AVX_VNNI_INT8,
};

static const loom_llvmir_target_profile_t* const kX86TargetProfiles[] = {
    &kX86_64ObjectProfile,
    &kX86_64PackedDotObjectProfile,
};

static const loom_llvmir_target_profile_provider_t kX86TargetProfileProvider = {
    .name = IREE_SVL("x86"),
    .profiles = kX86TargetProfiles,
    .profile_count = IREE_ARRAYSIZE(kX86TargetProfiles),
    .llc_target_name = IREE_SVL("x86"),
};

const loom_target_bundle_t* loom_llvmir_target_bundle_x86_64_object(void) {
  return &kX86_64ObjectBundle;
}

const loom_target_bundle_t* loom_llvmir_target_bundle_x86_64_packed_dot_object(
    void) {
  return &kX86_64PackedDotObjectBundle;
}

const loom_llvmir_target_env_t* loom_llvmir_target_env_x86_64_unknown_linux_gnu(
    void) {
  return &kX86_64UnknownLinuxGnuTargetEnv;
}

const loom_llvmir_target_profile_t* loom_llvmir_target_profile_x86_64_object(
    void) {
  return &kX86_64ObjectProfile;
}

const loom_llvmir_target_profile_t*
loom_llvmir_target_profile_x86_64_packed_dot_object(void) {
  return &kX86_64PackedDotObjectProfile;
}

const loom_llvmir_target_profile_provider_t*
loom_llvmir_x86_target_profile_provider(void) {
  return &kX86TargetProfileProvider;
}

iree_status_t loom_llvmir_target_profile_initialize_x86_64_object(
    loom_llvmir_target_profile_t* out_profile) {
  if (out_profile == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile output is required");
  }
  *out_profile = kX86_64ObjectProfile;
  return iree_ok_status();
}
