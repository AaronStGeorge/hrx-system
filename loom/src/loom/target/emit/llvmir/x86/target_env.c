// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/x86/target_env.h"

#include "loom/target/arch/x86/packed_dot_contract.h"

static const loom_llvmir_target_env_t kX86_64UnknownLinuxGnuTargetEnv = {
    .name = IREE_SVL("x86_64-unknown-linux-gnu"),
    .target_triple = IREE_SVL("x86_64-unknown-linux-gnu"),
    .data_layout = IREE_SVL("e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:"
                            "64-i128:128-f80:128-n8:16:32:64-S128"),
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
    .target_features = IREE_SVL("+avx512bf16,+avx512vl,+avxvnni,+avxvnniint8"),
    .exported_linkage = LOOM_LLVMIR_LINKAGE_DSO_LOCAL,
    .kernel_calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
    .x86_packed_dot_feature_bits = LOOM_X86_PACKED_DOT_FEATURE_AVX512_BF16 |
                                   LOOM_X86_PACKED_DOT_FEATURE_AVX512_VL |
                                   LOOM_X86_PACKED_DOT_FEATURE_AVX_VNNI |
                                   LOOM_X86_PACKED_DOT_FEATURE_AVX_VNNI_INT8,
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
