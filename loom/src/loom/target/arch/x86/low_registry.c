// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/low_registry.h"

#include <stdint.h>

#include "loom/target/arch/x86/avx512_descriptors.h"
#include "loom/target/arch/x86/feature_bits.h"
#include "loom/target/arch/x86/packed_dot_descriptors.h"

#define LOOM_X86_64_TARGET_TRIPLE IREE_SVL("x86_64-unknown-linux-gnu")
#define LOOM_X86_64_DATA_LAYOUT                     \
  IREE_SVL(                                         \
      "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:" \
      "64-i128:128-f80:128-n8:16:32:64-S128")
#define LOOM_X86_64_AVX512_FEATURES \
  IREE_SVL("+avx512f,+avx512bw,+avx512dq,+avx512vl,+fma")
#define LOOM_X86_64_PACKED_DOT_FEATURES \
  IREE_SVL("+avx512bf16,+avx512vl,+avxvnni,+avxvnniint8")
#define LOOM_X86_64_PACKED_DOT_DESCRIPTOR_SET IREE_SVL("x86.packed_dot.core")

static const loom_target_snapshot_t kX86_64Avx512Snapshot = {
    .name = IREE_SVL("x86_64-avx512-low"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    .target_triple = LOOM_X86_64_TARGET_TRIPLE,
    .data_layout = LOOM_X86_64_DATA_LAYOUT,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .target_cpu = IREE_SVL("x86-64-v4"),
    .target_features = LOOM_X86_64_AVX512_FEATURES,
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

static const loom_target_snapshot_t kX86_64PackedDotSnapshot = {
    .name = IREE_SVL("x86_64-packed-dot-low"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
    .target_triple = LOOM_X86_64_TARGET_TRIPLE,
    .data_layout = LOOM_X86_64_DATA_LAYOUT,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .target_cpu = IREE_SVL("x86-64-v4"),
    .target_features = LOOM_X86_64_PACKED_DOT_FEATURES,
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

static const loom_target_config_t kX86_64Avx512Config = {
    .name = IREE_SVL("x86.avx512.core"),
    .contract_set_key = IREE_SVL("x86.avx512.core"),
};

static const loom_target_config_t kX86_64PackedDotConfig = {
    .name = IREE_SVL("x86.packed_dot.core"),
    .contract_set_key = LOOM_X86_64_PACKED_DOT_DESCRIPTOR_SET,
    .contract_feature_bits =
        LOOM_X86_FEATURE_AVX512_BF16 | LOOM_X86_FEATURE_AVX512_VL |
        LOOM_X86_FEATURE_AVX_VNNI | LOOM_X86_FEATURE_AVX_VNNI_INT8,
};

const loom_target_bundle_t loom_x86_low_target_bundle_avx512_core = {
    .name = IREE_SVL("x86-avx512"),
    .snapshot = &kX86_64Avx512Snapshot,
    .export_plan = &kX86_64ObjectExportPlan,
    .config = &kX86_64Avx512Config,
};

const loom_target_bundle_t loom_x86_low_target_bundle_packed_dot_core = {
    .name = IREE_SVL("x86-packed-dot"),
    .snapshot = &kX86_64PackedDotSnapshot,
    .export_plan = &kX86_64ObjectExportPlan,
    .config = &kX86_64PackedDotConfig,
};

static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
    loom_x86_avx512_core_descriptor_set,
    loom_x86_packed_dot_core_descriptor_set,
};

static const loom_target_bundle_t* const kLowTargetBundles[] = {
    &loom_x86_low_target_bundle_avx512_core,
    &loom_x86_low_target_bundle_packed_dot_core,
};

void loom_x86_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kLowDescriptorSetProviders,
      IREE_ARRAYSIZE(kLowDescriptorSetProviders), kLowTargetBundles,
      IREE_ARRAYSIZE(kLowTargetBundles));
}
