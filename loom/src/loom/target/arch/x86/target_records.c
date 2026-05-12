// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/target_records.h"

#include <stdint.h>

#include "loom/target/arch/x86/feature_bits.h"

#define LOOM_X86_64_SCALAR_DESCRIPTOR_SET IREE_SVL("x86.scalar.core")
#define LOOM_X86_64_SIMD128_DESCRIPTOR_SET IREE_SVL("x86.simd128.core")
#define LOOM_X86_64_AVX2_DESCRIPTOR_SET IREE_SVL("x86.avx2.core")
#define LOOM_X86_64_PACKED_DOT_DESCRIPTOR_SET IREE_SVL("x86.packed_dot.core")
#define LOOM_X86_64_AVX512_PACKED_DOT_DESCRIPTOR_SET \
  IREE_SVL("x86.avx512_packed_dot.core")

static const loom_target_snapshot_t kX86_64ScalarSnapshot = {
    .name = IREE_SVL("x86_64-scalar-low"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
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

static const loom_target_snapshot_t kX86_64Simd128Snapshot = {
    .name = IREE_SVL("x86_64-simd128-low"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
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

static const loom_target_snapshot_t kX86_64Avx2Snapshot = {
    .name = IREE_SVL("x86_64-avx2-low"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
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

static const loom_target_snapshot_t kX86_64Avx512Snapshot = {
    .name = IREE_SVL("x86_64-avx512-low"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
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

static const loom_target_snapshot_t kX86_64PackedDotSnapshot = {
    .name = IREE_SVL("x86_64-packed-dot-low"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
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

static const loom_target_snapshot_t kX86_64Avx512PackedDotSnapshot = {
    .name = IREE_SVL("x86_64-avx512-packed-dot-low"),
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,
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

static const loom_target_export_plan_t kX86_64ObjectExportPlan = {
    .name = IREE_SVL("x86_64-object"),
    .abi_kind = LOOM_TARGET_ABI_OBJECT_FUNCTION,
    .linkage = LOOM_TARGET_LINKAGE_DSO_LOCAL,
};

static const loom_target_config_t kX86_64ScalarConfig = {
    .name = IREE_SVL("x86.scalar.core"),
    .contract_set_key = LOOM_X86_64_SCALAR_DESCRIPTOR_SET,
};

static const loom_target_config_t kX86_64Simd128Config = {
    .name = IREE_SVL("x86.simd128.core"),
    .contract_set_key = LOOM_X86_64_SIMD128_DESCRIPTOR_SET,
};

static const loom_target_config_t kX86_64Avx2Config = {
    .name = IREE_SVL("x86.avx2.core"),
    .contract_set_key = LOOM_X86_64_AVX2_DESCRIPTOR_SET,
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

static const loom_target_config_t kX86_64Avx512PackedDotConfig = {
    .name = IREE_SVL("x86.avx512_packed_dot.core"),
    .contract_set_key = LOOM_X86_64_AVX512_PACKED_DOT_DESCRIPTOR_SET,
    .contract_feature_bits =
        LOOM_X86_FEATURE_AVX512_VNNI | LOOM_X86_FEATURE_AVX512_BF16 |
        LOOM_X86_FEATURE_AVX512_VL | LOOM_X86_FEATURE_AVX_VNNI |
        LOOM_X86_FEATURE_AVX_VNNI_INT8,
};

const loom_target_bundle_t loom_x86_low_target_bundle_scalar_core = {
    .name = IREE_SVL("x86-scalar"),
    .snapshot = &kX86_64ScalarSnapshot,
    .export_plan = &kX86_64ObjectExportPlan,
    .config = &kX86_64ScalarConfig,
};

const loom_target_bundle_t loom_x86_low_target_bundle_simd128_core = {
    .name = IREE_SVL("x86-simd128"),
    .snapshot = &kX86_64Simd128Snapshot,
    .export_plan = &kX86_64ObjectExportPlan,
    .config = &kX86_64Simd128Config,
};

const loom_target_bundle_t loom_x86_low_target_bundle_avx2_core = {
    .name = IREE_SVL("x86-avx2"),
    .snapshot = &kX86_64Avx2Snapshot,
    .export_plan = &kX86_64ObjectExportPlan,
    .config = &kX86_64Avx2Config,
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

const loom_target_bundle_t loom_x86_low_target_bundle_avx512_packed_dot_core = {
    .name = IREE_SVL("x86-avx512-packed-dot"),
    .snapshot = &kX86_64Avx512PackedDotSnapshot,
    .export_plan = &kX86_64ObjectExportPlan,
    .config = &kX86_64Avx512PackedDotConfig,
};

static const loom_target_bundle_t* const kX86TargetBundleValues[] = {
    NULL,
    &loom_x86_low_target_bundle_avx512_core,
    &loom_x86_low_target_bundle_packed_dot_core,
    &loom_x86_low_target_bundle_avx512_packed_dot_core,
    &loom_x86_low_target_bundle_scalar_core,
    &loom_x86_low_target_bundle_simd128_core,
    &loom_x86_low_target_bundle_avx2_core,
};

const loom_target_bundle_table_t loom_x86_target_bundles = {
    .values = kX86TargetBundleValues,
    .count = IREE_ARRAYSIZE(kX86TargetBundleValues),
};
