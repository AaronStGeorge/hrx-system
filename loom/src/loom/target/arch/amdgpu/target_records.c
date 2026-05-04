// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/target_records.h"

#include <stdint.h>

#include "loom/target/arch/amdgpu/target_info.h"

// clang-format off
#define LOOM_AMDGPU_LOW_SNAPSHOT(symbol, snapshot_name, wavefront_size)      \
  static const loom_target_snapshot_t symbol = {                             \
      .name = IREE_SVL(snapshot_name),                                       \
      .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE,               \
      .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,                    \
      .default_pointer_bitwidth = 64,                                        \
      .index_bitwidth = 32,                                                  \
      .offset_bitwidth = 64,                                                 \
      .max_workgroup_size = {.x = 1024, .y = 1024, .z = 1024},               \
      .max_flat_workgroup_size = 1024,                                       \
      .subgroup_size = wavefront_size,                                       \
      .max_grid_size = {.x = INT32_MAX, .y = UINT16_MAX, .z = UINT16_MAX},   \
      .max_flat_grid_size = UINT32_MAX,                                      \
      .max_workgroup_count = {.x = INT32_MAX, .y = UINT16_MAX, .z = UINT16_MAX}, \
      .memory_spaces = {                                                     \
          .generic = 0,                                                      \
          .global = 1,                                                       \
          .workgroup = 3,                                                    \
          .constant = 4,                                                     \
          .private_memory = 5,                                               \
          .host = UINT32_MAX,                                                \
          .descriptor = 7,                                                   \
      },                                                                     \
  }

LOOM_AMDGPU_LOW_SNAPSHOT(kAmdgpuCdna3Snapshot, "amdgpu-cdna3-low", 64);
LOOM_AMDGPU_LOW_SNAPSHOT(kAmdgpuCdna4Snapshot, "amdgpu-cdna4-low", 64);
LOOM_AMDGPU_LOW_SNAPSHOT(kAmdgpuRdna3Snapshot, "amdgpu-rdna3-low", 32);
LOOM_AMDGPU_LOW_SNAPSHOT(kAmdgpuRdna4Snapshot, "amdgpu-rdna4-low", 32);
LOOM_AMDGPU_LOW_SNAPSHOT(kAmdgpuRdna4Gfx125xSnapshot, "amdgpu-rdna4-gfx125x-low", 32);

static const loom_target_export_plan_t kAmdgpuHalExportPlan = {
  .name = IREE_SVL("amdgpu-hal"),
  .abi_kind = LOOM_TARGET_ABI_HAL_KERNEL,
  .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
  .hal_kernel = {
    .binding_alignment = 16,
    .required_workgroup_size = {.x = 0, .y = 0, .z = 0},
    .flat_workgroup_size_min = 0,
    .flat_workgroup_size_max = 0,
    .buffer_resource_flags = LOOM_AMDGPU_HAL_BUFFER_RESOURCE_FLAGS,
  },
};

#define LOOM_AMDGPU_LOW_CONFIG(symbol, key) \
  static const loom_target_config_t symbol = { \
      .name = IREE_SVL(key), \
      .contract_set_key = IREE_SVL(key), \
  }

LOOM_AMDGPU_LOW_CONFIG(kAmdgpuCdna3Config, "amdgpu.cdna3.core");
LOOM_AMDGPU_LOW_CONFIG(kAmdgpuCdna4Config, "amdgpu.cdna4.core");
LOOM_AMDGPU_LOW_CONFIG(kAmdgpuRdna3Config, "amdgpu.rdna3.core");
LOOM_AMDGPU_LOW_CONFIG(kAmdgpuRdna4Config, "amdgpu.rdna4.core");
LOOM_AMDGPU_LOW_CONFIG(kAmdgpuRdna4Gfx125xConfig, "amdgpu.rdna4.gfx125x.core");

static const loom_target_bundle_t kAmdgpuLowTargetBundleCdna3Core = {
  .name = IREE_SVL("amdgpu-cdna3"),
  .snapshot = &kAmdgpuCdna3Snapshot,
  .export_plan = &kAmdgpuHalExportPlan,
  .config = &kAmdgpuCdna3Config,
};

static const loom_target_bundle_t kAmdgpuLowTargetBundleCdna4Core = {
  .name = IREE_SVL("amdgpu-cdna4"),
  .snapshot = &kAmdgpuCdna4Snapshot,
  .export_plan = &kAmdgpuHalExportPlan,
  .config = &kAmdgpuCdna4Config,
};

static const loom_target_bundle_t kAmdgpuLowTargetBundleRdna3Core = {
  .name = IREE_SVL("amdgpu-rdna3"),
  .snapshot = &kAmdgpuRdna3Snapshot,
  .export_plan = &kAmdgpuHalExportPlan,
  .config = &kAmdgpuRdna3Config,
};

static const loom_target_bundle_t kAmdgpuLowTargetBundleRdna4Core = {
  .name = IREE_SVL("amdgpu-rdna4"),
  .snapshot = &kAmdgpuRdna4Snapshot,
  .export_plan = &kAmdgpuHalExportPlan,
  .config = &kAmdgpuRdna4Config,
};

static const loom_target_bundle_t kAmdgpuLowTargetBundleRdna4Gfx125xCore = {
  .name = IREE_SVL("amdgpu-rdna4-gfx125x"),
  .snapshot = &kAmdgpuRdna4Gfx125xSnapshot,
  .export_plan = &kAmdgpuHalExportPlan,
  .config = &kAmdgpuRdna4Gfx125xConfig,
};

static const loom_target_bundle_t* const kAmdgpuTargetBundleValues[] = {
  NULL,
  &kAmdgpuLowTargetBundleCdna3Core,
  &kAmdgpuLowTargetBundleCdna4Core,
  &kAmdgpuLowTargetBundleRdna3Core,
  &kAmdgpuLowTargetBundleRdna4Core,
  &kAmdgpuLowTargetBundleRdna4Gfx125xCore,
};

const loom_target_bundle_table_t loom_amdgpu_target_bundles = {
  .values = kAmdgpuTargetBundleValues,
  .count = IREE_ARRAYSIZE(kAmdgpuTargetBundleValues),
};

// clang-format on

const loom_target_bundle_t* loom_amdgpu_target_bundle_for_descriptor_set(
    uint16_t descriptor_set_ordinal) {
  const loom_target_bundle_t* const
      bundles[LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_COUNT] = {
          [LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA3] =
              &kAmdgpuLowTargetBundleCdna3Core,
          [LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_CDNA4] =
              &kAmdgpuLowTargetBundleCdna4Core,
          [LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA3] =
              &kAmdgpuLowTargetBundleRdna3Core,
          [LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4] =
              &kAmdgpuLowTargetBundleRdna4Core,
          [LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_RDNA4_GFX125X] =
              &kAmdgpuLowTargetBundleRdna4Gfx125xCore,
      };
  if (descriptor_set_ordinal >= IREE_ARRAYSIZE(bundles)) {
    return NULL;
  }
  return bundles[descriptor_set_ordinal];
}
