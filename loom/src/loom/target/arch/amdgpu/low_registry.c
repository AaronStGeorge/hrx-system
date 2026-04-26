// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/low_registry.h"

#include <stdint.h>

#include "loom/target/arch/amdgpu/gfx11_descriptors.h"
#include "loom/target/arch/amdgpu/gfx1250_descriptors.h"
#include "loom/target/arch/amdgpu/gfx12_descriptors.h"
#include "loom/target/arch/amdgpu/gfx950_descriptors.h"

#define LOOM_AMDGPU_TARGET_TRIPLE IREE_SVL("amdgcn-amd-amdhsa")
#define LOOM_AMDGPU_DATA_LAYOUT                              \
  IREE_SVL(                                                  \
      "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:" \
      "32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:"  \
      "192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-"   \
      "v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:" \
      "2048-n32:64-S32-A5-G1-ni:7:8:9")

#define LOOM_AMDGPU_LOW_SNAPSHOT(symbol, snapshot_name, cpu)   \
  static const loom_target_snapshot_t symbol = {               \
      .name = IREE_SVL(snapshot_name),                         \
      .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE, \
      .target_triple = LOOM_AMDGPU_TARGET_TRIPLE,              \
      .data_layout = LOOM_AMDGPU_DATA_LAYOUT,                  \
      .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,      \
      .target_cpu = IREE_SVL(cpu),                             \
      .default_pointer_bitwidth = 64,                          \
      .index_bitwidth = 32,                                    \
      .offset_bitwidth = 64,                                   \
      .max_workgroup_size =                                    \
          {                                                    \
              .x = 1024,                                       \
              .y = 1024,                                       \
              .z = 1024,                                       \
          },                                                   \
      .max_flat_workgroup_size = 1024,                         \
      .max_grid_size =                                         \
          {                                                    \
              .x = INT32_MAX,                                  \
              .y = UINT16_MAX,                                 \
              .z = UINT16_MAX,                                 \
          },                                                   \
      .max_flat_grid_size = UINT32_MAX,                        \
      .max_workgroup_count =                                   \
          {                                                    \
              .x = INT32_MAX,                                  \
              .y = UINT16_MAX,                                 \
              .z = UINT16_MAX,                                 \
          },                                                   \
      .memory_spaces =                                         \
          {                                                    \
              .generic = 0,                                    \
              .global = 1,                                     \
              .workgroup = 3,                                  \
              .constant = 4,                                   \
              .private_memory = 5,                             \
              .host = UINT32_MAX,                              \
              .descriptor = 7,                                 \
          },                                                   \
  }

LOOM_AMDGPU_LOW_SNAPSHOT(kAmdgpuGfx950Snapshot, "amdgpu-gfx950-low", "gfx950");
LOOM_AMDGPU_LOW_SNAPSHOT(kAmdgpuGfx11Snapshot, "amdgpu-gfx11-low", "gfx1100");
LOOM_AMDGPU_LOW_SNAPSHOT(kAmdgpuGfx12Snapshot, "amdgpu-gfx12-low", "gfx1200");
LOOM_AMDGPU_LOW_SNAPSHOT(kAmdgpuGfx1250Snapshot, "amdgpu-gfx1250-low",
                         "gfx1250");

static const loom_target_export_plan_t kAmdgpuHalExportPlan = {
    .name = IREE_SVL("amdgpu-hal"),
    .abi_kind = LOOM_TARGET_ABI_HAL_KERNEL,
    .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
    .hal_kernel =
        {
            .binding_alignment = 16,
            .required_workgroup_size = {.x = 0, .y = 0, .z = 0},
            .flat_workgroup_size_min = 64,
            .flat_workgroup_size_max = 64,
            .buffer_resource_flags = 159744,
        },
};

#define LOOM_AMDGPU_LOW_CONFIG(symbol, key)    \
  static const loom_target_config_t symbol = { \
      .name = IREE_SVL(key),                   \
      .contract_set_key = IREE_SVL(key),       \
  }

LOOM_AMDGPU_LOW_CONFIG(kAmdgpuGfx950Config, "amdgpu.gfx950.core");
LOOM_AMDGPU_LOW_CONFIG(kAmdgpuGfx11Config, "amdgpu.gfx11.core");
LOOM_AMDGPU_LOW_CONFIG(kAmdgpuGfx12Config, "amdgpu.gfx12.core");
LOOM_AMDGPU_LOW_CONFIG(kAmdgpuGfx1250Config, "amdgpu.gfx1250.core");

const loom_target_bundle_t loom_amdgpu_low_target_bundle_gfx950_core = {
    .name = IREE_SVL("amdgpu-gfx950"),
    .snapshot = &kAmdgpuGfx950Snapshot,
    .export_plan = &kAmdgpuHalExportPlan,
    .config = &kAmdgpuGfx950Config,
};

const loom_target_bundle_t loom_amdgpu_low_target_bundle_gfx11_core = {
    .name = IREE_SVL("amdgpu-gfx11"),
    .snapshot = &kAmdgpuGfx11Snapshot,
    .export_plan = &kAmdgpuHalExportPlan,
    .config = &kAmdgpuGfx11Config,
};

const loom_target_bundle_t loom_amdgpu_low_target_bundle_gfx12_core = {
    .name = IREE_SVL("amdgpu-gfx12"),
    .snapshot = &kAmdgpuGfx12Snapshot,
    .export_plan = &kAmdgpuHalExportPlan,
    .config = &kAmdgpuGfx12Config,
};

const loom_target_bundle_t loom_amdgpu_low_target_bundle_gfx1250_core = {
    .name = IREE_SVL("amdgpu-gfx1250"),
    .snapshot = &kAmdgpuGfx1250Snapshot,
    .export_plan = &kAmdgpuHalExportPlan,
    .config = &kAmdgpuGfx1250Config,
};

static const loom_low_descriptor_set_provider_t kLowDescriptorSetProviders[] = {
    loom_amdgpu_gfx950_core_descriptor_set,
    loom_amdgpu_gfx11_core_descriptor_set,
    loom_amdgpu_gfx12_core_descriptor_set,
    loom_amdgpu_gfx1250_core_descriptor_set,
};

static const loom_target_bundle_t* const kLowTargetBundles[] = {
    &loom_amdgpu_low_target_bundle_gfx950_core,
    &loom_amdgpu_low_target_bundle_gfx11_core,
    &loom_amdgpu_low_target_bundle_gfx12_core,
    &loom_amdgpu_low_target_bundle_gfx1250_core,
};

void loom_amdgpu_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry) {
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, kLowDescriptorSetProviders,
      IREE_ARRAYSIZE(kLowDescriptorSetProviders), kLowTargetBundles,
      IREE_ARRAYSIZE(kLowTargetBundles));
}
