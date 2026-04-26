// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared target payload types.
//
// This header intentionally contains only target-neutral data shapes. It must
// not include target-family descriptors, presets, feature catalogs, scheduling
// tables, or emitter-specific helpers. Those live under opt-in target packages
// so AOT and JIT builds can exclude every target family they do not use.

#ifndef LOOM_TARGET_TYPES_H_
#define LOOM_TARGET_TYPES_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_target_codegen_format_e {
  LOOM_TARGET_CODEGEN_FORMAT_UNKNOWN = 0,
  LOOM_TARGET_CODEGEN_FORMAT_LLVMIR = 1,
  LOOM_TARGET_CODEGEN_FORMAT_SPIRV = 2,
  LOOM_TARGET_CODEGEN_FORMAT_VM = 3,
  LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE = 4,
  LOOM_TARGET_CODEGEN_FORMAT_WASM = 5,
} loom_target_codegen_format_t;

typedef enum loom_target_artifact_format_e {
  LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN = 0,
  LOOM_TARGET_ARTIFACT_FORMAT_ELF = 1,
  LOOM_TARGET_ARTIFACT_FORMAT_COFF = 2,
  LOOM_TARGET_ARTIFACT_FORMAT_MACHO = 3,
  LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY = 4,
  LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE = 5,
  LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY = 6,
} loom_target_artifact_format_t;

typedef enum loom_target_artifact_abi_kind_e {
  LOOM_TARGET_ARTIFACT_ABI_KIND_UNKNOWN = 0,
  LOOM_TARGET_ARTIFACT_ABI_KIND_OBJECT_FILE = 1,
  LOOM_TARGET_ARTIFACT_ABI_KIND_HAL_EXECUTABLE = 2,
  LOOM_TARGET_ARTIFACT_ABI_KIND_VM_MODULE = 3,
  LOOM_TARGET_ARTIFACT_ABI_KIND_WASM_MODULE = 4,
  LOOM_TARGET_ARTIFACT_ABI_KIND_SPIRV_MODULE = 5,
} loom_target_artifact_abi_kind_t;

typedef enum loom_target_abi_kind_e {
  LOOM_TARGET_ABI_UNKNOWN = 0,
  LOOM_TARGET_ABI_OBJECT_FUNCTION = 1,
  LOOM_TARGET_ABI_HAL_KERNEL = 2,
  LOOM_TARGET_ABI_VM_MODULE_FUNCTION = 3,
  LOOM_TARGET_ABI_SHADER_ENTRY_POINT = 4,
  LOOM_TARGET_ABI_WASM_FUNCTION = 5,
} loom_target_abi_kind_t;

typedef enum loom_target_linkage_e {
  LOOM_TARGET_LINKAGE_DEFAULT = 0,
  LOOM_TARGET_LINKAGE_DSO_LOCAL = 1,
} loom_target_linkage_t;

typedef struct loom_target_memory_space_map_t {
  // Generic/default pointer address space.
  uint32_t generic;
  // Device or process global memory address space.
  uint32_t global;
  // Workgroup/shared memory address space.
  uint32_t workgroup;
  // Constant memory address space.
  uint32_t constant;
  // Per-invocation/private memory address space.
  uint32_t private_memory;
  // Host-visible memory address space, or UINT32_MAX when unavailable.
  uint32_t host;
  // Descriptor/resource identity address space, or UINT32_MAX when unavailable.
  uint32_t descriptor;
} loom_target_memory_space_map_t;

typedef struct loom_target_workgroup_size_t {
  // Workgroup size along the x dimension.
  uint32_t x;
  // Workgroup size along the y dimension.
  uint32_t y;
  // Workgroup size along the z dimension.
  uint32_t z;
} loom_target_workgroup_size_t;

typedef struct loom_target_grid_size_t {
  // Maximum dispatched grid size along the x dimension.
  uint32_t x;
  // Maximum dispatched grid size along the y dimension.
  uint32_t y;
  // Maximum dispatched grid size along the z dimension.
  uint32_t z;
} loom_target_grid_size_t;

typedef struct loom_target_workgroup_count_limit_t {
  // Maximum dispatch workgroup count along the x dimension.
  uint32_t x;
  // Maximum dispatch workgroup count along the y dimension.
  uint32_t y;
  // Maximum dispatch workgroup count along the z dimension.
  uint32_t z;
} loom_target_workgroup_count_limit_t;

typedef struct loom_target_snapshot_t {
  // Stable snapshot name for diagnostics and tests.
  iree_string_view_t name;
  // Primary emitter family for this snapshot.
  loom_target_codegen_format_t codegen_format;
  // ISA, object, or toolchain target triple when the backend uses one.
  iree_string_view_t target_triple;
  // Target data layout string when the backend exposes one.
  iree_string_view_t data_layout;
  // Linkable or loadable artifact format produced for this snapshot.
  loom_target_artifact_format_t artifact_format;
  // Target CPU, GPU chip, or architecture variant name.
  iree_string_view_t target_cpu;
  // Downstream feature string used by emitter/tool adapters.
  iree_string_view_t target_features;
  // Default pointer bit width for target-independent layout decisions.
  uint32_t default_pointer_bitwidth;
  // Index bit width chosen for lowered index values.
  uint32_t index_bitwidth;
  // Offset bit width chosen for lowered byte offsets.
  uint32_t offset_bitwidth;
  // Maximum API/hardware local workgroup size per dimension. Zero dimensions
  // mean the target has not supplied a tighter limit.
  loom_target_workgroup_size_t max_workgroup_size;
  // Maximum flat local workgroup size. Zero means the target has not supplied a
  // tighter total-workitem limit across the workgroup dimensions.
  uint32_t max_flat_workgroup_size;
  // Maximum API/hardware dispatched grid size per dimension, in workitems.
  // Zero dimensions mean the target has not supplied a tighter limit.
  loom_target_grid_size_t max_grid_size;
  // Maximum flat dispatched grid size, in workitems. Zero means the target has
  // not supplied a tighter total-workitem limit across the grid dimensions.
  uint64_t max_flat_grid_size;
  // Maximum API/hardware dispatch workgroup count per dimension. Zero
  // dimensions mean the target has not supplied a tighter limit.
  loom_target_workgroup_count_limit_t max_workgroup_count;
  // Address space assignments used by target-specific ABI lowering.
  loom_target_memory_space_map_t memory_spaces;
} loom_target_snapshot_t;

typedef struct loom_target_hal_kernel_abi_t {
  // ABI-required byte alignment for binding pointer parameters.
  uint32_t binding_alignment;
  // Function-selected fixed workgroup size for a kernel entry point. A zero
  // dimension means the function has not selected a fixed size yet.
  loom_target_workgroup_size_t required_workgroup_size;
  // Optimization lower flat workgroup size advertised to the backend.
  uint32_t flat_workgroup_size_min;
  // Optimization upper flat workgroup size advertised to the backend.
  uint32_t flat_workgroup_size_max;
  // ABI-required raw buffer resource flags for global binding resources.
  uint32_t buffer_resource_flags;
} loom_target_hal_kernel_abi_t;

typedef struct loom_target_export_plan_t {
  // Stable export plan name for diagnostics and tests.
  iree_string_view_t name;
  // Output symbol exposed by this export, or empty to preserve the source name.
  iree_string_view_t export_symbol;
  // Callable/package ABI used for this export.
  loom_target_abi_kind_t abi_kind;
  // ABI-required linkage for exported object functions or entry points.
  loom_target_linkage_t linkage;
  // HAL kernel ABI facts when abi_kind is LOOM_TARGET_ABI_HAL_KERNEL.
  loom_target_hal_kernel_abi_t hal_kernel;
} loom_target_export_plan_t;

typedef struct loom_target_config_t {
  // Stable config name for diagnostics and tests.
  iree_string_view_t name;
  // Provider-defined target-contract selection key, or empty for default
  // policy.
  iree_string_view_t contract_set_key;
  // Provider-defined target-contract feature bitset, or zero for default
  // policy.
  uint64_t contract_feature_bits;
} loom_target_config_t;

typedef struct loom_target_bundle_t {
  // Stable bundle/profile name for diagnostics and tests.
  iree_string_view_t name;
  // Frozen codegen target snapshot selected for this bundle.
  const loom_target_snapshot_t* snapshot;
  // Export and ABI plan selected for this bundle.
  const loom_target_export_plan_t* export_plan;
  // Legalization and target-contract config selected for this bundle.
  const loom_target_config_t* config;
} loom_target_bundle_t;

typedef struct loom_target_bundle_storage_t {
  // Materialized target snapshot owned by this storage object.
  loom_target_snapshot_t snapshot;
  // Materialized function ABI/export contract owned by this storage object.
  loom_target_export_plan_t export_plan;
  // Materialized target configuration owned by this storage object.
  loom_target_config_t config;
  // Bundle view pointing at snapshot, export_plan, and config above.
  loom_target_bundle_t bundle;
} loom_target_bundle_storage_t;

// Rebinds |storage->bundle| to the payloads embedded in |storage|. Call this
// after copying a storage object by value; the embedded bundle is a view and
// its pointers must refer to the same enclosing storage object.
static inline void loom_target_bundle_storage_rebind(
    loom_target_bundle_storage_t* storage) {
  IREE_ASSERT_ARGUMENT(storage);
  storage->bundle.snapshot = &storage->snapshot;
  storage->bundle.export_plan = &storage->export_plan;
  storage->bundle.config = &storage->config;
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_TYPES_H_
