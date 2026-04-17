// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared target record types.
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
} loom_target_codegen_format_t;

typedef enum loom_target_artifact_format_e {
  LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN = 0,
  LOOM_TARGET_ARTIFACT_FORMAT_ELF = 1,
  LOOM_TARGET_ARTIFACT_FORMAT_COFF = 2,
  LOOM_TARGET_ARTIFACT_FORMAT_MACHO = 3,
  LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY = 4,
  LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE = 5,
} loom_target_artifact_format_t;

typedef enum loom_target_abi_kind_e {
  LOOM_TARGET_ABI_UNKNOWN = 0,
  LOOM_TARGET_ABI_OBJECT_FUNCTION = 1,
  LOOM_TARGET_ABI_HAL_KERNEL = 2,
  LOOM_TARGET_ABI_VM_MODULE_FUNCTION = 3,
  LOOM_TARGET_ABI_SHADER_ENTRY_POINT = 4,
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
  // Address space assignments used by target-specific ABI lowering.
  loom_target_memory_space_map_t memory_spaces;
} loom_target_snapshot_t;

typedef struct loom_target_hal_kernel_abi_t {
  // ABI-required byte alignment for binding pointer parameters.
  uint32_t binding_alignment;
  // ABI-required fixed workgroup size for each kernel entry point.
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
  // Source symbol selected for export, or empty for reusable fixture plans.
  iree_string_view_t source_symbol;
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

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_TYPES_H_
