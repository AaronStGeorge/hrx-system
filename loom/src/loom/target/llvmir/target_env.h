// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVM target environment descriptions.
//
// Target environments collect the pieces of target state that should be chosen
// by target/ABI selection, not by the generic LLVM IR module builder. The
// module stores a copied target_config snapshot; these descriptions are the
// reusable source data used while constructing that snapshot and
// target-specific pointer/address-space types.

#ifndef LOOM_TARGET_LLVMIR_TARGET_ENV_H_
#define LOOM_TARGET_LLVMIR_TARGET_ENV_H_

#include "loom/target/llvmir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_llvmir_object_format_e {
  LOOM_LLVMIR_OBJECT_FORMAT_UNKNOWN = 0,
  LOOM_LLVMIR_OBJECT_FORMAT_ELF = 1,
  LOOM_LLVMIR_OBJECT_FORMAT_COFF = 2,
  LOOM_LLVMIR_OBJECT_FORMAT_MACHO = 3,
} loom_llvmir_object_format_t;

typedef enum loom_llvmir_target_profile_kind_e {
  LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT = 0,
  LOOM_LLVMIR_TARGET_PROFILE_HAL_KERNEL = 1,
} loom_llvmir_target_profile_kind_t;

enum {
  LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT = 5,
};

typedef struct loom_llvmir_target_address_spaces_t {
  // Generic/default pointer address space.
  uint32_t generic;
  // Device or process global memory address space.
  uint32_t global;
  // Workgroup/shared memory address space.
  uint32_t local;
  // Constant memory address space.
  uint32_t constant;
  // Per-invocation/private memory address space.
  uint32_t private_memory;
  // Target-specific buffer-resource pointer address space, or UINT32_MAX.
  uint32_t buffer_resource;
} loom_llvmir_target_address_spaces_t;

typedef struct loom_llvmir_workgroup_size_t {
  // Workgroup size along the x dimension.
  uint32_t x;
  // Workgroup size along the y dimension.
  uint32_t y;
  // Workgroup size along the z dimension.
  uint32_t z;
} loom_llvmir_workgroup_size_t;

typedef struct loom_llvmir_target_env_t {
  // Stable target environment name for diagnostics and tests.
  iree_string_view_t name;
  // LLVM target triple.
  iree_string_view_t target_triple;
  // LLVM data layout, or empty when supplied by a downstream toolchain.
  iree_string_view_t data_layout;
  // Linkable object format produced by this target.
  loom_llvmir_object_format_t object_format;
  // Default pointer bit width for target-independent layout decisions.
  uint32_t default_pointer_bitwidth;
  // Index bit width chosen for lowered index values.
  uint32_t index_bitwidth;
  // Offset bit width chosen for lowered byte offsets.
  uint32_t offset_bitwidth;
  // Address space assignments used by target-specific ABI lowering.
  loom_llvmir_target_address_spaces_t address_spaces;
} loom_llvmir_target_env_t;

typedef struct loom_llvmir_amdgpu_hal_abi_t {
  // ABI-required byte alignment for HAL binding pointer parameters.
  uint32_t binding_alignment;
  // ABI-required fixed workgroup size attached to each kernel entry point.
  loom_llvmir_workgroup_size_t required_workgroup_size;
  // Optimization default lower flat workgroup size advertised to LLVM.
  uint32_t flat_workgroup_size_min;
  // Optimization default upper flat workgroup size advertised to LLVM.
  uint32_t flat_workgroup_size_max;
  // ABI-required raw buffer resource flags for global binding resources.
  uint32_t buffer_resource_flags;
} loom_llvmir_amdgpu_hal_abi_t;

typedef struct loom_llvmir_target_profile_t {
  // Stable target profile name for diagnostics and tests.
  iree_string_view_t name;
  // Target environment selected by this ABI profile.
  const loom_llvmir_target_env_t* target_env;
  // ABI family represented by this profile.
  loom_llvmir_target_profile_kind_t kind;
  // Optimization default target CPU, or empty when supplied per variant.
  iree_string_view_t target_cpu;
  // Optimization default target feature string, or empty when supplied per
  // variant.
  iree_string_view_t target_features;
  // ABI-required linkage for exported object functions or kernel entry points.
  loom_llvmir_linkage_t exported_linkage;
  // ABI-required calling convention for kernel entry points.
  loom_llvmir_calling_convention_t kernel_calling_convention;
  // ABI-required metadata attachment name for fixed workgroup size.
  iree_string_view_t required_workgroup_size_metadata_name;
  // AMDGPU HAL-specific ABI parameters.
  loom_llvmir_amdgpu_hal_abi_t amdgpu_hal;
} loom_llvmir_target_profile_t;

const loom_llvmir_target_env_t* loom_llvmir_target_env_x86_64_unknown_linux_gnu(
    void);

const loom_llvmir_target_env_t* loom_llvmir_target_env_amdgcn_amd_amdhsa(void);

const loom_llvmir_target_profile_t* loom_llvmir_target_profile_x86_64_object(
    void);

const loom_llvmir_target_profile_t* loom_llvmir_target_profile_amdgpu_hal(void);

// Looks up a built-in target profile by name. Empty names resolve to the
// default host object profile.
iree_status_t loom_llvmir_target_profile_lookup(
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile);

// Initializes |out_profile| with a shallow copy of the built-in x86_64 object
// profile. String views and |target_env| point at immutable static storage;
// callers may overwrite profile fields for one lowering invocation.
iree_status_t loom_llvmir_target_profile_initialize_x86_64_object(
    loom_llvmir_target_profile_t* out_profile);

// Initializes |out_profile| with a shallow copy of the built-in AMDGPU HAL
// profile. String views and |target_env| point at immutable static storage;
// callers may overwrite profile fields for one lowering invocation.
iree_status_t loom_llvmir_target_profile_initialize_amdgpu_hal(
    loom_llvmir_target_profile_t* out_profile);

iree_status_t loom_llvmir_target_env_module_config(
    const loom_llvmir_target_env_t* target_env, iree_string_view_t source_name,
    loom_llvmir_target_config_t* out_config);

iree_status_t loom_llvmir_target_profile_module_config(
    const loom_llvmir_target_profile_t* profile, iree_string_view_t source_name,
    loom_llvmir_target_config_t* out_config);

// Writes the ABI-required parameter attrs for a HAL kernel binding pointer.
// The caller provides temporary storage; loom_llvmir_function_add_parameter()
// copies the attrs into the target module when attaching them to a parameter.
iree_status_t loom_llvmir_target_profile_kernel_binding_attrs(
    const loom_llvmir_target_profile_t* profile, loom_llvmir_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* out_attr_count);

// Adds the ABI and optimization attr group for a HAL kernel entry point.
// Attribute values are derived from |profile| at materialization time so
// profile copies can safely customize workgroup-size policy.
iree_status_t loom_llvmir_target_profile_add_kernel_attr_group(
    loom_llvmir_module_t* module, const loom_llvmir_target_profile_t* profile,
    loom_llvmir_attr_group_id_t* out_group_id);

// Attaches the fixed-workgroup-size metadata required by a HAL kernel entry
// point.
iree_status_t loom_llvmir_target_profile_attach_kernel_metadata(
    loom_llvmir_function_t* function,
    const loom_llvmir_target_profile_t* profile);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_TARGET_ENV_H_
