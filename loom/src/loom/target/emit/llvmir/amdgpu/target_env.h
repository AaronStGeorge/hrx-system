// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVM AMDGPU target environment/profile presets and HAL ABI helpers.

#ifndef LOOM_TARGET_LLVMIR_AMDGPU_TARGET_ENV_H_
#define LOOM_TARGET_LLVMIR_AMDGPU_TARGET_ENV_H_

#include "loom/target/emit/llvmir/target_env.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_llvmir_target_env_t* loom_llvmir_target_env_amdgcn_amd_amdhsa(void);

const loom_llvmir_target_profile_t* loom_llvmir_target_profile_amdgpu_hal(void);

// Initializes |out_profile| with a shallow copy of the built-in AMDGPU HAL
// profile. String views and |target_env| point at immutable static storage;
// callers may overwrite profile fields for one lowering invocation.
iree_status_t loom_llvmir_target_profile_initialize_amdgpu_hal(
    loom_llvmir_target_profile_t* out_profile);

// Writes the ABI-required parameter attrs for a HAL kernel binding pointer.
// The caller provides temporary storage; loom_llvmir_function_add_parameter()
// copies the attrs into the target module when attaching them to a parameter.
iree_status_t loom_llvmir_target_profile_kernel_binding_attrs(
    const loom_llvmir_target_profile_t* profile, loom_llvmir_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* out_attr_count);

// Adds the ABI and optimization attr group for an AMDGPU HAL kernel entry
// point. Attribute values are derived from |profile| at materialization time so
// profile copies can safely customize workgroup-size policy.
iree_status_t loom_llvmir_target_profile_add_kernel_attr_group(
    loom_llvmir_module_t* module, const loom_llvmir_target_profile_t* profile,
    loom_llvmir_attr_group_id_t* out_group_id);

// Attaches the fixed-workgroup-size metadata required by an AMDGPU HAL kernel
// entry point.
iree_status_t loom_llvmir_target_profile_attach_kernel_metadata(
    loom_llvmir_function_t* function,
    const loom_llvmir_target_profile_t* profile);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_AMDGPU_TARGET_ENV_H_
