// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/target_env.h"

#include "loom/target/llvmir/types.h"

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

static const loom_llvmir_target_env_t kAmdgcnAmdAmdhsaTargetEnv = {
    .name = IREE_SVL("amdgcn-amd-amdhsa"),
    .target_triple = IREE_SVL("amdgcn-amd-amdhsa"),
    .data_layout =
        IREE_SVL("e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-"
                 "p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:"
                 "256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:"
                 "256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-"
                 "A5-G1-ni:7:8:9"),
    .object_format = LOOM_LLVMIR_OBJECT_FORMAT_ELF,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 32,
    .offset_bitwidth = 64,
    .address_spaces =
        {
            .generic = 0,
            .global = 1,
            .local = 3,
            .constant = 4,
            .private_memory = 5,
            .buffer_resource = 7,
        },
};

static const loom_llvmir_attr_t kAmdgpuHalKernelBindingAttrs[] = {
    {
        .kind = LOOM_LLVMIR_ATTR_INREG,
        .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
    },
    {
        .kind = LOOM_LLVMIR_ATTR_NOALIAS,
        .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
    },
    {
        .kind = LOOM_LLVMIR_ATTR_NOUNDEF,
        .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
    },
    {
        .kind = LOOM_LLVMIR_ATTR_NONNULL,
        .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
    },
    {
        .kind = LOOM_LLVMIR_ATTR_ALIGN,
        .value = 16,
        .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
    },
};

static const loom_llvmir_attr_t kAmdgpuHalKernelFunctionAttrs[] = {
    {
        .kind = LOOM_LLVMIR_ATTR_ALWAYSINLINE,
        .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
    },
    {
        .kind = LOOM_LLVMIR_ATTR_STRING_KEY_VALUE,
        .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
        .key = IREE_SVL("amdgpu-flat-work-group-size"),
        .string_value = IREE_SVL("64,64"),
    },
    {
        .kind = LOOM_LLVMIR_ATTR_STRING_KEY,
        .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
        .key = IREE_SVL("uniform-work-group-size"),
    },
};

static const loom_llvmir_target_profile_t kX86_64ObjectProfile = {
    .name = IREE_SVL("x86_64-object"),
    .target_env = &kX86_64UnknownLinuxGnuTargetEnv,
    .kind = LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT,
    .exported_linkage = LOOM_LLVMIR_LINKAGE_DSO_LOCAL,
    .kernel_calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
};

static const loom_llvmir_target_profile_t kAmdgpuHalProfile = {
    .name = IREE_SVL("amdgpu-hal"),
    .target_env = &kAmdgcnAmdAmdhsaTargetEnv,
    .kind = LOOM_LLVMIR_TARGET_PROFILE_HAL_KERNEL,
    .exported_linkage = LOOM_LLVMIR_LINKAGE_DEFAULT,
    .kernel_calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_AMDGPU_KERNEL,
    .kernel_binding_attrs = kAmdgpuHalKernelBindingAttrs,
    .kernel_binding_attr_count = IREE_ARRAYSIZE(kAmdgpuHalKernelBindingAttrs),
    .kernel_function_attrs = kAmdgpuHalKernelFunctionAttrs,
    .kernel_function_attr_count = IREE_ARRAYSIZE(kAmdgpuHalKernelFunctionAttrs),
    .required_workgroup_size_metadata_name = IREE_SVL("reqd_work_group_size"),
    .amdgpu_hal =
        {
            .binding_alignment = 16,
            .required_workgroup_size = {.x = 64, .y = 1, .z = 1},
            .flat_workgroup_size_min = 64,
            .flat_workgroup_size_max = 64,
            .buffer_resource_flags = 159744,
        },
};

const loom_llvmir_target_env_t* loom_llvmir_target_env_x86_64_unknown_linux_gnu(
    void) {
  return &kX86_64UnknownLinuxGnuTargetEnv;
}

const loom_llvmir_target_env_t* loom_llvmir_target_env_amdgcn_amd_amdhsa(void) {
  return &kAmdgcnAmdAmdhsaTargetEnv;
}

iree_status_t loom_llvmir_target_env_module_config(
    const loom_llvmir_target_env_t* target_env, iree_string_view_t source_name,
    loom_llvmir_target_config_t* out_config) {
  if (out_config == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target config output is required");
  }
  *out_config = (loom_llvmir_target_config_t){0};
  if (target_env == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target environment is required");
  }
  *out_config = (loom_llvmir_target_config_t){
      .source_name = source_name,
      .target_triple = target_env->target_triple,
      .data_layout = target_env->data_layout,
      .default_pointer_bitwidth = target_env->default_pointer_bitwidth,
      .index_bitwidth = target_env->index_bitwidth,
      .offset_bitwidth = target_env->offset_bitwidth,
  };
  return iree_ok_status();
}

const loom_llvmir_target_profile_t* loom_llvmir_target_profile_x86_64_object(
    void) {
  return &kX86_64ObjectProfile;
}

const loom_llvmir_target_profile_t* loom_llvmir_target_profile_amdgpu_hal(
    void) {
  return &kAmdgpuHalProfile;
}

iree_status_t loom_llvmir_target_profile_module_config(
    const loom_llvmir_target_profile_t* profile, iree_string_view_t source_name,
    loom_llvmir_target_config_t* out_config) {
  if (profile == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile is required");
  }
  return loom_llvmir_target_env_module_config(profile->target_env, source_name,
                                              out_config);
}

iree_status_t loom_llvmir_target_profile_add_kernel_attr_group(
    loom_llvmir_module_t* module, const loom_llvmir_target_profile_t* profile,
    loom_llvmir_attr_group_id_t* out_group_id) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_group_id);
  *out_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  if (profile == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile is required");
  }
  if (profile->kernel_function_attr_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile has no kernel attrs");
  }
  return loom_llvmir_module_add_attr_group(
      module, profile->kernel_function_attrs,
      profile->kernel_function_attr_count, out_group_id);
}

iree_status_t loom_llvmir_target_profile_attach_kernel_metadata(
    loom_llvmir_function_t* function,
    const loom_llvmir_target_profile_t* profile) {
  IREE_ASSERT_ARGUMENT(function);
  if (profile == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile is required");
  }
  if (iree_string_view_is_empty(
          profile->required_workgroup_size_metadata_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM target profile has no required workgroup size metadata");
  }
  if (profile->amdgpu_hal.required_workgroup_size.x > INT32_MAX ||
      profile->amdgpu_hal.required_workgroup_size.y > INT32_MAX ||
      profile->amdgpu_hal.required_workgroup_size.z > INT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "LLVM target profile workgroup size exceeds metadata range");
  }
  int32_t workgroup_size_values[] = {
      (int32_t)profile->amdgpu_hal.required_workgroup_size.x,
      (int32_t)profile->amdgpu_hal.required_workgroup_size.y,
      (int32_t)profile->amdgpu_hal.required_workgroup_size.z,
  };
  loom_llvmir_metadata_id_t workgroup_size = LOOM_LLVMIR_METADATA_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_metadata_i32_tuple(
      function->module,
      &(loom_llvmir_metadata_i32_tuple_t){
          .values = workgroup_size_values,
          .value_count = IREE_ARRAYSIZE(workgroup_size_values),
      },
      &workgroup_size));
  return loom_llvmir_function_add_metadata_attachment(
      function, &(loom_llvmir_metadata_attachment_t){
                    .name = profile->required_workgroup_size_metadata_name,
                    .metadata_id = workgroup_size,
                });
}
