// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/llvmir/target_env.h"

#include <stdio.h>

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

static const loom_llvmir_target_profile_t* const kBuiltinTargetProfiles[] = {
    &kX86_64ObjectProfile,
    &kAmdgpuHalProfile,
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

iree_status_t loom_llvmir_target_profile_lookup(
    iree_string_view_t profile_name,
    const loom_llvmir_target_profile_t** out_profile) {
  if (out_profile == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile output is required");
  }
  *out_profile = NULL;
  profile_name = iree_string_view_trim(profile_name);
  if (iree_string_view_is_empty(profile_name)) {
    *out_profile = loom_llvmir_target_profile_x86_64_object();
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kBuiltinTargetProfiles);
       ++i) {
    const loom_llvmir_target_profile_t* profile = kBuiltinTargetProfiles[i];
    if (iree_string_view_equal(profile_name, profile->name)) {
      *out_profile = profile;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown LLVMIR target profile '%.*s'",
                          (int)profile_name.size, profile_name.data);
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

iree_status_t loom_llvmir_target_profile_initialize_amdgpu_hal(
    loom_llvmir_target_profile_t* out_profile) {
  if (out_profile == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile output is required");
  }
  *out_profile = kAmdgpuHalProfile;
  return iree_ok_status();
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

iree_status_t loom_llvmir_target_profile_kernel_binding_attrs(
    const loom_llvmir_target_profile_t* profile, loom_llvmir_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* out_attr_count) {
  if (out_attr_count == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM binding attr count output is required");
  }
  *out_attr_count = 0;
  if (profile == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile is required");
  }
  if (attrs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM binding attr storage is required");
  }
  if (profile->kind != LOOM_LLVMIR_TARGET_PROFILE_HAL_KERNEL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile is not a HAL kernel profile");
  }
  if (attr_capacity <
      LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM binding attr storage is too small");
  }
  if (profile->amdgpu_hal.binding_alignment == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL binding alignment must be non-zero");
  }

  attrs[0] = (loom_llvmir_attr_t){
      .kind = LOOM_LLVMIR_ATTR_INREG,
      .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
  };
  attrs[1] = (loom_llvmir_attr_t){
      .kind = LOOM_LLVMIR_ATTR_NOALIAS,
      .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
  };
  attrs[2] = (loom_llvmir_attr_t){
      .kind = LOOM_LLVMIR_ATTR_NOUNDEF,
      .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
  };
  attrs[3] = (loom_llvmir_attr_t){
      .kind = LOOM_LLVMIR_ATTR_NONNULL,
      .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
  };
  attrs[4] = (loom_llvmir_attr_t){
      .kind = LOOM_LLVMIR_ATTR_ALIGN,
      .value = profile->amdgpu_hal.binding_alignment,
      .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
  };
  *out_attr_count = LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_target_profile_flat_workgroup_size_attr(
    const loom_llvmir_target_profile_t* profile, char* storage,
    iree_host_size_t storage_capacity, iree_string_view_t* out_value) {
  if (profile->amdgpu_hal.flat_workgroup_size_min == 0 ||
      profile->amdgpu_hal.flat_workgroup_size_max == 0 ||
      profile->amdgpu_hal.flat_workgroup_size_min >
          profile->amdgpu_hal.flat_workgroup_size_max) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL flat workgroup size range must be non-zero and ordered");
  }
  int length = snprintf(storage, storage_capacity, "%u,%u",
                        profile->amdgpu_hal.flat_workgroup_size_min,
                        profile->amdgpu_hal.flat_workgroup_size_max);
  if (length <= 0 || (iree_host_size_t)length >= storage_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HAL flat workgroup size attr overflow");
  }
  *out_value = iree_make_string_view(storage, (iree_host_size_t)length);
  return iree_ok_status();
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
  if (profile->kind != LOOM_LLVMIR_TARGET_PROFILE_HAL_KERNEL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile is not a HAL kernel profile");
  }
  char flat_workgroup_size_storage[32];
  iree_string_view_t flat_workgroup_size = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_flat_workgroup_size_attr(
      profile, flat_workgroup_size_storage,
      IREE_ARRAYSIZE(flat_workgroup_size_storage), &flat_workgroup_size));
  loom_llvmir_attr_t attrs[] = {
      {
          .kind = LOOM_LLVMIR_ATTR_ALWAYSINLINE,
          .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
      },
      {
          .kind = LOOM_LLVMIR_ATTR_STRING_KEY_VALUE,
          .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
          .key = IREE_SVL("amdgpu-flat-work-group-size"),
          .string_value = flat_workgroup_size,
      },
      {
          .kind = LOOM_LLVMIR_ATTR_STRING_KEY,
          .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
          .key = IREE_SVL("uniform-work-group-size"),
      },
  };
  return loom_llvmir_module_add_attr_group(module, attrs, IREE_ARRAYSIZE(attrs),
                                           out_group_id);
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
  if (profile->amdgpu_hal.required_workgroup_size.x == 0 ||
      profile->amdgpu_hal.required_workgroup_size.y == 0 ||
      profile->amdgpu_hal.required_workgroup_size.z == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LLVM target profile workgroup size dimensions must be non-zero");
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
