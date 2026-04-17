// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/target_env.h"

#include <limits.h>
#include <stdio.h>

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

static iree_status_t loom_llvmir_object_format_from_artifact(
    loom_target_artifact_format_t artifact_format,
    loom_llvmir_object_format_t* out_object_format) {
  switch (artifact_format) {
    case LOOM_TARGET_ARTIFACT_FORMAT_ELF:
      *out_object_format = LOOM_LLVMIR_OBJECT_FORMAT_ELF;
      return iree_ok_status();
    case LOOM_TARGET_ARTIFACT_FORMAT_COFF:
      *out_object_format = LOOM_LLVMIR_OBJECT_FORMAT_COFF;
      return iree_ok_status();
    case LOOM_TARGET_ARTIFACT_FORMAT_MACHO:
      *out_object_format = LOOM_LLVMIR_OBJECT_FORMAT_MACHO;
      return iree_ok_status();
    case LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN:
    case LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY:
    case LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target artifact format is not an LLVM object");
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown target artifact format");
}

static iree_status_t loom_llvmir_linkage_from_target(
    loom_target_linkage_t linkage, loom_llvmir_linkage_t* out_linkage) {
  switch (linkage) {
    case LOOM_TARGET_LINKAGE_DEFAULT:
      *out_linkage = LOOM_LLVMIR_LINKAGE_DEFAULT;
      return iree_ok_status();
    case LOOM_TARGET_LINKAGE_DSO_LOCAL:
      *out_linkage = LOOM_LLVMIR_LINKAGE_DSO_LOCAL;
      return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown target linkage");
}

static iree_status_t loom_llvmir_validate_target_snapshot(
    const loom_target_snapshot_t* snapshot) {
  if (snapshot == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target snapshot is required");
  }
  if (snapshot->codegen_format != LOOM_TARGET_CODEGEN_FORMAT_LLVMIR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target snapshot is not an LLVMIR codegen target");
  }
  if (snapshot->default_pointer_bitwidth == 0 ||
      snapshot->index_bitwidth == 0 || snapshot->offset_bitwidth == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target snapshot pointer, index, and offset bit "
                            "widths must be non-zero");
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_validate_hal_kernel_export_plan(
    const loom_target_snapshot_t* snapshot,
    const loom_target_export_plan_t* export_plan) {
  if (!iree_string_view_equal(snapshot->target_triple,
                              IREE_SV("amdgcn-amd-amdhsa"))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "LLVMIR HAL kernel profile derivation currently supports AMDGPU only");
  }
  if (export_plan->hal_kernel.binding_alignment == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL kernel binding alignment must be non-zero");
  }
  if (export_plan->hal_kernel.required_workgroup_size.x == 0 ||
      export_plan->hal_kernel.required_workgroup_size.y == 0 ||
      export_plan->hal_kernel.required_workgroup_size.z == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL kernel required workgroup dimensions must be non-zero");
  }
  if (export_plan->hal_kernel.flat_workgroup_size_min == 0 ||
      export_plan->hal_kernel.flat_workgroup_size_max == 0 ||
      export_plan->hal_kernel.flat_workgroup_size_min >
          export_plan->hal_kernel.flat_workgroup_size_max) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL kernel flat workgroup size range must be non-zero and ordered");
  }
  return iree_ok_status();
}

iree_status_t loom_llvmir_target_profile_storage_initialize_from_bundle(
    const loom_target_bundle_t* bundle,
    loom_llvmir_target_profile_storage_t* out_storage) {
  if (out_storage == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile storage output is required");
  }
  *out_storage = (loom_llvmir_target_profile_storage_t){0};
  if (bundle == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target bundle is required");
  }
  const loom_target_snapshot_t* snapshot = bundle->snapshot;
  const loom_target_export_plan_t* export_plan = bundle->export_plan;
  const loom_target_config_t* config = bundle->config;
  IREE_RETURN_IF_ERROR(loom_llvmir_validate_target_snapshot(snapshot));
  if (export_plan == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target export plan is required");
  }
  if (config == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target config is required");
  }

  loom_llvmir_object_format_t object_format = LOOM_LLVMIR_OBJECT_FORMAT_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_llvmir_object_format_from_artifact(
      snapshot->artifact_format, &object_format));
  loom_llvmir_linkage_t exported_linkage = LOOM_LLVMIR_LINKAGE_DEFAULT;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_linkage_from_target(export_plan->linkage, &exported_linkage));

  out_storage->target_env = (loom_llvmir_target_env_t){
      .name = snapshot->name,
      .target_triple = snapshot->target_triple,
      .data_layout = snapshot->data_layout,
      .object_format = object_format,
      .default_pointer_bitwidth = snapshot->default_pointer_bitwidth,
      .index_bitwidth = snapshot->index_bitwidth,
      .offset_bitwidth = snapshot->offset_bitwidth,
      .address_spaces =
          {
              .generic = snapshot->memory_spaces.generic,
              .global = snapshot->memory_spaces.global,
              .local = snapshot->memory_spaces.workgroup,
              .constant = snapshot->memory_spaces.constant,
              .private_memory = snapshot->memory_spaces.private_memory,
              .buffer_resource = snapshot->memory_spaces.descriptor,
          },
  };

  out_storage->profile = (loom_llvmir_target_profile_t){
      .name = bundle->name,
      .target_env = &out_storage->target_env,
      .target_cpu = snapshot->target_cpu,
      .target_features = snapshot->target_features,
      .x86_packed_dot_feature_bits = config->contract_feature_bits,
      .exported_linkage = exported_linkage,
      .kernel_calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
  };

  switch (export_plan->abi_kind) {
    case LOOM_TARGET_ABI_OBJECT_FUNCTION:
      out_storage->profile.kind = LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT;
      return iree_ok_status();
    case LOOM_TARGET_ABI_HAL_KERNEL: {
      IREE_RETURN_IF_ERROR(
          loom_llvmir_validate_hal_kernel_export_plan(snapshot, export_plan));
      out_storage->profile.kind = LOOM_LLVMIR_TARGET_PROFILE_HAL_KERNEL;
      out_storage->profile.kernel_calling_convention =
          LOOM_LLVMIR_CALLING_CONVENTION_AMDGPU_KERNEL;
      out_storage->profile.required_workgroup_size_metadata_name =
          IREE_SV("reqd_work_group_size");
      out_storage->profile.amdgpu_hal = (loom_llvmir_amdgpu_hal_abi_t){
          .binding_alignment = export_plan->hal_kernel.binding_alignment,
          .required_workgroup_size =
              {
                  .x = export_plan->hal_kernel.required_workgroup_size.x,
                  .y = export_plan->hal_kernel.required_workgroup_size.y,
                  .z = export_plan->hal_kernel.required_workgroup_size.z,
              },
          .flat_workgroup_size_min =
              export_plan->hal_kernel.flat_workgroup_size_min,
          .flat_workgroup_size_max =
              export_plan->hal_kernel.flat_workgroup_size_max,
          .buffer_resource_flags =
              export_plan->hal_kernel.buffer_resource_flags,
      };
      return iree_ok_status();
    }
    case LOOM_TARGET_ABI_UNKNOWN:
    case LOOM_TARGET_ABI_VM_MODULE_FUNCTION:
    case LOOM_TARGET_ABI_SHADER_ENTRY_POINT:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "target ABI kind does not have an LLVMIR profile adapter yet");
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown target ABI kind");
}

static iree_status_t loom_llvmir_target_profile_append_llc_argument(
    iree_string_view_t prefix, iree_string_view_t value,
    loom_llvmir_target_profile_llc_arguments_t* arguments) {
  if (iree_string_view_is_empty(value)) {
    return iree_ok_status();
  }
  if (arguments->count >= IREE_ARRAYSIZE(arguments->values)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LLVM llc argument storage is too small");
  }
  char* storage = arguments->storage[arguments->count];
  iree_host_size_t storage_length =
      IREE_ARRAYSIZE(arguments->storage[arguments->count]);
  int length = snprintf(storage, storage_length, "%.*s%.*s", (int)prefix.size,
                        prefix.data, (int)value.size, value.data);
  if (length <= 0 || (iree_host_size_t)length >= storage_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LLVM llc argument exceeds storage");
  }
  arguments->values[arguments->count] =
      iree_make_string_view(storage, (iree_host_size_t)length);
  ++arguments->count;
  return iree_ok_status();
}

iree_status_t loom_llvmir_target_profile_llc_arguments(
    const loom_llvmir_target_profile_t* profile,
    loom_llvmir_target_profile_llc_arguments_t* out_arguments) {
  if (out_arguments == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM llc argument output is required");
  }
  *out_arguments = (loom_llvmir_target_profile_llc_arguments_t){0};
  if (profile == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target profile is required");
  }
  if (profile->target_env == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LLVM target environment is required");
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_append_llc_argument(
      IREE_SV("-mtriple="), profile->target_env->target_triple, out_arguments));
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_append_llc_argument(
      IREE_SV("-mcpu="), profile->target_cpu, out_arguments));
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_append_llc_argument(
      IREE_SV("-mattr="), profile->target_features, out_arguments));
  return iree_ok_status();
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
      loom_llvmir_function_module(function),
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
