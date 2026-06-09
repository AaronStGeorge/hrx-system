// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/target_env.h"

#include <stdio.h>

#include "loom/target/launch.h"

void loom_llvmir_target_env_module_config(
    const loom_llvmir_target_env_t* target_env, iree_string_view_t source_name,
    loom_llvmir_target_config_t* out_config) {
  *out_config = (loom_llvmir_target_config_t){0};
  *out_config = (loom_llvmir_target_config_t){
      .source_name = source_name,
      .target_triple = target_env->target_triple,
      .data_layout = target_env->data_layout,
      .default_pointer_bitwidth = target_env->default_pointer_bitwidth,
      .index_bitwidth = target_env->index_bitwidth,
      .offset_bitwidth = target_env->offset_bitwidth,
  };
}

void loom_llvmir_target_profile_module_config(
    const loom_llvmir_target_profile_t* profile, iree_string_view_t source_name,
    loom_llvmir_target_config_t* out_config) {
  loom_llvmir_target_env_module_config(profile->target_env, source_name,
                                       out_config);
}

static loom_llvmir_object_format_t loom_llvmir_object_format_from_artifact(
    loom_target_artifact_format_t artifact_format) {
  switch (artifact_format) {
    case LOOM_TARGET_ARTIFACT_FORMAT_ELF:
      return LOOM_LLVMIR_OBJECT_FORMAT_ELF;
    case LOOM_TARGET_ARTIFACT_FORMAT_COFF:
      return LOOM_LLVMIR_OBJECT_FORMAT_COFF;
    case LOOM_TARGET_ARTIFACT_FORMAT_MACHO:
      return LOOM_LLVMIR_OBJECT_FORMAT_MACHO;
    case LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN:
    case LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY:
    case LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE:
    case LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY:
      return LOOM_LLVMIR_OBJECT_FORMAT_UNKNOWN;
  }
  return LOOM_LLVMIR_OBJECT_FORMAT_UNKNOWN;
}

static loom_llvmir_linkage_t loom_llvmir_linkage_from_target(
    loom_target_linkage_t linkage) {
  switch (linkage) {
    case LOOM_TARGET_LINKAGE_DEFAULT:
      return LOOM_LLVMIR_LINKAGE_DEFAULT;
    case LOOM_TARGET_LINKAGE_DSO_LOCAL:
      return LOOM_LLVMIR_LINKAGE_DSO_LOCAL;
  }
  return LOOM_LLVMIR_LINKAGE_DEFAULT;
}

static void loom_llvmir_resolve_hal_kernel_flat_workgroup_range(
    const loom_target_snapshot_t* snapshot,
    const loom_target_hal_kernel_abi_t* hal_kernel, uint32_t* out_min,
    uint32_t* out_max) {
  const uint32_t flat_min = hal_kernel->flat_workgroup_size_min;
  const uint32_t flat_max = hal_kernel->flat_workgroup_size_max;
  if (flat_min != 0) {
    *out_min = flat_min;
    *out_max = flat_max;
    return;
  }
  *out_min = 1;
  *out_max = snapshot->max_flat_workgroup_size;
}

void loom_llvmir_target_profile_storage_initialize_from_bundle(
    const loom_target_bundle_t* bundle,
    const loom_llvmir_target_profile_t* projected_profile,
    loom_llvmir_target_profile_storage_t* out_storage) {
  *out_storage = (loom_llvmir_target_profile_storage_t){0};
  const loom_target_snapshot_t* snapshot = bundle->snapshot;
  const loom_target_export_plan_t* export_plan = bundle->export_plan;

  out_storage->target_env = (loom_llvmir_target_env_t){
      .name = snapshot->name,
      .target_triple = projected_profile->target_env->target_triple,
      .data_layout = projected_profile->target_env->data_layout,
      .object_format =
          loom_llvmir_object_format_from_artifact(snapshot->artifact_format),
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
      .name = projected_profile->name,
      .target_env = &out_storage->target_env,
      .target_cpu = projected_profile->target_cpu,
      .target_features = projected_profile->target_features,
      .x86_packed_dot_feature_bits =
          projected_profile->x86_packed_dot_feature_bits,
      .exported_linkage = loom_llvmir_linkage_from_target(export_plan->linkage),
      .kernel_calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
  };

  if (export_plan->abi_kind == LOOM_TARGET_ABI_OBJECT_FUNCTION) {
    out_storage->profile.kind = LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT;
    return;
  }
  if (export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL) {
    uint32_t flat_workgroup_size_min = 0;
    uint32_t flat_workgroup_size_max = 0;
    loom_llvmir_resolve_hal_kernel_flat_workgroup_range(
        snapshot, &export_plan->hal_kernel, &flat_workgroup_size_min,
        &flat_workgroup_size_max);
    out_storage->profile.kind = LOOM_LLVMIR_TARGET_PROFILE_HAL_KERNEL;
    out_storage->profile.kernel_calling_convention =
        projected_profile->kernel_calling_convention;
    out_storage->profile.required_workgroup_size_metadata_name =
        projected_profile->required_workgroup_size_metadata_name;
    out_storage->profile.amdgpu_hal = (loom_llvmir_amdgpu_hal_abi_t){
        .required_workgroup_size =
            {
                .x = export_plan->hal_kernel.required_workgroup_size.x,
                .y = export_plan->hal_kernel.required_workgroup_size.y,
                .z = export_plan->hal_kernel.required_workgroup_size.z,
            },
        .flat_workgroup_size_min = flat_workgroup_size_min,
        .flat_workgroup_size_max = flat_workgroup_size_max,
        .buffer_resource_flags = export_plan->hal_kernel.buffer_resource_flags,
    };
  }
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
  *out_arguments = (loom_llvmir_target_profile_llc_arguments_t){0};
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_append_llc_argument(
      IREE_SV("-mtriple="), profile->target_env->target_triple, out_arguments));
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_append_llc_argument(
      IREE_SV("-mcpu="), profile->target_cpu, out_arguments));
  IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_append_llc_argument(
      IREE_SV("-mattr="), profile->target_features, out_arguments));
  return iree_ok_status();
}

void loom_llvmir_target_profile_kernel_binding_attrs(
    const loom_llvmir_target_profile_t* profile, loom_llvmir_attr_t* attrs,
    iree_host_size_t* out_attr_count) {
  IREE_ASSERT_ARGUMENT(profile);
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_ASSERT_ARGUMENT(out_attr_count);
  *out_attr_count = 0;
  attrs[0] = (loom_llvmir_attr_t){
      .kind = LOOM_LLVMIR_ATTR_INREG,
      .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
  };
  attrs[1] = (loom_llvmir_attr_t){
      .kind = LOOM_LLVMIR_ATTR_NOUNDEF,
      .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
  };
  *out_attr_count = 2;
}

static iree_string_view_t loom_llvmir_target_profile_flat_workgroup_size_attr(
    const loom_llvmir_target_profile_t* profile, char* storage,
    iree_host_size_t storage_capacity) {
  int length = snprintf(storage, storage_capacity, "%u,%u",
                        profile->amdgpu_hal.flat_workgroup_size_min,
                        profile->amdgpu_hal.flat_workgroup_size_max);
  return iree_make_string_view(storage, (iree_host_size_t)length);
}

iree_status_t loom_llvmir_target_profile_add_kernel_attr_group(
    loom_llvmir_module_t* module, const loom_llvmir_target_profile_t* profile,
    loom_llvmir_attr_group_id_t* out_group_id) {
  *out_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  char flat_workgroup_size_storage[32];
  iree_string_view_t flat_workgroup_size =
      loom_llvmir_target_profile_flat_workgroup_size_attr(
          profile, flat_workgroup_size_storage,
          IREE_ARRAYSIZE(flat_workgroup_size_storage));
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
  const loom_target_workgroup_size_t required_workgroup_size = {
      .x = profile->amdgpu_hal.required_workgroup_size.x,
      .y = profile->amdgpu_hal.required_workgroup_size.y,
      .z = profile->amdgpu_hal.required_workgroup_size.z,
  };
  if (loom_target_workgroup_size_is_empty(&required_workgroup_size)) {
    return iree_ok_status();
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
