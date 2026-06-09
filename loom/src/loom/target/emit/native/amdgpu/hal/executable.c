// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/hal/executable.h"

#include <string.h>

#include "iree/base/internal/flatcc/building.h"
#include "iree/schemas/amdgpu_executable_def_builder.h"

static const loom_amdgpu_hal_kernel_binding_flags_t
    LOOM_AMDGPU_HAL_KERNEL_BINDING_KNOWN_FLAGS =
        LOOM_AMDGPU_HAL_KERNEL_BINDING_READ_ONLY |
        LOOM_AMDGPU_HAL_KERNEL_BINDING_INDIRECT;

static iree_status_t loom_amdgpu_hal_executable_validate(
    iree_string_view_t isa, iree_const_byte_span_t hsaco,
    const loom_amdgpu_hal_kernel_export_t* exports,
    iree_host_size_t export_count) {
  if (iree_string_view_is_empty(isa)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL executable ISA is required");
  }
  if (!hsaco.data || hsaco.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL executable HSACO image is required");
  }
  if (!exports || export_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL executable requires at least one "
                            "exported kernel");
  }

  for (iree_host_size_t i = 0; i < export_count; ++i) {
    const loom_amdgpu_hal_kernel_export_t* export = &exports[i];
    if (iree_string_view_is_empty(export->symbol_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU HAL executable export %" PRIhsz
                              " symbol name is required",
                              i);
    }
    if (export->workgroup_size.x == 0 || export->workgroup_size.y == 0 ||
        export->workgroup_size.z == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HAL executable export '%.*s' workgroup size must be nonzero",
          (int)export->symbol_name.size, export->symbol_name.data);
    }
    if (!export->binding_flags && export->binding_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU HAL executable export '%.*s' binding flags are required",
          (int)export->symbol_name.size, export->symbol_name.data);
    }
    for (iree_host_size_t j = 0; j < export->binding_count; ++j) {
      const loom_amdgpu_hal_kernel_binding_flags_t unknown_flags =
          export->binding_flags[j] &
          ~LOOM_AMDGPU_HAL_KERNEL_BINDING_KNOWN_FLAGS;
      if (unknown_flags != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU HAL executable export '%.*s' binding %" PRIhsz
            " has unknown flags 0x%" PRIx64,
            (int)export->symbol_name.size, export->symbol_name.data, j,
            unknown_flags);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_executable_build_flatbuffer(
    flatcc_builder_t* builder, iree_string_view_t isa,
    iree_const_byte_span_t hsaco,
    const loom_amdgpu_hal_kernel_export_t* exports,
    iree_host_size_t export_count, iree_allocator_t allocator,
    uint8_t** out_flatbuffer_data, iree_host_size_t* out_flatbuffer_length) {
  *out_flatbuffer_data = NULL;
  *out_flatbuffer_length = 0;

  flatbuffers_string_ref_t isa_ref =
      flatbuffers_string_create(builder, isa.data, isa.size);
  if (!isa_ref) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build AMDGPU executable ISA string");
  }

  flatbuffers_string_ref_t hsaco_ref = flatbuffers_string_create(
      builder, (const char*)hsaco.data, hsaco.data_length);
  if (!hsaco_ref) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build AMDGPU executable HSACO image");
  }
  iree_hal_amdgpu_ModuleDef_ref_t module_ref =
      iree_hal_amdgpu_ModuleDef_create(builder, hsaco_ref);
  if (!module_ref) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build AMDGPU executable module");
  }
  iree_hal_amdgpu_ModuleDef_vec_ref_t modules_ref =
      iree_hal_amdgpu_ModuleDef_vec_create(builder, &module_ref, 1);
  if (!modules_ref) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build AMDGPU executable module vector");
  }

  iree_hal_amdgpu_ExportDef_ref_t* export_refs = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, export_count, sizeof(*export_refs), (void**)&export_refs));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < export_count;
       ++i) {
    const loom_amdgpu_hal_kernel_export_t* export = &exports[i];
    flatbuffers_string_ref_t symbol_ref = flatbuffers_string_create(
        builder, export->symbol_name.data, export->symbol_name.size);
    if (!symbol_ref) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to build AMDGPU export symbol string");
      break;
    }

    iree_hal_amdgpu_Dims_t workgroup_size = {0};
    iree_hal_amdgpu_Dims_assign(&workgroup_size, export->workgroup_size.x,
                                export->workgroup_size.y,
                                export->workgroup_size.z);

    iree_hal_amdgpu_BindingBits_vec_ref_t binding_flags_ref = 0;
    if (export->binding_count != 0) {
      binding_flags_ref = iree_hal_amdgpu_BindingBits_vec_create(
          builder,
          (const iree_hal_amdgpu_BindingBits_enum_t*)export->binding_flags,
          export->binding_count);
      if (!binding_flags_ref) {
        status = iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "failed to build AMDGPU export binding flags vector");
        break;
      }
    }

    if (iree_hal_amdgpu_ExportDef_start(builder) ||
        iree_hal_amdgpu_ExportDef_symbol_name_add(builder, symbol_ref) ||
        iree_hal_amdgpu_ExportDef_workgroup_size_add(builder,
                                                     &workgroup_size) ||
        iree_hal_amdgpu_ExportDef_constant_count_add(builder,
                                                     export->constant_count)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to build AMDGPU executable export "
                                "fields");
      break;
    }
    if (binding_flags_ref && iree_hal_amdgpu_ExportDef_binding_flags_add(
                                 builder, binding_flags_ref)) {
      status = iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "failed to build AMDGPU executable binding flags field");
      break;
    }
    export_refs[i] = iree_hal_amdgpu_ExportDef_end(builder);
    if (!export_refs[i]) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to build AMDGPU executable export");
    }
  }

  iree_hal_amdgpu_ExportDef_vec_ref_t exports_ref = 0;
  if (iree_status_is_ok(status)) {
    exports_ref = iree_hal_amdgpu_ExportDef_vec_create(builder, export_refs,
                                                       export_count);
    if (!exports_ref) {
      status =
          iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                           "failed to build AMDGPU executable export vector");
    }
  }
  iree_allocator_free(allocator, export_refs);
  IREE_RETURN_IF_ERROR(status);

  if (iree_hal_amdgpu_ExecutableDef_start_as_root(builder) ||
      iree_hal_amdgpu_ExecutableDef_isa_add(builder, isa_ref) ||
      iree_hal_amdgpu_ExecutableDef_exports_add(builder, exports_ref) ||
      iree_hal_amdgpu_ExecutableDef_modules_add(builder, modules_ref)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build AMDGPU executable fields");
  }
  flatbuffers_ref_t root_ref =
      iree_hal_amdgpu_ExecutableDef_end_as_root(builder);
  if (!root_ref) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build AMDGPU executable FlatBuffer");
  }

  size_t flatbuffer_length = 0;
  void* flatbuffer_data =
      flatcc_builder_finalize_buffer(builder, &flatbuffer_length);
  if (!flatbuffer_data || flatbuffer_length > IREE_HOST_SIZE_MAX) {
    flatcc_builder_free(flatbuffer_data);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to finalize AMDGPU executable FlatBuffer");
  }

  *out_flatbuffer_data = flatbuffer_data;
  *out_flatbuffer_length = (iree_host_size_t)flatbuffer_length;
  return iree_ok_status();
}

void loom_amdgpu_hal_executable_deinitialize(
    loom_amdgpu_hal_executable_t* executable, iree_allocator_t allocator) {
  if (!executable) {
    return;
  }
  iree_allocator_free(allocator, (void*)executable->executable_format.data);
  iree_allocator_free(allocator, executable->data);
  *executable = (loom_amdgpu_hal_executable_t){0};
}

iree_status_t loom_amdgpu_package_hal_executable(
    iree_string_view_t isa, iree_const_byte_span_t hsaco,
    const loom_amdgpu_hal_kernel_export_t* exports,
    iree_host_size_t export_count, iree_allocator_t allocator,
    loom_amdgpu_hal_executable_t* out_executable) {
  *out_executable = (loom_amdgpu_hal_executable_t){0};

  iree_status_t status =
      loom_amdgpu_hal_executable_validate(isa, hsaco, exports, export_count);

  flatcc_builder_t builder;
  bool builder_initialized = false;
  uint8_t* flatbuffer_data = NULL;
  iree_host_size_t flatbuffer_length = 0;
  if (iree_status_is_ok(status)) {
    if (flatcc_builder_init(&builder) != 0) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to initialize FlatBuffer builder");
    } else {
      builder_initialized = true;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_hal_executable_build_flatbuffer(
        &builder, isa, hsaco, exports, export_count, allocator,
        &flatbuffer_data, &flatbuffer_length);
  }

  if (iree_status_is_ok(status)) {
    iree_host_size_t executable_length = 0;
    if (!iree_host_size_checked_add(sizeof(iree_flatbuffer_file_header_t),
                                    flatbuffer_length, &executable_length)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "AMDGPU HAL executable container size exceeds host size");
    } else {
      status = iree_allocator_malloc(allocator, executable_length,
                                     (void**)&out_executable->data);
      if (iree_status_is_ok(status)) {
        iree_flatbuffer_file_header_t header = {0};
        memcpy(&header.magic, iree_hal_amdgpu_ExecutableDef_file_identifier,
               sizeof(header.magic));
        header.version = 0;
        header.content_size = flatbuffer_length;

        memcpy(out_executable->data, &header, sizeof(header));
        memcpy(out_executable->data + sizeof(header), flatbuffer_data,
               flatbuffer_length);
        out_executable->data_length = executable_length;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    void* executable_format_data = NULL;
    status = iree_allocator_clone(allocator,
                                  iree_make_const_byte_span(isa.data, isa.size),
                                  &executable_format_data);
    if (iree_status_is_ok(status)) {
      out_executable->executable_format =
          iree_make_string_view(executable_format_data, isa.size);
    }
  }

  flatcc_builder_free(flatbuffer_data);
  if (builder_initialized) {
    flatcc_builder_clear(&builder);
  }
  if (!iree_status_is_ok(status)) {
    loom_amdgpu_hal_executable_deinitialize(out_executable, allocator);
  }
  return status;
}
