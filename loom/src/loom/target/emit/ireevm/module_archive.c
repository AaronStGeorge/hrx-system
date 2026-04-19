// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/module_archive.h"

#include <limits.h>
#include <string.h>

#include "iree/base/internal/flatcc/building.h"
#include "iree/schemas/bytecode_module_def_builder.h"
#include "iree/vm/bytecode/utils/isa.h"

typedef struct loom_ireevm_module_build_t {
  // FlatCC table references for each function signature.
  iree_vm_FunctionSignatureDef_ref_t* signature_references;
  // FlatCC table references for each exported function.
  iree_vm_ExportFunctionDef_ref_t* export_references;
  // FlatCC struct payloads for each function descriptor.
  iree_vm_FunctionDescriptor_t* descriptors;
  // Concatenated padded VM function bytecode data.
  uint8_t* bytecode_data;
  // Number of bytes in |bytecode_data|.
  iree_host_size_t bytecode_data_length;
} loom_ireevm_module_build_t;

static void loom_ireevm_module_build_deinitialize(
    loom_ireevm_module_build_t* build, iree_allocator_t allocator) {
  iree_allocator_free(allocator, build->signature_references);
  iree_allocator_free(allocator, build->export_references);
  iree_allocator_free(allocator, build->descriptors);
  iree_allocator_free(allocator, build->bytecode_data);
  *build = (loom_ireevm_module_build_t){0};
}

static iree_status_t loom_ireevm_module_archive_validate_inputs(
    iree_string_view_t module_name,
    const loom_ireevm_module_archive_function_t* functions,
    iree_host_size_t function_count, loom_ireevm_module_build_t* build) {
  if (iree_string_view_is_empty(module_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM module name is required");
  }
  if (!functions || function_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM module requires at least one function");
  }
  if (function_count > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "VM module function count exceeds i32");
  }

  iree_host_size_t bytecode_data_length = 0;
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    const loom_ireevm_module_archive_function_t* function = &functions[i];
    if (iree_string_view_is_empty(function->export_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM function export name is required");
    }
    if (iree_string_view_is_empty(function->calling_convention)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM function calling convention is required for '%.*s'",
          (int)function->export_name.size, function->export_name.data);
    }
    if (!function->bytecode || !function->bytecode->data ||
        function->bytecode->data_length == 0 ||
        function->bytecode->bytecode_length == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "VM function bytecode is required for '%.*s'",
                              (int)function->export_name.size,
                              function->export_name.data);
    }
    if (function->bytecode->block_count > INT16_MAX ||
        function->bytecode->i32_register_count > INT16_MAX ||
        function->bytecode->ref_register_count > INT16_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VM function descriptor counts exceed i16 for '%.*s'",
          (int)function->export_name.size, function->export_name.data);
    }
    if (function->bytecode->data_length > INT32_MAX ||
        function->bytecode->bytecode_length > INT32_MAX ||
        bytecode_data_length > INT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "VM function bytecode length exceeds i32 for '%.*s'",
          (int)function->export_name.size, function->export_name.data);
    }
    if (!iree_host_size_checked_add(bytecode_data_length,
                                    function->bytecode->data_length,
                                    &bytecode_data_length) ||
        bytecode_data_length > INT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "VM module bytecode data exceeds i32");
    }
  }
  build->bytecode_data_length = bytecode_data_length;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_module_build_allocate(
    iree_host_size_t function_count, iree_allocator_t allocator,
    loom_ireevm_module_build_t* build) {
  iree_status_t status = iree_allocator_malloc_array(
      allocator, function_count, sizeof(*build->signature_references),
      (void**)&build->signature_references);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, function_count,
                                         sizeof(*build->export_references),
                                         (void**)&build->export_references);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, function_count,
                                         sizeof(*build->descriptors),
                                         (void**)&build->descriptors);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, build->bytecode_data_length,
                                   (void**)&build->bytecode_data);
  }
  if (!iree_status_is_ok(status)) {
    loom_ireevm_module_build_deinitialize(build, allocator);
  }
  return status;
}

static iree_status_t loom_ireevm_module_build_populate(
    flatcc_builder_t* builder, iree_string_view_t module_name,
    const loom_ireevm_module_archive_function_t* functions,
    iree_host_size_t function_count, loom_ireevm_module_build_t* build,
    iree_allocator_t allocator, loom_ireevm_module_archive_t* out_archive) {
  iree_host_size_t bytecode_offset = 0;
  for (iree_host_size_t i = 0; i < function_count; ++i) {
    const loom_ireevm_module_archive_function_t* function = &functions[i];
    const loom_ireevm_function_bytecode_t* bytecode = function->bytecode;

    flatbuffers_string_ref_t calling_convention_ref =
        flatbuffers_string_create(builder, function->calling_convention.data,
                                  function->calling_convention.size);
    if (!calling_convention_ref) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to build VM calling convention string");
    }
    if (iree_vm_FunctionSignatureDef_start(builder) ||
        iree_vm_FunctionSignatureDef_calling_convention_add(
            builder, calling_convention_ref)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to start VM function signature");
    }
    build->signature_references[i] = iree_vm_FunctionSignatureDef_end(builder);
    if (!build->signature_references[i]) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to build VM function signature");
    }

    flatbuffers_string_ref_t export_name_ref = flatbuffers_string_create(
        builder, function->export_name.data, function->export_name.size);
    if (!export_name_ref) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to build VM export name string");
    }
    if (iree_vm_ExportFunctionDef_start(builder) ||
        iree_vm_ExportFunctionDef_local_name_add(builder, export_name_ref)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to start VM function export");
    }
    if (i != 0 &&
        iree_vm_ExportFunctionDef_internal_ordinal_add(builder, (int32_t)i)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to build VM function export ordinal");
    }
    build->export_references[i] = iree_vm_ExportFunctionDef_end(builder);
    if (!build->export_references[i]) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "failed to build VM function export");
    }

    iree_vm_FunctionDescriptor_assign(
        &build->descriptors[i], (int32_t)bytecode_offset,
        (int32_t)bytecode->bytecode_length, (iree_vm_FeatureBits_enum_t)0, 0,
        (int16_t)bytecode->block_count, (int16_t)bytecode->i32_register_count,
        (int16_t)bytecode->ref_register_count);
    memcpy(build->bytecode_data + bytecode_offset, bytecode->data,
           bytecode->data_length);
    bytecode_offset += bytecode->data_length;
  }
  IREE_ASSERT(bytecode_offset == build->bytecode_data_length);

  flatbuffers_string_ref_t module_name_ref =
      flatbuffers_string_create(builder, module_name.data, module_name.size);
  if (!module_name_ref) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build VM module name string");
  }
  iree_vm_ExportFunctionDef_vec_ref_t exports_reference =
      iree_vm_ExportFunctionDef_vec_create(builder, build->export_references,
                                           function_count);
  iree_vm_FunctionSignatureDef_vec_ref_t signatures_reference =
      iree_vm_FunctionSignatureDef_vec_create(
          builder, build->signature_references, function_count);
  iree_vm_FunctionDescriptor_vec_ref_t descriptors_reference =
      iree_vm_FunctionDescriptor_vec_create(builder, build->descriptors,
                                            function_count);
  flatbuffers_uint8_vec_ref_t bytecode_data_reference =
      flatbuffers_uint8_vec_create(builder, build->bytecode_data,
                                   build->bytecode_data_length);
  if (!exports_reference || !signatures_reference || !descriptors_reference ||
      !bytecode_data_reference) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build VM module vectors");
  }

  const uint32_t bytecode_version =
      ((uint32_t)IREE_VM_BYTECODE_VERSION_MAJOR << 16) |
      (uint32_t)IREE_VM_BYTECODE_VERSION_MINOR;
  if (iree_vm_BytecodeModuleDef_start_as_root_with_size(builder) ||
      iree_vm_BytecodeModuleDef_name_add(builder, module_name_ref) ||
      iree_vm_BytecodeModuleDef_exported_functions_add(builder,
                                                       exports_reference) ||
      iree_vm_BytecodeModuleDef_function_signatures_add(builder,
                                                        signatures_reference) ||
      iree_vm_BytecodeModuleDef_function_descriptors_add(
          builder, descriptors_reference) ||
      iree_vm_BytecodeModuleDef_bytecode_version_add(builder,
                                                     bytecode_version) ||
      iree_vm_BytecodeModuleDef_bytecode_data_add(builder,
                                                  bytecode_data_reference)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build VM module FlatBuffer fields");
  }
  flatbuffers_ref_t root_reference =
      iree_vm_BytecodeModuleDef_end_as_root(builder);
  if (!root_reference) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to build VM module FlatBuffer");
  }

  size_t flatbuffer_length = 0;
  void* flatbuffer_data =
      flatcc_builder_finalize_buffer(builder, &flatbuffer_length);
  if (!flatbuffer_data || flatbuffer_length > IREE_HOST_SIZE_MAX) {
    flatcc_builder_free(flatbuffer_data);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "failed to finalize VM module FlatBuffer");
  }

  iree_status_t status =
      iree_allocator_malloc(allocator, (iree_host_size_t)flatbuffer_length,
                            (void**)&out_archive->data);
  if (iree_status_is_ok(status)) {
    memcpy(out_archive->data, flatbuffer_data, flatbuffer_length);
    out_archive->data_length = (iree_host_size_t)flatbuffer_length;
  }
  flatcc_builder_free(flatbuffer_data);
  return status;
}

void loom_ireevm_module_archive_deinitialize(
    loom_ireevm_module_archive_t* archive, iree_allocator_t allocator) {
  if (!archive) {
    return;
  }
  iree_allocator_free(allocator, archive->data);
  *archive = (loom_ireevm_module_archive_t){0};
}

iree_status_t loom_ireevm_emit_module_archive(
    iree_string_view_t module_name,
    const loom_ireevm_module_archive_function_t* functions,
    iree_host_size_t function_count, iree_allocator_t allocator,
    loom_ireevm_module_archive_t* out_archive) {
  if (!out_archive) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "VM module archive output is required");
  }
  *out_archive = (loom_ireevm_module_archive_t){0};

  loom_ireevm_module_build_t build = {0};
  iree_status_t status = loom_ireevm_module_archive_validate_inputs(
      module_name, functions, function_count, &build);
  if (iree_status_is_ok(status)) {
    status =
        loom_ireevm_module_build_allocate(function_count, allocator, &build);
  }

  flatcc_builder_t builder;
  bool builder_initialized = false;
  if (iree_status_is_ok(status)) {
    if (flatcc_builder_init(&builder) != 0) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to initialize FlatBuffer builder");
    } else {
      builder_initialized = true;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_module_build_populate(&builder, module_name, functions,
                                               function_count, &build,
                                               allocator, out_archive);
  }

  if (builder_initialized) {
    flatcc_builder_clear(&builder);
  }
  loom_ireevm_module_build_deinitialize(&build, allocator);
  if (!iree_status_is_ok(status)) {
    loom_ireevm_module_archive_deinitialize(out_archive, allocator);
  }
  return status;
}
