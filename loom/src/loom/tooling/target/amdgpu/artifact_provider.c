// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/artifact_provider.h"

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/ops/target.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/hal/executable.h"
#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"
#include "loom/tooling/execution/hal/runtime.h"

typedef struct loom_amdgpu_hal_artifact_storage_t {
  // Target-native HSACO artifact and export metadata.
  loom_amdgpu_hal_kernel_library_t kernel_library;
  // IREE HAL executable package consumed by the AMDGPU loader.
  loom_amdgpu_hal_executable_t hal_executable;
} loom_amdgpu_hal_artifact_storage_t;

static iree_status_t loom_amdgpu_hal_artifact_provider_format_target_id(
    const loom_amdgpu_processor_info_t* processor,
    iree_string_builder_t* builder) {
  iree_string_builder_reset(builder);
  return loom_amdgpu_amdhsa_target_id_append(processor,
                                             iree_string_view_empty(), builder);
}

static iree_status_t loom_amdgpu_hal_artifact_processor_is_supported(
    const loom_amdgpu_processor_info_t* processor, bool* out_supported) {
  *out_supported = false;
  if (iree_string_view_is_empty(processor->processor) ||
      iree_string_view_is_empty(processor->descriptor_set_key) ||
      processor->descriptor_set_ordinal ==
          LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE ||
      processor->elf_machine_flags == 0 ||
      processor->kernel_descriptor_profile ==
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE) {
    return iree_ok_status();
  }

  const loom_amdgpu_descriptor_set_info_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_info_lookup_descriptor_set_by_ordinal(
      processor->descriptor_set_ordinal, &descriptor_set));
  *out_supported = descriptor_set->supports_descriptor_packet_encoding;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_artifact_provider_select_device_target(
    const loom_run_hal_artifact_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_target);

  *out_target = (loom_run_hal_device_target_t){0};

  iree_string_builder_t target_id_builder;
  iree_string_builder_initialize(allocator, &target_id_builder);

  iree_status_t status = iree_ok_status();
  const iree_host_size_t processor_count =
      loom_amdgpu_target_info_processor_count();
  for (iree_host_size_t i = 0;
       i < processor_count && out_target->data == NULL &&
       iree_status_is_ok(status);
       ++i) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_processor_at(i);
    bool emit_supported = false;
    status = loom_amdgpu_hal_artifact_processor_is_supported(processor,
                                                             &emit_supported);
    if (!iree_status_is_ok(status) || !emit_supported) {
      continue;
    }
    const loom_target_bundle_t* target_bundle =
        loom_amdgpu_target_bundle_for_descriptor_set(
            processor->descriptor_set_ordinal);
    if (target_bundle == NULL) {
      continue;
    }
    status = loom_amdgpu_hal_artifact_provider_format_target_id(
        processor, &target_id_builder);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (iree_hal_executable_cache_can_prepare_format(
            runtime->executable_cache,
            IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION,
            iree_string_builder_view(&target_id_builder))) {
      *out_target = (loom_run_hal_device_target_t){
          .data = processor,
          .target_bundle = target_bundle,
          .target_key = processor->processor,
      };
    }
  }

  iree_string_builder_deinitialize(&target_id_builder);
  if (iree_status_is_ok(status) && out_target->data == NULL) {
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "selected %.*s HAL device has no Loom-supported native target",
        (int)provider->target_family_name.size,
        provider->target_family_name.data);
  }
  return status;
}

static iree_status_t loom_amdgpu_hal_artifact_provider_validate_target_symbol(
    loom_module_t* module, iree_string_view_t symbol_name,
    const loom_amdgpu_processor_info_t* processor, loom_symbol_ref_t target_ref,
    bool* out_reusable) {
  *out_reusable = false;
  const loom_symbol_t* symbol = &module->symbols.entries[target_ref.symbol_id];
  if (symbol->defining_op == NULL) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_target_isa(symbol->defining_op)) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "AMDGPU target assignment symbol '@%.*s' already names a non-AMDGPU "
        "target op",
        (int)symbol_name.size, symbol_name.data);
  }
  const iree_string_view_t existing_processor =
      loom_amdgpu_target_record_processor_name(module, symbol->defining_op);
  if (!iree_string_view_equal(existing_processor, processor->processor)) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "AMDGPU target assignment symbol '@%.*s' selects processor '%.*s', "
        "but the selected HAL device requires '%.*s'",
        (int)symbol_name.size, symbol_name.data, (int)existing_processor.size,
        existing_processor.data, (int)processor->processor.size,
        processor->processor.data);
  }
  *out_reusable = true;
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_hal_artifact_provider_resolve_device_target_ref(
    const loom_run_hal_artifact_provider_t* provider, loom_module_t* module,
    const loom_run_hal_device_target_t* target,
    loom_symbol_ref_t* out_target_ref) {
  IREE_ASSERT_ARGUMENT(provider);
  if (out_target_ref == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL target resolver requires an output "
                            "target ref");
  }
  *out_target_ref = loom_symbol_ref_null();
  if (module == NULL || module->body == NULL ||
      module->body->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL target resolver requires a module "
                            "with a body block");
  }
  if (target == NULL || target->data == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU HAL target resolver requires a selected "
                            "device target");
  }

  const loom_amdgpu_processor_info_t* processor =
      (const loom_amdgpu_processor_info_t*)target->data;
  if (iree_string_view_is_empty(processor->processor)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU HAL target resolver selected a processor "
                            "row with no processor name");
  }

  loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, processor->processor, &symbol_name_id));
  uint16_t symbol_id = loom_module_find_symbol(module, symbol_name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_module_add_symbol(module, symbol_name_id, &symbol_id));
  }
  *out_target_ref = (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};

  bool reusable = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_artifact_provider_validate_target_symbol(
      module, processor->processor, processor, *out_target_ref, &reusable));
  if (reusable) {
    return iree_ok_status();
  }

  loom_block_t* module_block = loom_module_block(module);
  loom_builder_t builder = {0};
  loom_builder_initialize(module, &module->arena, module_block, &builder);
  if (module_block->first_op != NULL) {
    loom_builder_set_before(&builder, module_block->first_op);
  }
  loom_op_t* target_op = NULL;
  return loom_amdgpu_target_record_build_for_processor(
      &builder, processor, *out_target_ref, LOOM_LOCATION_UNKNOWN, &target_op);
}

static loom_target_compile_report_row_storage_t
loom_amdgpu_hal_artifact_provider_report_row_storage(
    loom_target_compile_report_t* report) {
  if (report == NULL) {
    return (loom_target_compile_report_row_storage_t){0};
  }
  return (loom_target_compile_report_row_storage_t){
      .pressure_rows = report->pressure_rows,
      .pressure_row_capacity = report->pressure_row_capacity,
      .spill_rows = report->spill_rows,
      .spill_row_capacity = report->spill_row_capacity,
      .source_low_rows = report->source_low_rows,
      .source_low_row_capacity = report->source_low_row_capacity,
      .target_legalization_rows = report->target_legalization_rows,
      .target_legalization_row_capacity =
          report->target_legalization_row_capacity,
  };
}

static iree_status_t loom_amdgpu_hal_artifact_provider_emit_artifact(
    const loom_run_hal_artifact_provider_t* provider, loom_module_t* module,
    const loom_run_hal_device_target_t* target,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_source_resolver_t source_resolver, uint32_t max_errors,
    loom_run_candidate_artifact_flags_t artifact_flags,
    loom_target_compile_report_t* report, iree_allocator_t allocator,
    bool* out_emitted, loom_run_hal_artifact_t* out_artifact) {
  IREE_ASSERT_ARGUMENT(provider);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(out_emitted);
  IREE_ASSERT_ARGUMENT(out_artifact);

  *out_emitted = false;
  *out_artifact = (loom_run_hal_artifact_t){0};

  const loom_amdgpu_processor_info_t* processor =
      (const loom_amdgpu_processor_info_t*)target->data;

  loom_amdgpu_hal_artifact_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *storage = (loom_amdgpu_hal_artifact_storage_t){0};

  const loom_amdgpu_hal_kernel_library_options_t library_options = {
      .processor = processor ? processor->processor : iree_string_view_empty(),
      .target_selection =
          {
              .bundle = target->target_bundle,
              .data = target->data,
          },
      .diagnostic_sink = diagnostic_sink,
      .source_resolver = source_resolver,
      .max_errors = max_errors,
      .report = report,
      .report_row_storage =
          loom_amdgpu_hal_artifact_provider_report_row_storage(report),
      .capture_target_listing = iree_all_bits_set(
          artifact_flags, LOOM_RUN_CANDIDATE_ARTIFACT_FLAG_TARGET_LISTING),
  };
  bool library_emitted = false;
  iree_status_t status = loom_amdgpu_emit_hal_kernel_library(
      module, &library_options, allocator, &library_emitted,
      &storage->kernel_library);
  if (iree_status_is_ok(status) && library_emitted) {
    status = loom_amdgpu_package_hal_executable(
        storage->kernel_library.executable_format,
        iree_make_const_byte_span(storage->kernel_library.hsaco_data,
                                  storage->kernel_library.hsaco_data_length),
        storage->kernel_library.exports, storage->kernel_library.export_count,
        allocator, &storage->hal_executable);
  }
  if (iree_status_is_ok(status) && library_emitted) {
    *out_artifact = (loom_run_hal_artifact_t){
        .executable_format = storage->hal_executable.executable_format,
        .target_bundle = target->target_bundle,
        .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
        .target_artifact_data = iree_make_const_byte_span(
            storage->kernel_library.hsaco_data,
            storage->kernel_library.hsaco_data_length),
        .target_listing_format = storage->kernel_library.target_listing_format,
        .target_listing_data = iree_make_const_byte_span(
            (const uint8_t*)storage->kernel_library.target_listing_data,
            storage->kernel_library.target_listing_data_length),
        .executable_data = iree_make_const_byte_span(
            storage->hal_executable.data, storage->hal_executable.data_length),
        .storage = storage,
    };
    *out_emitted = true;
  } else {
    loom_amdgpu_hal_executable_deinitialize(&storage->hal_executable,
                                            allocator);
    loom_amdgpu_hal_kernel_library_deinitialize(&storage->kernel_library,
                                                allocator);
    iree_allocator_free(allocator, storage);
  }
  return status;
}

static void loom_amdgpu_hal_artifact_provider_deinitialize_artifact(
    const loom_run_hal_artifact_provider_t* provider,
    loom_run_hal_artifact_t* artifact, iree_allocator_t allocator) {
  (void)provider;
  if (artifact == NULL || artifact->storage == NULL) {
    return;
  }
  loom_amdgpu_hal_artifact_storage_t* storage =
      (loom_amdgpu_hal_artifact_storage_t*)artifact->storage;
  loom_amdgpu_hal_executable_deinitialize(&storage->hal_executable, allocator);
  loom_amdgpu_hal_kernel_library_deinitialize(&storage->kernel_library,
                                              allocator);
  iree_allocator_free(allocator, storage);
  *artifact = (loom_run_hal_artifact_t){0};
}

const loom_run_hal_artifact_provider_t loom_amdgpu_hal_artifact_provider = {
    .name = IREE_SVL("amdgpu-hal"),
    .hal_driver_name = IREE_SVL("amdgpu"),
    .target_family_name = IREE_SVL("AMDGPU"),
    .select_device_target =
        loom_amdgpu_hal_artifact_provider_select_device_target,
    .resolve_device_target_ref =
        loom_amdgpu_hal_artifact_provider_resolve_device_target_ref,
    .emit_artifact = loom_amdgpu_hal_artifact_provider_emit_artifact,
    .deinitialize_artifact =
        loom_amdgpu_hal_artifact_provider_deinitialize_artifact,
};
