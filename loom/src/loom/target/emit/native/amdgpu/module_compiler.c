// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/module_compiler.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/hal_kernel_abi.h"
#include "loom/target/arch/amdgpu/hal_resource_materialization.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/lower.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/arch/amdgpu/wait_packets.h"
#include "loom/target/arch/amdgpu/wait_plan.h"
#include "loom/target/compile_report_low.h"
#include "loom/target/emit/native/amdgpu/kernel_hsaco.h"
#include "loom/target/module_compiler.h"

enum {
  LOOM_AMDGPU_MODULE_COMPILE_DEFAULT_MAX_ERRORS = 20,
};

static bool loom_amdgpu_module_compile_bundle_is_compatible(
    void* user_data, const loom_target_module_compile_entry_t* entry) {
  (void)user_data;
  const loom_target_bundle_t* bundle = &entry->bundle_storage.bundle;
  return bundle && bundle->snapshot && bundle->export_plan &&
         bundle->snapshot->codegen_format ==
             LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE &&
         bundle->snapshot->artifact_format == LOOM_TARGET_ARTIFACT_FORMAT_ELF &&
         iree_string_view_equal(bundle->snapshot->target_triple,
                                IREE_SV("amdgcn-amd-amdhsa")) &&
         bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL;
}

static iree_status_t loom_amdgpu_module_compile_append_target_id(
    const loom_target_snapshot_t* snapshot, iree_string_builder_t* builder) {
  if (!snapshot || iree_string_view_is_empty(snapshot->target_triple) ||
      iree_string_view_is_empty(snapshot->target_cpu)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target snapshot requires triple and CPU");
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, snapshot->target_triple));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("--")));
  return iree_string_builder_append_string(builder, snapshot->target_cpu);
}

static iree_status_t loom_amdgpu_module_compile_append_descriptor_symbol(
    const loom_target_module_compile_entry_t* entry,
    iree_string_builder_t* builder) {
  iree_string_view_t symbol = iree_string_view_empty();
  const loom_target_export_plan_t* export_plan =
      &entry->bundle_storage.export_plan;
  if (!iree_string_view_is_empty(export_plan->export_symbol)) {
    symbol = export_plan->export_symbol;
  } else {
    symbol = entry->function_name;
  }
  if (iree_string_view_is_empty(symbol)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU HAL executable export symbol is required");
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(builder, symbol));
  return iree_string_builder_append_string(builder, IREE_SV(".kd"));
}

static iree_status_t loom_amdgpu_module_compile_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref,
    iree_string_view_t* out_name) {
  IREE_ASSERT_ARGUMENT(out_name);
  *out_name = iree_string_view_empty();
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU compile symbol reference is invalid");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU compile symbol has no module string");
  }
  *out_name = module->strings.entries[symbol->name_id];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_module_compile_apply_target_cpu(
    loom_target_module_compile_entry_t* entry, iree_string_view_t target_cpu) {
  if (iree_string_view_is_empty(target_cpu)) {
    return iree_ok_status();
  }
  const loom_amdgpu_processor_info_t* processor = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_lookup_processor(target_cpu, &processor));
  if (iree_string_view_is_empty(processor->low_preset_key) ||
      !iree_string_view_equal(processor->descriptor_set_key,
                              entry->bundle_storage.config.contract_set_key)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU target CPU '%.*s' is not compatible with descriptor set '%.*s'",
        (int)target_cpu.size, target_cpu.data,
        (int)entry->bundle_storage.config.contract_set_key.size,
        entry->bundle_storage.config.contract_set_key.data);
  }

  entry->bundle_storage.snapshot.target_cpu = processor->target_cpu;
  loom_target_ir_bundle_storage_rebind(&entry->bundle_storage);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_module_compile_read_stream_contents(
    iree_io_stream_t* stream, iree_allocator_t allocator,
    iree_const_byte_span_t* out_contents) {
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_contents = iree_const_byte_span_empty();
  const iree_io_stream_pos_t stream_length = iree_io_stream_length(stream);
  if (stream_length < 0 || (uint64_t)stream_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU HSACO stream length is out of range");
  }
  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, (iree_host_size_t)stream_length, (void**)&data));
  iree_status_t status =
      iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_read(stream, (iree_host_size_t)stream_length, data,
                                 /*out_buffer_length=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    *out_contents =
        iree_make_const_byte_span(data, (iree_host_size_t)stream_length);
  } else {
    iree_allocator_free(allocator, data);
  }
  return status;
}

static iree_status_t loom_amdgpu_module_compile_emit_hsaco(
    const loom_low_packetization_t* packetization,
    const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout,
    loom_amdgpu_kernel_hsaco_summary_t* out_summary,
    iree_const_byte_span_t* out_hsaco, iree_arena_allocator_t* sidecar_arena,
    iree_allocator_t allocator) {
  IREE_ASSERT_ARGUMENT(out_summary);
  IREE_ASSERT_ARGUMENT(out_hsaco);
  *out_summary = (loom_amdgpu_kernel_hsaco_summary_t){0};
  *out_hsaco = iree_const_byte_span_empty();

  loom_amdgpu_wait_plan_t wait_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_build(&packetization->schedule,
                                                   sidecar_arena, &wait_plan));
  loom_amdgpu_wait_packet_plan_t wait_packets = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_plan_build(
      &wait_plan, sidecar_arena, &wait_packets));

  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      32 * 1024, allocator, &stream));
  const loom_amdgpu_kernel_hsaco_options_t hsaco_options = {
      .abi_layout = abi_layout,
      .wait_packets = &wait_packets,
      .summary = out_summary,
  };
  iree_status_t status = loom_amdgpu_emit_kernel_hsaco(
      &packetization->schedule, &packetization->allocation, &hsaco_options,
      stream, sidecar_arena);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_module_compile_read_stream_contents(stream, allocator,
                                                             out_hsaco);
  }
  iree_io_stream_release(stream);
  return status;
}

static iree_status_t loom_amdgpu_module_compile_lower_function(
    loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_target_module_compile_entry_t* entry,
    loom_func_like_t source_function,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    uint32_t max_errors, loom_low_lower_result_t* out_result) {
  loom_low_lower_policy_registry_t policy_registry = {0};
  loom_amdgpu_low_lower_policy_registry_initialize(&policy_registry);
  const loom_low_lower_policy_t* policy = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_policy_registry_lookup_for_bundle(
      &policy_registry, &entry->bundle_storage.bundle, &policy));

  const loom_low_lower_options_t lower_options = {
      .target_ref = entry->target_ref,
      .bundle = &entry->bundle_storage.bundle,
      .descriptor_registry = &low_registry->registry,
      .descriptor_requirements =
          LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
      .policy = policy,
      .emitter = loom_target_module_compile_emitter(diagnostic_emitter),
      .max_errors = max_errors,
  };
  IREE_RETURN_IF_ERROR(loom_low_lower_function(module, source_function,
                                               &lower_options, out_result));
  if (out_result->error_count > 0 || out_result->low_func_op == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU source-to-low lowering failed with %" PRIu32 " error%s",
        out_result->error_count, out_result->error_count == 1 ? "" : "s");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_module_compile_select_low_function(
    loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_target_module_compile_entry_t* entry,
    loom_func_like_t source_function,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    uint32_t max_errors, loom_op_t** out_low_function_op) {
  IREE_ASSERT_ARGUMENT(out_low_function_op);
  *out_low_function_op = NULL;

  if (loom_low_func_def_isa(source_function.op)) {
    *out_low_function_op = source_function.op;
    return iree_ok_status();
  }
  if (!loom_func_def_isa(source_function.op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL executable compilation requires the export source to be "
        "a func.def or low.func.def");
  }

  loom_low_lower_result_t lower_result = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_module_compile_lower_function(
      module, low_registry, entry, source_function, diagnostic_emitter,
      max_errors, &lower_result));

  *out_low_function_op = lower_result.low_func_op;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_module_compile_low_function(
    loom_module_t* module,
    const loom_amdgpu_module_compile_options_t* amdgpu_options,
    const loom_target_module_compile_options_t* target_options,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_target_module_compile_entry_t* entry,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* sidecar_arena, loom_target_compile_report_t* report,
    loom_amdgpu_hal_executable_t* out_executable, iree_allocator_t allocator) {
  (void)amdgpu_options;
  const uint32_t max_errors = loom_target_module_compile_max_errors(
      target_options, LOOM_AMDGPU_MODULE_COMPILE_DEFAULT_MAX_ERRORS);
  loom_op_t* low_function_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_module_compile_select_low_function(
      module, low_registry, entry, entry->function, diagnostic_emitter,
      max_errors, &low_function_op));
  if (report != NULL) {
    loom_symbol_ref_t lowered_ref = entry->function_ref;
    if (low_function_op != entry->function.op) {
      lowered_ref =
          loom_func_like_callee(loom_func_like_cast(module, low_function_op));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_module_compile_symbol_name(
        module, lowered_ref, &report->lowered_symbol));
  }

  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_target_low_descriptor_set_select_for_bundle(
      &low_registry->registry, &entry->bundle_storage.bundle,
      LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION, &descriptor_set));
  loom_amdgpu_hal_resource_materialization_result_t materialization = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_materialize(
      module, low_function_op, &entry->bundle_storage.bundle, descriptor_set,
      &materialization, sidecar_arena));

  const loom_low_allocation_fixed_value_t* fixed_values = NULL;
  iree_host_size_t fixed_value_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
      module, low_function_op, descriptor_set, &fixed_values,
      &fixed_value_count, sidecar_arena));

  IREE_RETURN_IF_ERROR(loom_target_module_compile_verify_module(
      module, target_options, LOOM_AMDGPU_MODULE_COMPILE_DEFAULT_MAX_ERRORS));
  IREE_RETURN_IF_ERROR(loom_target_module_compile_verify_low_module(
      module, low_registry,
      LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION, diagnostic_emitter,
      max_errors));

  loom_low_packetization_t packetization = {0};
  const loom_low_packetization_options_t packetization_options = {
      .descriptor_registry = &low_registry->registry,
      .allocation_fixed_values = fixed_values,
      .allocation_fixed_value_count = fixed_value_count,
      .emitter = loom_target_module_compile_emitter(diagnostic_emitter),
  };
  IREE_RETURN_IF_ERROR(loom_low_packetize_function(
      module, low_function_op, &packetization_options, sidecar_arena,
      &packetization));
  if (report != NULL) {
    loom_target_compile_report_record_low_packetization(report, &packetization);
  }

  loom_amdgpu_kernel_hsaco_summary_t hsaco_summary = {0};
  iree_const_byte_span_t hsaco = iree_const_byte_span_empty();
  iree_status_t status = loom_amdgpu_module_compile_emit_hsaco(
      &packetization, &materialization.abi_layout, &hsaco_summary, &hsaco,
      sidecar_arena, allocator);
  if (iree_status_is_ok(status) && report != NULL) {
    loom_target_compile_report_record_emission(
        report, hsaco_summary.instruction_count, hsaco_summary.text_byte_count,
        hsaco_summary.text_storage_byte_count);
    loom_target_compile_report_record_memory(
        report, hsaco_summary.private_segment_fixed_size,
        hsaco_summary.group_segment_fixed_size);
  }

  iree_string_builder_t target_id_builder;
  iree_string_builder_initialize(allocator, &target_id_builder);
  iree_string_builder_t descriptor_symbol_builder;
  iree_string_builder_initialize(allocator, &descriptor_symbol_builder);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_module_compile_append_target_id(
        entry->bundle_storage.bundle.snapshot, &target_id_builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_module_compile_append_descriptor_symbol(
        entry, &descriptor_symbol_builder);
  }
  if (iree_status_is_ok(status)) {
    const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout =
        &materialization.abi_layout;
    loom_amdgpu_hal_executable_binding_flags_t* binding_flags = NULL;
    if (abi_layout->resource_count != 0) {
      status = iree_arena_allocate_array(
          sidecar_arena, abi_layout->resource_count, sizeof(*binding_flags),
          (void**)&binding_flags);
      if (iree_status_is_ok(status)) {
        memset(binding_flags, 0,
               abi_layout->resource_count * sizeof(*binding_flags));
      }
    }
    if (iree_status_is_ok(status)) {
      const loom_target_hal_kernel_abi_t* hal_kernel =
          &entry->bundle_storage.bundle.export_plan->hal_kernel;
      const loom_amdgpu_hal_executable_export_t export_def = {
          .symbol_name = iree_string_builder_view(&descriptor_symbol_builder),
          .workgroup_size = hal_kernel->required_workgroup_size,
          .constant_count = 0,
          .binding_flags = binding_flags,
          .binding_count = abi_layout->resource_count,
      };
      status = loom_amdgpu_emit_hal_executable(
          iree_string_builder_view(&target_id_builder), hsaco, &export_def, 1,
          allocator, out_executable);
    }
  }

  iree_string_builder_deinitialize(&descriptor_symbol_builder);
  iree_string_builder_deinitialize(&target_id_builder);
  iree_allocator_free(allocator, (void*)hsaco.data);
  return status;
}

iree_status_t loom_amdgpu_compile_hal_executable(
    loom_module_t* module, const loom_amdgpu_module_compile_options_t* options,
    iree_allocator_t allocator, loom_amdgpu_hal_executable_t* out_executable) {
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = (loom_amdgpu_hal_executable_t){0};
  loom_target_compile_report_t* report = options ? options->report : NULL;
  if (report != NULL) {
    loom_target_compile_report_initialize(report);
    loom_target_compile_report_set_row_storage(
        report, options ? &options->report_row_storage : NULL);
    report->artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE;
    report->entry_symbol =
        options ? options->entry_symbol : iree_string_view_empty();
  }
  if (!module) {
    iree_status_t status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "module is required");
    if (report != NULL) {
      loom_target_compile_report_record_status(report, status);
    }
    return status;
  }

  const loom_target_module_compile_options_t target_options = {
      .entry_symbol =
          options ? options->entry_symbol : iree_string_view_empty(),
      .diagnostic_sink =
          options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .source_resolver =
          options ? options->source_resolver : (loom_source_resolver_t){0},
      .max_errors = options ? options->max_errors : 0,
  };
  loom_target_low_descriptor_registry_t low_registry = {0};
  loom_amdgpu_low_descriptor_registry_initialize(&low_registry);
  loom_target_module_compile_diagnostic_emitter_t diagnostic_emitter = {0};
  loom_target_module_compile_diagnostic_emitter_initialize(
      module, &target_options, LOOM_EMITTER_VERIFIER, &diagnostic_emitter);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool, &sidecar_arena);

  loom_target_module_compile_entry_t entry = {0};
  iree_status_t status = loom_target_module_compile_verify_module(
      module, &target_options, LOOM_AMDGPU_MODULE_COMPILE_DEFAULT_MAX_ERRORS);
  if (iree_status_is_ok(status)) {
    status = loom_target_module_compile_select_entry(
        module, &target_options, &low_registry,
        loom_amdgpu_module_compile_bundle_is_compatible, NULL,
        IREE_SV("AMDGPU HAL-native"), &sidecar_arena, &entry);
  }
  if (iree_status_is_ok(status) && options != NULL) {
    status = loom_amdgpu_module_compile_apply_target_cpu(&entry,
                                                         options->target_cpu);
  }
  if (iree_status_is_ok(status) && report != NULL) {
    loom_target_compile_report_record_target_bundle(
        report, &entry.bundle_storage.bundle);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_module_compile_low_function(
        module, options, &target_options, &low_registry, &entry,
        &diagnostic_emitter, &sidecar_arena, report, out_executable, allocator);
  }
  if (iree_status_is_ok(status) && report != NULL) {
    loom_target_compile_report_record_artifact_size(
        report, out_executable->data_length);
  }

  if (!iree_status_is_ok(status)) {
    loom_amdgpu_hal_executable_deinitialize(out_executable, allocator);
  }
  if (report != NULL) {
    loom_target_compile_report_record_status(report, status);
  }
  iree_arena_deinitialize(&sidecar_arena);
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}
