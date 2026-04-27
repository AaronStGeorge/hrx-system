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
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/hal_kernel_abi.h"
#include "loom/target/arch/amdgpu/hal_resource_materialization.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/lower.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/arch/amdgpu/wait_packets.h"
#include "loom/target/arch/amdgpu/wait_plan.h"
#include "loom/target/compile_report_low.h"
#include "loom/target/emit/native/amdgpu/kernel_hsaco.h"
#include "loom/target/launch.h"
#include "loom/target/module_compiler.h"

enum {
  LOOM_AMDGPU_MODULE_COMPILE_DEFAULT_MAX_ERRORS = 20,
};

static bool loom_amdgpu_module_compile_bundle_is_compatible(
    void* user_data, const loom_target_module_compile_entry_t* entry) {
  (void)user_data;
  if (!loom_kernel_def_isa(entry->func.op) &&
      !loom_low_kernel_def_isa(entry->func.op)) {
    return false;
  }
  const loom_target_bundle_t* bundle = &entry->bundle_storage.bundle;
  return bundle && bundle->snapshot && bundle->export_plan &&
         bundle->snapshot->codegen_format ==
             LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE &&
         bundle->snapshot->artifact_format == LOOM_TARGET_ARTIFACT_FORMAT_ELF &&
         iree_string_view_equal(bundle->snapshot->target_triple,
                                IREE_SV("amdgcn-amd-amdhsa")) &&
         bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL;
}

typedef struct loom_amdgpu_module_compile_kernel_plan_t {
  // Target-resolved function entry used to build this kernel.
  const loom_target_module_compile_entry_t* entry;
  // Selected or lowered low.kernel.def op for packetization.
  loom_op_t* low_function_op;
  // ABI/resource materialization result captured before packetization.
  loom_amdgpu_hal_resource_materialization_result_t materialization;
  // Fixed allocator values derived from the HAL ABI live-ins.
  const loom_low_allocation_fixed_value_t* fixed_values;
  // Number of entries in |fixed_values|.
  iree_host_size_t fixed_value_count;
} loom_amdgpu_module_compile_kernel_plan_t;

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

static iree_status_t loom_amdgpu_module_compile_set_target_profile_cpu(
    loom_module_t* module, const loom_target_module_compile_entry_t* entry,
    iree_string_view_t target_cpu) {
  if (!loom_symbol_ref_is_valid(entry->target_ref) ||
      entry->target_ref.module_id != 0 ||
      entry->target_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target CPU override requires a valid "
                            "target profile symbol");
  }
  loom_symbol_t* symbol = &module->symbols.entries[entry->target_ref.symbol_id];
  if (!symbol->defining_op || !loom_target_profile_isa(symbol->defining_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target CPU override requires a "
                            "target.profile symbol");
  }
  loom_string_id_t target_cpu_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("target_cpu"),
                                                 &target_cpu_name_id));
  loom_string_id_t target_cpu_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, target_cpu, &target_cpu_id));
  const loom_named_attr_update_t update = loom_named_attr_replace(
      target_cpu_name_id, loom_attr_string(target_cpu_id));
  loom_attribute_t overrides = {0};
  IREE_RETURN_IF_ERROR(loom_module_replace_canonical_attr_dict(
      module, loom_target_profile_overrides(symbol->defining_op),
      loom_make_named_attr_update_slice(&update, 1), &overrides));
  loom_op_attrs(symbol->defining_op)[loom_target_profile_overrides_ATTR_INDEX] =
      overrides;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_module_compile_apply_target_cpu(
    loom_module_t* module, loom_target_module_compile_entry_t* entry,
    iree_string_view_t target_cpu) {
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

  IREE_RETURN_IF_ERROR(loom_amdgpu_module_compile_set_target_profile_cpu(
      module, entry, processor->target_cpu));
  entry->bundle_storage.snapshot.target_cpu = processor->target_cpu;
  loom_target_bundle_storage_rebind(&entry->bundle_storage);
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

static iree_status_t loom_amdgpu_module_compile_build_hsaco_contribution(
    const loom_low_packetization_t* packetization,
    const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout,
    loom_amdgpu_kernel_hsaco_contribution_t* out_contribution,
    iree_arena_allocator_t* sidecar_arena) {
  loom_amdgpu_wait_plan_t wait_plan = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_plan_build(&packetization->schedule,
                                                   sidecar_arena, &wait_plan));
  loom_amdgpu_wait_packet_plan_t wait_packets = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_packet_plan_build(
      &wait_plan, sidecar_arena, &wait_packets));

  const loom_amdgpu_kernel_hsaco_options_t hsaco_options = {
      .abi_layout = abi_layout,
      .wait_packets = &wait_packets,
  };
  return loom_amdgpu_build_kernel_hsaco_contribution(
      &packetization->schedule, &packetization->allocation, &hsaco_options,
      out_contribution, sidecar_arena);
}

static iree_status_t loom_amdgpu_module_compile_write_hsaco(
    const loom_amdgpu_kernel_hsaco_contribution_t* contributions,
    iree_host_size_t contribution_count, iree_const_byte_span_t* out_hsaco,
    iree_arena_allocator_t* sidecar_arena, iree_allocator_t allocator) {
  IREE_ASSERT_ARGUMENT(out_hsaco);
  *out_hsaco = iree_const_byte_span_empty();

  iree_io_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      32 * 1024, allocator, &stream));
  iree_status_t status = loom_amdgpu_write_kernel_hsaco_contributions(
      contributions, contribution_count, stream, sidecar_arena);
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
    uint32_t max_errors, iree_arena_allocator_t* sidecar_arena,
    loom_target_compile_report_t* report, loom_low_lower_result_t* out_result) {
  loom_low_lower_policy_registry_t policy_registry = {0};
  loom_amdgpu_low_lower_policy_registry_initialize(&policy_registry);
  const loom_low_lower_policy_t* policy = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_policy_registry_lookup_for_bundle(
      &policy_registry, &entry->bundle_storage.bundle, &policy));

  loom_low_lower_report_storage_t report_storage = {0};
  IREE_RETURN_IF_ERROR(loom_target_compile_report_allocate_low_lowering_rows(
      report, sidecar_arena, &report_storage));
  static const loom_target_low_legality_provider_t* const
      kLowLegalityProviders[] = {
          &loom_amdgpu_low_legality_provider_storage,
      };
  const loom_low_lower_options_t lower_options = {
      .target_ref = entry->target_ref,
      .bundle = &entry->bundle_storage.bundle,
      .descriptor_registry = &low_registry->registry,
      .descriptor_requirements =
          LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
      .legality_provider_list =
          {
              .count = IREE_ARRAYSIZE(kLowLegalityProviders),
              .values = kLowLegalityProviders,
          },
      .policy = policy,
      .emitter = loom_target_module_compile_emitter(diagnostic_emitter),
      .max_errors = max_errors,
      .report_enabled = report != NULL,
      .report_storage = report_storage,
  };
  IREE_RETURN_IF_ERROR(loom_low_lower_function(module, source_function,
                                               &lower_options, out_result));
  if (report != NULL) {
    loom_target_compile_report_record_low_lowering(report, out_result);
  }
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
    uint32_t max_errors, iree_arena_allocator_t* sidecar_arena,
    loom_target_compile_report_t* report, loom_op_t** out_low_function_op) {
  IREE_ASSERT_ARGUMENT(out_low_function_op);
  *out_low_function_op = NULL;

  if (loom_low_kernel_def_isa(source_function.op)) {
    *out_low_function_op = source_function.op;
    return iree_ok_status();
  }
  if (!loom_kernel_def_isa(source_function.op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU HAL executable compilation requires the export source to be "
        "a kernel.def or low.kernel.def");
  }

  loom_low_lower_result_t lower_result = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_module_compile_lower_function(
      module, low_registry, entry, source_function, diagnostic_emitter,
      max_errors, sidecar_arena, report, &lower_result));

  *out_low_function_op = lower_result.low_func_op;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_module_compile_prepare_kernel_plan(
    loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_target_module_compile_entry_t* entry,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    uint32_t max_errors, iree_arena_allocator_t* sidecar_arena,
    loom_target_compile_report_t* report,
    loom_amdgpu_module_compile_kernel_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_amdgpu_module_compile_kernel_plan_t){
      .entry = entry,
  };

  loom_op_t* low_function_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_module_compile_select_low_function(
      module, low_registry, entry, entry->func, diagnostic_emitter, max_errors,
      sidecar_arena, report, &low_function_op));
  out_plan->low_function_op = low_function_op;
  if (report != NULL) {
    loom_symbol_ref_t lowered_ref = entry->func_ref;
    if (low_function_op != entry->func.op) {
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_resource_materialize(
      module, low_function_op, &entry->bundle_storage.bundle, descriptor_set,
      &out_plan->materialization, sidecar_arena));

  return loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
      module, low_function_op, descriptor_set, &out_plan->fixed_values,
      &out_plan->fixed_value_count, sidecar_arena);
}

static iree_status_t loom_amdgpu_module_compile_build_kernel_contribution(
    loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    const loom_amdgpu_module_compile_kernel_plan_t* plan,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* sidecar_arena, loom_target_compile_report_t* report,
    loom_amdgpu_kernel_hsaco_contribution_t* out_contribution,
    loom_amdgpu_hal_executable_export_t* out_export) {
  IREE_ASSERT_ARGUMENT(out_contribution);
  IREE_ASSERT_ARGUMENT(out_export);
  *out_contribution = (loom_amdgpu_kernel_hsaco_contribution_t){0};
  *out_export = (loom_amdgpu_hal_executable_export_t){0};

  loom_low_packetization_t packetization = {0};
  const loom_low_packetization_options_t packetization_options = {
      .descriptor_registry = &low_registry->registry,
      .allocation_fixed_values = plan->fixed_values,
      .allocation_fixed_value_count = plan->fixed_value_count,
      .emitter = loom_target_module_compile_emitter(diagnostic_emitter),
  };
  IREE_RETURN_IF_ERROR(loom_low_packetize_function(
      module, plan->low_function_op, &packetization_options, sidecar_arena,
      &packetization));
  if (report != NULL) {
    loom_target_compile_report_record_low_packetization(report, &packetization);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_module_compile_build_hsaco_contribution(
      &packetization, &plan->materialization.abi_layout, out_contribution,
      sidecar_arena));
  if (report != NULL) {
    loom_target_compile_report_record_emission(
        report, out_contribution->summary.instruction_count,
        out_contribution->summary.text_byte_count,
        out_contribution->summary.text_storage_byte_count);
    loom_target_compile_report_record_memory(
        report, out_contribution->summary.private_segment_fixed_size,
        out_contribution->summary.group_segment_fixed_size);
  }

  const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout =
      &plan->materialization.abi_layout;
  loom_amdgpu_hal_executable_binding_flags_t* binding_flags = NULL;
  if (abi_layout->resource_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        sidecar_arena, abi_layout->resource_count, sizeof(*binding_flags),
        (void**)&binding_flags));
    memset(binding_flags, 0,
           abi_layout->resource_count * sizeof(*binding_flags));
  }

  const loom_target_hal_kernel_abi_t* hal_kernel =
      &plan->entry->bundle_storage.bundle.export_plan->hal_kernel;
  IREE_RETURN_IF_ERROR(loom_target_require_concrete_hal_kernel_launch(
      hal_kernel, IREE_SV("AMDGPU HAL executable export")));
  *out_export = (loom_amdgpu_hal_executable_export_t){
      .symbol_name = out_contribution->kernel.metadata.descriptor_symbol,
      .workgroup_size = hal_kernel->required_workgroup_size,
      .constant_count = 0,
      .binding_flags = binding_flags,
      .binding_count = abi_layout->resource_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_module_compile_entries(
    loom_module_t* module,
    const loom_target_module_compile_options_t* target_options,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_target_module_compile_entry_list_t entries,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_arena_allocator_t* sidecar_arena, loom_target_compile_report_t* report,
    loom_amdgpu_hal_executable_t* out_executable, iree_allocator_t allocator) {
  if (entries.count == 0 || entries.values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU compilation requires at least one entry");
  }

  const uint32_t max_errors = loom_target_module_compile_max_errors(
      target_options, LOOM_AMDGPU_MODULE_COMPILE_DEFAULT_MAX_ERRORS);
  loom_target_compile_report_t* single_entry_report =
      entries.count == 1 ? report : NULL;
  loom_amdgpu_module_compile_kernel_plan_t* plans = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      sidecar_arena, entries.count, sizeof(*plans), (void**)&plans));
  for (uint16_t i = 0; i < entries.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_module_compile_prepare_kernel_plan(
        module, low_registry, &entries.values[i], diagnostic_emitter,
        max_errors, sidecar_arena, single_entry_report, &plans[i]));
  }

  IREE_RETURN_IF_ERROR(loom_target_module_compile_verify_module(
      module, target_options, LOOM_AMDGPU_MODULE_COMPILE_DEFAULT_MAX_ERRORS));
  IREE_RETURN_IF_ERROR(loom_target_module_compile_verify_low_module(
      module, low_registry,
      LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION, diagnostic_emitter,
      max_errors));

  loom_amdgpu_kernel_hsaco_contribution_t* contributions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(sidecar_arena, entries.count,
                                                 sizeof(*contributions),
                                                 (void**)&contributions));
  loom_amdgpu_hal_executable_export_t* exports = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      sidecar_arena, entries.count, sizeof(*exports), (void**)&exports));
  for (uint16_t i = 0; i < entries.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_module_compile_build_kernel_contribution(
        module, low_registry, &plans[i], diagnostic_emitter, sidecar_arena,
        single_entry_report, &contributions[i], &exports[i]));
  }

  iree_const_byte_span_t hsaco = iree_const_byte_span_empty();
  iree_status_t status = loom_amdgpu_module_compile_write_hsaco(
      contributions, entries.count, &hsaco, sidecar_arena, allocator);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_emit_hal_executable(contributions[0].target, hsaco,
                                             exports, entries.count, allocator,
                                             out_executable);
  }
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
  const loom_target_module_compile_entry_predicate_t entry_predicate = {
      .fn = loom_amdgpu_module_compile_bundle_is_compatible,
      .user_data = NULL,
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool, &sidecar_arena);

  loom_target_module_compile_entry_t single_entry = {0};
  loom_target_module_compile_entry_list_t entries = {0};
  const iree_string_view_t artifact_symbol =
      options ? loom_target_module_compile_normalize_symbol_name(
                    options->artifact_symbol)
              : iree_string_view_empty();
  const iree_string_view_t entry_symbol =
      loom_target_module_compile_entry_symbol_name(&target_options);
  iree_status_t status = loom_target_module_compile_verify_module(
      module, &target_options, LOOM_AMDGPU_MODULE_COMPILE_DEFAULT_MAX_ERRORS);
  if (iree_status_is_ok(status) && !iree_string_view_is_empty(entry_symbol) &&
      !iree_string_view_is_empty(artifact_symbol)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "select either an AMDGPU entry symbol or an "
                              "AMDGPU artifact symbol, not both");
  }
  if (iree_status_is_ok(status)) {
    if (!iree_string_view_is_empty(artifact_symbol)) {
      status = loom_target_module_compile_select_artifact_entries(
          module, artifact_symbol, &low_registry, entry_predicate,
          IREE_SV("AMDGPU HAL-native"), &sidecar_arena, &entries);
    } else {
      status = loom_target_module_compile_select_entry(
          module, &target_options, &low_registry, entry_predicate,
          IREE_SV("AMDGPU HAL-native"), &sidecar_arena, &single_entry);
      if (iree_status_is_ok(status)) {
        entries = (loom_target_module_compile_entry_list_t){
            .values = &single_entry,
            .count = 1,
        };
      }
    }
  }
  if (iree_status_is_ok(status) && options != NULL) {
    for (uint16_t i = 0; i < entries.count && iree_status_is_ok(status); ++i) {
      status = loom_amdgpu_module_compile_apply_target_cpu(
          module, &entries.values[i], options->target_cpu);
    }
  }
  if (iree_status_is_ok(status) && report != NULL) {
    loom_target_compile_report_record_target_bundle(
        report, &entries.values[0].bundle_storage.bundle);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_module_compile_entries(
        module, &target_options, &low_registry, entries, &diagnostic_emitter,
        &sidecar_arena, report, out_executable, allocator);
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
