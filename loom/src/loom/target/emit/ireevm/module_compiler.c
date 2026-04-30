// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/module_compiler.h"

#include <inttypes.h>

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/preparation.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/builtin_registry.h"
#include "loom/pass/value_facts.h"
#include "loom/target/compile_report_low.h"
#include "loom/target/emit/ireevm/function_bytecode.h"
#include "loom/target/emit/ireevm/low_registry.h"
#include "loom/target/emit/ireevm/lower.h"
#include "loom/target/module_compiler.h"

enum {
  LOOM_IREEVM_MODULE_COMPILE_DEFAULT_MAX_ERRORS = 20,
};

static iree_string_view_t loom_ireevm_module_compile_module_name(
    const loom_ireevm_module_compile_options_t* options) {
  if (options && !iree_string_view_is_empty(options->module_name)) {
    return options->module_name;
  }
  return IREE_SV("loom");
}

static bool loom_ireevm_module_compile_bundle_is_compatible(
    void* user_data, const loom_target_module_compile_entry_t* entry) {
  (void)user_data;
  const loom_target_bundle_t* bundle = &entry->bundle_storage.bundle;
  return bundle && bundle->snapshot && bundle->export_plan &&
         bundle->snapshot->codegen_format == LOOM_TARGET_CODEGEN_FORMAT_VM &&
         bundle->snapshot->artifact_format ==
             LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE &&
         bundle->export_plan->abi_kind == LOOM_TARGET_ABI_VM_MODULE_FUNCTION;
}

static iree_status_t loom_ireevm_module_compile_append_cconv_type(
    loom_type_t type, iree_string_builder_t* builder) {
  if (!loom_type_is_scalar(type)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "IREE VM archive compiler currently supports only scalar i1/i32 ABI "
        "values");
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type != LOOM_SCALAR_TYPE_I1 &&
      scalar_type != LOOM_SCALAR_TYPE_I32) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "IREE VM archive compiler currently supports only scalar i1/i32 ABI "
        "values");
  }
  return iree_string_builder_append_string(builder, IREE_SV("i"));
}

static iree_status_t loom_ireevm_module_compile_build_cconv(
    const loom_module_t* module, loom_func_like_t function,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("0")));

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function, &argument_count);
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ireevm_module_compile_append_cconv_type(
        loom_module_value_type(module, argument_ids[i]), builder));
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("_")));
  const loom_value_slice_t result_ids = loom_func_def_results(function.op);
  for (uint16_t i = 0; i < result_ids.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ireevm_module_compile_append_cconv_type(
        loom_module_value_type(module, result_ids.values[i]), builder));
  }
  return iree_ok_status();
}

static iree_string_view_t loom_ireevm_module_compile_export_name(
    const loom_target_module_compile_entry_t* entry) {
  const loom_target_export_plan_t* export_plan =
      &entry->bundle_storage.export_plan;
  if (!iree_string_view_is_empty(export_plan->export_symbol)) {
    return export_plan->export_symbol;
  }
  return entry->func_name;
}

static iree_status_t loom_ireevm_module_compile_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref,
    iree_string_view_t* out_name) {
  IREE_ASSERT_ARGUMENT(out_name);
  *out_name = iree_string_view_empty();
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "lowered IREE VM function symbol is invalid");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "lowered IREE VM function symbol name is invalid");
  }
  *out_name = module->strings.entries[symbol->name_id];
  return iree_ok_status();
}

static iree_status_t loom_ireevm_module_compile_lower_function(
    loom_module_t* module,
    const loom_target_low_descriptor_registry_t* registry,
    const loom_target_module_compile_entry_t* entry,
    loom_func_like_t source_function,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    uint32_t max_errors, iree_arena_allocator_t* table_arena,
    loom_pass_value_fact_owner_t* value_facts,
    loom_target_compile_report_t* report, loom_low_lower_result_t* out_result) {
  loom_low_lower_policy_registry_t policy_registry = {0};
  loom_ireevm_low_lower_policy_registry_initialize(&policy_registry);
  const loom_low_lower_policy_t* policy = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_policy_registry_lookup_for_bundle(
      &policy_registry, &entry->bundle_storage.bundle, &policy));

  loom_low_lower_report_storage_t report_storage = {0};
  IREE_RETURN_IF_ERROR(loom_target_compile_report_allocate_low_lowering_rows(
      report, table_arena, &report_storage));
  loom_value_fact_table_t* fact_table = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_fact_owner_acquire(
      value_facts, module,
      loom_pass_value_fact_scope_function_for_target(
          source_function, &entry->bundle_storage.bundle),
      &fact_table));
  const loom_low_lower_options_t lower_options = {
      .target_ref = entry->target_ref,
      .bundle = &entry->bundle_storage.bundle,
      .descriptor_registry = &registry->registry,
      .policy = policy,
      .fact_table = fact_table,
      .emitter = loom_target_module_compile_emitter(diagnostic_emitter),
      .max_errors = max_errors,
      .report_enabled = report != NULL,
      .report_storage = report_storage,
      .table_arena = table_arena,
  };
  iree_status_t status = loom_low_lower_function(module, source_function,
                                                 &lower_options, out_result);
  loom_pass_value_fact_owner_invalidate(value_facts);
  IREE_RETURN_IF_ERROR(status);
  if (report != NULL) {
    loom_target_compile_report_record_low_lowering(report, out_result);
  }
  if (out_result->error_count > 0 || out_result->low_func_op == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source-to-low lowering failed with %" PRIu32 " error%s",
        out_result->error_count, out_result->error_count == 1 ? "" : "s");
  }
  return iree_ok_status();
}

iree_status_t loom_ireevm_compile_module_archive(
    loom_module_t* module, const loom_ireevm_module_compile_options_t* options,
    iree_allocator_t allocator, loom_ireevm_module_archive_t* out_archive) {
  IREE_ASSERT_ARGUMENT(out_archive);
  *out_archive = (loom_ireevm_module_archive_t){0};
  loom_target_compile_report_t* report = options ? options->report : NULL;
  if (report != NULL) {
    loom_target_compile_report_initialize(report);
    loom_target_compile_report_set_row_storage(
        report, options ? &options->report_row_storage : NULL);
    report->artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;
    report->module_name = loom_ireevm_module_compile_module_name(options);
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
  const uint32_t max_errors = loom_target_module_compile_max_errors(
      &target_options, LOOM_IREEVM_MODULE_COMPILE_DEFAULT_MAX_ERRORS);
  loom_target_low_descriptor_registry_t low_registry = {0};
  loom_ireevm_low_descriptor_registry_initialize(&low_registry);
  loom_target_module_compile_diagnostic_emitter_t diagnostic_emitter = {0};
  loom_target_module_compile_diagnostic_emitter_initialize(
      module, &target_options, LOOM_EMITTER_VERIFIER, &diagnostic_emitter);
  const loom_target_module_compile_entry_predicate_t entry_predicate = {
      .fn = loom_ireevm_module_compile_bundle_is_compatible,
      .user_data = NULL,
  };

  loom_target_module_compile_entry_t entry = {0};
  loom_func_like_t source_function = {0};
  loom_low_lower_result_t lower_result = {0};
  loom_ireevm_function_bytecode_t bytecode = {0};
  iree_string_builder_t calling_convention_builder;
  iree_string_builder_initialize(allocator, &calling_convention_builder);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);
  iree_arena_allocator_t table_arena;
  iree_arena_initialize(&block_pool, &table_arena);
  loom_pass_value_fact_owner_t value_facts;
  loom_pass_value_fact_owner_initialize(&block_pool, &value_facts);

  iree_status_t status = loom_target_module_compile_verify_module(
      module, &target_options, LOOM_IREEVM_MODULE_COMPILE_DEFAULT_MAX_ERRORS);
  if (iree_status_is_ok(status)) {
    status = loom_target_module_compile_select_entry(
        module, &target_options, &low_registry, entry_predicate,
        IREE_SV("IREE VM"), &table_arena, &entry);
  }
  if (iree_status_is_ok(status) && report != NULL) {
    loom_target_compile_report_record_target_bundle(
        report, &entry.bundle_storage.bundle);
  }
  if (iree_status_is_ok(status)) {
    source_function = entry.func;
  }
  if (iree_status_is_ok(status) &&
      source_function.op->kind != LOOM_OP_FUNC_DEF) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "IREE VM archive compiler requires the export source to be a func.def");
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_module_compile_build_cconv(
        module, source_function, &calling_convention_builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_module_compile_lower_function(
        module, &low_registry, &entry, source_function, &diagnostic_emitter,
        max_errors, &table_arena, &value_facts, report, &lower_result);
  }
  if (iree_status_is_ok(status) && report != NULL) {
    status = loom_ireevm_module_compile_symbol_name(
        module, lower_result.low_func_ref, &report->lowered_symbol);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_module_compile_verify_module(
        module, &target_options, LOOM_IREEVM_MODULE_COMPILE_DEFAULT_MAX_ERRORS);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* low_functions[] = {
        lower_result.low_func_op,
    };
    const loom_low_preparation_options_t preparation_options = {
        .pass_registry = loom_pass_builtin_registry(),
        .descriptor_registry = &low_registry.registry,
        .diagnostic_emitter =
            loom_target_module_compile_emitter(&diagnostic_emitter),
    };
    status = loom_low_prepare_functions_for_packetization(
        module, low_functions, IREE_ARRAYSIZE(low_functions),
        &preparation_options, &block_pool);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_module_compile_verify_module(
        module, &target_options, LOOM_IREEVM_MODULE_COMPILE_DEFAULT_MAX_ERRORS);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_module_compile_verify_low_module(
        module, &low_registry, &diagnostic_emitter, max_errors);
  }
  loom_low_emission_frame_t frame = {0};
  if (iree_status_is_ok(status)) {
    const loom_low_emission_frame_options_t frame_options = {
        .descriptor_registry = &low_registry.registry,
        .memory_access_table = lower_result.memory_access_table,
        .emitter = loom_target_module_compile_emitter(&diagnostic_emitter),
    };
    status = loom_low_emission_frame_build(
        module, lower_result.low_func_op, &frame_options, &table_arena, &frame);
  }
  if (iree_status_is_ok(status) && report != NULL) {
    status =
        loom_target_compile_report_record_low_emission_frame(report, &frame);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_emit_function_bytecode(
        &frame.schedule, &frame.allocation, allocator, &bytecode);
  }
  if (iree_status_is_ok(status) && report != NULL) {
    loom_target_compile_report_record_emission(
        report,
        frame.schedule.scheduled_node_count + frame.schedule.block_count,
        bytecode.bytecode_length, bytecode.data_length);
  }
  if (iree_status_is_ok(status)) {
    const loom_ireevm_module_archive_function_t functions[] = {
        {
            .export_name = loom_ireevm_module_compile_export_name(&entry),
            .calling_convention =
                iree_string_builder_view(&calling_convention_builder),
            .bytecode = &bytecode,
        },
    };
    status = loom_ireevm_emit_module_archive(
        loom_ireevm_module_compile_module_name(options), functions,
        IREE_ARRAYSIZE(functions), allocator, out_archive);
  }
  if (iree_status_is_ok(status) && report != NULL) {
    loom_target_compile_report_record_artifact_size(report,
                                                    out_archive->data_length);
  }

  if (!iree_status_is_ok(status)) {
    loom_ireevm_module_archive_deinitialize(out_archive, allocator);
  }
  if (report != NULL) {
    loom_target_compile_report_record_status(report, status);
  }
  loom_ireevm_function_bytecode_deinitialize(&bytecode, allocator);
  loom_pass_value_fact_owner_deinitialize(&value_facts);
  iree_arena_deinitialize(&table_arena);
  iree_arena_block_pool_deinitialize(&block_pool);
  iree_string_builder_deinitialize(&calling_convention_builder);
  return status;
}
