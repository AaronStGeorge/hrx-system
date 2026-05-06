// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/archive_emitter.h"

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/compile_report_low.h"
#include "loom/target/emit/ireevm/function_bytecode.h"
#include "loom/target/emit/ireevm/provider.h"
#include "loom/target/entry_selection.h"
#include "loom/target/provider.h"

enum {
  LOOM_IREEVM_ARCHIVE_EMIT_DEFAULT_MAX_ERRORS = 20,
};

static iree_string_view_t loom_ireevm_archive_emit_module_name(
    const loom_ireevm_archive_emit_options_t* options) {
  if (options && !iree_string_view_is_empty(options->module_name)) {
    return options->module_name;
  }
  return IREE_SV("loom");
}

static bool loom_ireevm_archive_emit_bundle_is_compatible(
    void* user_data, const loom_target_entry_t* entry) {
  if (!loom_low_func_def_isa(entry->func.op)) {
    return false;
  }
  const loom_target_bundle_t* bundle = &entry->bundle_storage.bundle;
  return bundle && bundle->snapshot && bundle->export_plan &&
         bundle->snapshot->codegen_format == LOOM_TARGET_CODEGEN_FORMAT_VM &&
         bundle->snapshot->artifact_format ==
             LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE &&
         bundle->export_plan->abi_kind == LOOM_TARGET_ABI_VM_MODULE_FUNCTION;
}

static bool loom_ireevm_archive_emit_is_vm_i32_register(
    const loom_module_t* module, loom_type_t type) {
  if (!loom_type_is_register(type) ||
      loom_type_register_unit_count(type) != 1) {
    return false;
  }
  const loom_string_id_t register_class_id = loom_type_register_class_id(type);
  if (register_class_id == LOOM_STRING_ID_INVALID ||
      register_class_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[register_class_id],
                                IREE_SV("vm.i32"));
}

static iree_status_t loom_ireevm_archive_emit_append_cconv_type(
    const loom_module_t* module, loom_type_t type,
    iree_string_builder_t* builder) {
  if (loom_ireevm_archive_emit_is_vm_i32_register(module, type)) {
    return iree_string_builder_append_string(builder, IREE_SV("i"));
  }
  if (loom_type_is_scalar(type)) {
    const loom_scalar_type_t scalar_type = loom_type_element_type(type);
    if (scalar_type == LOOM_SCALAR_TYPE_I1 ||
        scalar_type == LOOM_SCALAR_TYPE_I32) {
      return iree_string_builder_append_string(builder, IREE_SV("i"));
    }
  }
  return iree_make_status(
      IREE_STATUS_INTERNAL,
      "IREE VM target legality accepted an unsupported ABI value type");
}

static iree_status_t loom_ireevm_archive_emit_build_cconv(
    const loom_module_t* module, loom_func_like_t function,
    iree_string_builder_t* builder) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("0")));

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function, &argument_count);
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ireevm_archive_emit_append_cconv_type(
        module, loom_module_value_type(module, argument_ids[i]), builder));
  }

  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(builder, IREE_SV("_")));
  const loom_value_id_t* result_ids = loom_op_const_results(function.op);
  for (uint16_t i = 0; i < function.op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ireevm_archive_emit_append_cconv_type(
        module, loom_module_value_type(module, result_ids[i]), builder));
  }
  return iree_ok_status();
}

static iree_string_view_t loom_ireevm_archive_emit_export_name(
    const loom_target_entry_t* entry) {
  const loom_target_export_plan_t* export_plan =
      &entry->bundle_storage.export_plan;
  if (!iree_string_view_is_empty(export_plan->export_symbol)) {
    return export_plan->export_symbol;
  }
  return entry->func_name;
}

static iree_status_t loom_ireevm_archive_emit_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "lowered IREE VM function symbol is invalid");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "lowered IREE VM function symbol name is invalid");
  }
  *out_name = module->strings.entries[symbol->name_id];
  return iree_ok_status();
}

typedef struct loom_ireevm_archive_emit_state_t {
  // Module being emitted.
  loom_module_t* module;
  // Caller-provided options, or NULL for defaults.
  const loom_ireevm_archive_emit_options_t* options;
  // Host allocator used for transient buffers and output ownership.
  iree_allocator_t allocator;
  // True after arena-backed state has been initialized.
  bool initialized;
  // Optional caller-owned structured compile report.
  loom_target_compile_report_t* report;
  // Target-neutral entry options derived from |options|.
  loom_target_entry_options_t target_options;
  // Normalized diagnostic cap used by target phases.
  uint32_t max_errors;
  // Target provider environment used to compose target-owned registries.
  loom_target_environment_t target_environment;
  // Target-low descriptor registry used for selection, verification, and
  // emission.
  loom_target_low_descriptor_registry_t low_registry;
  // Diagnostic materializer shared by target-low verification and emission.
  loom_target_entry_diagnostic_emitter_t diagnostic_emitter;
  // Predicate selecting target bundles the IREE VM archive path can emit.
  loom_target_entry_predicate_t entry_predicate;
  // Block pool backing all per-emission arena allocations.
  iree_arena_block_pool_t block_pool;
  // Arena for target selection, frames, and tables.
  iree_arena_allocator_t table_arena;
  // Builder for the VM calling convention string.
  iree_string_builder_t calling_convention_builder;
  // Function bytecode emitted before archive wrapping.
  loom_ireevm_function_bytecode_t bytecode;
} loom_ireevm_archive_emit_state_t;

static void loom_ireevm_archive_emit_state_deinitialize(
    loom_ireevm_archive_emit_state_t* state) {
  loom_ireevm_function_bytecode_deinitialize(&state->bytecode,
                                             state->allocator);
  iree_arena_deinitialize(&state->table_arena);
  iree_arena_block_pool_deinitialize(&state->block_pool);
  iree_string_builder_deinitialize(&state->calling_convention_builder);
  loom_target_environment_deinitialize(&state->target_environment);
  state->initialized = false;
}

static iree_status_t loom_ireevm_archive_emit_state_initialize(
    loom_module_t* module, const loom_ireevm_archive_emit_options_t* options,
    iree_allocator_t allocator, loom_ireevm_archive_emit_state_t* out_state) {
  *out_state = (loom_ireevm_archive_emit_state_t){
      .module = module,
      .options = options,
      .allocator = allocator,
      .report = options ? options->report : NULL,
      .target_options =
          {
              .entry_symbol =
                  options ? options->entry_symbol : iree_string_view_empty(),
              .diagnostic_sink = options ? options->diagnostic_sink
                                         : (loom_diagnostic_sink_t){0},
              .source_resolver = options ? options->source_resolver
                                         : (loom_source_resolver_t){0},
              .max_errors = options ? options->max_errors : 0,
          },
      .entry_predicate =
          {
              .fn = loom_ireevm_archive_emit_bundle_is_compatible,
              .user_data = NULL,
          },
  };
  out_state->max_errors = loom_target_entry_max_errors(
      &out_state->target_options, LOOM_IREEVM_ARCHIVE_EMIT_DEFAULT_MAX_ERRORS);

  if (out_state->report != NULL) {
    loom_target_compile_report_initialize(out_state->report);
    loom_target_compile_report_set_row_storage(
        out_state->report, options ? &options->report_row_storage : NULL);
    out_state->report->artifact_kind =
        LOOM_TARGET_COMPILE_ARTIFACT_KIND_VM_ARCHIVE;
    out_state->report->module_name =
        loom_ireevm_archive_emit_module_name(options);
    out_state->report->entry_symbol =
        options ? options->entry_symbol : iree_string_view_empty();
  }

  iree_string_builder_initialize(allocator,
                                 &out_state->calling_convention_builder);
  iree_arena_block_pool_initialize(32 * 1024, allocator,
                                   &out_state->block_pool);
  iree_arena_initialize(&out_state->block_pool, &out_state->table_arena);
  out_state->initialized = true;

  iree_status_t status = loom_target_environment_initialize(
      &loom_ireevm_target_provider_set, &out_state->target_environment);
  if (iree_status_is_ok(status)) {
    status = loom_target_environment_initialize_low_descriptor_registry(
        &out_state->target_environment, &out_state->low_registry);
  }
  if (iree_status_is_ok(status)) {
    loom_target_entry_diagnostic_emitter_initialize(
        module, &out_state->target_options, LOOM_EMITTER_VERIFIER,
        &out_state->diagnostic_emitter);
  } else {
    loom_ireevm_archive_emit_state_deinitialize(out_state);
  }
  return status;
}

static iree_status_t loom_ireevm_archive_emit_select_entry(
    loom_ireevm_archive_emit_state_t* state, bool* out_selected,
    loom_target_entry_t* out_entry) {
  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_module(
      state->module, &state->target_options,
      LOOM_IREEVM_ARCHIVE_EMIT_DEFAULT_MAX_ERRORS, &verify_result));
  if (verify_result.error_count != 0) {
    *out_selected = false;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_target_entry_select_entry(
      state->module, &state->target_options, state->entry_predicate,
      &state->diagnostic_emitter, IREE_SV("IREE VM"), &state->table_arena,
      out_selected, out_entry));
  if (!*out_selected) {
    return iree_ok_status();
  }
  if (state->report != NULL) {
    loom_target_compile_report_record_target_bundle(
        state->report, &out_entry->bundle_storage.bundle);
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_archive_emit_selected_entry(
    loom_ireevm_archive_emit_state_t* state, const loom_target_entry_t* entry,
    loom_ireevm_module_archive_t* out_archive, bool* out_emitted) {
  IREE_RETURN_IF_ERROR(loom_ireevm_archive_emit_build_cconv(
      state->module, entry->func, &state->calling_convention_builder));

  if (state->report != NULL) {
    IREE_RETURN_IF_ERROR(loom_ireevm_archive_emit_symbol_name(
        state->module, entry->func_ref, &state->report->lowered_symbol));
  }

  loom_verify_result_t verify_result = {0};
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_module(
      state->module, &state->target_options,
      LOOM_IREEVM_ARCHIVE_EMIT_DEFAULT_MAX_ERRORS, &verify_result));
  if (verify_result.error_count != 0) {
    return iree_ok_status();
  }

  loom_low_verify_result_t low_verify_result = {0};
  IREE_RETURN_IF_ERROR(loom_target_entry_verify_low_module(
      state->module, &state->low_registry, &state->diagnostic_emitter,
      state->max_errors, &low_verify_result));
  if (low_verify_result.error_count != 0) {
    return iree_ok_status();
  }

  loom_low_emission_frame_t frame = {0};
  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = &state->low_registry.registry,
      .memory_access_table = loom_low_memory_access_table_empty(),
      .emitter = loom_target_entry_emitter(&state->diagnostic_emitter),
  };
  IREE_RETURN_IF_ERROR(loom_low_emission_frame_build(
      state->module, entry->func.op, &frame_options, &state->table_arena,
      &frame));
  if (state->report != NULL) {
    IREE_RETURN_IF_ERROR(loom_target_compile_report_record_low_emission_frame(
        state->report, &frame));
  }

  IREE_RETURN_IF_ERROR(loom_ireevm_emit_function_bytecode(
      &frame.schedule, &frame.allocation, state->allocator, &state->bytecode));
  if (state->report != NULL) {
    loom_target_compile_report_record_emission(
        state->report,
        frame.schedule.scheduled_node_count + frame.schedule.block_count,
        state->bytecode.bytecode_length, state->bytecode.data_length);
  }

  const loom_ireevm_module_archive_function_t functions[] = {
      {
          .export_name = loom_ireevm_archive_emit_export_name(entry),
          .calling_convention =
              iree_string_builder_view(&state->calling_convention_builder),
          .bytecode = &state->bytecode,
      },
  };
  IREE_RETURN_IF_ERROR(loom_ireevm_emit_module_archive(
      loom_ireevm_archive_emit_module_name(state->options), functions,
      IREE_ARRAYSIZE(functions), state->allocator, out_archive));
  if (state->report != NULL) {
    loom_target_compile_report_record_artifact_size(state->report,
                                                    out_archive->data_length);
  }
  *out_emitted = true;
  return iree_ok_status();
}

static iree_status_t loom_ireevm_archive_emit(
    loom_ireevm_archive_emit_state_t* state,
    loom_ireevm_module_archive_t* out_archive, bool* out_emitted) {
  loom_target_entry_t entry = {0};
  bool selected = false;
  IREE_RETURN_IF_ERROR(
      loom_ireevm_archive_emit_select_entry(state, &selected, &entry));
  if (!selected) {
    return iree_ok_status();
  }
  return loom_ireevm_archive_emit_selected_entry(state, &entry, out_archive,
                                                 out_emitted);
}

iree_status_t loom_ireevm_emit_module_archive_from_ir(
    loom_module_t* module, const loom_ireevm_archive_emit_options_t* options,
    iree_allocator_t allocator, bool* out_emitted,
    loom_ireevm_module_archive_t* out_archive) {
  *out_emitted = false;
  *out_archive = (loom_ireevm_module_archive_t){0};

  loom_ireevm_archive_emit_state_t state = {0};
  iree_status_t status = loom_ireevm_archive_emit_state_initialize(
      module, options, allocator, &state);
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_archive_emit(&state, out_archive, out_emitted);
  }
  if (!iree_status_is_ok(status)) {
    loom_ireevm_module_archive_deinitialize(out_archive, allocator);
  }
  if (state.report != NULL) {
    loom_target_compile_report_record_status(state.report, status);
  }
  if (state.initialized) {
    loom_ireevm_archive_emit_state_deinitialize(&state);
  }
  return status;
}
