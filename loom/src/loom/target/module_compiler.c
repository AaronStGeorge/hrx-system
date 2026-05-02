// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/module_compiler.h"

#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/target/artifact_plan.h"
#include "loom/target/function_contract.h"
#include "loom/util/call_graph.h"

uint32_t loom_target_module_compile_max_errors(
    const loom_target_module_compile_options_t* options,
    uint32_t default_max_errors) {
  if (options && options->max_errors != 0) {
    return options->max_errors;
  }
  return default_max_errors;
}

iree_string_view_t loom_target_module_compile_normalize_symbol_name(
    iree_string_view_t symbol_name) {
  symbol_name = iree_string_view_trim(symbol_name);
  if (iree_string_view_starts_with_char(symbol_name, '@')) {
    symbol_name = iree_string_view_remove_prefix(symbol_name, 1);
  }
  return symbol_name;
}

iree_string_view_t loom_target_module_compile_entry_symbol_name(
    const loom_target_module_compile_options_t* options) {
  if (!options) {
    return iree_string_view_empty();
  }
  return loom_target_module_compile_normalize_symbol_name(
      options->entry_symbol);
}

static iree_string_view_t loom_target_module_compile_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static iree_string_view_t loom_target_module_compile_codegen_format_name(
    loom_target_codegen_format_t format) {
  switch (format) {
    case LOOM_TARGET_CODEGEN_FORMAT_LLVMIR:
      return IREE_SV("llvmir");
    case LOOM_TARGET_CODEGEN_FORMAT_SPIRV:
      return IREE_SV("spirv");
    case LOOM_TARGET_CODEGEN_FORMAT_VM:
      return IREE_SV("vm");
    case LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE:
      return IREE_SV("low_native");
    case LOOM_TARGET_CODEGEN_FORMAT_WASM:
      return IREE_SV("wasm");
    case LOOM_TARGET_CODEGEN_FORMAT_UNKNOWN:
      return IREE_SV("unknown");
  }
  return IREE_SV("unknown");
}

static iree_string_view_t loom_target_module_compile_artifact_format_name(
    loom_target_artifact_format_t format) {
  switch (format) {
    case LOOM_TARGET_ARTIFACT_FORMAT_ELF:
      return IREE_SV("elf");
    case LOOM_TARGET_ARTIFACT_FORMAT_COFF:
      return IREE_SV("coff");
    case LOOM_TARGET_ARTIFACT_FORMAT_MACHO:
      return IREE_SV("macho");
    case LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY:
      return IREE_SV("spirv_binary");
    case LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE:
      return IREE_SV("vm_bytecode");
    case LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY:
      return IREE_SV("wasm_binary");
    case LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN:
      return IREE_SV("unknown");
  }
  return IREE_SV("unknown");
}

static iree_string_view_t loom_target_module_compile_abi_kind_name(
    loom_target_abi_kind_t abi_kind) {
  switch (abi_kind) {
    case LOOM_TARGET_ABI_OBJECT_FUNCTION:
      return IREE_SV("object_function");
    case LOOM_TARGET_ABI_HAL_KERNEL:
      return IREE_SV("hal_kernel");
    case LOOM_TARGET_ABI_VM_MODULE_FUNCTION:
      return IREE_SV("vm_module_function");
    case LOOM_TARGET_ABI_SHADER_ENTRY_POINT:
      return IREE_SV("shader_entry_point");
    case LOOM_TARGET_ABI_WASM_FUNCTION:
      return IREE_SV("wasm_function");
    case LOOM_TARGET_ABI_UNKNOWN:
      return IREE_SV("unknown");
  }
  return IREE_SV("unknown");
}

static bool loom_target_module_compile_lookup_symbol_id(
    const loom_module_t* module, iree_string_view_t symbol_name,
    uint16_t* out_symbol_id) {
  *out_symbol_id = LOOM_SYMBOL_ID_INVALID;
  const loom_string_id_t symbol_name_id =
      loom_module_lookup_string(module, symbol_name);
  if (symbol_name_id == LOOM_STRING_ID_INVALID) {
    return false;
  }
  const uint16_t symbol_id = loom_module_find_symbol(module, symbol_name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return false;
  }
  *out_symbol_id = symbol_id;
  return true;
}

static iree_string_view_t loom_target_module_compile_symbol_name(
    const loom_module_t* module, loom_symbol_id_t symbol_id) {
  if (symbol_id >= module->symbols.count) {
    return IREE_SV("<unknown>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<unknown>");
  }
  return module->strings.entries[symbol->name_id];
}

static const loom_op_t* loom_target_module_compile_symbol_op(
    const loom_module_t* module, loom_symbol_id_t symbol_id) {
  if (symbol_id >= module->symbols.count) {
    return NULL;
  }
  return module->symbols.entries[symbol_id].defining_op;
}

void loom_target_module_compile_diagnostic_emitter_initialize(
    const loom_module_t* module,
    const loom_target_module_compile_options_t* options, loom_emitter_t emitter,
    loom_target_module_compile_diagnostic_emitter_t* out_emitter) {
  *out_emitter = (loom_target_module_compile_diagnostic_emitter_t){
      .module = module,
      .source_resolver =
          options ? options->source_resolver : (loom_source_resolver_t){0},
      .diagnostic_sink =
          options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .emitter = emitter,
  };
}

static bool loom_target_module_compile_resolve_emission_location(
    const loom_target_module_compile_diagnostic_emitter_t* emitter,
    const loom_op_t* op, loom_source_range_t* out_source_location) {
  if (!emitter || !emitter->module || !op) {
    return false;
  }
  if (!loom_source_resolve(emitter->source_resolver, emitter->module,
                           op->location, out_source_location)) {
    return false;
  }
  if (out_source_location->provenance ==
          LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE &&
      out_source_location->source.size > 0) {
    out_source_location->provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
  }
  return true;
}

static iree_host_size_t loom_target_module_compile_collect_related_locations(
    const loom_target_module_compile_diagnostic_emitter_t* emitter,
    const loom_diagnostic_related_op_t* related_ops,
    iree_host_size_t related_op_count,
    loom_diagnostic_related_location_t* out_related_locations,
    iree_host_size_t* out_omitted_count) {
  *out_omitted_count = 0;
  if (!related_ops || related_op_count == 0) {
    return 0;
  }
  iree_host_size_t related_location_count = 0;
  for (iree_host_size_t i = 0; i < related_op_count; ++i) {
    loom_source_range_t source_location = {
        .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
    };
    if (!loom_target_module_compile_resolve_emission_location(
            emitter, related_ops[i].op, &source_location)) {
      continue;
    }
    if (related_location_count >= LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS) {
      ++*out_omitted_count;
      continue;
    }
    out_related_locations[related_location_count++] =
        (loom_diagnostic_related_location_t){
            .label = related_ops[i].label,
            .source_location = source_location,
        };
  }
  return related_location_count;
}

static iree_status_t loom_target_module_compile_emit_diagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  IREE_ASSERT_ARGUMENT(user_data);
  IREE_ASSERT_ARGUMENT(emission);
  IREE_ASSERT(emission->error != NULL);
  loom_target_module_compile_diagnostic_emitter_t* emitter =
      (loom_target_module_compile_diagnostic_emitter_t*)user_data;

  loom_diagnostic_t diagnostic = {
      .severity = emission->error->severity,
      .error = emission->error,
      .params = emission->params,
      .param_count = emission->param_count,
      .emitter = emitter->emitter,
      .origin = {.provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE},
      .source_location = {.provenance =
                              LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE},
  };
  switch (diagnostic.severity) {
    case LOOM_DIAGNOSTIC_ERROR:
      ++emitter->error_count;
      break;
    case LOOM_DIAGNOSTIC_WARNING:
      ++emitter->warning_count;
      break;
    case LOOM_DIAGNOSTIC_REMARK:
      ++emitter->remark_count;
      break;
    case LOOM_DIAGNOSTIC_COUNT_:
      break;
  }

  loom_diagnostic_related_location_t
      related_locations[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  diagnostic.related_location_count =
      loom_target_module_compile_collect_related_locations(
          emitter, emission->related_ops, emission->related_op_count,
          related_locations, &diagnostic.related_location_omitted_count);
  if (diagnostic.related_location_count > 0) {
    diagnostic.related_locations = related_locations;
  }

  if (loom_target_module_compile_resolve_emission_location(
          emitter, emission->op, &diagnostic.source_location)) {
    diagnostic.origin = diagnostic.source_location;
  }
  return loom_diagnostic_emit(&emitter->diagnostic_sink, &diagnostic);
}

iree_diagnostic_emitter_t loom_target_module_compile_emitter(
    loom_target_module_compile_diagnostic_emitter_t* emitter) {
  return (iree_diagnostic_emitter_t){
      .fn = loom_target_module_compile_emit_diagnostic,
      .user_data = emitter,
  };
}

static iree_status_t loom_target_module_compile_emit(
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    const loom_op_t* op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(
      loom_target_module_compile_emitter(diagnostic_emitter), &emission);
}

iree_status_t loom_target_module_compile_verify_module(
    const loom_module_t* module,
    const loom_target_module_compile_options_t* options,
    uint32_t default_max_errors, loom_verify_result_t* out_result) {
  const loom_verify_options_t verify_options = {
      .sink = options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .max_errors =
          loom_target_module_compile_max_errors(options, default_max_errors),
      .source_resolver =
          options ? options->source_resolver : (loom_source_resolver_t){0},
  };
  *out_result = (loom_verify_result_t){0};
  return loom_verify_module(module, &verify_options, out_result);
}

iree_status_t loom_target_module_compile_verify_low_module(
    const loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    uint32_t max_errors, loom_low_verify_result_t* out_result) {
  const loom_low_verify_options_t low_verify_options = {
      .descriptor_registry = &low_registry->registry,
      .emitter = loom_target_module_compile_emitter(diagnostic_emitter),
      .max_errors = max_errors,
  };
  *out_result = (loom_low_verify_result_t){0};
  return loom_low_verify_module(module, &low_verify_options, out_result);
}

static iree_status_t loom_target_module_compile_initialize_fact_table(
    const loom_target_low_descriptor_registry_t* low_registry,
    iree_arena_allocator_t* arena, loom_symbol_fact_table_t* out_fact_table) {
  if (low_registry == NULL || low_registry->target_bundle_count == 0 ||
      low_registry->target_bundles == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "module compilation requires target profile presets");
  }
  loom_target_preset_registry_t* preset_registry = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*preset_registry),
                                           (void**)&preset_registry));
  *preset_registry = loom_target_low_descriptor_registry_presets(low_registry);

  loom_symbol_fact_resource_t* resource = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*resource), (void**)&resource));
  *resource = loom_target_profile_preset_registry_resource(preset_registry);
  const loom_symbol_fact_table_options_t fact_options = {
      .resources = loom_make_symbol_fact_resource_list(resource, 1),
  };
  loom_symbol_fact_table_initialize_with_options(out_fact_table, &fact_options,
                                                 arena);
  return iree_ok_status();
}

static iree_status_t loom_target_module_compile_lookup_func_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_id_t symbol_id,
    const loom_func_symbol_facts_t** out_func_facts) {
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(fact_table, module,
                                                     symbol_id, &base_facts));
  *out_func_facts = loom_func_symbol_facts_cast(base_facts);
  return iree_ok_status();
}

static void loom_target_module_compile_entry_from_facts(
    const loom_module_t* module, loom_symbol_id_t symbol_id,
    const loom_func_symbol_facts_t* func_facts,
    loom_target_module_compile_entry_t* out_entry) {
  out_entry->func = loom_func_like_cast(module, func_facts->func_op);
  out_entry->func_name = func_facts->name;
  out_entry->func_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  out_entry->target_ref = func_facts->target_symbol;
}

static void loom_target_module_compile_assign_entry(
    const loom_target_module_compile_entry_t* source,
    loom_target_module_compile_entry_t* out_entry) {
  *out_entry = *source;
  loom_target_bundle_storage_rebind(&out_entry->bundle_storage);
}

static iree_status_t loom_target_module_compile_emit_entry_not_found(
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t pipeline_name, iree_string_view_t symbol_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(pipeline_name),
      loom_param_string(symbol_name),
  };
  return loom_target_module_compile_emit(
      diagnostic_emitter, NULL,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TARGET, 20), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_target_module_compile_emit_artifact_not_found(
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t pipeline_name, iree_string_view_t artifact_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(pipeline_name),
      loom_param_string(artifact_name),
  };
  return loom_target_module_compile_emit(
      diagnostic_emitter, NULL,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TARGET, 21), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_target_module_compile_emit_entry_not_func(
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    const loom_op_t* op, iree_string_view_t pipeline_name,
    iree_string_view_t symbol_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(pipeline_name),
      loom_param_string(symbol_name),
  };
  return loom_target_module_compile_emit(
      diagnostic_emitter, op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TARGET, 22), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_target_module_compile_emit_missing_target_profile(
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    const loom_op_t* op, iree_string_view_t pipeline_name,
    iree_string_view_t function_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(pipeline_name),
      loom_param_string(function_name),
  };
  return loom_target_module_compile_emit(
      diagnostic_emitter, op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TARGET, 23), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_target_module_compile_emit_incompatible_bundle(
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    const loom_target_module_compile_entry_t* entry,
    iree_string_view_t pipeline_name) {
  const loom_target_bundle_t* bundle = &entry->bundle_storage.bundle;
  const loom_target_snapshot_t* snapshot = bundle->snapshot;
  const loom_target_export_plan_t* export_plan = bundle->export_plan;
  const loom_target_config_t* config = bundle->config;
  const loom_diagnostic_param_t params[] = {
      loom_param_string(pipeline_name),
      loom_param_string(loom_target_module_compile_nonempty(
          bundle->name, IREE_SV("<empty>"))),
      loom_param_string(loom_target_module_compile_nonempty(
          export_plan ? export_plan->name : iree_string_view_empty(),
          IREE_SV("<empty>"))),
      loom_param_string(loom_target_module_compile_nonempty(
          config ? config->name : iree_string_view_empty(),
          IREE_SV("<empty>"))),
      loom_param_string(entry->func_name),
      loom_param_string(loom_target_module_compile_codegen_format_name(
          snapshot ? snapshot->codegen_format
                   : LOOM_TARGET_CODEGEN_FORMAT_UNKNOWN)),
      loom_param_string(loom_target_module_compile_artifact_format_name(
          snapshot ? snapshot->artifact_format
                   : LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN)),
      loom_param_string(loom_target_module_compile_abi_kind_name(
          export_plan ? export_plan->abi_kind : LOOM_TARGET_ABI_UNKNOWN)),
      loom_param_string(loom_target_module_compile_nonempty(
          snapshot ? snapshot->target_triple : iree_string_view_empty(),
          IREE_SV("<empty>"))),
  };
  return loom_target_module_compile_emit(
      diagnostic_emitter, entry->func.op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TARGET, 24), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_target_module_compile_emit_no_compatible_entry(
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t pipeline_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(pipeline_name),
  };
  return loom_target_module_compile_emit(
      diagnostic_emitter, NULL,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TARGET, 25), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_target_module_compile_emit_ambiguous_entry(
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t pipeline_name, uint32_t candidate_count) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(pipeline_name),
      loom_param_u32(candidate_count),
  };
  return loom_target_module_compile_emit(
      diagnostic_emitter, NULL,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TARGET, 26), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_target_module_compile_emit_empty_artifact(
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    const loom_op_t* op, iree_string_view_t pipeline_name,
    iree_string_view_t artifact_name) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(pipeline_name),
      loom_param_string(artifact_name),
  };
  return loom_target_module_compile_emit(
      diagnostic_emitter, op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TARGET, 27), params,
      IREE_ARRAYSIZE(params));
}

iree_status_t loom_target_module_compile_emit_entry_artifact_conflict(
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t entry_kind, iree_string_view_t entry_symbol,
    iree_string_view_t artifact_symbol) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(entry_kind),
      loom_param_string(entry_symbol),
      loom_param_string(artifact_symbol),
  };
  return loom_target_module_compile_emit(
      diagnostic_emitter, NULL,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TARGET, 28), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_target_module_compile_try_entry(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_id_t symbol_id,
    loom_target_module_compile_entry_predicate_t predicate,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t pipeline_name, bool require_compatible,
    bool* out_compatible, loom_target_module_compile_entry_t* out_entry) {
  *out_compatible = false;
  const loom_func_symbol_facts_t* func_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_target_module_compile_lookup_func_facts(
      module, fact_table, symbol_id, &func_facts));
  if (!func_facts || !func_facts->has_body) {
    if (!require_compatible) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_target_module_compile_emit_entry_not_func(
        diagnostic_emitter,
        loom_target_module_compile_symbol_op(module, symbol_id), pipeline_name,
        loom_target_module_compile_symbol_name(module, symbol_id)));
    return iree_ok_status();
  }
  if (!loom_symbol_ref_is_valid(func_facts->target_symbol)) {
    if (!require_compatible) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_target_module_compile_emit_missing_target_profile(
        diagnostic_emitter, func_facts->func_op, pipeline_name,
        func_facts->name));
    return iree_ok_status();
  }

  loom_target_module_compile_entry_t entry = {0};
  loom_target_module_compile_entry_from_facts(module, symbol_id, func_facts,
                                              &entry);
  IREE_RETURN_IF_ERROR(loom_target_function_contract_resolve(
      module, fact_table, func_facts, &entry.bundle_storage));
  if (!predicate.fn(predicate.user_data, &entry)) {
    if (!require_compatible) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_target_module_compile_emit_incompatible_bundle(
        diagnostic_emitter, &entry, pipeline_name));
    return iree_ok_status();
  }

  loom_target_module_compile_assign_entry(&entry, out_entry);
  *out_compatible = true;
  return iree_ok_status();
}

static iree_status_t loom_target_module_compile_select_named_entry(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    iree_string_view_t entry_symbol,
    loom_target_module_compile_entry_predicate_t predicate,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t pipeline_name, bool* out_selected,
    loom_target_module_compile_entry_t* out_entry) {
  *out_selected = false;
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  if (!loom_target_module_compile_lookup_symbol_id(module, entry_symbol,
                                                   &symbol_id)) {
    return loom_target_module_compile_emit_entry_not_found(
        diagnostic_emitter, pipeline_name, entry_symbol);
  }
  bool compatible = false;
  IREE_RETURN_IF_ERROR(loom_target_module_compile_try_entry(
      module, fact_table, symbol_id, predicate, diagnostic_emitter,
      pipeline_name, /*require_compatible=*/true, &compatible, out_entry));
  *out_selected = compatible;
  return iree_ok_status();
}

static iree_status_t loom_target_module_compile_select_single_entry(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_target_module_compile_entry_predicate_t predicate,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t pipeline_name, bool* out_selected,
    loom_target_module_compile_entry_t* out_entry) {
  *out_selected = false;
  iree_host_size_t candidate_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (i > UINT16_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "symbol index exceeds entry ref range");
    }
    bool compatible = false;
    loom_target_module_compile_entry_t candidate = {0};
    IREE_RETURN_IF_ERROR(loom_target_module_compile_try_entry(
        module, fact_table, (loom_symbol_id_t)i, predicate, diagnostic_emitter,
        pipeline_name, /*require_compatible=*/false, &compatible, &candidate));
    if (!compatible) {
      continue;
    }
    ++candidate_count;
    if (candidate_count == 1) {
      loom_target_module_compile_assign_entry(&candidate, out_entry);
    }
  }

  if (candidate_count == 0) {
    return loom_target_module_compile_emit_no_compatible_entry(
        diagnostic_emitter, pipeline_name);
  }
  if (candidate_count > 1) {
    const uint32_t diagnostic_count =
        candidate_count > UINT32_MAX ? UINT32_MAX : (uint32_t)candidate_count;
    return loom_target_module_compile_emit_ambiguous_entry(
        diagnostic_emitter, pipeline_name, diagnostic_count);
  }
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_target_module_compile_select_entry(
    const loom_module_t* module,
    const loom_target_module_compile_options_t* options,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_target_module_compile_entry_predicate_t predicate,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t entry_kind, iree_arena_allocator_t* arena,
    bool* out_selected, loom_target_module_compile_entry_t* out_entry) {
  *out_entry = (loom_target_module_compile_entry_t){0};
  *out_selected = false;

  loom_symbol_fact_table_t fact_table = {0};
  IREE_RETURN_IF_ERROR(loom_target_module_compile_initialize_fact_table(
      low_registry, arena, &fact_table));

  iree_string_view_t entry_symbol =
      loom_target_module_compile_entry_symbol_name(options);
  if (!iree_string_view_is_empty(entry_symbol)) {
    return loom_target_module_compile_select_named_entry(
        module, &fact_table, entry_symbol, predicate, diagnostic_emitter,
        entry_kind, out_selected, out_entry);
  }
  return loom_target_module_compile_select_single_entry(
      module, &fact_table, predicate, diagnostic_emitter, entry_kind,
      out_selected, out_entry);
}

iree_status_t loom_target_module_compile_select_artifact_entries(
    const loom_module_t* module, iree_string_view_t artifact_symbol,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_target_module_compile_entry_predicate_t predicate,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    iree_string_view_t entry_kind, iree_arena_allocator_t* arena,
    bool* out_selected, loom_target_module_compile_entry_list_t* out_entries) {
  *out_entries = (loom_target_module_compile_entry_list_t){0};
  *out_selected = false;

  artifact_symbol =
      loom_target_module_compile_normalize_symbol_name(artifact_symbol);

  loom_symbol_fact_table_t fact_table = {0};
  IREE_RETURN_IF_ERROR(loom_target_module_compile_initialize_fact_table(
      low_registry, arena, &fact_table));

  uint16_t artifact_symbol_id = LOOM_SYMBOL_ID_INVALID;
  if (!loom_target_module_compile_lookup_symbol_id(module, artifact_symbol,
                                                   &artifact_symbol_id)) {
    return loom_target_module_compile_emit_artifact_not_found(
        diagnostic_emitter, entry_kind, artifact_symbol);
  }
  loom_call_graph_t call_graph = {0};
  IREE_RETURN_IF_ERROR(loom_call_graph_build(module, arena, &call_graph));
  loom_target_artifact_plan_t artifact_plan = {0};
  IREE_RETURN_IF_ERROR(loom_target_artifact_plan_build(
      module,
      (loom_symbol_ref_t){
          .module_id = 0,
          .symbol_id = artifact_symbol_id,
      },
      &fact_table, &call_graph, arena, &artifact_plan));
  if (artifact_plan.entry_count == 0) {
    return loom_target_module_compile_emit_empty_artifact(
        diagnostic_emitter,
        loom_target_module_compile_symbol_op(module, artifact_symbol_id),
        entry_kind, artifact_symbol);
  }

  loom_target_module_compile_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, artifact_plan.entry_count, sizeof(*entries), (void**)&entries));
  for (uint16_t i = 0; i < artifact_plan.entry_count; ++i) {
    bool compatible = false;
    IREE_RETURN_IF_ERROR(loom_target_module_compile_try_entry(
        module, &fact_table, artifact_plan.entry_symbol_ids[i], predicate,
        diagnostic_emitter, entry_kind, /*require_compatible=*/true,
        &compatible, &entries[i]));
    if (!compatible) {
      return iree_ok_status();
    }
  }

  *out_entries = (loom_target_module_compile_entry_list_t){
      .values = entries,
      .count = artifact_plan.entry_count,
  };
  *out_selected = true;
  return iree_ok_status();
}
