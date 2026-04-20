// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/module_compiler.h"

#include <inttypes.h>

#include "loom/codegen/low/verify.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/presets.h"

uint32_t loom_target_module_compile_max_errors(
    const loom_target_module_compile_options_t* options,
    uint32_t default_max_errors) {
  if (options && options->max_errors != 0) {
    return options->max_errors;
  }
  return default_max_errors;
}

iree_string_view_t loom_target_module_compile_target_symbol_name(
    const loom_target_module_compile_options_t* options) {
  if (!options) {
    return iree_string_view_empty();
  }
  iree_string_view_t target_symbol =
      iree_string_view_trim(options->target_symbol);
  if (iree_string_view_starts_with_char(target_symbol, '@')) {
    target_symbol = iree_string_view_remove_prefix(target_symbol, 1);
  }
  return target_symbol;
}

void loom_target_module_compile_diagnostic_emitter_initialize(
    const loom_module_t* module,
    const loom_target_module_compile_options_t* options, loom_emitter_t emitter,
    loom_target_module_compile_diagnostic_emitter_t* out_emitter) {
  IREE_ASSERT_ARGUMENT(out_emitter);
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
    loom_diagnostic_related_location_t* out_related_locations) {
  if (!related_ops || related_op_count == 0) {
    return 0;
  }
  iree_host_size_t related_location_count = 0;
  for (iree_host_size_t i = 0;
       i < related_op_count &&
       related_location_count < LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS;
       ++i) {
    loom_source_range_t source_location = {
        .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
    };
    if (!loom_target_module_compile_resolve_emission_location(
            emitter, related_ops[i].op, &source_location)) {
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
  loom_target_module_compile_diagnostic_emitter_t* emitter =
      (loom_target_module_compile_diagnostic_emitter_t*)user_data;
  if (!emitter || !emission || !emission->error) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostic emitter requires an emission");
  }

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

  loom_diagnostic_related_location_t
      related_locations[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  diagnostic.related_location_count =
      loom_target_module_compile_collect_related_locations(
          emitter, emission->related_ops, emission->related_op_count,
          related_locations);
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

iree_status_t loom_target_module_compile_expand_presets(
    loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry) {
  const loom_target_preset_registry_t preset_registry =
      loom_target_low_descriptor_registry_presets(low_registry);
  iree_host_size_t expanded_count = 0;
  return loom_target_expand_presets(module, &preset_registry, &expanded_count);
}

iree_status_t loom_target_module_compile_verify_module(
    const loom_module_t* module,
    const loom_target_module_compile_options_t* options,
    uint32_t default_max_errors) {
  const loom_verify_options_t verify_options = {
      .sink = options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .max_errors =
          loom_target_module_compile_max_errors(options, default_max_errors),
      .source_resolver =
          options ? options->source_resolver : (loom_source_resolver_t){0},
  };
  loom_verify_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_verify_module(module, &verify_options, &result));
  if (result.error_count > 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module verification failed with %" PRIu32 " error%s",
        result.error_count, result.error_count == 1 ? "" : "s");
  }
  return iree_ok_status();
}

iree_status_t loom_target_module_compile_verify_low_module(
    const loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_low_descriptor_requirement_flags_t descriptor_requirements,
    loom_target_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    uint32_t max_errors) {
  const loom_low_verify_options_t low_verify_options = {
      .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
      .descriptor_registry = &low_registry->registry,
      .descriptor_requirements = descriptor_requirements,
      .emitter = loom_target_module_compile_emitter(diagnostic_emitter),
      .max_errors = max_errors,
  };
  loom_low_verify_result_t result = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_verify_module(module, &low_verify_options, &result));
  if (result.error_count > 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low verification failed with %" PRIu32 " error%s",
                            result.error_count,
                            result.error_count == 1 ? "" : "s");
  }
  return iree_ok_status();
}

iree_status_t loom_target_module_compile_find_symbol_by_name(
    const loom_module_t* module, iree_string_view_t symbol_name,
    uint16_t* out_symbol_id) {
  IREE_ASSERT_ARGUMENT(out_symbol_id);
  *out_symbol_id = LOOM_SYMBOL_ID_INVALID;
  const loom_string_id_t symbol_name_id =
      loom_module_lookup_string(module, symbol_name);
  if (symbol_name_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "symbol @%.*s was not found",
                            (int)symbol_name.size, symbol_name.data);
  }
  const uint16_t symbol_id = loom_module_find_symbol(module, symbol_name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "symbol @%.*s was not found",
                            (int)symbol_name.size, symbol_name.data);
  }
  *out_symbol_id = symbol_id;
  return iree_ok_status();
}

static iree_status_t loom_target_module_compile_select_named_target(
    const loom_module_t* module, iree_string_view_t target_symbol,
    loom_target_module_compile_bundle_predicate_fn_t predicate,
    void* predicate_user_data, iree_string_view_t target_kind,
    loom_target_module_compile_target_t* out_target) {
  uint16_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_target_module_compile_find_symbol_by_name(
      module, target_symbol, &target_symbol_id));
  IREE_RETURN_IF_ERROR(loom_target_ir_bundle_from_symbol_name(
      module, target_symbol, &out_target->bundle_storage));
  if (!predicate(predicate_user_data, &out_target->bundle_storage.bundle)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target bundle @%.*s is not compatible with %.*s",
                            (int)target_symbol.size, target_symbol.data,
                            (int)target_kind.size, target_kind.data);
  }
  out_target->target_ref =
      (loom_symbol_ref_t){.module_id = 0, .symbol_id = target_symbol_id};
  return iree_ok_status();
}

static iree_status_t loom_target_module_compile_select_single_target(
    const loom_module_t* module,
    loom_target_module_compile_bundle_predicate_fn_t predicate,
    void* predicate_user_data, iree_string_view_t target_kind,
    loom_target_module_compile_target_t* out_target) {
  iree_host_size_t candidate_count = 0;

  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    if (i > UINT16_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "symbol index exceeds target ref range");
    }
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!symbol->defining_op || !loom_target_bundle_isa(symbol->defining_op)) {
      continue;
    }
    if (symbol->name_id >= module->strings.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target bundle has invalid symbol name id");
    }
    iree_string_view_t symbol_name = module->strings.entries[symbol->name_id];
    loom_target_ir_bundle_storage_t storage = {0};
    IREE_RETURN_IF_ERROR(
        loom_target_ir_bundle_from_symbol_name(module, symbol_name, &storage));
    if (!predicate(predicate_user_data, &storage.bundle)) {
      continue;
    }

    ++candidate_count;
    if (candidate_count == 1) {
      out_target->bundle_storage = storage;
      loom_target_ir_bundle_storage_rebind(&out_target->bundle_storage);
      out_target->target_ref =
          (loom_symbol_ref_t){.module_id = 0, .symbol_id = (uint16_t)i};
    }
  }

  if (candidate_count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "module contains no %.*s target bundle",
                            (int)target_kind.size, target_kind.data);
  }
  if (candidate_count > 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module contains %" PRIhsz
                            " %.*s target bundles; pass --loom-target=@name",
                            candidate_count, (int)target_kind.size,
                            target_kind.data);
  }
  return iree_ok_status();
}

iree_status_t loom_target_module_compile_select_target(
    const loom_module_t* module,
    const loom_target_module_compile_options_t* options,
    loom_target_module_compile_bundle_predicate_fn_t predicate,
    void* predicate_user_data, iree_string_view_t target_kind,
    loom_target_module_compile_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(predicate);
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = (loom_target_module_compile_target_t){0};
  iree_string_view_t target_symbol =
      loom_target_module_compile_target_symbol_name(options);
  if (!iree_string_view_is_empty(target_symbol)) {
    return loom_target_module_compile_select_named_target(
        module, target_symbol, predicate, predicate_user_data, target_kind,
        out_target);
  }
  return loom_target_module_compile_select_single_target(
      module, predicate, predicate_user_data, target_kind, out_target);
}

iree_status_t loom_target_module_compile_find_source_function(
    const loom_module_t* module, const loom_target_bundle_t* bundle,
    loom_func_like_t* out_function) {
  IREE_ASSERT_ARGUMENT(out_function);
  *out_function = (loom_func_like_t){0};
  if (!bundle || !bundle->export_plan ||
      iree_string_view_is_empty(bundle->export_plan->source_symbol)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected target bundle does not name an export source function");
  }
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_target_module_compile_find_symbol_by_name(
      module, bundle->export_plan->source_symbol, &symbol_id));
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  if (!symbol->defining_op) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "source function @%.*s has no definition",
                            (int)bundle->export_plan->source_symbol.size,
                            bundle->export_plan->source_symbol.data);
  }
  loom_func_like_t function = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(function) || !loom_func_like_body(function) ||
      (function.op->kind != LOOM_OP_FUNC_DEF &&
       function.op->kind != LOOM_OP_LOW_FUNC_DEF)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source symbol @%.*s is not a func.def or low.func.def with a body",
        (int)bundle->export_plan->source_symbol.size,
        bundle->export_plan->source_symbol.data);
  }
  *out_function = function;
  return iree_ok_status();
}
