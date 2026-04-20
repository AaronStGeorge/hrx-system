// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/module_compiler.h"

#include <inttypes.h>

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/ops.h"
#include "loom/target/emit/ireevm/function_bytecode.h"
#include "loom/target/emit/ireevm/low_registry.h"
#include "loom/target/emit/ireevm/lower.h"
#include "loom/target/ir_records.h"
#include "loom/target/presets.h"

enum {
  LOOM_IREEVM_MODULE_COMPILE_DEFAULT_MAX_ERRORS = 20,
};

typedef struct loom_ireevm_module_compile_target_t {
  // Materialized target.bundle records selected for this compilation.
  loom_target_ir_bundle_storage_t bundle_storage;
  // Module-local symbol reference for the selected target.bundle op.
  loom_symbol_ref_t target_ref;
} loom_ireevm_module_compile_target_t;

typedef struct loom_ireevm_module_compile_diagnostic_emitter_t {
  // Module containing op locations referenced by emitted diagnostics.
  const loom_module_t* module;
  // Source resolver for original source-backed locations.
  loom_source_resolver_t source_resolver;
  // Final diagnostic sink that owns rendering or capture policy.
  loom_diagnostic_sink_t diagnostic_sink;
  // Subsystem identity stored in materialized diagnostics.
  loom_emitter_t emitter;
} loom_ireevm_module_compile_diagnostic_emitter_t;

static uint32_t loom_ireevm_module_compile_max_errors(
    const loom_ireevm_module_compile_options_t* options) {
  if (options && options->max_errors != 0) {
    return options->max_errors;
  }
  return LOOM_IREEVM_MODULE_COMPILE_DEFAULT_MAX_ERRORS;
}

static iree_string_view_t loom_ireevm_module_compile_module_name(
    const loom_ireevm_module_compile_options_t* options) {
  if (options && !iree_string_view_is_empty(options->module_name)) {
    return options->module_name;
  }
  return IREE_SV("loom");
}

static iree_string_view_t loom_ireevm_module_compile_target_symbol_name(
    const loom_ireevm_module_compile_options_t* options) {
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

static bool loom_ireevm_module_compile_resolve_emission_location(
    const loom_ireevm_module_compile_diagnostic_emitter_t* emitter,
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

static iree_host_size_t loom_ireevm_module_compile_collect_related_locations(
    const loom_ireevm_module_compile_diagnostic_emitter_t* emitter,
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
    if (!loom_ireevm_module_compile_resolve_emission_location(
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

static iree_status_t loom_ireevm_module_compile_emit_diagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loom_ireevm_module_compile_diagnostic_emitter_t* emitter =
      (loom_ireevm_module_compile_diagnostic_emitter_t*)user_data;
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
      loom_ireevm_module_compile_collect_related_locations(
          emitter, emission->related_ops, emission->related_op_count,
          related_locations);
  if (diagnostic.related_location_count > 0) {
    diagnostic.related_locations = related_locations;
  }

  if (loom_ireevm_module_compile_resolve_emission_location(
          emitter, emission->op, &diagnostic.source_location)) {
    diagnostic.origin = diagnostic.source_location;
  }
  return loom_diagnostic_emit(&emitter->diagnostic_sink, &diagnostic);
}

static iree_diagnostic_emitter_t loom_ireevm_module_compile_emitter(
    loom_ireevm_module_compile_diagnostic_emitter_t* emitter) {
  return (iree_diagnostic_emitter_t){
      .fn = loom_ireevm_module_compile_emit_diagnostic,
      .user_data = emitter,
  };
}

static iree_status_t loom_ireevm_module_compile_expand_presets(
    loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry) {
  const loom_target_preset_registry_t preset_registry =
      loom_target_low_descriptor_registry_presets(low_registry);
  iree_host_size_t expanded_count = 0;
  return loom_target_expand_presets(module, &preset_registry, &expanded_count);
}

static iree_status_t loom_ireevm_module_compile_verify_module(
    const loom_module_t* module,
    const loom_ireevm_module_compile_options_t* options) {
  const loom_verify_options_t verify_options = {
      .sink = options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .max_errors = loom_ireevm_module_compile_max_errors(options),
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

static iree_status_t loom_ireevm_module_compile_verify_low_module(
    const loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_ireevm_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    uint32_t max_errors) {
  const loom_low_verify_options_t low_verify_options = {
      .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
      .descriptor_registry = &low_registry->registry,
      .descriptor_requirements =
          LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
      .emitter = loom_ireevm_module_compile_emitter(diagnostic_emitter),
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

static bool loom_ireevm_module_compile_bundle_is_compatible(
    const loom_target_bundle_t* bundle) {
  return bundle && bundle->snapshot && bundle->export_plan &&
         bundle->snapshot->codegen_format == LOOM_TARGET_CODEGEN_FORMAT_VM &&
         bundle->snapshot->artifact_format ==
             LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE &&
         bundle->export_plan->abi_kind == LOOM_TARGET_ABI_VM_MODULE_FUNCTION;
}

static iree_status_t loom_ireevm_module_compile_find_symbol_by_name(
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

static iree_status_t loom_ireevm_module_compile_select_named_target(
    const loom_module_t* module, iree_string_view_t target_symbol,
    loom_ireevm_module_compile_target_t* out_target) {
  uint16_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_ireevm_module_compile_find_symbol_by_name(
      module, target_symbol, &target_symbol_id));
  IREE_RETURN_IF_ERROR(loom_target_ir_bundle_from_symbol_name(
      module, target_symbol, &out_target->bundle_storage));
  if (!loom_ireevm_module_compile_bundle_is_compatible(
          &out_target->bundle_storage.bundle)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target bundle @%.*s does not produce an IREE VM bytecode module "
        "function",
        (int)target_symbol.size, target_symbol.data);
  }
  out_target->target_ref =
      (loom_symbol_ref_t){.module_id = 0, .symbol_id = target_symbol_id};
  return iree_ok_status();
}

static iree_status_t loom_ireevm_module_compile_select_single_target(
    const loom_module_t* module,
    loom_ireevm_module_compile_target_t* out_target) {
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
    if (!loom_ireevm_module_compile_bundle_is_compatible(&storage.bundle)) {
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
                            "module contains no IREE VM target bundle");
  }
  if (candidate_count > 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module contains %" PRIhsz
                            " IREE VM target bundles; pass --loom-target=@name",
                            candidate_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_ireevm_module_compile_select_target(
    const loom_module_t* module,
    const loom_ireevm_module_compile_options_t* options,
    loom_ireevm_module_compile_target_t* out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = (loom_ireevm_module_compile_target_t){0};
  iree_string_view_t target_symbol =
      loom_ireevm_module_compile_target_symbol_name(options);
  if (!iree_string_view_is_empty(target_symbol)) {
    return loom_ireevm_module_compile_select_named_target(module, target_symbol,
                                                          out_target);
  }
  return loom_ireevm_module_compile_select_single_target(module, out_target);
}

static iree_status_t loom_ireevm_module_compile_find_source_function(
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
  IREE_RETURN_IF_ERROR(loom_ireevm_module_compile_find_symbol_by_name(
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
      function.op->kind != LOOM_OP_FUNC_DEF) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source symbol @%.*s is not a func.def with a body",
                            (int)bundle->export_plan->source_symbol.size,
                            bundle->export_plan->source_symbol.data);
  }
  *out_function = function;
  return iree_ok_status();
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
    const loom_target_bundle_t* bundle) {
  if (bundle && bundle->export_plan &&
      !iree_string_view_is_empty(bundle->export_plan->export_symbol)) {
    return bundle->export_plan->export_symbol;
  }
  if (bundle && bundle->export_plan) {
    return bundle->export_plan->source_symbol;
  }
  return iree_string_view_empty();
}

static iree_status_t loom_ireevm_module_compile_lower_function(
    loom_module_t* module,
    const loom_target_low_descriptor_registry_t* registry,
    const loom_ireevm_module_compile_target_t* target,
    loom_func_like_t source_function,
    loom_ireevm_module_compile_diagnostic_emitter_t* diagnostic_emitter,
    uint32_t max_errors, loom_low_lower_result_t* out_result) {
  loom_low_lower_policy_registry_t policy_registry = {0};
  loom_ireevm_low_lower_policy_registry_initialize(&policy_registry);
  const loom_low_lower_policy_t* policy = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_policy_registry_lookup_for_bundle(
      &policy_registry, &target->bundle_storage.bundle, &policy));

  const loom_low_lower_options_t lower_options = {
      .target_ref = target->target_ref,
      .bundle = &target->bundle_storage.bundle,
      .descriptor_registry = &registry->registry,
      .descriptor_requirements =
          LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
      .policy = policy,
      .emitter = loom_ireevm_module_compile_emitter(diagnostic_emitter),
      .max_errors = max_errors,
  };
  IREE_RETURN_IF_ERROR(loom_low_lower_function(module, source_function,
                                               &lower_options, out_result));
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
  if (!module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "module is required");
  }

  const uint32_t max_errors = loom_ireevm_module_compile_max_errors(options);
  loom_target_low_descriptor_registry_t low_registry = {0};
  loom_ireevm_low_descriptor_registry_initialize(&low_registry);
  loom_ireevm_module_compile_diagnostic_emitter_t diagnostic_emitter = {
      .module = module,
      .source_resolver =
          options ? options->source_resolver : (loom_source_resolver_t){0},
      .diagnostic_sink =
          options ? options->diagnostic_sink : (loom_diagnostic_sink_t){0},
      .emitter = LOOM_EMITTER_VERIFIER,
  };

  loom_ireevm_module_compile_target_t target = {0};
  loom_func_like_t source_function = {0};
  loom_low_lower_result_t lower_result = {0};
  loom_ireevm_function_bytecode_t bytecode = {0};
  iree_string_builder_t calling_convention_builder;
  iree_string_builder_initialize(allocator, &calling_convention_builder);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(32 * 1024, allocator, &block_pool);
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool, &sidecar_arena);

  iree_status_t status =
      loom_ireevm_module_compile_expand_presets(module, &low_registry);
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_module_compile_verify_module(module, options);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_module_compile_select_target(module, options, &target);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_module_compile_find_source_function(
        module, &target.bundle_storage.bundle, &source_function);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_module_compile_build_cconv(
        module, source_function, &calling_convention_builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_module_compile_lower_function(
        module, &low_registry, &target, source_function, &diagnostic_emitter,
        max_errors, &lower_result);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_module_compile_verify_module(module, options);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_module_compile_verify_low_module(
        module, &low_registry, &diagnostic_emitter, max_errors);
  }
  loom_low_packetization_t packetization = {0};
  if (iree_status_is_ok(status)) {
    const loom_low_packetization_options_t packetization_options = {
        .descriptor_registry = &low_registry.registry,
        .emitter = loom_ireevm_module_compile_emitter(&diagnostic_emitter),
    };
    status = loom_low_packetize_function(module, lower_result.low_func_op,
                                         &packetization_options, &sidecar_arena,
                                         &packetization);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ireevm_emit_function_bytecode(&packetization.schedule,
                                                &packetization.allocation,
                                                allocator, &bytecode);
  }
  if (iree_status_is_ok(status)) {
    const loom_ireevm_module_archive_function_t functions[] = {
        {
            .export_name = loom_ireevm_module_compile_export_name(
                &target.bundle_storage.bundle),
            .calling_convention =
                iree_string_builder_view(&calling_convention_builder),
            .bytecode = &bytecode,
        },
    };
    status = loom_ireevm_emit_module_archive(
        loom_ireevm_module_compile_module_name(options), functions,
        IREE_ARRAYSIZE(functions), allocator, out_archive);
  }

  if (!iree_status_is_ok(status)) {
    loom_ireevm_module_archive_deinitialize(out_archive, allocator);
  }
  loom_ireevm_function_bytecode_deinitialize(&bytecode, allocator);
  iree_arena_deinitialize(&sidecar_arena);
  iree_arena_block_pool_deinitialize(&block_pool);
  iree_string_builder_deinitialize(&calling_convention_builder);
  return status;
}
