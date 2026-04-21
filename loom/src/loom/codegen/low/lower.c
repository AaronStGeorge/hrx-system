// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/builder.h"
#include "loom/error/error_defs.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"

struct loom_low_lower_context_t {
  // Module being mutated by this lowering run.
  loom_module_t* module;
  // Source function being lowered.
  loom_func_like_t source_function;
  // Caller-owned lowering options.
  const loom_low_lower_options_t* options;
  // Target-low lowering policy selected by the caller.
  const loom_low_lower_policy_t* policy;
  // Descriptor set selected by source legality.
  const loom_low_descriptor_set_t* descriptor_set;
  // Result object receiving counters and emitted low function metadata.
  loom_low_lower_result_t* result;
  // Scratch arena for transient maps and remapped operand lists.
  iree_arena_allocator_t arena;
  // Builder used while emitting the low function.
  loom_builder_t builder;
  // Emitted low.func.def operation, or NULL before emission starts.
  loom_op_t* low_func_op;
  // Number of source values captured by |value_map|.
  iree_host_size_t value_map_count;
  // Source value id to emitted low value id map.
  loom_value_id_t* value_map;
  // Source block ordinal to emitted low block pointer map.
  loom_block_t** block_map;
  // Source function argument ABI mappings.
  loom_low_lower_abi_argument_t* argument_map;
  // Number of entries in |argument_map|.
  uint16_t argument_map_count;
};

static bool loom_low_lower_type_is_none(loom_type_t type) {
  return loom_type_kind(type) == LOOM_TYPE_NONE;
}

static iree_string_view_t loom_low_lower_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static iree_string_view_t loom_low_lower_policy_name(
    const loom_low_lower_policy_t* policy) {
  if (!policy) {
    return IREE_SV("<missing>");
  }
  return loom_low_lower_nonempty(policy->name, IREE_SV("<unnamed>"));
}

static iree_status_t loom_low_lower_policy_verify(
    const loom_low_lower_policy_t* policy) {
  if (policy == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low lowering policy is required");
  }
  if (iree_string_view_is_empty(iree_string_view_trim(policy->name))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low lowering policy name is required");
  }
  if (policy->map_type.fn == NULL || policy->can_lower_op.fn == NULL ||
      policy->try_lower_op.fn == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "complete target-low lowering policy is required");
  }
  return iree_ok_status();
}

static bool loom_low_lower_abi_argument_kind_is_known(
    loom_low_lower_abi_argument_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT:
    case LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_resource_import_kind_is_known(
    loom_low_resource_import_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_STATE:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_IMPORT:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BUFFER_RESOURCE:
      return true;
    default:
      return false;
  }
}

void loom_low_lower_policy_registry_initialize_from_entries(
    loom_low_lower_policy_registry_t* out_registry,
    const loom_low_lower_policy_registry_entry_t* entries,
    iree_host_size_t entry_count) {
  IREE_ASSERT_ARGUMENT(out_registry);
  *out_registry = (loom_low_lower_policy_registry_t){
      .entries = entries,
      .entry_count = entry_count,
  };
}

static iree_status_t loom_low_lower_policy_registry_verify_tables(
    const loom_low_lower_policy_registry_t* registry) {
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low lowering policy registry is required");
  }
  if (registry->entry_count != 0 && registry->entries == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low lowering policy registry entries are required");
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_policy_registry_verify_entry(
    const loom_low_lower_policy_registry_entry_t* entry) {
  if (iree_string_view_is_empty(
          iree_string_view_trim(entry->contract_set_key))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low lowering policy contract key is "
                            "required");
  }
  return loom_low_lower_policy_verify(entry->policy);
}

iree_status_t loom_low_lower_policy_registry_verify(
    const loom_low_lower_policy_registry_t* registry) {
  IREE_RETURN_IF_ERROR(loom_low_lower_policy_registry_verify_tables(registry));
  for (iree_host_size_t i = 0; i < registry->entry_count; ++i) {
    const loom_low_lower_policy_registry_entry_t* entry = &registry->entries[i];
    IREE_RETURN_IF_ERROR(loom_low_lower_policy_registry_verify_entry(entry));
    for (iree_host_size_t j = i + 1; j < registry->entry_count; ++j) {
      const loom_low_lower_policy_registry_entry_t* other_entry =
          &registry->entries[j];
      IREE_RETURN_IF_ERROR(
          loom_low_lower_policy_registry_verify_entry(other_entry));
      if (iree_string_view_equal(entry->contract_set_key,
                                 other_entry->contract_set_key)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "target-low lowering policy registry has duplicate contract key "
            "'%.*s'",
            (int)entry->contract_set_key.size, entry->contract_set_key.data);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_policy_registry_lookup(
    const loom_low_lower_policy_registry_t* registry,
    iree_string_view_t contract_set_key,
    const loom_low_lower_policy_t** out_policy) {
  if (out_policy == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low lowering policy output is required");
  }
  *out_policy = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_policy_registry_verify_tables(registry));
  if (iree_string_view_is_empty(iree_string_view_trim(contract_set_key))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target-low lowering policy lookup contract key is required");
  }

  const loom_low_lower_policy_t* match = NULL;
  for (iree_host_size_t i = 0; i < registry->entry_count; ++i) {
    const loom_low_lower_policy_registry_entry_t* entry = &registry->entries[i];
    IREE_RETURN_IF_ERROR(loom_low_lower_policy_registry_verify_entry(entry));
    if (!iree_string_view_equal(entry->contract_set_key, contract_set_key)) {
      continue;
    }
    if (match != NULL) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "target-low lowering policy registry has duplicate contract key "
          "'%.*s'",
          (int)contract_set_key.size, contract_set_key.data);
    }
    match = entry->policy;
  }
  if (match == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "target-low lowering policy registry has no contract key '%.*s'",
        (int)contract_set_key.size, contract_set_key.data);
  }
  *out_policy = match;
  return iree_ok_status();
}

iree_status_t loom_low_lower_policy_registry_lookup_for_bundle(
    const loom_low_lower_policy_registry_t* registry,
    const loom_target_bundle_t* bundle,
    const loom_low_lower_policy_t** out_policy) {
  if (bundle == NULL) {
    if (out_policy != NULL) {
      *out_policy = NULL;
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target bundle is required");
  }
  if (bundle->config == NULL) {
    if (out_policy != NULL) {
      *out_policy = NULL;
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target bundle config is required");
  }
  return loom_low_lower_policy_registry_lookup(
      registry, bundle->config->contract_set_key, out_policy);
}

bool loom_low_lower_policy_registry_has_bundle(
    const loom_low_lower_policy_registry_t* registry,
    const loom_target_bundle_t* bundle) {
  if (!registry || !bundle || !bundle->config) {
    return false;
  }
  iree_string_view_t contract_set_key = bundle->config->contract_set_key;
  if (iree_string_view_is_empty(iree_string_view_trim(contract_set_key))) {
    return false;
  }
  for (iree_host_size_t i = 0; i < registry->entry_count; ++i) {
    if (iree_string_view_equal(registry->entries[i].contract_set_key,
                               contract_set_key)) {
      return true;
    }
  }
  return false;
}

static iree_string_view_t loom_low_lower_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id < module->strings.count) {
    return module->strings.entries[symbol->name_id];
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_lower_function_name(
    const loom_low_lower_context_t* context) {
  if (!loom_func_like_isa(context->source_function)) {
    return IREE_SV("<module>");
  }
  return loom_low_lower_symbol_name(
      context->module, loom_func_like_callee(context->source_function));
}

static iree_string_view_t loom_low_lower_target_key(
    const loom_target_bundle_t* bundle) {
  if (!bundle) {
    return IREE_SV("<missing>");
  }
  return loom_low_lower_nonempty(bundle->name, IREE_SV("<empty>"));
}

static iree_string_view_t loom_low_lower_export_name(
    const loom_target_bundle_t* bundle) {
  if (!bundle || !bundle->export_plan) {
    return IREE_SV("<missing>");
  }
  return loom_low_lower_nonempty(bundle->export_plan->name, IREE_SV("<empty>"));
}

static iree_string_view_t loom_low_lower_config_key(
    const loom_target_bundle_t* bundle) {
  if (!bundle || !bundle->config) {
    return IREE_SV("<missing>");
  }
  return loom_low_lower_nonempty(bundle->config->name, IREE_SV("<empty>"));
}

static bool loom_low_lower_should_stop(
    const loom_low_lower_context_t* context) {
  return context->options->max_errors != 0 &&
         context->result->error_count >= context->options->max_errors;
}

static iree_status_t loom_low_lower_emit(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         const loom_error_def_t* error,
                                         const loom_diagnostic_param_t* params,
                                         iree_host_size_t param_count) {
  if (error->severity == LOOM_DIAGNOSTIC_ERROR) {
    if (loom_low_lower_should_stop(context)) {
      return iree_ok_status();
    }
    ++context->result->error_count;
  } else if (error->severity == LOOM_DIAGNOSTIC_REMARK) {
    ++context->result->remark_count;
  }
  loom_diagnostic_emission_t emission = {
      .op = source_op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(context->options->emitter, &emission);
}

iree_status_t loom_low_lower_emit_reject(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         iree_string_view_t subject_kind,
                                         iree_string_view_t subject_name,
                                         iree_string_view_t reason) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_target_key(context->options->bundle)),
      loom_param_string(loom_low_lower_export_name(context->options->bundle)),
      loom_param_string(loom_low_lower_config_key(context->options->bundle)),
      loom_param_string(loom_low_lower_function_name(context)),
      loom_param_string(subject_kind),
      loom_param_string(subject_name),
      loom_param_string(reason),
  };
  return loom_low_lower_emit(
      context, source_op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 1),
      params, IREE_ARRAYSIZE(params));
}

loom_module_t* loom_low_lower_context_module(
    loom_low_lower_context_t* context) {
  return context->module;
}

loom_builder_t* loom_low_lower_context_builder(
    loom_low_lower_context_t* context) {
  return &context->builder;
}

loom_func_like_t loom_low_lower_context_source_function(
    const loom_low_lower_context_t* context) {
  return context->source_function;
}

loom_op_t* loom_low_lower_context_low_function(
    const loom_low_lower_context_t* context) {
  return context->low_func_op;
}

const loom_target_bundle_t* loom_low_lower_context_bundle(
    const loom_low_lower_context_t* context) {
  return context->options->bundle;
}

const loom_low_descriptor_set_t* loom_low_lower_context_descriptor_set(
    const loom_low_lower_context_t* context) {
  return context->descriptor_set;
}

iree_status_t loom_low_lower_register_class_string_id(
    loom_low_lower_context_t* context, uint16_t reg_class_id,
    loom_string_id_t* out_string_id) {
  IREE_ASSERT_ARGUMENT(context);
  return loom_low_build_register_class_string_id(
      context->module, context->descriptor_set, reg_class_id, out_string_id);
}

iree_status_t loom_low_lower_make_register_type(
    loom_low_lower_context_t* context, uint16_t reg_class_id,
    uint32_t unit_count, loom_type_t* out_type) {
  IREE_ASSERT_ARGUMENT(context);
  return loom_low_build_register_type(context->module, context->descriptor_set,
                                      reg_class_id, unit_count, out_type);
}

iree_status_t loom_low_lower_emit_descriptor_op(
    loom_low_lower_context_t* context, uint64_t descriptor_id,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_types,
    iree_host_size_t result_count, const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count, loom_location_id_t location,
    loom_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(context);
  return loom_low_build_descriptor_op(
      &context->builder, context->descriptor_set, descriptor_id, operands,
      operand_count, attrs, result_types, result_count, tied_results,
      tied_result_count, location, out_op);
}

iree_status_t loom_low_lower_emit_descriptor_const(
    loom_low_lower_context_t* context, uint64_t descriptor_id,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(context);
  return loom_low_build_descriptor_const(&context->builder,
                                         context->descriptor_set, descriptor_id,
                                         attrs, result_type, location, out_op);
}

iree_status_t loom_low_lower_map_type(loom_low_lower_context_t* context,
                                      const loom_op_t* source_op,
                                      loom_type_t source_type,
                                      loom_type_t* out_low_type) {
  if (!out_low_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low type output is required");
  }
  *out_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      context->policy->map_type.fn(context->policy->map_type.user_data, context,
                                   source_op, source_type, out_low_type));
  return iree_ok_status();
}

iree_status_t loom_low_lower_map_value(loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_value_id_t source_value_id,
                                       loom_type_t* out_low_type) {
  if (!out_low_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low type output is required");
  }
  *out_low_type = loom_type_none();
  if (source_value_id >= context->module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source value %u is out of range",
                            (unsigned)source_value_id);
  }
  const loom_type_t source_type =
      loom_module_value_type(context->module, source_value_id);
  if (context->policy->map_value.fn == NULL) {
    return loom_low_lower_map_type(context, source_op, source_type,
                                   out_low_type);
  }
  return context->policy->map_value.fn(context->policy->map_value.user_data,
                                       context, source_op, source_value_id,
                                       source_type, out_low_type);
}

static iree_status_t loom_low_lower_map_direct_argument(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_semantic_type = loom_type_none(),
  };
  return loom_low_lower_map_value(context, source_op, source_argument_id,
                                  &out_argument->abi_type);
}

static iree_status_t loom_low_lower_map_argument(
    loom_low_lower_context_t* context, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  uint32_t previous_error_count = context->result->error_count;
  if (context->policy->map_argument.fn == NULL) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_map_direct_argument(context, context->source_function.op,
                                           source_argument_id, out_argument));
  } else {
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
        .abi_type = loom_type_none(),
        .resource_semantic_type = loom_type_none(),
    };
    IREE_RETURN_IF_ERROR(context->policy->map_argument.fn(
        context->policy->map_argument.user_data, context,
        context->source_function.op, source_argument_index, source_argument_id,
        out_argument));
  }

  if (!loom_low_lower_abi_argument_kind_is_known(out_argument->kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low argument ABI policy produced an "
                            "unknown argument kind");
  }
  if (loom_low_lower_type_is_none(out_argument->abi_type)) {
    if (context->result->error_count == previous_error_count) {
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_reject(
          context, context->source_function.op, IREE_SV("argument"),
          IREE_SV("<unknown>"),
          IREE_SV("target-low argument ABI policy did not produce a register "
                  "type")));
    }
    return iree_ok_status();
  }
  if (!loom_type_is_register(out_argument->abi_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low argument ABI policy produced a "
                            "non-register ABI type");
  }

  if (out_argument->kind == LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
    return iree_ok_status();
  }
  if (!loom_low_lower_resource_import_kind_is_known(
          out_argument->resource_import_kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low argument ABI policy produced an "
                            "unknown resource import kind");
  }
  if (out_argument->resource_index < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low argument ABI policy produced a "
                            "negative resource index");
  }
  if (loom_low_lower_type_is_none(out_argument->resource_semantic_type)) {
    out_argument->resource_semantic_type =
        loom_module_value_type(context->module, source_argument_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_initialize_argument_map(
    loom_low_lower_context_t* context) {
  if (context->argument_map != NULL) {
    return iree_ok_status();
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  context->argument_map_count = argument_count;
  if (argument_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->arena, argument_count, sizeof(*context->argument_map),
      (void**)&context->argument_map));
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_map_argument(
        context, i, source_arguments[i], &context->argument_map[i]));
  }
  return iree_ok_status();
}

static uint16_t loom_low_lower_direct_argument_count(
    const loom_low_lower_context_t* context) {
  uint16_t direct_argument_count = 0;
  for (uint16_t i = 0; i < context->argument_map_count; ++i) {
    if (context->argument_map[i].kind == LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
      ++direct_argument_count;
    }
  }
  return direct_argument_count;
}

iree_status_t loom_low_lower_lookup_value(loom_low_lower_context_t* context,
                                          loom_value_id_t source_value_id,
                                          loom_value_id_t* out_low_value_id) {
  if (!out_low_value_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low value output is required");
  }
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  if (source_value_id >= context->value_map_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source value %u is outside the value map captured for lowering",
        (unsigned)source_value_id);
  }
  loom_value_id_t low_value_id = context->value_map[source_value_id];
  if (low_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source value %u has no target-low mapping",
                            (unsigned)source_value_id);
  }
  *out_low_value_id = low_value_id;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_copy_value_name(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_value_id_t low_value_id) {
  const loom_value_t* source_value =
      loom_module_value(context->module, source_value_id);
  if (source_value->name_id == LOOM_STRING_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_module_set_value_name(context->module, low_value_id,
                                    source_value->name_id);
}

iree_status_t loom_low_lower_bind_value(loom_low_lower_context_t* context,
                                        loom_value_id_t source_value_id,
                                        loom_value_id_t low_value_id) {
  if (source_value_id >= context->value_map_count ||
      low_value_id >= context->module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source or low value id is out of range");
  }
  loom_value_id_t existing = context->value_map[source_value_id];
  if (existing != LOOM_VALUE_ID_INVALID && existing != low_value_id) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source value %u is already bound to low value %u",
                            (unsigned)source_value_id, (unsigned)existing);
  }
  context->value_map[source_value_id] = low_value_id;
  return loom_low_lower_copy_value_name(context, source_value_id, low_value_id);
}

static iree_status_t loom_low_lower_validate_options(
    loom_module_t* module, loom_func_like_t source_function,
    const loom_low_lower_options_t* options,
    loom_low_lower_result_t* out_result) {
  if (!module || !loom_func_like_isa(source_function) || !options ||
      !out_result) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, source function, options, and result are required");
  }
  if (source_function.op->kind != LOOM_OP_FUNC_DEF) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source-to-low lowering currently requires "
                            "a func.def source function");
  }
  if (!loom_symbol_ref_is_valid(options->target_ref) ||
      options->target_ref.module_id != 0 ||
      options->target_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module-local target bundle symbol is required");
  }
  const loom_symbol_t* target_symbol =
      &module->symbols.entries[options->target_ref.symbol_id];
  if (!target_symbol->defining_op ||
      target_symbol->defining_op->kind != LOOM_OP_TARGET_BUNDLE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target symbol must define a target.bundle op");
  }
  if (!options->bundle || !options->bundle->snapshot ||
      !options->bundle->export_plan || !options->bundle->config) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "complete target bundle is required");
  }
  if (!options->descriptor_registry) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry is required");
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_policy_verify(options->policy));
  IREE_RETURN_IF_ERROR(loom_target_low_legality_provider_list_verify(
      options->legality_provider_list));
  return iree_ok_status();
}

static iree_status_t loom_low_lower_create_derived_symbol(
    loom_low_lower_context_t* context, iree_string_view_t base_name,
    iree_string_view_t suffix, bool append_index, uint32_t index,
    loom_symbol_ref_t* out_symbol_ref) {
  *out_symbol_ref = loom_symbol_ref_null();
  if (iree_string_view_is_empty(base_name) ||
      iree_string_view_equal(base_name, IREE_SV("<unnamed>"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "base symbol name is required");
  }

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = iree_string_builder_append_string(&builder, base_name);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&builder, suffix);
  }
  if (iree_status_is_ok(status) && append_index) {
    status = iree_string_builder_append_format(&builder, "%u", (unsigned)index);
  }

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  if (iree_status_is_ok(status)) {
    status = loom_module_intern_string(
        context->module,
        iree_make_string_view(iree_string_builder_buffer(&builder),
                              iree_string_builder_size(&builder)),
        &name_id);
  }
  iree_string_builder_deinitialize(&builder);
  IREE_RETURN_IF_ERROR(status);

  if (loom_module_find_symbol(context->module, name_id) !=
      LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "target-low derived symbol already exists");
  }

  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_add_symbol(context->module, name_id, &symbol_id));
  *out_symbol_ref = (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
  return iree_ok_status();
}

static iree_status_t loom_low_lower_create_low_symbol(
    loom_low_lower_context_t* context, loom_symbol_ref_t* out_symbol_ref) {
  loom_symbol_ref_t source_ref =
      loom_func_like_callee(context->source_function);
  iree_string_view_t source_name =
      loom_low_lower_symbol_name(context->module, source_ref);
  iree_string_view_t suffix = context->options->low_function_suffix;
  if (iree_string_view_is_empty(suffix)) {
    suffix = IREE_SV("__low");
  }
  return loom_low_lower_create_derived_symbol(
      context, source_name, suffix, /*append_index=*/false, 0, out_symbol_ref);
}

static iree_status_t loom_low_lower_intern_type_id(
    loom_low_lower_context_t* context, loom_type_t type,
    loom_type_id_t* out_type_id) {
  return loom_module_intern_type_id(context->module, type, out_type_id);
}

static iree_status_t loom_low_lower_check_mapped_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_type_t* out_low_type) {
  uint32_t previous_error_count = context->result->error_count;
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_value_id, out_low_type));
  if (loom_low_lower_type_is_none(*out_low_type)) {
    if (context->result->error_count == previous_error_count) {
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_reject(
          context, source_op, IREE_SV("value"), IREE_SV("<unknown>"),
          IREE_SV("target-low value policy did not produce a register type")));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_check_function_signature(
    loom_low_lower_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_low_lower_initialize_argument_map(context));

  const loom_value_id_t* result_ids =
      loom_op_const_results(context->source_function.op);
  for (uint16_t i = 0; i < context->source_function.op->result_count; ++i) {
    loom_type_t low_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
        context, context->source_function.op, result_ids[i], &low_type));
  }

  uint16_t predicate_count = 0;
  (void)loom_func_like_predicates(context->source_function, &predicate_count);
  if (predicate_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_reject(
        context, context->source_function.op, IREE_SV("function"),
        loom_low_lower_function_name(context),
        IREE_SV("function predicates need value remapping before target-low "
                "lowering")));
  }
  if (context->source_function.op->tied_result_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_reject(
        context, context->source_function.op, IREE_SV("function"),
        loom_low_lower_function_name(context),
        IREE_SV("tied function results need explicit ABI ownership lowering")));
  }
  return iree_ok_status();
}

static bool loom_low_lower_op_is_structural(loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_CFG_BR:
    case LOOM_OP_CFG_COND_BR:
    case LOOM_OP_FUNC_RETURN:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_low_lower_preflight_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (source_op->region_count != 0) {
    return loom_low_lower_emit_reject(
        context, source_op, IREE_SV("op"),
        loom_op_name(context->module, source_op),
        IREE_SV("nested regions must be lowered away before target-low "
                "source lowering"));
  }
  if (loom_low_lower_op_is_structural(source_op->kind)) {
    return iree_ok_status();
  }

  bool handled = false;
  IREE_RETURN_IF_ERROR(context->policy->can_lower_op.fn(
      context->policy->can_lower_op.user_data, context, source_op, &handled));
  if (!handled) {
    return loom_low_lower_emit_reject(
        context, source_op, IREE_SV("op"),
        loom_op_name(context->module, source_op),
        IREE_SV("the selected target-low lowering policy has no descriptor "
                "mapping for this op"));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_preflight_body(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  IREE_RETURN_IF_ERROR(loom_low_lower_check_function_signature(context));
  if (loom_low_lower_should_stop(context)) {
    return iree_ok_status();
  }

  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(source_body, block_index);
    if (block_index != 0) {
      for (uint16_t i = 0; i < block->arg_count; ++i) {
        loom_type_t low_type = loom_type_none();
        IREE_RETURN_IF_ERROR(loom_low_lower_check_mapped_value(
            context, context->source_function.op, block->arg_ids[i],
            &low_type));
        if (loom_low_lower_should_stop(context)) {
          return iree_ok_status();
        }
      }
    }
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_low_lower_preflight_op(context, op));
      if (loom_low_lower_should_stop(context)) {
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_map_signature_types(
    loom_low_lower_context_t* context, loom_type_t** out_arg_types,
    iree_host_size_t* out_arg_count, loom_type_t** out_result_types,
    iree_host_size_t* out_result_count) {
  IREE_RETURN_IF_ERROR(loom_low_lower_initialize_argument_map(context));
  *out_arg_types = NULL;
  *out_arg_count = 0;
  *out_result_types = NULL;
  *out_result_count = 0;

  uint16_t argument_count = 0;
  (void)loom_func_like_arg_ids(context->source_function, &argument_count);
  loom_type_t* arg_types = NULL;
  const uint16_t direct_argument_count =
      loom_low_lower_direct_argument_count(context);
  if (direct_argument_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&context->arena, direct_argument_count,
                                  sizeof(*arg_types), (void**)&arg_types));
    uint16_t direct_argument_index = 0;
    for (uint16_t i = 0; i < argument_count; ++i) {
      if (context->argument_map[i].kind != LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
        continue;
      }
      arg_types[direct_argument_index] = context->argument_map[i].abi_type;
      if (loom_low_lower_type_is_none(arg_types[direct_argument_index])) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "preflight accepted an unmapped function argument type");
      }
      ++direct_argument_index;
    }
  }

  const uint16_t result_count = context->source_function.op->result_count;
  loom_type_t* result_types = NULL;
  if (result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &context->arena, result_count, sizeof(*result_types),
        (void**)&result_types));
    const loom_value_id_t* result_ids =
        loom_op_const_results(context->source_function.op);
    for (uint16_t i = 0; i < result_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_map_value(context, context->source_function.op,
                                   result_ids[i], &result_types[i]));
      if (loom_low_lower_type_is_none(result_types[i])) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "preflight accepted an unmapped function result type");
      }
    }
  }

  *out_arg_types = arg_types;
  *out_arg_count = direct_argument_count;
  *out_result_types = result_types;
  *out_result_count = result_count;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_create_function_op(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_symbol_ref_t low_func_ref) {
  loom_type_t* arg_types = NULL;
  iree_host_size_t arg_count = 0;
  loom_type_t* result_types = NULL;
  iree_host_size_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_lower_map_signature_types(
      context, &arg_types, &arg_count, &result_types, &result_count));

  loom_low_func_def_build_flags_t build_flags = 0;
  uint8_t visibility = loom_func_like_visibility(context->source_function);
  uint8_t cc = loom_func_like_cc(context->source_function);
  uint8_t purity = loom_func_like_purity(context->source_function);
  if (visibility != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY;
  }
  if (cc != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_CC;
  }
  if (purity != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_PURITY;
  }
  loom_builder_initialize(context->module, &context->module->arena,
                          loom_module_block(context->module),
                          &context->builder);
  IREE_RETURN_IF_ERROR(loom_low_func_def_build(
      &context->builder, build_flags, visibility, cc, purity,
      /*allocation=*/0, /*schedule=*/0, context->options->target_ref,
      low_func_ref, arg_types, arg_count, result_types, result_count,
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*predicates=*/NULL, /*predicates_count=*/0,
      context->source_function.op->location, &context->low_func_op));

  loom_region_t* low_body = loom_low_func_def_body(context->low_func_op);
  low_body->flags = source_body->flags;
  context->result->low_func_op = context->low_func_op;
  context->result->low_func_ref = low_func_ref;

  const loom_value_id_t* source_results =
      loom_op_const_results(context->source_function.op);
  const loom_value_id_t* low_results =
      loom_op_const_results(context->low_func_op);
  for (uint16_t i = 0; i < context->source_function.op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_copy_value_name(
        context, source_results[i], low_results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_map_blocks(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  loom_region_t* low_body = loom_low_func_def_body(context->low_func_op);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->arena, source_body->block_count, sizeof(*context->block_map),
      (void**)&context->block_map));
  memset(
      context->block_map, 0,
      (iree_host_size_t)source_body->block_count * sizeof(*context->block_map));

  for (uint16_t i = 0; i < source_body->block_count; ++i) {
    loom_block_t* source_block = loom_region_block(source_body, i);
    loom_block_t* low_block = NULL;
    if (i == 0) {
      low_block = loom_region_entry_block(low_body);
    } else {
      IREE_RETURN_IF_ERROR(
          loom_region_append_block(context->module, low_body, &low_block));
    }
    low_block->label_id = source_block->label_id;
    context->block_map[i] = low_block;
  }

  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    loom_block_t* source_block = loom_region_block(source_body, block_index);
    loom_block_t* low_block = context->block_map[block_index];
    if (block_index == 0) {
      const uint16_t direct_argument_count =
          loom_low_lower_direct_argument_count(context);
      if (low_block->arg_count != direct_argument_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "emitted low entry block argument count does not match direct "
            "source arguments");
      }
      uint16_t direct_argument_index = 0;
      for (uint16_t arg_index = 0; arg_index < source_block->arg_count;
           ++arg_index) {
        if (context->argument_map[arg_index].kind !=
            LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
            context, source_block->arg_ids[arg_index],
            low_block->arg_ids[direct_argument_index]));
        ++direct_argument_index;
      }
      continue;
    }

    for (uint16_t arg_index = 0; arg_index < source_block->arg_count;
         ++arg_index) {
      loom_type_t low_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_low_lower_map_value(
          context, context->source_function.op,
          source_block->arg_ids[arg_index], &low_type));
      if (loom_low_lower_type_is_none(low_type)) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "preflight accepted an unmapped block type");
      }
      loom_value_id_t low_arg = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
          &context->builder, low_block, low_type, &low_arg));
      IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
          context, source_block->arg_ids[arg_index], low_arg));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_argument_resource_imports(
    loom_low_lower_context_t* context) {
  uint16_t argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  if (argument_count == 0 ||
      argument_count == loom_low_lower_direct_argument_count(context)) {
    return iree_ok_status();
  }

  loom_region_t* low_body = loom_low_func_def_body(context->low_func_op);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < argument_count && iree_status_is_ok(status); ++i) {
    if (context->argument_map[i].kind != LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE) {
      continue;
    }
    loom_type_t semantic_type = context->argument_map[i].resource_semantic_type;
    if (loom_low_lower_type_is_none(semantic_type)) {
      semantic_type =
          loom_module_value_type(context->module, source_arguments[i]);
    }
    loom_type_id_t semantic_type_id = LOOM_TYPE_ID_INVALID;
    status = loom_low_lower_intern_type_id(context, semantic_type,
                                           &semantic_type_id);
    if (!iree_status_is_ok(status)) {
      break;
    }
    loom_op_t* resource_op = NULL;
    status = loom_low_resource_build(
        &context->builder,
        (uint8_t)context->argument_map[i].resource_import_kind,
        context->argument_map[i].resource_index, semantic_type_id,
        context->argument_map[i].abi_type,
        context->source_function.op->location, &resource_op);
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_bind_value(context, source_arguments[i],
                                         loom_low_resource_result(resource_op));
    }
  }

  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_emit_preamble(
    loom_low_lower_context_t* context) {
  if (context->policy->emit_preamble.fn == NULL) {
    return iree_ok_status();
  }

  loom_region_t* low_body = loom_low_func_def_body(context->low_func_op);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  loom_builder_set_block(&context->builder, loom_region_entry_block(low_body));
  iree_status_t status = context->policy->emit_preamble.fn(
      context->policy->emit_preamble.user_data, context);
  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

static iree_status_t loom_low_lower_find_block_index(loom_region_t* region,
                                                     const loom_block_t* block,
                                                     uint16_t* out_index) {
  for (uint16_t i = 0; i < region->block_count; ++i) {
    if (loom_region_block(region, i) == block) {
      *out_index = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "branch target block is outside the source region");
}

static iree_status_t loom_low_lower_remap_values(
    loom_low_lower_context_t* context, const loom_value_id_t* source_values,
    iree_host_size_t value_count, loom_value_id_t** out_low_values) {
  *out_low_values = NULL;
  if (value_count == 0) {
    return iree_ok_status();
  }
  loom_value_id_t* low_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &context->arena, value_count, sizeof(*low_values), (void**)&low_values));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, source_values[i], &low_values[i]));
  }
  *out_low_values = low_values;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_structural_op(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    const loom_op_t* source_op, bool* out_handled) {
  *out_handled = true;
  switch (source_op->kind) {
    case LOOM_OP_FUNC_RETURN: {
      loom_value_slice_t values = loom_func_return_operands(source_op);
      loom_value_id_t* low_values = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(
          context, values.values, values.count, &low_values));
      loom_op_t* low_return_op = NULL;
      return loom_low_return_build(&context->builder, low_values, values.count,
                                   source_op->location, &low_return_op);
    }
    case LOOM_OP_CFG_BR: {
      uint16_t source_dest_index = 0;
      IREE_RETURN_IF_ERROR(loom_low_lower_find_block_index(
          source_body, loom_cfg_br_dest(source_op), &source_dest_index));
      loom_value_slice_t args = loom_cfg_br_args(source_op);
      loom_value_id_t* low_args = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_remap_values(context, args.values,
                                                       args.count, &low_args));
      loom_op_t* low_br_op = NULL;
      return loom_low_br_build(&context->builder,
                               context->block_map[source_dest_index], low_args,
                               args.count, source_op->location, &low_br_op);
    }
    case LOOM_OP_CFG_COND_BR: {
      uint16_t source_true_index = 0;
      IREE_RETURN_IF_ERROR(loom_low_lower_find_block_index(
          source_body, loom_cfg_cond_br_true_dest(source_op),
          &source_true_index));
      uint16_t source_false_index = 0;
      IREE_RETURN_IF_ERROR(loom_low_lower_find_block_index(
          source_body, loom_cfg_cond_br_false_dest(source_op),
          &source_false_index));
      loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, loom_cfg_cond_br_condition(source_op), &low_condition));
      loom_op_t* low_cond_br_op = NULL;
      return loom_low_cond_br_build(&context->builder, low_condition,
                                    context->block_map[source_true_index],
                                    context->block_map[source_false_index],
                                    source_op->location, &low_cond_br_op);
    }
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}

static iree_status_t loom_low_lower_validate_op_results_bound(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_id_t* source_results = loom_op_const_results(source_op);
  for (uint16_t i = 0; i < source_op->result_count; ++i) {
    if (source_results[i] >= context->value_map_count ||
        context->value_map[source_results[i]] == LOOM_VALUE_ID_INVALID) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "target-low policy '%.*s' did not bind result %u of %.*s",
          (int)loom_low_lower_policy_name(context->policy).size,
          loom_low_lower_policy_name(context->policy).data, (unsigned)i,
          (int)loom_op_name(context->module, source_op).size,
          loom_op_name(context->module, source_op).data);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_emit_body(loom_low_lower_context_t* context,
                                              loom_region_t* source_body) {
  loom_region_t* low_body = loom_low_func_def_body(context->low_func_op);
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &context->builder, context->low_func_op, low_body);
  iree_status_t status = iree_ok_status();

  for (uint16_t block_index = 0;
       block_index < source_body->block_count && iree_status_is_ok(status);
       ++block_index) {
    loom_block_t* source_block = loom_region_block(source_body, block_index);
    loom_builder_set_block(&context->builder, context->block_map[block_index]);
    loom_op_t* source_op = NULL;
    loom_block_for_each_op(source_block, source_op) {
      bool handled = false;
      status = loom_low_lower_structural_op(context, source_body, source_op,
                                            &handled);
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (!handled) {
        status = context->policy->try_lower_op.fn(
            context->policy->try_lower_op.user_data, context, source_op,
            &handled);
        if (!iree_status_is_ok(status)) {
          break;
        }
        if (!handled) {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "target-low policy '%.*s' failed to lower preflighted op %.*s",
              (int)loom_low_lower_policy_name(context->policy).size,
              loom_low_lower_policy_name(context->policy).data,
              (int)loom_op_name(context->module, source_op).size,
              loom_op_name(context->module, source_op).data);
          break;
        }
      }
      status = loom_low_lower_validate_op_results_bound(context, source_op);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
  }

  loom_builder_restore(&context->builder, saved_ip);
  return status;
}

iree_status_t loom_low_lower_function(loom_module_t* module,
                                      loom_func_like_t source_function,
                                      const loom_low_lower_options_t* options,
                                      loom_low_lower_result_t* out_result) {
  if (out_result) {
    *out_result = (loom_low_lower_result_t){
        .low_func_ref = loom_symbol_ref_null(),
    };
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_validate_options(module, source_function,
                                                       options, out_result));

  loom_region_t* source_body = loom_func_like_body(source_function);
  if (!source_body) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source function must have a body");
  }

  loom_target_low_legality_options_t legality_options = {
      .bundle = options->bundle,
      .descriptor_registry = options->descriptor_registry,
      .descriptor_requirements = options->descriptor_requirements,
      .provider_list = options->legality_provider_list,
      .emitter = options->emitter,
      .max_errors = options->max_errors,
  };
  loom_target_low_legality_result_t legality_result = {};
  IREE_RETURN_IF_ERROR(loom_target_low_verify_function_legality(
      module, source_function, &legality_options, &legality_result));

  out_result->error_count = legality_result.error_count;
  out_result->remark_count = legality_result.remark_count;
  out_result->descriptor_set = legality_result.descriptor_set;
  if (out_result->error_count != 0) {
    return iree_ok_status();
  }

  loom_low_lower_context_t context = {
      .module = module,
      .source_function = source_function,
      .options = options,
      .policy = options->policy,
      .descriptor_set = legality_result.descriptor_set,
      .result = out_result,
      .value_map_count = module->values.count,
  };
  iree_arena_initialize(module->arena.block_pool, &context.arena);

  iree_status_t status = iree_arena_allocate_array(
      &context.arena, context.value_map_count, sizeof(*context.value_map),
      (void**)&context.value_map);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < context.value_map_count; ++i) {
      context.value_map[i] = LOOM_VALUE_ID_INVALID;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_preflight_body(&context, source_body);
  }
  if (iree_status_is_ok(status) && context.result->error_count == 0) {
    loom_symbol_ref_t low_func_ref = loom_symbol_ref_null();
    status = loom_low_lower_create_low_symbol(&context, &low_func_ref);
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_create_function_op(&context, source_body,
                                                 low_func_ref);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_map_blocks(&context, source_body);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_emit_preamble(&context);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_emit_argument_resource_imports(&context);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_emit_body(&context, source_body);
    }
  }

  iree_arena_deinitialize(&context.arena);
  return status;
}
