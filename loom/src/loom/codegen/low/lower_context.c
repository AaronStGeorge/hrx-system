// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower_internal.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"

static iree_string_view_t loom_low_lower_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
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

iree_string_view_t loom_low_lower_context_function_name(
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

bool loom_low_lower_context_should_stop(
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
    if (loom_low_lower_context_should_stop(context)) {
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
      loom_param_string(loom_low_lower_context_function_name(context)),
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

const loom_value_fact_table_t* loom_low_lower_context_fact_table(
    const loom_low_lower_context_t* context) {
  return &context->fact_table;
}

iree_host_size_t loom_low_lower_context_selected_plan_count(
    const loom_low_lower_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  return context->selected_plan_count;
}

loom_low_lower_selected_plan_view_t loom_low_lower_context_selected_plan_view(
    const loom_low_lower_context_t* context, iree_host_size_t index) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_LT(index, context->selected_plan_count);
  return (loom_low_lower_selected_plan_view_t){
      .source_op = context->selected_plans[index].source_op,
      .plan = context->selected_plans[index].plan,
  };
}

iree_status_t loom_low_lower_allocate_scratch_array(
    loom_low_lower_context_t* context, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_ptr);
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  return iree_arena_allocate_array(&context->arena, count, element_size,
                                   out_ptr);
}

iree_status_t loom_low_lower_allocate_plan_data(
    loom_low_lower_context_t* context, iree_host_size_t data_length,
    void** out_data) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_GT(data_length, 0);
  IREE_ASSERT_ARGUMENT(out_data);
  *out_data = NULL;
  return loom_low_lower_allocate_scratch_array(context, 1, data_length,
                                               out_data);
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
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_low_type);
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
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_low_type);
  *out_low_type = loom_type_none();
  IREE_ASSERT_LT(source_value_id, context->module->values.count);
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

iree_status_t loom_low_lower_lookup_value(loom_low_lower_context_t* context,
                                          loom_value_id_t source_value_id,
                                          loom_value_id_t* out_low_value_id) {
  IREE_ASSERT_ARGUMENT(out_low_value_id);
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_LT(source_value_id, context->value_map_count);
  loom_value_id_t low_value_id = context->value_map[source_value_id];
  IREE_ASSERT(low_value_id != LOOM_VALUE_ID_INVALID);
  IREE_ASSERT(low_value_id != LOOM_LOW_LOWER_VALUE_ID_ELIDED);
  *out_low_value_id = low_value_id;
  return iree_ok_status();
}

iree_status_t loom_low_lower_copy_value_name(loom_low_lower_context_t* context,
                                             loom_value_id_t source_value_id,
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
  IREE_ASSERT_LT(source_value_id, context->value_map_count);
  IREE_ASSERT_LT(low_value_id, context->module->values.count);
  loom_value_id_t existing = context->value_map[source_value_id];
  IREE_ASSERT(existing == LOOM_VALUE_ID_INVALID || existing == low_value_id);
  context->value_map[source_value_id] = low_value_id;
  return loom_low_lower_copy_value_name(context, source_value_id, low_value_id);
}

iree_status_t loom_low_lower_elide_value(loom_low_lower_context_t* context,
                                         loom_value_id_t source_value_id) {
  IREE_ASSERT_LT(source_value_id, context->value_map_count);
  loom_value_id_t existing = context->value_map[source_value_id];
  IREE_ASSERT(existing == LOOM_VALUE_ID_INVALID ||
              existing == LOOM_LOW_LOWER_VALUE_ID_ELIDED);
  context->value_map[source_value_id] = LOOM_LOW_LOWER_VALUE_ID_ELIDED;
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

iree_status_t loom_low_lower_create_function_symbol(
    loom_low_lower_context_t* context, iree_string_view_t suffix,
    bool append_index, uint32_t index, loom_symbol_ref_t* out_symbol_ref) {
  IREE_ASSERT_ARGUMENT(out_symbol_ref);
  *out_symbol_ref = loom_symbol_ref_null();
  IREE_ASSERT(context->low_func_op != NULL);
  iree_string_view_t low_function_name = loom_low_lower_symbol_name(
      context->module, loom_low_func_def_callee(context->low_func_op));
  return loom_low_lower_create_derived_symbol(
      context, low_function_name, suffix, append_index, index, out_symbol_ref);
}
