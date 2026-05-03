// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/low_legality.h"

#include <stdint.h>

#include "iree/base/internal/arena.h"
#include "loom/error/error_defs.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/util/walk.h"

typedef uint8_t loom_target_low_legality_t;

enum loom_target_low_legality_e {
  LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED = 0,
  LOOM_TARGET_LOW_LEGALITY_CORE = 1,
  LOOM_TARGET_LOW_LEGALITY_PROVIDER = 2,
  LOOM_TARGET_LOW_LEGALITY_SOURCE_ONLY = 3,
  LOOM_TARGET_LOW_LEGALITY_MODULE_METADATA = 4,
};

struct loom_target_low_legality_context_t {
  // Source module being checked.
  const loom_module_t* module;
  // Source function being checked.
  loom_func_like_t function;
  // Caller-owned verification options.
  const loom_target_low_legality_options_t* options;
  // Descriptor set selected by options.bundle.
  const loom_low_descriptor_set_t* descriptor_set;
  // Source facts visible to target-specific legality providers.
  const loom_value_fact_table_t* fact_table;
  // Result object receiving counters and selected descriptor set.
  loom_target_low_legality_result_t* result;
  // Scratch arena for the IR walker.
  iree_arena_allocator_t arena;
};

static iree_string_view_t loom_target_low_legality_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static iree_string_view_t loom_target_low_legality_symbol_name(
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

iree_string_view_t loom_target_low_legality_function_name(
    const loom_target_low_legality_context_t* context) {
  if (!loom_func_like_isa(context->function)) {
    return IREE_SV("<module>");
  }
  return loom_target_low_legality_symbol_name(
      context->module, loom_func_like_callee(context->function));
}

static iree_string_view_t loom_target_low_legality_target_key(
    const loom_target_bundle_t* bundle) {
  return loom_target_low_legality_nonempty(bundle->name, IREE_SV("<empty>"));
}

static iree_string_view_t loom_target_low_legality_export_name(
    const loom_target_bundle_t* bundle) {
  return loom_target_low_legality_nonempty(bundle->export_plan->name,
                                           IREE_SV("<empty>"));
}

static iree_string_view_t loom_target_low_legality_config_key(
    const loom_target_bundle_t* bundle) {
  return loom_target_low_legality_nonempty(bundle->config->name,
                                           IREE_SV("<empty>"));
}

static bool loom_target_low_legality_should_stop(
    const loom_target_low_legality_context_t* context) {
  return context->options->max_errors != 0 &&
         context->result->error_count >= context->options->max_errors;
}

static iree_status_t loom_target_low_legality_emit(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  if (error->severity == LOOM_DIAGNOSTIC_ERROR) {
    if (loom_target_low_legality_should_stop(context)) {
      return iree_ok_status();
    }
    ++context->result->error_count;
  } else if (error->severity == LOOM_DIAGNOSTIC_REMARK) {
    ++context->result->remark_count;
  }
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(context->options->emitter, &emission);
}

iree_status_t loom_target_low_legality_reject(
    loom_target_low_legality_context_t* context,
    const loom_target_low_legality_provider_t* provider, const loom_op_t* op,
    iree_string_view_t subject_kind, iree_string_view_t subject_name,
    iree_string_view_t reason) {
  (void)provider;
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_target_low_legality_target_key(context->options->bundle)),
      loom_param_string(
          loom_target_low_legality_export_name(context->options->bundle)),
      loom_param_string(
          loom_target_low_legality_config_key(context->options->bundle)),
      loom_param_string(loom_target_low_legality_function_name(context)),
      loom_param_string(subject_kind),
      loom_param_string(subject_name),
      loom_param_string(reason),
  };
  return loom_target_low_legality_emit(
      context, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 1), params,
      IREE_ARRAYSIZE(params));
}

iree_status_t loom_target_low_legality_emit_error_ref(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_error_ref_t error_ref, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  const loom_error_def_t* error = loom_error_def_lookup_ref(error_ref);
  IREE_ASSERT(error != NULL);
  return loom_target_low_legality_emit(context, op, error, params, param_count);
}

#define LOOM_TARGET_LOW_LEGALITY_CONTEXT_PARAM_COUNT 5

static void loom_target_low_legality_make_context_params(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_diagnostic_param_t* params) {
  params[0] = loom_param_string(
      loom_target_low_legality_target_key(context->options->bundle));
  params[1] = loom_param_string(
      loom_target_low_legality_export_name(context->options->bundle));
  params[2] = loom_param_string(
      loom_target_low_legality_config_key(context->options->bundle));
  params[3] =
      loom_param_string(loom_target_low_legality_function_name(context));
  params[4] = loom_param_string(loom_op_name(context->module, op));
}

static iree_status_t loom_target_low_legality_emit_target_context_error(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    uint16_t error_code, const loom_diagnostic_param_t* extra_params,
    iree_host_size_t extra_param_count) {
  IREE_ASSERT_LE(extra_param_count, 4);
  loom_diagnostic_param_t
      params[LOOM_TARGET_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 4];
  loom_target_low_legality_make_context_params(context, op, params);
  for (iree_host_size_t i = 0; i < extra_param_count; ++i) {
    params[LOOM_TARGET_LOW_LEGALITY_CONTEXT_PARAM_COUNT + i] = extra_params[i];
  }
  return loom_target_low_legality_emit(
      context, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TARGET, error_code),
      params, LOOM_TARGET_LOW_LEGALITY_CONTEXT_PARAM_COUNT + extra_param_count);
}

static iree_status_t loom_target_low_legality_emit_no_target_contract(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  return loom_target_low_legality_emit_target_context_error(
      context, op, 1, /*extra_params=*/NULL, /*extra_param_count=*/0);
}

static iree_status_t loom_target_low_legality_emit_type_error(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    uint16_t error_code, loom_type_t type) {
  const loom_diagnostic_param_t params[] = {
      loom_param_type(type),
  };
  return loom_target_low_legality_emit_target_context_error(
      context, op, error_code, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_target_low_legality_reject_error_ref(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_target_contract_rejection_t* rejection) {
  const loom_error_def_t* error =
      loom_error_def_lookup_ref(rejection->error_ref);
  IREE_ASSERT(error != NULL);
  return loom_target_low_legality_emit(context, op, error, rejection->params,
                                       rejection->param_count);
}

iree_status_t loom_target_low_legality_record_contract(
    loom_target_low_legality_context_t* context,
    const loom_target_low_legality_provider_t* provider, const loom_op_t* op,
    iree_string_view_t contract_key, iree_string_view_t decision,
    iree_string_view_t reason) {
  (void)provider;
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_target_low_legality_target_key(context->options->bundle)),
      loom_param_string(
          loom_target_low_legality_export_name(context->options->bundle)),
      loom_param_string(
          loom_target_low_legality_config_key(context->options->bundle)),
      loom_param_string(loom_target_low_legality_function_name(context)),
      loom_param_string(contract_key),
      loom_param_string(decision),
      loom_param_string(reason),
  };
  return loom_target_low_legality_emit(
      context, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 2), params,
      IREE_ARRAYSIZE(params));
}

iree_status_t loom_target_low_legality_record_memory_access(
    loom_target_low_legality_context_t* context,
    const loom_target_low_legality_provider_t* provider, const loom_op_t* op,
    iree_string_view_t memory_space, iree_string_view_t operation_kind,
    iree_string_view_t packet_key, iree_string_view_t decision,
    uint32_t element_bytes, uint32_t vector_lanes,
    uint32_t dynamic_stride_bytes, uint32_t vector_lane_stride_bytes,
    uint32_t bank_stride_words, uint32_t bank_conflict_degree,
    iree_string_view_t reason) {
  (void)provider;
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_target_low_legality_target_key(context->options->bundle)),
      loom_param_string(
          loom_target_low_legality_export_name(context->options->bundle)),
      loom_param_string(
          loom_target_low_legality_config_key(context->options->bundle)),
      loom_param_string(loom_target_low_legality_function_name(context)),
      loom_param_string(memory_space),
      loom_param_string(operation_kind),
      loom_param_string(packet_key),
      loom_param_string(decision),
      loom_param_u32(element_bytes),
      loom_param_u32(vector_lanes),
      loom_param_u32(dynamic_stride_bytes),
      loom_param_u32(vector_lane_stride_bytes),
      loom_param_u32(bank_stride_words),
      loom_param_u32(bank_conflict_degree),
      loom_param_string(reason),
  };
  return loom_target_low_legality_emit(
      context, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 17), params,
      IREE_ARRAYSIZE(params));
}

const loom_module_t* loom_target_low_legality_module(
    const loom_target_low_legality_context_t* context) {
  return context->module;
}

loom_func_like_t loom_target_low_legality_function(
    const loom_target_low_legality_context_t* context) {
  return context->function;
}

const loom_target_bundle_t* loom_target_low_legality_bundle(
    const loom_target_low_legality_context_t* context) {
  return context->options->bundle;
}

const loom_low_descriptor_set_t* loom_target_low_legality_descriptor_set(
    const loom_target_low_legality_context_t* context) {
  return context->descriptor_set;
}

const loom_value_fact_table_t* loom_target_low_legality_fact_table(
    const loom_target_low_legality_context_t* context) {
  return context->fact_table;
}

loom_target_low_legality_diagnostic_flags_t
loom_target_low_legality_diagnostic_flags(
    const loom_target_low_legality_context_t* context) {
  return context->options->diagnostic_flags;
}

static bool loom_target_low_legality_codegen_format_is_low(
    loom_target_codegen_format_t codegen_format) {
  switch (codegen_format) {
    case LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE:
    case LOOM_TARGET_CODEGEN_FORMAT_VM:
    case LOOM_TARGET_CODEGEN_FORMAT_WASM:
      return true;
    default:
      return false;
  }
}

static bool loom_target_low_legality_abi_is_low(
    loom_target_abi_kind_t abi_kind) {
  switch (abi_kind) {
    case LOOM_TARGET_ABI_OBJECT_FUNCTION:
    case LOOM_TARGET_ABI_HAL_KERNEL:
    case LOOM_TARGET_ABI_VM_MODULE_FUNCTION:
    case LOOM_TARGET_ABI_WASM_FUNCTION:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_target_low_legality_validate_options(
    const loom_target_low_legality_options_t* options,
    const loom_low_descriptor_set_t** out_descriptor_set) {
  *out_descriptor_set = NULL;
  if (!loom_target_low_legality_codegen_format_is_low(
          options->bundle->snapshot->codegen_format)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target bundle '%.*s' codegen format is not target-low",
        (int)options->bundle->name.size, options->bundle->name.data);
  }
  if (!loom_target_low_legality_abi_is_low(
          options->bundle->export_plan->abi_kind)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target bundle '%.*s' ABI is not accepted by target-low legality",
        (int)options->bundle->name.size, options->bundle->name.data);
  }
  if (options->bundle->snapshot->default_pointer_bitwidth == 0 ||
      options->bundle->snapshot->index_bitwidth == 0 ||
      options->bundle->snapshot->offset_bitwidth == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target bundle '%.*s' pointer, index, and offset bit widths must be "
        "non-zero",
        (int)options->bundle->name.size, options->bundle->name.data);
  }
  if (iree_any_bit_set(options->diagnostic_flags,
                       ~LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_ALL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low legality diagnostics use unknown "
                            "flag bits 0x%08x",
                            (unsigned)options->diagnostic_flags);
  }
  return loom_target_low_descriptor_set_select_for_bundle(
      options->descriptor_registry, options->bundle, out_descriptor_set);
}

static iree_status_t loom_target_low_legality_verify_scalar_type(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_type_t type) {
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I1:
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
    case LOOM_SCALAR_TYPE_F32:
    case LOOM_SCALAR_TYPE_F64:
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
      return loom_target_low_legality_emit_type_error(context, op, 58, type);
    case LOOM_SCALAR_TYPE_COUNT_:
      break;
  }
  return loom_target_low_legality_emit_type_error(context, op, 59, type);
}

static const loom_type_descriptor_t*
loom_target_low_legality_resolve_dialect_type(const loom_module_t* module,
                                              loom_type_t type,
                                              iree_string_view_t* out_name) {
  *out_name = IREE_SV("<unknown>");
  if (!loom_type_is_dialect(type)) {
    return NULL;
  }
  loom_string_id_t name_id = loom_type_dialect_name_id(type);
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return NULL;
  }
  iree_string_view_t name = module->strings.entries[name_id];
  *out_name = name;
  const loom_type_descriptor_t* descriptor = loom_type_registry_lookup(name);
  if (descriptor == NULL ||
      descriptor->param_count != loom_type_dialect_param_count(type)) {
    return NULL;
  }
  return descriptor;
}

static bool loom_target_low_legality_op_accepts_type_contract(
    const loom_module_t* module, const loom_op_t* op,
    const loom_type_descriptor_t* descriptor) {
  loom_op_semantics_t op_semantics = loom_op_semantics(module, op);
  return loom_contract_family_set_has_any(
      op_semantics.contract_families, descriptor->semantics.contract_families);
}

static iree_string_view_t loom_target_low_legality_type_semantic_name(
    loom_type_semantic_t semantic) {
  switch (semantic) {
    case LOOM_TYPE_SEMANTIC_CONTROL_TOKEN:
      return IREE_SV("control token");
    case LOOM_TYPE_SEMANTIC_TARGET_CONTRACT_VALUE:
      return IREE_SV("target contract value");
    case LOOM_TYPE_SEMANTIC_ORDINARY:
    default:
      return IREE_SV("registered");
  }
}

static iree_status_t loom_target_low_legality_verify_registered_type(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_type_t type, bool* out_handled) {
  *out_handled = false;
  iree_string_view_t type_name = iree_string_view_empty();
  const loom_type_descriptor_t* descriptor =
      loom_target_low_legality_resolve_dialect_type(context->module, type,
                                                    &type_name);
  if (descriptor == NULL ||
      descriptor->semantics.semantic == LOOM_TYPE_SEMANTIC_ORDINARY) {
    return iree_ok_status();
  }
  *out_handled = true;
  if (descriptor->semantics.contract_families != 0 &&
      loom_target_low_legality_op_accepts_type_contract(context->module, op,
                                                        descriptor)) {
    return iree_ok_status();
  }
  switch (descriptor->semantics.semantic) {
    case LOOM_TYPE_SEMANTIC_CONTROL_TOKEN:
    case LOOM_TYPE_SEMANTIC_TARGET_CONTRACT_VALUE: {
      (void)type_name;
      const loom_diagnostic_param_t params[] = {
          loom_param_type(type),
          loom_param_string(loom_target_low_legality_type_semantic_name(
              descriptor->semantics.semantic)),
      };
      return loom_target_low_legality_emit_target_context_error(
          context, op, 60, params, IREE_ARRAYSIZE(params));
    }
    default:
      return loom_target_low_legality_emit_type_error(context, op, 63, type);
  }
}

static iree_status_t loom_target_low_legality_verify_type(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_type_t type) {
  if (loom_type_is_scalar(type)) {
    return loom_target_low_legality_verify_scalar_type(context, op, type);
  }
  if (loom_type_is_buffer(type) || loom_type_is_view(type) ||
      loom_type_is_register(type) || loom_type_is_encoding(type)) {
    return iree_ok_status();
  }
  bool registered_type_handled = false;
  IREE_RETURN_IF_ERROR(loom_target_low_legality_verify_registered_type(
      context, op, type, &registered_type_handled));
  if (registered_type_handled) {
    return iree_ok_status();
  }
  if (loom_type_is_vector(type)) {
    if (!loom_type_is_all_static(type) || loom_type_rank(type) != 1) {
      return loom_target_low_legality_emit_type_error(context, op, 61, type);
    }
    uint64_t element_count = 0;
    if (!loom_type_static_element_count(type, &element_count) ||
        element_count > UINT32_MAX) {
      return loom_target_low_legality_emit_type_error(context, op, 62, type);
    }
    return loom_target_low_legality_verify_scalar_type(
        context, op, loom_type_scalar(loom_type_element_type(type)));
  }
  return loom_target_low_legality_emit_type_error(context, op, 63, type);
}

static iree_status_t loom_target_low_legality_verify_value(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_value_id_t value_id) {
  if (value_id >= context->module->values.count) {
    const loom_diagnostic_param_t params[] = {
        loom_param_u64(value_id),
    };
    return loom_target_low_legality_emit_target_context_error(
        context, op, 64, params, IREE_ARRAYSIZE(params));
  }
  const loom_type_t type = loom_module_value_type(context->module, value_id);
  return loom_target_low_legality_verify_type(context, op, type);
}

static iree_status_t loom_target_low_legality_verify_op_value_types(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_target_low_legality_verify_value(context, op, operands[i]));
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_target_low_legality_verify_value(context, op, results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_low_legality_try_provider_op(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  const uint8_t dialect_id = loom_op_dialect_id(op->kind);
  for (iree_host_size_t i = 0; i < context->options->provider_list.count; ++i) {
    const loom_target_low_legality_provider_t* provider =
        context->options->provider_list.values[i];
    if (!loom_target_low_legality_builtin_dialect_bits_contain(
            provider->builtin_dialect_bits, dialect_id)) {
      continue;
    }
    bool handled = false;
    IREE_RETURN_IF_ERROR(
        provider->try_verify_op(provider, context, op, &handled));
    if (handled) {
      *out_handled = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_low_legality_reject_contract_query(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_target_contract_query_result_t* result) {
  if (result->rejection != NULL) {
    if (loom_error_ref_is_set(result->rejection->error_ref)) {
      return loom_target_low_legality_reject_error_ref(context, op,
                                                       result->rejection);
    }
    return loom_target_low_legality_reject(
        context, NULL, op, result->rejection->subject_kind,
        result->rejection->subject_name, result->rejection->reason);
  }
  return loom_target_low_legality_emit_no_target_contract(context, op);
}

static iree_status_t loom_target_low_legality_try_contract_query_op(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  if (loom_target_contract_query_callback_is_empty(
          context->options->contract_query)) {
    return iree_ok_status();
  }

  const loom_target_contract_query_environment_t environment = {
      .module = context->module,
      .function = context->function,
      .bundle = context->options->bundle,
      .descriptor_set = context->descriptor_set,
      .fact_table = context->fact_table,
      .arena = &context->arena,
  };
  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();
  IREE_RETURN_IF_ERROR(context->options->contract_query.fn(
      context->options->contract_query.user_data, &environment, op, &result));
  switch (result.outcome) {
    case LOOM_TARGET_CONTRACT_QUERY_UNHANDLED:
      return iree_ok_status();
    case LOOM_TARGET_CONTRACT_QUERY_LEGAL:
      *out_handled = true;
      return iree_ok_status();
    case LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED:
    case LOOM_TARGET_CONTRACT_QUERY_INVALID_IR:
      *out_handled = true;
      return loom_target_low_legality_reject_contract_query(context, op,
                                                            &result);
    default:
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "target contract query returned unknown outcome "
                              "%d",
                              (int)result.outcome);
  }
}

static iree_status_t loom_target_low_legality_reject_source_only_op(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_WHILE:
    case LOOM_OP_SCF_SWITCH:
      return loom_target_low_legality_emit_target_context_error(
          context, op, 66, /*extra_params=*/NULL, /*extra_param_count=*/0);
    case LOOM_OP_SCF_CONDITION:
    case LOOM_OP_SCF_YIELD:
      return loom_target_low_legality_emit_target_context_error(
          context, op, 67, /*extra_params=*/NULL, /*extra_param_count=*/0);
    default:
      return loom_target_low_legality_emit_target_context_error(
          context, op, 68, /*extra_params=*/NULL, /*extra_param_count=*/0);
  }
}

static iree_status_t loom_target_low_legality_verify_op_class(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, op);
  if (loom_traits_are_fact_identity(traits)) {
    return iree_ok_status();
  }
  if (loom_traits_are_value_alias(traits)) {
    return iree_ok_status();
  }
  switch (op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT:
    case LOOM_OP_CFG_BR:
    case LOOM_OP_CFG_COND_BR:
    case LOOM_OP_FUNC_RETURN:
    case LOOM_OP_KERNEL_RETURN:
      return iree_ok_status();
    default:
      break;
  }
  loom_op_semantics_t semantics = loom_op_semantics(context->module, op);
  loom_target_low_legality_t legality = LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED;
  if (semantics.contract_families != 0) {
    legality = LOOM_TARGET_LOW_LEGALITY_PROVIDER;
  } else {
    switch (semantics.phase) {
      case LOOM_OP_PHASE_EXECUTABLE:
        legality = LOOM_TARGET_LOW_LEGALITY_CORE;
        break;
      case LOOM_OP_PHASE_SOURCE_STRUCTURE:
        legality = LOOM_TARGET_LOW_LEGALITY_SOURCE_ONLY;
        break;
      case LOOM_OP_PHASE_MODULE_METADATA:
        legality = LOOM_TARGET_LOW_LEGALITY_MODULE_METADATA;
        break;
      case LOOM_OP_PHASE_UNSPECIFIED:
      default:
        legality = LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED;
        break;
    }
  }
  switch (legality) {
    case LOOM_TARGET_LOW_LEGALITY_CORE:
      return iree_ok_status();
    case LOOM_TARGET_LOW_LEGALITY_PROVIDER:
      return loom_target_low_legality_emit_target_context_error(
          context, op, 65, /*extra_params=*/NULL, /*extra_param_count=*/0);
    case LOOM_TARGET_LOW_LEGALITY_SOURCE_ONLY:
      return loom_target_low_legality_reject_source_only_op(context, op);
    case LOOM_TARGET_LOW_LEGALITY_MODULE_METADATA:
      return loom_target_low_legality_emit_target_context_error(
          context, op, 69, /*extra_params=*/NULL, /*extra_param_count=*/0);
    case LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED:
      return loom_target_low_legality_emit_no_target_contract(context, op);
    default: {
      iree_string_view_t op_name = loom_op_name(context->module, op);
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "op '%.*s' has unknown target-low legality class %u",
          (int)op_name.size, op_name.data, (unsigned)legality);
    }
  }
}

static iree_status_t loom_target_low_legality_verify_op(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_verify_op_value_types(context, op));

  bool contract_handled = false;
  IREE_RETURN_IF_ERROR(loom_target_low_legality_try_contract_query_op(
      context, op, &contract_handled));
  if (contract_handled) {
    return iree_ok_status();
  }

  bool provider_handled = false;
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_try_provider_op(context, op, &provider_handled));
  if (provider_handled) {
    return iree_ok_status();
  }

  return loom_target_low_legality_verify_op_class(context, op);
}

static bool loom_target_low_legality_skip_children_after_rejection(
    loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_WHILE:
    case LOOM_OP_SCF_SWITCH:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_target_low_legality_walk_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* walk_context,
    loom_walk_result_t* out_result) {
  (void)walk_context;
  loom_target_low_legality_context_t* context =
      (loom_target_low_legality_context_t*)user_data;
  *out_result = LOOM_WALK_CONTINUE;
  uint32_t previous_error_count = context->result->error_count;
  IREE_RETURN_IF_ERROR(loom_target_low_legality_verify_op(context, op));
  if (loom_target_low_legality_should_stop(context)) {
    *out_result = LOOM_WALK_ABORT;
  } else if (context->result->error_count != previous_error_count &&
             loom_target_low_legality_skip_children_after_rejection(op->kind)) {
    *out_result = LOOM_WALK_SKIP;
  }
  return iree_ok_status();
}

static iree_status_t loom_target_low_legality_verify_function_signature(
    loom_target_low_legality_context_t* context) {
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(context->function, &argument_count);
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_target_low_legality_verify_value(
        context, context->function.op, argument_ids[i]));
  }
  const loom_value_id_t* result_ids =
      loom_op_const_results(context->function.op);
  for (uint16_t i = 0; i < context->function.op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_target_low_legality_verify_value(
        context, context->function.op, result_ids[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_target_low_verify_function_legality(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_low_legality_options_t* options,
    loom_target_low_legality_result_t* out_result) {
  *out_result = (loom_target_low_legality_result_t){0};
  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_validate_options(options, &descriptor_set));
  out_result->descriptor_set = descriptor_set;

  loom_target_low_legality_context_t context = {
      .module = module,
      .function = function,
      .options = options,
      .descriptor_set = descriptor_set,
      .result = out_result,
  };
  iree_arena_initialize(module->arena.block_pool, &context.arena);
  context.fact_table = options->fact_table;

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = loom_target_low_legality_verify_function_signature(&context);
  }
  loom_region_t* body = loom_func_like_body(function);
  if (iree_status_is_ok(status) && body) {
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    status = loom_walk_region(
        module, body, LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_target_low_legality_walk_op, &context},
        &context.arena, &walk_result);
  }

  iree_arena_deinitialize(&context.arena);
  return status;
}
