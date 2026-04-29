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
#include "loom/ops/func/ops.h"
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

static iree_string_view_t loom_target_low_legality_function_name(
    const loom_target_low_legality_context_t* context) {
  if (!loom_func_like_isa(context->function)) {
    return IREE_SV("<module>");
  }
  return loom_target_low_legality_symbol_name(
      context->module, loom_func_like_callee(context->function));
}

static iree_string_view_t loom_target_low_legality_target_key(
    const loom_target_bundle_t* bundle) {
  if (!bundle) {
    return IREE_SV("<missing>");
  }
  return loom_target_low_legality_nonempty(bundle->name, IREE_SV("<empty>"));
}

static iree_string_view_t loom_target_low_legality_export_name(
    const loom_target_bundle_t* bundle) {
  if (!bundle || !bundle->export_plan) {
    return IREE_SV("<missing>");
  }
  return loom_target_low_legality_nonempty(bundle->export_plan->name,
                                           IREE_SV("<empty>"));
}

static iree_string_view_t loom_target_low_legality_config_key(
    const loom_target_bundle_t* bundle) {
  if (!bundle || !bundle->config) {
    return IREE_SV("<missing>");
  }
  return loom_target_low_legality_nonempty(bundle->config->name,
                                           IREE_SV("<empty>"));
}

static bool loom_target_low_legality_should_stop(
    const loom_target_low_legality_context_t* context) {
  return context->options->max_errors != 0 &&
         context->result->error_count >= context->options->max_errors;
}

iree_status_t loom_target_low_legality_provider_list_verify(
    loom_target_low_legality_provider_list_t list) {
  if (loom_target_low_legality_provider_list_is_empty(list)) {
    return iree_ok_status();
  }
  if (list.values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low legality provider list is required");
  }
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    const loom_target_low_legality_provider_t* provider = list.values[i];
    if (provider == NULL || provider->try_verify_op == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target-low legality provider is invalid");
    }
  }
  return iree_ok_status();
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
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_low_legality_options_t* options,
    loom_target_low_legality_result_t* out_result,
    const loom_low_descriptor_set_t** out_descriptor_set) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(options->fact_table);
  IREE_ASSERT_ARGUMENT(out_result);
  IREE_ASSERT_ARGUMENT(out_descriptor_set);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low legality function is required");
  }
  *out_descriptor_set = NULL;
  if (!options->bundle || !options->bundle->snapshot ||
      !options->bundle->export_plan || !options->bundle->config) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "complete target-low bundle is required");
  }
  if (!options->descriptor_registry) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "low descriptor registry is required");
  }
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
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_provider_list_verify(options->provider_list));
  if (iree_any_bit_set(options->diagnostic_flags,
                       ~LOOM_TARGET_LOW_LEGALITY_DIAGNOSTIC_ALL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target-low legality diagnostics use unknown "
                            "flag bits 0x%08x",
                            (unsigned)options->diagnostic_flags);
  }
  return loom_target_low_descriptor_set_select_for_bundle(
      options->descriptor_registry, options->bundle,
      options->descriptor_requirements, out_descriptor_set);
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
      return loom_target_low_legality_reject(
          context, NULL, op, IREE_SV("type"), IREE_SV("fp8"),
          IREE_SV("FP8 scalar types require explicit decode or a selected "
                  "target-low contract"));
    case LOOM_SCALAR_TYPE_COUNT_:
      break;
  }
  return loom_target_low_legality_reject(context, NULL, op, IREE_SV("type"),
                                         IREE_SV("scalar"),
                                         IREE_SV("unknown scalar type"));
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
      return loom_target_low_legality_reject(
          context, NULL, op, IREE_SV("type"), type_name,
          IREE_SV("control tokens must be produced or consumed by ops from a "
                  "matching target contract family before target-low "
                  "lowering"));
    case LOOM_TYPE_SEMANTIC_TARGET_CONTRACT_VALUE:
      return loom_target_low_legality_reject(
          context, NULL, op, IREE_SV("type"), type_name,
          IREE_SV("target contract values must be produced or consumed by ops "
                  "from a matching target contract family before target-low "
                  "lowering"));
    default:
      return loom_target_low_legality_reject(
          context, NULL, op, IREE_SV("type"), type_name,
          IREE_SV("registered type has no target-low legality mapping yet"));
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
      return loom_target_low_legality_reject(
          context, NULL, op, IREE_SV("type"), IREE_SV("vector"),
          IREE_SV("source-to-low legality requires specialized static "
                  "one-dimensional vectors"));
    }
    uint64_t element_count = 0;
    if (!loom_type_static_element_count(type, &element_count) ||
        element_count > UINT32_MAX) {
      return loom_target_low_legality_reject(
          context, NULL, op, IREE_SV("type"), IREE_SV("vector"),
          IREE_SV("vector lane count is not representable"));
    }
    return loom_target_low_legality_verify_scalar_type(
        context, op, loom_type_scalar(loom_type_element_type(type)));
  }
  return loom_target_low_legality_reject(
      context, NULL, op, IREE_SV("type"), IREE_SV("value"),
      IREE_SV("type has no target-low legality mapping yet"));
}

static iree_status_t loom_target_low_legality_verify_value(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_value_id_t value_id) {
  if (value_id >= context->module->values.count) {
    return loom_target_low_legality_reject(
        context, NULL, op, IREE_SV("value"), IREE_SV("<invalid>"),
        IREE_SV("operation references an invalid SSA value"));
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
  for (iree_host_size_t i = 0; i < context->options->provider_list.count; ++i) {
    const loom_target_low_legality_provider_t* provider =
        context->options->provider_list.values[i];
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

static iree_status_t loom_target_low_legality_reject_op(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t reason) {
  return loom_target_low_legality_reject(context, NULL, op, IREE_SV("op"),
                                         loom_op_name(context->module, op),
                                         reason);
}

static iree_status_t loom_target_low_legality_reject_source_only_op(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_WHILE:
    case LOOM_OP_SCF_SWITCH:
      return loom_target_low_legality_reject_op(
          context, op,
          IREE_SV("structured SCF control flow must be lowered to CFG before "
                  "target-low lowering"));
    case LOOM_OP_SCF_CONDITION:
    case LOOM_OP_SCF_YIELD:
      return loom_target_low_legality_reject_op(
          context, op,
          IREE_SV("SCF terminators must be lowered with their parent "
                  "structured control-flow op before target-low lowering"));
    default:
      return loom_target_low_legality_reject_op(
          context, op,
          IREE_SV("source-only op must be lowered before target-low legality"));
  }
}

static iree_status_t loom_target_low_legality_verify_op_class(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  if (loom_traits_are_fact_identity(
          loom_op_effective_traits(context->module, op))) {
    return iree_ok_status();
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
      return loom_target_low_legality_reject_op(
          context, op,
          IREE_SV("target-low legality requires a target provider for this "
                  "op"));
    case LOOM_TARGET_LOW_LEGALITY_SOURCE_ONLY:
      return loom_target_low_legality_reject_source_only_op(context, op);
    case LOOM_TARGET_LOW_LEGALITY_MODULE_METADATA:
      return loom_target_low_legality_reject_op(
          context, op,
          IREE_SV("target record ops are module metadata and cannot appear "
                  "inside executable regions"));
    case LOOM_TARGET_LOW_LEGALITY_UNSUPPORTED:
      return loom_target_low_legality_reject_op(
          context, op,
          IREE_SV("no target-low lowering rule or legality provider is "
                  "registered"));
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
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_target_low_legality_result_t){0};
  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_target_low_legality_validate_options(
      module, function, options, out_result, &descriptor_set));
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
