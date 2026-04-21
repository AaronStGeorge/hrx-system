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
#include "loom/ops/encoding/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/util/fact_table.h"
#include "loom/util/walk.h"

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
  // Locally computed fact table when the caller did not provide one.
  loom_value_fact_table_t local_fact_table;
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
  if (!module || !loom_func_like_isa(function) || !options || !out_result ||
      !out_descriptor_set) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "module, function, options, result, and descriptor "
                            "output are required");
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
  return loom_target_low_legality_verify_type(
      context, op, loom_module_value_type(context->module, value_id));
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

static bool loom_target_low_legality_op_is_supported_core(loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_BUFFER_ALLOCA:
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
    case LOOM_OP_BUFFER_VIEW:
    case LOOM_OP_CFG_BR:
    case LOOM_OP_CFG_COND_BR:
    case LOOM_OP_ENCODING_ASSUME_SPEC:
    case LOOM_OP_ENCODING_DEFINE:
    case LOOM_OP_ENCODING_LAYOUT_ASSUME_DENSE:
    case LOOM_OP_ENCODING_LAYOUT_ASSUME_STRIDED:
    case LOOM_OP_ENCODING_LAYOUT_DENSE:
    case LOOM_OP_ENCODING_LAYOUT_STRIDED:
    case LOOM_OP_FUNC_CALL:
    case LOOM_OP_FUNC_RETURN:
    case LOOM_OP_INDEX_ADD:
    case LOOM_OP_INDEX_CAST:
    case LOOM_OP_INDEX_CMP:
    case LOOM_OP_INDEX_CONSTANT:
    case LOOM_OP_INDEX_DIV:
    case LOOM_OP_INDEX_MADD:
    case LOOM_OP_INDEX_MUL:
    case LOOM_OP_INDEX_REM:
    case LOOM_OP_INDEX_SUB:
    case LOOM_OP_KERNEL_WORKGROUP_ID:
    case LOOM_OP_KERNEL_WORKITEM_ID:
    case LOOM_OP_LOW_BR:
    case LOOM_OP_LOW_COND_BR:
    case LOOM_OP_LOW_CONCAT:
    case LOOM_OP_LOW_CONST:
    case LOOM_OP_LOW_COPY:
    case LOOM_OP_LOW_FRAME_INDEX:
    case LOOM_OP_LOW_FUNC_DECL:
    case LOOM_OP_LOW_FUNC_DEF:
    case LOOM_OP_LOW_INVOKE:
    case LOOM_OP_LOW_OP:
    case LOOM_OP_LOW_RELOAD:
    case LOOM_OP_LOW_RESOURCE:
    case LOOM_OP_LOW_RETURN:
    case LOOM_OP_LOW_SLOT:
    case LOOM_OP_LOW_SPILL:
    case LOOM_OP_SCALAR_ADDF:
    case LOOM_OP_SCALAR_ADDI:
    case LOOM_OP_SCALAR_ANDI:
    case LOOM_OP_SCALAR_BITCAST:
    case LOOM_OP_SCALAR_CMPF:
    case LOOM_OP_SCALAR_CMPI:
    case LOOM_OP_SCALAR_CONSTANT:
    case LOOM_OP_SCALAR_DIVF:
    case LOOM_OP_SCALAR_DIVSI:
    case LOOM_OP_SCALAR_DIVUI:
    case LOOM_OP_SCALAR_EXTF:
    case LOOM_OP_SCALAR_EXTSI:
    case LOOM_OP_SCALAR_EXTUI:
    case LOOM_OP_SCALAR_FMAF:
    case LOOM_OP_SCALAR_FMAI:
    case LOOM_OP_SCALAR_FPTOUI:
    case LOOM_OP_SCALAR_FPTOSI:
    case LOOM_OP_SCALAR_FPTRUNC:
    case LOOM_OP_SCALAR_MULF:
    case LOOM_OP_SCALAR_MULI:
    case LOOM_OP_SCALAR_NEGF:
    case LOOM_OP_SCALAR_ORI:
    case LOOM_OP_SCALAR_REMF:
    case LOOM_OP_SCALAR_REMSI:
    case LOOM_OP_SCALAR_REMUI:
    case LOOM_OP_SCALAR_SHLI:
    case LOOM_OP_SCALAR_SHRSI:
    case LOOM_OP_SCALAR_SHRUI:
    case LOOM_OP_SCALAR_SITOFP:
    case LOOM_OP_SCALAR_SUBF:
    case LOOM_OP_SCALAR_SUBI:
    case LOOM_OP_SCALAR_TRUNCI:
    case LOOM_OP_SCALAR_UITOFP:
    case LOOM_OP_SCALAR_XORI:
    case LOOM_OP_SCF_SELECT:
    case LOOM_OP_VECTOR_ADDF:
    case LOOM_OP_VECTOR_ADDI:
    case LOOM_OP_VECTOR_ANDI:
    case LOOM_OP_VECTOR_BITCAST:
    case LOOM_OP_VECTOR_CMPF:
    case LOOM_OP_VECTOR_CMPI:
    case LOOM_OP_VECTOR_CONSTANT:
    case LOOM_OP_VECTOR_DIVF:
    case LOOM_OP_VECTOR_DIVSI:
    case LOOM_OP_VECTOR_DIVUI:
    case LOOM_OP_VECTOR_EXTF:
    case LOOM_OP_VECTOR_EXTRACT:
    case LOOM_OP_VECTOR_EXTSI:
    case LOOM_OP_VECTOR_EXTUI:
    case LOOM_OP_VECTOR_FPTOUI:
    case LOOM_OP_VECTOR_FPTOSI:
    case LOOM_OP_VECTOR_FPTRUNC:
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
    case LOOM_OP_VECTOR_FMAF:
    case LOOM_OP_VECTOR_FMAI:
    case LOOM_OP_VECTOR_INSERT:
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VECTOR_MULF:
    case LOOM_OP_VECTOR_MULI:
    case LOOM_OP_VECTOR_NEGF:
    case LOOM_OP_VECTOR_ORI:
    case LOOM_OP_VECTOR_POISON:
    case LOOM_OP_VECTOR_REMF:
    case LOOM_OP_VECTOR_REMSI:
    case LOOM_OP_VECTOR_REMUI:
    case LOOM_OP_VECTOR_SELECT:
    case LOOM_OP_VECTOR_SHLI:
    case LOOM_OP_VECTOR_SHRSI:
    case LOOM_OP_VECTOR_SHRUI:
    case LOOM_OP_VECTOR_SHUFFLE:
    case LOOM_OP_VECTOR_SITOFP:
    case LOOM_OP_VECTOR_SPLAT:
    case LOOM_OP_VECTOR_STORE:
    case LOOM_OP_VECTOR_SUBF:
    case LOOM_OP_VECTOR_SUBI:
    case LOOM_OP_VECTOR_TRUNCI:
    case LOOM_OP_VECTOR_UITOFP:
    case LOOM_OP_VECTOR_XORI:
    case LOOM_OP_VIEW_LOAD:
    case LOOM_OP_VIEW_PREFETCH:
    case LOOM_OP_VIEW_REFINE:
    case LOOM_OP_VIEW_STORE:
    case LOOM_OP_VIEW_SUBVIEW:
      return true;
    default:
      return false;
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

  switch (op->kind) {
    case LOOM_OP_FUNC_DEF:
    case LOOM_OP_FUNC_DECL:
      return iree_ok_status();
    case LOOM_OP_TARGET_BUNDLE:
    case LOOM_OP_TARGET_CONFIG:
    case LOOM_OP_TARGET_EXPORT:
    case LOOM_OP_TARGET_PRESET:
    case LOOM_OP_TARGET_SNAPSHOT:
      return loom_target_low_legality_reject(
          context, NULL, op, IREE_SV("op"), loom_op_name(context->module, op),
          IREE_SV("target record ops are module metadata and cannot appear "
                  "inside executable regions"));
    case LOOM_OP_VECTOR_DOT2F:
    case LOOM_OP_VECTOR_DOT4F8:
    case LOOM_OP_VECTOR_DOT4I:
    case LOOM_OP_VECTOR_DOT8I4:
      return loom_target_low_legality_reject(
          context, NULL, op, IREE_SV("op"), loom_op_name(context->module, op),
          IREE_SV("op requires an explicit target-low contract provider"));
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_WHILE:
    case LOOM_OP_SCF_SWITCH:
      return loom_target_low_legality_reject(
          context, NULL, op, IREE_SV("op"), loom_op_name(context->module, op),
          IREE_SV("structured SCF control flow must be lowered to CFG before "
                  "target-low lowering"));
    case LOOM_OP_SCF_CONDITION:
    case LOOM_OP_SCF_YIELD:
      return loom_target_low_legality_reject(
          context, NULL, op, IREE_SV("op"), loom_op_name(context->module, op),
          IREE_SV("SCF terminators must be lowered with their parent "
                  "structured control-flow op before target-low lowering"));
    default:
      break;
  }

  if (loom_target_low_legality_op_is_supported_core(op->kind)) {
    return iree_ok_status();
  }
  return loom_target_low_legality_reject(
      context, NULL, op, IREE_SV("op"), loom_op_name(context->module, op),
      IREE_SV(
          "no target-low lowering rule or legality provider is registered"));
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
  if (out_result) {
    *out_result = (loom_target_low_legality_result_t){0};
  }
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

  iree_status_t status = iree_ok_status();
  if (options->fact_table != NULL) {
    context.fact_table = options->fact_table;
  } else {
    status = loom_value_fact_table_initialize(
        &context.local_fact_table, &context.arena, module->values.count);
    if (iree_status_is_ok(status)) {
      status = loom_value_fact_table_compute(&context.local_fact_table, module,
                                             function);
    }
    if (iree_status_is_ok(status)) {
      context.fact_table = &context.local_fact_table;
    }
  }
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
