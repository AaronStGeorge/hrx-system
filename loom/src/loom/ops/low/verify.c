// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

typedef struct loom_low_callee_signature_t {
  const loom_op_t* definition_op;
  const loom_value_id_t* argument_ids;
  uint16_t argument_count;
  const loom_value_id_t* result_ids;
  uint16_t result_count;
} loom_low_callee_signature_t;

static iree_status_t loom_low_emit_related(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count, const loom_diagnostic_related_op_t* related,
    iree_host_size_t related_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related,
      .related_op_count = related_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_low_emit(iree_diagnostic_emitter_t emitter,
                                   const loom_op_t* op,
                                   const loom_error_def_t* error,
                                   const loom_diagnostic_param_t* params,
                                   iree_host_size_t param_count) {
  return loom_low_emit_related(emitter, op, error, params, param_count, NULL,
                               0);
}

static bool loom_low_descriptor_key_segment_start(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool loom_low_descriptor_key_segment_continue(char c) {
  return loom_low_descriptor_key_segment_start(c) || (c >= '0' && c <= '9');
}

static bool loom_low_descriptor_key_is_valid(iree_string_view_t key) {
  if (iree_string_view_is_empty(key)) return false;
  bool saw_separator = false;
  bool expect_segment_start = true;
  for (iree_host_size_t i = 0; i < key.size; ++i) {
    char c = key.data[i];
    if (c == '.') {
      if (expect_segment_start) return false;
      saw_separator = true;
      expect_segment_start = true;
      continue;
    }
    if (expect_segment_start) {
      if (!loom_low_descriptor_key_segment_start(c)) return false;
      expect_segment_start = false;
      continue;
    }
    if (!loom_low_descriptor_key_segment_continue(c)) return false;
  }
  return saw_separator && !expect_segment_start;
}

static iree_string_view_t loom_low_string_or_empty(const loom_module_t* module,
                                                   loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_low_symbol_name(const loom_module_t* module,
                                               loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  iree_string_view_t name = loom_low_string_or_empty(module, symbol->name_id);
  return iree_string_view_is_empty(name) ? IREE_SV("<unnamed>") : name;
}

static iree_string_view_t loom_low_symbol_definition_name(
    const loom_symbol_t* symbol) {
  if (!symbol || !symbol->definition) return IREE_SV("unresolved");
  return loom_symbol_definition_descriptor_name(symbol->definition);
}

static const loom_symbol_t* loom_low_lookup_defined_symbol(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (!symbol->definition || !symbol->defining_op) return NULL;
  return symbol;
}

static bool loom_low_function_isa(const loom_op_t* op) {
  return loom_low_func_def_isa(op) || loom_low_func_decl_isa(op);
}

static iree_status_t loom_low_emit_callee_related(
    iree_diagnostic_emitter_t emitter, const loom_op_t* invoke_op,
    const loom_op_t* definition_op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("defined here"),
      .op = definition_op,
  }};
  return loom_low_emit_related(emitter, invoke_op, error, params, param_count,
                               related,
                               definition_op ? IREE_ARRAYSIZE(related) : 0);
}

static iree_status_t loom_low_emit_invoke_callee_kind_mismatch(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_symbol_t* symbol, iree_diagnostic_emitter_t emitter) {
  loom_symbol_ref_t callee = loom_low_invoke_callee(invoke_op);
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(loom_low_symbol_name(module, callee)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    loom_low_invoke_callee_ATTR_INDEX)),
      loom_param_string(loom_low_symbol_definition_name(symbol)),
      loom_param_string(IREE_SV("low function")),
  };
  return loom_low_emit_callee_related(
      emitter, invoke_op, symbol->defining_op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3), params,
      IREE_ARRAYSIZE(params));
}

static bool loom_low_load_callee_signature(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_low_callee_signature_t* out_signature) {
  if (!loom_low_function_isa(symbol->defining_op)) return false;
  loom_func_like_t callee = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(callee)) return false;
  out_signature->definition_op = symbol->defining_op;
  out_signature->argument_ids =
      loom_func_like_arg_ids(callee, &out_signature->argument_count);
  out_signature->result_ids = loom_op_const_results(callee.op);
  out_signature->result_count = callee.op->result_count;
  return true;
}

static iree_status_t loom_low_emit_descriptor_key_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t attr_index,
    iree_string_view_t key) {
  loom_diagnostic_field_ref_t attr_ref =
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, attr_index);
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(loom_param_string(IREE_SV("opcode")), attr_ref),
      loom_param_string(key),
      loom_param_string(
          IREE_SV("a namespace-qualified descriptor key with non-empty "
                  "identifier segments")),
  };
  return loom_low_emit(emitter, op,
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 27),
                       params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_invoke_count_mismatch(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter, iree_string_view_t field_kind,
    uint16_t actual_count, uint16_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_symbol_name(module, loom_low_invoke_callee(invoke_op))),
      loom_param_string(field_kind),
      loom_param_u32(actual_count),
      loom_param_u32(expected_count),
  };
  return loom_low_emit_callee_related(
      emitter, invoke_op, signature->definition_op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 13), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_invoke_type_mismatch(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter,
    loom_diagnostic_field_kind_t field_ref_kind, iree_string_view_t field_kind,
    uint16_t field_index, loom_type_t actual_type,
    iree_string_view_t callee_field_kind, loom_type_t expected_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_symbol_name(module, loom_low_invoke_callee(invoke_op))),
      loom_param_with_field_ref(
          loom_param_string(field_kind),
          loom_diagnostic_field_ref(field_ref_kind, field_index)),
      loom_param_u32(field_index),
      loom_param_type(actual_type),
      loom_param_string(callee_field_kind),
      loom_param_type(expected_type),
  };
  return loom_low_emit_callee_related(
      emitter, invoke_op, signature->definition_op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 14), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_verify_descriptor_key(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_string_id_t opcode_id,
    uint16_t attr_index) {
  iree_string_view_t key = loom_low_string_or_empty(module, opcode_id);
  if (loom_low_descriptor_key_is_valid(key)) return iree_ok_status();
  return loom_low_emit_descriptor_key_error(emitter, op, attr_index, key);
}

static iree_status_t loom_low_verify_invoke_argument_count(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  if (invoke_op->operand_count == signature->argument_count) {
    return iree_ok_status();
  }
  return loom_low_emit_invoke_count_mismatch(
      module, invoke_op, signature, emitter, IREE_SV("operand"),
      invoke_op->operand_count, signature->argument_count);
}

static iree_status_t loom_low_verify_invoke_result_count(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  if (invoke_op->result_count == signature->result_count) {
    return iree_ok_status();
  }
  return loom_low_emit_invoke_count_mismatch(
      module, invoke_op, signature, emitter, IREE_SV("result"),
      invoke_op->result_count, signature->result_count);
}

static iree_status_t loom_low_verify_invoke_argument_types(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  uint16_t compare_count = invoke_op->operand_count;
  if (compare_count > signature->argument_count) {
    compare_count = signature->argument_count;
  }
  const loom_value_id_t* invoke_operands = loom_op_const_operands(invoke_op);
  for (uint16_t i = 0; i < compare_count; ++i) {
    loom_type_t actual_type =
        loom_module_value_type(module, invoke_operands[i]);
    loom_type_t expected_type =
        loom_module_value_type(module, signature->argument_ids[i]);
    if (loom_type_equal(actual_type, expected_type)) continue;
    IREE_RETURN_IF_ERROR(loom_low_emit_invoke_type_mismatch(
        module, invoke_op, signature, emitter, LOOM_DIAGNOSTIC_FIELD_OPERAND,
        IREE_SV("operand"), i, actual_type, IREE_SV("argument"),
        expected_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_invoke_result_types(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  uint16_t compare_count = invoke_op->result_count;
  if (compare_count > signature->result_count) {
    compare_count = signature->result_count;
  }
  const loom_value_id_t* invoke_results = loom_op_const_results(invoke_op);
  for (uint16_t i = 0; i < compare_count; ++i) {
    loom_type_t actual_type = loom_module_value_type(module, invoke_results[i]);
    loom_type_t expected_type =
        loom_module_value_type(module, signature->result_ids[i]);
    if (loom_type_equal(actual_type, expected_type)) continue;
    IREE_RETURN_IF_ERROR(loom_low_emit_invoke_type_mismatch(
        module, invoke_op, signature, emitter, LOOM_DIAGNOSTIC_FIELD_RESULT,
        IREE_SV("result"), i, actual_type, IREE_SV("result"), expected_type));
  }
  return iree_ok_status();
}

iree_status_t loom_low_op_verify(const loom_module_t* module,
                                 const loom_op_t* op,
                                 iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_descriptor_key(module, op, emitter,
                                        loom_low_op_opcode(op),
                                        loom_low_op_opcode_ATTR_INDEX);
}

iree_status_t loom_low_const_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_descriptor_key(module, op, emitter,
                                        loom_low_const_opcode(op),
                                        loom_low_const_opcode_ATTR_INDEX);
}

iree_status_t loom_low_invoke_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  const loom_symbol_t* symbol =
      loom_low_lookup_defined_symbol(module, loom_low_invoke_callee(op));
  if (!symbol) {
    return iree_ok_status();
  }

  loom_low_callee_signature_t signature = {0};
  if (!loom_low_load_callee_signature(module, symbol, &signature)) {
    return loom_low_emit_invoke_callee_kind_mismatch(module, op, symbol,
                                                     emitter);
  }

  IREE_RETURN_IF_ERROR(
      loom_low_verify_invoke_argument_count(module, op, &signature, emitter));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_invoke_result_count(module, op, &signature, emitter));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_invoke_argument_types(module, op, &signature, emitter));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_invoke_result_types(module, op, &signature, emitter));
  return iree_ok_status();
}
