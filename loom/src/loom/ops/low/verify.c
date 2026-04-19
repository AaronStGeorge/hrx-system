// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/successor_verify.h"

typedef struct loom_low_callee_signature_t {
  // Defining low.func op for related diagnostic locations.
  const loom_op_t* definition_op;
  // Callee entry block argument value IDs in signature order.
  const loom_value_id_t* argument_ids;
  // Number of callee arguments.
  uint16_t argument_count;
  // Callee result value IDs in signature order.
  const loom_value_id_t* result_ids;
  // Number of callee results.
  uint16_t result_count;
} loom_low_callee_signature_t;

typedef enum loom_low_abi_field_kind_e {
  // Adapter operand entry mapping a caller value to callee ABI argument.
  LOOM_LOW_ABI_FIELD_OPERAND = 0,
  // Adapter result entry mapping a callee ABI result to caller value.
  LOOM_LOW_ABI_FIELD_RESULT = 1,
} loom_low_abi_field_kind_t;

typedef struct loom_low_abi_entry_t {
  // Mapping record operation that defines this entry.
  const loom_op_t* op;
  // Symbol reference naming this mapping record.
  loom_symbol_ref_t symbol;
  // Adapter symbol reference this mapping belongs to.
  loom_symbol_ref_t adapter;
  // Slot index within the adapter operand or result list.
  int64_t index;
  // Per-slot conversion rule.
  uint8_t conversion;
  // Semantic boundary type carried by low.invoke.
  loom_type_t semantic_type;
  // Callee register ABI type carried by low.func.
  loom_type_t abi_type;
} loom_low_abi_entry_t;

typedef enum loom_low_invoke_caller_kind_e {
  // Invoke is not nested under a function-like body.
  LOOM_LOW_INVOKE_CALLER_NONE = 0,
  // Invoke is nested under an ordinary semantic function-like body.
  LOOM_LOW_INVOKE_CALLER_SEMANTIC = 1,
  // Invoke is nested under a target-bound low function body.
  LOOM_LOW_INVOKE_CALLER_LOW = 2,
} loom_low_invoke_caller_kind_t;

typedef struct loom_low_invoke_caller_t {
  // Enclosing function-like op, or NULL when there is no valid caller.
  const loom_op_t* op;
  // Caller classification used by low.invoke context verification.
  loom_low_invoke_caller_kind_t kind;
} loom_low_invoke_caller_t;

typedef struct loom_low_abi_metadata_counts_t {
  // Number of low.abi.effect records attached to the adapter.
  uint32_t effect_count;
  // Number of low.abi.clobber records attached to the adapter.
  uint32_t clobber_count;
} loom_low_abi_metadata_counts_t;

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

static bool loom_low_qualified_key_segment_start(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool loom_low_qualified_key_segment_continue(char c) {
  return loom_low_qualified_key_segment_start(c) || (c >= '0' && c <= '9');
}

static bool loom_low_qualified_key_is_valid(iree_string_view_t key) {
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
      if (!loom_low_qualified_key_segment_start(c)) return false;
      expect_segment_start = false;
      continue;
    }
    if (!loom_low_qualified_key_segment_continue(c)) return false;
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

static iree_string_view_t loom_low_op_name(const loom_module_t* module,
                                           const loom_op_t* op) {
  if (!op) return IREE_SV("<null>");
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable) return IREE_SV("<unknown>");
  return loom_op_vtable_name(vtable);
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

static loom_symbol_ref_t loom_low_function_target(const loom_op_t* op) {
  if (loom_low_func_def_isa(op)) return loom_low_func_def_target(op);
  if (loom_low_func_decl_isa(op)) return loom_low_func_decl_target(op);
  return loom_symbol_ref_null();
}

static loom_symbol_ref_t loom_low_function_symbol(const loom_op_t* op) {
  if (loom_low_func_def_isa(op)) return loom_low_func_def_callee(op);
  if (loom_low_func_decl_isa(op)) return loom_low_func_decl_callee(op);
  return loom_symbol_ref_null();
}

static bool loom_low_symbol_ref_equal(loom_symbol_ref_t lhs,
                                      loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static bool loom_low_optional_attr_is_present(const loom_op_t* op,
                                              uint16_t attr_index) {
  return attr_index < op->attribute_count &&
         !loom_attr_is_absent(loom_op_attrs(op)[attr_index]);
}

static loom_type_t loom_low_type_attr(const loom_module_t* module,
                                      loom_type_id_t type_id) {
  if (type_id == LOOM_TYPE_ID_INVALID || type_id >= module->types.count) {
    return loom_type_none();
  }
  return module->types.entries[type_id];
}

static iree_string_view_t loom_low_abi_conversion_name(uint8_t conversion) {
  switch (conversion) {
    case LOOM_LOW_ABI_ADAPTER_CONVERSION_DIRECT:
      return IREE_SV("direct");
    case LOOM_LOW_ABI_ADAPTER_CONVERSION_MAPPED:
      return IREE_SV("mapped");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_low_abi_value_conversion_name(
    uint8_t conversion) {
  switch (conversion) {
    case LOOM_LOW_CONVERSION_DIRECT:
      return IREE_SV("direct");
    case LOOM_LOW_CONVERSION_SCALAR_TO_REGISTER:
      return IREE_SV("scalar_to_register");
    case LOOM_LOW_CONVERSION_REGISTER_TO_SCALAR:
      return IREE_SV("register_to_scalar");
    case LOOM_LOW_CONVERSION_VALUE_TO_REGISTER:
      return IREE_SV("value_to_register");
    case LOOM_LOW_CONVERSION_REGISTER_TO_VALUE:
      return IREE_SV("register_to_value");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_low_abi_field_kind_name(
    loom_low_abi_field_kind_t field_kind) {
  switch (field_kind) {
    case LOOM_LOW_ABI_FIELD_OPERAND:
      return IREE_SV("operand");
    case LOOM_LOW_ABI_FIELD_RESULT:
      return IREE_SV("result");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_low_invoke_caller_kind_name(
    loom_low_invoke_caller_kind_t caller_kind) {
  switch (caller_kind) {
    case LOOM_LOW_INVOKE_CALLER_NONE:
      return IREE_SV("module");
    case LOOM_LOW_INVOKE_CALLER_SEMANTIC:
      return IREE_SV("semantic function");
    case LOOM_LOW_INVOKE_CALLER_LOW:
      return IREE_SV("low function");
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t loom_low_emit_function_contract_error(
    const loom_module_t* module, const loom_op_t* op, uint16_t attr_index,
    iree_string_view_t contract_name, iree_string_view_t reason,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_symbol_name(module, loom_low_function_symbol(op))),
      loom_param_with_field_ref(
          loom_param_string(contract_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
      loom_param_string(reason),
  };
  return loom_low_emit(emitter, op,
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 23),
                       params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_verify_optional_enum_is_named(
    const loom_module_t* module, const loom_op_t* op, uint16_t attr_index,
    uint8_t value, iree_string_view_t contract_name, iree_string_view_t reason,
    iree_diagnostic_emitter_t emitter) {
  if (!loom_low_optional_attr_is_present(op, attr_index) || value != 0) {
    return iree_ok_status();
  }
  return loom_low_emit_function_contract_error(module, op, attr_index,
                                               contract_name, reason, emitter);
}

static iree_status_t loom_low_verify_function_exactness_modes(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  if (loom_low_func_def_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_low_verify_optional_enum_is_named(
        module, op, loom_low_func_def_allocation_ATTR_INDEX,
        loom_low_func_def_allocation(op), IREE_SV("allocation"),
        IREE_SV("explicit allocation mode must name virtual, assigned, or "
                "fixed"),
        emitter));
    IREE_RETURN_IF_ERROR(loom_low_verify_optional_enum_is_named(
        module, op, loom_low_func_def_schedule_ATTR_INDEX,
        loom_low_func_def_schedule(op), IREE_SV("schedule"),
        IREE_SV("explicit schedule mode must name free, constrained, or "
                "locked"),
        emitter));
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_enum_is_named(
      module, op, loom_low_func_decl_allocation_ATTR_INDEX,
      loom_low_func_decl_allocation(op), IREE_SV("allocation"),
      IREE_SV("explicit allocation mode must name virtual, assigned, or fixed"),
      emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_enum_is_named(
      module, op, loom_low_func_decl_schedule_ATTR_INDEX,
      loom_low_func_decl_schedule(op), IREE_SV("schedule"),
      IREE_SV("explicit schedule mode must name free, constrained, or locked"),
      emitter));
  return iree_ok_status();
}

static bool loom_low_function_code_import_is_present(const loom_op_t* op,
                                                     uint16_t* out_attr_index) {
  uint16_t import_kind_attr_index = loom_low_func_def_import_kind_ATTR_INDEX;
  uint16_t code_symbol_attr_index = loom_low_func_def_code_symbol_ATTR_INDEX;
  if (loom_low_func_decl_isa(op)) {
    import_kind_attr_index = loom_low_func_decl_import_kind_ATTR_INDEX;
    code_symbol_attr_index = loom_low_func_decl_code_symbol_ATTR_INDEX;
  }
  if (loom_low_optional_attr_is_present(op, import_kind_attr_index)) {
    *out_attr_index = import_kind_attr_index;
    return true;
  }
  if (loom_low_optional_attr_is_present(op, code_symbol_attr_index)) {
    *out_attr_index = code_symbol_attr_index;
    return true;
  }
  return false;
}

static iree_status_t loom_low_verify_no_code_import_on_def(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  uint16_t attr_index = 0;
  if (!loom_low_function_code_import_is_present(op, &attr_index)) {
    return iree_ok_status();
  }
  return loom_low_emit_function_contract_error(
      module, op, attr_index, IREE_SV("import"),
      IREE_SV("imported code belongs on low.func.decl; low.func.def owns an "
              "inline body"),
      emitter);
}

static iree_status_t loom_low_verify_decl_code_import(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  bool import_kind_present = loom_low_optional_attr_is_present(
      op, loom_low_func_decl_import_kind_ATTR_INDEX);
  bool code_symbol_present = loom_low_optional_attr_is_present(
      op, loom_low_func_decl_code_symbol_ATTR_INDEX);
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_enum_is_named(
      module, op, loom_low_func_decl_import_kind_ATTR_INDEX,
      loom_low_func_decl_import_kind(op), IREE_SV("import"),
      IREE_SV("import kind must name vm, native, rocasm, or object"), emitter));
  if (import_kind_present && code_symbol_present) {
    iree_string_view_t code_symbol =
        loom_low_string_or_empty(module, loom_low_func_decl_code_symbol(op));
    if (!iree_string_view_is_empty(code_symbol)) {
      return iree_ok_status();
    }
    return loom_low_emit_function_contract_error(
        module, op, loom_low_func_decl_code_symbol_ATTR_INDEX,
        IREE_SV("import"), IREE_SV("code symbol must not be empty"), emitter);
  }
  if (import_kind_present) {
    return loom_low_emit_function_contract_error(
        module, op, loom_low_func_decl_import_kind_ATTR_INDEX,
        IREE_SV("import"), IREE_SV("import kind requires a code symbol string"),
        emitter);
  }
  if (code_symbol_present) {
    return loom_low_emit_function_contract_error(
        module, op, loom_low_func_decl_code_symbol_ATTR_INDEX,
        IREE_SV("import"), IREE_SV("code symbol requires an import kind"),
        emitter);
  }
  return iree_ok_status();
}

static bool loom_low_abi_field_kind_matches_op(
    loom_low_abi_field_kind_t field_kind, const loom_op_t* op) {
  switch (field_kind) {
    case LOOM_LOW_ABI_FIELD_OPERAND:
      return loom_low_abi_operand_isa(op);
    case LOOM_LOW_ABI_FIELD_RESULT:
      return loom_low_abi_result_isa(op);
    default:
      return false;
  }
}

static loom_symbol_ref_t loom_low_abi_entry_symbol(const loom_op_t* op) {
  if (loom_low_abi_operand_isa(op)) return loom_low_abi_operand_symbol(op);
  if (loom_low_abi_result_isa(op)) return loom_low_abi_result_symbol(op);
  return loom_symbol_ref_null();
}

static loom_symbol_ref_t loom_low_abi_entry_adapter(const loom_op_t* op) {
  if (loom_low_abi_operand_isa(op)) return loom_low_abi_operand_adapter(op);
  if (loom_low_abi_result_isa(op)) return loom_low_abi_result_adapter(op);
  return loom_symbol_ref_null();
}

static int64_t loom_low_abi_entry_index(const loom_op_t* op) {
  if (loom_low_abi_operand_isa(op)) return loom_low_abi_operand_index(op);
  if (loom_low_abi_result_isa(op)) return loom_low_abi_result_index(op);
  return -1;
}

static uint8_t loom_low_abi_entry_conversion(const loom_op_t* op) {
  if (loom_low_abi_operand_isa(op)) return loom_low_abi_operand_conversion(op);
  if (loom_low_abi_result_isa(op)) return loom_low_abi_result_conversion(op);
  return UINT8_MAX;
}

static loom_type_t loom_low_abi_entry_semantic_type(const loom_module_t* module,
                                                    const loom_op_t* op) {
  if (loom_low_abi_operand_isa(op)) {
    return loom_low_type_attr(module, loom_low_abi_operand_semantic_type(op));
  }
  if (loom_low_abi_result_isa(op)) {
    return loom_low_type_attr(module, loom_low_abi_result_semantic_type(op));
  }
  return loom_type_none();
}

static loom_type_t loom_low_abi_entry_abi_type(const loom_module_t* module,
                                               const loom_op_t* op) {
  if (loom_low_abi_operand_isa(op)) {
    return loom_low_type_attr(module, loom_low_abi_operand_abi_type(op));
  }
  if (loom_low_abi_result_isa(op)) {
    return loom_low_type_attr(module, loom_low_abi_result_abi_type(op));
  }
  return loom_type_none();
}

static loom_low_abi_entry_t loom_low_abi_entry_load(const loom_module_t* module,
                                                    const loom_op_t* op) {
  return (loom_low_abi_entry_t){
      .op = op,
      .symbol = loom_low_abi_entry_symbol(op),
      .adapter = loom_low_abi_entry_adapter(op),
      .index = loom_low_abi_entry_index(op),
      .conversion = loom_low_abi_entry_conversion(op),
      .semantic_type = loom_low_abi_entry_semantic_type(module, op),
      .abi_type = loom_low_abi_entry_abi_type(module, op),
  };
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

static iree_status_t loom_low_emit_symbol_kind_mismatch(
    const loom_module_t* module, const loom_op_t* op, loom_symbol_ref_t ref,
    uint16_t attr_index, const loom_symbol_t* symbol,
    iree_string_view_t expected_kind, iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(loom_low_symbol_name(module, ref)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
      loom_param_string(loom_low_symbol_definition_name(symbol)),
      loom_param_string(expected_kind),
  };
  loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("defined here"),
      .op = symbol ? symbol->defining_op : NULL,
  }};
  return loom_low_emit_related(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3), params,
      IREE_ARRAYSIZE(params), related,
      symbol && symbol->defining_op ? IREE_ARRAYSIZE(related) : 0);
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

static loom_low_invoke_caller_t loom_low_find_invoke_caller(
    const loom_module_t* module, const loom_op_t* invoke_op) {
  const loom_op_t* parent = invoke_op->parent_op;
  while (parent) {
    loom_func_like_t func = loom_func_like_cast(module, (loom_op_t*)parent);
    if (loom_func_like_isa(func) && loom_func_like_body(func)) {
      return (loom_low_invoke_caller_t){
          .op = parent,
          .kind = loom_low_func_def_isa(parent)
                      ? LOOM_LOW_INVOKE_CALLER_LOW
                      : LOOM_LOW_INVOKE_CALLER_SEMANTIC,
      };
    }
    parent = parent->parent_op;
  }
  return (loom_low_invoke_caller_t){
      .op = NULL,
      .kind = LOOM_LOW_INVOKE_CALLER_NONE,
  };
}

static const loom_op_t* loom_low_find_enclosing_low_func_def(
    const loom_module_t* module, const loom_op_t* nested_op) {
  const loom_op_t* parent = nested_op->parent_op;
  while (parent) {
    if (loom_low_func_def_isa(parent)) return parent;
    loom_func_like_t func = loom_func_like_cast(module, (loom_op_t*)parent);
    if (loom_func_like_isa(func) && loom_func_like_body(func)) return NULL;
    parent = parent->parent_op;
  }
  return NULL;
}

static bool loom_low_func_like_is_pure(const loom_module_t* module,
                                       const loom_op_t* op) {
  loom_func_like_t func = loom_func_like_cast(module, (loom_op_t*)op);
  if (!loom_func_like_isa(func)) return false;
  if (loom_func_like_purity(func) != 0) return true;
  loom_region_t* body = loom_func_like_body(func);
  return body && !loom_region_has_read_effects(body) &&
         !loom_region_has_write_effects(body);
}

static bool loom_low_abi_metadata_matches_adapter(
    const loom_op_t* op, loom_symbol_ref_t adapter_ref) {
  if (loom_low_abi_effect_isa(op)) {
    return loom_low_symbol_ref_equal(loom_low_abi_effect_adapter(op),
                                     adapter_ref);
  }
  if (loom_low_abi_clobber_isa(op)) {
    return loom_low_symbol_ref_equal(loom_low_abi_clobber_adapter(op),
                                     adapter_ref);
  }
  return false;
}

static loom_low_abi_metadata_counts_t loom_low_count_abi_metadata(
    const loom_module_t* module, loom_symbol_ref_t adapter_ref) {
  loom_low_abi_metadata_counts_t counts = {0};
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    const loom_op_t* op = symbol->defining_op;
    if (!op || !loom_low_abi_metadata_matches_adapter(op, adapter_ref)) {
      continue;
    }
    if (loom_low_abi_effect_isa(op)) ++counts.effect_count;
    if (loom_low_abi_clobber_isa(op)) ++counts.clobber_count;
  }
  return counts;
}

static iree_status_t loom_low_emit_descriptor_key_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t attr_index,
    iree_string_view_t field_name, iree_string_view_t key,
    iree_string_view_t expected) {
  loom_diagnostic_field_ref_t attr_ref =
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, attr_index);
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(loom_param_string(field_name), attr_ref),
      loom_param_string(key),
      loom_param_string(expected),
  };
  return loom_low_emit(emitter, op,
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 27),
                       params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_verify_qualified_key_attr(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_string_id_t string_id,
    uint16_t attr_index, iree_string_view_t field_name,
    iree_string_view_t expected) {
  iree_string_view_t key = loom_low_string_or_empty(module, string_id);
  if (loom_low_qualified_key_is_valid(key)) return iree_ok_status();
  return loom_low_emit_descriptor_key_error(emitter, op, attr_index, field_name,
                                            key, expected);
}

static bool loom_low_is_power_of_two_i64(int64_t value) {
  return value > 0 && (((uint64_t)value & ((uint64_t)value - 1)) == 0);
}

static iree_status_t loom_low_emit_structural_storage_error(
    const loom_module_t* module, const loom_op_t* op,
    loom_diagnostic_field_ref_t field_ref, iree_string_view_t field_name,
    iree_string_view_t reason, const loom_diagnostic_related_op_t* related,
    iree_host_size_t related_count, iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_op_name(module, op)),
      loom_param_with_field_ref(loom_param_string(field_name), field_ref),
      loom_param_string(reason),
  };
  return loom_low_emit_related(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 24),
      params, IREE_ARRAYSIZE(params), related, related_count);
}

static iree_status_t loom_low_emit_structural_storage_attr_error(
    const loom_module_t* module, const loom_op_t* op, uint16_t attr_index,
    iree_string_view_t field_name, iree_string_view_t reason,
    iree_diagnostic_emitter_t emitter) {
  return loom_low_emit_structural_storage_error(
      module, op,
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, attr_index),
      field_name, reason, NULL, 0, emitter);
}

static iree_status_t loom_low_verify_slot_function(
    const loom_module_t* module, const loom_op_t* op,
    loom_symbol_ref_t function_ref, uint16_t attr_index,
    iree_diagnostic_emitter_t emitter) {
  const loom_symbol_t* symbol =
      loom_low_lookup_defined_symbol(module, function_ref);
  if (!symbol) return iree_ok_status();
  if (loom_low_func_def_isa(symbol->defining_op)) return iree_ok_status();
  return loom_low_emit_symbol_kind_mismatch(
      module, op, function_ref, attr_index, symbol,
      IREE_SV("low function definition"), emitter);
}

static iree_status_t loom_low_load_slot(const loom_module_t* module,
                                        const loom_op_t* op,
                                        loom_symbol_ref_t slot_ref,
                                        uint16_t slot_attr_index,
                                        iree_diagnostic_emitter_t emitter,
                                        const loom_op_t** out_slot_op) {
  *out_slot_op = NULL;
  const loom_symbol_t* slot_symbol =
      loom_low_lookup_defined_symbol(module, slot_ref);
  if (!slot_symbol) return iree_ok_status();
  if (!loom_low_slot_isa(slot_symbol->defining_op)) {
    return loom_low_emit_symbol_kind_mismatch(module, op, slot_ref,
                                              slot_attr_index, slot_symbol,
                                              IREE_SV("low slot"), emitter);
  }
  *out_slot_op = slot_symbol->defining_op;
  return iree_ok_status();
}

static iree_status_t loom_low_verify_slot_use(
    const loom_module_t* module, const loom_op_t* op,
    loom_symbol_ref_t slot_ref, uint16_t slot_attr_index, int64_t offset,
    uint16_t offset_attr_index, iree_diagnostic_emitter_t emitter) {
  const loom_op_t* slot_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_load_slot(module, op, slot_ref, slot_attr_index,
                                          emitter, &slot_op));
  if (!slot_op) return iree_ok_status();

  const loom_op_t* enclosing_func =
      loom_low_find_enclosing_low_func_def(module, op);
  if (!enclosing_func) {
    return loom_low_emit_structural_storage_error(
        module, op, loom_diagnostic_field_ref_none(),
        IREE_SV("enclosing function"),
        IREE_SV("slot traffic must be nested under a low function body"), NULL,
        0, emitter);
  }

  if (!loom_low_symbol_ref_equal(loom_low_slot_function(slot_op),
                                 loom_low_func_def_callee(enclosing_func))) {
    loom_diagnostic_related_op_t related[] = {{
        .label = IREE_SV("slot defined here"),
        .op = slot_op,
    }};
    return loom_low_emit_structural_storage_error(
        module, op,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                  slot_attr_index),
        IREE_SV("slot"),
        IREE_SV("slot owner must match the enclosing low function"), related,
        IREE_ARRAYSIZE(related), emitter);
  }

  if (offset < 0) {
    return loom_low_emit_structural_storage_attr_error(
        module, op, offset_attr_index, IREE_SV("offset"),
        IREE_SV("offset must be non-negative"), emitter);
  }

  int64_t slot_size = loom_low_slot_size(slot_op);
  if (slot_size > 0 && offset >= slot_size) {
    loom_diagnostic_related_op_t related[] = {{
        .label = IREE_SV("slot defined here"),
        .op = slot_op,
    }};
    return loom_low_emit_structural_storage_error(
        module, op,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                  offset_attr_index),
        IREE_SV("offset"),
        IREE_SV("offset must address a byte inside the referenced slot"),
        related, IREE_ARRAYSIZE(related), emitter);
  }

  return iree_ok_status();
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

static iree_status_t loom_low_emit_adapter_callee_mismatch(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature, const loom_op_t* adapter_op,
    iree_diagnostic_emitter_t emitter) {
  loom_symbol_ref_t invoke_callee = loom_low_invoke_callee(invoke_op);
  loom_symbol_ref_t adapter_ref = loom_low_invoke_adapter(invoke_op);
  loom_symbol_ref_t adapter_callee = loom_low_abi_adapter_callee(adapter_op);
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(loom_low_symbol_name(module, invoke_callee)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    loom_low_invoke_callee_ATTR_INDEX)),
      loom_param_with_field_ref(
          loom_param_string(loom_low_symbol_name(module, adapter_ref)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    loom_low_invoke_adapter_ATTR_INDEX)),
      loom_param_string(loom_low_symbol_name(module, adapter_callee)),
  };
  loom_diagnostic_related_op_t related[] = {
      {
          .label = IREE_SV("callee defined here"),
          .op = signature->definition_op,
      },
      {
          .label = IREE_SV("adapter defined here"),
          .op = adapter_op,
      },
  };
  return loom_low_emit_related(
      emitter, invoke_op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 15),
      params, IREE_ARRAYSIZE(params), related, IREE_ARRAYSIZE(related));
}

static iree_status_t loom_low_emit_adapter_count_mismatch(
    const loom_module_t* module, const loom_op_t* op,
    loom_symbol_ref_t callee_ref, const loom_op_t* callee_op,
    loom_symbol_ref_t adapter_ref, const loom_op_t* adapter_op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t field_kind,
    int64_t actual_count, int64_t expected_count, uint8_t conversion) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_symbol_name(module, callee_ref)),
      loom_param_string(loom_low_symbol_name(module, adapter_ref)),
      loom_param_string(field_kind),
      loom_param_i64(actual_count),
      loom_param_i64(expected_count),
      loom_param_string(loom_low_abi_conversion_name(conversion)),
  };
  loom_diagnostic_related_op_t related[] = {
      {0},
      {0},
  };
  iree_host_size_t related_count = 0;
  if (callee_op) {
    related[related_count++] = (loom_diagnostic_related_op_t){
        .label = IREE_SV("callee defined here"),
        .op = callee_op,
    };
  }
  if (adapter_op) {
    related[related_count++] = (loom_diagnostic_related_op_t){
        .label = IREE_SV("adapter defined here"),
        .op = adapter_op,
    };
  }
  return loom_low_emit_related(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 16),
      params, IREE_ARRAYSIZE(params), related, related_count);
}

static iree_status_t loom_low_emit_adapter_type_mismatch(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature, const loom_op_t* adapter_op,
    iree_diagnostic_emitter_t emitter,
    loom_diagnostic_field_kind_t field_ref_kind, iree_string_view_t field_kind,
    uint16_t field_index, loom_type_t actual_type,
    iree_string_view_t callee_field_kind, loom_type_t expected_type,
    uint8_t conversion) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_symbol_name(module, loom_low_invoke_callee(invoke_op))),
      loom_param_string(
          loom_low_symbol_name(module, loom_low_invoke_adapter(invoke_op))),
      loom_param_string(loom_low_abi_conversion_name(conversion)),
      loom_param_with_field_ref(
          loom_param_string(field_kind),
          loom_diagnostic_field_ref(field_ref_kind, field_index)),
      loom_param_u32(field_index),
      loom_param_type(actual_type),
      loom_param_string(callee_field_kind),
      loom_param_type(expected_type),
  };
  loom_diagnostic_related_op_t related[] = {
      {
          .label = IREE_SV("callee defined here"),
          .op = signature->definition_op,
      },
      {
          .label = IREE_SV("adapter defined here"),
          .op = adapter_op,
      },
  };
  return loom_low_emit_related(
      emitter, invoke_op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 17),
      params, IREE_ARRAYSIZE(params), related, IREE_ARRAYSIZE(related));
}

static iree_status_t loom_low_emit_abi_entry_error(
    const loom_module_t* module, const loom_op_t* op,
    loom_symbol_ref_t adapter_ref, const loom_op_t* adapter_op,
    iree_string_view_t entry_name, const loom_op_t* entry_op,
    loom_low_abi_field_kind_t field_kind, int64_t field_index,
    iree_string_view_t reason, iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_symbol_name(module, adapter_ref)),
      loom_param_string(entry_name),
      loom_param_string(loom_low_abi_field_kind_name(field_kind)),
      loom_param_i64(field_index),
      loom_param_string(reason),
  };
  loom_diagnostic_related_op_t related[] = {
      {0},
      {0},
  };
  iree_host_size_t related_count = 0;
  if (adapter_op) {
    related[related_count++] = (loom_diagnostic_related_op_t){
        .label = IREE_SV("adapter defined here"),
        .op = adapter_op,
    };
  }
  if (entry_op) {
    related[related_count++] = (loom_diagnostic_related_op_t){
        .label = IREE_SV("entry defined here"),
        .op = entry_op,
    };
  }
  return loom_low_emit_related(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 18),
      params, IREE_ARRAYSIZE(params), related, related_count);
}

static iree_status_t loom_low_emit_abi_entry_type_mismatch(
    const loom_module_t* module, const loom_op_t* op,
    loom_symbol_ref_t adapter_ref, const loom_op_t* adapter_op,
    const loom_low_abi_entry_t* entry, loom_low_abi_field_kind_t field_kind,
    uint16_t field_index, loom_type_t actual_type,
    iree_string_view_t expected_field_kind, loom_type_t expected_type,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_symbol_name(module, adapter_ref)),
      loom_param_string(loom_low_symbol_name(module, entry->symbol)),
      loom_param_string(loom_low_abi_value_conversion_name(entry->conversion)),
      loom_param_string(loom_low_abi_field_kind_name(field_kind)),
      loom_param_u32(field_index),
      loom_param_type(actual_type),
      loom_param_string(expected_field_kind),
      loom_param_type(expected_type),
  };
  loom_diagnostic_related_op_t related[] = {
      {0},
      {0},
  };
  iree_host_size_t related_count = 0;
  if (adapter_op) {
    related[related_count++] = (loom_diagnostic_related_op_t){
        .label = IREE_SV("adapter defined here"),
        .op = adapter_op,
    };
  }
  if (entry->op) {
    related[related_count++] = (loom_diagnostic_related_op_t){
        .label = IREE_SV("entry defined here"),
        .op = entry->op,
    };
  }
  return loom_low_emit_related(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 19),
      params, IREE_ARRAYSIZE(params), related, related_count);
}

static iree_status_t loom_low_emit_invoke_context_error(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_invoke_caller_t* caller, const loom_op_t* callee_op,
    const loom_op_t* adapter_op, iree_string_view_t reason,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_symbol_name(module, loom_low_invoke_callee(invoke_op))),
      loom_param_string(loom_low_invoke_caller_kind_name(caller->kind)),
      loom_param_string(reason),
  };
  loom_diagnostic_related_op_t related[] = {
      {0},
      {0},
      {0},
  };
  iree_host_size_t related_count = 0;
  if (caller->op) {
    related[related_count++] = (loom_diagnostic_related_op_t){
        .label = IREE_SV("caller defined here"),
        .op = caller->op,
    };
  }
  if (callee_op) {
    related[related_count++] = (loom_diagnostic_related_op_t){
        .label = IREE_SV("callee defined here"),
        .op = callee_op,
    };
  }
  if (adapter_op) {
    related[related_count++] = (loom_diagnostic_related_op_t){
        .label = IREE_SV("adapter defined here"),
        .op = adapter_op,
    };
  }
  return loom_low_emit_related(
      emitter, invoke_op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 20),
      params, IREE_ARRAYSIZE(params), related, related_count);
}

static iree_status_t loom_low_emit_abi_metadata_error(
    const loom_module_t* module, const loom_op_t* op,
    loom_symbol_ref_t adapter_ref, const loom_op_t* adapter_op,
    iree_string_view_t record_kind, loom_symbol_ref_t record_ref,
    iree_string_view_t reason, iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_symbol_name(module, adapter_ref)),
      loom_param_string(record_kind),
      loom_param_string(loom_low_symbol_name(module, record_ref)),
      loom_param_string(reason),
  };
  loom_diagnostic_related_op_t related[] = {
      {0},
      {0},
  };
  iree_host_size_t related_count = 0;
  if (adapter_op) {
    related[related_count++] = (loom_diagnostic_related_op_t){
        .label = IREE_SV("adapter defined here"),
        .op = adapter_op,
    };
  }
  related[related_count++] = (loom_diagnostic_related_op_t){
      .label = IREE_SV("metadata defined here"),
      .op = op,
  };
  return loom_low_emit_related(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 21),
      params, IREE_ARRAYSIZE(params), related, related_count);
}

static iree_status_t loom_low_emit_invoke_purity_effect_error(
    const loom_module_t* module, const loom_op_t* invoke_op,
    iree_string_view_t boundary_name, iree_string_view_t reason,
    const loom_op_t* related_op, iree_string_view_t related_label,
    loom_low_abi_metadata_counts_t counts, iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(
          loom_low_symbol_name(module, loom_low_invoke_callee(invoke_op))),
      loom_param_string(boundary_name),
      loom_param_string(reason),
      loom_param_u32(counts.effect_count),
      loom_param_u32(counts.clobber_count),
  };
  loom_diagnostic_related_op_t related[] = {
      {
          .label = related_label,
          .op = related_op,
      },
  };
  return loom_low_emit_related(
      emitter, invoke_op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 22),
      params, IREE_ARRAYSIZE(params), related,
      related_op ? IREE_ARRAYSIZE(related) : 0);
}

static iree_status_t loom_low_verify_descriptor_key(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_string_id_t opcode_id,
    uint16_t attr_index) {
  return loom_low_verify_qualified_key_attr(
      module, op, emitter, opcode_id, attr_index, IREE_SV("opcode"),
      IREE_SV("a namespace-qualified descriptor key with non-empty identifier "
              "segments"));
}

static iree_status_t loom_low_verify_abi_metadata_adapter(
    const loom_module_t* module, const loom_op_t* op,
    loom_symbol_ref_t adapter_ref, uint16_t adapter_attr_index,
    iree_diagnostic_emitter_t emitter, const loom_op_t** out_adapter_op) {
  *out_adapter_op = NULL;
  const loom_symbol_t* adapter_symbol =
      loom_low_lookup_defined_symbol(module, adapter_ref);
  if (!adapter_symbol) return iree_ok_status();
  const loom_op_t* adapter_op = adapter_symbol->defining_op;
  if (!loom_low_abi_adapter_isa(adapter_op)) {
    IREE_RETURN_IF_ERROR(loom_low_emit_symbol_kind_mismatch(
        module, op, adapter_ref, adapter_attr_index, adapter_symbol,
        IREE_SV("low ABI adapter"), emitter));
    return iree_ok_status();
  }
  *out_adapter_op = adapter_op;
  return iree_ok_status();
}

static iree_status_t loom_low_verify_abi_effect_resource(
    const loom_module_t* module, const loom_op_t* op,
    loom_symbol_ref_t adapter_ref, const loom_op_t* adapter_op,
    iree_diagnostic_emitter_t emitter) {
  bool resource_is_present = loom_low_optional_attr_is_present(
      op, loom_low_abi_effect_resource_ATTR_INDEX);
  if (resource_is_present) {
    IREE_RETURN_IF_ERROR(loom_low_verify_qualified_key_attr(
        module, op, emitter, loom_low_abi_effect_resource(op),
        loom_low_abi_effect_resource_ATTR_INDEX, IREE_SV("resource"),
        IREE_SV("a namespace-qualified ABI resource key with non-empty "
                "identifier segments")));
  }

  switch (loom_low_abi_effect_kind(op)) {
    case LOOM_LOW_ABI_EFFECT_KIND_READ:
    case LOOM_LOW_ABI_EFFECT_KIND_WRITE:
    case LOOM_LOW_ABI_EFFECT_KIND_READWRITE:
      if (!resource_is_present) {
        return loom_low_emit_abi_metadata_error(
            module, op, adapter_ref, adapter_op, IREE_SV("effect"),
            loom_low_abi_effect_symbol(op),
            IREE_SV("read/write/readwrite effects require a resource key"),
            emitter);
      }
      return iree_ok_status();
    case LOOM_LOW_ABI_EFFECT_KIND_CALL:
    case LOOM_LOW_ABI_EFFECT_KIND_UNKNOWN:
      return iree_ok_status();
    default:
      return loom_low_emit_abi_metadata_error(
          module, op, adapter_ref, adapter_op, IREE_SV("effect"),
          loom_low_abi_effect_symbol(op), IREE_SV("unknown effect kind"),
          emitter);
  }
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

static iree_status_t loom_low_verify_adapter_signature_count(
    const loom_module_t* module, const loom_op_t* adapter_op,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter, iree_string_view_t field_kind,
    int64_t actual_count, int64_t expected_count) {
  if (actual_count == expected_count) return iree_ok_status();
  return loom_low_emit_adapter_count_mismatch(
      module, adapter_op, loom_low_abi_adapter_callee(adapter_op),
      signature->definition_op, loom_low_abi_adapter_symbol(adapter_op),
      adapter_op, emitter, field_kind, actual_count, expected_count,
      loom_low_abi_adapter_conversion(adapter_op));
}

static iree_status_t loom_low_verify_invoke_adapter_count(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature, const loom_op_t* adapter_op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t field_kind,
    int64_t actual_count, int64_t expected_count) {
  if (actual_count == expected_count) return iree_ok_status();
  return loom_low_emit_adapter_count_mismatch(
      module, invoke_op, loom_low_invoke_callee(invoke_op),
      signature->definition_op, loom_low_invoke_adapter(invoke_op), adapter_op,
      emitter, field_kind, actual_count, expected_count,
      loom_low_abi_adapter_conversion(adapter_op));
}

static iree_status_t loom_low_verify_invoke_adapter_argument_types(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature, const loom_op_t* adapter_op,
    iree_diagnostic_emitter_t emitter) {
  uint8_t conversion = loom_low_abi_adapter_conversion(adapter_op);
  if (conversion != LOOM_LOW_ABI_ADAPTER_CONVERSION_DIRECT) {
    return iree_ok_status();
  }
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
    IREE_RETURN_IF_ERROR(loom_low_emit_adapter_type_mismatch(
        module, invoke_op, signature, adapter_op, emitter,
        LOOM_DIAGNOSTIC_FIELD_OPERAND, IREE_SV("operand"), i, actual_type,
        IREE_SV("argument"), expected_type, conversion));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_invoke_adapter_result_types(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature, const loom_op_t* adapter_op,
    iree_diagnostic_emitter_t emitter) {
  uint8_t conversion = loom_low_abi_adapter_conversion(adapter_op);
  if (conversion != LOOM_LOW_ABI_ADAPTER_CONVERSION_DIRECT) {
    return iree_ok_status();
  }
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
    IREE_RETURN_IF_ERROR(loom_low_emit_adapter_type_mismatch(
        module, invoke_op, signature, adapter_op, emitter,
        LOOM_DIAGNOSTIC_FIELD_RESULT, IREE_SV("result"), i, actual_type,
        IREE_SV("result"), expected_type, conversion));
  }
  return iree_ok_status();
}

static bool loom_low_find_abi_entry(const loom_module_t* module,
                                    loom_symbol_ref_t adapter_ref,
                                    loom_low_abi_field_kind_t field_kind,
                                    uint16_t field_index,
                                    loom_low_abi_entry_t* out_entry,
                                    loom_low_abi_entry_t* out_duplicate) {
  bool found = false;
  if (out_entry) memset(out_entry, 0, sizeof(*out_entry));
  if (out_duplicate) memset(out_duplicate, 0, sizeof(*out_duplicate));
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    const loom_op_t* entry_op = symbol->defining_op;
    if (!entry_op ||
        !loom_low_abi_field_kind_matches_op(field_kind, entry_op)) {
      continue;
    }
    loom_low_abi_entry_t entry = loom_low_abi_entry_load(module, entry_op);
    if (!loom_low_symbol_ref_equal(entry.adapter, adapter_ref) ||
        entry.index != field_index) {
      continue;
    }
    if (found) {
      if (out_duplicate) *out_duplicate = entry;
      return true;
    }
    found = true;
    if (out_entry) *out_entry = entry;
  }
  return found;
}

static iree_status_t loom_low_verify_abi_entry_conversion_shape(
    const loom_module_t* module, const loom_op_t* op,
    const loom_low_abi_entry_t* entry, loom_low_abi_field_kind_t field_kind,
    const loom_op_t* adapter_op, iree_diagnostic_emitter_t emitter) {
  switch (entry->conversion) {
    case LOOM_LOW_CONVERSION_DIRECT:
      if (loom_type_equal(entry->semantic_type, entry->abi_type)) {
        return iree_ok_status();
      }
      return loom_low_emit_abi_entry_type_mismatch(
          module, op, entry->adapter, adapter_op, entry, field_kind,
          (uint16_t)entry->index, entry->semantic_type, IREE_SV("ABI"),
          entry->abi_type, emitter);
    case LOOM_LOW_CONVERSION_SCALAR_TO_REGISTER:
      if (field_kind != LOOM_LOW_ABI_FIELD_OPERAND) {
        return loom_low_emit_abi_entry_error(
            module, op, entry->adapter, adapter_op,
            loom_low_symbol_name(module, entry->symbol), entry->op, field_kind,
            entry->index,
            IREE_SV("scalar_to_register is only valid for operand entries"),
            emitter);
      }
      if (loom_type_is_scalar(entry->semantic_type) &&
          loom_type_is_register(entry->abi_type)) {
        return iree_ok_status();
      }
      return loom_low_emit_abi_entry_error(
          module, op, entry->adapter, adapter_op,
          loom_low_symbol_name(module, entry->symbol), entry->op, field_kind,
          entry->index,
          IREE_SV("scalar_to_register requires a scalar semantic type and "
                  "register ABI type"),
          emitter);
    case LOOM_LOW_CONVERSION_REGISTER_TO_SCALAR:
      if (field_kind != LOOM_LOW_ABI_FIELD_RESULT) {
        return loom_low_emit_abi_entry_error(
            module, op, entry->adapter, adapter_op,
            loom_low_symbol_name(module, entry->symbol), entry->op, field_kind,
            entry->index,
            IREE_SV("register_to_scalar is only valid for result entries"),
            emitter);
      }
      if (loom_type_is_scalar(entry->semantic_type) &&
          loom_type_is_register(entry->abi_type)) {
        return iree_ok_status();
      }
      return loom_low_emit_abi_entry_error(
          module, op, entry->adapter, adapter_op,
          loom_low_symbol_name(module, entry->symbol), entry->op, field_kind,
          entry->index,
          IREE_SV("register_to_scalar requires a register ABI type and scalar "
                  "semantic type"),
          emitter);
    case LOOM_LOW_CONVERSION_VALUE_TO_REGISTER:
      if (field_kind != LOOM_LOW_ABI_FIELD_OPERAND) {
        return loom_low_emit_abi_entry_error(
            module, op, entry->adapter, adapter_op,
            loom_low_symbol_name(module, entry->symbol), entry->op, field_kind,
            entry->index,
            IREE_SV("value_to_register is only valid for operand entries"),
            emitter);
      }
      if (!loom_type_is_register(entry->semantic_type) &&
          loom_type_is_register(entry->abi_type)) {
        return iree_ok_status();
      }
      return loom_low_emit_abi_entry_error(
          module, op, entry->adapter, adapter_op,
          loom_low_symbol_name(module, entry->symbol), entry->op, field_kind,
          entry->index,
          IREE_SV("value_to_register requires a non-register semantic type and "
                  "register ABI type"),
          emitter);
    case LOOM_LOW_CONVERSION_REGISTER_TO_VALUE:
      if (field_kind != LOOM_LOW_ABI_FIELD_RESULT) {
        return loom_low_emit_abi_entry_error(
            module, op, entry->adapter, adapter_op,
            loom_low_symbol_name(module, entry->symbol), entry->op, field_kind,
            entry->index,
            IREE_SV("register_to_value is only valid for result entries"),
            emitter);
      }
      if (!loom_type_is_register(entry->semantic_type) &&
          loom_type_is_register(entry->abi_type)) {
        return iree_ok_status();
      }
      return loom_low_emit_abi_entry_error(
          module, op, entry->adapter, adapter_op,
          loom_low_symbol_name(module, entry->symbol), entry->op, field_kind,
          entry->index,
          IREE_SV("register_to_value requires a register ABI type and "
                  "non-register semantic type"),
          emitter);
    default:
      return loom_low_emit_abi_entry_error(
          module, op, entry->adapter, adapter_op,
          loom_low_symbol_name(module, entry->symbol), entry->op, field_kind,
          entry->index, IREE_SV("unknown conversion rule"), emitter);
  }
}

static iree_status_t loom_low_verify_abi_entry_record(
    const loom_module_t* module, const loom_op_t* op,
    loom_low_abi_field_kind_t field_kind, iree_diagnostic_emitter_t emitter) {
  loom_low_abi_entry_t entry = loom_low_abi_entry_load(module, op);
  const loom_symbol_t* adapter_symbol =
      loom_low_lookup_defined_symbol(module, entry.adapter);
  if (!adapter_symbol) {
    return iree_ok_status();
  }
  const loom_op_t* adapter_op = adapter_symbol->defining_op;
  uint16_t adapter_attr_index = field_kind == LOOM_LOW_ABI_FIELD_OPERAND
                                    ? loom_low_abi_operand_adapter_ATTR_INDEX
                                    : loom_low_abi_result_adapter_ATTR_INDEX;
  if (!loom_low_abi_adapter_isa(adapter_op)) {
    IREE_RETURN_IF_ERROR(loom_low_emit_symbol_kind_mismatch(
        module, op, entry.adapter, adapter_attr_index, adapter_symbol,
        IREE_SV("low ABI adapter"), emitter));
    return iree_ok_status();
  }

  if (loom_low_abi_adapter_conversion(adapter_op) !=
      LOOM_LOW_ABI_ADAPTER_CONVERSION_MAPPED) {
    return loom_low_emit_abi_entry_error(
        module, op, entry.adapter, adapter_op,
        loom_low_symbol_name(module, entry.symbol), op, field_kind, entry.index,
        IREE_SV("only mapped adapters accept operand/result mapping entries"),
        emitter);
  }

  int64_t expected_count = field_kind == LOOM_LOW_ABI_FIELD_OPERAND
                               ? loom_low_abi_adapter_operand_count(adapter_op)
                               : loom_low_abi_adapter_result_count(adapter_op);
  if (entry.index < 0 || entry.index >= expected_count) {
    return loom_low_emit_abi_entry_error(
        module, op, entry.adapter, adapter_op,
        loom_low_symbol_name(module, entry.symbol), op, field_kind, entry.index,
        IREE_SV("index is outside the adapter arity"), emitter);
  }

  return loom_low_verify_abi_entry_conversion_shape(
      module, op, &entry, field_kind, adapter_op, emitter);
}

static iree_status_t loom_low_verify_adapter_mapped_entry(
    const loom_module_t* module, const loom_op_t* adapter_op,
    loom_low_abi_field_kind_t field_kind, uint16_t field_index,
    loom_type_t expected_abi_type, iree_diagnostic_emitter_t emitter) {
  loom_symbol_ref_t adapter_ref = loom_low_abi_adapter_symbol(adapter_op);
  loom_low_abi_entry_t entry = {0};
  loom_low_abi_entry_t duplicate = {0};
  bool found = loom_low_find_abi_entry(module, adapter_ref, field_kind,
                                       field_index, &entry, &duplicate);
  if (!found) {
    return loom_low_emit_abi_entry_error(
        module, adapter_op, adapter_ref, adapter_op, IREE_SV("<missing>"), NULL,
        field_kind, field_index,
        IREE_SV("mapped adapter is missing this entry"), emitter);
  }
  if (duplicate.op) {
    return loom_low_emit_abi_entry_error(
        module, adapter_op, adapter_ref, adapter_op,
        loom_low_symbol_name(module, duplicate.symbol), duplicate.op,
        field_kind, field_index,
        IREE_SV("mapped adapter has more than one entry for this index"),
        emitter);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_abi_entry_conversion_shape(
      module, adapter_op, &entry, field_kind, adapter_op, emitter));
  if (loom_type_equal(entry.abi_type, expected_abi_type)) {
    return iree_ok_status();
  }
  iree_string_view_t expected_field_kind =
      field_kind == LOOM_LOW_ABI_FIELD_OPERAND ? IREE_SV("callee argument")
                                               : IREE_SV("callee result");
  return loom_low_emit_abi_entry_type_mismatch(
      module, adapter_op, adapter_ref, adapter_op, &entry, field_kind,
      field_index, entry.abi_type, expected_field_kind, expected_abi_type,
      emitter);
}

static iree_status_t loom_low_verify_adapter_mapped_entries(
    const loom_module_t* module, const loom_op_t* adapter_op,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  uint8_t conversion = loom_low_abi_adapter_conversion(adapter_op);
  if (conversion != LOOM_LOW_ABI_ADAPTER_CONVERSION_MAPPED) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < signature->argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_adapter_mapped_entry(
        module, adapter_op, LOOM_LOW_ABI_FIELD_OPERAND, i,
        loom_module_value_type(module, signature->argument_ids[i]), emitter));
  }
  for (uint16_t i = 0; i < signature->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_adapter_mapped_entry(
        module, adapter_op, LOOM_LOW_ABI_FIELD_RESULT, i,
        loom_module_value_type(module, signature->result_ids[i]), emitter));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_invoke_mapped_entry(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature, const loom_op_t* adapter_op,
    loom_low_abi_field_kind_t field_kind, uint16_t field_index,
    loom_type_t actual_semantic_type, iree_diagnostic_emitter_t emitter) {
  loom_symbol_ref_t adapter_ref = loom_low_invoke_adapter(invoke_op);
  loom_low_abi_entry_t entry = {0};
  loom_low_abi_entry_t duplicate = {0};
  bool found = loom_low_find_abi_entry(module, adapter_ref, field_kind,
                                       field_index, &entry, &duplicate);
  if (!found) {
    return loom_low_emit_abi_entry_error(
        module, invoke_op, adapter_ref, adapter_op, IREE_SV("<missing>"), NULL,
        field_kind, field_index,
        IREE_SV("mapped adapter is missing this entry"), emitter);
  }
  if (duplicate.op) {
    return loom_low_emit_abi_entry_error(
        module, invoke_op, adapter_ref, adapter_op,
        loom_low_symbol_name(module, duplicate.symbol), duplicate.op,
        field_kind, field_index,
        IREE_SV("mapped adapter has more than one entry for this index"),
        emitter);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_abi_entry_conversion_shape(
      module, invoke_op, &entry, field_kind, adapter_op, emitter));
  if (!loom_type_equal(actual_semantic_type, entry.semantic_type)) {
    return loom_low_emit_abi_entry_type_mismatch(
        module, invoke_op, adapter_ref, adapter_op, &entry, field_kind,
        field_index, actual_semantic_type, IREE_SV("adapter semantic"),
        entry.semantic_type, emitter);
  }

  loom_type_t expected_abi_type =
      field_kind == LOOM_LOW_ABI_FIELD_OPERAND
          ? loom_module_value_type(module, signature->argument_ids[field_index])
          : loom_module_value_type(module, signature->result_ids[field_index]);
  if (loom_type_equal(entry.abi_type, expected_abi_type)) {
    return iree_ok_status();
  }
  iree_string_view_t expected_field_kind =
      field_kind == LOOM_LOW_ABI_FIELD_OPERAND ? IREE_SV("callee argument")
                                               : IREE_SV("callee result");
  return loom_low_emit_abi_entry_type_mismatch(
      module, invoke_op, adapter_ref, adapter_op, &entry, field_kind,
      field_index, entry.abi_type, expected_field_kind, expected_abi_type,
      emitter);
}

static iree_status_t loom_low_verify_invoke_adapter_mapped_types(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature, const loom_op_t* adapter_op,
    iree_diagnostic_emitter_t emitter) {
  uint8_t conversion = loom_low_abi_adapter_conversion(adapter_op);
  if (conversion != LOOM_LOW_ABI_ADAPTER_CONVERSION_MAPPED) {
    return iree_ok_status();
  }
  uint16_t compare_count = invoke_op->operand_count;
  if (compare_count > signature->argument_count) {
    compare_count = signature->argument_count;
  }
  const loom_value_id_t* invoke_operands = loom_op_const_operands(invoke_op);
  for (uint16_t i = 0; i < compare_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_invoke_mapped_entry(
        module, invoke_op, signature, adapter_op, LOOM_LOW_ABI_FIELD_OPERAND, i,
        loom_module_value_type(module, invoke_operands[i]), emitter));
  }
  compare_count = invoke_op->result_count;
  if (compare_count > signature->result_count) {
    compare_count = signature->result_count;
  }
  const loom_value_id_t* invoke_results = loom_op_const_results(invoke_op);
  for (uint16_t i = 0; i < compare_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_invoke_mapped_entry(
        module, invoke_op, signature, adapter_op, LOOM_LOW_ABI_FIELD_RESULT, i,
        loom_module_value_type(module, invoke_results[i]), emitter));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_invoke_caller_context(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature, bool adapter_is_present,
    const loom_op_t* adapter_op, iree_diagnostic_emitter_t emitter) {
  loom_low_invoke_caller_t caller =
      loom_low_find_invoke_caller(module, invoke_op);
  if (caller.kind == LOOM_LOW_INVOKE_CALLER_NONE) {
    return loom_low_emit_invoke_context_error(
        module, invoke_op, &caller, signature->definition_op, adapter_op,
        IREE_SV("low.invoke must be nested under a function-like body"),
        emitter);
  }

  if (caller.kind == LOOM_LOW_INVOKE_CALLER_SEMANTIC) {
    if (!adapter_is_present) {
      return loom_low_emit_invoke_context_error(
          module, invoke_op, &caller, signature->definition_op, adapter_op,
          IREE_SV("semantic function bodies require an explicit ABI adapter"),
          emitter);
    }
    return iree_ok_status();
  }

  if (caller.kind == LOOM_LOW_INVOKE_CALLER_LOW) {
    if (adapter_op && loom_low_abi_adapter_conversion(adapter_op) ==
                          LOOM_LOW_ABI_ADAPTER_CONVERSION_MAPPED) {
      return loom_low_emit_invoke_context_error(
          module, invoke_op, &caller, signature->definition_op, adapter_op,
          IREE_SV("mapped ABI adapters cross the semantic-to-low boundary and "
                  "are not valid in low function bodies"),
          emitter);
    }
    loom_symbol_ref_t caller_target = loom_low_function_target(caller.op);
    loom_symbol_ref_t callee_target =
        loom_low_function_target(signature->definition_op);
    if (!loom_low_symbol_ref_equal(caller_target, callee_target)) {
      return loom_low_emit_invoke_context_error(
          module, invoke_op, &caller, signature->definition_op, adapter_op,
          IREE_SV("callee target must match enclosing low function target"),
          emitter);
    }
    return iree_ok_status();
  }

  return iree_ok_status();
}

static iree_status_t loom_low_verify_invoke_purity(
    const loom_module_t* module, const loom_op_t* invoke_op,
    const loom_low_callee_signature_t* signature, const loom_op_t* adapter_op,
    iree_diagnostic_emitter_t emitter) {
  if (loom_low_invoke_purity(invoke_op) == 0) return iree_ok_status();

  if (adapter_op) {
    loom_symbol_ref_t adapter_ref = loom_low_invoke_adapter(invoke_op);
    loom_low_abi_metadata_counts_t counts =
        loom_low_count_abi_metadata(module, adapter_ref);
    if (counts.effect_count == 0 && counts.clobber_count == 0) {
      return iree_ok_status();
    }
    return loom_low_emit_invoke_purity_effect_error(
        module, invoke_op, loom_low_symbol_name(module, adapter_ref),
        IREE_SV("ABI adapter declares observable effects or clobbers"),
        adapter_op, IREE_SV("adapter defined here"), counts, emitter);
  }

  if (loom_low_func_like_is_pure(module, signature->definition_op)) {
    return iree_ok_status();
  }

  loom_low_abi_metadata_counts_t counts = {0};
  return loom_low_emit_invoke_purity_effect_error(
      module, invoke_op, IREE_SV("<direct>"),
      IREE_SV("direct callee has no pure contract"), signature->definition_op,
      IREE_SV("callee defined here"), counts, emitter);
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

iree_status_t loom_low_br_verify(const loom_module_t* module,
                                 const loom_op_t* op,
                                 iree_diagnostic_emitter_t emitter) {
  loom_value_slice_t args = loom_low_br_args(op);
  return loom_ops_verify_successor_args(module, emitter, op, IREE_SV("low.br"),
                                        0, loom_low_br_dest(op), args.values,
                                        args.count);
}

iree_status_t loom_low_cond_br_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_ops_verify_successor_args(
      module, emitter, op, IREE_SV("low.cond_br"), 0,
      loom_low_cond_br_true_dest(op), NULL, 0));
  return loom_ops_verify_successor_args(
      module, emitter, op, IREE_SV("low.cond_br"), 1,
      loom_low_cond_br_false_dest(op), NULL, 0);
}

iree_status_t loom_low_func_def_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(
      loom_low_verify_function_exactness_modes(module, op, emitter));
  return loom_low_verify_no_code_import_on_def(module, op, emitter);
}

iree_status_t loom_low_func_decl_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(
      loom_low_verify_function_exactness_modes(module, op, emitter));
  return loom_low_verify_decl_code_import(module, op, emitter);
}

iree_status_t loom_low_slot_verify(const loom_module_t* module,
                                   const loom_op_t* op,
                                   iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_slot_function(
      module, op, loom_low_slot_function(op), loom_low_slot_function_ATTR_INDEX,
      emitter));

  uint8_t space = loom_low_slot_space(op);
  if (space < LOOM_LOW_SLOT_SPACE_STACK ||
      space >= LOOM_LOW_SLOT_SPACE_COUNT_) {
    IREE_RETURN_IF_ERROR(loom_low_emit_structural_storage_attr_error(
        module, op, loom_low_slot_space_ATTR_INDEX, IREE_SV("space"),
        IREE_SV("space must name a supported low slot space"), emitter));
  }

  if (loom_low_slot_size(op) <= 0) {
    IREE_RETURN_IF_ERROR(loom_low_emit_structural_storage_attr_error(
        module, op, loom_low_slot_size_ATTR_INDEX, IREE_SV("size"),
        IREE_SV("size must be positive"), emitter));
  }

  if (!loom_low_is_power_of_two_i64(loom_low_slot_align(op))) {
    IREE_RETURN_IF_ERROR(loom_low_emit_structural_storage_attr_error(
        module, op, loom_low_slot_align_ATTR_INDEX, IREE_SV("align"),
        IREE_SV("alignment must be a positive power of two"), emitter));
  }

  return iree_ok_status();
}

iree_status_t loom_low_spill_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_slot_use(
      module, op, loom_low_spill_slot(op), loom_low_spill_slot_ATTR_INDEX,
      loom_low_spill_offset(op), loom_low_spill_offset_ATTR_INDEX, emitter);
}

iree_status_t loom_low_reload_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_slot_use(
      module, op, loom_low_reload_slot(op), loom_low_reload_slot_ATTR_INDEX,
      loom_low_reload_offset(op), loom_low_reload_offset_ATTR_INDEX, emitter);
}

iree_status_t loom_low_frame_index_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_slot_use(
      module, op, loom_low_frame_index_slot(op),
      loom_low_frame_index_slot_ATTR_INDEX, loom_low_frame_index_offset(op),
      loom_low_frame_index_offset_ATTR_INDEX, emitter);
}

iree_status_t loom_low_abi_adapter_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  const loom_symbol_t* symbol =
      loom_low_lookup_defined_symbol(module, loom_low_abi_adapter_callee(op));
  if (!symbol) {
    return iree_ok_status();
  }

  loom_low_callee_signature_t signature = {0};
  if (!loom_low_load_callee_signature(module, symbol, &signature)) {
    return loom_low_emit_symbol_kind_mismatch(
        module, op, loom_low_abi_adapter_callee(op),
        loom_low_abi_adapter_callee_ATTR_INDEX, symbol, IREE_SV("low function"),
        emitter);
  }

  IREE_RETURN_IF_ERROR(loom_low_verify_adapter_signature_count(
      module, op, &signature, emitter, IREE_SV("adapter operand"),
      loom_low_abi_adapter_operand_count(op), signature.argument_count));
  IREE_RETURN_IF_ERROR(loom_low_verify_adapter_signature_count(
      module, op, &signature, emitter, IREE_SV("adapter result"),
      loom_low_abi_adapter_result_count(op), signature.result_count));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_adapter_mapped_entries(module, op, &signature, emitter));
  return iree_ok_status();
}

iree_status_t loom_low_abi_operand_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_abi_entry_record(module, op,
                                          LOOM_LOW_ABI_FIELD_OPERAND, emitter);
}

iree_status_t loom_low_abi_result_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_abi_entry_record(module, op, LOOM_LOW_ABI_FIELD_RESULT,
                                          emitter);
}

iree_status_t loom_low_abi_effect_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  const loom_op_t* adapter_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_verify_abi_metadata_adapter(
      module, op, loom_low_abi_effect_adapter(op),
      loom_low_abi_effect_adapter_ATTR_INDEX, emitter, &adapter_op));
  return loom_low_verify_abi_effect_resource(
      module, op, loom_low_abi_effect_adapter(op), adapter_op, emitter);
}

iree_status_t loom_low_abi_clobber_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  const loom_op_t* adapter_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_verify_abi_metadata_adapter(
      module, op, loom_low_abi_clobber_adapter(op),
      loom_low_abi_clobber_adapter_ATTR_INDEX, emitter, &adapter_op));
  return loom_low_verify_qualified_key_attr(
      module, op, emitter, loom_low_abi_clobber_resource(op),
      loom_low_abi_clobber_resource_ATTR_INDEX, IREE_SV("resource"),
      IREE_SV("a namespace-qualified ABI clobber key with non-empty "
              "identifier segments"));
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

  bool adapter_is_present =
      loom_low_optional_attr_is_present(op, loom_low_invoke_adapter_ATTR_INDEX);
  const loom_op_t* adapter_op = NULL;
  if (adapter_is_present) {
    const loom_symbol_t* adapter_symbol =
        loom_low_lookup_defined_symbol(module, loom_low_invoke_adapter(op));
    if (!adapter_symbol) {
      return iree_ok_status();
    }
    adapter_op = adapter_symbol->defining_op;
    if (!loom_low_abi_adapter_isa(adapter_op)) {
      IREE_RETURN_IF_ERROR(loom_low_emit_symbol_kind_mismatch(
          module, op, loom_low_invoke_adapter(op),
          loom_low_invoke_adapter_ATTR_INDEX, adapter_symbol,
          IREE_SV("low ABI adapter"), emitter));
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_low_verify_invoke_caller_context(
      module, op, &signature, adapter_is_present, adapter_op, emitter));

  if (adapter_op) {
    if (!loom_low_symbol_ref_equal(loom_low_invoke_callee(op),
                                   loom_low_abi_adapter_callee(adapter_op))) {
      IREE_RETURN_IF_ERROR(loom_low_emit_adapter_callee_mismatch(
          module, op, &signature, adapter_op, emitter));
    }
    IREE_RETURN_IF_ERROR(loom_low_verify_invoke_adapter_count(
        module, op, &signature, adapter_op, emitter, IREE_SV("invoke operand"),
        op->operand_count, loom_low_abi_adapter_operand_count(adapter_op)));
    IREE_RETURN_IF_ERROR(loom_low_verify_invoke_adapter_count(
        module, op, &signature, adapter_op, emitter, IREE_SV("invoke result"),
        op->result_count, loom_low_abi_adapter_result_count(adapter_op)));
    IREE_RETURN_IF_ERROR(loom_low_verify_invoke_adapter_argument_types(
        module, op, &signature, adapter_op, emitter));
    IREE_RETURN_IF_ERROR(loom_low_verify_invoke_adapter_result_types(
        module, op, &signature, adapter_op, emitter));
    IREE_RETURN_IF_ERROR(loom_low_verify_invoke_adapter_mapped_types(
        module, op, &signature, adapter_op, emitter));
    IREE_RETURN_IF_ERROR(loom_low_verify_invoke_purity(module, op, &signature,
                                                       adapter_op, emitter));
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_low_verify_invoke_argument_count(module, op, &signature, emitter));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_invoke_result_count(module, op, &signature, emitter));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_invoke_argument_types(module, op, &signature, emitter));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_invoke_result_types(module, op, &signature, emitter));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_invoke_purity(module, op, &signature, NULL, emitter));
  return iree_ok_status();
}

loom_trait_flags_t loom_low_invoke_effective_traits(const loom_op_t* op) {
  if (loom_low_invoke_purity(op) != 0) return LOOM_TRAIT_PURE;
  return LOOM_TRAIT_UNKNOWN_EFFECTS;
}
