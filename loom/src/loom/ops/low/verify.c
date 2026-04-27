// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/function_contract_verify.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/successor_verify.h"
#include "loom/util/stable_id.h"

typedef struct loom_low_callee_signature_t {
  // Defining function-like op for related diagnostic locations.
  const loom_op_t* definition_op;
  // Callee argument value IDs in signature order.
  const loom_value_id_t* argument_ids;
  // Number of callee arguments.
  uint16_t argument_count;
  // Callee result value IDs in signature order.
  const loom_value_id_t* result_ids;
  // Number of callee results.
  uint16_t result_count;
} loom_low_callee_signature_t;

typedef struct loom_low_effect_counts_t {
  // Number of explicit effect records attached to the boundary.
  uint32_t effect_count;
  // Number of explicit clobber records attached to the boundary.
  uint32_t clobber_count;
} loom_low_effect_counts_t;

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
  if (iree_string_view_is_empty(key)) {
    return false;
  }
  bool saw_separator = false;
  bool expect_segment_start = true;
  for (iree_host_size_t i = 0; i < key.size; ++i) {
    char c = key.data[i];
    if (c == '.') {
      if (expect_segment_start) {
        return false;
      }
      saw_separator = true;
      expect_segment_start = true;
      continue;
    }
    if (expect_segment_start) {
      if (!loom_low_qualified_key_segment_start(c)) {
        return false;
      }
      expect_segment_start = false;
      continue;
    }
    if (!loom_low_qualified_key_segment_continue(c)) {
      return false;
    }
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
  if (!op) {
    return IREE_SV("<null>");
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable) {
    return IREE_SV("<unknown>");
  }
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
  if (!symbol || !symbol->definition) {
    return IREE_SV("unresolved");
  }
  return loom_symbol_definition_descriptor_name(symbol->definition);
}

static const loom_symbol_t* loom_low_lookup_defined_symbol(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (!symbol->definition || !symbol->defining_op) {
    return NULL;
  }
  return symbol;
}

static bool loom_low_function_isa(const loom_op_t* op) {
  return loom_low_func_def_isa(op) || loom_low_func_decl_isa(op);
}

static bool loom_low_executable_def_isa(const loom_op_t* op) {
  return loom_low_func_def_isa(op) || loom_low_kernel_def_isa(op);
}

static loom_symbol_ref_t loom_low_function_symbol(const loom_op_t* op) {
  if (loom_low_func_def_isa(op)) {
    return loom_low_func_def_callee(op);
  }
  if (loom_low_kernel_def_isa(op)) {
    return loom_low_kernel_def_callee(op);
  }
  if (loom_low_func_decl_isa(op)) {
    return loom_low_func_decl_callee(op);
  }
  return loom_symbol_ref_null();
}

static loom_symbol_ref_t loom_low_function_target(const loom_op_t* op) {
  if (loom_low_func_def_isa(op)) {
    return loom_low_func_def_target(op);
  }
  if (loom_low_kernel_def_isa(op)) {
    return loom_low_kernel_def_target(op);
  }
  if (loom_low_func_decl_isa(op)) {
    return loom_low_func_decl_target(op);
  }
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

static bool loom_low_function_explicit_abi(const loom_op_t* op,
                                           uint8_t* out_abi,
                                           uint16_t* out_abi_attr_index) {
  if (loom_low_func_def_isa(op) &&
      loom_low_optional_attr_is_present(op, loom_low_func_def_abi_ATTR_INDEX)) {
    *out_abi = loom_low_func_def_abi(op);
    *out_abi_attr_index = loom_low_func_def_abi_ATTR_INDEX;
    return true;
  }
  if (loom_low_func_decl_isa(op) &&
      loom_low_optional_attr_is_present(op,
                                        loom_low_func_decl_abi_ATTR_INDEX)) {
    *out_abi = loom_low_func_decl_abi(op);
    *out_abi_attr_index = loom_low_func_decl_abi_ATTR_INDEX;
    return true;
  }
  return false;
}

static loom_type_t loom_low_type_attr(const loom_module_t* module,
                                      loom_type_id_t type_id) {
  if (type_id == LOOM_TYPE_ID_INVALID || type_id >= module->types.count) {
    return loom_type_none();
  }
  return module->types.entries[type_id];
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
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 16),
                       params, IREE_ARRAYSIZE(params));
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

static iree_status_t loom_low_verify_contract_attr_present(
    const loom_module_t* module, const loom_op_t* op, uint16_t attr_index,
    iree_string_view_t contract_name, iree_string_view_t reason,
    iree_diagnostic_emitter_t emitter) {
  if (loom_low_optional_attr_is_present(op, attr_index)) {
    return iree_ok_status();
  }
  return loom_low_emit_function_contract_error(module, op, attr_index,
                                               contract_name, reason, emitter);
}

static iree_status_t loom_low_verify_positive_u32_attr(
    const loom_module_t* module, const loom_op_t* op, uint16_t attr_index,
    int64_t value, iree_string_view_t contract_name,
    iree_diagnostic_emitter_t emitter) {
  if (!loom_low_optional_attr_is_present(op, attr_index)) {
    return iree_ok_status();
  }
  if (value > 0 && value <= UINT32_MAX) {
    return iree_ok_status();
  }
  return loom_low_emit_function_contract_error(
      module, op, attr_index, contract_name, IREE_SV("positive u32"), emitter);
}

static iree_status_t loom_low_verify_kernel_contract(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const bool has_export_symbol = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_export_symbol_ATTR_INDEX);
  const bool has_artifact = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_artifact_ATTR_INDEX);
  const bool has_export_ordinal = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_export_ordinal_ATTR_INDEX);
  const bool has_export_linkage = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_export_linkage_ATTR_INDEX);
  if (!has_export_symbol &&
      (has_artifact || has_export_ordinal || has_export_linkage)) {
    return loom_low_verify_contract_attr_present(
        module, op, loom_low_kernel_def_export_symbol_ATTR_INDEX,
        IREE_SV("export"),
        IREE_SV("present when artifact, ordinal, or linkage is present"),
        emitter);
  }
  if (!has_artifact && (has_export_ordinal || has_export_linkage)) {
    return loom_low_verify_contract_attr_present(
        module, op, loom_low_kernel_def_artifact_ATTR_INDEX,
        IREE_SV("artifact"),
        IREE_SV("present when ordinal or linkage is present"), emitter);
  }
  if (has_artifact) {
    IREE_RETURN_IF_ERROR(loom_low_verify_contract_attr_present(
        module, op, loom_low_kernel_def_target_ATTR_INDEX, IREE_SV("target"),
        IREE_SV("present when artifact is present"), emitter));
  }

  const bool has_workgroup_size_x = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_workgroup_size_x_ATTR_INDEX);
  const bool has_workgroup_size_y = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_workgroup_size_y_ATTR_INDEX);
  const bool has_workgroup_size_z = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_workgroup_size_z_ATTR_INDEX);
  if (has_workgroup_size_x || has_workgroup_size_y || has_workgroup_size_z) {
    IREE_RETURN_IF_ERROR(loom_low_verify_contract_attr_present(
        module, op, loom_low_kernel_def_workgroup_size_x_ATTR_INDEX,
        IREE_SV("workgroup_size_x"),
        IREE_SV("present with workgroup_size_y and workgroup_size_z"),
        emitter));
    IREE_RETURN_IF_ERROR(loom_low_verify_contract_attr_present(
        module, op, loom_low_kernel_def_workgroup_size_y_ATTR_INDEX,
        IREE_SV("workgroup_size_y"),
        IREE_SV("present with workgroup_size_x and workgroup_size_z"),
        emitter));
    IREE_RETURN_IF_ERROR(loom_low_verify_contract_attr_present(
        module, op, loom_low_kernel_def_workgroup_size_z_ATTR_INDEX,
        IREE_SV("workgroup_size_z"),
        IREE_SV("present with workgroup_size_x and workgroup_size_y"),
        emitter));
    IREE_RETURN_IF_ERROR(loom_low_verify_positive_u32_attr(
        module, op, loom_low_kernel_def_workgroup_size_x_ATTR_INDEX,
        loom_low_kernel_def_workgroup_size_x(op), IREE_SV("workgroup_size_x"),
        emitter));
    IREE_RETURN_IF_ERROR(loom_low_verify_positive_u32_attr(
        module, op, loom_low_kernel_def_workgroup_size_y_ATTR_INDEX,
        loom_low_kernel_def_workgroup_size_y(op), IREE_SV("workgroup_size_y"),
        emitter));
    IREE_RETURN_IF_ERROR(loom_low_verify_positive_u32_attr(
        module, op, loom_low_kernel_def_workgroup_size_z_ATTR_INDEX,
        loom_low_kernel_def_workgroup_size_z(op), IREE_SV("workgroup_size_z"),
        emitter));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_exactness_modes(
    const loom_module_t* module, const loom_op_t* op,
    uint16_t allocation_attr_index, uint8_t allocation,
    uint16_t schedule_attr_index, uint8_t schedule,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_enum_is_named(
      module, op, allocation_attr_index, allocation, IREE_SV("allocation"),
      IREE_SV("explicit allocation mode must name virtual, assigned, or fixed"),
      emitter));
  return loom_low_verify_optional_enum_is_named(
      module, op, schedule_attr_index, schedule, IREE_SV("schedule"),
      IREE_SV("explicit schedule mode must name free, constrained, or locked"),
      emitter);
}

static iree_status_t loom_low_verify_function_exactness_modes(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  if (loom_low_func_def_isa(op)) {
    return loom_low_verify_exactness_modes(
        module, op, loom_low_func_def_allocation_ATTR_INDEX,
        loom_low_func_def_allocation(op), loom_low_func_def_schedule_ATTR_INDEX,
        loom_low_func_def_schedule(op), emitter);
  }
  return loom_low_verify_exactness_modes(
      module, op, loom_low_func_decl_allocation_ATTR_INDEX,
      loom_low_func_decl_allocation(op), loom_low_func_decl_schedule_ATTR_INDEX,
      loom_low_func_decl_schedule(op), emitter);
}

static iree_status_t loom_low_verify_kernel_exactness_modes(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_exactness_modes(
      module, op, loom_low_kernel_def_allocation_ATTR_INDEX,
      loom_low_kernel_def_allocation(op),
      loom_low_kernel_def_schedule_ATTR_INDEX, loom_low_kernel_def_schedule(op),
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

static bool loom_low_load_func_like_signature(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_low_callee_signature_t* out_signature) {
  loom_func_like_t func =
      loom_func_like_cast(module, symbol ? symbol->defining_op : NULL);
  if (!loom_func_like_isa(func)) {
    return false;
  }
  out_signature->definition_op = symbol->defining_op;
  out_signature->argument_ids =
      loom_func_like_arg_ids(func, &out_signature->argument_count);
  out_signature->result_ids = loom_op_const_results(func.op);
  out_signature->result_count = func.op->result_count;
  return true;
}

static bool loom_low_load_low_signature(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_low_callee_signature_t* out_signature) {
  if (!symbol || !loom_low_function_isa(symbol->defining_op)) {
    return false;
  }
  return loom_low_load_func_like_signature(module, symbol, out_signature);
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
  if (loom_low_qualified_key_is_valid(key)) {
    return iree_ok_status();
  }
  return loom_low_emit_descriptor_key_error(emitter, op, attr_index, field_name,
                                            key, expected);
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
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 17),
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

static iree_status_t loom_low_verify_descriptor_id(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_string_id_t opcode_id,
    int64_t descriptor_id, uint16_t descriptor_id_attr_index) {
  iree_string_view_t key = loom_low_string_or_empty(module, opcode_id);
  uint64_t expected_id = loom_stable_id_from_string(key);
  if (descriptor_id == (int64_t)expected_id) {
    return iree_ok_status();
  }
  return loom_low_emit_structural_storage_error(
      module, op,
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                descriptor_id_attr_index),
      IREE_SV("descriptor_id"),
      IREE_SV("descriptor ID must match the stable ID derived from opcode"),
      NULL, 0, emitter);
}

static iree_status_t loom_low_verify_same_register_unit_count(
    const loom_module_t* module, const loom_op_t* op, loom_value_id_t source_id,
    loom_value_id_t result_id, iree_diagnostic_emitter_t emitter) {
  const loom_type_t source_type = loom_module_value_type(module, source_id);
  const loom_type_t result_type = loom_module_value_type(module, result_id);
  if (!loom_type_is_register(source_type) ||
      !loom_type_is_register(result_type)) {
    return iree_ok_status();
  }
  if (loom_type_register_unit_count(source_type) ==
      loom_type_register_unit_count(result_type)) {
    return iree_ok_status();
  }
  return loom_low_emit_structural_storage_error(
      module, op, loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0),
      IREE_SV("result"),
      IREE_SV("result register unit count must match the source"), NULL, 0,
      emitter);
}

static iree_status_t loom_low_verify_slice_register_range(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const int64_t offset = loom_low_slice_offset(op);
  if (offset < 0) {
    return loom_low_emit_structural_storage_attr_error(
        module, op, loom_low_slice_offset_ATTR_INDEX, IREE_SV("offset"),
        IREE_SV("offset must be non-negative"), emitter);
  }

  const loom_type_t source_type =
      loom_module_value_type(module, loom_low_slice_source(op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_low_slice_result(op));
  if (!loom_type_is_register(source_type) ||
      !loom_type_is_register(result_type)) {
    return iree_ok_status();
  }

  const uint64_t source_unit_count = loom_type_register_unit_count(source_type);
  const uint64_t result_unit_count = loom_type_register_unit_count(result_type);
  const uint64_t offset_unit_count = (uint64_t)offset;
  if (offset_unit_count > source_unit_count ||
      result_unit_count > source_unit_count - offset_unit_count) {
    return loom_low_emit_structural_storage_attr_error(
        module, op, loom_low_slice_offset_ATTR_INDEX, IREE_SV("offset"),
        IREE_SV("offset plus result register units must fit within the source "
                "register range"),
        emitter);
  }
  return iree_ok_status();
}

static const loom_op_t* loom_low_find_enclosing_low_executable_def(
    const loom_module_t* module, const loom_op_t* nested_op) {
  const loom_op_t* parent = nested_op->parent_op;
  while (parent) {
    if (loom_low_executable_def_isa(parent)) {
      return parent;
    }
    loom_func_like_t func = loom_func_like_cast(module, (loom_op_t*)parent);
    if (loom_func_like_isa(func) && loom_func_like_body(func)) {
      return NULL;
    }
    parent = parent->parent_op;
  }
  return NULL;
}

static iree_status_t loom_low_verify_slot_function(
    const loom_module_t* module, const loom_op_t* op,
    loom_symbol_ref_t function_ref, uint16_t attr_index,
    iree_diagnostic_emitter_t emitter) {
  const loom_symbol_t* symbol =
      loom_low_lookup_defined_symbol(module, function_ref);
  if (!symbol) {
    return iree_ok_status();
  }
  if (loom_low_executable_def_isa(symbol->defining_op)) {
    return iree_ok_status();
  }
  return loom_low_emit_symbol_kind_mismatch(
      module, op, function_ref, attr_index, symbol,
      IREE_SV("low function or low kernel definition"), emitter);
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
  if (!slot_symbol) {
    return iree_ok_status();
  }
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
  if (!slot_op) {
    return iree_ok_status();
  }

  const loom_op_t* enclosing_func =
      loom_low_find_enclosing_low_executable_def(module, op);
  if (!enclosing_func) {
    return loom_low_emit_structural_storage_error(
        module, op, loom_diagnostic_field_ref_none(),
        IREE_SV("enclosing function"),
        IREE_SV("slot traffic must be nested under a low function or low "
                "kernel body"),
        NULL, 0, emitter);
  }

  if (!loom_low_symbol_ref_equal(loom_low_slot_function(slot_op),
                                 loom_low_function_symbol(enclosing_func))) {
    loom_diagnostic_related_op_t related[] = {{
        .label = IREE_SV("slot defined here"),
        .op = slot_op,
    }};
    return loom_low_emit_structural_storage_error(
        module, op,
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                  slot_attr_index),
        IREE_SV("slot"),
        IREE_SV("slot owner must match the enclosing low entry"), related,
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

static bool loom_low_resource_kind_is_known(uint8_t kind) {
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

static iree_string_view_t loom_low_resource_export_abi_reason(uint8_t kind) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER:
      return IREE_SV(
          "native_pointer resources require object_function export "
          "ABI");
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_STATE:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_IMPORT:
      return IREE_SV("VM resources require vm_module_function export ABI");
    case LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BUFFER_RESOURCE:
      return IREE_SV("HAL buffer resources require hal_kernel export ABI");
    default:
      return IREE_SV(
          "resource import_kind must name a supported target resource");
  }
}

static bool loom_low_resource_matches_export_abi(uint8_t kind, uint8_t abi) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER:
      return abi == LOOM_LOW_ABI_OBJECT_FUNCTION;
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_STATE:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_IMPORT:
      return abi == LOOM_LOW_ABI_VM_MODULE_FUNCTION;
    case LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BUFFER_RESOURCE:
      return abi == LOOM_LOW_ABI_HAL_KERNEL;
    default:
      return false;
  }
}

static iree_status_t loom_low_verify_resource_function_abi(
    const loom_module_t* module, const loom_op_t* resource_op,
    const loom_op_t* function_op, uint16_t function_abi_attr_index, uint8_t abi,
    uint8_t kind, iree_diagnostic_emitter_t emitter) {
  if (loom_low_resource_matches_export_abi(kind, abi)) {
    return iree_ok_status();
  }
  loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("function ABI defined here"),
      .op = function_op,
      .field_ref = loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                             function_abi_attr_index),
  }};
  return loom_low_emit_structural_storage_error(
      module, resource_op,
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                loom_low_resource_import_kind_ATTR_INDEX),
      IREE_SV("import_kind"), loom_low_resource_export_abi_reason(kind),
      related, IREE_ARRAYSIZE(related), emitter);
}

static iree_status_t loom_low_verify_resource_op(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const uint8_t import_kind = loom_low_resource_import_kind(op);
  if (!loom_low_resource_kind_is_known(import_kind)) {
    return loom_low_emit_structural_storage_attr_error(
        module, op, loom_low_resource_import_kind_ATTR_INDEX,
        IREE_SV("import_kind"),
        IREE_SV("import_kind must name a supported target resource"), emitter);
  }

  if (loom_low_resource_index(op) < 0) {
    return loom_low_emit_structural_storage_attr_error(
        module, op, loom_low_resource_index_ATTR_INDEX, IREE_SV("index"),
        IREE_SV("index must be non-negative"), emitter);
  }

  loom_attribute_t valid_byte_count =
      loom_op_attrs(op)[loom_low_resource_valid_byte_count_ATTR_INDEX];
  if (!loom_attr_is_absent(valid_byte_count) &&
      loom_low_resource_valid_byte_count(op) < 0) {
    return loom_low_emit_structural_storage_attr_error(
        module, op, loom_low_resource_valid_byte_count_ATTR_INDEX,
        IREE_SV("valid_byte_count"),
        IREE_SV("valid_byte_count must be non-negative"), emitter);
  }

  loom_attribute_t cache_swizzle_stride =
      loom_op_attrs(op)[loom_low_resource_cache_swizzle_stride_ATTR_INDEX];
  if (!loom_attr_is_absent(cache_swizzle_stride)) {
    const int64_t stride = loom_low_resource_cache_swizzle_stride(op);
    if (stride < 0 || stride > 0x3FFF) {
      return loom_low_emit_structural_storage_attr_error(
          module, op, loom_low_resource_cache_swizzle_stride_ATTR_INDEX,
          IREE_SV("cache_swizzle_stride"),
          IREE_SV("cache_swizzle_stride must fit a 14-bit byte stride"),
          emitter);
    }
  }

  const loom_type_t semantic_type =
      loom_low_type_attr(module, loom_low_resource_semantic_type(op));
  if (loom_type_kind(semantic_type) == LOOM_TYPE_NONE) {
    return loom_low_emit_structural_storage_attr_error(
        module, op, loom_low_resource_semantic_type_ATTR_INDEX,
        IREE_SV("semantic_type"),
        IREE_SV("semantic_type must name a valid Loom type"), emitter);
  }

  const loom_type_t result_type =
      loom_module_value_type(module, loom_low_resource_result(op));
  if (!loom_type_is_register(result_type)) {
    return loom_low_emit_structural_storage_error(
        module, op, loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0),
        IREE_SV("result"), IREE_SV("resource result must be a register type"),
        NULL, 0, emitter);
  }

  const loom_op_t* enclosing_func =
      loom_low_find_enclosing_low_executable_def(module, op);
  if (!enclosing_func) {
    return loom_low_emit_structural_storage_error(
        module, op, loom_diagnostic_field_ref_none(),
        IREE_SV("enclosing function"),
        IREE_SV("resource imports must be nested under a low function or low "
                "kernel body"),
        NULL, 0, emitter);
  }

  uint8_t function_abi = 0;
  uint16_t function_abi_attr_index = 0;
  if (loom_low_function_explicit_abi(enclosing_func, &function_abi,
                                     &function_abi_attr_index)) {
    return loom_low_verify_resource_function_abi(
        module, op, enclosing_func, function_abi_attr_index, function_abi,
        import_kind, emitter);
  }

  return iree_ok_status();
}

static iree_status_t loom_low_verify_function_preamble(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_diagnostic_emitter_t emitter) {
  loom_region_t* body = NULL;
  if (loom_low_func_def_isa(function_op)) {
    body = loom_low_func_def_body(function_op);
  } else if (loom_low_kernel_def_isa(function_op)) {
    body = loom_low_kernel_def_body(function_op);
  }
  if (body == NULL || body->block_count == 0) {
    return iree_ok_status();
  }

  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  bool preamble_open = true;
  const loom_op_t* nested_op = NULL;
  loom_block_for_each_op(entry_block, nested_op) {
    if (loom_low_live_in_isa(nested_op) || loom_low_resource_isa(nested_op)) {
      if (!preamble_open) {
        return loom_low_emit_structural_storage_error(
            module, nested_op, loom_diagnostic_field_ref_none(),
            IREE_SV("position"),
            IREE_SV("live-ins and resources must form an entry-block prefix "
                    "before ordinary low packets"),
            NULL, 0, emitter);
      }
      continue;
    }
    preamble_open = false;
  }

  for (uint16_t block_index = 1; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    loom_block_for_each_op(block, nested_op) {
      if (!loom_low_live_in_isa(nested_op) &&
          !loom_low_resource_isa(nested_op)) {
        continue;
      }
      return loom_low_emit_structural_storage_error(
          module, nested_op, loom_diagnostic_field_ref_none(),
          IREE_SV("position"),
          IREE_SV("live-ins and resources must appear in the low entry block"),
          NULL, 0, emitter);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_kernel_returns(
    const loom_module_t* module, const loom_op_t* kernel_op,
    iree_diagnostic_emitter_t emitter) {
  loom_region_t* body = loom_low_kernel_def_body(kernel_op);
  if (body == NULL) {
    return iree_ok_status();
  }
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* nested_op = NULL;
    loom_block_for_each_op(block, nested_op) {
      if (!loom_low_return_isa(nested_op) || nested_op->operand_count == 0) {
        continue;
      }
      return loom_low_emit_structural_storage_error(
          module, nested_op,
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 0),
          IREE_SV("values"),
          IREE_SV("low kernel entries must return no values"), NULL, 0,
          emitter);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_emit_callee_related(
    iree_diagnostic_emitter_t emitter, const loom_op_t* call_op,
    const loom_op_t* definition_op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("defined here"),
      .op = definition_op,
  }};
  return loom_low_emit_related(emitter, call_op, error, params, param_count,
                               related,
                               definition_op ? IREE_ARRAYSIZE(related) : 0);
}

static iree_status_t loom_low_emit_call_callee_kind_mismatch(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_ref_t callee, uint16_t callee_attr_index,
    const loom_symbol_t* symbol, iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(loom_low_symbol_name(module, callee)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    callee_attr_index)),
      loom_param_string(loom_low_symbol_definition_name(symbol)),
      loom_param_string(IREE_SV("low function")),
  };
  return loom_low_emit_callee_related(
      emitter, call_op, symbol->defining_op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_call_count_mismatch(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_ref_t callee, const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter, iree_string_view_t field_kind,
    uint16_t actual_count, uint16_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_symbol_name(module, callee)),
      loom_param_string(field_kind),
      loom_param_u32(actual_count),
      loom_param_u32(expected_count),
  };
  return loom_low_emit_callee_related(
      emitter, call_op, signature->definition_op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 13), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_call_type_mismatch(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_ref_t callee, const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter,
    loom_diagnostic_field_kind_t field_ref_kind, iree_string_view_t field_kind,
    uint16_t field_index, loom_type_t actual_type,
    iree_string_view_t callee_field_kind, loom_type_t expected_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_symbol_name(module, callee)),
      loom_param_with_field_ref(
          loom_param_string(field_kind),
          loom_diagnostic_field_ref(field_ref_kind, field_index)),
      loom_param_u32(field_index),
      loom_param_type(actual_type),
      loom_param_string(callee_field_kind),
      loom_param_type(expected_type),
  };
  return loom_low_emit_callee_related(
      emitter, call_op, signature->definition_op,
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 14), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_call_purity_effect_error(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_ref_t callee, iree_string_view_t boundary_name,
    iree_string_view_t reason, const loom_op_t* related_op,
    iree_string_view_t related_label, loom_low_effect_counts_t counts,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_symbol_name(module, callee)),
      loom_param_string(boundary_name),
      loom_param_string(reason),
      loom_param_u32(counts.effect_count),
      loom_param_u32(counts.clobber_count),
  };
  loom_diagnostic_related_op_t related[] = {{
      .label = related_label,
      .op = related_op,
  }};
  return loom_low_emit_related(
      emitter, call_op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 15),
      params, IREE_ARRAYSIZE(params), related,
      related_op ? IREE_ARRAYSIZE(related) : 0);
}

static bool loom_low_func_like_is_pure(const loom_module_t* module,
                                       const loom_op_t* op) {
  loom_func_like_t func = loom_func_like_cast(module, (loom_op_t*)op);
  if (!loom_func_like_isa(func)) {
    return false;
  }
  if (loom_func_like_purity(func) != 0) {
    return true;
  }
  loom_region_t* body = loom_func_like_body(func);
  return body && !loom_region_has_read_effects(body) &&
         !loom_region_has_write_effects(body);
}

static iree_status_t loom_low_verify_call_argument_count(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_ref_t callee, const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  if (call_op->operand_count == signature->argument_count) {
    return iree_ok_status();
  }
  return loom_low_emit_call_count_mismatch(
      module, call_op, callee, signature, emitter, IREE_SV("operand"),
      call_op->operand_count, signature->argument_count);
}

static iree_status_t loom_low_verify_call_result_count(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_ref_t callee, const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  if (call_op->result_count == signature->result_count) {
    return iree_ok_status();
  }
  return loom_low_emit_call_count_mismatch(
      module, call_op, callee, signature, emitter, IREE_SV("result"),
      call_op->result_count, signature->result_count);
}

static iree_status_t loom_low_verify_call_argument_types(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_ref_t callee, const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  uint16_t compare_count = call_op->operand_count;
  if (compare_count > signature->argument_count) {
    compare_count = signature->argument_count;
  }
  const loom_value_id_t* call_operands = loom_op_const_operands(call_op);
  for (uint16_t i = 0; i < compare_count; ++i) {
    loom_type_t actual_type = loom_module_value_type(module, call_operands[i]);
    loom_type_t expected_type =
        loom_module_value_type(module, signature->argument_ids[i]);
    if (loom_type_equal(actual_type, expected_type)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_emit_call_type_mismatch(
        module, call_op, callee, signature, emitter,
        LOOM_DIAGNOSTIC_FIELD_OPERAND, IREE_SV("operand"), i, actual_type,
        IREE_SV("argument"), expected_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_call_result_types(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_ref_t callee, const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  uint16_t compare_count = call_op->result_count;
  if (compare_count > signature->result_count) {
    compare_count = signature->result_count;
  }
  const loom_value_id_t* call_results = loom_op_const_results(call_op);
  for (uint16_t i = 0; i < compare_count; ++i) {
    loom_type_t actual_type = loom_module_value_type(module, call_results[i]);
    loom_type_t expected_type =
        loom_module_value_type(module, signature->result_ids[i]);
    if (loom_type_equal(actual_type, expected_type)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_emit_call_type_mismatch(
        module, call_op, callee, signature, emitter,
        LOOM_DIAGNOSTIC_FIELD_RESULT, IREE_SV("result"), i, actual_type,
        IREE_SV("result"), expected_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_call_purity(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_ref_t callee, uint8_t purity,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  if (purity == 0) {
    return iree_ok_status();
  }

  const loom_op_t* purity_op = signature->definition_op;
  if (loom_low_func_like_is_pure(module, purity_op)) {
    return iree_ok_status();
  }

  loom_low_effect_counts_t counts = {0};
  return loom_low_emit_call_purity_effect_error(
      module, call_op, callee, loom_low_symbol_name(module, callee),
      IREE_SV("callee has no pure contract"), purity_op,
      IREE_SV("contract defined here"), counts, emitter);
}

static iree_status_t loom_low_verify_func_call_context(
    const loom_module_t* module, const loom_op_t* call_op,
    const loom_low_callee_signature_t* callee_signature,
    iree_diagnostic_emitter_t emitter) {
  const loom_op_t* caller_op =
      loom_low_find_enclosing_low_executable_def(module, call_op);
  if (!caller_op) {
    return loom_low_emit_structural_storage_error(
        module, call_op, loom_diagnostic_field_ref_none(),
        IREE_SV("enclosing low entry"),
        IREE_SV("low.func.call must be nested under a low function or low "
                "kernel body"),
        NULL, 0, emitter);
  }

  loom_symbol_ref_t caller_target = loom_low_function_target(caller_op);
  loom_symbol_ref_t callee_target =
      loom_low_function_target(callee_signature->definition_op);
  if (loom_low_symbol_ref_equal(caller_target, callee_target)) {
    return iree_ok_status();
  }
  loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("callee defined here"),
      .op = callee_signature->definition_op,
  }};
  return loom_low_emit_structural_storage_error(
      module, call_op,
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                loom_low_func_call_callee_ATTR_INDEX),
      IREE_SV("callee"),
      IREE_SV("callee target must match enclosing low entry target"), related,
      IREE_ARRAYSIZE(related), emitter);
}

iree_status_t loom_low_op_verify(const loom_module_t* module,
                                 const loom_op_t* op,
                                 iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_key(
      module, op, emitter, loom_low_op_opcode(op),
      loom_low_op_opcode_ATTR_INDEX));
  return loom_low_verify_descriptor_id(
      module, op, emitter, loom_low_op_opcode(op),
      loom_low_op_descriptor_id(op), loom_low_op_descriptor_id_ATTR_INDEX);
}

iree_status_t loom_low_const_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_descriptor_key(
      module, op, emitter, loom_low_const_opcode(op),
      loom_low_const_opcode_ATTR_INDEX));
  return loom_low_verify_descriptor_id(module, op, emitter,
                                       loom_low_const_opcode(op),
                                       loom_low_const_descriptor_id(op),
                                       loom_low_const_descriptor_id_ATTR_INDEX);
}

iree_status_t loom_low_copy_verify(const loom_module_t* module,
                                   const loom_op_t* op,
                                   iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_same_register_unit_count(
      module, op, loom_low_copy_source(op), loom_low_copy_result(op), emitter);
}

iree_status_t loom_low_slice_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_slice_register_range(module, op, emitter);
}

iree_status_t loom_low_live_in_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_qualified_key_attr(
      module, op, emitter, loom_low_live_in_source(op),
      loom_low_live_in_source_ATTR_INDEX, IREE_SV("source"),
      IREE_SV("qualified target live-in key")));

  const loom_type_t result_type =
      loom_module_value_type(module, loom_low_live_in_result(op));
  if (!loom_type_is_register(result_type)) {
    return loom_low_emit_structural_storage_error(
        module, op, loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0),
        IREE_SV("result"), IREE_SV("live-in result must be a register type"),
        NULL, 0, emitter);
  }

  const loom_op_t* enclosing_func =
      loom_low_find_enclosing_low_executable_def(module, op);
  if (!enclosing_func) {
    return loom_low_emit_structural_storage_error(
        module, op, loom_diagnostic_field_ref_none(),
        IREE_SV("enclosing function"),
        IREE_SV("live-ins must be nested under a low function or low kernel "
                "body"),
        NULL, 0, emitter);
  }
  return iree_ok_status();
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
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_function_exactness_modes(module, op, emitter));
  return loom_low_verify_function_preamble(module, op, emitter);
}

iree_status_t loom_low_kernel_def_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_kernel_contract(module, op, emitter));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_kernel_exactness_modes(module, op, emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_kernel_returns(module, op, emitter));
  return loom_low_verify_function_preamble(module, op, emitter);
}

iree_status_t loom_low_func_decl_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));
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

iree_status_t loom_low_resource_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_resource_op(module, op, emitter);
}

iree_status_t loom_low_func_call_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_symbol_ref_t callee = loom_low_func_call_callee(op);
  const loom_symbol_t* symbol = loom_low_lookup_defined_symbol(module, callee);
  if (!symbol) {
    return iree_ok_status();
  }

  loom_low_callee_signature_t low_signature = {0};
  if (!loom_low_load_low_signature(module, symbol, &low_signature)) {
    return loom_low_emit_call_callee_kind_mismatch(
        module, op, callee, loom_low_func_call_callee_ATTR_INDEX, symbol,
        emitter);
  }

  IREE_RETURN_IF_ERROR(
      loom_low_verify_func_call_context(module, op, &low_signature, emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_call_argument_count(
      module, op, callee, &low_signature, emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_call_result_count(
      module, op, callee, &low_signature, emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_call_argument_types(
      module, op, callee, &low_signature, emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_call_result_types(
      module, op, callee, &low_signature, emitter));
  return loom_low_verify_call_purity(module, op, callee,
                                     loom_low_func_call_purity(op),
                                     &low_signature, emitter);
}

iree_status_t loom_low_invoke_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  loom_symbol_ref_t callee = loom_low_invoke_callee(op);
  const loom_symbol_t* symbol = loom_low_lookup_defined_symbol(module, callee);
  if (!symbol) {
    return iree_ok_status();
  }

  loom_low_callee_signature_t low_signature = {0};
  if (!loom_low_load_low_signature(module, symbol, &low_signature)) {
    return loom_low_emit_call_callee_kind_mismatch(
        module, op, callee, loom_low_invoke_callee_ATTR_INDEX, symbol, emitter);
  }

  return loom_low_verify_call_purity(
      module, op, callee, loom_low_invoke_purity(op), &low_signature, emitter);
}

loom_trait_flags_t loom_low_func_call_effective_traits(const loom_op_t* op) {
  if (loom_low_func_call_purity(op) != 0) {
    return LOOM_TRAIT_PURE;
  }
  return LOOM_TRAIT_UNKNOWN_EFFECTS;
}

loom_trait_flags_t loom_low_invoke_effective_traits(const loom_op_t* op) {
  if (loom_low_invoke_purity(op) != 0) {
    return LOOM_TRAIT_PURE;
  }
  return LOOM_TRAIT_UNKNOWN_EFFECTS;
}
