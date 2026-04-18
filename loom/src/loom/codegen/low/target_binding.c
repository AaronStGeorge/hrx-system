// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/target_binding.h"

#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"

static iree_status_t loom_low_emit(iree_diagnostic_emitter_t emitter,
                                   const loom_op_t* op,
                                   const loom_error_def_t* error,
                                   const loom_diagnostic_param_t* params,
                                   iree_host_size_t param_count,
                                   const loom_diagnostic_related_op_t* related,
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

static const loom_symbol_t* loom_low_lookup_defined_symbol(
    const loom_module_t* module, loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->definition == NULL || symbol->defining_op == NULL) return NULL;
  return symbol;
}

static iree_string_view_t loom_low_symbol_name(const loom_module_t* module,
                                               loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id < module->strings.count) {
    return module->strings.entries[symbol->name_id];
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_function_name(const loom_module_t* module,
                                                 const loom_op_t* low_func_op) {
  if (loom_low_func_def_isa(low_func_op)) {
    return loom_low_symbol_name(module, loom_low_func_def_callee(low_func_op));
  }
  if (loom_low_func_decl_isa(low_func_op)) {
    return loom_low_symbol_name(module, loom_low_func_decl_callee(low_func_op));
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_low_symbol_definition_name(
    const loom_symbol_t* symbol) {
  if (!symbol || !symbol->definition) return IREE_SV("unresolved");
  return loom_symbol_definition_descriptor_name(symbol->definition);
}

static iree_status_t loom_low_emit_symbol_kind_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_module_t* module,
    const loom_op_t* op, loom_symbol_ref_t ref, const loom_symbol_t* symbol,
    uint16_t attr_index, iree_string_view_t expected_kind) {
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
  return loom_low_emit(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3), params,
      IREE_ARRAYSIZE(params), related,
      symbol && symbol->defining_op ? IREE_ARRAYSIZE(related) : 0);
}

static iree_status_t loom_low_emit_unresolved_symbol(
    iree_diagnostic_emitter_t emitter, const loom_module_t* module,
    const loom_op_t* op, loom_symbol_ref_t ref, uint16_t attr_index) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(loom_low_symbol_name(module, ref)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
  };
  return loom_low_emit(emitter, op,
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 2),
                       params, IREE_ARRAYSIZE(params), NULL, 0);
}

static iree_status_t loom_low_emit_missing_descriptor_set(
    iree_diagnostic_emitter_t emitter, const loom_module_t* module,
    const loom_op_t* low_func_op, const loom_op_t* config_op,
    uint16_t target_attr_index, iree_string_view_t descriptor_set_key,
    iree_string_view_t target_name) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_function_name(module, low_func_op)),
      loom_param_with_field_ref(
          loom_param_string(target_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    target_attr_index)),
      loom_param_string(descriptor_set_key),
  };
  loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("descriptor set selected here"),
      .op = config_op,
      .field_ref = loom_diagnostic_field_ref(
          LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
          loom_target_config_contract_set_key_ATTR_INDEX),
  }};
  return loom_low_emit(emitter, low_func_op,
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_LOWERING, 3),
                       params, IREE_ARRAYSIZE(params), related,
                       IREE_ARRAYSIZE(related));
}

static iree_status_t loom_low_emit_target_config_feature_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* config_op,
    int64_t feature_bits) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("contract_feature_bits")),
          loom_diagnostic_field_ref(
              LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
              loom_target_config_contract_feature_bits_ATTR_INDEX)),
      loom_param_i64(feature_bits),
      loom_param_string(IREE_SV("a non-negative feature bitset")),
  };
  return loom_low_emit(emitter, config_op,
                       loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14),
                       params, IREE_ARRAYSIZE(params), NULL, 0);
}

static bool loom_low_get_function_target_ref(const loom_op_t* low_func_op,
                                             loom_symbol_ref_t* out_target_ref,
                                             uint16_t* out_target_attr_index) {
  if (loom_low_func_def_isa(low_func_op)) {
    *out_target_ref = loom_low_func_def_target(low_func_op);
    *out_target_attr_index = loom_low_func_def_target_ATTR_INDEX;
    return true;
  }
  if (loom_low_func_decl_isa(low_func_op)) {
    *out_target_ref = loom_low_func_decl_target(low_func_op);
    *out_target_attr_index = loom_low_func_decl_target_ATTR_INDEX;
    return true;
  }
  return false;
}

static iree_status_t loom_low_resolve_bundle_record(
    const loom_module_t* module, const loom_op_t* bundle_op,
    loom_symbol_ref_t record_ref, uint16_t record_attr_index,
    loom_op_kind_t expected_kind, iree_string_view_t expected_kind_name,
    iree_diagnostic_emitter_t emitter,
    const loom_symbol_t** out_record_symbol) {
  *out_record_symbol = NULL;
  const loom_symbol_t* record_symbol =
      loom_low_lookup_defined_symbol(module, record_ref);
  if (!record_symbol) {
    return loom_low_emit_unresolved_symbol(emitter, module, bundle_op,
                                           record_ref, record_attr_index);
  }
  if (record_symbol->defining_op->kind != expected_kind) {
    return loom_low_emit_symbol_kind_mismatch(
        emitter, module, bundle_op, record_ref, record_symbol,
        record_attr_index, expected_kind_name);
  }
  *out_record_symbol = record_symbol;
  return iree_ok_status();
}

iree_status_t loom_low_resolve_function_target(
    const loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_descriptor_registry_t* registry,
    iree_diagnostic_emitter_t emitter, loom_low_resolved_target_t* out_target) {
  if (!module || !low_func_op || !out_target) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, low function op, and output target are required");
  }
  *out_target = (loom_low_resolved_target_t){0};
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  uint16_t target_attr_index = 0;
  if (!loom_low_get_function_target_ref(low_func_op, &target_ref,
                                        &target_attr_index)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def or low.func.decl");
  }

  const loom_symbol_t* target_symbol =
      loom_low_lookup_defined_symbol(module, target_ref);
  if (!target_symbol) {
    return loom_low_emit_unresolved_symbol(emitter, module, low_func_op,
                                           target_ref, target_attr_index);
  }

  out_target->target_symbol = target_symbol;
  out_target->target_op = target_symbol->defining_op;
  out_target->target_name = loom_low_symbol_name(module, target_ref);

  if (!loom_target_bundle_isa(target_symbol->defining_op)) {
    return loom_low_emit_symbol_kind_mismatch(
        emitter, module, low_func_op, target_ref, target_symbol,
        target_attr_index, IREE_SV("target bundle"));
  }

  const loom_op_t* bundle_op = target_symbol->defining_op;
  const loom_symbol_t* snapshot_symbol = NULL;
  IREE_RETURN_IF_ERROR(loom_low_resolve_bundle_record(
      module, bundle_op, loom_target_bundle_snapshot(bundle_op),
      loom_target_bundle_snapshot_ATTR_INDEX, LOOM_OP_TARGET_SNAPSHOT,
      IREE_SV("target snapshot"), emitter, &snapshot_symbol));
  const loom_symbol_t* export_plan_symbol = NULL;
  IREE_RETURN_IF_ERROR(loom_low_resolve_bundle_record(
      module, bundle_op, loom_target_bundle_export_plan(bundle_op),
      loom_target_bundle_export_plan_ATTR_INDEX, LOOM_OP_TARGET_EXPORT,
      IREE_SV("target export plan"), emitter, &export_plan_symbol));
  const loom_symbol_t* config_symbol = NULL;
  IREE_RETURN_IF_ERROR(loom_low_resolve_bundle_record(
      module, bundle_op, loom_target_bundle_config(bundle_op),
      loom_target_bundle_config_ATTR_INDEX, LOOM_OP_TARGET_CONFIG,
      IREE_SV("target config"), emitter, &config_symbol));
  // Resolver diagnostics are emitted through |emitter| and return OK so the
  // verifier can continue collecting errors.
  if (!snapshot_symbol || !export_plan_symbol || !config_symbol) {
    return iree_ok_status();
  }

  out_target->snapshot_op = snapshot_symbol->defining_op;
  out_target->export_plan_op = export_plan_symbol->defining_op;
  const loom_op_t* config_op = config_symbol->defining_op;
  out_target->config_op = config_op;
  int64_t feature_bits = loom_target_config_contract_feature_bits(config_op);
  if (feature_bits < 0) {
    return loom_low_emit_target_config_feature_constraint(emitter, config_op,
                                                          feature_bits);
  }
  IREE_RETURN_IF_ERROR(loom_target_ir_bundle_from_ops(
      module, bundle_op, out_target->snapshot_op, out_target->export_plan_op,
      config_op, &out_target->bundle_storage));
  out_target->descriptor_set_key =
      out_target->bundle_storage.config.contract_set_key;
  out_target->feature_bits =
      out_target->bundle_storage.config.contract_feature_bits;

  const loom_low_descriptor_set_t* descriptor_set = NULL;
  IREE_RETURN_IF_ERROR(loom_low_descriptor_registry_lookup(
      registry, out_target->descriptor_set_key, &descriptor_set));
  if (!descriptor_set) {
    return loom_low_emit_missing_descriptor_set(
        emitter, module, low_func_op, config_op, target_attr_index,
        out_target->descriptor_set_key, out_target->target_name);
  }
  out_target->descriptor_set = descriptor_set;
  return iree_ok_status();
}
