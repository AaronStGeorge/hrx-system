// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ops/target/ops.h"

static iree_status_t loom_target_emit(iree_diagnostic_emitter_t emitter,
                                      const loom_op_t* op,
                                      const loom_error_def_t* error,
                                      const loom_diagnostic_param_t* params,
                                      iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_target_emit_attr_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_target_emit(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14),
      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_target_verify_known_enum(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, uint8_t value) {
  if (value != 0) return iree_ok_status();
  return loom_target_emit_attr_constraint(emitter, op, attr_name, value,
                                          IREE_SV("a known nonzero case"));
}

iree_status_t loom_target_profile_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  loom_string_id_t preset_id = loom_target_profile_preset(op);
  if (preset_id < module->strings.count &&
      !iree_string_view_is_empty(
          iree_string_view_trim(module->strings.entries[preset_id]))) {
    return iree_ok_status();
  }
  return loom_target_emit_attr_constraint(emitter, op, IREE_SV("preset"), 0,
                                          IREE_SV("a non-empty preset key"));
}

static const loom_symbol_t* loom_target_lookup_symbol(
    const loom_module_t* module, loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->definition == NULL || symbol->defining_op == NULL) return NULL;
  return symbol;
}

static iree_string_view_t loom_target_symbol_name(const loom_module_t* module,
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

static iree_string_view_t loom_target_symbol_definition_name(
    const loom_symbol_t* symbol) {
  if (!symbol || !symbol->definition) return IREE_SV("unresolved");
  return loom_symbol_definition_descriptor_name(symbol->definition);
}

static iree_status_t loom_target_emit_record_kind_mismatch(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_symbol_ref_t ref,
    const loom_symbol_t* symbol, iree_string_view_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_target_symbol_name(module, ref)),
      loom_param_string(loom_target_symbol_definition_name(symbol)),
      loom_param_string(expected_kind),
  };
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("defined here"),
      .op = symbol->defining_op,
  }};
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 3),
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_target_verify_record_ref(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_symbol_ref_t ref,
    loom_op_kind_t expected_op_kind, iree_string_view_t expected_kind) {
  const loom_symbol_t* symbol = loom_target_lookup_symbol(module, ref);
  if (!symbol) return iree_ok_status();
  if (symbol->defining_op->kind == expected_op_kind) return iree_ok_status();
  return loom_target_emit_record_kind_mismatch(module, op, emitter, ref, symbol,
                                               expected_kind);
}

iree_status_t loom_target_artifact_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_target_verify_record_ref(
      module, op, emitter, loom_target_artifact_target(op),
      LOOM_OP_TARGET_PROFILE, IREE_SV("target profile")));
  uint8_t format = loom_target_artifact_artifact_format(op);
  if (format != 0) {
    IREE_RETURN_IF_ERROR(loom_target_verify_known_enum(
        emitter, op, IREE_SV("artifact_format"), format));
  }
  uint8_t abi = loom_target_artifact_abi(op);
  if (abi != 0) {
    IREE_RETURN_IF_ERROR(
        loom_target_verify_known_enum(emitter, op, IREE_SV("abi"), abi));
  }
  return iree_ok_status();
}
