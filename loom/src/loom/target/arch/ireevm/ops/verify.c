// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/function_contract_verify.h"
#include "loom/target/arch/ireevm/ops/ops.h"

static iree_status_t loom_ireevm_emit_string_attr_value_error(
    const loom_op_t* op, uint16_t attr_index, iree_string_view_t attr_name,
    iree_string_view_t actual_value, iree_string_view_t expected_constraint,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(attr_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
      loom_param_string(actual_value),
      loom_param_string(expected_constraint),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_STRUCTURE_027,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_ireevm_verify_non_empty_string_attr(
    const loom_module_t* module, const loom_op_t* op, uint16_t attr_index,
    loom_string_id_t string_id, iree_string_view_t attr_name,
    iree_string_view_t expected_constraint, iree_diagnostic_emitter_t emitter) {
  if (string_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ireevm import attribute %.*s string id %u is "
                            "invalid",
                            (int)attr_name.size, attr_name.data,
                            (uint32_t)string_id);
  }

  iree_string_view_t value = module->strings.entries[string_id];
  if (!iree_string_view_is_empty(value)) {
    return iree_ok_status();
  }
  return loom_ireevm_emit_string_attr_value_error(
      op, attr_index, attr_name, value, expected_constraint, emitter);
}

iree_status_t loom_ireevm_import_decl_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));
  return loom_ireevm_verify_non_empty_string_attr(
      module, op, loom_ireevm_import_decl_import_symbol_ATTR_INDEX,
      loom_ireevm_import_decl_import_symbol(op), IREE_SV("symbol"),
      IREE_SV("non-empty import symbol"), emitter);
}
