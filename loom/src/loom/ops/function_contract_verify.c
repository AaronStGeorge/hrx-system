// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/function_contract_verify.h"

#include "loom/error/error_defs.h"
#include "loom/ops/op_defs.h"

static iree_status_t loom_function_contract_emit_attr_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_i64(0),
      loom_param_string(expected_constraint),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14),
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static bool loom_function_contract_attr_present(loom_func_like_t function,
                                                uint8_t attr_index) {
  if (attr_index == LOOM_ATTR_INDEX_NONE) {
    return false;
  }
  return !loom_attr_is_absent(loom_op_attrs(function.op)[attr_index]);
}

iree_status_t loom_function_contract_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  loom_func_like_t function = loom_func_like_cast(module, (loom_op_t*)op);
  if (!loom_func_like_isa(function)) {
    return iree_ok_status();
  }

  const bool has_target =
      loom_symbol_ref_is_valid(loom_func_like_target(function));
  const bool has_abi = loom_function_contract_attr_present(
      function, function.vtable->abi_attr_index);
  const bool has_abi_attrs = loom_func_like_abi_attrs(function).count > 0;
  const bool has_export_symbol =
      loom_func_like_export_symbol(function) != LOOM_STRING_ID_INVALID;
  const bool has_export_attrs = loom_func_like_export_attrs(function).count > 0;
  if (!has_target &&
      (has_abi || has_abi_attrs || has_export_symbol || has_export_attrs)) {
    return loom_function_contract_emit_attr_constraint(
        emitter, op, IREE_SV("target"),
        IREE_SV("present when ABI or export attrs are present"));
  }

  return iree_ok_status();
}
