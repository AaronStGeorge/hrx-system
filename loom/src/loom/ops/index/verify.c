// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"

static iree_status_t loom_index_emit(iree_diagnostic_emitter_t emitter,
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

static bool loom_index_type_is_address(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET;
}

static bool loom_index_type_is_integer_or_address(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return loom_scalar_type_is_integer(scalar_type) ||
         scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET;
}

static iree_status_t loom_index_emit_operand_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t operand_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(operand_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_index_emit(emitter, op, &loom_err_type_003, params,
                         IREE_ARRAYSIZE(params));
}

static iree_status_t loom_index_emit_result_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t result_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(result_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_index_emit(emitter, op, &loom_err_type_004, params,
                         IREE_ARRAYSIZE(params));
}

iree_status_t loom_index_constant_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_index_constant_result(op));
  if (!loom_index_type_is_address(result_type)) return iree_ok_status();
  loom_attribute_t value = loom_index_constant_value(op);
  loom_attr_kind_t expected_kind = LOOM_ATTR_ANY;
  if (loom_attr_matches_scalar_type(value, loom_type_element_type(result_type),
                                    &expected_kind)) {
    return iree_ok_status();
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("value")),
      loom_param_u32(value.kind),
      loom_param_u32(expected_kind),
  };
  return loom_index_emit(emitter, op, &loom_err_type_005, params,
                         IREE_ARRAYSIZE(params));
}

iree_status_t loom_index_cast_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  loom_type_t input_type =
      loom_module_value_type(module, loom_index_cast_input(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_index_cast_result(op));

  if (!loom_index_type_is_integer_or_address(input_type)) {
    return loom_index_emit_operand_constraint(
        emitter, op, IREE_SV("input"), input_type,
        IREE_SV("integer, index, or offset"));
  }
  if (!loom_index_type_is_integer_or_address(result_type)) {
    return loom_index_emit_result_constraint(
        emitter, op, IREE_SV("result"), result_type,
        IREE_SV("integer, index, or offset"));
  }
  if (!loom_index_type_is_address(input_type) &&
      !loom_index_type_is_address(result_type)) {
    return loom_index_emit_result_constraint(
        emitter, op, IREE_SV("result"), result_type,
        IREE_SV("index or offset boundary"));
  }
  return iree_ok_status();
}
