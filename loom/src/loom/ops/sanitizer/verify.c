// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/sanitizer/ops.h"

static iree_status_t loom_sanitizer_emit(iree_diagnostic_emitter_t emitter,
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

static bool loom_sanitizer_type_accepts_integer_predicates(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return loom_scalar_type_is_integer(scalar_type) ||
         scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET;
}

static bool loom_sanitizer_value_is_assert_operand(loom_value_slice_t values,
                                                   loom_value_id_t value_id) {
  for (uint16_t i = 0; i < values.count; ++i) {
    if (values.values[i] == value_id) return true;
  }
  return false;
}

static void loom_sanitizer_format_predicate_arg(char* buffer,
                                                iree_host_size_t capacity,
                                                uint16_t predicate_index,
                                                uint8_t argument_index) {
  iree_snprintf(buffer, capacity, "predicates[%u].arg[%u]", predicate_index,
                argument_index);
}

static iree_status_t loom_sanitizer_emit_unlisted_predicate_value(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, uint16_t predicate_index, uint8_t argument_index) {
  char field_name[40];
  loom_sanitizer_format_predicate_arg(field_name, sizeof(field_name),
                                      predicate_index, argument_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(iree_make_cstring_view(field_name)),
      loom_param_string(IREE_SV("an asserted value operand")),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_STRUCTURE_032, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_emit_predicate_value_type(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    uint16_t predicate_index, uint8_t argument_index, loom_type_t actual_type) {
  char field_name[40];
  loom_sanitizer_format_predicate_arg(field_name, sizeof(field_name),
                                      predicate_index, argument_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(iree_make_cstring_view(field_name)),
      loom_param_type(actual_type),
      loom_param_string(IREE_SV("integer, index, or offset value")),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                             IREE_ARRAYSIZE(params));
}

iree_status_t loom_sanitizer_assert_value_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_slice_t values = loom_sanitizer_assert_value_values(op);
  loom_attribute_t predicates = loom_sanitizer_assert_value_predicates(op);
  for (uint16_t predicate_index = 0; predicate_index < predicates.count;
       ++predicate_index) {
    const loom_predicate_t* predicate =
        &predicates.predicate_list[predicate_index];
    for (uint8_t argument_index = 0; argument_index < predicate->arg_count;
         ++argument_index) {
      if (predicate->arg_tags[argument_index] != LOOM_PRED_ARG_VALUE) continue;
      if (predicate->args[argument_index] < 0) {
        return loom_sanitizer_emit_unlisted_predicate_value(
            module, emitter, op, predicate_index, argument_index);
      }
      loom_value_id_t value_id =
          (loom_value_id_t)predicate->args[argument_index];
      if (!loom_sanitizer_value_is_assert_operand(values, value_id)) {
        return loom_sanitizer_emit_unlisted_predicate_value(
            module, emitter, op, predicate_index, argument_index);
      }
      loom_type_t value_type = loom_module_value_type(module, value_id);
      if (!loom_sanitizer_type_accepts_integer_predicates(value_type)) {
        return loom_sanitizer_emit_predicate_value_type(
            emitter, op, predicate_index, argument_index, value_type);
      }
    }
  }
  return iree_ok_status();
}
