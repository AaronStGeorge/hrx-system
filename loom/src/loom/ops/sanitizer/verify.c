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
#include "loom/ops/view/access_verifier.h"

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Predicate assertions
//===----------------------------------------------------------------------===//

static bool loom_sanitizer_type_accepts_integer_predicates(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return loom_scalar_type_is_integer(scalar_type) ||
         scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET;
}

static bool loom_sanitizer_type_accepts_float_predicates(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_scalar_type_is_float(loom_type_element_type(type));
}

static bool loom_sanitizer_type_accepts_predicate(loom_type_t type,
                                                  uint8_t predicate_kind) {
  switch ((loom_predicate_kind_t)predicate_kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
    case LOOM_PREDICATE_LT:
    case LOOM_PREDICATE_LE:
    case LOOM_PREDICATE_GT:
    case LOOM_PREDICATE_GE:
    case LOOM_PREDICATE_MUL:
    case LOOM_PREDICATE_MIN:
    case LOOM_PREDICATE_MAX:
    case LOOM_PREDICATE_POW2:
    case LOOM_PREDICATE_RANGE:
      return loom_sanitizer_type_accepts_integer_predicates(type);
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
      return loom_sanitizer_type_accepts_float_predicates(type);
    case LOOM_PREDICATE_COUNT_:
      return false;
  }
  return false;
}

static iree_string_view_t loom_sanitizer_predicate_expected_type(
    uint8_t predicate_kind) {
  switch ((loom_predicate_kind_t)predicate_kind) {
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
      return IREE_SV("floating-point value");
    default:
      return IREE_SV("integer, index, or offset value");
  }
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
    const loom_op_t* op, uint16_t predicate_index, uint8_t argument_index,
    iree_string_view_t expected_constraint) {
  char field_name[40];
  loom_sanitizer_format_predicate_arg(field_name, sizeof(field_name),
                                      predicate_index, argument_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(iree_make_cstring_view(field_name)),
      loom_param_string(expected_constraint),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_STRUCTURE_032, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_emit_predicate_value_type(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    uint16_t predicate_index, uint8_t argument_index, loom_type_t actual_type,
    iree_string_view_t expected_type) {
  char field_name[40];
  loom_sanitizer_format_predicate_arg(field_name, sizeof(field_name),
                                      predicate_index, argument_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(iree_make_cstring_view(field_name)),
      loom_param_type(actual_type),
      loom_param_string(expected_type),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_verify_predicates_reference_values(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_value_slice_t values,
    loom_attribute_t predicates, iree_string_view_t expected_constraint) {
  for (uint16_t predicate_index = 0; predicate_index < predicates.count;
       ++predicate_index) {
    const loom_predicate_t* predicate =
        &predicates.predicate_list[predicate_index];
    for (uint8_t argument_index = 0; argument_index < predicate->arg_count;
         ++argument_index) {
      if (predicate->arg_tags[argument_index] != LOOM_PRED_ARG_VALUE) continue;
      if (predicate->args[argument_index] < 0) {
        return loom_sanitizer_emit_unlisted_predicate_value(
            module, emitter, op, predicate_index, argument_index,
            expected_constraint);
      }
      loom_value_id_t value_id =
          (loom_value_id_t)predicate->args[argument_index];
      if (!loom_sanitizer_value_is_assert_operand(values, value_id)) {
        return loom_sanitizer_emit_unlisted_predicate_value(
            module, emitter, op, predicate_index, argument_index,
            expected_constraint);
      }
      loom_type_t value_type = loom_module_value_type(module, value_id);
      if (!loom_sanitizer_type_accepts_predicate(value_type, predicate->kind)) {
        return loom_sanitizer_emit_predicate_value_type(
            emitter, op, predicate_index, argument_index, value_type,
            loom_sanitizer_predicate_expected_type(predicate->kind));
      }
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Layout assertions
//===----------------------------------------------------------------------===//

static iree_status_t loom_sanitizer_verify_type_has_encoding(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    iree_string_view_t field_name, loom_type_t type) {
  if (!loom_type_is_view(type) || loom_type_has_encoding(type)) {
    return iree_ok_status();
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_string(IREE_SV("view type")),
  };
  return loom_sanitizer_emit(emitter, op, LOOM_ERR_ENCODING_001, params,
                             IREE_ARRAYSIZE(params));
}

static iree_status_t loom_sanitizer_verify_static_dimensions(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_type_t source_type, loom_type_t result_type) {
  uint8_t rank = loom_type_rank(source_type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    if (loom_type_dim_is_dynamic_at(source_type, axis) ||
        loom_type_dim_is_dynamic_at(result_type, axis)) {
      continue;
    }
    int64_t source_size = loom_type_dim_static_size_at(source_type, axis);
    int64_t result_size = loom_type_dim_static_size_at(result_type, axis);
    if (source_size == result_size) continue;

    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("source static dimension")),
        loom_param_i64(source_size),
        loom_param_string(IREE_SV("result static dimension")),
        loom_param_i64(result_size),
    };
    return loom_sanitizer_emit(emitter, op, LOOM_ERR_SHAPE_001, params,
                               IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Op verifiers
//===----------------------------------------------------------------------===//

iree_status_t loom_sanitizer_assert_access_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_sanitizer_assert_access_view(op));
  return loom_view_verify_element_access(
      module, op, emitter, IREE_SV("view"), view_type,
      loom_sanitizer_assert_access_static_indices(op),
      loom_sanitizer_assert_access_indices(op).count);
}

iree_status_t loom_sanitizer_assert_value_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_sanitizer_verify_predicates_reference_values(
      module, op, emitter, loom_sanitizer_assert_value_values(op),
      loom_sanitizer_assert_value_predicates(op),
      IREE_SV("an asserted value operand"));
}

iree_status_t loom_sanitizer_assert_op_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_sanitizer_verify_predicates_reference_values(
      module, op, emitter, loom_sanitizer_assert_op_values(op),
      loom_sanitizer_assert_op_predicates(op), IREE_SV("an assertion operand"));
}

iree_status_t loom_sanitizer_assert_layout_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_sanitizer_assert_layout_view(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_sanitizer_assert_layout_result(op));

  IREE_RETURN_IF_ERROR(loom_sanitizer_verify_type_has_encoding(
      op, emitter, IREE_SV("view type layout"), source_type));
  IREE_RETURN_IF_ERROR(loom_sanitizer_verify_type_has_encoding(
      op, emitter, IREE_SV("result type layout"), result_type));

  if (!loom_type_is_view(source_type) || !loom_type_is_view(result_type) ||
      !loom_type_rank_equals(source_type, result_type)) {
    return iree_ok_status();
  }
  return loom_sanitizer_verify_static_dimensions(op, emitter, source_type,
                                                 result_type);
}
