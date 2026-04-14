// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/view/ops.h"

static iree_status_t loom_view_emit(iree_diagnostic_emitter_t emitter,
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

static uint16_t loom_view_dynamic_sentinel_count(loom_attribute_t values) {
  uint16_t dynamic_count = 0;
  for (uint16_t i = 0; i < values.count; ++i) {
    if (values.i64_array[i] == INT64_MIN) ++dynamic_count;
  }
  return dynamic_count;
}

static iree_status_t loom_view_verify_dynamic_index_count(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_attribute_t static_values,
    uint16_t dynamic_count) {
  uint16_t expected_dynamic_count =
      loom_view_dynamic_sentinel_count(static_values);
  if (dynamic_count == expected_dynamic_count) return iree_ok_status();

  iree_string_view_t op_name = loom_op_name(module, op);
  loom_diagnostic_param_t params[] = {
      loom_param_string(op_name),
      loom_param_u32(dynamic_count),
      loom_param_u32(expected_dynamic_count),
  };
  return loom_view_emit(emitter, op, &loom_err_structure_001, params,
                        IREE_ARRAYSIZE(params));
}

static iree_status_t loom_view_verify_static_index_count_matches_rank(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t operand_name,
    loom_attribute_t static_values, loom_type_t shaped_type) {
  uint8_t rank = loom_type_rank(shaped_type);
  if (static_values.count == rank) return iree_ok_status();

  loom_diagnostic_param_t params[] = {
      loom_param_string(operand_name),
      loom_param_u32(static_values.count),
      loom_param_i64(rank),
  };
  return loom_view_emit(emitter, op, &loom_err_subrange_001, params,
                        IREE_ARRAYSIZE(params));
}

static iree_status_t loom_view_verify_type_has_encoding(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    iree_string_view_t field_name, loom_type_t type) {
  if (!loom_type_is_view(type) || loom_type_has_encoding(type)) {
    return iree_ok_status();
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_string(IREE_SV("view type")),
  };
  return loom_view_emit(emitter, op, &loom_err_encoding_001, params,
                        IREE_ARRAYSIZE(params));
}

static iree_status_t loom_view_refine_verify_static_dimensions(
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
    return loom_view_emit(emitter, op, &loom_err_shape_001, params,
                          IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

iree_status_t loom_view_subview_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_attribute_t static_offsets = loom_view_subview_static_offsets(op);
  IREE_RETURN_IF_ERROR(loom_view_verify_dynamic_index_count(
      module, op, emitter, static_offsets,
      loom_view_subview_offsets(op).count));

  loom_type_t source_type =
      loom_module_value_type(module, loom_view_subview_source(op));
  if (!loom_type_is_view(source_type)) return iree_ok_status();
  return loom_view_verify_static_index_count_matches_rank(
      module, op, emitter, IREE_SV("source"), static_offsets, source_type);
}

iree_status_t loom_view_refine_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_view_refine_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_view_refine_result(op));

  IREE_RETURN_IF_ERROR(loom_view_verify_type_has_encoding(
      op, emitter, IREE_SV("source type layout"), source_type));
  IREE_RETURN_IF_ERROR(loom_view_verify_type_has_encoding(
      op, emitter, IREE_SV("result type layout"), result_type));

  if (!loom_type_is_view(source_type) || !loom_type_is_view(result_type) ||
      !loom_type_rank_equals(source_type, result_type)) {
    return iree_ok_status();
  }
  return loom_view_refine_verify_static_dimensions(op, emitter, source_type,
                                                   result_type);
}

iree_status_t loom_view_prefetch_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_attribute_t static_indices = loom_view_prefetch_static_indices(op);
  IREE_RETURN_IF_ERROR(loom_view_verify_dynamic_index_count(
      module, op, emitter, static_indices,
      loom_view_prefetch_indices(op).count));

  loom_type_t view_type =
      loom_module_value_type(module, loom_view_prefetch_view(op));
  if (!loom_type_is_view(view_type)) return iree_ok_status();
  return loom_view_verify_static_index_count_matches_rank(
      module, op, emitter, IREE_SV("view"), static_indices, view_type);
}
