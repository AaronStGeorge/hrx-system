// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_defs.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/scf/ops.h"

static iree_status_t loom_scf_emit(iree_diagnostic_emitter_t emitter,
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

static uint32_t loom_scf_saturating_u32(iree_host_size_t value) {
  if (value > UINT32_MAX) return UINT32_MAX;
  return (uint32_t)value;
}

static iree_status_t loom_scf_emit_attribute_kind_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, loom_attr_kind_t actual_kind,
    loom_attr_kind_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_u32(actual_kind),
      loom_param_u32(expected_kind),
  };
  return loom_scf_emit(emitter, op, &loom_err_type_005, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_scf_emit_attribute_value_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(attr_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_scf_emit(emitter, op, &loom_err_structure_014, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_scf_emit_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, iree_host_size_t actual_count,
    iree_string_view_t expected_field_name, iree_host_size_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_u32(loom_scf_saturating_u32(actual_count)),
      loom_param_string(expected_field_name),
      loom_param_u32(loom_scf_saturating_u32(expected_count)),
  };
  return loom_scf_emit(emitter, op, &loom_err_structure_013, params,
                       IREE_ARRAYSIZE(params));
}

static void loom_scf_format_lookup_field_name(char* buffer,
                                              iree_host_size_t buffer_capacity,
                                              const char* prefix,
                                              iree_host_size_t index) {
  iree_snprintf(buffer, buffer_capacity, "%s[%" PRIhsz "]", prefix, index);
}

static iree_status_t loom_scf_emit_lookup_type_mismatch(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_host_size_t value_index,
    iree_host_size_t result_index) {
  loom_value_slice_t values = loom_scf_lookup_values(op);
  loom_value_slice_t results = loom_scf_lookup_results(op);
  char value_name[32];
  char result_name[32];
  loom_scf_format_lookup_field_name(value_name, sizeof(value_name), "table",
                                    value_index);
  loom_scf_format_lookup_field_name(result_name, sizeof(result_name), "results",
                                    result_index);
  loom_diagnostic_param_t params[] = {
      loom_param_string(iree_make_cstring_view(value_name)),
      loom_param_type(
          loom_module_value_type(module, values.values[value_index])),
      loom_param_string(iree_make_cstring_view(result_name)),
      loom_param_type(
          loom_module_value_type(module, results.values[result_index])),
  };
  return loom_scf_emit(emitter, op, &loom_err_type_001, params,
                       IREE_ARRAYSIZE(params));
}

iree_status_t loom_scf_lookup_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  if (op->result_count == 0) {
    return loom_scf_emit_count_mismatch(emitter, op, IREE_SV("results"), 0,
                                        IREE_SV("at least one lookup result"),
                                        1);
  }

  loom_attribute_t case_keys = loom_scf_lookup_case_keys(op);
  if (case_keys.kind != LOOM_ATTR_I64_ARRAY) {
    return loom_scf_emit_attribute_kind_mismatch(
        emitter, op, IREE_SV("case_keys"), case_keys.kind, LOOM_ATTR_I64_ARRAY);
  }
  if (case_keys.count > 0 && !case_keys.i64_array) {
    return loom_scf_emit_attribute_value_constraint(
        emitter, op, IREE_SV("case_keys"), case_keys.count,
        IREE_SV("non-null i64_array storage when count is non-zero"));
  }

  for (uint16_t i = 1; i < case_keys.count; ++i) {
    if (case_keys.i64_array[i] <= case_keys.i64_array[i - 1]) {
      return loom_scf_emit_attribute_value_constraint(
          emitter, op, IREE_SV("case_keys"), case_keys.i64_array[i],
          IREE_SV("strictly increasing sorted unique case key"));
    }
  }

  loom_value_slice_t values = loom_scf_lookup_values(op);
  iree_host_size_t row_count = (iree_host_size_t)case_keys.count + 1;
  iree_host_size_t expected_value_count =
      row_count * (iree_host_size_t)op->result_count;
  if (values.count != expected_value_count) {
    return loom_scf_emit_count_mismatch(
        emitter, op, IREE_SV("table"), values.count,
        IREE_SV("(case key count + default row) * result count"),
        expected_value_count);
  }

  loom_value_slice_t results = loom_scf_lookup_results(op);
  for (iree_host_size_t i = 0; i < values.count; ++i) {
    iree_host_size_t result_index = i % op->result_count;
    loom_type_t value_type = loom_module_value_type(module, values.values[i]);
    loom_type_t result_type =
        loom_module_value_type(module, results.values[result_index]);
    if (!loom_type_equal(value_type, result_type)) {
      return loom_scf_emit_lookup_type_mismatch(module, emitter, op, i,
                                                result_index);
    }
  }

  return iree_ok_status();
}
