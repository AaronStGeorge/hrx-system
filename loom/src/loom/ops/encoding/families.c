// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/families.h"

#include <stdint.h>

#include "loom/error/error_defs.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/encoding/numeric_transform.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/encoding/roles.h"
#include "loom/ops/op_defs.h"

iree_status_t loom_encoding_register_physical_storage_family(
    loom_context_t* context);

static iree_string_view_t loom_encoding_turboquant_kv_name(void) {
  return IREE_SV("turboquant_kv");
}

static iree_string_view_t loom_encoding_amdgpu_matrix_operand_name(void) {
  return IREE_SV("amdgpu_matrix_operand");
}

static bool loom_encoding_string_id_equal(const loom_module_t* module,
                                          loom_string_id_t string_id,
                                          iree_string_view_t expected) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[string_id], expected);
}

static const loom_named_attr_t* loom_encoding_find_param(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    if (loom_encoding_string_id_equal(module, entry->name_id, name)) {
      return entry;
    }
  }
  return NULL;
}

static bool loom_encoding_dynamic_param_value(
    const loom_encoding_define_param_view_t* params,
    const loom_named_attr_t* name_entry, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (!name_entry || name_entry->value.kind != LOOM_ATTR_I64) return false;
  int64_t ordinal = name_entry->value.i64;
  if (ordinal < 0 || ordinal >= params->dynamic_values.count) return false;
  *out_value = params->dynamic_values.values[ordinal];
  return true;
}

static bool loom_encoding_string_attr_value(const loom_module_t* module,
                                            loom_attribute_t attr,
                                            iree_string_view_t* out_value) {
  if (attr.kind != LOOM_ATTR_STRING ||
      attr.string_id == LOOM_STRING_ID_INVALID ||
      attr.string_id >= module->strings.count) {
    return false;
  }
  *out_value = module->strings.entries[attr.string_id];
  return true;
}

static bool loom_encoding_amdgpu_matrix_static_param_name_is_supported(
    const loom_module_t* module, loom_string_id_t name_id) {
  return loom_encoding_string_id_equal(module, name_id, IREE_SV("format")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("packed_elements")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("packed_registers")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("scale")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("scale_conversion")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("scale_format")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("scale_placement")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("zero_scale_fallback"));
}

static bool loom_encoding_amdgpu_matrix_format_supported(
    iree_string_view_t value) {
  return iree_string_view_equal(value, IREE_SV("fp8")) ||
         iree_string_view_equal(value, IREE_SV("bf8")) ||
         iree_string_view_equal(value, IREE_SV("fp6")) ||
         iree_string_view_equal(value, IREE_SV("bf6")) ||
         iree_string_view_equal(value, IREE_SV("fp4"));
}

static bool loom_encoding_amdgpu_matrix_scale_supported(
    iree_string_view_t value) {
  return iree_string_view_equal(value, IREE_SV("none")) ||
         iree_string_view_equal(value, IREE_SV("scale32")) ||
         iree_string_view_equal(value, IREE_SV("scale16"));
}

static bool loom_encoding_amdgpu_matrix_scale_format_supported(
    iree_string_view_t value) {
  return iree_string_view_equal(value, IREE_SV("none")) ||
         iree_string_view_equal(value, IREE_SV("e8")) ||
         iree_string_view_equal(value, IREE_SV("e5m3")) ||
         iree_string_view_equal(value, IREE_SV("e4m3"));
}

static bool loom_encoding_amdgpu_matrix_scale_placement_supported(
    iree_string_view_t value) {
  return iree_string_view_equal(value, IREE_SV("none")) ||
         iree_string_view_equal(value, IREE_SV("explicit")) ||
         iree_string_view_equal(value, IREE_SV("row0")) ||
         iree_string_view_equal(value, IREE_SV("row1"));
}

static bool loom_encoding_amdgpu_matrix_scale_conversion_supported(
    iree_string_view_t value) {
  return iree_string_view_equal(value, IREE_SV("none")) ||
         iree_string_view_equal(value, IREE_SV("lane_local")) ||
         iree_string_view_equal(value, IREE_SV("convergent"));
}

static iree_status_t loom_encoding_emit(iree_diagnostic_emitter_t emitter,
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

static iree_status_t loom_encoding_emit_param_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, const loom_error_def_t* error,
    iree_string_view_t param_name) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
  };
  return loom_encoding_emit(emitter, op, error, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_static_kind_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, iree_string_view_t param_name,
    loom_attr_kind_t actual_kind, iree_string_view_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_u32(actual_kind),
      loom_param_string(expected_kind),
  };
  return loom_encoding_emit(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 10),
      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_static_value_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, iree_string_view_t param_name,
    iree_string_view_t actual_value, iree_string_view_t expected_values) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_string(actual_value),
      loom_param_string(expected_values),
  };
  return loom_encoding_emit(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 13),
      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_attribute_constraint_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t param_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(param_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_encoding_emit(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14),
      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_dynamic_type_error(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t encoding_name,
    iree_string_view_t param_name, loom_value_id_t value_id,
    iree_string_view_t expected_type) {
  loom_type_t actual_type = loom_module_value_type(module, value_id);
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_type(actual_type),
      loom_param_string(expected_type),
  };
  return loom_encoding_emit(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 9), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_role_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, iree_string_view_t param_name,
    iree_string_view_t expected_role) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_name),
      loom_param_string(expected_role),
  };
  return loom_encoding_emit(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 11),
      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_mutually_exclusive_param_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t encoding_name, iree_string_view_t param_a,
    iree_string_view_t param_b) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(encoding_name),
      loom_param_string(param_a),
      loom_param_string(param_b),
  };
  return loom_encoding_emit(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 14),
      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_amdgpu_matrix_require_static_param(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    const loom_named_attr_t** out_param) {
  *out_param = NULL;
  const loom_named_attr_t* entry =
      loom_encoding_find_param(module, params->static_attrs, param_name);
  if (!entry) {
    return loom_encoding_emit_param_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
        loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 7), param_name);
  }
  *out_param = entry;
  return iree_ok_status();
}

static iree_status_t loom_encoding_amdgpu_matrix_require_static_string(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    iree_string_view_t* out_value, bool* out_ok) {
  *out_value = iree_string_view_empty();
  *out_ok = false;
  const loom_named_attr_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_amdgpu_matrix_require_static_param(
      module, op, params, emitter, param_name, &entry));
  if (!entry) return iree_ok_status();
  if (!loom_encoding_string_attr_value(module, entry->value, out_value)) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(), param_name,
        (loom_attr_kind_t)entry->value.kind, IREE_SV("string"));
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_amdgpu_matrix_require_static_i64(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    int64_t* out_value, bool* out_ok) {
  *out_value = 0;
  *out_ok = false;
  const loom_named_attr_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_amdgpu_matrix_require_static_param(
      module, op, params, emitter, param_name, &entry));
  if (!entry) return iree_ok_status();
  if (entry->value.kind != LOOM_ATTR_I64) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(), param_name,
        (loom_attr_kind_t)entry->value.kind, IREE_SV("i64"));
  }
  *out_value = loom_attr_as_i64(entry->value);
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_amdgpu_matrix_require_static_bool(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    bool* out_value, bool* out_ok) {
  *out_value = false;
  *out_ok = false;
  const loom_named_attr_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_amdgpu_matrix_require_static_param(
      module, op, params, emitter, param_name, &entry));
  if (!entry) return iree_ok_status();
  if (entry->value.kind != LOOM_ATTR_BOOL) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(), param_name,
        (loom_attr_kind_t)entry->value.kind, IREE_SV("bool"));
  }
  *out_value = loom_attr_as_bool(entry->value);
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_amdgpu_matrix_verify_param_names(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  for (iree_host_size_t i = 0; i < params->static_attrs.count; ++i) {
    const loom_named_attr_t* entry = &params->static_attrs.entries[i];
    if (!loom_encoding_amdgpu_matrix_static_param_name_is_supported(
            module, entry->name_id)) {
      iree_string_view_t param_name = module->strings.entries[entry->name_id];
      return loom_encoding_emit_param_error(
          emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
          loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 8), param_name);
    }
  }
  for (iree_host_size_t i = 0; i < params->dynamic_names.count; ++i) {
    const loom_named_attr_t* entry = &params->dynamic_names.entries[i];
    iree_string_view_t param_name = module->strings.entries[entry->name_id];
    return loom_encoding_emit_param_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
        loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 8), param_name);
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_amdgpu_matrix_verify_static_strings(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;

  iree_string_view_t format = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_amdgpu_matrix_require_static_string(
      module, op, params, emitter, IREE_SV("format"), &format, out_ok));
  if (!*out_ok) return iree_ok_status();
  if (!loom_encoding_amdgpu_matrix_format_supported(format)) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
        IREE_SV("format"), format,
        IREE_SV("'fp8', 'bf8', 'fp6', 'bf6', or 'fp4'"));
  }

  iree_string_view_t scale = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_amdgpu_matrix_require_static_string(
      module, op, params, emitter, IREE_SV("scale"), &scale, out_ok));
  if (!*out_ok) return iree_ok_status();
  if (!loom_encoding_amdgpu_matrix_scale_supported(scale)) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
        IREE_SV("scale"), scale, IREE_SV("'none', 'scale32', or 'scale16'"));
  }

  iree_string_view_t scale_format = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_amdgpu_matrix_require_static_string(
      module, op, params, emitter, IREE_SV("scale_format"), &scale_format,
      out_ok));
  if (!*out_ok) return iree_ok_status();
  if (!loom_encoding_amdgpu_matrix_scale_format_supported(scale_format)) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
        IREE_SV("scale_format"), scale_format,
        IREE_SV("'none', 'e8', 'e5m3', or 'e4m3'"));
  }

  iree_string_view_t scale_placement = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_amdgpu_matrix_require_static_string(
      module, op, params, emitter, IREE_SV("scale_placement"), &scale_placement,
      out_ok));
  if (!*out_ok) return iree_ok_status();
  if (!loom_encoding_amdgpu_matrix_scale_placement_supported(scale_placement)) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
        IREE_SV("scale_placement"), scale_placement,
        IREE_SV("'none', 'explicit', 'row0', or 'row1'"));
  }

  iree_string_view_t scale_conversion = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_amdgpu_matrix_require_static_string(
      module, op, params, emitter, IREE_SV("scale_conversion"),
      &scale_conversion, out_ok));
  if (!*out_ok) return iree_ok_status();
  if (!loom_encoding_amdgpu_matrix_scale_conversion_supported(
          scale_conversion)) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
        IREE_SV("scale_conversion"), scale_conversion,
        IREE_SV("'none', 'lane_local', or 'convergent'"));
  }

  bool zero_scale_fallback = false;
  IREE_RETURN_IF_ERROR(loom_encoding_amdgpu_matrix_require_static_bool(
      module, op, params, emitter, IREE_SV("zero_scale_fallback"),
      &zero_scale_fallback, out_ok));
  if (!*out_ok) return iree_ok_status();

  bool scale_is_none = iree_string_view_equal(scale, IREE_SV("none"));
  if (scale_is_none && !iree_string_view_equal(scale_format, IREE_SV("none"))) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
        IREE_SV("scale_format"), scale_format,
        IREE_SV("'none' when scale is 'none'"));
  }
  if (scale_is_none &&
      !iree_string_view_equal(scale_placement, IREE_SV("none"))) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
        IREE_SV("scale_placement"), scale_placement,
        IREE_SV("'none' when scale is 'none'"));
  }
  if (scale_is_none &&
      !iree_string_view_equal(scale_conversion, IREE_SV("none"))) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
        IREE_SV("scale_conversion"), scale_conversion,
        IREE_SV("'none' when scale is 'none'"));
  }
  if (scale_is_none && zero_scale_fallback) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
        IREE_SV("zero_scale_fallback"), IREE_SV("true"),
        IREE_SV("false when scale is 'none'"));
  }
  if (!scale_is_none &&
      iree_string_view_equal(scale_placement, IREE_SV("none"))) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
        IREE_SV("scale_placement"), scale_placement,
        IREE_SV("'explicit', 'row0', or 'row1' when scaled"));
  }
  if (!scale_is_none &&
      iree_string_view_equal(scale_conversion, IREE_SV("none"))) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_amdgpu_matrix_operand_name(),
        IREE_SV("scale_conversion"), scale_conversion,
        IREE_SV("'lane_local' or 'convergent' when scaled"));
  }

  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_amdgpu_matrix_verify_define(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter) {
  bool param_ok = false;
  IREE_RETURN_IF_ERROR(loom_encoding_amdgpu_matrix_verify_param_names(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_encoding_amdgpu_matrix_verify_static_strings(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();

  int64_t packed_elements = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_amdgpu_matrix_require_static_i64(
      module, op, params, emitter, IREE_SV("packed_elements"), &packed_elements,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  if (packed_elements <= 0 || packed_elements > UINT16_MAX) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("packed_elements"), packed_elements,
        IREE_SV("positive and <= 65535"));
  }

  int64_t packed_registers = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_amdgpu_matrix_require_static_i64(
      module, op, params, emitter, IREE_SV("packed_registers"),
      &packed_registers, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (packed_registers <= 0 || packed_registers > UINT16_MAX) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("packed_registers"), packed_registers,
        IREE_SV("positive and <= 65535"));
  }

  return iree_ok_status();
}

static bool loom_encoding_turboquant_static_param_name_is_supported(
    const loom_module_t* module, loom_string_id_t name_id) {
  return loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("first_stage_bits")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("logical_element")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("logical_elems")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("pack_order")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("qjl_rows")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("record_bytes")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("residual_bits")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("scalar_quantizer")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("transform_family"));
}

static bool loom_encoding_turboquant_dynamic_param_name_is_supported(
    const loom_module_t* module, loom_string_id_t name_id) {
  return loom_encoding_string_id_equal(module, name_id, IREE_SV("centroids")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("qjl_transform")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("thresholds")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("transform"));
}

static iree_status_t loom_encoding_turboquant_require_static_param(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    const loom_named_attr_t** out_param) {
  *out_param = NULL;
  const loom_named_attr_t* entry =
      loom_encoding_find_param(module, params->static_attrs, param_name);
  if (!entry) {
    return loom_encoding_emit_param_error(
        emitter, op, loom_encoding_turboquant_kv_name(),
        loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 7), param_name);
  }
  *out_param = entry;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_require_dynamic_param(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    const loom_named_attr_t** out_param) {
  *out_param = NULL;
  const loom_named_attr_t* entry =
      loom_encoding_find_param(module, params->dynamic_names, param_name);
  if (!entry) {
    return loom_encoding_emit_param_error(
        emitter, op, loom_encoding_turboquant_kv_name(),
        loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 7), param_name);
  }
  *out_param = entry;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_require_static_i64(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    int64_t* out_value, bool* out_ok) {
  *out_value = 0;
  *out_ok = false;
  const loom_named_attr_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_param(
      module, op, params, emitter, param_name, &entry));
  if (!entry) return iree_ok_status();
  if (entry->value.kind != LOOM_ATTR_I64) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, loom_encoding_turboquant_kv_name(), param_name,
        (loom_attr_kind_t)entry->value.kind, IREE_SV("i64"));
  }
  *out_value = loom_attr_as_i64(entry->value);
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_require_static_string(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    iree_string_view_t* out_value, bool* out_ok) {
  *out_value = iree_string_view_empty();
  *out_ok = false;
  const loom_named_attr_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_param(
      module, op, params, emitter, param_name, &entry));
  if (!entry) return iree_ok_status();
  if (!loom_encoding_string_attr_value(module, entry->value, out_value)) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, loom_encoding_turboquant_kv_name(), param_name,
        (loom_attr_kind_t)entry->value.kind, IREE_SV("string"));
  }
  *out_ok = true;
  return iree_ok_status();
}

static bool loom_encoding_turboquant_transform_family_supported(
    iree_string_view_t value) {
  return iree_string_view_equal(value, IREE_SV("hadamard")) ||
         iree_string_view_equal(value, IREE_SV("hadamard_sign")) ||
         iree_string_view_equal(value, IREE_SV("sign_permute_hadamard"));
}

static bool loom_encoding_numeric_transform_static_param_name_is_supported(
    const loom_module_t* module, loom_string_id_t name_id) {
  return loom_encoding_string_id_equal(module, name_id, IREE_SV("family")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("input_elems")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("normalization")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("output_elems"));
}

static bool loom_encoding_numeric_transform_dynamic_param_name_is_supported(
    const loom_module_t* module, loom_string_id_t name_id) {
  return loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("input_elems")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("matrix")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("output_elems")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("permutation")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("seed")) ||
         loom_encoding_string_id_equal(module, name_id, IREE_SV("signs"));
}

static bool loom_encoding_numeric_transform_family_supported(
    iree_string_view_t value) {
  return loom_encoding_numeric_transform_family_from_name(value) !=
         LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_UNKNOWN;
}

static iree_string_view_t loom_encoding_numeric_transform_family_quoted(
    iree_string_view_t value) {
  if (iree_string_view_equal(value, IREE_SV("hadamard"))) {
    return IREE_SV("'hadamard'");
  }
  if (iree_string_view_equal(value, IREE_SV("hadamard_sign"))) {
    return IREE_SV("'hadamard_sign'");
  }
  if (iree_string_view_equal(value, IREE_SV("jl_dense"))) {
    return IREE_SV("'jl_dense'");
  }
  if (iree_string_view_equal(value, IREE_SV("sign_permute_hadamard"))) {
    return IREE_SV("'sign_permute_hadamard'");
  }
  return value;
}

static bool loom_encoding_numeric_transform_normalization_supported(
    iree_string_view_t value) {
  loom_encoding_numeric_transform_normalization_t normalization =
      LOOM_ENCODING_NUMERIC_TRANSFORM_NORMALIZATION_NONE;
  return loom_encoding_numeric_transform_normalization_from_name(
      value, &normalization);
}

static bool loom_encoding_numeric_transform_param_is_extent(
    const loom_module_t* module, loom_string_id_t name_id) {
  return loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("input_elems")) ||
         loom_encoding_string_id_equal(module, name_id,
                                       IREE_SV("output_elems"));
}

static iree_status_t loom_encoding_numeric_transform_verify_static_params(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  for (iree_host_size_t i = 0; i < params->static_attrs.count; ++i) {
    const loom_named_attr_t* entry = &params->static_attrs.entries[i];
    iree_string_view_t param_name = module->strings.entries[entry->name_id];
    if (!loom_encoding_numeric_transform_static_param_name_is_supported(
            module, entry->name_id)) {
      return loom_encoding_emit_param_error(
          emitter, op, loom_encoding_numeric_transform_name(),
          loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 8), param_name);
    }

    if (loom_encoding_numeric_transform_param_is_extent(module,
                                                        entry->name_id)) {
      if (entry->value.kind != LOOM_ATTR_I64) {
        return loom_encoding_emit_static_kind_error(
            emitter, op, loom_encoding_numeric_transform_name(), param_name,
            (loom_attr_kind_t)entry->value.kind, IREE_SV("i64"));
      }
      int64_t extent = loom_attr_as_i64(entry->value);
      if (extent <= 0) {
        return loom_encoding_emit_attribute_constraint_error(
            emitter, op, param_name, extent, IREE_SV("positive extent"));
      }
      continue;
    }

    iree_string_view_t string_value = iree_string_view_empty();
    if (!loom_encoding_string_attr_value(module, entry->value, &string_value)) {
      return loom_encoding_emit_static_kind_error(
          emitter, op, loom_encoding_numeric_transform_name(), param_name,
          (loom_attr_kind_t)entry->value.kind, IREE_SV("string"));
    }

    if (loom_encoding_string_id_equal(module, entry->name_id,
                                      IREE_SV("family"))) {
      if (loom_encoding_numeric_transform_family_supported(string_value)) {
        continue;
      }
      return loom_encoding_emit_static_value_error(
          emitter, op, loom_encoding_numeric_transform_name(), param_name,
          string_value,
          IREE_SV("'hadamard', 'hadamard_sign', 'jl_dense', or "
                  "'sign_permute_hadamard'"));
    }

    if (loom_encoding_numeric_transform_normalization_supported(string_value)) {
      continue;
    }
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_numeric_transform_name(), param_name,
        string_value, IREE_SV("'none' or 'orthonormal'"));
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_numeric_transform_verify_dynamic_params(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  for (iree_host_size_t i = 0; i < params->dynamic_names.count; ++i) {
    const loom_named_attr_t* entry = &params->dynamic_names.entries[i];
    iree_string_view_t param_name = module->strings.entries[entry->name_id];
    if (!loom_encoding_numeric_transform_dynamic_param_name_is_supported(
            module, entry->name_id)) {
      return loom_encoding_emit_param_error(
          emitter, op, loom_encoding_numeric_transform_name(),
          loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 8), param_name);
    }

    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    if (!loom_encoding_dynamic_param_value(params, entry, &value_id)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "malformed encoding.define operand dictionary for parameter '%.*s'",
          (int)param_name.size, param_name.data);
    }

    loom_type_t actual_type = loom_module_value_type(module, value_id);
    if (loom_encoding_numeric_transform_param_is_extent(module,
                                                        entry->name_id) ||
        loom_encoding_string_id_equal(module, entry->name_id,
                                      IREE_SV("seed"))) {
      if (loom_type_equal(actual_type,
                          loom_type_scalar(LOOM_SCALAR_TYPE_INDEX))) {
        continue;
      }
      return loom_encoding_emit_dynamic_type_error(
          module, emitter, op, loom_encoding_numeric_transform_name(),
          param_name, value_id, IREE_SV("index"));
    }

    if (loom_encoding_string_id_equal(module, entry->name_id,
                                      IREE_SV("signs"))) {
      if (loom_type_is_vector(actual_type) &&
          loom_type_element_type(actual_type) == LOOM_SCALAR_TYPE_I1) {
        continue;
      }
      return loom_encoding_emit_dynamic_type_error(
          module, emitter, op, loom_encoding_numeric_transform_name(),
          param_name, value_id, IREE_SV("i1 vector"));
    }

    if (loom_encoding_string_id_equal(module, entry->name_id,
                                      IREE_SV("permutation"))) {
      if (loom_type_is_vector(actual_type) &&
          loom_type_satisfies_constraint(
              actual_type,
              LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT)) {
        continue;
      }
      return loom_encoding_emit_dynamic_type_error(
          module, emitter, op, loom_encoding_numeric_transform_name(),
          param_name, value_id,
          IREE_SV("vector with index or non-i1 integer elements"));
    }

    if (loom_type_is_vector(actual_type) &&
        loom_scalar_type_is_float(loom_type_element_type(actual_type))) {
      continue;
    }
    return loom_encoding_emit_dynamic_type_error(
        module, emitter, op, loom_encoding_numeric_transform_name(), param_name,
        value_id, IREE_SV("floating-point vector"));
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_numeric_transform_require_param(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    bool family_must_be_static, bool* out_ok) {
  *out_ok = false;
  const loom_named_attr_t* static_param =
      loom_encoding_find_param(module, params->static_attrs, param_name);
  if (static_param) {
    *out_ok = true;
    return iree_ok_status();
  }
  if (!family_must_be_static &&
      loom_encoding_find_param(module, params->dynamic_names, param_name)) {
    *out_ok = true;
    return iree_ok_status();
  }
  return loom_encoding_emit_param_error(
      emitter, op, loom_encoding_numeric_transform_name(),
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 7), param_name);
}

static iree_status_t loom_encoding_numeric_transform_require_dynamic_param(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    bool* out_ok) {
  *out_ok = false;
  if (loom_encoding_find_param(module, params->dynamic_names, param_name)) {
    *out_ok = true;
    return iree_ok_status();
  }
  return loom_encoding_emit_param_error(
      emitter, op, loom_encoding_numeric_transform_name(),
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 7), param_name);
}

static bool loom_encoding_numeric_transform_has_dynamic_param(
    const loom_module_t* module,
    const loom_encoding_define_param_view_t* params,
    iree_string_view_t param_name) {
  return loom_encoding_find_param(module, params->dynamic_names, param_name) !=
         NULL;
}

static iree_status_t loom_encoding_numeric_transform_reject_dynamic_param(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    bool* out_ok) {
  *out_ok = false;
  if (!loom_encoding_find_param(module, params->dynamic_names, param_name)) {
    *out_ok = true;
    return iree_ok_status();
  }
  return loom_encoding_emit_param_error(
      emitter, op, loom_encoding_numeric_transform_name(),
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 8), param_name);
}

static iree_status_t loom_encoding_numeric_transform_verify_no_dynamic_payload(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
      module, op, params, emitter, IREE_SV("signs"), out_ok));
  if (!*out_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
      module, op, params, emitter, IREE_SV("permutation"), out_ok));
  if (!*out_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
      module, op, params, emitter, IREE_SV("matrix"), out_ok));
  if (!*out_ok) return iree_ok_status();
  return loom_encoding_numeric_transform_reject_dynamic_param(
      module, op, params, emitter, IREE_SV("seed"), out_ok);
}

static iree_status_t loom_encoding_numeric_transform_verify_no_matrix(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  return loom_encoding_numeric_transform_reject_dynamic_param(
      module, op, params, emitter, IREE_SV("matrix"), out_ok);
}

static iree_status_t
loom_encoding_numeric_transform_verify_hadamard_sign_payload(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  bool has_signs = loom_encoding_numeric_transform_has_dynamic_param(
      module, params, IREE_SV("signs"));
  bool has_seed = loom_encoding_numeric_transform_has_dynamic_param(
      module, params, IREE_SV("seed"));
  if (!has_signs && !has_seed) {
    return loom_encoding_emit_param_error(
        emitter, op, loom_encoding_numeric_transform_name(),
        loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 7),
        IREE_SV("signs or seed"));
  }
  if (has_signs && has_seed) {
    return loom_encoding_emit_mutually_exclusive_param_error(
        emitter, op, loom_encoding_numeric_transform_name(), IREE_SV("signs"),
        IREE_SV("seed"));
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_numeric_transform_verify_jl_normalization(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  const loom_named_attr_t* entry = loom_encoding_find_param(
      module, params->static_attrs, IREE_SV("normalization"));
  if (!entry) {
    *out_ok = true;
    return iree_ok_status();
  }

  iree_string_view_t value = iree_string_view_empty();
  if (!loom_encoding_string_attr_value(module, entry->value, &value)) {
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    *out_ok = true;
    return iree_ok_status();
  }
  return loom_encoding_emit_static_value_error(
      emitter, op, loom_encoding_numeric_transform_name(),
      IREE_SV("normalization"), value, IREE_SV("'none'"));
}

static iree_status_t loom_encoding_numeric_transform_verify_family_params(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  const loom_named_attr_t* family_entry =
      loom_encoding_find_param(module, params->static_attrs, IREE_SV("family"));
  if (!family_entry) return iree_ok_status();

  iree_string_view_t family_name = iree_string_view_empty();
  if (!loom_encoding_string_attr_value(module, family_entry->value,
                                       &family_name)) {
    return iree_ok_status();
  }

  switch (loom_encoding_numeric_transform_family_from_name(family_name)) {
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD:
      return loom_encoding_numeric_transform_verify_no_dynamic_payload(
          module, op, params, emitter, out_ok);
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD_SIGN: {
      IREE_RETURN_IF_ERROR(
          loom_encoding_numeric_transform_verify_hadamard_sign_payload(
              module, op, params, emitter, out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          module, op, params, emitter, IREE_SV("permutation"), out_ok));
      if (!*out_ok) return iree_ok_status();
      return loom_encoding_numeric_transform_verify_no_matrix(
          module, op, params, emitter, out_ok);
    }
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD: {
      IREE_RETURN_IF_ERROR(
          loom_encoding_numeric_transform_require_dynamic_param(
              module, op, params, emitter, IREE_SV("signs"), out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(
          loom_encoding_numeric_transform_require_dynamic_param(
              module, op, params, emitter, IREE_SV("permutation"), out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          module, op, params, emitter, IREE_SV("seed"), out_ok));
      if (!*out_ok) return iree_ok_status();
      return loom_encoding_numeric_transform_verify_no_matrix(
          module, op, params, emitter, out_ok);
    }
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_JL_DENSE: {
      IREE_RETURN_IF_ERROR(
          loom_encoding_numeric_transform_require_dynamic_param(
              module, op, params, emitter, IREE_SV("matrix"), out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          module, op, params, emitter, IREE_SV("signs"), out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          module, op, params, emitter, IREE_SV("permutation"), out_ok));
      if (!*out_ok) return iree_ok_status();
      IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_reject_dynamic_param(
          module, op, params, emitter, IREE_SV("seed"), out_ok));
      if (!*out_ok) return iree_ok_status();
      return loom_encoding_numeric_transform_verify_jl_normalization(
          module, op, params, emitter, out_ok);
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_encoding_numeric_transform_verify_define(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter) {
  bool param_ok = false;
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_verify_static_params(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_verify_dynamic_params(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_require_param(
      module, op, params, emitter, IREE_SV("family"),
      /*family_must_be_static=*/true, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_require_param(
      module, op, params, emitter, IREE_SV("input_elems"),
      /*family_must_be_static=*/false, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_require_param(
      module, op, params, emitter, IREE_SV("output_elems"),
      /*family_must_be_static=*/false, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_numeric_transform_verify_family_params(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();

  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_verify_static_param_kinds(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  for (iree_host_size_t i = 0; i < params->static_attrs.count; ++i) {
    const loom_named_attr_t* entry = &params->static_attrs.entries[i];
    iree_string_view_t param_name = module->strings.entries[entry->name_id];
    if (!loom_encoding_turboquant_static_param_name_is_supported(
            module, entry->name_id)) {
      return loom_encoding_emit_param_error(
          emitter, op, loom_encoding_turboquant_kv_name(),
          loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 8), param_name);
    }
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_verify_dynamic_param_names(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, bool* out_ok) {
  *out_ok = false;
  for (iree_host_size_t i = 0; i < params->dynamic_names.count; ++i) {
    const loom_named_attr_t* entry = &params->dynamic_names.entries[i];
    iree_string_view_t param_name = module->strings.entries[entry->name_id];
    if (!loom_encoding_turboquant_dynamic_param_name_is_supported(
            module, entry->name_id)) {
      return loom_encoding_emit_param_error(
          emitter, op, loom_encoding_turboquant_kv_name(),
          loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 8), param_name);
    }
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_ceil_bits_to_bytes(
    int64_t bit_count, int64_t* out_byte_count) {
  *out_byte_count = 0;
  int64_t padded_bits = 0;
  if (!iree_checked_add_i64(bit_count, 7, &padded_bits)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "bit count overflows int64");
  }
  *out_byte_count = padded_bits / 8;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_min_record_bytes(
    int64_t logical_elems, int64_t first_stage_bits, int64_t qjl_rows,
    int64_t residual_bits, int64_t* out_record_bytes) {
  *out_record_bytes = 0;
  int64_t code_bits = 0;
  int64_t code_bytes = 0;
  int64_t residual_total_bits = 0;
  int64_t residual_bytes = 0;
  if (!iree_checked_mul_i64(logical_elems, first_stage_bits, &code_bits) ||
      !iree_checked_mul_i64(qjl_rows, residual_bits, &residual_total_bits)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "turboquant record bit count overflows int64");
  }
  IREE_RETURN_IF_ERROR(
      loom_encoding_turboquant_ceil_bits_to_bytes(code_bits, &code_bytes));
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_ceil_bits_to_bytes(
      residual_total_bits, &residual_bytes));
  if (!iree_checked_add_i64(code_bytes, residual_bytes, out_record_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "turboquant record byte count overflows int64");
  }
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_verify_dynamic_view(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    int64_t expected_static_length, bool* out_ok) {
  *out_ok = false;
  const loom_named_attr_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_dynamic_param(
      module, op, params, emitter, param_name, &entry));
  if (!entry) return iree_ok_status();

  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_dynamic_param_value(params, entry, &value_id)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "malformed encoding.define operand dictionary for parameter '%.*s'",
        (int)param_name.size, param_name.data);
  }

  loom_type_t actual_type = loom_module_value_type(module, value_id);
  if (!loom_type_is_view(actual_type) || loom_type_rank(actual_type) != 1 ||
      !loom_scalar_type_is_float(loom_type_element_type(actual_type))) {
    return loom_encoding_emit_dynamic_type_error(
        module, emitter, op, loom_encoding_turboquant_kv_name(), param_name,
        value_id, IREE_SV("rank-1 floating-point view"));
  }

  if (!loom_type_dim_is_dynamic_at(actual_type, 0)) {
    int64_t actual_length = loom_type_dim_static_size_at(actual_type, 0);
    if (actual_length != expected_static_length) {
      return loom_encoding_emit_attribute_constraint_error(
          emitter, op, param_name, actual_length,
          IREE_SV("static view length matching schema"));
    }
  }
  *out_ok = true;
  return iree_ok_status();
}

static iree_status_t loom_encoding_turboquant_verify_dynamic_transform(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    bool* out_ok) {
  *out_ok = false;
  const loom_named_attr_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_dynamic_param(
      module, op, params, emitter, param_name, &entry));
  if (!entry) return iree_ok_status();

  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_dynamic_param_value(params, entry, &value_id)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "malformed encoding.define operand dictionary for parameter '%.*s'",
        (int)param_name.size, param_name.data);
  }

  loom_type_t actual_type = loom_module_value_type(module, value_id);
  if (!loom_type_is_encoding(actual_type)) {
    return loom_encoding_emit_dynamic_type_error(
        module, emitter, op, loom_encoding_turboquant_kv_name(), param_name,
        value_id, IREE_SV("encoding<transform>"));
  }
  if (loom_type_encoding_role(actual_type) !=
      LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM) {
    return loom_encoding_emit_role_error(
        emitter, op, loom_encoding_turboquant_kv_name(), param_name,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM));
  }
  *out_ok = true;
  return iree_ok_status();
}

static bool loom_encoding_try_get_numeric_transform_family(
    const loom_module_t* module, loom_value_id_t value_id,
    iree_string_view_t* out_family) {
  *out_family = iree_string_view_empty();
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;
  const loom_op_t* define_op = loom_value_def_op(value);
  if (!define_op || !loom_encoding_define_isa(define_op)) return false;

  loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(module, define_op);
  if (!params.spec ||
      !loom_encoding_string_id_equal(module, params.spec->name_id,
                                     loom_encoding_numeric_transform_name())) {
    return false;
  }
  const loom_named_attr_t* family_entry =
      loom_encoding_find_param(module, params.static_attrs, IREE_SV("family"));
  if (!family_entry) return false;
  return loom_encoding_string_attr_value(module, family_entry->value,
                                         out_family);
}

static iree_status_t loom_encoding_turboquant_verify_transform_family(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter, iree_string_view_t param_name,
    iree_string_view_t expected_family, bool* out_ok) {
  *out_ok = false;
  const loom_named_attr_t* entry =
      loom_encoding_find_param(module, params->dynamic_names, param_name);
  if (!entry) {
    *out_ok = true;
    return iree_ok_status();
  }

  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_dynamic_param_value(params, entry, &value_id)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "malformed encoding.define operand dictionary for parameter '%.*s'",
        (int)param_name.size, param_name.data);
  }

  iree_string_view_t actual_family = iree_string_view_empty();
  if (!loom_encoding_try_get_numeric_transform_family(module, value_id,
                                                      &actual_family)) {
    *out_ok = true;
    return iree_ok_status();
  }
  if (iree_string_view_equal(actual_family, expected_family)) {
    *out_ok = true;
    return iree_ok_status();
  }

  iree_string_view_t expected =
      loom_encoding_numeric_transform_family_quoted(expected_family);
  return loom_encoding_emit_static_value_error(
      emitter, op, loom_encoding_turboquant_kv_name(), param_name,
      actual_family, expected);
}

static iree_status_t loom_encoding_turboquant_verify_define(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter) {
  bool param_ok = false;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_static_param_kinds(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_dynamic_param_names(
      module, op, params, emitter, &param_ok));
  if (!param_ok) return iree_ok_status();

  int64_t first_stage_bits = 0;
  int64_t logical_elems = 0;
  int64_t qjl_rows = 0;
  int64_t record_bytes = 0;
  int64_t residual_bits = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      module, op, params, emitter, IREE_SV("first_stage_bits"),
      &first_stage_bits, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      module, op, params, emitter, IREE_SV("logical_elems"), &logical_elems,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      module, op, params, emitter, IREE_SV("qjl_rows"), &qjl_rows, &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      module, op, params, emitter, IREE_SV("record_bytes"), &record_bytes,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_i64(
      module, op, params, emitter, IREE_SV("residual_bits"), &residual_bits,
      &param_ok));
  if (!param_ok) return iree_ok_status();

  if (first_stage_bits < 1 || first_stage_bits > 8) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("first_stage_bits"), first_stage_bits,
        IREE_SV("between 1 and 8"));
  }
  if (logical_elems <= 0) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("logical_elems"), logical_elems,
        IREE_SV("positive"));
  }
  if (qjl_rows <= 0) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("qjl_rows"), qjl_rows, IREE_SV("positive"));
  }
  if (record_bytes <= 0) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("record_bytes"), record_bytes,
        IREE_SV("positive"));
  }
  if (residual_bits != 1) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("residual_bits"), residual_bits,
        IREE_SV("equal to 1"));
  }

  iree_string_view_t logical_element = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_string(
      module, op, params, emitter, IREE_SV("logical_element"), &logical_element,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  loom_scalar_type_t logical_scalar = LOOM_SCALAR_TYPE_COUNT_;
  if (!loom_scalar_type_parse(logical_element, &logical_scalar) ||
      !loom_scalar_type_is_float(logical_scalar)) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_turboquant_kv_name(),
        IREE_SV("logical_element"), logical_element,
        IREE_SV("floating-point scalar type"));
  }

  iree_string_view_t pack_order = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_string(
      module, op, params, emitter, IREE_SV("pack_order"), &pack_order,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  if (!iree_string_view_equal(pack_order, IREE_SV("lsb0"))) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_turboquant_kv_name(), IREE_SV("pack_order"),
        pack_order, IREE_SV("'lsb0'"));
  }

  iree_string_view_t scalar_quantizer = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_string(
      module, op, params, emitter, IREE_SV("scalar_quantizer"),
      &scalar_quantizer, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (!iree_string_view_equal(scalar_quantizer, IREE_SV("lloyd_max"))) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_turboquant_kv_name(),
        IREE_SV("scalar_quantizer"), scalar_quantizer, IREE_SV("'lloyd_max'"));
  }

  iree_string_view_t transform_family = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_require_static_string(
      module, op, params, emitter, IREE_SV("transform_family"),
      &transform_family, &param_ok));
  if (!param_ok) return iree_ok_status();
  if (!loom_encoding_turboquant_transform_family_supported(transform_family)) {
    return loom_encoding_emit_static_value_error(
        emitter, op, loom_encoding_turboquant_kv_name(),
        IREE_SV("transform_family"), transform_family,
        IREE_SV("'hadamard', 'hadamard_sign', or 'sign_permute_hadamard'"));
  }

  int64_t minimum_record_bytes = 0;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_min_record_bytes(
      logical_elems, first_stage_bits, qjl_rows, residual_bits,
      &minimum_record_bytes));
  if (record_bytes < minimum_record_bytes) {
    return loom_encoding_emit_attribute_constraint_error(
        emitter, op, IREE_SV("record_bytes"), record_bytes,
        IREE_SV("large enough for first-stage codes and QJL residual bits"));
  }

  int64_t centroid_count = 1LL << first_stage_bits;
  int64_t threshold_count = centroid_count - 1;
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_dynamic_view(
      module, op, params, emitter, IREE_SV("centroids"), centroid_count,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_dynamic_view(
      module, op, params, emitter, IREE_SV("thresholds"), threshold_count,
      &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_dynamic_transform(
      module, op, params, emitter, IREE_SV("qjl_transform"), &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_transform_family(
      module, op, params, emitter, IREE_SV("qjl_transform"),
      IREE_SV("jl_dense"), &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_dynamic_transform(
      module, op, params, emitter, IREE_SV("transform"), &param_ok));
  if (!param_ok) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_encoding_turboquant_verify_transform_family(
      module, op, params, emitter, IREE_SV("transform"), transform_family,
      &param_ok));
  if (!param_ok) return iree_ok_status();

  return iree_ok_status();
}

static const loom_encoding_vtable_t loom_encoding_dense_vtable = {
    .name = IREE_SVL("dense"),
    .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
};

static const loom_encoding_vtable_t loom_encoding_strided_vtable = {
    .name = IREE_SVL("strided"),
    .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
};

static const loom_encoding_vtable_t loom_encoding_q8_0_vtable = {
    .name = IREE_SVL("q8_0"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_ggml_q4_0_vtable = {
    .name = IREE_SVL("ggml_q4_0"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_ggml_q8_0_vtable = {
    .name = IREE_SVL("ggml_q8_0"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_q6_k_vtable = {
    .name = IREE_SVL("q6_k"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_ggml_q6_k_vtable = {
    .name = IREE_SVL("ggml_q6_k"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_ggml_iq_grid_vtable = {
    .name = IREE_SVL("ggml_iq_grid"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_loom_fp4_table_vtable = {
    .name = IREE_SVL("loom_fp4_table"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_ieee_fp8_e4m3_vtable = {
    .name = IREE_SVL("ieee_fp8_e4m3"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
};

static const loom_encoding_vtable_t loom_encoding_amdgpu_matrix_operand_vtable =
    {
        .name = IREE_SVL("amdgpu_matrix_operand"),
        .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
        .verify_define = loom_encoding_amdgpu_matrix_verify_define,
};

static const loom_encoding_vtable_t loom_encoding_numeric_transform_vtable = {
    .name = IREE_SVL("numeric_transform"),
    .role = LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM,
    .verify_define = loom_encoding_numeric_transform_verify_define,
};

static const loom_encoding_vtable_t loom_encoding_orthogonal_transform_vtable =
    {
        .name = IREE_SVL("orthogonal_transform"),
        .role = LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM,
};

static const loom_encoding_vtable_t loom_encoding_turboquant_kv_vtable = {
    .name = IREE_SVL("turboquant_kv"),
    .role = LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
    .verify_define = loom_encoding_turboquant_verify_define,
};

static const loom_encoding_vtable_t* const loom_encoding_builtin_vtables[] = {
    &loom_encoding_dense_vtable,
    &loom_encoding_strided_vtable,
    &loom_encoding_q8_0_vtable,
    &loom_encoding_ggml_q4_0_vtable,
    &loom_encoding_ggml_q8_0_vtable,
    &loom_encoding_q6_k_vtable,
    &loom_encoding_ggml_q6_k_vtable,
    &loom_encoding_ggml_iq_grid_vtable,
    &loom_encoding_loom_fp4_table_vtable,
    &loom_encoding_ieee_fp8_e4m3_vtable,
    &loom_encoding_amdgpu_matrix_operand_vtable,
    &loom_encoding_numeric_transform_vtable,
    &loom_encoding_orthogonal_transform_vtable,
    &loom_encoding_turboquant_kv_vtable,
};

iree_status_t loom_context_register_builtin_encoding_vtables(
    loom_context_t* context) {
  IREE_RETURN_IF_ERROR(loom_encoding_register_physical_storage_family(context));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(loom_encoding_builtin_vtables); ++i) {
    IREE_RETURN_IF_ERROR(loom_context_register_encoding_vtable(
        context, loom_encoding_builtin_vtables[i]));
  }
  return iree_ok_status();
}
