// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/storage.h"

#include <stdint.h>

#include "loom/error/error_defs.h"
#include "loom/ir/context.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/encoding/roles.h"
#include "loom/util/fact_table.h"

static iree_string_view_t loom_encoding_physical_storage_name(void) {
  return IREE_SV("physical_storage");
}

static iree_string_view_t loom_encoding_amdgpu_matrix_operand_name(void) {
  return IREE_SV("amdgpu_matrix_operand");
}

static iree_string_view_t loom_encoding_layout_param_name(void) {
  return IREE_SV("layout");
}

static iree_string_view_t loom_encoding_schema_param_name(void) {
  return IREE_SV("schema");
}

static iree_string_view_t loom_encoding_stride_param_name(void) {
  return IREE_SV("stride");
}

static iree_string_view_t loom_encoding_strides_param_name(void) {
  return IREE_SV("strides");
}

static iree_string_view_t loom_encoding_format_param_name(void) {
  return IREE_SV("format");
}

static iree_string_view_t loom_encoding_packed_elements_param_name(void) {
  return IREE_SV("packed_elements");
}

static iree_string_view_t loom_encoding_packed_registers_param_name(void) {
  return IREE_SV("packed_registers");
}

static iree_string_view_t loom_encoding_scale_param_name(void) {
  return IREE_SV("scale");
}

static iree_string_view_t loom_encoding_scale_format_param_name(void) {
  return IREE_SV("scale_format");
}

static iree_string_view_t loom_encoding_scale_placement_param_name(void) {
  return IREE_SV("scale_placement");
}

static iree_string_view_t loom_encoding_scale_conversion_param_name(void) {
  return IREE_SV("scale_conversion");
}

static iree_string_view_t loom_encoding_zero_scale_fallback_param_name(void) {
  return IREE_SV("zero_scale_fallback");
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

static bool loom_encoding_name_equal(const loom_module_t* module,
                                     const loom_encoding_t* encoding,
                                     iree_string_view_t expected) {
  if (!encoding) return false;
  return loom_encoding_string_id_equal(module, encoding->name_id, expected);
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

static bool loom_encoding_string_attr_value(const loom_module_t* module,
                                            loom_attribute_t attr,
                                            iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  if (attr.kind != LOOM_ATTR_STRING ||
      attr.string_id == LOOM_STRING_ID_INVALID ||
      attr.string_id >= module->strings.count) {
    return false;
  }
  *out_value = module->strings.entries[attr.string_id];
  return true;
}

static bool loom_encoding_static_string_param(const loom_module_t* module,
                                              const loom_encoding_t* encoding,
                                              iree_string_view_t param_name,
                                              iree_string_view_t* out_value) {
  const loom_named_attr_t* entry = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), param_name);
  return entry &&
         loom_encoding_string_attr_value(module, entry->value, out_value);
}

static bool loom_encoding_static_u16_param(const loom_module_t* module,
                                           const loom_encoding_t* encoding,
                                           iree_string_view_t param_name,
                                           uint16_t* out_value) {
  *out_value = 0;
  const loom_named_attr_t* entry = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), param_name);
  if (!entry || entry->value.kind != LOOM_ATTR_I64) return false;
  int64_t value = loom_attr_as_i64(entry->value);
  if (value <= 0 || value > UINT16_MAX) return false;
  *out_value = (uint16_t)value;
  return true;
}

static bool loom_encoding_static_bool_param(const loom_module_t* module,
                                            const loom_encoding_t* encoding,
                                            iree_string_view_t param_name,
                                            bool* out_value) {
  *out_value = false;
  const loom_named_attr_t* entry = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), param_name);
  if (!entry || entry->value.kind != LOOM_ATTR_BOOL) return false;
  *out_value = loom_attr_as_bool(entry->value);
  return true;
}

static bool loom_encoding_dynamic_param_value(
    const loom_encoding_define_param_view_t* params,
    const loom_named_attr_t* name_entry, loom_value_id_t* out_value) {
  if (!name_entry || name_entry->value.kind != LOOM_ATTR_I64) return false;
  int64_t ordinal = name_entry->value.i64;
  if (ordinal < 0 || ordinal >= params->dynamic_values.count) return false;
  *out_value = params->dynamic_values.values[ordinal];
  return true;
}

static loom_value_fact_address_layout_t loom_encoding_dense_address_layout(
    void) {
  return (loom_value_fact_address_layout_t){
      .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE,
      .rank = 0,
      .strides = NULL,
  };
}

static loom_value_fact_matrix_format_t loom_encoding_matrix_format_from_name(
    iree_string_view_t value) {
  if (iree_string_view_equal(value, IREE_SV("fp8"))) {
    return LOOM_VALUE_FACT_MATRIX_FORMAT_FP8;
  }
  if (iree_string_view_equal(value, IREE_SV("bf8"))) {
    return LOOM_VALUE_FACT_MATRIX_FORMAT_BF8;
  }
  if (iree_string_view_equal(value, IREE_SV("fp6"))) {
    return LOOM_VALUE_FACT_MATRIX_FORMAT_FP6;
  }
  if (iree_string_view_equal(value, IREE_SV("bf6"))) {
    return LOOM_VALUE_FACT_MATRIX_FORMAT_BF6;
  }
  if (iree_string_view_equal(value, IREE_SV("fp4"))) {
    return LOOM_VALUE_FACT_MATRIX_FORMAT_FP4;
  }
  return LOOM_VALUE_FACT_MATRIX_FORMAT_UNKNOWN;
}

static loom_value_fact_matrix_scale_kind_t
loom_encoding_matrix_scale_kind_from_name(iree_string_view_t value) {
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_NONE;
  }
  if (iree_string_view_equal(value, IREE_SV("scale32"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_32;
  }
  if (iree_string_view_equal(value, IREE_SV("scale16"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_16;
  }
  return LOOM_VALUE_FACT_MATRIX_SCALE_UNKNOWN;
}

static loom_value_fact_matrix_scale_format_t
loom_encoding_matrix_scale_format_from_name(iree_string_view_t value) {
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE;
  }
  if (iree_string_view_equal(value, IREE_SV("e8"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_E8;
  }
  if (iree_string_view_equal(value, IREE_SV("e5m3"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_E5M3;
  }
  if (iree_string_view_equal(value, IREE_SV("e4m3"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_E4M3;
  }
  return LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_UNKNOWN;
}

static loom_value_fact_matrix_scale_placement_t
loom_encoding_matrix_scale_placement_from_name(iree_string_view_t value) {
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_NONE;
  }
  if (iree_string_view_equal(value, IREE_SV("explicit"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_EXPLICIT;
  }
  if (iree_string_view_equal(value, IREE_SV("row0"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_ROW0;
  }
  if (iree_string_view_equal(value, IREE_SV("row1"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_ROW1;
  }
  return LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_UNKNOWN;
}

static loom_value_fact_matrix_scale_conversion_t
loom_encoding_matrix_scale_conversion_from_name(iree_string_view_t value) {
  if (iree_string_view_equal(value, IREE_SV("none"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_NONE;
  }
  if (iree_string_view_equal(value, IREE_SV("lane_local"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_LANE_LOCAL;
  }
  if (iree_string_view_equal(value, IREE_SV("convergent"))) {
    return LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_CONVERGENT;
  }
  return LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_UNKNOWN;
}

static bool loom_encoding_physical_storage_define_isa(
    const loom_module_t* module, const loom_op_t* op) {
  if (!module || !op || !loom_encoding_define_isa(op)) return false;
  const loom_encoding_t* spec =
      loom_module_encoding(module, loom_encoding_define_spec(op));
  return loom_encoding_name_equal(module, spec,
                                  loom_encoding_physical_storage_name());
}

static bool loom_encoding_strided_op_address_layout(
    const loom_op_t* op, loom_value_facts_t* stride_storage,
    iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  loom_attribute_t static_strides =
      loom_encoding_layout_strided_static_strides(op);
  if (static_strides.kind != LOOM_ATTR_I64_ARRAY ||
      static_strides.count > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK ||
      static_strides.count > stride_capacity ||
      (static_strides.count > 0 && !stride_storage)) {
    return false;
  }

  loom_value_slice_t dynamic_strides = loom_encoding_layout_strided_strides(op);
  uint16_t expected_dynamic_count = 0;
  for (uint16_t i = 0; i < static_strides.count; ++i) {
    if (static_strides.i64_array[i] == INT64_MIN) {
      ++expected_dynamic_count;
    }
  }
  if (expected_dynamic_count != dynamic_strides.count) return false;

  for (uint16_t i = 0; i < static_strides.count; ++i) {
    int64_t stride = static_strides.i64_array[i];
    if (stride == INT64_MIN) {
      stride_storage[i] = loom_value_facts_make(0, INT64_MAX, 1);
    } else {
      if (stride < 0) return false;
      stride_storage[i] = loom_value_facts_exact_i64(stride);
    }
  }

  *out_layout = (loom_value_fact_address_layout_t){
      .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      .rank = (uint8_t)static_strides.count,
      .strides = static_strides.count > 0 ? stride_storage : NULL,
  };
  return true;
}

static bool loom_encoding_assume_strided_op_address_layout(
    const loom_op_t* op, loom_value_facts_t* stride_storage,
    iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  int64_t rank = loom_encoding_layout_assume_strided_rank(op);
  if (rank < 0 || rank > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK ||
      (iree_host_size_t)rank > stride_capacity ||
      (rank > 0 && !stride_storage)) {
    return false;
  }
  for (int64_t i = 0; i < rank; ++i) {
    stride_storage[i] = loom_value_facts_make(0, INT64_MAX, 1);
  }
  *out_layout = (loom_value_fact_address_layout_t){
      .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      .rank = (uint8_t)rank,
      .strides = rank > 0 ? stride_storage : NULL,
  };
  return true;
}

static bool loom_encoding_query_value_address_layout_rec(
    const loom_module_t* module, loom_value_id_t value_id, uint8_t depth,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  if (!module || depth > 4 || value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= module->values.count) {
    return false;
  }

  loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_encoding(type)) return false;

  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return false;

  const loom_op_t* op = loom_value_def_op(value);
  if (loom_encoding_layout_dense_isa(op) ||
      loom_encoding_layout_assume_dense_isa(op)) {
    *out_layout = loom_encoding_dense_address_layout();
    return true;
  }
  if (loom_encoding_layout_strided_isa(op)) {
    return loom_encoding_strided_op_address_layout(op, stride_storage,
                                                   stride_capacity, out_layout);
  }
  if (loom_encoding_layout_assume_strided_isa(op)) {
    return loom_encoding_assume_strided_op_address_layout(
        op, stride_storage, stride_capacity, out_layout);
  }
  if (loom_encoding_assume_spec_isa(op)) {
    if (loom_encoding_query_static_address_layout(
            module, loom_encoding_assume_spec_spec(op), stride_storage,
            stride_capacity, out_layout)) {
      return true;
    }
    return loom_encoding_query_value_address_layout_rec(
        module, loom_encoding_assume_spec_enc(op), (uint8_t)(depth + 1),
        stride_storage, stride_capacity, out_layout);
  }
  if (loom_encoding_define_isa(op)) {
    if (loom_encoding_query_static_address_layout(
            module, loom_encoding_define_spec(op), stride_storage,
            stride_capacity, out_layout)) {
      return true;
    }
  }
  if (!loom_encoding_physical_storage_define_isa(module, op)) return false;

  loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(module, op);
  const loom_named_attr_t* layout_entry = loom_encoding_find_param(
      module, params.dynamic_names, loom_encoding_layout_param_name());
  loom_value_id_t layout_value = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_dynamic_param_value(&params, layout_entry,
                                         &layout_value)) {
    return false;
  }
  return loom_encoding_query_value_address_layout_rec(
      module, layout_value, (uint8_t)(depth + 1), stride_storage,
      stride_capacity, out_layout);
}

bool loom_encoding_query_value_address_layout(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  if (!out_layout) return false;
  *out_layout = (loom_value_fact_address_layout_t){0};
  return loom_encoding_query_value_address_layout_rec(
      module, value_id, /*depth=*/0, stride_storage, stride_capacity,
      out_layout);
}

static bool loom_encoding_static_dense_layout_isa(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  return loom_encoding_name_equal(module, encoding, IREE_SV("dense"));
}

static bool loom_encoding_static_strided_layout_isa(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  return loom_encoding_name_equal(module, encoding, IREE_SV("strided"));
}

static bool loom_encoding_static_strided_layout(
    const loom_module_t* module, const loom_encoding_t* encoding,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  const loom_named_attr_t* strides =
      loom_encoding_find_param(module, loom_encoding_attrs(encoding),
                               loom_encoding_strides_param_name());
  if (strides) {
    if (strides->value.kind != LOOM_ATTR_I64_ARRAY ||
        strides->value.count > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK ||
        strides->value.count > stride_capacity ||
        (strides->value.count > 0 && !stride_storage)) {
      return false;
    }
    for (uint16_t i = 0; i < strides->value.count; ++i) {
      int64_t stride = strides->value.i64_array[i];
      if (stride < 0) return false;
      stride_storage[i] = loom_value_facts_exact_i64(stride);
    }
    *out_layout = (loom_value_fact_address_layout_t){
        .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
        .rank = (uint8_t)strides->value.count,
        .strides = strides->value.count > 0 ? stride_storage : NULL,
    };
    return true;
  }

  const loom_named_attr_t* stride = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), loom_encoding_stride_param_name());
  if (!stride || stride->value.kind != LOOM_ATTR_I64 || stride_capacity < 1 ||
      !stride_storage || stride->value.i64 < 0) {
    return false;
  }
  stride_storage[0] = loom_value_facts_exact_i64(stride->value.i64);
  *out_layout = (loom_value_fact_address_layout_t){
      .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
      .rank = 1,
      .strides = stride_storage,
  };
  return true;
}

static bool loom_encoding_static_amdgpu_matrix_schema(
    const loom_module_t* module, uint16_t encoding_id,
    const loom_encoding_t* encoding,
    loom_value_fact_storage_schema_t* out_schema) {
  out_schema->static_spec_encoding_id = encoding_id;

  iree_string_view_t format_name = iree_string_view_empty();
  iree_string_view_t scale_name = iree_string_view_empty();
  iree_string_view_t scale_format_name = iree_string_view_empty();
  iree_string_view_t scale_placement_name = iree_string_view_empty();
  iree_string_view_t scale_conversion_name = iree_string_view_empty();
  uint16_t packed_elements = 0;
  uint16_t packed_registers = 0;
  bool zero_scale_fallback = false;
  if (!loom_encoding_static_string_param(
          module, encoding, loom_encoding_format_param_name(), &format_name) ||
      !loom_encoding_static_string_param(
          module, encoding, loom_encoding_scale_param_name(), &scale_name) ||
      !loom_encoding_static_string_param(
          module, encoding, loom_encoding_scale_format_param_name(),
          &scale_format_name) ||
      !loom_encoding_static_string_param(
          module, encoding, loom_encoding_scale_placement_param_name(),
          &scale_placement_name) ||
      !loom_encoding_static_string_param(
          module, encoding, loom_encoding_scale_conversion_param_name(),
          &scale_conversion_name) ||
      !loom_encoding_static_u16_param(
          module, encoding, loom_encoding_packed_elements_param_name(),
          &packed_elements) ||
      !loom_encoding_static_u16_param(
          module, encoding, loom_encoding_packed_registers_param_name(),
          &packed_registers) ||
      !loom_encoding_static_bool_param(
          module, encoding, loom_encoding_zero_scale_fallback_param_name(),
          &zero_scale_fallback)) {
    return true;
  }

  loom_value_fact_matrix_storage_schema_t matrix = {
      .format = loom_encoding_matrix_format_from_name(format_name),
      .scale_kind = loom_encoding_matrix_scale_kind_from_name(scale_name),
      .scale_format =
          loom_encoding_matrix_scale_format_from_name(scale_format_name),
      .scale_placement =
          loom_encoding_matrix_scale_placement_from_name(scale_placement_name),
      .scale_conversion = loom_encoding_matrix_scale_conversion_from_name(
          scale_conversion_name),
      .packed_register_count = packed_registers,
      .packed_element_count = packed_elements,
      .zero_scale_fallback = zero_scale_fallback,
  };
  if (matrix.format == LOOM_VALUE_FACT_MATRIX_FORMAT_UNKNOWN ||
      matrix.scale_kind == LOOM_VALUE_FACT_MATRIX_SCALE_UNKNOWN ||
      matrix.scale_format == LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_UNKNOWN ||
      matrix.scale_placement ==
          LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_UNKNOWN ||
      matrix.scale_conversion ==
          LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_UNKNOWN) {
    return true;
  }

  bool scaled = matrix.scale_kind != LOOM_VALUE_FACT_MATRIX_SCALE_NONE;
  if (!scaled &&
      (matrix.scale_format != LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE ||
       matrix.scale_placement != LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_NONE ||
       matrix.scale_conversion !=
           LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_NONE ||
       matrix.zero_scale_fallback)) {
    return true;
  }
  if (scaled &&
      (matrix.scale_placement == LOOM_VALUE_FACT_MATRIX_SCALE_PLACEMENT_NONE ||
       matrix.scale_conversion ==
           LOOM_VALUE_FACT_MATRIX_SCALE_CONVERSION_NONE)) {
    return true;
  }

  out_schema->matrix = matrix;
  return true;
}

static bool loom_encoding_query_static_address_layout_rec(
    const loom_module_t* module, uint16_t encoding_id, uint8_t depth,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  if (!module || depth > 4) return false;
  const loom_encoding_t* encoding = loom_module_encoding(module, encoding_id);
  if (loom_encoding_static_dense_layout_isa(module, encoding)) {
    *out_layout = loom_encoding_dense_address_layout();
    return true;
  }
  if (loom_encoding_static_strided_layout_isa(module, encoding)) {
    return loom_encoding_static_strided_layout(module, encoding, stride_storage,
                                               stride_capacity, out_layout);
  }
  if (!loom_encoding_name_equal(module, encoding,
                                loom_encoding_physical_storage_name())) {
    return false;
  }

  const loom_named_attr_t* layout = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), loom_encoding_layout_param_name());
  if (!layout || layout->value.kind != LOOM_ATTR_ENCODING) return false;
  return loom_encoding_query_static_address_layout_rec(
      module, loom_attr_as_encoding_id(layout->value), (uint8_t)(depth + 1),
      stride_storage, stride_capacity, out_layout);
}

bool loom_encoding_query_static_address_layout(
    const loom_module_t* module, uint16_t encoding_id,
    loom_value_facts_t* stride_storage, iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  if (!out_layout) return false;
  *out_layout = (loom_value_fact_address_layout_t){0};
  return loom_encoding_query_static_address_layout_rec(
      module, encoding_id, /*depth=*/0, stride_storage, stride_capacity,
      out_layout);
}

static bool loom_encoding_query_static_storage_schema_rec(
    const loom_module_t* module, uint16_t encoding_id, uint8_t depth,
    loom_value_fact_storage_schema_t* out_schema) {
  if (!module || depth > 4) return false;
  const loom_encoding_t* encoding = loom_module_encoding(module, encoding_id);
  if (!encoding) return false;
  if (loom_encoding_name_equal(module, encoding,
                               loom_encoding_physical_storage_name())) {
    const loom_named_attr_t* schema =
        loom_encoding_find_param(module, loom_encoding_attrs(encoding),
                                 loom_encoding_schema_param_name());
    if (!schema || schema->value.kind != LOOM_ATTR_ENCODING) return false;
    return loom_encoding_query_static_storage_schema_rec(
        module, loom_attr_as_encoding_id(schema->value), (uint8_t)(depth + 1),
        out_schema);
  }

  if (loom_encoding_static_role(module, encoding) !=
      LOOM_ENCODING_ROLE_STORAGE_SCHEMA) {
    return false;
  }

  *out_schema = (loom_value_fact_storage_schema_t){
      .static_spec_encoding_id = encoding_id,
  };
  if (loom_encoding_name_equal(module, encoding,
                               loom_encoding_amdgpu_matrix_operand_name())) {
    return loom_encoding_static_amdgpu_matrix_schema(module, encoding_id,
                                                     encoding, out_schema);
  }
  return true;
}

bool loom_encoding_query_static_storage_schema(
    const loom_module_t* module, uint16_t encoding_id,
    loom_value_fact_storage_schema_t* out_schema) {
  if (!out_schema) return false;
  *out_schema = (loom_value_fact_storage_schema_t){0};
  return loom_encoding_query_static_storage_schema_rec(module, encoding_id,
                                                       /*depth=*/0, out_schema);
}

static bool loom_encoding_value_address_layout(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_address_layout_t* out_layout) {
  *out_layout = (loom_value_fact_address_layout_t){0};
  loom_value_fact_encoding_summary_t summary = {0};
  if (!loom_value_facts_query_encoding_summary(context, facts, &summary)) {
    return false;
  }
  if (summary.address_layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_UNKNOWN) {
    return false;
  }
  *out_layout = summary.address_layout;
  return true;
}

static bool loom_encoding_value_storage_schema(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_storage_schema_t* out_schema) {
  *out_schema = (loom_value_fact_storage_schema_t){0};
  loom_value_fact_encoding_summary_t summary = {0};
  if (!loom_value_facts_query_encoding_summary(context, facts, &summary)) {
    return false;
  }
  if (summary.storage_schema.static_spec_encoding_id == 0 &&
      summary.storage_schema.matrix.format ==
          LOOM_VALUE_FACT_MATRIX_FORMAT_UNKNOWN) {
    return false;
  }
  *out_schema = summary.storage_schema;
  return true;
}

bool loom_encoding_query_type_address_layout(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t type, loom_value_facts_t* stride_storage,
    iree_host_size_t stride_capacity,
    loom_value_fact_address_layout_t* out_layout) {
  if (!out_layout) return false;
  *out_layout = (loom_value_fact_address_layout_t){0};
  if (!module || !loom_type_has_encoding(type)) return false;

  if (loom_type_has_static_encoding(type)) {
    return loom_encoding_query_static_address_layout(
        module, type.encoding_id, stride_storage, stride_capacity, out_layout);
  }

  if (!loom_type_has_ssa_encoding(type) || !context || !context->table) {
    return false;
  }
  loom_value_id_t value_id = loom_type_encoding_value_id(type);
  loom_value_facts_t facts =
      loom_value_fact_table_lookup(context->table, value_id);
  return loom_encoding_value_address_layout(context, facts, out_layout);
}

bool loom_encoding_query_type_storage_schema(
    const loom_fact_context_t* context, const loom_module_t* module,
    loom_type_t type, loom_value_fact_storage_schema_t* out_schema) {
  if (!out_schema) return false;
  *out_schema = (loom_value_fact_storage_schema_t){0};
  if (!module || !loom_type_has_encoding(type)) return false;

  if (loom_type_has_static_encoding(type)) {
    return loom_encoding_query_static_storage_schema(module, type.encoding_id,
                                                     out_schema);
  }

  if (!loom_type_has_ssa_encoding(type) || !context || !context->table) {
    return false;
  }
  loom_value_id_t value_id = loom_type_encoding_value_id(type);
  loom_value_facts_t facts =
      loom_value_fact_table_lookup(context->table, value_id);
  return loom_encoding_value_storage_schema(context, facts, out_schema);
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
    const loom_error_def_t* error, iree_string_view_t param_name) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_encoding_physical_storage_name()),
      loom_param_string(param_name),
  };
  return loom_encoding_emit(emitter, op, error, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_static_kind_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t param_name, loom_attr_kind_t actual_kind,
    iree_string_view_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_encoding_physical_storage_name()),
      loom_param_string(param_name),
      loom_param_u32(actual_kind),
      loom_param_string(expected_kind),
  };
  return loom_encoding_emit(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 10),
      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_dynamic_type_error(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t param_name,
    loom_value_id_t value_id) {
  loom_type_t actual_type = loom_module_value_type(module, value_id);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_encoding_physical_storage_name()),
      loom_param_string(param_name),
      loom_param_type(actual_type),
      loom_param_string(IREE_SV("encoding")),
  };
  return loom_encoding_emit(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 9), params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_emit_role_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t param_name, iree_string_view_t expected_role) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_encoding_physical_storage_name()),
      loom_param_string(param_name),
      loom_param_string(expected_role),
  };
  return loom_encoding_emit(
      emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 11),
      params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_encoding_physical_storage_verify_static_param(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, const loom_named_attr_t* entry,
    iree_string_view_t param_name, loom_encoding_role_t expected_role,
    iree_string_view_t expected_role_name) {
  if (entry->value.kind != LOOM_ATTR_ENCODING) {
    return loom_encoding_emit_static_kind_error(
        emitter, op, param_name, (loom_attr_kind_t)entry->value.kind,
        IREE_SV("encoding"));
  }

  const loom_encoding_t* nested =
      loom_module_encoding(module, loom_attr_as_encoding_id(entry->value));
  loom_encoding_role_t actual_role = loom_encoding_static_role(module, nested);
  if (actual_role != expected_role) {
    return loom_encoding_emit_role_error(emitter, op, param_name,
                                         expected_role_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_encoding_physical_storage_verify_dynamic_param(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter,
    const loom_encoding_define_param_view_t* params,
    const loom_named_attr_t* entry, iree_string_view_t param_name,
    loom_encoding_role_t expected_role, iree_string_view_t expected_role_name) {
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  if (!loom_encoding_dynamic_param_value(params, entry, &value_id)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "malformed encoding.define operand dictionary for parameter '%.*s'",
        (int)param_name.size, param_name.data);
  }

  if (!loom_type_is_encoding(loom_module_value_type(module, value_id))) {
    return loom_encoding_emit_dynamic_type_error(module, emitter, op,
                                                 param_name, value_id);
  }

  loom_encoding_role_t actual_role = loom_encoding_value_role(module, value_id);
  if (actual_role != expected_role) {
    return loom_encoding_emit_role_error(emitter, op, param_name,
                                         expected_role_name);
  }
  return iree_ok_status();
}

static iree_status_t loom_encoding_physical_storage_verify_define(
    const loom_module_t* module, const loom_op_t* op,
    const loom_encoding_define_param_view_t* params,
    iree_diagnostic_emitter_t emitter) {
  for (iree_host_size_t i = 0; i < params->static_attrs.count; ++i) {
    const loom_named_attr_t* entry = &params->static_attrs.entries[i];
    if (!loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_layout_param_name()) &&
        !loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_schema_param_name())) {
      iree_string_view_t param_name = module->strings.entries[entry->name_id];
      return loom_encoding_emit_param_error(
          emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 8),
          param_name);
    }
  }
  for (iree_host_size_t i = 0; i < params->dynamic_names.count; ++i) {
    const loom_named_attr_t* entry = &params->dynamic_names.entries[i];
    if (!loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_layout_param_name()) &&
        !loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_schema_param_name())) {
      iree_string_view_t param_name = module->strings.entries[entry->name_id];
      return loom_encoding_emit_param_error(
          emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 8),
          param_name);
    }
  }

  const loom_named_attr_t* static_layout = loom_encoding_find_param(
      module, params->static_attrs, loom_encoding_layout_param_name());
  const loom_named_attr_t* dynamic_layout = loom_encoding_find_param(
      module, params->dynamic_names, loom_encoding_layout_param_name());
  if (!static_layout && !dynamic_layout) {
    return loom_encoding_emit_param_error(
        emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 7),
        loom_encoding_layout_param_name());
  }

  const loom_named_attr_t* static_schema = loom_encoding_find_param(
      module, params->static_attrs, loom_encoding_schema_param_name());
  const loom_named_attr_t* dynamic_schema = loom_encoding_find_param(
      module, params->dynamic_names, loom_encoding_schema_param_name());
  if (!static_schema && !dynamic_schema) {
    return loom_encoding_emit_param_error(
        emitter, op, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 7),
        loom_encoding_schema_param_name());
  }

  if (static_layout) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_static_param(
        module, op, emitter, static_layout, loom_encoding_layout_param_name(),
        LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT)));
  }
  if (dynamic_layout) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_dynamic_param(
        module, op, emitter, params, dynamic_layout,
        loom_encoding_layout_param_name(), LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT)));
  }

  if (static_schema) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_static_param(
        module, op, emitter, static_schema, loom_encoding_schema_param_name(),
        LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_STORAGE_SCHEMA)));
  }
  if (dynamic_schema) {
    IREE_RETURN_IF_ERROR(loom_encoding_physical_storage_verify_dynamic_param(
        module, op, emitter, params, dynamic_schema,
        loom_encoding_schema_param_name(), LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
        loom_encoding_role_description(LOOM_ENCODING_ROLE_STORAGE_SCHEMA)));
  }

  return iree_ok_status();
}

static iree_status_t loom_encoding_physical_storage_verify_static(
    const loom_module_t* module, const loom_encoding_t* encoding) {
  const loom_named_attr_t* layout = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), loom_encoding_layout_param_name());
  const loom_named_attr_t* schema = loom_encoding_find_param(
      module, loom_encoding_attrs(encoding), loom_encoding_schema_param_name());

  for (iree_host_size_t i = 0; i < encoding->attribute_count; ++i) {
    const loom_named_attr_t* entry = &encoding->attributes[i];
    if (!loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_layout_param_name()) &&
        !loom_encoding_string_id_equal(module, entry->name_id,
                                       loom_encoding_schema_param_name())) {
      iree_string_view_t param_name = module->strings.entries[entry->name_id];
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "encoding 'physical_storage' does not support parameter '%.*s'",
          (int)param_name.size, param_name.data);
    }
  }

  if (layout && layout->value.kind != LOOM_ATTR_ENCODING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "encoding 'physical_storage' parameter 'layout' must be an encoding");
  }
  if (schema && schema->value.kind != LOOM_ATTR_ENCODING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "encoding 'physical_storage' parameter 'schema' must be an encoding");
  }

  if (layout) {
    const loom_encoding_t* layout_encoding =
        loom_module_encoding(module, loom_attr_as_encoding_id(layout->value));
    if (loom_encoding_static_role(module, layout_encoding) !=
        LOOM_ENCODING_ROLE_ADDRESS_LAYOUT) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "encoding 'physical_storage' parameter 'layout' must be an address "
          "layout encoding");
    }
  }

  if (schema) {
    const loom_encoding_t* schema_encoding =
        loom_module_encoding(module, loom_attr_as_encoding_id(schema->value));
    if (loom_encoding_static_role(module, schema_encoding) !=
        LOOM_ENCODING_ROLE_STORAGE_SCHEMA) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "encoding 'physical_storage' parameter 'schema' must be a storage "
          "schema encoding");
    }
  }

  return iree_ok_status();
}

static const loom_encoding_vtable_t loom_encoding_physical_storage_vtable = {
    .name = IREE_SVL("physical_storage"),
    .role = LOOM_ENCODING_ROLE_PHYSICAL_STORAGE,
    .verify = loom_encoding_physical_storage_verify_static,
    .verify_define = loom_encoding_physical_storage_verify_define,
};

iree_status_t loom_encoding_register_physical_storage_family(
    loom_context_t* context) {
  return loom_context_register_encoding_vtable(
      context, &loom_encoding_physical_storage_vtable);
}
