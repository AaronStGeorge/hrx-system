// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/amdgpu/matrix_facts.h"

static bool loom_amdgpu_matrix_numeric_type_from_fact(
    loom_value_fact_matrix_format_t format,
    loom_amdgpu_matrix_numeric_type_t* out_numeric_type) {
  *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN;
  switch (format) {
    case LOOM_VALUE_FACT_MATRIX_FORMAT_FP8:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_FP8;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_BF8:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_BF8;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_FP6:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_FP6;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_BF6:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_BF6;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_FP4:
      *out_numeric_type = LOOM_AMDGPU_MATRIX_NUMERIC_FP4;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_UNKNOWN:
    default:
      return false;
  }
}

bool loom_amdgpu_matrix_format_selector_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_amdgpu_matrix_format_selector_t* out_selector) {
  if (!out_selector) return false;
  switch (schema.matrix.format) {
    case LOOM_VALUE_FACT_MATRIX_FORMAT_FP8:
      *out_selector = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP8;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_BF8:
      *out_selector = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_BF8;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_FP6:
      *out_selector = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP6;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_BF6:
      *out_selector = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_BF6;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_FP4:
      *out_selector = LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP4;
      return true;
    case LOOM_VALUE_FACT_MATRIX_FORMAT_UNKNOWN:
    default:
      return false;
  }
}

bool loom_amdgpu_matrix_payload_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_amdgpu_matrix_payload_shape_t* out_payload) {
  if (!out_payload) return false;
  *out_payload = (loom_amdgpu_matrix_payload_shape_t){0};
  loom_amdgpu_matrix_numeric_type_t numeric_type =
      LOOM_AMDGPU_MATRIX_NUMERIC_UNKNOWN;
  if (!loom_amdgpu_matrix_numeric_type_from_fact(schema.matrix.format,
                                                 &numeric_type)) {
    return false;
  }
  out_payload->numeric_type = numeric_type;
  out_payload->register_count = schema.matrix.packed_register_count;
  out_payload->element_count = schema.matrix.packed_element_count;
  return true;
}

bool loom_amdgpu_matrix_scale_kind_from_storage_schema(
    loom_value_fact_storage_schema_t schema,
    loom_amdgpu_matrix_scale_kind_t* out_scale_kind) {
  if (!out_scale_kind) return false;
  switch (schema.matrix.scale_kind) {
    case LOOM_VALUE_FACT_MATRIX_SCALE_NONE:
      *out_scale_kind = LOOM_AMDGPU_MATRIX_SCALE_NONE;
      return true;
    case LOOM_VALUE_FACT_MATRIX_SCALE_32:
      *out_scale_kind = LOOM_AMDGPU_MATRIX_SCALE_32;
      return true;
    case LOOM_VALUE_FACT_MATRIX_SCALE_16:
      *out_scale_kind = LOOM_AMDGPU_MATRIX_SCALE_16;
      return true;
    case LOOM_VALUE_FACT_MATRIX_SCALE_UNKNOWN:
    default:
      return false;
  }
}

loom_amdgpu_matrix_contract_flags_t
loom_amdgpu_matrix_available_flags_from_storage_schema(
    loom_value_fact_storage_schema_t schema) {
  loom_amdgpu_matrix_contract_flags_t flags = 0;
  loom_amdgpu_matrix_format_selector_t selector =
      LOOM_AMDGPU_MATRIX_FORMAT_SELECTOR_FP8;
  if (loom_amdgpu_matrix_format_selector_from_storage_schema(schema,
                                                             &selector)) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_MATRIX_FORMATS;
  }
  if (schema.matrix.scale_kind != LOOM_VALUE_FACT_MATRIX_SCALE_UNKNOWN &&
      schema.matrix.scale_kind != LOOM_VALUE_FACT_MATRIX_SCALE_NONE) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALED;
  }
  if (schema.matrix.scale_format !=
          LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_UNKNOWN &&
      schema.matrix.scale_format != LOOM_VALUE_FACT_MATRIX_SCALE_FORMAT_NONE) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_SCALE_FORMATS;
  }
  if (schema.matrix.zero_scale_fallback) {
    flags |= LOOM_AMDGPU_MATRIX_CONTRACT_FLAG_ZERO_SCALE_FALLBACK;
  }
  return flags;
}
