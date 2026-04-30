// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/encoding_auxiliary.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// Key vocabulary
//===----------------------------------------------------------------------===//

static const iree_string_view_t loom_vector_encoding_auxiliary_key_names
    [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_] = {
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE] = IREE_SVL("scale"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SECONDARY_SCALE] =
            IREE_SVL("secondary_scale"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE2] = IREE_SVL("scale2"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE3] = IREE_SVL("scale3"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE4] = IREE_SVL("scale4"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE5] = IREE_SVL("scale5"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE6] = IREE_SVL("scale6"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE7] = IREE_SVL("scale7"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_ZERO_POINT] =
            IREE_SVL("zero_point"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_MINIMUM] = IREE_SVL("minimum"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIAS] = IREE_SVL("bias"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SUM_CORRECTION] =
            IREE_SVL("sum_correction"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_CODEBOOK] = IREE_SVL("codebook"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SPARSITY] = IREE_SVL("sparsity"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_METADATA] = IREE_SVL("metadata"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_INDICES] = IREE_SVL("indices"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_OFFSETS] = IREE_SVL("offsets"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_MASK] = IREE_SVL("mask"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SIGNS] = IREE_SVL("signs"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_RESIDUAL] = IREE_SVL("residual"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_AMAX] = IREE_SVL("amax"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_THRESHOLDS] =
            IREE_SVL("thresholds"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_CENTROIDS] = IREE_SVL("centroids"),
        [LOOM_VECTOR_ENCODING_AUXILIARY_KEY_OUTLIERS] = IREE_SVL("outliers"),
};

iree_string_view_t loom_vector_encoding_auxiliary_key_name(
    loom_vector_encoding_auxiliary_key_t key) {
  if (key >= LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_) {
    return iree_string_view_empty();
  }
  return loom_vector_encoding_auxiliary_key_names[key];
}

bool loom_vector_encoding_auxiliary_key_lookup(
    iree_string_view_t name, loom_vector_encoding_auxiliary_key_t* out_key) {
  for (uint8_t i = 0; i < LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_; ++i) {
    if (!iree_string_view_equal(name,
                                loom_vector_encoding_auxiliary_key_names[i])) {
      continue;
    }
    *out_key = (loom_vector_encoding_auxiliary_key_t)i;
    return true;
  }
  return false;
}

bool loom_vector_encoding_auxiliary_scale_key(
    uint16_t index, loom_vector_encoding_auxiliary_key_t* out_key) {
  static const loom_vector_encoding_auxiliary_key_t scale_keys[] = {
      LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE,
      LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SECONDARY_SCALE,
      LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE2,
      LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE3,
      LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE4,
      LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE5,
      LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE6,
      LOOM_VECTOR_ENCODING_AUXILIARY_KEY_SCALE7,
  };
  if (index >= IREE_ARRAYSIZE(scale_keys)) {
    return false;
  }
  *out_key = scale_keys[index];
  return true;
}

//===----------------------------------------------------------------------===//
// View resolution
//===----------------------------------------------------------------------===//

void loom_vector_encoding_auxiliary_view_initialize(
    loom_vector_encoding_auxiliary_view_t* out_view) {
  memset(out_view, 0, sizeof(*out_view));
  for (uint8_t i = 0; i < LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_; ++i) {
    out_view->values[i] = LOOM_VALUE_ID_INVALID;
  }
}

bool loom_vector_encoding_auxiliary_view_resolve(
    const loom_module_t* module, loom_value_slice_t auxiliary_values,
    loom_named_attr_slice_t auxiliary_names,
    loom_vector_encoding_auxiliary_view_t* out_view,
    iree_string_view_t* out_unknown_key) {
  loom_vector_encoding_auxiliary_view_initialize(out_view);
  if (out_unknown_key) {
    *out_unknown_key = iree_string_view_empty();
  }
  for (iree_host_size_t i = 0; i < auxiliary_names.count; ++i) {
    const loom_named_attr_t* entry = &auxiliary_names.entries[i];
    if (entry->name_id == LOOM_STRING_ID_INVALID ||
        entry->name_id >= module->strings.count) {
      continue;
    }
    iree_string_view_t key_name = module->strings.entries[entry->name_id];
    loom_vector_encoding_auxiliary_key_t key = 0;
    if (!loom_vector_encoding_auxiliary_key_lookup(key_name, &key)) {
      if (out_unknown_key) {
        *out_unknown_key = key_name;
      }
      return false;
    }
    int64_t ordinal =
        entry->value.kind == LOOM_ATTR_I64 ? entry->value.i64 : -1;
    iree_host_size_t ordinal_index = (iree_host_size_t)ordinal;
    if (ordinal >= 0 && ordinal_index < auxiliary_values.count) {
      out_view->values[key] = auxiliary_values.values[ordinal_index];
    }
    out_view->present_keys |= loom_vector_encoding_auxiliary_key_flag(key);
  }
  return true;
}

bool loom_vector_encoding_auxiliary_required_keys_from_schema(
    loom_value_fact_encoded_operand_schema_t schema,
    loom_vector_encoding_auxiliary_key_flags_t* out_required_keys,
    uint16_t* out_unsupported_scale_index) {
  loom_vector_encoding_auxiliary_key_flags_t required_keys = 0;
  if (out_unsupported_scale_index) {
    *out_unsupported_scale_index = UINT16_MAX;
  }

  for (uint16_t i = 0; i < schema.scale_operand_count; ++i) {
    loom_vector_encoding_auxiliary_key_t key = 0;
    if (!loom_vector_encoding_auxiliary_scale_key(i, &key)) {
      if (out_unsupported_scale_index) {
        *out_unsupported_scale_index = i;
      }
      *out_required_keys = required_keys;
      return false;
    }
    required_keys |= loom_vector_encoding_auxiliary_key_flag(key);
  }

  if (iree_any_bit_set(schema.scale_topology,
                       LOOM_VALUE_FACT_SCALE_TOPOLOGY_RUNTIME_AMAX_DERIVED)) {
    required_keys |= LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_AMAX;
  }
  if (iree_any_bit_set(schema.affine_policy,
                       LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_ZERO_POINT)) {
    required_keys |= LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_ZERO_POINT;
  }
  if (iree_any_bit_set(schema.affine_policy,
                       LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN)) {
    required_keys |= LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_MINIMUM;
  }
  if (iree_any_bit_set(schema.affine_policy,
                       LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_BIAS)) {
    required_keys |= LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_BIAS;
  }
  if (iree_any_bit_set(schema.affine_policy,
                       LOOM_VALUE_FACT_AFFINE_POLICY_SUM_CORRECTION)) {
    required_keys |= LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SUM_CORRECTION;
  }
  if (iree_any_bit_set(schema.codebook_policy,
                       LOOM_VALUE_FACT_CODEBOOK_POLICY_DYNAMIC_TABLE_OPERAND)) {
    required_keys |= LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_CODEBOOK;
  }
  if (schema.sparsity_policy != 0) {
    required_keys |= LOOM_VECTOR_ENCODING_AUXILIARY_KEY_BIT_SPARSITY;
  }

  *out_required_keys = required_keys;
  return true;
}
