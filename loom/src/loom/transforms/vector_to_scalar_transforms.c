// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <math.h>

#include "loom/error/error_defs.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/params.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/vector_to_scalar_internal.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Numeric transform descriptor decoding
//===----------------------------------------------------------------------===//

typedef enum loom_vector_to_scalar_transform_family_e {
  LOOM_VECTOR_TO_SCALAR_TRANSFORM_UNKNOWN = 0,
  LOOM_VECTOR_TO_SCALAR_TRANSFORM_HADAMARD = 1,
  LOOM_VECTOR_TO_SCALAR_TRANSFORM_HADAMARD_SIGN = 2,
  LOOM_VECTOR_TO_SCALAR_TRANSFORM_SIGN_PERMUTE_HADAMARD = 3,
  LOOM_VECTOR_TO_SCALAR_TRANSFORM_JL_DENSE = 4,
} loom_vector_to_scalar_transform_family_t;

typedef enum loom_vector_to_scalar_transform_normalization_e {
  LOOM_VECTOR_TO_SCALAR_TRANSFORM_NORMALIZATION_NONE = 0,
  LOOM_VECTOR_TO_SCALAR_TRANSFORM_NORMALIZATION_ORTHONORMAL = 1,
} loom_vector_to_scalar_transform_normalization_t;

typedef struct loom_vector_to_scalar_transform_descriptor_t {
  // Static numeric_transform family selected by the encoding descriptor.
  loom_vector_to_scalar_transform_family_t family;
  // Static normalization convention requested by the transform descriptor.
  loom_vector_to_scalar_transform_normalization_t normalization;
  // Optional dynamic i1 vector carrying per-lane negative-sign bits.
  loom_value_id_t signs;
  // Optional dynamic integer/index vector carrying per-lane permutation slots.
  loom_value_id_t permutation;
  // Optional dynamic floating-point matrix used by jl_dense transforms.
  loom_value_id_t matrix;
  // Optional dynamic seed for deterministic sign/permutation generation.
  loom_value_id_t seed;
} loom_vector_to_scalar_transform_descriptor_t;

static iree_status_t loom_vector_to_scalar_emit_transform_unimplemented(
    loom_vector_to_scalar_state_t* state, iree_string_view_t reason) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(state->rewriter->module, state->op)),
      loom_param_string(IREE_SV("vector-to-scalar")),
      loom_param_string(reason),
  };
  loom_diagnostic_emission_t emission = {
      .op = state->op,
      .error = &loom_err_lowering_001,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(
      iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission));
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "vector-to-scalar cannot lower vector.transform: "
                          "%.*s",
                          (int)reason.size, reason.data);
}

static bool loom_vector_to_scalar_string_id_equal(const loom_module_t* module,
                                                  loom_string_id_t string_id,
                                                  iree_string_view_t expected) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[string_id], expected);
}

static const loom_named_attr_t* loom_vector_to_scalar_find_named_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* entry = &attrs.entries[i];
    if (loom_vector_to_scalar_string_id_equal(module, entry->name_id, name)) {
      return entry;
    }
  }
  return NULL;
}

static bool loom_vector_to_scalar_string_attr_value(
    const loom_module_t* module, loom_attribute_t attr,
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

static bool loom_vector_to_scalar_dynamic_param_value(
    const loom_encoding_define_param_view_t* params,
    const loom_named_attr_t* name_entry, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (!name_entry || name_entry->value.kind != LOOM_ATTR_I64) return false;
  int64_t ordinal = name_entry->value.i64;
  if (ordinal < 0 || ordinal >= params->dynamic_values.count) return false;
  *out_value = params->dynamic_values.values[ordinal];
  return true;
}

static loom_vector_to_scalar_transform_family_t
loom_vector_to_scalar_transform_family_from_name(iree_string_view_t name) {
  if (iree_string_view_equal(name, IREE_SV("hadamard"))) {
    return LOOM_VECTOR_TO_SCALAR_TRANSFORM_HADAMARD;
  }
  if (iree_string_view_equal(name, IREE_SV("hadamard_sign"))) {
    return LOOM_VECTOR_TO_SCALAR_TRANSFORM_HADAMARD_SIGN;
  }
  if (iree_string_view_equal(name, IREE_SV("sign_permute_hadamard"))) {
    return LOOM_VECTOR_TO_SCALAR_TRANSFORM_SIGN_PERMUTE_HADAMARD;
  }
  if (iree_string_view_equal(name, IREE_SV("jl_dense"))) {
    return LOOM_VECTOR_TO_SCALAR_TRANSFORM_JL_DENSE;
  }
  return LOOM_VECTOR_TO_SCALAR_TRANSFORM_UNKNOWN;
}

static iree_status_t loom_vector_to_scalar_read_transform_descriptor(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_transform_descriptor_t* out_descriptor) {
  *out_descriptor = (loom_vector_to_scalar_transform_descriptor_t){
      .signs = LOOM_VALUE_ID_INVALID,
      .permutation = LOOM_VALUE_ID_INVALID,
      .matrix = LOOM_VALUE_ID_INVALID,
      .seed = LOOM_VALUE_ID_INVALID,
  };

  loom_value_id_t transform_value = loom_vector_transform_transform(state->op);
  if (transform_value == LOOM_VALUE_ID_INVALID ||
      transform_value >= state->rewriter->module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.transform operand id is out of range");
  }
  const loom_value_t* transform =
      loom_module_value(state->rewriter->module, transform_value);
  if (loom_value_is_block_arg(transform)) {
    return loom_vector_to_scalar_emit_transform_unimplemented(
        state, IREE_SV("transform descriptor is not locally defined"));
  }

  const loom_op_t* define_op = loom_value_def_op(transform);
  if (!define_op || !loom_encoding_define_isa(define_op)) {
    return loom_vector_to_scalar_emit_transform_unimplemented(
        state, IREE_SV("transform descriptor is not an encoding.define"));
  }

  loom_encoding_define_param_view_t params =
      loom_encoding_define_param_view(state->rewriter->module, define_op);
  if (!params.spec || !loom_vector_to_scalar_string_id_equal(
                          state->rewriter->module, params.spec->name_id,
                          IREE_SV("numeric_transform"))) {
    return loom_vector_to_scalar_emit_transform_unimplemented(
        state, IREE_SV("transform descriptor is not #numeric_transform"));
  }

  const loom_named_attr_t* family_param = loom_vector_to_scalar_find_named_attr(
      state->rewriter->module, params.static_attrs, IREE_SV("family"));
  iree_string_view_t family_name = iree_string_view_empty();
  if (!family_param ||
      !loom_vector_to_scalar_string_attr_value(
          state->rewriter->module, family_param->value, &family_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "vector.transform numeric_transform family must be a static string");
  }
  out_descriptor->family =
      loom_vector_to_scalar_transform_family_from_name(family_name);
  if (out_descriptor->family == LOOM_VECTOR_TO_SCALAR_TRANSFORM_UNKNOWN) {
    return loom_vector_to_scalar_emit_transform_unimplemented(
        state, IREE_SV("unknown numeric_transform family"));
  }

  out_descriptor->normalization =
      LOOM_VECTOR_TO_SCALAR_TRANSFORM_NORMALIZATION_NONE;
  const loom_named_attr_t* normalization_param =
      loom_vector_to_scalar_find_named_attr(state->rewriter->module,
                                            params.static_attrs,
                                            IREE_SV("normalization"));
  if (normalization_param) {
    iree_string_view_t normalization = iree_string_view_empty();
    if (!loom_vector_to_scalar_string_attr_value(state->rewriter->module,
                                                 normalization_param->value,
                                                 &normalization)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "numeric_transform normalization must be a "
                              "static string");
    }
    if (iree_string_view_equal(normalization, IREE_SV("orthonormal"))) {
      out_descriptor->normalization =
          LOOM_VECTOR_TO_SCALAR_TRANSFORM_NORMALIZATION_ORTHONORMAL;
    } else if (!iree_string_view_equal(normalization, IREE_SV("none"))) {
      return loom_vector_to_scalar_emit_transform_unimplemented(
          state, IREE_SV("unknown numeric_transform normalization"));
    }
  }

  const loom_named_attr_t* signs_param = loom_vector_to_scalar_find_named_attr(
      state->rewriter->module, params.dynamic_names, IREE_SV("signs"));
  if (signs_param && !loom_vector_to_scalar_dynamic_param_value(
                         &params, signs_param, &out_descriptor->signs)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "malformed encoding.define operand dictionary for parameter 'signs'");
  }

  const loom_named_attr_t* permutation_param =
      loom_vector_to_scalar_find_named_attr(state->rewriter->module,
                                            params.dynamic_names,
                                            IREE_SV("permutation"));
  if (permutation_param &&
      !loom_vector_to_scalar_dynamic_param_value(
          &params, permutation_param, &out_descriptor->permutation)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "malformed encoding.define operand dictionary for "
                            "parameter 'permutation'");
  }

  const loom_named_attr_t* matrix_param = loom_vector_to_scalar_find_named_attr(
      state->rewriter->module, params.dynamic_names, IREE_SV("matrix"));
  if (matrix_param && !loom_vector_to_scalar_dynamic_param_value(
                          &params, matrix_param, &out_descriptor->matrix)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "malformed encoding.define operand dictionary for parameter 'matrix'");
  }

  const loom_named_attr_t* seed_param = loom_vector_to_scalar_find_named_attr(
      state->rewriter->module, params.dynamic_names, IREE_SV("seed"));
  if (seed_param && !loom_vector_to_scalar_dynamic_param_value(
                        &params, seed_param, &out_descriptor->seed)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "malformed encoding.define operand dictionary for parameter 'seed'");
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Numeric transform lane programs
//===----------------------------------------------------------------------===//

static bool loom_vector_to_scalar_transform_uses_signs(
    loom_vector_to_scalar_transform_family_t family) {
  return family == LOOM_VECTOR_TO_SCALAR_TRANSFORM_HADAMARD_SIGN ||
         family == LOOM_VECTOR_TO_SCALAR_TRANSFORM_SIGN_PERMUTE_HADAMARD;
}

static bool loom_vector_to_scalar_transform_has_signs(
    const loom_vector_to_scalar_transform_descriptor_t* descriptor) {
  return descriptor->signs != LOOM_VALUE_ID_INVALID;
}

static bool loom_vector_to_scalar_transform_has_permutation(
    const loom_vector_to_scalar_transform_descriptor_t* descriptor) {
  return descriptor->permutation != LOOM_VALUE_ID_INVALID;
}

static bool loom_vector_to_scalar_transform_has_matrix(
    const loom_vector_to_scalar_transform_descriptor_t* descriptor) {
  return descriptor->matrix != LOOM_VALUE_ID_INVALID;
}

static bool loom_vector_to_scalar_transform_has_seed(
    const loom_vector_to_scalar_transform_descriptor_t* descriptor) {
  return descriptor->seed != LOOM_VALUE_ID_INVALID;
}

static iree_status_t loom_vector_to_scalar_build_negf_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t input,
    loom_value_id_t* out_result) {
  return loom_vector_to_scalar_build_generic_lane_op(
      state, LOOM_OP_SCALAR_NEGF, 0, &input, 1, NULL, 0,
      state->result_scalar_type, out_result);
}

static iree_status_t loom_vector_to_scalar_apply_sign_bit(
    loom_vector_to_scalar_state_t* state, loom_value_id_t sign,
    loom_value_id_t input, loom_value_id_t* out_result) {
  loom_value_id_t negated = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_negf_lane(state, input, &negated));
  return loom_vector_to_scalar_build_select_lane(state, sign, negated, input,
                                                 out_result);
}

static iree_status_t loom_vector_to_scalar_build_addf_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result) {
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_ADDF, lhs, rhs, state->result_scalar_type,
      out_result);
}

static iree_status_t loom_vector_to_scalar_build_mulf_lane(
    loom_vector_to_scalar_state_t* state, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_value_id_t* out_result) {
  return loom_vector_to_scalar_build_scalar_binary(
      state, LOOM_OP_SCALAR_MULF, lhs, rhs, state->result_scalar_type,
      out_result);
}

static iree_status_t loom_vector_to_scalar_apply_orthonormal_scale(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_transform_descriptor_t* descriptor,
    int64_t input_extent, loom_value_id_t input, loom_value_id_t* out_result) {
  if (descriptor->normalization !=
      LOOM_VECTOR_TO_SCALAR_TRANSFORM_NORMALIZATION_ORTHONORMAL) {
    *out_result = input;
    return iree_ok_status();
  }

  double scale = 1.0 / sqrt((double)input_extent);
  loom_value_id_t scale_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_attr_constant(
      &state->rewriter->builder, state->result_scalar_type, state->location,
      loom_attr_f64(scale), &scale_value));
  return loom_vector_to_scalar_build_mulf_lane(state, input, scale_value,
                                               out_result);
}

static int64_t loom_vector_to_scalar_static_last_axis_extent(loom_type_t type) {
  uint8_t rank = loom_type_rank(type);
  if (rank == 0 || loom_type_dim_is_dynamic_at(type, (uint8_t)(rank - 1))) {
    return -1;
  }
  return (int64_t)loom_type_dim_static_size_at(type, (uint8_t)(rank - 1));
}

static iree_status_t loom_vector_to_scalar_transform_source_indices_from_term(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t result_indices,
    loom_vector_to_scalar_index_term_t last_index,
    loom_vector_to_scalar_index_list_t* out_indices) {
  uint8_t rank = result_indices.rank;
  if (rank == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vector.transform requires rank >= 1");
  }
  loom_vector_to_scalar_index_term_t* source_terms = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->rewriter->arena, rank, sizeof(loom_vector_to_scalar_index_term_t),
      (void**)&source_terms));
  for (uint8_t axis = 0; axis < rank - 1; ++axis) {
    source_terms[axis] =
        loom_vector_to_scalar_lane_term(state, result_indices, axis);
  }
  source_terms[rank - 1] = last_index;
  return loom_vector_to_scalar_terms_to_index_list(state, source_terms, rank,
                                                   out_indices);
}

static iree_status_t loom_vector_to_scalar_cast_value_to_index(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_value_id_t* out_index) {
  loom_type_t value_type =
      loom_module_value_type(state->rewriter->module, value);
  if (loom_type_element_type(value_type) == LOOM_SCALAR_TYPE_INDEX) {
    *out_index = value;
    return iree_ok_status();
  }
  loom_op_t* cast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      &state->rewriter->builder, value, value_type,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), state->location, &cast_op));
  *out_index = loom_index_cast_result(cast_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_permutation_lane_term(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_transform_descriptor_t* descriptor,
    loom_vector_to_scalar_index_list_t indices,
    loom_vector_to_scalar_index_term_t* out_term) {
  loom_value_id_t permutation_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
      state, descriptor->permutation, indices, &permutation_lane));
  loom_value_id_t permutation_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_cast_value_to_index(
      state, permutation_lane, &permutation_index));
  *out_term = loom_vector_to_scalar_value_term(state, permutation_index);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_transform_accumulate(
    loom_vector_to_scalar_state_t* state, loom_value_id_t term,
    loom_value_id_t* accumulator) {
  if (*accumulator == LOOM_VALUE_ID_INVALID) {
    *accumulator = term;
    return iree_ok_status();
  }
  return loom_vector_to_scalar_build_addf_lane(state, *accumulator, term,
                                               accumulator);
}

static iree_status_t loom_vector_to_scalar_build_scalar_i64_constant(
    loom_vector_to_scalar_state_t* state, int64_t value,
    loom_value_id_t* out_value) {
  return loom_vector_to_scalar_build_scalar_constant(
      &state->rewriter->builder, loom_type_scalar(LOOM_SCALAR_TYPE_I64),
      state->location, value, out_value);
}

static iree_status_t loom_vector_to_scalar_build_scalar_i64_binary(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t* out_result) {
  return loom_vector_to_scalar_build_scalar_binary(
      state, kind, lhs, rhs, loom_type_scalar(LOOM_SCALAR_TYPE_I64),
      out_result);
}

static iree_status_t loom_vector_to_scalar_build_scalar_i64_binary_constant(
    loom_vector_to_scalar_state_t* state, loom_op_kind_t kind,
    loom_value_id_t lhs, int64_t rhs, loom_value_id_t* out_result) {
  loom_value_id_t rhs_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_scalar_i64_constant(state, rhs, &rhs_value));
  return loom_vector_to_scalar_build_scalar_i64_binary(state, kind, lhs,
                                                       rhs_value, out_result);
}

static iree_status_t loom_vector_to_scalar_build_i64_nonzero(
    loom_vector_to_scalar_state_t* state, loom_value_id_t value,
    loom_value_id_t* out_condition) {
  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_build_scalar_i64_constant(state, 0, &zero));
  loom_op_t* cmp_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      &state->rewriter->builder, LOOM_SCALAR_CMPI_PREDICATE_NE, value, zero,
      loom_type_scalar(LOOM_SCALAR_TYPE_I64),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), state->location, &cmp_op));
  *out_condition = loom_scalar_cmpi_result(cmp_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_to_scalar_build_hadamard_dynamic_phase(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t output_index, int64_t input_index,
    loom_value_id_t* out_condition) {
  *out_condition = LOOM_VALUE_ID_INVALID;
  if (input_index == 0) return iree_ok_status();

  loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_value_id_t output_i64 = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_index_term_as_scalar(
      state, output_index, i64_type, &output_i64));

  loom_value_id_t parity = LOOM_VALUE_ID_INVALID;
  for (uint8_t bit = 0; bit < 63; ++bit) {
    int64_t bit_mask = (int64_t)1 << bit;
    if ((input_index & bit_mask) == 0) continue;

    loom_value_id_t shifted = output_i64;
    if (bit != 0) {
      IREE_RETURN_IF_ERROR(
          loom_vector_to_scalar_build_scalar_i64_binary_constant(
              state, LOOM_OP_SCALAR_SHRUI, output_i64, bit, &shifted));
    }
    loom_value_id_t low_bit = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
        state, LOOM_OP_SCALAR_ANDI, shifted, 1, &low_bit));
    if (parity == LOOM_VALUE_ID_INVALID) {
      parity = low_bit;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary(
        state, LOOM_OP_SCALAR_XORI, parity, low_bit, &parity));
  }
  return loom_vector_to_scalar_build_i64_nonzero(state, parity, out_condition);
}

static iree_status_t loom_vector_to_scalar_apply_hadamard_phase(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t output_index, int64_t input_index,
    loom_value_id_t input, loom_value_id_t* out_result) {
  if (!output_index.is_dynamic) {
    if (loom_count_ones_u64_width(
            (uint64_t)(output_index.static_value & input_index), 64) &
        1) {
      return loom_vector_to_scalar_build_negf_lane(state, input, out_result);
    }
    *out_result = input;
    return iree_ok_status();
  }

  loom_value_id_t negate_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_hadamard_dynamic_phase(
      state, output_index, input_index, &negate_condition));
  if (negate_condition == LOOM_VALUE_ID_INVALID) {
    *out_result = input;
    return iree_ok_status();
  }
  return loom_vector_to_scalar_apply_sign_bit(state, negate_condition, input,
                                              out_result);
}

// Seeded signs use the low bit of the SplitMix64 finalizer for
// seed + input lane. This keeps the reference path deterministic and
// expressible entirely in scalar integer IR.
static iree_status_t loom_vector_to_scalar_build_seed_sign_bit(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_transform_descriptor_t* descriptor,
    int64_t input_index, loom_value_id_t* out_sign) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t i64_type = loom_type_scalar(LOOM_SCALAR_TYPE_I64);
  loom_op_t* seed_cast_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      &state->rewriter->builder, descriptor->seed, index_type, i64_type,
      state->location, &seed_cast_op));
  loom_value_id_t mixed = loom_index_cast_result(seed_cast_op);

  if (input_index != 0) {
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
        state, LOOM_OP_SCALAR_ADDI, mixed, input_index, &mixed));
  }
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_ADDI, mixed, -7046029254386353131LL, &mixed));

  loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_SHRUI, mixed, 30, &shifted));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary(
      state, LOOM_OP_SCALAR_XORI, mixed, shifted, &mixed));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_MULI, mixed, -4658895280553007687LL, &mixed));

  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_SHRUI, mixed, 27, &shifted));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary(
      state, LOOM_OP_SCALAR_XORI, mixed, shifted, &mixed));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_MULI, mixed, -7723592293110705685LL, &mixed));

  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_SHRUI, mixed, 31, &shifted));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary(
      state, LOOM_OP_SCALAR_XORI, mixed, shifted, &mixed));
  IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_scalar_i64_binary_constant(
      state, LOOM_OP_SCALAR_ANDI, mixed, 1, &mixed));

  return loom_vector_to_scalar_build_i64_nonzero(state, mixed, out_sign);
}

static iree_status_t loom_vector_to_scalar_transform_hadamard_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_transform_descriptor_t* descriptor,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_type_t source_type = loom_module_value_type(
      state->rewriter->module, loom_vector_transform_source(state->op));
  int64_t input_extent =
      loom_vector_to_scalar_static_last_axis_extent(source_type);
  if (input_extent <= 0) {
    return loom_vector_to_scalar_emit_transform_unimplemented(
        state, IREE_SV("dynamic transform extents require specialization"));
  }

  if (descriptor->family == LOOM_VECTOR_TO_SCALAR_TRANSFORM_HADAMARD &&
      (loom_vector_to_scalar_transform_has_signs(descriptor) ||
       loom_vector_to_scalar_transform_has_permutation(descriptor) ||
       loom_vector_to_scalar_transform_has_matrix(descriptor) ||
       loom_vector_to_scalar_transform_has_seed(descriptor))) {
    return loom_vector_to_scalar_emit_transform_unimplemented(
        state, IREE_SV("hadamard does not consume dynamic transform "
                       "parameters"));
  }

  if (descriptor->family ==
      LOOM_VECTOR_TO_SCALAR_TRANSFORM_SIGN_PERMUTE_HADAMARD) {
    if (!loom_vector_to_scalar_transform_has_signs(descriptor) ||
        !loom_vector_to_scalar_transform_has_permutation(descriptor) ||
        loom_vector_to_scalar_transform_has_matrix(descriptor) ||
        loom_vector_to_scalar_transform_has_seed(descriptor)) {
      return loom_vector_to_scalar_emit_transform_unimplemented(
          state, IREE_SV("sign_permute_hadamard requires explicit signs and "
                         "permutation only"));
    }
  }

  if (descriptor->family == LOOM_VECTOR_TO_SCALAR_TRANSFORM_HADAMARD_SIGN) {
    bool has_signs = loom_vector_to_scalar_transform_has_signs(descriptor);
    bool has_seed = loom_vector_to_scalar_transform_has_seed(descriptor);
    if (has_signs == has_seed) {
      return loom_vector_to_scalar_emit_transform_unimplemented(
          state, IREE_SV("hadamard_sign requires exactly one sign source"));
    }
    if (loom_vector_to_scalar_transform_has_permutation(descriptor) ||
        loom_vector_to_scalar_transform_has_matrix(descriptor)) {
      return loom_vector_to_scalar_emit_transform_unimplemented(
          state, IREE_SV("hadamard_sign does not consume permutation or matrix "
                         "parameters"));
    }
  }

  uint8_t last_axis = (uint8_t)(indices.rank - 1);
  loom_vector_to_scalar_index_term_t output_index =
      loom_vector_to_scalar_lane_term(state, indices, last_axis);
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  for (int64_t input_index = 0; input_index < input_extent; ++input_index) {
    loom_vector_to_scalar_index_list_t transform_indices = {0};
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_transform_source_indices_from_term(
            state, indices, loom_vector_to_scalar_static_term(input_index),
            &transform_indices));

    loom_vector_to_scalar_index_list_t source_indices = transform_indices;
    if (descriptor->family ==
        LOOM_VECTOR_TO_SCALAR_TRANSFORM_SIGN_PERMUTE_HADAMARD) {
      loom_vector_to_scalar_index_term_t source_last_index = {0};
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_permutation_lane_term(
          state, descriptor, transform_indices, &source_last_index));
      IREE_RETURN_IF_ERROR(
          loom_vector_to_scalar_transform_source_indices_from_term(
              state, indices, source_last_index, &source_indices));
    }

    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, loom_vector_transform_source(state->op), source_indices,
        &source_lane));

    if (loom_vector_to_scalar_transform_uses_signs(descriptor->family)) {
      loom_value_id_t sign_lane = LOOM_VALUE_ID_INVALID;
      if (loom_vector_to_scalar_transform_has_signs(descriptor)) {
        IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
            state, descriptor->signs, source_indices, &sign_lane));
      } else {
        IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_seed_sign_bit(
            state, descriptor, input_index, &sign_lane));
      }
      IREE_RETURN_IF_ERROR(loom_vector_to_scalar_apply_sign_bit(
          state, sign_lane, source_lane, &source_lane));
    }

    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_apply_hadamard_phase(
        state, output_index, input_index, source_lane, &source_lane));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_transform_accumulate(
        state, source_lane, &accumulator));
  }

  return loom_vector_to_scalar_apply_orthonormal_scale(
      state, descriptor, input_extent, accumulator, out_lane);
}

static iree_status_t loom_vector_to_scalar_transform_matrix_indices_from_terms(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_term_t output_index,
    loom_vector_to_scalar_index_term_t input_index,
    loom_vector_to_scalar_index_list_t* out_indices) {
  loom_vector_to_scalar_index_term_t terms[2] = {
      output_index,
      input_index,
  };
  return loom_vector_to_scalar_terms_to_index_list(
      state, terms, IREE_ARRAYSIZE(terms), out_indices);
}

static iree_status_t loom_vector_to_scalar_transform_jl_dense_lane(
    loom_vector_to_scalar_state_t* state,
    const loom_vector_to_scalar_transform_descriptor_t* descriptor,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  if (descriptor->matrix == LOOM_VALUE_ID_INVALID) {
    return loom_vector_to_scalar_emit_transform_unimplemented(
        state, IREE_SV("jl_dense requires an explicit matrix parameter"));
  }
  if (descriptor->normalization !=
      LOOM_VECTOR_TO_SCALAR_TRANSFORM_NORMALIZATION_NONE) {
    return loom_vector_to_scalar_emit_transform_unimplemented(
        state, IREE_SV("jl_dense normalization is carried by the explicit "
                       "matrix"));
  }
  if (loom_vector_to_scalar_transform_has_signs(descriptor) ||
      loom_vector_to_scalar_transform_has_permutation(descriptor) ||
      loom_vector_to_scalar_transform_has_seed(descriptor)) {
    return loom_vector_to_scalar_emit_transform_unimplemented(
        state, IREE_SV("jl_dense does not consume signs, permutation, or seed "
                       "parameters"));
  }

  loom_type_t source_type = loom_module_value_type(
      state->rewriter->module, loom_vector_transform_source(state->op));
  int64_t input_extent =
      loom_vector_to_scalar_static_last_axis_extent(source_type);
  if (input_extent <= 0) {
    return loom_vector_to_scalar_emit_transform_unimplemented(
        state, IREE_SV("dynamic transform extents require specialization"));
  }

  uint8_t last_axis = (uint8_t)(indices.rank - 1);
  loom_vector_to_scalar_index_term_t output_index =
      loom_vector_to_scalar_lane_term(state, indices, last_axis);
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  for (int64_t input_index = 0; input_index < input_extent; ++input_index) {
    loom_vector_to_scalar_index_list_t source_indices = {0};
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_transform_source_indices_from_term(
            state, indices, loom_vector_to_scalar_static_term(input_index),
            &source_indices));
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, loom_vector_transform_source(state->op), source_indices,
        &source_lane));

    loom_vector_to_scalar_index_list_t matrix_indices = {0};
    IREE_RETURN_IF_ERROR(
        loom_vector_to_scalar_transform_matrix_indices_from_terms(
            state, output_index, loom_vector_to_scalar_static_term(input_index),
            &matrix_indices));
    loom_value_id_t matrix_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_materialize_lane(
        state, descriptor->matrix, matrix_indices, &matrix_lane));

    loom_value_id_t product = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_build_mulf_lane(
        state, matrix_lane, source_lane, &product));
    IREE_RETURN_IF_ERROR(loom_vector_to_scalar_transform_accumulate(
        state, product, &accumulator));
  }

  *out_lane = accumulator;
  return iree_ok_status();
}

iree_status_t loom_vector_to_scalar_build_transform_lane(
    loom_vector_to_scalar_state_t* state,
    loom_vector_to_scalar_index_list_t indices, loom_value_id_t* out_lane) {
  loom_vector_to_scalar_transform_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_to_scalar_read_transform_descriptor(state, &descriptor));
  switch (descriptor.family) {
    case LOOM_VECTOR_TO_SCALAR_TRANSFORM_HADAMARD:
    case LOOM_VECTOR_TO_SCALAR_TRANSFORM_HADAMARD_SIGN:
    case LOOM_VECTOR_TO_SCALAR_TRANSFORM_SIGN_PERMUTE_HADAMARD:
      return loom_vector_to_scalar_transform_hadamard_lane(state, &descriptor,
                                                           indices, out_lane);
    case LOOM_VECTOR_TO_SCALAR_TRANSFORM_JL_DENSE:
      return loom_vector_to_scalar_transform_jl_dense_lane(state, &descriptor,
                                                           indices, out_lane);
    default:
      return loom_vector_to_scalar_emit_transform_unimplemented(
          state, IREE_SV("unknown numeric_transform family"));
  }
}
