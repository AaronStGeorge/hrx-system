// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lowering for target-neutral Loom vector ops into structured LLVMIR ops.

#include "loom/ops/vector/ops.h"
#include "loom/target/emit/llvmir/lower/internal.h"

static iree_status_t loom_llvmir_lowering_vector_lane_count(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_type_t type, uint32_t* out_lane_count) {
  if (!loom_type_is_vector(type) || !loom_type_is_all_static(type) ||
      loom_type_rank(type) != 1) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "only static one-dimensional vectors lower today");
  }
  uint64_t lane_count = 0;
  if (!loom_type_static_element_count(type, &lane_count) || lane_count == 0 ||
      lane_count > UINT32_MAX) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "vector lane count is not representable in LLVMIR");
  }
  *out_lane_count = (uint32_t)lane_count;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_i32_constant(
    loom_llvmir_lowering_state_t* state, uint32_t value,
    loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(state->target_module, 32, &i32_type));
  return loom_llvmir_module_add_integer_constant(state->target_module, i32_type,
                                                 value, out_value_id);
}

static iree_status_t loom_llvmir_lowering_i32_vector_constant(
    loom_llvmir_lowering_state_t* state, const uint64_t* values,
    iree_host_size_t value_count, loom_llvmir_value_id_t* out_value_id) {
  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t vector_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(state->target_module, 32, &i32_type));
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_vector_type(
      state->target_module, (uint32_t)value_count, i32_type, &vector_type));
  return loom_llvmir_module_add_integer_vector_constant(
      state->target_module, vector_type, values, value_count, out_value_id);
}

static iree_status_t loom_llvmir_lowering_repeated_i32_vector_constant(
    loom_llvmir_lowering_state_t* state, uint32_t lane_count, uint32_t value,
    loom_llvmir_value_id_t* out_value_id) {
  uint64_t* values = NULL;
  iree_status_t status = iree_allocator_malloc(
      state->allocator, (iree_host_size_t)lane_count * sizeof(*values),
      (void**)&values);
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < lane_count; ++i) values[i] = value;
    status = loom_llvmir_lowering_i32_vector_constant(state, values, lane_count,
                                                      out_value_id);
  }
  if (values) iree_allocator_free(state->allocator, values);
  return status;
}

static iree_status_t loom_llvmir_lowering_add_poison_value(
    loom_llvmir_lowering_state_t* state, loom_type_t source_type,
    loom_llvmir_type_id_t* out_type_id, loom_llvmir_value_id_t* out_value_id) {
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lower_type(state, source_type, out_type_id));
  return loom_llvmir_module_add_poison_constant(state->target_module,
                                                *out_type_id, out_value_id);
}

static iree_status_t loom_llvmir_lowering_vector_lane_index(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    uint32_t lane_count, loom_llvmir_value_id_t* out_index) {
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "vector lane ops require one index term");
  }
  int64_t static_index = static_indices.i64_array[0];
  if (static_index != INT64_MIN) {
    if (dynamic_indices.count != 0) {
      return loom_llvmir_lowering_unsupported_op(
          state, op, "static vector lane index cannot also have operands");
    }
    if (static_index < 0 || static_index > UINT32_MAX) {
      return loom_llvmir_lowering_unsupported_op(
          state, op, "static vector lane index is not representable");
    }
    if ((uint32_t)static_index >= lane_count) {
      return loom_llvmir_lowering_unsupported_op(
          state, op, "static vector lane index is out of bounds");
    }
    return loom_llvmir_lowering_i32_constant(state, (uint32_t)static_index,
                                             out_index);
  }
  if (dynamic_indices.count != 1) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "dynamic vector lane index sentinel needs one operand");
  }
  return loom_llvmir_lowering_lookup_value(
      state, loom_value_slice_get(dynamic_indices, 0), out_index);
}

static iree_status_t loom_llvmir_lowering_shuffle_mask(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_attribute_t source_lanes, uint32_t input_lane_count,
    uint32_t result_lane_count, loom_llvmir_value_id_t* out_mask) {
  if (source_lanes.kind != LOOM_ATTR_I64_ARRAY ||
      source_lanes.count != result_lane_count) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "vector.shuffle needs one static source lane per result");
  }
  uint64_t* values = NULL;
  iree_status_t status = iree_allocator_malloc(
      state->allocator, (iree_host_size_t)result_lane_count * sizeof(*values),
      (void**)&values);
  for (uint32_t i = 0; i < result_lane_count && iree_status_is_ok(status);
       ++i) {
    int64_t source_lane = source_lanes.i64_array[i];
    if (source_lane < 0 || source_lane >= input_lane_count) {
      status = loom_llvmir_lowering_unsupported_op(
          state, op, "vector.shuffle source lane is out of bounds");
    } else {
      values[i] = (uint64_t)source_lane;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_lowering_i32_vector_constant(
        state, values, result_lane_count, out_mask);
  }
  if (values) iree_allocator_free(state->allocator, values);
  return status;
}

static iree_status_t loom_llvmir_lowering_splat_value(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op, loom_value_id_t result_id,
    loom_llvmir_value_id_t scalar, loom_type_t result_source_type,
    loom_llvmir_value_id_t* out_result) {
  uint32_t lane_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_lane_count(
      state, op, result_source_type, &lane_count));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_value_id_t poison = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_add_poison_value(
      state, result_source_type, &result_type, &poison));
  loom_llvmir_value_id_t zero_index = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_i32_constant(state, 0, &zero_index));

  loom_llvmir_value_id_t first_lane = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_insert_element(target_block,
                                       &(loom_llvmir_insert_element_desc_t){
                                           .result_type = result_type,
                                           .vector = poison,
                                           .element = scalar,
                                           .index = zero_index,
                                       },
                                       &first_lane));

  loom_llvmir_value_id_t mask = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_repeated_i32_vector_constant(
      state, lane_count, 0, &mask));
  return loom_llvmir_build_shuffle_vector(
      target_block,
      &(loom_llvmir_shuffle_vector_desc_t){
          .result_name = loom_llvmir_lowering_value_name(state, result_id),
          .result_type = result_type,
          .lhs = first_lane,
          .rhs = poison,
          .mask = mask,
      },
      out_result);
}

iree_status_t loom_llvmir_lowering_lower_vector_constant(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_attribute_t value_attr) {
  loom_value_id_t result_id = loom_vector_constant_result(op);
  loom_type_t result_source_type =
      loom_module_value_type(state->source_module, result_id);
  uint32_t lane_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_lane_count(
      state, op, result_source_type, &lane_count));
  loom_scalar_type_t element_type = loom_type_element_type(result_source_type);
  if (element_type != LOOM_SCALAR_TYPE_INDEX &&
      element_type != LOOM_SCALAR_TYPE_OFFSET &&
      !loom_scalar_type_is_integer(element_type)) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "vector.constant only supports integer-like vectors today");
  }
  uint64_t value = 0;
  if (value_attr.kind == LOOM_ATTR_BOOL) {
    value = value_attr.raw;
  } else if (value_attr.kind == LOOM_ATTR_I64) {
    value = (uint64_t)value_attr.i64;
  } else {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "integer vector constants require an i64 or bool attribute");
  }

  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lower_type(state, result_source_type, &result_type));
  uint64_t* values = NULL;
  iree_status_t status = iree_allocator_malloc(
      state->allocator, (iree_host_size_t)lane_count * sizeof(*values),
      (void**)&values);
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < lane_count; ++i) values[i] = value;
    loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
    status = loom_llvmir_module_add_integer_vector_constant(
        state->target_module, result_type, values, lane_count, &result);
    if (iree_status_is_ok(status)) {
      status = loom_llvmir_lowering_map_value(state, result_id, result);
    }
  }
  if (values) iree_allocator_free(state->allocator, values);
  return status;
}

iree_status_t loom_llvmir_lowering_lower_vector_poison(
    loom_llvmir_lowering_state_t* state, const loom_op_t* op) {
  loom_value_id_t result_id = loom_vector_poison_result(op);
  loom_type_t result_source_type =
      loom_module_value_type(state->source_module, result_id);
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_add_poison_value(
      state, result_source_type, &result_type, &result));
  return loom_llvmir_lowering_map_value(state, result_id, result);
}

iree_status_t loom_llvmir_lowering_lower_vector_splat(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t scalar = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_vector_splat_scalar(op), &scalar));
  loom_value_id_t result_id = loom_vector_splat_result(op);
  loom_type_t result_source_type =
      loom_module_value_type(state->source_module, result_id);
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_splat_value(
      state, target_block, op, result_id, scalar, result_source_type, &result));
  return loom_llvmir_lowering_map_value(state, result_id, result);
}

iree_status_t loom_llvmir_lowering_lower_vector_from_elements(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_value_id_t result_id = loom_vector_from_elements_result(op);
  loom_type_t result_source_type =
      loom_module_value_type(state->source_module, result_id);
  uint32_t lane_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_lane_count(
      state, op, result_source_type, &lane_count));
  loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  if (elements.count != lane_count) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "vector.from_elements operand count must match lanes");
  }

  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_value_id_t current = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_add_poison_value(
      state, result_source_type, &result_type, &current));
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_llvmir_value_id_t element = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
        state, loom_value_slice_get(elements, i), &element));
    loom_llvmir_value_id_t index = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_i32_constant(state, i, &index));
    loom_llvmir_value_id_t inserted = LOOM_LLVMIR_VALUE_ID_INVALID;
    iree_string_view_t result_name =
        i + 1 == lane_count ? loom_llvmir_lowering_value_name(state, result_id)
                            : iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        loom_llvmir_build_insert_element(target_block,
                                         &(loom_llvmir_insert_element_desc_t){
                                             .result_name = result_name,
                                             .result_type = result_type,
                                             .vector = current,
                                             .element = element,
                                             .index = index,
                                         },
                                         &inserted));
    current = inserted;
  }
  return loom_llvmir_lowering_map_value(state, result_id, current);
}

iree_status_t loom_llvmir_lowering_lower_vector_extract(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_value_id_t source_id = loom_vector_extract_source(op);
  loom_type_t source_type =
      loom_module_value_type(state->source_module, source_id);
  uint32_t lane_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_lane_count(
      state, op, source_type, &lane_count));
  loom_value_id_t result_id = loom_vector_extract_result(op);
  loom_type_t result_source_type =
      loom_module_value_type(state->source_module, result_id);
  if (!loom_type_is_scalar(result_source_type)) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "vector.extract only supports scalar lane extraction");
  }

  loom_llvmir_value_id_t source = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_value(state, source_id, &source));
  loom_llvmir_value_id_t index = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_lane_index(
      state, op, loom_vector_extract_static_indices(op),
      loom_vector_extract_indices(op), lane_count, &index));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lower_type(state, result_source_type, &result_type));

  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_extract_element(
      target_block,
      &(loom_llvmir_extract_element_desc_t){
          .result_name = loom_llvmir_lowering_value_name(state, result_id),
          .result_type = result_type,
          .vector = source,
          .index = index,
      },
      &result));
  return loom_llvmir_lowering_map_value(state, result_id, result);
}

iree_status_t loom_llvmir_lowering_lower_vector_insert(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_value_id_t value_id = loom_vector_insert_value(op);
  loom_type_t value_source_type =
      loom_module_value_type(state->source_module, value_id);
  if (!loom_type_is_scalar(value_source_type)) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "vector.insert only supports scalar lane insertion");
  }
  loom_value_id_t result_id = loom_vector_insert_result(op);
  loom_type_t result_source_type =
      loom_module_value_type(state->source_module, result_id);
  uint32_t lane_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_lane_count(
      state, op, result_source_type, &lane_count));

  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_value(state, value_id, &value));
  loom_llvmir_value_id_t dest = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_vector_insert_dest(op), &dest));
  loom_llvmir_value_id_t index = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_lane_index(
      state, op, loom_vector_insert_static_indices(op),
      loom_vector_insert_indices(op), lane_count, &index));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lower_type(state, result_source_type, &result_type));

  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_insert_element(
      target_block,
      &(loom_llvmir_insert_element_desc_t){
          .result_name = loom_llvmir_lowering_value_name(state, result_id),
          .result_type = result_type,
          .vector = dest,
          .element = value,
          .index = index,
      },
      &result));
  return loom_llvmir_lowering_map_value(state, result_id, result);
}

iree_status_t loom_llvmir_lowering_lower_vector_shuffle(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_value_id_t source_id = loom_vector_shuffle_source(op);
  loom_type_t source_type =
      loom_module_value_type(state->source_module, source_id);
  uint32_t input_lane_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_lane_count(
      state, op, source_type, &input_lane_count));
  loom_value_id_t result_id = loom_vector_shuffle_result(op);
  loom_type_t result_source_type =
      loom_module_value_type(state->source_module, result_id);
  uint32_t result_lane_count = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_lane_count(
      state, op, result_source_type, &result_lane_count));

  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lower_type(state, result_source_type, &result_type));
  loom_llvmir_value_id_t source = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_value(state, source_id, &source));
  loom_llvmir_value_id_t poison = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_poison_constant(
      state->target_module, result_type, &poison));
  loom_llvmir_value_id_t mask = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_shuffle_mask(
      state, op, loom_vector_shuffle_source_lanes(op), input_lane_count,
      result_lane_count, &mask));

  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_shuffle_vector(
      target_block,
      &(loom_llvmir_shuffle_vector_desc_t){
          .result_name = loom_llvmir_lowering_value_name(state, result_id),
          .result_type = result_type,
          .lhs = source,
          .rhs = poison,
          .mask = mask,
      },
      &result));
  return loom_llvmir_lowering_map_value(state, result_id, result);
}

static iree_status_t loom_llvmir_lowering_vector_integer_flags(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint8_t instance_flags, loom_llvmir_integer_arithmetic_flags_t* out_flags) {
  const uint8_t known_flags =
      LOOM_VECTOR_INTOVERFLOWFLAGS_NSW | LOOM_VECTOR_INTOVERFLOWFLAGS_NUW;
  if (iree_any_bit_set(instance_flags, (uint8_t)~known_flags)) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "unknown vector integer overflow flags");
  }
  loom_llvmir_integer_arithmetic_flags_t flags =
      LOOM_LLVMIR_INTEGER_ARITHMETIC_NONE;
  if (iree_any_bit_set(instance_flags, LOOM_VECTOR_INTOVERFLOWFLAGS_NUW)) {
    flags |= LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_UNSIGNED_WRAP;
  }
  if (iree_any_bit_set(instance_flags, LOOM_VECTOR_INTOVERFLOWFLAGS_NSW)) {
    flags |= LOOM_LLVMIR_INTEGER_ARITHMETIC_NO_SIGNED_WRAP;
  }
  *out_flags = flags;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_lowering_vector_fast_math_flags(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint8_t instance_flags, loom_llvmir_fast_math_flags_t* out_flags) {
  const uint8_t known_flags = LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NNAN |
                              LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NINF |
                              LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NSZ;
  if (iree_any_bit_set(instance_flags, (uint8_t)~known_flags)) {
    return loom_llvmir_lowering_unsupported_op(
        state, op, "unknown vector floating-point assumption flags");
  }
  loom_llvmir_fast_math_flags_t flags = LOOM_LLVMIR_FAST_MATH_NONE;
  if (iree_any_bit_set(instance_flags, LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NNAN)) {
    flags |= LOOM_LLVMIR_FAST_MATH_NO_NANS;
  }
  if (iree_any_bit_set(instance_flags, LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NINF)) {
    flags |= LOOM_LLVMIR_FAST_MATH_NO_INFS;
  }
  if (iree_any_bit_set(instance_flags, LOOM_VECTOR_FLOATASSUMPTIONFLAGS_NSZ)) {
    flags |= LOOM_LLVMIR_FAST_MATH_NO_SIGNED_ZEROS;
  }
  *out_flags = flags;
  return iree_ok_status();
}

static bool loom_llvmir_lowering_vector_op_is_float_binop(const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_VECTOR_ADDF:
    case LOOM_OP_VECTOR_SUBF:
    case LOOM_OP_VECTOR_MULF:
    case LOOM_OP_VECTOR_DIVF:
    case LOOM_OP_VECTOR_REMF:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_llvmir_lowering_vector_binop_from_op(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_binop_t* out_binop) {
  switch (op->kind) {
    case LOOM_OP_VECTOR_ADDF:
      *out_binop = LOOM_LLVMIR_BINOP_FADD;
      return iree_ok_status();
    case LOOM_OP_VECTOR_SUBF:
      *out_binop = LOOM_LLVMIR_BINOP_FSUB;
      return iree_ok_status();
    case LOOM_OP_VECTOR_MULF:
      *out_binop = LOOM_LLVMIR_BINOP_FMUL;
      return iree_ok_status();
    case LOOM_OP_VECTOR_DIVF:
      *out_binop = LOOM_LLVMIR_BINOP_FDIV;
      return iree_ok_status();
    case LOOM_OP_VECTOR_REMF:
      *out_binop = LOOM_LLVMIR_BINOP_FREM;
      return iree_ok_status();
    case LOOM_OP_VECTOR_ADDI:
      *out_binop = LOOM_LLVMIR_BINOP_ADD;
      return iree_ok_status();
    case LOOM_OP_VECTOR_SUBI:
      *out_binop = LOOM_LLVMIR_BINOP_SUB;
      return iree_ok_status();
    case LOOM_OP_VECTOR_MULI:
      *out_binop = LOOM_LLVMIR_BINOP_MUL;
      return iree_ok_status();
    case LOOM_OP_VECTOR_DIVSI:
      *out_binop = LOOM_LLVMIR_BINOP_SDIV;
      return iree_ok_status();
    case LOOM_OP_VECTOR_DIVUI:
      *out_binop = LOOM_LLVMIR_BINOP_UDIV;
      return iree_ok_status();
    case LOOM_OP_VECTOR_REMSI:
      *out_binop = LOOM_LLVMIR_BINOP_SREM;
      return iree_ok_status();
    case LOOM_OP_VECTOR_REMUI:
      *out_binop = LOOM_LLVMIR_BINOP_UREM;
      return iree_ok_status();
    case LOOM_OP_VECTOR_ANDI:
      *out_binop = LOOM_LLVMIR_BINOP_AND;
      return iree_ok_status();
    case LOOM_OP_VECTOR_ORI:
      *out_binop = LOOM_LLVMIR_BINOP_OR;
      return iree_ok_status();
    case LOOM_OP_VECTOR_XORI:
      *out_binop = LOOM_LLVMIR_BINOP_XOR;
      return iree_ok_status();
    case LOOM_OP_VECTOR_SHLI:
      *out_binop = LOOM_LLVMIR_BINOP_SHL;
      return iree_ok_status();
    case LOOM_OP_VECTOR_SHRSI:
      *out_binop = LOOM_LLVMIR_BINOP_ASHR;
      return iree_ok_status();
    case LOOM_OP_VECTOR_SHRUI:
      *out_binop = LOOM_LLVMIR_BINOP_LSHR;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "vector op has no direct LLVM binary opcode mapping");
  }
}

iree_status_t loom_llvmir_lowering_lower_vector_binop(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operands[2] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));
  loom_llvmir_binop_t binop = LOOM_LLVMIR_BINOP_ADD;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_vector_binop_from_op(state, op, &binop));

  loom_llvmir_binop_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.op = binop;
  desc.lhs = operands[0];
  desc.rhs = operands[1];
  if (loom_llvmir_lowering_vector_op_is_float_binop(op)) {
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_fast_math_flags(
        state, op, op->instance_flags, &desc.fast_math_flags));
  } else {
    IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_integer_flags(
        state, op, op->instance_flags, &desc.integer_flags));
  }
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_binop(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

iree_status_t loom_llvmir_lowering_lower_vector_negf(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_vector_negf_input(op), &value));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module, loom_vector_negf_result(op)),
      &result_type));

  loom_llvmir_unop_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_vector_negf_result(op));
  desc.result_type = result_type;
  desc.op = LOOM_LLVMIR_UNOP_FNEG;
  desc.value = value;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_fast_math_flags(
      state, op, op->instance_flags, &desc.fast_math_flags));
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_unop(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_lowering_vector_icmp_predicate(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint8_t source_predicate, loom_llvmir_icmp_predicate_t* out_predicate) {
  switch (source_predicate) {
    case LOOM_VECTOR_CMPI_PREDICATE_EQ:
      *out_predicate = LOOM_LLVMIR_ICMP_EQ;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_NE:
      *out_predicate = LOOM_LLVMIR_ICMP_NE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_SLT:
      *out_predicate = LOOM_LLVMIR_ICMP_SLT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_SLE:
      *out_predicate = LOOM_LLVMIR_ICMP_SLE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_SGT:
      *out_predicate = LOOM_LLVMIR_ICMP_SGT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_SGE:
      *out_predicate = LOOM_LLVMIR_ICMP_SGE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_ULT:
      *out_predicate = LOOM_LLVMIR_ICMP_ULT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_ULE:
      *out_predicate = LOOM_LLVMIR_ICMP_ULE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_UGT:
      *out_predicate = LOOM_LLVMIR_ICMP_UGT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPI_PREDICATE_UGE:
      *out_predicate = LOOM_LLVMIR_ICMP_UGE;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "unknown vector integer comparison predicate");
  }
}

iree_status_t loom_llvmir_lowering_lower_vector_icmp(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operands[2] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));
  loom_llvmir_icmp_predicate_t predicate = LOOM_LLVMIR_ICMP_EQ;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_icmp_predicate(
      state, op, loom_vector_cmpi_predicate(op), &predicate));

  loom_llvmir_icmp_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.predicate = predicate;
  desc.lhs = operands[0];
  desc.rhs = operands[1];
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_icmp(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_lowering_vector_fcmp_predicate(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    uint8_t source_predicate, loom_llvmir_fcmp_predicate_t* out_predicate) {
  switch (source_predicate) {
    case LOOM_VECTOR_CMPF_PREDICATE_OEQ:
      *out_predicate = LOOM_LLVMIR_FCMP_OEQ;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_OGT:
      *out_predicate = LOOM_LLVMIR_FCMP_OGT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_OGE:
      *out_predicate = LOOM_LLVMIR_FCMP_OGE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_OLT:
      *out_predicate = LOOM_LLVMIR_FCMP_OLT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_OLE:
      *out_predicate = LOOM_LLVMIR_FCMP_OLE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_ONE:
      *out_predicate = LOOM_LLVMIR_FCMP_ONE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_ORD:
      *out_predicate = LOOM_LLVMIR_FCMP_ORD;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_UEQ:
      *out_predicate = LOOM_LLVMIR_FCMP_UEQ;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_UGT:
      *out_predicate = LOOM_LLVMIR_FCMP_UGT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_UGE:
      *out_predicate = LOOM_LLVMIR_FCMP_UGE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_ULT:
      *out_predicate = LOOM_LLVMIR_FCMP_ULT;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_ULE:
      *out_predicate = LOOM_LLVMIR_FCMP_ULE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_UNE:
      *out_predicate = LOOM_LLVMIR_FCMP_UNE;
      return iree_ok_status();
    case LOOM_VECTOR_CMPF_PREDICATE_UNO:
      *out_predicate = LOOM_LLVMIR_FCMP_UNO;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "unknown vector floating-point comparison predicate");
  }
}

iree_status_t loom_llvmir_lowering_lower_vector_fcmp(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operands[2] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));
  loom_llvmir_fcmp_predicate_t predicate = LOOM_LLVMIR_FCMP_FALSE;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_vector_fcmp_predicate(
      state, op, loom_vector_cmpf_predicate(op), &predicate));

  loom_llvmir_fcmp_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.predicate = predicate;
  desc.lhs = operands[0];
  desc.rhs = operands[1];
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_fcmp(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

iree_status_t loom_llvmir_lowering_lower_vector_select(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operands[3] = {LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID,
                                        LOOM_LLVMIR_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_lookup_operands(state, op, operands));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_vector_select_result(op)),
      &result_type));

  loom_llvmir_select_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_vector_select_result(op));
  desc.result_type = result_type;
  desc.condition = operands[0];
  desc.true_value = operands[1];
  desc.false_value = operands[2];
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_select(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}

static iree_status_t loom_llvmir_lowering_vector_cast_op_from_kind(
    const loom_llvmir_lowering_state_t* state, const loom_op_t* op,
    loom_llvmir_cast_op_t* out_cast_op) {
  switch (op->kind) {
    case LOOM_OP_VECTOR_SITOFP:
      *out_cast_op = LOOM_LLVMIR_CAST_SIGNED_INT_TO_FP;
      return iree_ok_status();
    case LOOM_OP_VECTOR_UITOFP:
      *out_cast_op = LOOM_LLVMIR_CAST_UNSIGNED_INT_TO_FP;
      return iree_ok_status();
    case LOOM_OP_VECTOR_FPTOSI:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_TO_SIGNED_INT;
      return iree_ok_status();
    case LOOM_OP_VECTOR_FPTOUI:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_TO_UNSIGNED_INT;
      return iree_ok_status();
    case LOOM_OP_VECTOR_EXTF:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_EXTEND;
      return iree_ok_status();
    case LOOM_OP_VECTOR_FPTRUNC:
      *out_cast_op = LOOM_LLVMIR_CAST_FP_TRUNCATE;
      return iree_ok_status();
    case LOOM_OP_VECTOR_EXTSI:
      *out_cast_op = LOOM_LLVMIR_CAST_SIGN_EXTEND;
      return iree_ok_status();
    case LOOM_OP_VECTOR_EXTUI:
      *out_cast_op = LOOM_LLVMIR_CAST_ZERO_EXTEND;
      return iree_ok_status();
    case LOOM_OP_VECTOR_TRUNCI:
      *out_cast_op = LOOM_LLVMIR_CAST_TRUNCATE;
      return iree_ok_status();
    case LOOM_OP_VECTOR_BITCAST:
      *out_cast_op = LOOM_LLVMIR_CAST_BITCAST;
      return iree_ok_status();
    default:
      return loom_llvmir_lowering_unsupported_op(
          state, op, "vector op has no direct LLVM cast opcode mapping");
  }
}

iree_status_t loom_llvmir_lowering_lower_vector_cast(
    loom_llvmir_lowering_state_t* state, loom_llvmir_block_t* target_block,
    const loom_op_t* op) {
  loom_llvmir_value_id_t operand = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lookup_value(
      state, loom_op_const_operands(op)[0], &operand));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_lowering_lower_type(
      state,
      loom_module_value_type(state->source_module,
                             loom_op_const_results(op)[0]),
      &result_type));
  loom_llvmir_cast_op_t cast_op = LOOM_LLVMIR_CAST_TRUNCATE;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_lowering_vector_cast_op_from_kind(state, op, &cast_op));

  loom_llvmir_cast_desc_t desc = {0};
  desc.result_name =
      loom_llvmir_lowering_value_name(state, loom_op_const_results(op)[0]);
  desc.result_type = result_type;
  desc.op = cast_op;
  desc.value = operand;
  loom_llvmir_value_id_t result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_cast(target_block, &desc, &result));
  return loom_llvmir_lowering_map_single_result(state, op, result);
}
