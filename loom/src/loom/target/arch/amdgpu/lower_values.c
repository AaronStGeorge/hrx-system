// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"

static bool loom_amdgpu_iota_i32_lane_value(int64_t base, int64_t step,
                                            uint32_t lane, int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  int64_t scaled_step = 0;
  if (!iree_checked_mul_i64((int64_t)lane, step, &scaled_step)) {
    return false;
  }
  int64_t value = 0;
  if (!iree_checked_add_i64(base, scaled_step, &value) || value < INT32_MIN ||
      value > INT32_MAX) {
    return false;
  }
  *out_value = value;
  return true;
}

static bool loom_amdgpu_iota_i32_lanes_fit(int64_t base, int64_t step,
                                           uint32_t lane_count) {
  for (uint32_t i = 0; i < lane_count; ++i) {
    int64_t unused = 0;
    if (!loom_amdgpu_iota_i32_lane_value(base, step, i, &unused)) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_can_lower_scalar_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_id_t result = loom_scalar_constant_result(source_op);
  const loom_attribute_t value = loom_scalar_constant_value(source_op);
  if (loom_amdgpu_value_is_i32(context, result)) {
    return loom_amdgpu_attr_is_i32_immediate(value);
  }
  if (loom_amdgpu_value_is_f32(context, result)) {
    return loom_amdgpu_attr_is_f32_immediate(value);
  }
  return false;
}

static bool loom_amdgpu_can_lower_index_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_value_is_address_scalar(
             context, loom_index_constant_result(source_op)) &&
         loom_amdgpu_attr_is_i32_immediate(
             loom_index_constant_value(source_op));
}

static bool loom_amdgpu_can_lower_vector_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_constant_result(source_op));
  const loom_attribute_t value = loom_vector_constant_value(source_op);
  if (loom_amdgpu_vector_i32_lane_count(result_type) != 0) {
    return loom_amdgpu_attr_is_i32_immediate(value);
  }
  if (loom_amdgpu_vector_f32_lane_count(result_type) != 0) {
    return loom_amdgpu_attr_is_f32_immediate(value);
  }
  return false;
}

static bool loom_amdgpu_can_lower_vector_iota(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint32_t lane_count = loom_amdgpu_vector_i32_lane_count(
      loom_module_value_type(module, loom_vector_iota_result(source_op)));
  int64_t base = 0;
  int64_t step = 0;
  return lane_count != 0 &&
         loom_amdgpu_value_as_i32_constant(
             context, loom_vector_iota_base(source_op), &base) &&
         loom_amdgpu_value_as_i32_constant(
             context, loom_vector_iota_step(source_op), &step) &&
         loom_amdgpu_iota_i32_lanes_fit(base, step, lane_count);
}

static bool loom_amdgpu_select_vector_extract_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_extract_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_amdgpu_vector_extract_plan_t){0};
  if (loom_vector_extract_indices(source_op).count != 0) {
    return false;
  }
  loom_attribute_t static_indices =
      loom_vector_extract_static_indices(source_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      static_indices.i64_array[0] < 0 ||
      static_indices.i64_array[0] > UINT32_MAX) {
    return false;
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_extract_source(source_op));
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(source_type);
  const uint32_t lane_offset = (uint32_t)static_indices.i64_array[0];
  if (lane_count == 0 || lane_offset >= lane_count) {
    return false;
  }

  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_extract_result(source_op));
  if (loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_I32 &&
      !loom_amdgpu_type_is_i32(result_type)) {
    return false;
  }
  if (loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_F32 &&
      !loom_amdgpu_type_is_f32(result_type)) {
    return false;
  }

  *out_plan = (loom_amdgpu_vector_extract_plan_t){
      .source = loom_vector_extract_source(source_op),
      .result = loom_vector_extract_result(source_op),
      .lane_offset = lane_offset,
      .lane_count = lane_count,
  };
  return true;
}

static bool loom_amdgpu_can_lower_vector_from_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_op);
  if (elements.count == 0 ||
      elements.count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return false;
  }
  const loom_value_id_t result = loom_vector_from_elements_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (loom_amdgpu_vector_32bit_lane_count(result_type) != elements.count) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(result_type);
  for (uint32_t i = 0; i < elements.count; ++i) {
    const loom_value_id_t element = elements.values[i];
    const loom_type_t source_type = loom_module_value_type(module, element);
    if (!loom_type_is_scalar(source_type) ||
        loom_type_element_type(source_type) != element_type) {
      return false;
    }
    if (element_type == LOOM_SCALAR_TYPE_I32 &&
        !loom_amdgpu_value_can_materialize_as_vgpr_i32(context, element)) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_amdgpu_select_value_plan(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
#define LOOM_AMDGPU_SELECT_IF(condition)                              \
  do {                                                                \
    if (condition) {                                                  \
      *out_plan = loom_low_lower_plan_make(source_op->kind, 0, NULL); \
    }                                                                 \
    return iree_ok_status();                                          \
  } while (false)
  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
      LOOM_AMDGPU_SELECT_IF(
          loom_amdgpu_can_lower_index_constant(context, source_op));
    case LOOM_OP_SCALAR_CONSTANT:
      LOOM_AMDGPU_SELECT_IF(
          loom_amdgpu_can_lower_scalar_constant(context, source_op));
    case LOOM_OP_VECTOR_CONSTANT:
      LOOM_AMDGPU_SELECT_IF(
          loom_amdgpu_can_lower_vector_constant(context, source_op));
    case LOOM_OP_VECTOR_IOTA:
      LOOM_AMDGPU_SELECT_IF(
          loom_amdgpu_can_lower_vector_iota(context, source_op));
    case LOOM_OP_VECTOR_EXTRACT: {
      loom_amdgpu_vector_extract_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      if (loom_amdgpu_select_vector_extract_plan(context, source_op,
                                                 plan_data)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, 0, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
      LOOM_AMDGPU_SELECT_IF(
          loom_amdgpu_can_lower_vector_from_elements(context, source_op));
    default:
      return iree_ok_status();
  }
#undef LOOM_AMDGPU_SELECT_IF
}

static iree_status_t loom_amdgpu_bind_vgpr_u32_lane_constants(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, const uint32_t* lane_bit_patterns,
    uint32_t lane_count) {
  IREE_ASSERT_ARGUMENT(lane_bit_patterns);
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t low_lane_values[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32,
        lane_bit_patterns[i], lane_type, &low_lane_values[i]));
  }

  if (lane_count == 1) {
    return loom_low_lower_bind_value(context, source_result,
                                     low_lane_values[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_lane_values, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_bind_vgpr_i32_lane_constants(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, const int64_t* lane_values,
    uint32_t lane_count) {
  IREE_ASSERT_ARGUMENT(lane_values);
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  uint32_t lane_bit_patterns[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    IREE_ASSERT_GE(lane_values[i], INT32_MIN);
    IREE_ASSERT_LE(lane_values[i], INT32_MAX);
    lane_bit_patterns[i] = (uint32_t)(int32_t)lane_values[i];
  }
  return loom_amdgpu_bind_vgpr_u32_lane_constants(
      context, source_op, source_result, lane_bit_patterns, lane_count);
}

static iree_status_t loom_amdgpu_bind_vgpr_f32_lane_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, uint32_t lane_bit_pattern,
    uint32_t lane_count) {
  uint32_t lane_bit_patterns[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    lane_bit_patterns[i] = lane_bit_pattern;
  }
  return loom_amdgpu_bind_vgpr_u32_lane_constants(
      context, source_op, source_result, lane_bit_patterns, lane_count);
}

static iree_status_t loom_amdgpu_lower_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, uint32_t bit_pattern,
    loom_value_id_t source_result) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(context, source_op,
                                                  descriptor_id, bit_pattern,
                                                  result_type, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static iree_status_t loom_amdgpu_lower_i32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_attribute_t source_attr,
    loom_value_id_t source_result) {
  const int64_t source_value = source_attr.i64;
  return loom_amdgpu_lower_u32_constant(context, source_op, descriptor_id,
                                        (uint32_t)(int32_t)source_value,
                                        source_result);
}

static iree_status_t loom_amdgpu_lower_index_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_constant(context, source_op,
                                        LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32,
                                        loom_index_constant_value(source_op),
                                        loom_index_constant_result(source_op));
}

static iree_status_t loom_amdgpu_lower_scalar_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source_result = loom_scalar_constant_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source_result);
  if (loom_amdgpu_type_is_f32(source_type)) {
    return loom_amdgpu_lower_u32_constant(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32,
        loom_amdgpu_attr_f32_bit_pattern(loom_scalar_constant_value(source_op)),
        source_result);
  }

  const uint64_t descriptor_id =
      loom_amdgpu_value_prefers_vgpr(context, source_result)
          ? LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32
          : LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32;
  return loom_amdgpu_lower_i32_constant(context, source_op, descriptor_id,
                                        loom_scalar_constant_value(source_op),
                                        source_result);
}

static iree_status_t loom_amdgpu_lower_vector_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source_result = loom_vector_constant_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source_result);
  const uint32_t i32_lane_count =
      loom_amdgpu_vector_i32_lane_count(source_type);
  if (i32_lane_count == 1) {
    return loom_amdgpu_lower_i32_constant(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32,
        loom_vector_constant_value(source_op), source_result);
  }
  if (i32_lane_count > 1) {
    const int64_t source_value = loom_vector_constant_value(source_op).i64;
    int64_t lane_values[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
    for (uint32_t i = 0; i < i32_lane_count; ++i) {
      lane_values[i] = source_value;
    }
    return loom_amdgpu_bind_vgpr_i32_lane_constants(
        context, source_op, source_result, lane_values, i32_lane_count);
  }

  const uint32_t f32_lane_count =
      loom_amdgpu_vector_f32_lane_count(source_type);
  if (f32_lane_count != 0) {
    return loom_amdgpu_bind_vgpr_f32_lane_constant(
        context, source_op, source_result,
        loom_amdgpu_attr_f32_bit_pattern(loom_vector_constant_value(source_op)),
        f32_lane_count);
  }

  IREE_ASSERT_UNREACHABLE();
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_iota(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source_result = loom_vector_iota_result(source_op);
  const uint32_t lane_count = loom_amdgpu_vector_i32_lane_count(
      loom_module_value_type(module, source_result));
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  int64_t base = 0;
  int64_t step = 0;
  const bool has_i32_base = loom_amdgpu_value_as_i32_constant(
      context, loom_vector_iota_base(source_op), &base);
  const bool has_i32_step = loom_amdgpu_value_as_i32_constant(
      context, loom_vector_iota_step(source_op), &step);
  IREE_ASSERT(has_i32_base);
  IREE_ASSERT(has_i32_step);

  int64_t lane_values[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    const bool lane_value_fits =
        loom_amdgpu_iota_i32_lane_value(base, step, i, &lane_values[i]);
    IREE_ASSERT(lane_value_fits);
  }
  return loom_amdgpu_bind_vgpr_i32_lane_constants(
      context, source_op, source_result, lane_values, lane_count);
}

static iree_status_t loom_amdgpu_lower_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  if (plan->lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, low_source);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op,
                                                  low_source, plan->lane_offset,
                                                  result_type, &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}

static iree_status_t loom_amdgpu_lower_vector_from_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_op);
  const loom_value_id_t source_result =
      loom_vector_from_elements_result(source_op);
  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < elements.count; ++i) {
    const loom_value_id_t element = elements.values[i];
    if (loom_amdgpu_value_is_i32(context, element)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
          context, source_op, element, &lanes[i]));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, element, &lanes[i]));
    }
  }
  if (elements.count == 1) {
    return loom_low_lower_bind_value(context, source_result, lanes[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lanes, elements.count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_value_op(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan) {
  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
      return loom_amdgpu_lower_index_constant(context, source_op);
    case LOOM_OP_SCALAR_CONSTANT:
      return loom_amdgpu_lower_scalar_constant(context, source_op);
    case LOOM_OP_VECTOR_CONSTANT:
      return loom_amdgpu_lower_vector_constant(context, source_op);
    case LOOM_OP_VECTOR_IOTA:
      return loom_amdgpu_lower_vector_iota(context, source_op);
    case LOOM_OP_VECTOR_EXTRACT:
      return loom_amdgpu_lower_vector_extract(
          context, source_op,
          (const loom_amdgpu_vector_extract_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
      return loom_amdgpu_lower_vector_from_elements(context, source_op);
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
}
