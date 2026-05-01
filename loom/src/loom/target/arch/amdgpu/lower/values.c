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
#include "loom/target/arch/amdgpu/lower/internal.h"

typedef struct loom_amdgpu_constant_plan_t {
  // Source result value receiving the emitted low constant.
  loom_value_id_t result;
  // Descriptor row selected for the constant move packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Module string ID for the descriptor's imm32 attribute.
  loom_string_id_t imm32_attr_name_id;
  // Number of result lanes receiving the same 32-bit bit pattern.
  uint32_t lane_count;
  // Immediate bit pattern emitted into each selected result lane.
  uint32_t bit_pattern;
} loom_amdgpu_constant_plan_t;

typedef struct loom_amdgpu_vector_iota_plan_t {
  // Descriptor row selected for each lane constant packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Module string ID for the descriptor's imm32 attribute.
  loom_string_id_t imm32_attr_name_id;
  // Result vector receiving the generated i32 lane constants.
  loom_value_id_t result;
  // Static number of generated lanes.
  uint32_t lane_count;
  // Precomputed lane bit patterns emitted as VGPR constants.
  uint32_t lane_bit_patterns[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
} loom_amdgpu_vector_iota_plan_t;

typedef struct loom_amdgpu_vector_from_elements_plan_t {
  // Result vector assembled from the selected source elements.
  loom_value_id_t result;
  // Static source element count.
  uint32_t element_count;
  // Source and result scalar element type.
  loom_scalar_type_t element_type;
  // Source scalar values in result lane order.
  loom_value_id_t elements[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
} loom_amdgpu_vector_from_elements_plan_t;

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

static iree_status_t loom_amdgpu_resolve_imm32_descriptor(
    loom_low_lower_context_t* context, uint64_t descriptor_id,
    loom_low_lower_resolved_descriptor_t* out_descriptor,
    loom_string_id_t* out_imm32_attr_name_id, bool* out_present) {
  IREE_ASSERT_ARGUMENT(out_imm32_attr_name_id);
  *out_imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor_if_present(
      context, descriptor_id, out_descriptor, out_present));
  if (!*out_present) {
    return iree_ok_status();
  }
  return loom_amdgpu_intern(context, IREE_SV("imm32"), out_imm32_attr_name_id);
}

static iree_status_t loom_amdgpu_select_u32_bit_pattern_constant_plan(
    loom_low_lower_context_t* context, uint32_t bit_pattern,
    loom_value_id_t result, uint64_t descriptor_id,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_selected = false;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  loom_string_id_t imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_imm32_descriptor(
      context, descriptor_id, &descriptor, &imm32_attr_name_id,
      &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_constant_plan_t){
      .result = result,
      .descriptor = descriptor,
      .imm32_attr_name_id = imm32_attr_name_id,
      .lane_count = 1,
      .bit_pattern = bit_pattern,
  };
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_i32_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, uint64_t descriptor_id,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  IREE_ASSERT_ARGUMENT(out_selected);
  if (!loom_amdgpu_attr_is_i32_immediate(value)) {
    *out_selected = false;
    return iree_ok_status();
  }
  return loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, (uint32_t)(int32_t)value.i64, result, descriptor_id, out_plan,
      out_selected);
}

static iree_status_t loom_amdgpu_select_f32_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, uint32_t lane_count,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_selected = false;
  if (!loom_amdgpu_attr_is_f32_immediate(value)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, loom_amdgpu_attr_f32_bit_pattern(value), result,
      LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32, out_plan, out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  out_plan->lane_count = lane_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_index_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_selected = false;
  const loom_value_id_t result = loom_index_constant_result(source_op);
  const loom_attribute_t value = loom_index_constant_value(source_op);
  if (!loom_amdgpu_value_is_address_scalar(context, result) ||
      !loom_amdgpu_attr_is_u32_address_immediate(value)) {
    return iree_ok_status();
  }
  return loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, (uint32_t)value.i64, result, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32,
      out_plan, out_selected);
}

static iree_status_t loom_amdgpu_select_scalar_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_selected = false;
  const loom_value_id_t result = loom_scalar_constant_result(source_op);
  const loom_attribute_t value = loom_scalar_constant_value(source_op);
  if (loom_amdgpu_value_is_f32(context, result)) {
    return loom_amdgpu_select_f32_constant_plan(
        context, value, result, /*lane_count=*/1, out_plan, out_selected);
  }
  if (!loom_amdgpu_value_is_i32(context, result)) {
    return iree_ok_status();
  }
  const uint64_t descriptor_id = loom_amdgpu_value_prefers_vgpr(context, result)
                                     ? LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32
                                     : LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32;
  return loom_amdgpu_select_i32_constant_plan(
      context, value, result, descriptor_id, out_plan, out_selected);
}

static iree_status_t loom_amdgpu_select_vector_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_constant_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const loom_attribute_t value = loom_vector_constant_value(source_op);
  const uint32_t i32_lane_count =
      loom_amdgpu_vector_i32_lane_count(result_type);
  if (i32_lane_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_i32_constant_plan(
        context, value, result, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32, out_plan,
        out_selected));
    if (!*out_selected) {
      return iree_ok_status();
    }
    out_plan->lane_count = i32_lane_count;
    return iree_ok_status();
  }
  const uint32_t f32_lane_count =
      loom_amdgpu_vector_f32_lane_count(result_type);
  if (f32_lane_count != 0) {
    return loom_amdgpu_select_f32_constant_plan(
        context, value, result, f32_lane_count, out_plan, out_selected);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_vector_iota_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_iota_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_iota_plan_t){0};
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_iota_result(source_op);
  const uint32_t lane_count =
      loom_amdgpu_vector_i32_lane_count(loom_module_value_type(module, result));
  if (lane_count == 0) {
    return iree_ok_status();
  }
  int64_t base = 0;
  int64_t step = 0;
  if (!loom_amdgpu_value_as_i32_constant(
          context, loom_vector_iota_base(source_op), &base) ||
      !loom_amdgpu_value_as_i32_constant(
          context, loom_vector_iota_step(source_op), &step)) {
    return iree_ok_status();
  }

  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_imm32_descriptor(
      context, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32, &out_plan->descriptor,
      &out_plan->imm32_attr_name_id, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }

  for (uint32_t i = 0; i < lane_count; ++i) {
    int64_t lane_value = 0;
    if (!loom_amdgpu_iota_i32_lane_value(base, step, i, &lane_value)) {
      return iree_ok_status();
    }
    out_plan->lane_bit_patterns[i] = (uint32_t)(int32_t)lane_value;
  }
  out_plan->result = result;
  out_plan->lane_count = lane_count;
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_select_vector_extract_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_extract_plan_t* out_plan) {
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

static bool loom_amdgpu_select_vector_from_elements_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_from_elements_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_vector_from_elements_plan_t){0};
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
  if (element_type != LOOM_SCALAR_TYPE_I32 &&
      element_type != LOOM_SCALAR_TYPE_F32) {
    return false;
  }
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
    out_plan->elements[i] = element;
  }
  out_plan->result = result;
  out_plan->element_count = elements.count;
  out_plan->element_type = element_type;
  return true;
}

iree_status_t loom_amdgpu_select_value_plan(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op,
                                            loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
    case LOOM_OP_SCALAR_CONSTANT:
    case LOOM_OP_VECTOR_CONSTANT: {
      loom_amdgpu_constant_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      bool selected = false;
      if (source_op->kind == LOOM_OP_INDEX_CONSTANT) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_select_index_constant_plan(
            context, source_op, plan_data, &selected));
      } else if (source_op->kind == LOOM_OP_SCALAR_CONSTANT) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_select_scalar_constant_plan(
            context, source_op, plan_data, &selected));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_constant_plan(
            context, source_op, plan_data, &selected));
      }
      if (selected) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_IOTA: {
      loom_amdgpu_vector_iota_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_vector_iota_plan(
          context, source_op, plan_data, &selected));
      if (selected) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_EXTRACT: {
      loom_amdgpu_vector_extract_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      if (loom_amdgpu_select_vector_extract_plan(context, source_op,
                                                 plan_data)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_FROM_ELEMENTS: {
      loom_amdgpu_vector_from_elements_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      if (loom_amdgpu_select_vector_from_elements_plan(context, source_op,
                                                       plan_data)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_bind_vgpr_u32_lane_constants(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t imm32_attr_name_id, const uint32_t* lane_bit_patterns,
    uint32_t lane_count) {
  IREE_ASSERT_ARGUMENT(lane_bit_patterns);
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t low_lane_values[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
        context, source_op, descriptor, imm32_attr_name_id,
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

static iree_status_t loom_amdgpu_lower_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t imm32_attr_name_id, uint32_t bit_pattern,
    loom_value_id_t source_result) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_resolved_const_u32(
      context, source_op, descriptor, imm32_attr_name_id, bit_pattern,
      result_type, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static iree_status_t loom_amdgpu_lower_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_constant_plan_t* plan) {
  if (plan->lane_count == 1) {
    return loom_amdgpu_lower_u32_constant(context, source_op, &plan->descriptor,
                                          plan->imm32_attr_name_id,
                                          plan->bit_pattern, plan->result);
  }
  uint32_t lane_bit_patterns[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->lane_count; ++i) {
    lane_bit_patterns[i] = plan->bit_pattern;
  }
  return loom_amdgpu_bind_vgpr_u32_lane_constants(
      context, source_op, plan->result, &plan->descriptor,
      plan->imm32_attr_name_id, lane_bit_patterns, plan->lane_count);
}

static iree_status_t loom_amdgpu_lower_vector_iota(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_iota_plan_t* plan) {
  return loom_amdgpu_bind_vgpr_u32_lane_constants(
      context, source_op, plan->result, &plan->descriptor,
      plan->imm32_attr_name_id, plan->lane_bit_patterns, plan->lane_count);
}

static iree_status_t loom_amdgpu_lower_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
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
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_from_elements_plan_t* plan) {
  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  switch (plan->element_type) {
    case LOOM_SCALAR_TYPE_I32:
      for (uint32_t i = 0; i < plan->element_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
            context, source_op, plan->elements[i], &lanes[i]));
      }
      break;
    case LOOM_SCALAR_TYPE_F32:
      for (uint32_t i = 0; i < plan->element_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_low_lower_lookup_value(context, plan->elements[i], &lanes[i]));
      }
      break;
    default:
      IREE_CHECK_UNREACHABLE();
  }
  if (plan->element_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, lanes[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lanes, plan->element_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_value_op(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan) {
  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
    case LOOM_OP_SCALAR_CONSTANT:
    case LOOM_OP_VECTOR_CONSTANT:
      return loom_amdgpu_lower_constant_plan(
          context, source_op,
          (const loom_amdgpu_constant_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_IOTA:
      return loom_amdgpu_lower_vector_iota(
          context, source_op,
          (const loom_amdgpu_vector_iota_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_EXTRACT:
      return loom_amdgpu_lower_vector_extract(
          context, source_op,
          (const loom_amdgpu_vector_extract_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
      return loom_amdgpu_lower_vector_from_elements(
          context, source_op,
          (const loom_amdgpu_vector_from_elements_plan_t*)plan.target_data);
    default:
      IREE_CHECK_UNREACHABLE();
  }
}
