// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/internal.h"
#include "loom/target/arch/amdgpu/target_refs.h"

typedef struct loom_amdgpu_constant_plan_t {
  // Source result value receiving the emitted low constant.
  loom_value_id_t result;
  // Descriptor row selected for the constant move packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Module string ID for the descriptor's imm32 attribute.
  loom_string_id_t imm32_attr_name_id;
  // Number of 32-bit registers receiving the same bit pattern.
  uint32_t register_count;
  // Immediate bit pattern emitted into each selected result register.
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

typedef struct loom_amdgpu_cast_alias_plan_t {
  // Source value whose existing low mapping can represent the cast result.
  loom_value_id_t source;
  // Result value receiving the low source alias.
  loom_value_id_t result;
} loom_amdgpu_cast_alias_plan_t;

typedef struct loom_amdgpu_scalar_trunci_plan_t {
  // Source integer value being truncated.
  loom_value_id_t source;
  // Result integer value receiving the low 32 bits of source.
  loom_value_id_t result;
} loom_amdgpu_scalar_trunci_plan_t;

static bool loom_amdgpu_iota_i32_lane_value(int64_t base, int64_t step,
                                            uint32_t lane, int64_t* out_value) {
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
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor,
    loom_string_id_t* out_imm32_attr_name_id, bool* out_present) {
  *out_imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, descriptor_ref, out_descriptor, out_present));
  if (!*out_present) {
    return iree_ok_status();
  }
  return loom_amdgpu_intern(context, IREE_SV("imm32"), out_imm32_attr_name_id);
}

static iree_status_t loom_amdgpu_select_u32_bit_pattern_constant_plan(
    loom_low_lower_context_t* context, uint32_t bit_pattern,
    loom_value_id_t result, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  loom_string_id_t imm32_attr_name_id = LOOM_STRING_ID_INVALID;
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_imm32_descriptor(
      context, descriptor_ref, &descriptor, &imm32_attr_name_id,
      &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_constant_plan_t){
      .result = result,
      .descriptor = descriptor,
      .imm32_attr_name_id = imm32_attr_name_id,
      .register_count = 1,
      .bit_pattern = bit_pattern,
  };
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_i32_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  if (!loom_amdgpu_attr_is_i32_immediate(value)) {
    *out_selected = false;
    return iree_ok_status();
  }
  return loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, (uint32_t)(int32_t)value.i64, result, descriptor_ref, out_plan,
      out_selected);
}

static iree_status_t loom_amdgpu_select_f32_constant_plan(
    loom_low_lower_context_t* context, loom_attribute_t value,
    loom_value_id_t result, uint32_t lane_count,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  if (!loom_amdgpu_attr_is_f32_immediate(value)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, loom_amdgpu_attr_f32_bit_pattern(value), result,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  out_plan->register_count = lane_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_packed_16bit_float_constant_plan(
    loom_low_lower_context_t* context, loom_type_t result_type,
    loom_attribute_t value, loom_value_id_t result,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  uint32_t unused_payload_bit_count = 0;
  uint32_t register_count = 0;
  if (!loom_amdgpu_type_packed_16bit_float_storage(
          result_type, &unused_payload_bit_count, &register_count) ||
      !loom_amdgpu_attr_is_16bit_float_immediate(value)) {
    return iree_ok_status();
  }
  const uint32_t lane_bit_pattern = loom_amdgpu_attr_16bit_float_bit_pattern(
      loom_type_element_type(result_type), value);
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, lane_bit_pattern | (lane_bit_pattern << 16), result,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected));
  if (!*out_selected) {
    return iree_ok_status();
  }
  out_plan->register_count = register_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_index_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  *out_selected = false;
  const loom_value_id_t result = loom_index_constant_result(source_op);
  const loom_attribute_t value = loom_index_constant_value(source_op);
  if (!loom_amdgpu_value_is_address_scalar(context, result) ||
      !loom_amdgpu_attr_is_u32_address_immediate(value)) {
    return iree_ok_status();
  }
  return loom_amdgpu_select_u32_bit_pattern_constant_plan(
      context, (uint32_t)value.i64, result,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, out_plan, out_selected);
}

static iree_status_t loom_amdgpu_select_scalar_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  *out_selected = false;
  const loom_value_id_t result = loom_scalar_constant_result(source_op);
  const loom_attribute_t value = loom_scalar_constant_value(source_op);
  if (loom_amdgpu_value_is_f32(context, result)) {
    return loom_amdgpu_select_f32_constant_plan(
        context, value, result, /*lane_count=*/1, out_plan, out_selected);
  }
  if (loom_amdgpu_value_is_16bit_float(context, result)) {
    if (!loom_amdgpu_attr_is_16bit_float_immediate(value)) {
      return iree_ok_status();
    }
    const loom_type_t result_type =
        loom_module_value_type(loom_low_lower_context_module(context), result);
    return loom_amdgpu_select_u32_bit_pattern_constant_plan(
        context,
        loom_amdgpu_attr_16bit_float_bit_pattern(
            loom_type_element_type(result_type), value),
        result, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan, out_selected);
  }
  if (!loom_amdgpu_value_is_i32(context, result)) {
    return iree_ok_status();
  }
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      loom_amdgpu_value_prefers_vgpr(context, result)
          ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
          : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
  return loom_amdgpu_select_i32_constant_plan(
      context, value, result, descriptor_ref, out_plan, out_selected);
}

static iree_status_t loom_amdgpu_select_vector_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_constant_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_constant_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const loom_attribute_t value = loom_vector_constant_value(source_op);
  const uint32_t i32_lane_count =
      loom_amdgpu_vector_i32_lane_count(result_type);
  if (i32_lane_count != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_select_i32_constant_plan(
        context, value, result, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, out_plan,
        out_selected));
    if (!*out_selected) {
      return iree_ok_status();
    }
    out_plan->register_count = i32_lane_count;
    return iree_ok_status();
  }
  const uint32_t f32_lane_count =
      loom_amdgpu_vector_f32_lane_count(result_type);
  if (f32_lane_count != 0) {
    return loom_amdgpu_select_f32_constant_plan(
        context, value, result, f32_lane_count, out_plan, out_selected);
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_packed_16bit_float_constant_plan(
      context, result_type, value, result, out_plan, out_selected));
  if (*out_selected) {
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_select_vector_iota_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_iota_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_iota_plan_t){0};
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
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, &out_plan->descriptor,
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
  loom_attribute_t static_indices =
      loom_vector_extract_static_indices(source_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1) {
    return false;
  }
  const loom_value_slice_t indices = loom_vector_extract_indices(source_op);

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_extract_source(source_op));
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(source_type);
  if (lane_count == 0) {
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

  bool is_dynamic = false;
  uint32_t lane_offset = 0;
  loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
  if (static_indices.i64_array[0] == INT64_MIN) {
    if (indices.count != 1) {
      return false;
    }
    is_dynamic = true;
    dynamic_index = indices.values[0];
  } else {
    if (indices.count != 0 || static_indices.i64_array[0] < 0 ||
        static_indices.i64_array[0] > UINT32_MAX) {
      return false;
    }
    lane_offset = (uint32_t)static_indices.i64_array[0];
    if (lane_offset >= lane_count) {
      return false;
    }
  }

  *out_plan = (loom_amdgpu_vector_extract_plan_t){
      .source = loom_vector_extract_source(source_op),
      .dynamic_index = dynamic_index,
      .result = loom_vector_extract_result(source_op),
      .lane_offset = lane_offset,
      .lane_count = lane_count,
      .is_dynamic = is_dynamic,
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

static iree_status_t loom_amdgpu_select_index_cast_alias_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_cast_alias_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_cast_alias_plan_t){0};
  *out_selected = false;
  const loom_value_id_t source = loom_index_cast_input(source_op);
  const loom_value_id_t result = loom_index_cast_result(source_op);

  loom_type_t source_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source, &source_low_type));
  if (!loom_type_is_register(source_low_type)) {
    return iree_ok_status();
  }
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  if (!loom_type_equal(source_low_type, result_low_type)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_cast_alias_plan_t){
      .source = source,
      .result = result,
  };
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_select_scalar_trunci_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_trunci_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_scalar_trunci_plan_t){0};
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source = loom_scalar_trunci_input(source_op);
  const loom_value_id_t result = loom_scalar_trunci_result(source_op);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  if (!loom_amdgpu_type_is_i64(source_type) ||
      !loom_amdgpu_type_is_i32(result_type)) {
    return false;
  }
  *out_plan = (loom_amdgpu_scalar_trunci_plan_t){
      .source = source,
      .result = result,
  };
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
    case LOOM_OP_INDEX_CAST: {
      loom_amdgpu_cast_alias_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_select_index_cast_alias_plan(
          context, source_op, plan_data, &selected));
      if (selected) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    case LOOM_OP_SCALAR_TRUNCI: {
      loom_amdgpu_scalar_trunci_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      if (loom_amdgpu_select_scalar_trunci_plan(context, source_op,
                                                plan_data)) {
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
  if (plan->register_count == 1) {
    return loom_amdgpu_lower_u32_constant(context, source_op, &plan->descriptor,
                                          plan->imm32_attr_name_id,
                                          plan->bit_pattern, plan->result);
  }
  uint32_t lane_bit_patterns[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->register_count; ++i) {
    lane_bit_patterns[i] = plan->bit_pattern;
  }
  return loom_amdgpu_bind_vgpr_u32_lane_constants(
      context, source_op, plan->result, &plan->descriptor,
      plan->imm32_attr_name_id, lane_bit_patterns, plan->register_count);
}

static iree_status_t loom_amdgpu_lower_vector_iota(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_iota_plan_t* plan) {
  return loom_amdgpu_bind_vgpr_u32_lane_constants(
      context, source_op, plan->result, &plan->descriptor,
      plan->imm32_attr_name_id, plan->lane_bit_patterns, plan->lane_count);
}

static iree_status_t loom_amdgpu_extract_32bit_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t lane_count, uint32_t lane_offset,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (lane_count == 1) {
    *out_lane = low_source;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_source, lane_offset,
                                    lane_type, out_lane);
}

static loom_type_t loom_amdgpu_low_register_lane_type(
    const loom_module_t* module, loom_value_id_t low_value) {
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  if (!loom_type_is_register(low_type)) {
    return loom_type_none();
  }
  return loom_type_register(loom_type_register_class_id(low_type), 1);
}

static iree_status_t loom_amdgpu_lower_static_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  if (plan->lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, low_source);
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t result_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(result_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
        context, source_op, plan->result, &result_type));
  }
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_32bit_lane(
      context, source_op, low_source, plan->lane_count, plan->lane_offset,
      result_type, &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}

static iree_status_t loom_amdgpu_lower_dynamic_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  if (plan->lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, low_source);
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_vgpr_type(context, &source_lane_type));
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));

  loom_value_id_t selected_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_32bit_lane(
      context, source_op, low_source, plan->lane_count, 0, source_lane_type,
      &selected_lane));
  if (!loom_type_equal(source_lane_type, lane_type)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
        context, source_op, selected_lane, &selected_lane));
  }

  loom_value_id_t index_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, plan->dynamic_index, &index_lane));
  for (uint32_t i = 1; i < plan->lane_count; ++i) {
    loom_value_id_t ordinal = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, i, lane_type,
        &ordinal));

    const loom_value_id_t compare_operands[] = {
        index_lane,
        ordinal,
    };
    loom_op_t* compare_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        compare_operands, IREE_ARRAYSIZE(compare_operands),
        loom_make_named_attr_slice(NULL, 0), &mask_lane_type, 1, &compare_op));

    loom_value_id_t table_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_32bit_lane(
        context, source_op, low_source, plan->lane_count, i, source_lane_type,
        &table_lane));
    if (!loom_type_equal(source_lane_type, lane_type)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_b32_copy(
          context, source_op, table_lane, &table_lane));
    }
    const loom_value_id_t select_operands[] = {
        selected_lane,
        table_lane,
        loom_value_slice_get(loom_low_op_results(compare_op), 0),
    };
    loom_op_t* select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
        select_operands, IREE_ARRAYSIZE(select_operands),
        loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &select_op));
    selected_lane = loom_value_slice_get(loom_low_op_results(select_op), 0);
  }
  return loom_low_lower_bind_value(context, plan->result, selected_lane);
}

static iree_status_t loom_amdgpu_lower_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_extract_plan_t* plan) {
  return plan->is_dynamic ? loom_amdgpu_lower_dynamic_vector_extract(
                                context, source_op, plan)
                          : loom_amdgpu_lower_static_vector_extract(
                                context, source_op, plan);
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

static iree_status_t loom_amdgpu_lower_cast_alias(
    loom_low_lower_context_t* context,
    const loom_amdgpu_cast_alias_plan_t* plan) {
  return loom_low_lower_bind_value_alias(context, plan->source, plan->result);
}

static iree_status_t loom_amdgpu_lower_scalar_trunci(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_trunci_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op,
                                                  low_source, /*lane_offset=*/0,
                                                  result_type, &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
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
    case LOOM_OP_INDEX_CAST:
      return loom_amdgpu_lower_cast_alias(
          context, (const loom_amdgpu_cast_alias_plan_t*)plan.target_data);
    case LOOM_OP_SCALAR_TRUNCI:
      return loom_amdgpu_lower_scalar_trunci(
          context, source_op,
          (const loom_amdgpu_scalar_trunci_plan_t*)plan.target_data);
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
