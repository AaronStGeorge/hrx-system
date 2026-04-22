// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"

static bool loom_amdgpu_vector_reduce_descriptor_id(
    loom_scalar_type_t element_type, uint8_t kind,
    uint64_t* out_descriptor_id) {
  IREE_ASSERT_ARGUMENT(out_descriptor_id);
  *out_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  switch (element_type) {
    case LOOM_SCALAR_TYPE_I32:
      switch ((loom_vector_reduce_kind_t)kind) {
        case LOOM_VECTOR_REDUCE_KIND_ADDI:
          *out_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32;
          return true;
        case LOOM_VECTOR_REDUCE_KIND_MULI:
          *out_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32;
          return true;
        case LOOM_VECTOR_REDUCE_KIND_ANDI:
          *out_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32;
          return true;
        case LOOM_VECTOR_REDUCE_KIND_ORI:
          *out_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32;
          return true;
        case LOOM_VECTOR_REDUCE_KIND_XORI:
          *out_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32;
          return true;
        default:
          return false;
      }
    case LOOM_SCALAR_TYPE_F32:
      switch ((loom_vector_reduce_kind_t)kind) {
        case LOOM_VECTOR_REDUCE_KIND_ADDF:
          *out_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_F32;
          return true;
        case LOOM_VECTOR_REDUCE_KIND_MULF:
          *out_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_F32;
          return true;
        case LOOM_VECTOR_REDUCE_KIND_MINNUMF:
          *out_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_MIN_F32;
          return true;
        case LOOM_VECTOR_REDUCE_KIND_MAXNUMF:
          *out_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_MAX_F32;
          return true;
        default:
          return false;
      }
    default:
      return false;
  }
}

bool loom_amdgpu_select_vector_reduce_descriptor_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t* out_descriptor_id) {
  IREE_ASSERT_ARGUMENT(out_descriptor_id);
  *out_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source_input = loom_vector_reduce_input(source_op);
  const loom_value_id_t source_init = loom_vector_reduce_init(source_op);
  const loom_value_id_t source_result = loom_vector_reduce_result(source_op);
  const loom_type_t input_type = loom_module_value_type(module, source_input);
  const loom_type_t init_type = loom_module_value_type(module, source_init);
  const loom_type_t result_type = loom_module_value_type(module, source_result);
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(input_type);
  if (lane_count == 0 || !loom_type_is_scalar(init_type) ||
      !loom_type_equal(init_type, result_type) ||
      loom_type_element_type(input_type) !=
          loom_type_element_type(result_type)) {
    return false;
  }

  const loom_scalar_type_t element_type = loom_type_element_type(input_type);
  if (!loom_amdgpu_vector_reduce_descriptor_id(
          element_type, loom_vector_reduce_kind(source_op),
          out_descriptor_id)) {
    return false;
  }
  if (element_type == LOOM_SCALAR_TYPE_I32) {
    return loom_amdgpu_value_can_materialize_as_vgpr_i32(context, source_init);
  }
  return element_type == LOOM_SCALAR_TYPE_F32;
}

iree_status_t loom_amdgpu_lower_vector_reduce(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              uint64_t descriptor_id) {
  IREE_ASSERT_NE(descriptor_id, LOOM_LOW_DESCRIPTOR_ID_NONE);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source_input = loom_vector_reduce_input(source_op);
  const loom_value_id_t source_init = loom_vector_reduce_init(source_op);
  const loom_value_id_t source_result = loom_vector_reduce_result(source_op);
  const loom_type_t input_type = loom_module_value_type(module, source_input);
  const loom_scalar_type_t element_type = loom_type_element_type(input_type);
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(input_type);
  IREE_ASSERT_GT(lane_count, 0);

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  bool result_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &result_is_vgpr));
  IREE_ASSERT(result_is_vgpr);

  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  if (element_type == LOOM_SCALAR_TYPE_I32) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, source_init, &accumulator));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, source_init, &accumulator));
  }

  loom_value_id_t low_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_input, &low_input));
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t lane = low_input;
    if (lane_count != 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_input, i, result_type, &lane));
    }
    loom_value_id_t operands[] = {
        accumulator,
        lane,
    };
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, descriptor_id, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(NULL, 0), &result_type, 1, &lane_op));
    accumulator = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }
  return loom_low_lower_bind_value(context, source_result, accumulator);
}
