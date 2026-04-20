// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/hal_kernel_abi.h"

#define LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES 4u

static bool loom_amdgpu_type_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static bool loom_amdgpu_type_is_address_scalar(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         (loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX ||
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET);
}

static bool loom_amdgpu_type_is_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

static bool loom_amdgpu_type_is_vector_1xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 1;
}

static bool loom_amdgpu_type_is_vector_4xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_amdgpu_type_is_vector_1xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 1;
}

static bool loom_amdgpu_type_is_vector_4xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_amdgpu_type_is_vector_1x32_or_4x32(loom_type_t type) {
  return loom_amdgpu_type_is_vector_1xi32(type) ||
         loom_amdgpu_type_is_vector_4xi32(type) ||
         loom_amdgpu_type_is_vector_1xf32(type) ||
         loom_amdgpu_type_is_vector_4xf32(type);
}

static uint32_t loom_amdgpu_vector_32bit_lane_count(loom_type_t type) {
  if (loom_amdgpu_type_is_vector_1xi32(type) ||
      loom_amdgpu_type_is_vector_1xf32(type)) {
    return 1;
  }
  if (loom_amdgpu_type_is_vector_4xi32(type) ||
      loom_amdgpu_type_is_vector_4xf32(type)) {
    return 4;
  }
  return 0;
}

static uint32_t loom_amdgpu_vector_i32_lane_count(loom_type_t type) {
  if (loom_amdgpu_type_is_vector_1xi32(type)) {
    return 1;
  }
  if (loom_amdgpu_type_is_vector_4xi32(type)) {
    return 4;
  }
  return 0;
}

static uint32_t loom_amdgpu_vector_f32_lane_count(loom_type_t type) {
  if (loom_amdgpu_type_is_vector_1xf32(type)) {
    return 1;
  }
  if (loom_amdgpu_type_is_vector_4xf32(type)) {
    return 4;
  }
  return 0;
}

static bool loom_amdgpu_type_is_32bit_view(loom_type_t type) {
  return loom_type_is_view(type) &&
         (loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 ||
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32);
}

static bool loom_amdgpu_value_is_i32(loom_low_lower_context_t* context,
                                     loom_value_id_t value_id) {
  return loom_amdgpu_type_is_i32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static bool loom_amdgpu_value_is_address_scalar(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_address_scalar(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static bool loom_amdgpu_value_is_f32(loom_low_lower_context_t* context,
                                     loom_value_id_t value_id) {
  return loom_amdgpu_type_is_f32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static bool loom_amdgpu_value_is_vector_1x32_or_4x32(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_vector_1x32_or_4x32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static bool loom_amdgpu_value_is_32bit_view(loom_low_lower_context_t* context,
                                            loom_value_id_t value_id) {
  return loom_amdgpu_type_is_32bit_view(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static iree_status_t loom_amdgpu_make_register_type(
    loom_low_lower_context_t* context, iree_string_view_t register_class,
    uint32_t unit_count, loom_type_t* out_type) {
  loom_string_id_t register_class_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                register_class, &register_class_id));
  *out_type = loom_type_register(register_class_id, unit_count);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_make_sgpr_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, IREE_SV("amdgpu.sgpr"), 1,
                                        out_type);
}

static iree_status_t loom_amdgpu_make_sgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, IREE_SV("amdgpu.sgpr"),
                                        unit_count, out_type);
}

static iree_status_t loom_amdgpu_make_vgpr_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, IREE_SV("amdgpu.vgpr"), 1,
                                        out_type);
}

static bool loom_amdgpu_low_type_register_class_is(
    loom_low_lower_context_t* context, loom_type_t type,
    iree_string_view_t expected_register_class) {
  if (!loom_type_is_register(type)) {
    return false;
  }
  loom_string_id_t register_class_id = loom_type_register_class_id(type);
  loom_module_t* module = loom_low_lower_context_module(context);
  if (register_class_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[register_class_id],
                                expected_register_class);
}

static bool loom_amdgpu_value_prefers_vgpr(loom_low_lower_context_t* context,
                                           loom_value_id_t source_value_id) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (source_value_id >= module->values.count) {
    return false;
  }

  loom_type_t source_type = loom_module_value_type(module, source_value_id);
  if (loom_amdgpu_type_is_f32(source_type) ||
      loom_amdgpu_type_is_vector_1x32_or_4x32(source_type)) {
    return true;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL) {
    return false;
  }

  switch (defining_op->kind) {
    case LOOM_OP_KERNEL_WORKITEM_ID:
      return loom_kernel_workitem_id_dimension(defining_op) ==
             LOOM_KERNEL_WORKITEM_ID_DIMENSION_X;
    case LOOM_OP_VECTOR_EXTRACT:
      return true;
    case LOOM_OP_SCALAR_ADDI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_addi_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_addi_rhs(defining_op));
    case LOOM_OP_SCALAR_SUBI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_subi_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_subi_rhs(defining_op));
    case LOOM_OP_SCALAR_MULI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_muli_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_muli_rhs(defining_op));
    case LOOM_OP_SCALAR_ANDI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_andi_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_andi_rhs(defining_op));
    case LOOM_OP_SCALAR_ORI:
      return loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_ori_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_ori_rhs(defining_op));
    case LOOM_OP_SCALAR_XORI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_xori_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_xori_rhs(defining_op));
    case LOOM_OP_SCALAR_SHLI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_shli_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_shli_rhs(defining_op));
    case LOOM_OP_SCALAR_SHRSI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_shrsi_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_shrsi_rhs(defining_op));
    case LOOM_OP_SCALAR_SHRUI:
      return loom_amdgpu_value_prefers_vgpr(
                 context, loom_scalar_shrui_lhs(defining_op)) ||
             loom_amdgpu_value_prefers_vgpr(context,
                                            loom_scalar_shrui_rhs(defining_op));
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_map_type(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_type_t source_type,
                                          loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_amdgpu_type_is_i32(source_type)) {
    return loom_amdgpu_make_sgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_address_scalar(source_type)) {
    return loom_amdgpu_make_sgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_vector_1xi32(source_type) ||
      loom_amdgpu_type_is_vector_1xf32(source_type)) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_vector_4xi32(source_type) ||
      loom_amdgpu_type_is_vector_4xf32(source_type)) {
    return loom_amdgpu_make_register_type(context, IREE_SV("amdgpu.vgpr"), 4,
                                          out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("AMDGPU lowering currently supports only i32 scalar values, "
              "address scalar values, vector<1xi32>/vector<1xf32> VGPR "
              "values, and vector<4xi32>/vector<4xf32> VGPR ranges"));
}

static iree_status_t loom_amdgpu_map_value(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_value_id_t source_value_id,
                                           loom_type_t source_type,
                                           loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_amdgpu_type_is_f32(source_type)) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  if ((loom_amdgpu_type_is_i32(source_type) ||
       loom_amdgpu_type_is_address_scalar(source_type)) &&
      loom_amdgpu_value_prefers_vgpr(context, source_value_id)) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  return loom_amdgpu_map_type(user_data, context, source_op, source_type,
                              out_low_type);
}

static iree_status_t loom_amdgpu_make_hal_buffer_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  loom_string_id_t hal_buffer_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(loom_low_lower_context_module(context),
                                IREE_SV("hal.buffer"), &hal_buffer_id));
  *out_type = loom_type_dialect_opaque(hal_buffer_id);
  return iree_ok_status();
}

static uint32_t loom_amdgpu_hal_buffer_resource_index(
    loom_low_lower_context_t* context, uint16_t source_argument_index) {
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids = loom_func_like_arg_ids(
      loom_low_lower_context_source_function(context), &argument_count);
  uint32_t resource_index = 0;
  for (uint16_t i = 0; i < source_argument_index && i < argument_count; ++i) {
    loom_type_t type = loom_module_value_type(
        loom_low_lower_context_module(context), argument_ids[i]);
    if (loom_type_is_buffer(type)) {
      ++resource_index;
    }
  }
  return resource_index;
}

static iree_status_t loom_amdgpu_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)user_data;
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  if (bundle->export_plan->abi_kind == LOOM_TARGET_ABI_HAL_KERNEL &&
      loom_type_is_buffer(source_type)) {
    loom_type_t resource_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 4, &resource_type));
    loom_type_t semantic_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_hal_buffer_type(context, &semantic_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE,
        .abi_type = resource_type,
        .resource_kind = LOOM_LOW_ABI_RESOURCE_KIND_HAL_BUFFER_RESOURCE,
        .resource_index = loom_amdgpu_hal_buffer_resource_index(
            context, source_argument_index),
        .resource_semantic_type = semantic_type,
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_semantic_type = loom_type_none(),
  };
  return loom_amdgpu_map_value(user_data, context, source_function_op,
                               source_argument_id, source_type,
                               &out_argument->abi_type);
}

static bool loom_amdgpu_attr_is_i32_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_I64 && value.i64 >= INT32_MIN &&
         value.i64 <= INT32_MAX;
}

static bool loom_amdgpu_attr_is_f32_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_F64;
}

static uint32_t loom_amdgpu_attr_f32_bit_pattern(loom_attribute_t value) {
  const float f32_value = (float)loom_attr_as_f64(value);
  uint32_t bit_pattern = 0;
  memcpy(&bit_pattern, &f32_value, sizeof(bit_pattern));
  return bit_pattern;
}

static bool loom_amdgpu_value_is_exact_index_constant(
    loom_low_lower_context_t* context, loom_value_id_t value_id,
    int64_t expected_value) {
  const loom_value_t* value =
      loom_module_value(loom_low_lower_context_module(context), value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_index_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_index_constant_value(defining_op);
  return attr.kind == LOOM_ATTR_I64 && attr.i64 == expected_value;
}

static bool loom_amdgpu_value_as_i32_constant(loom_low_lower_context_t* context,
                                              loom_value_id_t value_id,
                                              int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  if (!loom_amdgpu_value_is_i32(context, value_id)) {
    return false;
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_scalar_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_scalar_constant_value(defining_op);
  if (!loom_amdgpu_attr_is_i32_immediate(attr)) {
    return false;
  }
  *out_value = attr.i64;
  return true;
}

static bool loom_amdgpu_value_can_materialize_as_vgpr_i32(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  int64_t unused_value = 0;
  return loom_amdgpu_value_prefers_vgpr(context, value_id) ||
         loom_amdgpu_value_as_i32_constant(context, value_id, &unused_value);
}

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

static bool loom_amdgpu_can_lower_buffer_view(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  return loom_amdgpu_value_is_32bit_view(context,
                                         loom_buffer_view_result(source_op)) &&
         loom_amdgpu_value_is_exact_index_constant(
             context, loom_buffer_view_byte_offset(source_op), 0);
}

static bool loom_amdgpu_value_is_workitem_id_x(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  return defining_op && loom_kernel_workitem_id_isa(defining_op) &&
         loom_kernel_workitem_id_dimension(defining_op) ==
             LOOM_KERNEL_WORKITEM_ID_DIMENSION_X;
}

typedef struct loom_amdgpu_memory_access_t {
  loom_value_id_t dynamic_index;
  uint32_t dynamic_index_byte_stride;
  int64_t static_byte_offset;
  uint32_t vgpr_count;
} loom_amdgpu_memory_access_t;

static bool loom_amdgpu_memory_access_find_dynamic_axis(
    loom_attribute_t static_indices, uint8_t* out_dynamic_axis) {
  IREE_ASSERT_ARGUMENT(out_dynamic_axis);
  *out_dynamic_axis = UINT8_MAX;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return false;
  }
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (static_indices.i64_array[i] != INT64_MIN) {
      continue;
    }
    if (*out_dynamic_axis != UINT8_MAX || i > UINT8_MAX) {
      return false;
    }
    *out_dynamic_axis = (uint8_t)i;
  }
  return true;
}

static bool loom_amdgpu_memory_access_static_byte_offset(
    const loom_vector_memory_access_t* vector_access,
    loom_attribute_t static_indices, uint8_t dynamic_axis,
    int64_t* out_static_byte_offset) {
  IREE_ASSERT_ARGUMENT(out_static_byte_offset);
  *out_static_byte_offset = 0;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return false;
  }
  if (static_indices.count > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK) {
    return false;
  }

  int64_t static_origin[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (i == dynamic_axis) {
      if (static_indices.i64_array[i] != INT64_MIN) {
        return false;
      }
      continue;
    }
    if (static_indices.i64_array[i] == INT64_MIN) {
      return false;
    }
    static_origin[i] = static_indices.i64_array[i];
  }
  loom_attribute_t static_origin_attr =
      loom_attr_i64_array(static_origin, static_indices.count);
  int64_t lane_indices[] = {0};
  return loom_vector_memory_access_static_lane_byte_offset(
      vector_access, static_origin_attr, lane_indices,
      IREE_ARRAYSIZE(lane_indices), out_static_byte_offset);
}

static bool loom_amdgpu_memory_access_static_byte_offset_is_usable(
    int64_t static_byte_offset, uint32_t vgpr_count) {
  if (static_byte_offset < 0) {
    return false;
  }
  return vgpr_count != 4 || (static_byte_offset & 15) == 0;
}

static bool loom_amdgpu_memory_access_select(
    loom_low_lower_context_t* context, loom_value_slice_t dynamic_indices,
    loom_attribute_t static_indices, loom_type_t view_type,
    loom_type_t vector_type, loom_amdgpu_memory_access_t* out_access) {
  IREE_ASSERT_ARGUMENT(out_access);
  *out_access = (loom_amdgpu_memory_access_t){
      .dynamic_index = LOOM_VALUE_ID_INVALID,
  };

  loom_vector_memory_access_t vector_access;
  if (!loom_vector_memory_access_describe(
          loom_low_lower_context_module(context), view_type, vector_type,
          &vector_access)) {
    return false;
  }
  if (vector_access.layout_kind != LOOM_VECTOR_MEMORY_LAYOUT_DENSE ||
      vector_access.static_element_byte_count != 4 ||
      vector_access.vector_rank != 1) {
    return false;
  }

  out_access->vgpr_count = loom_amdgpu_vector_32bit_lane_count(vector_type);
  if (out_access->vgpr_count == 0) {
    return false;
  }

  int64_t vector_axis_stride = 0;
  if (!loom_vector_memory_access_static_axis_stride(
          &vector_access, vector_access.first_vector_axis,
          &vector_axis_stride) ||
      vector_axis_stride != 1) {
    return false;
  }

  if (dynamic_indices.count == 0) {
    int64_t lane_indices[] = {0};
    if (!loom_vector_memory_access_static_lane_byte_offset(
            &vector_access, static_indices, lane_indices,
            IREE_ARRAYSIZE(lane_indices), &out_access->static_byte_offset)) {
      return false;
    }
    if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
            out_access->static_byte_offset, out_access->vgpr_count)) {
      return false;
    }
    return true;
  }

  if (dynamic_indices.count != 1) {
    return false;
  }
  uint8_t dynamic_axis = UINT8_MAX;
  if (!loom_amdgpu_memory_access_find_dynamic_axis(static_indices,
                                                   &dynamic_axis) ||
      dynamic_axis == UINT8_MAX || dynamic_axis >= vector_access.view_rank) {
    return false;
  }
  if (!loom_amdgpu_value_is_workitem_id_x(context, dynamic_indices.values[0])) {
    return false;
  }

  int64_t axis_stride = 0;
  if (!loom_vector_memory_access_static_axis_stride(
          &vector_access, dynamic_axis, &axis_stride)) {
    return false;
  }
  int64_t dynamic_index_byte_stride = 0;
  if (!iree_checked_mul_i64(axis_stride,
                            vector_access.static_element_byte_count,
                            &dynamic_index_byte_stride) ||
      dynamic_index_byte_stride < 0 || dynamic_index_byte_stride > UINT32_MAX) {
    return false;
  }
  if (out_access->vgpr_count == 4 && (dynamic_index_byte_stride & 15) != 0) {
    return false;
  }
  int64_t static_byte_offset = 0;
  if (!loom_amdgpu_memory_access_static_byte_offset(
          &vector_access, static_indices, dynamic_axis, &static_byte_offset)) {
    return false;
  }
  if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
          static_byte_offset, out_access->vgpr_count)) {
    return false;
  }
  out_access->dynamic_index = dynamic_indices.values[0];
  out_access->dynamic_index_byte_stride = (uint32_t)dynamic_index_byte_stride;
  out_access->static_byte_offset = static_byte_offset;
  return true;
}

static bool loom_amdgpu_load_memory_access_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_t* out_access) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  return loom_amdgpu_memory_access_select(
      context, loom_vector_load_indices(source_op),
      loom_vector_load_static_indices(source_op),
      loom_module_value_type(module, loom_vector_load_view(source_op)),
      loom_module_value_type(module, loom_vector_load_result(source_op)),
      out_access);
}

static bool loom_amdgpu_store_memory_access_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_access_t* out_access) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  return loom_amdgpu_memory_access_select(
      context, loom_vector_store_indices(source_op),
      loom_vector_store_static_indices(source_op),
      loom_module_value_type(module, loom_vector_store_view(source_op)),
      loom_module_value_type(module, loom_vector_store_value(source_op)),
      out_access);
}

static bool loom_amdgpu_can_lower_vector_load(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  loom_amdgpu_memory_access_t access;
  return loom_amdgpu_value_is_32bit_view(context,
                                         loom_vector_load_view(source_op)) &&
         loom_amdgpu_value_is_vector_1x32_or_4x32(
             context, loom_vector_load_result(source_op)) &&
         loom_amdgpu_load_memory_access_select(context, source_op, &access);
}

static bool loom_amdgpu_can_lower_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_amdgpu_memory_access_t access;
  return loom_amdgpu_value_is_vector_1x32_or_4x32(
             context, loom_vector_store_value(source_op)) &&
         loom_amdgpu_value_is_32bit_view(context,
                                         loom_vector_store_view(source_op)) &&
         loom_amdgpu_store_memory_access_select(context, source_op, &access);
}

static bool loom_amdgpu_can_lower_i32_binary(loom_low_lower_context_t* context,
                                             loom_value_id_t lhs,
                                             loom_value_id_t rhs,
                                             loom_value_id_t result,
                                             bool allow_uniform_result) {
  if (!loom_amdgpu_value_is_i32(context, lhs) ||
      !loom_amdgpu_value_is_i32(context, rhs) ||
      !loom_amdgpu_value_is_i32(context, result)) {
    return false;
  }
  const bool lhs_vgpr = loom_amdgpu_value_prefers_vgpr(context, lhs);
  const bool rhs_vgpr = loom_amdgpu_value_prefers_vgpr(context, rhs);
  const bool result_vgpr = loom_amdgpu_value_prefers_vgpr(context, result);
  if (result_vgpr) {
    return loom_amdgpu_value_can_materialize_as_vgpr_i32(context, lhs) &&
           loom_amdgpu_value_can_materialize_as_vgpr_i32(context, rhs);
  }
  return allow_uniform_result && !lhs_vgpr && !rhs_vgpr;
}

static bool loom_amdgpu_can_lower_vector_i32_binary(
    loom_low_lower_context_t* context, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_value_id_t result) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t result_type = loom_module_value_type(module, result);
  return loom_amdgpu_vector_i32_lane_count(result_type) != 0 &&
         loom_type_equal(loom_module_value_type(module, lhs), result_type) &&
         loom_type_equal(loom_module_value_type(module, rhs), result_type);
}

static bool loom_amdgpu_can_lower_f32_binary(loom_low_lower_context_t* context,
                                             loom_value_id_t lhs,
                                             loom_value_id_t rhs,
                                             loom_value_id_t result) {
  return loom_amdgpu_value_is_f32(context, lhs) &&
         loom_amdgpu_value_is_f32(context, rhs) &&
         loom_amdgpu_value_is_f32(context, result);
}

static bool loom_amdgpu_can_lower_vector_f32_binary(
    loom_low_lower_context_t* context, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_value_id_t result) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t result_type = loom_module_value_type(module, result);
  return (loom_amdgpu_type_is_vector_1xf32(result_type) ||
          loom_amdgpu_type_is_vector_4xf32(result_type)) &&
         loom_type_equal(loom_module_value_type(module, lhs), result_type) &&
         loom_type_equal(loom_module_value_type(module, rhs), result_type);
}

static bool loom_amdgpu_can_lower_vector_f32_ternary(
    loom_low_lower_context_t* context, loom_value_id_t a, loom_value_id_t b,
    loom_value_id_t c, loom_value_id_t result) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t result_type = loom_module_value_type(module, result);
  return (loom_amdgpu_type_is_vector_1xf32(result_type) ||
          loom_amdgpu_type_is_vector_4xf32(result_type)) &&
         loom_type_equal(loom_module_value_type(module, a), result_type) &&
         loom_type_equal(loom_module_value_type(module, b), result_type) &&
         loom_type_equal(loom_module_value_type(module, c), result_type);
}

static bool loom_amdgpu_can_lower_f32_ternary(loom_low_lower_context_t* context,
                                              loom_value_id_t a,
                                              loom_value_id_t b,
                                              loom_value_id_t c,
                                              loom_value_id_t result) {
  return loom_amdgpu_value_is_f32(context, a) &&
         loom_amdgpu_value_is_f32(context, b) &&
         loom_amdgpu_value_is_f32(context, c) &&
         loom_amdgpu_value_is_f32(context, result);
}

static bool loom_amdgpu_vector_extract_select(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              uint32_t* out_lane_offset) {
  IREE_ASSERT_ARGUMENT(out_lane_offset);
  *out_lane_offset = 0;
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
  *out_lane_offset = lane_offset;
  return true;
}

static bool loom_amdgpu_can_lower_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  uint32_t unused_lane_offset = 0;
  return loom_amdgpu_vector_extract_select(context, source_op,
                                           &unused_lane_offset);
}

static bool loom_amdgpu_can_lower_vector_from_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source_result =
      loom_vector_from_elements_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, source_result);
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(result_type);
  if (lane_count == 0) {
    return false;
  }
  loom_value_slice_t elements = loom_vector_from_elements_elements(source_op);
  if (elements.count != lane_count) {
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

static bool loom_amdgpu_can_lower_workitem_id(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  return loom_kernel_workitem_id_dimension(source_op) ==
             LOOM_KERNEL_WORKITEM_ID_DIMENSION_X &&
         loom_amdgpu_value_is_address_scalar(
             context, loom_kernel_workitem_id_result(source_op));
}

static iree_status_t loom_amdgpu_can_lower_op(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              bool* out_handled) {
  (void)user_data;
  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
      *out_handled = loom_amdgpu_can_lower_index_constant(context, source_op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_CONSTANT:
      *out_handled = loom_amdgpu_can_lower_scalar_constant(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_CONSTANT:
      *out_handled = loom_amdgpu_can_lower_vector_constant(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_IOTA:
      *out_handled = loom_amdgpu_can_lower_vector_iota(context, source_op);
      return iree_ok_status();
    case LOOM_OP_BUFFER_VIEW:
      *out_handled = loom_amdgpu_can_lower_buffer_view(context, source_op);
      return iree_ok_status();
    case LOOM_OP_KERNEL_WORKITEM_ID:
      *out_handled = loom_amdgpu_can_lower_workitem_id(context, source_op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ADDI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_addi_lhs(source_op),
          loom_scalar_addi_rhs(source_op), loom_scalar_addi_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SUBI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_subi_lhs(source_op),
          loom_scalar_subi_rhs(source_op), loom_scalar_subi_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_MULI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_muli_lhs(source_op),
          loom_scalar_muli_rhs(source_op), loom_scalar_muli_result(source_op),
          /*allow_uniform_result=*/false);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ANDI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_andi_lhs(source_op),
          loom_scalar_andi_rhs(source_op), loom_scalar_andi_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ORI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_ori_lhs(source_op),
          loom_scalar_ori_rhs(source_op), loom_scalar_ori_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_XORI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_xori_lhs(source_op),
          loom_scalar_xori_rhs(source_op), loom_scalar_xori_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHLI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_shli_lhs(source_op),
          loom_scalar_shli_rhs(source_op), loom_scalar_shli_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHRSI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_shrsi_lhs(source_op),
          loom_scalar_shrsi_rhs(source_op), loom_scalar_shrsi_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_SHRUI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_shrui_lhs(source_op),
          loom_scalar_shrui_rhs(source_op), loom_scalar_shrui_result(source_op),
          /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ADDF:
      *out_handled = loom_amdgpu_can_lower_f32_binary(
          context, loom_scalar_addf_lhs(source_op),
          loom_scalar_addf_rhs(source_op), loom_scalar_addf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_SCALAR_SUBF:
      *out_handled = loom_amdgpu_can_lower_f32_binary(
          context, loom_scalar_subf_lhs(source_op),
          loom_scalar_subf_rhs(source_op), loom_scalar_subf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_SCALAR_MULF:
      *out_handled = loom_amdgpu_can_lower_f32_binary(
          context, loom_scalar_mulf_lhs(source_op),
          loom_scalar_mulf_rhs(source_op), loom_scalar_mulf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_SCALAR_FMAF:
      *out_handled = loom_amdgpu_can_lower_f32_ternary(
          context, loom_scalar_fmaf_a(source_op), loom_scalar_fmaf_b(source_op),
          loom_scalar_fmaf_c(source_op), loom_scalar_fmaf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_ADDI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_addi_lhs(source_op),
          loom_vector_addi_rhs(source_op), loom_vector_addi_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_SUBI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_subi_lhs(source_op),
          loom_vector_subi_rhs(source_op), loom_vector_subi_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_MULI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_muli_lhs(source_op),
          loom_vector_muli_rhs(source_op), loom_vector_muli_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_ANDI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_andi_lhs(source_op),
          loom_vector_andi_rhs(source_op), loom_vector_andi_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_ORI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_ori_lhs(source_op),
          loom_vector_ori_rhs(source_op), loom_vector_ori_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_XORI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_xori_lhs(source_op),
          loom_vector_xori_rhs(source_op), loom_vector_xori_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_SHLI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_shli_lhs(source_op),
          loom_vector_shli_rhs(source_op), loom_vector_shli_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_SHRSI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_shrsi_lhs(source_op),
          loom_vector_shrsi_rhs(source_op),
          loom_vector_shrsi_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_SHRUI:
      *out_handled = loom_amdgpu_can_lower_vector_i32_binary(
          context, loom_vector_shrui_lhs(source_op),
          loom_vector_shrui_rhs(source_op),
          loom_vector_shrui_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_ADDF:
      *out_handled = loom_amdgpu_can_lower_vector_f32_binary(
          context, loom_vector_addf_lhs(source_op),
          loom_vector_addf_rhs(source_op), loom_vector_addf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_SUBF:
      *out_handled = loom_amdgpu_can_lower_vector_f32_binary(
          context, loom_vector_subf_lhs(source_op),
          loom_vector_subf_rhs(source_op), loom_vector_subf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_MULF:
      *out_handled = loom_amdgpu_can_lower_vector_f32_binary(
          context, loom_vector_mulf_lhs(source_op),
          loom_vector_mulf_rhs(source_op), loom_vector_mulf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_FMAF:
      *out_handled = loom_amdgpu_can_lower_vector_f32_ternary(
          context, loom_vector_fmaf_a(source_op), loom_vector_fmaf_b(source_op),
          loom_vector_fmaf_c(source_op), loom_vector_fmaf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_EXTRACT:
      *out_handled = loom_amdgpu_can_lower_vector_extract(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
      *out_handled =
          loom_amdgpu_can_lower_vector_from_elements(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_LOAD:
      *out_handled = loom_amdgpu_can_lower_vector_load(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_STORE:
      *out_handled = loom_amdgpu_can_lower_vector_store(context, source_op);
      return iree_ok_status();
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_intern(loom_low_lower_context_t* context,
                                        iree_string_view_t string,
                                        loom_string_id_t* out_string_id) {
  return loom_module_intern_string(loom_low_lower_context_module(context),
                                   string, out_string_id);
}

static iree_status_t loom_amdgpu_low_result_type(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, loom_type_t* out_low_type) {
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_result, out_low_type));
  if (!loom_type_is_register(*out_low_type)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU source type did not map to a register");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_low_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t opcode, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attrs,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(out_op);
  *out_op = NULL;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, opcode, &opcode_id));
  return loom_low_op_build(loom_low_lower_context_builder(context), opcode_id,
                           operands, operand_count, attrs, result_types,
                           result_count,
                           /*tied_results=*/NULL, /*tied_result_count=*/0,
                           source_op->location, out_op);
}

static iree_status_t loom_amdgpu_emit_const_u32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t opcode, uint32_t value, loom_type_t result_type,
    loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_string_id_t opcode_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, opcode, &opcode_id));
  loom_string_id_t value_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("imm32"), &value_name_id));
  loom_named_attr_t attrs[] = {
      {
          .name_id = value_name_id,
          .value = loom_attr_i64(value),
      },
  };
  loom_op_t* low_const = NULL;
  IREE_RETURN_IF_ERROR(loom_low_const_build(
      loom_low_lower_context_builder(context), opcode_id,
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), result_type,
      source_op->location, &low_const));
  *out_value_id = loom_low_const_result(low_const);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_i32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  IREE_ASSERT_ARGUMENT(out_low_value);
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  if (loom_amdgpu_low_type_register_class_is(context, low_type,
                                             IREE_SV("amdgpu.vgpr"))) {
    *out_low_value = low_value;
    return iree_ok_status();
  }

  int64_t value = 0;
  if (!loom_amdgpu_value_as_i32_constant(context, source_value, &value)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted AMDGPU i32 value that cannot "
                            "materialize as a VGPR operand");
  }
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(
      context, source_op, IREE_SV("amdgpu.v_mov_b32"), (uint32_t)(int32_t)value,
      vgpr_type, out_low_value);
}

static iree_status_t loom_amdgpu_bind_vgpr_u32_lane_constants(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, const uint32_t* lane_bit_patterns,
    uint32_t lane_count) {
  IREE_ASSERT_ARGUMENT(lane_bit_patterns);
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "constant lane count");
  }

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t low_lane_values[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, IREE_SV("amdgpu.v_mov_b32"), lane_bit_patterns[i],
        lane_type, &low_lane_values[i]));
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
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "constant lane count");
  }

  uint32_t lane_bit_patterns[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    if (lane_values[i] < INT32_MIN || lane_values[i] > INT32_MAX) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "preflight accepted out-of-range AMDGPU vector "
                              "constant lane");
    }
    lane_bit_patterns[i] = (uint32_t)(int32_t)lane_values[i];
  }
  return loom_amdgpu_bind_vgpr_u32_lane_constants(
      context, source_op, source_result, lane_bit_patterns, lane_count);
}

static iree_status_t loom_amdgpu_bind_vgpr_f32_lane_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, uint32_t lane_bit_pattern,
    uint32_t lane_count) {
  uint32_t lane_bit_patterns[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    lane_bit_patterns[i] = lane_bit_pattern;
  }
  return loom_amdgpu_bind_vgpr_u32_lane_constants(
      context, source_op, source_result, lane_bit_patterns, lane_count);
}

static iree_status_t loom_amdgpu_lower_u32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t opcode, uint32_t bit_pattern,
    loom_value_id_t source_result) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, opcode, bit_pattern, result_type, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static iree_status_t loom_amdgpu_lower_i32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t opcode, loom_attribute_t source_attr,
    loom_value_id_t source_result) {
  const int64_t source_value = source_attr.i64;
  return loom_amdgpu_lower_u32_constant(context, source_op, opcode,
                                        (uint32_t)(int32_t)source_value,
                                        source_result);
}

static iree_status_t loom_amdgpu_lower_index_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_constant(context, source_op,
                                        IREE_SV("amdgpu.s_mov_b32"),
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
        context, source_op, IREE_SV("amdgpu.v_mov_b32"),
        loom_amdgpu_attr_f32_bit_pattern(loom_scalar_constant_value(source_op)),
        source_result);
  }

  const iree_string_view_t opcode =
      loom_amdgpu_value_prefers_vgpr(context, source_result)
          ? IREE_SV("amdgpu.v_mov_b32")
          : IREE_SV("amdgpu.s_mov_b32");
  return loom_amdgpu_lower_i32_constant(context, source_op, opcode,
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
        context, source_op, IREE_SV("amdgpu.v_mov_b32"),
        loom_vector_constant_value(source_op), source_result);
  }
  if (i32_lane_count > 1) {
    const int64_t source_value = loom_vector_constant_value(source_op).i64;
    int64_t lane_values[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
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

  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "preflight accepted unsupported AMDGPU vector "
                          "constant lane count");
}

static iree_status_t loom_amdgpu_lower_vector_iota(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source_result = loom_vector_iota_result(source_op);
  const uint32_t lane_count = loom_amdgpu_vector_i32_lane_count(
      loom_module_value_type(module, source_result));
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "iota lane count");
  }
  int64_t base = 0;
  int64_t step = 0;
  if (!loom_amdgpu_value_as_i32_constant(
          context, loom_vector_iota_base(source_op), &base) ||
      !loom_amdgpu_value_as_i32_constant(
          context, loom_vector_iota_step(source_op), &step)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "iota base/step");
  }

  int64_t lane_values[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    if (!loom_amdgpu_iota_i32_lane_value(base, step, i, &lane_values[i])) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "preflight accepted out-of-range AMDGPU vector "
                              "iota lane");
    }
  }
  return loom_amdgpu_bind_vgpr_i32_lane_constants(
      context, source_op, source_result, lane_values, lane_count);
}

static iree_status_t loom_amdgpu_lower_binary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t opcode, loom_value_id_t source_lhs,
    loom_value_id_t source_rhs, loom_value_id_t source_result) {
  loom_value_id_t low_operands[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_lhs, &low_operands[0]));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_rhs, &low_operands[1]));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, opcode, low_operands, IREE_ARRAYSIZE(low_operands),
      loom_make_named_attr_slice(NULL, 0), &result_type, 1, &low_op));
  return loom_low_lower_bind_value(
      context, source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_amdgpu_lower_ternary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t opcode, loom_value_id_t source_a,
    loom_value_id_t source_b, loom_value_id_t source_c,
    loom_value_id_t source_result) {
  loom_value_id_t low_operands[3] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_a, &low_operands[0]));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_b, &low_operands[1]));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_c, &low_operands[2]));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, opcode, low_operands, IREE_ARRAYSIZE(low_operands),
      loom_make_named_attr_slice(NULL, 0), &result_type, 1, &low_op));
  return loom_low_lower_bind_value(
      context, source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_amdgpu_emit_low_slice(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t offset, loom_type_t result_type,
    loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(
      loom_low_lower_context_builder(context), low_source, offset, result_type,
      source_op->location, &slice_op));
  *out_value_id = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_binary_op_ordered(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t opcode, loom_value_id_t source_lhs,
    loom_value_id_t source_rhs, loom_value_id_t source_result,
    bool swap_operands) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(
      loom_module_value_type(module, source_result));
  if (lane_count == 1) {
    if (swap_operands) {
      return loom_amdgpu_lower_binary_op(context, source_op, opcode, source_rhs,
                                         source_lhs, source_result);
    }
    return loom_amdgpu_lower_binary_op(context, source_op, opcode, source_lhs,
                                       source_rhs, source_result);
  }
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "binary lane count");
  }

  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_lhs, &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_rhs, &low_rhs));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t lane_lhs = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, low_lhs,
                                                    i, lane_type, &lane_lhs));
    loom_value_id_t lane_rhs = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, low_rhs,
                                                    i, lane_type, &lane_rhs));
    loom_value_id_t operands[] = {
        swap_operands ? lane_rhs : lane_lhs,
        swap_operands ? lane_lhs : lane_rhs,
    };
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, opcode, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &lane_op));
    lane_results[i] = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_vector_binary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t opcode, loom_value_id_t source_lhs,
    loom_value_id_t source_rhs, loom_value_id_t source_result) {
  return loom_amdgpu_lower_vector_binary_op_ordered(
      context, source_op, opcode, source_lhs, source_rhs, source_result,
      /*swap_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_vector_ternary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t opcode, loom_value_id_t source_a,
    loom_value_id_t source_b, loom_value_id_t source_c,
    loom_value_id_t source_result) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(
      loom_module_value_type(module, source_result));
  if (lane_count == 1) {
    return loom_amdgpu_lower_ternary_op(context, source_op, opcode, source_a,
                                        source_b, source_c, source_result);
  }
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "ternary lane count");
  }

  loom_value_id_t low_a = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, source_a, &low_a));
  loom_value_id_t low_b = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, source_b, &low_b));
  loom_value_id_t low_c = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, source_c, &low_c));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t lane_a = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, low_a,
                                                    i, lane_type, &lane_a));
    loom_value_id_t lane_b = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, low_b,
                                                    i, lane_type, &lane_b));
    loom_value_id_t lane_c = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, low_c,
                                                    i, lane_type, &lane_c));
    loom_value_id_t operands[] = {
        lane_a,
        lane_b,
        lane_c,
    };
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, opcode, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &lane_op));
    lane_results[i] = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_i32_binary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t scalar_opcode, iree_string_view_t vector_opcode,
    loom_value_id_t source_lhs, loom_value_id_t source_rhs,
    loom_value_id_t source_result, bool swap_vector_operands) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  if (loom_amdgpu_low_type_register_class_is(context, result_type,
                                             IREE_SV("amdgpu.vgpr"))) {
    const loom_value_id_t first_source =
        swap_vector_operands ? source_rhs : source_lhs;
    const loom_value_id_t second_source =
        swap_vector_operands ? source_lhs : source_rhs;
    loom_value_id_t low_operands[2] = {
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, first_source, &low_operands[0]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, second_source, &low_operands[1]));
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, vector_opcode, low_operands,
        IREE_ARRAYSIZE(low_operands), loom_make_named_attr_slice(NULL, 0),
        &result_type, 1, &low_op));
    return loom_low_lower_bind_value(
        context, source_result,
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }
  if (iree_string_view_is_empty(scalar_opcode)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted uniform AMDGPU scalar integer "
                            "op without a scalar descriptor");
  }
  return loom_amdgpu_lower_binary_op(context, source_op, scalar_opcode,
                                     source_lhs, source_rhs, source_result);
}

static iree_status_t loom_amdgpu_lower_addi(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, IREE_SV("amdgpu.s_add_u32"),
      IREE_SV("amdgpu.v_add_u32"), loom_scalar_addi_lhs(source_op),
      loom_scalar_addi_rhs(source_op), loom_scalar_addi_result(source_op),
      /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_subi(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, IREE_SV("amdgpu.s_sub_u32"),
      IREE_SV("amdgpu.v_sub_u32"), loom_scalar_subi_lhs(source_op),
      loom_scalar_subi_rhs(source_op), loom_scalar_subi_result(source_op),
      /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_muli(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, iree_string_view_empty(),
      IREE_SV("amdgpu.v_mul_lo_u32"), loom_scalar_muli_lhs(source_op),
      loom_scalar_muli_rhs(source_op), loom_scalar_muli_result(source_op),
      /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_andi(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, IREE_SV("amdgpu.s_and_b32"),
      IREE_SV("amdgpu.v_and_b32"), loom_scalar_andi_lhs(source_op),
      loom_scalar_andi_rhs(source_op), loom_scalar_andi_result(source_op),
      /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_ori(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, IREE_SV("amdgpu.s_or_b32"),
      IREE_SV("amdgpu.v_or_b32"), loom_scalar_ori_lhs(source_op),
      loom_scalar_ori_rhs(source_op), loom_scalar_ori_result(source_op),
      /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_xori(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, IREE_SV("amdgpu.s_xor_b32"),
      IREE_SV("amdgpu.v_xor_b32"), loom_scalar_xori_lhs(source_op),
      loom_scalar_xori_rhs(source_op), loom_scalar_xori_result(source_op),
      /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_shli(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, IREE_SV("amdgpu.s_lshl_b32"),
      IREE_SV("amdgpu.v_lshlrev_b32"), loom_scalar_shli_lhs(source_op),
      loom_scalar_shli_rhs(source_op), loom_scalar_shli_result(source_op),
      /*swap_vector_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_shrsi(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, IREE_SV("amdgpu.s_ashr_i32"),
      IREE_SV("amdgpu.v_ashrrev_i32"), loom_scalar_shrsi_lhs(source_op),
      loom_scalar_shrsi_rhs(source_op), loom_scalar_shrsi_result(source_op),
      /*swap_vector_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_shrui(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, IREE_SV("amdgpu.s_lshr_b32"),
      IREE_SV("amdgpu.v_lshrrev_b32"), loom_scalar_shrui_lhs(source_op),
      loom_scalar_shrui_rhs(source_op), loom_scalar_shrui_result(source_op),
      /*swap_vector_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_vector_addi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, IREE_SV("amdgpu.v_add_u32"),
      loom_vector_addi_lhs(source_op), loom_vector_addi_rhs(source_op),
      loom_vector_addi_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_subi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, IREE_SV("amdgpu.v_sub_u32"),
      loom_vector_subi_lhs(source_op), loom_vector_subi_rhs(source_op),
      loom_vector_subi_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_muli(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, IREE_SV("amdgpu.v_mul_lo_u32"),
      loom_vector_muli_lhs(source_op), loom_vector_muli_rhs(source_op),
      loom_vector_muli_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_andi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, IREE_SV("amdgpu.v_and_b32"),
      loom_vector_andi_lhs(source_op), loom_vector_andi_rhs(source_op),
      loom_vector_andi_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_ori(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, IREE_SV("amdgpu.v_or_b32"),
      loom_vector_ori_lhs(source_op), loom_vector_ori_rhs(source_op),
      loom_vector_ori_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_xori(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, IREE_SV("amdgpu.v_xor_b32"),
      loom_vector_xori_lhs(source_op), loom_vector_xori_rhs(source_op),
      loom_vector_xori_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_shli(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op_ordered(
      context, source_op, IREE_SV("amdgpu.v_lshlrev_b32"),
      loom_vector_shli_lhs(source_op), loom_vector_shli_rhs(source_op),
      loom_vector_shli_result(source_op), /*swap_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_vector_shrsi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op_ordered(
      context, source_op, IREE_SV("amdgpu.v_ashrrev_i32"),
      loom_vector_shrsi_lhs(source_op), loom_vector_shrsi_rhs(source_op),
      loom_vector_shrsi_result(source_op), /*swap_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_vector_shrui(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op_ordered(
      context, source_op, IREE_SV("amdgpu.v_lshrrev_b32"),
      loom_vector_shrui_lhs(source_op), loom_vector_shrui_rhs(source_op),
      loom_vector_shrui_result(source_op), /*swap_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_vector_addf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, IREE_SV("amdgpu.v_add_f32"),
      loom_vector_addf_lhs(source_op), loom_vector_addf_rhs(source_op),
      loom_vector_addf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_subf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, IREE_SV("amdgpu.v_sub_f32"),
      loom_vector_subf_lhs(source_op), loom_vector_subf_rhs(source_op),
      loom_vector_subf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_mulf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, IREE_SV("amdgpu.v_mul_f32"),
      loom_vector_mulf_lhs(source_op), loom_vector_mulf_rhs(source_op),
      loom_vector_mulf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_fmaf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_ternary_op(
      context, source_op, IREE_SV("amdgpu.v_fma_f32"),
      loom_vector_fmaf_a(source_op), loom_vector_fmaf_b(source_op),
      loom_vector_fmaf_c(source_op), loom_vector_fmaf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_addf(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, IREE_SV("amdgpu.v_add_f32"),
      loom_scalar_addf_lhs(source_op), loom_scalar_addf_rhs(source_op),
      loom_scalar_addf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_subf(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, IREE_SV("amdgpu.v_sub_f32"),
      loom_scalar_subf_lhs(source_op), loom_scalar_subf_rhs(source_op),
      loom_scalar_subf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_mulf(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, IREE_SV("amdgpu.v_mul_f32"),
      loom_scalar_mulf_lhs(source_op), loom_scalar_mulf_rhs(source_op),
      loom_scalar_mulf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_fmaf(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_ternary_op(
      context, source_op, IREE_SV("amdgpu.v_fma_f32"),
      loom_scalar_fmaf_a(source_op), loom_scalar_fmaf_b(source_op),
      loom_scalar_fmaf_c(source_op), loom_scalar_fmaf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  uint32_t lane_offset = 0;
  if (!loom_amdgpu_vector_extract_select(context, source_op, &lane_offset)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.extract");
  }

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_extract_source(source_op), &low_source));
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_extract_source(source_op));
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(source_type);
  if (lane_count == 1) {
    return loom_low_lower_bind_value(
        context, loom_vector_extract_result(source_op), low_source);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, loom_vector_extract_result(source_op), &result_type));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_source, lane_offset, result_type, &low_result));
  return loom_low_lower_bind_value(
      context, loom_vector_extract_result(source_op), low_result);
}

static iree_status_t loom_amdgpu_lower_vector_from_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!loom_amdgpu_can_lower_vector_from_elements(context, source_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.from_elements");
  }
  loom_value_slice_t elements = loom_vector_from_elements_elements(source_op);
  const loom_value_id_t source_result =
      loom_vector_from_elements_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_result_type =
      loom_module_value_type(module, source_result);
  const bool result_is_i32_vector =
      loom_type_element_type(source_result_type) == LOOM_SCALAR_TYPE_I32;
  loom_value_id_t low_elements[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES] = {0};
  for (uint32_t i = 0; i < elements.count; ++i) {
    if (result_is_i32_vector) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
          context, source_op, elements.values[i], &low_elements[i]));
    } else {
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, elements.values[i], &low_elements[i]));
    }
  }
  if (elements.count == 1) {
    return loom_low_lower_bind_value(context, source_result, low_elements[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_elements, elements.count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, source_result,
                                   loom_low_concat_result(concat_op));
}

static iree_status_t loom_amdgpu_lower_buffer_view(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_buffer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_buffer_view_buffer(source_op), &low_buffer));
  return loom_low_lower_bind_value(context, loom_buffer_view_result(source_op),
                                   low_buffer);
}

static iree_status_t loom_amdgpu_emit_workitem_id_x_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_low_value_id) {
  IREE_ASSERT_ARGUMENT(out_low_value_id);
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(
      context, IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE),
      &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), source_id,
                             loom_make_named_attr_slice(NULL, 0), vgpr_type,
                             source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_preamble(
    void* user_data, loom_low_lower_context_t* context) {
  (void)user_data;
  loom_func_like_t source_function =
      loom_low_lower_context_source_function(context);
  loom_region_t* source_body = loom_func_like_body(source_function);
  if (source_body == NULL) {
    return iree_ok_status();
  }
  const loom_op_t* first_workitem_id_x_op = NULL;
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_const_block(source_body, block_index);
    const loom_op_t* source_op = NULL;
    loom_block_for_each_op(source_block, source_op) {
      if (loom_kernel_workitem_id_isa(source_op) &&
          loom_kernel_workitem_id_dimension(source_op) ==
              LOOM_KERNEL_WORKITEM_ID_DIMENSION_X) {
        first_workitem_id_x_op = source_op;
        break;
      }
    }
    if (first_workitem_id_x_op != NULL) {
      break;
    }
  }
  if (first_workitem_id_x_op == NULL) {
    return iree_ok_status();
  }

  loom_value_id_t low_workitem_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workitem_id_x_live_in(
      context, first_workitem_id_x_op, &low_workitem_id));
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_const_block(source_body, block_index);
    const loom_op_t* source_op = NULL;
    loom_block_for_each_op(source_block, source_op) {
      if (!loom_kernel_workitem_id_isa(source_op) ||
          loom_kernel_workitem_id_dimension(source_op) !=
              LOOM_KERNEL_WORKITEM_ID_DIMENSION_X) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
          context, loom_kernel_workitem_id_result(source_op), low_workitem_id));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_workitem_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  return loom_low_lower_lookup_value(
      context, loom_kernel_workitem_id_result(source_op), &low_result);
}

static iree_status_t loom_amdgpu_emit_memory_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_memory_access_t* access, loom_value_id_t* out_low_vaddr) {
  IREE_ASSERT_ARGUMENT(out_low_vaddr);
  *out_low_vaddr = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  if (access->dynamic_index == LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      IREE_SV("amdgpu.v_mov_b32"), 0, vgpr_type,
                                      out_low_vaddr);
  }

  loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, access->dynamic_index, &low_index));
  loom_value_id_t low_byte_stride = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, IREE_SV("amdgpu.v_mov_b32"),
      access->dynamic_index_byte_stride, vgpr_type, &low_byte_stride));
  loom_value_id_t operands[] = {
      low_index,
      low_byte_stride,
  };
  loom_op_t* low_offset_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, IREE_SV("amdgpu.v_mul_lo_u32"), operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &vgpr_type,
      1, &low_offset_op));
  *out_low_vaddr = loom_value_slice_get(loom_low_op_results(low_offset_op), 0);
  return iree_ok_status();
}

static iree_string_view_t loom_amdgpu_buffer_load_opcode(
    const loom_amdgpu_memory_access_t* access) {
  return access->vgpr_count == 4 ? IREE_SV("amdgpu.buffer_load_b128")
                                 : IREE_SV("amdgpu.buffer_load_dword");
}

static iree_string_view_t loom_amdgpu_buffer_store_opcode(
    const loom_amdgpu_memory_access_t* access) {
  return access->vgpr_count == 4 ? IREE_SV("amdgpu.buffer_store_b128")
                                 : IREE_SV("amdgpu.buffer_store_dword");
}

static iree_status_t loom_amdgpu_lower_vector_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_load_view(source_op), &low_resource));

  loom_amdgpu_memory_access_t access;
  if (!loom_amdgpu_load_memory_access_select(context, source_op, &access)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector.load "
                            "memory access");
  }
  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_memory_vaddr(context, source_op, &access, &low_vaddr));

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(context, source_op,
                                                  IREE_SV("amdgpu.s_mov_b32"),
                                                  0, sgpr_type, &low_soffset));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, loom_vector_load_result(source_op), &result_type));

  loom_string_id_t offset_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("offset"), &offset_id));
  loom_value_id_t operands[] = {
      low_resource,
      low_vaddr,
      low_soffset,
  };
  loom_named_attr_t attrs[] = {
      {
          .name_id = offset_id,
          .value = loom_attr_i64(access.static_byte_offset),
      },
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, loom_amdgpu_buffer_load_opcode(&access), operands,
      IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), &result_type, 1,
      &low_op));
  return loom_low_lower_bind_value(
      context, loom_vector_load_result(source_op),
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_amdgpu_lower_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_value(source_op), &low_value));
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_view(source_op), &low_resource));

  loom_amdgpu_memory_access_t access;
  if (!loom_amdgpu_store_memory_access_select(context, source_op, &access)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.store memory access");
  }
  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_memory_vaddr(context, source_op, &access, &low_vaddr));

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(context, source_op,
                                                  IREE_SV("amdgpu.s_mov_b32"),
                                                  0, sgpr_type, &low_soffset));

  loom_string_id_t offset_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("offset"), &offset_id));
  loom_value_id_t operands[] = {
      low_value,
      low_resource,
      low_vaddr,
      low_soffset,
  };
  loom_named_attr_t attrs[] = {
      {
          .name_id = offset_id,
          .value = loom_attr_i64(access.static_byte_offset),
      },
  };
  loom_op_t* low_op = NULL;
  return loom_amdgpu_emit_low_op(
      context, source_op, loom_amdgpu_buffer_store_opcode(&access), operands,
      IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)),
      /*result_types=*/NULL, /*result_count=*/0, &low_op);
}

static iree_status_t loom_amdgpu_try_lower_op(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              bool* out_handled) {
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_can_lower_op(user_data, context, source_op, out_handled));
  if (!*out_handled) {
    return iree_ok_status();
  }

  switch (source_op->kind) {
    case LOOM_OP_INDEX_CONSTANT:
      return loom_amdgpu_lower_index_constant(context, source_op);
    case LOOM_OP_SCALAR_CONSTANT:
      return loom_amdgpu_lower_scalar_constant(context, source_op);
    case LOOM_OP_VECTOR_CONSTANT:
      return loom_amdgpu_lower_vector_constant(context, source_op);
    case LOOM_OP_VECTOR_IOTA:
      return loom_amdgpu_lower_vector_iota(context, source_op);
    case LOOM_OP_BUFFER_VIEW:
      return loom_amdgpu_lower_buffer_view(context, source_op);
    case LOOM_OP_KERNEL_WORKITEM_ID:
      return loom_amdgpu_lower_workitem_id(context, source_op);
    case LOOM_OP_SCALAR_ADDI:
      return loom_amdgpu_lower_addi(context, source_op);
    case LOOM_OP_SCALAR_SUBI:
      return loom_amdgpu_lower_subi(context, source_op);
    case LOOM_OP_SCALAR_MULI:
      return loom_amdgpu_lower_muli(context, source_op);
    case LOOM_OP_SCALAR_ANDI:
      return loom_amdgpu_lower_andi(context, source_op);
    case LOOM_OP_SCALAR_ORI:
      return loom_amdgpu_lower_ori(context, source_op);
    case LOOM_OP_SCALAR_XORI:
      return loom_amdgpu_lower_xori(context, source_op);
    case LOOM_OP_SCALAR_SHLI:
      return loom_amdgpu_lower_shli(context, source_op);
    case LOOM_OP_SCALAR_SHRSI:
      return loom_amdgpu_lower_shrsi(context, source_op);
    case LOOM_OP_SCALAR_SHRUI:
      return loom_amdgpu_lower_shrui(context, source_op);
    case LOOM_OP_SCALAR_ADDF:
      return loom_amdgpu_lower_addf(context, source_op);
    case LOOM_OP_SCALAR_SUBF:
      return loom_amdgpu_lower_subf(context, source_op);
    case LOOM_OP_SCALAR_MULF:
      return loom_amdgpu_lower_mulf(context, source_op);
    case LOOM_OP_SCALAR_FMAF:
      return loom_amdgpu_lower_fmaf(context, source_op);
    case LOOM_OP_VECTOR_ADDI:
      return loom_amdgpu_lower_vector_addi(context, source_op);
    case LOOM_OP_VECTOR_SUBI:
      return loom_amdgpu_lower_vector_subi(context, source_op);
    case LOOM_OP_VECTOR_MULI:
      return loom_amdgpu_lower_vector_muli(context, source_op);
    case LOOM_OP_VECTOR_ANDI:
      return loom_amdgpu_lower_vector_andi(context, source_op);
    case LOOM_OP_VECTOR_ORI:
      return loom_amdgpu_lower_vector_ori(context, source_op);
    case LOOM_OP_VECTOR_XORI:
      return loom_amdgpu_lower_vector_xori(context, source_op);
    case LOOM_OP_VECTOR_SHLI:
      return loom_amdgpu_lower_vector_shli(context, source_op);
    case LOOM_OP_VECTOR_SHRSI:
      return loom_amdgpu_lower_vector_shrsi(context, source_op);
    case LOOM_OP_VECTOR_SHRUI:
      return loom_amdgpu_lower_vector_shrui(context, source_op);
    case LOOM_OP_VECTOR_ADDF:
      return loom_amdgpu_lower_vector_addf(context, source_op);
    case LOOM_OP_VECTOR_SUBF:
      return loom_amdgpu_lower_vector_subf(context, source_op);
    case LOOM_OP_VECTOR_MULF:
      return loom_amdgpu_lower_vector_mulf(context, source_op);
    case LOOM_OP_VECTOR_FMAF:
      return loom_amdgpu_lower_vector_fmaf(context, source_op);
    case LOOM_OP_VECTOR_EXTRACT:
      return loom_amdgpu_lower_vector_extract(context, source_op);
    case LOOM_OP_VECTOR_FROM_ELEMENTS:
      return loom_amdgpu_lower_vector_from_elements(context, source_op);
    case LOOM_OP_VECTOR_LOAD:
      return loom_amdgpu_lower_vector_load(context, source_op);
    case LOOM_OP_VECTOR_STORE:
      return loom_amdgpu_lower_vector_store(context, source_op);
    default:
      return iree_ok_status();
  }
}

static const loom_low_lower_policy_t kAmdgpuLowLowerPolicy = {
    .name = IREE_SVL("amdgpu-register-lower"),
    .map_type = {.fn = loom_amdgpu_map_type, .user_data = NULL},
    .map_value = {.fn = loom_amdgpu_map_value, .user_data = NULL},
    .map_argument = {.fn = loom_amdgpu_map_argument, .user_data = NULL},
    .emit_preamble = {.fn = loom_amdgpu_emit_preamble, .user_data = NULL},
    .can_lower_op = {.fn = loom_amdgpu_can_lower_op, .user_data = NULL},
    .try_lower_op = {.fn = loom_amdgpu_try_lower_op, .user_data = NULL},
};

const loom_low_lower_policy_t* loom_amdgpu_low_lower_policy(void) {
  return &kAmdgpuLowLowerPolicy;
}

void loom_amdgpu_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx950.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx11.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx12.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("amdgpu.gfx1250.core"),
          .policy = &kAmdgpuLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
