// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/hal_kernel_abi.h"
#include "loom/target/arch/amdgpu/lower_internal.h"
#include "loom/util/fact_table.h"

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
  int64_t unused_byte_offset = 0;
  return loom_amdgpu_value_is_32bit_view(context,
                                         loom_buffer_view_result(source_op)) &&
         loom_amdgpu_module_value_as_exact_index_constant(
             loom_low_lower_context_module(context),
             loom_buffer_view_byte_offset(source_op), &unused_byte_offset) &&
         unused_byte_offset >= 0;
}

static bool loom_amdgpu_can_lower_buffer_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (loom_buffer_alloca_memory_space(source_op) !=
      LOOM_BUFFER_MEMORY_SPACE_WORKGROUP) {
    return false;
  }
  const int64_t base_alignment = loom_buffer_alloca_base_alignment(source_op);
  if (base_alignment <= 0 || base_alignment > UINT32_MAX ||
      !loom_amdgpu_u32_is_power_of_two((uint32_t)base_alignment)) {
    return false;
  }
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  int64_t byte_length = 0;
  return loom_amdgpu_value_facts_as_exact_non_negative_i64(
             loom_value_fact_table_lookup(
                 fact_table, loom_buffer_alloca_byte_length(source_op)),
             &byte_length) &&
         byte_length > 0;
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

static bool loom_amdgpu_can_lower_address_binary(
    loom_low_lower_context_t* context, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_value_id_t result, bool allow_uniform_result) {
  if (!loom_amdgpu_value_is_address_scalar(context, lhs) ||
      !loom_amdgpu_value_is_address_scalar(context, rhs) ||
      !loom_amdgpu_value_is_address_scalar(context, result)) {
    return false;
  }
  const bool lhs_vgpr = loom_amdgpu_value_prefers_vgpr(context, lhs);
  const bool rhs_vgpr = loom_amdgpu_value_prefers_vgpr(context, rhs);
  const bool result_vgpr = loom_amdgpu_value_prefers_vgpr(context, result);
  if (result_vgpr) {
    return loom_amdgpu_value_can_materialize_as_vgpr_address(context, lhs) &&
           loom_amdgpu_value_can_materialize_as_vgpr_address(context, rhs);
  }
  return allow_uniform_result && !lhs_vgpr && !rhs_vgpr;
}

static bool loom_amdgpu_can_lower_index_madd(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op) {
  const loom_value_id_t a = loom_index_madd_a(source_op);
  const loom_value_id_t b = loom_index_madd_b(source_op);
  const loom_value_id_t c = loom_index_madd_c(source_op);
  const loom_value_id_t result = loom_index_madd_result(source_op);
  return loom_amdgpu_value_is_address_scalar(context, a) &&
         loom_amdgpu_value_is_address_scalar(context, b) &&
         loom_amdgpu_value_is_address_scalar(context, c) &&
         loom_amdgpu_value_is_address_scalar(context, result) &&
         loom_amdgpu_value_prefers_vgpr(context, result) &&
         loom_amdgpu_value_can_materialize_as_vgpr_address(context, a) &&
         loom_amdgpu_value_can_materialize_as_vgpr_address(context, b) &&
         loom_amdgpu_value_can_materialize_as_vgpr_address(context, c);
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
  return loom_amdgpu_vector_f32_lane_count(result_type) != 0 &&
         loom_type_equal(loom_module_value_type(module, lhs), result_type) &&
         loom_type_equal(loom_module_value_type(module, rhs), result_type);
}

static bool loom_amdgpu_can_lower_vector_f32_ternary(
    loom_low_lower_context_t* context, loom_value_id_t a, loom_value_id_t b,
    loom_value_id_t c, loom_value_id_t result) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t result_type = loom_module_value_type(module, result);
  return loom_amdgpu_vector_f32_lane_count(result_type) != 0 &&
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

static bool loom_amdgpu_can_lower_i32_to_f32(loom_low_lower_context_t* context,
                                             loom_value_id_t input,
                                             loom_value_id_t result) {
  return loom_amdgpu_value_is_i32(context, input) &&
         loom_amdgpu_value_is_f32(context, result);
}

static bool loom_amdgpu_can_lower_vector_i32_to_f32(
    loom_low_lower_context_t* context, loom_value_id_t input,
    loom_value_id_t result) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint32_t input_lane_count =
      loom_amdgpu_vector_i32_lane_count(loom_module_value_type(module, input));
  const uint32_t result_lane_count =
      loom_amdgpu_vector_f32_lane_count(loom_module_value_type(module, result));
  return input_lane_count != 0 && input_lane_count == result_lane_count;
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
  const loom_kernel_dimension_t dimension =
      loom_kernel_workitem_id_dimension(source_op);
  return dimension < LOOM_KERNEL_DIMENSION_COUNT_ &&
         loom_amdgpu_value_is_address_scalar(
             context, loom_kernel_workitem_id_result(source_op));
}

static bool loom_amdgpu_can_lower_workgroup_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_kernel_dimension_t dimension =
      loom_kernel_workgroup_id_dimension(source_op);
  return dimension < LOOM_KERNEL_DIMENSION_COUNT_ &&
         loom_amdgpu_value_is_address_scalar(
             context, loom_kernel_workgroup_id_result(source_op));
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
    case LOOM_OP_BUFFER_ALLOCA:
      *out_handled = loom_amdgpu_can_lower_buffer_alloca(context, source_op);
      return iree_ok_status();
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
      *out_handled = true;
      return iree_ok_status();
    case LOOM_OP_BUFFER_VIEW:
      *out_handled = loom_amdgpu_can_lower_buffer_view(context, source_op);
      return iree_ok_status();
    case LOOM_OP_KERNEL_WORKITEM_ID:
      *out_handled = loom_amdgpu_can_lower_workitem_id(context, source_op);
      return iree_ok_status();
    case LOOM_OP_KERNEL_WORKGROUP_ID:
      *out_handled = loom_amdgpu_can_lower_workgroup_id(context, source_op);
      return iree_ok_status();
    case LOOM_OP_INDEX_ADD:
      *out_handled = loom_amdgpu_can_lower_address_binary(
          context, loom_index_add_lhs(source_op), loom_index_add_rhs(source_op),
          loom_index_add_result(source_op), /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_INDEX_SUB:
      *out_handled = loom_amdgpu_can_lower_address_binary(
          context, loom_index_sub_lhs(source_op), loom_index_sub_rhs(source_op),
          loom_index_sub_result(source_op), /*allow_uniform_result=*/true);
      return iree_ok_status();
    case LOOM_OP_INDEX_MUL:
      *out_handled = loom_amdgpu_can_lower_address_binary(
          context, loom_index_mul_lhs(source_op), loom_index_mul_rhs(source_op),
          loom_index_mul_result(source_op), /*allow_uniform_result=*/false);
      return iree_ok_status();
    case LOOM_OP_INDEX_MADD:
      *out_handled = loom_amdgpu_can_lower_index_madd(context, source_op);
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
    case LOOM_OP_SCALAR_MINNUMF:
      *out_handled = loom_amdgpu_can_lower_f32_binary(
          context, loom_scalar_minnumf_lhs(source_op),
          loom_scalar_minnumf_rhs(source_op),
          loom_scalar_minnumf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_SCALAR_MAXNUMF:
      *out_handled = loom_amdgpu_can_lower_f32_binary(
          context, loom_scalar_maxnumf_lhs(source_op),
          loom_scalar_maxnumf_rhs(source_op),
          loom_scalar_maxnumf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_SCALAR_FMAF:
      *out_handled = loom_amdgpu_can_lower_f32_ternary(
          context, loom_scalar_fmaf_a(source_op), loom_scalar_fmaf_b(source_op),
          loom_scalar_fmaf_c(source_op), loom_scalar_fmaf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_SCALAR_SITOFP:
      *out_handled = loom_amdgpu_can_lower_i32_to_f32(
          context, loom_scalar_sitofp_input(source_op),
          loom_scalar_sitofp_result(source_op));
      return iree_ok_status();
    case LOOM_OP_SCALAR_UITOFP:
      *out_handled = loom_amdgpu_can_lower_i32_to_f32(
          context, loom_scalar_uitofp_input(source_op),
          loom_scalar_uitofp_result(source_op));
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
    case LOOM_OP_VECTOR_MINNUMF:
      *out_handled = loom_amdgpu_can_lower_vector_f32_binary(
          context, loom_vector_minnumf_lhs(source_op),
          loom_vector_minnumf_rhs(source_op),
          loom_vector_minnumf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_MAXNUMF:
      *out_handled = loom_amdgpu_can_lower_vector_f32_binary(
          context, loom_vector_maxnumf_lhs(source_op),
          loom_vector_maxnumf_rhs(source_op),
          loom_vector_maxnumf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_FMAF:
      *out_handled = loom_amdgpu_can_lower_vector_f32_ternary(
          context, loom_vector_fmaf_a(source_op), loom_vector_fmaf_b(source_op),
          loom_vector_fmaf_c(source_op), loom_vector_fmaf_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_SITOFP:
      *out_handled = loom_amdgpu_can_lower_vector_i32_to_f32(
          context, loom_vector_sitofp_input(source_op),
          loom_vector_sitofp_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_UITOFP:
      *out_handled = loom_amdgpu_can_lower_vector_i32_to_f32(
          context, loom_vector_uitofp_input(source_op),
          loom_vector_uitofp_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_BITFIELD_EXTRACTU:
    case LOOM_OP_VECTOR_BITFIELD_EXTRACTS:
      *out_handled =
          loom_amdgpu_can_lower_vector_bitfield_extract(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_BITFIELD_INSERT:
      *out_handled =
          loom_amdgpu_can_lower_vector_bitfield_insert(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_BITPACK:
      *out_handled = loom_amdgpu_can_lower_vector_bitpack(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_BITUNPACKS:
    case LOOM_OP_VECTOR_BITUNPACKU:
      *out_handled = loom_amdgpu_can_lower_vector_bitunpack(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_BITCAST:
      *out_handled = loom_amdgpu_can_lower_vector_bitcast(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_SLICE:
      *out_handled = loom_amdgpu_can_lower_vector_slice(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_REDUCE:
      *out_handled = loom_amdgpu_can_lower_vector_reduce(context, source_op);
      return iree_ok_status();
    case LOOM_OP_VECTOR_DOT4I:
      *out_handled = loom_amdgpu_can_lower_vector_dot4i(context, source_op);
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

static iree_status_t loom_amdgpu_bind_vgpr_u32_lane_constants(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, const uint32_t* lane_bit_patterns,
    uint32_t lane_count) {
  IREE_ASSERT_ARGUMENT(lane_bit_patterns);
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "constant lane count");
  }

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
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "constant lane count");
  }

  uint32_t lane_bit_patterns[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
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
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
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

  int64_t lane_values[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
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
    uint64_t descriptor_id, loom_value_id_t source_lhs,
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
      context, source_op, descriptor_id, low_operands,
      IREE_ARRAYSIZE(low_operands), loom_make_named_attr_slice(NULL, 0),
      &result_type, 1, &low_op));
  return loom_low_lower_bind_value(
      context, source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_amdgpu_lower_ternary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t source_a, loom_value_id_t source_b,
    loom_value_id_t source_c, loom_value_id_t source_result) {
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
      context, source_op, descriptor_id, low_operands,
      IREE_ARRAYSIZE(low_operands), loom_make_named_attr_slice(NULL, 0),
      &result_type, 1, &low_op));
  return loom_low_lower_bind_value(
      context, source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_amdgpu_lower_vector_binary_op_ordered(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t source_lhs,
    loom_value_id_t source_rhs, loom_value_id_t source_result,
    bool swap_operands) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(
      loom_module_value_type(module, source_result));
  if (lane_count == 1) {
    if (swap_operands) {
      return loom_amdgpu_lower_binary_op(context, source_op, descriptor_id,
                                         source_rhs, source_lhs, source_result);
    }
    return loom_amdgpu_lower_binary_op(context, source_op, descriptor_id,
                                       source_lhs, source_rhs, source_result);
  }
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
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
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
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
        context, source_op, descriptor_id, operands, IREE_ARRAYSIZE(operands),
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
    uint64_t descriptor_id, loom_value_id_t source_lhs,
    loom_value_id_t source_rhs, loom_value_id_t source_result) {
  return loom_amdgpu_lower_vector_binary_op_ordered(
      context, source_op, descriptor_id, source_lhs, source_rhs, source_result,
      /*swap_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_vector_ternary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t source_a, loom_value_id_t source_b,
    loom_value_id_t source_c, loom_value_id_t source_result) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(
      loom_module_value_type(module, source_result));
  if (lane_count == 1) {
    return loom_amdgpu_lower_ternary_op(context, source_op, descriptor_id,
                                        source_a, source_b, source_c,
                                        source_result);
  }
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
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
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
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
        context, source_op, descriptor_id, operands, IREE_ARRAYSIZE(operands),
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
    uint64_t scalar_descriptor_id, uint64_t vector_descriptor_id,
    loom_value_id_t source_lhs, loom_value_id_t source_rhs,
    loom_value_id_t source_result, bool swap_vector_operands) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  bool result_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &result_is_vgpr));
  if (result_is_vgpr) {
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
        context, source_op, vector_descriptor_id, low_operands,
        IREE_ARRAYSIZE(low_operands), loom_make_named_attr_slice(NULL, 0),
        &result_type, 1, &low_op));
    return loom_low_lower_bind_value(
        context, source_result,
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }
  if (scalar_descriptor_id == LOOM_LOW_DESCRIPTOR_ID_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted uniform AMDGPU scalar integer "
                            "op without a scalar descriptor");
  }
  return loom_amdgpu_lower_binary_op(context, source_op, scalar_descriptor_id,
                                     source_lhs, source_rhs, source_result);
}

static iree_status_t loom_amdgpu_lower_addi(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32,
      LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32, loom_scalar_addi_lhs(source_op),
      loom_scalar_addi_rhs(source_op), loom_scalar_addi_result(source_op),
      /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_subi(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_SUB_U32,
      LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32, loom_scalar_subi_lhs(source_op),
      loom_scalar_subi_rhs(source_op), loom_scalar_subi_result(source_op),
      /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_muli(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, LOOM_LOW_DESCRIPTOR_ID_NONE,
      LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32, loom_scalar_muli_lhs(source_op),
      loom_scalar_muli_rhs(source_op), loom_scalar_muli_result(source_op),
      /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_andi(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_AND_B32,
      LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32, loom_scalar_andi_lhs(source_op),
      loom_scalar_andi_rhs(source_op), loom_scalar_andi_result(source_op),
      /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_ori(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_OR_B32,
      LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32, loom_scalar_ori_lhs(source_op),
      loom_scalar_ori_rhs(source_op), loom_scalar_ori_result(source_op),
      /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_xori(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_XOR_B32,
      LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32, loom_scalar_xori_lhs(source_op),
      loom_scalar_xori_rhs(source_op), loom_scalar_xori_result(source_op),
      /*swap_vector_operands=*/false);
}

static iree_status_t loom_amdgpu_lower_shli(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHL_B32,
      LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32, loom_scalar_shli_lhs(source_op),
      loom_scalar_shli_rhs(source_op), loom_scalar_shli_result(source_op),
      /*swap_vector_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_shrsi(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_ASHR_I32,
      LOOM_AMDGPU_DESCRIPTOR_ID_V_ASHRREV_I32, loom_scalar_shrsi_lhs(source_op),
      loom_scalar_shrsi_rhs(source_op), loom_scalar_shrsi_result(source_op),
      /*swap_vector_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_shrui(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHR_B32,
      LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32, loom_scalar_shrui_lhs(source_op),
      loom_scalar_shrui_rhs(source_op), loom_scalar_shrui_result(source_op),
      /*swap_vector_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_address_binary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t scalar_descriptor_id, uint64_t vector_descriptor_id,
    loom_value_id_t source_lhs, loom_value_id_t source_rhs,
    loom_value_id_t source_result) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  bool result_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &result_is_vgpr));
  if (result_is_vgpr) {
    loom_value_id_t low_operands[2] = {
        LOOM_VALUE_ID_INVALID,
        LOOM_VALUE_ID_INVALID,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_address(
        context, source_op, source_lhs, &low_operands[0]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_address(
        context, source_op, source_rhs, &low_operands[1]));
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, vector_descriptor_id, low_operands,
        IREE_ARRAYSIZE(low_operands), loom_make_named_attr_slice(NULL, 0),
        &result_type, 1, &low_op));
    return loom_low_lower_bind_value(
        context, source_result,
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }
  if (scalar_descriptor_id == LOOM_LOW_DESCRIPTOR_ID_NONE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "preflight accepted uniform AMDGPU address op without a scalar "
        "descriptor");
  }
  return loom_amdgpu_lower_binary_op(context, source_op, scalar_descriptor_id,
                                     source_lhs, source_rhs, source_result);
}

static iree_status_t loom_amdgpu_lower_index_add(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_address_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32,
      LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32, loom_index_add_lhs(source_op),
      loom_index_add_rhs(source_op), loom_index_add_result(source_op));
}

static iree_status_t loom_amdgpu_lower_index_sub(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_address_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_SUB_U32,
      LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32, loom_index_sub_lhs(source_op),
      loom_index_sub_rhs(source_op), loom_index_sub_result(source_op));
}

static iree_status_t loom_amdgpu_lower_index_mul(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_address_binary_op(
      context, source_op, LOOM_LOW_DESCRIPTOR_ID_NONE,
      LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32, loom_index_mul_lhs(source_op),
      loom_index_mul_rhs(source_op), loom_index_mul_result(source_op));
}

static iree_status_t loom_amdgpu_lower_index_madd(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, loom_index_madd_result(source_op), &result_type));
  loom_value_id_t low_a = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_address(
      context, source_op, loom_index_madd_a(source_op), &low_a));
  loom_value_id_t low_b = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_address(
      context, source_op, loom_index_madd_b(source_op), &low_b));
  loom_value_id_t low_product = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32, low_a, low_b,
      result_type, &low_product));
  loom_value_id_t low_c = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_address(
      context, source_op, loom_index_madd_c(source_op), &low_c));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32, low_product,
      low_c, result_type, &low_result));
  return loom_low_lower_bind_value(context, loom_index_madd_result(source_op),
                                   low_result);
}

static iree_status_t loom_amdgpu_lower_vector_addi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32,
      loom_vector_addi_lhs(source_op), loom_vector_addi_rhs(source_op),
      loom_vector_addi_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_subi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32,
      loom_vector_subi_lhs(source_op), loom_vector_subi_rhs(source_op),
      loom_vector_subi_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_muli(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32,
      loom_vector_muli_lhs(source_op), loom_vector_muli_rhs(source_op),
      loom_vector_muli_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_andi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32,
      loom_vector_andi_lhs(source_op), loom_vector_andi_rhs(source_op),
      loom_vector_andi_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_ori(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32,
      loom_vector_ori_lhs(source_op), loom_vector_ori_rhs(source_op),
      loom_vector_ori_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_xori(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32,
      loom_vector_xori_lhs(source_op), loom_vector_xori_rhs(source_op),
      loom_vector_xori_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_shli(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op_ordered(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32,
      loom_vector_shli_lhs(source_op), loom_vector_shli_rhs(source_op),
      loom_vector_shli_result(source_op), /*swap_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_vector_shrsi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op_ordered(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_ASHRREV_I32,
      loom_vector_shrsi_lhs(source_op), loom_vector_shrsi_rhs(source_op),
      loom_vector_shrsi_result(source_op), /*swap_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_vector_shrui(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op_ordered(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32,
      loom_vector_shrui_lhs(source_op), loom_vector_shrui_rhs(source_op),
      loom_vector_shrui_result(source_op), /*swap_operands=*/true);
}

static iree_status_t loom_amdgpu_lower_vector_addf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_F32,
      loom_vector_addf_lhs(source_op), loom_vector_addf_rhs(source_op),
      loom_vector_addf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_subf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_F32,
      loom_vector_subf_lhs(source_op), loom_vector_subf_rhs(source_op),
      loom_vector_subf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_mulf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_F32,
      loom_vector_mulf_lhs(source_op), loom_vector_mulf_rhs(source_op),
      loom_vector_mulf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_minnumf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MIN_F32,
      loom_vector_minnumf_lhs(source_op), loom_vector_minnumf_rhs(source_op),
      loom_vector_minnumf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_maxnumf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MAX_F32,
      loom_vector_maxnumf_lhs(source_op), loom_vector_maxnumf_rhs(source_op),
      loom_vector_maxnumf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_fmaf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_ternary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_FMA_F32,
      loom_vector_fmaf_a(source_op), loom_vector_fmaf_b(source_op),
      loom_vector_fmaf_c(source_op), loom_vector_fmaf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_i32_to_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t source_input,
    loom_value_id_t source_result) {
  loom_value_id_t low_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_input, &low_input));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_id, &low_input, 1,
      loom_make_named_attr_slice(NULL, 0), &result_type, 1, &low_op));
  return loom_low_lower_bind_value(
      context, source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_amdgpu_lower_vector_i32_to_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t source_input,
    loom_value_id_t source_result) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint32_t lane_count = loom_amdgpu_vector_i32_lane_count(
      loom_module_value_type(module, source_input));
  if (lane_count == 1) {
    return loom_amdgpu_lower_i32_to_f32(context, source_op, descriptor_id,
                                        source_input, source_result);
  }
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector "
                            "integer-to-float conversion lane count");
  }

  loom_value_id_t low_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_input, &low_input));
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t lane_input = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_input, i, lane_type, &lane_input));
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, descriptor_id, &lane_input, 1,
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

static iree_status_t loom_amdgpu_lower_sitofp(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_to_f32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_ID_V_CVT_F32_I32,
                                      loom_scalar_sitofp_input(source_op),
                                      loom_scalar_sitofp_result(source_op));
}

static iree_status_t loom_amdgpu_lower_uitofp(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_to_f32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_ID_V_CVT_F32_U32,
                                      loom_scalar_uitofp_input(source_op),
                                      loom_scalar_uitofp_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_sitofp(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_i32_to_f32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_CVT_F32_I32,
      loom_vector_sitofp_input(source_op),
      loom_vector_sitofp_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_uitofp(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_vector_i32_to_f32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_CVT_F32_U32,
      loom_vector_uitofp_input(source_op),
      loom_vector_uitofp_result(source_op));
}

static iree_status_t loom_amdgpu_lower_addf(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_F32,
      loom_scalar_addf_lhs(source_op), loom_scalar_addf_rhs(source_op),
      loom_scalar_addf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_subf(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_F32,
      loom_scalar_subf_lhs(source_op), loom_scalar_subf_rhs(source_op),
      loom_scalar_subf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_mulf(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_F32,
      loom_scalar_mulf_lhs(source_op), loom_scalar_mulf_rhs(source_op),
      loom_scalar_mulf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_minnumf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MIN_F32,
      loom_scalar_minnumf_lhs(source_op), loom_scalar_minnumf_rhs(source_op),
      loom_scalar_minnumf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_maxnumf(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MAX_F32,
      loom_scalar_maxnumf_lhs(source_op), loom_scalar_maxnumf_rhs(source_op),
      loom_scalar_maxnumf_result(source_op));
}

static iree_status_t loom_amdgpu_lower_fmaf(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_ternary_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_FMA_F32,
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
  loom_value_id_t low_elements[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {0};
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

static iree_status_t loom_amdgpu_lower_buffer_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!loom_amdgpu_can_lower_buffer_alloca(context, source_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "buffer.alloca");
  }

  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  int64_t byte_length = 0;
  if (!loom_amdgpu_value_facts_as_exact_non_negative_i64(
          loom_value_fact_table_lookup(
              fact_table, loom_buffer_alloca_byte_length(source_op)),
          &byte_length) ||
      byte_length <= 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted AMDGPU buffer.alloca with "
                            "unsupported byte length");
  }

  loom_symbol_ref_t slot_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_low_lower_create_function_symbol(
      context, IREE_SV("__lds"), /*append_index=*/true,
      loom_buffer_alloca_result(source_op), &slot_ref));

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_op_t* low_func_op = loom_low_lower_context_low_function(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_after(builder, low_func_op);
  loom_op_t* slot_op = NULL;
  iree_status_t status = loom_low_slot_build(
      builder, slot_ref, loom_low_func_def_callee(low_func_op),
      LOOM_LOW_SLOT_SPACE_LDS, byte_length,
      loom_buffer_alloca_base_alignment(source_op), source_op->location,
      &slot_op);
  loom_builder_restore(builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_op_t* frame_index_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_frame_index_build(builder, slot_ref, /*offset=*/0, vgpr_type,
                                 source_op->location, &frame_index_op));
  return loom_low_lower_bind_value(context,
                                   loom_buffer_alloca_result(source_op),
                                   loom_low_frame_index_result(frame_index_op));
}

static iree_status_t loom_amdgpu_lower_buffer_view(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_buffer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_buffer_view_buffer(source_op), &low_buffer));
  return loom_low_lower_bind_value(context, loom_buffer_view_result(source_op),
                                   low_buffer);
}

static iree_status_t loom_amdgpu_lower_buffer_assume_memory_space(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_buffer = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_buffer_assume_memory_space_buffer(source_op), &low_buffer));
  return loom_low_lower_bind_value(
      context, loom_buffer_assume_memory_space_result(source_op), low_buffer);
}

static iree_status_t loom_amdgpu_workitem_id_source(
    loom_kernel_dimension_t dimension, iree_string_view_t* out_source) {
  IREE_ASSERT_ARGUMENT(out_source);
  *out_source = iree_string_view_empty();
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      *out_source = IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE);
      return iree_ok_status();
    case LOOM_KERNEL_DIMENSION_Y:
      *out_source = IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Y_SOURCE);
      return iree_ok_status();
    case LOOM_KERNEL_DIMENSION_Z:
      *out_source = IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Z_SOURCE);
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown AMDGPU workitem-id dimension %u",
                              (unsigned)dimension);
  }
}

static iree_status_t loom_amdgpu_workgroup_id_source(
    loom_kernel_dimension_t dimension, iree_string_view_t* out_source) {
  IREE_ASSERT_ARGUMENT(out_source);
  *out_source = iree_string_view_empty();
  switch (dimension) {
    case LOOM_KERNEL_DIMENSION_X:
      *out_source = IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_X_SOURCE);
      return iree_ok_status();
    case LOOM_KERNEL_DIMENSION_Y:
      *out_source = IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_Y_SOURCE);
      return iree_ok_status();
    case LOOM_KERNEL_DIMENSION_Z:
      *out_source = IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_Z_SOURCE);
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown AMDGPU workgroup-id dimension %u",
                              (unsigned)dimension);
  }
}

static iree_status_t loom_amdgpu_emit_workitem_id_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_kernel_dimension_t dimension, loom_value_id_t* out_low_value_id) {
  IREE_ASSERT_ARGUMENT(out_low_value_id);
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  iree_string_view_t source = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_workitem_id_source(dimension, &source));
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, source, &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), source_id,
                             loom_make_named_attr_slice(NULL, 0), vgpr_type,
                             source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_workgroup_id_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_kernel_dimension_t dimension, loom_value_id_t* out_low_value_id) {
  IREE_ASSERT_ARGUMENT(out_low_value_id);
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  iree_string_view_t source = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_workgroup_id_source(dimension, &source));
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, source, &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), source_id,
                             loom_make_named_attr_slice(NULL, 0), sgpr_type,
                             source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_m0_live_in(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t* out_low_value_id) {
  IREE_ASSERT_ARGUMENT(out_low_value_id);
  *out_low_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_descriptor_implicit_resource_type(
      context, descriptor_id, &m0_type));
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(
      context, IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_M0_SOURCE), &source_id));
  loom_op_t* live_in_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_live_in_build(loom_low_lower_context_builder(context), source_id,
                             loom_make_named_attr_slice(NULL, 0), m0_type,
                             source_op->location, &live_in_op));
  *out_low_value_id = loom_low_live_in_result(live_in_op);
  loom_string_id_t value_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("m0"), &value_name_id));
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_name(loom_low_lower_context_module(context),
                                 *out_low_value_id, value_name_id));
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lookup_m0_live_in(loom_low_lower_context_t* context,
                                            loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_op_t* low_function = loom_low_lower_context_low_function(context);
  loom_region_t* body =
      low_function ? loom_low_func_def_body(low_function) : NULL;
  if (body == NULL || body->block_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU M0 lookup requires an emitted low function body");
  }
  loom_string_id_t source_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(
      context, IREE_SV(LOOM_AMDGPU_HAL_KERNEL_ABI_M0_SOURCE), &source_id));
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(entry_block, op) {
    if (!loom_low_live_in_isa(op)) {
      break;
    }
    if (loom_low_live_in_source(op) == source_id) {
      *out_value_id = loom_low_live_in_result(op);
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU lowering selected an M0-consuming packet without an M0 live-in");
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
  const loom_op_t* first_workitem_id_ops[LOOM_KERNEL_DIMENSION_COUNT_] = {0};
  const loom_op_t* first_workgroup_id_ops[LOOM_KERNEL_DIMENSION_COUNT_] = {0};
  const loom_op_t* first_m0_op = NULL;
  uint64_t m0_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_const_block(source_body, block_index);
    const loom_op_t* source_op = NULL;
    loom_block_for_each_op(source_block, source_op) {
      if (first_m0_op == NULL) {
        uint64_t descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
        if (loom_amdgpu_source_op_selects_m0_descriptor(context, source_op,
                                                        &descriptor_id)) {
          first_m0_op = source_op;
          m0_descriptor_id = descriptor_id;
        }
      }
      if (!loom_kernel_workitem_id_isa(source_op)) {
        if (!loom_kernel_workgroup_id_isa(source_op)) {
          continue;
        }
        const loom_kernel_dimension_t dimension =
            loom_kernel_workgroup_id_dimension(source_op);
        if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_ ||
            first_workgroup_id_ops[dimension] != NULL) {
          continue;
        }
        first_workgroup_id_ops[dimension] = source_op;
      } else {
        const loom_kernel_dimension_t dimension =
            loom_kernel_workitem_id_dimension(source_op);
        if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_ ||
            first_workitem_id_ops[dimension] != NULL) {
          continue;
        }
        first_workitem_id_ops[dimension] = source_op;
      }
    }
  }

  loom_value_id_t low_workitem_ids[LOOM_KERNEL_DIMENSION_COUNT_] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  loom_value_id_t low_workgroup_ids[LOOM_KERNEL_DIMENSION_COUNT_] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  if (first_m0_op != NULL) {
    loom_value_id_t unused_m0 = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_m0_live_in(
        context, first_m0_op, m0_descriptor_id, &unused_m0));
  }
  for (uint32_t i = 0; i < LOOM_KERNEL_DIMENSION_COUNT_; ++i) {
    if (first_workgroup_id_ops[i] != NULL) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workgroup_id_live_in(
          context, first_workgroup_id_ops[i], (loom_kernel_dimension_t)i,
          &low_workgroup_ids[i]));
    }
    if (first_workitem_id_ops[i] == NULL) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_workitem_id_live_in(
        context, first_workitem_id_ops[i], (loom_kernel_dimension_t)i,
        &low_workitem_ids[i]));
  }

  for (uint16_t block_index = 0; block_index < source_body->block_count;
       ++block_index) {
    const loom_block_t* source_block =
        loom_region_const_block(source_body, block_index);
    const loom_op_t* source_op = NULL;
    loom_block_for_each_op(source_block, source_op) {
      if (!loom_kernel_workitem_id_isa(source_op)) {
        if (!loom_kernel_workgroup_id_isa(source_op)) {
          continue;
        }
        const loom_kernel_dimension_t dimension =
            loom_kernel_workgroup_id_dimension(source_op);
        if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_ ||
            low_workgroup_ids[dimension] == LOOM_VALUE_ID_INVALID) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
            context, loom_kernel_workgroup_id_result(source_op),
            low_workgroup_ids[dimension]));
      } else {
        const loom_kernel_dimension_t dimension =
            loom_kernel_workitem_id_dimension(source_op);
        if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_ ||
            low_workitem_ids[dimension] == LOOM_VALUE_ID_INVALID) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
            context, loom_kernel_workitem_id_result(source_op),
            low_workitem_ids[dimension]));
      }
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

static iree_status_t loom_amdgpu_lower_workgroup_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  return loom_low_lower_lookup_value(
      context, loom_kernel_workgroup_id_result(source_op), &low_result);
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
    case LOOM_OP_BUFFER_ALLOCA:
      return loom_amdgpu_lower_buffer_alloca(context, source_op);
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
      return loom_amdgpu_lower_buffer_assume_memory_space(context, source_op);
    case LOOM_OP_BUFFER_VIEW:
      return loom_amdgpu_lower_buffer_view(context, source_op);
    case LOOM_OP_KERNEL_WORKITEM_ID:
      return loom_amdgpu_lower_workitem_id(context, source_op);
    case LOOM_OP_KERNEL_WORKGROUP_ID:
      return loom_amdgpu_lower_workgroup_id(context, source_op);
    case LOOM_OP_INDEX_ADD:
      return loom_amdgpu_lower_index_add(context, source_op);
    case LOOM_OP_INDEX_SUB:
      return loom_amdgpu_lower_index_sub(context, source_op);
    case LOOM_OP_INDEX_MUL:
      return loom_amdgpu_lower_index_mul(context, source_op);
    case LOOM_OP_INDEX_MADD:
      return loom_amdgpu_lower_index_madd(context, source_op);
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
    case LOOM_OP_SCALAR_MINNUMF:
      return loom_amdgpu_lower_minnumf(context, source_op);
    case LOOM_OP_SCALAR_MAXNUMF:
      return loom_amdgpu_lower_maxnumf(context, source_op);
    case LOOM_OP_SCALAR_FMAF:
      return loom_amdgpu_lower_fmaf(context, source_op);
    case LOOM_OP_SCALAR_SITOFP:
      return loom_amdgpu_lower_sitofp(context, source_op);
    case LOOM_OP_SCALAR_UITOFP:
      return loom_amdgpu_lower_uitofp(context, source_op);
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
    case LOOM_OP_VECTOR_MINNUMF:
      return loom_amdgpu_lower_vector_minnumf(context, source_op);
    case LOOM_OP_VECTOR_MAXNUMF:
      return loom_amdgpu_lower_vector_maxnumf(context, source_op);
    case LOOM_OP_VECTOR_FMAF:
      return loom_amdgpu_lower_vector_fmaf(context, source_op);
    case LOOM_OP_VECTOR_SITOFP:
      return loom_amdgpu_lower_vector_sitofp(context, source_op);
    case LOOM_OP_VECTOR_UITOFP:
      return loom_amdgpu_lower_vector_uitofp(context, source_op);
    case LOOM_OP_VECTOR_BITFIELD_EXTRACTU:
    case LOOM_OP_VECTOR_BITFIELD_EXTRACTS:
      return loom_amdgpu_lower_vector_bitfield_extract(context, source_op);
    case LOOM_OP_VECTOR_BITFIELD_INSERT:
      return loom_amdgpu_lower_vector_bitfield_insert(context, source_op);
    case LOOM_OP_VECTOR_BITPACK:
      return loom_amdgpu_lower_vector_bitpack(context, source_op);
    case LOOM_OP_VECTOR_BITUNPACKS:
    case LOOM_OP_VECTOR_BITUNPACKU:
      return loom_amdgpu_lower_vector_bitunpack(context, source_op);
    case LOOM_OP_VECTOR_BITCAST:
      return loom_amdgpu_lower_vector_bitcast(context, source_op);
    case LOOM_OP_VECTOR_SLICE:
      return loom_amdgpu_lower_vector_slice(context, source_op);
    case LOOM_OP_VECTOR_REDUCE:
      return loom_amdgpu_lower_vector_reduce(context, source_op);
    case LOOM_OP_VECTOR_DOT4I:
      return loom_amdgpu_lower_vector_dot4i(context, source_op);
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

static iree_status_t loom_amdgpu_low_legality_verify_buffer_view(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_type_is_32bit_view(
          loom_module_value_type(module, loom_buffer_view_result(op)))) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("memory"), loom_op_name(module, op),
        IREE_SV("AMDGPU buffer memory lowering currently requires typed views "
                "over 32-bit elements"));
  }
  int64_t unused_byte_offset = 0;
  if (!loom_amdgpu_module_value_as_exact_index_constant(
          module, loom_buffer_view_byte_offset(op), &unused_byte_offset) ||
      unused_byte_offset < 0) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("memory"), loom_op_name(module, op),
        IREE_SV("AMDGPU HAL buffer views currently require exact non-negative "
                "static byte offsets"));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_low_legality_try_verify_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  switch (op->kind) {
    case LOOM_OP_BUFFER_VIEW:
      return loom_amdgpu_low_legality_verify_buffer_view(provider, context, op,
                                                         out_handled);
    case LOOM_OP_VECTOR_BITPACK:
    case LOOM_OP_VECTOR_BITUNPACKS:
    case LOOM_OP_VECTOR_BITUNPACKU:
      return loom_amdgpu_low_legality_verify_vector_bitstream(provider, context,
                                                              op, out_handled);
    case LOOM_OP_VECTOR_BITCAST:
    case LOOM_OP_VECTOR_SLICE:
      return loom_amdgpu_low_legality_verify_vector_structural(
          provider, context, op, out_handled);
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VECTOR_STORE:
      return loom_amdgpu_low_legality_verify_vector_memory(provider, context,
                                                           op, out_handled);
    default:
      if (loom_amdgpu_op_is_vector_dot(op->kind)) {
        return loom_amdgpu_low_legality_verify_vector_dot(provider, context, op,
                                                          out_handled);
      }
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

const loom_target_low_legality_provider_t
    loom_amdgpu_low_legality_provider_storage = {
        .name = IREE_SVL("amdgpu"),
        .try_verify_op = loom_amdgpu_low_legality_try_verify_op,
};

const loom_low_lower_policy_t* loom_amdgpu_low_lower_policy(void) {
  return &kAmdgpuLowLowerPolicy;
}

const loom_target_low_legality_provider_t* loom_amdgpu_low_legality_provider(
    void) {
  return &loom_amdgpu_low_legality_provider_storage;
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
