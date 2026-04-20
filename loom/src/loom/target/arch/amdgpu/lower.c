// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/hal_kernel_abi.h"

static bool loom_amdgpu_type_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static bool loom_amdgpu_type_is_address_scalar(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         (loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX ||
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET);
}

static bool loom_amdgpu_type_is_vector_1xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 1;
}

static bool loom_amdgpu_type_is_i32_view(loom_type_t type) {
  return loom_type_is_view(type) && loom_type_rank(type) == 1 &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
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

static bool loom_amdgpu_value_is_vector_1xi32(loom_low_lower_context_t* context,
                                              loom_value_id_t value_id) {
  return loom_amdgpu_type_is_vector_1xi32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static bool loom_amdgpu_value_is_i32_view(loom_low_lower_context_t* context,
                                          loom_value_id_t value_id) {
  return loom_amdgpu_type_is_i32_view(
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
  if (loom_amdgpu_type_is_vector_1xi32(source_type)) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("AMDGPU lowering currently supports only i32 scalar values, "
              "address scalar values, and vector<1xi32> VGPR values"));
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
  return loom_amdgpu_map_type(user_data, context, source_function_op,
                              source_type, &out_argument->abi_type);
}

static bool loom_amdgpu_attr_is_i32_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_I64 && value.i64 >= INT32_MIN &&
         value.i64 <= INT32_MAX;
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

static bool loom_amdgpu_can_lower_scalar_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_value_is_i32(context,
                                  loom_scalar_constant_result(source_op)) &&
         loom_amdgpu_attr_is_i32_immediate(
             loom_scalar_constant_value(source_op));
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
  return loom_amdgpu_value_is_vector_1xi32(
             context, loom_vector_constant_result(source_op)) &&
         loom_amdgpu_attr_is_i32_immediate(
             loom_vector_constant_value(source_op));
}

static bool loom_amdgpu_can_lower_buffer_view(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  return loom_amdgpu_value_is_i32_view(context,
                                       loom_buffer_view_result(source_op)) &&
         loom_amdgpu_value_is_exact_index_constant(
             context, loom_buffer_view_byte_offset(source_op), 0);
}

static bool loom_amdgpu_indices_are_zero(loom_value_slice_t dynamic_indices,
                                         loom_attribute_t static_indices) {
  if (dynamic_indices.count != 0) {
    return false;
  }
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return false;
  }
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (static_indices.i64_array[i] != 0) {
      return false;
    }
  }
  return true;
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

static bool loom_amdgpu_memory_index_select(
    loom_low_lower_context_t* context, loom_value_slice_t dynamic_indices,
    loom_attribute_t static_indices, loom_value_id_t* out_dynamic_index) {
  IREE_ASSERT_ARGUMENT(out_dynamic_index);
  *out_dynamic_index = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_indices_are_zero(dynamic_indices, static_indices)) {
    return true;
  }
  if (dynamic_indices.count != 1) {
    return false;
  }
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      static_indices.i64_array[0] != INT64_MIN) {
    return false;
  }
  if (!loom_amdgpu_value_is_workitem_id_x(context, dynamic_indices.values[0])) {
    return false;
  }
  *out_dynamic_index = dynamic_indices.values[0];
  return true;
}

static bool loom_amdgpu_load_memory_index_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_dynamic_index) {
  return loom_amdgpu_memory_index_select(
      context, loom_vector_load_indices(source_op),
      loom_vector_load_static_indices(source_op), out_dynamic_index);
}

static bool loom_amdgpu_store_memory_index_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t* out_dynamic_index) {
  return loom_amdgpu_memory_index_select(
      context, loom_vector_store_indices(source_op),
      loom_vector_store_static_indices(source_op), out_dynamic_index);
}

static bool loom_amdgpu_can_lower_vector_load(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op) {
  loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
  return loom_amdgpu_value_is_i32_view(context,
                                       loom_vector_load_view(source_op)) &&
         loom_amdgpu_value_is_vector_1xi32(
             context, loom_vector_load_result(source_op)) &&
         loom_amdgpu_load_memory_index_select(context, source_op,
                                              &dynamic_index);
}

static bool loom_amdgpu_can_lower_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
  return loom_amdgpu_value_is_vector_1xi32(
             context, loom_vector_store_value(source_op)) &&
         loom_amdgpu_value_is_i32_view(context,
                                       loom_vector_store_view(source_op)) &&
         loom_amdgpu_store_memory_index_select(context, source_op,
                                               &dynamic_index);
}

static bool loom_amdgpu_can_lower_i32_binary(loom_low_lower_context_t* context,
                                             loom_value_id_t lhs,
                                             loom_value_id_t rhs,
                                             loom_value_id_t result) {
  return loom_amdgpu_value_is_i32(context, lhs) &&
         loom_amdgpu_value_is_i32(context, rhs) &&
         loom_amdgpu_value_is_i32(context, result);
}

static bool loom_amdgpu_can_lower_vector_1xi32_binary(
    loom_low_lower_context_t* context, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_value_id_t result) {
  return loom_amdgpu_value_is_vector_1xi32(context, lhs) &&
         loom_amdgpu_value_is_vector_1xi32(context, rhs) &&
         loom_amdgpu_value_is_vector_1xi32(context, result);
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
    case LOOM_OP_BUFFER_VIEW:
      *out_handled = loom_amdgpu_can_lower_buffer_view(context, source_op);
      return iree_ok_status();
    case LOOM_OP_KERNEL_WORKITEM_ID:
      *out_handled = loom_amdgpu_can_lower_workitem_id(context, source_op);
      return iree_ok_status();
    case LOOM_OP_SCALAR_ADDI:
      *out_handled = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_addi_lhs(source_op),
          loom_scalar_addi_rhs(source_op), loom_scalar_addi_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_ADDI:
      *out_handled = loom_amdgpu_can_lower_vector_1xi32_binary(
          context, loom_vector_addi_lhs(source_op),
          loom_vector_addi_rhs(source_op), loom_vector_addi_result(source_op));
      return iree_ok_status();
    case LOOM_OP_VECTOR_MULI:
      *out_handled = loom_amdgpu_can_lower_vector_1xi32_binary(
          context, loom_vector_muli_lhs(source_op),
          loom_vector_muli_rhs(source_op), loom_vector_muli_result(source_op));
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
  IREE_RETURN_IF_ERROR(loom_low_lower_map_type(
      context, source_op,
      loom_module_value_type(loom_low_lower_context_module(context),
                             source_result),
      out_low_type));
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

static iree_status_t loom_amdgpu_lower_i32_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    iree_string_view_t opcode, loom_attribute_t source_attr,
    loom_value_id_t source_result) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, source_result, &result_type));

  const int64_t source_value = source_attr.i64;
  const uint32_t bit_pattern = (uint32_t)(int32_t)source_value;
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, opcode, bit_pattern, result_type, &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
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
  return loom_amdgpu_lower_i32_constant(context, source_op,
                                        IREE_SV("amdgpu.s_mov_b32"),
                                        loom_scalar_constant_value(source_op),
                                        loom_scalar_constant_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_constant(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_i32_constant(context, source_op,
                                        IREE_SV("amdgpu.v_mov_b32"),
                                        loom_vector_constant_value(source_op),
                                        loom_vector_constant_result(source_op));
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

static iree_status_t loom_amdgpu_lower_addi(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, IREE_SV("amdgpu.s_add_u32"),
      loom_scalar_addi_lhs(source_op), loom_scalar_addi_rhs(source_op),
      loom_scalar_addi_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_addi(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, IREE_SV("amdgpu.v_add_u32"),
      loom_vector_addi_lhs(source_op), loom_vector_addi_rhs(source_op),
      loom_vector_addi_result(source_op));
}

static iree_status_t loom_amdgpu_lower_vector_muli(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  return loom_amdgpu_lower_binary_op(
      context, source_op, IREE_SV("amdgpu.v_mul_lo_u32"),
      loom_vector_muli_lhs(source_op), loom_vector_muli_rhs(source_op),
      loom_vector_muli_result(source_op));
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
    loom_value_id_t source_dynamic_index, loom_value_id_t* out_low_vaddr) {
  IREE_ASSERT_ARGUMENT(out_low_vaddr);
  *out_low_vaddr = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  if (source_dynamic_index == LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      IREE_SV("amdgpu.v_mov_b32"), 0, vgpr_type,
                                      out_low_vaddr);
  }

  loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_dynamic_index, &low_index));
  loom_value_id_t low_element_size = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, IREE_SV("amdgpu.v_mov_b32"), 4, vgpr_type,
      &low_element_size));
  loom_value_id_t operands[] = {
      low_index,
      low_element_size,
  };
  loom_op_t* low_offset_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, IREE_SV("amdgpu.v_mul_lo_u32"), operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &vgpr_type,
      1, &low_offset_op));
  *out_low_vaddr = loom_value_slice_get(loom_low_op_results(low_offset_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_load_view(source_op), &low_resource));

  loom_value_id_t source_dynamic_index = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_load_memory_index_select(context, source_op,
                                            &source_dynamic_index)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU vector.load "
                            "indices");
  }
  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
      context, source_op, source_dynamic_index, &low_vaddr));

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
          .value = loom_attr_i64(0),
      },
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, IREE_SV("amdgpu.buffer_load_dword"), operands,
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

  loom_value_id_t source_dynamic_index = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_store_memory_index_select(context, source_op,
                                             &source_dynamic_index)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.store indices");
  }
  loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_vaddr(
      context, source_op, source_dynamic_index, &low_vaddr));

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
          .value = loom_attr_i64(0),
      },
  };
  loom_op_t* low_op = NULL;
  return loom_amdgpu_emit_low_op(
      context, source_op, IREE_SV("amdgpu.buffer_store_dword"), operands,
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
    case LOOM_OP_BUFFER_VIEW:
      return loom_amdgpu_lower_buffer_view(context, source_op);
    case LOOM_OP_KERNEL_WORKITEM_ID:
      return loom_amdgpu_lower_workitem_id(context, source_op);
    case LOOM_OP_SCALAR_ADDI:
      return loom_amdgpu_lower_addi(context, source_op);
    case LOOM_OP_VECTOR_ADDI:
      return loom_amdgpu_lower_vector_addi(context, source_op);
    case LOOM_OP_VECTOR_MULI:
      return loom_amdgpu_lower_vector_muli(context, source_op);
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
