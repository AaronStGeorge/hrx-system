// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower/internal.h"

bool loom_amdgpu_type_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

bool loom_amdgpu_type_is_i1(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

bool loom_amdgpu_type_is_address_scalar(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         (loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX ||
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET);
}

bool loom_amdgpu_type_is_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

uint32_t loom_amdgpu_static_vector_lane_count(loom_type_t type,
                                              loom_scalar_type_t element_type,
                                              uint32_t max_lane_count) {
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type) ||
      loom_type_element_type(type) != element_type) {
    return 0;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count < 1 || lane_count > (int64_t)max_lane_count) {
    return 0;
  }
  return (uint32_t)lane_count;
}

static uint32_t loom_amdgpu_vector_lane_count(loom_type_t type,
                                              loom_scalar_type_t element_type) {
  return loom_amdgpu_static_vector_lane_count(
      type, element_type, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
}

static bool loom_amdgpu_type_is_vector_32bit_lane_range(loom_type_t type) {
  return loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_I32) != 0 ||
         loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_F32) != 0;
}

uint32_t loom_amdgpu_vector_32bit_lane_count(loom_type_t type) {
  const uint32_t i32_lane_count =
      loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_I32);
  return i32_lane_count != 0
             ? i32_lane_count
             : loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_F32);
}

uint32_t loom_amdgpu_vector_i32_lane_count(loom_type_t type) {
  return loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_I32);
}

uint32_t loom_amdgpu_vector_f32_lane_count(loom_type_t type) {
  return loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_F32);
}

uint32_t loom_amdgpu_vector_i1_lane_count(loom_type_t type) {
  return loom_amdgpu_static_vector_lane_count(
      type, LOOM_SCALAR_TYPE_I1, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
}

uint32_t loom_amdgpu_vector_i8_lane_count(loom_type_t type) {
  return loom_amdgpu_static_vector_lane_count(type, LOOM_SCALAR_TYPE_I8,
                                              LOOM_AMDGPU_MAX_PACKED_I8_LANES);
}

bool loom_amdgpu_type_packed_integer_storage(loom_type_t type,
                                             uint32_t* out_payload_bit_count,
                                             uint32_t* out_register_count) {
  IREE_ASSERT_ARGUMENT(out_payload_bit_count);
  IREE_ASSERT_ARGUMENT(out_register_count);
  *out_payload_bit_count = 0;
  *out_register_count = 0;
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count < 1 || lane_count > INT32_MAX) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  if (!loom_scalar_type_is_integer(element_type)) {
    return false;
  }
  const int32_t element_bit_count = loom_scalar_type_bitwidth(element_type);
  if (element_bit_count <= 0) {
    return false;
  }
  int64_t total_bit_count = 0;
  if (!iree_checked_mul_i64(lane_count, element_bit_count, &total_bit_count) ||
      total_bit_count <= 0) {
    return false;
  }
  const int64_t register_count = (total_bit_count + 31) / 32;
  if (register_count < 1 ||
      register_count > (int64_t)LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return false;
  }
  *out_payload_bit_count = (uint32_t)total_bit_count;
  *out_register_count = (uint32_t)register_count;
  return true;
}

bool loom_amdgpu_type_packed_16bit_float_storage(
    loom_type_t type, uint32_t* out_payload_bit_count,
    uint32_t* out_register_count) {
  IREE_ASSERT_ARGUMENT(out_payload_bit_count);
  IREE_ASSERT_ARGUMENT(out_register_count);
  *out_payload_bit_count = 0;
  *out_register_count = 0;
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count < 1 ||
      lane_count > (int64_t)LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  if (element_type != LOOM_SCALAR_TYPE_F16 &&
      element_type != LOOM_SCALAR_TYPE_BF16) {
    return false;
  }
  const uint32_t register_count = (uint32_t)((lane_count + 1) / 2);
  *out_payload_bit_count = (uint32_t)lane_count * 16u;
  *out_register_count = register_count;
  return true;
}

bool loom_amdgpu_type_is_byte_addressable_view(loom_type_t type) {
  if (!loom_type_is_view(type)) {
    return false;
  }
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  return element_bit_count > 0 && (element_bit_count % 8) == 0;
}

bool loom_amdgpu_value_is_i32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id) {
  return loom_amdgpu_type_is_i32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_address_scalar(loom_low_lower_context_t* context,
                                         loom_value_id_t value_id) {
  return loom_amdgpu_type_is_address_scalar(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_f32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id) {
  return loom_amdgpu_type_is_f32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_byte_addressable_view(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_byte_addressable_view(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static iree_status_t loom_amdgpu_make_register_type(
    loom_low_lower_context_t* context, uint16_t reg_class_id,
    uint32_t unit_count, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(context, reg_class_id, unit_count,
                                           out_type);
}

iree_status_t loom_amdgpu_make_sgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                                        1, out_type);
}

iree_status_t loom_amdgpu_make_sgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                                        unit_count, out_type);
}

iree_status_t loom_amdgpu_make_vgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                        1, out_type);
}

iree_status_t loom_amdgpu_make_scc_type(loom_low_lower_context_t* context,
                                        loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_SCC,
                                        1, out_type);
}

iree_status_t loom_amdgpu_make_vgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                        unit_count, out_type);
}

iree_status_t loom_amdgpu_make_descriptor_implicit_resource_type(
    loom_low_lower_context_t* context, uint64_t descriptor_id,
    loom_type_t* out_type) {
  IREE_ASSERT_ARGUMENT(out_type);
  *out_type = loom_type_none();
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_lookup_descriptor_by_id(descriptor_set,
                                                      descriptor_id);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU descriptor set does not contain requested "
                            "implicit-resource descriptor");
  }
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  if (descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU implicit-resource descriptor ordinal is invalid");
  }

  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand = &operands[i];
    if (operand->role != LOOM_LOW_OPERAND_ROLE_RESOURCE ||
        !iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      continue;
    }
    for (uint16_t j = 0; j < operand->reg_class_alt_count; ++j) {
      const uint32_t alt_index = operand->reg_class_alt_start + j;
      if (alt_index >= descriptor_set->reg_class_alt_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU implicit-resource descriptor has an invalid register-class "
            "alternative span");
      }
      const loom_low_reg_class_alt_t* alt =
          &descriptor_set->reg_class_alts[alt_index];
      if (iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
        continue;
      }
      if (alt->reg_class_id >= descriptor_set->reg_class_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU implicit-resource descriptor references an invalid "
            "register class");
      }
      return loom_amdgpu_make_register_type(context, alt->reg_class_id,
                                            operand->unit_count, out_type);
    }
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU descriptor has no explicit implicit resource operand");
}

iree_status_t loom_amdgpu_low_type_register_class_is(
    loom_low_lower_context_t* context, loom_type_t type, uint16_t reg_class_id,
    bool* out_match) {
  IREE_ASSERT_ARGUMENT(out_match);
  *out_match = false;
  if (!loom_type_is_register(type)) {
    return iree_ok_status();
  }
  loom_string_id_t expected_class_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_register_class_string_id(
      context, reg_class_id, &expected_class_id));
  *out_match = loom_type_register_class_id(type) == expected_class_id;
  return iree_ok_status();
}

bool loom_amdgpu_module_value_prefers_vgpr(const loom_module_t* module,
                                           loom_value_id_t source_value_id) {
  if (source_value_id >= module->values.count) {
    return false;
  }

  loom_type_t source_type = loom_module_value_type(module, source_value_id);
  if (loom_amdgpu_type_is_f32(source_type) ||
      loom_amdgpu_type_is_vector_32bit_lane_range(source_type)) {
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
      return loom_kernel_workitem_id_dimension(defining_op) <
             LOOM_KERNEL_DIMENSION_COUNT_;
    case LOOM_OP_INDEX_ADD:
      return loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_index_add_lhs(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_index_add_rhs(defining_op));
    case LOOM_OP_INDEX_SUB:
      return loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_index_sub_lhs(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_index_sub_rhs(defining_op));
    case LOOM_OP_INDEX_MUL:
      return loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_index_mul_lhs(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_index_mul_rhs(defining_op));
    case LOOM_OP_INDEX_MADD:
      return loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_index_madd_a(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_index_madd_b(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_index_madd_c(defining_op));
    case LOOM_OP_VECTOR_EXTRACT:
      return true;
    case LOOM_OP_VECTOR_REDUCE:
      return loom_amdgpu_type_is_vector_32bit_lane_range(loom_module_value_type(
          module, loom_vector_reduce_input(defining_op)));
    case LOOM_OP_VIEW_ATOMIC_CMPXCHG:
    case LOOM_OP_VIEW_ATOMIC_RMW:
      return true;
    case LOOM_OP_SCALAR_ADDI:
      return loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_addi_lhs(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_addi_rhs(defining_op));
    case LOOM_OP_SCALAR_SUBI:
      return loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_subi_lhs(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_subi_rhs(defining_op));
    case LOOM_OP_SCALAR_MULI:
      return loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_muli_lhs(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_muli_rhs(defining_op));
    case LOOM_OP_SCALAR_ANDI:
      return loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_andi_lhs(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_andi_rhs(defining_op));
    case LOOM_OP_SCALAR_ORI:
      return loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_ori_lhs(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_ori_rhs(defining_op));
    case LOOM_OP_SCALAR_XORI:
      return loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_xori_lhs(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_xori_rhs(defining_op));
    case LOOM_OP_SCALAR_SHLI:
      return loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_shli_lhs(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_shli_rhs(defining_op));
    case LOOM_OP_SCALAR_SHRSI:
      return loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_shrsi_lhs(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_shrsi_rhs(defining_op));
    case LOOM_OP_SCALAR_SHRUI:
      return loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_shrui_lhs(defining_op)) ||
             loom_amdgpu_module_value_prefers_vgpr(
                 module, loom_scalar_shrui_rhs(defining_op));
    default:
      return false;
  }
}

bool loom_amdgpu_value_prefers_vgpr(loom_low_lower_context_t* context,
                                    loom_value_id_t source_value_id) {
  return loom_amdgpu_module_value_prefers_vgpr(
      loom_low_lower_context_module(context), source_value_id);
}

iree_status_t loom_amdgpu_map_type(void* user_data,
                                   loom_low_lower_context_t* context,
                                   const loom_op_t* source_op,
                                   loom_type_t source_type,
                                   loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_amdgpu_type_is_i1(source_type)) {
    return loom_amdgpu_make_scc_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_i32(source_type)) {
    return loom_amdgpu_make_sgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_address_scalar(source_type)) {
    return loom_amdgpu_make_sgpr_type(context, out_low_type);
  }
  const uint32_t vector_lane_count =
      loom_amdgpu_vector_32bit_lane_count(source_type);
  if (vector_lane_count == 1) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  if (vector_lane_count > 1) {
    return loom_amdgpu_make_register_type(context,
                                          LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                          vector_lane_count, out_low_type);
  }
  const uint32_t mask_lane_count =
      loom_amdgpu_vector_i1_lane_count(source_type);
  if (mask_lane_count != 0) {
    return loom_amdgpu_make_sgpr_range_type(context, mask_lane_count * 2u,
                                            out_low_type);
  }
  uint32_t unused_payload_bit_count = 0;
  uint32_t packed_register_count = 0;
  if (loom_amdgpu_type_packed_16bit_float_storage(
          source_type, &unused_payload_bit_count, &packed_register_count)) {
    return loom_amdgpu_make_register_type(context,
                                          LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                          packed_register_count, out_low_type);
  }
  if (loom_amdgpu_type_packed_integer_storage(
          source_type, &unused_payload_bit_count, &packed_register_count)) {
    if (packed_register_count == 1) {
      return loom_amdgpu_make_vgpr_type(context, out_low_type);
    }
    return loom_amdgpu_make_register_type(context,
                                          LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                          packed_register_count, out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("AMDGPU lowering currently supports only i1 and i32 scalar "
              "values, "
              "address scalar values, rank-1 static i32/f32 vectors with "
              "1 to 8 lanes, rank-1 static i1 mask vectors with 1 to 8 lanes, "
              "rank-1 static f16/bf16 vectors that fit in 1 to 8 packed "
              "32-bit registers, "
              "and rank-1 static integer vectors that fit in 1 to 4 packed "
              "32-bit registers"));
}

iree_status_t loom_amdgpu_map_value(void* user_data,
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

bool loom_amdgpu_module_value_as_exact_index_constant(
    const loom_module_t* module, loom_value_id_t value_id, int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_index_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_index_constant_value(defining_op);
  if (attr.kind != LOOM_ATTR_I64) {
    return false;
  }
  *out_value = attr.i64;
  return true;
}

bool loom_amdgpu_module_value_as_i32_constant(const loom_module_t* module,
                                              loom_value_id_t value_id,
                                              int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  if (!loom_amdgpu_type_is_i32(loom_module_value_type(module, value_id))) {
    return false;
  }
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

bool loom_amdgpu_module_value_as_f32_constant(const loom_module_t* module,
                                              loom_value_id_t value_id,
                                              uint32_t* out_bit_pattern) {
  IREE_ASSERT_ARGUMENT(out_bit_pattern);
  *out_bit_pattern = 0;
  if (!loom_amdgpu_type_is_f32(loom_module_value_type(module, value_id))) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_scalar_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_scalar_constant_value(defining_op);
  if (!loom_amdgpu_attr_is_f32_immediate(attr)) {
    return false;
  }
  *out_bit_pattern = loom_amdgpu_attr_f32_bit_pattern(attr);
  return true;
}

bool loom_amdgpu_value_facts_as_exact_non_negative_i64(loom_value_facts_t facts,
                                                       int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts) ||
      facts.range_lo < 0) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

bool loom_amdgpu_u32_is_power_of_two(uint32_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool loom_amdgpu_attr_is_i32_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_I64 && value.i64 >= INT32_MIN &&
         value.i64 <= INT32_MAX;
}

bool loom_amdgpu_attr_is_f32_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_F64;
}

uint32_t loom_amdgpu_attr_f32_bit_pattern(loom_attribute_t value) {
  const float f32_value = (float)loom_attr_as_f64(value);
  uint32_t bit_pattern = 0;
  memcpy(&bit_pattern, &f32_value, sizeof(bit_pattern));
  return bit_pattern;
}

bool loom_amdgpu_value_as_i32_constant(loom_low_lower_context_t* context,
                                       loom_value_id_t value_id,
                                       int64_t* out_value) {
  return loom_amdgpu_module_value_as_i32_constant(
      loom_low_lower_context_module(context), value_id, out_value);
}

bool loom_amdgpu_value_as_f32_constant(loom_low_lower_context_t* context,
                                       loom_value_id_t value_id,
                                       uint32_t* out_bit_pattern) {
  return loom_amdgpu_module_value_as_f32_constant(
      loom_low_lower_context_module(context), value_id, out_bit_pattern);
}

bool loom_amdgpu_value_as_address_constant(loom_low_lower_context_t* context,
                                           loom_value_id_t value_id,
                                           int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  if (!loom_amdgpu_value_is_address_scalar(context, value_id)) {
    return false;
  }
  return loom_amdgpu_module_value_as_exact_index_constant(
      loom_low_lower_context_module(context), value_id, out_value);
}

bool loom_amdgpu_value_can_materialize_as_vgpr_i32(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  int64_t unused_value = 0;
  const loom_module_t* module = loom_low_lower_context_module(context);
  return loom_amdgpu_module_value_prefers_vgpr(module, value_id) ||
         loom_amdgpu_module_value_as_i32_constant(module, value_id,
                                                  &unused_value);
}

bool loom_amdgpu_value_can_materialize_as_vgpr_address(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  int64_t unused_value = 0;
  return loom_amdgpu_value_prefers_vgpr(context, value_id) ||
         loom_amdgpu_value_as_address_constant(context, value_id,
                                               &unused_value);
}
