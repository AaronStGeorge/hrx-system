// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"
#include "loom/util/fact_table.h"

bool loom_amdgpu_type_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
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

bool loom_amdgpu_type_is_32bit_view(loom_type_t type) {
  return loom_type_is_view(type) &&
         (loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 ||
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32);
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

bool loom_amdgpu_value_is_vector_32bit_lane_range(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_vector_32bit_lane_range(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_32bit_view(loom_low_lower_context_t* context,
                                     loom_value_id_t value_id) {
  return loom_amdgpu_type_is_32bit_view(
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
      IREE_SV("AMDGPU lowering currently supports only i32 scalar values, "
              "address scalar values, rank-1 static i32/f32 vectors with "
              "1 to 8 lanes, rank-1 static i1 mask vectors with 1 to 8 lanes, "
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

typedef struct loom_amdgpu_buffer_argument_extent_t {
  // Module containing the source function being inspected.
  const loom_module_t* module;
  // True once a buffer.view derived from the source argument is found.
  bool found_view;
  // True when a derived view has no exact static dense byte extent.
  bool found_unbounded_view;
  // Maximum byte count required by all statically boundable derived views.
  int64_t valid_byte_count;
} loom_amdgpu_buffer_argument_extent_t;

static bool loom_amdgpu_view_static_dense_byte_extent(
    const loom_module_t* module, loom_type_t view_type,
    int64_t* out_byte_extent) {
  IREE_ASSERT_ARGUMENT(out_byte_extent);
  *out_byte_extent = 0;
  if (!loom_type_is_view(view_type)) {
    return false;
  }

  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_type_address_layout(
          /*context=*/NULL, module, view_type, stride_storage,
          IREE_ARRAYSIZE(stride_storage), &layout) ||
      layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
    return false;
  }

  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(view_type));
  if (element_bit_count <= 0 || (element_bit_count % 8) != 0) {
    return false;
  }

  int64_t element_count = 1;
  const uint8_t rank = loom_type_rank(view_type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(view_type, i)) {
      return false;
    }
    const int64_t dim_size = loom_type_dim_static_size_at(view_type, i);
    if (dim_size < 0 ||
        !iree_checked_mul_i64(element_count, dim_size, &element_count)) {
      return false;
    }
  }

  const int64_t element_byte_count = element_bit_count / 8;
  return iree_checked_mul_i64(element_count, element_byte_count,
                              out_byte_extent);
}

static void loom_amdgpu_buffer_argument_extent_include_view(
    loom_amdgpu_buffer_argument_extent_t* state,
    const loom_op_t* buffer_view_op) {
  if (state->found_unbounded_view) {
    return;
  }
  state->found_view = true;
  int64_t base_byte_offset = 0;
  int64_t view_byte_extent = 0;
  int64_t valid_byte_count = 0;
  if (!loom_amdgpu_module_value_as_exact_index_constant(
          state->module, loom_buffer_view_byte_offset(buffer_view_op),
          &base_byte_offset) ||
      base_byte_offset < 0 ||
      !loom_amdgpu_view_static_dense_byte_extent(
          state->module,
          loom_module_value_type(state->module,
                                 loom_buffer_view_result(buffer_view_op)),
          &view_byte_extent) ||
      !iree_checked_add_i64(base_byte_offset, view_byte_extent,
                            &valid_byte_count)) {
    state->found_unbounded_view = true;
    return;
  }
  state->valid_byte_count = iree_max(state->valid_byte_count, valid_byte_count);
}

static bool loom_amdgpu_source_buffer_argument_valid_byte_count(
    loom_low_lower_context_t* context, loom_value_id_t source_argument_id,
    int64_t* out_valid_byte_count) {
  IREE_ASSERT_ARGUMENT(out_valid_byte_count);
  *out_valid_byte_count = 0;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_t* source_argument =
      loom_module_value(module, source_argument_id);
  loom_amdgpu_buffer_argument_extent_t state = {
      .module = module,
  };
  const loom_use_t* use = NULL;
  loom_value_for_each_use(source_argument, use) {
    const loom_op_t* user_op = loom_use_user_op(*use);
    if (loom_use_operand_index(*use) == 0 && loom_buffer_view_isa(user_op)) {
      loom_amdgpu_buffer_argument_extent_include_view(&state, user_op);
    } else {
      state.found_unbounded_view = true;
    }
  }
  if (!state.found_view || state.found_unbounded_view) {
    return false;
  }
  *out_valid_byte_count = state.valid_byte_count;
  return true;
}

iree_status_t loom_amdgpu_map_argument(
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
    loom_low_resource_build_flags_t resource_build_flags = 0;
    int64_t resource_valid_byte_count = 0;
    if (loom_amdgpu_source_buffer_argument_valid_byte_count(
            context, source_argument_id, &resource_valid_byte_count)) {
      resource_build_flags |= LOOM_LOW_RESOURCE_BUILD_FLAG_HAS_VALID_BYTE_COUNT;
    }
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE,
        .abi_type = resource_type,
        .resource_import_kind =
            LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BUFFER_RESOURCE,
        .resource_index = loom_amdgpu_hal_buffer_resource_index(
            context, source_argument_index),
        .resource_semantic_type = semantic_type,
        .resource_build_flags = resource_build_flags,
        .resource_valid_byte_count = resource_valid_byte_count,
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

static bool loom_amdgpu_value_as_address_constant(
    loom_low_lower_context_t* context, loom_value_id_t value_id,
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
  return loom_amdgpu_value_prefers_vgpr(context, value_id) ||
         loom_amdgpu_value_as_i32_constant(context, value_id, &unused_value);
}

bool loom_amdgpu_value_can_materialize_as_vgpr_address(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  int64_t unused_value = 0;
  return loom_amdgpu_value_prefers_vgpr(context, value_id) ||
         loom_amdgpu_value_as_address_constant(context, value_id,
                                               &unused_value);
}

iree_status_t loom_amdgpu_intern(loom_low_lower_context_t* context,
                                 iree_string_view_t string,
                                 loom_string_id_t* out_string_id) {
  return loom_module_intern_string(loom_low_lower_context_module(context),
                                   string, out_string_id);
}

iree_status_t loom_amdgpu_low_result_type(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t source_result,
                                          loom_type_t* out_low_type) {
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_result, out_low_type));
  if (!loom_type_is_register(*out_low_type)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU source type did not map to a register");
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_low_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attrs,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(out_op);
  *out_op = NULL;
  return loom_low_lower_emit_descriptor_op(
      context, descriptor_id, operands, operand_count, attrs, result_types,
      result_count, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, out_op);
}

iree_status_t loom_amdgpu_emit_const_u32(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         uint64_t descriptor_id, uint32_t value,
                                         loom_type_t result_type,
                                         loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
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
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_const(
      context, descriptor_id,
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), result_type,
      source_op->location, &low_const));
  *out_value_id = loom_low_const_result(low_const);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_vgpr_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t lane_type, loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_id, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_vgpr_shift(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          uint64_t descriptor_id,
                                          uint32_t shift, loom_value_id_t value,
                                          loom_type_t lane_type,
                                          loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;
  if (shift == 0) {
    *out_value = value;
    return iree_ok_status();
  }

  loom_value_id_t shift_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32, shift, lane_type,
      &shift_value));
  return loom_amdgpu_emit_vgpr_binary(context, source_op, descriptor_id,
                                      shift_value, value, lane_type, out_value);
}

iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_i32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  IREE_ASSERT_ARGUMENT(out_low_value);
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr) {
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
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32,
      (uint32_t)(int32_t)value, vgpr_type, out_low_value);
}

iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  IREE_ASSERT_ARGUMENT(out_low_value);
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr) {
    *out_low_value = low_value;
    return iree_ok_status();
  }

  int64_t value = 0;
  if (!loom_amdgpu_value_as_address_constant(context, source_value, &value) ||
      value < 0 || value > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "preflight accepted AMDGPU address value that cannot materialize as a "
        "VGPR operand");
  }
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32,
                                    (uint32_t)value, vgpr_type, out_low_value);
}

iree_status_t loom_amdgpu_emit_low_slice(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_value_id_t low_source,
                                         uint32_t offset,
                                         loom_type_t result_type,
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

bool loom_amdgpu_low_legality_bundle_is_amdgpu(
    const loom_target_bundle_t* bundle) {
  return bundle != NULL && bundle->config != NULL &&
         iree_string_view_starts_with(bundle->config->contract_set_key,
                                      IREE_SV("amdgpu."));
}
