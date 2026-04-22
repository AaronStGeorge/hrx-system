// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"

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

iree_status_t loom_amdgpu_select_integer_plan(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
  bool selected = false;
  switch (source_op->kind) {
    case LOOM_OP_INDEX_ADD:
      selected = loom_amdgpu_can_lower_address_binary(
          context, loom_index_add_lhs(source_op), loom_index_add_rhs(source_op),
          loom_index_add_result(source_op), /*allow_uniform_result=*/true);
      break;
    case LOOM_OP_INDEX_SUB:
      selected = loom_amdgpu_can_lower_address_binary(
          context, loom_index_sub_lhs(source_op), loom_index_sub_rhs(source_op),
          loom_index_sub_result(source_op), /*allow_uniform_result=*/true);
      break;
    case LOOM_OP_INDEX_MUL:
      selected = loom_amdgpu_can_lower_address_binary(
          context, loom_index_mul_lhs(source_op), loom_index_mul_rhs(source_op),
          loom_index_mul_result(source_op), /*allow_uniform_result=*/false);
      break;
    case LOOM_OP_INDEX_MADD:
      selected = loom_amdgpu_can_lower_index_madd(context, source_op);
      break;
    case LOOM_OP_SCALAR_ADDI:
      selected = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_addi_lhs(source_op),
          loom_scalar_addi_rhs(source_op), loom_scalar_addi_result(source_op),
          /*allow_uniform_result=*/true);
      break;
    case LOOM_OP_SCALAR_SUBI:
      selected = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_subi_lhs(source_op),
          loom_scalar_subi_rhs(source_op), loom_scalar_subi_result(source_op),
          /*allow_uniform_result=*/true);
      break;
    case LOOM_OP_SCALAR_MULI:
      selected = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_muli_lhs(source_op),
          loom_scalar_muli_rhs(source_op), loom_scalar_muli_result(source_op),
          /*allow_uniform_result=*/false);
      break;
    case LOOM_OP_SCALAR_ANDI:
      selected = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_andi_lhs(source_op),
          loom_scalar_andi_rhs(source_op), loom_scalar_andi_result(source_op),
          /*allow_uniform_result=*/true);
      break;
    case LOOM_OP_SCALAR_ORI:
      selected = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_ori_lhs(source_op),
          loom_scalar_ori_rhs(source_op), loom_scalar_ori_result(source_op),
          /*allow_uniform_result=*/true);
      break;
    case LOOM_OP_SCALAR_XORI:
      selected = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_xori_lhs(source_op),
          loom_scalar_xori_rhs(source_op), loom_scalar_xori_result(source_op),
          /*allow_uniform_result=*/true);
      break;
    case LOOM_OP_SCALAR_SHLI:
      selected = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_shli_lhs(source_op),
          loom_scalar_shli_rhs(source_op), loom_scalar_shli_result(source_op),
          /*allow_uniform_result=*/true);
      break;
    case LOOM_OP_SCALAR_SHRSI:
      selected = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_shrsi_lhs(source_op),
          loom_scalar_shrsi_rhs(source_op), loom_scalar_shrsi_result(source_op),
          /*allow_uniform_result=*/true);
      break;
    case LOOM_OP_SCALAR_SHRUI:
      selected = loom_amdgpu_can_lower_i32_binary(
          context, loom_scalar_shrui_lhs(source_op),
          loom_scalar_shrui_rhs(source_op), loom_scalar_shrui_result(source_op),
          /*allow_uniform_result=*/true);
      break;
    default:
      return iree_ok_status();
  }
  if (selected) {
    *out_plan = loom_low_lower_plan_make(source_op->kind, 0, NULL);
  }
  return iree_ok_status();
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
    IREE_ASSERT_UNREACHABLE();
    return iree_ok_status();
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
    IREE_ASSERT_UNREACHABLE();
    return iree_ok_status();
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

iree_status_t loom_amdgpu_lower_integer_op(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op) {
  switch (source_op->kind) {
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
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
}
