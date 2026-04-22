// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"

iree_status_t loom_amdgpu_intern(loom_low_lower_context_t* context,
                                 iree_string_view_t string,
                                 loom_string_id_t* out_string_id) {
  IREE_ASSERT_ARGUMENT(out_string_id);
  return loom_module_intern_string(loom_low_lower_context_module(context),
                                   string, out_string_id);
}

iree_status_t loom_amdgpu_low_result_type(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t source_result,
                                          loom_type_t* out_low_type) {
  IREE_ASSERT_ARGUMENT(out_low_type);
  *out_low_type = loom_type_none();
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
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU i32 value cannot materialize as a VGPR operand");
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
        "AMDGPU address value cannot materialize as a VGPR operand");
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
