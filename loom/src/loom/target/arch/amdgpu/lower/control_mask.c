// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/control_mask.h"

#include <string.h>

#include "loom/codegen/low/lower/control_plan.h"
#include "loom/ir/module.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

bool loom_amdgpu_low_type_is_native_i1_mask(loom_low_lower_context_t* context,
                                            loom_type_t low_type) {
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  return is_sgpr && loom_low_register_type_unit_count(low_type) == 2;
}

static uint8_t loom_amdgpu_mask_merge_constant(
    loom_low_lower_context_t* context, loom_value_slice_t args,
    uint16_t index) {
  bool value = false;
  if (index < args.count &&
      loom_amdgpu_value_as_i1_constant(context, args.values[index], &value)) {
    return value ? LOOM_AMDGPU_MASK_VALUE_TRUE : LOOM_AMDGPU_MASK_VALUE_FALSE;
  }
  return LOOM_AMDGPU_MASK_VALUE_DYNAMIC;
}

iree_status_t loom_amdgpu_prepare_masked_merge(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_block_t* source_merge_dest,
    loom_low_lower_block_ref_t merge_low_dest, loom_value_slice_t false_args,
    loom_value_slice_t true_args,
    const loom_amdgpu_mask_merge_constants_t** out_constants) {
  *out_constants = NULL;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const uint16_t count =
      loom_low_lower_control_carrier_argument_offset(context, merge_low_dest);
  loom_amdgpu_mask_merge_constants_t* constants = NULL;
  const uint32_t previous_error_count =
      loom_low_lower_context_error_count(context);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0;
       i < count && iree_status_is_ok(status) &&
       loom_low_lower_context_error_count(context) == previous_error_count;
       ++i) {
    const loom_type_t type =
        loom_low_lower_control_argument_type(context, merge_low_dest, i);
    if (i < source_merge_dest->arg_count &&
        loom_amdgpu_type_is_i1(
            loom_block_arg_type(module, source_merge_dest, i)) &&
        loom_amdgpu_low_type_is_native_i1_mask(context, type)) {
      const loom_amdgpu_mask_merge_constants_t choice = {
          .false_value =
              loom_amdgpu_mask_merge_constant(context, false_args, i),
          .true_value = loom_amdgpu_mask_merge_constant(context, true_args, i),
      };
      if (choice.false_value == LOOM_AMDGPU_MASK_VALUE_DYNAMIC &&
          choice.true_value == LOOM_AMDGPU_MASK_VALUE_DYNAMIC) {
        continue;
      }
      if (constants == NULL) {
        status = loom_low_lower_allocate_function_array(
            context, count, sizeof(*constants), (void**)&constants);
        if (iree_status_is_ok(status)) {
          memset(constants, 0, count * sizeof(*constants));
        }
      }
      if (iree_status_is_ok(status)) {
        constants[i] = choice;
      }
      continue;
    }
    const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
        context, type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
    const uint32_t lane_count = loom_low_register_type_unit_count(type);
    if (!is_vgpr || lane_count == 0 ||
        lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
      status = loom_low_lower_emit_branch_constraint(
          context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
    }
  }
  if (iree_status_is_ok(status)) {
    *out_constants = constants;
  }
  return status;
}

static iree_status_t loom_amdgpu_emit_zero_vgpr_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t value_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, value_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (!is_vgpr) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
  }

  const uint32_t lane_count = loom_low_register_type_unit_count(value_type);
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
  }
  loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(value_type, 1);
  if (lane_count == 1) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      lane_type, out_value);
  }

  loom_value_id_t zero_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
      &zero_lane));
  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    lanes[i] = zero_lane;
  }
  return loom_amdgpu_build_low_register_range(
      context, source_op, lanes, lane_count, value_type, out_value);
}

iree_status_t loom_amdgpu_emit_zero_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t value_type, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_low_type_is_native_i1_mask(context, value_type)) {
    return loom_low_lower_emit_branch_constraint(
        context, source_op, IREE_SV("masked_region_merge_vgpr_values"));
  }
  loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(value_type, 1);
  loom_value_id_t zero_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0, lane_type,
      &zero_lane));
  const loom_value_id_t lanes[] = {zero_lane, zero_lane};
  return loom_amdgpu_build_low_register_range(
      context, source_op, lanes, IREE_ARRAYSIZE(lanes), value_type, out_value);
}

iree_status_t loom_amdgpu_emit_zero_merge_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t value_type, loom_value_id_t* out_value) {
  if (loom_amdgpu_low_type_is_native_i1_mask(context, value_type)) {
    return loom_amdgpu_emit_zero_native_i1_mask(context, source_op, value_type,
                                                out_value);
  }
  return loom_amdgpu_emit_zero_vgpr_value(context, source_op, value_type,
                                          out_value);
}

static iree_status_t loom_amdgpu_emit_masked_merge_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_condition, loom_type_t lane_type,
    loom_value_id_t* out_merged_value) {
  const loom_value_id_t select_operands[] = {
      low_false_value,
      low_true_value,
      low_condition,
  };
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      select_operands, IREE_ARRAYSIZE(select_operands),
      loom_named_attr_slice_empty(), &lane_type, 1, &select_op));
  *out_merged_value = loom_value_slice_get(loom_low_op_results(select_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_native_i1_mask_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &op));
  *out_result = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_native_i1_mask_merge(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_amdgpu_mask_merge_constants_t constants,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_condition, loom_type_t mask_type,
    loom_value_id_t* out_merged_value) {
  *out_merged_value = LOOM_VALUE_ID_INVALID;
  const bool true_is_constant =
      constants.true_value != LOOM_AMDGPU_MASK_VALUE_DYNAMIC;
  const bool true_constant =
      constants.true_value == LOOM_AMDGPU_MASK_VALUE_TRUE;
  const bool false_is_constant =
      constants.false_value != LOOM_AMDGPU_MASK_VALUE_DYNAMIC;
  const bool false_constant =
      constants.false_value == LOOM_AMDGPU_MASK_VALUE_TRUE;
  if (true_is_constant && false_is_constant) {
    if (true_constant && false_constant) {
      *out_merged_value = saved_exec;
      return iree_ok_status();
    }
    if (!true_constant && !false_constant) {
      return loom_amdgpu_emit_zero_native_i1_mask(context, source_op, mask_type,
                                                  out_merged_value);
    }
  }

  if (low_true_value == low_false_value) {
    *out_merged_value = low_true_value;
    return iree_ok_status();
  }

  if (true_is_constant && true_constant && false_is_constant &&
      !false_constant) {
    return loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, saved_exec,
        low_condition, mask_type, out_merged_value);
  }

  loom_value_id_t true_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64, saved_exec,
      low_condition, mask_type, &true_condition));

  if (true_is_constant && !true_constant && false_is_constant &&
      false_constant) {
    return loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
        true_condition, saved_exec, mask_type, out_merged_value);
  }

  loom_value_id_t true_mask = LOOM_VALUE_ID_INVALID;
  bool true_mask_is_zero = false;
  if (true_is_constant && true_constant) {
    true_mask = true_condition;
  } else if (true_is_constant && !true_constant) {
    true_mask_is_zero = true;
  } else if (low_true_value == low_condition) {
    true_mask = true_condition;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
        true_condition, low_true_value, mask_type, &true_mask));
  }

  if ((false_is_constant && !false_constant) ||
      low_false_value == low_condition) {
    if (true_mask_is_zero) {
      return loom_amdgpu_emit_zero_native_i1_mask(context, source_op, mask_type,
                                                  out_merged_value);
    }
    *out_merged_value = true_mask;
    return iree_ok_status();
  }

  loom_value_id_t false_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64, true_condition,
      saved_exec, mask_type, &false_condition));
  loom_value_id_t false_mask = LOOM_VALUE_ID_INVALID;
  if (false_is_constant && false_constant) {
    false_mask = false_condition;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
        false_condition, low_false_value, mask_type, &false_mask));
  }
  if (true_mask_is_zero) {
    *out_merged_value = false_mask;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_native_i1_mask_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64, true_mask,
      false_mask, mask_type, out_merged_value);
}

iree_status_t loom_amdgpu_emit_masked_merge_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_amdgpu_mask_merge_constants_t constants,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_condition, loom_type_t value_type,
    loom_value_id_t* out_merged_value) {
  *out_merged_value = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_low_type_is_native_i1_mask(context, value_type)) {
    return loom_amdgpu_emit_native_i1_mask_merge(
        context, source_op, saved_exec, constants, low_false_value,
        low_true_value, low_condition, value_type, out_merged_value);
  }

  const uint32_t lane_count = loom_low_register_type_unit_count(value_type);
  loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(value_type, 1);
  if (lane_count == 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, low_false_value, &low_false_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, low_true_value, &low_true_value));
    return loom_amdgpu_emit_masked_merge_lane(
        context, source_op, low_false_value, low_true_value, low_condition,
        lane_type, out_merged_value);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_false_value, &low_false_value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_true_value, &low_true_value));

  loom_value_id_t lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t false_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_false_value, i, lane_type, &false_lane));
    loom_value_id_t true_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_true_value, i, lane_type, &true_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_masked_merge_lane(
        context, source_op, false_lane, true_lane, low_condition, lane_type,
        &lanes[i]));
  }

  return loom_amdgpu_build_low_register_range(
      context, source_op, lanes, lane_count, value_type, out_merged_value);
}
