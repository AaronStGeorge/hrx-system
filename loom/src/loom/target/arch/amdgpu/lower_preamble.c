// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/context.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/hal_kernel_abi.h"
#include "loom/target/arch/amdgpu/lower_internal.h"

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

iree_status_t loom_amdgpu_select_preamble_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
  bool selected = false;
  switch (source_op->kind) {
    case LOOM_OP_KERNEL_WORKITEM_ID:
      selected = loom_amdgpu_can_lower_workitem_id(context, source_op);
      break;
    case LOOM_OP_KERNEL_WORKGROUP_ID:
      selected = loom_amdgpu_can_lower_workgroup_id(context, source_op);
      break;
    default:
      return iree_ok_status();
  }
  if (selected) {
    *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
  }
  return iree_ok_status();
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
  IREE_ASSERT(body != NULL);
  IREE_ASSERT_GT(body->block_count, 0);
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
  IREE_ASSERT_UNREACHABLE();
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_preamble(void* user_data,
                                        loom_low_lower_context_t* context) {
  (void)user_data;
  const loom_op_t* first_workitem_id_ops[LOOM_KERNEL_DIMENSION_COUNT_] = {0};
  const loom_op_t* first_workgroup_id_ops[LOOM_KERNEL_DIMENSION_COUNT_] = {0};
  const loom_op_t* first_m0_op = NULL;
  uint64_t m0_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  const iree_host_size_t plan_count =
      loom_low_lower_context_selected_plan_count(context);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    const loom_op_t* source_op = selected_plan.source_op;
    const loom_low_lower_plan_t plan = selected_plan.plan;
    switch (plan.id) {
      case LOOM_OP_VECTOR_LOAD:
      case LOOM_OP_VECTOR_STORE: {
        if (first_m0_op != NULL) {
          continue;
        }
        const loom_amdgpu_memory_access_plan_t* access =
            (const loom_amdgpu_memory_access_plan_t*)plan.target_data;
        IREE_ASSERT(access != NULL);
        if (access->address_form != LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
          continue;
        }
        first_m0_op = source_op;
        m0_descriptor_id = access->descriptor_id;
        break;
      }
      case LOOM_OP_KERNEL_WORKITEM_ID: {
        const loom_kernel_dimension_t dimension =
            loom_kernel_workitem_id_dimension(source_op);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        if (first_workitem_id_ops[dimension] == NULL) {
          first_workitem_id_ops[dimension] = source_op;
        }
        break;
      }
      case LOOM_OP_KERNEL_WORKGROUP_ID: {
        const loom_kernel_dimension_t dimension =
            loom_kernel_workgroup_id_dimension(source_op);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        if (first_workgroup_id_ops[dimension] == NULL) {
          first_workgroup_id_ops[dimension] = source_op;
        }
        break;
      }
      default:
        break;
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

  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    const loom_op_t* source_op = selected_plan.source_op;
    const loom_low_lower_plan_t plan = selected_plan.plan;
    switch (plan.id) {
      case LOOM_OP_KERNEL_WORKITEM_ID: {
        const loom_kernel_dimension_t dimension =
            loom_kernel_workitem_id_dimension(source_op);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        IREE_ASSERT_NE(low_workitem_ids[dimension], LOOM_VALUE_ID_INVALID);
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
            context, loom_kernel_workitem_id_result(source_op),
            low_workitem_ids[dimension]));
        break;
      }
      case LOOM_OP_KERNEL_WORKGROUP_ID: {
        const loom_kernel_dimension_t dimension =
            loom_kernel_workgroup_id_dimension(source_op);
        IREE_ASSERT_LT(dimension, LOOM_KERNEL_DIMENSION_COUNT_);
        IREE_ASSERT_NE(low_workgroup_ids[dimension], LOOM_VALUE_ID_INVALID);
        IREE_RETURN_IF_ERROR(loom_low_lower_bind_value(
            context, loom_kernel_workgroup_id_result(source_op),
            low_workgroup_ids[dimension]));
        break;
      }
      default:
        break;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_preamble_op(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  switch (source_op->kind) {
    case LOOM_OP_KERNEL_WORKITEM_ID: {
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      return loom_low_lower_lookup_value(
          context, loom_kernel_workitem_id_result(source_op), &low_result);
    }
    case LOOM_OP_KERNEL_WORKGROUP_ID: {
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      return loom_low_lower_lookup_value(
          context, loom_kernel_workgroup_id_result(source_op), &low_result);
    }
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
}
