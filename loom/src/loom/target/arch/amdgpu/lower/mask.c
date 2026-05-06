// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/compare_candidates.h"
#include "loom/target/arch/amdgpu/lower/internal.h"
#include "loom/target/arch/amdgpu/target_refs.h"

static bool loom_amdgpu_vector_compare_descriptor_ref(
    loom_op_kind_t op_kind, uint8_t predicate,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  for (iree_host_size_t i = 0; i < kLoomAmdgpuCompareDescriptorCandidateCount;
       ++i) {
    const loom_amdgpu_compare_descriptor_candidate_t* row =
        &kLoomAmdgpuCompareDescriptorCandidates[i];
    if (row->op_kind == op_kind && row->predicate == predicate) {
      *out_descriptor_ref = row->descriptor_ref;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_amdgpu_select_vector_compare_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_value_id_t result,
    loom_scalar_type_t payload_element_type, uint8_t predicate,
    loom_amdgpu_vector_compare_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_compare_plan_t){0};
  *out_selected = false;
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_vector_compare_descriptor_ref(source_op->kind, predicate,
                                                 &descriptor_ref)) {
    return iree_ok_status();
  }
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, descriptor_ref, &descriptor, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t lhs_type = loom_module_value_type(module, lhs);
  const uint32_t lhs_lane_count = loom_amdgpu_static_vector_lane_count(
      lhs_type, payload_element_type, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  if (lhs_lane_count == 0 ||
      !loom_type_equal(loom_module_value_type(module, rhs), lhs_type) ||
      loom_amdgpu_vector_i1_lane_count(
          loom_module_value_type(module, result)) != lhs_lane_count) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_vector_compare_plan_t){
      .descriptor = descriptor,
      .lhs = lhs,
      .rhs = rhs,
      .result = result,
      .lane_count = lhs_lane_count,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_cmpi_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_compare_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_select_vector_compare_plan(
      context, source_op, loom_vector_cmpi_lhs(source_op),
      loom_vector_cmpi_rhs(source_op), loom_vector_cmpi_result(source_op),
      LOOM_SCALAR_TYPE_I32, loom_vector_cmpi_predicate(source_op), out_plan,
      out_selected);
}

iree_status_t loom_amdgpu_select_vector_cmpf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_compare_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_select_vector_compare_plan(
      context, source_op, loom_vector_cmpf_lhs(source_op),
      loom_vector_cmpf_rhs(source_op), loom_vector_cmpf_result(source_op),
      LOOM_SCALAR_TYPE_F32, loom_vector_cmpf_predicate(source_op), out_plan,
      out_selected);
}

iree_status_t loom_amdgpu_select_scalar_cmpf_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_compare_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_compare_plan_t){0};
  *out_selected = false;
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_vector_compare_descriptor_ref(
          source_op->kind, loom_scalar_cmpf_predicate(source_op),
          &descriptor_ref)) {
    return iree_ok_status();
  }
  loom_low_lower_resolved_descriptor_t descriptor = {0};
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, descriptor_ref, &descriptor, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t lhs = loom_scalar_cmpf_lhs(source_op);
  const loom_value_id_t rhs = loom_scalar_cmpf_rhs(source_op);
  const loom_value_id_t result = loom_scalar_cmpf_result(source_op);
  if (!loom_amdgpu_type_is_f32(loom_module_value_type(module, lhs)) ||
      !loom_type_equal(loom_module_value_type(module, rhs),
                       loom_module_value_type(module, lhs)) ||
      !loom_amdgpu_type_is_i1(loom_module_value_type(module, result))) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_vector_compare_plan_t){
      .descriptor = descriptor,
      .lhs = lhs,
      .rhs = rhs,
      .result = result,
      .lane_count = 1,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_select_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_select_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_select_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_vector_select_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(result_type);
  const loom_value_id_t condition = loom_vector_select_condition(source_op);
  const loom_value_id_t true_value = loom_vector_select_true_value(source_op);
  const loom_value_id_t false_value = loom_vector_select_false_value(source_op);
  if (lane_count == 0 ||
      loom_amdgpu_vector_i1_lane_count(
          loom_module_value_type(module, condition)) != lane_count ||
      !loom_type_equal(loom_module_value_type(module, true_value),
                       result_type) ||
      !loom_type_equal(loom_module_value_type(module, false_value),
                       result_type)) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_vector_select_plan_t){
      .condition = condition,
      .true_value = true_value,
      .false_value = false_value,
      .result = result,
      .lane_count = lane_count,
  };
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_slice_lane_if_needed(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t lane_count, uint32_t unit_offset,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (lane_count == 1) {
    *out_lane = low_source;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, low_source, unit_offset,
                                    lane_type, out_lane);
}

static iree_status_t loom_amdgpu_lower_vector_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan) {
  const uint32_t lane_count = plan->lane_count;
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->lhs, &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->rhs, &low_rhs));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t lane_lhs = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
        context, source_op, low_lhs, lane_count, i, lane_type, &lane_lhs));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, lane_lhs, &lane_lhs));
    loom_value_id_t lane_rhs = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
        context, source_op, low_rhs, lane_count, i, lane_type, &lane_rhs));
    IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, lane_rhs, &lane_rhs));
    const loom_value_id_t operands[] = {
        lane_lhs,
        lane_rhs,
    };
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(NULL, 0), &mask_lane_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &lane_op));
    lane_results[i] = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }

  if (lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, lane_results[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_vector_cmpi(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan) {
  return loom_amdgpu_lower_vector_compare(context, source_op, plan);
}

iree_status_t loom_amdgpu_lower_vector_cmpf(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan) {
  return loom_amdgpu_lower_vector_compare(context, source_op, plan);
}

iree_status_t loom_amdgpu_lower_scalar_cmpf(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_compare_plan_t* plan) {
  return loom_amdgpu_lower_vector_compare(context, source_op, plan);
}

iree_status_t loom_amdgpu_lower_vector_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan) {
  const uint32_t lane_count = plan->lane_count;
  IREE_ASSERT_GT(lane_count, 0);
  IREE_ASSERT_LE(lane_count, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);

  loom_value_id_t low_condition = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->condition, &low_condition));
  loom_value_id_t low_true_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->true_value, &low_true_value));
  loom_value_id_t low_false_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->false_value,
                                                   &low_false_value));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t lane_condition = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
        context, source_op, low_condition, lane_count, i * 2u, mask_lane_type,
        &lane_condition));
    loom_value_id_t lane_true_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
        context, source_op, low_true_value, lane_count, i, lane_type,
        &lane_true_value));
    loom_value_id_t lane_false_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
        context, source_op, low_false_value, lane_count, i, lane_type,
        &lane_false_value));
    const loom_value_id_t operands[] = {
        lane_false_value,
        lane_true_value,
        lane_condition,
    };
    loom_op_t* lane_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
        &lane_type, 1, &lane_op));
    lane_results[i] = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  }

  if (lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, lane_results[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}
