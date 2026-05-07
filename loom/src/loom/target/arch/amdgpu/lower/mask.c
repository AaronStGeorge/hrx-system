// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/facts.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/compare_candidates.h"
#include "loom/target/arch/amdgpu/lower/internal.h"
#include "loom/target/arch/amdgpu/target_refs.h"
#include "loom/util/fact_table.h"
#include "loom/util/math.h"

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

static bool loom_amdgpu_i64_value_as_u32_bits(int64_t value,
                                              uint32_t* out_bits) {
  if (value < INT32_MIN || value > UINT32_MAX) {
    return false;
  }
  *out_bits = (uint32_t)value;
  return true;
}

static bool loom_amdgpu_value_facts_as_u32_bits(loom_value_facts_t facts,
                                                uint32_t* out_bits) {
  int64_t value = 0;
  return loom_value_facts_as_exact_i64(facts, &value) &&
         loom_amdgpu_i64_value_as_u32_bits(value, out_bits);
}

static bool loom_amdgpu_source_lane_as_u32_bits(
    const loom_value_fact_table_t* fact_table, loom_value_id_t source,
    uint32_t lane, uint32_t* out_bits) {
  *out_bits = 0;
  if (fact_table == NULL) {
    return false;
  }

  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, source);
  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(&fact_table->context, facts,
                                             &uniform)) {
    return loom_amdgpu_value_facts_as_u32_bits(uniform.element, out_bits);
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (loom_value_facts_query_small_static_lanes(&fact_table->context, facts,
                                                &lanes)) {
    return lane < lanes.count &&
           loom_amdgpu_value_facts_as_u32_bits(lanes.lanes[lane], out_bits);
  }

  loom_value_fact_vector_iota_t iota = {0};
  if (loom_value_facts_query_vector_iota(&fact_table->context, facts, &iota)) {
    int64_t base = 0;
    int64_t step = 0;
    int64_t delta = 0;
    int64_t value = 0;
    return loom_value_facts_as_exact_i64(iota.base, &base) &&
           loom_value_facts_as_exact_i64(iota.step, &step) &&
           loom_checked_mul_i64((int64_t)lane, step, &delta) &&
           loom_checked_add_i64(base, delta, &value) &&
           loom_amdgpu_i64_value_as_u32_bits(value, out_bits);
  }

  return loom_amdgpu_value_facts_as_u32_bits(facts, out_bits);
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

  loom_low_lower_resolved_descriptor_t register_descriptor = {0};
  bool register_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32, &register_descriptor,
      &register_descriptor_present));
  if (!register_descriptor_present) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_descriptor_t src1_inline_descriptor = {0};
  bool src1_inline_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC1_INLINE,
      &src1_inline_descriptor, &src1_inline_descriptor_present));
  loom_string_id_t true_value_attr_name_id = LOOM_STRING_ID_INVALID;
  if (src1_inline_descriptor_present) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, IREE_SV("true_value"),
                                            &true_value_attr_name_id));
  }

  *out_plan = (loom_amdgpu_vector_select_plan_t){
      .condition_kind = LOOM_AMDGPU_SELECT_CONDITION_KIND_VECTOR_MASK,
      .register_descriptor = register_descriptor,
      .src1_inline_descriptor = src1_inline_descriptor,
      .true_value_attr_name_id = true_value_attr_name_id,
      .condition = condition,
      .true_value = true_value,
      .false_value = false_value,
      .result = result,
      .lane_count = lane_count,
  };
  *out_selected = true;
  return iree_ok_status();
}

static uint32_t loom_amdgpu_select_32bit_lane_count(loom_type_t type) {
  if (loom_amdgpu_type_is_i32(type) ||
      loom_amdgpu_type_is_address_scalar(type) ||
      loom_amdgpu_type_is_f32(type)) {
    return 1;
  }
  return loom_amdgpu_vector_32bit_lane_count(type);
}

static iree_status_t loom_amdgpu_low_type_is_register_class(
    loom_low_lower_context_t* context, loom_type_t type, uint16_t reg_class_id,
    uint32_t unit_count, bool* out_match) {
  *out_match = false;
  bool is_class = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, type, reg_class_id, &is_class));
  *out_match = is_class && loom_type_register_unit_count(type) == unit_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_resolve_select_vgpr_descriptors(
    loom_low_lower_context_t* context,
    loom_low_lower_resolved_descriptor_t* out_register_descriptor,
    loom_low_lower_resolved_descriptor_t* out_src1_inline_descriptor,
    loom_string_id_t* out_true_value_attr_name_id, bool* out_present) {
  *out_register_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  *out_src1_inline_descriptor = (loom_low_lower_resolved_descriptor_t){0};
  *out_true_value_attr_name_id = LOOM_STRING_ID_INVALID;
  *out_present = false;

  bool register_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      out_register_descriptor, &register_descriptor_present));
  if (!register_descriptor_present) {
    return iree_ok_status();
  }

  bool src1_inline_descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32_SRC1_INLINE,
      out_src1_inline_descriptor, &src1_inline_descriptor_present));
  if (src1_inline_descriptor_present) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, IREE_SV("true_value"),
                                            out_true_value_attr_name_id));
  }

  *out_present = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_scf_select_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_select_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_vector_select_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t result = loom_scf_select_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const uint32_t lane_count = loom_amdgpu_select_32bit_lane_count(result_type);
  const loom_value_id_t condition = loom_scf_select_condition(source_op);
  const loom_value_id_t true_value = loom_scf_select_true_value(source_op);
  const loom_value_id_t false_value = loom_scf_select_false_value(source_op);
  if (lane_count == 0 ||
      !loom_amdgpu_type_is_i1(loom_module_value_type(module, condition)) ||
      !loom_type_equal(loom_module_value_type(module, true_value),
                       result_type) ||
      !loom_type_equal(loom_module_value_type(module, false_value),
                       result_type)) {
    return iree_ok_status();
  }

  loom_type_t condition_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op, condition,
                                                &condition_low_type));
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));

  bool condition_is_scc = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class(
      context, condition_low_type, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1,
      &condition_is_scc));
  bool result_is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1,
      &result_is_sgpr));
  if (condition_is_scc && result_is_sgpr && lane_count == 1) {
    loom_low_lower_resolved_descriptor_t scc_descriptor = {0};
    bool scc_descriptor_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32, &scc_descriptor,
        &scc_descriptor_present));
    if (!scc_descriptor_present) {
      return iree_ok_status();
    }
    *out_plan = (loom_amdgpu_vector_select_plan_t){
        .condition_kind = LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC,
        .scc_descriptor = scc_descriptor,
        .condition = condition,
        .true_value = true_value,
        .false_value = false_value,
        .result = result,
        .lane_count = lane_count,
    };
    *out_selected = true;
    return iree_ok_status();
  }

  bool condition_is_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class(
      context, condition_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2,
      &condition_is_mask));
  if (!condition_is_mask) {
    return iree_ok_status();
  }
  bool result_is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_is_register_class(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, lane_count,
      &result_is_vgpr));
  if (!result_is_vgpr) {
    return iree_ok_status();
  }

  loom_low_lower_resolved_descriptor_t register_descriptor = {0};
  loom_low_lower_resolved_descriptor_t src1_inline_descriptor = {0};
  loom_string_id_t true_value_attr_name_id = LOOM_STRING_ID_INVALID;
  bool descriptors_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_select_vgpr_descriptors(
      context, &register_descriptor, &src1_inline_descriptor,
      &true_value_attr_name_id, &descriptors_present));
  if (!descriptors_present) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_vector_select_plan_t){
      .condition_kind = LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK,
      .register_descriptor = register_descriptor,
      .src1_inline_descriptor = src1_inline_descriptor,
      .true_value_attr_name_id = true_value_attr_name_id,
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

static iree_status_t loom_amdgpu_emit_vector_select_src1_inline_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_select_plan_t* plan, loom_value_id_t false_value,
    loom_value_id_t condition, uint32_t true_bits, loom_type_t lane_type,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      false_value,
      condition,
  };
  const loom_named_attr_t attrs[] = {
      {
          .name_id = plan->true_value_attr_name_id,
          .value = loom_attr_i64(true_bits),
      },
  };
  loom_op_t* lane_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->src1_inline_descriptor, operands,
      IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), &lane_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &lane_op));
  *out_result = loom_value_slice_get(loom_low_op_results(lane_op), 0);
  return iree_ok_status();
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

iree_status_t loom_amdgpu_lower_select(
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

  if (plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SCC) {
    const loom_value_id_t operands[] = {
        low_true_value,
        low_false_value,
        low_condition,
    };
    loom_type_t result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
        context, source_op, plan->result, &result_type));
    loom_op_t* select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->scc_descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(NULL, 0), &result_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &select_op));
    return loom_low_lower_bind_value(
        context, plan->result,
        loom_value_slice_get(loom_low_op_results(select_op), 0));
  }

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t lane_condition = LOOM_VALUE_ID_INVALID;
    if (plan->condition_kind == LOOM_AMDGPU_SELECT_CONDITION_KIND_SCALAR_MASK) {
      lane_condition = low_condition;
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
          context, source_op, low_condition, lane_count, i * 2u, mask_lane_type,
          &lane_condition));
    }
    uint32_t lane_true_bits = 0;
    if (plan->src1_inline_descriptor.descriptor != NULL &&
        loom_amdgpu_source_lane_as_u32_bits(fact_table, plan->true_value, i,
                                            &lane_true_bits) &&
        lane_true_bits <= 64) {
      loom_value_id_t lane_false_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_slice_lane_if_needed(
          context, source_op, low_false_value, lane_count, i, lane_type,
          &lane_false_value));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vector_select_src1_inline_lane(
          context, source_op, plan, lane_false_value, lane_condition,
          lane_true_bits, lane_type, &lane_results[i]));
      continue;
    }

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
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &plan->register_descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(NULL, 0), &lane_type, 1,
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
