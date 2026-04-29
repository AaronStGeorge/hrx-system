// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower/internal.h"

static bool loom_amdgpu_table_lookup_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_table_lookup_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_amdgpu_table_lookup_plan_t){0};
  if (!loom_vector_table_lookup_isa(source_op)) {
    return false;
  }

  out_plan->table = loom_vector_table_lookup_table(source_op);
  out_plan->indices = loom_vector_table_lookup_indices(source_op);
  out_plan->result = loom_vector_table_lookup_result(source_op);

  const loom_type_t table_type =
      loom_module_value_type(module, out_plan->table);
  const loom_type_t indices_type =
      loom_module_value_type(module, out_plan->indices);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);

  const uint32_t table_lane_count = loom_amdgpu_static_vector_lane_count(
      table_type, LOOM_SCALAR_TYPE_F32, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  const uint32_t result_lane_count = loom_amdgpu_static_vector_lane_count(
      result_type, LOOM_SCALAR_TYPE_F32,
      LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
  if (table_lane_count == 0 || result_lane_count == 0) {
    return false;
  }

  if (loom_amdgpu_static_vector_lane_count(
          indices_type, LOOM_SCALAR_TYPE_I32,
          LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) == result_lane_count) {
    out_plan->index_kind = LOOM_AMDGPU_TABLE_INDEX_KIND_I32;
    out_plan->index_register_count = result_lane_count;
  } else {
    uint32_t index_payload_bit_count = 0;
    uint32_t index_register_count = 0;
    if (loom_amdgpu_static_vector_lane_count(indices_type, LOOM_SCALAR_TYPE_I8,
                                             LOOM_AMDGPU_MAX_PACKED_I8_LANES) !=
            result_lane_count ||
        !loom_amdgpu_type_packed_integer_storage(
            indices_type, &index_payload_bit_count, &index_register_count) ||
        index_payload_bit_count != result_lane_count * 8u) {
      return false;
    }
    out_plan->index_kind = LOOM_AMDGPU_TABLE_INDEX_KIND_PACKED_I8;
    out_plan->index_register_count = index_register_count;
  }

  out_plan->table_lane_count = table_lane_count;
  out_plan->result_lane_count = result_lane_count;
  return true;
}

iree_status_t loom_amdgpu_select_vector_table_lookup_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_table_lookup_plan_t* out_plan, bool* out_selected) {
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_selected = loom_amdgpu_table_lookup_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_table_lookup_slice_if_needed(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, uint32_t register_count, uint32_t register_offset,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  IREE_ASSERT_ARGUMENT(out_lane);
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (register_count == 1) {
    *out_lane = source;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_low_slice(context, source_op, source, register_offset,
                                    lane_type, out_lane);
}

static iree_status_t loom_amdgpu_table_lookup_extract_i32_index_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_indices,
    uint32_t result_lane, loom_type_t lane_type,
    loom_value_id_t* out_index_lane) {
  return loom_amdgpu_table_lookup_slice_if_needed(
      context, source_op, low_indices, plan->index_register_count, result_lane,
      lane_type, out_index_lane);
}

static iree_status_t loom_amdgpu_table_lookup_extract_i8_index_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_indices,
    uint32_t result_lane, loom_type_t lane_type,
    loom_value_id_t* out_index_lane) {
  IREE_ASSERT_ARGUMENT(out_index_lane);
  *out_index_lane = LOOM_VALUE_ID_INVALID;
  const uint32_t register_offset = result_lane / 4u;
  const uint32_t byte_offset = result_lane & 3u;
  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_slice_if_needed(
      context, source_op, low_indices, plan->index_register_count,
      register_offset, lane_type, &source_register));

  loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32_LIT,
      byte_offset * 8u, source_register, lane_type, &shifted));

  return loom_amdgpu_emit_vgpr_binary_literal(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32_LIT, shifted,
      0xFFu, lane_type, out_index_lane);
}

static iree_status_t loom_amdgpu_table_lookup_extract_index_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_indices,
    uint32_t result_lane, loom_type_t lane_type,
    loom_value_id_t* out_index_lane) {
  if (plan->index_kind == LOOM_AMDGPU_TABLE_INDEX_KIND_I32) {
    return loom_amdgpu_table_lookup_extract_i32_index_lane(
        context, source_op, plan, low_indices, result_lane, lane_type,
        out_index_lane);
  }
  return loom_amdgpu_table_lookup_extract_i8_index_lane(
      context, source_op, plan, low_indices, result_lane, lane_type,
      out_index_lane);
}

static iree_status_t loom_amdgpu_table_lookup_extract_table_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan, loom_value_id_t low_table,
    uint32_t table_lane, loom_type_t lane_type,
    loom_value_id_t* out_table_lane) {
  return loom_amdgpu_table_lookup_slice_if_needed(
      context, source_op, low_table, plan->table_lane_count, table_lane,
      lane_type, out_table_lane);
}

static iree_status_t loom_amdgpu_table_lookup_select_table_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan,
    const loom_value_id_t* table_lanes, const loom_value_id_t* ordinals,
    loom_value_id_t index_lane, loom_type_t lane_type,
    loom_type_t mask_lane_type, loom_value_id_t* out_selected_lane) {
  IREE_ASSERT_ARGUMENT(table_lanes);
  IREE_ASSERT_ARGUMENT(ordinals);
  IREE_ASSERT_ARGUMENT(out_selected_lane);
  *out_selected_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t selected_lane = table_lanes[0];
  for (uint32_t i = 1; i < plan->table_lane_count; ++i) {
    const loom_value_id_t compare_operands[] = {
        index_lane,
        ordinals[i],
    };
    loom_op_t* compare_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_EQ_I32,
        compare_operands, IREE_ARRAYSIZE(compare_operands),
        loom_make_named_attr_slice(NULL, 0), &mask_lane_type, 1, &compare_op));
    const loom_value_id_t select_operands[] = {
        selected_lane,
        table_lanes[i],
        loom_value_slice_get(loom_low_op_results(compare_op), 0),
    };
    loom_op_t* select_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_CNDMASK_B32,
        select_operands, IREE_ARRAYSIZE(select_operands),
        loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &select_op));
    selected_lane = loom_value_slice_get(loom_low_op_results(select_op), 0);
  }
  *out_selected_lane = selected_lane;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_table_lookup(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_table_lookup_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  loom_value_id_t low_table = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->table, &low_table));
  loom_value_id_t low_indices = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->indices, &low_indices));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_type_t mask_lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_lane_type));
  loom_value_id_t table_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  loom_value_id_t ordinals[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {0};
  for (uint32_t i = 0; i < plan->table_lane_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_extract_table_lane(
        context, source_op, plan, low_table, i, lane_type, &table_lanes[i]));
    if (i == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32, i, lane_type,
        &ordinals[i]));
  }

  loom_value_id_t result_lanes[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  for (uint32_t i = 0; i < plan->result_lane_count; ++i) {
    loom_value_id_t index_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_extract_index_lane(
        context, source_op, plan, low_indices, i, lane_type, &index_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_table_lookup_select_table_lane(
        context, source_op, plan, table_lanes, ordinals, index_lane, lane_type,
        mask_lane_type, &result_lanes[i]));
  }

  if (plan->result_lane_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, result_lanes[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), result_lanes,
      plan->result_lane_count, result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_low_legality_verify_vector_table(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  loom_amdgpu_table_lookup_plan_t unused_plan = {0};
  if (loom_amdgpu_table_lookup_plan_from_op(module, op, &unused_plan)) {
    return iree_ok_status();
  }
  return loom_target_low_legality_reject(
      context, provider, op, IREE_SV("shape"), loom_op_name(module, op),
      IREE_SV("AMDGPU vector.table.lookup currently requires a static rank-1 "
              "f32 table, static rank-1 i32 or packed i8 indices, and a "
              "static rank-1 f32 result within the scalarized register lane "
              "limit"));
}
