// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/attribute.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/internal.h"
#include "loom/target/arch/amdgpu/matrix_contract.h"
#include "loom/target/arch/amdgpu/target_id.h"
#include "loom/target/arch/amdgpu/target_refs.h"
#include "loom/util/fact_table.h"

enum {
  LOOM_AMDGPU_FRAGMENT_VIEW_RANK = 2,
};

static bool loom_amdgpu_fragment_memory_role_from_vector_role(
    loom_vector_role_t role, loom_amdgpu_matrix_operand_role_t* out_role) {
  *out_role = LOOM_AMDGPU_MATRIX_OPERAND_ROLE_UNKNOWN;
  switch (role) {
    case LOOM_VECTOR_ROLE_LHS:
      *out_role = LOOM_AMDGPU_MATRIX_OPERAND_ROLE_LHS;
      return true;
    case LOOM_VECTOR_ROLE_RHS:
      *out_role = LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RHS;
      return true;
    case LOOM_VECTOR_ROLE_INIT:
      *out_role = LOOM_AMDGPU_MATRIX_OPERAND_ROLE_ACCUMULATOR;
      return true;
    case LOOM_VECTOR_ROLE_RESULT:
      *out_role = LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RESULT;
      return true;
    case LOOM_VECTOR_ROLE_COUNT_:
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_fragment_memory_target_layout(
    loom_low_lower_context_t* context,
    const loom_amdgpu_matrix_fragment_layout_t** out_layout,
    bool* out_selected) {
  *out_layout = NULL;
  *out_selected = false;
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_processor_from_ref(
          loom_low_lower_context_module(context),
          loom_low_lower_context_target_ref(context));
  if (processor == NULL) {
    return iree_ok_status();
  }

  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  if (!loom_amdgpu_matrix_feature_bits_for_profile(
          processor->matrix_feature_profile, &feature_bits) ||
      !iree_all_bits_set(feature_bits, LOOM_AMDGPU_MATRIX_FEATURE_WMMA_GFX11)) {
    return iree_ok_status();
  }

  uint32_t wavefront_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_wavefront_size(
      loom_low_lower_context_bundle(context), &wavefront_size));
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(
          LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_RDNA3_WMMAR3_F32_16X16X16_F16);
  if (layout == NULL || wavefront_size != layout->wave_size) {
    return iree_ok_status();
  }
  *out_layout = layout;
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_fragment_memory_exact_nonnegative_i64(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    int64_t* out_value) {
  return loom_amdgpu_value_facts_as_exact_non_negative_i64(
      loom_value_fact_table_lookup(fact_table, value_id), out_value);
}

static bool loom_amdgpu_fragment_memory_shape_matches(
    const loom_value_fact_table_t* fact_table,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_amdgpu_matrix_operand_role_t role, loom_value_id_t rows,
    loom_value_id_t columns) {
  int64_t row_count = 0;
  int64_t column_count = 0;
  if (!loom_amdgpu_fragment_memory_exact_nonnegative_i64(fact_table, rows,
                                                         &row_count) ||
      !loom_amdgpu_fragment_memory_exact_nonnegative_i64(fact_table, columns,
                                                         &column_count)) {
    return false;
  }

  const loom_amdgpu_matrix_tile_shape_t shape = layout->tile_shape;
  switch (role) {
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_LHS:
      return row_count == shape.result_row_count &&
             column_count == shape.reduction_count;
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RHS:
      return row_count == shape.reduction_count &&
             column_count == shape.result_column_count;
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_ACCUMULATOR:
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RESULT:
      return row_count == shape.result_row_count &&
             column_count == shape.result_column_count;
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_payload_matches(
    loom_type_t payload_type,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  if (role_layout == NULL || !loom_type_is_vector(payload_type) ||
      loom_type_rank(payload_type) != 1 ||
      !loom_type_is_all_static(payload_type)) {
    return false;
  }

  switch (role_layout->role) {
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_LHS:
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RHS: {
      if (loom_type_element_type(payload_type) != LOOM_SCALAR_TYPE_F16) {
        return false;
      }
      uint32_t payload_bit_count = 0;
      uint32_t register_count = 0;
      return loom_amdgpu_type_packed_16bit_float_storage(
                 payload_type, &payload_bit_count, &register_count) &&
             register_count == role_layout->register_count &&
             payload_bit_count == (uint32_t)role_layout->register_count * 32u;
    }
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_ACCUMULATOR:
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RESULT:
      return loom_type_element_type(payload_type) == LOOM_SCALAR_TYPE_F32 &&
             loom_amdgpu_vector_f32_lane_count(payload_type) ==
                 role_layout->register_count;
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_view_matches(
    const loom_module_t* module, loom_type_t view_type,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    loom_vector_memory_access_t* out_access) {
  *out_access = (loom_vector_memory_access_t){0};
  if (!loom_type_is_view(view_type) ||
      loom_type_rank(view_type) != LOOM_AMDGPU_FRAGMENT_VIEW_RANK ||
      !loom_type_is_all_static(view_type)) {
    return false;
  }

  loom_scalar_type_t expected_element_type = LOOM_SCALAR_TYPE_COUNT_;
  switch (role_layout->role) {
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_LHS:
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RHS:
      expected_element_type = LOOM_SCALAR_TYPE_F16;
      break;
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_ACCUMULATOR:
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RESULT:
      expected_element_type = LOOM_SCALAR_TYPE_F32;
      break;
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
  if (loom_type_element_type(view_type) != expected_element_type) {
    return false;
  }

  loom_type_t scalar_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, expected_element_type,
                          loom_dim_pack_static(1), /*encoding_id=*/0);
  if (!loom_vector_memory_access_describe(module, view_type, scalar_vector_type,
                                          out_access)) {
    return false;
  }
  return out_access->static_element_byte_count > 0 &&
         out_access->static_element_byte_count <= UINT16_MAX &&
         (out_access->layout_kind == LOOM_VECTOR_MEMORY_LAYOUT_DENSE ||
          out_access->layout_kind == LOOM_VECTOR_MEMORY_LAYOUT_STRIDED);
}

static bool loom_amdgpu_fragment_memory_fill_view_strides(
    const loom_vector_memory_access_t* access,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    uint32_t* out_axis_byte_strides) {
  for (uint8_t axis = 0; axis < access->view_rank; ++axis) {
    int64_t element_stride = 0;
    if (!loom_vector_memory_access_static_axis_stride(access, axis,
                                                      &element_stride) ||
        element_stride <= 0) {
      return false;
    }
    int64_t byte_stride = 0;
    if (!iree_checked_mul_i64(element_stride, access->static_element_byte_count,
                              &byte_stride) ||
        byte_stride <= 0 || byte_stride > UINT32_MAX) {
      return false;
    }
    out_axis_byte_strides[axis] = (uint32_t)byte_stride;
  }

  if (role_layout->elements_per_register > 1) {
    uint8_t packed_axis = UINT8_MAX;
    switch (role_layout->map_kind) {
      case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
        packed_axis = 1;
        break;
      case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION:
        packed_axis = 0;
        break;
      default:
        return false;
    }
    if (packed_axis >= access->view_rank ||
        out_axis_byte_strides[packed_axis] !=
            (uint32_t)access->static_element_byte_count) {
      return false;
    }
  }

  return true;
}

static bool loom_amdgpu_fragment_memory_fill_origins(
    loom_attribute_t static_indices, loom_value_slice_t dynamic_indices,
    loom_amdgpu_fragment_origin_plan_t* out_origins) {
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count != LOOM_AMDGPU_FRAGMENT_VIEW_RANK) {
    return false;
  }

  uint16_t dynamic_ordinal = 0;
  for (uint16_t axis = 0; axis < static_indices.count; ++axis) {
    out_origins[axis] = (loom_amdgpu_fragment_origin_plan_t){
        .dynamic_index = LOOM_VALUE_ID_INVALID,
        .static_index = 0,
    };
    const int64_t static_index = static_indices.i64_array[axis];
    if (static_index == INT64_MIN) {
      if (dynamic_ordinal >= dynamic_indices.count) {
        return false;
      }
      out_origins[axis].dynamic_index =
          loom_value_slice_get(dynamic_indices, dynamic_ordinal++);
    } else if (static_index < 0) {
      return false;
    } else {
      out_origins[axis].static_index = static_index;
    }
  }
  return dynamic_ordinal == dynamic_indices.count;
}

static bool loom_amdgpu_fragment_memory_descriptor_ref(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      *out_descriptor_ref =
          operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_LOAD
              ? LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32_SADDR
              : LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      *out_descriptor_ref = operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_LOAD
                                ? LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32
                                : LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_fragment_memory_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_fragment_memory_plan_t){0};
  *out_selected = false;
  if (!loom_kernel_def_isa(
          loom_low_lower_context_source_function(context).op)) {
    return iree_ok_status();
  }

  loom_vector_role_t vector_role = LOOM_VECTOR_ROLE_COUNT_;
  loom_value_id_t view = LOOM_VALUE_ID_INVALID;
  loom_value_id_t payload = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rows = LOOM_VALUE_ID_INVALID;
  loom_value_id_t columns = LOOM_VALUE_ID_INVALID;
  loom_attribute_t static_indices = loom_attr_absent();
  loom_value_slice_t dynamic_indices = {0};
  loom_attribute_t cache_scope = loom_attr_absent();
  loom_attribute_t cache_temporal = loom_attr_absent();
  if (operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_LOAD) {
    vector_role = loom_vector_fragment_load_role(source_op);
    view = loom_vector_fragment_load_view(source_op);
    payload = loom_vector_fragment_load_result(source_op);
    rows = loom_vector_fragment_load_rows(source_op);
    columns = loom_vector_fragment_load_columns(source_op);
    static_indices = loom_vector_fragment_load_static_indices(source_op);
    dynamic_indices = loom_vector_fragment_load_indices(source_op);
    cache_scope = loom_op_attrs(
        source_op)[loom_vector_fragment_load_cache_scope_ATTR_INDEX];
    cache_temporal = loom_op_attrs(
        source_op)[loom_vector_fragment_load_cache_temporal_ATTR_INDEX];
  } else {
    vector_role = loom_vector_fragment_store_role(source_op);
    view = loom_vector_fragment_store_view(source_op);
    payload = loom_vector_fragment_store_value(source_op);
    rows = loom_vector_fragment_store_rows(source_op);
    columns = loom_vector_fragment_store_columns(source_op);
    static_indices = loom_vector_fragment_store_static_indices(source_op);
    dynamic_indices = loom_vector_fragment_store_indices(source_op);
    cache_scope = loom_op_attrs(
        source_op)[loom_vector_fragment_store_cache_scope_ATTR_INDEX];
    cache_temporal = loom_op_attrs(
        source_op)[loom_vector_fragment_store_cache_temporal_ATTR_INDEX];
  }
  if (!loom_attr_is_absent(cache_scope) ||
      !loom_attr_is_absent(cache_temporal)) {
    return iree_ok_status();
  }

  const loom_amdgpu_matrix_fragment_layout_t* layout = NULL;
  bool target_selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_target_layout(
      context, &layout, &target_selected));
  if (!target_selected) {
    return iree_ok_status();
  }

  loom_amdgpu_matrix_operand_role_t role =
      LOOM_AMDGPU_MATRIX_OPERAND_ROLE_UNKNOWN;
  if (!loom_amdgpu_fragment_memory_role_from_vector_role(vector_role, &role) ||
      !loom_amdgpu_fragment_memory_shape_matches(
          loom_low_lower_context_fact_table(context), layout, role, rows,
          columns)) {
    return iree_ok_status();
  }

  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, role);
  if (role_layout == NULL ||
      role_layout->register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return iree_ok_status();
  }
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t payload_type = loom_module_value_type(module, payload);
  if (!loom_amdgpu_fragment_memory_payload_matches(payload_type, role_layout)) {
    return iree_ok_status();
  }

  loom_vector_memory_access_t access = {0};
  const loom_type_t view_type = loom_module_value_type(module, view);
  if (!loom_amdgpu_fragment_memory_view_matches(module, view_type, role_layout,
                                                &access)) {
    return iree_ok_status();
  }
  uint32_t axis_byte_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  if (!loom_amdgpu_fragment_memory_fill_view_strides(&access, role_layout,
                                                     axis_byte_strides)) {
    return iree_ok_status();
  }

  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_value_facts_query_view_reference(
          &loom_low_lower_context_fact_table(context)->context,
          loom_value_fact_table_lookup(
              loom_low_lower_context_fact_table(context), view),
          &view_reference)) {
    return iree_ok_status();
  }
  int64_t base_byte_offset = 0;
  if (!loom_amdgpu_value_facts_as_exact_non_negative_i64(
          view_reference.base_byte_offset, &base_byte_offset)) {
    return iree_ok_status();
  }

  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_fragment_memory_descriptor_ref(
          operation_kind, view_reference.memory_space, &descriptor_ref)) {
    return iree_ok_status();
  }
  loom_low_lower_resolved_descriptor_t unused_descriptor = {0};
  bool descriptor_present = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
      context, descriptor_ref, &unused_descriptor, &descriptor_present));
  if (!descriptor_present) {
    return iree_ok_status();
  }

  loom_amdgpu_fragment_origin_plan_t
      origins[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  if (!loom_amdgpu_fragment_memory_fill_origins(static_indices, dynamic_indices,
                                                origins)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_fragment_memory_plan_t){
      .operation_kind = operation_kind,
      .role = role,
      .layout_kind = layout->kind,
      .view = view,
      .payload = payload,
      .memory_space = view_reference.memory_space,
      .root_value_id = view_reference.root_value_id,
      .alias_scope_id = view_reference.alias_scope_id,
      .base_byte_offset = (uint64_t)base_byte_offset,
      .view_rank = access.view_rank,
      .register_count = role_layout->register_count,
      .elements_per_register = role_layout->elements_per_register,
      .element_byte_count = (uint16_t)access.static_element_byte_count,
      .descriptor_ref = descriptor_ref,
  };
  for (uint8_t axis = 0; axis < access.view_rank; ++axis) {
    out_plan->origins[axis] = origins[axis];
    out_plan->axis_byte_strides[axis] = axis_byte_strides[axis];
  }
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_fragment_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_fragment_memory_select(context, source_op,
                                            LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
                                            out_plan, out_selected);
}

iree_status_t loom_amdgpu_select_vector_fragment_store_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_fragment_memory_select(context, source_op,
                                            LOOM_AMDGPU_MEMORY_OPERATION_STORE,
                                            out_plan, out_selected);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_scaled_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, uint32_t scale, loom_type_t vgpr_type,
    loom_value_id_t* out_low_term) {
  *out_low_term = LOOM_VALUE_ID_INVALID;
  if (scale == 0) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      vgpr_type, out_low_term);
  }
  if (scale == 1) {
    *out_low_term = low_value;
    return iree_ok_status();
  }
  if (loom_amdgpu_u32_is_power_of_two(scale)) {
    uint32_t shift = 0;
    uint32_t value = scale;
    while (value > 1u) {
      value >>= 1u;
      ++shift;
    }
    return loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, shift,
        low_value, vgpr_type, out_low_term);
  }

  loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, scale,
      vgpr_type, &low_scale));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32, low_value,
      low_scale, vgpr_type, out_low_term);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_add_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_term, loom_type_t vgpr_type,
    loom_value_id_t* inout_low_accumulator) {
  if (*inout_low_accumulator == LOOM_VALUE_ID_INVALID) {
    *inout_low_accumulator = low_term;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
      *inout_low_accumulator, low_term, vgpr_type, inout_low_accumulator);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_dynamic_origin_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_type_t vgpr_type,
    loom_value_id_t* inout_low_accumulator) {
  for (uint8_t axis = 0; axis < plan->view_rank; ++axis) {
    const loom_amdgpu_fragment_origin_plan_t* origin = &plan->origins[axis];
    if (origin->dynamic_index == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_address(
        context, source_op, origin->dynamic_index, &low_index));
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_scaled_term(
        context, source_op, low_index, plan->axis_byte_strides[axis], vgpr_type,
        &low_term));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_term(
        context, source_op, low_term, vgpr_type, inout_low_accumulator));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_fragment_memory_static_origin_offset(
    const loom_amdgpu_fragment_memory_plan_t* plan,
    uint64_t* inout_static_byte_offset) {
  if (*inout_static_byte_offset > INT64_MAX) {
    return false;
  }
  int64_t static_byte_offset = (int64_t)*inout_static_byte_offset;
  for (uint8_t axis = 0; axis < plan->view_rank; ++axis) {
    const loom_amdgpu_fragment_origin_plan_t* origin = &plan->origins[axis];
    if (origin->dynamic_index != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    int64_t axis_offset = 0;
    if (!iree_checked_mul_i64(origin->static_index,
                              plan->axis_byte_strides[axis], &axis_offset) ||
        !iree_checked_add_i64(static_byte_offset, axis_offset,
                              &static_byte_offset)) {
      return false;
    }
  }
  *inout_static_byte_offset = (uint64_t)static_byte_offset;
  return true;
}

static bool loom_amdgpu_fragment_memory_register_terms(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint32_t* out_lane_mod_stride, uint32_t* out_lane_div_stride,
    uint64_t* out_static_byte_offset) {
  const loom_amdgpu_matrix_tile_shape_t shape = layout->tile_shape;
  *out_lane_mod_stride = 0;
  *out_lane_div_stride = 0;
  *out_static_byte_offset = 0;
  switch (plan->role) {
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_LHS:
      *out_lane_mod_stride = plan->axis_byte_strides[0];
      *out_static_byte_offset = (uint64_t)register_index *
                                plan->elements_per_register *
                                plan->axis_byte_strides[1];
      return true;
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RHS:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      *out_static_byte_offset = (uint64_t)register_index *
                                plan->elements_per_register *
                                plan->axis_byte_strides[0];
      return true;
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_ACCUMULATOR:
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_RESULT:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      *out_lane_div_stride = plan->axis_byte_strides[0];
      *out_static_byte_offset =
          (uint64_t)register_index *
          (shape.result_row_count / plan->register_count) *
          plan->axis_byte_strides[0];
      return true;
    case LOOM_AMDGPU_MATRIX_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
}

static iree_status_t loom_amdgpu_emit_fragment_memory_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    loom_value_id_t low_lane_mod, loom_value_id_t low_lane_div,
    loom_value_id_t low_resource, loom_type_t vgpr_type,
    loom_value_id_t* out_low_vaddr) {
  *out_low_vaddr = LOOM_VALUE_ID_INVALID;
  uint32_t lane_mod_stride = 0;
  uint32_t lane_div_stride = 0;
  if (plan->base_byte_offset > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU fragment memory base exceeds i64");
  }
  int64_t static_byte_offset_i64 = (int64_t)plan->base_byte_offset;
  uint64_t register_static_offset = 0;
  if (!loom_amdgpu_fragment_memory_register_terms(
          layout, plan, register_index, &lane_mod_stride, &lane_div_stride,
          &register_static_offset)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU fragment memory register map is invalid");
  }
  if (register_static_offset > INT64_MAX ||
      !iree_checked_add_i64(static_byte_offset_i64,
                            (int64_t)register_static_offset,
                            &static_byte_offset_i64)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU fragment memory byte offset exceeds i64");
  }
  uint64_t static_byte_offset = (uint64_t)static_byte_offset_i64;
  if (!loom_amdgpu_fragment_memory_static_origin_offset(plan,
                                                        &static_byte_offset) ||
      static_byte_offset > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU fragment memory byte offset exceeds u32");
  }

  loom_value_id_t low_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_dynamic_origin_terms(
      context, source_op, plan, vgpr_type, &low_accumulator));

  if (lane_div_stride != 0) {
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_scaled_term(
        context, source_op, low_lane_div, lane_div_stride, vgpr_type,
        &low_term));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_term(
        context, source_op, low_term, vgpr_type, &low_accumulator));
  }
  if (lane_mod_stride != 0) {
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_scaled_term(
        context, source_op, low_lane_mod, lane_mod_stride, vgpr_type,
        &low_term));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_term(
        context, source_op, low_term, vgpr_type, &low_accumulator));
  }

  if (plan->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_term(
        context, source_op, low_resource, vgpr_type, &low_accumulator));
  }

  if (static_byte_offset != 0) {
    if (low_accumulator == LOOM_VALUE_ID_INVALID) {
      return loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          (uint32_t)static_byte_offset, vgpr_type, out_low_vaddr);
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_literal(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
        low_accumulator, (uint32_t)static_byte_offset, vgpr_type,
        &low_accumulator));
  }

  if (low_accumulator == LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      vgpr_type, out_low_vaddr);
  }
  *out_low_vaddr = low_accumulator;
  return iree_ok_status();
}

static loom_low_memory_space_t loom_amdgpu_fragment_memory_low_space(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      return LOOM_LOW_MEMORY_SPACE_GLOBAL;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      return LOOM_LOW_MEMORY_SPACE_WORKGROUP;
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    default:
      return LOOM_LOW_MEMORY_SPACE_GENERIC;
  }
}

static iree_status_t loom_amdgpu_record_fragment_memory_packet(
    loom_low_lower_context_t* context, const loom_op_t* low_op,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  loom_low_memory_access_summary_t summary = {
      .memory_space = loom_amdgpu_fragment_memory_low_space(plan->memory_space),
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
  };
  if (summary.memory_space != LOOM_LOW_MEMORY_SPACE_GENERIC) {
    summary.precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE;
  }
  if (plan->alias_scope_id != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE) {
    summary.alias_root_id = plan->alias_scope_id;
    summary.precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT;
  }
  return loom_low_lower_record_memory_access_summary(context, low_op, &summary);
}

static iree_status_t loom_amdgpu_make_fragment_memory_attrs(
    loom_low_lower_context_t* context, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, iree_host_size_t* out_attr_count) {
  *out_attr_count = 0;
  return loom_amdgpu_append_i64_attr(context, IREE_SV("offset"), 0, attrs,
                                     attr_capacity, out_attr_count);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_lane_ids(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t vgpr_type, loom_value_id_t* out_low_lane_mod,
    loom_value_id_t* out_low_lane_div) {
  *out_low_lane_mod = LOOM_VALUE_ID_INVALID;
  *out_low_lane_div = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, vgpr_type, &low_lane));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_literal(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_lane,
      15, vgpr_type, out_low_lane_mod));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 4,
      low_lane, vgpr_type, out_low_lane_div);
}

static iree_status_t loom_amdgpu_emit_fragment_load_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_value_id_t low_vaddr,
    loom_value_id_t low_resource, loom_type_t vgpr_type,
    loom_value_id_t* out_low_register) {
  *out_low_register = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_fragment_memory_attrs(
      context, attrs, IREE_ARRAYSIZE(attrs), &attr_count));

  loom_value_id_t operands[2] = {low_vaddr, low_resource};
  iree_host_size_t operand_count =
      plan->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ? 1 : 2;
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, plan->descriptor_ref, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &vgpr_type, 1, &low_op));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_record_fragment_memory_packet(context, low_op, plan));
  *out_low_register = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_store_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_value_id_t low_vaddr,
    loom_value_id_t low_payload_register, loom_value_id_t low_resource) {
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_fragment_memory_attrs(
      context, attrs, IREE_ARRAYSIZE(attrs), &attr_count));

  loom_value_id_t operands[3] = {
      low_vaddr,
      low_payload_register,
      low_resource,
  };
  iree_host_size_t operand_count =
      plan->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ? 2 : 3;
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, plan->descriptor_ref, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, &low_op));
  return loom_amdgpu_record_fragment_memory_packet(context, low_op, plan);
}

iree_status_t loom_amdgpu_lower_vector_fragment_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(plan->layout_kind);
  if (layout == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU fragment memory plan has no layout");
  }

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t low_lane_mod = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_lane_div = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_ids(
      context, source_op, vgpr_type, &low_lane_mod, &low_lane_div));

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->view, &low_resource));

  loom_value_id_t low_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  for (uint16_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, layout, plan, i, low_lane_mod, low_lane_div,
        low_resource, vgpr_type, &low_vaddr));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_load_packet(
        context, source_op, plan, low_vaddr, low_resource, vgpr_type,
        &low_registers[i]));
  }

  if (plan->register_count == 1) {
    return loom_low_lower_bind_value(context, plan->payload, low_registers[0]);
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
      context, plan->register_count, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_registers,
      plan->register_count, result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->payload,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_vector_fragment_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(plan->layout_kind);
  if (layout == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU fragment memory plan has no layout");
  }

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t low_lane_mod = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_lane_div = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_ids(
      context, source_op, vgpr_type, &low_lane_mod, &low_lane_div));

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->view, &low_resource));
  loom_value_id_t low_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->payload, &low_payload));

  for (uint16_t i = 0; i < plan->register_count; ++i) {
    loom_value_id_t low_payload_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_payload, i, vgpr_type, &low_payload_register));
    loom_value_id_t low_vaddr = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, layout, plan, i, low_lane_mod, low_lane_div,
        low_resource, vgpr_type, &low_vaddr));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_store_packet(
        context, source_op, plan, low_vaddr, low_payload_register,
        low_resource));
  }
  return iree_ok_status();
}
