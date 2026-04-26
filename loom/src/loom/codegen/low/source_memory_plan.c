// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_memory_plan.h"

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

static bool loom_low_source_memory_static_view_vector_type(
    loom_type_t view_type, loom_type_t* out_vector_type,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  IREE_ASSERT_ARGUMENT(out_vector_type);
  *out_vector_type = loom_type_none();
  if (!loom_type_is_view(view_type) ||
      loom_type_rank(view_type) > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED;
    return false;
  }

  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(view_type));
  if (element_bit_count <= 0 || (element_bit_count % 8) != 0) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH;
    return false;
  }

  int64_t lane_count = 1;
  const uint8_t rank = loom_type_rank(view_type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (loom_type_dim_is_dynamic_at(view_type, i)) {
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_LANE_COUNT;
      return false;
    }
    const int64_t dim_size = loom_type_dim_static_size_at(view_type, i);
    if (dim_size <= 0 ||
        !iree_checked_mul_i64(lane_count, dim_size, &lane_count) ||
        lane_count > UINT32_MAX) {
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_LANE_COUNT;
      return false;
    }
  }

  *out_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(view_type),
                          loom_dim_pack_static(lane_count), /*encoding_id=*/0);
  return true;
}

static bool loom_low_source_memory_access_exact_i64(loom_value_facts_t facts,
                                                    int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_low_source_memory_access_value_as_workitem_id(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_kernel_dimension_t* out_dimension) {
  IREE_ASSERT_ARGUMENT(out_dimension);
  *out_dimension = LOOM_KERNEL_DIMENSION_COUNT_;
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_kernel_workitem_id_isa(defining_op)) {
    return false;
  }
  const loom_kernel_dimension_t dimension =
      loom_kernel_workitem_id_dimension(defining_op);
  if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_) {
    return false;
  }
  *out_dimension = dimension;
  return true;
}

static bool loom_low_source_memory_access_value_as_workgroup_id(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_kernel_dimension_t* out_dimension) {
  IREE_ASSERT_ARGUMENT(out_dimension);
  *out_dimension = LOOM_KERNEL_DIMENSION_COUNT_;
  if (value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_kernel_workgroup_id_isa(defining_op)) {
    return false;
  }
  const loom_kernel_dimension_t dimension =
      loom_kernel_workgroup_id_dimension(defining_op);
  if (dimension >= LOOM_KERNEL_DIMENSION_COUNT_) {
    return false;
  }
  *out_dimension = dimension;
  return true;
}

static bool loom_low_source_memory_access_collect_dynamic_axes(
    loom_attribute_t static_indices, uint8_t* out_dynamic_axes,
    uint8_t* out_dynamic_axis_count) {
  IREE_ASSERT_ARGUMENT(out_dynamic_axes);
  IREE_ASSERT_ARGUMENT(out_dynamic_axis_count);
  *out_dynamic_axis_count = 0;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return false;
  }
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (static_indices.i64_array[i] != INT64_MIN) {
      continue;
    }
    if (*out_dynamic_axis_count >=
            LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY ||
        i > UINT8_MAX) {
      return false;
    }
    out_dynamic_axes[*out_dynamic_axis_count] = (uint8_t)i;
    *out_dynamic_axis_count += 1;
  }
  return true;
}

static bool loom_low_source_memory_access_axis_is_dynamic(
    const uint8_t* dynamic_axes, uint8_t dynamic_axis_count, uint16_t axis) {
  for (uint8_t i = 0; i < dynamic_axis_count; ++i) {
    if (dynamic_axes[i] == axis) {
      return true;
    }
  }
  return false;
}

static bool loom_low_source_memory_access_static_byte_offset(
    const loom_vector_memory_access_t* vector_access,
    loom_attribute_t static_indices, const uint8_t* dynamic_axes,
    uint8_t dynamic_axis_count, int64_t* out_static_byte_offset) {
  IREE_ASSERT_ARGUMENT(out_static_byte_offset);
  *out_static_byte_offset = 0;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK) {
    return false;
  }

  int64_t static_origin[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (loom_low_source_memory_access_axis_is_dynamic(dynamic_axes,
                                                      dynamic_axis_count, i)) {
      if (static_indices.i64_array[i] != INT64_MIN) {
        return false;
      }
      continue;
    }
    if (static_indices.i64_array[i] == INT64_MIN) {
      return false;
    }
    static_origin[i] = static_indices.i64_array[i];
  }
  loom_attribute_t static_origin_attr =
      loom_attr_i64_array(static_origin, static_indices.count);
  int64_t lane_indices[] = {0};
  return loom_vector_memory_access_static_lane_byte_offset(
      vector_access, static_origin_attr, lane_indices,
      IREE_ARRAYSIZE(lane_indices), out_static_byte_offset);
}

static void loom_low_source_memory_access_fold_exact_dynamic_indices(
    const loom_value_fact_table_t* fact_table,
    loom_value_slice_t dynamic_indices, loom_attribute_t static_indices,
    int64_t* folded_static_indices, uint8_t* dynamic_axes,
    loom_value_id_t* dynamic_index_values, uint8_t* dynamic_axis_count) {
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    folded_static_indices[i] = static_indices.i64_array[i];
  }

  uint8_t folded_dynamic_axis_count = 0;
  const uint8_t original_dynamic_axis_count = *dynamic_axis_count;
  for (uint8_t i = 0; i < original_dynamic_axis_count; ++i) {
    const uint8_t dynamic_axis = dynamic_axes[i];
    const loom_value_id_t dynamic_index = dynamic_indices.values[i];
    int64_t exact_index = 0;
    if (dynamic_axis < static_indices.count &&
        loom_low_source_memory_access_exact_i64(
            loom_value_fact_table_lookup(fact_table, dynamic_index),
            &exact_index) &&
        exact_index != INT64_MIN) {
      folded_static_indices[dynamic_axis] = exact_index;
      continue;
    }
    dynamic_axes[folded_dynamic_axis_count] = dynamic_axis;
    dynamic_index_values[folded_dynamic_axis_count] = dynamic_index;
    ++folded_dynamic_axis_count;
  }
  *dynamic_axis_count = folded_dynamic_axis_count;
}

static bool loom_low_source_memory_access_add_view_base_byte_offset(
    const loom_value_fact_table_t* fact_table, loom_value_id_t view_value_id,
    loom_low_source_memory_access_plan_t* plan,
    loom_low_source_memory_access_diagnostic_t* diagnostic,
    int64_t* inout_static_byte_offset) {
  IREE_ASSERT_ARGUMENT(fact_table);
  IREE_ASSERT_ARGUMENT(inout_static_byte_offset);
  loom_value_fact_view_reference_t view_reference = {0};
  if (!loom_value_facts_query_view_reference(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table, view_value_id),
          &view_reference)) {
    diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE;
    return false;
  }

  int64_t view_base_byte_offset = 0;
  if (!loom_low_source_memory_access_exact_i64(view_reference.base_byte_offset,
                                               &view_base_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE;
    return false;
  }
  int64_t static_byte_offset = 0;
  if (!iree_checked_add_i64(*inout_static_byte_offset, view_base_byte_offset,
                            &static_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW;
    return false;
  }
  *inout_static_byte_offset = static_byte_offset;
  plan->memory_space = view_reference.memory_space;
  plan->root_value_id = view_reference.root_value_id;
  return true;
}

static bool loom_low_source_memory_access_vector_lane_count(
    loom_type_t vector_type, uint32_t* out_lane_count) {
  IREE_ASSERT_ARGUMENT(out_lane_count);
  *out_lane_count = 0;
  if (!loom_type_is_vector(vector_type) || loom_type_rank(vector_type) != 1 ||
      loom_type_dim_is_dynamic_at(vector_type, 0)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(vector_type, 0);
  if (lane_count <= 0 || lane_count > UINT32_MAX) {
    return false;
  }
  *out_lane_count = (uint32_t)lane_count;
  return true;
}

static bool loom_low_source_memory_access_power_of_two_shift(
    int64_t value, uint32_t* out_shift) {
  IREE_ASSERT_ARGUMENT(out_shift);
  *out_shift = LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE;
  if (value <= 0 || value > UINT32_MAX) {
    return false;
  }
  uint32_t remaining_value = (uint32_t)value;
  if ((remaining_value & (remaining_value - 1)) != 0) {
    return false;
  }
  uint32_t shift = 0;
  while (remaining_value > 1) {
    remaining_value >>= 1;
    ++shift;
  }
  *out_shift = shift;
  return true;
}

static bool loom_low_source_memory_operation_kind_from_op(
    const loom_op_t* source_op,
    loom_low_source_memory_operation_kind_t* out_operation_kind) {
  IREE_ASSERT_ARGUMENT(out_operation_kind);
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VIEW_LOAD:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
      return true;
    case LOOM_OP_VECTOR_STORE:
    case LOOM_OP_VIEW_STORE:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
      return true;
    case LOOM_OP_VIEW_ATOMIC_REDUCE:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_REDUCE;
      return true;
    case LOOM_OP_VIEW_ATOMIC_RMW:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW;
      return true;
    case LOOM_OP_VIEW_ATOMIC_CMPXCHG:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_CMPXCHG;
      return true;
    case LOOM_OP_VIEW_PREFETCH:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_PREFETCH;
      return true;
    default:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
      return false;
  }
}

static loom_type_t loom_low_source_memory_element_vector_type(
    loom_type_t view_type) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR,
                             loom_type_element_type(view_type),
                             loom_dim_pack_static(1), /*encoding_id=*/0);
}

static loom_type_t loom_low_source_memory_access_payload_vector_type(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_memory_access_t access, loom_type_t view_type) {
  const loom_value_id_t value_id = loom_memory_access_value(access);
  if (value_id != LOOM_VALUE_ID_INVALID && value_id < module->values.count) {
    const loom_type_t value_type = loom_module_value_type(module, value_id);
    if (loom_type_is_vector(value_type)) {
      return value_type;
    }
  }
  if (source_op->result_count == 1) {
    const loom_value_id_t result_id = loom_op_const_results(source_op)[0];
    if (result_id < module->values.count) {
      const loom_type_t result_type = loom_module_value_type(module, result_id);
      if (loom_type_is_vector(result_type)) {
        return result_type;
      }
    }
  }
  return loom_low_source_memory_element_vector_type(view_type);
}

static bool loom_low_source_memory_access_plan_from_components(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_id_t view_value_id, loom_value_slice_t dynamic_indices,
    loom_attribute_t static_indices, loom_type_t view_type,
    loom_type_t vector_type, loom_vector_memory_cache_policy_t cache_policy,
    loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  *out_plan = (loom_low_source_memory_access_plan_t){
      .operation_kind = operation_kind,
      .view_value_id = view_value_id,
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = LOOM_VALUE_ID_INVALID,
      .cache_policy = cache_policy,
  };
  *out_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};

  loom_vector_memory_access_t vector_access;
  if (!loom_vector_memory_access_describe(module, view_type, vector_type,
                                          &vector_access)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED;
    return false;
  }
  switch (vector_access.layout_kind) {
    case LOOM_VECTOR_MEMORY_LAYOUT_DENSE:
    case LOOM_VECTOR_MEMORY_LAYOUT_STRIDED:
      break;
    case LOOM_VECTOR_MEMORY_LAYOUT_UNKNOWN:
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_LAYOUT;
      return false;
  }
  if (vector_access.static_element_byte_count <= 0 ||
      vector_access.static_element_byte_count > UINT32_MAX) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH;
    return false;
  }
  out_plan->element_byte_count =
      (uint32_t)vector_access.static_element_byte_count;
  if (vector_access.vector_rank != 1) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_RANK;
    return false;
  }
  if (!loom_low_source_memory_access_vector_lane_count(
          vector_type, &out_plan->vector_lane_count)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_LANE_COUNT;
    return false;
  }

  int64_t vector_axis_stride = 0;
  if (!loom_vector_memory_access_static_axis_stride(
          &vector_access, vector_access.first_vector_axis,
          &vector_axis_stride) ||
      !iree_checked_mul_i64(vector_axis_stride,
                            vector_access.static_element_byte_count,
                            &out_plan->vector_lane_byte_stride)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }

  uint8_t dynamic_axes[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY] = {0};
  uint8_t dynamic_axis_count = 0;
  if (!loom_low_source_memory_access_collect_dynamic_axes(
          static_indices, dynamic_axes, &dynamic_axis_count) ||
      dynamic_axis_count != dynamic_indices.count) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT;
    return false;
  }
  int64_t folded_static_indices[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  loom_value_id_t
      dynamic_index_values[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY] = {0};
  loom_low_source_memory_access_fold_exact_dynamic_indices(
      fact_table, dynamic_indices, static_indices, folded_static_indices,
      dynamic_axes, dynamic_index_values, &dynamic_axis_count);
  static_indices =
      loom_attr_i64_array(folded_static_indices, static_indices.count);

  int64_t static_byte_offset = 0;
  if (!loom_low_source_memory_access_static_byte_offset(
          &vector_access, static_indices, dynamic_axes, dynamic_axis_count,
          &static_byte_offset)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_STATIC_OFFSET;
    return false;
  }
  if (!loom_low_source_memory_access_add_view_base_byte_offset(
          fact_table, view_value_id, out_plan, out_diagnostic,
          &static_byte_offset)) {
    return false;
  }

  for (uint8_t i = 0; i < dynamic_axis_count; ++i) {
    const uint8_t dynamic_axis = dynamic_axes[i];
    if (dynamic_axis >= vector_access.view_rank) {
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS;
      return false;
    }

    int64_t axis_stride = 0;
    int64_t byte_stride = 0;
    if (!loom_vector_memory_access_static_axis_stride(
            &vector_access, dynamic_axis, &axis_stride) ||
        !iree_checked_mul_i64(axis_stride,
                              vector_access.static_element_byte_count,
                              &byte_stride)) {
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE;
      return false;
    }

    const loom_value_id_t dynamic_index = dynamic_index_values[i];
    loom_kernel_dimension_t dynamic_index_dimension =
        LOOM_KERNEL_DIMENSION_COUNT_;
    loom_low_source_memory_dynamic_index_source_t dynamic_index_source =
        LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE;
    if (loom_low_source_memory_access_value_as_workitem_id(
            module, dynamic_index, &dynamic_index_dimension)) {
      dynamic_index_source =
          LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID;
    } else if (loom_low_source_memory_access_value_as_workgroup_id(
                   module, dynamic_index, &dynamic_index_dimension)) {
      dynamic_index_source =
          LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID;
    }

    uint32_t byte_shift = LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE;
    (void)loom_low_source_memory_access_power_of_two_shift(byte_stride,
                                                           &byte_shift);
    out_plan->dynamic_terms[i] = (loom_low_source_memory_dynamic_term_t){
        .index = dynamic_index,
        .source = dynamic_index_source,
        .dimension = dynamic_index_dimension,
        .axis = dynamic_axis,
        .byte_stride = byte_stride,
        .byte_shift = byte_shift,
    };
  }
  out_plan->dynamic_term_count = dynamic_axis_count;
  out_plan->static_byte_offset = static_byte_offset;
  return true;
}

bool loom_low_source_memory_access_plan_build(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op, loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(fact_table);
  IREE_ASSERT_ARGUMENT(source_op);
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT_ARGUMENT(out_diagnostic);
  *out_plan = (loom_low_source_memory_access_plan_t){0};
  *out_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};

  loom_low_source_memory_operation_kind_t operation_kind =
      LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  if (!loom_low_source_memory_operation_kind_from_op(source_op,
                                                     &operation_kind)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP;
    return false;
  }

  loom_memory_access_t access = loom_memory_access_cast(module, source_op);
  if (!loom_memory_access_isa(access)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP;
    return false;
  }
  const loom_value_id_t view_value_id = loom_memory_access_view(access);
  if (view_value_id >= module->values.count) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE;
    return false;
  }

  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_attrs(
          loom_memory_access_cache_scope(access),
          loom_memory_access_cache_temporal(access), &cache_policy)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_CACHE_POLICY;
    return false;
  }

  const loom_type_t view_type = loom_module_value_type(module, view_value_id);
  const loom_type_t vector_type =
      loom_low_source_memory_access_payload_vector_type(module, source_op,
                                                        access, view_type);
  return loom_low_source_memory_access_plan_from_components(
      module, fact_table, operation_kind, view_value_id,
      loom_memory_access_dynamic_indices(access),
      loom_memory_access_static_indices(access), view_type, vector_type,
      cache_policy, out_plan, out_diagnostic);
}

bool loom_low_source_memory_access_plan_build_view(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_id_t view_value_id,
    loom_vector_memory_cache_policy_t cache_policy,
    loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(fact_table);
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT_ARGUMENT(out_diagnostic);
  *out_plan = (loom_low_source_memory_access_plan_t){0};
  *out_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  if (view_value_id >= module->values.count) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE;
    return false;
  }

  const loom_type_t result_view_type =
      loom_module_value_type(module, view_value_id);
  loom_type_t vector_type = loom_type_none();
  if (!loom_low_source_memory_static_view_vector_type(
          result_view_type, &vector_type, out_diagnostic)) {
    return false;
  }

  loom_value_id_t access_view_id = view_value_id;
  loom_value_slice_t dynamic_indices = {0};
  int64_t zero_indices[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  loom_attribute_t static_indices =
      loom_attr_i64_array(zero_indices, loom_type_rank(result_view_type));
  loom_type_t access_view_type = result_view_type;

  loom_value_id_t projection_view_id = view_value_id;
  const loom_value_t* view_value = loom_module_value(module, view_value_id);
  if (!loom_value_is_block_arg(view_value)) {
    const loom_op_t* defining_op = loom_value_def_op(view_value);
    if (defining_op && loom_view_refine_isa(defining_op)) {
      projection_view_id = loom_view_refine_source(defining_op);
    }
  }

  const loom_value_t* projection_view =
      loom_module_value(module, projection_view_id);
  if (!loom_value_is_block_arg(projection_view)) {
    const loom_op_t* defining_op = loom_value_def_op(projection_view);
    if (defining_op && loom_view_subview_isa(defining_op)) {
      access_view_id = loom_view_subview_source(defining_op);
      access_view_type = loom_module_value_type(module, access_view_id);
      dynamic_indices = loom_view_subview_offsets(defining_op);
      static_indices = loom_view_subview_static_offsets(defining_op);
    }
  }

  return loom_low_source_memory_access_plan_from_components(
      module, fact_table, operation_kind, access_view_id, dynamic_indices,
      static_indices, access_view_type, vector_type, cache_policy, out_plan,
      out_diagnostic);
}

iree_string_view_t loom_low_source_memory_access_rejection_detail(
    loom_low_source_memory_access_rejection_flags_t rejection_bits) {
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP)) {
    return IREE_SV("source op is not a supported memory access");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DESCRIBE_FAILED)) {
    return IREE_SV("source view/vector types do not form a memory access");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_LAYOUT)) {
    return IREE_SV("source view memory layout is unknown");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_ELEMENT_WIDTH)) {
    return IREE_SV("source memory element width is not byte-addressable");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_RANK)) {
    return IREE_SV("source memory lowering requires a rank-1 vector payload");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_LANE_COUNT)) {
    return IREE_SV(
        "source memory lowering requires a static vector lane count");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE)) {
    return IREE_SV("source vector lane stride is not statically known");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_STATIC_OFFSET)) {
    return IREE_SV("source static memory offset is not statically known");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT)) {
    return IREE_SV(
        "source memory access dynamic index list does not match dynamic axes");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS)) {
    return IREE_SV("source memory access does not identify one dynamic axis");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE)) {
    return IREE_SV("source dynamic memory stride is not statically known");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_SOURCE)) {
    return IREE_SV("source view does not have a known storage root");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE)) {
    return IREE_SV("source view base byte offset is not exact");
  }
  if (iree_any_bit_set(
          rejection_bits,
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_VIEW_BASE_OVERFLOW)) {
    return IREE_SV("source view base byte offset overflows the access offset");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_CACHE_POLICY)) {
    return IREE_SV("source memory cache policy is malformed");
  }
  return IREE_SV("source memory access is not representable");
}
