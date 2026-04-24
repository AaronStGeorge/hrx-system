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

static bool loom_low_source_memory_access_find_dynamic_axis(
    loom_attribute_t static_indices, uint8_t* out_dynamic_axis) {
  IREE_ASSERT_ARGUMENT(out_dynamic_axis);
  *out_dynamic_axis = UINT8_MAX;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY) {
    return false;
  }
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (static_indices.i64_array[i] != INT64_MIN) {
      continue;
    }
    if (*out_dynamic_axis != UINT8_MAX || i > UINT8_MAX) {
      return false;
    }
    *out_dynamic_axis = (uint8_t)i;
  }
  return true;
}

static bool loom_low_source_memory_access_static_byte_offset(
    const loom_vector_memory_access_t* vector_access,
    loom_attribute_t static_indices, uint8_t dynamic_axis,
    int64_t* out_static_byte_offset) {
  IREE_ASSERT_ARGUMENT(out_static_byte_offset);
  *out_static_byte_offset = 0;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      static_indices.count > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK) {
    return false;
  }

  int64_t static_origin[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (i == dynamic_axis) {
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

static bool loom_low_source_memory_access_add_view_base_byte_offset(
    const loom_value_fact_table_t* fact_table, loom_value_id_t view_value_id,
    loom_low_source_memory_access_plan_t* plan,
    loom_low_source_memory_access_diagnostic_t* diagnostic,
    int64_t* inout_static_byte_offset) {
  IREE_ASSERT_ARGUMENT(inout_static_byte_offset);
  loom_value_fact_view_reference_t view_reference = {0};
  if (fact_table == NULL ||
      !loom_value_facts_query_view_reference(
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
      .dynamic_index = LOOM_VALUE_ID_INVALID,
      .dynamic_index_source = LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_NONE,
      .dynamic_index_dimension = LOOM_KERNEL_DIMENSION_COUNT_,
      .dynamic_axis = UINT8_MAX,
      .dynamic_index_byte_shift = LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE,
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

  if (dynamic_indices.count == 0) {
    int64_t lane_indices[] = {0};
    int64_t static_byte_offset = 0;
    if (!loom_vector_memory_access_static_lane_byte_offset(
            &vector_access, static_indices, lane_indices,
            IREE_ARRAYSIZE(lane_indices), &static_byte_offset)) {
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_STATIC_OFFSET;
      return false;
    }
    if (!loom_low_source_memory_access_add_view_base_byte_offset(
            fact_table, view_value_id, out_plan, out_diagnostic,
            &static_byte_offset)) {
      return false;
    }
    out_plan->static_byte_offset = static_byte_offset;
    return true;
  }

  if (dynamic_indices.count != 1) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_COUNT;
    return false;
  }
  uint8_t dynamic_axis = UINT8_MAX;
  if (!loom_low_source_memory_access_find_dynamic_axis(static_indices,
                                                       &dynamic_axis) ||
      dynamic_axis == UINT8_MAX || dynamic_axis >= vector_access.view_rank) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_AXIS;
    return false;
  }

  int64_t axis_stride = 0;
  if (!loom_vector_memory_access_static_axis_stride(
          &vector_access, dynamic_axis, &axis_stride) ||
      !iree_checked_mul_i64(axis_stride,
                            vector_access.static_element_byte_count,
                            &out_plan->dynamic_index_byte_stride)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_DYNAMIC_STRIDE;
    return false;
  }
  int64_t static_byte_offset = 0;
  if (!loom_low_source_memory_access_static_byte_offset(
          &vector_access, static_indices, dynamic_axis, &static_byte_offset)) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_STATIC_OFFSET;
    return false;
  }
  if (!loom_low_source_memory_access_add_view_base_byte_offset(
          fact_table, view_value_id, out_plan, out_diagnostic,
          &static_byte_offset)) {
    return false;
  }

  const loom_value_id_t dynamic_index = dynamic_indices.values[0];
  loom_kernel_dimension_t dynamic_index_dimension =
      LOOM_KERNEL_DIMENSION_COUNT_;
  if (loom_low_source_memory_access_value_as_workitem_id(
          module, dynamic_index, &dynamic_index_dimension)) {
    out_plan->dynamic_index_source =
        LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID;
  } else if (loom_low_source_memory_access_value_as_workgroup_id(
                 module, dynamic_index, &dynamic_index_dimension)) {
    out_plan->dynamic_index_source =
        LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID;
  } else {
    out_plan->dynamic_index_source =
        LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE;
  }
  (void)loom_low_source_memory_access_power_of_two_shift(
      out_plan->dynamic_index_byte_stride, &out_plan->dynamic_index_byte_shift);
  out_plan->dynamic_index = dynamic_index;
  out_plan->dynamic_index_dimension = dynamic_index_dimension;
  out_plan->dynamic_axis = dynamic_axis;
  out_plan->static_byte_offset = static_byte_offset;
  return true;
}

bool loom_low_source_memory_access_plan_build(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op, loom_low_source_memory_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_diagnostic) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(source_op);
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT_ARGUMENT(out_diagnostic);
  *out_plan = (loom_low_source_memory_access_plan_t){0};
  *out_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  if (source_op->kind != LOOM_OP_VECTOR_LOAD &&
      source_op->kind != LOOM_OP_VECTOR_STORE &&
      source_op->kind != LOOM_OP_VIEW_PREFETCH) {
    out_diagnostic->rejection_bits |=
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP;
    return false;
  }
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_LOAD: {
      loom_vector_memory_cache_policy_t cache_policy = {0};
      if (!loom_vector_memory_cache_policy_from_op(source_op, &cache_policy)) {
        out_diagnostic->rejection_bits |=
            LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_CACHE_POLICY;
        return false;
      }
      return loom_low_source_memory_access_plan_from_components(
          module, fact_table, LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD,
          loom_vector_load_view(source_op), loom_vector_load_indices(source_op),
          loom_vector_load_static_indices(source_op),
          loom_module_value_type(module, loom_vector_load_view(source_op)),
          loom_module_value_type(module, loom_vector_load_result(source_op)),
          cache_policy, out_plan, out_diagnostic);
    }
    case LOOM_OP_VECTOR_STORE: {
      loom_vector_memory_cache_policy_t cache_policy = {0};
      if (!loom_vector_memory_cache_policy_from_op(source_op, &cache_policy)) {
        out_diagnostic->rejection_bits |=
            LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_CACHE_POLICY;
        return false;
      }
      return loom_low_source_memory_access_plan_from_components(
          module, fact_table, LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE,
          loom_vector_store_view(source_op),
          loom_vector_store_indices(source_op),
          loom_vector_store_static_indices(source_op),
          loom_module_value_type(module, loom_vector_store_view(source_op)),
          loom_module_value_type(module, loom_vector_store_value(source_op)),
          cache_policy, out_plan, out_diagnostic);
    }
    case LOOM_OP_VIEW_PREFETCH: {
      const loom_value_id_t view_value_id = loom_view_prefetch_view(source_op);
      const loom_type_t view_type =
          loom_module_value_type(module, view_value_id);
      const loom_type_t element_vector_type = loom_type_shaped_1d(
          LOOM_TYPE_VECTOR, loom_type_element_type(view_type),
          loom_dim_pack_static(1), /*encoding_id=*/0);
      return loom_low_source_memory_access_plan_from_components(
          module, fact_table, LOOM_LOW_SOURCE_MEMORY_OPERATION_PREFETCH,
          view_value_id, loom_view_prefetch_indices(source_op),
          loom_view_prefetch_static_indices(source_op), view_type,
          element_vector_type, (loom_vector_memory_cache_policy_t){0}, out_plan,
          out_diagnostic);
    }
    default:
      out_diagnostic->rejection_bits |=
          LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP;
      return false;
  }
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
    return IREE_SV("source memory access requires zero or one dynamic index");
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
