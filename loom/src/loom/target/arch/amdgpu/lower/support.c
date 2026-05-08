// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "loom/analysis/view_regions.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/context.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/internal.h"
#include "loom/target/arch/amdgpu/target_refs.h"
#include "loom/util/math.h"

bool loom_amdgpu_type_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

bool loom_amdgpu_type_is_i64(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I64;
}

bool loom_amdgpu_type_is_i1(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

bool loom_amdgpu_type_is_address_scalar(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         (loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX ||
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET);
}

bool loom_amdgpu_type_is_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

bool loom_amdgpu_type_is_16bit_float(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  return element_type == LOOM_SCALAR_TYPE_F16 ||
         element_type == LOOM_SCALAR_TYPE_BF16;
}

uint32_t loom_amdgpu_static_vector_lane_count(loom_type_t type,
                                              loom_scalar_type_t element_type,
                                              uint32_t max_lane_count) {
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type) ||
      loom_type_element_type(type) != element_type) {
    return 0;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count < 1 || lane_count > (int64_t)max_lane_count) {
    return 0;
  }
  return (uint32_t)lane_count;
}

uint32_t loom_amdgpu_static_vector_register_count(
    loom_type_t type, loom_scalar_type_t element_type,
    uint32_t max_register_count) {
  if (!loom_type_is_vector(type) || !loom_type_is_all_static(type) ||
      loom_type_element_type(type) != element_type) {
    return 0;
  }
  uint64_t element_count = 0;
  if (!loom_type_static_element_count(type, &element_count) ||
      element_count < 1 || element_count > max_register_count) {
    return 0;
  }
  return (uint32_t)element_count;
}

static uint32_t loom_amdgpu_vector_lane_count(loom_type_t type,
                                              loom_scalar_type_t element_type) {
  return loom_amdgpu_static_vector_lane_count(
      type, element_type, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
}

static uint32_t loom_amdgpu_vector_register_count(
    loom_type_t type, loom_scalar_type_t element_type) {
  return loom_amdgpu_static_vector_register_count(
      type, element_type, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
}

static bool loom_amdgpu_type_is_vector_32bit_register_range(loom_type_t type) {
  return loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_I32) != 0 ||
         loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_F32) != 0;
}

bool loom_amdgpu_type_is_32bit_memory_payload(loom_type_t type) {
  return loom_amdgpu_type_is_i32(type) || loom_amdgpu_type_is_f32(type) ||
         loom_amdgpu_static_vector_lane_count(
             type, LOOM_SCALAR_TYPE_I32, LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES) !=
             0 ||
         loom_amdgpu_static_vector_lane_count(
             type, LOOM_SCALAR_TYPE_F32, LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES) !=
             0;
}

static bool loom_amdgpu_type_is_offset(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET;
}

uint32_t loom_amdgpu_vector_32bit_lane_count(loom_type_t type) {
  const uint32_t i32_lane_count =
      loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_I32);
  return i32_lane_count != 0
             ? i32_lane_count
             : loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_F32);
}

uint32_t loom_amdgpu_vector_32bit_register_count(loom_type_t type) {
  const uint32_t i32_register_count =
      loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_I32);
  return i32_register_count != 0
             ? i32_register_count
             : loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_F32);
}

bool loom_amdgpu_static_vector_flat_register_from_indices(
    loom_type_t type, const int64_t* indices, uint32_t* out_ordinal) {
  uint32_t ordinal = 0;
  const uint8_t rank = loom_type_rank(type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    const int64_t dimension_size = loom_type_dim_static_size_at(type, axis);
    if (dimension_size < 1 || indices[axis] < 0 ||
        indices[axis] >= dimension_size ||
        ordinal >
            (UINT32_MAX - (uint32_t)indices[axis]) / (uint32_t)dimension_size) {
      return false;
    }
    ordinal = ordinal * (uint32_t)dimension_size + (uint32_t)indices[axis];
  }
  *out_ordinal = ordinal;
  return true;
}

uint32_t loom_amdgpu_vector_i32_lane_count(loom_type_t type) {
  return loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_I32);
}

uint32_t loom_amdgpu_vector_i32_register_count(loom_type_t type) {
  return loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_I32);
}

uint32_t loom_amdgpu_vector_f32_lane_count(loom_type_t type) {
  return loom_amdgpu_vector_lane_count(type, LOOM_SCALAR_TYPE_F32);
}

uint32_t loom_amdgpu_vector_f32_register_count(loom_type_t type) {
  return loom_amdgpu_vector_register_count(type, LOOM_SCALAR_TYPE_F32);
}

uint32_t loom_amdgpu_vector_i1_lane_count(loom_type_t type) {
  return loom_amdgpu_static_vector_lane_count(
      type, LOOM_SCALAR_TYPE_I1, LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES);
}

uint32_t loom_amdgpu_vector_i8_lane_count(loom_type_t type) {
  return loom_amdgpu_static_vector_lane_count(type, LOOM_SCALAR_TYPE_I8,
                                              LOOM_AMDGPU_MAX_PACKED_I8_LANES);
}

bool loom_amdgpu_type_packed_integer_storage(loom_type_t type,
                                             uint32_t* out_payload_bit_count,
                                             uint32_t* out_register_count) {
  *out_payload_bit_count = 0;
  *out_register_count = 0;
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count < 1 || lane_count > INT32_MAX) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  if (!loom_scalar_type_is_integer(element_type)) {
    return false;
  }
  const int32_t element_bit_count = loom_scalar_type_bitwidth(element_type);
  if (element_bit_count <= 0) {
    return false;
  }
  int64_t total_bit_count = 0;
  if (!iree_checked_mul_i64(lane_count, element_bit_count, &total_bit_count) ||
      total_bit_count <= 0) {
    return false;
  }
  const int64_t register_count = (total_bit_count + 31) / 32;
  if (register_count < 1 ||
      register_count > (int64_t)LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return false;
  }
  *out_payload_bit_count = (uint32_t)total_bit_count;
  *out_register_count = (uint32_t)register_count;
  return true;
}

bool loom_amdgpu_type_packed_16bit_float_storage(
    loom_type_t type, uint32_t* out_payload_bit_count,
    uint32_t* out_register_count) {
  *out_payload_bit_count = 0;
  *out_register_count = 0;
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count < 1 ||
      lane_count > (int64_t)LOOM_AMDGPU_MAX_PACKED_16BIT_FLOAT_LANES) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  if (element_type != LOOM_SCALAR_TYPE_F16 &&
      element_type != LOOM_SCALAR_TYPE_BF16) {
    return false;
  }
  const uint32_t register_count = (uint32_t)((lane_count + 1) / 2);
  *out_payload_bit_count = (uint32_t)lane_count * 16u;
  *out_register_count = register_count;
  return true;
}

bool loom_amdgpu_type_is_byte_addressable_view(loom_type_t type) {
  if (!loom_type_is_view(type)) {
    return false;
  }
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  return element_bit_count > 0 && (element_bit_count % 8) == 0;
}

static iree_string_view_t loom_amdgpu_nonempty(iree_string_view_t value,
                                               iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

void loom_amdgpu_low_legality_make_context_params(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_diagnostic_param_t* params) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  params[0] =
      loom_param_string(loom_amdgpu_nonempty(bundle->name, IREE_SV("<empty>")));
  params[1] = loom_param_string(
      loom_amdgpu_nonempty(bundle->export_plan->name, IREE_SV("<empty>")));
  params[2] = loom_param_string(
      loom_amdgpu_nonempty(bundle->config->name, IREE_SV("<empty>")));
  params[3] =
      loom_param_string(loom_target_low_legality_function_name(context));
  params[4] = loom_param_string(
      loom_op_name(loom_target_low_legality_module(context), op));
}

iree_status_t loom_amdgpu_low_legality_reject(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t constraint_key) {
  loom_diagnostic_param_t
      params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT + 1];
  loom_amdgpu_low_legality_make_context_params(context, op, params);
  params[LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT] =
      loom_param_string(constraint_key);
  return loom_target_low_legality_emit_error_ref(
      context, op, LOOM_ERR_AMDGPU_023_REF, params, IREE_ARRAYSIZE(params));
}

bool loom_amdgpu_value_is_i32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id) {
  return loom_amdgpu_type_is_i32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_address_scalar(loom_low_lower_context_t* context,
                                         loom_value_id_t value_id) {
  return loom_amdgpu_type_is_address_scalar(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_f32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id) {
  return loom_amdgpu_type_is_f32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_16bit_float(loom_low_lower_context_t* context,
                                      loom_value_id_t value_id) {
  return loom_amdgpu_type_is_16bit_float(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_is_byte_addressable_view(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_byte_addressable_view(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

static iree_status_t loom_amdgpu_make_register_type(
    loom_low_lower_context_t* context, uint16_t reg_class_id,
    uint32_t unit_count, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(context, reg_class_id, unit_count,
                                           out_type);
}

iree_status_t loom_amdgpu_make_sgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                                        1, out_type);
}

iree_status_t loom_amdgpu_make_sgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_SGPR,
                                        unit_count, out_type);
}

iree_status_t loom_amdgpu_make_vgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                        1, out_type);
}

iree_status_t loom_amdgpu_make_scc_type(loom_low_lower_context_t* context,
                                        loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_SCC,
                                        1, out_type);
}

iree_status_t loom_amdgpu_make_vgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type) {
  return loom_amdgpu_make_register_type(context, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                        unit_count, out_type);
}

iree_status_t loom_amdgpu_make_descriptor_row_implicit_resource_type(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    loom_type_t* out_type) {
  *out_type = loom_type_none();
  IREE_ASSERT(descriptor != NULL);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_operand_t* operands =
      &descriptor_set->operands[descriptor->operand_start];
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const loom_low_operand_t* operand = &operands[i];
    if (operand->role != LOOM_LOW_OPERAND_ROLE_RESOURCE ||
        !iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_IMPLICIT)) {
      continue;
    }
    for (uint16_t j = 0; j < operand->reg_class_alt_count; ++j) {
      const uint32_t alt_index = operand->reg_class_alt_start + j;
      if (alt_index >= descriptor_set->reg_class_alt_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU implicit-resource descriptor has an invalid register-class "
            "alternative span");
      }
      const loom_low_reg_class_alt_t* alt =
          &descriptor_set->reg_class_alts[alt_index];
      if (iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE)) {
        continue;
      }
      if (alt->reg_class_id >= descriptor_set->reg_class_count) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU implicit-resource descriptor references an invalid "
            "register class");
      }
      return loom_amdgpu_make_register_type(context, alt->reg_class_id,
                                            operand->unit_count, out_type);
    }
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU descriptor has no explicit implicit resource operand");
}

iree_status_t loom_amdgpu_low_type_register_class_is(
    loom_low_lower_context_t* context, loom_type_t type, uint16_t reg_class_id,
    bool* out_match) {
  *out_match = false;
  if (!loom_type_is_register(type)) {
    return iree_ok_status();
  }
  loom_string_id_t expected_class_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_register_class_string_id(
      context, reg_class_id, &expected_class_id));
  *out_match = loom_type_register_class_id(type) == expected_class_id;
  return iree_ok_status();
}

bool loom_amdgpu_source_value_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id);

static bool loom_amdgpu_source_memory_root_is_read_only(
    const loom_low_source_memory_access_plan_t* plan,
    const loom_view_region_table_t* view_regions) {
  if (view_regions == NULL ||
      plan->alias_scope_id == LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE) {
    return false;
  }
  const loom_view_access_flags_t access_flags =
      loom_view_region_table_root_access_flags(view_regions,
                                               plan->root_value_id);
  return iree_all_bits_set(access_flags, LOOM_VIEW_ACCESS_READ) &&
         !iree_any_bit_set(access_flags, LOOM_VIEW_ACCESS_WRITE);
}

static bool loom_amdgpu_source_memory_terms_prefer_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    const loom_low_source_memory_access_plan_t* plan) {
  for (uint8_t i = 0; i < plan->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term = &plan->dynamic_terms[i];
    switch (term->source) {
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_NONE:
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKGROUP_ID:
        break;
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID:
        return true;
      case LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE:
        if (loom_amdgpu_source_value_prefers_vgpr(module, fact_table,
                                                  view_regions, term->index)) {
          return true;
        }
        break;
    }
    for (uint8_t stride_ordinal = 0; stride_ordinal < term->stride_value_count;
         ++stride_ordinal) {
      if (loom_amdgpu_source_value_prefers_vgpr(
              module, fact_table, view_regions,
              term->stride_values[stride_ordinal])) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_amdgpu_source_memory_access_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions, const loom_op_t* source_op,
    loom_type_t source_type) {
  if (fact_table == NULL || view_regions == NULL) {
    return true;
  }
  if (!loom_amdgpu_type_is_32bit_memory_payload(source_type)) {
    return true;
  }
  loom_low_source_memory_access_plan_t plan = {0};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  if (!loom_low_source_memory_access_plan_build(module, fact_table, source_op,
                                                &plan, &diagnostic)) {
    return true;
  }
  if (plan.operation_kind != LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD ||
      (plan.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
       plan.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT) ||
      iree_any_bit_set(
          plan.cache_policy.build_flags,
          LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE |
              LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL) ||
      !loom_amdgpu_source_memory_root_is_read_only(&plan, view_regions) ||
      loom_amdgpu_source_memory_terms_prefer_vgpr(module, fact_table,
                                                  view_regions, &plan)) {
    return true;
  }
  return false;
}

static bool loom_amdgpu_vector_extract_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions, const loom_op_t* source_op) {
  const loom_attribute_t static_indices =
      loom_vector_extract_static_indices(source_op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      static_indices.i64_array[0] == INT64_MIN) {
    return true;
  }
  return loom_amdgpu_source_value_prefers_vgpr(
      module, fact_table, view_regions, loom_vector_extract_source(source_op));
}

static bool loom_amdgpu_source_value_is_native_i1_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id);

bool loom_amdgpu_source_value_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id) {
  if (source_value_id >= module->values.count) {
    return false;
  }

  loom_type_t source_type = loom_module_value_type(module, source_value_id);
  if (loom_amdgpu_type_is_f32(source_type) ||
      loom_amdgpu_type_is_16bit_float(source_type)) {
    return true;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_amdgpu_type_is_vector_32bit_register_range(source_type);
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL) {
    return loom_amdgpu_type_is_vector_32bit_register_range(source_type);
  }

  if (loom_traits_are_fact_identity(
          loom_op_effective_traits(module, defining_op))) {
    const uint16_t result_index = loom_value_def_index(value);
    if (result_index >= defining_op->operand_count) {
      return false;
    }
    const loom_value_id_t source_identity_value_id =
        loom_op_const_operands(defining_op)[result_index];
    if (source_identity_value_id == source_value_id) {
      return false;
    }
    return loom_amdgpu_source_value_prefers_vgpr(
        module, fact_table, view_regions, source_identity_value_id);
  }

  switch (defining_op->kind) {
    case LOOM_OP_KERNEL_WORKITEM_ID:
      return loom_kernel_workitem_id_dimension(defining_op) <
             LOOM_KERNEL_DIMENSION_COUNT_;
    case LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID:
      return loom_kernel_workitem_dispatch_id_dimension(defining_op) <
             LOOM_KERNEL_DIMENSION_COUNT_;
    case LOOM_OP_KERNEL_SUBGROUP_ID:
    case LOOM_OP_KERNEL_SUBGROUP_LANE_ID:
      return true;
    case LOOM_OP_KERNEL_SUBGROUP_BROADCAST:
    case LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST:
    case LOOM_OP_KERNEL_SUBGROUP_REDUCE:
    case LOOM_OP_KERNEL_SUBGROUP_SCAN:
      return loom_value_def_index(value) == 0;
    case LOOM_OP_KERNEL_SUBGROUP_SHUFFLE:
      return loom_value_def_index(value) == 0;
    case LOOM_OP_INDEX_CAST:
      return loom_amdgpu_source_value_prefers_vgpr(
          module, fact_table, view_regions, loom_index_cast_input(defining_op));
    case LOOM_OP_INDEX_ADD:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_add_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_add_rhs(defining_op));
    case LOOM_OP_INDEX_SUB:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_sub_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_sub_rhs(defining_op));
    case LOOM_OP_INDEX_MUL:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_mul_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_mul_rhs(defining_op));
    case LOOM_OP_INDEX_MIN:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_min_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_min_rhs(defining_op));
    case LOOM_OP_INDEX_MAX:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_max_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_max_rhs(defining_op));
    case LOOM_OP_INDEX_ANDI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_andi_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_andi_rhs(defining_op));
    case LOOM_OP_INDEX_ORI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_ori_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_ori_rhs(defining_op));
    case LOOM_OP_INDEX_XORI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_xori_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_xori_rhs(defining_op));
    case LOOM_OP_INDEX_SHLI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_shli_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_shli_rhs(defining_op));
    case LOOM_OP_INDEX_SHRSI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_shrsi_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_shrsi_rhs(defining_op));
    case LOOM_OP_INDEX_SHRUI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_shrui_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_shrui_rhs(defining_op));
    case LOOM_OP_INDEX_MADD:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_madd_a(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_madd_b(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_index_madd_c(defining_op));
    case LOOM_OP_VECTOR_EXTRACT:
      return loom_amdgpu_vector_extract_prefers_vgpr(module, fact_table,
                                                     view_regions, defining_op);
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VIEW_LOAD:
      return loom_amdgpu_source_memory_access_prefers_vgpr(
          module, fact_table, view_regions, defining_op, source_type);
    case LOOM_OP_VECTOR_REDUCE:
      return loom_amdgpu_type_is_vector_32bit_register_range(
          loom_module_value_type(module,
                                 loom_vector_reduce_input(defining_op)));
    case LOOM_OP_VIEW_ATOMIC_CMPXCHG:
    case LOOM_OP_VIEW_ATOMIC_RMW:
      return true;
    case LOOM_OP_SCF_SELECT:
      return loom_amdgpu_source_value_is_native_i1_mask(
                 module, fact_table, view_regions,
                 loom_scf_select_condition(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scf_select_true_value(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scf_select_false_value(defining_op));
    case LOOM_OP_SCALAR_ADDI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_addi_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_addi_rhs(defining_op));
    case LOOM_OP_SCALAR_SUBI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_subi_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_subi_rhs(defining_op));
    case LOOM_OP_SCALAR_MULI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_muli_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_muli_rhs(defining_op));
    case LOOM_OP_SCALAR_MINSI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_minsi_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_minsi_rhs(defining_op));
    case LOOM_OP_SCALAR_MAXSI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_maxsi_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_maxsi_rhs(defining_op));
    case LOOM_OP_SCALAR_MINUI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_minui_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_minui_rhs(defining_op));
    case LOOM_OP_SCALAR_MAXUI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_maxui_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_maxui_rhs(defining_op));
    case LOOM_OP_SCALAR_ANDI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_andi_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_andi_rhs(defining_op));
    case LOOM_OP_SCALAR_ORI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_ori_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_ori_rhs(defining_op));
    case LOOM_OP_SCALAR_XORI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_xori_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_xori_rhs(defining_op));
    case LOOM_OP_SCALAR_SHLI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_shli_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_shli_rhs(defining_op));
    case LOOM_OP_SCALAR_SHRSI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_shrsi_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_shrsi_rhs(defining_op));
    case LOOM_OP_SCALAR_SHRUI:
      return loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_shrui_lhs(defining_op)) ||
             loom_amdgpu_source_value_prefers_vgpr(
                 module, fact_table, view_regions,
                 loom_scalar_shrui_rhs(defining_op));
    case LOOM_OP_SCALAR_TRUNCI:
      return loom_amdgpu_source_value_prefers_vgpr(
          module, fact_table, view_regions,
          loom_scalar_trunci_input(defining_op));
    default:
      return loom_amdgpu_type_is_vector_32bit_register_range(source_type);
  }
}

bool loom_amdgpu_module_value_prefers_vgpr(const loom_module_t* module,
                                           loom_value_id_t source_value_id) {
  return loom_amdgpu_source_value_prefers_vgpr(
      module, /*fact_table=*/NULL, /*view_regions=*/NULL, source_value_id);
}

bool loom_amdgpu_value_prefers_vgpr(loom_low_lower_context_t* context,
                                    loom_value_id_t source_value_id) {
  return loom_amdgpu_module_value_prefers_vgpr(
      loom_low_lower_context_module(context), source_value_id);
}

static iree_status_t loom_amdgpu_context_value_prefers_vgpr(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    bool* out_prefers_vgpr) {
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  *out_prefers_vgpr = loom_amdgpu_source_value_prefers_vgpr(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context), view_regions,
      source_value_id);
  return iree_ok_status();
}

static bool loom_amdgpu_scf_select_use_needs_native_i1_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions, const loom_value_t* value) {
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    if (loom_use_operand_index(*use) != 0) {
      continue;
    }
    const loom_op_t* user_op = loom_use_user_op(*use);
    if (!loom_scf_select_isa(user_op)) {
      continue;
    }
    const loom_type_t result_type =
        loom_module_value_type(module, loom_scf_select_result(user_op));
    if (loom_amdgpu_type_is_f32(result_type) ||
        loom_amdgpu_type_is_vector_32bit_register_range(result_type)) {
      return true;
    }
    if (loom_amdgpu_source_value_prefers_vgpr(
            module, fact_table, view_regions,
            loom_scf_select_true_value(user_op)) ||
        loom_amdgpu_source_value_prefers_vgpr(
            module, fact_table, view_regions,
            loom_scf_select_false_value(user_op))) {
      return true;
    }
  }
  return false;
}

static bool loom_amdgpu_source_value_is_native_i1_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id) {
  if (source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_amdgpu_scf_select_use_needs_native_i1_mask(module, fact_table,
                                                      view_regions, value)) {
    return true;
  }
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op) {
    return false;
  }

  if (loom_traits_are_fact_identity(
          loom_op_effective_traits(module, defining_op))) {
    const uint16_t result_index = loom_value_def_index(value);
    if (result_index >= defining_op->operand_count) {
      return false;
    }
    const loom_value_id_t source_identity_value_id =
        loom_op_const_operands(defining_op)[result_index];
    return source_identity_value_id != source_value_id &&
           loom_amdgpu_source_value_is_native_i1_mask(
               module, fact_table, view_regions, source_identity_value_id);
  }

  if (loom_scalar_cmpi_isa(defining_op) && loom_value_def_index(value) == 0) {
    return loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_scalar_cmpi_lhs(defining_op)) ||
           loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_scalar_cmpi_rhs(defining_op));
  }

  if (loom_scalar_cmpf_isa(defining_op) && loom_value_def_index(value) == 0) {
    return loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_scalar_cmpf_lhs(defining_op)) ||
           loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_scalar_cmpf_rhs(defining_op));
  }

  if (loom_index_cmp_isa(defining_op) && loom_value_def_index(value) == 0) {
    return loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_index_cmp_lhs(defining_op)) ||
           loom_amdgpu_source_value_prefers_vgpr(
               module, fact_table, view_regions,
               loom_index_cmp_rhs(defining_op));
  }

  return loom_kernel_subgroup_shuffle_isa(defining_op) &&
         loom_value_def_index(value) == 1;
}

bool loom_amdgpu_module_value_is_native_i1_mask(
    const loom_module_t* module, loom_value_id_t source_value_id) {
  return loom_amdgpu_source_value_is_native_i1_mask(
      module, /*fact_table=*/NULL, /*view_regions=*/NULL, source_value_id);
}

static iree_status_t loom_amdgpu_context_value_is_native_i1_mask(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    bool* out_is_native_mask) {
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  *out_is_native_mask = loom_amdgpu_source_value_is_native_i1_mask(
      loom_low_lower_context_module(context),
      loom_low_lower_context_fact_table(context), view_regions,
      source_value_id);
  return iree_ok_status();
}

static bool loom_amdgpu_value_is_subgroup_lane_mask_result(
    const loom_module_t* module, loom_value_id_t source_value_id) {
  if (source_value_id >= module->values.count ||
      !loom_amdgpu_type_is_i64(
          loom_module_value_type(module, source_value_id))) {
    return false;
  }

  const loom_value_t* value = loom_module_value(module, source_value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL || loom_value_def_index(value) != 0) {
    return false;
  }
  return loom_kernel_subgroup_active_mask_isa(defining_op) ||
         loom_kernel_subgroup_vote_ballot_isa(defining_op);
}

static bool loom_amdgpu_offset_value_needs_64bit(
    const loom_value_fact_table_t* fact_table, const loom_module_t* module,
    loom_value_id_t source_value_id, loom_type_t source_type) {
  if (!loom_amdgpu_type_is_offset(source_type)) {
    return false;
  }
  if (fact_table == NULL || module == NULL ||
      source_value_id >= module->values.count) {
    return true;
  }
  return !loom_value_facts_fit_unsigned_bit_count(
      loom_value_fact_table_lookup(fact_table, source_value_id), 32);
}

iree_status_t loom_amdgpu_map_type(void* user_data,
                                   loom_low_lower_context_t* context,
                                   const loom_op_t* source_op,
                                   loom_type_t source_type,
                                   loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_amdgpu_type_is_i1(source_type)) {
    return loom_amdgpu_make_scc_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_i32(source_type)) {
    return loom_amdgpu_make_sgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_i64(source_type)) {
    return loom_amdgpu_make_sgpr_range_type(context, 2, out_low_type);
  }
  if (loom_amdgpu_type_is_address_scalar(source_type)) {
    return loom_amdgpu_make_sgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_16bit_float(source_type)) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  const uint32_t vector_register_count =
      loom_amdgpu_vector_32bit_register_count(source_type);
  if (vector_register_count == 1) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  if (vector_register_count > 1) {
    return loom_amdgpu_make_register_type(context,
                                          LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                          vector_register_count, out_low_type);
  }
  const uint32_t mask_lane_count =
      loom_amdgpu_vector_i1_lane_count(source_type);
  if (mask_lane_count != 0) {
    return loom_amdgpu_make_sgpr_range_type(context, mask_lane_count * 2u,
                                            out_low_type);
  }
  uint32_t unused_payload_bit_count = 0;
  uint32_t packed_register_count = 0;
  if (loom_amdgpu_type_packed_16bit_float_storage(
          source_type, &unused_payload_bit_count, &packed_register_count)) {
    return loom_amdgpu_make_register_type(context,
                                          LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                          packed_register_count, out_low_type);
  }
  if (loom_amdgpu_type_packed_integer_storage(
          source_type, &unused_payload_bit_count, &packed_register_count)) {
    if (packed_register_count == 1) {
      return loom_amdgpu_make_vgpr_type(context, out_low_type);
    }
    return loom_amdgpu_make_register_type(context,
                                          LOOM_AMDGPU_REG_CLASS_ID_VGPR,
                                          packed_register_count, out_low_type);
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

iree_status_t loom_amdgpu_map_value(void* user_data,
                                    loom_low_lower_context_t* context,
                                    const loom_op_t* source_op,
                                    loom_value_id_t source_value_id,
                                    loom_type_t source_type,
                                    loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_amdgpu_type_is_i1(source_type)) {
    bool is_native_mask = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_is_native_i1_mask(
        context, source_value_id, &is_native_mask));
    if (is_native_mask) {
      return loom_amdgpu_make_sgpr_range_type(context, 2, out_low_type);
    }
  }
  if (loom_amdgpu_value_is_subgroup_lane_mask_result(
          loom_low_lower_context_module(context), source_value_id)) {
    return loom_amdgpu_make_sgpr_range_type(context, 2, out_low_type);
  }
  if (loom_amdgpu_type_is_f32(source_type)) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_16bit_float(source_type)) {
    return loom_amdgpu_make_vgpr_type(context, out_low_type);
  }
  if (loom_amdgpu_type_is_i64(source_type)) {
    bool prefers_vgpr = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_prefers_vgpr(
        context, source_value_id, &prefers_vgpr));
    if (prefers_vgpr) {
      return loom_amdgpu_make_vgpr_range_type(context, 2, out_low_type);
    }
    return loom_amdgpu_make_sgpr_range_type(context, 2, out_low_type);
  }
  if (loom_amdgpu_type_is_i32(source_type) ||
      loom_amdgpu_type_is_address_scalar(source_type)) {
    bool prefers_vgpr = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_context_value_prefers_vgpr(
        context, source_value_id, &prefers_vgpr));
    if (loom_amdgpu_offset_value_needs_64bit(
            loom_low_lower_context_fact_table(context),
            loom_low_lower_context_module(context), source_value_id,
            source_type)) {
      if (prefers_vgpr) {
        return loom_amdgpu_make_vgpr_range_type(context, 2, out_low_type);
      }
      return loom_amdgpu_make_sgpr_range_type(context, 2, out_low_type);
    }
    if (prefers_vgpr) {
      return loom_amdgpu_make_vgpr_type(context, out_low_type);
    }
  }
  return loom_amdgpu_map_type(user_data, context, source_op, source_type,
                              out_low_type);
}

static iree_status_t loom_amdgpu_map_contract_register(
    const loom_target_contract_query_environment_t* environment,
    uint16_t descriptor_register_class_id, uint32_t register_unit_count,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  if (descriptor_register_class_id >=
      environment->descriptor_set->reg_class_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU contract register class %" PRIu16
                            " is outside the selected descriptor set",
                            descriptor_register_class_id);
  }
  *out_mapped_value = loom_low_lower_rule_mapped_value_register(
      descriptor_register_class_id, register_unit_count);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_map_contract_value(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)user_data;
  (void)source_op;
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  const loom_type_t source_type =
      loom_module_value_type(environment->module, source_value_id);
  if (loom_amdgpu_type_is_i1(source_type) &&
      loom_amdgpu_source_value_is_native_i1_mask(
          environment->module, environment->fact_table,
          environment->view_regions, source_value_id)) {
    return loom_amdgpu_map_contract_register(
        environment, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, out_mapped_value);
  }
  if (loom_amdgpu_value_is_subgroup_lane_mask_result(environment->module,
                                                     source_value_id)) {
    return loom_amdgpu_map_contract_register(
        environment, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2, out_mapped_value);
  }
  if (loom_amdgpu_type_is_i1(source_type)) {
    return loom_amdgpu_map_contract_register(
        environment, LOOM_AMDGPU_REG_CLASS_ID_SCC, 1, out_mapped_value);
  }
  if (loom_amdgpu_type_is_f32(source_type)) {
    return loom_amdgpu_map_contract_register(
        environment, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, out_mapped_value);
  }
  if (loom_amdgpu_type_is_16bit_float(source_type)) {
    return loom_amdgpu_map_contract_register(
        environment, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, out_mapped_value);
  }
  if (loom_amdgpu_type_is_i64(source_type)) {
    const bool prefers_vgpr = loom_amdgpu_source_value_prefers_vgpr(
        environment->module, environment->fact_table, environment->view_regions,
        source_value_id);
    return loom_amdgpu_map_contract_register(
        environment,
        prefers_vgpr ? LOOM_AMDGPU_REG_CLASS_ID_VGPR
                     : LOOM_AMDGPU_REG_CLASS_ID_SGPR,
        2, out_mapped_value);
  }
  if (loom_amdgpu_type_is_i32(source_type) ||
      loom_amdgpu_type_is_address_scalar(source_type)) {
    const bool prefers_vgpr = loom_amdgpu_source_value_prefers_vgpr(
        environment->module, environment->fact_table, environment->view_regions,
        source_value_id);
    if (loom_amdgpu_offset_value_needs_64bit(environment->fact_table,
                                             environment->module,
                                             source_value_id, source_type)) {
      return loom_amdgpu_map_contract_register(
          environment,
          prefers_vgpr ? LOOM_AMDGPU_REG_CLASS_ID_VGPR
                       : LOOM_AMDGPU_REG_CLASS_ID_SGPR,
          2, out_mapped_value);
    }
    if (prefers_vgpr) {
      return loom_amdgpu_map_contract_register(
          environment, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1, out_mapped_value);
    }
  }
  if (loom_amdgpu_type_is_i32(source_type) ||
      loom_amdgpu_type_is_address_scalar(source_type)) {
    return loom_amdgpu_map_contract_register(
        environment, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1, out_mapped_value);
  }

  const uint32_t vector_register_count =
      loom_amdgpu_vector_32bit_register_count(source_type);
  if (vector_register_count != 0) {
    return loom_amdgpu_map_contract_register(
        environment, LOOM_AMDGPU_REG_CLASS_ID_VGPR, vector_register_count,
        out_mapped_value);
  }
  const uint32_t mask_lane_count =
      loom_amdgpu_vector_i1_lane_count(source_type);
  if (mask_lane_count != 0) {
    return loom_amdgpu_map_contract_register(
        environment, LOOM_AMDGPU_REG_CLASS_ID_SGPR, mask_lane_count * 2u,
        out_mapped_value);
  }

  uint32_t unused_payload_bit_count = 0;
  uint32_t packed_register_count = 0;
  if (loom_amdgpu_type_packed_16bit_float_storage(
          source_type, &unused_payload_bit_count, &packed_register_count)) {
    return loom_amdgpu_map_contract_register(
        environment, LOOM_AMDGPU_REG_CLASS_ID_VGPR, packed_register_count,
        out_mapped_value);
  }
  if (loom_amdgpu_type_packed_integer_storage(
          source_type, &unused_payload_bit_count, &packed_register_count)) {
    return loom_amdgpu_map_contract_register(
        environment, LOOM_AMDGPU_REG_CLASS_ID_VGPR, packed_register_count,
        out_mapped_value);
  }
  return iree_ok_status();
}

bool loom_amdgpu_module_value_as_exact_index_constant(
    const loom_module_t* module, loom_value_id_t value_id, int64_t* out_value) {
  *out_value = 0;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_index_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_index_constant_value(defining_op);
  if (attr.kind != LOOM_ATTR_I64) {
    return false;
  }
  *out_value = attr.i64;
  return true;
}

bool loom_amdgpu_module_value_as_i32_constant(const loom_module_t* module,
                                              loom_value_id_t value_id,
                                              int64_t* out_value) {
  *out_value = 0;
  if (!loom_amdgpu_type_is_i32(loom_module_value_type(module, value_id))) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_scalar_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_scalar_constant_value(defining_op);
  if (!loom_amdgpu_attr_is_i32_immediate(attr)) {
    return false;
  }
  *out_value = attr.i64;
  return true;
}

bool loom_amdgpu_module_value_as_f32_constant(const loom_module_t* module,
                                              loom_value_id_t value_id,
                                              uint32_t* out_bit_pattern) {
  *out_bit_pattern = 0;
  if (!loom_amdgpu_type_is_f32(loom_module_value_type(module, value_id))) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_scalar_constant_isa(defining_op)) {
    return false;
  }
  loom_attribute_t attr = loom_scalar_constant_value(defining_op);
  if (!loom_amdgpu_attr_is_f32_immediate(attr)) {
    return false;
  }
  *out_bit_pattern = loom_amdgpu_attr_f32_bit_pattern(attr);
  return true;
}

bool loom_amdgpu_value_facts_as_exact_non_negative_i64(loom_value_facts_t facts,
                                                       int64_t* out_value) {
  *out_value = 0;
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts) ||
      facts.range_lo < 0) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

bool loom_amdgpu_u32_is_power_of_two(uint32_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool loom_amdgpu_attr_is_i32_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_I64 && value.i64 >= INT32_MIN &&
         value.i64 <= INT32_MAX;
}

bool loom_amdgpu_attr_is_u32_address_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_I64 && value.i64 >= 0 &&
         value.i64 <= UINT32_MAX;
}

bool loom_amdgpu_attr_is_f32_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_F64;
}

uint32_t loom_amdgpu_attr_f32_bit_pattern(loom_attribute_t value) {
  const float f32_value = (float)loom_attr_as_f64(value);
  uint32_t bit_pattern = 0;
  memcpy(&bit_pattern, &f32_value, sizeof(bit_pattern));
  return bit_pattern;
}

bool loom_amdgpu_attr_is_16bit_float_immediate(loom_attribute_t value) {
  return value.kind == LOOM_ATTR_F64;
}

uint32_t loom_amdgpu_attr_16bit_float_bit_pattern(loom_scalar_type_t type,
                                                  loom_attribute_t value) {
  const float f32_value = (float)loom_attr_as_f64(value);
  switch (type) {
    case LOOM_SCALAR_TYPE_F16:
      return iree_math_f32_to_f16(f32_value);
    case LOOM_SCALAR_TYPE_BF16:
      return iree_math_f32_to_bf16(f32_value);
    default:
      IREE_ASSERT_UNREACHABLE("expected f16 or bf16");
      return 0;
  }
}

bool loom_amdgpu_value_as_i32_constant(loom_low_lower_context_t* context,
                                       loom_value_id_t value_id,
                                       int64_t* out_value) {
  return loom_amdgpu_module_value_as_i32_constant(
      loom_low_lower_context_module(context), value_id, out_value);
}

bool loom_amdgpu_value_as_f32_constant(loom_low_lower_context_t* context,
                                       loom_value_id_t value_id,
                                       uint32_t* out_bit_pattern) {
  return loom_amdgpu_module_value_as_f32_constant(
      loom_low_lower_context_module(context), value_id, out_bit_pattern);
}

bool loom_amdgpu_value_as_address_constant(loom_low_lower_context_t* context,
                                           loom_value_id_t value_id,
                                           int64_t* out_value) {
  *out_value = 0;
  if (!loom_amdgpu_value_is_address_scalar(context, value_id)) {
    return false;
  }
  return loom_amdgpu_module_value_as_exact_index_constant(
      loom_low_lower_context_module(context), value_id, out_value);
}

bool loom_amdgpu_value_can_materialize_as_vgpr_i32(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_i32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_can_materialize_as_vgpr_f32(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_type_is_f32(
      loom_module_value_type(loom_low_lower_context_module(context), value_id));
}

bool loom_amdgpu_value_can_materialize_as_vgpr_address(
    loom_low_lower_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_value_is_address_scalar(context, value_id);
}
