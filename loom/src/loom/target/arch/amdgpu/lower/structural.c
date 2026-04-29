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

static bool loom_amdgpu_vector_bitcast_storage_shape(
    loom_type_t type, uint32_t* out_payload_bit_count,
    uint32_t* out_register_count) {
  IREE_ASSERT_ARGUMENT(out_payload_bit_count);
  IREE_ASSERT_ARGUMENT(out_register_count);
  *out_payload_bit_count = 0;
  *out_register_count = 0;

  const uint32_t lane_count = loom_amdgpu_vector_32bit_lane_count(type);
  if (lane_count != 0) {
    *out_payload_bit_count = 32u * lane_count;
    *out_register_count = lane_count;
    return true;
  }

  return loom_amdgpu_type_packed_integer_storage(type, out_payload_bit_count,
                                                 out_register_count) ||
         loom_amdgpu_type_packed_16bit_float_storage(
             type, out_payload_bit_count, out_register_count);
}

static bool loom_amdgpu_vector_bitcast_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_vector_bitcast_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_amdgpu_vector_bitcast_plan_t){0};
  if (!loom_vector_bitcast_isa(source_op)) {
    return false;
  }

  out_plan->source = loom_vector_bitcast_input(source_op);
  out_plan->result = loom_vector_bitcast_result(source_op);
  const loom_type_t input_type =
      loom_module_value_type(module, out_plan->source);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);

  uint32_t input_payload_bit_count = 0;
  uint32_t input_register_count = 0;
  uint32_t result_payload_bit_count = 0;
  uint32_t result_register_count = 0;
  return loom_amdgpu_vector_bitcast_storage_shape(
             input_type, &input_payload_bit_count, &input_register_count) &&
         loom_amdgpu_vector_bitcast_storage_shape(
             result_type, &result_payload_bit_count, &result_register_count) &&
         input_payload_bit_count == result_payload_bit_count &&
         input_register_count == result_register_count;
}

static bool loom_amdgpu_static_rank1_slice_shape(loom_type_t source_type,
                                                 loom_type_t result_type,
                                                 int64_t* out_lane_count) {
  IREE_ASSERT_ARGUMENT(out_lane_count);
  *out_lane_count = 0;
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      loom_type_rank(source_type) != 1 || loom_type_rank(result_type) != 1 ||
      !loom_type_is_all_static(source_type) ||
      !loom_type_is_all_static(result_type) ||
      !loom_type_element_type_equals(source_type, result_type)) {
    return false;
  }
  const int64_t source_lane_count =
      loom_type_dim_static_size_at(source_type, 0);
  const int64_t result_lane_count =
      loom_type_dim_static_size_at(result_type, 0);
  if (source_lane_count < 1 || result_lane_count < 1) {
    return false;
  }
  *out_lane_count = result_lane_count;
  return true;
}

static bool loom_amdgpu_packed_register_bit_slice_window_is_supported(
    uint32_t source_bit_offset, uint32_t result_payload_bit_count,
    uint32_t result_register_count) {
  for (uint32_t register_index = 0; register_index < result_register_count;
       ++register_index) {
    const uint32_t result_bit_offset = register_index * 32u;
    const uint32_t result_remaining_bit_count =
        result_payload_bit_count - result_bit_offset;
    const uint32_t needed_bit_count = iree_min(result_remaining_bit_count, 32u);
    const uint32_t source_register_bit_offset =
        (source_bit_offset + result_bit_offset) & 31u;
    if (source_register_bit_offset + needed_bit_count > 32u) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_vector_slice_plan_from_op(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_vector_slice_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_amdgpu_vector_slice_plan_t){0};
  if (!loom_vector_slice_isa(source_op) ||
      loom_vector_slice_offsets(source_op).count != 0) {
    return false;
  }
  loom_attribute_t static_offsets = loom_vector_slice_static_offsets(source_op);
  if (static_offsets.kind != LOOM_ATTR_I64_ARRAY || static_offsets.count != 1 ||
      static_offsets.i64_array[0] < 0 ||
      static_offsets.i64_array[0] == INT64_MIN ||
      static_offsets.i64_array[0] > UINT32_MAX) {
    return false;
  }

  out_plan->source = loom_vector_slice_source(source_op);
  out_plan->result = loom_vector_slice_result(source_op);
  const loom_type_t source_type =
      loom_module_value_type(module, out_plan->source);
  const loom_type_t result_type =
      loom_module_value_type(module, out_plan->result);

  const uint32_t lane_offset = (uint32_t)static_offsets.i64_array[0];
  int64_t lane_count_i64 = 0;
  if (!loom_amdgpu_static_rank1_slice_shape(source_type, result_type,
                                            &lane_count_i64) ||
      lane_count_i64 > UINT32_MAX) {
    return false;
  }
  const uint32_t lane_count = (uint32_t)lane_count_i64;
  if ((uint64_t)lane_offset + lane_count >
      (uint64_t)loom_type_dim_static_size_at(source_type, 0)) {
    return false;
  }

  const uint32_t source_32bit_lane_count =
      loom_amdgpu_vector_32bit_lane_count(source_type);
  const uint32_t result_32bit_lane_count =
      loom_amdgpu_vector_32bit_lane_count(result_type);
  if (source_32bit_lane_count != 0 && result_32bit_lane_count != 0) {
    out_plan->source_register_count = source_32bit_lane_count;
    out_plan->result_register_count = result_32bit_lane_count;
    out_plan->kind = LOOM_AMDGPU_VECTOR_SLICE_KIND_32BIT_LANES;
  } else {
    uint32_t source_payload_bit_count = 0;
    uint32_t result_payload_bit_count = 0;
    uint32_t source_register_count = 0;
    uint32_t result_register_count = 0;
    if ((!loom_amdgpu_type_packed_integer_storage(
             source_type, &source_payload_bit_count, &source_register_count) ||
         !loom_amdgpu_type_packed_integer_storage(
             result_type, &result_payload_bit_count, &result_register_count)) &&
        (!loom_amdgpu_type_packed_16bit_float_storage(
             source_type, &source_payload_bit_count, &source_register_count) ||
         !loom_amdgpu_type_packed_16bit_float_storage(
             result_type, &result_payload_bit_count, &result_register_count))) {
      return false;
    }
    const int32_t element_bit_count =
        loom_scalar_type_bitwidth(loom_type_element_type(source_type));
    if (element_bit_count <= 0) {
      return false;
    }
    const uint32_t source_bit_offset =
        lane_offset * (uint32_t)element_bit_count;
    if (!loom_amdgpu_packed_register_bit_slice_window_is_supported(
            source_bit_offset, result_payload_bit_count,
            result_register_count)) {
      return false;
    }
    out_plan->source_register_count = source_register_count;
    out_plan->result_register_count = result_register_count;
    out_plan->element_bit_count = (uint32_t)element_bit_count;
    out_plan->kind = LOOM_AMDGPU_VECTOR_SLICE_KIND_PACKED_REGISTER_BITS;
  }

  out_plan->lane_offset = lane_offset;
  out_plan->lane_count = lane_count;
  return true;
}

bool loom_amdgpu_select_vector_bitcast_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_bitcast_plan_t* out_plan) {
  return loom_amdgpu_vector_bitcast_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan);
}

iree_status_t loom_amdgpu_lower_vector_bitcast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_bitcast_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  loom_value_id_t low_input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_input));

  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, plan->result, &result_low_type));
  const loom_type_t input_low_type =
      loom_module_value_type(loom_low_lower_context_module(context), low_input);
  IREE_ASSERT(loom_type_equal(input_low_type, result_low_type));

  return loom_low_lower_bind_value(context, plan->result, low_input);
}

bool loom_amdgpu_select_vector_slice_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_vector_slice_plan_t* out_plan) {
  return loom_amdgpu_vector_slice_plan_from_op(
      loom_low_lower_context_module(context), source_op, out_plan);
}

static iree_status_t loom_amdgpu_lower_vector_slice_32bit_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_slice_plan_t* select, loom_value_id_t low_source,
    loom_type_t lane_type, loom_value_id_t* low_registers) {
  IREE_ASSERT_ARGUMENT(low_registers);
  for (uint32_t i = 0; i < select->result_register_count; ++i) {
    const uint32_t source_lane = select->lane_offset + i;
    if (select->source_register_count == 1) {
      low_registers[i] = low_source;
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_low_slice(context, source_op, low_source, source_lane,
                                   lane_type, &low_registers[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_vector_slice_packed_register_bits(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_slice_plan_t* select, loom_value_id_t low_source,
    loom_type_t lane_type, loom_value_id_t* low_registers) {
  IREE_ASSERT_ARGUMENT(low_registers);
  const uint32_t source_bit_offset =
      select->lane_offset * select->element_bit_count;
  for (uint32_t i = 0; i < select->result_register_count; ++i) {
    const uint32_t register_source_bit_offset = source_bit_offset + i * 32u;
    const uint32_t source_register_index = register_source_bit_offset / 32u;
    const uint32_t source_register_bit_offset =
        register_source_bit_offset & 31u;
    loom_value_id_t source_register = low_source;
    if (select->source_register_count != 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, source_register_index, lane_type,
          &source_register));
    }
    if (source_register_bit_offset == 0) {
      low_registers[i] = source_register;
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32_LIT,
        source_register_bit_offset, source_register, lane_type,
        &low_registers[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_slice(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_vector_slice_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_value_id_t low_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES];
  if (plan->kind == LOOM_AMDGPU_VECTOR_SLICE_KIND_32BIT_LANES) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_slice_32bit_lanes(
        context, source_op, plan, low_source, lane_type, low_registers));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lower_vector_slice_packed_register_bits(
        context, source_op, plan, low_source, lane_type, low_registers));
  }

  if (plan->result_register_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, low_registers[0]);
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context),
                            low_registers, plan->result_register_count,
                            result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->result,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_low_legality_verify_vector_structural(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  const loom_module_t* module = loom_target_low_legality_module(context);
  switch (op->kind) {
    case LOOM_OP_VECTOR_BITCAST: {
      loom_amdgpu_vector_bitcast_plan_t plan;
      if (loom_amdgpu_vector_bitcast_plan_from_op(module, op, &plan)) {
        return iree_ok_status();
      }
      return loom_target_low_legality_reject(
          context, provider, op, IREE_SV("shape"), loom_op_name(module, op),
          IREE_SV("AMDGPU vector.bitcast requires source and result payloads "
                  "with identical 32-bit register storage"));
    }
    case LOOM_OP_VECTOR_SLICE: {
      loom_amdgpu_vector_slice_plan_t unused_plan = {0};
      if (loom_amdgpu_vector_slice_plan_from_op(module, op, &unused_plan)) {
        return iree_ok_status();
      }
      return loom_target_low_legality_reject(
          context, provider, op, IREE_SV("shape"), loom_op_name(module, op),
          IREE_SV("AMDGPU vector.slice currently requires static rank-1 "
                  "32-bit lane slices or packed bit windows contained within "
                  "each selected 32-bit register"));
    }
    default:
      *out_handled = false;
      return iree_ok_status();
  }
}
