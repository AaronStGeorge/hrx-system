// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"

typedef struct loom_amdgpu_bitfield_extract_t {
  // Source vector value containing i32 lanes.
  loom_value_id_t source;
  // Result vector value containing i32 lanes.
  loom_value_id_t result;
  // Least-significant source bit of the extracted field.
  uint32_t offset;
  // Number of bits extracted from each lane.
  uint32_t width;
  // True when the extracted field is sign-extended.
  bool is_signed;
} loom_amdgpu_bitfield_extract_t;

typedef struct loom_amdgpu_bitfield_insert_t {
  // Field vector value containing i32 lanes.
  loom_value_id_t field;
  // Base vector value containing i32 lanes.
  loom_value_id_t base;
  // Result vector value containing i32 lanes.
  loom_value_id_t result;
  // Least-significant destination bit of the inserted field.
  uint32_t offset;
  // Number of low field bits inserted into each base lane.
  uint32_t width;
} loom_amdgpu_bitfield_insert_t;

static bool loom_amdgpu_bitfield_extract_select(
    const loom_op_t* source_op, loom_amdgpu_bitfield_extract_t* out_select) {
  IREE_ASSERT_ARGUMENT(out_select);
  *out_select = (loom_amdgpu_bitfield_extract_t){0};

  int64_t offset = 0;
  int64_t width = 0;
  if (loom_vector_bitfield_extractu_isa(source_op)) {
    offset = loom_vector_bitfield_extractu_offset(source_op);
    width = loom_vector_bitfield_extractu_width(source_op);
    out_select->source = loom_vector_bitfield_extractu_source(source_op);
    out_select->result = loom_vector_bitfield_extractu_result(source_op);
    out_select->is_signed = false;
  } else if (loom_vector_bitfield_extracts_isa(source_op)) {
    offset = loom_vector_bitfield_extracts_offset(source_op);
    width = loom_vector_bitfield_extracts_width(source_op);
    out_select->source = loom_vector_bitfield_extracts_source(source_op);
    out_select->result = loom_vector_bitfield_extracts_result(source_op);
    out_select->is_signed = true;
  } else {
    return false;
  }

  if (offset < 0 || offset > 31 || width < 1 || width > 32 ||
      offset + width > 32) {
    return false;
  }
  out_select->offset = (uint32_t)offset;
  out_select->width = (uint32_t)width;
  return true;
}

static bool loom_amdgpu_bitfield_insert_select(
    const loom_op_t* source_op, loom_amdgpu_bitfield_insert_t* out_select) {
  IREE_ASSERT_ARGUMENT(out_select);
  *out_select = (loom_amdgpu_bitfield_insert_t){0};
  if (!loom_vector_bitfield_insert_isa(source_op)) {
    return false;
  }

  const int64_t offset = loom_vector_bitfield_insert_offset(source_op);
  const int64_t width = loom_vector_bitfield_insert_width(source_op);
  if (offset < 0 || offset > 31 || width < 1 || width > 32 ||
      offset + width > 32) {
    return false;
  }
  out_select->field = loom_vector_bitfield_insert_field(source_op);
  out_select->base = loom_vector_bitfield_insert_base(source_op);
  out_select->result = loom_vector_bitfield_insert_result(source_op);
  out_select->offset = (uint32_t)offset;
  out_select->width = (uint32_t)width;
  return true;
}

bool loom_amdgpu_can_lower_vector_bitfield_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_amdgpu_bitfield_extract_t select = {0};
  if (!loom_amdgpu_bitfield_extract_select(source_op, &select)) {
    return false;
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type = loom_module_value_type(module, select.source);
  const loom_type_t result_type = loom_module_value_type(module, select.result);
  const uint32_t source_lane_count =
      loom_amdgpu_vector_i32_lane_count(source_type);
  return source_lane_count != 0 &&
         loom_amdgpu_vector_i32_lane_count(result_type) == source_lane_count;
}

bool loom_amdgpu_can_lower_vector_bitfield_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_amdgpu_bitfield_insert_t select = {0};
  if (!loom_amdgpu_bitfield_insert_select(source_op, &select)) {
    return false;
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t field_type = loom_module_value_type(module, select.field);
  const loom_type_t base_type = loom_module_value_type(module, select.base);
  const loom_type_t result_type = loom_module_value_type(module, select.result);
  const uint32_t base_lane_count = loom_amdgpu_vector_i32_lane_count(base_type);
  return base_lane_count != 0 &&
         loom_amdgpu_vector_i32_lane_count(field_type) == base_lane_count &&
         loom_amdgpu_vector_i32_lane_count(result_type) == base_lane_count;
}

static uint32_t loom_amdgpu_bitfield_low_mask(uint32_t width) {
  return width == 32 ? UINT32_MAX : (UINT32_C(1) << width) - 1u;
}

static iree_status_t loom_amdgpu_emit_vgpr_binary(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t lane_type, loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[] = {
      lhs,
      rhs,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_id, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), &lane_type, 1, &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vgpr_shift(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, uint32_t shift, loom_value_id_t value,
    loom_type_t lane_type, loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;
  if (shift == 0) {
    *out_value = value;
    return iree_ok_status();
  }

  loom_value_id_t shift_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32, shift, lane_type,
      &shift_value));
  return loom_amdgpu_emit_vgpr_binary(context, source_op, descriptor_id,
                                      shift_value, value, lane_type, out_value);
}

static iree_status_t loom_amdgpu_emit_extractu_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_extract_t* select, loom_value_id_t source_lane,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  IREE_ASSERT_ARGUMENT(out_lane);
  *out_lane = LOOM_VALUE_ID_INVALID;

  loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32,
      select->offset, source_lane, lane_type, &shifted));

  if (select->width == 32) {
    *out_lane = shifted;
    return iree_ok_status();
  }

  const uint32_t mask = loom_amdgpu_bitfield_low_mask(select->width);
  loom_value_id_t mask_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32, mask, lane_type,
      &mask_value));
  return loom_amdgpu_emit_vgpr_binary(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32,
                                      shifted, mask_value, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_extracts_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_extract_t* select, loom_value_id_t source_lane,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  IREE_ASSERT_ARGUMENT(out_lane);
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (select->offset == 0 && select->width == 32) {
    *out_lane = source_lane;
    return iree_ok_status();
  }

  loom_value_id_t shifted_left = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32,
      32u - select->offset - select->width, source_lane, lane_type,
      &shifted_left));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_ASHRREV_I32,
      32u - select->width, shifted_left, lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_bitfield_extract_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_extract_t* select, loom_value_id_t source_lane,
    loom_type_t lane_type, loom_value_id_t* out_lane) {
  if (select->is_signed) {
    return loom_amdgpu_emit_extracts_lane(context, source_op, select,
                                          source_lane, lane_type, out_lane);
  }
  return loom_amdgpu_emit_extractu_lane(context, source_op, select, source_lane,
                                        lane_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_insert_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bitfield_insert_t* select, loom_value_id_t field_lane,
    loom_value_id_t base_lane, loom_type_t lane_type,
    loom_value_id_t* out_lane) {
  IREE_ASSERT_ARGUMENT(out_lane);
  *out_lane = LOOM_VALUE_ID_INVALID;
  if (select->offset == 0 && select->width == 32) {
    *out_lane = field_lane;
    return iree_ok_status();
  }

  loom_value_id_t field_low_bits = field_lane;
  if (select->width < 32) {
    const uint32_t field_mask = loom_amdgpu_bitfield_low_mask(select->width);
    loom_value_id_t field_mask_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32, field_mask,
        lane_type, &field_mask_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32, field_lane,
        field_mask_value, lane_type, &field_low_bits));
  }

  loom_value_id_t shifted_field = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32,
      select->offset, field_low_bits, lane_type, &shifted_field));

  const uint32_t target_mask = loom_amdgpu_bitfield_low_mask(select->width)
                               << select->offset;
  const uint32_t clear_mask = ~target_mask;
  if (clear_mask == 0) {
    *out_lane = shifted_field;
    return iree_ok_status();
  }

  loom_value_id_t clear_mask_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32, clear_mask,
      lane_type, &clear_mask_value));
  loom_value_id_t cleared_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32, base_lane,
      clear_mask_value, lane_type, &cleared_base));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32, cleared_base,
      shifted_field, lane_type, out_lane);
}

iree_status_t loom_amdgpu_lower_vector_bitfield_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!loom_amdgpu_can_lower_vector_bitfield_extract(context, source_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.bitfield.extract");
  }

  loom_amdgpu_bitfield_extract_t select = {0};
  if (!loom_amdgpu_bitfield_extract_select(source_op, &select)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.bitfield.extract");
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type = loom_module_value_type(module, select.source);
  const uint32_t lane_count = loom_amdgpu_vector_i32_lane_count(source_type);
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.bitfield.extract lane count");
  }

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, select.source, &low_source));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  if (lane_count == 1) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_bitfield_extract_lane(
        context, source_op, &select, low_source, lane_type, &low_result));
    return loom_low_lower_bind_value(context, select.result, low_result);
  }

  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t source_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_source, i, lane_type, &source_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_bitfield_extract_lane(
        context, source_op, &select, source_lane, lane_type, &lane_results[i]));
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, select.result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, select.result,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_vector_bitfield_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!loom_amdgpu_can_lower_vector_bitfield_insert(context, source_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.bitfield.insert");
  }

  loom_amdgpu_bitfield_insert_t select = {0};
  if (!loom_amdgpu_bitfield_insert_select(source_op, &select)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.bitfield.insert");
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t base_type = loom_module_value_type(module, select.base);
  const uint32_t lane_count = loom_amdgpu_vector_i32_lane_count(base_type);
  if (lane_count == 0 || lane_count > LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "preflight accepted unsupported AMDGPU "
                            "vector.bitfield.insert lane count");
  }

  loom_value_id_t low_field = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, select.field, &low_field));
  loom_value_id_t low_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, select.base, &low_base));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  if (lane_count == 1) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_insert_lane(context, source_op, &select, low_field,
                                     low_base, lane_type, &low_result));
    return loom_low_lower_bind_value(context, select.result, low_result);
  }

  loom_value_id_t lane_results[LOOM_AMDGPU_MAX_VECTOR_32BIT_LANES];
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_value_id_t field_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_field, i, lane_type, &field_lane));
    loom_value_id_t base_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_base, i, lane_type, &base_lane));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_insert_lane(context, source_op, &select, field_lane,
                                     base_lane, lane_type, &lane_results[i]));
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, select.result, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), lane_results, lane_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, select.result,
                                   loom_low_concat_result(concat_op));
}
