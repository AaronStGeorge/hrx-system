// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower/internal.h"

iree_status_t loom_amdgpu_intern(loom_low_lower_context_t* context,
                                 iree_string_view_t string,
                                 loom_string_id_t* out_string_id) {
  IREE_ASSERT_ARGUMENT(out_string_id);
  return loom_module_intern_string(loom_low_lower_context_module(context),
                                   string, out_string_id);
}

iree_status_t loom_amdgpu_append_i64_attr(loom_low_lower_context_t* context,
                                          iree_string_view_t name,
                                          int64_t value,
                                          loom_named_attr_t* attrs,
                                          iree_host_size_t attr_capacity,
                                          iree_host_size_t* inout_attr_count) {
  IREE_ASSERT_ARGUMENT(attrs);
  IREE_ASSERT_ARGUMENT(inout_attr_count);
  if (*inout_attr_count >= attr_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU low attr capacity exceeded");
  }
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, name, &name_id));
  attrs[*inout_attr_count] = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  *inout_attr_count += 1;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_sgpr_byte_offset(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t dynamic_index, int64_t dynamic_index_byte_stride,
    uint32_t dynamic_index_byte_shift, uint32_t static_byte_offset,
    loom_value_id_t* out_low_offset) {
  IREE_ASSERT_ARGUMENT(out_low_offset);
  *out_low_offset = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  if (dynamic_index == LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32,
        static_byte_offset, sgpr_type, out_low_offset);
  }

  loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, dynamic_index, &low_index));
  loom_value_id_t low_dynamic_offset = low_index;
  if (dynamic_index_byte_stride != 1) {
    IREE_ASSERT(dynamic_index_byte_stride >= 0 &&
                dynamic_index_byte_stride <= UINT32_MAX);
    IREE_ASSERT(dynamic_index_byte_shift !=
                LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE);
    loom_value_id_t low_shift = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32,
        dynamic_index_byte_shift, sgpr_type, &low_shift));
    loom_value_id_t shift_operands[] = {low_index, low_shift};
    loom_op_t* low_shift_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHL_B32,
        shift_operands, IREE_ARRAYSIZE(shift_operands),
        loom_make_named_attr_slice(NULL, 0), &sgpr_type, 1, &low_shift_op));
    low_dynamic_offset =
        loom_value_slice_get(loom_low_op_results(low_shift_op), 0);
  }

  if (static_byte_offset == 0) {
    *out_low_offset = low_dynamic_offset;
    return iree_ok_status();
  }

  loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32,
      static_byte_offset, sgpr_type, &low_static_offset));
  loom_value_id_t add_operands[] = {low_dynamic_offset, low_static_offset};
  loom_op_t* low_add_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32, add_operands,
      IREE_ARRAYSIZE(add_operands), loom_make_named_attr_slice(NULL, 0),
      &sgpr_type, 1, &low_add_op));
  *out_low_offset = loom_value_slice_get(loom_low_op_results(low_add_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_sgpr_byte_offset_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_dynamic_index_kind_t* dynamic_term_kinds,
    uint32_t static_byte_offset, loom_value_id_t* out_low_offset) {
  IREE_ASSERT_ARGUMENT(source);
  IREE_ASSERT_ARGUMENT(dynamic_term_kinds);
  IREE_ASSERT_ARGUMENT(out_low_offset);
  *out_low_offset = LOOM_VALUE_ID_INVALID;

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));

  loom_value_id_t low_accumulator = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    switch (dynamic_term_kinds[i]) {
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET:
        break;
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR:
        continue;
      case LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE:
        IREE_ASSERT_UNREACHABLE();
        continue;
    }
    const loom_low_source_memory_dynamic_term_t* term =
        &source->dynamic_terms[i];
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_byte_offset(
        context, source_op, term->index, term->byte_stride, term->byte_shift,
        /*static_byte_offset=*/0, &low_term));
    if (low_accumulator == LOOM_VALUE_ID_INVALID) {
      low_accumulator = low_term;
      continue;
    }
    loom_value_id_t add_operands[] = {low_accumulator, low_term};
    loom_op_t* low_add_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32, add_operands,
        IREE_ARRAYSIZE(add_operands), loom_make_named_attr_slice(NULL, 0),
        &sgpr_type, 1, &low_add_op));
    low_accumulator = loom_value_slice_get(loom_low_op_results(low_add_op), 0);
  }

  if (low_accumulator == LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_emit_sgpr_byte_offset(
        context, source_op, LOOM_VALUE_ID_INVALID,
        /*dynamic_index_byte_stride=*/1,
        LOOM_LOW_SOURCE_MEMORY_ACCESS_BYTE_SHIFT_NONE, static_byte_offset,
        out_low_offset);
  }
  if (static_byte_offset == 0) {
    *out_low_offset = low_accumulator;
    return iree_ok_status();
  }

  loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32,
      static_byte_offset, sgpr_type, &low_static_offset));
  loom_value_id_t add_operands[] = {low_accumulator, low_static_offset};
  loom_op_t* low_add_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32, add_operands,
      IREE_ARRAYSIZE(add_operands), loom_make_named_attr_slice(NULL, 0),
      &sgpr_type, 1, &low_add_op));
  *out_low_offset = loom_value_slice_get(loom_low_op_results(low_add_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_result_type(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_value_id_t source_result,
                                          loom_type_t* out_low_type) {
  IREE_ASSERT_ARGUMENT(out_low_type);
  *out_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_result, out_low_type));
  if (!loom_type_is_register(*out_low_type)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU source type did not map to a register");
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_low_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, const loom_value_id_t* operands,
    iree_host_size_t operand_count, loom_named_attr_slice_t attrs,
    const loom_type_t* result_types, iree_host_size_t result_count,
    loom_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(out_op);
  *out_op = NULL;
  return loom_low_lower_emit_descriptor_op(
      context, descriptor_id, operands, operand_count, attrs, result_types,
      result_count, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, out_op);
}

iree_status_t loom_amdgpu_emit_explicit_packet_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_explicit_packet_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  if (plan->descriptor_id == LOOM_LOW_DESCRIPTOR_ID_NONE) {
    return iree_ok_status();
  }

  loom_named_attr_t attrs[LOOM_AMDGPU_EXPLICIT_PACKET_IMMEDIATE_CAPACITY] = {0};
  for (iree_host_size_t i = 0; i < plan->immediate_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_intern(context, plan->immediates[i].name,
                                            &attrs[i].name_id));
    attrs[i].value = loom_attr_i64(plan->immediates[i].value);
  }

  loom_op_t* low_op = NULL;
  return loom_amdgpu_emit_low_op(
      context, source_op, plan->descriptor_id, /*operands=*/NULL,
      /*operand_count=*/0,
      loom_make_named_attr_slice(attrs, plan->immediate_count),
      /*result_types=*/NULL, /*result_count=*/0, &low_op);
}

iree_status_t loom_amdgpu_emit_const_u32(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         uint64_t descriptor_id, uint32_t value,
                                         loom_type_t result_type,
                                         loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_string_id_t value_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_intern(context, IREE_SV("imm32"), &value_name_id));
  loom_named_attr_t attrs[] = {
      {
          .name_id = value_name_id,
          .value = loom_attr_i64(value),
      },
  };
  loom_op_t* low_const = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_const(
      context, descriptor_id,
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), result_type,
      source_op->location, &low_const));
  *out_value_id = loom_low_const_result(low_const);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_m0_u32(loom_low_lower_context_t* context,
                                      const loom_op_t* source_op,
                                      uint64_t consumer_descriptor_id,
                                      uint32_t value,
                                      loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32, value, sgpr_type,
      &low_value));

  loom_type_t m0_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_descriptor_implicit_resource_type(
      context, consumer_descriptor_id, &m0_type));
  loom_value_id_t operands[] = {low_value};
  loom_op_t* low_m0_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_S_MOV_B32_M0, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &m0_type,
      1, &low_m0_op));
  *out_value_id = loom_value_slice_get(loom_low_op_results(low_m0_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_vgpr_b32_copy(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_value_id_t low_source,
                                             loom_value_id_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_value_id_t operands[] = {low_source};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32_COPY, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &vgpr_type,
      1, &low_op));
  *out_value = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_emit_vgpr_binary(
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

iree_status_t loom_amdgpu_emit_vgpr_shift(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          uint64_t descriptor_id,
                                          uint32_t shift, loom_value_id_t value,
                                          loom_type_t lane_type,
                                          loom_value_id_t* out_value) {
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

iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_i32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  IREE_ASSERT_ARGUMENT(out_low_value);
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr) {
    *out_low_value = low_value;
    return iree_ok_status();
  }

  int64_t value = 0;
  if (loom_amdgpu_value_as_i32_constant(context, source_value, &value)) {
    loom_type_t vgpr_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
    return loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32,
        (uint32_t)(int32_t)value, vgpr_type, out_low_value);
  }

  bool is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &is_sgpr));
  if (is_sgpr && loom_type_register_unit_count(low_type) == 1) {
    return loom_amdgpu_emit_vgpr_b32_copy(context, source_op, low_value,
                                          out_low_value);
  }

  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AMDGPU i32 value cannot materialize as a VGPR operand");
}

iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value) {
  IREE_ASSERT_ARGUMENT(out_low_value);
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr) {
    *out_low_value = low_value;
    return iree_ok_status();
  }

  int64_t value = 0;
  if (!loom_amdgpu_value_as_address_constant(context, source_value, &value) ||
      value < 0 || value > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU address value cannot materialize as a VGPR operand");
  }
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_ID_V_MOV_B32,
                                    (uint32_t)value, vgpr_type, out_low_value);
}

iree_status_t loom_amdgpu_emit_low_slice(loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_value_id_t low_source,
                                         uint32_t offset,
                                         loom_type_t result_type,
                                         loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(
      loom_low_lower_context_builder(context), low_source, offset, result_type,
      source_op->location, &slice_op));
  *out_value_id = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

bool loom_amdgpu_low_legality_bundle_is_amdgpu(
    const loom_target_bundle_t* bundle) {
  return bundle != NULL && bundle->config != NULL &&
         iree_string_view_starts_with(bundle->config->contract_set_key,
                                      IREE_SV("amdgpu."));
}
