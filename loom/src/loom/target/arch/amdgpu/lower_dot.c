// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/context.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"

typedef uint32_t loom_amdgpu_dot_rejection_flags_t;

#define LOOM_AMDGPU_DOT_REJECTION_KIND ((uint32_t)1u << 0)
#define LOOM_AMDGPU_DOT_REJECTION_LHS_SHAPE ((uint32_t)1u << 1)
#define LOOM_AMDGPU_DOT_REJECTION_RHS_SHAPE ((uint32_t)1u << 2)
#define LOOM_AMDGPU_DOT_REJECTION_ACC_SHAPE ((uint32_t)1u << 3)
#define LOOM_AMDGPU_DOT_REJECTION_RESULT_SHAPE ((uint32_t)1u << 4)
#define LOOM_AMDGPU_DOT_REJECTION_DESCRIPTOR_MISSING ((uint32_t)1u << 5)

typedef struct loom_amdgpu_dot_diagnostic_t {
  // Rejection bits explaining why a source vector dot is not legal here.
  loom_amdgpu_dot_rejection_flags_t rejection_bits;
} loom_amdgpu_dot_diagnostic_t;

static bool loom_amdgpu_descriptor_set_has_descriptor(
    const loom_low_descriptor_set_t* descriptor_set, uint64_t descriptor_id) {
  return descriptor_set != NULL &&
         loom_low_descriptor_set_lookup_descriptor_by_id(
             descriptor_set, descriptor_id) != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
}

static bool loom_amdgpu_dot4i_select(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    uint32_t* out_group_count, loom_amdgpu_dot_diagnostic_t* out_diagnostic) {
  IREE_ASSERT_ARGUMENT(out_group_count);
  IREE_ASSERT_ARGUMENT(out_diagnostic);
  *out_group_count = 0;
  *out_diagnostic = (loom_amdgpu_dot_diagnostic_t){0};
  if (!loom_vector_dot4i_isa(source_op)) {
    out_diagnostic->rejection_bits |= LOOM_AMDGPU_DOT_REJECTION_KIND;
    return false;
  }
  if (loom_vector_dot4i_kind(source_op) != LOOM_VECTOR_DOT4I_KIND_S8S8) {
    out_diagnostic->rejection_bits |= LOOM_AMDGPU_DOT_REJECTION_KIND;
    return false;
  }

  const loom_type_t lhs_type =
      loom_module_value_type(module, loom_vector_dot4i_lhs(source_op));
  const loom_type_t rhs_type =
      loom_module_value_type(module, loom_vector_dot4i_rhs(source_op));
  const loom_type_t acc_type =
      loom_module_value_type(module, loom_vector_dot4i_acc(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_dot4i_result(source_op));

  const uint32_t lhs_lane_count = loom_amdgpu_vector_i8_lane_count(lhs_type);
  if (lhs_lane_count == 0 || (lhs_lane_count % 4) != 0) {
    out_diagnostic->rejection_bits |= LOOM_AMDGPU_DOT_REJECTION_LHS_SHAPE;
    return false;
  }
  const uint32_t group_count = lhs_lane_count / 4;
  if (loom_amdgpu_vector_i8_lane_count(rhs_type) != lhs_lane_count) {
    out_diagnostic->rejection_bits |= LOOM_AMDGPU_DOT_REJECTION_RHS_SHAPE;
    return false;
  }
  if (loom_amdgpu_vector_i32_lane_count(acc_type) != group_count) {
    out_diagnostic->rejection_bits |= LOOM_AMDGPU_DOT_REJECTION_ACC_SHAPE;
    return false;
  }
  if (loom_amdgpu_vector_i32_lane_count(result_type) != group_count) {
    out_diagnostic->rejection_bits |= LOOM_AMDGPU_DOT_REJECTION_RESULT_SHAPE;
    return false;
  }
  if (!loom_amdgpu_descriptor_set_has_descriptor(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT4_I32_I8)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_DOT_REJECTION_DESCRIPTOR_MISSING;
    return false;
  }

  *out_group_count = group_count;
  return true;
}

bool loom_amdgpu_can_lower_vector_dot4i(loom_low_lower_context_t* context,
                                        const loom_op_t* source_op) {
  uint32_t unused_group_count = 0;
  loom_amdgpu_dot_diagnostic_t diagnostic = {0};
  return loom_amdgpu_dot4i_select(
      loom_low_lower_context_module(context),
      loom_low_lower_context_descriptor_set(context), source_op,
      &unused_group_count, &diagnostic);
}

static iree_string_view_t loom_amdgpu_dot_rejection_detail(
    loom_amdgpu_dot_rejection_flags_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits, LOOM_AMDGPU_DOT_REJECTION_KIND)) {
    return IREE_SV(
        "AMDGPU source-to-low currently supports vector.dot4i<s8s8>; other "
        "vector dot forms require additional target contracts");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_AMDGPU_DOT_REJECTION_LHS_SHAPE)) {
    return IREE_SV(
        "AMDGPU vector.dot4i lowering requires rank-1 static i8 lhs vectors "
        "with a lane count that is a multiple of 4 and packs into at most 4 "
        "VGPRs");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_AMDGPU_DOT_REJECTION_RHS_SHAPE)) {
    return IREE_SV(
        "AMDGPU vector.dot4i lowering requires the rhs vector shape to match "
        "the lhs vector shape");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_AMDGPU_DOT_REJECTION_ACC_SHAPE)) {
    return IREE_SV(
        "AMDGPU vector.dot4i lowering requires one i32 accumulator lane per "
        "four i8 input lanes");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_DOT_REJECTION_RESULT_SHAPE)) {
    return IREE_SV(
        "AMDGPU vector.dot4i lowering requires one i32 result lane per four "
        "i8 input lanes");
  }
  if (iree_any_bit_set(rejection_bits,
                       LOOM_AMDGPU_DOT_REJECTION_DESCRIPTOR_MISSING)) {
    return IREE_SV(
        "selected AMDGPU descriptor set has no v_dot4_i32_i8 descriptor");
  }
  return IREE_SV("AMDGPU vector dot op is not target-legal");
}

iree_status_t loom_amdgpu_lower_vector_dot4i(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op) {
  uint32_t group_count = 0;
  loom_amdgpu_dot_diagnostic_t diagnostic = {0};
  if (!loom_amdgpu_dot4i_select(loom_low_lower_context_module(context),
                                loom_low_lower_context_descriptor_set(context),
                                source_op, &group_count, &diagnostic)) {
    const iree_string_view_t detail =
        loom_amdgpu_dot_rejection_detail(diagnostic.rejection_bits);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "%.*s",
                            (int)detail.size, detail.data);
  }

  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_dot4i_lhs(source_op), &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_dot4i_rhs(source_op), &low_rhs));
  loom_value_id_t low_acc = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_dot4i_acc(source_op), &low_acc));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, loom_vector_dot4i_result(source_op), &result_type));
  if (group_count == 1) {
    loom_value_id_t operands[] = {low_lhs, low_rhs, low_acc};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT4_I32_I8, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
        &result_type, 1, &low_op));
    return loom_low_lower_bind_value(
        context, loom_vector_dot4i_result(source_op),
        loom_value_slice_get(loom_low_op_results(low_op), 0));
  }

  loom_type_t group_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &group_type));
  loom_value_id_t group_results[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (uint32_t i = 0; i < group_count; ++i) {
    loom_value_id_t group_lhs = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, low_lhs,
                                                    i, group_type, &group_lhs));
    loom_value_id_t group_rhs = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, low_rhs,
                                                    i, group_type, &group_rhs));
    loom_value_id_t group_acc = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op, low_acc,
                                                    i, group_type, &group_acc));
    loom_value_id_t operands[] = {group_lhs, group_rhs, group_acc};
    loom_op_t* group_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT4_I32_I8, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
        &group_type, 1, &group_op));
    group_results[i] = loom_value_slice_get(loom_low_op_results(group_op), 0);
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), group_results, group_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, loom_vector_dot4i_result(source_op),
                                   loom_low_concat_result(concat_op));
}

bool loom_amdgpu_op_is_vector_dot(loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_VECTOR_DOT2F:
    case LOOM_OP_VECTOR_DOT4F8:
    case LOOM_OP_VECTOR_DOT4I:
    case LOOM_OP_VECTOR_DOT8I4:
      return true;
    default:
      return false;
  }
}

iree_status_t loom_amdgpu_low_legality_verify_vector_dot(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (!loom_vector_dot4i_isa(op)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("contract"),
        loom_op_name(loom_target_low_legality_module(context), op),
        IREE_SV("AMDGPU source-to-low currently supports vector.dot4i<s8s8>; "
                "other vector dot forms require additional target contracts"));
  }

  uint32_t unused_group_count = 0;
  loom_amdgpu_dot_diagnostic_t diagnostic = {0};
  if (!loom_amdgpu_dot4i_select(
          loom_target_low_legality_module(context),
          loom_target_low_legality_descriptor_set(context), op,
          &unused_group_count, &diagnostic)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("contract"),
        IREE_SV("amdgpu.v_dot4_i32_i8"),
        loom_amdgpu_dot_rejection_detail(diagnostic.rejection_bits));
  }

  return loom_target_low_legality_record_contract(
      context, provider, op, IREE_SV("amdgpu.v_dot4_i32_i8"),
      IREE_SV("selected"), IREE_SV("selected AMDGPU packed dot descriptor"));
}
