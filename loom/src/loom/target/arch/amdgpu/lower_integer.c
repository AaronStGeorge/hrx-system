// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"

typedef uint16_t loom_amdgpu_integer_rule_flags_t;

// Rule operands/results are address-domain index values.
#define LOOM_AMDGPU_INTEGER_RULE_FLAG_ADDRESS_VALUES ((uint16_t)1u << 0)
// Uniform scalar-result emission is valid for this rule.
#define LOOM_AMDGPU_INTEGER_RULE_FLAG_ALLOW_UNIFORM_RESULT ((uint16_t)1u << 1)
// VGPR descriptor operand order is source operand 1, then source operand 0.
#define LOOM_AMDGPU_INTEGER_RULE_FLAG_SWAP_VECTOR_OPERANDS_0_1 \
  ((uint16_t)1u << 2)

typedef enum loom_amdgpu_integer_rule_kind_e {
  // Rule lowers a two-operand integer/address operation.
  LOOM_AMDGPU_INTEGER_RULE_BINARY = 0,
  // Rule lowers index.madd as a multiply followed by an add.
  LOOM_AMDGPU_INTEGER_RULE_MADD = 1,
} loom_amdgpu_integer_rule_kind_t;

typedef struct loom_amdgpu_integer_rule_t {
  // Source op kind accepted by this row.
  loom_op_kind_t source_op_kind;
  // Emission shape used by this row.
  loom_amdgpu_integer_rule_kind_t kind;
  // Rule behavior flags.
  loom_amdgpu_integer_rule_flags_t flags;
  // Descriptor used when the result is uniform.
  uint64_t scalar_descriptor_id;
  // Descriptor used when the result is a VGPR.
  uint64_t vector_descriptor_id;
} loom_amdgpu_integer_rule_t;

static const loom_amdgpu_integer_rule_t kAmdgpuIntegerRules[] = {
    {
        .source_op_kind = LOOM_OP_SCALAR_ADDI,
        .kind = LOOM_AMDGPU_INTEGER_RULE_BINARY,
        .flags = LOOM_AMDGPU_INTEGER_RULE_FLAG_ALLOW_UNIFORM_RESULT,
        .scalar_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32,
        .vector_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SUBI,
        .kind = LOOM_AMDGPU_INTEGER_RULE_BINARY,
        .flags = LOOM_AMDGPU_INTEGER_RULE_FLAG_ALLOW_UNIFORM_RESULT,
        .scalar_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_S_SUB_U32,
        .vector_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_MULI,
        .kind = LOOM_AMDGPU_INTEGER_RULE_BINARY,
        .scalar_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
        .vector_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_ANDI,
        .kind = LOOM_AMDGPU_INTEGER_RULE_BINARY,
        .flags = LOOM_AMDGPU_INTEGER_RULE_FLAG_ALLOW_UNIFORM_RESULT,
        .scalar_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_S_AND_B32,
        .vector_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_ORI,
        .kind = LOOM_AMDGPU_INTEGER_RULE_BINARY,
        .flags = LOOM_AMDGPU_INTEGER_RULE_FLAG_ALLOW_UNIFORM_RESULT,
        .scalar_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_S_OR_B32,
        .vector_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_XORI,
        .kind = LOOM_AMDGPU_INTEGER_RULE_BINARY,
        .flags = LOOM_AMDGPU_INTEGER_RULE_FLAG_ALLOW_UNIFORM_RESULT,
        .scalar_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_S_XOR_B32,
        .vector_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SHLI,
        .kind = LOOM_AMDGPU_INTEGER_RULE_BINARY,
        .flags = LOOM_AMDGPU_INTEGER_RULE_FLAG_ALLOW_UNIFORM_RESULT |
                 LOOM_AMDGPU_INTEGER_RULE_FLAG_SWAP_VECTOR_OPERANDS_0_1,
        .scalar_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHL_B32,
        .vector_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SHRSI,
        .kind = LOOM_AMDGPU_INTEGER_RULE_BINARY,
        .flags = LOOM_AMDGPU_INTEGER_RULE_FLAG_ALLOW_UNIFORM_RESULT |
                 LOOM_AMDGPU_INTEGER_RULE_FLAG_SWAP_VECTOR_OPERANDS_0_1,
        .scalar_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_S_ASHR_I32,
        .vector_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_ASHRREV_I32,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SHRUI,
        .kind = LOOM_AMDGPU_INTEGER_RULE_BINARY,
        .flags = LOOM_AMDGPU_INTEGER_RULE_FLAG_ALLOW_UNIFORM_RESULT |
                 LOOM_AMDGPU_INTEGER_RULE_FLAG_SWAP_VECTOR_OPERANDS_0_1,
        .scalar_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHR_B32,
        .vector_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_ADD,
        .kind = LOOM_AMDGPU_INTEGER_RULE_BINARY,
        .flags = LOOM_AMDGPU_INTEGER_RULE_FLAG_ADDRESS_VALUES |
                 LOOM_AMDGPU_INTEGER_RULE_FLAG_ALLOW_UNIFORM_RESULT,
        .scalar_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32,
        .vector_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_SUB,
        .kind = LOOM_AMDGPU_INTEGER_RULE_BINARY,
        .flags = LOOM_AMDGPU_INTEGER_RULE_FLAG_ADDRESS_VALUES |
                 LOOM_AMDGPU_INTEGER_RULE_FLAG_ALLOW_UNIFORM_RESULT,
        .scalar_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_S_SUB_U32,
        .vector_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_MUL,
        .kind = LOOM_AMDGPU_INTEGER_RULE_BINARY,
        .flags = LOOM_AMDGPU_INTEGER_RULE_FLAG_ADDRESS_VALUES,
        .scalar_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
        .vector_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_MADD,
        .kind = LOOM_AMDGPU_INTEGER_RULE_MADD,
        .flags = LOOM_AMDGPU_INTEGER_RULE_FLAG_ADDRESS_VALUES,
        .scalar_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
        .vector_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
    },
};

static_assert(LOOM_OP_SCALAR_ADDI < LOOM_OP_SCALAR_SUBI &&
                  LOOM_OP_SCALAR_SUBI < LOOM_OP_SCALAR_MULI &&
                  LOOM_OP_SCALAR_MULI < LOOM_OP_SCALAR_ANDI &&
                  LOOM_OP_SCALAR_ANDI < LOOM_OP_SCALAR_ORI &&
                  LOOM_OP_SCALAR_ORI < LOOM_OP_SCALAR_XORI &&
                  LOOM_OP_SCALAR_XORI < LOOM_OP_SCALAR_SHLI &&
                  LOOM_OP_SCALAR_SHLI < LOOM_OP_SCALAR_SHRSI &&
                  LOOM_OP_SCALAR_SHRSI < LOOM_OP_SCALAR_SHRUI &&
                  LOOM_OP_SCALAR_SHRUI < LOOM_OP_INDEX_ADD &&
                  LOOM_OP_INDEX_ADD < LOOM_OP_INDEX_SUB &&
                  LOOM_OP_INDEX_SUB < LOOM_OP_INDEX_MUL &&
                  LOOM_OP_INDEX_MUL < LOOM_OP_INDEX_MADD,
              "AMDGPU integer rule rows must stay sorted by source op kind");

static const loom_amdgpu_integer_rule_t* loom_amdgpu_integer_rule_for_op(
    loom_op_kind_t source_op_kind) {
  uint16_t low = 0;
  uint16_t high = IREE_ARRAYSIZE(kAmdgpuIntegerRules);
  while (low < high) {
    const uint16_t mid = low + (uint16_t)((high - low) / 2);
    const loom_amdgpu_integer_rule_t* rule = &kAmdgpuIntegerRules[mid];
    if (rule->source_op_kind == source_op_kind) {
      return rule;
    }
    if (rule->source_op_kind < source_op_kind) {
      low = (uint16_t)(mid + 1);
    } else {
      high = mid;
    }
  }
  return NULL;
}

static bool loom_amdgpu_integer_rule_allows_uniform_result(
    const loom_amdgpu_integer_rule_t* rule) {
  return iree_any_bit_set(rule->flags,
                          LOOM_AMDGPU_INTEGER_RULE_FLAG_ALLOW_UNIFORM_RESULT);
}

static bool loom_amdgpu_integer_rule_uses_address_values(
    const loom_amdgpu_integer_rule_t* rule) {
  return iree_any_bit_set(rule->flags,
                          LOOM_AMDGPU_INTEGER_RULE_FLAG_ADDRESS_VALUES);
}

static bool loom_amdgpu_integer_rule_swaps_vector_operands(
    const loom_amdgpu_integer_rule_t* rule) {
  return iree_any_bit_set(
      rule->flags, LOOM_AMDGPU_INTEGER_RULE_FLAG_SWAP_VECTOR_OPERANDS_0_1);
}

static bool loom_amdgpu_select_integer_binary_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_integer_rule_t* rule,
    loom_amdgpu_integer_plan_t* out_plan) {
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  const loom_value_id_t* results = loom_op_const_results(source_op);
  const loom_value_id_t lhs = operands[0];
  const loom_value_id_t rhs = operands[1];
  const loom_value_id_t result = results[0];
  const bool uses_address_values =
      loom_amdgpu_integer_rule_uses_address_values(rule);
  if (uses_address_values &&
      (!loom_amdgpu_value_is_address_scalar(context, lhs) ||
       !loom_amdgpu_value_is_address_scalar(context, rhs) ||
       !loom_amdgpu_value_is_address_scalar(context, result))) {
    return false;
  }
  if (!uses_address_values && (!loom_amdgpu_value_is_i32(context, lhs) ||
                               !loom_amdgpu_value_is_i32(context, rhs) ||
                               !loom_amdgpu_value_is_i32(context, result))) {
    return false;
  }

  const bool lhs_vgpr = loom_amdgpu_value_prefers_vgpr(context, lhs);
  const bool rhs_vgpr = loom_amdgpu_value_prefers_vgpr(context, rhs);
  const bool result_vgpr = loom_amdgpu_value_prefers_vgpr(context, result);
  loom_amdgpu_integer_operand_kind_t operand_kind =
      LOOM_AMDGPU_INTEGER_OPERAND_KIND_NONE;
  uint64_t descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  loom_value_id_t descriptor_lhs = lhs;
  loom_value_id_t descriptor_rhs = rhs;
  if (result_vgpr) {
    if (rule->vector_descriptor_id == LOOM_LOW_DESCRIPTOR_ID_NONE) {
      return false;
    }
    const bool can_materialize_lhs =
        uses_address_values
            ? loom_amdgpu_value_can_materialize_as_vgpr_address(context, lhs)
            : loom_amdgpu_value_can_materialize_as_vgpr_i32(context, lhs);
    const bool can_materialize_rhs =
        uses_address_values
            ? loom_amdgpu_value_can_materialize_as_vgpr_address(context, rhs)
            : loom_amdgpu_value_can_materialize_as_vgpr_i32(context, rhs);
    if (!can_materialize_lhs || !can_materialize_rhs) {
      return false;
    }
    operand_kind = uses_address_values
                       ? LOOM_AMDGPU_INTEGER_OPERAND_KIND_ADDRESS_VGPR
                       : LOOM_AMDGPU_INTEGER_OPERAND_KIND_I32_VGPR;
    descriptor_id = rule->vector_descriptor_id;
    if (loom_amdgpu_integer_rule_swaps_vector_operands(rule)) {
      descriptor_lhs = rhs;
      descriptor_rhs = lhs;
    }
  } else {
    if (!loom_amdgpu_integer_rule_allows_uniform_result(rule) || lhs_vgpr ||
        rhs_vgpr || rule->scalar_descriptor_id == LOOM_LOW_DESCRIPTOR_ID_NONE) {
      return false;
    }
    operand_kind = uses_address_values
                       ? LOOM_AMDGPU_INTEGER_OPERAND_KIND_ADDRESS_DIRECT
                       : LOOM_AMDGPU_INTEGER_OPERAND_KIND_I32_DIRECT;
    descriptor_id = rule->scalar_descriptor_id;
  }

  *out_plan = (loom_amdgpu_integer_plan_t){
      .emit_kind = LOOM_AMDGPU_INTEGER_EMIT_KIND_BINARY,
      .operand_kind = operand_kind,
      .descriptor_id = descriptor_id,
      .secondary_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE,
      .source_lhs = descriptor_lhs,
      .source_rhs = descriptor_rhs,
      .source_accumulator = LOOM_VALUE_ID_INVALID,
      .source_result = result,
  };
  return true;
}

static bool loom_amdgpu_select_integer_madd_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_integer_plan_t* out_plan) {
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  const loom_value_id_t* results = loom_op_const_results(source_op);
  const loom_value_id_t a = operands[0];
  const loom_value_id_t b = operands[1];
  const loom_value_id_t c = operands[2];
  const loom_value_id_t result = results[0];
  if (loom_amdgpu_value_is_address_scalar(context, a) &&
      loom_amdgpu_value_is_address_scalar(context, b) &&
      loom_amdgpu_value_is_address_scalar(context, c) &&
      loom_amdgpu_value_is_address_scalar(context, result) &&
      loom_amdgpu_value_prefers_vgpr(context, result) &&
      loom_amdgpu_value_can_materialize_as_vgpr_address(context, a) &&
      loom_amdgpu_value_can_materialize_as_vgpr_address(context, b) &&
      loom_amdgpu_value_can_materialize_as_vgpr_address(context, c)) {
    *out_plan = (loom_amdgpu_integer_plan_t){
        .emit_kind = LOOM_AMDGPU_INTEGER_EMIT_KIND_MADD,
        .operand_kind = LOOM_AMDGPU_INTEGER_OPERAND_KIND_ADDRESS_VGPR,
        .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32,
        .secondary_descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32,
        .source_lhs = a,
        .source_rhs = b,
        .source_accumulator = c,
        .source_result = result,
    };
    return true;
  }
  return false;
}

iree_status_t loom_amdgpu_select_integer_plan(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
  const loom_amdgpu_integer_rule_t* rule =
      loom_amdgpu_integer_rule_for_op(source_op->kind);
  if (rule == NULL) {
    return iree_ok_status();
  }
  loom_amdgpu_integer_plan_t* plan_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  bool selected = false;
  switch (rule->kind) {
    case LOOM_AMDGPU_INTEGER_RULE_BINARY:
      selected = loom_amdgpu_select_integer_binary_plan(context, source_op,
                                                        rule, plan_data);
      break;
    case LOOM_AMDGPU_INTEGER_RULE_MADD:
      selected =
          loom_amdgpu_select_integer_madd_plan(context, source_op, plan_data);
      break;
    default:
      IREE_ASSERT_UNREACHABLE();
      break;
  }
  if (selected) {
    *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lower_integer_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_integer_plan_t* plan, loom_value_id_t source_value,
    loom_value_id_t* out_low_value) {
  switch (plan->operand_kind) {
    case LOOM_AMDGPU_INTEGER_OPERAND_KIND_I32_DIRECT:
    case LOOM_AMDGPU_INTEGER_OPERAND_KIND_ADDRESS_DIRECT:
      return loom_low_lower_lookup_value(context, source_value, out_low_value);
    case LOOM_AMDGPU_INTEGER_OPERAND_KIND_I32_VGPR:
      return loom_amdgpu_lookup_or_materialize_vgpr_i32(
          context, source_op, source_value, out_low_value);
    case LOOM_AMDGPU_INTEGER_OPERAND_KIND_ADDRESS_VGPR:
      return loom_amdgpu_lookup_or_materialize_vgpr_address(
          context, source_op, source_value, out_low_value);
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_lower_integer_binary_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_integer_plan_t* plan) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, plan->source_result, &result_type));
  loom_value_id_t low_operands[2] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_integer_operand(
      context, source_op, plan, plan->source_lhs, &low_operands[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_integer_operand(
      context, source_op, plan, plan->source_rhs, &low_operands[1]));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, plan->descriptor_id, low_operands,
      IREE_ARRAYSIZE(low_operands), loom_make_named_attr_slice(NULL, 0),
      &result_type, 1, &low_op));
  return loom_low_lower_bind_value(
      context, plan->source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_amdgpu_lower_integer_madd_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_integer_plan_t* plan) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, plan->source_result, &result_type));
  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_integer_operand(
      context, source_op, plan, plan->source_lhs, &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_integer_operand(
      context, source_op, plan, plan->source_rhs, &low_rhs));
  loom_value_id_t low_product = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, plan->descriptor_id, low_lhs, low_rhs, result_type,
      &low_product));
  loom_value_id_t low_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lower_integer_operand(
      context, source_op, plan, plan->source_accumulator, &low_accumulator));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, plan->secondary_descriptor_id, low_product,
      low_accumulator, result_type, &low_result));
  return loom_low_lower_bind_value(context, plan->source_result, low_result);
}

iree_status_t loom_amdgpu_lower_integer_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_integer_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  switch (plan->emit_kind) {
    case LOOM_AMDGPU_INTEGER_EMIT_KIND_BINARY:
      return loom_amdgpu_lower_integer_binary_plan(context, source_op, plan);
    case LOOM_AMDGPU_INTEGER_EMIT_KIND_MADD:
      return loom_amdgpu_lower_integer_madd_plan(context, source_op, plan);
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
}
