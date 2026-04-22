// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/lower.h"

#include <stdint.h>

#include "loom/codegen/low/lower_rules.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/wasm/descriptors.h"

static bool loom_wasm_type_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static bool loom_wasm_type_is_vector_4xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_wasm_type_is_vector_4xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static iree_status_t loom_wasm_make_i32_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_WASM_I32, 1, out_type);
}

static iree_status_t loom_wasm_make_v128_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_WASM_V128, 1, out_type);
}

static iree_status_t loom_wasm_map_type(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_type_t source_type,
                                        loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_wasm_type_is_i32(source_type)) {
    return loom_wasm_make_i32_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_vector_4xi32(source_type)) {
    return loom_wasm_make_v128_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_vector_4xf32(source_type)) {
    return loom_wasm_make_v128_register_type(context, out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("Wasm lowering currently supports only i32 scalar values and "
              "vector<4xi32>/vector<4xf32> SIMD values"));
}

enum loom_wasm_type_pattern_e {
  LOOM_WASM_TYPE_I32 = 0,
  LOOM_WASM_TYPE_V4I32 = 1,
  LOOM_WASM_TYPE_V4F32 = 2,
};

static const loom_low_lower_type_pattern_t loom_wasm_type_patterns[] = {
    [LOOM_WASM_TYPE_I32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I32),
        },
    [LOOM_WASM_TYPE_V4I32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0,
            .type_kind = LOOM_TYPE_VECTOR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I32),
            .rank = 1,
            .static_dim0 = 4,
        },
    [LOOM_WASM_TYPE_V4F32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0,
            .type_kind = LOOM_TYPE_VECTOR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_F32),
            .rank = 1,
            .static_dim0 = 4,
        },
};

enum loom_wasm_value_ref_e {
  LOOM_WASM_OPERAND0 = 0,
  LOOM_WASM_OPERAND1 = 1,
  LOOM_WASM_RESULT0 = 2,
};

static const loom_low_lower_value_ref_t loom_wasm_value_refs[] = {
    [LOOM_WASM_OPERAND0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
        },
    [LOOM_WASM_OPERAND1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
        },
    [LOOM_WASM_RESULT0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
            .index = 0,
        },
};

static const loom_low_lower_attr_copy_t loom_wasm_attr_copies[] = {
    {
        .target_name = IREE_SVL("i32_value"),
        .source_attr_index = 0,
    },
};

enum loom_wasm_diagnostic_e {
  LOOM_WASM_DIAGNOSTIC_I32 = 0,
  LOOM_WASM_DIAGNOSTIC_V4I32 = 1,
  LOOM_WASM_DIAGNOSTIC_V4F32 = 2,
  LOOM_WASM_DIAGNOSTIC_I64_ATTR = 3,
  LOOM_WASM_DIAGNOSTIC_I32_CONSTANT_RANGE = 4,
};

static const loom_low_lower_diagnostic_t loom_wasm_diagnostics[] = {
    [LOOM_WASM_DIAGNOSTIC_I32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("i32"),
            .reason = IREE_SVL("Wasm lowering requires i32 scalar values"),
        },
    [LOOM_WASM_DIAGNOSTIC_V4I32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("vector<4xi32>"),
            .reason =
                IREE_SVL("Wasm SIMD lowering requires vector<4xi32> values"),
        },
    [LOOM_WASM_DIAGNOSTIC_V4F32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("vector<4xf32>"),
            .reason =
                IREE_SVL("Wasm SIMD lowering requires vector<4xf32> values"),
        },
    [LOOM_WASM_DIAGNOSTIC_I64_ATTR] =
        {
            .subject_kind = IREE_SVL("attr"),
            .subject_name = IREE_SVL("value"),
            .reason = IREE_SVL("Wasm constant lowering requires an i64 value"),
        },
    [LOOM_WASM_DIAGNOSTIC_I32_CONSTANT_RANGE] =
        {
            .subject_kind = IREE_SVL("attr"),
            .subject_name = IREE_SVL("value"),
            .reason = IREE_SVL("Wasm i32 constants must fit in signed i32"),
        },
};

enum loom_wasm_guard_e {
  LOOM_WASM_CONST_VALUE_GUARD = 0,
  LOOM_WASM_CONST_RESULT_GUARD = 1,
  LOOM_WASM_CONST_RANGE_GUARD = 2,
  LOOM_WASM_SCALAR_LHS_GUARD = 3,
  LOOM_WASM_SCALAR_RHS_GUARD = 4,
  LOOM_WASM_SCALAR_RESULT_GUARD = 5,
  LOOM_WASM_SPLAT_SCALAR_GUARD = 6,
  LOOM_WASM_SPLAT_RESULT_GUARD = 7,
  LOOM_WASM_V4F32_LHS_GUARD = 8,
  LOOM_WASM_V4F32_RHS_GUARD = 9,
  LOOM_WASM_V4F32_RESULT_GUARD = 10,
  LOOM_WASM_V4I32_LHS_GUARD = 11,
  LOOM_WASM_V4I32_RHS_GUARD = 12,
  LOOM_WASM_V4I32_RESULT_GUARD = 13,
};

static const loom_low_lower_guard_t loom_wasm_guards[] = {
    [LOOM_WASM_CONST_VALUE_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_ATTR_KIND,
            .attr_index = 0,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_I64_ATTR,
            .attr_kind = LOOM_ATTR_I64,
        },
    [LOOM_WASM_CONST_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_RESULT0,
            .type_pattern_index = LOOM_WASM_TYPE_I32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_I32,
        },
    [LOOM_WASM_CONST_RANGE_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE,
            .attr_index = 0,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_I32_CONSTANT_RANGE,
            .minimum_i64 = INT32_MIN,
            .maximum_i64 = INT32_MAX,
        },
    [LOOM_WASM_SCALAR_LHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_OPERAND0,
            .type_pattern_index = LOOM_WASM_TYPE_I32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_I32,
        },
    [LOOM_WASM_SCALAR_RHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_OPERAND1,
            .type_pattern_index = LOOM_WASM_TYPE_I32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_I32,
        },
    [LOOM_WASM_SCALAR_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_RESULT0,
            .type_pattern_index = LOOM_WASM_TYPE_I32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_I32,
        },
    [LOOM_WASM_SPLAT_SCALAR_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_OPERAND0,
            .type_pattern_index = LOOM_WASM_TYPE_I32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_I32,
        },
    [LOOM_WASM_SPLAT_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_RESULT0,
            .type_pattern_index = LOOM_WASM_TYPE_V4I32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_V4I32,
        },
    [LOOM_WASM_V4F32_LHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_OPERAND0,
            .type_pattern_index = LOOM_WASM_TYPE_V4F32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_V4F32,
        },
    [LOOM_WASM_V4F32_RHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_OPERAND1,
            .type_pattern_index = LOOM_WASM_TYPE_V4F32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_V4F32,
        },
    [LOOM_WASM_V4F32_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_RESULT0,
            .type_pattern_index = LOOM_WASM_TYPE_V4F32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_V4F32,
        },
    [LOOM_WASM_V4I32_LHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_OPERAND0,
            .type_pattern_index = LOOM_WASM_TYPE_V4I32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_V4I32,
        },
    [LOOM_WASM_V4I32_RHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_OPERAND1,
            .type_pattern_index = LOOM_WASM_TYPE_V4I32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_V4I32,
        },
    [LOOM_WASM_V4I32_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_RESULT0,
            .type_pattern_index = LOOM_WASM_TYPE_V4I32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_V4I32,
        },
};

enum loom_wasm_emit_e {
  LOOM_WASM_EMIT_I32_CONST = 0,
  LOOM_WASM_EMIT_I32_ADD = 1,
  LOOM_WASM_EMIT_I32_SUB = 2,
  LOOM_WASM_EMIT_I32X4_SPLAT = 3,
  LOOM_WASM_EMIT_F32X4_ADD = 4,
  LOOM_WASM_EMIT_F32X4_MUL = 5,
  LOOM_WASM_EMIT_I32X4_ADD = 6,
  LOOM_WASM_EMIT_I32X4_SUB = 7,
  LOOM_WASM_EMIT_I32X4_MUL = 8,
};

static const loom_low_lower_emit_t loom_wasm_emits[] = {
    [LOOM_WASM_EMIT_I32_CONST] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST,
            .descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_CONST,
            .result_ref_start = LOOM_WASM_RESULT0,
            .result_ref_count = 1,
            .attr_copy_start = 0,
            .attr_copy_count = 1,
        },
    [LOOM_WASM_EMIT_I32_ADD] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_ADD,
            .operand_ref_start = LOOM_WASM_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_WASM_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_WASM_EMIT_I32_SUB] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_SUB,
            .operand_ref_start = LOOM_WASM_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_WASM_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_WASM_EMIT_I32X4_SPLAT] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_SPLAT,
            .operand_ref_start = LOOM_WASM_OPERAND0,
            .operand_ref_count = 1,
            .result_ref_start = LOOM_WASM_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_WASM_EMIT_F32X4_ADD] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32X4_ADD,
            .operand_ref_start = LOOM_WASM_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_WASM_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_WASM_EMIT_F32X4_MUL] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32X4_MUL,
            .operand_ref_start = LOOM_WASM_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_WASM_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_WASM_EMIT_I32X4_ADD] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_ADD,
            .operand_ref_start = LOOM_WASM_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_WASM_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_WASM_EMIT_I32X4_SUB] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_SUB,
            .operand_ref_start = LOOM_WASM_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_WASM_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_WASM_EMIT_I32X4_MUL] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_MUL,
            .operand_ref_start = LOOM_WASM_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_WASM_RESULT0,
            .result_ref_count = 1,
        },
};

static const loom_low_lower_rule_t loom_wasm_rules[] = {
    {
        .source_op_kind = LOOM_OP_SCALAR_ADDI,
        .guard_start = LOOM_WASM_SCALAR_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_WASM_EMIT_I32_ADD,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SUBI,
        .guard_start = LOOM_WASM_SCALAR_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_WASM_EMIT_I32_SUB,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_CONSTANT,
        .guard_start = LOOM_WASM_CONST_VALUE_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_WASM_EMIT_I32_CONST,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SPLAT,
        .guard_start = LOOM_WASM_SPLAT_SCALAR_GUARD,
        .guard_count = 2,
        .emit_start = LOOM_WASM_EMIT_I32X4_SPLAT,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_ADDF,
        .guard_start = LOOM_WASM_V4F32_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_WASM_EMIT_F32X4_ADD,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_MULF,
        .guard_start = LOOM_WASM_V4F32_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_WASM_EMIT_F32X4_MUL,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_ADDI,
        .guard_start = LOOM_WASM_V4I32_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_WASM_EMIT_I32X4_ADD,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SUBI,
        .guard_start = LOOM_WASM_V4I32_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_WASM_EMIT_I32X4_SUB,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_MULI,
        .guard_start = LOOM_WASM_V4I32_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_WASM_EMIT_I32X4_MUL,
        .emit_count = 1,
    },
};

static const loom_low_lower_rule_span_t loom_wasm_rule_spans[] = {
    {
        .source_op_kind = LOOM_OP_SCALAR_ADDI,
        .rule_start = 0,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SUBI,
        .rule_start = 1,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_CONSTANT,
        .rule_start = 2,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SPLAT,
        .rule_start = 3,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_ADDF,
        .rule_start = 4,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_MULF,
        .rule_start = 5,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_ADDI,
        .rule_start = 6,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SUBI,
        .rule_start = 7,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_MULI,
        .rule_start = 8,
        .rule_count = 1,
    },
};

static const loom_low_lower_rule_set_t loom_wasm_rule_set = {
    .spans = loom_wasm_rule_spans,
    .span_count = IREE_ARRAYSIZE(loom_wasm_rule_spans),
    .rules = loom_wasm_rules,
    .rule_count = IREE_ARRAYSIZE(loom_wasm_rules),
    .type_patterns = loom_wasm_type_patterns,
    .type_pattern_count = IREE_ARRAYSIZE(loom_wasm_type_patterns),
    .value_refs = loom_wasm_value_refs,
    .value_ref_count = IREE_ARRAYSIZE(loom_wasm_value_refs),
    .guards = loom_wasm_guards,
    .guard_count = IREE_ARRAYSIZE(loom_wasm_guards),
    .attr_copies = loom_wasm_attr_copies,
    .attr_copy_count = IREE_ARRAYSIZE(loom_wasm_attr_copies),
    .emits = loom_wasm_emits,
    .emit_count = IREE_ARRAYSIZE(loom_wasm_emits),
    .diagnostics = loom_wasm_diagnostics,
    .diagnostic_count = IREE_ARRAYSIZE(loom_wasm_diagnostics),
};

static const loom_low_lower_policy_t kWasmLowLowerPolicy = {
    .name = IREE_SVL("wasm-lower"),
    .map_type = {.fn = loom_wasm_map_type, .user_data = NULL},
    .rule_set = &loom_wasm_rule_set,
};

const loom_low_lower_policy_t* loom_wasm_low_lower_policy(void) {
  return &kWasmLowLowerPolicy;
}

void loom_wasm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("wasm.core.simd128"),
          .policy = &kWasmLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
