// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/lower.h"

#include <stdint.h>

#include "loom/codegen/low/lower_rules.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/wasm/descriptors.h"

static bool loom_wasm_type_is_address_i32(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
    case LOOM_SCALAR_TYPE_I32:
      return true;
    default:
      return false;
  }
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
  if (loom_wasm_type_is_address_i32(source_type)) {
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
      IREE_SV("Wasm lowering currently supports only i32/index/offset scalar "
              "values and vector<4xi32>/vector<4xf32> SIMD values"));
}

static iree_status_t loom_wasm_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)source_argument_index;
  IREE_ASSERT_ARGUMENT(out_argument);
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t address_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_wasm_make_i32_register_type(context, &address_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
        .abi_type = address_type,
        .resource_semantic_type = loom_type_none(),
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_semantic_type = loom_type_none(),
  };
  return loom_wasm_map_type(user_data, context, source_function_op, source_type,
                            &out_argument->abi_type);
}

enum loom_wasm_type_pattern_e {
  LOOM_WASM_TYPE_I32 = 0,
  LOOM_WASM_TYPE_V4I32 = 1,
  LOOM_WASM_TYPE_V4F32 = 2,
  LOOM_WASM_TYPE_ADDRESS_I32 = 3,
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
    [LOOM_WASM_TYPE_ADDRESS_I32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_INDEX) |
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_OFFSET),
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
  LOOM_WASM_DIAGNOSTIC_ADDRESS_I32 = 5,
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
    [LOOM_WASM_DIAGNOSTIC_ADDRESS_I32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("index_or_offset"),
            .reason = IREE_SVL(
                "Wasm address lowering requires index or offset scalar values"),
        },
};

enum loom_wasm_guard_e {
  LOOM_WASM_CONST_VALUE_GUARD = 0,
  LOOM_WASM_CONST_RESULT_GUARD = 1,
  LOOM_WASM_CONST_RANGE_GUARD = 2,
  LOOM_WASM_ADDRESS_CONST_VALUE_GUARD = 3,
  LOOM_WASM_ADDRESS_CONST_RESULT_GUARD = 4,
  LOOM_WASM_ADDRESS_CONST_RANGE_GUARD = 5,
  LOOM_WASM_SCALAR_LHS_GUARD = 6,
  LOOM_WASM_SCALAR_RHS_GUARD = 7,
  LOOM_WASM_SCALAR_RESULT_GUARD = 8,
  LOOM_WASM_ADDRESS_LHS_GUARD = 9,
  LOOM_WASM_ADDRESS_RHS_GUARD = 10,
  LOOM_WASM_ADDRESS_RESULT_GUARD = 11,
  LOOM_WASM_SPLAT_SCALAR_GUARD = 12,
  LOOM_WASM_SPLAT_RESULT_GUARD = 13,
  LOOM_WASM_V4F32_LHS_GUARD = 14,
  LOOM_WASM_V4F32_RHS_GUARD = 15,
  LOOM_WASM_V4F32_RESULT_GUARD = 16,
  LOOM_WASM_V4I32_LHS_GUARD = 17,
  LOOM_WASM_V4I32_RHS_GUARD = 18,
  LOOM_WASM_V4I32_RESULT_GUARD = 19,
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
    [LOOM_WASM_ADDRESS_CONST_VALUE_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_ATTR_KIND,
            .attr_index = 0,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_I64_ATTR,
            .attr_kind = LOOM_ATTR_I64,
        },
    [LOOM_WASM_ADDRESS_CONST_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_RESULT0,
            .type_pattern_index = LOOM_WASM_TYPE_ADDRESS_I32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_ADDRESS_I32,
        },
    [LOOM_WASM_ADDRESS_CONST_RANGE_GUARD] =
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
    [LOOM_WASM_ADDRESS_LHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_OPERAND0,
            .type_pattern_index = LOOM_WASM_TYPE_ADDRESS_I32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_ADDRESS_I32,
        },
    [LOOM_WASM_ADDRESS_RHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_OPERAND1,
            .type_pattern_index = LOOM_WASM_TYPE_ADDRESS_I32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_ADDRESS_I32,
        },
    [LOOM_WASM_ADDRESS_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_RESULT0,
            .type_pattern_index = LOOM_WASM_TYPE_ADDRESS_I32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_ADDRESS_I32,
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
  LOOM_WASM_EMIT_I32_MUL = 3,
  LOOM_WASM_EMIT_I32X4_SPLAT = 4,
  LOOM_WASM_EMIT_F32X4_ADD = 5,
  LOOM_WASM_EMIT_F32X4_MUL = 6,
  LOOM_WASM_EMIT_I32X4_ADD = 7,
  LOOM_WASM_EMIT_I32X4_SUB = 8,
  LOOM_WASM_EMIT_I32X4_MUL = 9,
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
    [LOOM_WASM_EMIT_I32_MUL] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_MUL,
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
    {
        .source_op_kind = LOOM_OP_INDEX_CONSTANT,
        .guard_start = LOOM_WASM_ADDRESS_CONST_VALUE_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_WASM_EMIT_I32_CONST,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_ADD,
        .guard_start = LOOM_WASM_ADDRESS_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_WASM_EMIT_I32_ADD,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_SUB,
        .guard_start = LOOM_WASM_ADDRESS_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_WASM_EMIT_I32_SUB,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_MUL,
        .guard_start = LOOM_WASM_ADDRESS_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_WASM_EMIT_I32_MUL,
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
    {
        .source_op_kind = LOOM_OP_INDEX_CONSTANT,
        .rule_start = 9,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_ADD,
        .rule_start = 10,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_SUB,
        .rule_start = 11,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_MUL,
        .rule_start = 12,
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

static bool loom_wasm_memory_space_is_linear(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      return true;
    default:
      return false;
  }
}

static bool loom_wasm_i64_fits_i32(int64_t value) {
  return value >= INT32_MIN && value <= INT32_MAX;
}

static bool loom_wasm_memory_access_shape_is_v128(
    const loom_low_source_memory_access_plan_t* plan) {
  return plan->element_byte_count == 4 && plan->vector_lane_count == 4 &&
         plan->vector_lane_byte_stride == 4;
}

static bool loom_wasm_source_value_is_block_argument(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return false;
  }
  return loom_value_is_block_arg(loom_module_value(module, value_id));
}

static bool loom_wasm_select_memory_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_source_memory_access_plan_t* out_plan) {
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_low_source_memory_access_plan_build(
          module, loom_low_lower_context_fact_table(context), source_op,
          out_plan, &diagnostic)) {
    return false;
  }
  if (!loom_wasm_memory_access_shape_is_v128(out_plan) ||
      !loom_wasm_memory_space_is_linear(out_plan->memory_space) ||
      !loom_wasm_i64_fits_i32(out_plan->static_byte_offset) ||
      !loom_wasm_source_value_is_block_argument(module,
                                                out_plan->root_value_id)) {
    return false;
  }
  if (loom_low_source_memory_access_is_dynamic(out_plan) &&
      (out_plan->dynamic_index_byte_stride <= 0 ||
       !loom_wasm_i64_fits_i32(out_plan->dynamic_index_byte_stride))) {
    return false;
  }
  return true;
}

static iree_status_t loom_wasm_select_op(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
    case LOOM_OP_BUFFER_VIEW:
      *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
      return iree_ok_status();
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VECTOR_STORE: {
      loom_low_source_memory_access_plan_t* plan = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan), (void**)&plan));
      if (loom_wasm_select_memory_access(context, source_op, plan)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_wasm_make_i32_value_attr(
    loom_low_lower_context_t* context, int64_t value,
    loom_named_attr_t* out_attr) {
  IREE_ASSERT_ARGUMENT(out_attr);
  IREE_ASSERT(loom_wasm_i64_fits_i32(value));
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context), IREE_SV("i32_value"), &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_i32_const(loom_low_lower_context_t* context,
                                              int64_t value,
                                              loom_location_id_t location,
                                              loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_wasm_make_i32_value_attr(context, value, &attr));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_wasm_make_i32_register_type(context, &result_type));
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_const(
      context, WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_CONST,
      loom_make_named_attr_slice(&attr, 1), result_type, location, &const_op));
  *out_value_id = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_i32_binary(
    loom_low_lower_context_t* context, uint64_t descriptor_id,
    loom_value_id_t lhs, loom_value_id_t rhs, loom_location_id_t location,
    loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_wasm_make_i32_register_type(context, &result_type));
  loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
      context, descriptor_id, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &result_type, 1, NULL, 0, location, &op));
  *out_value_id = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_address_offset(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* plan,
    loom_location_id_t location, loom_value_id_t* inout_address) {
  if (plan->static_byte_offset == 0) {
    return iree_ok_status();
  }
  loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_emit_i32_const(
      context, plan->static_byte_offset, location, &offset));
  return loom_wasm_emit_i32_binary(
      context, WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_ADD, *inout_address,
      offset, location, inout_address);
}

static iree_status_t loom_wasm_emit_dynamic_address_offset(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* plan,
    loom_location_id_t location, loom_value_id_t* inout_address) {
  if (!loom_low_source_memory_access_is_dynamic(plan)) {
    return iree_ok_status();
  }
  loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, plan->dynamic_index,
                                                   &dynamic_index));
  loom_value_id_t dynamic_offset = dynamic_index;
  if (plan->dynamic_index_byte_stride != 1) {
    loom_value_id_t stride = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_wasm_emit_i32_const(
        context, plan->dynamic_index_byte_stride, location, &stride));
    IREE_RETURN_IF_ERROR(loom_wasm_emit_i32_binary(
        context, WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_MUL, dynamic_index,
        stride, location, &dynamic_offset));
  }
  return loom_wasm_emit_i32_binary(
      context, WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_ADD, *inout_address,
      dynamic_offset, location, inout_address);
}

static iree_status_t loom_wasm_emit_memory_address(
    loom_low_lower_context_t* context,
    const loom_low_source_memory_access_plan_t* plan,
    loom_location_id_t location, loom_value_id_t* out_address) {
  IREE_ASSERT_ARGUMENT(out_address);
  *out_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->root_value_id, out_address));
  IREE_RETURN_IF_ERROR(loom_wasm_emit_dynamic_address_offset(
      context, plan, location, out_address));
  return loom_wasm_emit_address_offset(context, plan, location, out_address);
}

static iree_status_t loom_wasm_lower_buffer_alias(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  loom_value_id_t low_value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value_id, &low_value_id));
  return loom_low_lower_bind_value(context, result_value_id, low_value_id);
}

static iree_status_t loom_wasm_lower_vector_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* plan) {
  loom_value_id_t address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_emit_memory_address(
      context, plan, source_op->location, &address));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_wasm_make_v128_register_type(context, &result_type));
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
      context, WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_V128_LOAD, &address, 1,
      loom_named_attr_slice_empty(), &result_type, 1, NULL, 0,
      source_op->location, &load_op));
  return loom_low_lower_bind_value(
      context, loom_vector_load_result(source_op),
      loom_value_slice_get(loom_low_op_results(load_op), 0));
}

static iree_status_t loom_wasm_lower_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* plan) {
  loom_value_id_t address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_emit_memory_address(
      context, plan, source_op->location, &address));
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_value(source_op), &value));
  loom_value_id_t operands[] = {address, value};
  loom_op_t* store_op = NULL;
  return loom_low_lower_emit_descriptor_op(
      context, WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_V128_STORE, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), NULL, 0, NULL, 0,
      source_op->location, &store_op);
}

static iree_status_t loom_wasm_emit_op(void* user_data,
                                       loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_low_lower_plan_t plan) {
  (void)user_data;
  switch (plan.id) {
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
      return loom_wasm_lower_buffer_alias(
          context, loom_buffer_assume_memory_space_buffer(source_op),
          loom_buffer_assume_memory_space_result(source_op));
    case LOOM_OP_BUFFER_VIEW:
      return loom_wasm_lower_buffer_alias(context,
                                          loom_buffer_view_buffer(source_op),
                                          loom_buffer_view_result(source_op));
    case LOOM_OP_VECTOR_LOAD:
      return loom_wasm_lower_vector_load(
          context, source_op,
          (const loom_low_source_memory_access_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_STORE:
      return loom_wasm_lower_vector_store(
          context, source_op,
          (const loom_low_source_memory_access_plan_t*)plan.target_data);
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
}

static const loom_low_lower_policy_t kWasmLowLowerPolicy = {
    .name = IREE_SVL("wasm-lower"),
    .map_type = {.fn = loom_wasm_map_type, .user_data = NULL},
    .map_argument = {.fn = loom_wasm_map_argument, .user_data = NULL},
    .rule_set = &loom_wasm_rule_set,
    .select_op = {.fn = loom_wasm_select_op, .user_data = NULL},
    .emit_op = {.fn = loom_wasm_emit_op, .user_data = NULL},
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
