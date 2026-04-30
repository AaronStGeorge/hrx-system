// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/codegen/low/lower_rules.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/combining.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/wasm/descriptors.h"
#include "loom/target/emit/wasm/lower.h"

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

static bool loom_wasm_type_is_scalar_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

static bool loom_wasm_type_is_vector_4xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_wasm_type_is_vector_4xi1(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_wasm_type_is_vector_4xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_wasm_type_is_v128_source_vector(loom_type_t type) {
  return loom_wasm_type_is_vector_4xi32(type) ||
         loom_wasm_type_is_vector_4xi1(type) ||
         loom_wasm_type_is_vector_4xf32(type);
}

static iree_status_t loom_wasm_make_i32_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_WASM_I32, 1, out_type);
}

static iree_status_t loom_wasm_make_f32_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_WASM_F32, 1, out_type);
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
  if (loom_wasm_type_is_scalar_f32(source_type)) {
    return loom_wasm_make_f32_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_vector_4xi32(source_type)) {
    return loom_wasm_make_v128_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_vector_4xi1(source_type)) {
    return loom_wasm_make_v128_register_type(context, out_low_type);
  }
  if (loom_wasm_type_is_vector_4xf32(source_type)) {
    return loom_wasm_make_v128_register_type(context, out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("Wasm lowering currently supports only i32/index/offset scalar "
              "values, f32 scalar values, and vector<4xi1>/vector<4xi32>/"
              "vector<4xf32> SIMD values"));
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
  LOOM_WASM_TYPE_V4I1 = 3,
  LOOM_WASM_TYPE_ADDRESS_I32 = 4,
  LOOM_WASM_TYPE_F32 = 5,
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
    [LOOM_WASM_TYPE_V4I1] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0,
            .type_kind = LOOM_TYPE_VECTOR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I1),
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
    [LOOM_WASM_TYPE_F32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_F32),
        },
};

enum loom_wasm_value_ref_e {
  LOOM_WASM_OPERAND0 = 0,
  LOOM_WASM_OPERAND1 = 1,
  LOOM_WASM_OPERAND2 = 2,
  LOOM_WASM_RESULT0 = 3,
  LOOM_WASM_SELECT_TRUE_VALUE = 4,
  LOOM_WASM_SELECT_FALSE_VALUE = 5,
  LOOM_WASM_SELECT_CONDITION = 6,
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
    [LOOM_WASM_OPERAND2] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 2,
        },
    [LOOM_WASM_RESULT0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
            .index = 0,
        },
    [LOOM_WASM_SELECT_TRUE_VALUE] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
        },
    [LOOM_WASM_SELECT_FALSE_VALUE] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 2,
        },
    [LOOM_WASM_SELECT_CONDITION] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
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
  LOOM_WASM_DIAGNOSTIC_V4I1 = 6,
  LOOM_WASM_DIAGNOSTIC_F32 = 7,
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
    [LOOM_WASM_DIAGNOSTIC_V4I1] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("vector<4xi1>"),
            .reason = IREE_SVL(
                "Wasm SIMD mask lowering requires vector<4xi1> values"),
        },
    [LOOM_WASM_DIAGNOSTIC_F32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("f32"),
            .reason = IREE_SVL("Wasm lowering requires f32 scalar values"),
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
  LOOM_WASM_SCALAR_F32_LHS_GUARD = 12,
  LOOM_WASM_SCALAR_F32_RHS_GUARD = 13,
  LOOM_WASM_SCALAR_F32_RESULT_GUARD = 14,
  LOOM_WASM_SPLAT_SCALAR_GUARD = 15,
  LOOM_WASM_SPLAT_RESULT_GUARD = 16,
  LOOM_WASM_V4F32_LHS_GUARD = 17,
  LOOM_WASM_V4F32_RHS_GUARD = 18,
  LOOM_WASM_V4F32_RESULT_GUARD = 19,
  LOOM_WASM_V4I32_LHS_GUARD = 20,
  LOOM_WASM_V4I32_RHS_GUARD = 21,
  LOOM_WASM_V4I32_RESULT_GUARD = 22,
  LOOM_WASM_SELECT_V4I32_GUARD = 23,
  LOOM_WASM_SELECT_V4F32_GUARD = LOOM_WASM_SELECT_V4I32_GUARD + 4,
  LOOM_WASM_CMPI_EQ_GUARD = LOOM_WASM_SELECT_V4F32_GUARD + 4,
  LOOM_WASM_CMPI_NE_GUARD = LOOM_WASM_CMPI_EQ_GUARD + 4,
  LOOM_WASM_CMPI_SLT_GUARD = LOOM_WASM_CMPI_NE_GUARD + 4,
  LOOM_WASM_CMPI_SLE_GUARD = LOOM_WASM_CMPI_SLT_GUARD + 4,
  LOOM_WASM_CMPI_SGT_GUARD = LOOM_WASM_CMPI_SLE_GUARD + 4,
  LOOM_WASM_CMPI_SGE_GUARD = LOOM_WASM_CMPI_SGT_GUARD + 4,
  LOOM_WASM_CMPI_ULT_GUARD = LOOM_WASM_CMPI_SGE_GUARD + 4,
  LOOM_WASM_CMPI_ULE_GUARD = LOOM_WASM_CMPI_ULT_GUARD + 4,
  LOOM_WASM_CMPI_UGT_GUARD = LOOM_WASM_CMPI_ULE_GUARD + 4,
  LOOM_WASM_CMPI_UGE_GUARD = LOOM_WASM_CMPI_UGT_GUARD + 4,
  LOOM_WASM_CMPF_OEQ_GUARD = LOOM_WASM_CMPI_UGE_GUARD + 4,
  LOOM_WASM_CMPF_OGT_GUARD = LOOM_WASM_CMPF_OEQ_GUARD + 4,
  LOOM_WASM_CMPF_OGE_GUARD = LOOM_WASM_CMPF_OGT_GUARD + 4,
  LOOM_WASM_CMPF_OLT_GUARD = LOOM_WASM_CMPF_OGE_GUARD + 4,
  LOOM_WASM_CMPF_OLE_GUARD = LOOM_WASM_CMPF_OLT_GUARD + 4,
};

#define LOOM_WASM_VALUE_TYPE_GUARD(value_ref, type_pattern, diagnostic) \
  {                                                                     \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,                          \
      .value_ref_index = value_ref,                                     \
      .type_pattern_index = type_pattern,                               \
      .diagnostic_index = diagnostic,                                   \
  }

#define LOOM_WASM_ATTR_ENUM_GUARD(enum_value)             \
  {                                                       \
      .kind = LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ,          \
      .attr_index = 0,                                    \
      .diagnostic_index = LOOM_LOW_LOWER_DIAGNOSTIC_NONE, \
      .u64 = enum_value,                                  \
  }

#define LOOM_WASM_SELECT_GUARDS(base, value_type, diagnostic)                  \
  [base] = LOOM_WASM_VALUE_TYPE_GUARD(LOOM_WASM_OPERAND0, LOOM_WASM_TYPE_V4I1, \
                                      LOOM_WASM_DIAGNOSTIC_V4I1),              \
  [(base) + 1] =                                                               \
      LOOM_WASM_VALUE_TYPE_GUARD(LOOM_WASM_OPERAND1, value_type, diagnostic),  \
  [(base) + 2] =                                                               \
      LOOM_WASM_VALUE_TYPE_GUARD(LOOM_WASM_OPERAND2, value_type, diagnostic),  \
  [(base) + 3] =                                                               \
      LOOM_WASM_VALUE_TYPE_GUARD(LOOM_WASM_RESULT0, value_type, diagnostic)

#define LOOM_WASM_CMPI_GUARDS(base, predicate)                               \
  [base] = LOOM_WASM_ATTR_ENUM_GUARD(predicate),                             \
  [(base) + 1] = LOOM_WASM_VALUE_TYPE_GUARD(                                 \
      LOOM_WASM_OPERAND0, LOOM_WASM_TYPE_V4I32, LOOM_WASM_DIAGNOSTIC_V4I32), \
  [(base) + 2] = LOOM_WASM_VALUE_TYPE_GUARD(                                 \
      LOOM_WASM_OPERAND1, LOOM_WASM_TYPE_V4I32, LOOM_WASM_DIAGNOSTIC_V4I32), \
  [(base) + 3] = LOOM_WASM_VALUE_TYPE_GUARD(                                 \
      LOOM_WASM_RESULT0, LOOM_WASM_TYPE_V4I1, LOOM_WASM_DIAGNOSTIC_V4I1)

#define LOOM_WASM_CMPF_GUARDS(base, predicate)                               \
  [base] = LOOM_WASM_ATTR_ENUM_GUARD(predicate),                             \
  [(base) + 1] = LOOM_WASM_VALUE_TYPE_GUARD(                                 \
      LOOM_WASM_OPERAND0, LOOM_WASM_TYPE_V4F32, LOOM_WASM_DIAGNOSTIC_V4F32), \
  [(base) + 2] = LOOM_WASM_VALUE_TYPE_GUARD(                                 \
      LOOM_WASM_OPERAND1, LOOM_WASM_TYPE_V4F32, LOOM_WASM_DIAGNOSTIC_V4F32), \
  [(base) + 3] = LOOM_WASM_VALUE_TYPE_GUARD(                                 \
      LOOM_WASM_RESULT0, LOOM_WASM_TYPE_V4I1, LOOM_WASM_DIAGNOSTIC_V4I1)

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
    [LOOM_WASM_SCALAR_F32_LHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_OPERAND0,
            .type_pattern_index = LOOM_WASM_TYPE_F32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_F32,
        },
    [LOOM_WASM_SCALAR_F32_RHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_OPERAND1,
            .type_pattern_index = LOOM_WASM_TYPE_F32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_F32,
        },
    [LOOM_WASM_SCALAR_F32_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_WASM_RESULT0,
            .type_pattern_index = LOOM_WASM_TYPE_F32,
            .diagnostic_index = LOOM_WASM_DIAGNOSTIC_F32,
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
    LOOM_WASM_SELECT_GUARDS(LOOM_WASM_SELECT_V4I32_GUARD, LOOM_WASM_TYPE_V4I32,
                            LOOM_WASM_DIAGNOSTIC_V4I32),
    LOOM_WASM_SELECT_GUARDS(LOOM_WASM_SELECT_V4F32_GUARD, LOOM_WASM_TYPE_V4F32,
                            LOOM_WASM_DIAGNOSTIC_V4F32),
    LOOM_WASM_CMPI_GUARDS(LOOM_WASM_CMPI_EQ_GUARD,
                          LOOM_VECTOR_CMPI_PREDICATE_EQ),
    LOOM_WASM_CMPI_GUARDS(LOOM_WASM_CMPI_NE_GUARD,
                          LOOM_VECTOR_CMPI_PREDICATE_NE),
    LOOM_WASM_CMPI_GUARDS(LOOM_WASM_CMPI_SLT_GUARD,
                          LOOM_VECTOR_CMPI_PREDICATE_SLT),
    LOOM_WASM_CMPI_GUARDS(LOOM_WASM_CMPI_SLE_GUARD,
                          LOOM_VECTOR_CMPI_PREDICATE_SLE),
    LOOM_WASM_CMPI_GUARDS(LOOM_WASM_CMPI_SGT_GUARD,
                          LOOM_VECTOR_CMPI_PREDICATE_SGT),
    LOOM_WASM_CMPI_GUARDS(LOOM_WASM_CMPI_SGE_GUARD,
                          LOOM_VECTOR_CMPI_PREDICATE_SGE),
    LOOM_WASM_CMPI_GUARDS(LOOM_WASM_CMPI_ULT_GUARD,
                          LOOM_VECTOR_CMPI_PREDICATE_ULT),
    LOOM_WASM_CMPI_GUARDS(LOOM_WASM_CMPI_ULE_GUARD,
                          LOOM_VECTOR_CMPI_PREDICATE_ULE),
    LOOM_WASM_CMPI_GUARDS(LOOM_WASM_CMPI_UGT_GUARD,
                          LOOM_VECTOR_CMPI_PREDICATE_UGT),
    LOOM_WASM_CMPI_GUARDS(LOOM_WASM_CMPI_UGE_GUARD,
                          LOOM_VECTOR_CMPI_PREDICATE_UGE),
    LOOM_WASM_CMPF_GUARDS(LOOM_WASM_CMPF_OEQ_GUARD,
                          LOOM_VECTOR_CMPF_PREDICATE_OEQ),
    LOOM_WASM_CMPF_GUARDS(LOOM_WASM_CMPF_OGT_GUARD,
                          LOOM_VECTOR_CMPF_PREDICATE_OGT),
    LOOM_WASM_CMPF_GUARDS(LOOM_WASM_CMPF_OGE_GUARD,
                          LOOM_VECTOR_CMPF_PREDICATE_OGE),
    LOOM_WASM_CMPF_GUARDS(LOOM_WASM_CMPF_OLT_GUARD,
                          LOOM_VECTOR_CMPF_PREDICATE_OLT),
    LOOM_WASM_CMPF_GUARDS(LOOM_WASM_CMPF_OLE_GUARD,
                          LOOM_VECTOR_CMPF_PREDICATE_OLE),
};

#undef LOOM_WASM_CMPF_GUARDS
#undef LOOM_WASM_CMPI_GUARDS
#undef LOOM_WASM_SELECT_GUARDS
#undef LOOM_WASM_ATTR_ENUM_GUARD
#undef LOOM_WASM_VALUE_TYPE_GUARD

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
  LOOM_WASM_EMIT_I32X4_EQ = 10,
  LOOM_WASM_EMIT_I32X4_NE = 11,
  LOOM_WASM_EMIT_I32X4_LT_S = 12,
  LOOM_WASM_EMIT_I32X4_LE_S = 13,
  LOOM_WASM_EMIT_I32X4_GT_S = 14,
  LOOM_WASM_EMIT_I32X4_GE_S = 15,
  LOOM_WASM_EMIT_I32X4_LT_U = 16,
  LOOM_WASM_EMIT_I32X4_LE_U = 17,
  LOOM_WASM_EMIT_I32X4_GT_U = 18,
  LOOM_WASM_EMIT_I32X4_GE_U = 19,
  LOOM_WASM_EMIT_F32X4_EQ = 20,
  LOOM_WASM_EMIT_F32X4_GT = 21,
  LOOM_WASM_EMIT_F32X4_GE = 22,
  LOOM_WASM_EMIT_F32X4_LT = 23,
  LOOM_WASM_EMIT_F32X4_LE = 24,
  LOOM_WASM_EMIT_V128_BITSELECT = 25,
  LOOM_WASM_EMIT_F32_ADD = 26,
};

#define LOOM_WASM_UNARY_EMIT(descriptor_name)    \
  {                                              \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP, \
      .descriptor_id = descriptor_name,          \
      .operand_ref_start = LOOM_WASM_OPERAND0,   \
      .operand_ref_count = 1,                    \
      .result_ref_start = LOOM_WASM_RESULT0,     \
      .result_ref_count = 1,                     \
  }

#define LOOM_WASM_BINARY_EMIT(descriptor_name)   \
  {                                              \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP, \
      .descriptor_id = descriptor_name,          \
      .operand_ref_start = LOOM_WASM_OPERAND0,   \
      .operand_ref_count = 2,                    \
      .result_ref_start = LOOM_WASM_RESULT0,     \
      .result_ref_count = 1,                     \
  }

#define LOOM_WASM_SELECT_EMIT(descriptor_name)          \
  {                                                     \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,        \
      .descriptor_id = descriptor_name,                 \
      .operand_ref_start = LOOM_WASM_SELECT_TRUE_VALUE, \
      .operand_ref_count = 3,                           \
      .result_ref_start = LOOM_WASM_RESULT0,            \
      .result_ref_count = 1,                            \
  }

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
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_ADD),
    [LOOM_WASM_EMIT_I32_SUB] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_SUB),
    [LOOM_WASM_EMIT_I32_MUL] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_MUL),
    [LOOM_WASM_EMIT_I32X4_SPLAT] =
        LOOM_WASM_UNARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_SPLAT),
    [LOOM_WASM_EMIT_F32X4_ADD] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32X4_ADD),
    [LOOM_WASM_EMIT_F32X4_MUL] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32X4_MUL),
    [LOOM_WASM_EMIT_I32X4_ADD] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_ADD),
    [LOOM_WASM_EMIT_I32X4_SUB] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_SUB),
    [LOOM_WASM_EMIT_I32X4_MUL] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_MUL),
    [LOOM_WASM_EMIT_I32X4_EQ] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_EQ),
    [LOOM_WASM_EMIT_I32X4_NE] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_NE),
    [LOOM_WASM_EMIT_I32X4_LT_S] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_LT_S),
    [LOOM_WASM_EMIT_I32X4_LE_S] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_LE_S),
    [LOOM_WASM_EMIT_I32X4_GT_S] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_GT_S),
    [LOOM_WASM_EMIT_I32X4_GE_S] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_GE_S),
    [LOOM_WASM_EMIT_I32X4_LT_U] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_LT_U),
    [LOOM_WASM_EMIT_I32X4_LE_U] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_LE_U),
    [LOOM_WASM_EMIT_I32X4_GT_U] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_GT_U),
    [LOOM_WASM_EMIT_I32X4_GE_U] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_GE_U),
    [LOOM_WASM_EMIT_F32X4_EQ] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32X4_EQ),
    [LOOM_WASM_EMIT_F32X4_GT] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32X4_GT),
    [LOOM_WASM_EMIT_F32X4_GE] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32X4_GE),
    [LOOM_WASM_EMIT_F32X4_LT] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32X4_LT),
    [LOOM_WASM_EMIT_F32X4_LE] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32X4_LE),
    [LOOM_WASM_EMIT_V128_BITSELECT] = LOOM_WASM_SELECT_EMIT(
        WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_V128_BITSELECT),
    [LOOM_WASM_EMIT_F32_ADD] =
        LOOM_WASM_BINARY_EMIT(WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32_ADD),
};

#undef LOOM_WASM_SELECT_EMIT
#undef LOOM_WASM_BINARY_EMIT
#undef LOOM_WASM_UNARY_EMIT

enum loom_wasm_rule_e {
  LOOM_WASM_RULE_SCALAR_ADDI = 0,
  LOOM_WASM_RULE_SCALAR_SUBI,
  LOOM_WASM_RULE_SCALAR_ADDF,
  LOOM_WASM_RULE_SCALAR_CONSTANT,
  LOOM_WASM_RULE_VECTOR_SPLAT,
  LOOM_WASM_RULE_VECTOR_SELECT_V4I32,
  LOOM_WASM_RULE_VECTOR_SELECT_V4F32,
  LOOM_WASM_RULE_VECTOR_CMPI_EQ,
  LOOM_WASM_RULE_VECTOR_CMPI_NE,
  LOOM_WASM_RULE_VECTOR_CMPI_SLT,
  LOOM_WASM_RULE_VECTOR_CMPI_SLE,
  LOOM_WASM_RULE_VECTOR_CMPI_SGT,
  LOOM_WASM_RULE_VECTOR_CMPI_SGE,
  LOOM_WASM_RULE_VECTOR_CMPI_ULT,
  LOOM_WASM_RULE_VECTOR_CMPI_ULE,
  LOOM_WASM_RULE_VECTOR_CMPI_UGT,
  LOOM_WASM_RULE_VECTOR_CMPI_UGE,
  LOOM_WASM_RULE_VECTOR_CMPF_OEQ,
  LOOM_WASM_RULE_VECTOR_CMPF_OGT,
  LOOM_WASM_RULE_VECTOR_CMPF_OGE,
  LOOM_WASM_RULE_VECTOR_CMPF_OLT,
  LOOM_WASM_RULE_VECTOR_CMPF_OLE,
  LOOM_WASM_RULE_VECTOR_ADDF,
  LOOM_WASM_RULE_VECTOR_MULF,
  LOOM_WASM_RULE_VECTOR_ADDI,
  LOOM_WASM_RULE_VECTOR_SUBI,
  LOOM_WASM_RULE_VECTOR_MULI,
  LOOM_WASM_RULE_INDEX_CONSTANT,
  LOOM_WASM_RULE_INDEX_ADD,
  LOOM_WASM_RULE_INDEX_SUB,
  LOOM_WASM_RULE_INDEX_MUL,
};

#define LOOM_WASM_RULE(source_op, guard, guard_count_value, emit) \
  {                                                               \
      .source_op_kind = source_op,                                \
      .guard_start = guard,                                       \
      .guard_count = guard_count_value,                           \
      .emit_start = emit,                                         \
      .emit_count = 1,                                            \
  }

static const loom_low_lower_rule_t loom_wasm_rules[] = {
    [LOOM_WASM_RULE_SCALAR_ADDI] =
        LOOM_WASM_RULE(LOOM_OP_SCALAR_ADDI, LOOM_WASM_SCALAR_LHS_GUARD, 3,
                       LOOM_WASM_EMIT_I32_ADD),
    [LOOM_WASM_RULE_SCALAR_SUBI] =
        LOOM_WASM_RULE(LOOM_OP_SCALAR_SUBI, LOOM_WASM_SCALAR_LHS_GUARD, 3,
                       LOOM_WASM_EMIT_I32_SUB),
    [LOOM_WASM_RULE_SCALAR_ADDF] =
        LOOM_WASM_RULE(LOOM_OP_SCALAR_ADDF, LOOM_WASM_SCALAR_F32_LHS_GUARD, 3,
                       LOOM_WASM_EMIT_F32_ADD),
    [LOOM_WASM_RULE_SCALAR_CONSTANT] =
        LOOM_WASM_RULE(LOOM_OP_SCALAR_CONSTANT, LOOM_WASM_CONST_VALUE_GUARD, 3,
                       LOOM_WASM_EMIT_I32_CONST),
    [LOOM_WASM_RULE_VECTOR_SPLAT] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_SPLAT, LOOM_WASM_SPLAT_SCALAR_GUARD, 2,
                       LOOM_WASM_EMIT_I32X4_SPLAT),
    [LOOM_WASM_RULE_VECTOR_SELECT_V4I32] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_SELECT, LOOM_WASM_SELECT_V4I32_GUARD, 4,
                       LOOM_WASM_EMIT_V128_BITSELECT),
    [LOOM_WASM_RULE_VECTOR_SELECT_V4F32] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_SELECT, LOOM_WASM_SELECT_V4F32_GUARD, 4,
                       LOOM_WASM_EMIT_V128_BITSELECT),
    [LOOM_WASM_RULE_VECTOR_CMPI_EQ] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPI, LOOM_WASM_CMPI_EQ_GUARD, 4,
                       LOOM_WASM_EMIT_I32X4_EQ),
    [LOOM_WASM_RULE_VECTOR_CMPI_NE] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPI, LOOM_WASM_CMPI_NE_GUARD, 4,
                       LOOM_WASM_EMIT_I32X4_NE),
    [LOOM_WASM_RULE_VECTOR_CMPI_SLT] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPI, LOOM_WASM_CMPI_SLT_GUARD, 4,
                       LOOM_WASM_EMIT_I32X4_LT_S),
    [LOOM_WASM_RULE_VECTOR_CMPI_SLE] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPI, LOOM_WASM_CMPI_SLE_GUARD, 4,
                       LOOM_WASM_EMIT_I32X4_LE_S),
    [LOOM_WASM_RULE_VECTOR_CMPI_SGT] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPI, LOOM_WASM_CMPI_SGT_GUARD, 4,
                       LOOM_WASM_EMIT_I32X4_GT_S),
    [LOOM_WASM_RULE_VECTOR_CMPI_SGE] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPI, LOOM_WASM_CMPI_SGE_GUARD, 4,
                       LOOM_WASM_EMIT_I32X4_GE_S),
    [LOOM_WASM_RULE_VECTOR_CMPI_ULT] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPI, LOOM_WASM_CMPI_ULT_GUARD, 4,
                       LOOM_WASM_EMIT_I32X4_LT_U),
    [LOOM_WASM_RULE_VECTOR_CMPI_ULE] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPI, LOOM_WASM_CMPI_ULE_GUARD, 4,
                       LOOM_WASM_EMIT_I32X4_LE_U),
    [LOOM_WASM_RULE_VECTOR_CMPI_UGT] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPI, LOOM_WASM_CMPI_UGT_GUARD, 4,
                       LOOM_WASM_EMIT_I32X4_GT_U),
    [LOOM_WASM_RULE_VECTOR_CMPI_UGE] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPI, LOOM_WASM_CMPI_UGE_GUARD, 4,
                       LOOM_WASM_EMIT_I32X4_GE_U),
    [LOOM_WASM_RULE_VECTOR_CMPF_OEQ] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPF, LOOM_WASM_CMPF_OEQ_GUARD, 4,
                       LOOM_WASM_EMIT_F32X4_EQ),
    [LOOM_WASM_RULE_VECTOR_CMPF_OGT] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPF, LOOM_WASM_CMPF_OGT_GUARD, 4,
                       LOOM_WASM_EMIT_F32X4_GT),
    [LOOM_WASM_RULE_VECTOR_CMPF_OGE] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPF, LOOM_WASM_CMPF_OGE_GUARD, 4,
                       LOOM_WASM_EMIT_F32X4_GE),
    [LOOM_WASM_RULE_VECTOR_CMPF_OLT] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPF, LOOM_WASM_CMPF_OLT_GUARD, 4,
                       LOOM_WASM_EMIT_F32X4_LT),
    [LOOM_WASM_RULE_VECTOR_CMPF_OLE] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_CMPF, LOOM_WASM_CMPF_OLE_GUARD, 4,
                       LOOM_WASM_EMIT_F32X4_LE),
    [LOOM_WASM_RULE_VECTOR_ADDF] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_ADDF, LOOM_WASM_V4F32_LHS_GUARD, 3,
                       LOOM_WASM_EMIT_F32X4_ADD),
    [LOOM_WASM_RULE_VECTOR_MULF] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_MULF, LOOM_WASM_V4F32_LHS_GUARD, 3,
                       LOOM_WASM_EMIT_F32X4_MUL),
    [LOOM_WASM_RULE_VECTOR_ADDI] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_ADDI, LOOM_WASM_V4I32_LHS_GUARD, 3,
                       LOOM_WASM_EMIT_I32X4_ADD),
    [LOOM_WASM_RULE_VECTOR_SUBI] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_SUBI, LOOM_WASM_V4I32_LHS_GUARD, 3,
                       LOOM_WASM_EMIT_I32X4_SUB),
    [LOOM_WASM_RULE_VECTOR_MULI] =
        LOOM_WASM_RULE(LOOM_OP_VECTOR_MULI, LOOM_WASM_V4I32_LHS_GUARD, 3,
                       LOOM_WASM_EMIT_I32X4_MUL),
    [LOOM_WASM_RULE_INDEX_CONSTANT] = LOOM_WASM_RULE(
        LOOM_OP_INDEX_CONSTANT, LOOM_WASM_ADDRESS_CONST_VALUE_GUARD, 3,
        LOOM_WASM_EMIT_I32_CONST),
    [LOOM_WASM_RULE_INDEX_ADD] =
        LOOM_WASM_RULE(LOOM_OP_INDEX_ADD, LOOM_WASM_ADDRESS_LHS_GUARD, 3,
                       LOOM_WASM_EMIT_I32_ADD),
    [LOOM_WASM_RULE_INDEX_SUB] =
        LOOM_WASM_RULE(LOOM_OP_INDEX_SUB, LOOM_WASM_ADDRESS_LHS_GUARD, 3,
                       LOOM_WASM_EMIT_I32_SUB),
    [LOOM_WASM_RULE_INDEX_MUL] =
        LOOM_WASM_RULE(LOOM_OP_INDEX_MUL, LOOM_WASM_ADDRESS_LHS_GUARD, 3,
                       LOOM_WASM_EMIT_I32_MUL),
};

#undef LOOM_WASM_RULE

#define LOOM_WASM_RULE_SPAN(source_op, first_rule, count_value) \
  {                                                             \
      .source_op_kind = source_op,                              \
      .rule_start = first_rule,                                 \
      .rule_count = count_value,                                \
  }

static const loom_low_lower_rule_span_t loom_wasm_rule_spans[] = {
    LOOM_WASM_RULE_SPAN(LOOM_OP_SCALAR_ADDI, LOOM_WASM_RULE_SCALAR_ADDI, 1),
    LOOM_WASM_RULE_SPAN(LOOM_OP_SCALAR_SUBI, LOOM_WASM_RULE_SCALAR_SUBI, 1),
    LOOM_WASM_RULE_SPAN(LOOM_OP_SCALAR_ADDF, LOOM_WASM_RULE_SCALAR_ADDF, 1),
    LOOM_WASM_RULE_SPAN(LOOM_OP_SCALAR_CONSTANT, LOOM_WASM_RULE_SCALAR_CONSTANT,
                        1),
    LOOM_WASM_RULE_SPAN(LOOM_OP_VECTOR_SPLAT, LOOM_WASM_RULE_VECTOR_SPLAT, 1),
    LOOM_WASM_RULE_SPAN(LOOM_OP_VECTOR_SELECT,
                        LOOM_WASM_RULE_VECTOR_SELECT_V4I32, 2),
    LOOM_WASM_RULE_SPAN(LOOM_OP_VECTOR_CMPI, LOOM_WASM_RULE_VECTOR_CMPI_EQ, 10),
    LOOM_WASM_RULE_SPAN(LOOM_OP_VECTOR_CMPF, LOOM_WASM_RULE_VECTOR_CMPF_OEQ, 5),
    LOOM_WASM_RULE_SPAN(LOOM_OP_VECTOR_ADDF, LOOM_WASM_RULE_VECTOR_ADDF, 1),
    LOOM_WASM_RULE_SPAN(LOOM_OP_VECTOR_MULF, LOOM_WASM_RULE_VECTOR_MULF, 1),
    LOOM_WASM_RULE_SPAN(LOOM_OP_VECTOR_ADDI, LOOM_WASM_RULE_VECTOR_ADDI, 1),
    LOOM_WASM_RULE_SPAN(LOOM_OP_VECTOR_SUBI, LOOM_WASM_RULE_VECTOR_SUBI, 1),
    LOOM_WASM_RULE_SPAN(LOOM_OP_VECTOR_MULI, LOOM_WASM_RULE_VECTOR_MULI, 1),
    LOOM_WASM_RULE_SPAN(LOOM_OP_INDEX_CONSTANT, LOOM_WASM_RULE_INDEX_CONSTANT,
                        1),
    LOOM_WASM_RULE_SPAN(LOOM_OP_INDEX_ADD, LOOM_WASM_RULE_INDEX_ADD, 1),
    LOOM_WASM_RULE_SPAN(LOOM_OP_INDEX_SUB, LOOM_WASM_RULE_INDEX_SUB, 1),
    LOOM_WASM_RULE_SPAN(LOOM_OP_INDEX_MUL, LOOM_WASM_RULE_INDEX_MUL, 1),
};

#undef LOOM_WASM_RULE_SPAN

static const loom_low_lower_rule_set_t loom_wasm_rule_set = {
    .flags = LOOM_LOW_LOWER_RULE_SET_FLAG_TARGET_CONTRACT_QUERY,
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

static const loom_low_lower_rule_set_t* const kWasmRuleSets[] = {
    &loom_wasm_rule_set,
};

typedef struct loom_wasm_lane_plan_t {
  // Descriptor row selected for this lane packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Low register type produced by the lane packet.
  loom_type_t result_type;
  // Module string ID for the lane immediate attribute.
  loom_string_id_t lane_attr_name_id;
  // Zero-based v4 lane selected by the source vector op.
  uint8_t lane;
} loom_wasm_lane_plan_t;

typedef struct loom_wasm_shuffle_plan_t {
  // Descriptor row selected for the shuffle packet.
  loom_low_lower_resolved_descriptor_t descriptor;
  // Low register type produced by the shuffle packet.
  loom_type_t result_type;
  // Module string IDs for the lane immediate attributes.
  loom_string_id_t lane_attr_name_ids[16];
  // Wasm byte lane selected for each i8x16.shuffle immediate.
  uint8_t byte_lanes[16];
} loom_wasm_shuffle_plan_t;

typedef struct loom_wasm_memory_access_plan_t {
  // Target-independent source memory decomposition.
  loom_low_source_memory_access_plan_t source;
  // Descriptor row selected for the load/store memory packet.
  loom_low_lower_resolved_descriptor_t memory_descriptor;
  // Descriptor row selected for i32 address constants.
  loom_low_lower_resolved_descriptor_t i32_const_descriptor;
  // Descriptor row selected for i32 address additions.
  loom_low_lower_resolved_descriptor_t i32_add_descriptor;
  // Descriptor row selected for i32 address multiplies.
  loom_low_lower_resolved_descriptor_t i32_mul_descriptor;
  // Low i32 register type used by address arithmetic packets.
  loom_type_t i32_type;
  // Low v128 register type produced by load packets, or none for stores.
  loom_type_t load_result_type;
  // Module string ID for i32 const immediate attributes.
  loom_string_id_t i32_value_attr_name_id;
} loom_wasm_memory_access_plan_t;

typedef struct loom_wasm_reduce_plan_t {
  // Descriptor row selected for extracting one vector lane.
  loom_low_lower_resolved_descriptor_t extract_descriptor;
  // Descriptor row selected for accumulating one lane into the result.
  loom_low_lower_resolved_descriptor_t add_descriptor;
  // Low scalar register type produced by each lane extraction.
  loom_type_t lane_type;
  // Module string ID for the lane immediate attribute.
  loom_string_id_t lane_attr_name_id;
} loom_wasm_reduce_plan_t;

static bool loom_wasm_memory_space_is_linear(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
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

static iree_status_t loom_wasm_select_memory_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_source_memory_operation_kind_t operation_kind,
    uint64_t memory_descriptor_id, loom_wasm_memory_access_plan_t* out_plan,
    bool* out_selected) {
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_plan = (loom_wasm_memory_access_plan_t){
      .i32_type = loom_type_none(),
      .load_result_type = loom_type_none(),
      .i32_value_attr_name_id = LOOM_STRING_ID_INVALID,
  };
  *out_selected = false;

  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_low_source_memory_access_plan_build(
          module, loom_low_lower_context_fact_table(context), source_op,
          &out_plan->source, &diagnostic)) {
    return iree_ok_status();
  }
  if (out_plan->source.operation_kind != operation_kind ||
      !loom_wasm_memory_access_shape_is_v128(&out_plan->source) ||
      !loom_wasm_memory_space_is_linear(out_plan->source.memory_space) ||
      !loom_wasm_i64_fits_i32(out_plan->source.static_byte_offset) ||
      !loom_wasm_source_value_is_block_argument(
          module, out_plan->source.root_value_id)) {
    return iree_ok_status();
  }
  if (!loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
          &out_plan->source, out_plan->source.static_byte_offset, 32)) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < out_plan->source.dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &out_plan->source.dynamic_terms[i];
    if (term->byte_stride <= 0 || !loom_wasm_i64_fits_i32(term->byte_stride)) {
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, memory_descriptor_id, &out_plan->memory_descriptor));
  if (operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD) {
    IREE_RETURN_IF_ERROR(loom_wasm_make_v128_register_type(
        context, &out_plan->load_result_type));
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_CONST,
      &out_plan->i32_const_descriptor));
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_ADD,
      &out_plan->i32_add_descriptor));
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_MUL,
      &out_plan->i32_mul_descriptor));
  IREE_RETURN_IF_ERROR(
      loom_wasm_make_i32_register_type(context, &out_plan->i32_type));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("i32_value"), &out_plan->i32_value_attr_name_id));
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_wasm_select_static_v4_lane(loom_attribute_t static_indices,
                                            loom_value_slice_t dynamic_indices,
                                            uint8_t* out_lane) {
  IREE_ASSERT_ARGUMENT(out_lane);
  if (dynamic_indices.count != 0 ||
      static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1 ||
      static_indices.i64_array == NULL || static_indices.i64_array[0] < 0 ||
      static_indices.i64_array[0] > 3) {
    return false;
  }
  *out_lane = (uint8_t)static_indices.i64_array[0];
  return true;
}

static const iree_string_view_t kWasmShuffleLaneAttrNames[16] = {
    IREE_SVL("lane0"),  IREE_SVL("lane1"),  IREE_SVL("lane2"),
    IREE_SVL("lane3"),  IREE_SVL("lane4"),  IREE_SVL("lane5"),
    IREE_SVL("lane6"),  IREE_SVL("lane7"),  IREE_SVL("lane8"),
    IREE_SVL("lane9"),  IREE_SVL("lane10"), IREE_SVL("lane11"),
    IREE_SVL("lane12"), IREE_SVL("lane13"), IREE_SVL("lane14"),
    IREE_SVL("lane15"),
};

static iree_status_t loom_wasm_select_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_wasm_lane_plan_t* out_plan, bool* out_selected) {
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_plan = (loom_wasm_lane_plan_t){
      .result_type = loom_type_none(),
      .lane_attr_name_id = LOOM_STRING_ID_INVALID,
  };
  *out_selected = false;

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_extract_source(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_extract_result(source_op));
  uint64_t descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  if (loom_wasm_type_is_vector_4xi32(source_type) &&
      loom_wasm_type_is_address_i32(result_type)) {
    descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_EXTRACT_LANE;
    IREE_RETURN_IF_ERROR(
        loom_wasm_make_i32_register_type(context, &out_plan->result_type));
  } else if (loom_wasm_type_is_vector_4xf32(source_type) &&
             loom_wasm_type_is_scalar_f32(result_type)) {
    descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32X4_EXTRACT_LANE;
    IREE_RETURN_IF_ERROR(
        loom_wasm_make_f32_register_type(context, &out_plan->result_type));
  } else {
    return iree_ok_status();
  }
  if (!loom_wasm_select_static_v4_lane(
          loom_vector_extract_static_indices(source_op),
          loom_vector_extract_indices(source_op), &out_plan->lane)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, descriptor_id, &out_plan->descriptor));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("lane"),
                                                 &out_plan->lane_attr_name_id));
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_wasm_select_vector_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_wasm_lane_plan_t* out_plan, bool* out_selected) {
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_plan = (loom_wasm_lane_plan_t){
      .result_type = loom_type_none(),
      .lane_attr_name_id = LOOM_STRING_ID_INVALID,
  };
  *out_selected = false;

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t value_type =
      loom_module_value_type(module, loom_vector_insert_value(source_op));
  const loom_type_t dest_type =
      loom_module_value_type(module, loom_vector_insert_dest(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_insert_result(source_op));
  uint64_t descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  if (loom_wasm_type_is_address_i32(value_type) &&
      loom_wasm_type_is_vector_4xi32(dest_type) &&
      loom_wasm_type_is_vector_4xi32(result_type)) {
    descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_REPLACE_LANE;
  } else if (loom_wasm_type_is_scalar_f32(value_type) &&
             loom_wasm_type_is_vector_4xf32(dest_type) &&
             loom_wasm_type_is_vector_4xf32(result_type)) {
    descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32X4_REPLACE_LANE;
  } else {
    return iree_ok_status();
  }
  if (!loom_wasm_select_static_v4_lane(
          loom_vector_insert_static_indices(source_op),
          loom_vector_insert_indices(source_op), &out_plan->lane)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, descriptor_id, &out_plan->descriptor));
  IREE_RETURN_IF_ERROR(
      loom_wasm_make_v128_register_type(context, &out_plan->result_type));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("lane"),
                                                 &out_plan->lane_attr_name_id));
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_wasm_select_vector_shuffle(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_wasm_shuffle_plan_t* out_plan, bool* out_selected) {
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_plan = (loom_wasm_shuffle_plan_t){
      .result_type = loom_type_none(),
  };
  *out_selected = false;

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_shuffle_source(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_shuffle_result(source_op));
  if (!loom_wasm_type_is_v128_source_vector(source_type) ||
      !loom_wasm_type_is_v128_source_vector(result_type)) {
    return iree_ok_status();
  }
  loom_attribute_t source_lanes = loom_vector_shuffle_source_lanes(source_op);
  if (source_lanes.kind != LOOM_ATTR_I64_ARRAY || source_lanes.count != 4 ||
      source_lanes.i64_array == NULL) {
    return iree_ok_status();
  }
  for (uint8_t result_lane = 0; result_lane < 4; ++result_lane) {
    int64_t source_lane = source_lanes.i64_array[result_lane];
    if (source_lane < 0 || source_lane > 3) {
      return iree_ok_status();
    }
    for (uint8_t byte = 0; byte < 4; ++byte) {
      out_plan->byte_lanes[result_lane * 4 + byte] =
          (uint8_t)(source_lane * 4 + byte);
    }
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I8X16_SHUFFLE,
      &out_plan->descriptor));
  IREE_RETURN_IF_ERROR(
      loom_wasm_make_v128_register_type(context, &out_plan->result_type));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(out_plan->lane_attr_name_ids);
       ++i) {
    IREE_RETURN_IF_ERROR(
        loom_module_intern_string(module, kWasmShuffleLaneAttrNames[i],
                                  &out_plan->lane_attr_name_ids[i]));
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_wasm_select_vector_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_wasm_reduce_plan_t* out_plan, bool* out_selected) {
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT_ARGUMENT(out_selected);
  *out_plan = (loom_wasm_reduce_plan_t){
      .lane_type = loom_type_none(),
      .lane_attr_name_id = LOOM_STRING_ID_INVALID,
  };
  *out_selected = false;

  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t input_type =
      loom_module_value_type(module, loom_vector_reduce_input(source_op));
  const loom_type_t init_type =
      loom_module_value_type(module, loom_vector_reduce_init(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_reduce_result(source_op));
  uint64_t extract_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  uint64_t add_descriptor_id = LOOM_LOW_DESCRIPTOR_ID_NONE;
  loom_combining_kind_t reduce_kind = loom_vector_reduce_kind(source_op);
  switch (reduce_kind) {
    case LOOM_COMBINING_KIND_ADDI:
      if (!loom_wasm_type_is_vector_4xi32(input_type) ||
          !loom_wasm_type_is_address_i32(init_type) ||
          !loom_wasm_type_is_address_i32(result_type)) {
        return iree_ok_status();
      }
      extract_descriptor_id =
          WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32X4_EXTRACT_LANE;
      add_descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_I32_ADD;
      IREE_RETURN_IF_ERROR(
          loom_wasm_make_i32_register_type(context, &out_plan->lane_type));
      break;
    case LOOM_COMBINING_KIND_ADDF:
      if (!loom_wasm_type_is_vector_4xf32(input_type) ||
          !loom_wasm_type_is_scalar_f32(init_type) ||
          !loom_wasm_type_is_scalar_f32(result_type)) {
        return iree_ok_status();
      }
      extract_descriptor_id =
          WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32X4_EXTRACT_LANE;
      add_descriptor_id = WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_F32_ADD;
      IREE_RETURN_IF_ERROR(
          loom_wasm_make_f32_register_type(context, &out_plan->lane_type));
      break;
    default:
      return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, extract_descriptor_id, &out_plan->extract_descriptor));
  IREE_RETURN_IF_ERROR(loom_low_lower_resolve_descriptor(
      context, add_descriptor_id, &out_plan->add_descriptor));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("lane"),
                                                 &out_plan->lane_attr_name_id));
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_wasm_select_op(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_VIEW:
      *out_plan = loom_low_lower_plan_make(source_op->kind, NULL);
      return iree_ok_status();
    case LOOM_OP_VECTOR_LOAD:
    case LOOM_OP_VECTOR_STORE: {
      loom_wasm_memory_access_plan_t selected_plan = {0};
      bool selected = false;
      const bool is_load = source_op->kind == LOOM_OP_VECTOR_LOAD;
      IREE_RETURN_IF_ERROR(loom_wasm_select_memory_access(
          context, source_op,
          is_load ? LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD
                  : LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE,
          is_load ? WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_V128_LOAD
                  : WASM_CORE_SIMD128_DESCRIPTOR_ID_WASM_V128_STORE,
          &selected_plan, &selected));
      if (selected) {
        loom_wasm_memory_access_plan_t* plan = NULL;
        IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
            context, sizeof(*plan), (void**)&plan));
        *plan = selected_plan;
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_EXTRACT: {
      loom_wasm_lane_plan_t selected_plan = {0};
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_wasm_select_vector_extract(
          context, source_op, &selected_plan, &selected));
      if (selected) {
        loom_wasm_lane_plan_t* plan = NULL;
        IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
            context, sizeof(*plan), (void**)&plan));
        *plan = selected_plan;
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_INSERT: {
      loom_wasm_lane_plan_t selected_plan = {0};
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_wasm_select_vector_insert(
          context, source_op, &selected_plan, &selected));
      if (selected) {
        loom_wasm_lane_plan_t* plan = NULL;
        IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
            context, sizeof(*plan), (void**)&plan));
        *plan = selected_plan;
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_SHUFFLE: {
      loom_wasm_shuffle_plan_t selected_plan = {0};
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_wasm_select_vector_shuffle(
          context, source_op, &selected_plan, &selected));
      if (selected) {
        loom_wasm_shuffle_plan_t* plan = NULL;
        IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
            context, sizeof(*plan), (void**)&plan));
        *plan = selected_plan;
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
      }
      return iree_ok_status();
    }
    case LOOM_OP_VECTOR_REDUCE: {
      loom_wasm_reduce_plan_t selected_plan = {0};
      bool selected = false;
      IREE_RETURN_IF_ERROR(loom_wasm_select_vector_reduce(
          context, source_op, &selected_plan, &selected));
      if (selected) {
        loom_wasm_reduce_plan_t* plan = NULL;
        IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
            context, sizeof(*plan), (void**)&plan));
        *plan = selected_plan;
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static loom_named_attr_t loom_wasm_make_i64_attr(loom_string_id_t name_id,
                                                 int64_t value) {
  IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
  return (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
}

static loom_named_attr_t loom_wasm_make_i32_value_attr(loom_string_id_t name_id,
                                                       int64_t value) {
  IREE_ASSERT(loom_wasm_i64_fits_i32(value));
  return loom_wasm_make_i64_attr(name_id, value);
}

static iree_status_t loom_wasm_emit_resolved_i32_const(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t value_attr_name_id, loom_type_t result_type, int64_t value,
    loom_location_id_t location, loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr =
      loom_wasm_make_i32_value_attr(value_attr_name_id, value);
  loom_op_t* const_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, descriptor, loom_make_named_attr_slice(&attr, 1), result_type,
      location, &const_op));
  *out_value_id = loom_low_const_result(const_op);
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_resolved_typed_binary(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &result_type, 1, NULL, 0, location, &op));
  *out_value_id = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_resolved_typed_extract_lane(
    loom_low_lower_context_t* context,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_value_id_t source, uint8_t lane, loom_string_id_t lane_attr_name_id,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value_id) {
  IREE_ASSERT_ARGUMENT(out_value_id);
  *out_value_id = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attr = loom_wasm_make_i64_attr(lane_attr_name_id, lane);
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, descriptor, &source, 1, loom_make_named_attr_slice(&attr, 1),
      &result_type, 1, NULL, 0, location, &op));
  *out_value_id = loom_value_slice_get(loom_low_op_results(op), 0);
  return iree_ok_status();
}

static iree_status_t loom_wasm_make_shuffle_attrs(
    loom_low_lower_context_t* context, const loom_wasm_shuffle_plan_t* plan,
    loom_named_attr_slice_t* out_attrs) {
  IREE_ASSERT_ARGUMENT(out_attrs);
  loom_named_attr_t* attrs = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, IREE_ARRAYSIZE(plan->byte_lanes), sizeof(*attrs),
      (void**)&attrs));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(plan->byte_lanes); ++i) {
    attrs[i] = loom_wasm_make_i64_attr(plan->lane_attr_name_ids[i],
                                       plan->byte_lanes[i]);
  }
  *out_attrs =
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(plan->byte_lanes));
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_address_offset(
    loom_low_lower_context_t* context,
    const loom_wasm_memory_access_plan_t* plan, loom_location_id_t location,
    loom_value_id_t* inout_address) {
  if (plan->source.static_byte_offset == 0) {
    return iree_ok_status();
  }
  loom_value_id_t offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_emit_resolved_i32_const(
      context, &plan->i32_const_descriptor, plan->i32_value_attr_name_id,
      plan->i32_type, plan->source.static_byte_offset, location, &offset));
  return loom_wasm_emit_resolved_typed_binary(
      context, &plan->i32_add_descriptor, *inout_address, offset,
      plan->i32_type, location, inout_address);
}

static iree_status_t loom_wasm_emit_dynamic_address_offset(
    loom_low_lower_context_t* context,
    const loom_wasm_memory_access_plan_t* plan, loom_location_id_t location,
    loom_value_id_t* inout_address) {
  if (!loom_low_source_memory_access_is_dynamic(&plan->source)) {
    return iree_ok_status();
  }
  for (uint8_t i = 0; i < plan->source.dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &plan->source.dynamic_terms[i];
    loom_value_id_t dynamic_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, term->index, &dynamic_index));
    loom_value_id_t dynamic_offset = dynamic_index;
    if (term->byte_stride != 1) {
      loom_value_id_t stride = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_wasm_emit_resolved_i32_const(
          context, &plan->i32_const_descriptor, plan->i32_value_attr_name_id,
          plan->i32_type, term->byte_stride, location, &stride));
      IREE_RETURN_IF_ERROR(loom_wasm_emit_resolved_typed_binary(
          context, &plan->i32_mul_descriptor, dynamic_index, stride,
          plan->i32_type, location, &dynamic_offset));
    }
    IREE_RETURN_IF_ERROR(loom_wasm_emit_resolved_typed_binary(
        context, &plan->i32_add_descriptor, *inout_address, dynamic_offset,
        plan->i32_type, location, inout_address));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_memory_address(
    loom_low_lower_context_t* context,
    const loom_wasm_memory_access_plan_t* plan, loom_location_id_t location,
    loom_value_id_t* out_address) {
  IREE_ASSERT_ARGUMENT(out_address);
  *out_address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, plan->source.root_value_id, out_address));
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
    const loom_wasm_memory_access_plan_t* plan) {
  loom_value_id_t address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_emit_memory_address(
      context, plan, source_op->location, &address));
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->memory_descriptor, &address, 1,
      loom_named_attr_slice_empty(), &plan->load_result_type, 1, NULL, 0,
      source_op->location, &load_op));
  IREE_RETURN_IF_ERROR(loom_low_lower_record_source_memory_access(
      context, load_op, &plan->source));
  return loom_low_lower_bind_value(
      context, loom_vector_load_result(source_op),
      loom_value_slice_get(loom_low_op_results(load_op), 0));
}

static iree_status_t loom_wasm_lower_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_wasm_memory_access_plan_t* plan) {
  loom_value_id_t address = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_emit_memory_address(
      context, plan, source_op->location, &address));
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_value(source_op), &value));
  loom_value_id_t operands[] = {address, value};
  loom_op_t* store_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->memory_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), NULL, 0, NULL, 0, source_op->location,
      &store_op));
  return loom_low_lower_record_source_memory_access(context, store_op,
                                                    &plan->source);
}

static iree_status_t loom_wasm_lower_vector_extract(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_wasm_lane_plan_t* plan) {
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_extract_source(source_op), &source));
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_emit_resolved_typed_extract_lane(
      context, &plan->descriptor, source, plan->lane, plan->lane_attr_name_id,
      plan->result_type, source_op->location, &result));
  return loom_low_lower_bind_value(
      context, loom_vector_extract_result(source_op), result);
}

static iree_status_t loom_wasm_lower_vector_insert(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_wasm_lane_plan_t* plan) {
  loom_value_id_t dest = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_insert_dest(source_op), &dest));
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_insert_value(source_op), &value));
  loom_named_attr_t attr =
      loom_wasm_make_i64_attr(plan->lane_attr_name_id, plan->lane);
  loom_value_id_t operands[] = {dest, value};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(&attr, 1), &plan->result_type, 1, NULL, 0,
      source_op->location, &op));
  return loom_low_lower_bind_value(
      context, loom_vector_insert_result(source_op),
      loom_value_slice_get(loom_low_op_results(op), 0));
}

static iree_status_t loom_wasm_lower_vector_shuffle(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_wasm_shuffle_plan_t* plan) {
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_shuffle_source(source_op), &source));
  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  IREE_RETURN_IF_ERROR(loom_wasm_make_shuffle_attrs(context, plan, &attrs));
  loom_value_id_t operands[] = {source, source};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &plan->descriptor, operands, IREE_ARRAYSIZE(operands), attrs,
      &plan->result_type, 1, NULL, 0, source_op->location, &op));
  return loom_low_lower_bind_value(
      context, loom_vector_shuffle_result(source_op),
      loom_value_slice_get(loom_low_op_results(op), 0));
}

static iree_status_t loom_wasm_lower_vector_reduce(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_wasm_reduce_plan_t* plan) {
  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_reduce_input(source_op), &input));
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_reduce_init(source_op), &accumulator));
  for (uint8_t lane = 0; lane < 4; ++lane) {
    loom_value_id_t lane_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_wasm_emit_resolved_typed_extract_lane(
        context, &plan->extract_descriptor, input, lane,
        plan->lane_attr_name_id, plan->lane_type, source_op->location,
        &lane_value));
    IREE_RETURN_IF_ERROR(loom_wasm_emit_resolved_typed_binary(
        context, &plan->add_descriptor, accumulator, lane_value,
        plan->lane_type, source_op->location, &accumulator));
  }
  return loom_low_lower_bind_value(
      context, loom_vector_reduce_result(source_op), accumulator);
}

static iree_status_t loom_wasm_emit_op(void* user_data,
                                       loom_low_lower_context_t* context,
                                       const loom_op_t* source_op,
                                       loom_low_lower_plan_t plan) {
  (void)user_data;
  switch (plan.id) {
    case LOOM_OP_BUFFER_VIEW:
      return loom_wasm_lower_buffer_alias(context,
                                          loom_buffer_view_buffer(source_op),
                                          loom_buffer_view_result(source_op));
    case LOOM_OP_VECTOR_LOAD:
      return loom_wasm_lower_vector_load(
          context, source_op,
          (const loom_wasm_memory_access_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_STORE:
      return loom_wasm_lower_vector_store(
          context, source_op,
          (const loom_wasm_memory_access_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_EXTRACT:
      return loom_wasm_lower_vector_extract(
          context, source_op, (const loom_wasm_lane_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_INSERT:
      return loom_wasm_lower_vector_insert(
          context, source_op, (const loom_wasm_lane_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_SHUFFLE:
      return loom_wasm_lower_vector_shuffle(
          context, source_op,
          (const loom_wasm_shuffle_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_REDUCE:
      return loom_wasm_lower_vector_reduce(
          context, source_op, (const loom_wasm_reduce_plan_t*)plan.target_data);
    default:
      IREE_CHECK_UNREACHABLE();
  }
}

static const loom_low_lower_policy_t kWasmLowLowerPolicy = {
    .name = IREE_SVL("wasm-lower"),
    .map_type = {.fn = loom_wasm_map_type, .user_data = NULL},
    .map_argument = {.fn = loom_wasm_map_argument, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kWasmRuleSets),
            .values = kWasmRuleSets,
        },
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
