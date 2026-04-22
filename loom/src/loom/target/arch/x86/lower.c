// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/lower.h"

#include <stdint.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower_rules.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/avx512_descriptors.h"
#include "loom/target/arch/x86/packed_dot_contract.h"
#include "loom/target/arch/x86/packed_dot_descriptors.h"
#include "loom/target/arch/x86/packed_dot_vector.h"

#define LOOM_X86_CONTRACT_SET_AVX512_CORE IREE_SV("x86.avx512.core")
#define LOOM_X86_CONTRACT_SET_PACKED_DOT_CORE IREE_SV("x86.packed_dot.core")

static bool loom_x86_contract_set_key_is_avx512_core(
    iree_string_view_t contract_set_key) {
  return iree_string_view_equal(contract_set_key,
                                LOOM_X86_CONTRACT_SET_AVX512_CORE);
}

static bool loom_x86_contract_set_key_is_packed_dot_core(
    iree_string_view_t contract_set_key) {
  return iree_string_view_equal(contract_set_key,
                                LOOM_X86_CONTRACT_SET_PACKED_DOT_CORE);
}

static iree_string_view_t loom_x86_legality_contract_set_key(
    const loom_target_low_legality_context_t* context) {
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (bundle == NULL || bundle->config == NULL) {
    return iree_string_view_empty();
  }
  return bundle->config->contract_set_key;
}

static bool loom_x86_contract_set_key_is_x86(
    iree_string_view_t contract_set_key) {
  return loom_x86_contract_set_key_is_avx512_core(contract_set_key) ||
         loom_x86_contract_set_key_is_packed_dot_core(contract_set_key);
}

static bool loom_x86_scalar_type_has_packed_dot_register_width(
    loom_scalar_type_t scalar_type, uint32_t* out_bit_width) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I8:
      *out_bit_width = 8;
      return true;
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      *out_bit_width = 16;
      return true;
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_F32:
      *out_bit_width = 32;
      return true;
    default:
      *out_bit_width = 0;
      return false;
  }
}

static bool loom_x86_type_static_vector_bit_width(loom_type_t type,
                                                  uint32_t* out_bit_width) {
  *out_bit_width = 0;
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type)) {
    return false;
  }
  uint32_t element_bit_width = 0;
  if (!loom_x86_scalar_type_has_packed_dot_register_width(
          loom_type_element_type(type), &element_bit_width)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count <= 0 ||
      (uint64_t)lane_count > UINT32_MAX / element_bit_width) {
    return false;
  }
  *out_bit_width = (uint32_t)lane_count * element_bit_width;
  return true;
}

static bool loom_x86_type_is_vector_16xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 16;
}

static bool loom_x86_type_is_vector_16xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 16;
}

static bool loom_x86_type_is_address_gpr64(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  switch (loom_type_element_type(type)) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_x86_make_gpr64_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, X86_AVX512_CORE_REG_CLASS_ID_X86_GPR64, 1, out_type);
}

static iree_status_t loom_x86_make_zmm_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, X86_AVX512_CORE_REG_CLASS_ID_X86_ZMM, 1, out_type);
}

static iree_status_t loom_x86_map_avx512_type(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_type_t source_type,
                                              loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_x86_type_is_address_gpr64(source_type)) {
    return loom_x86_make_gpr64_register_type(context, out_low_type);
  }
  if (loom_x86_type_is_vector_16xi32(source_type) ||
      loom_x86_type_is_vector_16xf32(source_type)) {
    return loom_x86_make_zmm_register_type(context, out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("x86 AVX512 lowering currently supports only index/offset "
              "address values and vector<16xi32>/vector<16xf32> values"));
}

static iree_status_t loom_x86_map_avx512_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  (void)source_argument_index;
  IREE_ASSERT_ARGUMENT(out_argument);
  const loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t address_type = loom_type_none();
    IREE_RETURN_IF_ERROR(
        loom_x86_make_gpr64_register_type(context, &address_type));
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
  return loom_x86_map_avx512_type(user_data, context, source_function_op,
                                  source_type, &out_argument->abi_type);
}

static iree_status_t loom_x86_map_packed_dot_type(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_type_t source_type,
    loom_type_t* out_low_type) {
  (void)user_data;
  uint32_t vector_bit_width = 0;
  if (loom_x86_type_static_vector_bit_width(source_type, &vector_bit_width)) {
    switch (vector_bit_width) {
      case 128:
        return loom_low_lower_make_register_type(
            context, X86_PACKED_DOT_CORE_REG_CLASS_ID_X86_XMM, 1, out_low_type);
      case 256:
        return loom_low_lower_make_register_type(
            context, X86_PACKED_DOT_CORE_REG_CLASS_ID_X86_YMM, 1, out_low_type);
      case 512:
        return loom_low_lower_make_register_type(
            context, X86_PACKED_DOT_CORE_REG_CLASS_ID_X86_ZMM, 1, out_low_type);
      default:
        break;
    }
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("x86 packed-dot lowering requires a static 128-, 256-, or "
              "512-bit i8/i16/f16/bf16/i32/f32 vector"));
}

enum loom_x86_avx512_type_pattern_e {
  LOOM_X86_AVX512_TYPE_V16I32 = 0,
  LOOM_X86_AVX512_TYPE_V16F32 = 1,
  LOOM_X86_AVX512_TYPE_ADDRESS_GPR64 = 2,
};

static const loom_low_lower_type_pattern_t loom_x86_avx512_type_patterns[] = {
    [LOOM_X86_AVX512_TYPE_V16I32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0,
            .type_kind = LOOM_TYPE_VECTOR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I32),
            .rank = 1,
            .static_dim0 = 16,
        },
    [LOOM_X86_AVX512_TYPE_V16F32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0,
            .type_kind = LOOM_TYPE_VECTOR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_F32),
            .rank = 1,
            .static_dim0 = 16,
        },
    [LOOM_X86_AVX512_TYPE_ADDRESS_GPR64] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_INDEX) |
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_OFFSET),
        },
};

enum loom_x86_avx512_value_ref_e {
  LOOM_X86_AVX512_OPERAND0 = 0,
  LOOM_X86_AVX512_OPERAND1 = 1,
  LOOM_X86_AVX512_RESULT0 = 2,
  LOOM_X86_AVX512_OPERAND2 = 3,
  LOOM_X86_AVX512_FMA_ACCUMULATOR = 4,
  LOOM_X86_AVX512_FMA_LHS = 5,
  LOOM_X86_AVX512_FMA_RHS = 6,
  LOOM_X86_AVX512_TEMPORARY0 = 7,
  LOOM_X86_AVX512_TEMPORARY0_THEN_OPERAND2 = 8,
  LOOM_X86_AVX512_OPERAND2_AFTER_TEMPORARY0 = 9,
};

static const loom_low_lower_value_ref_t loom_x86_avx512_value_refs[] = {
    [LOOM_X86_AVX512_OPERAND0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
        },
    [LOOM_X86_AVX512_OPERAND1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
        },
    [LOOM_X86_AVX512_RESULT0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
            .index = 0,
        },
    [LOOM_X86_AVX512_OPERAND2] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 2,
        },
    [LOOM_X86_AVX512_FMA_ACCUMULATOR] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 2,
        },
    [LOOM_X86_AVX512_FMA_LHS] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
        },
    [LOOM_X86_AVX512_FMA_RHS] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
        },
    [LOOM_X86_AVX512_TEMPORARY0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 0,
        },
    [LOOM_X86_AVX512_TEMPORARY0_THEN_OPERAND2] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 0,
        },
    [LOOM_X86_AVX512_OPERAND2_AFTER_TEMPORARY0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 2,
        },
};

static const loom_low_lower_attr_copy_t loom_x86_avx512_attr_copies[] = {
    {
        .target_name = IREE_SVL("imm64"),
        .source_attr_index = 0,
    },
};

enum loom_x86_avx512_diagnostic_e {
  LOOM_X86_AVX512_DIAGNOSTIC_V16I32 = 0,
  LOOM_X86_AVX512_DIAGNOSTIC_V16F32 = 1,
  LOOM_X86_AVX512_DIAGNOSTIC_I64_ATTR = 2,
  LOOM_X86_AVX512_DIAGNOSTIC_ADDRESS_GPR64 = 3,
  LOOM_X86_AVX512_DIAGNOSTIC_IMM64_RANGE = 4,
};

static const loom_low_lower_diagnostic_t loom_x86_avx512_diagnostics[] = {
    [LOOM_X86_AVX512_DIAGNOSTIC_V16I32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("vector<16xi32>"),
            .reason =
                IREE_SVL("x86 AVX512 integer lowering requires vector<16xi32> "
                         "values"),
        },
    [LOOM_X86_AVX512_DIAGNOSTIC_V16F32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("vector<16xf32>"),
            .reason = IREE_SVL("x86 AVX512 floating-point lowering requires "
                               "vector<16xf32> values"),
        },
    [LOOM_X86_AVX512_DIAGNOSTIC_I64_ATTR] =
        {
            .subject_kind = IREE_SVL("attr"),
            .subject_name = IREE_SVL("value"),
            .reason = IREE_SVL("x86 AVX512 constant lowering requires an i64 "
                               "attribute value"),
        },
    [LOOM_X86_AVX512_DIAGNOSTIC_ADDRESS_GPR64] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("index_or_offset"),
            .reason = IREE_SVL("x86 AVX512 address lowering requires index or "
                               "offset scalar values"),
        },
    [LOOM_X86_AVX512_DIAGNOSTIC_IMM64_RANGE] =
        {
            .subject_kind = IREE_SVL("attr"),
            .subject_name = IREE_SVL("value"),
            .reason = IREE_SVL("x86 AVX512 imm64 constants must fit in the "
                               "signed 64-bit descriptor range"),
        },
};

enum loom_x86_avx512_guard_e {
  LOOM_X86_AVX512_I32_LHS_GUARD = 0,
  LOOM_X86_AVX512_I32_RHS_GUARD = 1,
  LOOM_X86_AVX512_I32_RESULT_GUARD = 2,
  LOOM_X86_AVX512_F32_LHS_GUARD = 3,
  LOOM_X86_AVX512_F32_RHS_GUARD = 4,
  LOOM_X86_AVX512_F32_RESULT_GUARD = 5,
  LOOM_X86_AVX512_F32_ACCUMULATOR_GUARD = 6,
  LOOM_X86_AVX512_ADDRESS_CONST_VALUE_GUARD = 7,
  LOOM_X86_AVX512_ADDRESS_CONST_RESULT_GUARD = 8,
  LOOM_X86_AVX512_ADDRESS_CONST_RANGE_GUARD = 9,
  LOOM_X86_AVX512_ADDRESS_LHS_GUARD = 10,
  LOOM_X86_AVX512_ADDRESS_RHS_GUARD = 11,
  LOOM_X86_AVX512_ADDRESS_RESULT_GUARD = 12,
  LOOM_X86_AVX512_ADDRESS_MADD_LHS_GUARD = 13,
  LOOM_X86_AVX512_ADDRESS_MADD_RHS_GUARD = 14,
  LOOM_X86_AVX512_ADDRESS_MADD_ACC_GUARD = 15,
  LOOM_X86_AVX512_ADDRESS_MADD_RESULT_GUARD = 16,
};

static const loom_low_lower_guard_t loom_x86_avx512_guards[] = {
    [LOOM_X86_AVX512_I32_LHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_OPERAND0,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_V16I32,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_V16I32,
        },
    [LOOM_X86_AVX512_I32_RHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_OPERAND1,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_V16I32,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_V16I32,
        },
    [LOOM_X86_AVX512_I32_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_RESULT0,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_V16I32,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_V16I32,
        },
    [LOOM_X86_AVX512_F32_LHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_OPERAND0,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_V16F32,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_V16F32,
        },
    [LOOM_X86_AVX512_F32_RHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_OPERAND1,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_V16F32,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_V16F32,
        },
    [LOOM_X86_AVX512_F32_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_RESULT0,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_V16F32,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_V16F32,
        },
    [LOOM_X86_AVX512_F32_ACCUMULATOR_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_OPERAND2,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_V16F32,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_V16F32,
        },
    [LOOM_X86_AVX512_ADDRESS_CONST_VALUE_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_ATTR_KIND,
            .attr_index = 0,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_I64_ATTR,
            .attr_kind = LOOM_ATTR_I64,
        },
    [LOOM_X86_AVX512_ADDRESS_CONST_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_RESULT0,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_ADDRESS_GPR64,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_ADDRESS_GPR64,
        },
    [LOOM_X86_AVX512_ADDRESS_CONST_RANGE_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE,
            .attr_index = 0,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_IMM64_RANGE,
            .minimum_i64 = INT64_MIN + 1,
            .maximum_i64 = INT64_MAX,
        },
    [LOOM_X86_AVX512_ADDRESS_LHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_OPERAND0,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_ADDRESS_GPR64,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_ADDRESS_GPR64,
        },
    [LOOM_X86_AVX512_ADDRESS_RHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_OPERAND1,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_ADDRESS_GPR64,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_ADDRESS_GPR64,
        },
    [LOOM_X86_AVX512_ADDRESS_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_RESULT0,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_ADDRESS_GPR64,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_ADDRESS_GPR64,
        },
    [LOOM_X86_AVX512_ADDRESS_MADD_LHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_OPERAND0,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_ADDRESS_GPR64,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_ADDRESS_GPR64,
        },
    [LOOM_X86_AVX512_ADDRESS_MADD_RHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_OPERAND1,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_ADDRESS_GPR64,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_ADDRESS_GPR64,
        },
    [LOOM_X86_AVX512_ADDRESS_MADD_ACC_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_OPERAND2,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_ADDRESS_GPR64,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_ADDRESS_GPR64,
        },
    [LOOM_X86_AVX512_ADDRESS_MADD_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_RESULT0,
            .type_pattern_index = LOOM_X86_AVX512_TYPE_ADDRESS_GPR64,
            .diagnostic_index = LOOM_X86_AVX512_DIAGNOSTIC_ADDRESS_GPR64,
        },
};

static const loom_tied_result_t loom_x86_avx512_tied_results[] = {
    {
        .result_index = 0,
        .operand_index = 0,
        .has_type_change = false,
    },
};

enum loom_x86_avx512_emit_e {
  LOOM_X86_AVX512_EMIT_VADDPS = 0,
  LOOM_X86_AVX512_EMIT_VSUBPS = 1,
  LOOM_X86_AVX512_EMIT_VMULPS = 2,
  LOOM_X86_AVX512_EMIT_VFMADD231PS = 3,
  LOOM_X86_AVX512_EMIT_VPADDD = 4,
  LOOM_X86_AVX512_EMIT_VPSUBD = 5,
  LOOM_X86_AVX512_EMIT_VPMULLD = 6,
  LOOM_X86_AVX512_EMIT_MOVIMM_GPR64 = 7,
  LOOM_X86_AVX512_EMIT_LEA_ADD_GPR64 = 8,
  LOOM_X86_AVX512_EMIT_IMUL_GPR64 = 9,
  LOOM_X86_AVX512_EMIT_IMUL_TEMP_GPR64 = 10,
  LOOM_X86_AVX512_EMIT_LEA_ADD_TEMP_GPR64 = 11,
};

static const loom_low_lower_emit_t loom_x86_avx512_emits[] = {
    [LOOM_X86_AVX512_EMIT_VADDPS] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDPS_ZMM,
            .operand_ref_start = LOOM_X86_AVX512_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_X86_AVX512_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_X86_AVX512_EMIT_VSUBPS] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VSUBPS_ZMM,
            .operand_ref_start = LOOM_X86_AVX512_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_X86_AVX512_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_X86_AVX512_EMIT_VMULPS] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VMULPS_ZMM,
            .operand_ref_start = LOOM_X86_AVX512_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_X86_AVX512_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_X86_AVX512_EMIT_VFMADD231PS] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VFMADD231PS_ZMM,
            .operand_ref_start = LOOM_X86_AVX512_FMA_ACCUMULATOR,
            .operand_ref_count = 3,
            .copy_operand_mask = 0x1,
            .result_ref_start = LOOM_X86_AVX512_RESULT0,
            .result_ref_count = 1,
            .tied_result_start = 0,
            .tied_result_count = 1,
        },
    [LOOM_X86_AVX512_EMIT_VPADDD] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPADDD_ZMM,
            .operand_ref_start = LOOM_X86_AVX512_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_X86_AVX512_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_X86_AVX512_EMIT_VPSUBD] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPSUBD_ZMM,
            .operand_ref_start = LOOM_X86_AVX512_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_X86_AVX512_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_X86_AVX512_EMIT_VPMULLD] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPMULLD_ZMM,
            .operand_ref_start = LOOM_X86_AVX512_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_X86_AVX512_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_X86_AVX512_EMIT_MOVIMM_GPR64] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_MOVIMM_GPR64,
            .result_ref_start = LOOM_X86_AVX512_RESULT0,
            .result_ref_count = 1,
            .attr_copy_start = 0,
            .attr_copy_count = 1,
        },
    [LOOM_X86_AVX512_EMIT_LEA_ADD_GPR64] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_LEA_ADD_GPR64,
            .operand_ref_start = LOOM_X86_AVX512_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_X86_AVX512_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_X86_AVX512_EMIT_IMUL_GPR64] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_IMUL_GPR64,
            .operand_ref_start = LOOM_X86_AVX512_OPERAND0,
            .operand_ref_count = 2,
            .copy_operand_mask = 0x1,
            .result_ref_start = LOOM_X86_AVX512_RESULT0,
            .result_ref_count = 1,
            .tied_result_start = 0,
            .tied_result_count = 1,
        },
    [LOOM_X86_AVX512_EMIT_IMUL_TEMP_GPR64] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .flags = LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_IMUL_GPR64,
            .operand_ref_start = LOOM_X86_AVX512_OPERAND0,
            .operand_ref_count = 2,
            .copy_operand_mask = 0x1,
            .result_ref_start = LOOM_X86_AVX512_RESULT0,
            .result_ref_count = 1,
            .result_bind_ref_start = LOOM_X86_AVX512_TEMPORARY0,
            .tied_result_start = 0,
            .tied_result_count = 1,
        },
    [LOOM_X86_AVX512_EMIT_LEA_ADD_TEMP_GPR64] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_LEA_ADD_GPR64,
            .operand_ref_start = LOOM_X86_AVX512_TEMPORARY0_THEN_OPERAND2,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_X86_AVX512_RESULT0,
            .result_ref_count = 1,
        },
};

static const loom_low_lower_rule_t loom_x86_avx512_rules[] = {
    {
        .source_op_kind = LOOM_OP_VECTOR_ADDF,
        .guard_start = LOOM_X86_AVX512_F32_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_X86_AVX512_EMIT_VADDPS,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SUBF,
        .guard_start = LOOM_X86_AVX512_F32_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_X86_AVX512_EMIT_VSUBPS,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_MULF,
        .guard_start = LOOM_X86_AVX512_F32_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_X86_AVX512_EMIT_VMULPS,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_FMAF,
        .guard_start = LOOM_X86_AVX512_F32_LHS_GUARD,
        .guard_count = 4,
        .emit_start = LOOM_X86_AVX512_EMIT_VFMADD231PS,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_ADDI,
        .guard_start = LOOM_X86_AVX512_I32_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_X86_AVX512_EMIT_VPADDD,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SUBI,
        .guard_start = LOOM_X86_AVX512_I32_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_X86_AVX512_EMIT_VPSUBD,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_MULI,
        .guard_start = LOOM_X86_AVX512_I32_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_X86_AVX512_EMIT_VPMULLD,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_CONSTANT,
        .guard_start = LOOM_X86_AVX512_ADDRESS_CONST_VALUE_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_X86_AVX512_EMIT_MOVIMM_GPR64,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_ADD,
        .guard_start = LOOM_X86_AVX512_ADDRESS_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_X86_AVX512_EMIT_LEA_ADD_GPR64,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_MUL,
        .guard_start = LOOM_X86_AVX512_ADDRESS_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_X86_AVX512_EMIT_IMUL_GPR64,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_MADD,
        .temporary_count = 1,
        .guard_start = LOOM_X86_AVX512_ADDRESS_MADD_LHS_GUARD,
        .guard_count = 4,
        .emit_start = LOOM_X86_AVX512_EMIT_IMUL_TEMP_GPR64,
        .emit_count = 2,
    },
};

static const loom_low_lower_rule_span_t loom_x86_avx512_rule_spans[] = {
    {
        .source_op_kind = LOOM_OP_VECTOR_ADDF,
        .rule_start = 0,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SUBF,
        .rule_start = 1,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_MULF,
        .rule_start = 2,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_FMAF,
        .rule_start = 3,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_ADDI,
        .rule_start = 4,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SUBI,
        .rule_start = 5,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_MULI,
        .rule_start = 6,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_CONSTANT,
        .rule_start = 7,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_ADD,
        .rule_start = 8,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_MUL,
        .rule_start = 9,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_MADD,
        .rule_start = 10,
        .rule_count = 1,
    },
};

static const loom_low_lower_rule_set_t loom_x86_avx512_rule_set = {
    .spans = loom_x86_avx512_rule_spans,
    .span_count = IREE_ARRAYSIZE(loom_x86_avx512_rule_spans),
    .rules = loom_x86_avx512_rules,
    .rule_count = IREE_ARRAYSIZE(loom_x86_avx512_rules),
    .type_patterns = loom_x86_avx512_type_patterns,
    .type_pattern_count = IREE_ARRAYSIZE(loom_x86_avx512_type_patterns),
    .value_refs = loom_x86_avx512_value_refs,
    .value_ref_count = IREE_ARRAYSIZE(loom_x86_avx512_value_refs),
    .guards = loom_x86_avx512_guards,
    .guard_count = IREE_ARRAYSIZE(loom_x86_avx512_guards),
    .attr_copies = loom_x86_avx512_attr_copies,
    .attr_copy_count = IREE_ARRAYSIZE(loom_x86_avx512_attr_copies),
    .tied_results = loom_x86_avx512_tied_results,
    .tied_result_count = IREE_ARRAYSIZE(loom_x86_avx512_tied_results),
    .emits = loom_x86_avx512_emits,
    .emit_count = IREE_ARRAYSIZE(loom_x86_avx512_emits),
    .diagnostics = loom_x86_avx512_diagnostics,
    .diagnostic_count = IREE_ARRAYSIZE(loom_x86_avx512_diagnostics),
};

static bool loom_x86_memory_space_is_object_memory(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      return true;
    default:
      return false;
  }
}

static bool loom_x86_i64_fits_disp32(int64_t value) {
  return value >= INT32_MIN && value <= INT32_MAX;
}

static bool loom_x86_memory_access_shape_is_zmm32(
    const loom_low_source_memory_access_plan_t* plan) {
  return plan->element_byte_count == 4 && plan->vector_lane_count == 16 &&
         plan->vector_lane_byte_stride == 4;
}

typedef enum loom_x86_memory_address_kind_e {
  LOOM_X86_MEMORY_ADDRESS_STATIC = 0,
  LOOM_X86_MEMORY_ADDRESS_INDEXED = 1,
} loom_x86_memory_address_kind_t;

typedef struct loom_x86_memory_access_plan_t {
  // Target-independent memory decomposition shared across targets.
  loom_low_source_memory_access_plan_t source;
  // Selected x86 addressing form.
  loom_x86_memory_address_kind_t address_kind;
  // x86 index scale for LOOM_X86_MEMORY_ADDRESS_INDEXED.
  uint8_t index_scale;
} loom_x86_memory_access_plan_t;

static bool loom_x86_source_value_is_block_argument(const loom_module_t* module,
                                                    loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return false;
  }
  return loom_value_is_block_arg(loom_module_value(module, value_id));
}

static bool loom_x86_dynamic_stride_as_address_scale(int64_t byte_stride,
                                                     uint8_t* out_scale) {
  IREE_ASSERT_ARGUMENT(out_scale);
  switch (byte_stride) {
    case 1:
    case 2:
    case 4:
    case 8:
      *out_scale = (uint8_t)byte_stride;
      return true;
    default:
      *out_scale = 0;
      return false;
  }
}

static bool loom_x86_select_memory_access(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_x86_memory_access_plan_t* out_plan) {
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_low_source_memory_access_plan_build(
          module, loom_low_lower_context_fact_table(context), source_op,
          &out_plan->source, &diagnostic)) {
    return false;
  }
  if (!loom_x86_memory_access_shape_is_zmm32(&out_plan->source) ||
      !loom_x86_memory_space_is_object_memory(out_plan->source.memory_space) ||
      !loom_x86_i64_fits_disp32(out_plan->source.static_byte_offset) ||
      !loom_x86_source_value_is_block_argument(
          module, out_plan->source.root_value_id)) {
    return false;
  }
  if (!loom_low_source_memory_access_is_dynamic(&out_plan->source)) {
    out_plan->address_kind = LOOM_X86_MEMORY_ADDRESS_STATIC;
    out_plan->index_scale = 0;
    return true;
  }
  if (out_plan->source.dynamic_index_source !=
      LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_VALUE) {
    return false;
  }
  out_plan->address_kind = LOOM_X86_MEMORY_ADDRESS_INDEXED;
  return loom_x86_dynamic_stride_as_address_scale(
      out_plan->source.dynamic_index_byte_stride, &out_plan->index_scale);
}

static iree_status_t loom_x86_select_avx512_op(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t* out_plan) {
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
      loom_x86_memory_access_plan_t* plan = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan), (void**)&plan));
      if (loom_x86_select_memory_access(context, source_op, plan)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan);
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_x86_make_disp32_attr(
    loom_low_lower_context_t* context, int64_t value,
    loom_named_attr_t* out_attr) {
  IREE_ASSERT_ARGUMENT(out_attr);
  IREE_ASSERT(loom_x86_i64_fits_disp32(value));
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context), IREE_SV("disp32"), &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

static iree_status_t loom_x86_make_scale_attr(loom_low_lower_context_t* context,
                                              uint8_t value,
                                              loom_named_attr_t* out_attr) {
  IREE_ASSERT_ARGUMENT(out_attr);
  IREE_ASSERT(value == 1 || value == 2 || value == 4 || value == 8);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      loom_low_lower_context_module(context), IREE_SV("scale"), &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

static iree_status_t loom_x86_lower_buffer_alias(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_value_id_t result_value_id) {
  loom_value_id_t low_value_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value_id, &low_value_id));
  return loom_low_lower_bind_value(context, result_value_id, low_value_id);
}

static iree_status_t loom_x86_lower_vector_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_x86_memory_access_plan_t* plan) {
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source.root_value_id, &base));

  loom_named_attr_t attrs[2] = {0};
  IREE_RETURN_IF_ERROR(loom_x86_make_disp32_attr(
      context, plan->source.static_byte_offset, &attrs[0]));
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_x86_make_zmm_register_type(context, &result_type));
  uint64_t descriptor_id =
      X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VMOVDQU32_LOAD_ZMM;
  loom_value_id_t operands[2] = {base, LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 1;
  iree_host_size_t attr_count = 1;
  if (plan->address_kind == LOOM_X86_MEMORY_ADDRESS_INDEXED) {
    descriptor_id =
        X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VMOVDQU32_LOAD_INDEXED_ZMM;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, plan->source.dynamic_index, &operands[1]));
    IREE_RETURN_IF_ERROR(
        loom_x86_make_scale_attr(context, plan->index_scale, &attrs[1]));
    operand_count = 2;
    attr_count = 2;
  }
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
      context, descriptor_id, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1, NULL, 0,
      source_op->location, &load_op));
  return loom_low_lower_bind_value(
      context, loom_vector_load_result(source_op),
      loom_value_slice_get(loom_low_op_results(load_op), 0));
}

static iree_status_t loom_x86_lower_vector_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_x86_memory_access_plan_t* plan) {
  loom_value_id_t base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source.root_value_id, &base));
  loom_value_id_t value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
      context, loom_vector_store_value(source_op), &value));

  loom_named_attr_t attrs[2] = {0};
  IREE_RETURN_IF_ERROR(loom_x86_make_disp32_attr(
      context, plan->source.static_byte_offset, &attrs[0]));
  uint64_t descriptor_id =
      X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VMOVDQU32_STORE_ZMM;
  loom_value_id_t operands[3] = {value, base, LOOM_VALUE_ID_INVALID};
  iree_host_size_t operand_count = 2;
  iree_host_size_t attr_count = 1;
  if (plan->address_kind == LOOM_X86_MEMORY_ADDRESS_INDEXED) {
    descriptor_id =
        X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VMOVDQU32_STORE_INDEXED_ZMM;
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, plan->source.dynamic_index, &operands[2]));
    IREE_RETURN_IF_ERROR(
        loom_x86_make_scale_attr(context, plan->index_scale, &attrs[1]));
    operand_count = 3;
    attr_count = 2;
  }
  loom_op_t* store_op = NULL;
  return loom_low_lower_emit_descriptor_op(
      context, descriptor_id, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), NULL, 0, NULL, 0,
      source_op->location, &store_op);
}

static iree_status_t loom_x86_emit_avx512_op(void* user_data,
                                             loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t plan) {
  (void)user_data;
  switch (plan.id) {
    case LOOM_OP_BUFFER_ASSUME_MEMORY_SPACE:
      return loom_x86_lower_buffer_alias(
          context, loom_buffer_assume_memory_space_buffer(source_op),
          loom_buffer_assume_memory_space_result(source_op));
    case LOOM_OP_BUFFER_VIEW:
      return loom_x86_lower_buffer_alias(context,
                                         loom_buffer_view_buffer(source_op),
                                         loom_buffer_view_result(source_op));
    case LOOM_OP_VECTOR_LOAD:
      return loom_x86_lower_vector_load(
          context, source_op,
          (const loom_x86_memory_access_plan_t*)plan.target_data);
    case LOOM_OP_VECTOR_STORE:
      return loom_x86_lower_vector_store(
          context, source_op,
          (const loom_x86_memory_access_plan_t*)plan.target_data);
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
}

static iree_string_view_t loom_x86_packed_dot_rejection_name(
    loom_x86_packed_dot_rejection_bits_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_X86_PACKED_DOT_REJECTION_FEATURES)) {
    return IREE_SV("features");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD)) {
    return IREE_SV("payload");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_SHAPE)) {
    return IREE_SV("shape");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_FAMILY)) {
    return IREE_SV("family");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_FLAGS)) {
    return IREE_SV("flags");
  }
  return IREE_SV("invalid-request");
}

static iree_string_view_t loom_x86_packed_dot_rejection_detail(
    loom_x86_packed_dot_rejection_bits_t rejection_bits) {
  if (iree_any_bit_set(rejection_bits,
                       LOOM_X86_PACKED_DOT_REJECTION_FEATURES)) {
    return IREE_SV(
        "target profile does not enable a matching x86 packed-dot feature");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD)) {
    return IREE_SV(
        "no x86 packed-dot descriptor matches the vector dot payload types");
  }
  if (iree_any_bit_set(rejection_bits, LOOM_X86_PACKED_DOT_REJECTION_SHAPE)) {
    return IREE_SV("no x86 packed-dot descriptor matches the vector dot shape");
  }
  return IREE_SV("no x86 packed-dot descriptor matches this vector dot op");
}

static iree_status_t loom_x86_select_packed_dot_descriptor(
    const loom_module_t* module, const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* op,
    loom_x86_packed_dot_match_diagnostic_t* out_diagnostic,
    const loom_x86_packed_dot_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  loom_x86_packed_dot_match_request_t request = {0};
  if (!loom_x86_packed_dot_match_request_from_vector_op(module, op, &request)) {
    if (out_diagnostic != NULL) {
      *out_diagnostic = (loom_x86_packed_dot_match_diagnostic_t){
          .rejection_bits = LOOM_X86_PACKED_DOT_REJECTION_INVALID_REQUEST,
      };
    }
    return iree_ok_status();
  }
  if (bundle == NULL || bundle->config == NULL || descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "x86 packed-dot selection requires a target bundle "
                            "and descriptor set");
  }
  request.feature_bits =
      (loom_x86_packed_dot_feature_bits_t)bundle->config->contract_feature_bits;

  loom_x86_packed_dot_match_diagnostic_t diagnostic = {0};
  const loom_x86_packed_dot_descriptor_t* descriptor =
      loom_x86_packed_dot_select(&request, &diagnostic);
  if (out_diagnostic != NULL) {
    *out_diagnostic = diagnostic;
  }
  if (descriptor == NULL) {
    return iree_ok_status();
  }

  uint32_t descriptor_ordinal = loom_low_descriptor_set_lookup_descriptor_by_id(
      descriptor_set, descriptor->stable_id);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "x86 packed-dot descriptor '%.*s' is missing",
                            (int)descriptor->name.size, descriptor->name.data);
  }
  *out_descriptor = descriptor;
  return iree_ok_status();
}

static bool loom_x86_op_is_vector_dot(loom_op_kind_t kind) {
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

typedef struct loom_x86_packed_dot_plan_t {
  // Stable descriptor ID selected for the packed-dot instruction.
  uint64_t descriptor_id;
  // Left-hand payload vector value.
  loom_value_id_t lhs;
  // Right-hand payload vector value.
  loom_value_id_t rhs;
  // Accumulator vector value.
  loom_value_id_t accumulator;
  // Result vector value.
  loom_value_id_t result;
} loom_x86_packed_dot_plan_t;

static bool loom_x86_init_packed_dot_plan(
    const loom_op_t* source_op, uint64_t descriptor_id,
    loom_x86_packed_dot_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_x86_packed_dot_plan_t){0};
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_DOT2F:
      *out_plan = (loom_x86_packed_dot_plan_t){
          .descriptor_id = descriptor_id,
          .lhs = loom_vector_dot2f_lhs(source_op),
          .rhs = loom_vector_dot2f_rhs(source_op),
          .accumulator = loom_vector_dot2f_acc(source_op),
          .result = loom_vector_dot2f_result(source_op),
      };
      return true;
    case LOOM_OP_VECTOR_DOT4I:
      *out_plan = (loom_x86_packed_dot_plan_t){
          .descriptor_id = descriptor_id,
          .lhs = loom_vector_dot4i_lhs(source_op),
          .rhs = loom_vector_dot4i_rhs(source_op),
          .accumulator = loom_vector_dot4i_acc(source_op),
          .result = loom_vector_dot4i_result(source_op),
      };
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_x86_select_packed_dot_op(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_VECTOR_DOT2F:
    case LOOM_OP_VECTOR_DOT4I: {
      const loom_x86_packed_dot_descriptor_t* descriptor = NULL;
      IREE_RETURN_IF_ERROR(loom_x86_select_packed_dot_descriptor(
          loom_low_lower_context_module(context),
          loom_low_lower_context_bundle(context),
          loom_low_lower_context_descriptor_set(context), source_op,
          /*out_diagnostic=*/NULL, &descriptor));
      if (descriptor != NULL) {
        loom_x86_packed_dot_plan_t* plan_data = NULL;
        IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
            context, sizeof(*plan_data), (void**)&plan_data));
        if (loom_x86_init_packed_dot_plan(source_op, descriptor->stable_id,
                                          plan_data)) {
          *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
        }
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_x86_lower_tied_ternary_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t descriptor_id, loom_value_id_t source_lhs,
    loom_value_id_t source_rhs, loom_value_id_t source_accumulator,
    loom_value_id_t source_result) {
  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_lhs, &low_lhs));
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_rhs, &low_rhs));
  loom_value_id_t low_accumulator = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(context, source_accumulator,
                                                   &low_accumulator));

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_map_value(context, source_op,
                                                source_result, &result_type));
  IREE_ASSERT(loom_type_is_register(result_type));

  loom_op_t* accumulator_copy_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_copy_build(
      loom_low_lower_context_builder(context), low_accumulator, result_type,
      source_op->location, &accumulator_copy_op));
  const loom_value_id_t copied_accumulator =
      loom_low_copy_result(accumulator_copy_op);

  loom_value_id_t low_operands[3] = {copied_accumulator, low_lhs, low_rhs};
  loom_tied_result_t tied_result = {
      .result_index = 0,
      .operand_index = 0,
      .has_type_change = false,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_descriptor_op(
      context, descriptor_id, low_operands, IREE_ARRAYSIZE(low_operands),
      loom_make_named_attr_slice(NULL, 0), &result_type, 1, &tied_result, 1,
      source_op->location, &low_op));
  return loom_low_lower_bind_value(
      context, source_result,
      loom_value_slice_get(loom_low_op_results(low_op), 0));
}

static iree_status_t loom_x86_lower_packed_dot_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_x86_packed_dot_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  return loom_x86_lower_tied_ternary_op(context, source_op, plan->descriptor_id,
                                        plan->lhs, plan->rhs, plan->accumulator,
                                        plan->result);
}

static iree_status_t loom_x86_emit_packed_dot_op(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan) {
  (void)user_data;
  switch (plan.id) {
    case LOOM_OP_VECTOR_DOT2F:
    case LOOM_OP_VECTOR_DOT4I:
      return loom_x86_lower_packed_dot_op(
          context, source_op,
          (const loom_x86_packed_dot_plan_t*)plan.target_data);
    default:
      IREE_ASSERT_UNREACHABLE();
      return iree_ok_status();
  }
}

static iree_status_t loom_x86_low_legality_verify_packed_dot(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  const iree_string_view_t contract_set_key =
      loom_x86_legality_contract_set_key(context);
  if (!loom_x86_contract_set_key_is_x86(contract_set_key)) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (!loom_x86_contract_set_key_is_packed_dot_core(contract_set_key)) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("contract"), contract_set_key,
        IREE_SV("x86 vector dot ops require the x86.packed_dot.core "
                "target-low contract set"));
  }

  loom_x86_packed_dot_match_diagnostic_t diagnostic = {0};
  const loom_x86_packed_dot_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_x86_select_packed_dot_descriptor(
      loom_target_low_legality_module(context),
      loom_target_low_legality_bundle(context),
      loom_target_low_legality_descriptor_set(context), op, &diagnostic,
      &descriptor));
  if (descriptor == NULL) {
    return loom_target_low_legality_reject(
        context, provider, op, IREE_SV("contract"),
        loom_x86_packed_dot_rejection_name(diagnostic.rejection_bits),
        loom_x86_packed_dot_rejection_detail(diagnostic.rejection_bits));
  }

  return loom_target_low_legality_record_contract(
      context, provider, op, descriptor->name, IREE_SV("selected"),
      IREE_SV("selected x86 packed-dot descriptor"));
}

static iree_status_t loom_x86_low_legality_try_verify_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  if (loom_x86_op_is_vector_dot(op->kind)) {
    return loom_x86_low_legality_verify_packed_dot(provider, context, op,
                                                   out_handled);
  }
  return iree_ok_status();
}

static const loom_low_lower_policy_t kX86Avx512LowLowerPolicy = {
    .name = IREE_SVL("x86-avx512-low-lower"),
    .map_type = {.fn = loom_x86_map_avx512_type, .user_data = NULL},
    .map_argument = {.fn = loom_x86_map_avx512_argument, .user_data = NULL},
    .rule_set = &loom_x86_avx512_rule_set,
    .select_op = {.fn = loom_x86_select_avx512_op, .user_data = NULL},
    .emit_op = {.fn = loom_x86_emit_avx512_op, .user_data = NULL},
};

static const loom_low_lower_policy_t kX86PackedDotLowLowerPolicy = {
    .name = IREE_SVL("x86-packed-dot-low-lower"),
    .map_type = {.fn = loom_x86_map_packed_dot_type, .user_data = NULL},
    .select_op = {.fn = loom_x86_select_packed_dot_op, .user_data = NULL},
    .emit_op = {.fn = loom_x86_emit_packed_dot_op, .user_data = NULL},
};

const loom_target_low_legality_provider_t
    loom_x86_low_legality_provider_storage = {
        .name = IREE_SVL("x86"),
        .try_verify_op = loom_x86_low_legality_try_verify_op,
};

const loom_low_lower_policy_t* loom_x86_avx512_low_lower_policy(void) {
  return &kX86Avx512LowLowerPolicy;
}

const loom_low_lower_policy_t* loom_x86_packed_dot_low_lower_policy(void) {
  return &kX86PackedDotLowLowerPolicy;
}

const loom_target_low_legality_provider_t* loom_x86_low_legality_provider(
    void) {
  return &loom_x86_low_legality_provider_storage;
}

void loom_x86_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("x86.avx512.core"),
          .policy = &kX86Avx512LowLowerPolicy,
      },
      {
          .contract_set_key = IREE_SVL("x86.packed_dot.core"),
          .policy = &kX86PackedDotLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
