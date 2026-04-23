// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 AVX512 source-to-low rule tables.

#include <stdint.h>

#include "loom/codegen/low/lower_rules.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/avx512_descriptors.h"

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

const loom_low_lower_rule_set_t loom_x86_avx512_rule_set = {
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
