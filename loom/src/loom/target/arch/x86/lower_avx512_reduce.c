// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 AVX512 source-to-low vector reduction rule tables.

#include "loom/codegen/low/lower_rules.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/avx512_descriptors.h"
#include "loom/target/arch/x86/lower_internal.h"

enum loom_x86_avx512_reduce_type_pattern_e {
  LOOM_X86_AVX512_REDUCE_TYPE_V4I32 = 0,
  LOOM_X86_AVX512_REDUCE_TYPE_SI32 = 1,
  LOOM_X86_AVX512_REDUCE_TYPE_V4F32 = 2,
  LOOM_X86_AVX512_REDUCE_TYPE_SF32 = 3,
  LOOM_X86_AVX512_REDUCE_TYPE_V16F32 = 4,
};

static const loom_low_lower_type_pattern_t
    loom_x86_avx512_reduce_type_patterns[] = {
        [LOOM_X86_AVX512_REDUCE_TYPE_V4I32] =
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
        [LOOM_X86_AVX512_REDUCE_TYPE_SI32] =
            {
                .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                         LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
                .type_kind = LOOM_TYPE_SCALAR,
                .element_type_mask =
                    LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I32),
            },
        [LOOM_X86_AVX512_REDUCE_TYPE_V4F32] =
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
        [LOOM_X86_AVX512_REDUCE_TYPE_SF32] =
            {
                .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                         LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
                .type_kind = LOOM_TYPE_SCALAR,
                .element_type_mask =
                    LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_F32),
            },
        [LOOM_X86_AVX512_REDUCE_TYPE_V16F32] =
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
};

enum loom_x86_avx512_reduce_value_ref_e {
  LOOM_X86_AVX512_REDUCE_INPUT = 0,
  LOOM_X86_AVX512_REDUCE_RESULT,
  LOOM_X86_AVX512_REDUCE_INIT,
  LOOM_X86_AVX512_REDUCE_LANE0,
  LOOM_X86_AVX512_REDUCE_ACC0,
  LOOM_X86_AVX512_REDUCE_LANE1,
  LOOM_X86_AVX512_REDUCE_ACC1,
  LOOM_X86_AVX512_REDUCE_LANE2,
  LOOM_X86_AVX512_REDUCE_ACC2,
  LOOM_X86_AVX512_REDUCE_LANE3,
  LOOM_X86_AVX512_REDUCE_F32_INPUT0,
  LOOM_X86_AVX512_REDUCE_F32_SHUFFLE0,
  LOOM_X86_AVX512_REDUCE_F32_PAIR_SUM,
  LOOM_X86_AVX512_REDUCE_F32_SHUFFLE1,
  LOOM_X86_AVX512_REDUCE_F32_INIT,
  LOOM_X86_AVX512_REDUCE_F32_VECTOR_SUM,
  LOOM_X86_AVX512_REDUCE_F32X16_INPUT,
  LOOM_X86_AVX512_REDUCE_F32X16_Q0,
  LOOM_X86_AVX512_REDUCE_F32X16_Q1,
  LOOM_X86_AVX512_REDUCE_F32X16_Q2,
  LOOM_X86_AVX512_REDUCE_F32X16_Q3,
  LOOM_X86_AVX512_REDUCE_F32X16_SUM01,
  LOOM_X86_AVX512_REDUCE_F32X16_SUM23,
  LOOM_X86_AVX512_REDUCE_F32X16_XMM_SUM,
  LOOM_X86_AVX512_REDUCE_F32X16_SHUFFLE0,
  LOOM_X86_AVX512_REDUCE_F32X16_PAIR_SUM,
  LOOM_X86_AVX512_REDUCE_F32X16_SHUFFLE1,
  LOOM_X86_AVX512_REDUCE_F32X16_INIT,
  LOOM_X86_AVX512_REDUCE_F32X16_VECTOR_SUM,
};

static const loom_low_lower_value_ref_t loom_x86_avx512_reduce_value_refs[] = {
    [LOOM_X86_AVX512_REDUCE_INPUT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
        },
    [LOOM_X86_AVX512_REDUCE_RESULT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
            .index = 0,
        },
    [LOOM_X86_AVX512_REDUCE_INIT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
        },
    [LOOM_X86_AVX512_REDUCE_LANE0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 0,
        },
    [LOOM_X86_AVX512_REDUCE_ACC0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 1,
        },
    [LOOM_X86_AVX512_REDUCE_LANE1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 2,
        },
    [LOOM_X86_AVX512_REDUCE_ACC1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 3,
        },
    [LOOM_X86_AVX512_REDUCE_LANE2] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 4,
        },
    [LOOM_X86_AVX512_REDUCE_ACC2] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 5,
        },
    [LOOM_X86_AVX512_REDUCE_LANE3] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 6,
        },
    [LOOM_X86_AVX512_REDUCE_F32_INPUT0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
        },
    [LOOM_X86_AVX512_REDUCE_F32_SHUFFLE0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 0,
        },
    [LOOM_X86_AVX512_REDUCE_F32_PAIR_SUM] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 1,
        },
    [LOOM_X86_AVX512_REDUCE_F32_SHUFFLE1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 2,
        },
    [LOOM_X86_AVX512_REDUCE_F32_INIT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
        },
    [LOOM_X86_AVX512_REDUCE_F32_VECTOR_SUM] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 3,
        },
    [LOOM_X86_AVX512_REDUCE_F32X16_INPUT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
        },
    [LOOM_X86_AVX512_REDUCE_F32X16_Q0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 0,
        },
    [LOOM_X86_AVX512_REDUCE_F32X16_Q1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 1,
        },
    [LOOM_X86_AVX512_REDUCE_F32X16_Q2] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 2,
        },
    [LOOM_X86_AVX512_REDUCE_F32X16_Q3] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 3,
        },
    [LOOM_X86_AVX512_REDUCE_F32X16_SUM01] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 4,
        },
    [LOOM_X86_AVX512_REDUCE_F32X16_SUM23] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 5,
        },
    [LOOM_X86_AVX512_REDUCE_F32X16_XMM_SUM] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 6,
        },
    [LOOM_X86_AVX512_REDUCE_F32X16_SHUFFLE0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 7,
        },
    [LOOM_X86_AVX512_REDUCE_F32X16_PAIR_SUM] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 8,
        },
    [LOOM_X86_AVX512_REDUCE_F32X16_SHUFFLE1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 9,
        },
    [LOOM_X86_AVX512_REDUCE_F32X16_INIT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
        },
    [LOOM_X86_AVX512_REDUCE_F32X16_VECTOR_SUM] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 10,
        },
};

enum loom_x86_avx512_reduce_attr_copy_e {
  LOOM_X86_AVX512_REDUCE_ATTR_LANE0 = 0,
  LOOM_X86_AVX512_REDUCE_ATTR_LANE1,
  LOOM_X86_AVX512_REDUCE_ATTR_LANE2,
  LOOM_X86_AVX512_REDUCE_ATTR_LANE3,
  LOOM_X86_AVX512_REDUCE_ATTR_F32_SWAP_PAIRS,
  LOOM_X86_AVX512_REDUCE_ATTR_F32_SWAP_NEIGHBORS,
};

#define LOOM_X86_AVX512_REDUCE_I64_ATTR(name_value, literal_value) \
  {                                                                \
      .kind = LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL,                \
      .target_name = IREE_SVL(name_value),                         \
      .literal_i64 = literal_value,                                \
  }

static const loom_low_lower_attr_copy_t loom_x86_avx512_reduce_attr_copies[] = {
    [LOOM_X86_AVX512_REDUCE_ATTR_LANE0] =
        LOOM_X86_AVX512_REDUCE_I64_ATTR("lane", 0),
    [LOOM_X86_AVX512_REDUCE_ATTR_LANE1] =
        LOOM_X86_AVX512_REDUCE_I64_ATTR("lane", 1),
    [LOOM_X86_AVX512_REDUCE_ATTR_LANE2] =
        LOOM_X86_AVX512_REDUCE_I64_ATTR("lane", 2),
    [LOOM_X86_AVX512_REDUCE_ATTR_LANE3] =
        LOOM_X86_AVX512_REDUCE_I64_ATTR("lane", 3),
    [LOOM_X86_AVX512_REDUCE_ATTR_F32_SWAP_PAIRS] =
        LOOM_X86_AVX512_REDUCE_I64_ATTR("control", 78),
    [LOOM_X86_AVX512_REDUCE_ATTR_F32_SWAP_NEIGHBORS] =
        LOOM_X86_AVX512_REDUCE_I64_ATTR("control", 177),
};

#undef LOOM_X86_AVX512_REDUCE_I64_ATTR

enum loom_x86_avx512_reduce_diagnostic_e {
  LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_KIND = 0,
  LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_V4I32 = 1,
  LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_I32 = 2,
  LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_DESCRIPTOR = 3,
  LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_V4F32 = 4,
  LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_F32 = 5,
  LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_V16F32 = 6,
};

static const loom_low_lower_diagnostic_t loom_x86_avx512_reduce_diagnostics[] =
    {
        [LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_KIND] =
            {
                .subject_kind = IREE_SVL("kind"),
                .subject_name = IREE_SVL("vector.reduce"),
                .reason = IREE_SVL("x86 AVX512VL vector.reduce lowering "
                                   "currently requires addi or addf"),
            },
        [LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_V4I32] =
            {
                .subject_kind = IREE_SVL("type"),
                .subject_name = IREE_SVL("vector<4xi32>"),
                .reason = IREE_SVL("x86 AVX512VL vector.reduce integer "
                                   "lowering requires vector<4xi32> input"),
            },
        [LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_I32] =
            {
                .subject_kind = IREE_SVL("type"),
                .subject_name = IREE_SVL("i32"),
                .reason = IREE_SVL("x86 AVX512VL vector.reduce integer "
                                   "lowering requires i32 init/result values"),
            },
        [LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_DESCRIPTOR] =
            {
                .subject_kind = IREE_SVL("descriptor"),
                .subject_name = IREE_SVL("x86.reduce.i32x4"),
                .reason = IREE_SVL("the selected x86 descriptor set does not "
                                   "contain the required reduce packets"),
            },
        [LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_V4F32] =
            {
                .subject_kind = IREE_SVL("type"),
                .subject_name = IREE_SVL("vector<4xf32>"),
                .reason = IREE_SVL("x86 AVX512VL vector.reduce floating-point "
                                   "lowering requires vector<4xf32> input"),
            },
        [LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_F32] =
            {
                .subject_kind = IREE_SVL("type"),
                .subject_name = IREE_SVL("f32"),
                .reason = IREE_SVL("x86 AVX512VL vector.reduce floating-point "
                                   "lowering requires f32 init/result values"),
            },
        [LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_V16F32] =
            {
                .subject_kind = IREE_SVL("type"),
                .subject_name = IREE_SVL("vector<16xf32>"),
                .reason = IREE_SVL("x86 AVX512 vector.reduce floating-point "
                                   "lowering requires vector<16xf32> input"),
            },
};

enum loom_x86_avx512_reduce_guard_e {
  LOOM_X86_AVX512_REDUCE_GUARD_KIND = 0,
  LOOM_X86_AVX512_REDUCE_GUARD_INPUT,
  LOOM_X86_AVX512_REDUCE_GUARD_INIT,
  LOOM_X86_AVX512_REDUCE_GUARD_RESULT,
  LOOM_X86_AVX512_REDUCE_GUARD_VPEXTRD,
  LOOM_X86_AVX512_REDUCE_GUARD_ADD_GPR32,
  LOOM_X86_AVX512_REDUCE_GUARD_KIND_ADDF,
  LOOM_X86_AVX512_REDUCE_GUARD_F32_INPUT,
  LOOM_X86_AVX512_REDUCE_GUARD_F32_INIT,
  LOOM_X86_AVX512_REDUCE_GUARD_F32_RESULT,
  LOOM_X86_AVX512_REDUCE_GUARD_VPERMILPS,
  LOOM_X86_AVX512_REDUCE_GUARD_VADDPS,
  LOOM_X86_AVX512_REDUCE_GUARD_VADDSS,
  LOOM_X86_AVX512_REDUCE_GUARD_KIND_ADDF_V16,
  LOOM_X86_AVX512_REDUCE_GUARD_F32X16_INPUT,
  LOOM_X86_AVX512_REDUCE_GUARD_F32X16_INIT,
  LOOM_X86_AVX512_REDUCE_GUARD_F32X16_RESULT,
  LOOM_X86_AVX512_REDUCE_GUARD_VEXTRACTF32X4,
  LOOM_X86_AVX512_REDUCE_GUARD_VPERMILPS_V16,
  LOOM_X86_AVX512_REDUCE_GUARD_VADDPS_V16,
  LOOM_X86_AVX512_REDUCE_GUARD_VADDSS_V16,
};

static const loom_low_lower_guard_t loom_x86_avx512_reduce_guards[] = {
    [LOOM_X86_AVX512_REDUCE_GUARD_KIND] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ,
            .attr_index = 0,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_KIND,
            .u64 = LOOM_VECTOR_REDUCE_KIND_ADDI,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_INPUT] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_REDUCE_INPUT,
            .type_pattern_index = LOOM_X86_AVX512_REDUCE_TYPE_V4I32,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_V4I32,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_INIT] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_REDUCE_INIT,
            .type_pattern_index = LOOM_X86_AVX512_REDUCE_TYPE_SI32,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_I32,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_RESULT] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_REDUCE_RESULT,
            .type_pattern_index = LOOM_X86_AVX512_REDUCE_TYPE_SI32,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_I32,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_VPEXTRD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_DESCRIPTOR,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPEXTRD_GPR32_XMM,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_ADD_GPR32] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_DESCRIPTOR,
            .descriptor_id = X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_ADD_GPR32,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_KIND_ADDF] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ,
            .attr_index = 0,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_KIND,
            .u64 = LOOM_VECTOR_REDUCE_KIND_ADDF,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_F32_INPUT] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_REDUCE_INPUT,
            .type_pattern_index = LOOM_X86_AVX512_REDUCE_TYPE_V4F32,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_V4F32,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_F32_INIT] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_REDUCE_INIT,
            .type_pattern_index = LOOM_X86_AVX512_REDUCE_TYPE_SF32,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_F32,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_F32_RESULT] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_REDUCE_RESULT,
            .type_pattern_index = LOOM_X86_AVX512_REDUCE_TYPE_SF32,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_F32,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_VPERMILPS] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_DESCRIPTOR,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPERMILPS_XMM,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_VADDPS] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_DESCRIPTOR,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDPS_XMM,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_VADDSS] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_DESCRIPTOR,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDSS_XMM,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_KIND_ADDF_V16] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ,
            .attr_index = 0,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_KIND,
            .u64 = LOOM_VECTOR_REDUCE_KIND_ADDF,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_F32X16_INPUT] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_REDUCE_INPUT,
            .type_pattern_index = LOOM_X86_AVX512_REDUCE_TYPE_V16F32,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_V16F32,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_F32X16_INIT] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_REDUCE_INIT,
            .type_pattern_index = LOOM_X86_AVX512_REDUCE_TYPE_SF32,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_F32,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_F32X16_RESULT] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_X86_AVX512_REDUCE_RESULT,
            .type_pattern_index = LOOM_X86_AVX512_REDUCE_TYPE_SF32,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_F32,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_VEXTRACTF32X4] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_DESCRIPTOR,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VEXTRACTF32X4_XMM_ZMM,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_VPERMILPS_V16] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_DESCRIPTOR,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPERMILPS_XMM,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_VADDPS_V16] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_DESCRIPTOR,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDPS_XMM,
        },
    [LOOM_X86_AVX512_REDUCE_GUARD_VADDSS_V16] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,
            .diagnostic_index = LOOM_X86_AVX512_REDUCE_DIAGNOSTIC_DESCRIPTOR,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDSS_XMM,
        },
};

static const loom_tied_result_t loom_x86_avx512_reduce_tied_results[] = {
    {
        .result_index = 0,
        .operand_index = 0,
        .has_type_change = false,
    },
};

enum loom_x86_avx512_reduce_emit_e {
  LOOM_X86_AVX512_REDUCE_EMIT_EXTRACT0 = 0,
  LOOM_X86_AVX512_REDUCE_EMIT_ADD0,
  LOOM_X86_AVX512_REDUCE_EMIT_EXTRACT1,
  LOOM_X86_AVX512_REDUCE_EMIT_ADD1,
  LOOM_X86_AVX512_REDUCE_EMIT_EXTRACT2,
  LOOM_X86_AVX512_REDUCE_EMIT_ADD2,
  LOOM_X86_AVX512_REDUCE_EMIT_EXTRACT3,
  LOOM_X86_AVX512_REDUCE_EMIT_ADD3,
  LOOM_X86_AVX512_REDUCE_EMIT_F32_SHUFFLE0,
  LOOM_X86_AVX512_REDUCE_EMIT_F32_ADD_PAIRS,
  LOOM_X86_AVX512_REDUCE_EMIT_F32_SHUFFLE1,
  LOOM_X86_AVX512_REDUCE_EMIT_F32_ADD_VECTOR,
  LOOM_X86_AVX512_REDUCE_EMIT_F32_ADD_INIT,
  LOOM_X86_AVX512_REDUCE_EMIT_F32X16_EXTRACT0,
  LOOM_X86_AVX512_REDUCE_EMIT_F32X16_EXTRACT1,
  LOOM_X86_AVX512_REDUCE_EMIT_F32X16_EXTRACT2,
  LOOM_X86_AVX512_REDUCE_EMIT_F32X16_EXTRACT3,
  LOOM_X86_AVX512_REDUCE_EMIT_F32X16_ADD01,
  LOOM_X86_AVX512_REDUCE_EMIT_F32X16_ADD23,
  LOOM_X86_AVX512_REDUCE_EMIT_F32X16_ADD_QUARTERS,
  LOOM_X86_AVX512_REDUCE_EMIT_F32X16_SHUFFLE0,
  LOOM_X86_AVX512_REDUCE_EMIT_F32X16_ADD_PAIRS,
  LOOM_X86_AVX512_REDUCE_EMIT_F32X16_SHUFFLE1,
  LOOM_X86_AVX512_REDUCE_EMIT_F32X16_ADD_VECTOR,
  LOOM_X86_AVX512_REDUCE_EMIT_F32X16_ADD_INIT,
};

#define LOOM_X86_AVX512_REDUCE_EXTRACT(lane_ref, attr_copy)           \
  {                                                                   \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,                      \
      .flags = LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,         \
      .descriptor_id =                                                \
          X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPEXTRD_GPR32_XMM, \
      .operand_ref_start = LOOM_X86_AVX512_REDUCE_INPUT,              \
      .operand_ref_count = 1,                                         \
      .result_ref_start = LOOM_X86_AVX512_REDUCE_RESULT,              \
      .result_ref_count = 1,                                          \
      .result_bind_ref_start = lane_ref,                              \
      .attr_copy_start = attr_copy,                                   \
      .attr_copy_count = 1,                                           \
  }

#define LOOM_X86_AVX512_REDUCE_ADD(operand_start, result_ref, flags_value) \
  {                                                                        \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,                           \
      .flags = flags_value,                                                \
      .descriptor_id = X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_ADD_GPR32, \
      .operand_ref_start = operand_start,                                  \
      .operand_ref_count = 2,                                              \
      .copy_operand_mask = 0x1,                                            \
      .result_ref_start = LOOM_X86_AVX512_REDUCE_RESULT,                   \
      .result_ref_count = 1,                                               \
      .result_bind_ref_start = result_ref,                                 \
      .tied_result_start = 0,                                              \
      .tied_result_count = 1,                                              \
  }

#define LOOM_X86_AVX512_REDUCE_F32X16_XMM_FLAGS    \
  (LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS | \
   LOOM_LOW_LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN)

#define LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP(                            \
    descriptor_id_value, operand_ref_start_value, operand_ref_count_value, \
    result_bind_ref_value)                                                 \
  {                                                                        \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,                           \
      .flags = LOOM_X86_AVX512_REDUCE_F32X16_XMM_FLAGS,                    \
      .descriptor_id = descriptor_id_value,                                \
      .operand_ref_start = operand_ref_start_value,                        \
      .operand_ref_count = operand_ref_count_value,                        \
      .result_type_pattern_start = LOOM_X86_AVX512_REDUCE_TYPE_V4F32,      \
      .result_ref_count = 1,                                               \
      .result_bind_ref_start = result_bind_ref_value,                      \
  }

#define LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP_ATTR(                       \
    descriptor_id_value, operand_ref_start_value, operand_ref_count_value, \
    result_bind_ref_value, attr_copy_value)                                \
  {                                                                        \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,                           \
      .flags = LOOM_X86_AVX512_REDUCE_F32X16_XMM_FLAGS,                    \
      .descriptor_id = descriptor_id_value,                                \
      .operand_ref_start = operand_ref_start_value,                        \
      .operand_ref_count = operand_ref_count_value,                        \
      .result_type_pattern_start = LOOM_X86_AVX512_REDUCE_TYPE_V4F32,      \
      .result_ref_count = 1,                                               \
      .result_bind_ref_start = result_bind_ref_value,                      \
      .attr_copy_start = attr_copy_value,                                  \
      .attr_copy_count = 1,                                                \
  }

static const loom_low_lower_emit_t loom_x86_avx512_reduce_emits[] = {
    [LOOM_X86_AVX512_REDUCE_EMIT_EXTRACT0] = LOOM_X86_AVX512_REDUCE_EXTRACT(
        LOOM_X86_AVX512_REDUCE_LANE0, LOOM_X86_AVX512_REDUCE_ATTR_LANE0),
    [LOOM_X86_AVX512_REDUCE_EMIT_ADD0] = LOOM_X86_AVX512_REDUCE_ADD(
        LOOM_X86_AVX512_REDUCE_INIT, LOOM_X86_AVX512_REDUCE_ACC0,
        LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS),
    [LOOM_X86_AVX512_REDUCE_EMIT_EXTRACT1] = LOOM_X86_AVX512_REDUCE_EXTRACT(
        LOOM_X86_AVX512_REDUCE_LANE1, LOOM_X86_AVX512_REDUCE_ATTR_LANE1),
    [LOOM_X86_AVX512_REDUCE_EMIT_ADD1] = LOOM_X86_AVX512_REDUCE_ADD(
        LOOM_X86_AVX512_REDUCE_ACC0, LOOM_X86_AVX512_REDUCE_ACC1,
        LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS),
    [LOOM_X86_AVX512_REDUCE_EMIT_EXTRACT2] = LOOM_X86_AVX512_REDUCE_EXTRACT(
        LOOM_X86_AVX512_REDUCE_LANE2, LOOM_X86_AVX512_REDUCE_ATTR_LANE2),
    [LOOM_X86_AVX512_REDUCE_EMIT_ADD2] = LOOM_X86_AVX512_REDUCE_ADD(
        LOOM_X86_AVX512_REDUCE_ACC1, LOOM_X86_AVX512_REDUCE_ACC2,
        LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS),
    [LOOM_X86_AVX512_REDUCE_EMIT_EXTRACT3] = LOOM_X86_AVX512_REDUCE_EXTRACT(
        LOOM_X86_AVX512_REDUCE_LANE3, LOOM_X86_AVX512_REDUCE_ATTR_LANE3),
    [LOOM_X86_AVX512_REDUCE_EMIT_ADD3] = LOOM_X86_AVX512_REDUCE_ADD(
        LOOM_X86_AVX512_REDUCE_ACC2, LOOM_X86_AVX512_REDUCE_RESULT, 0),
    [LOOM_X86_AVX512_REDUCE_EMIT_F32_SHUFFLE0] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .flags = LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPERMILPS_XMM,
            .operand_ref_start = LOOM_X86_AVX512_REDUCE_F32_INPUT0,
            .operand_ref_count = 1,
            .result_ref_start = LOOM_X86_AVX512_REDUCE_F32_INPUT0,
            .result_ref_count = 1,
            .result_bind_ref_start = LOOM_X86_AVX512_REDUCE_F32_SHUFFLE0,
            .attr_copy_start = LOOM_X86_AVX512_REDUCE_ATTR_F32_SWAP_PAIRS,
            .attr_copy_count = 1,
        },
    [LOOM_X86_AVX512_REDUCE_EMIT_F32_ADD_PAIRS] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .flags = LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDPS_XMM,
            .operand_ref_start = LOOM_X86_AVX512_REDUCE_F32_INPUT0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_X86_AVX512_REDUCE_F32_INPUT0,
            .result_ref_count = 1,
            .result_bind_ref_start = LOOM_X86_AVX512_REDUCE_F32_PAIR_SUM,
        },
    [LOOM_X86_AVX512_REDUCE_EMIT_F32_SHUFFLE1] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .flags = LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPERMILPS_XMM,
            .operand_ref_start = LOOM_X86_AVX512_REDUCE_F32_PAIR_SUM,
            .operand_ref_count = 1,
            .result_ref_start = LOOM_X86_AVX512_REDUCE_F32_INPUT0,
            .result_ref_count = 1,
            .result_bind_ref_start = LOOM_X86_AVX512_REDUCE_F32_SHUFFLE1,
            .attr_copy_start = LOOM_X86_AVX512_REDUCE_ATTR_F32_SWAP_NEIGHBORS,
            .attr_copy_count = 1,
        },
    [LOOM_X86_AVX512_REDUCE_EMIT_F32_ADD_VECTOR] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .flags = LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDPS_XMM,
            .operand_ref_start = LOOM_X86_AVX512_REDUCE_F32_PAIR_SUM,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_X86_AVX512_REDUCE_F32_INPUT0,
            .result_ref_count = 1,
            .result_bind_ref_start = LOOM_X86_AVX512_REDUCE_F32_VECTOR_SUM,
        },
    [LOOM_X86_AVX512_REDUCE_EMIT_F32_ADD_INIT] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDSS_XMM,
            .operand_ref_start = LOOM_X86_AVX512_REDUCE_F32_INIT,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_X86_AVX512_REDUCE_RESULT,
            .result_ref_count = 1,
        },
    [LOOM_X86_AVX512_REDUCE_EMIT_F32X16_EXTRACT0] =
        LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP_ATTR(
            X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VEXTRACTF32X4_XMM_ZMM,
            LOOM_X86_AVX512_REDUCE_F32X16_INPUT, 1,
            LOOM_X86_AVX512_REDUCE_F32X16_Q0,
            LOOM_X86_AVX512_REDUCE_ATTR_LANE0),
    [LOOM_X86_AVX512_REDUCE_EMIT_F32X16_EXTRACT1] =
        LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP_ATTR(
            X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VEXTRACTF32X4_XMM_ZMM,
            LOOM_X86_AVX512_REDUCE_F32X16_INPUT, 1,
            LOOM_X86_AVX512_REDUCE_F32X16_Q1,
            LOOM_X86_AVX512_REDUCE_ATTR_LANE1),
    [LOOM_X86_AVX512_REDUCE_EMIT_F32X16_EXTRACT2] =
        LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP_ATTR(
            X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VEXTRACTF32X4_XMM_ZMM,
            LOOM_X86_AVX512_REDUCE_F32X16_INPUT, 1,
            LOOM_X86_AVX512_REDUCE_F32X16_Q2,
            LOOM_X86_AVX512_REDUCE_ATTR_LANE2),
    [LOOM_X86_AVX512_REDUCE_EMIT_F32X16_EXTRACT3] =
        LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP_ATTR(
            X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VEXTRACTF32X4_XMM_ZMM,
            LOOM_X86_AVX512_REDUCE_F32X16_INPUT, 1,
            LOOM_X86_AVX512_REDUCE_F32X16_Q3,
            LOOM_X86_AVX512_REDUCE_ATTR_LANE3),
    [LOOM_X86_AVX512_REDUCE_EMIT_F32X16_ADD01] =
        LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP(
            X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDPS_XMM,
            LOOM_X86_AVX512_REDUCE_F32X16_Q0, 2,
            LOOM_X86_AVX512_REDUCE_F32X16_SUM01),
    [LOOM_X86_AVX512_REDUCE_EMIT_F32X16_ADD23] =
        LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP(
            X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDPS_XMM,
            LOOM_X86_AVX512_REDUCE_F32X16_Q2, 2,
            LOOM_X86_AVX512_REDUCE_F32X16_SUM23),
    [LOOM_X86_AVX512_REDUCE_EMIT_F32X16_ADD_QUARTERS] =
        LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP(
            X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDPS_XMM,
            LOOM_X86_AVX512_REDUCE_F32X16_SUM01, 2,
            LOOM_X86_AVX512_REDUCE_F32X16_XMM_SUM),
    [LOOM_X86_AVX512_REDUCE_EMIT_F32X16_SHUFFLE0] =
        LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP_ATTR(
            X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPERMILPS_XMM,
            LOOM_X86_AVX512_REDUCE_F32X16_XMM_SUM, 1,
            LOOM_X86_AVX512_REDUCE_F32X16_SHUFFLE0,
            LOOM_X86_AVX512_REDUCE_ATTR_F32_SWAP_PAIRS),
    [LOOM_X86_AVX512_REDUCE_EMIT_F32X16_ADD_PAIRS] =
        LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP(
            X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDPS_XMM,
            LOOM_X86_AVX512_REDUCE_F32X16_XMM_SUM, 2,
            LOOM_X86_AVX512_REDUCE_F32X16_PAIR_SUM),
    [LOOM_X86_AVX512_REDUCE_EMIT_F32X16_SHUFFLE1] =
        LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP_ATTR(
            X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VPERMILPS_XMM,
            LOOM_X86_AVX512_REDUCE_F32X16_PAIR_SUM, 1,
            LOOM_X86_AVX512_REDUCE_F32X16_SHUFFLE1,
            LOOM_X86_AVX512_REDUCE_ATTR_F32_SWAP_NEIGHBORS),
    [LOOM_X86_AVX512_REDUCE_EMIT_F32X16_ADD_VECTOR] =
        LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP(
            X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDPS_XMM,
            LOOM_X86_AVX512_REDUCE_F32X16_PAIR_SUM, 2,
            LOOM_X86_AVX512_REDUCE_F32X16_VECTOR_SUM),
    [LOOM_X86_AVX512_REDUCE_EMIT_F32X16_ADD_INIT] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id =
                X86_AVX512_CORE_DESCRIPTOR_ID_X86_AVX512_VADDSS_XMM,
            .operand_ref_start = LOOM_X86_AVX512_REDUCE_F32X16_INIT,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_X86_AVX512_REDUCE_RESULT,
            .result_ref_count = 1,
        },
};

#undef LOOM_X86_AVX512_REDUCE_ADD
#undef LOOM_X86_AVX512_REDUCE_EXTRACT
#undef LOOM_X86_AVX512_REDUCE_F32X16_XMM_FLAGS
#undef LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP
#undef LOOM_X86_AVX512_REDUCE_F32X16_XMM_TEMP_ATTR

enum loom_x86_avx512_reduce_rule_e {
  LOOM_X86_AVX512_REDUCE_RULE_ADDI = 0,
  LOOM_X86_AVX512_REDUCE_RULE_ADDF = 1,
  LOOM_X86_AVX512_REDUCE_RULE_ADDF_V16 = 2,
};

static const loom_low_lower_rule_t loom_x86_avx512_reduce_rules[] = {
    [LOOM_X86_AVX512_REDUCE_RULE_ADDI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_REDUCE,
            .temporary_count = 7,
            .guard_start = LOOM_X86_AVX512_REDUCE_GUARD_KIND,
            .guard_count = LOOM_X86_AVX512_REDUCE_GUARD_KIND_ADDF -
                           LOOM_X86_AVX512_REDUCE_GUARD_KIND,
            .emit_start = LOOM_X86_AVX512_REDUCE_EMIT_EXTRACT0,
            .emit_count = LOOM_X86_AVX512_REDUCE_EMIT_F32_SHUFFLE0 -
                          LOOM_X86_AVX512_REDUCE_EMIT_EXTRACT0,
        },
    [LOOM_X86_AVX512_REDUCE_RULE_ADDF] =
        {
            .source_op_kind = LOOM_OP_VECTOR_REDUCE,
            .temporary_count = 4,
            .guard_start = LOOM_X86_AVX512_REDUCE_GUARD_KIND_ADDF,
            .guard_count = LOOM_X86_AVX512_REDUCE_GUARD_KIND_ADDF_V16 -
                           LOOM_X86_AVX512_REDUCE_GUARD_KIND_ADDF,
            .emit_start = LOOM_X86_AVX512_REDUCE_EMIT_F32_SHUFFLE0,
            .emit_count = LOOM_X86_AVX512_REDUCE_EMIT_F32X16_EXTRACT0 -
                          LOOM_X86_AVX512_REDUCE_EMIT_F32_SHUFFLE0,
        },
    [LOOM_X86_AVX512_REDUCE_RULE_ADDF_V16] =
        {
            .source_op_kind = LOOM_OP_VECTOR_REDUCE,
            .temporary_count = 11,
            .guard_start = LOOM_X86_AVX512_REDUCE_GUARD_KIND_ADDF_V16,
            .guard_count = IREE_ARRAYSIZE(loom_x86_avx512_reduce_guards) -
                           LOOM_X86_AVX512_REDUCE_GUARD_KIND_ADDF_V16,
            .emit_start = LOOM_X86_AVX512_REDUCE_EMIT_F32X16_EXTRACT0,
            .emit_count = IREE_ARRAYSIZE(loom_x86_avx512_reduce_emits) -
                          LOOM_X86_AVX512_REDUCE_EMIT_F32X16_EXTRACT0,
        },
};

static const loom_low_lower_rule_span_t loom_x86_avx512_reduce_rule_spans[] = {
    {
        .source_op_kind = LOOM_OP_VECTOR_REDUCE,
        .rule_start = LOOM_X86_AVX512_REDUCE_RULE_ADDI,
        .rule_count = IREE_ARRAYSIZE(loom_x86_avx512_reduce_rules),
    },
};

const loom_low_lower_rule_set_t loom_x86_avx512_reduce_rule_set = {
    .spans = loom_x86_avx512_reduce_rule_spans,
    .span_count = IREE_ARRAYSIZE(loom_x86_avx512_reduce_rule_spans),
    .rules = loom_x86_avx512_reduce_rules,
    .rule_count = IREE_ARRAYSIZE(loom_x86_avx512_reduce_rules),
    .type_patterns = loom_x86_avx512_reduce_type_patterns,
    .type_pattern_count = IREE_ARRAYSIZE(loom_x86_avx512_reduce_type_patterns),
    .value_refs = loom_x86_avx512_reduce_value_refs,
    .value_ref_count = IREE_ARRAYSIZE(loom_x86_avx512_reduce_value_refs),
    .guards = loom_x86_avx512_reduce_guards,
    .guard_count = IREE_ARRAYSIZE(loom_x86_avx512_reduce_guards),
    .attr_copies = loom_x86_avx512_reduce_attr_copies,
    .attr_copy_count = IREE_ARRAYSIZE(loom_x86_avx512_reduce_attr_copies),
    .tied_results = loom_x86_avx512_reduce_tied_results,
    .tied_result_count = IREE_ARRAYSIZE(loom_x86_avx512_reduce_tied_results),
    .emits = loom_x86_avx512_reduce_emits,
    .emit_count = IREE_ARRAYSIZE(loom_x86_avx512_reduce_emits),
    .diagnostics = loom_x86_avx512_reduce_diagnostics,
    .diagnostic_count = IREE_ARRAYSIZE(loom_x86_avx512_reduce_diagnostics),
};
