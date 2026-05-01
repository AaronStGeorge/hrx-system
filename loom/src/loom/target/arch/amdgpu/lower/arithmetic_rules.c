// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower_rules.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower/internal.h"

enum loom_amdgpu_arithmetic_type_pattern_e {
  LOOM_AMDGPU_ARITHMETIC_TYPE_VI32 = 0,
  LOOM_AMDGPU_ARITHMETIC_TYPE_VF32 = 1,
  LOOM_AMDGPU_ARITHMETIC_TYPE_SI32 = 2,
  LOOM_AMDGPU_ARITHMETIC_TYPE_SF32 = 3,
  LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS = 4,
};

static const loom_low_lower_type_pattern_t kAmdgpuArithmeticTypePatterns[] = {
    [LOOM_AMDGPU_ARITHMETIC_TYPE_VI32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE,
            .type_kind = LOOM_TYPE_VECTOR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I32),
            .rank = 1,
            .static_dim0_min = 1,
            .static_dim0_max = LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES,
        },
    [LOOM_AMDGPU_ARITHMETIC_TYPE_VF32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE,
            .type_kind = LOOM_TYPE_VECTOR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_F32),
            .rank = 1,
            .static_dim0_min = 1,
            .static_dim0_max = LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES,
        },
    [LOOM_AMDGPU_ARITHMETIC_TYPE_SI32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I32),
        },
    [LOOM_AMDGPU_ARITHMETIC_TYPE_SF32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_F32),
        },
    [LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_INDEX) |
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_OFFSET),
        },
};

enum loom_amdgpu_arithmetic_value_ref_e {
  LOOM_AMDGPU_ARITHMETIC_OPERAND0 = 0,
  LOOM_AMDGPU_ARITHMETIC_OPERAND1 = 1,
  LOOM_AMDGPU_ARITHMETIC_OPERAND2 = 2,
  LOOM_AMDGPU_ARITHMETIC_RESULT0 = 3,
  LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND0 = 4,
  LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND1 = 5,
  LOOM_AMDGPU_ARITHMETIC_TEMPORARY0 = 6,
  LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND2 = 7,
};

enum loom_amdgpu_arithmetic_materializer_e {
  LOOM_AMDGPU_ARITHMETIC_MATERIALIZER_ADDRESS_VGPR = 1,
};

static const loom_low_lower_value_materializer_t
    kAmdgpuArithmeticMaterializers[] = {
        {
            .can_materialize =
                loom_amdgpu_value_can_materialize_as_vgpr_address,
            .materialize = loom_amdgpu_lookup_or_materialize_vgpr_address,
        },
};

static const loom_low_lower_value_ref_t kAmdgpuArithmeticValueRefs[] = {
    [LOOM_AMDGPU_ARITHMETIC_OPERAND0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
        },
    [LOOM_AMDGPU_ARITHMETIC_OPERAND1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_OPERAND2] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 2,
        },
    [LOOM_AMDGPU_ARITHMETIC_RESULT0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
            .index = 0,
        },
    [LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
            .materializer_index =
                LOOM_AMDGPU_ARITHMETIC_MATERIALIZER_ADDRESS_VGPR,
        },
    [LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
            .materializer_index =
                LOOM_AMDGPU_ARITHMETIC_MATERIALIZER_ADDRESS_VGPR,
        },
    [LOOM_AMDGPU_ARITHMETIC_TEMPORARY0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_TEMPORARY,
            .index = 0,
        },
    [LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND2] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 2,
            .materializer_index =
                LOOM_AMDGPU_ARITHMETIC_MATERIALIZER_ADDRESS_VGPR,
        },
};

enum loom_amdgpu_arithmetic_diagnostic_e {
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32 = 0,
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32 = 1,
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32 = 2,
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32 = 3,
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_DESCRIPTOR = 4,
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS = 5,
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_VGPR = 6,
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_RESULT_VGPR = 7,
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_U32 = 8,
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_EXACT = 9,
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_LITERAL_EXACT = 10,
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_LITERAL_I32_BITS = 11,
};

static const loom_low_lower_diagnostic_t kAmdgpuArithmeticDiagnostics[] = {
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("vector<i32>"),
            .reason = IREE_SVL("AMDGPU arithmetic lowering requires a rank-1 "
                               "static i32 vector with 1 to 8 lanes"),
        },
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("vector<f32>"),
            .reason = IREE_SVL("AMDGPU arithmetic lowering requires a rank-1 "
                               "static f32 vector with 1 to 8 lanes"),
        },
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("i32"),
            .reason =
                IREE_SVL("AMDGPU arithmetic lowering requires an i32 scalar"),
        },
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("f32"),
            .reason =
                IREE_SVL("AMDGPU arithmetic lowering requires an f32 scalar"),
        },
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_DESCRIPTOR] =
        {
            .subject_kind = IREE_SVL("descriptor"),
            .subject_name = IREE_SVL("amdgpu.arithmetic"),
            .reason = IREE_SVL("the selected AMDGPU descriptor set does not "
                               "contain the required arithmetic packet"),
        },
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("index"),
            .reason = IREE_SVL("AMDGPU index lowering requires index or offset "
                               "scalar values"),
        },
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_VGPR] =
        {
            .subject_kind = IREE_SVL("materializer"),
            .subject_name = IREE_SVL("address-vgpr"),
            .reason = IREE_SVL("AMDGPU index lowering requires address values "
                               "that can materialize as VGPR operands"),
        },
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_RESULT_VGPR] =
        {
            .subject_kind = IREE_SVL("register-class"),
            .subject_name = IREE_SVL("vgpr"),
            .reason = IREE_SVL("AMDGPU arithmetic lowering requires a VGPR "
                               "result for vector descriptor emission"),
        },
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_U32] =
        {
            .subject_kind = IREE_SVL("address-width"),
            .subject_name = IREE_SVL("u32"),
            .reason = IREE_SVL("AMDGPU index lowering requires address "
                               "results proven non-negative and 32-bit"),
        },
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_EXACT] =
        {
            .subject_kind = IREE_SVL("address-literal"),
            .subject_name = IREE_SVL("i64"),
            .reason = IREE_SVL("AMDGPU literal address operands require exact "
                               "integer value facts"),
        },
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_LITERAL_EXACT] =
        {
            .subject_kind = IREE_SVL("literal"),
            .subject_name = IREE_SVL("i64"),
            .reason = IREE_SVL("AMDGPU literal arithmetic operands require "
                               "exact integer value facts"),
        },
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_LITERAL_I32_BITS] =
        {
            .subject_kind = IREE_SVL("literal-bits"),
            .subject_name = IREE_SVL("i32"),
            .reason = IREE_SVL("AMDGPU literal arithmetic operands must fit "
                               "in a signed i32 bit pattern"),
        },
};

#define LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(value_ref, type_pattern, \
                                           diagnostic)              \
  {                                                                 \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,                      \
      .value_ref_index = value_ref,                                 \
      .type_pattern_index = type_pattern,                           \
      .diagnostic_index = diagnostic,                               \
  }

#define LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(descriptor)             \
  {                                                                     \
      .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,                \
      .diagnostic_index = LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_DESCRIPTOR, \
      .descriptor_id = descriptor,                                      \
  }

#define LOOM_AMDGPU_ARITHMETIC_GUARD_MATERIALIZABLE(value_ref, diagnostic) \
  {                                                                        \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_MATERIALIZABLE,                   \
      .value_ref_index = value_ref,                                        \
      .diagnostic_index = diagnostic,                                      \
  }

#define LOOM_AMDGPU_ARITHMETIC_GUARD_LOW_REG_CLASS(value_ref, reg_class, \
                                                   diagnostic)           \
  {                                                                      \
      .kind = LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS,             \
      .value_ref_index = value_ref,                                      \
      .diagnostic_index = diagnostic,                                    \
      .register_class_id = reg_class,                                    \
  }

#define LOOM_AMDGPU_ARITHMETIC_GUARD_UNSIGNED_BIT_COUNT(value_ref, bit_count, \
                                                        diagnostic)           \
  {                                                                           \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_UNSIGNED_BIT_COUNT,                  \
      .value_ref_index = value_ref,                                           \
      .u64 = bit_count,                                                       \
      .diagnostic_index = diagnostic,                                         \
  }

#define LOOM_AMDGPU_ARITHMETIC_GUARD_SIGNED_BIT_COUNT(value_ref, bit_count, \
                                                      diagnostic)           \
  {                                                                         \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_SIGNED_BIT_COUNT,                  \
      .value_ref_index = value_ref,                                         \
      .u64 = bit_count,                                                     \
      .diagnostic_index = diagnostic,                                       \
  }

#define LOOM_AMDGPU_ARITHMETIC_GUARD_EXACT_I64(value_ref, diagnostic) \
  {                                                                   \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_EXACT_I64,                   \
      .value_ref_index = value_ref,                                   \
      .diagnostic_index = diagnostic,                                 \
  }

#define LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_I32_LITERAL(value_ref,           \
                                                         descriptor)          \
  LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,         \
                                     LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,        \
                                     LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32), \
      LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(                                     \
          LOOM_AMDGPU_ARITHMETIC_OPERAND1, LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,  \
          LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),                            \
      LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(                                     \
          LOOM_AMDGPU_ARITHMETIC_RESULT0, LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,   \
          LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),                            \
      LOOM_AMDGPU_ARITHMETIC_GUARD_EXACT_I64(                                 \
          value_ref, LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_LITERAL_EXACT),        \
      LOOM_AMDGPU_ARITHMETIC_GUARD_SIGNED_BIT_COUNT(                          \
          value_ref, 32, LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_LITERAL_I32_BITS), \
      LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(descriptor)

static const loom_low_lower_guard_t kAmdgpuArithmeticGuards[] = {
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_F32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_F32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_F32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_MIN_F32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_MAX_F32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND2,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_FMA_F32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_ASHRREV_I32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_CVT_F32_I32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_VF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_VF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_CVT_F32_U32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_F32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_F32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_F32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_MIN_F32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_MAX_F32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND2,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_FMA_F32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_CVT_F32_I32),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,
                                       LOOM_AMDGPU_ARITHMETIC_TYPE_SF32,
                                       LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SF32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_CVT_F32_U32),

    LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_I32_LITERAL(
        LOOM_AMDGPU_ARITHMETIC_OPERAND0,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32_LIT),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_I32_LITERAL(
        LOOM_AMDGPU_ARITHMETIC_OPERAND1,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32_LIT),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_I32_LITERAL(
        LOOM_AMDGPU_ARITHMETIC_OPERAND0,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32_LIT),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_I32_LITERAL(
        LOOM_AMDGPU_ARITHMETIC_OPERAND1,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32_LIT),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_I32_LITERAL(
        LOOM_AMDGPU_ARITHMETIC_OPERAND0,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32_LIT),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_I32_LITERAL(
        LOOM_AMDGPU_ARITHMETIC_OPERAND1,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32_LIT),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_I32_LITERAL(
        LOOM_AMDGPU_ARITHMETIC_OPERAND0,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32_LIT),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_I32_LITERAL(
        LOOM_AMDGPU_ARITHMETIC_OPERAND1,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32_LIT),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_I32_LITERAL(
        LOOM_AMDGPU_ARITHMETIC_OPERAND1,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32_LIT),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_I32_LITERAL(
        LOOM_AMDGPU_ARITHMETIC_OPERAND1,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_ASHRREV_I32_LIT),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_I32_LITERAL(
        LOOM_AMDGPU_ARITHMETIC_OPERAND1,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32_LIT),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(
        LOOM_AMDGPU_ARITHMETIC_OPERAND0, LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(
        LOOM_AMDGPU_ARITHMETIC_OPERAND1, LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(
        LOOM_AMDGPU_ARITHMETIC_OPERAND2, LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(
        LOOM_AMDGPU_ARITHMETIC_RESULT0, LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS),
    LOOM_AMDGPU_ARITHMETIC_GUARD_UNSIGNED_BIT_COUNT(
        LOOM_AMDGPU_ARITHMETIC_RESULT0, 32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_LOW_REG_CLASS(
        LOOM_AMDGPU_ARITHMETIC_RESULT0, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_RESULT_VGPR),
    LOOM_AMDGPU_ARITHMETIC_GUARD_MATERIALIZABLE(
        LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_VGPR),
    LOOM_AMDGPU_ARITHMETIC_GUARD_MATERIALIZABLE(
        LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_VGPR),
    LOOM_AMDGPU_ARITHMETIC_GUARD_EXACT_I64(
        LOOM_AMDGPU_ARITHMETIC_OPERAND2,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_EXACT),
    LOOM_AMDGPU_ARITHMETIC_GUARD_UNSIGNED_BIT_COUNT(
        LOOM_AMDGPU_ARITHMETIC_OPERAND2, 32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32_LIT),

    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(
        LOOM_AMDGPU_ARITHMETIC_OPERAND0, LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(
        LOOM_AMDGPU_ARITHMETIC_OPERAND1, LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(
        LOOM_AMDGPU_ARITHMETIC_OPERAND2, LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS),
    LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(
        LOOM_AMDGPU_ARITHMETIC_RESULT0, LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS),
    LOOM_AMDGPU_ARITHMETIC_GUARD_UNSIGNED_BIT_COUNT(
        LOOM_AMDGPU_ARITHMETIC_RESULT0, 32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_LOW_REG_CLASS(
        LOOM_AMDGPU_ARITHMETIC_RESULT0, LOOM_AMDGPU_REG_CLASS_ID_VGPR,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_RESULT_VGPR),
    LOOM_AMDGPU_ARITHMETIC_GUARD_MATERIALIZABLE(
        LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_VGPR),
    LOOM_AMDGPU_ARITHMETIC_GUARD_MATERIALIZABLE(
        LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_VGPR),
    LOOM_AMDGPU_ARITHMETIC_GUARD_MATERIALIZABLE(
        LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND2,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_VGPR),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32),
};

#undef LOOM_AMDGPU_ARITHMETIC_GUARD_EXACT_I64
#undef LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_I32_LITERAL
#undef LOOM_AMDGPU_ARITHMETIC_GUARD_LOW_REG_CLASS
#undef LOOM_AMDGPU_ARITHMETIC_GUARD_SIGNED_BIT_COUNT
#undef LOOM_AMDGPU_ARITHMETIC_GUARD_UNSIGNED_BIT_COUNT
#undef LOOM_AMDGPU_ARITHMETIC_GUARD_MATERIALIZABLE
#undef LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR
#undef LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE

enum loom_amdgpu_arithmetic_guard_range_e {
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ADDF = 0,
  LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY = 4,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SUBF =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ADDF +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_MULF =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SUBF +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_MINNUMF =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_MULF +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_MAXNUMF =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_MINNUMF +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_FMAF =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_MAXNUMF +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_TERNARY = 5,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ADDI =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_FMAF +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_TERNARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SUBI =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ADDI +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_MULI =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SUBI +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ANDI =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_MULI +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ORI =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ANDI +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_XORI =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ORI +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHLI =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_XORI +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHRSI =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHLI +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHRUI =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHRSI +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SITOFP =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHRUI +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_UNARY = 3,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_UITOFP =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SITOFP +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_UNARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ADDF =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_UITOFP +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_UNARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SUBF =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ADDF +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_MULF =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SUBF +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_MINNUMF =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_MULF +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_MAXNUMF =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_MINNUMF +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_FMAF =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_MAXNUMF +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SITOFP =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_FMAF +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_TERNARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_UITOFP =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SITOFP +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_UNARY,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ADDI_OPERAND0_LITERAL =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_UITOFP +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_UNARY,
  LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL = 6,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ADDI_OPERAND1_LITERAL =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ADDI_OPERAND0_LITERAL +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ANDI_OPERAND0_LITERAL =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ADDI_OPERAND1_LITERAL +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ANDI_OPERAND1_LITERAL =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ANDI_OPERAND0_LITERAL +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ORI_OPERAND0_LITERAL =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ANDI_OPERAND1_LITERAL +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ORI_OPERAND1_LITERAL =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ORI_OPERAND0_LITERAL +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_XORI_OPERAND0_LITERAL =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ORI_OPERAND1_LITERAL +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_XORI_OPERAND1_LITERAL =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_XORI_OPERAND0_LITERAL +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHLI_SHIFT_LITERAL =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_XORI_OPERAND1_LITERAL +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHRSI_SHIFT_LITERAL =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHLI_SHIFT_LITERAL +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHRUI_SHIFT_LITERAL =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHRSI_SHIFT_LITERAL +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_MADD =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHRUI_SHIFT_LITERAL +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
  LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_INDEX_MADD_LITERAL = 12,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_MADD_VGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_MADD +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_INDEX_MADD_LITERAL,
  LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_INDEX_MADD = 11,
  LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_ =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_MADD_VGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_INDEX_MADD,
};

static_assert(IREE_ARRAYSIZE(kAmdgpuArithmeticGuards) ==
                  LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_,
              "AMDGPU arithmetic guard ranges must cover the guard table");

enum loom_amdgpu_arithmetic_attr_copy_e {
  LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND0_I32_BITS = 0,
  LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND1_I32_BITS = 1,
  LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND2_EXACT = 2,
};

#define LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_I32_BITS(value_ref)  \
  {                                                           \
      .kind = LOOM_LOW_LOWER_ATTR_COPY_VALUE_I32_AS_U32_BITS, \
      .target_name = IREE_SVL("imm32"),                       \
      .value_ref_index = value_ref,                           \
  }

static const loom_low_lower_attr_copy_t kAmdgpuArithmeticAttrCopies[] = {
    [LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND0_I32_BITS] =
        LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_I32_BITS(
            LOOM_AMDGPU_ARITHMETIC_OPERAND0),
    [LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND1_I32_BITS] =
        LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_I32_BITS(
            LOOM_AMDGPU_ARITHMETIC_OPERAND1),
    [LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND2_EXACT] =
        {
            .kind = LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64,
            .target_name = IREE_SVL("imm32"),
            .value_ref_index = LOOM_AMDGPU_ARITHMETIC_OPERAND2,
        },
};

#undef LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_I32_BITS

enum loom_amdgpu_arithmetic_emit_e {
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ADDF = 0,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SUBF = 1,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_MULF = 2,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_MINNUMF = 3,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_MAXNUMF = 4,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_FMAF = 5,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ADDI = 6,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SUBI = 7,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_MULI = 8,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ANDI = 9,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ORI = 10,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_XORI = 11,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHLI = 12,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHRSI = 13,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHRUI = 14,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SITOFP = 15,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_UITOFP = 16,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ADDF = 17,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SUBF = 18,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_MULF = 19,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_MINNUMF = 20,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_MAXNUMF = 21,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_FMAF = 22,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SITOFP = 23,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_UITOFP = 24,
  LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MADD_PRODUCT = 25,
  LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MADD_ADDEND = 26,
  LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MADD_LITERAL_PRODUCT = 27,
  LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MADD_LITERAL_ADDEND = 28,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ADDI_OPERAND0_LITERAL = 29,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ADDI_OPERAND1_LITERAL = 30,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ANDI_OPERAND0_LITERAL = 31,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ANDI_OPERAND1_LITERAL = 32,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ORI_OPERAND0_LITERAL = 33,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ORI_OPERAND1_LITERAL = 34,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_XORI_OPERAND0_LITERAL = 35,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_XORI_OPERAND1_LITERAL = 36,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHLI_SHIFT_LITERAL = 37,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHRSI_SHIFT_LITERAL = 38,
  LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHRUI_SHIFT_LITERAL = 39,
};

#define LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY(descriptor)      \
  {                                                         \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE,   \
      .descriptor_id = descriptor,                          \
      .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_OPERAND0, \
      .operand_ref_count = 2,                               \
      .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,   \
      .result_ref_count = 1,                                \
  }

#define LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_SWAPPED(descriptor) \
  {                                                            \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE,      \
      .flags = LOOM_LOW_LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1,     \
      .descriptor_id = descriptor,                             \
      .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_OPERAND0,    \
      .operand_ref_count = 2,                                  \
      .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,      \
      .result_ref_count = 1,                                   \
  }

#define LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_LITERAL(descriptor, operand_ref, \
                                                   attr_copy)               \
  {                                                                         \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE,                   \
      .descriptor_id = descriptor,                                          \
      .operand_ref_start = operand_ref,                                     \
      .operand_ref_count = 1,                                               \
      .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,                   \
      .result_ref_count = 1,                                                \
      .attr_copy_start = attr_copy,                                         \
      .attr_copy_count = 1,                                                 \
  }

#define LOOM_AMDGPU_ARITHMETIC_EMIT_UNARY(descriptor)       \
  {                                                         \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE,   \
      .descriptor_id = descriptor,                          \
      .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_OPERAND0, \
      .operand_ref_count = 1,                               \
      .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,   \
      .result_ref_count = 1,                                \
  }

#define LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(descriptor) \
  {                                                           \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,              \
      .descriptor_id = descriptor,                            \
      .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_OPERAND0,   \
      .operand_ref_count = 2,                                 \
      .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,     \
      .result_ref_count = 1,                                  \
  }

#define LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_UNARY(descriptor) \
  {                                                          \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,             \
      .descriptor_id = descriptor,                           \
      .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_OPERAND0,  \
      .operand_ref_count = 1,                                \
      .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,    \
      .result_ref_count = 1,                                 \
  }

static const loom_low_lower_emit_t kAmdgpuArithmeticEmits[] = {
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ADDF] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY(LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_F32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SUBF] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY(LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_F32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_MULF] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY(LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_F32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_MINNUMF] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY(LOOM_AMDGPU_DESCRIPTOR_ID_V_MIN_F32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_MAXNUMF] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY(LOOM_AMDGPU_DESCRIPTOR_ID_V_MAX_F32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_FMAF] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_FMA_F32,
            .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_OPERAND0,
            .operand_ref_count = 3,
            .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ADDI] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY(LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SUBI] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY(LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_MULI] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ANDI] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY(LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ORI] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY(LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_XORI] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY(LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHLI] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_SWAPPED(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHRSI] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_SWAPPED(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ASHRREV_I32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHRUI] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_SWAPPED(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SITOFP] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_UNARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_CVT_F32_I32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_UITOFP] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_UNARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_CVT_F32_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ADDF] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_F32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SUBF] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_F32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_MULF] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_F32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_MINNUMF] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_MIN_F32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_MAXNUMF] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_MAX_F32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_FMAF] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_FMA_F32,
            .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_OPERAND0,
            .operand_ref_count = 3,
            .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SITOFP] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_UNARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_CVT_F32_I32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_UITOFP] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_UNARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_CVT_F32_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MADD_PRODUCT] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .flags = LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32,
            .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,
            .result_ref_count = 1,
            .result_bind_ref_start = LOOM_AMDGPU_ARITHMETIC_TEMPORARY0,
        },
    [LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MADD_ADDEND] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32,
            .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_TEMPORARY0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MADD_LITERAL_PRODUCT] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .flags = LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32,
            .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,
            .result_ref_count = 1,
            .result_bind_ref_start = LOOM_AMDGPU_ARITHMETIC_TEMPORARY0,
        },
    [LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MADD_LITERAL_ADDEND] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32_LIT,
            .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_TEMPORARY0,
            .operand_ref_count = 1,
            .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,
            .result_ref_count = 1,
            .attr_copy_start = LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND2_EXACT,
            .attr_copy_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ADDI_OPERAND0_LITERAL] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_LITERAL(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32_LIT,
            LOOM_AMDGPU_ARITHMETIC_OPERAND1,
            LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND0_I32_BITS),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ADDI_OPERAND1_LITERAL] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_LITERAL(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32_LIT,
            LOOM_AMDGPU_ARITHMETIC_OPERAND0,
            LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND1_I32_BITS),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ANDI_OPERAND0_LITERAL] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_LITERAL(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32_LIT,
            LOOM_AMDGPU_ARITHMETIC_OPERAND1,
            LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND0_I32_BITS),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ANDI_OPERAND1_LITERAL] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_LITERAL(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32_LIT,
            LOOM_AMDGPU_ARITHMETIC_OPERAND0,
            LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND1_I32_BITS),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ORI_OPERAND0_LITERAL] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_LITERAL(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32_LIT,
            LOOM_AMDGPU_ARITHMETIC_OPERAND1,
            LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND0_I32_BITS),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ORI_OPERAND1_LITERAL] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_LITERAL(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32_LIT,
            LOOM_AMDGPU_ARITHMETIC_OPERAND0,
            LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND1_I32_BITS),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_XORI_OPERAND0_LITERAL] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_LITERAL(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32_LIT,
            LOOM_AMDGPU_ARITHMETIC_OPERAND1,
            LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND0_I32_BITS),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_XORI_OPERAND1_LITERAL] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_LITERAL(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32_LIT,
            LOOM_AMDGPU_ARITHMETIC_OPERAND0,
            LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND1_I32_BITS),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHLI_SHIFT_LITERAL] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_LITERAL(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32_LIT,
            LOOM_AMDGPU_ARITHMETIC_OPERAND0,
            LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND1_I32_BITS),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHRSI_SHIFT_LITERAL] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_LITERAL(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ASHRREV_I32_LIT,
            LOOM_AMDGPU_ARITHMETIC_OPERAND0,
            LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND1_I32_BITS),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHRUI_SHIFT_LITERAL] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_LITERAL(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32_LIT,
            LOOM_AMDGPU_ARITHMETIC_OPERAND0,
            LOOM_AMDGPU_ARITHMETIC_ATTR_COPY_OPERAND1_I32_BITS),
};

#undef LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_UNARY
#undef LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY
#undef LOOM_AMDGPU_ARITHMETIC_EMIT_UNARY
#undef LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_LITERAL
#undef LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_SWAPPED
#undef LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY

enum loom_amdgpu_arithmetic_rule_e {
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ADDF = 0,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SUBF = 1,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MULF = 2,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MINNUMF = 3,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MAXNUMF = 4,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_FMAF = 5,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ADDI_OPERAND0_LITERAL = 6,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ADDI_OPERAND1_LITERAL = 7,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ADDI = 8,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SUBI = 9,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MULI = 10,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ANDI_OPERAND0_LITERAL = 11,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ANDI_OPERAND1_LITERAL = 12,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ANDI = 13,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ORI_OPERAND0_LITERAL = 14,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ORI_OPERAND1_LITERAL = 15,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ORI = 16,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_XORI_OPERAND0_LITERAL = 17,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_XORI_OPERAND1_LITERAL = 18,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_XORI = 19,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHLI_SHIFT_LITERAL = 20,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHLI = 21,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRSI_SHIFT_LITERAL = 22,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRSI = 23,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRUI_SHIFT_LITERAL = 24,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRUI = 25,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SITOFP = 26,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_UITOFP = 27,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ADDF = 28,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SUBF = 29,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MULF = 30,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MINNUMF = 31,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MAXNUMF = 32,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_FMAF = 33,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SITOFP = 34,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_UITOFP = 35,
  LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_MADD_LITERAL = 36,
  LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_MADD = 37,
  LOOM_AMDGPU_ARITHMETIC_RULE_COUNT_ = 38,
};

static const loom_low_lower_rule_t kAmdgpuArithmeticRules[] = {
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ADDF] =
        {
            .source_op_kind = LOOM_OP_VECTOR_ADDF,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ADDF,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ADDF,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SUBF] =
        {
            .source_op_kind = LOOM_OP_VECTOR_SUBF,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SUBF,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SUBF,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MULF] =
        {
            .source_op_kind = LOOM_OP_VECTOR_MULF,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_MULF,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_MULF,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MINNUMF] =
        {
            .source_op_kind = LOOM_OP_VECTOR_MINNUMF,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_MINNUMF,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_MINNUMF,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MAXNUMF] =
        {
            .source_op_kind = LOOM_OP_VECTOR_MAXNUMF,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_MAXNUMF,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_MAXNUMF,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_FMAF] =
        {
            .source_op_kind = LOOM_OP_VECTOR_FMAF,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_FMAF,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_TERNARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_FMAF,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ADDI_OPERAND0_LITERAL] =
        {
            .source_op_kind = LOOM_OP_VECTOR_ADDI,
            .guard_start =
                LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ADDI_OPERAND0_LITERAL,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
            .emit_start =
                LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ADDI_OPERAND0_LITERAL,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ADDI_OPERAND1_LITERAL] =
        {
            .source_op_kind = LOOM_OP_VECTOR_ADDI,
            .guard_start =
                LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ADDI_OPERAND1_LITERAL,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
            .emit_start =
                LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ADDI_OPERAND1_LITERAL,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ADDI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_ADDI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ADDI,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ADDI,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SUBI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_SUBI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SUBI,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SUBI,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MULI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_MULI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_MULI,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_MULI,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ANDI_OPERAND0_LITERAL] =
        {
            .source_op_kind = LOOM_OP_VECTOR_ANDI,
            .guard_start =
                LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ANDI_OPERAND0_LITERAL,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
            .emit_start =
                LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ANDI_OPERAND0_LITERAL,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ANDI_OPERAND1_LITERAL] =
        {
            .source_op_kind = LOOM_OP_VECTOR_ANDI,
            .guard_start =
                LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ANDI_OPERAND1_LITERAL,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
            .emit_start =
                LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ANDI_OPERAND1_LITERAL,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ANDI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_ANDI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ANDI,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ANDI,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ORI_OPERAND0_LITERAL] =
        {
            .source_op_kind = LOOM_OP_VECTOR_ORI,
            .guard_start =
                LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ORI_OPERAND0_LITERAL,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
            .emit_start =
                LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ORI_OPERAND0_LITERAL,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ORI_OPERAND1_LITERAL] =
        {
            .source_op_kind = LOOM_OP_VECTOR_ORI,
            .guard_start =
                LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ORI_OPERAND1_LITERAL,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
            .emit_start =
                LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ORI_OPERAND1_LITERAL,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ORI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_ORI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ORI,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ORI,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_XORI_OPERAND0_LITERAL] =
        {
            .source_op_kind = LOOM_OP_VECTOR_XORI,
            .guard_start =
                LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_XORI_OPERAND0_LITERAL,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
            .emit_start =
                LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_XORI_OPERAND0_LITERAL,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_XORI_OPERAND1_LITERAL] =
        {
            .source_op_kind = LOOM_OP_VECTOR_XORI,
            .guard_start =
                LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_XORI_OPERAND1_LITERAL,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
            .emit_start =
                LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_XORI_OPERAND1_LITERAL,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_XORI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_XORI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_XORI,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_XORI,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHLI_SHIFT_LITERAL] =
        {
            .source_op_kind = LOOM_OP_VECTOR_SHLI,
            .guard_start =
                LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHLI_SHIFT_LITERAL,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHLI_SHIFT_LITERAL,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHLI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_SHLI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHLI,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHLI,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRSI_SHIFT_LITERAL] =
        {
            .source_op_kind = LOOM_OP_VECTOR_SHRSI,
            .guard_start =
                LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHRSI_SHIFT_LITERAL,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
            .emit_start =
                LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHRSI_SHIFT_LITERAL,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRSI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_SHRSI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHRSI,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHRSI,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRUI_SHIFT_LITERAL] =
        {
            .source_op_kind = LOOM_OP_VECTOR_SHRUI,
            .guard_start =
                LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHRUI_SHIFT_LITERAL,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_LITERAL,
            .emit_start =
                LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHRUI_SHIFT_LITERAL,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRUI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_SHRUI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHRUI,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHRUI,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SITOFP] =
        {
            .source_op_kind = LOOM_OP_VECTOR_SITOFP,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SITOFP,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_UNARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SITOFP,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_UITOFP] =
        {
            .source_op_kind = LOOM_OP_VECTOR_UITOFP,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_UITOFP,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_UNARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_UITOFP,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ADDF] =
        {
            .source_op_kind = LOOM_OP_SCALAR_ADDF,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ADDF,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ADDF,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SUBF] =
        {
            .source_op_kind = LOOM_OP_SCALAR_SUBF,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SUBF,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SUBF,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MULF] =
        {
            .source_op_kind = LOOM_OP_SCALAR_MULF,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_MULF,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_MULF,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MINNUMF] =
        {
            .source_op_kind = LOOM_OP_SCALAR_MINNUMF,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_MINNUMF,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_MINNUMF,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MAXNUMF] =
        {
            .source_op_kind = LOOM_OP_SCALAR_MAXNUMF,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_MAXNUMF,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_MAXNUMF,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_FMAF] =
        {
            .source_op_kind = LOOM_OP_SCALAR_FMAF,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_FMAF,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_TERNARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_FMAF,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SITOFP] =
        {
            .source_op_kind = LOOM_OP_SCALAR_SITOFP,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SITOFP,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_UNARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SITOFP,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_UITOFP] =
        {
            .source_op_kind = LOOM_OP_SCALAR_UITOFP,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_UITOFP,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_UNARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_UITOFP,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_MADD_LITERAL] =
        {
            .source_op_kind = LOOM_OP_INDEX_MADD,
            .temporary_count = 1,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_MADD,
            .guard_count =
                LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_INDEX_MADD_LITERAL,
            .emit_start =
                LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MADD_LITERAL_PRODUCT,
            .emit_count = 2,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_MADD] =
        {
            .source_op_kind = LOOM_OP_INDEX_MADD,
            .temporary_count = 1,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_MADD_VGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_INDEX_MADD,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MADD_PRODUCT,
            .emit_count = 2,
        },
};

static_assert(IREE_ARRAYSIZE(kAmdgpuArithmeticRules) ==
                  LOOM_AMDGPU_ARITHMETIC_RULE_COUNT_,
              "AMDGPU arithmetic rule indexes must cover the rule table");

static const loom_low_lower_rule_span_t kAmdgpuArithmeticRuleSpans[] = {
    {
        .source_op_kind = LOOM_OP_SCALAR_ADDF,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ADDF,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SUBF,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SUBF,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_MULF,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MULF,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_MINNUMF,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MINNUMF,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_MAXNUMF,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MAXNUMF,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_FMAF,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_FMAF,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SITOFP,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SITOFP,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_UITOFP,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_UITOFP,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_ADDF,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ADDF,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SUBF,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SUBF,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_MULF,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MULF,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_MINNUMF,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MINNUMF,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_MAXNUMF,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MAXNUMF,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_FMAF,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_FMAF,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_ADDI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ADDI_OPERAND0_LITERAL,
        .rule_count = 3,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SUBI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SUBI,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_MULI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MULI,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_ANDI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ANDI_OPERAND0_LITERAL,
        .rule_count = 3,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_ORI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ORI_OPERAND0_LITERAL,
        .rule_count = 3,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_XORI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_XORI_OPERAND0_LITERAL,
        .rule_count = 3,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SHLI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHLI_SHIFT_LITERAL,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SHRSI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRSI_SHIFT_LITERAL,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SHRUI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRUI_SHIFT_LITERAL,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SITOFP,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SITOFP,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_UITOFP,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_UITOFP,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_MADD,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_MADD_LITERAL,
        .rule_count = 2,
    },
};

const loom_low_lower_rule_set_t loom_amdgpu_arithmetic_rule_set = {
    .flags = LOOM_LOW_LOWER_RULE_SET_FLAG_TARGET_CONTRACT_QUERY,
    .spans = kAmdgpuArithmeticRuleSpans,
    .span_count = IREE_ARRAYSIZE(kAmdgpuArithmeticRuleSpans),
    .rules = kAmdgpuArithmeticRules,
    .rule_count = IREE_ARRAYSIZE(kAmdgpuArithmeticRules),
    .type_patterns = kAmdgpuArithmeticTypePatterns,
    .type_pattern_count = IREE_ARRAYSIZE(kAmdgpuArithmeticTypePatterns),
    .value_refs = kAmdgpuArithmeticValueRefs,
    .value_ref_count = IREE_ARRAYSIZE(kAmdgpuArithmeticValueRefs),
    .materializers = kAmdgpuArithmeticMaterializers,
    .materializer_count = IREE_ARRAYSIZE(kAmdgpuArithmeticMaterializers),
    .guards = kAmdgpuArithmeticGuards,
    .guard_count = IREE_ARRAYSIZE(kAmdgpuArithmeticGuards),
    .attr_copies = kAmdgpuArithmeticAttrCopies,
    .attr_copy_count = IREE_ARRAYSIZE(kAmdgpuArithmeticAttrCopies),
    .emits = kAmdgpuArithmeticEmits,
    .emit_count = IREE_ARRAYSIZE(kAmdgpuArithmeticEmits),
    .diagnostics = kAmdgpuArithmeticDiagnostics,
    .diagnostic_count = IREE_ARRAYSIZE(kAmdgpuArithmeticDiagnostics),
};
