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
#include "loom/target/arch/amdgpu/lower_internal.h"

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
  LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND0 = 4,
  LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND1 = 5,
  LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND0 = 6,
  LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND1 = 7,
  LOOM_AMDGPU_ARITHMETIC_TEMPORARY0 = 8,
  LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND2 = 9,
};

enum loom_amdgpu_arithmetic_materializer_e {
  LOOM_AMDGPU_ARITHMETIC_MATERIALIZER_I32_VGPR = 1,
  LOOM_AMDGPU_ARITHMETIC_MATERIALIZER_ADDRESS_VGPR = 2,
};

static bool loom_amdgpu_arithmetic_can_materialize_i32_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id) {
  (void)source_op;
  return loom_amdgpu_value_can_materialize_as_vgpr_i32(context,
                                                       source_value_id);
}

static iree_status_t loom_amdgpu_arithmetic_materialize_i32_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_value_id_t* out_low_value_id) {
  return loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, source_value_id, out_low_value_id);
}

static bool loom_amdgpu_arithmetic_can_materialize_address_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id) {
  (void)source_op;
  return loom_amdgpu_value_can_materialize_as_vgpr_address(context,
                                                           source_value_id);
}

static iree_status_t loom_amdgpu_arithmetic_materialize_address_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_value_id_t* out_low_value_id) {
  return loom_amdgpu_lookup_or_materialize_vgpr_address(
      context, source_op, source_value_id, out_low_value_id);
}

static const loom_low_lower_value_materializer_t
    kAmdgpuArithmeticMaterializers[] = {
        {
            .can_materialize = loom_amdgpu_arithmetic_can_materialize_i32_vgpr,
            .materialize = loom_amdgpu_arithmetic_materialize_i32_vgpr,
        },
        {
            .can_materialize =
                loom_amdgpu_arithmetic_can_materialize_address_vgpr,
            .materialize = loom_amdgpu_arithmetic_materialize_address_vgpr,
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
    [LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
            .materializer_index = LOOM_AMDGPU_ARITHMETIC_MATERIALIZER_I32_VGPR,
        },
    [LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
            .materializer_index = LOOM_AMDGPU_ARITHMETIC_MATERIALIZER_I32_VGPR,
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
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_I32_VGPR = 8,
  LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SGPR = 9,
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
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_I32_VGPR] =
        {
            .subject_kind = IREE_SVL("materializer"),
            .subject_name = IREE_SVL("i32-vgpr"),
            .reason = IREE_SVL("AMDGPU scalar integer lowering requires i32 "
                               "values that can materialize as VGPR operands"),
        },
    [LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SGPR] =
        {
            .subject_kind = IREE_SVL("register-class"),
            .subject_name = IREE_SVL("sgpr"),
            .reason = IREE_SVL("AMDGPU scalar descriptor emission requires "
                               "SGPR operands and results"),
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

#define LOOM_AMDGPU_ARITHMETIC_GUARDS_SGPR_BINARY(type_pattern, diagnostic, \
                                                  descriptor)               \
  LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,       \
                                     type_pattern, diagnostic),             \
      LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,   \
                                         type_pattern, diagnostic),         \
      LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,    \
                                         type_pattern, diagnostic),         \
      LOOM_AMDGPU_ARITHMETIC_GUARD_LOW_REG_CLASS(                           \
          LOOM_AMDGPU_ARITHMETIC_RESULT0, LOOM_AMDGPU_REG_CLASS_ID_SGPR,    \
          LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SGPR),                          \
      LOOM_AMDGPU_ARITHMETIC_GUARD_LOW_REG_CLASS(                           \
          LOOM_AMDGPU_ARITHMETIC_OPERAND0, LOOM_AMDGPU_REG_CLASS_ID_SGPR,   \
          LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SGPR),                          \
      LOOM_AMDGPU_ARITHMETIC_GUARD_LOW_REG_CLASS(                           \
          LOOM_AMDGPU_ARITHMETIC_OPERAND1, LOOM_AMDGPU_REG_CLASS_ID_SGPR,   \
          LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SGPR),                          \
      LOOM_AMDGPU_ARITHMETIC_GUARD_DESCRIPTOR(descriptor)

#define LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY(                           \
    type_pattern, diagnostic, operand0_ref, operand1_ref, materializer_diag, \
    descriptor)                                                              \
  LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND0,        \
                                     type_pattern, diagnostic),              \
      LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_OPERAND1,    \
                                         type_pattern, diagnostic),          \
      LOOM_AMDGPU_ARITHMETIC_GUARD_VALUE(LOOM_AMDGPU_ARITHMETIC_RESULT0,     \
                                         type_pattern, diagnostic),          \
      LOOM_AMDGPU_ARITHMETIC_GUARD_LOW_REG_CLASS(                            \
          LOOM_AMDGPU_ARITHMETIC_RESULT0, LOOM_AMDGPU_REG_CLASS_ID_VGPR,     \
          LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_RESULT_VGPR),                    \
      LOOM_AMDGPU_ARITHMETIC_GUARD_MATERIALIZABLE(operand0_ref,              \
                                                  materializer_diag),        \
      LOOM_AMDGPU_ARITHMETIC_GUARD_MATERIALIZABLE(operand1_ref,              \
                                                  materializer_diag),        \
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

    LOOM_AMDGPU_ARITHMETIC_GUARDS_SGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_I32_VGPR,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_SGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_DESCRIPTOR_ID_S_SUB_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_I32_VGPR,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_I32_VGPR,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_SGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_DESCRIPTOR_ID_S_AND_B32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_I32_VGPR,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_SGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_DESCRIPTOR_ID_S_OR_B32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_I32_VGPR,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_SGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_DESCRIPTOR_ID_S_XOR_B32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_I32_VGPR,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_SGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHL_B32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_I32_VGPR,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_SGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_DESCRIPTOR_ID_S_ASHR_I32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_I32_VGPR,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_ASHRREV_I32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_SGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHR_B32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_SI32,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_SI32,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_I32_VGPR,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32),

    LOOM_AMDGPU_ARITHMETIC_GUARDS_SGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS,
        LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_VGPR,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_SGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS,
        LOOM_AMDGPU_DESCRIPTOR_ID_S_SUB_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_VGPR,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32),
    LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY(
        LOOM_AMDGPU_ARITHMETIC_TYPE_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS,
        LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND0,
        LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND1,
        LOOM_AMDGPU_ARITHMETIC_DIAGNOSTIC_ADDRESS_VGPR,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),
};

#undef LOOM_AMDGPU_ARITHMETIC_GUARDS_VGPR_BINARY
#undef LOOM_AMDGPU_ARITHMETIC_GUARDS_SGPR_BINARY
#undef LOOM_AMDGPU_ARITHMETIC_GUARD_LOW_REG_CLASS
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
  LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_MADD =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_UITOFP +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_UNARY,
  LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_INDEX_MADD = 10,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ADDI_SGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_MADD +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_INDEX_MADD,
  LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG = 7,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ADDI_VGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ADDI_SGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SUBI_SGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ADDI_VGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SUBI_VGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SUBI_SGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_MULI_VGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SUBI_VGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ANDI_SGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_MULI_VGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ANDI_VGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ANDI_SGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ORI_SGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ANDI_VGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ORI_VGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ORI_SGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_XORI_SGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ORI_VGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_XORI_VGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_XORI_SGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHLI_SGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_XORI_VGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHLI_VGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHLI_SGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHRSI_SGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHLI_VGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHRSI_VGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHRSI_SGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHRUI_SGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHRSI_VGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHRUI_VGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHRUI_SGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_ADD_SGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHRUI_VGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_ADD_VGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_ADD_SGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_SUB_SGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_ADD_VGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_SUB_VGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_SUB_SGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_MUL_VGPR =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_SUB_VGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
  LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_ =
      LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_MUL_VGPR +
      LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
};

static_assert(IREE_ARRAYSIZE(kAmdgpuArithmeticGuards) ==
                  LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_,
              "AMDGPU arithmetic guard ranges must cover the guard table");

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
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ADDI_SGPR = 27,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ADDI_VGPR = 28,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SUBI_SGPR = 29,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SUBI_VGPR = 30,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_MULI_VGPR = 31,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ANDI_SGPR = 32,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ANDI_VGPR = 33,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ORI_SGPR = 34,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ORI_VGPR = 35,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_XORI_SGPR = 36,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_XORI_VGPR = 37,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHLI_SGPR = 38,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHLI_VGPR = 39,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHRSI_SGPR = 40,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHRSI_VGPR = 41,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHRUI_SGPR = 42,
  LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHRUI_VGPR = 43,
  LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_ADD_SGPR = 44,
  LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_ADD_VGPR = 45,
  LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_SUB_SGPR = 46,
  LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_SUB_VGPR = 47,
  LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MUL_VGPR = 48,
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

#define LOOM_AMDGPU_ARITHMETIC_EMIT_I32_VGPR_BINARY(descriptor)      \
  {                                                                  \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,                     \
      .descriptor_id = descriptor,                                   \
      .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND0, \
      .operand_ref_count = 2,                                        \
      .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,            \
      .result_ref_count = 1,                                         \
  }

#define LOOM_AMDGPU_ARITHMETIC_EMIT_I32_VGPR_BINARY_SWAPPED(descriptor) \
  {                                                                     \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,                        \
      .flags = LOOM_LOW_LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1,              \
      .descriptor_id = descriptor,                                      \
      .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_I32_VGPR_OPERAND0,    \
      .operand_ref_count = 2,                                           \
      .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,               \
      .result_ref_count = 1,                                            \
  }

#define LOOM_AMDGPU_ARITHMETIC_EMIT_ADDRESS_VGPR_BINARY(descriptor)      \
  {                                                                      \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,                         \
      .descriptor_id = descriptor,                                       \
      .operand_ref_start = LOOM_AMDGPU_ARITHMETIC_ADDRESS_VGPR_OPERAND0, \
      .operand_ref_count = 2,                                            \
      .result_ref_start = LOOM_AMDGPU_ARITHMETIC_RESULT0,                \
      .result_ref_count = 1,                                             \
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
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ADDI_SGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ADDI_VGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_I32_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SUBI_SGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_SUB_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SUBI_VGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_I32_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_MULI_VGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_I32_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ANDI_SGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_AND_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ANDI_VGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_I32_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ORI_SGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_OR_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ORI_VGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_I32_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_XORI_SGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_XOR_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_XORI_VGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_I32_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHLI_SGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHL_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHLI_VGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_I32_VGPR_BINARY_SWAPPED(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHRSI_SGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_ASHR_I32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHRSI_VGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_I32_VGPR_BINARY_SWAPPED(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ASHRREV_I32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHRUI_SGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHR_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHRUI_VGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_I32_VGPR_BINARY_SWAPPED(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_ADD_SGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_ADD_VGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_ADDRESS_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_SUB_SGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_SUB_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_SUB_VGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_ADDRESS_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32),
    [LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MUL_VGPR] =
        LOOM_AMDGPU_ARITHMETIC_EMIT_ADDRESS_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),
};

#undef LOOM_AMDGPU_ARITHMETIC_EMIT_ADDRESS_VGPR_BINARY
#undef LOOM_AMDGPU_ARITHMETIC_EMIT_I32_VGPR_BINARY_SWAPPED
#undef LOOM_AMDGPU_ARITHMETIC_EMIT_I32_VGPR_BINARY
#undef LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_UNARY
#undef LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_BINARY
#undef LOOM_AMDGPU_ARITHMETIC_EMIT_UNARY
#undef LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY_SWAPPED
#undef LOOM_AMDGPU_ARITHMETIC_EMIT_BINARY

enum loom_amdgpu_arithmetic_rule_e {
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ADDF = 0,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SUBF = 1,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MULF = 2,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MINNUMF = 3,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MAXNUMF = 4,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_FMAF = 5,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ADDI = 6,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SUBI = 7,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_MULI = 8,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ANDI = 9,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ORI = 10,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_XORI = 11,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHLI = 12,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRSI = 13,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRUI = 14,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SITOFP = 15,
  LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_UITOFP = 16,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ADDF = 17,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SUBF = 18,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MULF = 19,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MINNUMF = 20,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MAXNUMF = 21,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_FMAF = 22,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SITOFP = 23,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_UITOFP = 24,
  LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_MADD = 25,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ADDI_SGPR = 26,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ADDI_VGPR = 27,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SUBI_SGPR = 28,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SUBI_VGPR = 29,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MULI_VGPR = 30,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ANDI_SGPR = 31,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ANDI_VGPR = 32,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ORI_SGPR = 33,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ORI_VGPR = 34,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_XORI_SGPR = 35,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_XORI_VGPR = 36,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHLI_SGPR = 37,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHLI_VGPR = 38,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHRSI_SGPR = 39,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHRSI_VGPR = 40,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHRUI_SGPR = 41,
  LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHRUI_VGPR = 42,
  LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_ADD_SGPR = 43,
  LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_ADD_VGPR = 44,
  LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_SUB_SGPR = 45,
  LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_SUB_VGPR = 46,
  LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_MUL_VGPR = 47,
  LOOM_AMDGPU_ARITHMETIC_RULE_COUNT_ = 48,
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
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ANDI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_ANDI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_ANDI,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_ANDI,
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
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_XORI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_XORI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_XORI,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_XORI,
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
    [LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRSI] =
        {
            .source_op_kind = LOOM_OP_VECTOR_SHRSI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_VECTOR_SHRSI,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_VECTOR_SHRSI,
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
    [LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_MADD] =
        {
            .source_op_kind = LOOM_OP_INDEX_MADD,
            .temporary_count = 1,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_MADD,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_INDEX_MADD,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MADD_PRODUCT,
            .emit_count = 2,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ADDI_SGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_ADDI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ADDI_SGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ADDI_SGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ADDI_VGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_ADDI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ADDI_VGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ADDI_VGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SUBI_SGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_SUBI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SUBI_SGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SUBI_SGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SUBI_VGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_SUBI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SUBI_VGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SUBI_VGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MULI_VGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_MULI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_MULI_VGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_MULI_VGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ANDI_SGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_ANDI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ANDI_SGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ANDI_SGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ANDI_VGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_ANDI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ANDI_VGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ANDI_VGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ORI_SGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_ORI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ORI_SGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ORI_SGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ORI_VGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_ORI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_ORI_VGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_ORI_VGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_XORI_SGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_XORI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_XORI_SGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_XORI_SGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_XORI_VGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_XORI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_XORI_VGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_XORI_VGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHLI_SGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_SHLI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHLI_SGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHLI_SGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHLI_VGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_SHLI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHLI_VGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHLI_VGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHRSI_SGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_SHRSI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHRSI_SGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHRSI_SGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHRSI_VGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_SHRSI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHRSI_VGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHRSI_VGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHRUI_SGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_SHRUI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHRUI_SGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHRUI_SGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHRUI_VGPR] =
        {
            .source_op_kind = LOOM_OP_SCALAR_SHRUI,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_SCALAR_SHRUI_VGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_SCALAR_SHRUI_VGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_ADD_SGPR] =
        {
            .source_op_kind = LOOM_OP_INDEX_ADD,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_ADD_SGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_ADD_SGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_ADD_VGPR] =
        {
            .source_op_kind = LOOM_OP_INDEX_ADD,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_ADD_VGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_ADD_VGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_SUB_SGPR] =
        {
            .source_op_kind = LOOM_OP_INDEX_SUB,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_SUB_SGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_SUB_SGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_SUB_VGPR] =
        {
            .source_op_kind = LOOM_OP_INDEX_SUB,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_SUB_VGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_SUB_VGPR,
            .emit_count = 1,
        },
    [LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_MUL_VGPR] =
        {
            .source_op_kind = LOOM_OP_INDEX_MUL,
            .guard_start = LOOM_AMDGPU_ARITHMETIC_GUARDS_INDEX_MUL_VGPR,
            .guard_count = LOOM_AMDGPU_ARITHMETIC_GUARD_COUNT_BINARY_REG,
            .emit_start = LOOM_AMDGPU_ARITHMETIC_EMIT_INDEX_MUL_VGPR,
            .emit_count = 1,
        },
};

static_assert(IREE_ARRAYSIZE(kAmdgpuArithmeticRules) ==
                  LOOM_AMDGPU_ARITHMETIC_RULE_COUNT_,
              "AMDGPU arithmetic rule indexes must cover the rule table");

static const loom_low_lower_rule_span_t kAmdgpuArithmeticRuleSpans[] = {
    {
        .source_op_kind = LOOM_OP_SCALAR_ADDI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ADDI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SUBI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SUBI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_MULI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_MULI_VGPR,
        .rule_count = 1,
    },
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
        .source_op_kind = LOOM_OP_SCALAR_ANDI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ANDI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_ORI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_ORI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_XORI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_XORI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SHLI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHLI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SHRSI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHRSI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SHRUI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_SCALAR_SHRUI_SGPR,
        .rule_count = 2,
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
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ADDI,
        .rule_count = 1,
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
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ANDI,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_ORI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_ORI,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_XORI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_XORI,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SHLI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHLI,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SHRSI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRSI,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_SHRUI,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_VECTOR_SHRUI,
        .rule_count = 1,
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
        .source_op_kind = LOOM_OP_INDEX_ADD,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_ADD_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_SUB,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_SUB_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_MUL,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_MUL_VGPR,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_MADD,
        .rule_start = LOOM_AMDGPU_ARITHMETIC_RULE_INDEX_MADD,
        .rule_count = 1,
    },
};

static_assert(LOOM_OP_SCALAR_MULI < LOOM_OP_SCALAR_ADDF &&
                  LOOM_OP_SCALAR_UITOFP < LOOM_OP_SCALAR_ANDI &&
                  LOOM_OP_SCALAR_SHRUI < LOOM_OP_VECTOR_ADDF &&
                  LOOM_OP_VECTOR_UITOFP < LOOM_OP_INDEX_ADD &&
                  LOOM_OP_INDEX_MUL < LOOM_OP_INDEX_MADD,
              "AMDGPU arithmetic rule spans must stay sorted by source op "
              "kind");

const loom_low_lower_rule_set_t loom_amdgpu_arithmetic_rule_set = {
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
    .emits = kAmdgpuArithmeticEmits,
    .emit_count = IREE_ARRAYSIZE(kAmdgpuArithmeticEmits),
    .diagnostics = kAmdgpuArithmeticDiagnostics,
    .diagnostic_count = IREE_ARRAYSIZE(kAmdgpuArithmeticDiagnostics),
};
