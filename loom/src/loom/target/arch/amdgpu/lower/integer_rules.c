// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower_rules.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower/internal.h"

enum loom_amdgpu_integer_type_pattern_e {
  LOOM_AMDGPU_INTEGER_TYPE_SI32 = 0,
  LOOM_AMDGPU_INTEGER_TYPE_ADDRESS = 1,
};

static const loom_low_lower_type_pattern_t kAmdgpuIntegerTypePatterns[] = {
    [LOOM_AMDGPU_INTEGER_TYPE_SI32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I32),
        },
    [LOOM_AMDGPU_INTEGER_TYPE_ADDRESS] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_INDEX) |
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_OFFSET),
        },
};

enum loom_amdgpu_integer_value_ref_e {
  LOOM_AMDGPU_INTEGER_OPERAND0 = 0,
  LOOM_AMDGPU_INTEGER_OPERAND1 = 1,
  LOOM_AMDGPU_INTEGER_RESULT0 = 2,
  LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND0 = 3,
  LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND1 = 4,
  LOOM_AMDGPU_INTEGER_ADDRESS_VGPR_OPERAND0 = 5,
  LOOM_AMDGPU_INTEGER_ADDRESS_VGPR_OPERAND1 = 6,
};

enum loom_amdgpu_integer_materializer_e {
  LOOM_AMDGPU_INTEGER_MATERIALIZER_I32_VGPR = 1,
  LOOM_AMDGPU_INTEGER_MATERIALIZER_ADDRESS_VGPR = 2,
};

static bool loom_amdgpu_integer_can_materialize_i32_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id) {
  (void)source_op;
  return loom_amdgpu_value_can_materialize_as_vgpr_i32(context,
                                                       source_value_id);
}

static iree_status_t loom_amdgpu_integer_materialize_i32_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_value_id_t* out_low_value_id) {
  return loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, source_value_id, out_low_value_id);
}

static bool loom_amdgpu_integer_can_materialize_address_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id) {
  (void)source_op;
  return loom_amdgpu_value_can_materialize_as_vgpr_address(context,
                                                           source_value_id);
}

static iree_status_t loom_amdgpu_integer_materialize_address_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_value_id_t* out_low_value_id) {
  return loom_amdgpu_lookup_or_materialize_vgpr_address(
      context, source_op, source_value_id, out_low_value_id);
}

static const loom_low_lower_value_materializer_t kAmdgpuIntegerMaterializers[] =
    {
        {
            .can_materialize = loom_amdgpu_integer_can_materialize_i32_vgpr,
            .materialize = loom_amdgpu_integer_materialize_i32_vgpr,
        },
        {
            .can_materialize = loom_amdgpu_integer_can_materialize_address_vgpr,
            .materialize = loom_amdgpu_integer_materialize_address_vgpr,
        },
};

static const loom_low_lower_value_ref_t kAmdgpuIntegerValueRefs[] = {
    [LOOM_AMDGPU_INTEGER_OPERAND0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
        },
    [LOOM_AMDGPU_INTEGER_OPERAND1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
        },
    [LOOM_AMDGPU_INTEGER_RESULT0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
            .index = 0,
        },
    [LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
            .materializer_index = LOOM_AMDGPU_INTEGER_MATERIALIZER_I32_VGPR,
        },
    [LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
            .materializer_index = LOOM_AMDGPU_INTEGER_MATERIALIZER_I32_VGPR,
        },
    [LOOM_AMDGPU_INTEGER_ADDRESS_VGPR_OPERAND0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
            .materializer_index = LOOM_AMDGPU_INTEGER_MATERIALIZER_ADDRESS_VGPR,
        },
    [LOOM_AMDGPU_INTEGER_ADDRESS_VGPR_OPERAND1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
            .materializer_index = LOOM_AMDGPU_INTEGER_MATERIALIZER_ADDRESS_VGPR,
        },
};

enum loom_amdgpu_integer_diagnostic_e {
  LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32 = 0,
  LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS = 1,
  LOOM_AMDGPU_INTEGER_DIAGNOSTIC_DESCRIPTOR = 2,
  LOOM_AMDGPU_INTEGER_DIAGNOSTIC_RESULT_VGPR = 3,
  LOOM_AMDGPU_INTEGER_DIAGNOSTIC_I32_VGPR = 4,
  LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS_VGPR = 5,
  LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SGPR = 6,
  LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS_U32 = 7,
};

static const loom_low_lower_diagnostic_t kAmdgpuIntegerDiagnostics[] = {
    [LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("i32"),
            .reason =
                IREE_SVL("AMDGPU integer lowering requires an i32 scalar"),
        },
    [LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("index"),
            .reason = IREE_SVL("AMDGPU index lowering requires index or offset "
                               "scalar values"),
        },
    [LOOM_AMDGPU_INTEGER_DIAGNOSTIC_DESCRIPTOR] =
        {
            .subject_kind = IREE_SVL("descriptor"),
            .subject_name = IREE_SVL("amdgpu.integer"),
            .reason = IREE_SVL("the selected AMDGPU descriptor set does not "
                               "contain the required integer packet"),
        },
    [LOOM_AMDGPU_INTEGER_DIAGNOSTIC_RESULT_VGPR] =
        {
            .subject_kind = IREE_SVL("register-class"),
            .subject_name = IREE_SVL("vgpr"),
            .reason = IREE_SVL("AMDGPU integer lowering requires a VGPR result "
                               "for vector descriptor emission"),
        },
    [LOOM_AMDGPU_INTEGER_DIAGNOSTIC_I32_VGPR] =
        {
            .subject_kind = IREE_SVL("materializer"),
            .subject_name = IREE_SVL("i32-vgpr"),
            .reason = IREE_SVL("AMDGPU integer lowering requires i32 values "
                               "that can materialize as VGPR operands"),
        },
    [LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS_VGPR] =
        {
            .subject_kind = IREE_SVL("materializer"),
            .subject_name = IREE_SVL("address-vgpr"),
            .reason = IREE_SVL("AMDGPU index lowering requires address values "
                               "that can materialize as VGPR operands"),
        },
    [LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SGPR] =
        {
            .subject_kind = IREE_SVL("register-class"),
            .subject_name = IREE_SVL("sgpr"),
            .reason = IREE_SVL("AMDGPU scalar descriptor emission requires "
                               "SGPR operands and results"),
        },
    [LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS_U32] =
        {
            .subject_kind = IREE_SVL("address-width"),
            .subject_name = IREE_SVL("u32"),
            .reason = IREE_SVL("AMDGPU index lowering requires address "
                               "results proven non-negative and 32-bit"),
        },
};

#define LOOM_AMDGPU_INTEGER_GUARD_VALUE(value_ref, type_pattern, diagnostic) \
  {                                                                          \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,                               \
      .value_ref_index = value_ref,                                          \
      .type_pattern_index = type_pattern,                                    \
      .diagnostic_index = diagnostic,                                        \
  }

#define LOOM_AMDGPU_INTEGER_GUARD_DESCRIPTOR(descriptor)             \
  {                                                                  \
      .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,             \
      .diagnostic_index = LOOM_AMDGPU_INTEGER_DIAGNOSTIC_DESCRIPTOR, \
      .descriptor_id = descriptor,                                   \
  }

#define LOOM_AMDGPU_INTEGER_GUARD_LOW_REG_CLASS(value_ref, reg_class, \
                                                diagnostic)           \
  {                                                                   \
      .kind = LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS,          \
      .value_ref_index = value_ref,                                   \
      .diagnostic_index = diagnostic,                                 \
      .register_class_id = reg_class,                                 \
  }

#define LOOM_AMDGPU_INTEGER_GUARD_MATERIALIZABLE(value_ref, diagnostic) \
  {                                                                     \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_MATERIALIZABLE,                \
      .value_ref_index = value_ref,                                     \
      .diagnostic_index = diagnostic,                                   \
  }

#define LOOM_AMDGPU_INTEGER_GUARD_UNSIGNED_BIT_COUNT(value_ref, bit_count, \
                                                     diagnostic)           \
  {                                                                        \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_UNSIGNED_BIT_COUNT,               \
      .value_ref_index = value_ref,                                        \
      .u64 = bit_count,                                                    \
      .diagnostic_index = diagnostic,                                      \
  }

#define LOOM_AMDGPU_INTEGER_GUARDS_SGPR_BINARY(type_pattern, diagnostic,      \
                                               descriptor)                    \
  LOOM_AMDGPU_INTEGER_GUARD_VALUE(LOOM_AMDGPU_INTEGER_OPERAND0, type_pattern, \
                                  diagnostic),                                \
      LOOM_AMDGPU_INTEGER_GUARD_VALUE(LOOM_AMDGPU_INTEGER_OPERAND1,           \
                                      type_pattern, diagnostic),              \
      LOOM_AMDGPU_INTEGER_GUARD_VALUE(LOOM_AMDGPU_INTEGER_RESULT0,            \
                                      type_pattern, diagnostic),              \
      LOOM_AMDGPU_INTEGER_GUARD_LOW_REG_CLASS(                                \
          LOOM_AMDGPU_INTEGER_RESULT0, LOOM_AMDGPU_REG_CLASS_ID_SGPR,         \
          LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SGPR),                               \
      LOOM_AMDGPU_INTEGER_GUARD_LOW_REG_CLASS(                                \
          LOOM_AMDGPU_INTEGER_OPERAND0, LOOM_AMDGPU_REG_CLASS_ID_SGPR,        \
          LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SGPR),                               \
      LOOM_AMDGPU_INTEGER_GUARD_LOW_REG_CLASS(                                \
          LOOM_AMDGPU_INTEGER_OPERAND1, LOOM_AMDGPU_REG_CLASS_ID_SGPR,        \
          LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SGPR),                               \
      LOOM_AMDGPU_INTEGER_GUARD_DESCRIPTOR(descriptor)

#define LOOM_AMDGPU_INTEGER_GUARDS_SGPR_ADDRESS_BINARY(descriptor)             \
  LOOM_AMDGPU_INTEGER_GUARD_VALUE(LOOM_AMDGPU_INTEGER_OPERAND0,                \
                                  LOOM_AMDGPU_INTEGER_TYPE_ADDRESS,            \
                                  LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS),     \
      LOOM_AMDGPU_INTEGER_GUARD_VALUE(LOOM_AMDGPU_INTEGER_OPERAND1,            \
                                      LOOM_AMDGPU_INTEGER_TYPE_ADDRESS,        \
                                      LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS), \
      LOOM_AMDGPU_INTEGER_GUARD_VALUE(LOOM_AMDGPU_INTEGER_RESULT0,             \
                                      LOOM_AMDGPU_INTEGER_TYPE_ADDRESS,        \
                                      LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS), \
      LOOM_AMDGPU_INTEGER_GUARD_UNSIGNED_BIT_COUNT(                            \
          LOOM_AMDGPU_INTEGER_RESULT0, 32,                                     \
          LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS_U32),                         \
      LOOM_AMDGPU_INTEGER_GUARD_LOW_REG_CLASS(                                 \
          LOOM_AMDGPU_INTEGER_RESULT0, LOOM_AMDGPU_REG_CLASS_ID_SGPR,          \
          LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SGPR),                                \
      LOOM_AMDGPU_INTEGER_GUARD_LOW_REG_CLASS(                                 \
          LOOM_AMDGPU_INTEGER_OPERAND0, LOOM_AMDGPU_REG_CLASS_ID_SGPR,         \
          LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SGPR),                                \
      LOOM_AMDGPU_INTEGER_GUARD_LOW_REG_CLASS(                                 \
          LOOM_AMDGPU_INTEGER_OPERAND1, LOOM_AMDGPU_REG_CLASS_ID_SGPR,         \
          LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SGPR),                                \
      LOOM_AMDGPU_INTEGER_GUARD_DESCRIPTOR(descriptor)

#define LOOM_AMDGPU_INTEGER_GUARDS_VGPR_BINARY(type_pattern, diagnostic,      \
                                               operand0_ref, operand1_ref,    \
                                               materializer_diag, descriptor) \
  LOOM_AMDGPU_INTEGER_GUARD_VALUE(LOOM_AMDGPU_INTEGER_OPERAND0, type_pattern, \
                                  diagnostic),                                \
      LOOM_AMDGPU_INTEGER_GUARD_VALUE(LOOM_AMDGPU_INTEGER_OPERAND1,           \
                                      type_pattern, diagnostic),              \
      LOOM_AMDGPU_INTEGER_GUARD_VALUE(LOOM_AMDGPU_INTEGER_RESULT0,            \
                                      type_pattern, diagnostic),              \
      LOOM_AMDGPU_INTEGER_GUARD_LOW_REG_CLASS(                                \
          LOOM_AMDGPU_INTEGER_RESULT0, LOOM_AMDGPU_REG_CLASS_ID_VGPR,         \
          LOOM_AMDGPU_INTEGER_DIAGNOSTIC_RESULT_VGPR),                        \
      LOOM_AMDGPU_INTEGER_GUARD_MATERIALIZABLE(operand0_ref,                  \
                                               materializer_diag),            \
      LOOM_AMDGPU_INTEGER_GUARD_MATERIALIZABLE(operand1_ref,                  \
                                               materializer_diag),            \
      LOOM_AMDGPU_INTEGER_GUARD_DESCRIPTOR(descriptor)

#define LOOM_AMDGPU_INTEGER_GUARDS_VGPR_ADDRESS_BINARY(                        \
    operand0_ref, operand1_ref, descriptor)                                    \
  LOOM_AMDGPU_INTEGER_GUARD_VALUE(LOOM_AMDGPU_INTEGER_OPERAND0,                \
                                  LOOM_AMDGPU_INTEGER_TYPE_ADDRESS,            \
                                  LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS),     \
      LOOM_AMDGPU_INTEGER_GUARD_VALUE(LOOM_AMDGPU_INTEGER_OPERAND1,            \
                                      LOOM_AMDGPU_INTEGER_TYPE_ADDRESS,        \
                                      LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS), \
      LOOM_AMDGPU_INTEGER_GUARD_VALUE(LOOM_AMDGPU_INTEGER_RESULT0,             \
                                      LOOM_AMDGPU_INTEGER_TYPE_ADDRESS,        \
                                      LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS), \
      LOOM_AMDGPU_INTEGER_GUARD_UNSIGNED_BIT_COUNT(                            \
          LOOM_AMDGPU_INTEGER_RESULT0, 32,                                     \
          LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS_U32),                         \
      LOOM_AMDGPU_INTEGER_GUARD_LOW_REG_CLASS(                                 \
          LOOM_AMDGPU_INTEGER_RESULT0, LOOM_AMDGPU_REG_CLASS_ID_VGPR,          \
          LOOM_AMDGPU_INTEGER_DIAGNOSTIC_RESULT_VGPR),                         \
      LOOM_AMDGPU_INTEGER_GUARD_MATERIALIZABLE(                                \
          operand0_ref, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS_VGPR),          \
      LOOM_AMDGPU_INTEGER_GUARD_MATERIALIZABLE(                                \
          operand1_ref, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_ADDRESS_VGPR),          \
      LOOM_AMDGPU_INTEGER_GUARD_DESCRIPTOR(descriptor)

enum loom_amdgpu_integer_rule_e {
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_ADDI_SGPR = 0,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_ADDI_VGPR = 1,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_SUBI_SGPR = 2,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_SUBI_VGPR = 3,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_MULI_VGPR = 4,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_ANDI_SGPR = 5,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_ANDI_VGPR = 6,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_ORI_SGPR = 7,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_ORI_VGPR = 8,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_XORI_SGPR = 9,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_XORI_VGPR = 10,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHLI_SGPR = 11,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHLI_VGPR = 12,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRSI_SGPR = 13,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRSI_VGPR = 14,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRUI_SGPR = 15,
  LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRUI_VGPR = 16,
  LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_SGPR = 17,
  LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_VGPR = 18,
  LOOM_AMDGPU_INTEGER_RULE_INDEX_SUB_SGPR = 19,
  LOOM_AMDGPU_INTEGER_RULE_INDEX_SUB_VGPR = 20,
  LOOM_AMDGPU_INTEGER_RULE_INDEX_MUL_VGPR = 21,
  LOOM_AMDGPU_INTEGER_RULE_COUNT_ = 22,
};

#define LOOM_AMDGPU_INTEGER_GUARD_COUNT_BINARY 7
#define LOOM_AMDGPU_INTEGER_GUARD_COUNT_ADDRESS_BINARY 8
#define LOOM_AMDGPU_INTEGER_GUARD_START(rule)                     \
  ((rule) < LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_SGPR               \
       ? (rule) * LOOM_AMDGPU_INTEGER_GUARD_COUNT_BINARY          \
       : LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_SGPR *                \
                 LOOM_AMDGPU_INTEGER_GUARD_COUNT_BINARY +         \
             ((rule) - LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_SGPR) * \
                 LOOM_AMDGPU_INTEGER_GUARD_COUNT_ADDRESS_BINARY)

static const loom_low_lower_guard_t kAmdgpuIntegerGuards[] = {
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_ADDI_SGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_SGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_ADDI_VGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_VGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND0,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND1,
            LOOM_AMDGPU_INTEGER_DIAGNOSTIC_I32_VGPR,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_SUBI_SGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_SGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_DESCRIPTOR_ID_S_SUB_U32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_SUBI_VGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_VGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND0,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND1,
            LOOM_AMDGPU_INTEGER_DIAGNOSTIC_I32_VGPR,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_MULI_VGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_VGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND0,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND1,
            LOOM_AMDGPU_INTEGER_DIAGNOSTIC_I32_VGPR,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_ANDI_SGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_SGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_DESCRIPTOR_ID_S_AND_B32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_ANDI_VGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_VGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND0,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND1,
            LOOM_AMDGPU_INTEGER_DIAGNOSTIC_I32_VGPR,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_ORI_SGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_SGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_DESCRIPTOR_ID_S_OR_B32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_ORI_VGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_VGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND0,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND1,
            LOOM_AMDGPU_INTEGER_DIAGNOSTIC_I32_VGPR,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_XORI_SGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_SGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_DESCRIPTOR_ID_S_XOR_B32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_XORI_VGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_VGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND0,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND1,
            LOOM_AMDGPU_INTEGER_DIAGNOSTIC_I32_VGPR,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHLI_SGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_SGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHL_B32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHLI_VGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_VGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND0,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND1,
            LOOM_AMDGPU_INTEGER_DIAGNOSTIC_I32_VGPR,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRSI_SGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_SGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_DESCRIPTOR_ID_S_ASHR_I32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRSI_VGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_VGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND0,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND1,
            LOOM_AMDGPU_INTEGER_DIAGNOSTIC_I32_VGPR,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ASHRREV_I32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRUI_SGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_SGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHR_B32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(
        LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRUI_VGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_VGPR_BINARY(
            LOOM_AMDGPU_INTEGER_TYPE_SI32, LOOM_AMDGPU_INTEGER_DIAGNOSTIC_SI32,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND0,
            LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND1,
            LOOM_AMDGPU_INTEGER_DIAGNOSTIC_I32_VGPR,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_SGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_SGPR_ADDRESS_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_VGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_VGPR_ADDRESS_BINARY(
            LOOM_AMDGPU_INTEGER_ADDRESS_VGPR_OPERAND0,
            LOOM_AMDGPU_INTEGER_ADDRESS_VGPR_OPERAND1,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(LOOM_AMDGPU_INTEGER_RULE_INDEX_SUB_SGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_SGPR_ADDRESS_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_SUB_U32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(LOOM_AMDGPU_INTEGER_RULE_INDEX_SUB_VGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_VGPR_ADDRESS_BINARY(
            LOOM_AMDGPU_INTEGER_ADDRESS_VGPR_OPERAND0,
            LOOM_AMDGPU_INTEGER_ADDRESS_VGPR_OPERAND1,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32),
    [LOOM_AMDGPU_INTEGER_GUARD_START(LOOM_AMDGPU_INTEGER_RULE_INDEX_MUL_VGPR)] =
        LOOM_AMDGPU_INTEGER_GUARDS_VGPR_ADDRESS_BINARY(
            LOOM_AMDGPU_INTEGER_ADDRESS_VGPR_OPERAND0,
            LOOM_AMDGPU_INTEGER_ADDRESS_VGPR_OPERAND1,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),
};

static_assert(IREE_ARRAYSIZE(kAmdgpuIntegerGuards) ==
                  LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_SGPR *
                          LOOM_AMDGPU_INTEGER_GUARD_COUNT_BINARY +
                      (LOOM_AMDGPU_INTEGER_RULE_COUNT_ -
                       LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_SGPR) *
                          LOOM_AMDGPU_INTEGER_GUARD_COUNT_ADDRESS_BINARY,
              "AMDGPU integer guard rows must align with rule rows");

#undef LOOM_AMDGPU_INTEGER_GUARDS_VGPR_ADDRESS_BINARY
#undef LOOM_AMDGPU_INTEGER_GUARDS_VGPR_BINARY
#undef LOOM_AMDGPU_INTEGER_GUARDS_SGPR_ADDRESS_BINARY
#undef LOOM_AMDGPU_INTEGER_GUARDS_SGPR_BINARY
#undef LOOM_AMDGPU_INTEGER_GUARD_UNSIGNED_BIT_COUNT
#undef LOOM_AMDGPU_INTEGER_GUARD_MATERIALIZABLE
#undef LOOM_AMDGPU_INTEGER_GUARD_LOW_REG_CLASS
#undef LOOM_AMDGPU_INTEGER_GUARD_DESCRIPTOR
#undef LOOM_AMDGPU_INTEGER_GUARD_VALUE

#define LOOM_AMDGPU_INTEGER_EMIT_SCALAR_BINARY(descriptor) \
  {                                                        \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,           \
      .descriptor_id = descriptor,                         \
      .operand_ref_start = LOOM_AMDGPU_INTEGER_OPERAND0,   \
      .operand_ref_count = 2,                              \
      .result_ref_start = LOOM_AMDGPU_INTEGER_RESULT0,     \
      .result_ref_count = 1,                               \
  }

#define LOOM_AMDGPU_INTEGER_EMIT_I32_VGPR_BINARY(descriptor)      \
  {                                                               \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,                  \
      .descriptor_id = descriptor,                                \
      .operand_ref_start = LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND0, \
      .operand_ref_count = 2,                                     \
      .result_ref_start = LOOM_AMDGPU_INTEGER_RESULT0,            \
      .result_ref_count = 1,                                      \
  }

#define LOOM_AMDGPU_INTEGER_EMIT_I32_VGPR_BINARY_SWAPPED(descriptor) \
  {                                                                  \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,                     \
      .flags = LOOM_LOW_LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1,           \
      .descriptor_id = descriptor,                                   \
      .operand_ref_start = LOOM_AMDGPU_INTEGER_I32_VGPR_OPERAND0,    \
      .operand_ref_count = 2,                                        \
      .result_ref_start = LOOM_AMDGPU_INTEGER_RESULT0,               \
      .result_ref_count = 1,                                         \
  }

#define LOOM_AMDGPU_INTEGER_EMIT_ADDRESS_VGPR_BINARY(descriptor)      \
  {                                                                   \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,                      \
      .descriptor_id = descriptor,                                    \
      .operand_ref_start = LOOM_AMDGPU_INTEGER_ADDRESS_VGPR_OPERAND0, \
      .operand_ref_count = 2,                                         \
      .result_ref_start = LOOM_AMDGPU_INTEGER_RESULT0,                \
      .result_ref_count = 1,                                          \
  }

static const loom_low_lower_emit_t kAmdgpuIntegerEmits[] = {
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_ADDI_SGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_ADDI_VGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_I32_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SUBI_SGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_SUB_U32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SUBI_VGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_I32_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_MULI_VGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_I32_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_ANDI_SGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_AND_B32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_ANDI_VGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_I32_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_ORI_SGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_OR_B32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_ORI_VGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_I32_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_XORI_SGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_XOR_B32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_XORI_VGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_I32_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHLI_SGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHL_B32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHLI_VGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_I32_VGPR_BINARY_SWAPPED(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHLREV_B32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRSI_SGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_ASHR_I32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRSI_VGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_I32_VGPR_BINARY_SWAPPED(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ASHRREV_I32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRUI_SGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_LSHR_B32),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRUI_VGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_I32_VGPR_BINARY_SWAPPED(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_LSHRREV_B32),
    [LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_SGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_ADD_U32),
    [LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_VGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_ADDRESS_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32),
    [LOOM_AMDGPU_INTEGER_RULE_INDEX_SUB_SGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_SCALAR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_S_SUB_U32),
    [LOOM_AMDGPU_INTEGER_RULE_INDEX_SUB_VGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_ADDRESS_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_SUB_U32),
    [LOOM_AMDGPU_INTEGER_RULE_INDEX_MUL_VGPR] =
        LOOM_AMDGPU_INTEGER_EMIT_ADDRESS_VGPR_BINARY(
            LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),
};

static_assert(IREE_ARRAYSIZE(kAmdgpuIntegerEmits) ==
                  LOOM_AMDGPU_INTEGER_RULE_COUNT_,
              "AMDGPU integer emit rows must align with rule rows");

#undef LOOM_AMDGPU_INTEGER_EMIT_ADDRESS_VGPR_BINARY
#undef LOOM_AMDGPU_INTEGER_EMIT_I32_VGPR_BINARY_SWAPPED
#undef LOOM_AMDGPU_INTEGER_EMIT_I32_VGPR_BINARY
#undef LOOM_AMDGPU_INTEGER_EMIT_SCALAR_BINARY

#define LOOM_AMDGPU_INTEGER_RULE(source_op, rule)            \
  {                                                          \
      .source_op_kind = source_op,                           \
      .guard_start = LOOM_AMDGPU_INTEGER_GUARD_START(rule),  \
      .guard_count = LOOM_AMDGPU_INTEGER_GUARD_COUNT_BINARY, \
      .emit_start = rule,                                    \
      .emit_count = 1,                                       \
  }

static const loom_low_lower_rule_t kAmdgpuIntegerRules[] = {
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_ADDI_SGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_ADDI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_ADDI_SGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_ADDI_VGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_ADDI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_ADDI_VGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SUBI_SGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_SUBI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_SUBI_SGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SUBI_VGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_SUBI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_SUBI_VGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_MULI_VGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_MULI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_MULI_VGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_ANDI_SGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_ANDI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_ANDI_SGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_ANDI_VGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_ANDI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_ANDI_VGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_ORI_SGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_ORI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_ORI_SGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_ORI_VGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_ORI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_ORI_VGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_XORI_SGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_XORI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_XORI_SGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_XORI_VGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_XORI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_XORI_VGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHLI_SGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_SHLI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHLI_SGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHLI_VGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_SHLI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHLI_VGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRSI_SGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_SHRSI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRSI_SGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRSI_VGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_SHRSI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRSI_VGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRUI_SGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_SHRUI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRUI_SGPR),
    [LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRUI_VGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_SCALAR_SHRUI, LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRUI_VGPR),
    [LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_SGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_INDEX_ADD, LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_SGPR),
    [LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_VGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_INDEX_ADD, LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_VGPR),
    [LOOM_AMDGPU_INTEGER_RULE_INDEX_SUB_SGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_INDEX_SUB, LOOM_AMDGPU_INTEGER_RULE_INDEX_SUB_SGPR),
    [LOOM_AMDGPU_INTEGER_RULE_INDEX_SUB_VGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_INDEX_SUB, LOOM_AMDGPU_INTEGER_RULE_INDEX_SUB_VGPR),
    [LOOM_AMDGPU_INTEGER_RULE_INDEX_MUL_VGPR] = LOOM_AMDGPU_INTEGER_RULE(
        LOOM_OP_INDEX_MUL, LOOM_AMDGPU_INTEGER_RULE_INDEX_MUL_VGPR),
};

static_assert(IREE_ARRAYSIZE(kAmdgpuIntegerRules) ==
                  LOOM_AMDGPU_INTEGER_RULE_COUNT_,
              "AMDGPU integer rule indexes must cover the rule table");

#undef LOOM_AMDGPU_INTEGER_RULE

static const loom_low_lower_rule_span_t kAmdgpuIntegerRuleSpans[] = {
    {
        .source_op_kind = LOOM_OP_SCALAR_ADDI,
        .rule_start = LOOM_AMDGPU_INTEGER_RULE_SCALAR_ADDI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SUBI,
        .rule_start = LOOM_AMDGPU_INTEGER_RULE_SCALAR_SUBI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_MULI,
        .rule_start = LOOM_AMDGPU_INTEGER_RULE_SCALAR_MULI_VGPR,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_ANDI,
        .rule_start = LOOM_AMDGPU_INTEGER_RULE_SCALAR_ANDI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_ORI,
        .rule_start = LOOM_AMDGPU_INTEGER_RULE_SCALAR_ORI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_XORI,
        .rule_start = LOOM_AMDGPU_INTEGER_RULE_SCALAR_XORI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SHLI,
        .rule_start = LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHLI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SHRSI,
        .rule_start = LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRSI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SHRUI,
        .rule_start = LOOM_AMDGPU_INTEGER_RULE_SCALAR_SHRUI_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_ADD,
        .rule_start = LOOM_AMDGPU_INTEGER_RULE_INDEX_ADD_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_SUB,
        .rule_start = LOOM_AMDGPU_INTEGER_RULE_INDEX_SUB_SGPR,
        .rule_count = 2,
    },
    {
        .source_op_kind = LOOM_OP_INDEX_MUL,
        .rule_start = LOOM_AMDGPU_INTEGER_RULE_INDEX_MUL_VGPR,
        .rule_count = 1,
    },
};

static_assert(LOOM_OP_SCALAR_MULI < LOOM_OP_SCALAR_ANDI &&
                  LOOM_OP_SCALAR_SHRUI < LOOM_OP_INDEX_ADD &&
                  LOOM_OP_INDEX_MUL < LOOM_OP_INDEX_MADD,
              "AMDGPU integer rule spans must stay sorted by source op kind");

const loom_low_lower_rule_set_t loom_amdgpu_integer_rule_set = {
    .spans = kAmdgpuIntegerRuleSpans,
    .span_count = IREE_ARRAYSIZE(kAmdgpuIntegerRuleSpans),
    .rules = kAmdgpuIntegerRules,
    .rule_count = IREE_ARRAYSIZE(kAmdgpuIntegerRules),
    .type_patterns = kAmdgpuIntegerTypePatterns,
    .type_pattern_count = IREE_ARRAYSIZE(kAmdgpuIntegerTypePatterns),
    .value_refs = kAmdgpuIntegerValueRefs,
    .value_ref_count = IREE_ARRAYSIZE(kAmdgpuIntegerValueRefs),
    .materializers = kAmdgpuIntegerMaterializers,
    .materializer_count = IREE_ARRAYSIZE(kAmdgpuIntegerMaterializers),
    .guards = kAmdgpuIntegerGuards,
    .guard_count = IREE_ARRAYSIZE(kAmdgpuIntegerGuards),
    .emits = kAmdgpuIntegerEmits,
    .emit_count = IREE_ARRAYSIZE(kAmdgpuIntegerEmits),
    .diagnostics = kAmdgpuIntegerDiagnostics,
    .diagnostic_count = IREE_ARRAYSIZE(kAmdgpuIntegerDiagnostics),
};

#undef LOOM_AMDGPU_INTEGER_GUARD_START
#undef LOOM_AMDGPU_INTEGER_GUARD_COUNT_BINARY
