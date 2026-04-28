// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower_rules.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower/internal.h"

enum loom_amdgpu_compare_type_pattern_e {
  LOOM_AMDGPU_COMPARE_TYPE_SI1 = 0,
  LOOM_AMDGPU_COMPARE_TYPE_SI32 = 1,
};

static const loom_low_lower_type_pattern_t kAmdgpuCompareTypePatterns[] = {
    [LOOM_AMDGPU_COMPARE_TYPE_SI1] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I1),
        },
    [LOOM_AMDGPU_COMPARE_TYPE_SI32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I32),
        },
};

enum loom_amdgpu_compare_value_ref_e {
  LOOM_AMDGPU_COMPARE_OPERAND0 = 0,
  LOOM_AMDGPU_COMPARE_OPERAND1 = 1,
  LOOM_AMDGPU_COMPARE_RESULT0 = 2,
  LOOM_AMDGPU_COMPARE_I32_VGPR_OPERAND0 = 3,
  LOOM_AMDGPU_COMPARE_I32_VGPR_OPERAND1 = 4,
};

enum loom_amdgpu_compare_materializer_e {
  LOOM_AMDGPU_COMPARE_MATERIALIZER_I32_VGPR = 1,
};

static bool loom_amdgpu_compare_can_materialize_i32_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id) {
  (void)source_op;
  return loom_amdgpu_value_can_materialize_as_vgpr_i32(context,
                                                       source_value_id);
}

static iree_status_t loom_amdgpu_compare_materialize_i32_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_value_id_t* out_low_value_id) {
  return loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, source_value_id, out_low_value_id);
}

static const loom_low_lower_value_materializer_t kAmdgpuCompareMaterializers[] =
    {
        {
            .can_materialize = loom_amdgpu_compare_can_materialize_i32_vgpr,
            .materialize = loom_amdgpu_compare_materialize_i32_vgpr,
        },
};

static const loom_low_lower_value_ref_t kAmdgpuCompareValueRefs[] = {
    [LOOM_AMDGPU_COMPARE_OPERAND0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
        },
    [LOOM_AMDGPU_COMPARE_OPERAND1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
        },
    [LOOM_AMDGPU_COMPARE_RESULT0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
            .index = 0,
        },
    [LOOM_AMDGPU_COMPARE_I32_VGPR_OPERAND0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
            .materializer_index = LOOM_AMDGPU_COMPARE_MATERIALIZER_I32_VGPR,
        },
    [LOOM_AMDGPU_COMPARE_I32_VGPR_OPERAND1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
            .materializer_index = LOOM_AMDGPU_COMPARE_MATERIALIZER_I32_VGPR,
        },
};

enum loom_amdgpu_compare_diagnostic_e {
  LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I32 = 0,
  LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I1 = 1,
  LOOM_AMDGPU_COMPARE_DIAGNOSTIC_PREDICATE = 2,
  LOOM_AMDGPU_COMPARE_DIAGNOSTIC_DESCRIPTOR = 3,
  LOOM_AMDGPU_COMPARE_DIAGNOSTIC_SGPR = 4,
  LOOM_AMDGPU_COMPARE_DIAGNOSTIC_SCC = 5,
  LOOM_AMDGPU_COMPARE_DIAGNOSTIC_MASK = 6,
  LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I32_VGPR = 7,
};

static const loom_low_lower_diagnostic_t kAmdgpuCompareDiagnostics[] = {
    [LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("i32"),
            .reason = IREE_SVL(
                "AMDGPU scalar comparison lowering requires i32 operands"),
        },
    [LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I1] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("i1"),
            .reason = IREE_SVL(
                "AMDGPU scalar comparison lowering requires an i1 result"),
        },
    [LOOM_AMDGPU_COMPARE_DIAGNOSTIC_PREDICATE] =
        {
            .subject_kind = IREE_SVL("attr"),
            .subject_name = IREE_SVL("predicate"),
            .reason = IREE_SVL(
                "AMDGPU scalar comparison lowering requires a supported "
                "scalar.cmpi predicate"),
        },
    [LOOM_AMDGPU_COMPARE_DIAGNOSTIC_DESCRIPTOR] =
        {
            .subject_kind = IREE_SVL("descriptor"),
            .subject_name = IREE_SVL("amdgpu.compare"),
            .reason = IREE_SVL("the selected AMDGPU descriptor set does not "
                               "contain the required scalar compare packet"),
        },
    [LOOM_AMDGPU_COMPARE_DIAGNOSTIC_SGPR] =
        {
            .subject_kind = IREE_SVL("register-class"),
            .subject_name = IREE_SVL("sgpr"),
            .reason = IREE_SVL(
                "AMDGPU scalar comparison lowering requires SGPR operands"),
        },
    [LOOM_AMDGPU_COMPARE_DIAGNOSTIC_SCC] =
        {
            .subject_kind = IREE_SVL("register-class"),
            .subject_name = IREE_SVL("scc"),
            .reason = IREE_SVL(
                "AMDGPU scalar comparison lowering requires an SCC result"),
        },
    [LOOM_AMDGPU_COMPARE_DIAGNOSTIC_MASK] =
        {
            .subject_kind = IREE_SVL("register-class"),
            .subject_name = IREE_SVL("sgpr-mask"),
            .reason = IREE_SVL("AMDGPU divergent comparison lowering requires "
                               "an SGPR mask result"),
        },
    [LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I32_VGPR] =
        {
            .subject_kind = IREE_SVL("materializer"),
            .subject_name = IREE_SVL("i32-vgpr"),
            .reason = IREE_SVL("AMDGPU divergent comparison lowering requires "
                               "i32 values that can materialize as VGPR "
                               "operands"),
        },
};

#define LOOM_AMDGPU_COMPARE_GUARD_VALUE(value_ref, type_pattern, diagnostic) \
  {                                                                          \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,                               \
      .value_ref_index = value_ref,                                          \
      .type_pattern_index = type_pattern,                                    \
      .diagnostic_index = diagnostic,                                        \
  }

#define LOOM_AMDGPU_COMPARE_GUARD_ATTR_ENUM(predicate)              \
  {                                                                 \
      .kind = LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ,                    \
      .attr_index = 0,                                              \
      .diagnostic_index = LOOM_AMDGPU_COMPARE_DIAGNOSTIC_PREDICATE, \
      .u64 = predicate,                                             \
  }

#define LOOM_AMDGPU_COMPARE_GUARD_LOW_REG_CLASS(value_ref, reg_class, \
                                                diagnostic)           \
  {                                                                   \
      .kind = LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS,          \
      .value_ref_index = value_ref,                                   \
      .diagnostic_index = diagnostic,                                 \
      .register_class_id = reg_class,                                 \
  }

#define LOOM_AMDGPU_COMPARE_GUARD_MATERIALIZABLE(value_ref, diagnostic) \
  {                                                                     \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_MATERIALIZABLE,                \
      .value_ref_index = value_ref,                                     \
      .diagnostic_index = diagnostic,                                   \
  }

#define LOOM_AMDGPU_COMPARE_GUARD_DESCRIPTOR(descriptor)             \
  {                                                                  \
      .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,             \
      .diagnostic_index = LOOM_AMDGPU_COMPARE_DIAGNOSTIC_DESCRIPTOR, \
      .descriptor_id = descriptor,                                   \
  }

#define LOOM_AMDGPU_COMPARE_GUARDS_SCC(predicate, descriptor)              \
  LOOM_AMDGPU_COMPARE_GUARD_VALUE(LOOM_AMDGPU_COMPARE_OPERAND0,            \
                                  LOOM_AMDGPU_COMPARE_TYPE_SI32,           \
                                  LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I32),     \
      LOOM_AMDGPU_COMPARE_GUARD_VALUE(LOOM_AMDGPU_COMPARE_OPERAND1,        \
                                      LOOM_AMDGPU_COMPARE_TYPE_SI32,       \
                                      LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I32), \
      LOOM_AMDGPU_COMPARE_GUARD_VALUE(LOOM_AMDGPU_COMPARE_RESULT0,         \
                                      LOOM_AMDGPU_COMPARE_TYPE_SI1,        \
                                      LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I1),  \
      LOOM_AMDGPU_COMPARE_GUARD_ATTR_ENUM(predicate),                      \
      LOOM_AMDGPU_COMPARE_GUARD_LOW_REG_CLASS(                             \
          LOOM_AMDGPU_COMPARE_OPERAND0, LOOM_AMDGPU_REG_CLASS_ID_SGPR,     \
          LOOM_AMDGPU_COMPARE_DIAGNOSTIC_SGPR),                            \
      LOOM_AMDGPU_COMPARE_GUARD_LOW_REG_CLASS(                             \
          LOOM_AMDGPU_COMPARE_OPERAND1, LOOM_AMDGPU_REG_CLASS_ID_SGPR,     \
          LOOM_AMDGPU_COMPARE_DIAGNOSTIC_SGPR),                            \
      LOOM_AMDGPU_COMPARE_GUARD_LOW_REG_CLASS(                             \
          LOOM_AMDGPU_COMPARE_RESULT0, LOOM_AMDGPU_REG_CLASS_ID_SCC,       \
          LOOM_AMDGPU_COMPARE_DIAGNOSTIC_SCC),                             \
      LOOM_AMDGPU_COMPARE_GUARD_DESCRIPTOR(descriptor)

#define LOOM_AMDGPU_COMPARE_GUARDS_MASK(predicate, descriptor)             \
  LOOM_AMDGPU_COMPARE_GUARD_VALUE(LOOM_AMDGPU_COMPARE_OPERAND0,            \
                                  LOOM_AMDGPU_COMPARE_TYPE_SI32,           \
                                  LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I32),     \
      LOOM_AMDGPU_COMPARE_GUARD_VALUE(LOOM_AMDGPU_COMPARE_OPERAND1,        \
                                      LOOM_AMDGPU_COMPARE_TYPE_SI32,       \
                                      LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I32), \
      LOOM_AMDGPU_COMPARE_GUARD_VALUE(LOOM_AMDGPU_COMPARE_RESULT0,         \
                                      LOOM_AMDGPU_COMPARE_TYPE_SI1,        \
                                      LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I1),  \
      LOOM_AMDGPU_COMPARE_GUARD_ATTR_ENUM(predicate),                      \
      LOOM_AMDGPU_COMPARE_GUARD_LOW_REG_CLASS(                             \
          LOOM_AMDGPU_COMPARE_RESULT0, LOOM_AMDGPU_REG_CLASS_ID_SGPR,      \
          LOOM_AMDGPU_COMPARE_DIAGNOSTIC_MASK),                            \
      LOOM_AMDGPU_COMPARE_GUARD_MATERIALIZABLE(                            \
          LOOM_AMDGPU_COMPARE_I32_VGPR_OPERAND0,                           \
          LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I32_VGPR),                        \
      LOOM_AMDGPU_COMPARE_GUARD_MATERIALIZABLE(                            \
          LOOM_AMDGPU_COMPARE_I32_VGPR_OPERAND1,                           \
          LOOM_AMDGPU_COMPARE_DIAGNOSTIC_I32_VGPR),                        \
      LOOM_AMDGPU_COMPARE_GUARD_DESCRIPTOR(descriptor)

enum loom_amdgpu_compare_rule_e {
  LOOM_AMDGPU_COMPARE_RULE_EQ = 0,
  LOOM_AMDGPU_COMPARE_RULE_NE = 1,
  LOOM_AMDGPU_COMPARE_RULE_SLT = 2,
  LOOM_AMDGPU_COMPARE_RULE_SLE = 3,
  LOOM_AMDGPU_COMPARE_RULE_SGT = 4,
  LOOM_AMDGPU_COMPARE_RULE_SGE = 5,
  LOOM_AMDGPU_COMPARE_RULE_ULT = 6,
  LOOM_AMDGPU_COMPARE_RULE_ULE = 7,
  LOOM_AMDGPU_COMPARE_RULE_UGT = 8,
  LOOM_AMDGPU_COMPARE_RULE_UGE = 9,
  LOOM_AMDGPU_COMPARE_RULE_MASK_EQ = 10,
  LOOM_AMDGPU_COMPARE_RULE_MASK_NE = 11,
  LOOM_AMDGPU_COMPARE_RULE_MASK_SLT = 12,
  LOOM_AMDGPU_COMPARE_RULE_MASK_SLE = 13,
  LOOM_AMDGPU_COMPARE_RULE_MASK_SGT = 14,
  LOOM_AMDGPU_COMPARE_RULE_MASK_SGE = 15,
  LOOM_AMDGPU_COMPARE_RULE_MASK_ULT = 16,
  LOOM_AMDGPU_COMPARE_RULE_MASK_ULE = 17,
  LOOM_AMDGPU_COMPARE_RULE_MASK_UGT = 18,
  LOOM_AMDGPU_COMPARE_RULE_MASK_UGE = 19,
  LOOM_AMDGPU_COMPARE_RULE_COUNT_ = 20,
};

#define LOOM_AMDGPU_COMPARE_GUARD_COUNT 8
#define LOOM_AMDGPU_COMPARE_GUARD_START(rule) \
  ((rule) * LOOM_AMDGPU_COMPARE_GUARD_COUNT)

static const loom_low_lower_guard_t kAmdgpuCompareGuards[] = {
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_EQ)] =
        LOOM_AMDGPU_COMPARE_GUARDS_SCC(LOOM_SCALAR_CMPI_PREDICATE_EQ,
                                       LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_EQ_I32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_NE)] =
        LOOM_AMDGPU_COMPARE_GUARDS_SCC(LOOM_SCALAR_CMPI_PREDICATE_NE,
                                       LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_LG_I32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_SLT)] =
        LOOM_AMDGPU_COMPARE_GUARDS_SCC(LOOM_SCALAR_CMPI_PREDICATE_SLT,
                                       LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_LT_I32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_SLE)] =
        LOOM_AMDGPU_COMPARE_GUARDS_SCC(LOOM_SCALAR_CMPI_PREDICATE_SLE,
                                       LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_LE_I32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_SGT)] =
        LOOM_AMDGPU_COMPARE_GUARDS_SCC(LOOM_SCALAR_CMPI_PREDICATE_SGT,
                                       LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_GT_I32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_SGE)] =
        LOOM_AMDGPU_COMPARE_GUARDS_SCC(LOOM_SCALAR_CMPI_PREDICATE_SGE,
                                       LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_GE_I32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_ULT)] =
        LOOM_AMDGPU_COMPARE_GUARDS_SCC(LOOM_SCALAR_CMPI_PREDICATE_ULT,
                                       LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_LT_U32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_ULE)] =
        LOOM_AMDGPU_COMPARE_GUARDS_SCC(LOOM_SCALAR_CMPI_PREDICATE_ULE,
                                       LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_LE_U32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_UGT)] =
        LOOM_AMDGPU_COMPARE_GUARDS_SCC(LOOM_SCALAR_CMPI_PREDICATE_UGT,
                                       LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_GT_U32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_UGE)] =
        LOOM_AMDGPU_COMPARE_GUARDS_SCC(LOOM_SCALAR_CMPI_PREDICATE_UGE,
                                       LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_GE_U32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_MASK_EQ)] =
        LOOM_AMDGPU_COMPARE_GUARDS_MASK(LOOM_SCALAR_CMPI_PREDICATE_EQ,
                                        LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_EQ_I32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_MASK_NE)] =
        LOOM_AMDGPU_COMPARE_GUARDS_MASK(LOOM_SCALAR_CMPI_PREDICATE_NE,
                                        LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_NE_I32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_MASK_SLT)] =
        LOOM_AMDGPU_COMPARE_GUARDS_MASK(
            LOOM_SCALAR_CMPI_PREDICATE_SLT,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_SLT_I32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_MASK_SLE)] =
        LOOM_AMDGPU_COMPARE_GUARDS_MASK(
            LOOM_SCALAR_CMPI_PREDICATE_SLE,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_SLE_I32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_MASK_SGT)] =
        LOOM_AMDGPU_COMPARE_GUARDS_MASK(
            LOOM_SCALAR_CMPI_PREDICATE_SGT,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_SGT_I32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_MASK_SGE)] =
        LOOM_AMDGPU_COMPARE_GUARDS_MASK(
            LOOM_SCALAR_CMPI_PREDICATE_SGE,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_SGE_I32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_MASK_ULT)] =
        LOOM_AMDGPU_COMPARE_GUARDS_MASK(
            LOOM_SCALAR_CMPI_PREDICATE_ULT,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_ULT_U32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_MASK_ULE)] =
        LOOM_AMDGPU_COMPARE_GUARDS_MASK(
            LOOM_SCALAR_CMPI_PREDICATE_ULE,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_ULE_U32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_MASK_UGT)] =
        LOOM_AMDGPU_COMPARE_GUARDS_MASK(
            LOOM_SCALAR_CMPI_PREDICATE_UGT,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_UGT_U32),
    [LOOM_AMDGPU_COMPARE_GUARD_START(LOOM_AMDGPU_COMPARE_RULE_MASK_UGE)] =
        LOOM_AMDGPU_COMPARE_GUARDS_MASK(
            LOOM_SCALAR_CMPI_PREDICATE_UGE,
            LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_UGE_U32),
};

static_assert(IREE_ARRAYSIZE(kAmdgpuCompareGuards) ==
                  LOOM_AMDGPU_COMPARE_RULE_COUNT_ *
                      LOOM_AMDGPU_COMPARE_GUARD_COUNT,
              "AMDGPU compare guard rows must align with rule rows");

#undef LOOM_AMDGPU_COMPARE_GUARDS_MASK
#undef LOOM_AMDGPU_COMPARE_GUARDS_SCC
#undef LOOM_AMDGPU_COMPARE_GUARD_DESCRIPTOR
#undef LOOM_AMDGPU_COMPARE_GUARD_MATERIALIZABLE
#undef LOOM_AMDGPU_COMPARE_GUARD_LOW_REG_CLASS
#undef LOOM_AMDGPU_COMPARE_GUARD_ATTR_ENUM
#undef LOOM_AMDGPU_COMPARE_GUARD_VALUE

#define LOOM_AMDGPU_COMPARE_EMIT(descriptor, operand_start) \
  {                                                         \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,            \
      .descriptor_id = descriptor,                          \
      .operand_ref_start = operand_start,                   \
      .operand_ref_count = 2,                               \
      .result_ref_start = LOOM_AMDGPU_COMPARE_RESULT0,      \
      .result_ref_count = 1,                                \
  }

#define LOOM_AMDGPU_COMPARE_SCC_EMIT(descriptor) \
  LOOM_AMDGPU_COMPARE_EMIT(descriptor, LOOM_AMDGPU_COMPARE_OPERAND0)

#define LOOM_AMDGPU_COMPARE_MASK_EMIT(descriptor) \
  LOOM_AMDGPU_COMPARE_EMIT(descriptor, LOOM_AMDGPU_COMPARE_I32_VGPR_OPERAND0)

static const loom_low_lower_emit_t kAmdgpuCompareEmits[] = {
    [LOOM_AMDGPU_COMPARE_RULE_EQ] =
        LOOM_AMDGPU_COMPARE_SCC_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_EQ_I32),
    [LOOM_AMDGPU_COMPARE_RULE_NE] =
        LOOM_AMDGPU_COMPARE_SCC_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_LG_I32),
    [LOOM_AMDGPU_COMPARE_RULE_SLT] =
        LOOM_AMDGPU_COMPARE_SCC_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_LT_I32),
    [LOOM_AMDGPU_COMPARE_RULE_SLE] =
        LOOM_AMDGPU_COMPARE_SCC_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_LE_I32),
    [LOOM_AMDGPU_COMPARE_RULE_SGT] =
        LOOM_AMDGPU_COMPARE_SCC_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_GT_I32),
    [LOOM_AMDGPU_COMPARE_RULE_SGE] =
        LOOM_AMDGPU_COMPARE_SCC_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_GE_I32),
    [LOOM_AMDGPU_COMPARE_RULE_ULT] =
        LOOM_AMDGPU_COMPARE_SCC_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_LT_U32),
    [LOOM_AMDGPU_COMPARE_RULE_ULE] =
        LOOM_AMDGPU_COMPARE_SCC_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_LE_U32),
    [LOOM_AMDGPU_COMPARE_RULE_UGT] =
        LOOM_AMDGPU_COMPARE_SCC_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_GT_U32),
    [LOOM_AMDGPU_COMPARE_RULE_UGE] =
        LOOM_AMDGPU_COMPARE_SCC_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_S_CMP_GE_U32),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_EQ] =
        LOOM_AMDGPU_COMPARE_MASK_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_EQ_I32),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_NE] =
        LOOM_AMDGPU_COMPARE_MASK_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_NE_I32),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_SLT] =
        LOOM_AMDGPU_COMPARE_MASK_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_SLT_I32),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_SLE] =
        LOOM_AMDGPU_COMPARE_MASK_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_SLE_I32),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_SGT] =
        LOOM_AMDGPU_COMPARE_MASK_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_SGT_I32),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_SGE] =
        LOOM_AMDGPU_COMPARE_MASK_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_SGE_I32),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_ULT] =
        LOOM_AMDGPU_COMPARE_MASK_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_ULT_U32),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_ULE] =
        LOOM_AMDGPU_COMPARE_MASK_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_ULE_U32),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_UGT] =
        LOOM_AMDGPU_COMPARE_MASK_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_UGT_U32),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_UGE] =
        LOOM_AMDGPU_COMPARE_MASK_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_CMP_UGE_U32),
};

static_assert(IREE_ARRAYSIZE(kAmdgpuCompareEmits) ==
                  LOOM_AMDGPU_COMPARE_RULE_COUNT_,
              "AMDGPU compare emit rows must align with rule rows");

#undef LOOM_AMDGPU_COMPARE_EMIT
#undef LOOM_AMDGPU_COMPARE_MASK_EMIT
#undef LOOM_AMDGPU_COMPARE_SCC_EMIT

#define LOOM_AMDGPU_COMPARE_RULE(rule)                      \
  {                                                         \
      .source_op_kind = LOOM_OP_SCALAR_CMPI,                \
      .guard_start = LOOM_AMDGPU_COMPARE_GUARD_START(rule), \
      .guard_count = LOOM_AMDGPU_COMPARE_GUARD_COUNT,       \
      .emit_start = rule,                                   \
      .emit_count = 1,                                      \
  }

static const loom_low_lower_rule_t kAmdgpuCompareRules[] = {
    [LOOM_AMDGPU_COMPARE_RULE_EQ] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_EQ),
    [LOOM_AMDGPU_COMPARE_RULE_NE] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_NE),
    [LOOM_AMDGPU_COMPARE_RULE_SLT] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_SLT),
    [LOOM_AMDGPU_COMPARE_RULE_SLE] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_SLE),
    [LOOM_AMDGPU_COMPARE_RULE_SGT] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_SGT),
    [LOOM_AMDGPU_COMPARE_RULE_SGE] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_SGE),
    [LOOM_AMDGPU_COMPARE_RULE_ULT] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_ULT),
    [LOOM_AMDGPU_COMPARE_RULE_ULE] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_ULE),
    [LOOM_AMDGPU_COMPARE_RULE_UGT] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_UGT),
    [LOOM_AMDGPU_COMPARE_RULE_UGE] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_UGE),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_EQ] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_MASK_EQ),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_NE] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_MASK_NE),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_SLT] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_MASK_SLT),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_SLE] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_MASK_SLE),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_SGT] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_MASK_SGT),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_SGE] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_MASK_SGE),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_ULT] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_MASK_ULT),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_ULE] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_MASK_ULE),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_UGT] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_MASK_UGT),
    [LOOM_AMDGPU_COMPARE_RULE_MASK_UGE] =
        LOOM_AMDGPU_COMPARE_RULE(LOOM_AMDGPU_COMPARE_RULE_MASK_UGE),
};

static_assert(IREE_ARRAYSIZE(kAmdgpuCompareRules) ==
                  LOOM_AMDGPU_COMPARE_RULE_COUNT_,
              "AMDGPU compare rule indexes must cover the rule table");

#undef LOOM_AMDGPU_COMPARE_RULE

static const loom_low_lower_rule_span_t kAmdgpuCompareRuleSpans[] = {
    {
        .source_op_kind = LOOM_OP_SCALAR_CMPI,
        .rule_start = LOOM_AMDGPU_COMPARE_RULE_EQ,
        .rule_count = LOOM_AMDGPU_COMPARE_RULE_COUNT_,
    },
};

const loom_low_lower_rule_set_t loom_amdgpu_compare_rule_set = {
    .spans = kAmdgpuCompareRuleSpans,
    .span_count = IREE_ARRAYSIZE(kAmdgpuCompareRuleSpans),
    .rules = kAmdgpuCompareRules,
    .rule_count = IREE_ARRAYSIZE(kAmdgpuCompareRules),
    .type_patterns = kAmdgpuCompareTypePatterns,
    .type_pattern_count = IREE_ARRAYSIZE(kAmdgpuCompareTypePatterns),
    .value_refs = kAmdgpuCompareValueRefs,
    .value_ref_count = IREE_ARRAYSIZE(kAmdgpuCompareValueRefs),
    .guards = kAmdgpuCompareGuards,
    .guard_count = IREE_ARRAYSIZE(kAmdgpuCompareGuards),
    .emits = kAmdgpuCompareEmits,
    .emit_count = IREE_ARRAYSIZE(kAmdgpuCompareEmits),
    .diagnostics = kAmdgpuCompareDiagnostics,
    .diagnostic_count = IREE_ARRAYSIZE(kAmdgpuCompareDiagnostics),
    .materializers = kAmdgpuCompareMaterializers,
    .materializer_count = IREE_ARRAYSIZE(kAmdgpuCompareMaterializers),
};

#undef LOOM_AMDGPU_COMPARE_GUARD_START
#undef LOOM_AMDGPU_COMPARE_GUARD_COUNT
