// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower_rules.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower_internal.h"

enum loom_amdgpu_dot_type_pattern_e {
  LOOM_AMDGPU_DOT_TYPE_VF32 = 0,
  LOOM_AMDGPU_DOT_TYPE_SF32 = 1,
  LOOM_AMDGPU_DOT_TYPE_VI8 = 2,
  LOOM_AMDGPU_DOT_TYPE_VI32_PACKED = 3,
};

static const loom_low_lower_type_pattern_t kAmdgpuDotTypePatterns[] = {
    [LOOM_AMDGPU_DOT_TYPE_VF32] =
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
    [LOOM_AMDGPU_DOT_TYPE_SF32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_F32),
        },
    [LOOM_AMDGPU_DOT_TYPE_VI8] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE,
            .type_kind = LOOM_TYPE_VECTOR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I8),
            .rank = 1,
            .static_dim0_min = 4,
            .static_dim0_max = LOOM_AMDGPU_MAX_PACKED_I8_LANES,
        },
    [LOOM_AMDGPU_DOT_TYPE_VI32_PACKED] =
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
            .static_dim0_max = LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS,
        },
};

enum loom_amdgpu_dot_value_ref_e {
  LOOM_AMDGPU_DOT_LHS = 0,
  LOOM_AMDGPU_DOT_RHS = 1,
  LOOM_AMDGPU_DOT_INIT = 2,
  LOOM_AMDGPU_DOT_RESULT = 3,
};

static const loom_low_lower_value_ref_t kAmdgpuDotValueRefs[] = {
    [LOOM_AMDGPU_DOT_LHS] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
        },
    [LOOM_AMDGPU_DOT_RHS] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
        },
    [LOOM_AMDGPU_DOT_INIT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 2,
        },
    [LOOM_AMDGPU_DOT_RESULT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
            .index = 0,
        },
};

enum loom_amdgpu_dot_diagnostic_e {
  LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS = 0,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS = 1,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC = 2,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT = 3,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC_VGPR = 4,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT_VGPR = 5,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_DESCRIPTOR = 6,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_KIND = 7,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS_I8 = 8,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS_I8 = 9,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS_I32 = 10,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS_I32 = 11,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC_I32 = 12,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT_I32 = 13,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS_VGPR = 14,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS_VGPR = 15,
  LOOM_AMDGPU_DOT_DIAGNOSTIC_PACKED_DESCRIPTOR = 16,
};

static const loom_low_lower_diagnostic_t kAmdgpuDotDiagnostics[] = {
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("lhs"),
            .reason = IREE_SVL("AMDGPU vector.dotf lowering requires a "
                               "static rank-1 f32 lhs vector with 1 to 8 "
                               "lanes"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("rhs"),
            .reason = IREE_SVL("AMDGPU vector.dotf lowering requires a "
                               "static rank-1 f32 rhs vector matching the lhs "
                               "shape"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("accumulator"),
            .reason = IREE_SVL("AMDGPU vector.dotf lowering requires an f32 "
                               "scalar accumulator"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("result"),
            .reason = IREE_SVL("AMDGPU vector.dotf lowering requires an f32 "
                               "scalar result"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC_VGPR] =
        {
            .subject_kind = IREE_SVL("register-class"),
            .subject_name = IREE_SVL("accumulator"),
            .reason = IREE_SVL("AMDGPU vector.dotf lowering requires the "
                               "accumulator to map to a VGPR"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT_VGPR] =
        {
            .subject_kind = IREE_SVL("register-class"),
            .subject_name = IREE_SVL("result"),
            .reason = IREE_SVL("AMDGPU vector.dotf lowering requires the "
                               "result to map to a VGPR"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_DESCRIPTOR] =
        {
            .subject_kind = IREE_SVL("descriptor"),
            .subject_name = IREE_SVL("amdgpu.v_fma_f32"),
            .reason = IREE_SVL("the selected AMDGPU descriptor set does not "
                               "contain the f32 FMA packet"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_KIND] =
        {
            .subject_kind = IREE_SVL("kind"),
            .subject_name = IREE_SVL("vector.dot"),
            .reason = IREE_SVL("AMDGPU packed vector dot lowering requires a "
                               "supported signedness kind"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS_I8] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("lhs"),
            .reason = IREE_SVL("AMDGPU vector.dot4i lowering requires a "
                               "static rank-1 i8 lhs vector with a "
                               "multiple-of-4 lane count from 4 to 16"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS_I8] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("rhs"),
            .reason = IREE_SVL("AMDGPU vector.dot4i lowering requires a "
                               "static rank-1 i8 rhs vector with a "
                               "multiple-of-4 lane count matching the lhs "
                               "shape"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS_I32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("lhs"),
            .reason = IREE_SVL("AMDGPU vector.dot8i4 lowering requires a "
                               "static rank-1 i32 lhs vector with 1 to 4 "
                               "lanes"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS_I32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("rhs"),
            .reason = IREE_SVL("AMDGPU vector.dot8i4 lowering requires a "
                               "static rank-1 i32 rhs vector matching the lhs "
                               "shape"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC_I32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("accumulator"),
            .reason = IREE_SVL("AMDGPU packed vector dot lowering requires a "
                               "static rank-1 i32 accumulator vector matching "
                               "the result group count"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT_I32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("result"),
            .reason = IREE_SVL("AMDGPU packed vector dot lowering requires a "
                               "static rank-1 i32 result vector with 1 to 4 "
                               "lanes"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS_VGPR] =
        {
            .subject_kind = IREE_SVL("register-class"),
            .subject_name = IREE_SVL("lhs"),
            .reason = IREE_SVL("AMDGPU packed vector dot lowering requires "
                               "the lhs to map to VGPR storage"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS_VGPR] =
        {
            .subject_kind = IREE_SVL("register-class"),
            .subject_name = IREE_SVL("rhs"),
            .reason = IREE_SVL("AMDGPU packed vector dot lowering requires "
                               "the rhs to map to VGPR storage"),
        },
    [LOOM_AMDGPU_DOT_DIAGNOSTIC_PACKED_DESCRIPTOR] =
        {
            .subject_kind = IREE_SVL("descriptor"),
            .subject_name = IREE_SVL("amdgpu.packed_dot"),
            .reason = IREE_SVL("the selected AMDGPU descriptor set does not "
                               "contain the required packed dot packet"),
        },
};

#define LOOM_AMDGPU_DOT_GUARD_VALUE(value_ref, type_pattern, diagnostic) \
  {                                                                      \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,                           \
      .value_ref_index = value_ref,                                      \
      .type_pattern_index = type_pattern,                                \
      .diagnostic_index = diagnostic,                                    \
  }

#define LOOM_AMDGPU_DOT_GUARD_KIND(kind_value)             \
  {                                                        \
      .kind = LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ,           \
      .attr_index = 0,                                     \
      .diagnostic_index = LOOM_AMDGPU_DOT_DIAGNOSTIC_KIND, \
      .u64 = kind_value,                                   \
  }

#define LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS(value_ref, diagnostic) \
  {                                                                \
      .kind = LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS,       \
      .value_ref_index = value_ref,                                \
      .diagnostic_index = diagnostic,                              \
      .register_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR,          \
  }

#define LOOM_AMDGPU_DOT_GUARD_DIM0_MULTIPLE(value_ref, multiple, diagnostic) \
  {                                                                          \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_STATIC_DIM0_MULTIPLE,               \
      .value_ref_index = value_ref,                                          \
      .diagnostic_index = diagnostic,                                        \
      .u64 = multiple,                                                       \
  }

#define LOOM_AMDGPU_DOT_GUARD_LOW_UNIT_COUNT_EQ(lhs, rhs, diagnostic) \
  {                                                                   \
      .kind = LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT_EQ,  \
      .value_ref_index = lhs,                                         \
      .other_value_ref_index = rhs,                                   \
      .diagnostic_index = diagnostic,                                 \
  }

#define LOOM_AMDGPU_DOT_GUARD_DESCRIPTOR(descriptor, diagnostic) \
  {                                                              \
      .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,         \
      .diagnostic_index = diagnostic,                            \
      .descriptor_id = descriptor,                               \
  }

enum loom_amdgpu_dot_rule_e {
  LOOM_AMDGPU_DOT_RULE_DOTF = 0,
  LOOM_AMDGPU_DOT_RULE_DOT4I_S8S8 = 1,
  LOOM_AMDGPU_DOT_RULE_DOT4I_U8U8 = 2,
  LOOM_AMDGPU_DOT_RULE_DOT4I_U8S8 = 3,
  LOOM_AMDGPU_DOT_RULE_DOT4I_S8U8 = 4,
  LOOM_AMDGPU_DOT_RULE_DOT8I4_S4S4 = 5,
  LOOM_AMDGPU_DOT_RULE_DOT8I4_S4U4 = 6,
  LOOM_AMDGPU_DOT_RULE_DOT8I4_U4S4 = 7,
  LOOM_AMDGPU_DOT_RULE_DOT8I4_U4U4 = 8,
  LOOM_AMDGPU_DOT_RULE_COUNT_ = 9,
};

enum loom_amdgpu_dot_guard_range_e {
  LOOM_AMDGPU_DOT_GUARDS_DOTF = 0,
  LOOM_AMDGPU_DOT_GUARD_COUNT_DOTF = 8,
  LOOM_AMDGPU_DOT_GUARDS_DOT4I_S8S8 =
      LOOM_AMDGPU_DOT_GUARDS_DOTF + LOOM_AMDGPU_DOT_GUARD_COUNT_DOTF,
  LOOM_AMDGPU_DOT_GUARD_COUNT_DOT4I = 15,
  LOOM_AMDGPU_DOT_GUARDS_DOT4I_U8U8 =
      LOOM_AMDGPU_DOT_GUARDS_DOT4I_S8S8 + LOOM_AMDGPU_DOT_GUARD_COUNT_DOT4I,
  LOOM_AMDGPU_DOT_GUARDS_DOT4I_U8S8 =
      LOOM_AMDGPU_DOT_GUARDS_DOT4I_U8U8 + LOOM_AMDGPU_DOT_GUARD_COUNT_DOT4I,
  LOOM_AMDGPU_DOT_GUARDS_DOT4I_S8U8 =
      LOOM_AMDGPU_DOT_GUARDS_DOT4I_U8S8 + LOOM_AMDGPU_DOT_GUARD_COUNT_DOT4I,
  LOOM_AMDGPU_DOT_GUARDS_DOT8I4_S4S4 =
      LOOM_AMDGPU_DOT_GUARDS_DOT4I_S8U8 + LOOM_AMDGPU_DOT_GUARD_COUNT_DOT4I,
  LOOM_AMDGPU_DOT_GUARD_COUNT_DOT8I4 = 13,
  LOOM_AMDGPU_DOT_GUARDS_DOT8I4_S4U4 =
      LOOM_AMDGPU_DOT_GUARDS_DOT8I4_S4S4 + LOOM_AMDGPU_DOT_GUARD_COUNT_DOT8I4,
  LOOM_AMDGPU_DOT_GUARDS_DOT8I4_U4S4 =
      LOOM_AMDGPU_DOT_GUARDS_DOT8I4_S4U4 + LOOM_AMDGPU_DOT_GUARD_COUNT_DOT8I4,
  LOOM_AMDGPU_DOT_GUARDS_DOT8I4_U4U4 =
      LOOM_AMDGPU_DOT_GUARDS_DOT8I4_U4S4 + LOOM_AMDGPU_DOT_GUARD_COUNT_DOT8I4,
  LOOM_AMDGPU_DOT_GUARD_COUNT_ =
      LOOM_AMDGPU_DOT_GUARDS_DOT8I4_U4U4 + LOOM_AMDGPU_DOT_GUARD_COUNT_DOT8I4,
};

#define LOOM_AMDGPU_DOT4I_GUARDS(kind_value, descriptor)                      \
  LOOM_AMDGPU_DOT_GUARD_KIND(kind_value),                                     \
      LOOM_AMDGPU_DOT_GUARD_VALUE(LOOM_AMDGPU_DOT_LHS,                        \
                                  LOOM_AMDGPU_DOT_TYPE_VI8,                   \
                                  LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS_I8),         \
      LOOM_AMDGPU_DOT_GUARD_DIM0_MULTIPLE(LOOM_AMDGPU_DOT_LHS, 4,             \
                                          LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS_I8), \
      LOOM_AMDGPU_DOT_GUARD_VALUE(LOOM_AMDGPU_DOT_RHS,                        \
                                  LOOM_AMDGPU_DOT_TYPE_VI8,                   \
                                  LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS_I8),         \
      LOOM_AMDGPU_DOT_GUARD_DIM0_MULTIPLE(LOOM_AMDGPU_DOT_RHS, 4,             \
                                          LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS_I8), \
      LOOM_AMDGPU_DOT_GUARD_LOW_UNIT_COUNT_EQ(                                \
          LOOM_AMDGPU_DOT_LHS, LOOM_AMDGPU_DOT_RHS,                           \
          LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS_I8),                                 \
      LOOM_AMDGPU_DOT_GUARD_VALUE(LOOM_AMDGPU_DOT_INIT,                       \
                                  LOOM_AMDGPU_DOT_TYPE_VI32_PACKED,           \
                                  LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC_I32),        \
      LOOM_AMDGPU_DOT_GUARD_LOW_UNIT_COUNT_EQ(                                \
          LOOM_AMDGPU_DOT_LHS, LOOM_AMDGPU_DOT_INIT,                          \
          LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC_I32),                                \
      LOOM_AMDGPU_DOT_GUARD_VALUE(LOOM_AMDGPU_DOT_RESULT,                     \
                                  LOOM_AMDGPU_DOT_TYPE_VI32_PACKED,           \
                                  LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT_I32),     \
      LOOM_AMDGPU_DOT_GUARD_LOW_UNIT_COUNT_EQ(                                \
          LOOM_AMDGPU_DOT_LHS, LOOM_AMDGPU_DOT_RESULT,                        \
          LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT_I32),                             \
      LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS(                                    \
          LOOM_AMDGPU_DOT_LHS, LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS_VGPR),          \
      LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS(                                    \
          LOOM_AMDGPU_DOT_RHS, LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS_VGPR),          \
      LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS(                                    \
          LOOM_AMDGPU_DOT_INIT, LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC_VGPR),         \
      LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS(                                    \
          LOOM_AMDGPU_DOT_RESULT, LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT_VGPR),    \
      LOOM_AMDGPU_DOT_GUARD_DESCRIPTOR(                                       \
          descriptor, LOOM_AMDGPU_DOT_DIAGNOSTIC_PACKED_DESCRIPTOR)

#define LOOM_AMDGPU_DOT8I4_GUARDS(kind_value, descriptor)                  \
  LOOM_AMDGPU_DOT_GUARD_KIND(kind_value),                                  \
      LOOM_AMDGPU_DOT_GUARD_VALUE(LOOM_AMDGPU_DOT_LHS,                     \
                                  LOOM_AMDGPU_DOT_TYPE_VI32_PACKED,        \
                                  LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS_I32),     \
      LOOM_AMDGPU_DOT_GUARD_VALUE(LOOM_AMDGPU_DOT_RHS,                     \
                                  LOOM_AMDGPU_DOT_TYPE_VI32_PACKED,        \
                                  LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS_I32),     \
      LOOM_AMDGPU_DOT_GUARD_LOW_UNIT_COUNT_EQ(                             \
          LOOM_AMDGPU_DOT_LHS, LOOM_AMDGPU_DOT_RHS,                        \
          LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS_I32),                             \
      LOOM_AMDGPU_DOT_GUARD_VALUE(LOOM_AMDGPU_DOT_INIT,                    \
                                  LOOM_AMDGPU_DOT_TYPE_VI32_PACKED,        \
                                  LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC_I32),     \
      LOOM_AMDGPU_DOT_GUARD_LOW_UNIT_COUNT_EQ(                             \
          LOOM_AMDGPU_DOT_LHS, LOOM_AMDGPU_DOT_INIT,                       \
          LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC_I32),                             \
      LOOM_AMDGPU_DOT_GUARD_VALUE(LOOM_AMDGPU_DOT_RESULT,                  \
                                  LOOM_AMDGPU_DOT_TYPE_VI32_PACKED,        \
                                  LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT_I32),  \
      LOOM_AMDGPU_DOT_GUARD_LOW_UNIT_COUNT_EQ(                             \
          LOOM_AMDGPU_DOT_LHS, LOOM_AMDGPU_DOT_RESULT,                     \
          LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT_I32),                          \
      LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS(                                 \
          LOOM_AMDGPU_DOT_LHS, LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS_VGPR),       \
      LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS(                                 \
          LOOM_AMDGPU_DOT_RHS, LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS_VGPR),       \
      LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS(                                 \
          LOOM_AMDGPU_DOT_INIT, LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC_VGPR),      \
      LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS(                                 \
          LOOM_AMDGPU_DOT_RESULT, LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT_VGPR), \
      LOOM_AMDGPU_DOT_GUARD_DESCRIPTOR(                                    \
          descriptor, LOOM_AMDGPU_DOT_DIAGNOSTIC_PACKED_DESCRIPTOR)

static const loom_low_lower_guard_t kAmdgpuDotGuards[] = {
    [LOOM_AMDGPU_DOT_GUARDS_DOTF] = LOOM_AMDGPU_DOT_GUARD_VALUE(
        LOOM_AMDGPU_DOT_LHS, LOOM_AMDGPU_DOT_TYPE_VF32,
        LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS),
    LOOM_AMDGPU_DOT_GUARD_VALUE(LOOM_AMDGPU_DOT_RHS, LOOM_AMDGPU_DOT_TYPE_VF32,
                                LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS),
    LOOM_AMDGPU_DOT_GUARD_LOW_UNIT_COUNT_EQ(LOOM_AMDGPU_DOT_LHS,
                                            LOOM_AMDGPU_DOT_RHS,
                                            LOOM_AMDGPU_DOT_DIAGNOSTIC_RHS),
    LOOM_AMDGPU_DOT_GUARD_VALUE(LOOM_AMDGPU_DOT_INIT, LOOM_AMDGPU_DOT_TYPE_SF32,
                                LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC),
    LOOM_AMDGPU_DOT_GUARD_VALUE(LOOM_AMDGPU_DOT_RESULT,
                                LOOM_AMDGPU_DOT_TYPE_SF32,
                                LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT),
    LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS(LOOM_AMDGPU_DOT_INIT,
                                        LOOM_AMDGPU_DOT_DIAGNOSTIC_ACC_VGPR),
    LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS(LOOM_AMDGPU_DOT_RESULT,
                                        LOOM_AMDGPU_DOT_DIAGNOSTIC_RESULT_VGPR),
    LOOM_AMDGPU_DOT_GUARD_DESCRIPTOR(LOOM_AMDGPU_DESCRIPTOR_ID_V_FMA_F32,
                                     LOOM_AMDGPU_DOT_DIAGNOSTIC_DESCRIPTOR),
    [LOOM_AMDGPU_DOT_GUARDS_DOT4I_S8S8] = LOOM_AMDGPU_DOT4I_GUARDS(
        LOOM_VECTOR_DOT4I_KIND_S8S8, LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT4_I32_I8),
    [LOOM_AMDGPU_DOT_GUARDS_DOT4I_U8U8] = LOOM_AMDGPU_DOT4I_GUARDS(
        LOOM_VECTOR_DOT4I_KIND_U8U8, LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT4_U32_U8),
    [LOOM_AMDGPU_DOT_GUARDS_DOT4I_U8S8] =
        LOOM_AMDGPU_DOT4I_GUARDS(LOOM_VECTOR_DOT4I_KIND_U8S8,
                                 LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT4_I32_IU8_U8S8),
    [LOOM_AMDGPU_DOT_GUARDS_DOT4I_S8U8] =
        LOOM_AMDGPU_DOT4I_GUARDS(LOOM_VECTOR_DOT4I_KIND_S8U8,
                                 LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT4_I32_IU8_S8U8),
    [LOOM_AMDGPU_DOT_GUARDS_DOT8I4_S4S4] = LOOM_AMDGPU_DOT8I4_GUARDS(
        LOOM_VECTOR_DOT8I4_KIND_S4S4, LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT8_I32_I4),
    [LOOM_AMDGPU_DOT_GUARDS_DOT8I4_S4U4] = LOOM_AMDGPU_DOT8I4_GUARDS(
        LOOM_VECTOR_DOT8I4_KIND_S4U4,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT8_I32_IU4_S4U4),
    [LOOM_AMDGPU_DOT_GUARDS_DOT8I4_U4S4] = LOOM_AMDGPU_DOT8I4_GUARDS(
        LOOM_VECTOR_DOT8I4_KIND_U4S4,
        LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT8_I32_IU4_U4S4),
    [LOOM_AMDGPU_DOT_GUARDS_DOT8I4_U4U4] = LOOM_AMDGPU_DOT8I4_GUARDS(
        LOOM_VECTOR_DOT8I4_KIND_U4U4, LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT8_U32_U4),
};

static_assert(IREE_ARRAYSIZE(kAmdgpuDotGuards) == LOOM_AMDGPU_DOT_GUARD_COUNT_,
              "AMDGPU dot guard rows must cover the guard ranges");

#undef LOOM_AMDGPU_DOT_GUARD_DESCRIPTOR
#undef LOOM_AMDGPU_DOT_GUARD_LOW_UNIT_COUNT_EQ
#undef LOOM_AMDGPU_DOT_GUARD_DIM0_MULTIPLE
#undef LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS
#undef LOOM_AMDGPU_DOT_GUARD_KIND
#undef LOOM_AMDGPU_DOT_GUARD_VALUE
#undef LOOM_AMDGPU_DOT8I4_GUARDS
#undef LOOM_AMDGPU_DOT4I_GUARDS

#define LOOM_AMDGPU_PACKED_DOT_EMIT(descriptor)           \
  {                                                       \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE, \
      .descriptor_id = descriptor,                        \
      .operand_ref_start = LOOM_AMDGPU_DOT_LHS,           \
      .operand_ref_count = 3,                             \
      .result_ref_start = LOOM_AMDGPU_DOT_RESULT,         \
      .result_ref_count = 1,                              \
  }

static const loom_low_lower_emit_t kAmdgpuDotEmits[] = {
    [LOOM_AMDGPU_DOT_RULE_DOTF] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES,
            .descriptor_id = LOOM_AMDGPU_DESCRIPTOR_ID_V_FMA_F32,
            .operand_ref_start = LOOM_AMDGPU_DOT_LHS,
            .operand_ref_count = 3,
            .accumulator_operand_index = 2,
            .result_ref_start = LOOM_AMDGPU_DOT_RESULT,
            .result_ref_count = 1,
        },
    [LOOM_AMDGPU_DOT_RULE_DOT4I_S8S8] =
        LOOM_AMDGPU_PACKED_DOT_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT4_I32_I8),
    [LOOM_AMDGPU_DOT_RULE_DOT4I_U8U8] =
        LOOM_AMDGPU_PACKED_DOT_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT4_U32_U8),
    [LOOM_AMDGPU_DOT_RULE_DOT4I_U8S8] = LOOM_AMDGPU_PACKED_DOT_EMIT(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT4_I32_IU8_U8S8),
    [LOOM_AMDGPU_DOT_RULE_DOT4I_S8U8] = LOOM_AMDGPU_PACKED_DOT_EMIT(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT4_I32_IU8_S8U8),
    [LOOM_AMDGPU_DOT_RULE_DOT8I4_S4S4] =
        LOOM_AMDGPU_PACKED_DOT_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT8_I32_I4),
    [LOOM_AMDGPU_DOT_RULE_DOT8I4_S4U4] = LOOM_AMDGPU_PACKED_DOT_EMIT(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT8_I32_IU4_S4U4),
    [LOOM_AMDGPU_DOT_RULE_DOT8I4_U4S4] = LOOM_AMDGPU_PACKED_DOT_EMIT(
        LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT8_I32_IU4_U4S4),
    [LOOM_AMDGPU_DOT_RULE_DOT8I4_U4U4] =
        LOOM_AMDGPU_PACKED_DOT_EMIT(LOOM_AMDGPU_DESCRIPTOR_ID_V_DOT8_U32_U4),
};

static_assert(IREE_ARRAYSIZE(kAmdgpuDotEmits) == LOOM_AMDGPU_DOT_RULE_COUNT_,
              "AMDGPU dot emit rows must align with rule rows");

#undef LOOM_AMDGPU_PACKED_DOT_EMIT

#define LOOM_AMDGPU_DOT_RULE(source_kind, guard_start_value, \
                             guard_count_value, emit_index)  \
  {                                                          \
      .source_op_kind = source_kind,                         \
      .guard_start = guard_start_value,                      \
      .guard_count = guard_count_value,                      \
      .emit_start = emit_index,                              \
      .emit_count = 1,                                       \
  }

static const loom_low_lower_rule_t kAmdgpuDotRules[] = {
    [LOOM_AMDGPU_DOT_RULE_DOTF] = LOOM_AMDGPU_DOT_RULE(
        LOOM_OP_VECTOR_DOTF, LOOM_AMDGPU_DOT_GUARDS_DOTF,
        LOOM_AMDGPU_DOT_GUARD_COUNT_DOTF, LOOM_AMDGPU_DOT_RULE_DOTF),
    [LOOM_AMDGPU_DOT_RULE_DOT4I_S8S8] = LOOM_AMDGPU_DOT_RULE(
        LOOM_OP_VECTOR_DOT4I, LOOM_AMDGPU_DOT_GUARDS_DOT4I_S8S8,
        LOOM_AMDGPU_DOT_GUARD_COUNT_DOT4I, LOOM_AMDGPU_DOT_RULE_DOT4I_S8S8),
    [LOOM_AMDGPU_DOT_RULE_DOT4I_U8U8] = LOOM_AMDGPU_DOT_RULE(
        LOOM_OP_VECTOR_DOT4I, LOOM_AMDGPU_DOT_GUARDS_DOT4I_U8U8,
        LOOM_AMDGPU_DOT_GUARD_COUNT_DOT4I, LOOM_AMDGPU_DOT_RULE_DOT4I_U8U8),
    [LOOM_AMDGPU_DOT_RULE_DOT4I_U8S8] = LOOM_AMDGPU_DOT_RULE(
        LOOM_OP_VECTOR_DOT4I, LOOM_AMDGPU_DOT_GUARDS_DOT4I_U8S8,
        LOOM_AMDGPU_DOT_GUARD_COUNT_DOT4I, LOOM_AMDGPU_DOT_RULE_DOT4I_U8S8),
    [LOOM_AMDGPU_DOT_RULE_DOT4I_S8U8] = LOOM_AMDGPU_DOT_RULE(
        LOOM_OP_VECTOR_DOT4I, LOOM_AMDGPU_DOT_GUARDS_DOT4I_S8U8,
        LOOM_AMDGPU_DOT_GUARD_COUNT_DOT4I, LOOM_AMDGPU_DOT_RULE_DOT4I_S8U8),
    [LOOM_AMDGPU_DOT_RULE_DOT8I4_S4S4] = LOOM_AMDGPU_DOT_RULE(
        LOOM_OP_VECTOR_DOT8I4, LOOM_AMDGPU_DOT_GUARDS_DOT8I4_S4S4,
        LOOM_AMDGPU_DOT_GUARD_COUNT_DOT8I4, LOOM_AMDGPU_DOT_RULE_DOT8I4_S4S4),
    [LOOM_AMDGPU_DOT_RULE_DOT8I4_S4U4] = LOOM_AMDGPU_DOT_RULE(
        LOOM_OP_VECTOR_DOT8I4, LOOM_AMDGPU_DOT_GUARDS_DOT8I4_S4U4,
        LOOM_AMDGPU_DOT_GUARD_COUNT_DOT8I4, LOOM_AMDGPU_DOT_RULE_DOT8I4_S4U4),
    [LOOM_AMDGPU_DOT_RULE_DOT8I4_U4S4] = LOOM_AMDGPU_DOT_RULE(
        LOOM_OP_VECTOR_DOT8I4, LOOM_AMDGPU_DOT_GUARDS_DOT8I4_U4S4,
        LOOM_AMDGPU_DOT_GUARD_COUNT_DOT8I4, LOOM_AMDGPU_DOT_RULE_DOT8I4_U4S4),
    [LOOM_AMDGPU_DOT_RULE_DOT8I4_U4U4] = LOOM_AMDGPU_DOT_RULE(
        LOOM_OP_VECTOR_DOT8I4, LOOM_AMDGPU_DOT_GUARDS_DOT8I4_U4U4,
        LOOM_AMDGPU_DOT_GUARD_COUNT_DOT8I4, LOOM_AMDGPU_DOT_RULE_DOT8I4_U4U4),
};

static_assert(IREE_ARRAYSIZE(kAmdgpuDotRules) == LOOM_AMDGPU_DOT_RULE_COUNT_,
              "AMDGPU dot rule indexes must cover the rule table");

#undef LOOM_AMDGPU_DOT_RULE

static const loom_low_lower_rule_span_t kAmdgpuDotRuleSpans[] = {
    {
        .source_op_kind = LOOM_OP_VECTOR_DOTF,
        .rule_start = LOOM_AMDGPU_DOT_RULE_DOTF,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_DOT4I,
        .rule_start = LOOM_AMDGPU_DOT_RULE_DOT4I_S8S8,
        .rule_count = 4,
    },
    {
        .source_op_kind = LOOM_OP_VECTOR_DOT8I4,
        .rule_start = LOOM_AMDGPU_DOT_RULE_DOT8I4_S4S4,
        .rule_count = 4,
    },
};

const loom_low_lower_rule_set_t loom_amdgpu_dot_rule_set = {
    .spans = kAmdgpuDotRuleSpans,
    .span_count = IREE_ARRAYSIZE(kAmdgpuDotRuleSpans),
    .rules = kAmdgpuDotRules,
    .rule_count = IREE_ARRAYSIZE(kAmdgpuDotRules),
    .type_patterns = kAmdgpuDotTypePatterns,
    .type_pattern_count = IREE_ARRAYSIZE(kAmdgpuDotTypePatterns),
    .value_refs = kAmdgpuDotValueRefs,
    .value_ref_count = IREE_ARRAYSIZE(kAmdgpuDotValueRefs),
    .guards = kAmdgpuDotGuards,
    .guard_count = IREE_ARRAYSIZE(kAmdgpuDotGuards),
    .emits = kAmdgpuDotEmits,
    .emit_count = IREE_ARRAYSIZE(kAmdgpuDotEmits),
    .diagnostics = kAmdgpuDotDiagnostics,
    .diagnostic_count = IREE_ARRAYSIZE(kAmdgpuDotDiagnostics),
};
