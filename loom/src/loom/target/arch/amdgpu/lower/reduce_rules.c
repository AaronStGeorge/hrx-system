// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower_rules.h"
#include "loom/ops/combining.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/descriptor_ids.h"
#include "loom/target/arch/amdgpu/lower/internal.h"

enum loom_amdgpu_reduce_type_pattern_e {
  LOOM_AMDGPU_REDUCE_TYPE_VI32 = 0,
  LOOM_AMDGPU_REDUCE_TYPE_VF32 = 1,
  LOOM_AMDGPU_REDUCE_TYPE_SI32 = 2,
  LOOM_AMDGPU_REDUCE_TYPE_SF32 = 3,
};

#define LOOM_AMDGPU_REDUCE_VECTOR_PATTERN(element_type)                  \
  {                                                                      \
      .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |                   \
               LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT |                \
               LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK |                   \
               LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE,       \
      .type_kind = LOOM_TYPE_VECTOR,                                     \
      .element_type_mask = LOOM_LOW_LOWER_SCALAR_TYPE_BIT(element_type), \
      .rank = 1,                                                         \
      .static_dim0_min = 1,                                              \
      .static_dim0_max = LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES,         \
  }

#define LOOM_AMDGPU_REDUCE_SCALAR_PATTERN(element_type)                  \
  {                                                                      \
      .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |                   \
               LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,                 \
      .type_kind = LOOM_TYPE_SCALAR,                                     \
      .element_type_mask = LOOM_LOW_LOWER_SCALAR_TYPE_BIT(element_type), \
  }

static const loom_low_lower_type_pattern_t kAmdgpuReduceTypePatterns[] = {
    [LOOM_AMDGPU_REDUCE_TYPE_VI32] =
        LOOM_AMDGPU_REDUCE_VECTOR_PATTERN(LOOM_SCALAR_TYPE_I32),
    [LOOM_AMDGPU_REDUCE_TYPE_VF32] =
        LOOM_AMDGPU_REDUCE_VECTOR_PATTERN(LOOM_SCALAR_TYPE_F32),
    [LOOM_AMDGPU_REDUCE_TYPE_SI32] =
        LOOM_AMDGPU_REDUCE_SCALAR_PATTERN(LOOM_SCALAR_TYPE_I32),
    [LOOM_AMDGPU_REDUCE_TYPE_SF32] =
        LOOM_AMDGPU_REDUCE_SCALAR_PATTERN(LOOM_SCALAR_TYPE_F32),
};

#undef LOOM_AMDGPU_REDUCE_SCALAR_PATTERN
#undef LOOM_AMDGPU_REDUCE_VECTOR_PATTERN

enum loom_amdgpu_reduce_value_ref_e {
  LOOM_AMDGPU_REDUCE_I32_VGPR_INIT = 0,
  LOOM_AMDGPU_REDUCE_I32_INPUT = 1,
  LOOM_AMDGPU_REDUCE_F32_INIT = 2,
  LOOM_AMDGPU_REDUCE_F32_INPUT = 3,
  LOOM_AMDGPU_REDUCE_RESULT = 4,
};

enum loom_amdgpu_reduce_materializer_e {
  LOOM_AMDGPU_REDUCE_MATERIALIZER_I32_VGPR = 1,
};

static bool loom_amdgpu_reduce_can_materialize_i32_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id) {
  (void)source_op;
  return loom_amdgpu_value_can_materialize_as_vgpr_i32(context,
                                                       source_value_id);
}

static iree_status_t loom_amdgpu_reduce_materialize_i32_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_value_id_t* out_low_value_id) {
  return loom_amdgpu_lookup_or_materialize_vgpr_i32(
      context, source_op, source_value_id, out_low_value_id);
}

static const loom_low_lower_value_materializer_t kAmdgpuReduceMaterializers[] =
    {
        {
            .can_materialize = loom_amdgpu_reduce_can_materialize_i32_vgpr,
            .materialize = loom_amdgpu_reduce_materialize_i32_vgpr,
        },
};

static const loom_low_lower_value_ref_t kAmdgpuReduceValueRefs[] = {
    [LOOM_AMDGPU_REDUCE_I32_VGPR_INIT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
            .materializer_index = LOOM_AMDGPU_REDUCE_MATERIALIZER_I32_VGPR,
        },
    [LOOM_AMDGPU_REDUCE_I32_INPUT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
        },
    [LOOM_AMDGPU_REDUCE_F32_INIT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
        },
    [LOOM_AMDGPU_REDUCE_F32_INPUT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
        },
    [LOOM_AMDGPU_REDUCE_RESULT] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
            .index = 0,
        },
};

enum loom_amdgpu_reduce_diagnostic_e {
  LOOM_AMDGPU_REDUCE_DIAGNOSTIC_KIND = 0,
  LOOM_AMDGPU_REDUCE_DIAGNOSTIC_VI32 = 1,
  LOOM_AMDGPU_REDUCE_DIAGNOSTIC_VF32 = 2,
  LOOM_AMDGPU_REDUCE_DIAGNOSTIC_SI32 = 3,
  LOOM_AMDGPU_REDUCE_DIAGNOSTIC_SF32 = 4,
  LOOM_AMDGPU_REDUCE_DIAGNOSTIC_RESULT_VGPR = 5,
  LOOM_AMDGPU_REDUCE_DIAGNOSTIC_I32_VGPR = 6,
  LOOM_AMDGPU_REDUCE_DIAGNOSTIC_DESCRIPTOR = 7,
};

static const loom_low_lower_diagnostic_t kAmdgpuReduceDiagnostics[] = {
    [LOOM_AMDGPU_REDUCE_DIAGNOSTIC_KIND] =
        {
            .subject_kind = IREE_SVL("kind"),
            .subject_name = IREE_SVL("vector.reduce"),
            .reason = IREE_SVL("AMDGPU vector.reduce lowering requires a "
                               "supported reduction kind"),
        },
    [LOOM_AMDGPU_REDUCE_DIAGNOSTIC_VI32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("vector<i32>"),
            .reason = IREE_SVL("AMDGPU vector.reduce integer lowering "
                               "requires a static rank-1 i32 vector"),
        },
    [LOOM_AMDGPU_REDUCE_DIAGNOSTIC_VF32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("vector<f32>"),
            .reason = IREE_SVL("AMDGPU vector.reduce floating-point lowering "
                               "requires a static rank-1 f32 vector"),
        },
    [LOOM_AMDGPU_REDUCE_DIAGNOSTIC_SI32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("i32"),
            .reason = IREE_SVL("AMDGPU vector.reduce integer lowering "
                               "requires i32 init and result values"),
        },
    [LOOM_AMDGPU_REDUCE_DIAGNOSTIC_SF32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("f32"),
            .reason = IREE_SVL("AMDGPU vector.reduce floating-point lowering "
                               "requires f32 init and result values"),
        },
    [LOOM_AMDGPU_REDUCE_DIAGNOSTIC_RESULT_VGPR] =
        {
            .subject_kind = IREE_SVL("register-class"),
            .subject_name = IREE_SVL("vgpr"),
            .reason = IREE_SVL("AMDGPU vector.reduce lowering requires a VGPR "
                               "result"),
        },
    [LOOM_AMDGPU_REDUCE_DIAGNOSTIC_I32_VGPR] =
        {
            .subject_kind = IREE_SVL("materializer"),
            .subject_name = IREE_SVL("i32-vgpr"),
            .reason = IREE_SVL("AMDGPU vector.reduce integer lowering "
                               "requires an init value materializable as a "
                               "VGPR accumulator"),
        },
    [LOOM_AMDGPU_REDUCE_DIAGNOSTIC_DESCRIPTOR] =
        {
            .subject_kind = IREE_SVL("descriptor"),
            .subject_name = IREE_SVL("amdgpu.reduce"),
            .reason = IREE_SVL("the selected AMDGPU descriptor set does not "
                               "contain the required reduce packet"),
        },
};

#define LOOM_AMDGPU_REDUCE_GUARD_KIND(reduce_kind)            \
  {                                                           \
      .kind = LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ,              \
      .attr_index = 0,                                        \
      .diagnostic_index = LOOM_AMDGPU_REDUCE_DIAGNOSTIC_KIND, \
      .u64 = reduce_kind,                                     \
  }

#define LOOM_AMDGPU_REDUCE_GUARD_VALUE(value_ref, type_pattern, diagnostic) \
  {                                                                         \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,                              \
      .value_ref_index = value_ref,                                         \
      .type_pattern_index = type_pattern,                                   \
      .diagnostic_index = diagnostic,                                       \
  }

#define LOOM_AMDGPU_REDUCE_GUARD_LOW_REG_CLASS(value_ref, reg_class, \
                                               diagnostic)           \
  {                                                                  \
      .kind = LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS,         \
      .value_ref_index = value_ref,                                  \
      .diagnostic_index = diagnostic,                                \
      .register_class_id = reg_class,                                \
  }

#define LOOM_AMDGPU_REDUCE_GUARD_MATERIALIZABLE(value_ref, diagnostic) \
  {                                                                    \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_MATERIALIZABLE,               \
      .value_ref_index = value_ref,                                    \
      .diagnostic_index = diagnostic,                                  \
  }

#define LOOM_AMDGPU_REDUCE_GUARD_DESCRIPTOR(descriptor)             \
  {                                                                 \
      .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,            \
      .diagnostic_index = LOOM_AMDGPU_REDUCE_DIAGNOSTIC_DESCRIPTOR, \
      .descriptor_id = descriptor,                                  \
  }

#define LOOM_AMDGPU_REDUCE_GUARDS_I32(reduce_kind, descriptor)            \
  LOOM_AMDGPU_REDUCE_GUARD_KIND(reduce_kind),                             \
      LOOM_AMDGPU_REDUCE_GUARD_VALUE(LOOM_AMDGPU_REDUCE_I32_INPUT,        \
                                     LOOM_AMDGPU_REDUCE_TYPE_VI32,        \
                                     LOOM_AMDGPU_REDUCE_DIAGNOSTIC_VI32), \
      LOOM_AMDGPU_REDUCE_GUARD_VALUE(LOOM_AMDGPU_REDUCE_I32_VGPR_INIT,    \
                                     LOOM_AMDGPU_REDUCE_TYPE_SI32,        \
                                     LOOM_AMDGPU_REDUCE_DIAGNOSTIC_SI32), \
      LOOM_AMDGPU_REDUCE_GUARD_VALUE(LOOM_AMDGPU_REDUCE_RESULT,           \
                                     LOOM_AMDGPU_REDUCE_TYPE_SI32,        \
                                     LOOM_AMDGPU_REDUCE_DIAGNOSTIC_SI32), \
      LOOM_AMDGPU_REDUCE_GUARD_LOW_REG_CLASS(                             \
          LOOM_AMDGPU_REDUCE_RESULT, LOOM_AMDGPU_REG_CLASS_ID_VGPR,       \
          LOOM_AMDGPU_REDUCE_DIAGNOSTIC_RESULT_VGPR),                     \
      LOOM_AMDGPU_REDUCE_GUARD_MATERIALIZABLE(                            \
          LOOM_AMDGPU_REDUCE_I32_VGPR_INIT,                               \
          LOOM_AMDGPU_REDUCE_DIAGNOSTIC_I32_VGPR),                        \
      LOOM_AMDGPU_REDUCE_GUARD_DESCRIPTOR(descriptor)

#define LOOM_AMDGPU_REDUCE_GUARDS_F32(reduce_kind, descriptor)            \
  LOOM_AMDGPU_REDUCE_GUARD_KIND(reduce_kind),                             \
      LOOM_AMDGPU_REDUCE_GUARD_VALUE(LOOM_AMDGPU_REDUCE_F32_INPUT,        \
                                     LOOM_AMDGPU_REDUCE_TYPE_VF32,        \
                                     LOOM_AMDGPU_REDUCE_DIAGNOSTIC_VF32), \
      LOOM_AMDGPU_REDUCE_GUARD_VALUE(LOOM_AMDGPU_REDUCE_F32_INIT,         \
                                     LOOM_AMDGPU_REDUCE_TYPE_SF32,        \
                                     LOOM_AMDGPU_REDUCE_DIAGNOSTIC_SF32), \
      LOOM_AMDGPU_REDUCE_GUARD_VALUE(LOOM_AMDGPU_REDUCE_RESULT,           \
                                     LOOM_AMDGPU_REDUCE_TYPE_SF32,        \
                                     LOOM_AMDGPU_REDUCE_DIAGNOSTIC_SF32), \
      LOOM_AMDGPU_REDUCE_GUARD_LOW_REG_CLASS(                             \
          LOOM_AMDGPU_REDUCE_RESULT, LOOM_AMDGPU_REG_CLASS_ID_VGPR,       \
          LOOM_AMDGPU_REDUCE_DIAGNOSTIC_RESULT_VGPR),                     \
      LOOM_AMDGPU_REDUCE_GUARD_DESCRIPTOR(descriptor)

enum loom_amdgpu_reduce_rule_e {
  LOOM_AMDGPU_REDUCE_RULE_ADDI = 0,
  LOOM_AMDGPU_REDUCE_RULE_MULI = 1,
  LOOM_AMDGPU_REDUCE_RULE_ANDI = 2,
  LOOM_AMDGPU_REDUCE_RULE_ORI = 3,
  LOOM_AMDGPU_REDUCE_RULE_XORI = 4,
  LOOM_AMDGPU_REDUCE_RULE_ADDF = 5,
  LOOM_AMDGPU_REDUCE_RULE_MULF = 6,
  LOOM_AMDGPU_REDUCE_RULE_MINNUMF = 7,
  LOOM_AMDGPU_REDUCE_RULE_MAXNUMF = 8,
  LOOM_AMDGPU_REDUCE_RULE_COUNT_ = 9,
};

enum loom_amdgpu_reduce_guard_range_e {
  LOOM_AMDGPU_REDUCE_GUARDS_ADDI = 0,
  LOOM_AMDGPU_REDUCE_GUARD_COUNT_I32 = 7,
  LOOM_AMDGPU_REDUCE_GUARDS_MULI =
      LOOM_AMDGPU_REDUCE_GUARDS_ADDI + LOOM_AMDGPU_REDUCE_GUARD_COUNT_I32,
  LOOM_AMDGPU_REDUCE_GUARDS_ANDI =
      LOOM_AMDGPU_REDUCE_GUARDS_MULI + LOOM_AMDGPU_REDUCE_GUARD_COUNT_I32,
  LOOM_AMDGPU_REDUCE_GUARDS_ORI =
      LOOM_AMDGPU_REDUCE_GUARDS_ANDI + LOOM_AMDGPU_REDUCE_GUARD_COUNT_I32,
  LOOM_AMDGPU_REDUCE_GUARDS_XORI =
      LOOM_AMDGPU_REDUCE_GUARDS_ORI + LOOM_AMDGPU_REDUCE_GUARD_COUNT_I32,
  LOOM_AMDGPU_REDUCE_GUARDS_ADDF =
      LOOM_AMDGPU_REDUCE_GUARDS_XORI + LOOM_AMDGPU_REDUCE_GUARD_COUNT_I32,
  LOOM_AMDGPU_REDUCE_GUARD_COUNT_F32 = 6,
  LOOM_AMDGPU_REDUCE_GUARDS_MULF =
      LOOM_AMDGPU_REDUCE_GUARDS_ADDF + LOOM_AMDGPU_REDUCE_GUARD_COUNT_F32,
  LOOM_AMDGPU_REDUCE_GUARDS_MINNUMF =
      LOOM_AMDGPU_REDUCE_GUARDS_MULF + LOOM_AMDGPU_REDUCE_GUARD_COUNT_F32,
  LOOM_AMDGPU_REDUCE_GUARDS_MAXNUMF =
      LOOM_AMDGPU_REDUCE_GUARDS_MINNUMF + LOOM_AMDGPU_REDUCE_GUARD_COUNT_F32,
  LOOM_AMDGPU_REDUCE_GUARD_COUNT_ =
      LOOM_AMDGPU_REDUCE_GUARDS_MAXNUMF + LOOM_AMDGPU_REDUCE_GUARD_COUNT_F32,
};

static const loom_low_lower_guard_t kAmdgpuReduceGuards[] = {
    [LOOM_AMDGPU_REDUCE_GUARDS_ADDI] = LOOM_AMDGPU_REDUCE_GUARDS_I32(
        LOOM_COMBINING_KIND_ADDI, LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32),
    [LOOM_AMDGPU_REDUCE_GUARDS_MULI] = LOOM_AMDGPU_REDUCE_GUARDS_I32(
        LOOM_COMBINING_KIND_MULI, LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),
    [LOOM_AMDGPU_REDUCE_GUARDS_ANDI] = LOOM_AMDGPU_REDUCE_GUARDS_I32(
        LOOM_COMBINING_KIND_ANDI, LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32),
    [LOOM_AMDGPU_REDUCE_GUARDS_ORI] = LOOM_AMDGPU_REDUCE_GUARDS_I32(
        LOOM_COMBINING_KIND_ORI, LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32),
    [LOOM_AMDGPU_REDUCE_GUARDS_XORI] = LOOM_AMDGPU_REDUCE_GUARDS_I32(
        LOOM_COMBINING_KIND_XORI, LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32),
    [LOOM_AMDGPU_REDUCE_GUARDS_ADDF] = LOOM_AMDGPU_REDUCE_GUARDS_F32(
        LOOM_COMBINING_KIND_ADDF, LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_F32),
    [LOOM_AMDGPU_REDUCE_GUARDS_MULF] = LOOM_AMDGPU_REDUCE_GUARDS_F32(
        LOOM_COMBINING_KIND_MULF, LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_F32),
    [LOOM_AMDGPU_REDUCE_GUARDS_MINNUMF] = LOOM_AMDGPU_REDUCE_GUARDS_F32(
        LOOM_COMBINING_KIND_MINNUMF, LOOM_AMDGPU_DESCRIPTOR_ID_V_MIN_F32),
    [LOOM_AMDGPU_REDUCE_GUARDS_MAXNUMF] = LOOM_AMDGPU_REDUCE_GUARDS_F32(
        LOOM_COMBINING_KIND_MAXNUMF, LOOM_AMDGPU_DESCRIPTOR_ID_V_MAX_F32),
};

static_assert(IREE_ARRAYSIZE(kAmdgpuReduceGuards) ==
                  LOOM_AMDGPU_REDUCE_GUARD_COUNT_,
              "AMDGPU reduce guard rows must cover the guard ranges");

#undef LOOM_AMDGPU_REDUCE_GUARDS_F32
#undef LOOM_AMDGPU_REDUCE_GUARDS_I32
#undef LOOM_AMDGPU_REDUCE_GUARD_DESCRIPTOR
#undef LOOM_AMDGPU_REDUCE_GUARD_MATERIALIZABLE
#undef LOOM_AMDGPU_REDUCE_GUARD_LOW_REG_CLASS
#undef LOOM_AMDGPU_REDUCE_GUARD_VALUE
#undef LOOM_AMDGPU_REDUCE_GUARD_KIND

#define LOOM_AMDGPU_REDUCE_EMIT_I32(descriptor)                   \
  {                                                               \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES, \
      .descriptor_id = descriptor,                                \
      .operand_ref_start = LOOM_AMDGPU_REDUCE_I32_VGPR_INIT,      \
      .operand_ref_count = 2,                                     \
      .accumulator_operand_index = 0,                             \
      .result_ref_start = LOOM_AMDGPU_REDUCE_RESULT,              \
      .result_ref_count = 1,                                      \
  }

#define LOOM_AMDGPU_REDUCE_EMIT_F32(descriptor)                   \
  {                                                               \
      .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES, \
      .descriptor_id = descriptor,                                \
      .operand_ref_start = LOOM_AMDGPU_REDUCE_F32_INIT,           \
      .operand_ref_count = 2,                                     \
      .accumulator_operand_index = 0,                             \
      .result_ref_start = LOOM_AMDGPU_REDUCE_RESULT,              \
      .result_ref_count = 1,                                      \
  }

static const loom_low_lower_emit_t kAmdgpuReduceEmits[] = {
    [LOOM_AMDGPU_REDUCE_RULE_ADDI] =
        LOOM_AMDGPU_REDUCE_EMIT_I32(LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_U32),
    [LOOM_AMDGPU_REDUCE_RULE_MULI] =
        LOOM_AMDGPU_REDUCE_EMIT_I32(LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_LO_U32),
    [LOOM_AMDGPU_REDUCE_RULE_ANDI] =
        LOOM_AMDGPU_REDUCE_EMIT_I32(LOOM_AMDGPU_DESCRIPTOR_ID_V_AND_B32),
    [LOOM_AMDGPU_REDUCE_RULE_ORI] =
        LOOM_AMDGPU_REDUCE_EMIT_I32(LOOM_AMDGPU_DESCRIPTOR_ID_V_OR_B32),
    [LOOM_AMDGPU_REDUCE_RULE_XORI] =
        LOOM_AMDGPU_REDUCE_EMIT_I32(LOOM_AMDGPU_DESCRIPTOR_ID_V_XOR_B32),
    [LOOM_AMDGPU_REDUCE_RULE_ADDF] =
        LOOM_AMDGPU_REDUCE_EMIT_F32(LOOM_AMDGPU_DESCRIPTOR_ID_V_ADD_F32),
    [LOOM_AMDGPU_REDUCE_RULE_MULF] =
        LOOM_AMDGPU_REDUCE_EMIT_F32(LOOM_AMDGPU_DESCRIPTOR_ID_V_MUL_F32),
    [LOOM_AMDGPU_REDUCE_RULE_MINNUMF] =
        LOOM_AMDGPU_REDUCE_EMIT_F32(LOOM_AMDGPU_DESCRIPTOR_ID_V_MIN_F32),
    [LOOM_AMDGPU_REDUCE_RULE_MAXNUMF] =
        LOOM_AMDGPU_REDUCE_EMIT_F32(LOOM_AMDGPU_DESCRIPTOR_ID_V_MAX_F32),
};

static_assert(IREE_ARRAYSIZE(kAmdgpuReduceEmits) ==
                  LOOM_AMDGPU_REDUCE_RULE_COUNT_,
              "AMDGPU reduce emit rows must align with rule rows");

#undef LOOM_AMDGPU_REDUCE_EMIT_F32
#undef LOOM_AMDGPU_REDUCE_EMIT_I32

#define LOOM_AMDGPU_REDUCE_RULE(guard_start_value, guard_count_value, \
                                emit_index)                           \
  {                                                                   \
      .source_op_kind = LOOM_OP_VECTOR_REDUCE,                        \
      .guard_start = guard_start_value,                               \
      .guard_count = guard_count_value,                               \
      .emit_start = emit_index,                                       \
      .emit_count = 1,                                                \
  }

static const loom_low_lower_rule_t kAmdgpuReduceRules[] = {
    [LOOM_AMDGPU_REDUCE_RULE_ADDI] = LOOM_AMDGPU_REDUCE_RULE(
        LOOM_AMDGPU_REDUCE_GUARDS_ADDI, LOOM_AMDGPU_REDUCE_GUARD_COUNT_I32,
        LOOM_AMDGPU_REDUCE_RULE_ADDI),
    [LOOM_AMDGPU_REDUCE_RULE_MULI] = LOOM_AMDGPU_REDUCE_RULE(
        LOOM_AMDGPU_REDUCE_GUARDS_MULI, LOOM_AMDGPU_REDUCE_GUARD_COUNT_I32,
        LOOM_AMDGPU_REDUCE_RULE_MULI),
    [LOOM_AMDGPU_REDUCE_RULE_ANDI] = LOOM_AMDGPU_REDUCE_RULE(
        LOOM_AMDGPU_REDUCE_GUARDS_ANDI, LOOM_AMDGPU_REDUCE_GUARD_COUNT_I32,
        LOOM_AMDGPU_REDUCE_RULE_ANDI),
    [LOOM_AMDGPU_REDUCE_RULE_ORI] = LOOM_AMDGPU_REDUCE_RULE(
        LOOM_AMDGPU_REDUCE_GUARDS_ORI, LOOM_AMDGPU_REDUCE_GUARD_COUNT_I32,
        LOOM_AMDGPU_REDUCE_RULE_ORI),
    [LOOM_AMDGPU_REDUCE_RULE_XORI] = LOOM_AMDGPU_REDUCE_RULE(
        LOOM_AMDGPU_REDUCE_GUARDS_XORI, LOOM_AMDGPU_REDUCE_GUARD_COUNT_I32,
        LOOM_AMDGPU_REDUCE_RULE_XORI),
    [LOOM_AMDGPU_REDUCE_RULE_ADDF] = LOOM_AMDGPU_REDUCE_RULE(
        LOOM_AMDGPU_REDUCE_GUARDS_ADDF, LOOM_AMDGPU_REDUCE_GUARD_COUNT_F32,
        LOOM_AMDGPU_REDUCE_RULE_ADDF),
    [LOOM_AMDGPU_REDUCE_RULE_MULF] = LOOM_AMDGPU_REDUCE_RULE(
        LOOM_AMDGPU_REDUCE_GUARDS_MULF, LOOM_AMDGPU_REDUCE_GUARD_COUNT_F32,
        LOOM_AMDGPU_REDUCE_RULE_MULF),
    [LOOM_AMDGPU_REDUCE_RULE_MINNUMF] = LOOM_AMDGPU_REDUCE_RULE(
        LOOM_AMDGPU_REDUCE_GUARDS_MINNUMF, LOOM_AMDGPU_REDUCE_GUARD_COUNT_F32,
        LOOM_AMDGPU_REDUCE_RULE_MINNUMF),
    [LOOM_AMDGPU_REDUCE_RULE_MAXNUMF] = LOOM_AMDGPU_REDUCE_RULE(
        LOOM_AMDGPU_REDUCE_GUARDS_MAXNUMF, LOOM_AMDGPU_REDUCE_GUARD_COUNT_F32,
        LOOM_AMDGPU_REDUCE_RULE_MAXNUMF),
};

static_assert(IREE_ARRAYSIZE(kAmdgpuReduceRules) ==
                  LOOM_AMDGPU_REDUCE_RULE_COUNT_,
              "AMDGPU reduce rule indexes must cover the rule table");

#undef LOOM_AMDGPU_REDUCE_RULE

static const loom_low_lower_rule_span_t kAmdgpuReduceRuleSpans[] = {
    {
        .source_op_kind = LOOM_OP_VECTOR_REDUCE,
        .rule_start = LOOM_AMDGPU_REDUCE_RULE_ADDI,
        .rule_count = LOOM_AMDGPU_REDUCE_RULE_COUNT_,
    },
};

const loom_low_lower_rule_set_t loom_amdgpu_reduce_rule_set = {
    .spans = kAmdgpuReduceRuleSpans,
    .span_count = IREE_ARRAYSIZE(kAmdgpuReduceRuleSpans),
    .rules = kAmdgpuReduceRules,
    .rule_count = IREE_ARRAYSIZE(kAmdgpuReduceRules),
    .type_patterns = kAmdgpuReduceTypePatterns,
    .type_pattern_count = IREE_ARRAYSIZE(kAmdgpuReduceTypePatterns),
    .value_refs = kAmdgpuReduceValueRefs,
    .value_ref_count = IREE_ARRAYSIZE(kAmdgpuReduceValueRefs),
    .materializers = kAmdgpuReduceMaterializers,
    .materializer_count = IREE_ARRAYSIZE(kAmdgpuReduceMaterializers),
    .guards = kAmdgpuReduceGuards,
    .guard_count = IREE_ARRAYSIZE(kAmdgpuReduceGuards),
    .emits = kAmdgpuReduceEmits,
    .emit_count = IREE_ARRAYSIZE(kAmdgpuReduceEmits),
    .diagnostics = kAmdgpuReduceDiagnostics,
    .diagnostic_count = IREE_ARRAYSIZE(kAmdgpuReduceDiagnostics),
};
