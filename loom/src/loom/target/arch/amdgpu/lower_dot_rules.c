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
};

#define LOOM_AMDGPU_DOT_GUARD_VALUE(value_ref, type_pattern, diagnostic) \
  {                                                                      \
      .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,                           \
      .value_ref_index = value_ref,                                      \
      .type_pattern_index = type_pattern,                                \
      .diagnostic_index = diagnostic,                                    \
  }

#define LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS(value_ref, diagnostic) \
  {                                                                \
      .kind = LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS,       \
      .value_ref_index = value_ref,                                \
      .diagnostic_index = diagnostic,                              \
      .register_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR,          \
  }

#define LOOM_AMDGPU_DOT_GUARD_DESCRIPTOR(descriptor)             \
  {                                                              \
      .kind = LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE,         \
      .diagnostic_index = LOOM_AMDGPU_DOT_DIAGNOSTIC_DESCRIPTOR, \
      .descriptor_id = descriptor,                               \
  }

enum loom_amdgpu_dot_rule_e {
  LOOM_AMDGPU_DOT_RULE_DOTF = 0,
  LOOM_AMDGPU_DOT_RULE_COUNT_ = 1,
};

enum loom_amdgpu_dot_guard_range_e {
  LOOM_AMDGPU_DOT_GUARDS_DOTF = 0,
  LOOM_AMDGPU_DOT_GUARD_COUNT_DOTF = 7,
  LOOM_AMDGPU_DOT_GUARD_COUNT_ =
      LOOM_AMDGPU_DOT_GUARDS_DOTF + LOOM_AMDGPU_DOT_GUARD_COUNT_DOTF,
};

static const loom_low_lower_guard_t kAmdgpuDotGuards[] = {
    [LOOM_AMDGPU_DOT_GUARDS_DOTF] = LOOM_AMDGPU_DOT_GUARD_VALUE(
        LOOM_AMDGPU_DOT_LHS, LOOM_AMDGPU_DOT_TYPE_VF32,
        LOOM_AMDGPU_DOT_DIAGNOSTIC_LHS),
    LOOM_AMDGPU_DOT_GUARD_VALUE(LOOM_AMDGPU_DOT_RHS, LOOM_AMDGPU_DOT_TYPE_VF32,
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
    LOOM_AMDGPU_DOT_GUARD_DESCRIPTOR(LOOM_AMDGPU_DESCRIPTOR_ID_V_FMA_F32),
};

static_assert(IREE_ARRAYSIZE(kAmdgpuDotGuards) == LOOM_AMDGPU_DOT_GUARD_COUNT_,
              "AMDGPU dot guard rows must cover the guard ranges");

#undef LOOM_AMDGPU_DOT_GUARD_DESCRIPTOR
#undef LOOM_AMDGPU_DOT_GUARD_LOW_REG_CLASS
#undef LOOM_AMDGPU_DOT_GUARD_VALUE

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
};

static_assert(IREE_ARRAYSIZE(kAmdgpuDotEmits) == LOOM_AMDGPU_DOT_RULE_COUNT_,
              "AMDGPU dot emit rows must align with rule rows");

static const loom_low_lower_rule_t kAmdgpuDotRules[] = {
    [LOOM_AMDGPU_DOT_RULE_DOTF] =
        {
            .source_op_kind = LOOM_OP_VECTOR_DOTF,
            .guard_start = LOOM_AMDGPU_DOT_GUARDS_DOTF,
            .guard_count = LOOM_AMDGPU_DOT_GUARD_COUNT_DOTF,
            .emit_start = LOOM_AMDGPU_DOT_RULE_DOTF,
            .emit_count = 1,
        },
};

static_assert(IREE_ARRAYSIZE(kAmdgpuDotRules) == LOOM_AMDGPU_DOT_RULE_COUNT_,
              "AMDGPU dot rule indexes must cover the rule table");

static const loom_low_lower_rule_span_t kAmdgpuDotRuleSpans[] = {
    {
        .source_op_kind = LOOM_OP_VECTOR_DOTF,
        .rule_start = LOOM_AMDGPU_DOT_RULE_DOTF,
        .rule_count = LOOM_AMDGPU_DOT_RULE_COUNT_,
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
