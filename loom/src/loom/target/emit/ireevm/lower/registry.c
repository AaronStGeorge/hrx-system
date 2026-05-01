// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/codegen/low/lower_rules.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/emit/ireevm/descriptors.h"
#include "loom/target/emit/ireevm/lower.h"

static bool loom_ireevm_type_is_i1_or_i32(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  loom_scalar_type_t element_type = loom_type_element_type(type);
  return element_type == LOOM_SCALAR_TYPE_I1 ||
         element_type == LOOM_SCALAR_TYPE_I32;
}

static iree_status_t loom_ireevm_make_vm_i32_register_type(
    loom_low_lower_context_t* context, loom_type_t* out_type) {
  return loom_low_lower_make_register_type(
      context, IREE_VM_CORE_REG_CLASS_ID_VM_I32, 1, out_type);
}

static iree_status_t loom_ireevm_map_type(void* user_data,
                                          loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_type_t source_type,
                                          loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_ireevm_type_is_i1_or_i32(source_type)) {
    return loom_ireevm_make_vm_i32_register_type(context, out_low_type);
  }
  return loom_low_lower_emit_reject(
      context, source_op, IREE_SV("type"), IREE_SV("source"),
      IREE_SV("IREE VM lowering currently supports only i1/i32 scalar values"));
}

enum loom_ireevm_type_pattern_e {
  LOOM_IREEVM_TYPE_I1 = 0,
  LOOM_IREEVM_TYPE_I32 = 1,
  LOOM_IREEVM_TYPE_I1_OR_I32 = 2,
};

static const loom_low_lower_type_pattern_t loom_ireevm_type_patterns[] = {
    [LOOM_IREEVM_TYPE_I1] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I1),
        },
    [LOOM_IREEVM_TYPE_I32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I32),
        },
    [LOOM_IREEVM_TYPE_I1_OR_I32] =
        {
            .flags = LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND |
                     LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT,
            .type_kind = LOOM_TYPE_SCALAR,
            .element_type_mask =
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I1) |
                LOOM_LOW_LOWER_SCALAR_TYPE_BIT(LOOM_SCALAR_TYPE_I32),
        },
};

enum loom_ireevm_value_ref_e {
  LOOM_IREEVM_OPERAND0 = 0,
  LOOM_IREEVM_OPERAND1 = 1,
  LOOM_IREEVM_RESULT0 = 2,
};

static const loom_low_lower_value_ref_t loom_ireevm_value_refs[] = {
    [LOOM_IREEVM_OPERAND0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 0,
        },
    [LOOM_IREEVM_OPERAND1] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND,
            .index = 1,
        },
    [LOOM_IREEVM_RESULT0] =
        {
            .kind = LOOM_LOW_LOWER_VALUE_REF_RESULT,
            .index = 0,
        },
};

static const loom_low_lower_attr_copy_t loom_ireevm_attr_copies[] = {
    {
        .target_name = IREE_SVL("i32_value"),
        .source_attr_index = 0,
    },
};

enum loom_ireevm_diagnostic_e {
  LOOM_IREEVM_DIAGNOSTIC_I1_OR_I32 = 0,
  LOOM_IREEVM_DIAGNOSTIC_I32 = 1,
  LOOM_IREEVM_DIAGNOSTIC_I64_ATTR = 2,
  LOOM_IREEVM_DIAGNOSTIC_I1_CONSTANT_RANGE = 3,
  LOOM_IREEVM_DIAGNOSTIC_I32_CONSTANT_RANGE = 4,
  LOOM_IREEVM_DIAGNOSTIC_CMPI_EQ = 5,
};

static const loom_low_lower_diagnostic_t loom_ireevm_diagnostics[] = {
    [LOOM_IREEVM_DIAGNOSTIC_I1_OR_I32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("source"),
            .reason = IREE_SVL(
                "IREE VM lowering currently supports only i1/i32 scalar "
                "values"),
        },
    [LOOM_IREEVM_DIAGNOSTIC_I32] =
        {
            .subject_kind = IREE_SVL("type"),
            .subject_name = IREE_SVL("i32"),
            .reason = IREE_SVL("IREE VM lowering requires i32 scalar operands"),
        },
    [LOOM_IREEVM_DIAGNOSTIC_I64_ATTR] =
        {
            .subject_kind = IREE_SVL("attr"),
            .subject_name = IREE_SVL("value"),
            .reason =
                IREE_SVL("IREE VM constant lowering requires an i64 value"),
        },
    [LOOM_IREEVM_DIAGNOSTIC_I1_CONSTANT_RANGE] =
        {
            .subject_kind = IREE_SVL("attr"),
            .subject_name = IREE_SVL("value"),
            .reason =
                IREE_SVL("IREE VM i1 constants must be either zero or one"),
        },
    [LOOM_IREEVM_DIAGNOSTIC_I32_CONSTANT_RANGE] =
        {
            .subject_kind = IREE_SVL("attr"),
            .subject_name = IREE_SVL("value"),
            .reason = IREE_SVL("IREE VM i32 constants must fit in signed i32"),
        },
    [LOOM_IREEVM_DIAGNOSTIC_CMPI_EQ] =
        {
            .subject_kind = IREE_SVL("attr"),
            .subject_name = IREE_SVL("predicate"),
            .reason = IREE_SVL("IREE VM scalar.cmpi lowering supports eq only"),
        },
};

enum loom_ireevm_guard_e {
  LOOM_IREEVM_CONST_VALUE_GUARD = 0,
  LOOM_IREEVM_CONST_I1_RESULT_GUARD = 1,
  LOOM_IREEVM_CONST_I1_RANGE_GUARD = 2,
  LOOM_IREEVM_CONST_I32_VALUE_GUARD = 3,
  LOOM_IREEVM_CONST_I32_RESULT_GUARD = 4,
  LOOM_IREEVM_CONST_I32_RANGE_GUARD = 5,
  LOOM_IREEVM_BINARY_LHS_GUARD = 6,
  LOOM_IREEVM_BINARY_RHS_GUARD = 7,
  LOOM_IREEVM_BINARY_RESULT_GUARD = 8,
  LOOM_IREEVM_CMPI_EQ_GUARD = 9,
  LOOM_IREEVM_CMPI_LHS_GUARD = 10,
  LOOM_IREEVM_CMPI_RHS_GUARD = 11,
  LOOM_IREEVM_CMPI_RESULT_GUARD = 12,
};

static const loom_low_lower_guard_t loom_ireevm_guards[] = {
    [LOOM_IREEVM_CONST_VALUE_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_ATTR_KIND,
            .attr_index = 0,
            .diagnostic_index = LOOM_IREEVM_DIAGNOSTIC_I64_ATTR,
            .attr_kind = LOOM_ATTR_I64,
        },
    [LOOM_IREEVM_CONST_I1_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_IREEVM_RESULT0,
            .type_pattern_index = LOOM_IREEVM_TYPE_I1,
            .diagnostic_index = LOOM_IREEVM_DIAGNOSTIC_I1_OR_I32,
        },
    [LOOM_IREEVM_CONST_I1_RANGE_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE,
            .attr_index = 0,
            .diagnostic_index = LOOM_IREEVM_DIAGNOSTIC_I1_CONSTANT_RANGE,
            .minimum_i64 = 0,
            .maximum_i64 = 1,
        },
    [LOOM_IREEVM_CONST_I32_VALUE_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_ATTR_KIND,
            .attr_index = 0,
            .diagnostic_index = LOOM_IREEVM_DIAGNOSTIC_I64_ATTR,
            .attr_kind = LOOM_ATTR_I64,
        },
    [LOOM_IREEVM_CONST_I32_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_IREEVM_RESULT0,
            .type_pattern_index = LOOM_IREEVM_TYPE_I32,
            .diagnostic_index = LOOM_IREEVM_DIAGNOSTIC_I1_OR_I32,
        },
    [LOOM_IREEVM_CONST_I32_RANGE_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE,
            .attr_index = 0,
            .diagnostic_index = LOOM_IREEVM_DIAGNOSTIC_I32_CONSTANT_RANGE,
            .minimum_i64 = INT32_MIN,
            .maximum_i64 = INT32_MAX,
        },
    [LOOM_IREEVM_BINARY_LHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_IREEVM_OPERAND0,
            .type_pattern_index = LOOM_IREEVM_TYPE_I32,
            .diagnostic_index = LOOM_IREEVM_DIAGNOSTIC_I32,
        },
    [LOOM_IREEVM_BINARY_RHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_IREEVM_OPERAND1,
            .type_pattern_index = LOOM_IREEVM_TYPE_I32,
            .diagnostic_index = LOOM_IREEVM_DIAGNOSTIC_I32,
        },
    [LOOM_IREEVM_BINARY_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_IREEVM_RESULT0,
            .type_pattern_index = LOOM_IREEVM_TYPE_I32,
            .diagnostic_index = LOOM_IREEVM_DIAGNOSTIC_I32,
        },
    [LOOM_IREEVM_CMPI_EQ_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ,
            .attr_index = 0,
            .diagnostic_index = LOOM_IREEVM_DIAGNOSTIC_CMPI_EQ,
            .u64 = LOOM_SCALAR_CMPI_PREDICATE_EQ,
        },
    [LOOM_IREEVM_CMPI_LHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_IREEVM_OPERAND0,
            .type_pattern_index = LOOM_IREEVM_TYPE_I32,
            .diagnostic_index = LOOM_IREEVM_DIAGNOSTIC_I32,
        },
    [LOOM_IREEVM_CMPI_RHS_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_IREEVM_OPERAND1,
            .type_pattern_index = LOOM_IREEVM_TYPE_I32,
            .diagnostic_index = LOOM_IREEVM_DIAGNOSTIC_I32,
        },
    [LOOM_IREEVM_CMPI_RESULT_GUARD] =
        {
            .kind = LOOM_LOW_LOWER_GUARD_VALUE_TYPE,
            .value_ref_index = LOOM_IREEVM_RESULT0,
            .type_pattern_index = LOOM_IREEVM_TYPE_I1_OR_I32,
            .diagnostic_index = LOOM_IREEVM_DIAGNOSTIC_I1_OR_I32,
        },
};

enum loom_ireevm_emit_e {
  LOOM_IREEVM_EMIT_CONST_I32 = 0,
  LOOM_IREEVM_EMIT_ADD_I32 = 1,
  LOOM_IREEVM_EMIT_SUB_I32 = 2,
  LOOM_IREEVM_EMIT_CMP_EQ_I32 = 3,
};

static const loom_low_lower_emit_t loom_ireevm_emits[] = {
    [LOOM_IREEVM_EMIT_CONST_I32] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST,
            .descriptor_id = IREE_VM_CORE_DESCRIPTOR_ID_CONST_I32,
            .result_ref_start = LOOM_IREEVM_RESULT0,
            .result_ref_count = 1,
            .attr_copy_start = 0,
            .attr_copy_count = 1,
        },
    [LOOM_IREEVM_EMIT_ADD_I32] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = IREE_VM_CORE_DESCRIPTOR_ID_ADD_I32,
            .operand_ref_start = LOOM_IREEVM_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_IREEVM_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_IREEVM_EMIT_SUB_I32] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = IREE_VM_CORE_DESCRIPTOR_ID_SUB_I32,
            .operand_ref_start = LOOM_IREEVM_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_IREEVM_RESULT0,
            .result_ref_count = 1,
        },
    [LOOM_IREEVM_EMIT_CMP_EQ_I32] =
        {
            .kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP,
            .descriptor_id = IREE_VM_CORE_DESCRIPTOR_ID_CMP_EQ_I32,
            .operand_ref_start = LOOM_IREEVM_OPERAND0,
            .operand_ref_count = 2,
            .result_ref_start = LOOM_IREEVM_RESULT0,
            .result_ref_count = 1,
        },
};

static const loom_low_lower_rule_t loom_ireevm_rules[] = {
    {
        .source_op_kind = LOOM_OP_SCALAR_ADDI,
        .guard_start = LOOM_IREEVM_BINARY_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_IREEVM_EMIT_ADD_I32,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SUBI,
        .guard_start = LOOM_IREEVM_BINARY_LHS_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_IREEVM_EMIT_SUB_I32,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_CMPI,
        .guard_start = LOOM_IREEVM_CMPI_EQ_GUARD,
        .guard_count = 4,
        .emit_start = LOOM_IREEVM_EMIT_CMP_EQ_I32,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_CONSTANT,
        .guard_start = LOOM_IREEVM_CONST_I32_VALUE_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_IREEVM_EMIT_CONST_I32,
        .emit_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_CONSTANT,
        .guard_start = LOOM_IREEVM_CONST_VALUE_GUARD,
        .guard_count = 3,
        .emit_start = LOOM_IREEVM_EMIT_CONST_I32,
        .emit_count = 1,
    },
};

static const loom_low_lower_rule_span_t loom_ireevm_rule_spans[] = {
    {
        .source_op_kind = LOOM_OP_SCALAR_ADDI,
        .rule_start = 0,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_SUBI,
        .rule_start = 1,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_CMPI,
        .rule_start = 2,
        .rule_count = 1,
    },
    {
        .source_op_kind = LOOM_OP_SCALAR_CONSTANT,
        .rule_start = 3,
        .rule_count = 2,
    },
};

static const loom_low_lower_rule_set_t loom_ireevm_rule_set = {
    .flags = LOOM_LOW_LOWER_RULE_SET_FLAG_TARGET_CONTRACT_QUERY,
    .spans = loom_ireevm_rule_spans,
    .span_count = IREE_ARRAYSIZE(loom_ireevm_rule_spans),
    .rules = loom_ireevm_rules,
    .rule_count = IREE_ARRAYSIZE(loom_ireevm_rules),
    .type_patterns = loom_ireevm_type_patterns,
    .type_pattern_count = IREE_ARRAYSIZE(loom_ireevm_type_patterns),
    .value_refs = loom_ireevm_value_refs,
    .value_ref_count = IREE_ARRAYSIZE(loom_ireevm_value_refs),
    .guards = loom_ireevm_guards,
    .guard_count = IREE_ARRAYSIZE(loom_ireevm_guards),
    .attr_copies = loom_ireevm_attr_copies,
    .attr_copy_count = IREE_ARRAYSIZE(loom_ireevm_attr_copies),
    .emits = loom_ireevm_emits,
    .emit_count = IREE_ARRAYSIZE(loom_ireevm_emits),
    .diagnostics = loom_ireevm_diagnostics,
    .diagnostic_count = IREE_ARRAYSIZE(loom_ireevm_diagnostics),
};

static const loom_low_lower_rule_set_t* const kIreeVmRuleSets[] = {
    &loom_ireevm_rule_set,
};

static const loom_low_lower_policy_t kIreeVmLowLowerPolicy = {
    .name = IREE_SVL("iree-vm-lower"),
    .map_type = {.fn = loom_ireevm_map_type, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kIreeVmRuleSets),
            .values = kIreeVmRuleSets,
        },
};

const loom_low_lower_policy_t* loom_ireevm_low_lower_policy(void) {
  return &kIreeVmLowLowerPolicy;
}

void loom_ireevm_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("iree.vm.core"),
          .policy = &kIreeVmLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
