// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/descriptors.h"

#include <stddef.h>

typedef struct loom_ireevm_descriptor_strings_t {
  char empty[sizeof("")];
  char set_key[sizeof("iree.vm.core")];
  char target_key[sizeof("iree.vm")];
  char feature_key[sizeof("iree.vm.v1")];
  char source[sizeof("loom-hand-authored-iree-vm-core")];
  char reg_i32[sizeof("vm.i32")];
  char reg_i64[sizeof("vm.i64")];
  char reg_f32[sizeof("vm.f32")];
  char reg_f64[sizeof("vm.f64")];
  char reg_ref[sizeof("vm.ref")];
  char reg_list[sizeof("vm.list")];
  char field_dst[sizeof("dst")];
  char field_lhs[sizeof("lhs")];
  char field_rhs[sizeof("rhs")];
  char field_value[sizeof("value")];
  char field_cond[sizeof("cond")];
  char field_arg0[sizeof("arg0")];
  char immediate_i32_value[sizeof("i32_value")];
  char immediate_target_block[sizeof("target_block")];
  char immediate_true_block[sizeof("true_block")];
  char immediate_false_block[sizeof("false_block")];
  char immediate_callee[sizeof("callee_ordinal")];
  char resource_alu[sizeof("vm.alu")];
  char resource_control[sizeof("vm.control")];
  char resource_call[sizeof("vm.call")];
  char schedule_const[sizeof("vm.const")];
  char schedule_alu_i32[sizeof("vm.alu.i32")];
  char schedule_control[sizeof("vm.control")];
  char schedule_call[sizeof("vm.call")];
  char key_const_i32[sizeof("iree.vm.const.i32")];
  char key_add_i32[sizeof("iree.vm.add.i32")];
  char key_sub_i32[sizeof("iree.vm.sub.i32")];
  char key_cmp_eq_i32[sizeof("iree.vm.cmp.eq.i32")];
  char key_br[sizeof("iree.vm.br")];
  char key_cond_br_i32[sizeof("iree.vm.cond_br.i32")];
  char key_call_import_i32[sizeof("iree.vm.call.import.i32")];
  char key_return_i32[sizeof("iree.vm.return.i32")];
  char key_return_void[sizeof("iree.vm.return.void")];
  char mnemonic_const_i32[sizeof("vm.const.i32")];
  char mnemonic_add_i32[sizeof("vm.add.i32")];
  char mnemonic_sub_i32[sizeof("vm.sub.i32")];
  char mnemonic_cmp_eq_i32[sizeof("vm.cmp.eq.i32")];
  char mnemonic_br[sizeof("vm.br")];
  char mnemonic_cond_br_i32[sizeof("vm.cond_br.i32")];
  char mnemonic_call_import_i32[sizeof("vm.call.import.i32")];
  char mnemonic_return_i32[sizeof("vm.return.i32")];
  char mnemonic_return_void[sizeof("vm.return.void")];
  char semantic_const_i32[sizeof("integer.const.i32")];
  char semantic_add_i32[sizeof("integer.add.i32")];
  char semantic_sub_i32[sizeof("integer.sub.i32")];
  char semantic_cmp_eq_i32[sizeof("integer.cmp.eq.i32")];
  char semantic_branch[sizeof("control.branch")];
  char semantic_cond_branch[sizeof("control.cond_branch.i32")];
  char semantic_call[sizeof("call.import.i32")];
  char semantic_return_i32[sizeof("control.return.i32")];
  char semantic_return_void[sizeof("control.return.void")];
} loom_ireevm_descriptor_strings_t;

static const loom_ireevm_descriptor_strings_t kStrings = {
    "",
    "iree.vm.core",
    "iree.vm",
    "iree.vm.v1",
    "loom-hand-authored-iree-vm-core",
    "vm.i32",
    "vm.i64",
    "vm.f32",
    "vm.f64",
    "vm.ref",
    "vm.list",
    "dst",
    "lhs",
    "rhs",
    "value",
    "cond",
    "arg0",
    "i32_value",
    "target_block",
    "true_block",
    "false_block",
    "callee_ordinal",
    "vm.alu",
    "vm.control",
    "vm.call",
    "vm.const",
    "vm.alu.i32",
    "vm.control",
    "vm.call",
    "iree.vm.const.i32",
    "iree.vm.add.i32",
    "iree.vm.sub.i32",
    "iree.vm.cmp.eq.i32",
    "iree.vm.br",
    "iree.vm.cond_br.i32",
    "iree.vm.call.import.i32",
    "iree.vm.return.i32",
    "iree.vm.return.void",
    "vm.const.i32",
    "vm.add.i32",
    "vm.sub.i32",
    "vm.cmp.eq.i32",
    "vm.br",
    "vm.cond_br.i32",
    "vm.call.import.i32",
    "vm.return.i32",
    "vm.return.void",
    "integer.const.i32",
    "integer.add.i32",
    "integer.sub.i32",
    "integer.cmp.eq.i32",
    "control.branch",
    "control.cond_branch.i32",
    "call.import.i32",
    "control.return.i32",
    "control.return.void",
};

#define VM_STRING_OFFSET(field) \
  ((uint32_t)offsetof(loom_ireevm_descriptor_strings_t, field))

enum {
  VM_REG_I32 = 0,
  VM_REG_I64 = 1,
  VM_REG_F32 = 2,
  VM_REG_F64 = 3,
  VM_REG_REF = 4,
  VM_REG_LIST = 5,
};

enum {
  VM_ALT_I32 = 0,
  VM_ALT_I64 = 1,
  VM_ALT_F32 = 2,
  VM_ALT_F64 = 3,
  VM_ALT_REF = 4,
  VM_ALT_LIST = 5,
};

enum {
  VM_RESOURCE_ALU = 0,
  VM_RESOURCE_CONTROL = 1,
  VM_RESOURCE_CALL = 2,
};

enum {
  VM_SCHEDULE_CONST = 0,
  VM_SCHEDULE_ALU_I32 = 1,
  VM_SCHEDULE_CONTROL = 2,
  VM_SCHEDULE_CALL = 3,
};

enum {
  VM_EFFECT_CONTROL = 0,
  VM_EFFECT_CALL = 1,
};

#define VM_REG_CLASS(name_field, unit_bits, flags_value)  \
  {                                                       \
      .name_string_offset = VM_STRING_OFFSET(name_field), \
      .flags = (flags_value),                             \
      .alloc_unit_bits = (unit_bits),                     \
      .spill_class_id = LOOM_LOW_REG_CLASS_NONE,          \
  }

#define VM_ALT(reg_class_id_value)                    \
  {                                                   \
      .reg_class_id = (reg_class_id_value),           \
      .flags = LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED, \
  }

#define VM_VALUE(field_name, role_value, alt_start_value)       \
  {                                                             \
      .field_name_string_offset = VM_STRING_OFFSET(field_name), \
      .role = (role_value),                                     \
      .reg_class_alt_start = (alt_start_value),                 \
      .reg_class_alt_count = 1,                                 \
      .unit_count = 1,                                          \
  }

#define VM_DESCRIPTOR(key_field, mnemonic_field, semantic_field,             \
                      operand_start_value, operand_count_value,              \
                      result_count_value, immediate_start_value,             \
                      immediate_count_value, effect_start_value,             \
                      effect_count_value, schedule_class_value, flags_value) \
  {                                                                          \
      .key_string_offset = VM_STRING_OFFSET(key_field),                      \
      .mnemonic_string_offset = VM_STRING_OFFSET(mnemonic_field),            \
      .source_string_offset = VM_STRING_OFFSET(source),                      \
      .semantic_tag_string_offset = VM_STRING_OFFSET(semantic_field),        \
      .operand_start = (operand_start_value),                                \
      .operand_count = (operand_count_value),                                \
      .result_count = (result_count_value),                                  \
      .immediate_start = (immediate_start_value),                            \
      .immediate_count = (immediate_count_value),                            \
      .effect_start = (effect_start_value),                                  \
      .effect_count = (effect_count_value),                                  \
      .schedule_class_id = (schedule_class_value),                           \
      .flags = (flags_value),                                                \
  }

static const loom_low_reg_class_t kRegClasses[] = {
    VM_REG_CLASS(reg_i32, 32, LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY),
    VM_REG_CLASS(reg_i64, 64, LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY),
    VM_REG_CLASS(reg_f32, 32, LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY),
    VM_REG_CLASS(reg_f64, 64, LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY),
    VM_REG_CLASS(reg_ref, 64,
                 LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY |
                     LOOM_LOW_REG_CLASS_FLAG_REFERENCE),
    VM_REG_CLASS(reg_list, 64,
                 LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY |
                     LOOM_LOW_REG_CLASS_FLAG_REFERENCE),
};

static const loom_low_reg_class_alt_t kRegClassAlts[] = {
    VM_ALT(VM_REG_I32), VM_ALT(VM_REG_I64), VM_ALT(VM_REG_F32),
    VM_ALT(VM_REG_F64), VM_ALT(VM_REG_REF), VM_ALT(VM_REG_LIST),
};

static const loom_low_operand_t kOperands[] = {
    VM_VALUE(field_dst, LOOM_LOW_OPERAND_ROLE_RESULT, VM_ALT_I32),

    VM_VALUE(field_dst, LOOM_LOW_OPERAND_ROLE_RESULT, VM_ALT_I32),
    VM_VALUE(field_lhs, LOOM_LOW_OPERAND_ROLE_OPERAND, VM_ALT_I32),
    VM_VALUE(field_rhs, LOOM_LOW_OPERAND_ROLE_OPERAND, VM_ALT_I32),

    VM_VALUE(field_dst, LOOM_LOW_OPERAND_ROLE_RESULT, VM_ALT_I32),
    VM_VALUE(field_lhs, LOOM_LOW_OPERAND_ROLE_OPERAND, VM_ALT_I32),
    VM_VALUE(field_rhs, LOOM_LOW_OPERAND_ROLE_OPERAND, VM_ALT_I32),

    VM_VALUE(field_dst, LOOM_LOW_OPERAND_ROLE_RESULT, VM_ALT_I32),
    VM_VALUE(field_lhs, LOOM_LOW_OPERAND_ROLE_OPERAND, VM_ALT_I32),
    VM_VALUE(field_rhs, LOOM_LOW_OPERAND_ROLE_OPERAND, VM_ALT_I32),

    VM_VALUE(field_cond, LOOM_LOW_OPERAND_ROLE_PREDICATE, VM_ALT_I32),

    VM_VALUE(field_dst, LOOM_LOW_OPERAND_ROLE_RESULT, VM_ALT_I32),
    VM_VALUE(field_arg0, LOOM_LOW_OPERAND_ROLE_OPERAND, VM_ALT_I32),

    VM_VALUE(field_value, LOOM_LOW_OPERAND_ROLE_OPERAND, VM_ALT_I32),
};

static const loom_low_immediate_t kImmediates[] = {
    {
        .field_name_string_offset = VM_STRING_OFFSET(immediate_i32_value),
        .kind = LOOM_LOW_IMMEDIATE_KIND_SIGNED,
        .bit_width = 32,
        .signed_min = INT32_MIN,
        .unsigned_max = INT32_MAX,
    },
    {
        .field_name_string_offset = VM_STRING_OFFSET(immediate_target_block),
        .kind = LOOM_LOW_IMMEDIATE_KIND_ORDINAL,
        .flags = LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC,
        .bit_width = 32,
        .unsigned_max = UINT32_MAX,
    },
    {
        .field_name_string_offset = VM_STRING_OFFSET(immediate_true_block),
        .kind = LOOM_LOW_IMMEDIATE_KIND_ORDINAL,
        .flags = LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC,
        .bit_width = 32,
        .unsigned_max = UINT32_MAX,
    },
    {
        .field_name_string_offset = VM_STRING_OFFSET(immediate_false_block),
        .kind = LOOM_LOW_IMMEDIATE_KIND_ORDINAL,
        .flags = LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC,
        .bit_width = 32,
        .unsigned_max = UINT32_MAX,
    },
    {
        .field_name_string_offset = VM_STRING_OFFSET(immediate_callee),
        .kind = LOOM_LOW_IMMEDIATE_KIND_ORDINAL,
        .flags = LOOM_LOW_IMMEDIATE_FLAG_SYMBOLIC,
        .bit_width = 32,
        .unsigned_max = UINT32_MAX,
    },
};

static const loom_low_effect_t kEffects[] = {
    {
        .kind = LOOM_LOW_EFFECT_KIND_CONTROL,
        .flags = LOOM_LOW_EFFECT_FLAG_ORDERED,
    },
    {
        .kind = LOOM_LOW_EFFECT_KIND_CALL,
        .flags = LOOM_LOW_EFFECT_FLAG_ORDERED | LOOM_LOW_EFFECT_FLAG_DEPENDENCY,
    },
};

static const loom_low_resource_t kResources[] = {
    {
        .name_string_offset = VM_STRING_OFFSET(resource_alu),
        .capacity_per_cycle = 1,
        .kind = LOOM_LOW_RESOURCE_KIND_SCALAR_ALU,
    },
    {
        .name_string_offset = VM_STRING_OFFSET(resource_control),
        .capacity_per_cycle = 1,
        .kind = LOOM_LOW_RESOURCE_KIND_CONTROL,
    },
    {
        .name_string_offset = VM_STRING_OFFSET(resource_call),
        .capacity_per_cycle = 1,
        .kind = LOOM_LOW_RESOURCE_KIND_CONTROL,
    },
};

static const loom_low_issue_use_t kIssueUses[] = {
    {.resource_id = VM_RESOURCE_ALU, .cycles = 1, .units = 1},
    {.resource_id = VM_RESOURCE_CONTROL, .cycles = 1, .units = 1},
    {.resource_id = VM_RESOURCE_CALL, .cycles = 1, .units = 1},
};

static const loom_low_schedule_class_t kScheduleClasses[] = {
    {
        .name_string_offset = VM_STRING_OFFSET(schedule_const),
        .latency_kind = LOOM_LOW_LATENCY_KIND_EXACT,
        .model_quality = LOOM_LOW_MODEL_QUALITY_EXACT,
    },
    {
        .name_string_offset = VM_STRING_OFFSET(schedule_alu_i32),
        .latency_cycles = 1,
        .latency_kind = LOOM_LOW_LATENCY_KIND_EXACT,
        .issue_use_start = 0,
        .issue_use_count = 1,
        .model_quality = LOOM_LOW_MODEL_QUALITY_EXACT,
    },
    {
        .name_string_offset = VM_STRING_OFFSET(schedule_control),
        .latency_cycles = 1,
        .latency_kind = LOOM_LOW_LATENCY_KIND_EXACT,
        .issue_use_start = 1,
        .issue_use_count = 1,
        .flags = LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL,
        .model_quality = LOOM_LOW_MODEL_QUALITY_EXACT,
    },
    {
        .name_string_offset = VM_STRING_OFFSET(schedule_call),
        .latency_cycles = 1,
        .latency_kind = LOOM_LOW_LATENCY_KIND_VARIABLE,
        .issue_use_start = 2,
        .issue_use_count = 1,
        .flags = LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_CALL,
        .model_quality = LOOM_LOW_MODEL_QUALITY_FALLBACK,
    },
};

static const loom_low_descriptor_t kDescriptors[] = {
    VM_DESCRIPTOR(key_const_i32, mnemonic_const_i32, semantic_const_i32, 0, 1,
                  1, 0, 1, 0, 0, VM_SCHEDULE_CONST,
                  LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE),
    VM_DESCRIPTOR(key_add_i32, mnemonic_add_i32, semantic_add_i32, 1, 3, 1, 0,
                  0, 0, 0, VM_SCHEDULE_ALU_I32,
                  LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE),
    VM_DESCRIPTOR(key_sub_i32, mnemonic_sub_i32, semantic_sub_i32, 4, 3, 1, 0,
                  0, 0, 0, VM_SCHEDULE_ALU_I32,
                  LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE),
    VM_DESCRIPTOR(key_cmp_eq_i32, mnemonic_cmp_eq_i32, semantic_cmp_eq_i32, 7,
                  3, 1, 0, 0, 0, 0, VM_SCHEDULE_ALU_I32,
                  LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE),
    VM_DESCRIPTOR(key_br, mnemonic_br, semantic_branch, 0, 0, 0, 1, 1,
                  VM_EFFECT_CONTROL, 1, VM_SCHEDULE_CONTROL,
                  LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                      LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR),
    VM_DESCRIPTOR(key_cond_br_i32, mnemonic_cond_br_i32, semantic_cond_branch,
                  10, 1, 0, 2, 2, VM_EFFECT_CONTROL, 1, VM_SCHEDULE_CONTROL,
                  LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                      LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR),
    VM_DESCRIPTOR(key_call_import_i32, mnemonic_call_import_i32, semantic_call,
                  11, 2, 1, 4, 1, VM_EFFECT_CALL, 1, VM_SCHEDULE_CALL,
                  LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING),
    VM_DESCRIPTOR(key_return_i32, mnemonic_return_i32, semantic_return_i32, 13,
                  1, 0, 0, 0, VM_EFFECT_CONTROL, 1, VM_SCHEDULE_CONTROL,
                  LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                      LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR),
    VM_DESCRIPTOR(key_return_void, mnemonic_return_void, semantic_return_void,
                  0, 0, 0, 0, 0, VM_EFFECT_CONTROL, 1, VM_SCHEDULE_CONTROL,
                  LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                      LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR),
};

static const loom_low_descriptor_set_t kDescriptorSet = {
    .abi_version = LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION,
    .generator_version = 1,
    .fingerprint =
        {
            .low = UINT64_C(9708103317755296508),
            .high = UINT64_C(12545568297801738333),
        },
    .key_string_offset = VM_STRING_OFFSET(set_key),
    .target_key_string_offset = VM_STRING_OFFSET(target_key),
    .feature_key_string_offset = VM_STRING_OFFSET(feature_key),
    .string_data = (const char*)&kStrings,
    .string_data_length = sizeof(kStrings),
    .descriptors = kDescriptors,
    .descriptor_count = IREE_ARRAYSIZE(kDescriptors),
    .operands = kOperands,
    .operand_count = IREE_ARRAYSIZE(kOperands),
    .immediates = kImmediates,
    .immediate_count = IREE_ARRAYSIZE(kImmediates),
    .effects = kEffects,
    .effect_count = IREE_ARRAYSIZE(kEffects),
    .reg_classes = kRegClasses,
    .reg_class_count = IREE_ARRAYSIZE(kRegClasses),
    .reg_class_alts = kRegClassAlts,
    .reg_class_alt_count = IREE_ARRAYSIZE(kRegClassAlts),
    .schedule_classes = kScheduleClasses,
    .schedule_class_count = IREE_ARRAYSIZE(kScheduleClasses),
    .issue_uses = kIssueUses,
    .issue_use_count = IREE_ARRAYSIZE(kIssueUses),
    .resources = kResources,
    .resource_count = IREE_ARRAYSIZE(kResources),
};

const loom_low_descriptor_set_t* loom_ireevm_core_descriptor_set(void) {
  return &kDescriptorSet;
}
