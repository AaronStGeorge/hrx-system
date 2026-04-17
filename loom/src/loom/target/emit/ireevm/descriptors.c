// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/ireevm/descriptors.h"

static const uint8_t kStringData[] =
    "\x00"
    "\x0c"
    "iree.vm.core"
    "\x07"
    "iree.vm"
    "\x0a"
    "iree.vm.v1"
    "\x06"
    "vm.i32"
    "\x06"
    "vm.i64"
    "\x06"
    "vm.f32"
    "\x06"
    "vm.f64"
    "\x06"
    "vm.ref"
    "\x07"
    "vm.list"
    "\x03"
    "dst"
    "\x03"
    "lhs"
    "\x03"
    "rhs"
    "\x05"
    "value"
    "\x04"
    "cond"
    "\x04"
    "arg0"
    "\x09"
    "i32_value"
    "\x0c"
    "target_block"
    "\x0a"
    "true_block"
    "\x0b"
    "false_block"
    "\x0e"
    "callee_ordinal"
    "\x06"
    "vm.alu"
    "\x0a"
    "vm.control"
    "\x07"
    "vm.call"
    "\x08"
    "vm.const"
    "\x0a"
    "vm.alu.i32"
    "\x0a"
    "vm.control"
    "\x07"
    "vm.call"
    "\x11"
    "iree.vm.const.i32"
    "\x0f"
    "iree.vm.add.i32"
    "\x0f"
    "iree.vm.sub.i32"
    "\x12"
    "iree.vm.cmp.eq.i32"
    "\x0a"
    "iree.vm.br"
    "\x13"
    "iree.vm.cond_br.i32"
    "\x17"
    "iree.vm.call.import.i32"
    "\x12"
    "iree.vm.return.i32"
    "\x13"
    "iree.vm.return.void"
    "\x0c"
    "vm.const.i32"
    "\x0a"
    "vm.add.i32"
    "\x0a"
    "vm.sub.i32"
    "\x0d"
    "vm.cmp.eq.i32"
    "\x05"
    "vm.br"
    "\x0e"
    "vm.cond_br.i32"
    "\x12"
    "vm.call.import.i32"
    "\x0d"
    "vm.return.i32"
    "\x0e"
    "vm.return.void"
    "\x11"
    "integer.const.i32"
    "\x0f"
    "integer.add.i32"
    "\x0f"
    "integer.sub.i32"
    "\x12"
    "integer.cmp.eq.i32"
    "\x0e"
    "control.branch"
    "\x17"
    "control.cond_branch.i32"
    "\x0f"
    "call.import.i32"
    "\x12"
    "control.return.i32"
    "\x13"
    "control.return.void";

enum {
  VM_STRING_empty = 0,
  VM_STRING_set_key = VM_STRING_empty + sizeof(""),
  VM_STRING_target_key = VM_STRING_set_key + sizeof("iree.vm.core"),
  VM_STRING_feature_key = VM_STRING_target_key + sizeof("iree.vm"),
  VM_STRING_reg_i32 = VM_STRING_feature_key + sizeof("iree.vm.v1"),
  VM_STRING_reg_i64 = VM_STRING_reg_i32 + sizeof("vm.i32"),
  VM_STRING_reg_f32 = VM_STRING_reg_i64 + sizeof("vm.i64"),
  VM_STRING_reg_f64 = VM_STRING_reg_f32 + sizeof("vm.f32"),
  VM_STRING_reg_ref = VM_STRING_reg_f64 + sizeof("vm.f64"),
  VM_STRING_reg_list = VM_STRING_reg_ref + sizeof("vm.ref"),
  VM_STRING_field_dst = VM_STRING_reg_list + sizeof("vm.list"),
  VM_STRING_field_lhs = VM_STRING_field_dst + sizeof("dst"),
  VM_STRING_field_rhs = VM_STRING_field_lhs + sizeof("lhs"),
  VM_STRING_field_value = VM_STRING_field_rhs + sizeof("rhs"),
  VM_STRING_field_cond = VM_STRING_field_value + sizeof("value"),
  VM_STRING_field_arg0 = VM_STRING_field_cond + sizeof("cond"),
  VM_STRING_immediate_i32_value = VM_STRING_field_arg0 + sizeof("arg0"),
  VM_STRING_immediate_target_block =
      VM_STRING_immediate_i32_value + sizeof("i32_value"),
  VM_STRING_immediate_true_block =
      VM_STRING_immediate_target_block + sizeof("target_block"),
  VM_STRING_immediate_false_block =
      VM_STRING_immediate_true_block + sizeof("true_block"),
  VM_STRING_immediate_callee =
      VM_STRING_immediate_false_block + sizeof("false_block"),
  VM_STRING_resource_alu =
      VM_STRING_immediate_callee + sizeof("callee_ordinal"),
  VM_STRING_resource_control = VM_STRING_resource_alu + sizeof("vm.alu"),
  VM_STRING_resource_call = VM_STRING_resource_control + sizeof("vm.control"),
  VM_STRING_schedule_const = VM_STRING_resource_call + sizeof("vm.call"),
  VM_STRING_schedule_alu_i32 = VM_STRING_schedule_const + sizeof("vm.const"),
  VM_STRING_schedule_control =
      VM_STRING_schedule_alu_i32 + sizeof("vm.alu.i32"),
  VM_STRING_schedule_call = VM_STRING_schedule_control + sizeof("vm.control"),
  VM_STRING_key_const_i32 = VM_STRING_schedule_call + sizeof("vm.call"),
  VM_STRING_key_add_i32 = VM_STRING_key_const_i32 + sizeof("iree.vm.const.i32"),
  VM_STRING_key_sub_i32 = VM_STRING_key_add_i32 + sizeof("iree.vm.add.i32"),
  VM_STRING_key_cmp_eq_i32 = VM_STRING_key_sub_i32 + sizeof("iree.vm.sub.i32"),
  VM_STRING_key_br = VM_STRING_key_cmp_eq_i32 + sizeof("iree.vm.cmp.eq.i32"),
  VM_STRING_key_cond_br_i32 = VM_STRING_key_br + sizeof("iree.vm.br"),
  VM_STRING_key_call_import_i32 =
      VM_STRING_key_cond_br_i32 + sizeof("iree.vm.cond_br.i32"),
  VM_STRING_key_return_i32 =
      VM_STRING_key_call_import_i32 + sizeof("iree.vm.call.import.i32"),
  VM_STRING_key_return_void =
      VM_STRING_key_return_i32 + sizeof("iree.vm.return.i32"),
  VM_STRING_mnemonic_const_i32 =
      VM_STRING_key_return_void + sizeof("iree.vm.return.void"),
  VM_STRING_mnemonic_add_i32 =
      VM_STRING_mnemonic_const_i32 + sizeof("vm.const.i32"),
  VM_STRING_mnemonic_sub_i32 =
      VM_STRING_mnemonic_add_i32 + sizeof("vm.add.i32"),
  VM_STRING_mnemonic_cmp_eq_i32 =
      VM_STRING_mnemonic_sub_i32 + sizeof("vm.sub.i32"),
  VM_STRING_mnemonic_br =
      VM_STRING_mnemonic_cmp_eq_i32 + sizeof("vm.cmp.eq.i32"),
  VM_STRING_mnemonic_cond_br_i32 = VM_STRING_mnemonic_br + sizeof("vm.br"),
  VM_STRING_mnemonic_call_import_i32 =
      VM_STRING_mnemonic_cond_br_i32 + sizeof("vm.cond_br.i32"),
  VM_STRING_mnemonic_return_i32 =
      VM_STRING_mnemonic_call_import_i32 + sizeof("vm.call.import.i32"),
  VM_STRING_mnemonic_return_void =
      VM_STRING_mnemonic_return_i32 + sizeof("vm.return.i32"),
  VM_STRING_semantic_const_i32 =
      VM_STRING_mnemonic_return_void + sizeof("vm.return.void"),
  VM_STRING_semantic_add_i32 =
      VM_STRING_semantic_const_i32 + sizeof("integer.const.i32"),
  VM_STRING_semantic_sub_i32 =
      VM_STRING_semantic_add_i32 + sizeof("integer.add.i32"),
  VM_STRING_semantic_cmp_eq_i32 =
      VM_STRING_semantic_sub_i32 + sizeof("integer.sub.i32"),
  VM_STRING_semantic_branch =
      VM_STRING_semantic_cmp_eq_i32 + sizeof("integer.cmp.eq.i32"),
  VM_STRING_semantic_cond_branch =
      VM_STRING_semantic_branch + sizeof("control.branch"),
  VM_STRING_semantic_call =
      VM_STRING_semantic_cond_branch + sizeof("control.cond_branch.i32"),
  VM_STRING_semantic_return_i32 =
      VM_STRING_semantic_call + sizeof("call.import.i32"),
  VM_STRING_semantic_return_void =
      VM_STRING_semantic_return_i32 + sizeof("control.return.i32"),
  VM_STRING_END =
      VM_STRING_semantic_return_void + sizeof("control.return.void"),
};

static_assert(VM_STRING_END == sizeof(kStringData) - 1,
              "IREE VM descriptor string offsets must cover the table payload");

#define VM_STRING_OFFSET(field) ((loom_bstring_table_offset_t)VM_STRING_##field)

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
            .low = UINT64_C(3224395707371742904),
            .high = UINT64_C(187231647689077189),
        },
    .key_string_offset = VM_STRING_OFFSET(set_key),
    .target_key_string_offset = VM_STRING_OFFSET(target_key),
    .feature_key_string_offset = VM_STRING_OFFSET(feature_key),
    .string_table =
        {
            .data = kStringData,
            .data_length = sizeof(kStringData) - 1,
        },
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
