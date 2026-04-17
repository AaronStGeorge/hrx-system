// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptors.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

// clang-format off
static const uint8_t kTestStrings[] =
    LOOM_BSTRING_LITERAL("\x00", "")
    LOOM_BSTRING_LITERAL("\x09", "test.core")
    LOOM_BSTRING_LITERAL("\x0b", "test.target")
    LOOM_BSTRING_LITERAL("\x0d", "test.features")
    LOOM_BSTRING_LITERAL("\x08", "test.gpr")
    LOOM_BSTRING_LITERAL("\x03", "dst")
    LOOM_BSTRING_LITERAL("\x03", "lhs")
    LOOM_BSTRING_LITERAL("\x03", "rhs")
    LOOM_BSTRING_LITERAL("\x05", "value")
    LOOM_BSTRING_LITERAL("\x08", "test.alu")
    LOOM_BSTRING_LITERAL("\x0a", "test.const")
    LOOM_BSTRING_LITERAL("\x0c", "test.alu.i32")
    LOOM_BSTRING_LITERAL("\x0e", "test.const.i32")
    LOOM_BSTRING_LITERAL("\x0c", "test.add.i32")
    LOOM_BSTRING_LITERAL("\x09", "const.i32")
    LOOM_BSTRING_LITERAL("\x07", "add.i32")
    LOOM_BSTRING_LITERAL("\x11", "integer.const.i32")
    LOOM_BSTRING_LITERAL("\x0f", "integer.add.i32")
    LOOM_BSTRING_LITERAL("\x04", "mode")
    LOOM_BSTRING_LITERAL("\x09", "test.mode")
    LOOM_BSTRING_LITERAL("\x04", "fast")
    LOOM_BSTRING_LITERAL("\x04", "slow");
// clang-format on

enum {
  TEST_STRING_empty = 0,
  TEST_STRING_set_key = TEST_STRING_empty + sizeof(""),
  TEST_STRING_target_key = TEST_STRING_set_key + sizeof("test.core"),
  TEST_STRING_feature_key = TEST_STRING_target_key + sizeof("test.target"),
  TEST_STRING_reg_gpr = TEST_STRING_feature_key + sizeof("test.features"),
  TEST_STRING_field_dst = TEST_STRING_reg_gpr + sizeof("test.gpr"),
  TEST_STRING_field_lhs = TEST_STRING_field_dst + sizeof("dst"),
  TEST_STRING_field_rhs = TEST_STRING_field_lhs + sizeof("lhs"),
  TEST_STRING_field_value = TEST_STRING_field_rhs + sizeof("rhs"),
  TEST_STRING_resource_alu = TEST_STRING_field_value + sizeof("value"),
  TEST_STRING_schedule_const = TEST_STRING_resource_alu + sizeof("test.alu"),
  TEST_STRING_schedule_alu = TEST_STRING_schedule_const + sizeof("test.const"),
  TEST_STRING_descriptor_const =
      TEST_STRING_schedule_alu + sizeof("test.alu.i32"),
  TEST_STRING_descriptor_add =
      TEST_STRING_descriptor_const + sizeof("test.const.i32"),
  TEST_STRING_mnemonic_const =
      TEST_STRING_descriptor_add + sizeof("test.add.i32"),
  TEST_STRING_mnemonic_add = TEST_STRING_mnemonic_const + sizeof("const.i32"),
  TEST_STRING_semantic_const = TEST_STRING_mnemonic_add + sizeof("add.i32"),
  TEST_STRING_semantic_add =
      TEST_STRING_semantic_const + sizeof("integer.const.i32"),
  TEST_STRING_field_mode = TEST_STRING_semantic_add + sizeof("integer.add.i32"),
  TEST_STRING_enum_mode = TEST_STRING_field_mode + sizeof("mode"),
  TEST_STRING_enum_fast = TEST_STRING_enum_mode + sizeof("test.mode"),
  TEST_STRING_enum_slow = TEST_STRING_enum_fast + sizeof("fast"),
  TEST_STRING_END = TEST_STRING_enum_slow + sizeof("slow"),
};

static_assert(TEST_STRING_END == sizeof(kTestStrings) - 1,
              "test descriptor string offsets must cover the table payload");

#define TEST_STRING_OFFSET(field) \
  static_cast<loom_bstring_table_offset_t>(TEST_STRING_##field)

struct TestTables {
  loom_low_descriptor_t descriptors[2];
  loom_low_descriptor_ref_t descriptor_refs[2];
  loom_low_operand_t operands[4];
  loom_low_immediate_t immediates[1];
  loom_low_enum_domain_t enum_domains[1];
  loom_low_enum_value_t enum_values[2];
  loom_low_effect_t effects[1];
  loom_low_constraint_t constraints[1];
  loom_low_reg_class_t reg_classes[1];
  loom_low_reg_class_alt_t reg_class_alts[1];
  loom_low_schedule_class_t schedule_classes[2];
  loom_low_issue_use_t issue_uses[1];
  loom_low_resource_t resources[1];
  uint64_t feature_mask_words[1];
  loom_low_descriptor_set_t set;
};

void InitializeTestTables(TestTables* tables) {
  *tables = {};
  tables->reg_classes[0].name_string_offset = TEST_STRING_OFFSET(reg_gpr);
  tables->reg_classes[0].flags = LOOM_LOW_REG_CLASS_FLAG_VIRTUAL_ONLY;
  tables->reg_classes[0].alloc_unit_bits = 32;
  tables->reg_classes[0].spill_class_id = LOOM_LOW_REG_CLASS_NONE;

  tables->reg_class_alts[0].reg_class_id = 0;
  tables->reg_class_alts[0].flags = LOOM_LOW_REG_CLASS_ALT_FLAG_PREFERRED;

  tables->operands[0].field_name_string_offset = TEST_STRING_OFFSET(field_dst);
  tables->operands[0].role = LOOM_LOW_OPERAND_ROLE_RESULT;
  tables->operands[0].reg_class_alt_start = 0;
  tables->operands[0].reg_class_alt_count = 1;
  tables->operands[0].unit_count = 1;

  tables->operands[1].field_name_string_offset = TEST_STRING_OFFSET(field_dst);
  tables->operands[1].role = LOOM_LOW_OPERAND_ROLE_RESULT;
  tables->operands[1].reg_class_alt_start = 0;
  tables->operands[1].reg_class_alt_count = 1;
  tables->operands[1].unit_count = 1;

  tables->operands[2].field_name_string_offset = TEST_STRING_OFFSET(field_lhs);
  tables->operands[2].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
  tables->operands[2].reg_class_alt_start = 0;
  tables->operands[2].reg_class_alt_count = 1;
  tables->operands[2].unit_count = 1;
  tables->operands[2].read_stage = 0;

  tables->operands[3].field_name_string_offset = TEST_STRING_OFFSET(field_rhs);
  tables->operands[3].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
  tables->operands[3].reg_class_alt_start = 0;
  tables->operands[3].reg_class_alt_count = 1;
  tables->operands[3].unit_count = 1;
  tables->operands[3].read_stage = 0;

  tables->immediates[0].field_name_string_offset =
      TEST_STRING_OFFSET(field_value);
  tables->immediates[0].kind = LOOM_LOW_IMMEDIATE_KIND_SIGNED;
  tables->immediates[0].bit_width = 32;
  tables->immediates[0].enum_domain_id = LOOM_LOW_ENUM_DOMAIN_NONE;
  tables->immediates[0].signed_min = INT32_MIN;
  tables->immediates[0].unsigned_max = INT32_MAX;

  tables->enum_domains[0].name_string_offset = TEST_STRING_OFFSET(enum_mode);
  tables->enum_domains[0].value_start = 0;
  tables->enum_domains[0].value_count = 2;

  tables->enum_values[0].token_string_offset = TEST_STRING_OFFSET(enum_fast);
  tables->enum_values[0].value = 0;
  tables->enum_values[1].token_string_offset = TEST_STRING_OFFSET(enum_slow);
  tables->enum_values[1].value = 1;

  tables->resources[0].name_string_offset = TEST_STRING_OFFSET(resource_alu);
  tables->resources[0].capacity_per_cycle = 1;
  tables->resources[0].kind = LOOM_LOW_RESOURCE_KIND_SCALAR_ALU;

  tables->issue_uses[0].resource_id = 0;
  tables->issue_uses[0].cycles = 1;
  tables->issue_uses[0].units = 1;

  tables->schedule_classes[0].name_string_offset =
      TEST_STRING_OFFSET(schedule_const);
  tables->schedule_classes[0].latency_kind = LOOM_LOW_LATENCY_KIND_EXACT;
  tables->schedule_classes[0].model_quality = LOOM_LOW_MODEL_QUALITY_EXACT;

  tables->schedule_classes[1].name_string_offset =
      TEST_STRING_OFFSET(schedule_alu);
  tables->schedule_classes[1].latency_cycles = 1;
  tables->schedule_classes[1].latency_kind = LOOM_LOW_LATENCY_KIND_EXACT;
  tables->schedule_classes[1].issue_use_start = 0;
  tables->schedule_classes[1].issue_use_count = 1;
  tables->schedule_classes[1].model_quality = LOOM_LOW_MODEL_QUALITY_EXACT;

  tables->feature_mask_words[0] = UINT64_C(0x5);

  tables->descriptors[0].key_string_offset =
      TEST_STRING_OFFSET(descriptor_const);
  tables->descriptors[0].mnemonic_string_offset =
      TEST_STRING_OFFSET(mnemonic_const);
  tables->descriptors[0].semantic_tag_string_offset =
      TEST_STRING_OFFSET(semantic_const);
  tables->descriptors[0].operand_start = 0;
  tables->descriptors[0].operand_count = 1;
  tables->descriptors[0].result_count = 1;
  tables->descriptors[0].immediate_start = 0;
  tables->descriptors[0].immediate_count = 1;
  tables->descriptors[0].schedule_class_id = 0;
  tables->descriptors[0].flags = LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE;

  tables->descriptors[1].key_string_offset = TEST_STRING_OFFSET(descriptor_add);
  tables->descriptors[1].mnemonic_string_offset =
      TEST_STRING_OFFSET(mnemonic_add);
  tables->descriptors[1].semantic_tag_string_offset =
      TEST_STRING_OFFSET(semantic_add);
  tables->descriptors[1].feature_mask_word_start = 0;
  tables->descriptors[1].feature_mask_word_count = 1;
  tables->descriptors[1].operand_start = 1;
  tables->descriptors[1].operand_count = 3;
  tables->descriptors[1].result_count = 1;
  tables->descriptors[1].schedule_class_id = 1;
  tables->descriptors[1].flags = LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE;

  tables->descriptor_refs[0].key_string_offset =
      TEST_STRING_OFFSET(descriptor_add);
  tables->descriptor_refs[0].descriptor_ordinal = 1;
  tables->descriptor_refs[1].key_string_offset =
      TEST_STRING_OFFSET(descriptor_const);
  tables->descriptor_refs[1].descriptor_ordinal = 0;

  tables->set.abi_version = LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION;
  tables->set.generator_version = 7;
  tables->set.key_string_offset = TEST_STRING_OFFSET(set_key);
  tables->set.target_key_string_offset = TEST_STRING_OFFSET(target_key);
  tables->set.feature_key_string_offset = TEST_STRING_OFFSET(feature_key);
  tables->set.string_table.data = kTestStrings;
  tables->set.string_table.data_length = sizeof(kTestStrings) - 1;
  tables->set.descriptors = tables->descriptors;
  tables->set.descriptor_count = IREE_ARRAYSIZE(tables->descriptors);
  tables->set.descriptor_refs = tables->descriptor_refs;
  tables->set.descriptor_ref_count = IREE_ARRAYSIZE(tables->descriptor_refs);
  tables->set.operands = tables->operands;
  tables->set.operand_count = IREE_ARRAYSIZE(tables->operands);
  tables->set.immediates = tables->immediates;
  tables->set.immediate_count = IREE_ARRAYSIZE(tables->immediates);
  tables->set.enum_domains = tables->enum_domains;
  tables->set.enum_values = tables->enum_values;
  tables->set.effects = tables->effects;
  tables->set.constraints = tables->constraints;
  tables->set.reg_classes = tables->reg_classes;
  tables->set.reg_class_count = IREE_ARRAYSIZE(tables->reg_classes);
  tables->set.reg_class_alts = tables->reg_class_alts;
  tables->set.reg_class_alt_count = IREE_ARRAYSIZE(tables->reg_class_alts);
  tables->set.schedule_classes = tables->schedule_classes;
  tables->set.schedule_class_count = IREE_ARRAYSIZE(tables->schedule_classes);
  tables->set.issue_uses = tables->issue_uses;
  tables->set.issue_use_count = IREE_ARRAYSIZE(tables->issue_uses);
  tables->set.resources = tables->resources;
  tables->set.resource_count = IREE_ARRAYSIZE(tables->resources);
  tables->set.feature_mask_words = tables->feature_mask_words;
  tables->set.feature_mask_word_count =
      IREE_ARRAYSIZE(tables->feature_mask_words);
}

void AddAddDescriptorConstraint(TestTables* tables,
                                loom_low_constraint_kind_t kind,
                                uint16_t lhs_operand_index,
                                uint16_t rhs_operand_index) {
  tables->constraints[0].kind = kind;
  tables->constraints[0].lhs_operand_index = lhs_operand_index;
  tables->constraints[0].rhs_operand_index = rhs_operand_index;
  tables->descriptors[1].constraint_start = 0;
  tables->descriptors[1].constraint_count = 1;
  tables->set.constraint_count = 1;
}

void AddAddDescriptorEffect(TestTables* tables, loom_low_effect_kind_t kind,
                            loom_low_memory_space_t memory_space) {
  tables->effects[0].kind = kind;
  tables->effects[0].memory_space = memory_space;
  tables->descriptors[1].effect_start = 0;
  tables->descriptors[1].effect_count = 1;
  tables->set.effect_count = 1;
}

TEST(LowDescriptorsTest, VerifiesAndLooksUpDescriptors) {
  TestTables tables;
  InitializeTestTables(&tables);
  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));

  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  IREE_ASSERT_OK(loom_low_descriptor_set_lookup_descriptor(
      &tables.set, IREE_SV("test.add.i32"), &descriptor_ordinal));
  EXPECT_EQ(descriptor_ordinal, 1u);

  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(&tables.set, descriptor_ordinal);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->operand_count, 3u);
  EXPECT_EQ(descriptor->result_count, 1u);

  iree_status_t status = loom_low_descriptor_set_lookup_descriptor(
      &tables.set, IREE_SV("test.missing"), &descriptor_ordinal);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
}

TEST(LowDescriptorsTest, RejectsMalformedSpans) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.descriptors[1].operand_count = 4;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, status);
}

TEST(LowDescriptorsTest, RejectsNonResultRoleInResultPrefix) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.operands[1].role = LOOM_LOW_OPERAND_ROLE_OPERAND;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsResultRoleAfterResultPrefix) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.operands[2].role = LOOM_LOW_OPERAND_ROLE_RESULT;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsOperandResultRows) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.operands[2].role = LOOM_LOW_OPERAND_ROLE_OPERAND_RESULT;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsImplicitRowsWithoutImplicitFlag) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.operands[2].role = LOOM_LOW_OPERAND_ROLE_IMPLICIT;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, AcceptsImplicitRowsWithImplicitFlag) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.operands[2].role = LOOM_LOW_OPERAND_ROLE_IMPLICIT;
  tables.operands[2].flags = LOOM_LOW_OPERAND_FLAG_IMPLICIT;

  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
}

TEST(LowDescriptorsTest, AcceptsTiedResultOperandConstraint) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_TIED, 0, 1);

  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
}

TEST(LowDescriptorsTest, AcceptsCommutableOperandConstraint) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_COMMUTABLE, 1,
                             2);

  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
}

TEST(LowDescriptorsTest, AcceptsDestructiveResultOperandConstraint) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE, 0,
                             1);

  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
}

TEST(LowDescriptorsTest, AcceptsEarlyClobberResultConstraint) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER, 0,
                             LOOM_LOW_ID_NONE);

  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
}

TEST(LowDescriptorsTest, AcceptsFoldableResultConstraint) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_FOLDABLE, 0,
                             LOOM_LOW_ID_NONE);

  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
}

TEST(LowDescriptorsTest, AcceptsRematerializableResultConstraint) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_REMATERIALIZABLE,
                             0, LOOM_LOW_ID_NONE);

  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
}

TEST(LowDescriptorsTest, RejectsTiedConstraintWithoutRhs) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_TIED, 0,
                             LOOM_LOW_ID_NONE);

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsTiedConstraintWithoutResultLhs) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_TIED, 1, 2);

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsCommutableConstraintOnResult) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_COMMUTABLE, 0,
                             1);

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsDestructiveConstraintWithoutResultLhs) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_DESTRUCTIVE, 1,
                             2);

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsEarlyClobberConstraintOnOperand) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_EARLY_CLOBBER, 1,
                             LOOM_LOW_ID_NONE);

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsFoldableConstraintOnOperand) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorConstraint(&tables, LOOM_LOW_CONSTRAINT_KIND_FOLDABLE, 1,
                             LOOM_LOW_ID_NONE);

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, AcceptsDeadRemovableReadEffect) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorEffect(&tables, LOOM_LOW_EFFECT_KIND_READ,
                         LOOM_LOW_MEMORY_SPACE_GENERIC);
  tables.schedule_classes[1].flags = LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD;

  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
}

TEST(LowDescriptorsTest, AcceptsSideEffectingReadEffect) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorEffect(&tables, LOOM_LOW_EFFECT_KIND_READ,
                         LOOM_LOW_MEMORY_SPACE_GENERIC);
  tables.descriptors[1].flags = LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING;
  tables.schedule_classes[1].flags = LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD;

  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
}

TEST(LowDescriptorsTest, RejectsWriteEffectWithoutSideEffectingFlag) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorEffect(&tables, LOOM_LOW_EFFECT_KIND_WRITE,
                         LOOM_LOW_MEMORY_SPACE_GENERIC);
  tables.schedule_classes[1].flags = LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_STORE;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsSideEffectingWithoutEffects) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.descriptors[1].flags = LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsDeadRemovableSideEffectingDescriptor) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorEffect(&tables, LOOM_LOW_EFFECT_KIND_READ,
                         LOOM_LOW_MEMORY_SPACE_GENERIC);
  tables.descriptors[1].flags = LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                                LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE;
  tables.schedule_classes[1].flags = LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsReadEffectWithoutMemorySpace) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorEffect(&tables, LOOM_LOW_EFFECT_KIND_READ,
                         LOOM_LOW_MEMORY_SPACE_NONE);
  tables.descriptors[1].flags = LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING;
  tables.schedule_classes[1].flags = LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_LOAD;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsTerminatorWithoutControlEffect) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorEffect(&tables, LOOM_LOW_EFFECT_KIND_CALL,
                         LOOM_LOW_MEMORY_SPACE_NONE);
  tables.descriptors[1].flags = LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                                LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR;
  tables.schedule_classes[1].flags = LOOM_LOW_SCHEDULE_CLASS_FLAG_MAY_CALL;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsControlEffectWithoutTerminator) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorEffect(&tables, LOOM_LOW_EFFECT_KIND_CONTROL,
                         LOOM_LOW_MEMORY_SPACE_NONE);
  tables.descriptors[1].flags = LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING;
  tables.schedule_classes[1].flags = LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsScheduleClassMissingEffectFlag) {
  TestTables tables;
  InitializeTestTables(&tables);
  AddAddDescriptorEffect(&tables, LOOM_LOW_EFFECT_KIND_READ,
                         LOOM_LOW_MEMORY_SPACE_GENERIC);
  tables.descriptors[1].flags = LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsDuplicateKeys) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.descriptors[1].key_string_offset =
      tables.descriptors[0].key_string_offset;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsUnsortedDescriptorReferences) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.descriptor_refs[0].key_string_offset =
      TEST_STRING_OFFSET(descriptor_const);
  tables.descriptor_refs[0].descriptor_ordinal = 0;
  tables.descriptor_refs[1].key_string_offset =
      TEST_STRING_OFFSET(descriptor_add);
  tables.descriptor_refs[1].descriptor_ordinal = 1;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, LookupRejectsIncompleteDescriptorReferences) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.set.descriptor_ref_count = 0;

  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  iree_status_t status = loom_low_descriptor_set_lookup_descriptor(
      &tables.set, IREE_SV("test.add.i32"), &descriptor_ordinal);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
}

TEST(LowDescriptorsTest, RejectsUnknownScheduleModelData) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.schedule_classes[1].model_quality = LOOM_LOW_MODEL_QUALITY_UNKNOWN;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, AcceptsEnumImmediateDomain) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.immediates[0].field_name_string_offset =
      TEST_STRING_OFFSET(field_mode);
  tables.immediates[0].kind = LOOM_LOW_IMMEDIATE_KIND_ENUM;
  tables.immediates[0].enum_domain_id = 0;
  tables.set.enum_domain_count = 1;
  tables.set.enum_value_count = 2;

  IREE_ASSERT_OK(loom_low_descriptor_set_verify(&tables.set));
}

TEST(LowDescriptorsTest, RejectsEnumImmediateWithoutDomain) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.immediates[0].kind = LOOM_LOW_IMMEDIATE_KIND_ENUM;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsEnumImmediateDomainOutOfRange) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.immediates[0].kind = LOOM_LOW_IMMEDIATE_KIND_ENUM;
  tables.immediates[0].enum_domain_id = 1;
  tables.set.enum_domain_count = 1;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, status);
}

TEST(LowDescriptorsTest, RejectsNonEnumImmediateWithDomain) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.immediates[0].enum_domain_id = 0;
  tables.set.enum_domain_count = 1;
  tables.set.enum_value_count = 2;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsEnumDomainWithoutValues) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.enum_domains[0].value_count = 0;
  tables.set.enum_domain_count = 1;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsUnsortedEnumDomainValues) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.enum_values[0].token_string_offset = TEST_STRING_OFFSET(enum_slow);
  tables.enum_values[1].token_string_offset = TEST_STRING_OFFSET(enum_fast);
  tables.set.enum_domain_count = 1;
  tables.set.enum_value_count = 2;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsDescriptorWithoutScheduleClass) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.descriptors[1].schedule_class_id = LOOM_LOW_SCHEDULE_CLASS_NONE;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsUnknownLatencyKind) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.schedule_classes[1].latency_kind = LOOM_LOW_LATENCY_KIND_UNKNOWN;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsInvalidLatencyKind) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.schedule_classes[1].latency_kind =
      static_cast<loom_low_latency_kind_t>(99);

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsNegativeLatencyKind) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.schedule_classes[1].latency_kind =
      static_cast<loom_low_latency_kind_t>(-1);

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsInvalidScheduleModelQuality) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.schedule_classes[1].model_quality =
      static_cast<loom_low_model_quality_t>(99);

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsNegativeScheduleModelQuality) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.schedule_classes[1].model_quality =
      static_cast<loom_low_model_quality_t>(-1);

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsExactModelQualityWithoutExactLatency) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.schedule_classes[1].latency_kind = LOOM_LOW_LATENCY_KIND_ESTIMATE;
  tables.schedule_classes[1].model_quality = LOOM_LOW_MODEL_QUALITY_EXACT;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsFallbackModelWithoutVariableLatency) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.schedule_classes[1].latency_kind = LOOM_LOW_LATENCY_KIND_ESTIMATE;
  tables.schedule_classes[1].model_quality = LOOM_LOW_MODEL_QUALITY_FALLBACK;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsZeroIssueUseCycles) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.issue_uses[0].cycles = 0;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, RejectsZeroIssueUseUnits) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.issue_uses[0].units = 0;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(LowDescriptorsTest, FormatsManifestJson) {
  TestTables tables;
  InitializeTestTables(&tables);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(
      loom_low_descriptor_set_format_manifest_json(&tables.set, &builder));
  std::string json(iree_string_builder_buffer(&builder),
                   iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);

  EXPECT_NE(json.find("\"key\":\"test.core\""), std::string::npos);
  EXPECT_NE(json.find("\"key\":\"test.add.i32\""), std::string::npos);
  EXPECT_NE(json.find("\"schedule_class\":1"), std::string::npos);
}

}  // namespace
}  // namespace loom
