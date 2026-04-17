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
    LOOM_BSTRING_LITERAL("\x0f", "integer.add.i32");
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
  TEST_STRING_END = TEST_STRING_semantic_add + sizeof("integer.add.i32"),
};

static_assert(TEST_STRING_END == sizeof(kTestStrings) - 1,
              "test descriptor string offsets must cover the table payload");

#define TEST_STRING_OFFSET(field) \
  static_cast<loom_bstring_table_offset_t>(TEST_STRING_##field)

struct TestTables {
  loom_low_descriptor_t descriptors[2];
  loom_low_operand_t operands[4];
  loom_low_immediate_t immediates[1];
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
  tables->immediates[0].signed_min = INT32_MIN;
  tables->immediates[0].unsigned_max = INT32_MAX;

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

  tables->set.abi_version = LOOM_LOW_DESCRIPTOR_SET_ABI_VERSION;
  tables->set.generator_version = 7;
  tables->set.key_string_offset = TEST_STRING_OFFSET(set_key);
  tables->set.target_key_string_offset = TEST_STRING_OFFSET(target_key);
  tables->set.feature_key_string_offset = TEST_STRING_OFFSET(feature_key);
  tables->set.string_table.data = kTestStrings;
  tables->set.string_table.data_length = sizeof(kTestStrings) - 1;
  tables->set.descriptors = tables->descriptors;
  tables->set.descriptor_count = IREE_ARRAYSIZE(tables->descriptors);
  tables->set.operands = tables->operands;
  tables->set.operand_count = IREE_ARRAYSIZE(tables->operands);
  tables->set.immediates = tables->immediates;
  tables->set.immediate_count = IREE_ARRAYSIZE(tables->immediates);
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
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_NOT_FOUND);
  EXPECT_EQ(descriptor_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  iree_status_ignore(status);
}

TEST(LowDescriptorsTest, RejectsMalformedSpans) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.descriptors[1].operand_count = 4;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_OUT_OF_RANGE);
  iree_status_ignore(status);
}

TEST(LowDescriptorsTest, RejectsDuplicateKeys) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.descriptors[1].key_string_offset =
      tables.descriptors[0].key_string_offset;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(LowDescriptorsTest, RejectsUnknownScheduleModelData) {
  TestTables tables;
  InitializeTestTables(&tables);
  tables.schedule_classes[1].model_quality = LOOM_LOW_MODEL_QUALITY_UNKNOWN;

  iree_status_t status = loom_low_descriptor_set_verify(&tables.set);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
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
