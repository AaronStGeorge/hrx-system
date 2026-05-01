// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/test/contract_table.h"

#include "iree/testing/gtest.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"

namespace loom {
namespace {

const loom_target_contract_descriptor_rule_t* DescriptorRuleForCase(
    const loom_target_contract_table_t* table, loom_op_kind_t op_kind,
    uint16_t case_ordinal) {
  loom_target_contract_op_entry_t entry =
      loom_target_contract_table_lookup_kind(table, op_kind);
  EXPECT_FALSE(loom_target_contract_op_entry_is_empty(entry));
  EXPECT_LT(case_ordinal, entry.case_count);
  const loom_target_contract_case_t* contract_case =
      &table->cases[entry.case_start + case_ordinal];
  EXPECT_EQ(contract_case->system, LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE);
  EXPECT_NE(contract_case->row_index, LOOM_TARGET_CONTRACT_ROW_NONE);
  return &table->descriptor_rules[contract_case->row_index];
}

TEST(TestLowContractTableTest, GeneratedTableReferencesTestLowRules) {
  const loom_target_contract_table_t* table =
      &loom_test_low_core_contract_table;
  ASSERT_NE(table, nullptr);

  const loom_target_contract_descriptor_rule_t* scalar_addi =
      DescriptorRuleForCase(table, LOOM_OP_SCALAR_ADDI, 0);
  EXPECT_EQ(scalar_addi->rule_set_index, 0);
  EXPECT_EQ(scalar_addi->rule_index, 0);

  const loom_target_contract_descriptor_rule_t* vector_addi =
      DescriptorRuleForCase(table, LOOM_OP_VECTOR_ADDI, 0);
  EXPECT_EQ(vector_addi->rule_set_index, 0);
  EXPECT_EQ(vector_addi->rule_index, 15);

  loom_target_contract_op_entry_t reduce_entry =
      loom_target_contract_table_lookup_kind(table, LOOM_OP_VECTOR_REDUCE);
  EXPECT_EQ(reduce_entry.case_count, 2);
  EXPECT_EQ(DescriptorRuleForCase(table, LOOM_OP_VECTOR_REDUCE, 0)->rule_index,
            18);
  EXPECT_EQ(DescriptorRuleForCase(table, LOOM_OP_VECTOR_REDUCE, 1)->rule_index,
            19);

  const loom_target_contract_descriptor_rule_t* index_madd =
      DescriptorRuleForCase(table, LOOM_OP_INDEX_MADD, 0);
  EXPECT_EQ(index_madd->rule_set_index, 0);
  EXPECT_EQ(index_madd->rule_index, 21);
}

TEST(TestLowContractTableTest, GeneratedTableLeavesUncoveredOpsEmpty) {
  const loom_target_contract_table_t* table =
      &loom_test_low_core_contract_table;
  ASSERT_NE(table, nullptr);

  loom_target_contract_op_entry_t entry =
      loom_target_contract_table_lookup_kind(table, LOOM_OP_VECTOR_LOAD);
  EXPECT_TRUE(loom_target_contract_op_entry_is_empty(entry));
}

}  // namespace
}  // namespace loom
