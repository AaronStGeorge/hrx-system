// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/contract.h"

#include <cstdint>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "loom/ir/ir.h"

namespace {

constexpr uint8_t kTestDialectId = 7;
constexpr uint8_t kLegalOpIndex = 3;

const loom_target_contract_op_entry_t kOpEntries[] = {
    {LOOM_TARGET_CONTRACT_ROW_NONE, 0},
    {LOOM_TARGET_CONTRACT_ROW_NONE, 0},
    {LOOM_TARGET_CONTRACT_ROW_NONE, 0},
    {0, 1},
};

const loom_target_contract_dialect_table_t kDialectTables[] = {
    {
        0,
        nullptr,
    },
    {
        IREE_ARRAYSIZE(kOpEntries),
        kOpEntries,
    },
};

const loom_target_contract_case_t kCases[] = {
    {
        LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE,
        0,
        0,
    },
};

const loom_target_contract_descriptor_rule_t kDescriptorRules[] = {
    {
        0,
    },
};

const loom_target_contract_table_t kContractTable = {
    11,
    kTestDialectId - 1,
    IREE_ARRAYSIZE(kDialectTables),
    kDialectTables,
    IREE_ARRAYSIZE(kCases),
    kCases,
    IREE_ARRAYSIZE(kDescriptorRules),
    kDescriptorRules,
};

TEST(TargetContractTableTest, LookupKindSelectsDescriptorRuleCase) {
  loom_target_contract_op_entry_t entry =
      loom_target_contract_table_lookup_kind(
          &kContractTable, LOOM_OP_KIND(kTestDialectId, kLegalOpIndex));

  ASSERT_FALSE(loom_target_contract_op_entry_is_empty(entry));
  EXPECT_EQ(entry.case_start, 0);
  EXPECT_EQ(entry.case_count, 1);
  const loom_target_contract_case_t* contract_case =
      &kContractTable.cases[entry.case_start];
  EXPECT_EQ(contract_case->system, LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE);
  ASSERT_NE(contract_case->row_index, LOOM_TARGET_CONTRACT_ROW_NONE);
  const loom_target_contract_descriptor_rule_t* descriptor_rule =
      &kContractTable.descriptor_rules[contract_case->row_index];
  EXPECT_EQ(descriptor_rule->rule_set_index, 0);
  EXPECT_EQ(descriptor_rule->rule_index, 0);
}

TEST(TargetContractTableTest, LookupKindIgnoresUncoveredDialectSlot) {
  loom_target_contract_op_entry_t entry =
      loom_target_contract_table_lookup_kind(
          &kContractTable, LOOM_OP_KIND(kTestDialectId - 1, 0));

  EXPECT_TRUE(loom_target_contract_op_entry_is_empty(entry));
}

TEST(TargetContractTableTest, LookupKindIgnoresUncoveredOps) {
  loom_target_contract_op_entry_t entry =
      loom_target_contract_table_lookup_kind(&kContractTable,
                                             LOOM_OP_KIND(kTestDialectId, 1));

  EXPECT_TRUE(loom_target_contract_op_entry_is_empty(entry));
}

}  // namespace
