// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/contract_query.h"

#include <cstdint>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/attribute.h"
#include "loom/ir/ir.h"

namespace {

constexpr loom_op_kind_t kSourceOpKind = LOOM_OP_KIND(7, 3);
constexpr uint64_t kDescriptorId = UINT64_C(0x123456789abcdef0);

const loom_low_descriptor_t kDescriptor = {
    /*.key_string_offset=*/0,
    /*.stable_id=*/kDescriptorId,
};

iree_status_t ResolveTestDescriptorRef(
    void* user_data, const loom_low_lower_rule_match_context_t* context,
    const loom_low_lower_rule_set_t* rule_set,
    loom_low_lower_descriptor_ref_t descriptor_ref,
    const loom_low_descriptor_t** out_descriptor) {
  (void)user_data;
  (void)context;
  (void)rule_set;
  *out_descriptor = descriptor_ref == 0 ? &kDescriptor : nullptr;
  return iree_ok_status();
}

const loom_target_config_t kTargetConfig = {
    /*.name=*/IREE_SV("test-config"),
    /*.contract_set_key=*/{},
    /*.contract_feature_bits=*/0,
};

const loom_target_export_plan_t kTargetExportPlan = {
    /*.name=*/IREE_SV("test-export"),
};

const loom_target_bundle_t kTargetBundle = {
    /*.name=*/IREE_SV("test-target"),
    /*.snapshot=*/nullptr,
    /*.export_plan=*/&kTargetExportPlan,
    /*.config=*/&kTargetConfig,
};

loom_target_contract_fragment_t MakeContractFragment(
    loom_target_contract_op_entry_t* op_entries,
    loom_target_contract_dialect_table_t* dialects,
    loom_target_contract_fragment_case_t* cases,
    loom_target_contract_descriptor_rule_t* descriptor_rules) {
  op_entries[loom_op_dialect_index(kSourceOpKind)].case_start = 0;
  op_entries[loom_op_dialect_index(kSourceOpKind)].case_count = 1;
  dialects[0].op_count = 4;
  dialects[0].op_entries = op_entries;
  cases[0].system = LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE;
  cases[0].row_index = 0;
  descriptor_rules[0].rule_index = 0;

  loom_target_contract_fragment_t fragment = {};
  fragment.dialect_base_id = loom_op_dialect_id(kSourceOpKind);
  fragment.dialect_count = 1;
  fragment.flags = LOOM_TARGET_CONTRACT_FRAGMENT_FLAG_TARGET_QUERY;
  fragment.dialects = dialects;
  fragment.case_count = 1;
  fragment.cases = cases;
  fragment.descriptor_rule_count = 1;
  fragment.descriptor_rules = descriptor_rules;
  return fragment;
}

TEST(LowContractQueryTest, ContractIndexDescriptorRuleSelectsLegalCase) {
  loom_low_lower_rule_descriptor_ref_t descriptor_ref = {
      /*.key=*/IREE_SV("test.descriptor"),
  };
  loom_low_lower_emit_t emit = {};
  emit.kind = LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP;
  emit.descriptor_ref = 0;
  loom_low_lower_rule_t rule = {};
  rule.source_op_kind = kSourceOpKind;
  rule.emit_count = 1;
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.rules = &rule;
  rule_set.rule_count = 1;
  rule_set.descriptor_refs = &descriptor_ref;
  rule_set.descriptor_ref_count = 1;
  rule_set.emits = &emit;
  rule_set.emit_count = 1;
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};

  loom_target_contract_op_entry_t op_entries[4] = {};
  loom_target_contract_dialect_table_t dialects[1] = {};
  loom_target_contract_fragment_case_t cases[1] = {};
  loom_target_contract_descriptor_rule_t descriptor_rules[1] = {};
  loom_target_contract_fragment_t fragment =
      MakeContractFragment(op_entries, dialects, cases, descriptor_rules);
  const loom_target_contract_binding_t bindings[] = {
      {
          &fragment,
          0,
      },
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_target_contract_index_t index = {};
  IREE_ASSERT_OK(loom_target_contract_index_compose(
      bindings, IREE_ARRAYSIZE(bindings), &index, &arena));

  const loom_low_lower_contract_query_options_t options = {
      /*.contract_index=*/&index,
      /*.rule_sets=*/
      {
          /*.count=*/IREE_ARRAYSIZE(rule_sets),
          /*.values=*/rule_sets,
      },
      /*.map_value=*/{},
      /*.register_class=*/{},
      /*.can_materialize=*/{},
      /*.descriptor_ref=*/
      {
          /*.fn=*/ResolveTestDescriptorRef,
          /*.user_data=*/nullptr,
      },
  };
  const loom_target_contract_query_environment_t environment = {
      /*.module=*/nullptr,
      /*.function=*/{},
      /*.bundle=*/&kTargetBundle,
      /*.target_ref=*/{},
      /*.descriptor_set=*/nullptr,
      /*.fact_table=*/nullptr,
      /*.view_regions=*/nullptr,
      /*.arena=*/nullptr,
  };
  loom_op_t op = {};
  op.kind = kSourceOpKind;
  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();
  IREE_ASSERT_OK(loom_low_lower_query_target_contract(&environment, &options,
                                                      &op, &result));

  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_LEGAL);
  EXPECT_EQ(result.binding_index, 0);
  EXPECT_EQ(result.case_index, 0);
  EXPECT_EQ(result.rule_set_index, 0);
  EXPECT_EQ(result.rule_index, 0);
  EXPECT_EQ(result.selected_descriptor, &kDescriptor);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(LowContractQueryTest, ContractIndexDescriptorRuleReportsRejectedCase) {
  loom_low_lower_guard_t guard = {};
  guard.kind = LOOM_LOW_LOWER_GUARD_ATTR_KIND;
  guard.attr_kind = LOOM_ATTR_I64;
  guard.diagnostic_index = 0;
  loom_low_lower_diagnostic_param_t diagnostic_params[] = {
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_TARGET_KEY,
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_EXPORT_NAME,
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_CONFIG_KEY,
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_FUNCTION_NAME,
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
          /*.string_value=*/IREE_SV("test.source"),
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
          /*.string_value=*/IREE_SV("field"),
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
          /*.string_value=*/IREE_SV("value"),
      },
      {
          /*.kind=*/LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL,
          /*.string_value=*/IREE_SV("attr_kind"),
      },
  };
  loom_low_lower_diagnostic_t diagnostic = {};
  diagnostic.error_ref = LOOM_ERR_TARGET_003_REF;
  diagnostic.param_count = IREE_ARRAYSIZE(diagnostic_params);
  loom_low_lower_rule_t rule = {};
  rule.source_op_kind = kSourceOpKind;
  rule.guard_count = 1;
  loom_low_lower_rule_set_t rule_set = {};
  rule_set.rules = &rule;
  rule_set.rule_count = 1;
  rule_set.guards = &guard;
  rule_set.guard_count = 1;
  rule_set.diagnostic_params = diagnostic_params;
  rule_set.diagnostic_param_count = IREE_ARRAYSIZE(diagnostic_params);
  rule_set.diagnostics = &diagnostic;
  rule_set.diagnostic_count = 1;
  const loom_low_lower_rule_set_t* rule_sets[] = {&rule_set};

  loom_target_contract_op_entry_t op_entries[4] = {};
  loom_target_contract_dialect_table_t dialects[1] = {};
  loom_target_contract_fragment_case_t cases[1] = {};
  loom_target_contract_descriptor_rule_t descriptor_rules[1] = {};
  loom_target_contract_fragment_t fragment =
      MakeContractFragment(op_entries, dialects, cases, descriptor_rules);
  const loom_target_contract_binding_t bindings[] = {
      {
          &fragment,
          0,
      },
  };

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_target_contract_index_t index = {};
  IREE_ASSERT_OK(loom_target_contract_index_compose(
      bindings, IREE_ARRAYSIZE(bindings), &index, &arena));

  const loom_low_lower_contract_query_options_t options = {
      /*.contract_index=*/&index,
      /*.rule_sets=*/
      {
          /*.count=*/IREE_ARRAYSIZE(rule_sets),
          /*.values=*/rule_sets,
      },
  };
  const loom_target_contract_query_environment_t environment = {
      /*.module=*/nullptr,
      /*.function=*/{},
      /*.bundle=*/&kTargetBundle,
      /*.target_ref=*/{},
      /*.descriptor_set=*/nullptr,
      /*.fact_table=*/nullptr,
      /*.view_regions=*/nullptr,
      /*.arena=*/&arena,
  };
  loom_op_t op = {};
  op.kind = kSourceOpKind;
  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();
  IREE_ASSERT_OK(loom_low_lower_query_target_contract(&environment, &options,
                                                      &op, &result));

  EXPECT_EQ(result.outcome, LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED);
  EXPECT_EQ(result.binding_index, 0);
  EXPECT_EQ(result.case_index, 0);
  EXPECT_EQ(result.rule_set_index, 0);
  EXPECT_EQ(result.rule_index, UINT16_MAX);
  EXPECT_EQ(result.diagnostic_index, 0);
  ASSERT_NE(result.rejection, nullptr);
  EXPECT_EQ(result.rejection->error_ref, LOOM_ERR_TARGET_003_REF);
  ASSERT_EQ(result.rejection->param_count, 8);
  EXPECT_TRUE(iree_string_view_equal(result.rejection->params[4].string,
                                     IREE_SV("test.source")));
  EXPECT_TRUE(iree_string_view_equal(result.rejection->params[5].string,
                                     IREE_SV("field")));
  EXPECT_TRUE(iree_string_view_equal(result.rejection->params[6].string,
                                     IREE_SV("value")));
  EXPECT_TRUE(iree_string_view_equal(result.rejection->params[7].string,
                                     IREE_SV("attr_kind")));

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
