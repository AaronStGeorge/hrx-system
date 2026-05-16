// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/passes/target_legalize.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/contract.h"
#include "loom/codegen/low/lower_rules.h"
#include "loom/codegen/low/pass_environment.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/target/compile_report.h"
#include "loom/target/legalization.h"
#include "loom/target/test/contracts/core_lower_rules.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class TargetLegalizePassTest : public ::testing::Test {
 protected:
  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_test_low_descriptor_registry_initialize(&registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  ModulePtr Parse(iree_string_view_t source) {
    const loom_text_parse_options_t parse_options = {
        .max_errors = 20,
    };
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("target_legalize_test.loom"),
                                  &context_, &block_pool_, &parse_options,
                                  &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  iree_status_t RunTargetLegalize(
      loom_low_lower_policy_registry_t* policy_registry, loom_module_t* module,
      loom_target_compile_report_t* compile_report,
      iree_string_view_t options = iree_string_view_empty(),
      const loom_target_legalizer_provider_list_t* legalizer_provider_list =
          nullptr) {
    iree_arena_allocator_t instance_arena;
    iree_arena_initialize(&block_pool_, &instance_arena);
    loom_pass_value_fact_owner_t value_facts = {};
    loom_pass_value_fact_owner_initialize(&block_pool_, &value_facts);
    const loom_pass_info_t* pass_info = loom_low_target_legalize_pass_info();
    std::vector<int64_t> statistics(pass_info->statistic_count, 0);
    loom_low_pass_environment_storage_t low_pass_environment_storage;
    loom_pass_environment_t environment =
        loom_low_pass_environment_storage_initialize(
            &registry_.registry, policy_registry, nullptr,
            legalizer_provider_list, nullptr, compile_report,
            &low_pass_environment_storage);
    loom_pass_t pass = {
        .info = pass_info,
        .module_run = loom_low_target_legalize_run,
        .instance_arena = &instance_arena,
        .arena = &instance_arena,
        .statistics = statistics.data(),
        .environment = &environment,
        .value_facts = &value_facts,
    };

    iree_status_t status = loom_low_target_legalize_create(&pass, options);
    if (iree_status_is_ok(status)) {
      status = loom_low_target_legalize_run(&pass, module);
    }
    loom_pass_value_fact_owner_deinitialize(&value_facts);
    iree_arena_deinitialize(&instance_arena);
    return status;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_ = {};
};

struct ObserveContractQueryResultState {
  bool saw_query_result = false;
  loom_target_contract_query_outcome_t outcome =
      LOOM_TARGET_CONTRACT_QUERY_INVALID_IR;
  const loom_matrix_fragment_layout_t* selected_matrix_fragment_layout =
      nullptr;
};

static iree_status_t ObserveContractQueryResultLegalizer(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)op;
  auto* state = static_cast<ObserveContractQueryResultState*>(
      const_cast<void*>(entry->user_data));
  state->saw_query_result = context->contract_query_result != nullptr;
  if (context->contract_query_result != nullptr) {
    state->outcome = context->contract_query_result->outcome;
    state->selected_matrix_fragment_layout =
        context->contract_query_result->selected_matrix_fragment_layout;
  }
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  return iree_ok_status();
}

TEST_F(TargetLegalizePassTest, RecordsRewriteRowsInCompileReport) {
  ModulePtr module = Parse(IREE_SV(
      "test.target<low_core> @test_target\n"
      "func.def target(@test_target) @reduce(%v: vector<4xi32>, %init: i32) "
      "-> (i32) {\n"
      "  %r = vector.reduce.axes<addi> %v, %init axes [0] : vector<4xi32>, "
      "i32\n"
      "  func.return %r : i32\n"
      "}\n"));

  loom_low_lower_policy_t policy = *loom_test_low_lower_policy();
  const loom_low_lower_policy_registry_entry_t entries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = &policy,
      },
  };
  loom_low_lower_policy_registry_t policy_registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &policy_registry, entries, IREE_ARRAYSIZE(entries));

  loom_target_compile_report_legalization_row_t rows[8] = {};
  const loom_target_compile_report_row_storage_t row_storage = {
      .target_legalization_rows = rows,
      .target_legalization_row_capacity = IREE_ARRAYSIZE(rows),
  };
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report);
  loom_target_compile_report_set_row_storage(&report, &row_storage);

  IREE_ASSERT_OK(RunTargetLegalize(&policy_registry, module.get(), &report));
  EXPECT_EQ(report.target_legalization_rewritten_op_count, 1u);
  EXPECT_GE(report.target_legalization_row_total_count, 1u);
  ASSERT_GE(report.target_legalization_row_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(rows[0].function_name, IREE_SV("reduce")));
  EXPECT_TRUE(iree_string_view_equal(rows[0].source_op_name,
                                     IREE_SV("vector.reduce.axes")));
  EXPECT_TRUE(
      iree_string_view_equal(rows[0].legalizer_name, IREE_SV("vector")));
  EXPECT_EQ(rows[0].legalizer_strategy,
            LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_REFERENCE);
  EXPECT_EQ(rows[0].action,
            LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN);
  EXPECT_EQ(rows[0].mode, LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_EAGER);
  EXPECT_GT(rows[0].created_op_count, 0u);
  EXPECT_EQ(rows[0].erased_op_count, 1u);
}

TEST_F(TargetLegalizePassTest, LegalizerReceivesCurrentContractQueryResult) {
  ModulePtr module = Parse(IREE_SV(
      "test.target<low_core> @test_target\n"
      "func.def target(@test_target) @reduce(%v: vector<4xi32>, %init: i32) "
      "-> (i32) {\n"
      "  %r = vector.reduce.axes<addi> %v, %init axes [0] : vector<4xi32>, "
      "i32\n"
      "  func.return %r : i32\n"
      "}\n"));

  loom_low_lower_policy_t policy = *loom_test_low_lower_policy();
  const loom_low_lower_policy_registry_entry_t policy_entries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = &policy,
      },
  };
  loom_low_lower_policy_registry_t policy_registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &policy_registry, policy_entries, IREE_ARRAYSIZE(policy_entries));

  ObserveContractQueryResultState observer_state = {};
  const loom_target_legalizer_entry_t legalizer_entries[] = {
      {
          .root_kind = LOOM_OP_VECTOR_REDUCE_AXES,
          .legalize = ObserveContractQueryResultLegalizer,
          .user_data = &observer_state,
      },
  };
  const loom_target_legalizer_provider_t legalizer_provider = {
      .name = IREE_SVL("observer"),
      .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_UNKNOWN,
      .entries = legalizer_entries,
      .entry_count = IREE_ARRAYSIZE(legalizer_entries),
  };
  const loom_target_legalizer_provider_t* legalizer_providers[] = {
      &legalizer_provider,
  };
  const loom_target_legalizer_provider_list_t legalizer_provider_list =
      loom_target_legalizer_provider_list_make(
          legalizer_providers, IREE_ARRAYSIZE(legalizer_providers));

  IREE_ASSERT_OK(RunTargetLegalize(&policy_registry, module.get(), nullptr,
                                   iree_string_view_empty(),
                                   &legalizer_provider_list));
  EXPECT_TRUE(observer_state.saw_query_result);
  EXPECT_EQ(observer_state.outcome, LOOM_TARGET_CONTRACT_QUERY_UNHANDLED);
  EXPECT_EQ(observer_state.selected_matrix_fragment_layout, nullptr);
}

TEST_F(TargetLegalizePassTest, RecordsReferenceRejectReasonInCompileReport) {
  ModulePtr module = Parse(
      IREE_SV("test.target<low_core> @test_target\n"
              "func.def target(@test_target) @mma(%lhs_data: vector<16xf16>, "
              "%rhs_data: vector<16xf16>, %init_data: vector<8xf32>) -> "
              "(vector<8xf32>) {\n"
              "  %m = index.constant 16 : index\n"
              "  %n = index.constant 16 : index\n"
              "  %k = index.constant 16 : index\n"
              "  %lhs = vector.fragment<lhs> %lhs_data shape [%m, %k] : "
              "vector<16xf16>\n"
              "  %rhs = vector.fragment<rhs> %rhs_data shape [%k, %n] : "
              "vector<16xf16>\n"
              "  %init = vector.fragment<init> %init_data shape [%m, %n] : "
              "vector<8xf32>\n"
              "  %result = vector.mma %lhs, %rhs, %init : vector<16xf16>, "
              "vector<16xf16>, vector<8xf32>\n"
              "  func.return %result : vector<8xf32>\n"
              "}\n"));

  loom_low_lower_policy_t policy = *loom_test_low_lower_policy();
  const loom_low_lower_policy_registry_entry_t entries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = &policy,
      },
  };
  loom_low_lower_policy_registry_t policy_registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &policy_registry, entries, IREE_ARRAYSIZE(entries));

  loom_target_compile_report_legalization_row_t rows[8] = {};
  const loom_target_compile_report_row_storage_t row_storage = {
      .target_legalization_rows = rows,
      .target_legalization_row_capacity = IREE_ARRAYSIZE(rows),
  };
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report);
  loom_target_compile_report_set_row_storage(&report, &row_storage);

  IREE_ASSERT_OK(RunTargetLegalize(&policy_registry, module.get(), &report,
                                   IREE_SV("mode=final")));
  EXPECT_EQ(report.target_legalization_unsupported_op_count, 1u);

  const loom_target_compile_report_legalization_row_t* mma_row = nullptr;
  for (iree_host_size_t i = 0; i < report.target_legalization_row_count; ++i) {
    if (rows[i].source_op_kind == LOOM_OP_VECTOR_MMA &&
        rows[i].action ==
            LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_UNSUPPORTED_FINAL) {
      mma_row = &rows[i];
      break;
    }
  }
  ASSERT_NE(mma_row, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(mma_row->legalizer_name, IREE_SV("vector")));
  EXPECT_EQ(mma_row->legalizer_strategy,
            LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_REFERENCE);
  EXPECT_TRUE(iree_any_bit_set(mma_row->source_rejection_bits,
                               LOOM_CONTRACT_REJECTION_FRAGMENT));
  EXPECT_EQ(mma_row->source_rejection_detail,
            LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_LHS_FRAGMENT_OWNERSHIP);
}

}  // namespace
}  // namespace loom
