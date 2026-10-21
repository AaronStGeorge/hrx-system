// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_observation.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"

namespace loom {
namespace {

class FactObservationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    loom_pass_value_fact_owner_initialize(&block_pool_, &fact_owner_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("observations"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));

    loom_builder_t builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder);
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&builder, IREE_SV("sum"), &name));
    uint16_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    loom_op_t* function_op = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(
        &builder, 0, 0, 0, {.module_id = 0, .symbol_id = symbol}, nullptr, 0,
        nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN,
        &function_op));
    function_ = loom_func_like_cast(module_, function_op);
    loom_builder_enter_region(&builder, function_op,
                              loom_func_like_body(function_));
    loom_builder_set_block(
        &builder, loom_region_entry_block(loom_func_like_body(function_)));
    const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    IREE_ASSERT_OK(loom_test_constant_build(&builder, loom_attr_i64(2), type,
                                            LOOM_LOCATION_UNKNOWN, &seed_));
    loom_op_t* one = nullptr;
    IREE_ASSERT_OK(loom_test_constant_build(&builder, loom_attr_i64(1), type,
                                            LOOM_LOCATION_UNKNOWN, &one));
    loom_op_t* add = nullptr;
    IREE_ASSERT_OK(loom_test_addi_build(
        &builder, loom_test_constant_result(seed_),
        loom_test_constant_result(one), type, LOOM_LOCATION_UNKNOWN, &add));
    sum_ = loom_test_addi_result(add);
    loom_op_t* use = nullptr;
    IREE_ASSERT_OK(
        loom_test_use_build(&builder, &sum_, 1, LOOM_LOCATION_UNKNOWN, &use));
    loom_op_t* terminator = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&builder, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &terminator));

    IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
        &fact_owner_, module_, loom_pass_value_fact_scope_function(function_),
        &facts_));
    loom_rewriter_initialize(&rewriter_, module_, &arena_);
    loom_rewriter_attach_value_facts(&rewriter_, facts_);
    loom_builder_set_before(&rewriter_.builder, terminator);
    loom_symbolic_expr_context_initialize(module_, nullptr, facts_, &arena_,
                                          &expressions_);
  }

  void TearDown() override {
    loom_rewriter_deinitialize(&rewriter_);
    loom_pass_value_fact_owner_deinitialize(&fact_owner_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void ExpectSum(int64_t expected) {
    loom_symbolic_expr_t expression;
    IREE_ASSERT_OK(
        loom_symbolic_expr_from_value(&expressions_, sum_, &expression));
    int64_t value = 0;
    ASSERT_TRUE(loom_value_facts_as_exact_i64(expression.facts, &value));
    EXPECT_EQ(value, expected);
  }

  // Shared backing for independently owned module, fact, and rewrite arenas.
  iree_arena_block_pool_t block_pool_;
  // Rewrite worklists and symbolic memo storage.
  iree_arena_allocator_t arena_;
  // Operation semantics used by the test function.
  loom_context_t context_;
  // Module owned by the fixture.
  loom_module_t* module_ = nullptr;
  // Function whose facts are maintained during rewriting.
  loom_func_like_t function_;
  // Mutable input to the retained sum.
  loom_op_t* seed_ = nullptr;
  // Result observed through the symbolic cache.
  loom_value_id_t sum_ = LOOM_VALUE_ID_INVALID;
  // Owns the table across scope invalidation and reacquisition.
  loom_pass_value_fact_owner_t fact_owner_;
  // Table borrowed from the owner.
  loom_value_fact_table_t* facts_ = nullptr;
  // Maintains pending equations independently of the symbolic cache.
  loom_rewriter_t rewriter_ = {};
  // Retains derived answers until explicitly reset.
  loom_symbolic_expr_context_t expressions_;
};

TEST_F(FactObservationTest, IndependentConstructionRetainsObservationsAndMemo) {
  ExpectSum(3);
  const uint32_t memo_count = expressions_.memo.count;
  ASSERT_GT(memo_count, 0u);
#ifndef NDEBUG
  const auto observation = loom_value_fact_observation_capture(facts_);
#endif
  for (int64_t value = 40; value < 48; ++value) {
    loom_op_t* constant = nullptr;
    IREE_ASSERT_OK(
        loom_test_constant_build(&rewriter_.builder, loom_attr_i64(value),
                                 loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                 LOOM_LOCATION_UNKNOWN, &constant));
    // Check before the next query, which could otherwise repopulate a reset
    // memo.
    EXPECT_EQ(expressions_.memo.count, memo_count);
#ifndef NDEBUG
    loom_value_fact_observation_assert_valid(observation);
#endif
    int64_t defined_value = 0;
    ASSERT_TRUE(loom_value_facts_as_exact_i64(
        loom_rewriter_value_facts(&rewriter_,
                                  loom_test_constant_result(constant)),
        &defined_value));
    EXPECT_EQ(defined_value, value);
    ExpectSum(3);
  }
}

TEST_F(FactObservationTest, ResetBeforePublicationObservesCompletedBatch) {
  ExpectSum(3);
  IREE_ASSERT_OK(
      loom_test_constant_rewrite_value(&rewriter_, seed_, loom_attr_i64(9)));
  // Direct constant facts are already current while downstream equations wait.
  int64_t seed_value = 0;
  ASSERT_TRUE(loom_value_facts_as_exact_i64(
      loom_rewriter_value_facts(&rewriter_, loom_test_constant_result(seed_)),
      &seed_value));
  EXPECT_EQ(seed_value, 9);

  // Greedy rewriting retires its memo at the end of a successful rewrite,
  // before the next observation publishes pending equations.
  loom_symbolic_expr_context_reset(&expressions_);
  IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter_));
  ExpectSum(10);
}

#ifndef NDEBUG
TEST_F(FactObservationTest, CachedReadsRejectPendingAndExpiredObservations) {
  ExpectSum(3);
  const auto observation = loom_value_fact_observation_capture(facts_);
  IREE_ASSERT_OK(
      loom_test_constant_rewrite_value(&rewriter_, seed_, loom_attr_i64(9)));
  EXPECT_DEATH_IF_SUPPORTED(
      loom_value_fact_observation_assert_valid(observation),
      "fact observation has pending equations");
  loom_symbolic_expr_summary_t summary;
  EXPECT_DEATH_IF_SUPPORTED(loom_symbolic_expr_context_try_lookup_summary(
                                &expressions_, sum_, &summary),
                            "fact observation has pending equations");

  IREE_ASSERT_OK(loom_rewriter_propagate_value_facts(&rewriter_));
  EXPECT_DEATH_IF_SUPPORTED(
      loom_value_fact_observation_assert_valid(observation),
      "fact observation expired");
  loom_symbolic_expr_t expression;
  EXPECT_DEATH_IF_SUPPORTED(IREE_ASSERT_OK(loom_symbolic_expr_from_value(
                                &expressions_, sum_, &expression)),
                            "fact observation expired");
  loom_symbolic_expr_context_reset(&expressions_);
  ExpectSum(10);
}

TEST_F(FactObservationTest, ScopeRetirementExpiresCachedFacts) {
  ExpectSum(3);
  loom_pass_value_fact_owner_invalidate(&fact_owner_);
  loom_symbolic_expr_t expression;
  EXPECT_DEATH_IF_SUPPORTED(IREE_ASSERT_OK(loom_symbolic_expr_from_value(
                                &expressions_, sum_, &expression)),
                            "fact observation expired");
  IREE_ASSERT_OK(loom_pass_value_fact_owner_acquire(
      &fact_owner_, module_, loom_pass_value_fact_scope_function(function_),
      &facts_));
  loom_symbolic_expr_context_reset(&expressions_);
  ExpectSum(3);
}
#endif  // !NDEBUG

}  // namespace
}  // namespace loom
