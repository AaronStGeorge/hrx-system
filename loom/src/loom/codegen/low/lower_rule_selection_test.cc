// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower_rules.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/lower.h"

namespace loom {
namespace {

struct ModuleDeleter {
  void operator()(loom_module_t* module) const { loom_module_free(module); }
};

using ModulePtr = std::unique_ptr<loom_module_t, ModuleDeleter>;

class LowLowerRuleSelectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_VECTOR, loom_vector_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseSource(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("low_rule_selection_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    IREE_ASSERT(module != nullptr);
    return ModulePtr(module);
  }

  loom_func_like_t FirstFunction(loom_module_t* module) {
    loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_module_block(module), op) {
      loom_func_like_t function = loom_func_like_cast(module, op);
      if (loom_func_like_isa(function)) {
        return function;
      }
    }
    ADD_FAILURE() << "expected module to contain a function";
    return (loom_func_like_t){0};
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  const loom_op_t* FirstBodyOpOfKind(loom_module_t* module,
                                     loom_op_kind_t kind) {
    loom_func_like_t function = FirstFunction(module);
    loom_region_t* body = loom_func_like_body(function);
    if (body == nullptr) {
      ADD_FAILURE() << "expected function to have a body";
      return nullptr;
    }
    const loom_block_t* entry_block = loom_region_const_entry_block(body);
    loom_op_t* op = nullptr;
    loom_block_for_each_op(entry_block, op) {
      if (op->kind == kind) {
        return op;
      }
    }
    ADD_FAILURE() << "expected function body to contain op kind " << kind;
    return nullptr;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(LowLowerRuleSelectionTest, SelectsVectorRule) {
  ModulePtr module = ParseSource(
      "func.def @add_vectors(%lhs: vector<4xi32>, %rhs: vector<4xi32>) -> "
      "(vector<4xi32>) {\n"
      "  %sum = vector.addi %lhs, %rhs : vector<4xi32>\n"
      "  func.return %sum : vector<4xi32>\n"
      "}\n");
  const loom_op_t* source_op =
      FirstBodyOpOfKind(module.get(), LOOM_OP_VECTOR_ADDI);
  ASSERT_NE(source_op, nullptr);

  const loom_low_lower_rule_match_context_t match_context = {
      .module = module.get(),
      .descriptor_set = loom_test_low_core_descriptor_set(),
      .map_value =
          {
              .fn = loom_test_low_lower_rule_match_map_value,
              .user_data = nullptr,
          },
  };
  loom_low_lower_rule_selection_t selection = {};
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &loom_test_low_lower_rule_set, source_op, &selection));

  EXPECT_TRUE(selection.has_source_op_span);
  ASSERT_NE(selection.rule, nullptr);
  EXPECT_EQ(selection.diagnostic_index, LOOM_LOW_LOWER_DIAGNOSTIC_NONE);
  EXPECT_EQ(loom_low_lower_rule_set_selection_diagnostic(
                &loom_test_low_lower_rule_set, selection),
            nullptr);
  EXPECT_EQ(loom_low_lower_rule_first_descriptor_id(
                &loom_test_low_lower_rule_set, selection.rule),
            TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32);
}

TEST_F(LowLowerRuleSelectionTest, UsesMaterializerPredicate) {
  ModulePtr module = ParseSource(
      "func.def @madd(%lhs: index, %rhs: index, %acc: index) -> (index) {\n"
      "  %result = index.madd %lhs, %rhs, %acc : index\n"
      "  func.return %result : index\n"
      "}\n");
  const loom_op_t* source_op =
      FirstBodyOpOfKind(module.get(), LOOM_OP_INDEX_MADD);
  ASSERT_NE(source_op, nullptr);

  loom_low_lower_rule_match_context_t match_context = {
      .module = module.get(),
      .descriptor_set = loom_test_low_core_descriptor_set(),
      .map_value =
          {
              .fn = loom_test_low_lower_rule_match_map_value,
              .user_data = nullptr,
          },
  };
  loom_low_lower_rule_selection_t selection = {};
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &loom_test_low_lower_rule_set, source_op, &selection));
  EXPECT_EQ(selection.rule, nullptr);
  const loom_low_lower_diagnostic_t* diagnostic =
      loom_low_lower_rule_set_selection_diagnostic(
          &loom_test_low_lower_rule_set, selection);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_TRUE(iree_string_view_equal(diagnostic->subject_kind,
                                     IREE_SV("materializer")));
  EXPECT_TRUE(
      iree_string_view_equal(diagnostic->subject_name, IREE_SV("copy")));
  EXPECT_TRUE(iree_string_view_equal(
      diagnostic->reason,
      IREE_SV("test lowering requires copy-materializable values")));

  match_context.can_materialize.fn =
      loom_test_low_lower_rule_match_can_materialize;
  match_context.can_materialize.user_data = nullptr;
  IREE_ASSERT_OK(loom_low_lower_rule_set_select_with_match_context(
      &match_context, &loom_test_low_lower_rule_set, source_op, &selection));

  EXPECT_TRUE(selection.has_source_op_span);
  ASSERT_NE(selection.rule, nullptr);
  EXPECT_EQ(selection.diagnostic_index, LOOM_LOW_LOWER_DIAGNOSTIC_NONE);
  EXPECT_EQ(loom_low_lower_rule_first_descriptor_id(
                &loom_test_low_lower_rule_set, selection.rule),
            TEST_LOW_CORE_DESCRIPTOR_ID_TEST_ADD_I32);
}

}  // namespace
}  // namespace loom
