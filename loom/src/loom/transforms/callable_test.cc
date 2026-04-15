// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/callable.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class CallableInlineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder_);
    iree_arena_initialize(&block_pool_, &rewriter_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&rewriter_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using VTableFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(loom_dialect_id_t dialect_id, VTableFn vtables_fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = vtables_fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  loom_symbol_ref_t MakeSymbol(iree_string_view_t name) {
    loom_string_id_t name_id = InternString(name);
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
  }

  loom_string_id_t InternString(iree_string_view_t string) {
    loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_builder_intern_string(&module_builder_, string, &string_id));
    return string_id;
  }

  void SetValueName(loom_value_id_t value_id, iree_string_view_t name) {
    loom_string_id_t name_id = InternString(name);
    loom_module_value(module_, value_id)->name_id = name_id;
  }

  loom_builder_t BodyBuilder(loom_op_t* func_op) {
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    loom_builder_t builder = {};
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(loom_func_like_body(func)),
                            &builder);
    builder.ip.parent_op = func_op;
    return builder;
  }

  loom_op_t* BuildNegateFunction(loom_symbol_ref_t callee,
                                 loom_type_t value_type) {
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &module_builder_, 0, 0, 0, 0, callee, &value_type, 1, &value_type, 1,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    IREE_ASSERT_EQ(arg_count, 1);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* neg_op = nullptr;
    IREE_CHECK_OK(loom_test_neg_build(&body_builder, args[0], value_type,
                                      LOOM_LOCATION_UNKNOWN, &neg_op));
    loom_value_id_t negated = loom_test_neg_result(neg_op);
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, &negated, 1,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    return func_op;
  }

  loom_op_t* BuildNegateTemplate(loom_symbol_ref_t callee,
                                 loom_type_t value_type) {
    loom_string_id_t implements = InternString(IREE_SV("test.neg"));
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_template_build(&module_builder_, 0, implements, 0,
                                           0, 0, 0, callee, &value_type, 1,
                                           &value_type, 1, nullptr, 0, nullptr,
                                           0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    IREE_ASSERT_EQ(arg_count, 1);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* neg_op = nullptr;
    IREE_CHECK_OK(loom_test_neg_build(&body_builder, args[0], value_type,
                                      LOOM_LOCATION_UNKNOWN, &neg_op));
    loom_value_id_t negated = loom_test_neg_result(neg_op);
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, &negated, 1,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    return func_op;
  }

  loom_op_t* BuildSwapAndNegateFunction(loom_symbol_ref_t callee,
                                        loom_type_t value_type) {
    loom_type_t arg_types[2] = {value_type, value_type};
    loom_type_t result_types[2] = {value_type, value_type};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &module_builder_, 0, 0, 0, 0, callee, arg_types,
        IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    IREE_ASSERT_EQ(arg_count, 2);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* neg_op = nullptr;
    IREE_CHECK_OK(loom_test_neg_build(&body_builder, args[0], value_type,
                                      LOOM_LOCATION_UNKNOWN, &neg_op));
    loom_value_id_t returned[2] = {args[1], loom_test_neg_result(neg_op)};
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, returned,
                                         IREE_ARRAYSIZE(returned),
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    return func_op;
  }

  loom_op_t* BuildCaller(loom_symbol_ref_t caller, loom_symbol_ref_t callee,
                         loom_type_t value_type, loom_op_t** out_call_op) {
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &module_builder_, 0, 0, 0, 0, caller, &value_type, 1, &value_type, 1,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    IREE_ASSERT_EQ(arg_count, 1);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* call_op = nullptr;
    IREE_CHECK_OK(loom_func_call_build(&body_builder, 0, 0, callee, args, 1,
                                       &value_type, 1, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &call_op));
    loom_value_id_t call_result = loom_func_call_results(call_op).values[0];
    SetValueName(call_result, IREE_SV("call_result"));
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, &call_result, 1,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    *out_call_op = call_op;
    return func_op;
  }

  loom_op_t* BuildApplyCaller(loom_symbol_ref_t caller,
                              loom_symbol_ref_t callee, loom_type_t value_type,
                              loom_op_t** out_apply_op) {
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &module_builder_, 0, 0, 0, 0, caller, &value_type, 1, &value_type, 1,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    IREE_ASSERT_EQ(arg_count, 1);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* apply_op = nullptr;
    IREE_CHECK_OK(loom_func_apply_build(&body_builder, 0, 0, callee, args, 1,
                                        &value_type, 1, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &apply_op));
    loom_value_id_t apply_result = loom_func_apply_results(apply_op).values[0];
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, &apply_result, 1,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    *out_apply_op = apply_op;
    return func_op;
  }

  loom_op_t* BuildTwoResultCaller(loom_symbol_ref_t caller,
                                  loom_symbol_ref_t callee,
                                  loom_type_t value_type,
                                  loom_op_t** out_call_op) {
    loom_type_t arg_types[2] = {value_type, value_type};
    loom_type_t result_types[2] = {value_type, value_type};
    loom_op_t* func_op = nullptr;
    IREE_CHECK_OK(loom_func_def_build(
        &module_builder_, 0, 0, 0, 0, caller, arg_types,
        IREE_ARRAYSIZE(arg_types), result_types, IREE_ARRAYSIZE(result_types),
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_func_like_t func = loom_func_like_cast(module_, func_op);
    uint16_t arg_count = 0;
    const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
    IREE_ASSERT_EQ(arg_count, 2);

    loom_builder_t body_builder = BodyBuilder(func_op);
    loom_op_t* call_op = nullptr;
    IREE_CHECK_OK(loom_func_call_build(&body_builder, 0, 0, callee, args,
                                       IREE_ARRAYSIZE(arg_types), result_types,
                                       IREE_ARRAYSIZE(result_types), nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &call_op));
    loom_value_slice_t call_results = loom_func_call_results(call_op);
    SetValueName(call_results.values[0], IREE_SV("swapped"));
    SetValueName(call_results.values[1], IREE_SV("negated"));
    loom_op_t* return_op = nullptr;
    IREE_CHECK_OK(loom_func_return_build(&body_builder, call_results.values,
                                         call_results.count,
                                         LOOM_LOCATION_UNKNOWN, &return_op));
    *out_call_op = call_op;
    return func_op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t module_builder_ = {};
  iree_arena_allocator_t rewriter_arena_;
};

TEST_F(CallableInlineTest, InlinesDirectCallAndReplacesReturnOperand) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_op_t* callee_op = BuildNegateFunction(callee_ref, i32);
  loom_op_t* call_op = nullptr;
  loom_op_t* caller_op = BuildCaller(caller_ref, callee_ref, i32, &call_op);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_ASSERT_OK(loom_callable_inline_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_TRUE(iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD));
  loom_region_t* caller_body =
      loom_func_like_body(loom_func_like_cast(module_, caller_op));
  loom_block_t* caller_block = loom_region_entry_block(caller_body);
  ASSERT_EQ(caller_block->op_count, 2u);
  ASSERT_TRUE(loom_test_neg_isa(loom_block_op(caller_block, 0)));
  ASSERT_TRUE(loom_func_return_isa(loom_block_op(caller_block, 1)));
  EXPECT_EQ(loom_block_op(caller_block, 0)->parent_op, caller_op);

  loom_value_id_t negated =
      loom_test_neg_result(loom_block_op(caller_block, 0));
  EXPECT_TRUE(iree_string_view_equal(
      module_->strings.entries[loom_module_value(module_, negated)->name_id],
      IREE_SV("call_result")));
  loom_value_slice_t returned =
      loom_func_return_operands(loom_block_op(caller_block, 1));
  ASSERT_EQ(returned.count, 1u);
  EXPECT_EQ(returned.values[0], negated);

  loom_region_t* callee_body =
      loom_func_like_body(loom_func_like_cast(module_, callee_op));
  EXPECT_EQ(loom_region_entry_block(callee_body)->op_count, 2u);
}

TEST_F(CallableInlineTest, ConsumingInlineMovesBodyAndErasesCallee) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_op_t* callee_op = BuildNegateFunction(callee_ref, i32);
  loom_func_like_t callee = loom_func_like_cast(module_, callee_op);
  loom_block_t* callee_block =
      loom_region_entry_block(loom_func_like_body(callee));
  loom_op_t* moved_neg_op = loom_block_op(callee_block, 0);
  loom_op_t* call_op = nullptr;
  loom_op_t* caller_op = BuildCaller(caller_ref, callee_ref, i32, &call_op);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_ASSERT_OK(
      loom_callable_inline_consuming_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_TRUE(iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_TRUE(iree_any_bit_set(callee_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_EQ(module_->symbols.entries[callee_ref.symbol_id].defining_op,
            nullptr);
  EXPECT_EQ(module_->symbols.entries[callee_ref.symbol_id].kind,
            LOOM_SYMBOL_NONE);

  loom_func_like_t caller = loom_func_like_cast(module_, caller_op);
  loom_block_t* caller_block =
      loom_region_entry_block(loom_func_like_body(caller));
  ASSERT_EQ(caller_block->op_count, 2u);
  EXPECT_EQ(loom_block_op(caller_block, 0), moved_neg_op);
  ASSERT_TRUE(loom_func_return_isa(loom_block_op(caller_block, 1)));
  loom_value_id_t negated = loom_test_neg_result(moved_neg_op);
  loom_value_slice_t returned =
      loom_func_return_operands(loom_block_op(caller_block, 1));
  ASSERT_EQ(returned.count, 1u);
  EXPECT_EQ(returned.values[0], negated);
  EXPECT_TRUE(iree_string_view_equal(
      module_->strings.entries[loom_module_value(module_, negated)->name_id],
      IREE_SV("call_result")));
}

TEST_F(CallableInlineTest, ConsumingInlineRejectsPublicCallee) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_op_t* callee_op = BuildNegateFunction(callee_ref, i32);
  module_->symbols.entries[callee_ref.symbol_id].flags |=
      LOOM_SYMBOL_FLAG_PUBLIC;
  loom_op_t* call_op = nullptr;
  BuildCaller(caller_ref, callee_ref, i32, &call_op);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_callable_inline_consuming_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_FALSE(iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_FALSE(iree_any_bit_set(callee_op->flags, LOOM_OP_FLAG_DEAD));
}

TEST_F(CallableInlineTest, ConsumingInlineRejectsAdditionalCalleeReference) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  loom_symbol_ref_t other_caller_ref = MakeSymbol(IREE_SV("other_caller"));
  loom_op_t* callee_op = BuildNegateFunction(callee_ref, i32);
  loom_op_t* call_op = nullptr;
  BuildCaller(caller_ref, callee_ref, i32, &call_op);
  loom_op_t* other_call_op = nullptr;
  BuildCaller(other_caller_ref, callee_ref, i32, &other_call_op);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_callable_inline_consuming_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_FALSE(iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_FALSE(iree_any_bit_set(other_call_op->flags, LOOM_OP_FLAG_DEAD));
  EXPECT_FALSE(iree_any_bit_set(callee_op->flags, LOOM_OP_FLAG_DEAD));
}

TEST_F(CallableInlineTest, InlinesFuncApplyTemplate) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("template_negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  BuildNegateTemplate(callee_ref, i32);
  loom_op_t* apply_op = nullptr;
  loom_op_t* caller_op =
      BuildApplyCaller(caller_ref, callee_ref, i32, &apply_op);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_ASSERT_OK(loom_callable_inline_direct_call(&rewriter, apply_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_TRUE(iree_any_bit_set(apply_op->flags, LOOM_OP_FLAG_DEAD));
  loom_region_t* caller_body =
      loom_func_like_body(loom_func_like_cast(module_, caller_op));
  loom_block_t* caller_block = loom_region_entry_block(caller_body);
  ASSERT_EQ(caller_block->op_count, 2u);
  ASSERT_TRUE(loom_test_neg_isa(loom_block_op(caller_block, 0)));
  ASSERT_TRUE(loom_func_return_isa(loom_block_op(caller_block, 1)));
}

TEST_F(CallableInlineTest, InlinesMultiResultCall) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t callee_ref = MakeSymbol(IREE_SV("swap_and_negate"));
  loom_symbol_ref_t caller_ref = MakeSymbol(IREE_SV("caller"));
  BuildSwapAndNegateFunction(callee_ref, i32);
  loom_op_t* call_op = nullptr;
  loom_op_t* caller_op =
      BuildTwoResultCaller(caller_ref, callee_ref, i32, &call_op);
  loom_func_like_t caller = loom_func_like_cast(module_, caller_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(caller, &arg_count);
  ASSERT_EQ(arg_count, 2u);

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_ASSERT_OK(loom_callable_inline_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);

  EXPECT_TRUE(iree_any_bit_set(call_op->flags, LOOM_OP_FLAG_DEAD));
  loom_region_t* caller_body = loom_func_like_body(caller);
  loom_block_t* caller_block = loom_region_entry_block(caller_body);
  ASSERT_EQ(caller_block->op_count, 2u);
  ASSERT_TRUE(loom_test_neg_isa(loom_block_op(caller_block, 0)));
  ASSERT_TRUE(loom_func_return_isa(loom_block_op(caller_block, 1)));
  loom_value_id_t negated =
      loom_test_neg_result(loom_block_op(caller_block, 0));
  EXPECT_TRUE(iree_string_view_equal(
      module_->strings.entries[loom_module_value(module_, negated)->name_id],
      IREE_SV("negated")));
  loom_value_slice_t returned =
      loom_func_return_operands(loom_block_op(caller_block, 1));
  ASSERT_EQ(returned.count, 2u);
  EXPECT_EQ(returned.values[0], args[1]);
  EXPECT_EQ(returned.values[1], negated);
}

TEST_F(CallableInlineTest, RejectsRecursiveSelfInline) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_symbol_ref_t self_ref = MakeSymbol(IREE_SV("self"));
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_func_def_build(&module_builder_, 0, 0, 0, 0, self_ref,
                                     &i32, 1, &i32, 1, nullptr, 0, nullptr, 0,
                                     LOOM_LOCATION_UNKNOWN, &func_op));
  loom_func_like_t func = loom_func_like_cast(module_, func_op);
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  ASSERT_EQ(arg_count, 1u);

  loom_builder_t body_builder = BodyBuilder(func_op);
  loom_op_t* call_op = nullptr;
  IREE_ASSERT_OK(loom_func_call_build(&body_builder, 0, 0, self_ref, args, 1,
                                      &i32, 1, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &call_op));
  loom_value_id_t call_result = loom_func_call_results(call_op).values[0];
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_func_return_build(&body_builder, &call_result, 1,
                                        LOOM_LOCATION_UNKNOWN, &return_op));

  loom_rewriter_t rewriter = {};
  IREE_ASSERT_OK(
      loom_rewriter_initialize(&rewriter, module_, &rewriter_arena_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_callable_inline_direct_call(&rewriter, call_op));
  loom_rewriter_deinitialize(&rewriter);
}

}  // namespace
}  // namespace loom
