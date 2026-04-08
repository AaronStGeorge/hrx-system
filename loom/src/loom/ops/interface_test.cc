// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Unit tests for the FuncLike, LoopLike, and RegionBranch interface
// cast functions and inline accessors. These tests verify the
// .rodata interface vtables on cache line 3 of the op vtable are
// wired correctly: cast() returns a non-null fat reference for ops
// that implement the interface and {NULL, NULL} for ops that don't,
// and the inline accessors return the right block args / regions /
// operands derived from the interface vtable's stored indices.

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class InterfaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    // Register test, scalar, and scf dialects. test provides
    // test.func to host the body and test.constant for value
    // factories; scf provides the loop/branch ops under test;
    // scalar isn't strictly required but kept available for any
    // future expansion.
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 vtables, (uint16_t)count));
    vtables = loom_scalar_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_SCALAR,
                                                 vtables, (uint16_t)count));
    vtables = loom_scf_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_SCF,
                                                 vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));

    // Build a test.func as a body to host the ops we test.
    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("host_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};
    IREE_ASSERT_OK(loom_test_func_build(&module_builder, 0, 0, 0, callee, NULL,
                                        0, NULL, 0, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op_));
    func_like_ = loom_func_like_cast(module_, func_op_);
    body_ = loom_func_like_body(func_like_);
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body_), &builder_);
    builder_.ip.parent_op = func_op_;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Builds a test.constant of type i32 with the given value.
  loom_op_t* build_i32(int64_t value) {
    loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_test_constant_build(&builder_, loom_attr_i64(value),
                                            i32, LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  // Builds a test.constant of type index with the given value.
  loom_op_t* build_index(int64_t value) {
    loom_type_t index = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_test_constant_build(&builder_, loom_attr_i64(value),
                                            index, LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  // Builds a test.constant of type i1 with the given boolean value.
  loom_op_t* build_i1(bool value) {
    loom_type_t i1 = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
    loom_op_t* op = nullptr;
    IREE_EXPECT_OK(loom_test_constant_build(&builder_, loom_attr_bool(value),
                                            i1, LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* func_op_ = nullptr;
  loom_func_like_t func_like_;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
};

//===----------------------------------------------------------------------===//
// FuncLike interface
//===----------------------------------------------------------------------===//

TEST_F(InterfaceTest, FuncLikeCastReturnsValidForFunc) {
  loom_func_like_t func_like = loom_func_like_cast(module_, func_op_);
  EXPECT_TRUE(loom_func_like_isa(func_like));
  EXPECT_EQ(func_like.op, func_op_);
  EXPECT_NE(func_like.vtable, nullptr);
  EXPECT_EQ(loom_func_like_body(func_like), body_);
}

TEST_F(InterfaceTest, FuncLikeCastReturnsNullForNonFunc) {
  loom_op_t* constant_op = build_i32(42);
  loom_func_like_t func_like = loom_func_like_cast(module_, constant_op);
  EXPECT_FALSE(loom_func_like_isa(func_like));
  EXPECT_EQ(func_like.op, nullptr);
  EXPECT_EQ(func_like.vtable, nullptr);
}

TEST_F(InterfaceTest, FuncLikeCastReturnsNullForNullOp) {
  loom_func_like_t func_like = loom_func_like_cast(module_, nullptr);
  EXPECT_FALSE(loom_func_like_isa(func_like));
  EXPECT_EQ(func_like.op, nullptr);
  EXPECT_EQ(func_like.vtable, nullptr);
}

//===----------------------------------------------------------------------===//
// LoopLike interface
//===----------------------------------------------------------------------===//

TEST_F(InterfaceTest, LoopLikeCastReturnsValidForScfFor) {
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(8);
  loom_op_t* step = build_index(1);
  loom_value_id_t lower_id = loom_op_results(lower)[0];
  loom_value_id_t upper_id = loom_op_results(upper)[0];
  loom_value_id_t step_id = loom_op_results(step)[0];

  loom_op_t* for_op = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(&builder_, lower_id, upper_id, step_id,
                                    nullptr, 0, nullptr, 0, nullptr, 0,
                                    LOOM_LOCATION_UNKNOWN, &for_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, for_op);
  EXPECT_TRUE(loom_loop_like_isa(loop));
  EXPECT_EQ(loop.op, for_op);
  EXPECT_NE(loop.vtable, nullptr);
}

TEST_F(InterfaceTest, LoopLikeCastReturnsNullForNonLoop) {
  loom_op_t* constant_op = build_i32(42);
  loom_loop_like_t loop = loom_loop_like_cast(module_, constant_op);
  EXPECT_FALSE(loom_loop_like_isa(loop));
  EXPECT_EQ(loop.op, nullptr);
  EXPECT_EQ(loop.vtable, nullptr);
}

TEST_F(InterfaceTest, LoopLikeCastReturnsNullForNullOp) {
  loom_loop_like_t loop = loom_loop_like_cast(module_, nullptr);
  EXPECT_FALSE(loom_loop_like_isa(loop));
}

TEST_F(InterfaceTest, LoopLikeCastReturnsNullForScfIf) {
  // scf.if is region-branch-like, not loop-like.
  loom_op_t* condition = build_i1(true);
  loom_value_id_t condition_id = loom_op_results(condition)[0];
  loom_op_t* if_op = nullptr;
  IREE_ASSERT_OK(loom_scf_if_build(&builder_, condition_id, nullptr, 0, nullptr,
                                   0, LOOM_LOCATION_UNKNOWN, &if_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, if_op);
  EXPECT_FALSE(loom_loop_like_isa(loop));
}

TEST_F(InterfaceTest, LoopLikeAccessorsForScfFor) {
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(16);
  loom_op_t* step = build_index(1);
  loom_value_id_t lower_id = loom_op_results(lower)[0];
  loom_value_id_t upper_id = loom_op_results(upper)[0];
  loom_value_id_t step_id = loom_op_results(step)[0];

  loom_op_t* for_op = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(&builder_, lower_id, upper_id, step_id,
                                    nullptr, 0, nullptr, 0, nullptr, 0,
                                    LOOM_LOCATION_UNKNOWN, &for_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, for_op);
  ASSERT_TRUE(loom_loop_like_isa(loop));

  // body region matches the auto-created body region.
  loom_region_t* body = loom_loop_like_body(loop);
  ASSERT_NE(body, nullptr);
  EXPECT_EQ(body, loom_scf_for_body(for_op));

  // scf.for has no separate condition region.
  EXPECT_EQ(loom_loop_like_condition_region(loop), nullptr);

  // The IV is the first block arg of the body's entry block.
  loom_value_id_t iv = loom_loop_like_iv(loop);
  EXPECT_NE(iv, LOOM_VALUE_ID_INVALID);
  EXPECT_EQ(iv, loom_block_arg_id(loom_region_const_entry_block(body), 0));
}

TEST_F(InterfaceTest, LoopLikeIterArgsEmpty) {
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(8);
  loom_op_t* step = build_index(1);
  loom_op_t* for_op = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, loom_op_results(lower)[0], loom_op_results(upper)[0],
      loom_op_results(step)[0], nullptr, 0, nullptr, 0, nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &for_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, for_op);
  loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  EXPECT_EQ(iter_args.count, 0);
}

TEST_F(InterfaceTest, LoopLikeIterArgsNonEmpty) {
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(8);
  loom_op_t* step = build_index(1);
  loom_op_t* init0 = build_i32(10);
  loom_op_t* init1 = build_i32(20);
  loom_value_id_t init_ids[2] = {loom_op_results(init0)[0],
                                 loom_op_results(init1)[0]};
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t result_types[2] = {i32, i32};

  loom_op_t* for_op = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, loom_op_results(lower)[0], loom_op_results(upper)[0],
      loom_op_results(step)[0], init_ids, 2, result_types, 2, nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &for_op));

  loom_loop_like_t loop = loom_loop_like_cast(module_, for_op);
  loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  EXPECT_EQ(iter_args.count, 2);
  EXPECT_EQ(iter_args.values[0], init_ids[0]);
  EXPECT_EQ(iter_args.values[1], init_ids[1]);
}

//===----------------------------------------------------------------------===//
// RegionBranch interface
//===----------------------------------------------------------------------===//

TEST_F(InterfaceTest, RegionBranchCastReturnsValidForScfIf) {
  loom_op_t* condition = build_i1(true);
  loom_value_id_t condition_id = loom_op_results(condition)[0];
  loom_op_t* if_op = nullptr;
  IREE_ASSERT_OK(loom_scf_if_build(&builder_, condition_id, nullptr, 0, nullptr,
                                   0, LOOM_LOCATION_UNKNOWN, &if_op));

  loom_region_branch_t branch = loom_region_branch_cast(module_, if_op);
  EXPECT_TRUE(loom_region_branch_isa(branch));
  EXPECT_EQ(branch.op, if_op);
  EXPECT_NE(branch.vtable, nullptr);
}

TEST_F(InterfaceTest, RegionBranchCastReturnsNullForNonBranch) {
  loom_op_t* constant_op = build_i32(42);
  loom_region_branch_t branch = loom_region_branch_cast(module_, constant_op);
  EXPECT_FALSE(loom_region_branch_isa(branch));
  EXPECT_EQ(branch.op, nullptr);
  EXPECT_EQ(branch.vtable, nullptr);
}

TEST_F(InterfaceTest, RegionBranchCastReturnsNullForNullOp) {
  loom_region_branch_t branch = loom_region_branch_cast(module_, nullptr);
  EXPECT_FALSE(loom_region_branch_isa(branch));
}

TEST_F(InterfaceTest, RegionBranchCastReturnsNullForScfFor) {
  // scf.for is loop-like, not region-branch-like.
  loom_op_t* lower = build_index(0);
  loom_op_t* upper = build_index(8);
  loom_op_t* step = build_index(1);
  loom_op_t* for_op = nullptr;
  IREE_ASSERT_OK(loom_scf_for_build(
      &builder_, loom_op_results(lower)[0], loom_op_results(upper)[0],
      loom_op_results(step)[0], nullptr, 0, nullptr, 0, nullptr, 0,
      LOOM_LOCATION_UNKNOWN, &for_op));

  loom_region_branch_t branch = loom_region_branch_cast(module_, for_op);
  EXPECT_FALSE(loom_region_branch_isa(branch));
}

TEST_F(InterfaceTest, RegionBranchSelectorForScfIf) {
  loom_op_t* condition = build_i1(false);
  loom_value_id_t condition_id = loom_op_results(condition)[0];
  loom_op_t* if_op = nullptr;
  IREE_ASSERT_OK(loom_scf_if_build(&builder_, condition_id, nullptr, 0, nullptr,
                                   0, LOOM_LOCATION_UNKNOWN, &if_op));

  loom_region_branch_t branch = loom_region_branch_cast(module_, if_op);
  ASSERT_TRUE(loom_region_branch_isa(branch));
  EXPECT_EQ(loom_region_branch_selector(branch), condition_id);
}

}  // namespace
}  // namespace loom
