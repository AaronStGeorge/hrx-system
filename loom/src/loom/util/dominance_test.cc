// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/dominance.h"

#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class DominanceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_test_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));

    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};
    loom_op_t* func_op = NULL;
    IREE_ASSERT_OK(loom_test_func_build(&module_builder, 0, 0, 0, callee, NULL,
                                        0, NULL, 0, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op));
    func_like_ = loom_func_like_cast(module_, func_op);
    func_op_ = func_op;
    body_ = loom_func_like_body(func_like_);
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body_), &builder_);
    builder_.ip.parent_op = func_op;

    iree_arena_initialize(&block_pool_, &dom_arena_);
    loom_dominance_info_initialize(module_, &dom_arena_, &dom_info_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&dom_arena_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void enter_region(loom_op_t* parent_op, loom_region_t* region) {
    saved_ips_.push_back(
        loom_builder_enter_region(&builder_, parent_op, region));
  }

  void leave_region() {
    ASSERT_FALSE(saved_ips_.empty());
    loom_builder_restore(&builder_, saved_ips_.back());
    saved_ips_.pop_back();
  }

  void finalize() { IREE_ASSERT_OK(loom_module_compute_uses(module_)); }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_func_like_t func_like_;
  loom_op_t* func_op_ = nullptr;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
  iree_arena_allocator_t dom_arena_;
  loom_dominance_info_t dom_info_;
  std::vector<loom_builder_ip_t> saved_ips_;
};

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST_F(DominanceTest, SelfDominance) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* c1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &c1));
  finalize();

  EXPECT_TRUE(loom_dominates_op(&dom_info_, c1, c1));
}

TEST_F(DominanceTest, SameBlockOrdering) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* c1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &c1));
  loom_op_t* c2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &c2));
  loom_op_t* c3 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(3), i32,
                                          LOOM_LOCATION_UNKNOWN, &c3));
  finalize();

  EXPECT_TRUE(loom_dominates_op(&dom_info_, c1, c2));
  EXPECT_TRUE(loom_dominates_op(&dom_info_, c1, c3));
  EXPECT_TRUE(loom_dominates_op(&dom_info_, c2, c3));
  EXPECT_FALSE(loom_dominates_op(&dom_info_, c2, c1));
  EXPECT_FALSE(loom_dominates_op(&dom_info_, c3, c1));
  EXPECT_FALSE(loom_dominates_op(&dom_info_, c3, c2));
}

TEST_F(DominanceTest, OuterDominatesInner) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_op_t* outer_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &outer_const));
  loom_value_id_t arg = loom_test_constant_result(outer_const);
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, f32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  enter_region(map_op, loom_test_map_body(map_op));
  loom_op_t* inner_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_const));
  leave_region();
  finalize();

  // Outer dominates inner.
  EXPECT_TRUE(loom_dominates_op(&dom_info_, outer_const, inner_const));
  // Inner does NOT dominate outer.
  EXPECT_FALSE(loom_dominates_op(&dom_info_, inner_const, outer_const));
  // map_op dominates its body's ops (same scope, map comes first).
  EXPECT_TRUE(loom_dominates_op(&dom_info_, map_op, inner_const));
}

TEST_F(DominanceTest, IsolatedRegionBlocksDominance) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* outer_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &outer_const));
  loom_op_t* iso_op = NULL;
  IREE_ASSERT_OK(loom_test_isolated_region_build(
      &builder_, NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &iso_op));

  enter_region(iso_op, loom_test_isolated_region_body(iso_op));
  loom_op_t* inner_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_const));
  leave_region();
  finalize();

  // Outer does NOT dominate inner across isolation boundary.
  EXPECT_FALSE(loom_dominates_op(&dom_info_, outer_const, inner_const));
  // Inner does not dominate outer (never does, isolation or not).
  EXPECT_FALSE(loom_dominates_op(&dom_info_, inner_const, outer_const));
}

TEST_F(DominanceTest, NestingDepth) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_op_t* c0 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &c0));
  loom_value_id_t arg = loom_test_constant_result(c0);
  loom_op_t* map1 = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, f32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map1));

  enter_region(map1, loom_test_map_body(map1));
  loom_value_id_t inner_arg =
      loom_region_entry_arg_id(loom_test_map_body(map1), 0);
  loom_op_t* map2 = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &inner_arg, 1, f32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map2));

  enter_region(map2, loom_test_map_body(map2));
  loom_op_t* deep = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(42), i32,
                                          LOOM_LOCATION_UNKNOWN, &deep));
  leave_region();
  leave_region();
  finalize();

  // Depth counts parent_op hops: func.def is depth 0 (parent_op=NULL),
  // ops in the func body are depth 1 (parent_op=func.def), etc.
  EXPECT_EQ(loom_op_nesting_depth(c0), 1u);
  EXPECT_EQ(loom_op_nesting_depth(map1), 1u);
  EXPECT_EQ(loom_op_nesting_depth(map2), 2u);
  EXPECT_EQ(loom_op_nesting_depth(deep), 3u);

  // c0 dominates everything (outer scope).
  EXPECT_TRUE(loom_dominates_op(&dom_info_, c0, map2));
  EXPECT_TRUE(loom_dominates_op(&dom_info_, c0, deep));
  // map2 dominates deep (parent scope).
  EXPECT_TRUE(loom_dominates_op(&dom_info_, map2, deep));
  // deep does NOT dominate anything above it.
  EXPECT_FALSE(loom_dominates_op(&dom_info_, deep, map2));
  EXPECT_FALSE(loom_dominates_op(&dom_info_, deep, c0));
}

TEST_F(DominanceTest, ValueDominanceOpResult) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_op_t* c1 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &c1));
  loom_value_id_t v1 = loom_test_constant_result(c1);

  loom_op_t* c2 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &c2));
  finalize();

  // Value defined by c1 dominates c2.
  EXPECT_TRUE(loom_dominates_value(&dom_info_, v1, c2));
  // Value defined by c2 does NOT dominate c1.
  loom_value_id_t v2 = loom_test_constant_result(c2);
  EXPECT_FALSE(loom_dominates_value(&dom_info_, v2, c1));
}

TEST_F(DominanceTest, ValueDominanceBlockArg) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_op_t* c0 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &c0));
  loom_value_id_t arg = loom_test_constant_result(c0);
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, f32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  enter_region(map_op, loom_test_map_body(map_op));
  // The map body has a block arg (the element from the input tile).
  loom_value_id_t block_arg_id =
      loom_region_entry_arg_id(loom_test_map_body(map_op), 0);

  loom_op_t* inner_const = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner_const));
  leave_region();
  finalize();

  // Block arg dominates ops in the same block.
  EXPECT_TRUE(loom_dominates_value(&dom_info_, block_arg_id, inner_const));
}

// An op that owns a region dominates ops inside it. The ancestor
// case (a == b_at_a_depth after walking b up to a's depth) must
// return true even though a and b are not in the same block.
TEST_F(DominanceTest, AncestorOpDominatesNestedOps) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_op_t* c0 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &c0));
  loom_value_id_t arg = loom_test_constant_result(c0);
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, f32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  enter_region(map_op, loom_test_map_body(map_op));
  loom_op_t* inner = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner));
  leave_region();
  finalize();

  // map_op dominates inner (ancestor relationship, not same-block).
  EXPECT_TRUE(loom_dominates_op(&dom_info_, map_op, inner));
  // inner does NOT dominate map_op.
  EXPECT_FALSE(loom_dominates_op(&dom_info_, inner, map_op));
}

// An op AFTER a region-bearing op in the same block must NOT
// dominate ops inside the region.
TEST_F(DominanceTest, SiblingAfterRegionDoesNotDominateNested) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  // Build: c0, map(body: inner), after_map.
  loom_op_t* c0 = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32,
                                          LOOM_LOCATION_UNKNOWN, &c0));
  loom_value_id_t arg = loom_test_constant_result(c0);
  loom_op_t* map_op = NULL;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, &arg, 1, f32, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));

  enter_region(map_op, loom_test_map_body(map_op));
  loom_op_t* inner = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(1), i32,
                                          LOOM_LOCATION_UNKNOWN, &inner));
  leave_region();

  loom_op_t* after_map = NULL;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(2), i32,
                                          LOOM_LOCATION_UNKNOWN, &after_map));
  finalize();

  // after_map is in the same block as map_op but comes after it.
  // after_map must NOT dominate inner (which is inside map_op's body).
  EXPECT_FALSE(loom_dominates_op(&dom_info_, after_map, inner));
  // Conversely, inner must NOT dominate after_map (inner scope).
  EXPECT_FALSE(loom_dominates_op(&dom_info_, inner, after_map));

  // But c0 (before map_op) DOES dominate inner.
  EXPECT_TRUE(loom_dominates_op(&dom_info_, c0, inner));
  // And map_op itself dominates inner (ancestor).
  EXPECT_TRUE(loom_dominates_op(&dom_info_, map_op, inner));
}

}  // namespace
}  // namespace loom
