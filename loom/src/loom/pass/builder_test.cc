// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/builder.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class PassBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_op_registry_initialize_context(iree_allocator_system(),
                                                       &context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t AllocateModule(iree_string_view_t name, ModulePtr* out_module) {
    *out_module = nullptr;
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(loom_module_allocate(&context_, name, &block_pool_,
                                              nullptr, iree_allocator_system(),
                                              &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

static iree_status_t BuildOneRun(loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* run_op = nullptr;
  return loom_pass_ir_build_run(
      builder, IREE_SV("cse"), loom_make_named_attr_slice(nullptr, 0), &run_op);
}

TEST_F(PassBuilderTest, BuildsLinkedPipelineSymbol) {
  ModulePtr module;
  IREE_ASSERT_OK(AllocateModule(IREE_SV("pipeline"), &module));

  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_pass_ir_build_pipeline(
      module.get(), IREE_SV("cleanup"), LOOM_PASS_ANCHOR_MODULE, BuildOneRun,
      nullptr, &pipeline_op));

  loom_symbol_ref_t symbol_ref = loom_pass_pipeline_symbol(pipeline_op);
  ASSERT_TRUE(loom_symbol_ref_is_valid(symbol_ref));
  ASSERT_EQ(symbol_ref.module_id, 0u);
  ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  EXPECT_EQ(module->symbols.entries[symbol_ref.symbol_id].defining_op,
            pipeline_op);

  loom_block_t* body_block =
      loom_region_entry_block(loom_pass_pipeline_body(pipeline_op));
  ASSERT_NE(body_block, nullptr);
  ASSERT_EQ(body_block->op_count, 2u);
  EXPECT_TRUE(loom_pass_run_isa(body_block->first_op));
  EXPECT_TRUE(loom_pass_yield_isa(body_block->last_op));
}

TEST(PassBuilderStandaloneTest, RequiresPassDialect) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);

  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  IREE_ASSERT_OK(loom_context_finalize(&context));

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("no_pass"), &block_pool,
                                      nullptr, iree_allocator_system(),
                                      &module));

  loom_op_t* pipeline_op = nullptr;
  iree_status_t status = loom_pass_ir_build_pipeline(
      module, IREE_SV("cleanup"), LOOM_PASS_ANCHOR_MODULE, BuildOneRun, nullptr,
      &pipeline_op);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_FAILED_PRECONDITION);
  iree_status_ignore(status);
  EXPECT_EQ(pipeline_op, nullptr);

  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
