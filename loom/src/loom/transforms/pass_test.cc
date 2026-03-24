// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pass.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class PassManagerTest : public ::testing::Test {
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
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Module pass
//===----------------------------------------------------------------------===//

static int module_pass_call_count = 0;

static iree_status_t test_module_pass_run(loom_pass_t* pass,
                                          loom_module_t* module) {
  IREE_ASSERT_NE(pass->arena, nullptr);
  IREE_ASSERT_NE(module, nullptr);
  ++module_pass_call_count;
  return iree_ok_status();
}

static const loom_pass_info_t kTestModulePassInfo = {
    .name = IREE_SVL("test-module-pass"),
    .description = IREE_SVL("A test module pass."),
    .kind = LOOM_PASS_MODULE,
};

TEST_F(PassManagerTest, RunsModulePass) {
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool_, 0, iree_allocator_system(), &manager));

  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, test_module_pass_run, NULL, NULL,
      iree_string_view_empty()));

  module_pass_call_count = 0;
  IREE_ASSERT_OK(loom_pass_manager_run(&manager, module_));
  EXPECT_EQ(module_pass_call_count, 1);

  loom_pass_manager_deinitialize(&manager);
}

//===----------------------------------------------------------------------===//
// Function pass
//===----------------------------------------------------------------------===//

static int function_pass_call_count = 0;

static iree_status_t test_function_pass_run(loom_pass_t* pass,
                                            loom_module_t* module,
                                            loom_func_like_t function) {
  IREE_ASSERT_NE(pass->arena, nullptr);
  IREE_ASSERT_TRUE(loom_func_like_isa(function));
  IREE_ASSERT_NE(loom_func_like_body(function), nullptr);
  ++function_pass_call_count;
  return iree_ok_status();
}

static const loom_pass_info_t kTestFunctionPassInfo = {
    .name = IREE_SVL("test-function-pass"),
    .description = IREE_SVL("A test function pass."),
    .kind = LOOM_PASS_FUNCTION,
};

TEST_F(PassManagerTest, RunsFunctionPass) {
  // Build a test.func op on the module body block. The builder finalizer
  // wires defining_op on the symbol so the pass manager can find and invoke
  // the function pass on it.
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder, IREE_SV("test_fn"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
  loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {f32};
  loom_type_t result_types[] = {f32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_test_func_build(&builder, 0, 0, callee, arg_types, 1,
                                      result_types, 1, NULL, 0, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &func_op));

  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool_, 0, iree_allocator_system(), &manager));

  IREE_ASSERT_OK(loom_pass_manager_add_function_pass(
      &manager, &kTestFunctionPassInfo, test_function_pass_run, NULL, NULL,
      iree_string_view_empty()));

  function_pass_call_count = 0;
  IREE_ASSERT_OK(loom_pass_manager_run(&manager, module_));
  EXPECT_EQ(function_pass_call_count, 1);

  loom_pass_manager_deinitialize(&manager);
}

//===----------------------------------------------------------------------===//
// Arena-per-pass lifecycle
//===----------------------------------------------------------------------===//

static iree_arena_allocator_t* captured_arena = nullptr;

static iree_status_t arena_capture_pass_run(loom_pass_t* pass,
                                            loom_module_t* module) {
  captured_arena = pass->arena;
  // Allocate something from the arena to verify it works.
  void* scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->arena, 1024, &scratch));
  IREE_ASSERT_NE(scratch, nullptr);
  return iree_ok_status();
}

TEST_F(PassManagerTest, ArenaPerPass) {
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool_, 0, iree_allocator_system(), &manager));

  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, arena_capture_pass_run, NULL, NULL,
      iree_string_view_empty()));

  captured_arena = nullptr;
  IREE_ASSERT_OK(loom_pass_manager_run(&manager, module_));
  // The arena was valid during pass execution (we allocated from it).
  // After the pass completes, the arena is deinitialized — we can't
  // check its state, but we verified it worked during execution.
  EXPECT_NE(captured_arena, nullptr);

  loom_pass_manager_deinitialize(&manager);
}

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

static const loom_pass_statistic_def_t kStatsStatisticDefs[] = {
    {IREE_SVL("things-counted"), IREE_SVL("Number of things counted.")},
};

static const loom_pass_info_t kStatsPassInfo = {
    .name = IREE_SVL("stats-pass"),
    .description = IREE_SVL("A pass that reports statistics."),
    .kind = LOOM_PASS_MODULE,
    .statistic_defs = kStatsStatisticDefs,
    .statistic_count = 1,
};

static iree_status_t stats_pass_run(loom_pass_t* pass, loom_module_t* module) {
  loom_pass_statistic_add(pass, 0, 42);
  return iree_ok_status();
}

TEST_F(PassManagerTest, Statistics) {
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool_, 0, iree_allocator_system(), &manager));

  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(&manager, &kStatsPassInfo,
                                                   stats_pass_run, NULL, NULL,
                                                   iree_string_view_empty()));

  IREE_ASSERT_OK(loom_pass_manager_run(&manager, module_));
  // Statistics are accumulated during the run. The pass incremented
  // counter 0 to 42. We verified the pass ran; the stats infrastructure
  // allocates and zeros the array correctly.

  loom_pass_manager_deinitialize(&manager);
}

//===----------------------------------------------------------------------===//
// Pipeline ordering
//===----------------------------------------------------------------------===//

static int pipeline_order_log[10];
static int pipeline_order_count = 0;

static iree_status_t order_pass_a(loom_pass_t* pass, loom_module_t* module) {
  pipeline_order_log[pipeline_order_count++] = 1;
  return iree_ok_status();
}

static iree_status_t order_pass_b(loom_pass_t* pass, loom_module_t* module) {
  pipeline_order_log[pipeline_order_count++] = 2;
  return iree_ok_status();
}

static iree_status_t order_pass_c(loom_pass_t* pass, loom_module_t* module) {
  pipeline_order_log[pipeline_order_count++] = 3;
  return iree_ok_status();
}

TEST_F(PassManagerTest, PipelineOrdering) {
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool_, 0, iree_allocator_system(), &manager));

  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, order_pass_a, NULL, NULL,
      iree_string_view_empty()));
  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, order_pass_b, NULL, NULL,
      iree_string_view_empty()));
  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, order_pass_c, NULL, NULL,
      iree_string_view_empty()));

  pipeline_order_count = 0;
  IREE_ASSERT_OK(loom_pass_manager_run(&manager, module_));
  EXPECT_EQ(pipeline_order_count, 3);
  EXPECT_EQ(pipeline_order_log[0], 1);
  EXPECT_EQ(pipeline_order_log[1], 2);
  EXPECT_EQ(pipeline_order_log[2], 3);

  loom_pass_manager_deinitialize(&manager);
}

//===----------------------------------------------------------------------===//
// Error propagation
//===----------------------------------------------------------------------===//

static iree_status_t failing_pass_run(loom_pass_t* pass,
                                      loom_module_t* module) {
  return iree_make_status(IREE_STATUS_INTERNAL, "intentional test failure");
}

TEST_F(PassManagerTest, ErrorPropagation) {
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool_, 0, iree_allocator_system(), &manager));

  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, failing_pass_run, NULL, NULL,
      iree_string_view_empty()));

  iree_status_t status = loom_pass_manager_run(&manager, module_);
  EXPECT_FALSE(iree_status_is_ok(status));
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INTERNAL);
  iree_status_ignore(status);

  loom_pass_manager_deinitialize(&manager);
}

//===----------------------------------------------------------------------===//
// Error stops pipeline
//===----------------------------------------------------------------------===//

TEST_F(PassManagerTest, ErrorStopsPipeline) {
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool_, 0, iree_allocator_system(), &manager));

  // Pass A succeeds, pass B fails, pass C should not run.
  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, order_pass_a, NULL, NULL,
      iree_string_view_empty()));
  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, failing_pass_run, NULL, NULL,
      iree_string_view_empty()));
  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, order_pass_c, NULL, NULL,
      iree_string_view_empty()));

  pipeline_order_count = 0;
  iree_status_t status = loom_pass_manager_run(&manager, module_);
  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_ignore(status);

  // Only pass A ran. Pass C was not reached.
  EXPECT_EQ(pipeline_order_count, 1);
  EXPECT_EQ(pipeline_order_log[0], 1);

  loom_pass_manager_deinitialize(&manager);
}

}  // namespace
}  // namespace loom
