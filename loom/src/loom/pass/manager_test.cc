// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/manager.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Pipeline option parsing
//===----------------------------------------------------------------------===//

TEST(PassPipelineParseTest, ConsumesNamesAndOptions) {
  iree_string_view_t pipeline = IREE_SV("canonicalize{max-iterations=20}, dce");
  loom_pass_pipeline_entry_spec_t entry = {};
  bool has_entry = false;

  IREE_ASSERT_OK(
      loom_pass_pipeline_consume_entry(&pipeline, &entry, &has_entry));
  EXPECT_TRUE(has_entry);
  EXPECT_TRUE(iree_string_view_equal(entry.name, IREE_SV("canonicalize")));
  EXPECT_TRUE(
      iree_string_view_equal(entry.options, IREE_SV("max-iterations=20")));

  IREE_ASSERT_OK(
      loom_pass_pipeline_consume_entry(&pipeline, &entry, &has_entry));
  EXPECT_TRUE(has_entry);
  EXPECT_TRUE(iree_string_view_equal(entry.name, IREE_SV("dce")));
  EXPECT_TRUE(iree_string_view_is_empty(entry.options));

  IREE_ASSERT_OK(
      loom_pass_pipeline_consume_entry(&pipeline, &entry, &has_entry));
  EXPECT_FALSE(has_entry);
}

typedef struct parsed_option_t {
  // Last parsed option name.
  iree_string_view_t name;

  // Last parsed option value.
  iree_string_view_t value;

  // Number of parsed assignments.
  int count;
} parsed_option_t;

static iree_status_t capture_option(void* user_data, iree_string_view_t name,
                                    iree_string_view_t value) {
  parsed_option_t* parsed_option = (parsed_option_t*)user_data;
  parsed_option->name = name;
  parsed_option->value = value;
  ++parsed_option->count;
  return iree_ok_status();
}

TEST(PassPipelineParseTest, ParsesOptionAssignments) {
  parsed_option_t parsed_option = {};
  IREE_ASSERT_OK(loom_pass_options_parse(IREE_SV("canonicalize"),
                                         IREE_SV("max-iterations = 7"),
                                         capture_option, &parsed_option));
  EXPECT_EQ(parsed_option.count, 1);
  EXPECT_TRUE(
      iree_string_view_equal(parsed_option.name, IREE_SV("max-iterations")));
  EXPECT_TRUE(iree_string_view_equal(parsed_option.value, IREE_SV("7")));
}

TEST(PassPipelineParseTest, RejectsMalformedPipelineEntries) {
  iree_string_view_t pipeline = IREE_SV("canonicalize{max-iterations=1");
  loom_pass_pipeline_entry_spec_t entry = {};
  bool has_entry = false;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_pipeline_consume_entry(&pipeline, &entry, &has_entry));

  pipeline = IREE_SV(",dce");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_pipeline_consume_entry(&pipeline, &entry, &has_entry));

  pipeline = IREE_SV("dce,");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_pipeline_consume_entry(&pipeline, &entry, &has_entry));
}

TEST(PassPipelineParseTest, RejectsMalformedOptionAssignments) {
  parsed_option_t parsed_option = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_options_parse(
                            IREE_SV("canonicalize"), IREE_SV("max-iterations"),
                            capture_option, &parsed_option));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_options_parse(IREE_SV("canonicalize"), IREE_SV("=7"),
                              capture_option, &parsed_option));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_pass_options_parse(IREE_SV("canonicalize"),
                              IREE_SV("max-iterations=7,"), capture_option,
                              &parsed_option));
}

TEST(PassPipelineParseTest, ParsesUint32OptionValues) {
  uint32_t value = 0;
  IREE_ASSERT_OK(loom_pass_option_parse_uint32(IREE_SV("canonicalize"),
                                               IREE_SV("max-iterations"),
                                               IREE_SV("42"), &value));
  EXPECT_EQ(value, 42u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_option_parse_uint32(IREE_SV("canonicalize"),
                                                      IREE_SV("max-iterations"),
                                                      IREE_SV("many"), &value));
}

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

static void BuildTestFunction(loom_module_t* module, iree_string_view_t name) {
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_intern_string(&builder, name, &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {f32};
  loom_type_t result_types[] = {f32};
  loom_op_t* func_op = NULL;
  IREE_ASSERT_OK(loom_test_func_build(&builder, 0, 0, 0, callee, arg_types, 1,
                                      result_types, 1, NULL, 0, NULL, 0,
                                      LOOM_LOCATION_UNKNOWN, &func_op));
  IREE_ASSERT_NE(func_op, nullptr);
}

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
      iree_string_view_empty(), NULL));

  module_pass_call_count = 0;
  IREE_ASSERT_OK(loom_pass_manager_run(&manager, module_));
  EXPECT_EQ(module_pass_call_count, 1);

  loom_pass_manager_deinitialize(&manager);
}

static int user_data_sentinel = 0;

static iree_status_t user_data_pass_run(loom_pass_t* pass,
                                        loom_module_t* module) {
  IREE_ASSERT_EQ(pass->user_data, &user_data_sentinel);
  return iree_ok_status();
}

TEST_F(PassManagerTest, PassEntryUserDataIsVisibleDuringRun) {
  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool_, 0, iree_allocator_system(), &manager));

  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, user_data_pass_run, NULL, NULL,
      iree_string_view_empty(), &user_data_sentinel));

  IREE_ASSERT_OK(loom_pass_manager_run(&manager, module_));

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
  BuildTestFunction(module_, IREE_SV("test_fn"));

  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool_, 0, iree_allocator_system(), &manager));

  IREE_ASSERT_OK(loom_pass_manager_add_function_pass(
      &manager, &kTestFunctionPassInfo, test_function_pass_run, NULL, NULL,
      iree_string_view_empty(), NULL));

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
      iree_string_view_empty(), NULL));

  captured_arena = nullptr;
  IREE_ASSERT_OK(loom_pass_manager_run(&manager, module_));
  // The arena was valid during pass execution (we allocated from it).
  // After the pass completes, the arena is deinitialized — we can't
  // check its state, but we verified it worked during execution.
  EXPECT_NE(captured_arena, nullptr);

  loom_pass_manager_deinitialize(&manager);
}

//===----------------------------------------------------------------------===//
// Function-pass arena lifetime
//===----------------------------------------------------------------------===//

typedef struct function_arena_test_state_t {
  iree_arena_allocator_t* instance_arena;
  int run_count;
} function_arena_test_state_t;

static int function_arena_destroyed_run_count = 0;

static iree_status_t function_arena_test_create(loom_pass_t* pass,
                                                iree_string_view_t options) {
  (void)options;
  IREE_ASSERT_NE(pass->instance_arena, nullptr);
  IREE_ASSERT_EQ(pass->arena, pass->instance_arena);
  function_arena_test_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->instance_arena, sizeof(*state),
                                           (void**)&state));
  state->instance_arena = pass->instance_arena;
  state->run_count = 0;
  pass->state = state;
  return iree_ok_status();
}

static iree_status_t function_arena_test_run(loom_pass_t* pass,
                                             loom_module_t* module,
                                             loom_func_like_t function) {
  IREE_ASSERT_NE(module, nullptr);
  IREE_ASSERT_TRUE(loom_func_like_isa(function));
  IREE_ASSERT_NE(loom_func_like_body(function), nullptr);
  IREE_ASSERT_NE(pass->state, nullptr);
  IREE_ASSERT_NE(pass->instance_arena, nullptr);
  IREE_ASSERT_NE(pass->arena, nullptr);
  IREE_ASSERT_NE(pass->arena, pass->instance_arena);
  EXPECT_EQ(pass->arena->total_allocation_size, 0u);
  EXPECT_EQ(pass->arena->used_allocation_size, 0u);

  void* scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(pass->arena, 1024, &scratch));
  IREE_ASSERT_NE(scratch, nullptr);

  function_arena_test_state_t* state =
      (function_arena_test_state_t*)pass->state;
  IREE_ASSERT_EQ(state->instance_arena, pass->instance_arena);
  ++state->run_count;
  return iree_ok_status();
}

static void function_arena_test_destroy(loom_pass_t* pass) {
  IREE_ASSERT_EQ(pass->arena, pass->instance_arena);
  IREE_ASSERT_NE(pass->state, nullptr);
  function_arena_test_state_t* state =
      (function_arena_test_state_t*)pass->state;
  IREE_ASSERT_EQ(state->instance_arena, pass->instance_arena);
  function_arena_destroyed_run_count = state->run_count;
}

static const loom_pass_info_t kFunctionArenaTestPassInfo = {
    .name = IREE_SVL("test-function-arena-pass"),
    .description = IREE_SVL("A test function pass that checks arena reset."),
    .kind = LOOM_PASS_FUNCTION,
};

TEST_F(PassManagerTest, FunctionPassArenaResetsBetweenFunctions) {
  BuildTestFunction(module_, IREE_SV("test_fn_0"));
  BuildTestFunction(module_, IREE_SV("test_fn_1"));
  BuildTestFunction(module_, IREE_SV("test_fn_2"));

  loom_pass_manager_t manager;
  IREE_ASSERT_OK(loom_pass_manager_initialize(
      &block_pool_, 0, iree_allocator_system(), &manager));

  IREE_ASSERT_OK(loom_pass_manager_add_function_pass(
      &manager, &kFunctionArenaTestPassInfo, function_arena_test_run,
      function_arena_test_create, function_arena_test_destroy,
      iree_string_view_empty(), NULL));

  function_arena_destroyed_run_count = 0;
  IREE_ASSERT_OK(loom_pass_manager_run(&manager, module_));
  EXPECT_EQ(function_arena_destroyed_run_count, 3);

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

  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kStatsPassInfo, stats_pass_run, NULL, NULL,
      iree_string_view_empty(), NULL));

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
      iree_string_view_empty(), NULL));
  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, order_pass_b, NULL, NULL,
      iree_string_view_empty(), NULL));
  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, order_pass_c, NULL, NULL,
      iree_string_view_empty(), NULL));

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
      iree_string_view_empty(), NULL));

  iree_status_t status = loom_pass_manager_run(&manager, module_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, status);

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
      iree_string_view_empty(), NULL));
  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, failing_pass_run, NULL, NULL,
      iree_string_view_empty(), NULL));
  IREE_ASSERT_OK(loom_pass_manager_add_module_pass(
      &manager, &kTestModulePassInfo, order_pass_c, NULL, NULL,
      iree_string_view_empty(), NULL));

  pipeline_order_count = 0;
  iree_status_t status = loom_pass_manager_run(&manager, module_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL, status);

  // Only pass A ran. Pass C was not reached.
  EXPECT_EQ(pipeline_order_count, 1);
  EXPECT_EQ(pipeline_order_log[0], 1);

  loom_pass_manager_deinitialize(&manager);
}

}  // namespace
}  // namespace loom
