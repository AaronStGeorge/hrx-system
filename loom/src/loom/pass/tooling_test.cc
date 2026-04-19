// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/tooling.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/pass/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/test/registry.h"

namespace loom {
namespace {

class PassToolingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t pass_count = 0;
    const loom_op_vtable_t* const* pass_vtables =
        loom_pass_dialect_vtables(&pass_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_PASS, pass_vtables, (uint16_t)pass_count));

    iree_host_size_t test_count = 0;
    const loom_op_vtable_t* const* test_vtables =
        loom_test_dialect_vtables(&test_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, test_vtables, (uint16_t)test_count));

    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : modules_) {
      loom_module_free(module);
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source) {
    loom_text_parse_options_t options = {
        .diagnostic_sink = {loom_diagnostic_stderr_sink, NULL},
        .max_errors = 20,
    };
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(source, IREE_SV("pass_tooling.loom"),
                                   &context_, &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    if (module) {
      modules_.push_back(module);
    }
    return module;
  }

  static iree_status_t ConfigureTrace(
      void* user_data, const loom_pass_program_instruction_t* instruction,
      void** out_pass_user_data) {
    (void)instruction;
    *out_pass_user_data = user_data;
    return iree_ok_status();
  }

  loom_pass_tool_run_options_t ToolOptions(loom_test_pass_trace_t* trace) {
    return (loom_pass_tool_run_options_t){
        .registry = loom_test_pass_registry(),
        .block_pool = &block_pool_,
        .configure =
            {
                .fn = ConfigureTrace,
                .user_data = trace,
            },
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  std::vector<loom_module_t*> modules_;
};

static void ExpectTraceEvent(const loom_test_pass_trace_t& trace,
                             iree_host_size_t event_index,
                             iree_string_view_t pass_name,
                             iree_string_view_t symbol_name) {
  ASSERT_LT(event_index, trace.event_count);
  EXPECT_TRUE(
      iree_string_view_equal(trace.events[event_index].pass_name, pass_name));
  EXPECT_TRUE(iree_string_view_equal(trace.events[event_index].symbol_name,
                                     symbol_name));
}

TEST_F(PassToolingTest, RunsNamedModulePipeline) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @cleanup pipeline {\n"
                    "  test.module-noop\n"
                    "  for func {\n"
                    "    test.noop\n"
                    "  }\n"
                    "}\n"
                    "test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  loom_pass_tool_run_options_t options = ToolOptions(&trace);
  IREE_ASSERT_OK(loom_pass_tool_run_pipeline_symbol(module, IREE_SV("@cleanup"),
                                                    &options));

  EXPECT_EQ(trace.module_noop_invocation_count, 1);
  EXPECT_EQ(trace.noop_invocation_count, 2);
}

TEST_F(PassToolingTest, RunsNamedFunctionPipelineAcrossFunctions) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @function_cleanup pipeline {\n"
                    "  test.noop\n"
                    "}\n"
                    "test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  loom_pass_tool_run_options_t options = ToolOptions(&trace);
  IREE_ASSERT_OK(loom_pass_tool_run_pipeline_symbol(
      module, IREE_SV("function_cleanup"), &options));

  EXPECT_EQ(trace.noop_invocation_count, 2);
}

TEST_F(PassToolingTest, RunsFlatPipelineWithDescriptorOptions) {
  loom_module_t* module =
      Parse(IREE_SV("test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  loom_pass_tool_run_options_t options = ToolOptions(&trace);
  IREE_ASSERT_OK(loom_pass_tool_run_flat_pipeline(
      module,
      IREE_SV("test.module-noop,test.options{count=5,mode=beta,string=flat}"),
      &options));

  EXPECT_EQ(trace.module_noop_invocation_count, 1);
  EXPECT_EQ(trace.options_invocation_count, 2);
  EXPECT_EQ(trace.options_decoded_create_count, 2);
  EXPECT_EQ(trace.decoded_options_count_value, 5u);
  EXPECT_EQ(trace.decoded_options_mode_index, 1u);
  EXPECT_TRUE(iree_string_view_equal(trace.decoded_options_string_value,
                                     IREE_SV("flat")));
}

TEST_F(PassToolingTest, GroupsAdjacentFlatFunctionPassesPerFunction) {
  loom_module_t* module =
      Parse(IREE_SV("test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  loom_pass_tool_run_options_t options = ToolOptions(&trace);
  IREE_ASSERT_OK(loom_pass_tool_run_flat_pipeline(
      module, IREE_SV("test.noop,test.mark-changed"), &options));

  ASSERT_EQ(trace.event_count, 4u);
  ExpectTraceEvent(trace, 0, IREE_SV("test.noop"), IREE_SV("first"));
  ExpectTraceEvent(trace, 1, IREE_SV("test.mark-changed"), IREE_SV("first"));
  ExpectTraceEvent(trace, 2, IREE_SV("test.noop"), IREE_SV("second"));
  ExpectTraceEvent(trace, 3, IREE_SV("test.mark-changed"), IREE_SV("second"));
}

TEST_F(PassToolingTest, FlatModulePassesSeparateFunctionPassGroups) {
  loom_module_t* module =
      Parse(IREE_SV("test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  loom_pass_tool_run_options_t options = ToolOptions(&trace);
  IREE_ASSERT_OK(loom_pass_tool_run_flat_pipeline(
      module, IREE_SV("test.noop,test.module-noop,test.mark-changed"),
      &options));

  ASSERT_EQ(trace.event_count, 5u);
  ExpectTraceEvent(trace, 0, IREE_SV("test.noop"), IREE_SV("first"));
  ExpectTraceEvent(trace, 1, IREE_SV("test.noop"), IREE_SV("second"));
  ExpectTraceEvent(trace, 2, IREE_SV("test.module-noop"), IREE_SV("<module>"));
  ExpectTraceEvent(trace, 3, IREE_SV("test.mark-changed"), IREE_SV("first"));
  ExpectTraceEvent(trace, 4, IREE_SV("test.mark-changed"), IREE_SV("second"));
}

TEST_F(PassToolingTest, RejectsUnknownFlatPipelinePass) {
  loom_module_t* module = Parse(IREE_SV(""));
  ASSERT_NE(module, nullptr);

  loom_test_pass_trace_t trace = {};
  loom_pass_tool_run_options_t options = ToolOptions(&trace);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_pass_tool_run_flat_pipeline(
                            module, IREE_SV("missing.pass"), &options));
}

}  // namespace
}  // namespace loom
