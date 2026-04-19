// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/interpreter.h"

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

class ProgramStorage {
 public:
  ~ProgramStorage() { loom_pass_program_deinitialize(&program); }

  loom_pass_program_t program = {};
};

class PassInterpreterTest : public ::testing::Test {
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
    IREE_EXPECT_OK(loom_text_parse(source, IREE_SV("pass_interpreter.loom"),
                                   &context_, &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    if (module) {
      modules_.push_back(module);
    }
    return module;
  }

  const loom_op_t* Pipeline(loom_module_t* module, iree_host_size_t index) {
    const loom_op_t* op = loom_block_const_op(loom_module_block(module), index);
    EXPECT_TRUE(loom_pass_pipeline_isa(op));
    return op;
  }

  loom_func_like_t Function(loom_module_t* module, iree_host_size_t index) {
    const loom_op_t* op = loom_block_const_op(loom_module_block(module), index);
    loom_func_like_t function =
        loom_func_like_cast(module, const_cast<loom_op_t*>(op));
    EXPECT_TRUE(loom_func_like_isa(function));
    return function;
  }

  iree_status_t Compile(loom_module_t* module, const loom_op_t* pipeline_op,
                        loom_pass_program_t* out_program) {
    loom_pass_program_compile_options_t options = {
        .registry = loom_test_pass_registry(),
    };
    return loom_pass_program_compile_pipeline(module, pipeline_op, &options,
                                              &block_pool_, out_program);
  }

  static iree_status_t ConfigureTrace(
      void* user_data, const loom_pass_program_instruction_t* instruction,
      void** out_pass_user_data) {
    (void)instruction;
    *out_pass_user_data = user_data;
    return iree_ok_status();
  }

  loom_pass_interpreter_options_t InterpreterOptions(
      loom_test_pass_trace_t* trace) {
    return (loom_pass_interpreter_options_t){
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

TEST_F(PassInterpreterTest, RunsModuleAndFunctionPasses) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
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

  ProgramStorage storage;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 0), &storage.program));

  loom_test_pass_trace_t trace = {};
  loom_pass_interpreter_options_t options = InterpreterOptions(&trace);
  IREE_ASSERT_OK(
      loom_pass_interpreter_run_module(&storage.program, module, &options));

  EXPECT_EQ(trace.module_noop_invocation_count, 1);
  EXPECT_EQ(trace.noop_invocation_count, 2);
}

TEST_F(PassInterpreterTest, RunsFunctionRootProgram) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                    "  test.noop\n"
                    "}\n"
                    "test.func @first() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @second() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  ProgramStorage storage;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 0), &storage.program));

  loom_test_pass_trace_t trace = {};
  loom_pass_interpreter_options_t options = InterpreterOptions(&trace);
  IREE_ASSERT_OK(loom_pass_interpreter_run_function(
      &storage.program, module, Function(module, 1), &options));

  EXPECT_EQ(trace.noop_invocation_count, 1);
}

TEST_F(PassInterpreterTest, AppliesNamePredicateToCurrentFunction) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  for func {\n"
                    "    where name(value = \"matmul\") {\n"
                    "      test.noop\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                    "test.func @matmul() {\n"
                    "  test.yield\n"
                    "}\n"
                    "test.func @not_matmul() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  ProgramStorage storage;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 0), &storage.program));

  loom_test_pass_trace_t trace = {};
  loom_pass_interpreter_options_t options = InterpreterOptions(&trace);
  IREE_ASSERT_OK(
      loom_pass_interpreter_run_module(&storage.program, module, &options));

  EXPECT_EQ(trace.noop_invocation_count, 1);
}

TEST_F(PassInterpreterTest, ExecutesFixedRepeat) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  for func {\n"
                    "    repeat fixed(count = 3) {\n"
                    "      test.noop\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                    "test.func @main() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  ProgramStorage storage;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 0), &storage.program));

  loom_test_pass_trace_t trace = {};
  loom_pass_interpreter_options_t options = InterpreterOptions(&trace);
  IREE_ASSERT_OK(
      loom_pass_interpreter_run_module(&storage.program, module, &options));

  EXPECT_EQ(trace.noop_invocation_count, 3);
}

TEST_F(PassInterpreterTest, StopsRepeatUntilConvergedWhenBodyIsStable) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  for func {\n"
                    "    repeat until_converged(max_iterations = 4) {\n"
                    "      test.noop\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                    "test.func @main() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  ProgramStorage storage;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 0), &storage.program));

  loom_test_pass_trace_t trace = {};
  loom_pass_interpreter_options_t options = InterpreterOptions(&trace);
  IREE_ASSERT_OK(
      loom_pass_interpreter_run_module(&storage.program, module, &options));

  EXPECT_EQ(trace.noop_invocation_count, 1);
}

TEST_F(PassInterpreterTest, ReportsRepeatUntilConvergedExhaustion) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  for func {\n"
                    "    repeat until_converged(max_iterations = 2) {\n"
                    "      test.mark-changed\n"
                    "    }\n"
                    "  }\n"
                    "}\n"
                    "test.func @main() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  ProgramStorage storage;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 0), &storage.program));

  loom_test_pass_trace_t trace = {};
  loom_pass_interpreter_options_t options = InterpreterOptions(&trace);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_pass_interpreter_run_module(&storage.program, module, &options));

  EXPECT_EQ(trace.mark_changed_invocation_count, 2);
}

TEST_F(PassInterpreterTest, ExposesDecodedOptionsToPassInstance) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  for func {\n"
                    "    test.options(count = 5, mode = beta, string = "
                    "\"decoded\")\n"
                    "  }\n"
                    "}\n"
                    "test.func @main() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  ProgramStorage storage;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 0), &storage.program));

  loom_test_pass_trace_t trace = {};
  loom_pass_interpreter_options_t options = InterpreterOptions(&trace);
  IREE_ASSERT_OK(
      loom_pass_interpreter_run_module(&storage.program, module, &options));

  EXPECT_EQ(trace.options_invocation_count, 1);
  EXPECT_EQ(trace.options_decoded_create_count, 1);
  EXPECT_EQ(trace.decoded_options_count_value, 5u);
  EXPECT_EQ(trace.decoded_options_mode_index, 1u);
  EXPECT_TRUE(iree_string_view_equal(trace.decoded_options_string_value,
                                     IREE_SV("decoded")));
}

TEST_F(PassInterpreterTest, PropagatesDescriptorCallbackFailure) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  test.fail\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  ProgramStorage storage;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 0), &storage.program));

  loom_test_pass_trace_t trace = {};
  loom_pass_interpreter_options_t options = InterpreterOptions(&trace);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      loom_pass_interpreter_run_module(&storage.program, module, &options));
  EXPECT_EQ(trace.fail_invocation_count, 1);
}

TEST_F(PassInterpreterTest, ExecutesPassFailAndHalt) {
  loom_module_t* fail_module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  fail \"stop\"\n"
                    "}\n"));
  ASSERT_NE(fail_module, nullptr);
  ProgramStorage fail_storage;
  IREE_ASSERT_OK(
      Compile(fail_module, Pipeline(fail_module, 0), &fail_storage.program));

  loom_test_pass_trace_t trace = {};
  loom_pass_interpreter_options_t options = InterpreterOptions(&trace);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_pass_interpreter_run_module(
                            &fail_storage.program, fail_module, &options));

  loom_module_t* halt_module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  halt \"inspect\"\n"
                    "}\n"));
  ASSERT_NE(halt_module, nullptr);
  ProgramStorage halt_storage;
  IREE_ASSERT_OK(
      Compile(halt_module, Pipeline(halt_module, 0), &halt_storage.program));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        loom_pass_interpreter_run_module(
                            &halt_storage.program, halt_module, &options));
}

}  // namespace
}  // namespace loom
