// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/program.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/pass/ops.h"
#include "loom/pass/test/registry.h"

namespace loom {
namespace {

class ProgramStorage {
 public:
  ~ProgramStorage() { loom_pass_program_deinitialize(&program); }

  loom_pass_program_t program = {};
};

class PassProgramTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_pass_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_PASS,
                                                 vtables, (uint16_t)count));
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
    loom_module_t* module = NULL;
    IREE_EXPECT_OK(loom_text_parse(source, IREE_SV("pass_program.loom"),
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

  iree_status_t Compile(loom_module_t* module, const loom_op_t* pipeline_op,
                        loom_pass_program_t* out_program) {
    loom_pass_program_compile_options_t options = {
        .registry = loom_test_pass_registry(),
    };
    return loom_pass_program_compile_pipeline(module, pipeline_op, &options,
                                              &block_pool_, out_program);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  std::vector<loom_module_t*> modules_;
};

TEST_F(PassProgramTest, CompilesStructuredProgram) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @callee pipeline {\n"
                    "  test.module-noop\n"
                    "}\n"
                    "\n"
                    "pass.pipeline<module> @main pipeline {\n"
                    "  test.module-noop\n"
                    "  for func {\n"
                    "    where name(value = \"matmul\") {\n"
                    "      test.options(count = 7, mode = beta, string = "
                    "\"payload\")\n"
                    "    }\n"
                    "    repeat fixed(count = 2) {\n"
                    "      test.noop\n"
                    "    }\n"
                    "  }\n"
                    "  call @callee\n"
                    "  fail \"stop\"\n"
                    "  halt \"inspect\"\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  ProgramStorage storage;
  loom_pass_program_t& program = storage.program;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 1), &program));

  ASSERT_EQ(program.instruction_count, 9u);
  EXPECT_EQ(program.root_kind, LOOM_PASS_MODULE);

  const loom_pass_program_instruction_t& module_invoke =
      program.instructions[0];
  EXPECT_EQ(module_invoke.kind, LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE);
  EXPECT_EQ(module_invoke.anchor_kind, LOOM_PASS_MODULE);
  ASSERT_NE(module_invoke.invoke.descriptor, nullptr);
  EXPECT_TRUE(iree_string_view_equal(module_invoke.invoke.descriptor->key,
                                     IREE_SV("test.module-noop")));

  const loom_pass_program_instruction_t& for_instruction =
      program.instructions[1];
  EXPECT_EQ(for_instruction.kind,
            LOOM_PASS_PROGRAM_INSTRUCTION_FOR_EACH_SYMBOL);
  EXPECT_EQ(for_instruction.anchor_kind, LOOM_PASS_MODULE);
  EXPECT_EQ(for_instruction.for_each_symbol.symbol_kind, LOOM_PASS_FUNCTION);
  EXPECT_EQ(for_instruction.for_each_symbol.snapshot_kind,
            LOOM_PASS_PROGRAM_SYMBOL_SNAPSHOT_FUNCTIONS_BY_SYMBOL_ID);
  EXPECT_EQ(for_instruction.for_each_symbol.body_start, 2u);
  EXPECT_EQ(for_instruction.for_each_symbol.body_end, 6u);

  const loom_pass_program_instruction_t& where_instruction =
      program.instructions[2];
  EXPECT_EQ(where_instruction.kind, LOOM_PASS_PROGRAM_INSTRUCTION_WHERE);
  EXPECT_EQ(where_instruction.anchor_kind, LOOM_PASS_FUNCTION);
  EXPECT_TRUE(iree_string_view_equal(where_instruction.where.predicate,
                                     IREE_SV("name")));
  EXPECT_EQ(where_instruction.where.body_start, 3u);
  EXPECT_EQ(where_instruction.where.body_end, 4u);
  ASSERT_EQ(where_instruction.where.attrs.attr_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      where_instruction.where.attrs.attrs[0].name, IREE_SV("value")));
  EXPECT_EQ(where_instruction.where.attrs.attrs[0].value.kind,
            LOOM_ATTR_STRING);
  EXPECT_TRUE(iree_string_view_equal(
      where_instruction.where.attrs.attrs[0].value.string_value,
      IREE_SV("matmul")));

  const loom_pass_program_instruction_t& options_invoke =
      program.instructions[3];
  EXPECT_EQ(options_invoke.kind, LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE);
  EXPECT_EQ(options_invoke.anchor_kind, LOOM_PASS_FUNCTION);
  EXPECT_TRUE(iree_string_view_equal(options_invoke.invoke.descriptor->key,
                                     IREE_SV("test.options")));
  const loom_pass_decoded_options_t* decoded_options =
      options_invoke.invoke.decoded_options;
  ASSERT_NE(decoded_options, nullptr);
  ASSERT_EQ(decoded_options->option_count, 3u);
  ASSERT_NE(decoded_options->options, nullptr);
  EXPECT_TRUE(decoded_options->options[0].present);
  EXPECT_EQ(decoded_options->options[0].uint32_value, 7u);
  EXPECT_TRUE(decoded_options->options[1].present);
  EXPECT_EQ(decoded_options->options[1].enum_value_index, 1u);
  EXPECT_TRUE(decoded_options->options[2].present);
  EXPECT_TRUE(iree_string_view_equal(decoded_options->options[2].string_value,
                                     IREE_SV("payload")));

  const loom_pass_program_instruction_t& repeat_instruction =
      program.instructions[4];
  EXPECT_EQ(repeat_instruction.kind, LOOM_PASS_PROGRAM_INSTRUCTION_REPEAT);
  EXPECT_EQ(repeat_instruction.anchor_kind, LOOM_PASS_FUNCTION);
  EXPECT_EQ(repeat_instruction.repeat.mode, LOOM_PASS_REPEAT_MODE_FIXED);
  EXPECT_EQ(repeat_instruction.repeat.count, 2);
  EXPECT_EQ(repeat_instruction.repeat.body_start, 5u);
  EXPECT_EQ(repeat_instruction.repeat.body_end, 6u);

  const loom_pass_program_instruction_t& function_invoke =
      program.instructions[5];
  EXPECT_EQ(function_invoke.kind, LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE);
  EXPECT_EQ(function_invoke.anchor_kind, LOOM_PASS_FUNCTION);
  EXPECT_TRUE(iree_string_view_equal(function_invoke.invoke.descriptor->key,
                                     IREE_SV("test.noop")));

  const loom_pass_program_instruction_t& callee_invoke =
      program.instructions[6];
  EXPECT_EQ(callee_invoke.kind, LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE);
  EXPECT_TRUE(iree_string_view_equal(callee_invoke.invoke.descriptor->key,
                                     IREE_SV("test.module-noop")));
  EXPECT_EQ(callee_invoke.source.pipeline_op, Pipeline(module, 0));
  ASSERT_EQ(callee_invoke.source.call_stack_count, 1u);
  ASSERT_NE(callee_invoke.source.call_stack, nullptr);
  EXPECT_TRUE(loom_pass_call_isa(callee_invoke.source.call_stack[0]));

  const loom_pass_program_instruction_t& fail_instruction =
      program.instructions[7];
  EXPECT_EQ(fail_instruction.kind, LOOM_PASS_PROGRAM_INSTRUCTION_FAIL);
  EXPECT_TRUE(iree_string_view_equal(fail_instruction.message.message,
                                     IREE_SV("stop")));

  const loom_pass_program_instruction_t& halt_instruction =
      program.instructions[8];
  EXPECT_EQ(halt_instruction.kind, LOOM_PASS_PROGRAM_INSTRUCTION_HALT);
  EXPECT_TRUE(iree_string_view_equal(halt_instruction.message.message,
                                     IREE_SV("inspect")));
}

TEST_F(PassProgramTest, CompilesFunctionRootPipeline) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                    "  test.noop\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  ProgramStorage storage;
  loom_pass_program_t& program = storage.program;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 0), &program));

  ASSERT_EQ(program.instruction_count, 1u);
  EXPECT_EQ(program.root_kind, LOOM_PASS_FUNCTION);
  EXPECT_EQ(program.instructions[0].kind, LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE);
  EXPECT_EQ(program.instructions[0].anchor_kind, LOOM_PASS_FUNCTION);
  EXPECT_EQ(program.instructions[0].invoke.info->kind, LOOM_PASS_FUNCTION);
}

TEST_F(PassProgramTest, CompilesCanonicalRunOp) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline {\n"
                    "  pass.run<test.options> {count = 8, mode = alpha, "
                    "string = \"canonical\"}\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  ProgramStorage storage;
  loom_pass_program_t& program = storage.program;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 0), &program));

  ASSERT_EQ(program.instruction_count, 1u);
  const loom_pass_program_instruction_t& invoke = program.instructions[0];
  EXPECT_EQ(invoke.kind, LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE);
  EXPECT_EQ(invoke.anchor_kind, LOOM_PASS_FUNCTION);
  EXPECT_TRUE(iree_string_view_equal(invoke.invoke.descriptor->key,
                                     IREE_SV("test.options")));
  const loom_pass_decoded_options_t* decoded_options =
      invoke.invoke.decoded_options;
  ASSERT_NE(decoded_options, nullptr);
  ASSERT_EQ(decoded_options->option_count, 3u);
  EXPECT_TRUE(decoded_options->options[0].present);
  EXPECT_EQ(decoded_options->options[0].uint32_value, 8u);
  EXPECT_TRUE(decoded_options->options[1].present);
  EXPECT_EQ(decoded_options->options[1].enum_value_index, 0u);
  EXPECT_TRUE(decoded_options->options[2].present);
  EXPECT_TRUE(iree_string_view_equal(decoded_options->options[2].string_value,
                                     IREE_SV("canonical")));
}

TEST_F(PassProgramTest, RejectsUnknownPassBeforeExecution) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<func> @pipeline pipeline {\n"
                    "  definitely-not-a-pass\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  ProgramStorage storage;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Compile(module, Pipeline(module, 0), &storage.program));
}

TEST_F(PassProgramTest, RejectsCallCycleBeforeExecution) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @a pipeline {\n"
                    "  call @b\n"
                    "}\n"
                    "\n"
                    "pass.pipeline<module> @b pipeline {\n"
                    "  call @a\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  ProgramStorage storage;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Compile(module, Pipeline(module, 0), &storage.program));
}

}  // namespace
}  // namespace loom
