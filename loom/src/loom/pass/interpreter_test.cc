// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/interpreter.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/pass/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/pass/report.h"
#include "loom/pass/test/registry.h"

namespace loom {
namespace {

class ProgramStorage {
 public:
  ~ProgramStorage() { loom_pass_program_deinitialize(&program); }

  loom_pass_program_t program = {};
};

class ReportStorage {
 public:
  ReportStorage() {
    loom_pass_report_initialize(iree_allocator_system(), &report);
  }

  ~ReportStorage() { loom_pass_report_deinitialize(&report); }

  loom_pass_report_t report = {};
};

struct DiagnosticCapture {
  int emission_count = 0;
  const loom_op_t* op = nullptr;
  const loom_error_def_t* error = nullptr;
  iree_host_size_t param_count = 0;
  loom_diagnostic_param_t params[4] = {};
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

  static iree_status_t CaptureDiagnostic(
      void* user_data, const loom_diagnostic_emission_t* emission) {
    DiagnosticCapture* capture = static_cast<DiagnosticCapture*>(user_data);
    ++capture->emission_count;
    capture->op = emission->op;
    capture->error = emission->error;
    capture->param_count = emission->param_count;
    EXPECT_LE(emission->param_count, IREE_ARRAYSIZE(capture->params));
    for (iree_host_size_t i = 0;
         i < emission->param_count && i < IREE_ARRAYSIZE(capture->params);
         ++i) {
      capture->params[i] = emission->params[i];
    }
    return iree_ok_status();
  }

  loom_pass_interpreter_options_t InterpreterOptions(
      loom_test_pass_trace_t* trace,
      iree_diagnostic_emitter_t diagnostic_emitter = {},
      loom_pass_report_t* report = nullptr) {
    return (loom_pass_interpreter_options_t){
        .block_pool = &block_pool_,
        .diagnostic_emitter = diagnostic_emitter,
        .configure =
            {
                .fn = ConfigureTrace,
                .user_data = trace,
            },
        .report = report,
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

TEST_F(PassInterpreterTest, AppendsExecutionReportRecords) {
  loom_module_t* module =
      Parse(IREE_SV("pass.pipeline<module> @pipeline pipeline {\n"
                    "  test.module-noop\n"
                    "  for func {\n"
                    "    test.mark-changed\n"
                    "  }\n"
                    "}\n"
                    "test.func @main() {\n"
                    "  test.yield\n"
                    "}\n"));
  ASSERT_NE(module, nullptr);

  ProgramStorage storage;
  IREE_ASSERT_OK(Compile(module, Pipeline(module, 0), &storage.program));

  ReportStorage report_storage;
  loom_test_pass_trace_t trace = {};
  loom_pass_interpreter_options_t options =
      InterpreterOptions(&trace, {}, &report_storage.report);
  IREE_ASSERT_OK(
      loom_pass_interpreter_run_module(&storage.program, module, &options));

  ASSERT_EQ(report_storage.report.invocation_count, 2u);
  const loom_pass_report_invocation_t& module_record =
      report_storage.report.invocations[0];
  EXPECT_TRUE(iree_string_view_equal(module_record.pass_key,
                                     IREE_SV("test.module-noop")));
  EXPECT_TRUE(iree_string_view_equal(module_record.pipeline_symbol,
                                     IREE_SV("pipeline")));
  EXPECT_TRUE(
      iree_string_view_equal(module_record.symbol_name, IREE_SV("<none>")));
  EXPECT_EQ(module_record.anchor_kind, LOOM_PASS_MODULE);
  EXPECT_FALSE(module_record.changed);
  EXPECT_GE(module_record.duration_nanoseconds, 0);
  ASSERT_EQ(module_record.statistic_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(module_record.statistics[0].name,
                                     IREE_SV("invocations")));
  EXPECT_EQ(module_record.statistics[0].value, 1);

  const loom_pass_report_invocation_t& function_record =
      report_storage.report.invocations[1];
  EXPECT_TRUE(iree_string_view_equal(function_record.pass_key,
                                     IREE_SV("test.mark-changed")));
  EXPECT_TRUE(
      iree_string_view_equal(function_record.symbol_name, IREE_SV("main")));
  EXPECT_EQ(function_record.anchor_kind, LOOM_PASS_FUNCTION);
  EXPECT_TRUE(function_record.changed);
  EXPECT_GE(function_record.duration_nanoseconds, 0);
  ASSERT_EQ(function_record.statistic_count, 2u);
  EXPECT_TRUE(iree_string_view_equal(function_record.statistics[0].name,
                                     IREE_SV("invocations")));
  EXPECT_EQ(function_record.statistics[0].value, 1);
  EXPECT_TRUE(iree_string_view_equal(function_record.statistics[1].name,
                                     IREE_SV("synthetic-events")));
  EXPECT_EQ(function_record.statistics[1].value, 1);
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
  DiagnosticCapture diagnostic_capture;
  loom_pass_interpreter_options_t options =
      InterpreterOptions(&trace, (iree_diagnostic_emitter_t){
                                     .fn = CaptureDiagnostic,
                                     .user_data = &diagnostic_capture,
                                 });
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      loom_pass_interpreter_run_module(&storage.program, module, &options));
  EXPECT_EQ(trace.fail_invocation_count, 1);
  EXPECT_EQ(diagnostic_capture.emission_count, 1);
  ASSERT_NE(diagnostic_capture.error, nullptr);
  EXPECT_EQ(diagnostic_capture.error->domain, LOOM_ERROR_DOMAIN_STRUCTURE);
  EXPECT_EQ(diagnostic_capture.error->code, 28);
  EXPECT_EQ(diagnostic_capture.op, storage.program.instructions[0].source.op);
  ASSERT_EQ(diagnostic_capture.param_count, 4);
  EXPECT_EQ(diagnostic_capture.params[0].kind, LOOM_PARAM_STRING);
  EXPECT_TRUE(iree_string_view_equal(diagnostic_capture.params[0].string,
                                     IREE_SV("test.fail")));
  EXPECT_EQ(diagnostic_capture.params[1].kind, LOOM_PARAM_STRING);
  EXPECT_TRUE(iree_string_view_equal(diagnostic_capture.params[1].string,
                                     IREE_SV("module")));
  EXPECT_EQ(diagnostic_capture.params[2].kind, LOOM_PARAM_STRING);
  EXPECT_TRUE(iree_string_view_equal(diagnostic_capture.params[2].string,
                                     IREE_SV("<none>")));
  EXPECT_EQ(diagnostic_capture.params[3].kind, LOOM_PARAM_STRING);
  EXPECT_TRUE(iree_string_view_equal(diagnostic_capture.params[3].string,
                                     IREE_SV("pipeline")));
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
