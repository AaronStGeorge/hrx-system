// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/verify.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/diagnostic.h"
#include "loom/error/error_defs.h"
#include "loom/error/json_sink.h"
#include "loom/error/renderer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Test infrastructure
//===----------------------------------------------------------------------===//

// Collects diagnostics into a vector for assertions.
struct DiagnosticCollector {
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
  std::vector<std::string> remarks;
};

static iree_status_t CollectDiagnostic(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  auto* collector = static_cast<DiagnosticCollector*>(user_data);
  // Render message from error def + params.
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loom_type_formatter_t type_formatter = {loom_type_format_minimal, NULL};
  loom_diagnostic_render_message(diagnostic->error, diagnostic->params,
                                 diagnostic->param_count, type_formatter,
                                 &stream);
  std::string message(iree_string_builder_buffer(&builder),
                      iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  switch (diagnostic->severity) {
    case LOOM_DIAGNOSTIC_ERROR:
      collector->errors.push_back(std::move(message));
      break;
    case LOOM_DIAGNOSTIC_WARNING:
      collector->warnings.push_back(std::move(message));
      break;
    case LOOM_DIAGNOSTIC_REMARK:
      collector->remarks.push_back(std::move(message));
      break;
    default:
      break;
  }
  return iree_ok_status();
}

// Registers test dialect vtables on the context.
static void RegisterTestDialect(loom_context_t* context) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
  IREE_ASSERT_OK(loom_context_register_dialect(context, LOOM_DIALECT_TEST,
                                               vtables, (uint16_t)count));
}

class VerifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterTestDialect(&context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("verify_test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);

    collector_ = {};
    memset(&options_, 0, sizeof(options_));
    options_.sink.fn = CollectDiagnostic;
    options_.sink.user_data = &collector_;
    options_.max_errors = 100;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Creates a value in the module's value table.
  loom_value_id_t DefineValue(loom_scalar_type_t scalar_type) {
    loom_type_t type = {0};
    type.header = loom_type_make_header(LOOM_TYPE_SCALAR, scalar_type, 0,
                                        LOOM_TYPE_FLAG_INLINE_DIMS);
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_EXPECT_OK(loom_builder_define_value(&builder_, type, &value_id));
    return value_id;
  }

  // Creates a test.func at module level and switches the builder to its
  // body block. Returns |arg_count| value IDs (the function's block
  // arguments) via |out_args|. Subsequent builder ops go inside the
  // function body. Call TerminateFunc() before Verify() to add a
  // terminator to the body block.
  void EnterTestFunc(const loom_type_t* arg_types, iree_host_size_t arg_count,
                     loom_value_id_t* out_args) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder_, IREE_SV("test"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};
    loom_op_t* func_op = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(
        &builder_, 0, 0, callee, arg_types, arg_count, nullptr, 0, nullptr, 0,
        nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_region_t* body = loom_test_func_body(func_op);
    loom_builder_set_block(&builder_, loom_region_entry_block(body));
    for (iree_host_size_t i = 0; i < arg_count; ++i) {
      out_args[i] = loom_region_entry_arg_id(body, (uint16_t)i);
    }
  }

  // Adds a test.yield terminator to the current block. Call after
  // building all ops inside the test.func body.
  void TerminateFunc() {
    loom_op_t* yield_op = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &yield_op));
  }

  loom_symbol_ref_t DefineTestCallee() {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_EXPECT_OK(
        loom_builder_intern_string(&builder_, IREE_SV("callee"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_EXPECT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
  }

  loom_verify_result_t Verify() {
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module_, &options_, &result));
    return result;
  }

  // Runs verification with the caret formatter and returns the output.
  std::string VerifyAndFormatCaret() {
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    auto sink = [](void* user_data,
                   const loom_diagnostic_t* d) -> iree_status_t {
      auto* builder = static_cast<iree_string_builder_t*>(user_data);
      loom_output_stream_t stream;
      loom_output_stream_for_builder(builder, &stream);
      return loom_diagnostic_format(d, &stream);
    };
    memset(&options_, 0, sizeof(options_));
    options_.sink.fn = sink;
    options_.sink.user_data = &output;
    options_.max_errors = 20;
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module_, &options_, &result));
    std::string text(iree_string_builder_buffer(&output),
                     iree_string_builder_size(&output));
    iree_string_builder_deinitialize(&output);
    return text;
  }

  // Runs verification with the JSON sink and returns the output.
  std::string VerifyAndFormatJson() {
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    loom_output_stream_t json_stream;
    loom_output_stream_for_builder(&output, &json_stream);
    loom_json_sink_options_t json_options;
    memset(&json_options, 0, sizeof(json_options));
    json_options.stream = &json_stream;
    json_options.type_formatter = {loom_type_format_minimal, nullptr};
    memset(&options_, 0, sizeof(options_));
    options_.sink.fn = loom_diagnostic_json_sink;
    options_.sink.user_data = &json_options;
    options_.max_errors = 20;
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module_, &options_, &result));
    std::string text(iree_string_builder_buffer(&output),
                     iree_string_builder_size(&output));
    iree_string_builder_deinitialize(&output);
    return text;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
  DiagnosticCollector collector_;
  loom_verify_options_t options_;
};

//===----------------------------------------------------------------------===//
// Structural checks
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, EmptyModulePasses) {
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, ValidAddiPasses) {
  // Build a valid test.addi: %r = test.addi %a, %b : i32
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, args[0], args[1], i32_type,
                                      LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, LoopBodyImplicitTerminatorPasses) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t arg_types[] = {index_type, index_type, index_type};
  loom_value_id_t args[3];
  EnterTestFunc(arg_types, 3, args);

  loom_op_t* loop_op = nullptr;
  IREE_ASSERT_OK(loom_test_loop_build(&builder_, args[0], args[1], args[2],
                                      nullptr, 0, nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &loop_op));
  ASSERT_NE(loop_op, nullptr);
  loom_region_t* body = loom_test_loop_body(loop_op);
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(loom_region_entry_block(body)->op_count, 0u);

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u)
      << (collector_.errors.empty() ? "(no errors)" : collector_.errors[0]);
}

TEST_F(VerifyTest, WrongOperandCountDetected) {
  // Manually create an op with wrong operand count.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI,
                                          /*operand_count=*/1,  // Should be 2.
                                          /*result_count=*/1,
                                          /*region_count=*/0,
                                          /*tied_result_count=*/0,
                                          /*attribute_count=*/0,
                                          LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_results(op)[0] = result_val;

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found_count_error = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("has 1 operands, expected 2") != std::string::npos) {
      found_count_error = true;
      break;
    }
  }
  EXPECT_TRUE(found_count_error)
      << "Expected operand count error, got: "
      << (collector_.errors.empty() ? "(no errors)" : collector_.errors[0]);
}

TEST_F(VerifyTest, OpAfterTerminatorDetected) {
  EnterTestFunc(nullptr, 0, nullptr);
  TerminateFunc();

  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* constant_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &constant_op));

  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("'test.constant' appears after a block terminator") !=
        std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected op-after-terminator error, got: "
                     << (collector_.errors.empty() ? "(no errors)"
                                                   : collector_.errors[0]);
}

//===----------------------------------------------------------------------===//
// Type constraint checks
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, TypeConstraintViolationDetected) {
  // test.addi expects INTEGER operands. Give it f32 values.
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {f32_type, f32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found_type_error = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("expected integer") != std::string::npos) {
      found_type_error = true;
      break;
    }
  }
  EXPECT_TRUE(found_type_error) << "Expected type constraint error";
}

//===----------------------------------------------------------------------===//
// Semantic constraint checks
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, SameTypeConstraintPasses) {
  // test.addi has SameType(lhs, rhs, result). All i32 → passes.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, args[0], args[1], i32_type,
                                      LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, SameTypeConstraintViolation) {
  // Build addi with mismatched types: i32 operands but f32 result.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);

  // Manually build to set a different result type.
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found_same_type = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("does not match") != std::string::npos) {
      found_same_type = true;
      break;
    }
  }
  EXPECT_TRUE(found_same_type) << "Expected type mismatch error";
}

//===----------------------------------------------------------------------===//
// SSA dominance checks
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, UndefinedOperandDetected) {
  // Create a test.func with no args, then build an op inside it that
  // references value IDs that exist in the value table but were never
  // defined by a prior op or as a block arg.
  EnterTestFunc(nullptr, 0, nullptr);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_value_id_t lhs = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t rhs = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_op_operands(op)[0] = lhs;
  loom_op_operands(op)[1] = rhs;
  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_op_results(op)[0] = result_val;

  // The verifier should catch the undefined operands.
  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found_undefined = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("undefined value") != std::string::npos) {
      found_undefined = true;
      break;
    }
  }
  EXPECT_TRUE(found_undefined)
      << "Expected undefined value error, got: "
      << (collector_.errors.empty() ? "(no errors)" : collector_.errors[0]);
}

//===----------------------------------------------------------------------===//
// Constraint table structure
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, ConstraintTableSize) {
  static_assert(sizeof(loom_constraint_t) == 16,
                "loom_constraint_t must be 16 bytes");
}

TEST_F(VerifyTest, AddiHasConstraints) {
  EXPECT_EQ(loom_test_addi_vtable.constraint_count, 1);
  EXPECT_NE(loom_test_addi_vtable.constraints, nullptr);
  EXPECT_EQ(loom_test_addi_vtable.constraints[0].relation,
            LOOM_RELATION_PAIRWISE_EQ);
  EXPECT_EQ(loom_test_addi_vtable.constraints[0].property, LOOM_PROPERTY_TYPE);
  EXPECT_EQ(loom_test_addi_vtable.constraints[0].arg_count, 3);
}

TEST_F(VerifyTest, MapHasRegionConstraints) {
  // test.map has 5 constraints including region-value relationships.
  EXPECT_EQ(loom_test_map_vtable.constraint_count, 5);
  EXPECT_NE(loom_test_map_vtable.constraints, nullptr);
  EXPECT_EQ(loom_test_map_vtable.constraints[0].relation,
            LOOM_RELATION_ALL_SAME);
  EXPECT_EQ(loom_test_map_vtable.constraints[0].property, LOOM_PROPERTY_SHAPE);
  EXPECT_EQ(loom_test_map_vtable.constraints[1].relation,
            LOOM_RELATION_REGION_ARG_COUNT);
}

TEST_F(VerifyTest, ConstantHasNoConstraints) {
  EXPECT_EQ(loom_test_constant_vtable.constraint_count, 0);
  EXPECT_EQ(loom_test_constant_vtable.constraints, nullptr);
}

//===----------------------------------------------------------------------===//
// Structured diagnostic output (print-op fallback)
//===----------------------------------------------------------------------===//

// Collects the full loom_diagnostic_t for structured field inspection.
struct StructuredDiagnosticCollector {
  struct Entry {
    loom_diagnostic_severity_t severity;
    std::string rendered_message;
    const loom_error_def_t* error;
    loom_emitter_t emitter;
    iree_host_size_t param_count;
    bool has_source_range;
    std::string source_text;
    std::string filename;
  };
  std::vector<Entry> entries;
};

static iree_status_t CollectStructured(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  auto* collector = static_cast<StructuredDiagnosticCollector*>(user_data);
  StructuredDiagnosticCollector::Entry entry;
  entry.severity = diagnostic->severity;
  entry.error = diagnostic->error;
  entry.emitter = diagnostic->emitter;
  entry.param_count = diagnostic->param_count;
  entry.has_source_range = diagnostic->origin.source.size > 0;
  if (entry.has_source_range) {
    entry.source_text = std::string(diagnostic->origin.source.data,
                                    diagnostic->origin.source.size);
    entry.filename = std::string(diagnostic->origin.filename.data,
                                 diagnostic->origin.filename.size);
  }
  // Render message from error def + params for test assertions.
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loom_type_formatter_t type_formatter = {loom_type_format_minimal, NULL};
  loom_diagnostic_render_message(diagnostic->error, diagnostic->params,
                                 diagnostic->param_count, type_formatter,
                                 &stream);
  entry.rendered_message = std::string(iree_string_builder_buffer(&builder),
                                       iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  collector->entries.push_back(std::move(entry));
  return iree_ok_status();
}

TEST_F(VerifyTest, StructuredDiagnosticHasErrorDef) {
  // Build an addi with wrong operand count to trigger a structured error.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 1, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_results(op)[0] = result_val;

  // Use the structured collector instead of the default one.
  StructuredDiagnosticCollector structured;
  options_.sink.fn = CollectStructured;
  options_.sink.user_data = &structured;

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  ASSERT_GT(structured.entries.size(), 0u);

  // The first error should be a structured STRUCTURE_001 diagnostic.
  const auto& entry = structured.entries[0];
  EXPECT_EQ(entry.severity, LOOM_DIAGNOSTIC_ERROR);
  EXPECT_EQ(entry.emitter, LOOM_EMITTER_VERIFIER);
  ASSERT_NE(entry.error, nullptr);
  EXPECT_EQ(entry.error->domain, LOOM_ERROR_DOMAIN_STRUCTURE);
  EXPECT_EQ(entry.error->code, 1);
  EXPECT_GT(entry.param_count, 0);
}

TEST_F(VerifyTest, PrintOpFallbackProvidesSourceRange) {
  // Build a valid op so we can trigger a SameType constraint error.
  // addi with i32 operands but f32 result → SameType violation.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  // No source resolver configured → the verifier must fall back to
  // printing the op for the source range.
  StructuredDiagnosticCollector structured;
  options_.sink.fn = CollectStructured;
  options_.sink.user_data = &structured;

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);

  // Find the SameType error (there may also be type constraint errors).
  bool found_with_source = false;
  for (const auto& entry : structured.entries) {
    if (entry.has_source_range) {
      found_with_source = true;
      // The source text should be the printed op (contains "test.addi").
      EXPECT_NE(entry.source_text.find("test.addi"), std::string::npos)
          << "Printed op source: " << entry.source_text;
      // The filename should be the verifier pseudo-filename.
      EXPECT_EQ(entry.filename, "<verifier>");
      break;
    }
  }
  EXPECT_TRUE(found_with_source)
      << "Expected at least one diagnostic with a print-op source range";
}

//===----------------------------------------------------------------------===//
// Golden output tests
//===----------------------------------------------------------------------===//
//
// Exact-match tests for diagnostic output. One per scenario. When
// formatting changes, the diff shows exactly what changed.

// Scenario: no source resolver, op has no location (post-pass).
// The verifier prints the op and uses it as the source line.
TEST_F(VerifyTest, GoldenCaretNoSource) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  TerminateFunc();
  std::string output = VerifyAndFormatCaret();
  EXPECT_EQ(output,
            "<verifier>:1:1: error [TYPE/004]: result 'result 0' has type f32,"
            " expected integer\n"
            " 1 | %2 = test.addi %0, %1 : f32\n"
            "   | ^^                         \n"
            "   = help: 'result 0' must satisfy type constraint 'integer'\n"
            "<verifier>:1:1: error [TYPE/001]: 'operand 0' type i32 does not"
            " match 'result 0' type f32\n"
            " 1 | %2 = test.addi %0, %1 : f32\n"
            "   | ^^             ^^          \n"
            "   = help: Ensure 'operand 0' and 'result 0' have the same"
            " type\n");
}

TEST_F(VerifyTest, GoldenJsonNoSource) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  TerminateFunc();
  std::string output = VerifyAndFormatJson();
  EXPECT_EQ(
      output,
      "{\"severity\":\"error\",\"domain\":\"TYPE\",\"code\":4,"
      "\"emitter\":\"verifier\","
      "\"message\":\"result 'result 0' has type f32, expected integer\","
      "\"fix_hint\":\"'result 0' must satisfy type constraint 'integer'\","
      "\"params\":{\"result_name\":\"result 0\","
      "\"actual_type\":\"f32\",\"expected_constraint\":\"integer\"}}\n"
      "{\"severity\":\"error\",\"domain\":\"TYPE\",\"code\":1,"
      "\"emitter\":\"verifier\","
      "\"message\":\"'operand 0' type i32 does not match"
      " 'result 0' type f32\","
      "\"fix_hint\":\"Ensure 'operand 0' and 'result 0'"
      " have the same type\","
      "\"params\":{\"field_a\":\"operand 0\",\"type_a\":\"i32\","
      "\"field_b\":\"result 0\",\"type_b\":\"f32\"}}\n");
}

// Scenario: source resolver provides original source text.
// The diagnostic uses the original source for carets, not the printed op.
static bool FakeSourceResolver(void* user_data, const loom_module_t* module,
                               loom_location_id_t location,
                               loom_source_range_t* out_range) {
  // Pretend every op comes from this source line.
  static const char source[] = "  %result = test.addi %input_a, %input_b : i32";
  (void)module;
  if (location == LOOM_LOCATION_UNKNOWN) return false;
  out_range->filename = IREE_SV("model.loom");
  out_range->source = iree_make_cstring_view(source);
  out_range->start = 2;
  out_range->end = 46;  // strlen(source) = 46.
  out_range->start_line = 42;
  out_range->start_column = 3;
  out_range->end_line = 42;
  out_range->end_column = 47;
  return true;
}

TEST_F(VerifyTest, GoldenCaretWithSource) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0,
                                          /*location=*/1,  // Non-zero so
                                                           // resolver fires.
                                          &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  // Wire up the fake resolver.
  iree_string_builder_t output;
  iree_string_builder_initialize(iree_allocator_system(), &output);
  auto sink = [](void* user_data, const loom_diagnostic_t* d) -> iree_status_t {
    auto* builder = static_cast<iree_string_builder_t*>(user_data);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(builder, &stream);
    return loom_diagnostic_format(d, &stream);
  };
  memset(&options_, 0, sizeof(options_));
  options_.sink.fn = sink;
  options_.sink.user_data = &output;
  options_.max_errors = 20;
  options_.source_resolver.fn = FakeSourceResolver;

  TerminateFunc();
  loom_verify_result_t result = {};
  IREE_EXPECT_OK(loom_verify_module(module_, &options_, &result));
  std::string text(iree_string_builder_buffer(&output),
                   iree_string_builder_size(&output));
  iree_string_builder_deinitialize(&output);

  EXPECT_EQ(
      text,
      "model.loom:42:3: error [TYPE/004]: result 'result 0' has type f32,"
      " expected integer\n"
      " 42 |   %result = test.addi %input_a, %input_b : i32\n"
      "    |   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"  // 44 carets
      "    = help: 'result 0' must satisfy type constraint 'integer'\n"
      "model.loom:42:3: error [TYPE/001]: 'operand 0' type i32 does not"
      " match 'result 0' type f32\n"
      " 42 |   %result = test.addi %input_a, %input_b : i32\n"
      "    |   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"  // 44 carets
      "    = help: Ensure 'operand 0' and 'result 0' have the same type\n");
}

TEST_F(VerifyTest, GoldenJsonWithSource) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, /*location=*/1, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  iree_string_builder_t output;
  iree_string_builder_initialize(iree_allocator_system(), &output);
  loom_output_stream_t json_stream;
  loom_output_stream_for_builder(&output, &json_stream);
  loom_json_sink_options_t json_options;
  memset(&json_options, 0, sizeof(json_options));
  json_options.stream = &json_stream;
  json_options.type_formatter = {loom_type_format_minimal, nullptr};
  memset(&options_, 0, sizeof(options_));
  options_.sink.fn = loom_diagnostic_json_sink;
  options_.sink.user_data = &json_options;
  options_.max_errors = 20;
  options_.source_resolver.fn = FakeSourceResolver;

  TerminateFunc();
  loom_verify_result_t result = {};
  IREE_EXPECT_OK(loom_verify_module(module_, &options_, &result));
  std::string text(iree_string_builder_buffer(&output),
                   iree_string_builder_size(&output));
  iree_string_builder_deinitialize(&output);

  // JSON output is the same regardless of source availability — the
  // source range is diagnostic-formatter metadata, not in the JSON.
  EXPECT_EQ(
      text,
      "{\"severity\":\"error\",\"domain\":\"TYPE\",\"code\":4,"
      "\"emitter\":\"verifier\","
      "\"message\":\"result 'result 0' has type f32, expected integer\","
      "\"fix_hint\":\"'result 0' must satisfy type constraint 'integer'\","
      "\"params\":{\"result_name\":\"result 0\","
      "\"actual_type\":\"f32\",\"expected_constraint\":\"integer\"}}\n"
      "{\"severity\":\"error\",\"domain\":\"TYPE\",\"code\":1,"
      "\"emitter\":\"verifier\","
      "\"message\":\"'operand 0' type i32 does not match"
      " 'result 0' type f32\","
      "\"fix_hint\":\"Ensure 'operand 0' and 'result 0'"
      " have the same type\","
      "\"params\":{\"field_a\":\"operand 0\",\"type_a\":\"i32\","
      "\"field_b\":\"result 0\",\"type_b\":\"f32\"}}\n");
}

TEST_F(VerifyTest, StructuredDominanceError) {
  // Create an op referencing undefined values to trigger dominance errors.
  EnterTestFunc(nullptr, 0, nullptr);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_value_id_t lhs = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t rhs = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_op_operands(op)[0] = lhs;
  loom_op_operands(op)[1] = rhs;
  loom_op_results(op)[0] = result_val;

  StructuredDiagnosticCollector structured;
  options_.sink.fn = CollectStructured;
  options_.sink.user_data = &structured;

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);

  // Find a DOMINANCE error.
  bool found_dominance = false;
  for (const auto& entry : structured.entries) {
    if (entry.error && entry.error->domain == LOOM_ERROR_DOMAIN_DOMINANCE) {
      found_dominance = true;
      EXPECT_EQ(entry.error->code, 1);  // DOMINANCE_001: undefined value.
      EXPECT_NE(entry.rendered_message.find("undefined value"),
                std::string::npos)
          << "message: " << entry.rendered_message;
      // Should have the printed op as source even for dominance errors.
      EXPECT_TRUE(entry.has_source_range)
          << "Dominance error should have print-op fallback source";
      break;
    }
  }
  EXPECT_TRUE(found_dominance) << "Expected a DOMINANCE domain error";
}

//===----------------------------------------------------------------------===//
// Tied result validation
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, DuplicateTiedResultIndexDetected) {
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {f32_type, f32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_type_t result_types[] = {f32_type, f32_type};
  loom_tied_result_t tied_results[] = {
      {.result_index = 0, .operand_index = 0, .has_type_change = false},
      {.result_index = 0, .operand_index = 1, .has_type_change = false},
  };
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_invoke_build(
      &builder_, DefineTestCallee(), args, 2, result_types, 2, tied_results,
      IREE_ARRAYSIZE(tied_results), LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found_duplicate_result = false;
  for (const auto& message : collector_.errors) {
    if (message.find("result 0 of 'test.invoke' is tied more than once") !=
        std::string::npos) {
      found_duplicate_result = true;
      break;
    }
  }
  EXPECT_TRUE(found_duplicate_result)
      << "Expected duplicate tied result error, got: "
      << (collector_.errors.empty() ? "(no errors)" : collector_.errors[0]);
}

TEST_F(VerifyTest, DuplicateTiedOperandIndexDetected) {
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {f32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_type_t result_types[] = {f32_type, f32_type};
  loom_tied_result_t tied_results[] = {
      {.result_index = 0, .operand_index = 0, .has_type_change = false},
      {.result_index = 1, .operand_index = 0, .has_type_change = false},
  };
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_invoke_build(
      &builder_, DefineTestCallee(), args, 1, result_types, 2, tied_results,
      IREE_ARRAYSIZE(tied_results), LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found_duplicate_operand = false;
  for (const auto& message : collector_.errors) {
    if (message.find(
            "operand 0 of 'test.invoke' is tied by more than one result") !=
        std::string::npos) {
      found_duplicate_operand = true;
      break;
    }
  }
  EXPECT_TRUE(found_duplicate_operand)
      << "Expected duplicate tied operand error, got: "
      << (collector_.errors.empty() ? "(no errors)" : collector_.errors[0]);
}

TEST_F(VerifyTest, AmbiguousRepeatedOperandValueDetected) {
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {f32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_value_id_t operands[] = {args[0], args[0]};
  loom_type_t result_types[] = {f32_type};
  loom_tied_result_t tied_results[] = {
      {.result_index = 0, .operand_index = 1, .has_type_change = false},
  };
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_invoke_build(
      &builder_, DefineTestCallee(), operands, IREE_ARRAYSIZE(operands),
      result_types, 1, tied_results, IREE_ARRAYSIZE(tied_results),
      LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found_ambiguous_value = false;
  for (const auto& message : collector_.errors) {
    if (message.find("operand 1 of 'test.invoke' ties value") !=
            std::string::npos &&
        message.find("multiple operand slots") != std::string::npos) {
      found_ambiguous_value = true;
      break;
    }
  }
  EXPECT_TRUE(found_ambiguous_value)
      << "Expected ambiguous tied operand-value error, got: "
      << (collector_.errors.empty() ? "(no errors)" : collector_.errors[0]);
}

TEST_F(VerifyTest, FuncDefTiedResultUsesEntryBlockArgsWithoutConsumingThem) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("identity"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
  loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};

  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_tied_result_t tied_results[] = {
      {.result_index = 0, .operand_index = 0, .has_type_change = false},
  };
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, callee, &f32_type, 1,
                                      &f32_type, 1, tied_results,
                                      IREE_ARRAYSIZE(tied_results), nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &func_op));

  loom_region_t* body = loom_test_func_body(func_op);
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(body->block_count, 1u);
  ASSERT_EQ(loom_region_entry_arg_count(body), 1u);
  loom_builder_set_block(&builder_, loom_region_entry_block(body));

  loom_value_id_t arg_id = loom_region_entry_arg_id(body, 0);
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &arg_id, 1,
                                       LOOM_LOCATION_UNKNOWN, &return_op));

  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u)
      << (collector_.errors.empty() ? "" : collector_.errors[0]);
}

TEST_F(VerifyTest, FuncDeclTiedResultUsesSignatureOperandsWithoutDominance) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_intern_string(
      &builder_, IREE_SV("extern_identity"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
  loom_symbol_ref_t callee = {.module_id = 0, .symbol_id = symbol_id};

  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_tied_result_t tied_results[] = {
      {.result_index = 0, .operand_index = 0, .has_type_change = false},
  };
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_decl_build(
      &builder_, 0, 0, callee, &f32_type, 1, &f32_type, 1, tied_results,
      IREE_ARRAYSIZE(tied_results), LOOM_LOCATION_UNKNOWN, &func_op));

  EXPECT_EQ(func_op->operand_count, 1u);
  EXPECT_EQ(func_op->tied_result_count, 1u);

  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u)
      << (collector_.errors.empty() ? "" : collector_.errors[0]);
}

TEST_F(VerifyTest, RejectsNonLocalSymbolRef) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("foreign"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
  loom_symbol_ref_t callee = {.module_id = 1, .symbol_id = symbol_id};

  loom_op_t* decl_op = nullptr;
  IREE_ASSERT_OK(loom_test_decl_build(&builder_, 0, 0, callee, nullptr, 0,
                                      nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &decl_op));

  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  ASSERT_FALSE(collector_.errors.empty());
  EXPECT_THAT(collector_.errors[0],
              ::testing::HasSubstr(
                  "symbol references in in-memory IR must use module_id 0"));
}

//===----------------------------------------------------------------------===//
// SSA encoding reference validation
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, ValidSsaEncodingPasses) {
  // Create a test.func with an encoding arg. The encoding value ID is
  // needed to construct the tile type, so we pass just the encoding
  // type to EnterTestFunc and add the tile arg manually afterwards.
  loom_type_t encoding_type = loom_type_encoding();
  loom_type_t arg_types[] = {encoding_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);
  loom_value_id_t enc_value = args[0];

  // Create a tile type that references the encoding via SSA.
  loom_type_t tile_type =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, 256, 0);
  tile_type.encoding_id = enc_value;
  tile_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  // Add a second block arg with the encoded tile type.
  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(&builder_, builder_.ip.block,
                                               tile_type, &input));

  // Build test.convert (accepts ANY types) with the encoded tile.
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_convert_build(&builder_, input, tile_type,
                                         LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, SsaEncodingOutOfRange) {
  // Create a tile type with an SSA encoding value_id that is way out of range.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_type_t tile_type =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, 256, 0);
  tile_type.encoding_id = 9999;
  tile_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  // Build test.convert with the bad encoding type on the result.
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_convert_build(&builder_, args[0], tile_type,
                                         LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("encoding value") != std::string::npos &&
        msg.find("9999") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected encoding out-of-range error";
}

TEST_F(VerifyTest, SsaEncodingNotDefined) {
  // Create a test.func with an i32 arg.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  // Create an encoding value but do NOT add it as a block arg --
  // it won't be in scope when used.
  loom_type_t encoding_type = loom_type_encoding();
  loom_value_id_t enc_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_define_value(&builder_, encoding_type, &enc_value));

  loom_type_t tile_type =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, 256, 0);
  tile_type.encoding_id = enc_value;
  tile_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  // Build a test.convert with a normal input but the bad operand type.
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_convert_build(&builder_, args[0], i32_type,
                                         LOOM_LOCATION_UNKNOWN, &op));
  // Inject the undefined encoding ref into the operand's type.
  loom_module_set_value_type(module_, args[0], tile_type);

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("not defined") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected encoding not-defined error";
}

TEST_F(VerifyTest, SsaEncodingWrongType) {
  // Create two i32 args: one to use as a wrong-type encoding reference,
  // one as the input to convert.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_type_t tile_type =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, 256, 0);
  tile_type.encoding_id = args[0];
  tile_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  // Build a test.convert with a normal input but inject the bad type.
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_convert_build(&builder_, args[1], i32_type,
                                         LOOM_LOCATION_UNKNOWN, &op));
  // Inject the wrong-type encoding ref into the operand's type.
  loom_module_set_value_type(module_, args[1], tile_type);

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("expected 'encoding'") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected encoding wrong-type error";
}

//===----------------------------------------------------------------------===//
// Variadic field constraint checks
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, VariadicSameTypeConstraintPasses) {
  // test.reduce has SameType("inputs", "result") where inputs is variadic.
  // All i32 → passes.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type, i32_type};
  loom_value_id_t args[3];
  EnterTestFunc(arg_types, 3, args);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_reduce_build(&builder_, args, 3, i32_type,
                                        LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, VariadicSameTypeMismatchInVariadic) {
  // test.reduce with inputs [i32, f32, i32] and result i32.
  // The second input has type f32 → constraint violation.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {i32_type, f32_type, i32_type};
  loom_value_id_t args[3];
  EnterTestFunc(arg_types, 3, args);

  // Manually build to control operand types precisely.
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_REDUCE, 3, 1,
                                          0, 0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_operands(op)[2] = args[2];
  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_op_results(op)[0] = result_val;

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("does not match") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected type mismatch within variadic inputs";
}

TEST_F(VerifyTest, VariadicSameTypeMismatchAgainstResult) {
  // test.reduce with inputs [i32, i32] and result f32.
  // All inputs match each other but not the result → constraint violation.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_REDUCE, 2, 1,
                                          0, 0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = DefineValue(LOOM_SCALAR_TYPE_F32);

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("does not match") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected type mismatch between variadic and result";
}

TEST_F(VerifyTest, VariadicSameTypeSingleInput) {
  // test.reduce with a single input [i32] and result i32.
  // Should pass: one variadic element matching the result.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_reduce_build(&builder_, args, 1, i32_type,
                                        LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

//===----------------------------------------------------------------------===//
// Adversarial input tests
//===----------------------------------------------------------------------===//
// These tests exercise the verifier on pathological IR that a malicious
// or buggy producer could construct. They verify that the verifier
// handles resource limits gracefully (returns RESOURCE_EXHAUSTED)
// and that arena allocation/growth works on large inputs.

TEST_F(VerifyTest, LargeModuleManyValues) {
  // Create a module with many values and ops to exercise the arena
  // allocation path and defined_stack dynamic growth. With 4000 values,
  // the initial defined_stack capacity (value_count/4 = 1000) will be
  // exceeded, forcing iree_arena_grow_array to fire.
  constexpr int kValueCount = 4000;
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  std::vector<loom_type_t> arg_types(kValueCount, i32_type);
  std::vector<loom_value_id_t> args(kValueCount);
  EnterTestFunc(arg_types.data(), kValueCount, args.data());

  // Build ops that use pairs of these values.
  for (int i = 0; i + 1 < kValueCount; i += 2) {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_test_addi_build(&builder_, args[i], args[i + 1],
                                        i32_type, LOOM_LOCATION_UNKNOWN, &op));
  }

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, DeepNestingAtLimit) {
  // Build IR nested to exactly LOOM_VERIFY_MAX_SCOPE_DEPTH (32) levels.
  // The module body is scope 0, the test.func body is scope 1, so we
  // can nest 30 additional isolated_region ops (scopes 2..31).
  constexpr int kNestingDepth = 30;

  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  EnterTestFunc(nullptr, 0, nullptr);

  // Build nested isolated_region ops. Each has one result and one body
  // region. We use save/restore to build inside each region.
  loom_builder_ip_t saved_ips[kNestingDepth];
  loom_op_t* ops[kNestingDepth];

  for (int depth = 0; depth < kNestingDepth; ++depth) {
    loom_type_t result_types[] = {i32_type};
    IREE_ASSERT_OK(
        loom_test_isolated_region_build(&builder_, result_types, 1, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &ops[depth]));
    // Switch to building inside the region body.
    loom_region_t* body = loom_test_isolated_region_body(ops[depth]);
    ASSERT_NE(body, nullptr);
    ASSERT_GT(body->block_count, 0);
    saved_ips[depth] = loom_builder_enter_region(&builder_, ops[depth], body);
  }

  // Restore all the way back out.
  for (int depth = kNestingDepth - 1; depth >= 0; --depth) {
    loom_builder_restore(&builder_, saved_ips[depth]);
  }

  // The nested isolated_region ops don't have proper terminators, so
  // the verifier reports structural errors. The point of this test is
  // that at the LOOM_VERIFY_MAX_SCOPE_DEPTH limit the verifier
  // completes the walk without RESOURCE_EXHAUSTED.
  TerminateFunc();
  loom_verify_result_t result = {};
  IREE_EXPECT_OK(loom_verify_module(module_, &options_, &result));
}

TEST_F(VerifyTest, DeepNestingExceedsLimit) {
  // Build IR nested to LOOM_VERIFY_MAX_SCOPE_DEPTH + 1 levels.
  // The verifier should return RESOURCE_EXHAUSTED (not crash, not
  // silently succeed). Module body is scope 0, func body is scope 1,
  // then 31 isolated_region regions push scopes 2..32 (exceeding 32).
  constexpr int kNestingDepth = 31;

  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  EnterTestFunc(nullptr, 0, nullptr);

  loom_builder_ip_t saved_ips[kNestingDepth];
  loom_op_t* ops[kNestingDepth];

  for (int depth = 0; depth < kNestingDepth; ++depth) {
    loom_type_t result_types[] = {i32_type};
    IREE_ASSERT_OK(
        loom_test_isolated_region_build(&builder_, result_types, 1, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &ops[depth]));
    loom_region_t* body = loom_test_isolated_region_body(ops[depth]);
    ASSERT_NE(body, nullptr);
    ASSERT_GT(body->block_count, 0);
    saved_ips[depth] = loom_builder_enter_region(&builder_, ops[depth], body);
  }

  for (int depth = kNestingDepth - 1; depth >= 0; --depth) {
    loom_builder_restore(&builder_, saved_ips[depth]);
  }

  TerminateFunc();
  loom_verify_result_t result = {};
  iree_status_t status = loom_verify_module(module_, &options_, &result);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
}

TEST_F(VerifyTest, DefinedStackGrowsPastInitialCapacity) {
  // The defined_stack starts at value_count/4 capacity. If we create
  // a module with N values but define them all in a single scope (no
  // nesting), the stack must grow to hold all N. This exercises
  // iree_arena_grow_array.
  //
  // 80 block args → value_count starts at ~80, initial capacity =
  // 80/4 = 20. We define 80 values in one scope → growth required.
  constexpr int kArgCount = 80;
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  std::vector<loom_type_t> arg_types(kArgCount, i32_type);
  std::vector<loom_value_id_t> args(kArgCount);
  EnterTestFunc(arg_types.data(), kArgCount, args.data());

  // Build some ops so the results also go onto the defined stack.
  for (int i = 0; i + 1 < kArgCount; i += 2) {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_test_addi_build(&builder_, args[i], args[i + 1],
                                        i32_type, LOOM_LOCATION_UNKNOWN, &op));
  }

  TerminateFunc();
  auto result = Verify();
  // Should succeed -- the stack grew dynamically.
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, MaxErrorsStopsWalk) {
  // Set max_errors to 2, build IR with many errors, verify that
  // the verifier stops after 2 and doesn't waste time on the rest.
  options_.max_errors = 2;
  EnterTestFunc(nullptr, 0, nullptr);

  // Build 10 ops with undefined operands (each generates a dominance error).
  for (int i = 0; i < 10; ++i) {
    loom_value_id_t fake_lhs = DefineValue(LOOM_SCALAR_TYPE_I32);
    loom_value_id_t fake_rhs = DefineValue(LOOM_SCALAR_TYPE_I32);
    loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_I32);
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_builder_allocate_op(
        &builder_, LOOM_OP_TEST_ADDI, 2, 1, /*region_count=*/0,
        /*tied_result_count=*/0, /*attribute_count=*/0, LOOM_LOCATION_UNKNOWN,
        &op));
    loom_op_operands(op)[0] = fake_lhs;
    loom_op_operands(op)[1] = fake_rhs;
    loom_op_results(op)[0] = result_val;
  }

  TerminateFunc();
  auto result = Verify();
  // Should have exactly 2 errors (stopped at limit).
  EXPECT_EQ(result.error_count, 2u);
}

//===----------------------------------------------------------------------===//
// Constraint relation coverage: all 8 relations
//===----------------------------------------------------------------------===//
// These tests exercise each constraint relation end-to-end (positive
// and/or negative paths) to ensure the verifier catches violations.

// --- ALL_SAME (relation 1): input tiles must have identical shapes ---

TEST_F(VerifyTest, AllSameViolationDifferentShapes) {
  // test.map has AllShapesMatch("inputs"). Build with two inputs of
  // different tile shapes → violation.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);
  loom_type_t tile8 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(8), 0);

  loom_type_t arg_types[] = {tile4, tile8};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0], args[1]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 2, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("shapes do not match") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected AllSame shape mismatch error";
}

TEST_F(VerifyTest, AllSamePassesIdenticalShapes) {
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4, tile4};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0], args[1]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 2, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  // Yield the first block arg (the builder auto-creates f32 block args
  // for the tile inputs).
  loom_region_t* body = loom_test_map_body(op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, op, body);
  loom_value_id_t yield_val = loom_region_entry_arg_id(body, 0);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

// --- REGION_ARG_COUNT (relation 4): block args count must match inputs ---

TEST_F(VerifyTest, RegionArgCountViolation) {
  // Build test.map with 2 inputs, then manually change the region to
  // have 3 block args instead of 2.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4, tile4};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0], args[1]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 2, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  // Add a third block arg to create a count mismatch.
  loom_region_t* body = loom_test_map_body(op);
  loom_type_t scalar_f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t extra_arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body), scalar_f32, &extra_arg));

  // Add a yield using the first block arg so we don't trip unrelated checks.
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, op, body);
  loom_value_id_t yield_val = loom_region_entry_arg_id(body, 0);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("block arguments") != std::string::npos &&
        msg.find("expected") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected region arg count mismatch error";
}

// --- REGION_ARG_MATCH (relation 5): block arg types must match element types
// ---

TEST_F(VerifyTest, RegionArgMatchViolation) {
  // Build test.map with f32 tile input. The builder auto-creates block
  // args with the element type (f32). We then manually change the first
  // block arg's type to i32.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 1, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  // Change the block arg type to i32.
  loom_region_t* body = loom_test_map_body(op);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_module_set_value_type(module_, loom_region_entry_arg_id(body, 0),
                             i32_type);

  // Yield the (now i32) block arg. This yields an i32 for a tile<4xf32>
  // result, which will also trigger a yield type mismatch, but the specific
  // error we care about is the block arg type mismatch.
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, op, body);
  loom_value_id_t yield_val = loom_region_entry_arg_id(body, 0);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("expected element type") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected region arg type mismatch error";
}

// --- YIELD_COUNT (relation 6): yield operands count must match results ---

TEST_F(VerifyTest, YieldCountViolation) {
  // Build test.map with 1 result, but yield 2 values.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 1, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  // Yield 2 values when result count is 1.
  loom_region_t* body = loom_test_map_body(op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, op, body);
  loom_type_t scalar_f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t val_a = LOOM_VALUE_ID_INVALID;
  loom_value_id_t val_b = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, scalar_f32, &val_a));
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, scalar_f32, &val_b));
  loom_value_id_t yield_vals[] = {val_a, val_b};
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, yield_vals, 2,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("yield has") != std::string::npos &&
        msg.find("expected") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected yield count mismatch error";
}

TEST_F(VerifyTest, YieldCountViolationWithImplicitTerminator) {
  // test.map has one result but the body is empty. The verifier should
  // interpret that as an implicit zero-yield terminator and reject the
  // result/yield count mismatch.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_op_t* map_op = nullptr;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, args, 1, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));
  ASSERT_NE(map_op, nullptr);
  loom_region_t* body = loom_test_map_body(map_op);
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(loom_region_entry_block(body)->op_count, 0u);

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("yield has 0 operands, expected 1") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected implicit-yield count mismatch error, got: "
                     << (collector_.errors.empty() ? "(no errors)"
                                                   : collector_.errors[0]);
}

// --- YIELD_MATCH (relation 7): yield types must match result types ---

TEST_F(VerifyTest, YieldMatchViolation) {
  // Build test.map with result tile<4xf32>, but yield an i32 value.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 1, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  // Yield an i32 when result is tile<f32>.
  loom_region_t* body = loom_test_map_body(op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, op, body);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t yield_val = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, i32_type, &yield_val));
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("yielded value has type") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected yield type mismatch error";
}

// --- COUNT_MATCHES_RANK (relation 2): offset count must match source rank ---

TEST_F(VerifyTest, CountMatchesRankViolation) {
  // test.slice with rank-2 source but only 1 dynamic offset → violation.
  loom_type_t tile_2d =
      loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(8), 0);
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  loom_type_t arg_types[] = {tile_2d, index_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);
  loom_value_id_t source = args[0];
  loom_value_id_t offset = args[1];

  // Provide 1 dynamic offset for a rank-2 source. static_offsets has 2
  // entries so the builder creates a valid op structurally.
  int64_t static_offsets[] = {0, 0};
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_slice_build(&builder_, source, &offset, 1,
                                       static_offsets, 2, tile_2d, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("offset count") != std::string::npos &&
        msg.find("does not match rank") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected offset count vs rank error";
}

TEST_F(VerifyTest, CountMatchesRankPasses) {
  // test.slice with rank-2 source and 2 dynamic offsets → passes.
  loom_type_t tile_2d =
      loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(8), 0);
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  loom_type_t arg_types[] = {tile_2d, index_type, index_type};
  loom_value_id_t args[3];
  EnterTestFunc(arg_types, 3, args);
  loom_value_id_t source = args[0];
  loom_value_id_t offset_a = args[1];
  loom_value_id_t offset_b = args[2];

  int64_t static_offsets[] = {0, 0};
  loom_value_id_t offsets[] = {offset_a, offset_b};
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_slice_build(&builder_, source, offsets, 2,
                                       static_offsets, 2, tile_2d, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

// --- ATTR_IN_RANGE_RANK (relation 3): dim index must be in [0, rank) ---

TEST_F(VerifyTest, AttrInRangeRankViolation) {
  // test.dim with rank-1 source but dim_index=5 → out of bounds.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);
  loom_value_id_t source = args[0];
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t result_val = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, index_type, &result_val));

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_DIM, 1, 1, 0,
                                          0, 1, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = source;
  loom_op_results(op)[0] = result_val;
  loom_op_attrs(op)[0] = loom_attr_i64(5);

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("out of bounds") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected dim index out of bounds error";
}

TEST_F(VerifyTest, AttrInRangeRankNegativeIndex) {
  // test.dim with rank-2 source but dim_index=-1 → out of bounds.
  loom_type_t tile_2d =
      loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(8), 0);

  loom_type_t arg_types[] = {tile_2d};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);
  loom_value_id_t source = args[0];
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t result_val = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, index_type, &result_val));

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_DIM, 1, 1, 0,
                                          0, 1, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = source;
  loom_op_results(op)[0] = result_val;
  loom_op_attrs(op)[0] = loom_attr_i64(-1);

  TerminateFunc();
  auto result = Verify();
  EXPECT_GT(result.error_count, 0u);
  bool found = false;
  for (const auto& msg : collector_.errors) {
    if (msg.find("out of bounds") != std::string::npos) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Expected negative dim index out of bounds error";
}

TEST_F(VerifyTest, AttrInRangeRankPasses) {
  // test.dim with rank-2 source and dim_index=1 → valid.
  loom_type_t tile_2d =
      loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(8), 0);

  loom_type_t arg_types[] = {tile_2d};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);
  loom_value_id_t source = args[0];
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t result_val = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, index_type, &result_val));

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_DIM, 1, 1, 0,
                                          0, 1, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = source;
  loom_op_results(op)[0] = result_val;
  loom_op_attrs(op)[0] = loom_attr_i64(1);

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

// --- Pairwise EQ positive: test.map with correct types ---

TEST_F(VerifyTest, PairwiseEqPassesTestMap) {
  // Build a complete, valid test.map with yield.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 1, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  // Yield the body's block arg (f32 element type matching the tile result).
  loom_region_t* body = loom_test_map_body(op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, op, body);
  loom_value_id_t yield_val = loom_region_entry_arg_id(body, 0);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

}  // namespace
}  // namespace loom
