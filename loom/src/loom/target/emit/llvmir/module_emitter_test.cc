// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/module_emitter.h"

#include <memory>
#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/llvmir/descriptors/low_registry.h"
#include "loom/target/emit/llvmir/text_writer.h"
#include "loom/target/emit/llvmir/verify.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnosticEmission;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::DiagnosticEmissionCapture;
using ModulePtr = ::loom::testing::ModulePtr;
using LlvmirModulePtr =
    std::unique_ptr<loom_llvmir_module_t, decltype(&loom_llvmir_module_free)>;

class LlvmirModuleEmitterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_llvmir_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseModule(const char* source) {
    DiagnosticCapture parse_capture;
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/parse_capture.sink(),
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("llvmir_module_emitter_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    EXPECT_TRUE(parse_capture.diagnostics.empty());
    EXPECT_NE(module, nullptr);
    return ModulePtr(module);
  }

  iree_status_t EmitLowModule(loom_module_t* module,
                              DiagnosticEmissionCapture* capture,
                              LlvmirModulePtr* out_module) {
    iree_arena_allocator_t scratch_arena;
    iree_arena_initialize(&block_pool_, &scratch_arena);
    loom_llvmir_module_t* raw_module = nullptr;
    iree_status_t status = loom_llvmir_emit_low_module(
        module, &low_registry_.registry, loom_target_selection_empty(),
        capture->emitter(), &scratch_arena, nullptr, &raw_module,
        iree_allocator_system());
    if (iree_status_is_ok(status) && raw_module != nullptr) {
      out_module->reset(raw_module);
    }
    iree_arena_deinitialize(&scratch_arena);
    return status;
  }

  iree_status_t WriteText(loom_llvmir_module_t* module, std::string* out_text) {
    IREE_RETURN_IF_ERROR(loom_llvmir_verify_module(module));
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    iree_status_t status = loom_llvmir_text_write_module(module, &stream);
    if (iree_status_is_ok(status)) {
      iree_string_view_t view = iree_string_builder_view(&builder);
      out_text->assign(view.data, view.size);
    }
    iree_string_builder_deinitialize(&builder);
    return status;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

TEST_F(LlvmirModuleEmitterTest, EmitsMultipleLowFunctionsInModuleOrder) {
  ModulePtr module = ParseModule(R"(
llvmir.target<object> @target {
  triple = "loom-direct64-unknown-none",
  data_layout = "e-p:64:64-i64:64-n8:16:32:64-S128"
}

low.func.def target(@target) abi(object_function) @first(%lhs: reg<llvmir.i32>, %rhs: reg<llvmir.i32>) -> (reg<llvmir.i32>) asm<llvmir.generic.core> {
  %sum = add.i32 %lhs, %rhs
  return %sum
}

low.func.def target(@target) abi(object_function) @second(%input_view: reg<llvmir.ptr>, %output_view: reg<llvmir.ptr>, %bounded_i: reg<llvmir.i64>) -> (reg<llvmir.i32>) asm<llvmir.generic.core> {
  %loaded = load.indexed.i32 %input_view, %bounded_i, 16, 4
  store.indexed.i32 %loaded, %output_view, %bounded_i, 32, 4
  return %loaded
}
)");

  DiagnosticEmissionCapture capture;
  LlvmirModulePtr llvmir_module(nullptr, loom_llvmir_module_free);
  IREE_ASSERT_OK(EmitLowModule(module.get(), &capture, &llvmir_module));
  ASSERT_NE(llvmir_module, nullptr);
  EXPECT_TRUE(capture.emissions.empty());

  std::string text;
  IREE_ASSERT_OK(WriteText(llvmir_module.get(), &text));
  const size_t first_position = text.find("@first(");
  const size_t second_position = text.find("@second(");
  ASSERT_NE(first_position, std::string::npos) << text;
  ASSERT_NE(second_position, std::string::npos) << text;
  EXPECT_LT(first_position, second_position) << text;
  EXPECT_NE(text.find("%sum = add i32 %lhs, %rhs"), std::string::npos) << text;
  EXPECT_NE(text.find("getelementptr i8, ptr %input_view"), std::string::npos)
      << text;
  EXPECT_NE(text.find("store i32 %loaded"), std::string::npos) << text;
}

TEST_F(LlvmirModuleEmitterTest, ReportsUnsupportedFunctionShapeAsDiagnostic) {
  ModulePtr module = ParseModule(R"(
llvmir.target<object> @target {
  triple = "loom-direct64-unknown-none",
  data_layout = "e-p:64:64-i64:64-n8:16:32:64-S128"
}

low.func.def target(@target) abi(object_function) @multi_result(%lhs: reg<llvmir.i32>, %rhs: reg<llvmir.i32>) -> (reg<llvmir.i32>, reg<llvmir.i32>) asm<llvmir.generic.core> {
  %sum = add.i32 %lhs, %rhs
  return %sum, %lhs
}
)");

  DiagnosticEmissionCapture capture;
  LlvmirModulePtr llvmir_module(nullptr, loom_llvmir_module_free);
  IREE_ASSERT_OK(EmitLowModule(module.get(), &capture, &llvmir_module));
  EXPECT_EQ(llvmir_module, nullptr);
  ASSERT_EQ(capture.emissions.size(), 1u);

  const CapturedDiagnosticEmission& emission = capture.emissions[0];
  EXPECT_EQ(emission.error, LOOM_ERR_TARGET_054);
  ASSERT_EQ(emission.string_params.size(), 3u);
  EXPECT_EQ(emission.string_params[0], "multi_result");
  EXPECT_EQ(emission.string_params[1], "llvmir.low");
  EXPECT_EQ(emission.string_params[2], "function_result");
  ASSERT_EQ(emission.u32_params.size(), 2u);
  EXPECT_EQ(emission.u32_params[0], 2u);
  EXPECT_EQ(emission.u32_params[1], 1u);
}

}  // namespace
}  // namespace loom
