// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/module_compiler.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_defs.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/testing/diagnostic_matchers.h"

namespace loom {
namespace {

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::FindDiagnostic;
using ::loom::testing::GetStringParam;

iree_status_t InitializeAmdgpuContext(loom_context_t* context) {
  loom_context_initialize(iree_allocator_system(), context);
  iree_status_t status = loom_op_registry_register_all_dialects(context);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_ops_register_dialect(context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(context);
  }
  if (!iree_status_is_ok(status)) {
    loom_context_deinitialize(context);
  }
  return status;
}

class AmdgpuModuleCompilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(InitializeAmdgpuContext(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void ParseGfx11Kernel(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx1100> @gfx_target\n"
        "kernel.def target(@gfx_target) @loom_kernel() {\n"
        "  %c1 = index.constant 1 : index\n"
        "  %c64 = index.constant 64 : index\n"
        "  kernel.launch.config workgroups(%c1, %c1, %c1) "
        "workgroup_size(%c64, %c1, %c1) : index\n"
        "} launch {\n"
        "  kernel.return\n"
        "}\n";
    DiagnosticCapture parse_capture;
    loom_text_parse_options_t parse_options = {
        .diagnostic_sink = parse_capture.sink(),
        .max_errors = 20,
    };
    IREE_ASSERT_OK(loom_text_parse(
        iree_make_cstring_view(kSource), IREE_SV("amdgpu_compile_test.loom"),
        &context_, &block_pool_, &parse_options, out_module));
    ASSERT_TRUE(parse_capture.diagnostics.empty());
    ASSERT_NE(*out_module, nullptr);
  }

  void ParseGfx942Kernel(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx942> @gfx_target\n"
        "kernel.def target(@gfx_target) @loom_kernel() {\n"
        "  %c1 = index.constant 1 : index\n"
        "  %c64 = index.constant 64 : index\n"
        "  kernel.launch.config workgroups(%c1, %c1, %c1) "
        "workgroup_size(%c64, %c1, %c1) : index\n"
        "} launch {\n"
        "  kernel.return\n"
        "}\n";
    DiagnosticCapture parse_capture;
    loom_text_parse_options_t parse_options = {
        .diagnostic_sink = parse_capture.sink(),
        .max_errors = 20,
    };
    IREE_ASSERT_OK(loom_text_parse(
        iree_make_cstring_view(kSource), IREE_SV("amdgpu_compile_test.loom"),
        &context_, &block_pool_, &parse_options, out_module));
    ASSERT_TRUE(parse_capture.diagnostics.empty());
    ASSERT_NE(*out_module, nullptr);
  }

  void CompileWithTargetCpu(iree_string_view_t target_cpu,
                            DiagnosticCapture* capture, bool* out_compiled) {
    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(ParseGfx11Kernel(&module));

    loom_amdgpu_hal_executable_t executable = {};
    loom_amdgpu_module_compile_options_t options = {
        .target_cpu = target_cpu,
        .diagnostic_sink = capture->sink(),
        .max_errors = 20,
    };
    iree_status_t status = loom_amdgpu_compile_hal_executable(
        module, &options, iree_allocator_system(), out_compiled, &executable);
    loom_amdgpu_hal_executable_deinitialize(&executable,
                                            iree_allocator_system());
    loom_module_free(module);
    IREE_ASSERT_OK(status);
  }

  void CompileGfx942Kernel(DiagnosticCapture* capture, bool* out_compiled) {
    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(ParseGfx942Kernel(&module));

    loom_amdgpu_hal_executable_t executable = {};
    loom_amdgpu_module_compile_options_t options = {
        .target_cpu = IREE_SV("gfx942"),
        .diagnostic_sink = capture->sink(),
        .max_errors = 20,
    };
    iree_status_t status = loom_amdgpu_compile_hal_executable(
        module, &options, iree_allocator_system(), out_compiled, &executable);
    loom_amdgpu_hal_executable_deinitialize(&executable,
                                            iree_allocator_system());
    loom_module_free(module);
    IREE_ASSERT_OK(status);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
};

TEST_F(AmdgpuModuleCompilerTest, UnknownTargetCpuEmitsDiagnostic) {
  DiagnosticCapture capture;
  bool compiled = true;
  ASSERT_NO_FATAL_FAILURE(
      CompileWithTargetCpu(IREE_SV("gfx9999"), &capture, &compiled));

  EXPECT_FALSE(compiled);
  ASSERT_EQ(capture.diagnostics.size(), 1u);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_AMDGPU, 3));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "gfx9999");
}

TEST_F(AmdgpuModuleCompilerTest, TargetCpuWithoutDescriptorSetEmitsDiagnostic) {
  DiagnosticCapture capture;
  bool compiled = true;
  ASSERT_NO_FATAL_FAILURE(
      CompileWithTargetCpu(IREE_SV("gfx908"), &capture, &compiled));

  EXPECT_FALSE(compiled);
  ASSERT_EQ(capture.diagnostics.size(), 1u);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_AMDGPU, 4));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "gfx908");
}

TEST_F(AmdgpuModuleCompilerTest, CompilesGfx942Kernel) {
  DiagnosticCapture capture;
  bool compiled = false;
  ASSERT_NO_FATAL_FAILURE(CompileGfx942Kernel(&capture, &compiled));

  EXPECT_TRUE(compiled);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(AmdgpuModuleCompilerTest, DescriptorSetMismatchEmitsDiagnostic) {
  DiagnosticCapture capture;
  bool compiled = true;
  ASSERT_NO_FATAL_FAILURE(
      CompileWithTargetCpu(IREE_SV("gfx950"), &capture, &compiled));

  EXPECT_FALSE(compiled);
  ASSERT_EQ(capture.diagnostics.size(), 1u);
  const CapturedDiagnostic* diagnostic = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_AMDGPU, 5));
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "gfx950");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "amdgpu.cdna4.core");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "gfx_target");
  EXPECT_EQ(GetStringParam(*diagnostic, 3), "amdgpu.rdna3.core");
}

}  // namespace
}  // namespace loom
