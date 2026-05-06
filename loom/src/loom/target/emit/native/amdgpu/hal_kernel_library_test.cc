// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
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

class AmdgpuHalKernelLibraryTest : public ::testing::Test {
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
    IREE_ASSERT_OK(loom_text_parse(iree_make_cstring_view(kSource),
                                   IREE_SV("amdgpu_emit_test.loom"), &context_,
                                   &block_pool_, &parse_options, out_module));
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
    IREE_ASSERT_OK(loom_text_parse(iree_make_cstring_view(kSource),
                                   IREE_SV("amdgpu_emit_test.loom"), &context_,
                                   &block_pool_, &parse_options, out_module));
    ASSERT_TRUE(parse_capture.diagnostics.empty());
    ASSERT_NE(*out_module, nullptr);
  }

  void EmitWithProcessor(iree_string_view_t processor,
                         DiagnosticCapture* capture, bool* out_emitted) {
    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(ParseGfx11Kernel(&module));

    loom_amdgpu_hal_kernel_library_t library = {};
    loom_amdgpu_hal_kernel_library_options_t options = {
        .processor = processor,
        .diagnostic_sink = capture->sink(),
        .max_errors = 20,
    };
    iree_status_t status = loom_amdgpu_emit_hal_kernel_library(
        module, &options, iree_allocator_system(), out_emitted, &library);
    loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                                iree_allocator_system());
    loom_module_free(module);
    IREE_ASSERT_OK(status);
  }

  void EmitGfx942Kernel(DiagnosticCapture* capture, bool* out_emitted) {
    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(ParseGfx942Kernel(&module));

    loom_amdgpu_hal_kernel_library_t library = {};
    loom_amdgpu_hal_kernel_library_options_t options = {
        .processor = IREE_SV("gfx942"),
        .diagnostic_sink = capture->sink(),
        .max_errors = 20,
    };
    iree_status_t status = loom_amdgpu_emit_hal_kernel_library(
        module, &options, iree_allocator_system(), out_emitted, &library);
    loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                                iree_allocator_system());
    loom_module_free(module);
    IREE_ASSERT_OK(status);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
};

TEST_F(AmdgpuHalKernelLibraryTest, UnknownProcessorEmitsDiagnostic) {
  DiagnosticCapture capture;
  bool emitted = true;
  ASSERT_NO_FATAL_FAILURE(
      EmitWithProcessor(IREE_SV("gfx9999"), &capture, &emitted));

  EXPECT_FALSE(emitted);
  ASSERT_EQ(capture.diagnostics.size(), 1u);
  const CapturedDiagnostic* diagnostic =
      FindDiagnostic(capture, LOOM_ERR_AMDGPU_003);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "gfx9999");
}

TEST_F(AmdgpuHalKernelLibraryTest,
       ProcessorWithoutDescriptorSetEmitsDiagnostic) {
  DiagnosticCapture capture;
  bool emitted = true;
  ASSERT_NO_FATAL_FAILURE(
      EmitWithProcessor(IREE_SV("gfx908"), &capture, &emitted));

  EXPECT_FALSE(emitted);
  ASSERT_EQ(capture.diagnostics.size(), 1u);
  const CapturedDiagnostic* diagnostic =
      FindDiagnostic(capture, LOOM_ERR_AMDGPU_004);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "gfx908");
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsGfx942Kernel) {
  DiagnosticCapture capture;
  bool emitted = false;
  ASSERT_NO_FATAL_FAILURE(EmitGfx942Kernel(&capture, &emitted));

  EXPECT_TRUE(emitted);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(AmdgpuHalKernelLibraryTest, DescriptorSetMismatchEmitsDiagnostic) {
  DiagnosticCapture capture;
  bool emitted = true;
  ASSERT_NO_FATAL_FAILURE(
      EmitWithProcessor(IREE_SV("gfx950"), &capture, &emitted));

  EXPECT_FALSE(emitted);
  ASSERT_EQ(capture.diagnostics.size(), 1u);
  const CapturedDiagnostic* diagnostic =
      FindDiagnostic(capture, LOOM_ERR_AMDGPU_005);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(GetStringParam(*diagnostic, 0), "gfx950");
  EXPECT_EQ(GetStringParam(*diagnostic, 1), "amdgpu.cdna4.core");
  EXPECT_EQ(GetStringParam(*diagnostic, 2), "gfx_target");
  EXPECT_EQ(GetStringParam(*diagnostic, 3), "amdgpu.rdna3.core");
}

}  // namespace
}  // namespace loom
