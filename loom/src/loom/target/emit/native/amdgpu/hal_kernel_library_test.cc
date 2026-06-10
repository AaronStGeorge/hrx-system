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
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/target_info.h"
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

std::string StringViewToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

uint32_t LoadLeU32(const uint8_t* bytes, size_t offset) {
  return (uint32_t)bytes[offset] | ((uint32_t)bytes[offset + 1] << 8) |
         ((uint32_t)bytes[offset + 2] << 16) |
         ((uint32_t)bytes[offset + 3] << 24);
}

class AmdgpuHalKernelLibraryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(InitializeAmdgpuContext(&context_));
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void ParseGfx11Kernel(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx1100> @gfx_target\n"
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  low.return\n"
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

  void ParseGfx11MultiKernel(loom_module_t** out_module) {
    static const char kSource[] =
        "amdgpu.target<gfx1100> @gfx_target\n"
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@first_kernel() {\n"
        "  low.return\n"
        "}\n"
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@second_kernel() {\n"
        "  low.return\n"
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
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  low.return\n"
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

  void ParseKernelForProcessor(const loom_amdgpu_processor_info_t* processor,
                               loom_module_t** out_module) {
    const loom_amdgpu_target_record_info_t* record_info =
        loom_amdgpu_target_record_default_info_for_descriptor_set(
            processor->descriptor_set_ordinal);
    ASSERT_NE(record_info, nullptr) << StringViewToString(processor->processor);

    std::string source = "amdgpu.target<";
    source.append(record_info->default_processor_name.data,
                  record_info->default_processor_name.size);
    source += "> @gfx_target";
    if (!iree_string_view_equal(processor->processor,
                                record_info->default_processor_name)) {
      source += " {processor = \"";
      source.append(processor->processor.data, processor->processor.size);
      source += "\"}";
    }
    source +=
        "\n"
        "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  low.return\n"
        "}\n";

    DiagnosticCapture parse_capture;
    loom_text_parse_options_t parse_options = {
        .diagnostic_sink = parse_capture.sink(),
        .max_errors = 20,
    };
    IREE_ASSERT_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("amdgpu_emit_test.loom"), &context_,
                        &block_pool_, &parse_options, out_module))
        << StringViewToString(processor->processor);
    ASSERT_TRUE(parse_capture.diagnostics.empty())
        << StringViewToString(processor->processor);
    ASSERT_NE(*out_module, nullptr) << StringViewToString(processor->processor);
  }

  bool IsDescriptorSetLinked(iree_string_view_t descriptor_set_key) const {
    return loom_low_descriptor_registry_lookup(&low_registry_.registry,
                                               descriptor_set_key) != nullptr;
  }

  bool IsProcessorDescriptorSetLinked(
      const loom_amdgpu_processor_info_t* processor) const {
    return IsDescriptorSetLinked(processor->descriptor_set_key);
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
  loom_target_low_descriptor_registry_t low_registry_ = {};
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
  if (!IsDescriptorSetLinked(IREE_SV("amdgpu.cdna3.core"))) {
    GTEST_SKIP() << "amdgpu.cdna3.core is not linked in this build";
  }
  DiagnosticCapture capture;
  bool emitted = false;
  ASSERT_NO_FATAL_FAILURE(EmitGfx942Kernel(&capture, &emitted));

  EXPECT_TRUE(emitted);
  EXPECT_TRUE(capture.diagnostics.empty());
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsEveryLinkedSupportedProcessor) {
  iree_host_size_t linked_supported_count = 0;
  const iree_host_size_t processor_count =
      loom_amdgpu_target_info_processor_count();
  for (iree_host_size_t i = 0; i < processor_count; ++i) {
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_processor_at(i);
    ASSERT_NE(processor, nullptr);
    bool hsaco_supported = false;
    IREE_ASSERT_OK(loom_amdgpu_target_info_processor_supports_hsaco(
        processor, &hsaco_supported));
    if (!hsaco_supported) {
      continue;
    }
    if (!IsProcessorDescriptorSetLinked(processor)) {
      continue;
    }
    ++linked_supported_count;

    loom_module_t* module = nullptr;
    ASSERT_NO_FATAL_FAILURE(ParseKernelForProcessor(processor, &module));

    DiagnosticCapture capture;
    loom_amdgpu_hal_kernel_library_t library = {};
    loom_amdgpu_hal_kernel_library_options_t options = {
        .diagnostic_sink = capture.sink(),
        .max_errors = 20,
    };
    bool emitted = false;
    IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
        module, &options, iree_allocator_system(), &emitted, &library))
        << StringViewToString(processor->processor);

    EXPECT_TRUE(emitted) << StringViewToString(processor->processor);
    EXPECT_TRUE(capture.diagnostics.empty())
        << StringViewToString(processor->processor);
    EXPECT_NE(library.hsaco_data, nullptr)
        << StringViewToString(processor->processor);
    EXPECT_GT(library.hsaco_data_length, 64u)
        << StringViewToString(processor->processor);
    if (library.hsaco_data_length > 64u) {
      EXPECT_EQ(LoadLeU32(library.hsaco_data, 48),
                processor->elf_machine_flags | processor->elf_feature_flags)
          << StringViewToString(processor->processor);
    }
    EXPECT_NE(iree_string_view_find(library.executable_format,
                                    processor->processor, 0),
              IREE_STRING_VIEW_NPOS)
        << StringViewToString(processor->processor);
    ASSERT_EQ(library.export_count, 1u)
        << StringViewToString(processor->processor);
    EXPECT_TRUE(iree_string_view_equal(library.exports[0].symbol_name,
                                       IREE_SV("loom_kernel.kd")))
        << StringViewToString(processor->processor);
    EXPECT_EQ(library.exports[0].workgroup_size.x, 64u)
        << StringViewToString(processor->processor);
    EXPECT_EQ(library.exports[0].workgroup_size.y, 1u)
        << StringViewToString(processor->processor);
    EXPECT_EQ(library.exports[0].workgroup_size.z, 1u)
        << StringViewToString(processor->processor);

    loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                                iree_allocator_system());
    loom_module_free(module);
  }
  EXPECT_GE(linked_supported_count, 1u);
}

TEST_F(AmdgpuHalKernelLibraryTest, EmitsAllCompatibleKernels) {
  loom_module_t* module = nullptr;
  ASSERT_NO_FATAL_FAILURE(ParseGfx11MultiKernel(&module));

  DiagnosticCapture capture;
  loom_amdgpu_hal_kernel_library_t library = {};
  loom_amdgpu_hal_kernel_library_options_t options = {
      .diagnostic_sink = capture.sink(),
      .max_errors = 20,
  };
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_emit_hal_kernel_library(
      module, &options, iree_allocator_system(), &emitted, &library));

  EXPECT_TRUE(emitted);
  EXPECT_TRUE(capture.diagnostics.empty());
  EXPECT_EQ(library.export_count, 2u);

  loom_amdgpu_hal_kernel_library_deinitialize(&library,
                                              iree_allocator_system());
  loom_module_free(module);
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
