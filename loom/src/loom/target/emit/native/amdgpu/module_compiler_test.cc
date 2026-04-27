// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/module_compiler.h"

#include <string>

#include "iree/hal/utils/executable_header.h"
#include "iree/schemas/amdgpu_executable_def_reader.h"
#include "iree/schemas/amdgpu_executable_def_verifier.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"

namespace loom {
namespace {

struct ExpectedHalExport {
  // Expected HSA kernel descriptor symbol in HAL export order.
  const char* symbol_name;
  // Expected number of HAL buffer bindings for the export.
  iree_host_size_t binding_count;
};

std::string FlatbufferString(flatbuffers_string_t string) {
  return std::string(string, flatbuffers_string_len(string));
}

void InitializeAmdgpuModuleCompilerContext(loom_context_t* context) {
  IREE_ASSERT_OK(
      loom_op_registry_initialize_context(iree_allocator_system(), context));
}

iree_status_t ParseLowNoopModule(loom_context_t* context,
                                 iree_arena_block_pool_t* block_pool,
                                 loom_module_t** out_module) {
  static const char kSource[] =
      "target.profile @gfx_target preset(\"amdgpu-gfx11\")\n"
      "low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) "
      "@loom_kernel() {\n"
      "  low.return\n"
      "}\n";
  loom_text_parse_options_t parse_options = {
      .max_errors = 20,
  };
  return loom_text_parse(iree_make_cstring_view(kSource),
                         IREE_SV("amdgpu_module_compiler.loom"), context,
                         block_pool, &parse_options, out_module);
}

iree_status_t ParseLowMultiExportArtifactModule(
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_module_t** out_module) {
  static const char kSource[] =
      "target.profile @gfx_target preset(\"amdgpu-gfx11\")\n"
      "target.artifact @gfx_artifact target(@gfx_target)\n"
      "low.kernel.def target(@gfx_target) "
      "export(\"second_kernel\") artifact(@gfx_artifact) ordinal(1) "
      "workgroup_size(64, 1, 1) "
      "@second() {\n"
      "  low.return\n"
      "}\n"
      "low.kernel.def target(@gfx_target) "
      "export(\"first_kernel\") artifact(@gfx_artifact) ordinal(0) "
      "workgroup_size(64, 1, 1) "
      "@first() {\n"
      "  low.return\n"
      "}\n";
  loom_text_parse_options_t parse_options = {
      .max_errors = 20,
  };
  return loom_text_parse(iree_make_cstring_view(kSource),
                         IREE_SV("amdgpu_module_compiler.loom"), context,
                         block_pool, &parse_options, out_module);
}

void ExpectHalExecutableHasExports(
    const loom_amdgpu_hal_executable_t& executable,
    const std::string& expected_format, const ExpectedHalExport* expected,
    iree_host_size_t expected_count) {
  EXPECT_EQ(std::string(executable.executable_format.data,
                        executable.executable_format.size),
            expected_format);

  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_hal_read_executable_flatbuffer_header(
      iree_make_const_byte_span(executable.data, executable.data_length),
      /*unsafe_infer_size=*/false,
      iree_hal_amdgpu_ExecutableDef_file_identifier, &flatbuffer_data));
  const int verify_ret = iree_hal_amdgpu_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  ASSERT_EQ(verify_ret, flatcc_verify_ok)
      << flatcc_verify_error_string(verify_ret);

  iree_hal_amdgpu_ExecutableDef_table_t executable_def =
      iree_hal_amdgpu_ExecutableDef_as_root(flatbuffer_data.data);
  iree_hal_amdgpu_ExportDef_vec_t exports =
      iree_hal_amdgpu_ExecutableDef_exports_get(executable_def);
  ASSERT_EQ(iree_hal_amdgpu_ExportDef_vec_len(exports), expected_count);
  for (iree_host_size_t i = 0; i < expected_count; ++i) {
    iree_hal_amdgpu_ExportDef_table_t export_table =
        iree_hal_amdgpu_ExportDef_vec_at(exports, i);
    EXPECT_EQ(FlatbufferString(
                  iree_hal_amdgpu_ExportDef_symbol_name_get(export_table)),
              expected[i].symbol_name);
    iree_hal_amdgpu_BindingBits_vec_t binding_flags =
        iree_hal_amdgpu_ExportDef_binding_flags_get(export_table);
    EXPECT_EQ(iree_hal_amdgpu_BindingBits_vec_len(binding_flags),
              expected[i].binding_count);
  }

  iree_hal_amdgpu_ModuleDef_vec_t modules =
      iree_hal_amdgpu_ExecutableDef_modules_get(executable_def);
  ASSERT_EQ(iree_hal_amdgpu_ModuleDef_vec_len(modules), 1u);
  iree_hal_amdgpu_ModuleDef_table_t module_def =
      iree_hal_amdgpu_ModuleDef_vec_at(modules, 0);
  flatbuffers_string_t hsaco = iree_hal_amdgpu_ModuleDef_image_get(module_def);
  ASSERT_GE(flatbuffers_string_len(hsaco), 4u);
  EXPECT_EQ(std::string(hsaco, 4), std::string("\x7f"
                                               "ELF",
                                               4));
  const std::string hsaco_image(hsaco, flatbuffers_string_len(hsaco));
  for (iree_host_size_t i = 0; i < expected_count; ++i) {
    EXPECT_NE(hsaco_image.find(expected[i].symbol_name), std::string::npos);
  }
}

void ExpectHalExecutableHasSingleExport(
    const loom_amdgpu_hal_executable_t& executable,
    const std::string& expected_format, const std::string& expected_symbol,
    iree_host_size_t expected_binding_count) {
  const ExpectedHalExport expected = {
      .symbol_name = expected_symbol.c_str(),
      .binding_count = expected_binding_count,
  };
  ExpectHalExecutableHasExports(executable, expected_format, &expected, 1);
}

TEST(AmdgpuModuleCompilerTest, CompilesLowNoopToHalExecutable) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context = {};
  InitializeAmdgpuModuleCompilerContext(&context);

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseLowNoopModule(&context, &block_pool, &module));
  ASSERT_NE(module, nullptr);

  loom_amdgpu_hal_executable_t executable = {};
  IREE_ASSERT_OK(loom_amdgpu_compile_hal_executable(
      module, /*options=*/nullptr, iree_allocator_system(), &executable));
  ExpectHalExecutableHasSingleExport(executable, "amdgcn-amd-amdhsa--gfx1100",
                                     "loom_kernel.kd",
                                     /*expected_binding_count=*/0);

  loom_amdgpu_hal_executable_deinitialize(&executable, iree_allocator_system());
  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(AmdgpuModuleCompilerTest, CompilesArtifactEntriesToOneHalExecutable) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context = {};
  InitializeAmdgpuModuleCompilerContext(&context);

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(
      ParseLowMultiExportArtifactModule(&context, &block_pool, &module));
  ASSERT_NE(module, nullptr);

  loom_amdgpu_hal_executable_t executable = {};
  const loom_amdgpu_module_compile_options_t options = {
      .artifact_symbol = IREE_SV("@gfx_artifact"),
  };
  IREE_ASSERT_OK(loom_amdgpu_compile_hal_executable(
      module, &options, iree_allocator_system(), &executable));
  const ExpectedHalExport expected_exports[] = {
      {
          .symbol_name = "first_kernel.kd",
          .binding_count = 0,
      },
      {
          .symbol_name = "second_kernel.kd",
          .binding_count = 0,
      },
  };
  ExpectHalExecutableHasExports(executable, "amdgcn-amd-amdhsa--gfx1100",
                                expected_exports,
                                IREE_ARRAYSIZE(expected_exports));

  loom_amdgpu_hal_executable_deinitialize(&executable, iree_allocator_system());
  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(AmdgpuModuleCompilerTest, RejectsAmbiguousEntryAndArtifactSelection) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context = {};
  InitializeAmdgpuModuleCompilerContext(&context);

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(
      ParseLowMultiExportArtifactModule(&context, &block_pool, &module));
  ASSERT_NE(module, nullptr);

  loom_amdgpu_hal_executable_t executable = {};
  const loom_amdgpu_module_compile_options_t options = {
      .entry_symbol = IREE_SV("@first"),
      .artifact_symbol = IREE_SV("@gfx_artifact"),
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_compile_hal_executable(module, &options,
                                         iree_allocator_system(), &executable));

  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(AmdgpuModuleCompilerTest, TargetCpuOptionSpecializesHalFormat) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context = {};
  InitializeAmdgpuModuleCompilerContext(&context);

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseLowNoopModule(&context, &block_pool, &module));
  ASSERT_NE(module, nullptr);

  loom_amdgpu_hal_executable_t executable = {};
  const loom_amdgpu_module_compile_options_t options = {
      .target_cpu = IREE_SV("gfx1101"),
  };
  IREE_ASSERT_OK(loom_amdgpu_compile_hal_executable(
      module, &options, iree_allocator_system(), &executable));
  EXPECT_EQ(std::string(executable.executable_format.data,
                        executable.executable_format.size),
            "amdgcn-amd-amdhsa--gfx1101");

  loom_amdgpu_hal_executable_deinitialize(&executable, iree_allocator_system());
  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
