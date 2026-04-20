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
#include "loom/ir/module.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

std::string FlatbufferString(flatbuffers_string_t string) {
  return std::string(string, flatbuffers_string_len(string));
}

iree_status_t ParseLowNoopModule(loom_context_t* context,
                                 iree_arena_block_pool_t* block_pool,
                                 loom_module_t** out_module) {
  static const char kSource[] =
      "target.preset @gfx_target {key = \"amdgpu-gfx11\", source = "
      "@loom_kernel}\n"
      "low.func.def target(@gfx_target) @loom_kernel() {\n"
      "  low.return\n"
      "}\n";
  loom_text_parse_options_t parse_options = {
      .max_errors = 20,
  };
  return loom_text_parse(iree_make_cstring_view(kSource),
                         IREE_SV("amdgpu_module_compiler.loom"), context,
                         block_pool, &parse_options, out_module);
}

iree_status_t ParseSemanticVectorModule(loom_context_t* context,
                                        iree_arena_block_pool_t* block_pool,
                                        loom_module_t** out_module) {
  static const char kSource[] =
      "target.preset @gfx_target {key = \"amdgpu-gfx11\", source = "
      "@loom_kernel}\n"
      "func.def @loom_kernel() {\n"
      "  %lhs = vector.constant 3 : vector<1xi32>\n"
      "  %rhs = vector.constant 7 : vector<1xi32>\n"
      "  %sum = vector.addi %lhs, %rhs : vector<1xi32>\n"
      "  %product = vector.muli %sum, %rhs : vector<1xi32>\n"
      "  func.return\n"
      "}\n";
  loom_text_parse_options_t parse_options = {
      .max_errors = 20,
  };
  return loom_text_parse(iree_make_cstring_view(kSource),
                         IREE_SV("amdgpu_module_compiler.loom"), context,
                         block_pool, &parse_options, out_module);
}

iree_status_t ParseSemanticBufferModule(loom_context_t* context,
                                        iree_arena_block_pool_t* block_pool,
                                        loom_module_t** out_module) {
  static const char kSource[] =
      "target.preset @gfx_target {key = \"amdgpu-gfx11\", source = "
      "@loom_kernel}\n"
      "func.def @loom_kernel(%output: buffer) {\n"
      "  func.return\n"
      "}\n";
  loom_text_parse_options_t parse_options = {
      .max_errors = 20,
  };
  return loom_text_parse(iree_make_cstring_view(kSource),
                         IREE_SV("amdgpu_module_compiler.loom"), context,
                         block_pool, &parse_options, out_module);
}

void ExpectHalExecutableHasSingleExport(
    const loom_amdgpu_hal_executable_t& executable,
    const std::string& expected_format, const std::string& expected_symbol,
    iree_host_size_t expected_binding_count) {
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
  ASSERT_EQ(iree_hal_amdgpu_ExportDef_vec_len(exports), 1u);
  iree_hal_amdgpu_ExportDef_table_t export_table =
      iree_hal_amdgpu_ExportDef_vec_at(exports, 0);
  EXPECT_EQ(
      FlatbufferString(iree_hal_amdgpu_ExportDef_symbol_name_get(export_table)),
      expected_symbol);
  iree_hal_amdgpu_BindingBits_vec_t binding_flags =
      iree_hal_amdgpu_ExportDef_binding_flags_get(export_table);
  EXPECT_EQ(iree_hal_amdgpu_BindingBits_vec_len(binding_flags),
            expected_binding_count);

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
}

TEST(AmdgpuModuleCompilerTest, CompilesLowNoopToHalExecutable) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context = {};
  IREE_ASSERT_OK(
      loom_testing_context_initialize_all(iree_allocator_system(), &context));

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

TEST(AmdgpuModuleCompilerTest, CompilesSemanticBufferResourceToHalExecutable) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context = {};
  IREE_ASSERT_OK(
      loom_testing_context_initialize_all(iree_allocator_system(), &context));

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseSemanticBufferModule(&context, &block_pool, &module));
  ASSERT_NE(module, nullptr);

  loom_amdgpu_hal_executable_t executable = {};
  IREE_ASSERT_OK(loom_amdgpu_compile_hal_executable(
      module, /*options=*/nullptr, iree_allocator_system(), &executable));
  ExpectHalExecutableHasSingleExport(executable, "amdgcn-amd-amdhsa--gfx1100",
                                     "loom_kernel.kd",
                                     /*expected_binding_count=*/1);

  loom_amdgpu_hal_executable_deinitialize(&executable, iree_allocator_system());
  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(AmdgpuModuleCompilerTest, CompilesSemanticVectorSourceToHalExecutable) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context = {};
  IREE_ASSERT_OK(
      loom_testing_context_initialize_all(iree_allocator_system(), &context));

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseSemanticVectorModule(&context, &block_pool, &module));
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

TEST(AmdgpuModuleCompilerTest, TargetCpuOptionSpecializesHalFormat) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context = {};
  IREE_ASSERT_OK(
      loom_testing_context_initialize_all(iree_allocator_system(), &context));

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
