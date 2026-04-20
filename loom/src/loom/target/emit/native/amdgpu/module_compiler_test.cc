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

iree_status_t ParseLowMemoryAluStressModule(loom_context_t* context,
                                            iree_arena_block_pool_t* block_pool,
                                            loom_module_t** out_module) {
  static const char kSource[] =
      "target.preset @gfx_target {key = \"amdgpu-gfx11\", source = "
      "@loom_kernel}\n"
      "low.func.def target(@gfx_target) @loom_kernel() {\n"
      "  %tid = low.live_in<amdgpu.workitem_id.x> : reg<amdgpu.vgpr>\n"
      "  %zero_v = low.const<amdgpu.v_mov_b32> {imm32 = 0} : "
      "reg<amdgpu.vgpr>\n"
      "  %four = low.const<amdgpu.v_mov_b32> {imm32 = 4} : "
      "reg<amdgpu.vgpr>\n"
      "  %byte_offset = low.op<amdgpu.v_mul_lo_u32>(%tid, %four) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  %source = low.resource @binding0 : reg<amdgpu.sgpr x4>\n"
      "  %wide_target = low.resource @binding1 : reg<amdgpu.sgpr x4>\n"
      "  %scalar_target = low.resource @binding2 : reg<amdgpu.sgpr x4>\n"
      "  %zero_s = low.const<amdgpu.s_mov_b32> {imm32 = 0} : "
      "reg<amdgpu.sgpr>\n"
      "  %wide = low.op<amdgpu.buffer_load_b128>(%source, %zero_v, "
      "%zero_s) {offset = 0} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, "
      "reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr x4>\n"
      "  %seven = low.const<amdgpu.v_mov_b32> {imm32 = 7} : "
      "reg<amdgpu.vgpr>\n"
      "  %sum = low.op<amdgpu.v_add_u32>(%tid, %seven) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  low.op<amdgpu.buffer_store_b128>(%wide, %wide_target, %zero_v, "
      "%zero_s) {offset = 0} : (reg<amdgpu.vgpr x4>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.op<amdgpu.buffer_store_dword>(%sum, %scalar_target, "
      "%byte_offset, %zero_s) {offset = 0} : (reg<amdgpu.vgpr>, "
      "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n"
      "low.abi.resource @binding0 {function = @loom_kernel, kind = "
      "hal_buffer_resource, index = 0, semantic_type = hal.buffer, "
      "abi_type = reg<amdgpu.sgpr x4>}\n"
      "low.abi.resource @binding1 {function = @loom_kernel, kind = "
      "hal_buffer_resource, index = 1, semantic_type = hal.buffer, "
      "abi_type = reg<amdgpu.sgpr x4>}\n"
      "low.abi.resource @binding2 {function = @loom_kernel, kind = "
      "hal_buffer_resource, index = 2, semantic_type = hal.buffer, "
      "abi_type = reg<amdgpu.sgpr x4>}\n";
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

iree_status_t ParseSemanticBufferStoreModule(
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_module_t** out_module) {
  static const char kSource[] =
      "target.preset @gfx_target {key = \"amdgpu-gfx11\", source = "
      "@loom_kernel}\n"
      "func.def @loom_kernel(%output: buffer) {\n"
      "  %zero = index.constant 0 : offset\n"
      "  %view = buffer.view %output[%zero] : buffer -> view<1xi32, #dense>\n"
      "  %value = vector.constant 42 : vector<1xi32>\n"
      "  vector.store %value, %view[0] : vector<1xi32>, view<1xi32, #dense>\n"
      "  func.return\n"
      "}\n";
  loom_text_parse_options_t parse_options = {
      .max_errors = 20,
  };
  return loom_text_parse(iree_make_cstring_view(kSource),
                         IREE_SV("amdgpu_module_compiler.loom"), context,
                         block_pool, &parse_options, out_module);
}

iree_status_t ParseSemanticBufferLoadAddStoreModule(
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_module_t** out_module) {
  static const char kSource[] =
      "target.preset @gfx_target {key = \"amdgpu-gfx11\", source = "
      "@loom_kernel}\n"
      "func.def @loom_kernel(%input: buffer, %output: buffer) {\n"
      "  %zero = index.constant 0 : offset\n"
      "  %input_view = buffer.view %input[%zero] : buffer -> "
      "view<1xi32, #dense>\n"
      "  %output_view = buffer.view %output[%zero] : buffer -> "
      "view<1xi32, #dense>\n"
      "  %loaded = vector.load %input_view[0] : view<1xi32, #dense> -> "
      "vector<1xi32>\n"
      "  %value = vector.constant 7 : vector<1xi32>\n"
      "  %sum = vector.addi %loaded, %value : vector<1xi32>\n"
      "  vector.store %sum, %output_view[0] : vector<1xi32>, "
      "view<1xi32, #dense>\n"
      "  func.return\n"
      "}\n";
  loom_text_parse_options_t parse_options = {
      .max_errors = 20,
  };
  return loom_text_parse(iree_make_cstring_view(kSource),
                         IREE_SV("amdgpu_module_compiler.loom"), context,
                         block_pool, &parse_options, out_module);
}

iree_status_t ParseSemanticWorkitemIndexedLoadAddStoreModule(
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    loom_module_t** out_module) {
  static const char kSource[] =
      "target.preset @gfx_target {key = \"amdgpu-gfx11\", source = "
      "@loom_kernel}\n"
      "func.def @loom_kernel(%input: buffer, %output: buffer) {\n"
      "  %tid = kernel.workitem.id<x> : index\n"
      "  %zero = index.constant 0 : offset\n"
      "  %input_view = buffer.view %input[%zero] : buffer -> "
      "view<64xi32, #dense>\n"
      "  %output_view = buffer.view %output[%zero] : buffer -> "
      "view<64xi32, #dense>\n"
      "  %loaded = vector.load %input_view[%tid] : view<64xi32, #dense> -> "
      "vector<1xi32>\n"
      "  %value = vector.constant 7 : vector<1xi32>\n"
      "  %sum = vector.addi %loaded, %value : vector<1xi32>\n"
      "  vector.store %sum, %output_view[%tid] : vector<1xi32>, "
      "view<64xi32, #dense>\n"
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

TEST(AmdgpuModuleCompilerTest, CompilesLowMemoryAluStressToHalExecutable) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context = {};
  IREE_ASSERT_OK(
      loom_testing_context_initialize_all(iree_allocator_system(), &context));

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseLowMemoryAluStressModule(&context, &block_pool, &module));
  ASSERT_NE(module, nullptr);

  loom_amdgpu_hal_executable_t executable = {};
  IREE_ASSERT_OK(loom_amdgpu_compile_hal_executable(
      module, /*options=*/nullptr, iree_allocator_system(), &executable));
  ExpectHalExecutableHasSingleExport(executable, "amdgcn-amd-amdhsa--gfx1100",
                                     "loom_kernel.kd",
                                     /*expected_binding_count=*/3);

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

TEST(AmdgpuModuleCompilerTest, CompilesSemanticBufferStoreToHalExecutable) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context = {};
  IREE_ASSERT_OK(
      loom_testing_context_initialize_all(iree_allocator_system(), &context));

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(
      ParseSemanticBufferStoreModule(&context, &block_pool, &module));
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

TEST(AmdgpuModuleCompilerTest,
     CompilesSemanticBufferLoadAddStoreToHalExecutable) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context = {};
  IREE_ASSERT_OK(
      loom_testing_context_initialize_all(iree_allocator_system(), &context));

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(
      ParseSemanticBufferLoadAddStoreModule(&context, &block_pool, &module));
  ASSERT_NE(module, nullptr);

  loom_amdgpu_hal_executable_t executable = {};
  IREE_ASSERT_OK(loom_amdgpu_compile_hal_executable(
      module, /*options=*/nullptr, iree_allocator_system(), &executable));
  ExpectHalExecutableHasSingleExport(executable, "amdgcn-amd-amdhsa--gfx1100",
                                     "loom_kernel.kd",
                                     /*expected_binding_count=*/2);

  loom_amdgpu_hal_executable_deinitialize(&executable, iree_allocator_system());
  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(AmdgpuModuleCompilerTest,
     CompilesSemanticWorkitemIndexedLoadAddStoreToHalExecutable) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context = {};
  IREE_ASSERT_OK(
      loom_testing_context_initialize_all(iree_allocator_system(), &context));

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(ParseSemanticWorkitemIndexedLoadAddStoreModule(
      &context, &block_pool, &module));
  ASSERT_NE(module, nullptr);

  loom_amdgpu_hal_executable_t executable = {};
  IREE_ASSERT_OK(loom_amdgpu_compile_hal_executable(
      module, /*options=*/nullptr, iree_allocator_system(), &executable));
  ExpectHalExecutableHasSingleExport(executable, "amdgcn-amd-amdhsa--gfx1100",
                                     "loom_kernel.kd",
                                     /*expected_binding_count=*/2);

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
