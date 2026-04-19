// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_assembly.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/presets.h"
#include "loom/target/tool/llvm.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

std::string ToString(const loom_llvm_tool_output_t& output) {
  return output.data ? std::string(output.data, output.length) : std::string();
}

bool IsToolUnavailable(iree_status_t status) {
  return iree_status_is_not_found(status) ||
         iree_status_is_unavailable(status) ||
         iree_status_is_unimplemented(status);
}

bool VersionTextListsAmdgcnTarget(const std::string& version_text) {
  return version_text.find("amdgcn") != std::string::npos ||
         version_text.find("AMDGPU") != std::string::npos ||
         version_text.find("AMD GCN") != std::string::npos;
}

class AmdgpuKernelAssemblyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
    loom_amdgpu_low_descriptor_registry_initialize(&target_registry_);
  }

  void TearDown() override {
    ResetModule();
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void ResetModule() {
    if (module_ != nullptr) {
      loom_module_free(module_);
      module_ = nullptr;
    }
  }

  loom_module_t* ParseSource(const std::string& source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("amdgpu_kernel_assembly_test.loom"), &context_,
                        &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  const loom_op_t* FindFirstLowFunction(loom_module_t* module) {
    loom_block_t* block = loom_module_block(module);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (loom_low_func_def_isa(op)) {
        return op;
      }
    }
    return nullptr;
  }

  void VerifyAndPacketizeCurrentModule(
      iree_arena_allocator_t* arena,
      loom_low_packetization_t* out_packetization) {
    ASSERT_NE(module_, nullptr);
    loom_low_verify_options_t verify_options = {
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = &target_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .max_errors = 20,
    };
    loom_low_verify_result_t verify_result = {};
    IREE_ASSERT_OK(
        loom_low_verify_module(module_, &verify_options, &verify_result));
    EXPECT_EQ(verify_result.error_count, 0u);

    const loom_op_t* low_function = FindFirstLowFunction(module_);
    ASSERT_NE(low_function, nullptr);
    loom_low_packetization_options_t packetization_options = {
        .descriptor_registry = &target_registry_.registry,
    };
    IREE_ASSERT_OK(loom_low_packetize_function(module_, low_function,
                                               &packetization_options, arena,
                                               out_packetization));
  }

  void BuildSidecarsFromSource(const std::string& source,
                               iree_arena_allocator_t* arena,
                               loom_low_packetization_t* out_packetization) {
    ResetModule();
    module_ = ParseSource(source);
    VerifyAndPacketizeCurrentModule(arena, out_packetization);
  }

  void BuildSidecarsForPreset(const char* preset_key, const char* target_symbol,
                              const char* function_symbol, const char* body,
                              iree_arena_allocator_t* arena,
                              loom_low_packetization_t* out_packetization) {
    std::string source = "target.preset @";
    source += target_symbol;
    source += " {key = \"";
    source += preset_key;
    source += "\", source = @";
    source += function_symbol;
    source += "}\n";
    source += body;
    ResetModule();
    module_ = ParseSource(source);
    ASSERT_NE(module_, nullptr);

    const loom_target_preset_registry_t preset_registry =
        loom_target_low_descriptor_registry_presets(&target_registry_);
    iree_host_size_t expanded_preset_count = 0;
    IREE_ASSERT_OK(loom_target_expand_presets(module_, &preset_registry,
                                              &expanded_preset_count));
    EXPECT_EQ(expanded_preset_count, 1u);
    VerifyAndPacketizeCurrentModule(arena, out_packetization);
  }

  void EmitKernelForPreset(const char* preset_key, const char* target_cpu,
                           std::string* out_output) {
    ASSERT_NE(out_output, nullptr);
    iree_arena_allocator_t sidecar_arena;
    iree_arena_initialize(&block_pool_, &sidecar_arena);
    loom_low_packetization_t packetization = {};
    BuildSidecarsForPreset(
        preset_key, "gfx_target", "loom_kernel",
        "low.func.def target(@gfx_target) @loom_kernel() {\n"
        "  %c0 = low.op<amdgpu.s_mov_b32>() {imm32 = 7} : () -> "
        "reg<amdgpu.sgpr>\n"
        "  %c1 = low.op<amdgpu.s_mov_b32>() {imm32 = 5} : () -> "
        "reg<amdgpu.sgpr>\n"
        "  %sum = low.op<amdgpu.s_add_u32>(%c0, %c1) : "
        "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
        "  low.return\n"
        "}\n",
        &sidecar_arena, &packetization);

    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_ASSERT_OK(loom_amdgpu_emit_kernel_assembly(
        &packetization.schedule, &packetization.allocation, &builder));
    const std::string output(iree_string_builder_view(&builder).data,
                             iree_string_builder_view(&builder).size);
    EXPECT_NE(output.find(std::string(".amdgcn_target "
                                      "\"amdgcn-amd-amdhsa--") +
                          target_cpu + "\""),
              std::string::npos);
    iree_string_builder_deinitialize(&builder);
    iree_arena_deinitialize(&sidecar_arena);
    *out_output = output;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t target_registry_ = {};
};

TEST_F(AmdgpuKernelAssemblyTest, EmitsKernelEnvelopeForGfx11) {
  std::string output;
  EmitKernelForPreset("amdgpu-gfx11", "gfx1100", &output);
  EXPECT_NE(output.find(".text\n"), std::string::npos);
  EXPECT_NE(output.find(".amdhsa_code_object_version 5\n"), std::string::npos);
  EXPECT_NE(output.find(".protected loom_kernel\n"), std::string::npos);
  EXPECT_NE(output.find(".globl loom_kernel\n"), std::string::npos);
  EXPECT_NE(output.find(".type loom_kernel,@function\n"), std::string::npos);
  EXPECT_NE(output.find("loom_kernel:\n.Lbb0:\n"), std::string::npos);
  EXPECT_NE(output.find("  s_mov_b32 s"), std::string::npos);
  EXPECT_NE(output.find("  s_add_u32 s"), std::string::npos);
  EXPECT_NE(output.find("  s_endpgm\n"), std::string::npos);
  EXPECT_NE(output.find(".Lfunc_end0:\n"), std::string::npos);
  EXPECT_NE(output.find(".size loom_kernel, .Lfunc_end0-loom_kernel\n"),
            std::string::npos);
  EXPECT_NE(output.find(".rodata\n.p2align 6\n"), std::string::npos);
  EXPECT_NE(output.find(".amdhsa_kernel loom_kernel\n"), std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_group_segment_fixed_size 0\n"),
            std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_private_segment_fixed_size 0\n"),
            std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_kernarg_size 0\n"), std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_user_sgpr_count 0\n"), std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_system_sgpr_workgroup_id_x 0\n"),
            std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_next_free_vgpr 0\n"), std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_next_free_sgpr "), std::string::npos);
  EXPECT_EQ(output.find("  .amdhsa_next_free_sgpr 0\n"), std::string::npos);
  EXPECT_NE(output.find(".end_amdhsa_kernel\n"), std::string::npos);
}

TEST_F(AmdgpuKernelAssemblyTest, AssemblesKernelEnvelopeForGfx11WithLlvmMc) {
  std::string assembly;
  EmitKernelForPreset("amdgpu-gfx11", "gfx1100", &assembly);

  loom_llvm_toolchain_t toolchain;
  loom_llvm_toolchain_initialize_from_environment(&toolchain);

  loom_llvm_tool_output_t version_text = {};
  iree_status_t status =
      loom_llvm_tool_query_version(&toolchain, LOOM_LLVM_TOOL_LLVM_MC,
                                   iree_allocator_system(), &version_text);
  if (IsToolUnavailable(status)) {
    iree_status_ignore(status);
    GTEST_SKIP() << "llvm-mc is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  std::string version = ToString(version_text);
  loom_llvm_tool_output_deinitialize(&version_text, iree_allocator_system());
  if (!VersionTextListsAmdgcnTarget(version)) {
    GTEST_SKIP() << "llvm-mc does not report amdgcn target support";
  }

  const iree_string_view_t arguments[] = {
      IREE_SV("--triple=amdgcn-amd-amdhsa"),
      IREE_SV("--mcpu=gfx1100"),
  };
  loom_llvm_tool_output_t object = {};
  IREE_ASSERT_OK(loom_llvm_tool_assemble_native_object(
      &toolchain, iree_make_string_view(assembly.data(), assembly.size()),
      arguments, IREE_ARRAYSIZE(arguments), iree_allocator_system(), &object));
  EXPECT_GT(object.length, 0u);
  loom_llvm_tool_output_deinitialize(&object, iree_allocator_system());
}

TEST_F(AmdgpuKernelAssemblyTest, EmitsTargetIdsForCurrentAmdgpuPresets) {
  struct Case {
    const char* preset_key;
    const char* target_cpu;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", "gfx950"},
      {"amdgpu-gfx11", "gfx1100"},
      {"amdgpu-gfx12", "gfx1200"},
      {"amdgpu-gfx1250", "gfx1250"},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    std::string output;
    EmitKernelForPreset(test_case.preset_key, test_case.target_cpu, &output);
  }
}

TEST_F(AmdgpuKernelAssemblyTest, RejectsFunctionArgumentsBeforeAbiLowering) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildSidecarsForPreset("amdgpu-gfx11", "gfx_target", "loom_kernel",
                         "low.func.def target(@gfx_target) @loom_kernel(%arg : "
                         "reg<amdgpu.sgpr>) {\n"
                         "  low.return\n"
                         "}\n",
                         &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = loom_amdgpu_emit_kernel_assembly(
      &packetization.schedule, &packetization.allocation, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(AmdgpuKernelAssemblyTest, RejectsNonDefaultLinkage) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildSidecarsFromSource(
      "target.snapshot @gfx1100 {artifact_format = elf, codegen_format = "
      "low_native, data_layout = \"\", default_pointer_bitwidth = 64, "
      "index_bitwidth = 32, memory_space_constant = 4, "
      "memory_space_descriptor = 7, memory_space_generic = 0, "
      "memory_space_global = 1, memory_space_host = 4294967295, "
      "memory_space_private = 5, memory_space_workgroup = 3, "
      "offset_bitwidth = 64, target_cpu = \"gfx1100\", target_features = \"\", "
      "target_triple = \"amdgcn-amd-amdhsa\"}\n"
      "target.export @gfx_export {abi = hal_kernel, export_symbol = "
      "\"loom_kernel\", hal_binding_alignment = 16, "
      "hal_buffer_resource_flags = 159744, hal_flat_workgroup_size_max = 64, "
      "hal_flat_workgroup_size_min = 64, hal_workgroup_size_x = 64, "
      "hal_workgroup_size_y = 1, hal_workgroup_size_z = 1, linkage = "
      "dso_local, source = @loom_kernel}\n"
      "target.config @gfx_config {contract_feature_bits = 0, contract_set_key "
      "= \"amdgpu.gfx11.core\"}\n"
      "target.bundle @gfx_target {config = @gfx_config, export_plan = "
      "@gfx_export, snapshot = @gfx1100}\n"
      "low.func.def target(@gfx_target) @loom_kernel() {\n"
      "  low.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = loom_amdgpu_emit_kernel_assembly(
      &packetization.schedule, &packetization.allocation, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

}  // namespace
}  // namespace loom
