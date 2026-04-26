// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/emit/native/amdgpu/kernel_assembly.h"
#include "loom/target/tool/llvm.h"

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

class AmdgpuKernelAssemblyLlvmToolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
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
                        IREE_SV("amdgpu_kernel_assembly_llvm_tool_test.loom"),
                        &context_, &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  loom_op_t* FindFirstLowFunction(loom_module_t* module) {
    loom_block_t* block = loom_module_block(module);
    loom_op_t* op = nullptr;
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

    loom_op_t* low_function = FindFirstLowFunction(module_);
    ASSERT_NE(low_function, nullptr);
    loom_low_packetization_options_t packetization_options = {
        .descriptor_registry = &target_registry_.registry,
    };
    IREE_ASSERT_OK(loom_low_packetize_function(module_, low_function,
                                               &packetization_options, arena,
                                               out_packetization));
  }

  void BuildSidecarsForPreset(const char* preset_key, const char* target_symbol,
                              const char* body, iree_arena_allocator_t* arena,
                              loom_low_packetization_t* out_packetization) {
    std::string source = "target.profile @";
    source += target_symbol;
    source += " preset(\"";
    source += preset_key;
    source += "\")\n";
    source += body;
    ResetModule();
    module_ = ParseSource(source);
    ASSERT_NE(module_, nullptr);
    VerifyAndPacketizeCurrentModule(arena, out_packetization);
  }

  void EmitKernelForPreset(const char* preset_key, const char* target_cpu,
                           std::string* out_output) {
    ASSERT_NE(out_output, nullptr);
    iree_arena_allocator_t sidecar_arena;
    iree_arena_initialize(&block_pool_, &sidecar_arena);
    loom_low_packetization_t packetization = {};
    BuildSidecarsForPreset(
        preset_key, "gfx_target",
        "low.func.def target(@gfx_target) @loom_kernel() {\n"
        "  %kernarg = low.live_in<amdgpu.kernarg_segment_ptr> : "
        "reg<amdgpu.sgpr x2>\n"
        "  %c0 = low.op<amdgpu.s_mov_b32>() {imm32 = 7} : () -> "
        "reg<amdgpu.sgpr>\n"
        "  %c1 = low.op<amdgpu.s_mov_b32>() {imm32 = 5} : () -> "
        "reg<amdgpu.sgpr>\n"
        "  %sum = low.op<amdgpu.s_add_u32>(%c0, %c1) : "
        "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
        "  %loaded = low.op<amdgpu.s_load_dwordx2>(%kernarg, %sum) "
        "{offset = 0} : (reg<amdgpu.sgpr x2>, reg<amdgpu.sgpr>) -> "
        "reg<amdgpu.sgpr x2>\n"
        "  low.return\n"
        "}\n",
        &sidecar_arena, &packetization);

    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_ASSERT_OK(loom_amdgpu_emit_kernel_assembly(&packetization.schedule,
                                                    &packetization.allocation,
                                                    &builder, &sidecar_arena));
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

// These are external LLVM tool smoke tests for assembler syntax compatibility.
// They do not define the Loom assembly emission contract.
TEST_F(AmdgpuKernelAssemblyLlvmToolTest,
       AssemblesKernelEnvelopeForGfx11WithLlvmMc) {
  std::string assembly;
  EmitKernelForPreset("amdgpu-gfx11", "gfx1100", &assembly);

  loom_llvm_toolchain_t toolchain;
  loom_llvm_toolchain_initialize_from_environment(&toolchain);

  loom_llvm_tool_output_t version_text = {};
  iree_status_t status =
      loom_llvm_tool_query_version(&toolchain, LOOM_LLVM_TOOL_LLVM_MC,
                                   iree_allocator_system(), &version_text);
  if (IsToolUnavailable(status)) {
    IREE_EXPECT_NOT_OK(status);
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

TEST_F(AmdgpuKernelAssemblyLlvmToolTest,
       DisassemblesGfx11ObjectWithLlvmObjdump) {
  std::string assembly;
  EmitKernelForPreset("amdgpu-gfx11", "gfx1100", &assembly);

  loom_llvm_toolchain_t toolchain;
  loom_llvm_toolchain_initialize_from_environment(&toolchain);

  loom_llvm_tool_output_t llvm_mc_version_text = {};
  iree_status_t status = loom_llvm_tool_query_version(
      &toolchain, LOOM_LLVM_TOOL_LLVM_MC, iree_allocator_system(),
      &llvm_mc_version_text);
  if (IsToolUnavailable(status)) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "llvm-mc is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  std::string llvm_mc_version = ToString(llvm_mc_version_text);
  loom_llvm_tool_output_deinitialize(&llvm_mc_version_text,
                                     iree_allocator_system());
  if (!VersionTextListsAmdgcnTarget(llvm_mc_version)) {
    GTEST_SKIP() << "llvm-mc does not report amdgcn target support";
  }

  const iree_string_view_t mc_arguments[] = {
      IREE_SV("--triple=amdgcn-amd-amdhsa"),
      IREE_SV("--mcpu=gfx1100"),
  };
  loom_llvm_tool_output_t object = {};
  IREE_ASSERT_OK(loom_llvm_tool_assemble_native_object(
      &toolchain, iree_make_string_view(assembly.data(), assembly.size()),
      mc_arguments, IREE_ARRAYSIZE(mc_arguments), iree_allocator_system(),
      &object));

  loom_llvm_tool_output_t objdump_version_text = {};
  status = loom_llvm_tool_query_version(&toolchain, LOOM_LLVM_TOOL_LLVM_OBJDUMP,
                                        iree_allocator_system(),
                                        &objdump_version_text);
  if (IsToolUnavailable(status)) {
    IREE_EXPECT_NOT_OK(status);
    loom_llvm_tool_output_deinitialize(&object, iree_allocator_system());
    GTEST_SKIP() << "llvm-objdump is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  std::string objdump_version = ToString(objdump_version_text);
  loom_llvm_tool_output_deinitialize(&objdump_version_text,
                                     iree_allocator_system());
  if (!VersionTextListsAmdgcnTarget(objdump_version)) {
    loom_llvm_tool_output_deinitialize(&object, iree_allocator_system());
    GTEST_SKIP() << "llvm-objdump does not report amdgcn target support";
  }

  const iree_string_view_t objdump_arguments[] = {
      IREE_SV("--disassemble"),
      IREE_SV("--mcpu=gfx1100"),
  };
  loom_llvm_tool_output_t disassembly = {};
  IREE_ASSERT_OK(loom_llvm_tool_disassemble_object(
      &toolchain, iree_make_const_byte_span(object.data, object.length),
      objdump_arguments, IREE_ARRAYSIZE(objdump_arguments),
      iree_allocator_system(), &disassembly));
  const std::string disassembly_text = ToString(disassembly);
  EXPECT_NE(disassembly_text.find("s_mov_b32 s2, 7"), std::string::npos)
      << disassembly_text;
  EXPECT_NE(disassembly_text.find("s_mov_b32 s3, 5"), std::string::npos)
      << disassembly_text;
  EXPECT_NE(disassembly_text.find("s_add_u32 s2, s2, s3"), std::string::npos)
      << disassembly_text;
  EXPECT_NE(disassembly_text.find("s_load_b64 s[0:1], s[0:1], s2"),
            std::string::npos)
      << disassembly_text;
  EXPECT_NE(disassembly_text.find("s_endpgm"), std::string::npos)
      << disassembly_text;

  loom_llvm_tool_output_deinitialize(&disassembly, iree_allocator_system());
  loom_llvm_tool_output_deinitialize(&object, iree_allocator_system());
}

}  // namespace
}  // namespace loom
