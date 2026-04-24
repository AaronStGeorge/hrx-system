// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/assembly.h"

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
#include "loom/target/arch/x86/low_registry.h"
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

bool VersionTextListsX86Target(const std::string& version_text) {
  return version_text.find("x86") != std::string::npos ||
         version_text.find("X86") != std::string::npos;
}

class X86AssemblyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
    loom_x86_low_descriptor_registry_initialize(&target_registry_);
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
                        IREE_SV("x86_assembly_test.loom"), &context_,
                        &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
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

  void BuildSidecars(const char* body, iree_arena_allocator_t* arena,
                     loom_low_packetization_t* out_packetization) {
    std::string source = "target.profile @x86_target preset(\"x86-avx512\")\n";
    source += body;
    ResetModule();
    module_ = ParseSource(source);
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

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t target_registry_ = {};
};

TEST_F(X86AssemblyTest, DisassemblesAvx512ObjectWithLlvmObjdump) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildSidecars(
      "low.func.def target(@x86_target) @x86_fragment(%base : "
      "reg<x86.gpr64>, %acc : reg<x86.zmm>, %lhs : reg<x86.zmm>, "
      "%rhs : reg<x86.zmm>) {\n"
      "  %loaded = low.op<x86.avx512.vmovdqu32.load.zmm>(%base) "
      "{disp32 = 0} : (reg<x86.gpr64>) -> reg<x86.zmm>\n"
      "  %sum = low.op<x86.avx512.vpaddd.zmm>(%loaded, %lhs) : "
      "(reg<x86.zmm>, reg<x86.zmm>) -> reg<x86.zmm>\n"
      "  %dot = low.op<x86.avx512.vpdpbusd.zmm>(%acc, %sum, %rhs) : "
      "(reg<x86.zmm>, reg<x86.zmm>, reg<x86.zmm>) -> %acc as "
      "reg<x86.zmm>\n"
      "  low.op<x86.avx512.vmovdqu32.store.zmm>(%dot, %base) "
      "{disp32 = 64} : (reg<x86.zmm>, reg<x86.gpr64>)\n"
      "  low.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_x86_emit_assembly_fragment(
      &packetization.schedule, &packetization.allocation, &builder));
  std::string assembly =
      ".intel_syntax noprefix\n"
      ".text\n"
      ".globl x86_fragment\n"
      "x86_fragment:\n";
  const iree_string_view_t fragment = iree_string_builder_view(&builder);
  assembly.append(fragment.data, fragment.size);

  loom_llvm_toolchain_t toolchain;
  loom_llvm_toolchain_initialize_from_environment(&toolchain);

  loom_llvm_tool_output_t llvm_mc_version_text = {};
  iree_status_t status = loom_llvm_tool_query_version(
      &toolchain, LOOM_LLVM_TOOL_LLVM_MC, iree_allocator_system(),
      &llvm_mc_version_text);
  if (IsToolUnavailable(status)) {
    IREE_EXPECT_NOT_OK(status);
    iree_string_builder_deinitialize(&builder);
    iree_arena_deinitialize(&sidecar_arena);
    GTEST_SKIP() << "llvm-mc is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  std::string llvm_mc_version = ToString(llvm_mc_version_text);
  loom_llvm_tool_output_deinitialize(&llvm_mc_version_text,
                                     iree_allocator_system());
  if (!VersionTextListsX86Target(llvm_mc_version)) {
    iree_string_builder_deinitialize(&builder);
    iree_arena_deinitialize(&sidecar_arena);
    GTEST_SKIP() << "llvm-mc does not report x86 target support";
  }

  const iree_string_view_t mc_arguments[] = {
      IREE_SV("--triple=x86_64-unknown-linux-gnu"),
      IREE_SV("--mattr=+avx512f,+avx512bw,+avx512vl,+avx512vnni"),
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
    iree_string_builder_deinitialize(&builder);
    iree_arena_deinitialize(&sidecar_arena);
    GTEST_SKIP() << "llvm-objdump is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  std::string objdump_version = ToString(objdump_version_text);
  loom_llvm_tool_output_deinitialize(&objdump_version_text,
                                     iree_allocator_system());
  if (!VersionTextListsX86Target(objdump_version)) {
    loom_llvm_tool_output_deinitialize(&object, iree_allocator_system());
    iree_string_builder_deinitialize(&builder);
    iree_arena_deinitialize(&sidecar_arena);
    GTEST_SKIP() << "llvm-objdump does not report x86 target support";
  }

  const iree_string_view_t objdump_arguments[] = {
      IREE_SV("--disassemble"),
      IREE_SV("--x86-asm-syntax=intel"),
  };
  loom_llvm_tool_output_t disassembly = {};
  IREE_ASSERT_OK(loom_llvm_tool_disassemble_object(
      &toolchain, iree_make_const_byte_span(object.data, object.length),
      objdump_arguments, IREE_ARRAYSIZE(objdump_arguments),
      iree_allocator_system(), &disassembly));
  const std::string disassembly_text = ToString(disassembly);
  EXPECT_NE(disassembly_text.find("vmovdqu32\tzmm3, zmmword ptr [rax]"),
            std::string::npos)
      << disassembly_text;
  EXPECT_NE(disassembly_text.find("vpaddd\tzmm0, zmm3, zmm0"),
            std::string::npos)
      << disassembly_text;
  EXPECT_NE(disassembly_text.find("vpdpbusd\tzmm1, zmm0, zmm2"),
            std::string::npos)
      << disassembly_text;
  EXPECT_NE(disassembly_text.find("vmovdqu32\tzmmword ptr [rax + 0x40], zmm1"),
            std::string::npos)
      << disassembly_text;
  EXPECT_NE(disassembly_text.find("\tret"), std::string::npos)
      << disassembly_text;

  loom_llvm_tool_output_deinitialize(&disassembly, iree_allocator_system());
  loom_llvm_tool_output_deinitialize(&object, iree_allocator_system());
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

}  // namespace
}  // namespace loom
