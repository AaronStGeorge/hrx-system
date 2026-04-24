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
#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/x86/low_registry.h"
#include "loom/target/arch/x86/lower.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

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

  void LowerSourceAndBuildSidecars(
      const char* preset_key, const char* body, iree_arena_allocator_t* arena,
      loom_low_packetization_t* out_packetization) {
    std::string source = "target.profile @x86_target preset(\"";
    source += preset_key;
    source += "\")\n";
    source += body;
    ResetModule();
    module_ = ParseSource(source);
    ASSERT_NE(module_, nullptr);

    loom_low_lower_policy_registry_t policy_registry = {};
    loom_x86_low_lower_policy_registry_initialize(&policy_registry);
    iree_arena_allocator_t selection_arena;
    iree_arena_initialize(&block_pool_, &selection_arena);
    const loom_low_source_selection_options_t selection_options = {
        .descriptor_registry = &target_registry_.registry,
        .policy_registry = &policy_registry,
        .lowering_kind = IREE_SV("x86 source-to-low"),
    };
    loom_low_source_selection_t selection = {};
    IREE_ASSERT_OK(loom_low_select_source_func(module_, &selection_options,
                                               &selection_arena, &selection));
    const loom_target_low_legality_provider_t* legality_providers[] = {
        loom_x86_low_legality_provider(),
    };
    const loom_target_low_legality_provider_list_t legality_provider_list =
        loom_target_low_legality_provider_list_make(
            legality_providers, IREE_ARRAYSIZE(legality_providers));
    const loom_low_lower_options_t lower_options = {
        .target_ref = selection.target_ref,
        .bundle = selection.target_bundle,
        .descriptor_registry = &target_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .legality_provider_list = legality_provider_list,
        .policy = selection.policy,
        .max_errors = 20,
    };
    loom_low_lower_result_t lower_result = {};
    IREE_ASSERT_OK(loom_low_lower_function(module_, selection.func,
                                           &lower_options, &lower_result));
    iree_arena_deinitialize(&selection_arena);
    EXPECT_EQ(lower_result.error_count, 0u);
    ASSERT_NE(lower_result.low_func_op, nullptr);

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

    loom_low_packetization_options_t packetization_options = {
        .descriptor_registry = &target_registry_.registry,
    };
    IREE_ASSERT_OK(loom_low_packetize_function(
        module_, lower_result.low_func_op, &packetization_options, arena,
        out_packetization));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t target_registry_ = {};
};

TEST_F(X86AssemblyTest, EmitsAvx512Fragment) {
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
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  EXPECT_NE(output.find(".Lbb0:"), std::string::npos);
  EXPECT_NE(output.find("vmovdqu32 zmm"), std::string::npos);
  EXPECT_NE(output.find("[rax]"), std::string::npos);
  EXPECT_NE(output.find("vpaddd zmm"), std::string::npos);
  EXPECT_NE(output.find("vpdpbusd zmm"), std::string::npos);
  EXPECT_NE(output.find("vmovdqu32 [rax + 64], zmm"), std::string::npos);
  EXPECT_NE(output.find("ret"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(X86AssemblyTest, EmitsMaterializedAvx512Copy) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildSidecars(
      "low.func.def target(@x86_target) @x86_fragment(%lhs : "
      "reg<x86.zmm>, %rhs : reg<x86.zmm>, %base : reg<x86.gpr64>) {\n"
      "  %copy = low.copy %lhs : reg<x86.zmm> -> reg<x86.zmm>\n"
      "  %sum = low.op<x86.avx512.vpaddd.zmm>(%copy, %rhs) : "
      "(reg<x86.zmm>, reg<x86.zmm>) -> reg<x86.zmm>\n"
      "  %second = low.op<x86.avx512.vpaddd.zmm>(%lhs, %sum) : "
      "(reg<x86.zmm>, reg<x86.zmm>) -> reg<x86.zmm>\n"
      "  low.op<x86.avx512.vmovdqu32.store.zmm>(%second, %base) "
      "{disp32 = 128} : (reg<x86.zmm>, reg<x86.gpr64>)\n"
      "  low.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_x86_emit_assembly_fragment(
      &packetization.schedule, &packetization.allocation, &builder));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  EXPECT_NE(output.find("vmovdqa32 zmm"), std::string::npos);
  EXPECT_NE(output.find("vpaddd zmm"), std::string::npos);
  EXPECT_NE(output.find("vmovdqu32 [rax + 128], zmm"), std::string::npos);
  EXPECT_NE(output.find("ret"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(X86AssemblyTest, EmitsAvx512AddressConstant) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildSidecars(
      "low.func.def target(@x86_target) @x86_fragment(%value : "
      "reg<x86.zmm>) {\n"
      "  %addr = low.const<x86.avx512.movimm.gpr64> {imm64 = 1024} : "
      "reg<x86.gpr64>\n"
      "  low.op<x86.avx512.vmovdqu32.store.zmm>(%value, %addr) "
      "{disp32 = -16} : (reg<x86.zmm>, reg<x86.gpr64>)\n"
      "  low.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_x86_emit_assembly_fragment(
      &packetization.schedule, &packetization.allocation, &builder));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  EXPECT_NE(output.find("mov rax, 1024"), std::string::npos);
  EXPECT_NE(output.find("vmovdqu32 [rax - 16], zmm"), std::string::npos);
  EXPECT_NE(output.find("ret"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(X86AssemblyTest, EmitsAvx512IndexedMemory) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildSidecars(
      "low.func.def target(@x86_target) @x86_fragment(%base : "
      "reg<x86.gpr64>, %index : reg<x86.gpr64>, %value : reg<x86.zmm>) {\n"
      "  %loaded = low.op<x86.avx512.vmovdqu32.load.indexed.zmm>(%base, "
      "%index) {disp32 = 32, scale = 4} : (reg<x86.gpr64>, "
      "reg<x86.gpr64>) -> reg<x86.zmm>\n"
      "  low.op<x86.avx512.vmovdqu32.store.indexed.zmm>(%value, %base, "
      "%index) {disp32 = -64, scale = 8} : (reg<x86.zmm>, "
      "reg<x86.gpr64>, reg<x86.gpr64>)\n"
      "  low.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_x86_emit_assembly_fragment(
      &packetization.schedule, &packetization.allocation, &builder));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  EXPECT_NE(output.find("vmovdqu32 zmm"), std::string::npos);
  EXPECT_NE(output.find("[rax + rcx * 4 + 32]"), std::string::npos);
  EXPECT_NE(output.find("vmovdqu32 [rax + rcx * 8 - 64], zmm"),
            std::string::npos);
  EXPECT_NE(output.find("ret"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(X86AssemblyTest, EmitsAvx512LeaAdd) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildSidecars(
      "low.func.def target(@x86_target) @x86_fragment(%lhs : "
      "reg<x86.gpr64>, %rhs : reg<x86.gpr64>, %value : reg<x86.zmm>) {\n"
      "  %sum = low.op<x86.avx512.lea.add.gpr64>(%lhs, %rhs) : "
      "(reg<x86.gpr64>, reg<x86.gpr64>) -> reg<x86.gpr64>\n"
      "  low.op<x86.avx512.vmovdqu32.store.zmm>(%value, %sum) {disp32 = 0} "
      ": (reg<x86.zmm>, reg<x86.gpr64>)\n"
      "  low.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_x86_emit_assembly_fragment(
      &packetization.schedule, &packetization.allocation, &builder));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  EXPECT_NE(output.find("lea "), std::string::npos);
  EXPECT_NE(output.find("["), std::string::npos);
  EXPECT_NE(output.find(" + "), std::string::npos);
  EXPECT_NE(output.find("ret"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(X86AssemblyTest, EmitsAvx512IndexMaddFromSourceLowering) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  LowerSourceAndBuildSidecars(
      "x86-avx512",
      "func.def target(@x86_target) @x86_source(%input: buffer, "
      "%output: buffer, %row: index, %stride: index, %col: index) {\n"
      "  %zero = index.constant 0 : offset\n"
      "  %linear = index.madd %row, %stride, %col : index\n"
      "  %input_view = buffer.view %input[%zero] : buffer -> "
      "view<32xf32, #dense>\n"
      "  %output_view = buffer.view %output[%zero] : buffer -> "
      "view<32xf32, #dense>\n"
      "  %loaded = vector.load %input_view[%linear] : view<32xf32, #dense> "
      "-> vector<16xf32>\n"
      "  vector.store %loaded, %output_view[%linear] : vector<16xf32>, "
      "view<32xf32, #dense>\n"
      "  func.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_x86_emit_assembly_fragment(
      &packetization.schedule, &packetization.allocation, &builder));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  EXPECT_NE(output.find("imul "), std::string::npos);
  EXPECT_NE(output.find("lea "), std::string::npos);
  EXPECT_NE(output.find("vmovdqu32 zmm"), std::string::npos);
  EXPECT_NE(output.find("vmovdqu32 ["), std::string::npos);
  EXPECT_NE(output.find("ret"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(X86AssemblyTest, DropsDeadAvx512FragmentFromSourceLowering) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  LowerSourceAndBuildSidecars(
      "x86-avx512",
      "func.def target(@x86_target) @x86_source(%lhs: vector<16xi32>, %rhs: "
      "vector<16xi32>, %flhs: vector<16xf32>, %frhs: vector<16xf32>) {\n"
      "  %sum = vector.addi %lhs, %rhs : vector<16xi32>\n"
      "  %diff = vector.subi %sum, %rhs : vector<16xi32>\n"
      "  %product = vector.muli %diff, %rhs : vector<16xi32>\n"
      "  %fsum = vector.addf %flhs, %frhs : vector<16xf32>\n"
      "  %fdiff = vector.subf %fsum, %flhs : vector<16xf32>\n"
      "  %fproduct = vector.mulf %fdiff, %frhs : vector<16xf32>\n"
      "  %ffma = vector.fmaf %flhs, %frhs, %fproduct : vector<16xf32>\n"
      "  func.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_x86_emit_assembly_fragment(
      &packetization.schedule, &packetization.allocation, &builder));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  EXPECT_NE(output.find(".Lbb0:"), std::string::npos);
  EXPECT_EQ(output.find("vpaddd zmm"), std::string::npos);
  EXPECT_EQ(output.find("vpsubd zmm"), std::string::npos);
  EXPECT_EQ(output.find("vpmulld zmm"), std::string::npos);
  EXPECT_EQ(output.find("vaddps zmm"), std::string::npos);
  EXPECT_EQ(output.find("vsubps zmm"), std::string::npos);
  EXPECT_EQ(output.find("vmulps zmm"), std::string::npos);
  EXPECT_EQ(output.find("vfmadd231ps zmm"), std::string::npos);
  EXPECT_NE(output.find("ret"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(X86AssemblyTest, DropsDeadPackedDotFragmentFromSourceLowering) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  LowerSourceAndBuildSidecars(
      "x86-packed-dot",
      "func.def target(@x86_target) @x86_source(%lhs: vector<32xi8>, %rhs: "
      "vector<32xi8>, "
      "%acc: vector<8xi32>) {\n"
      "  %dot = vector.dot4i<u8s8> %lhs, %rhs, %acc : vector<32xi8>, "
      "vector<32xi8>, vector<8xi32>\n"
      "  func.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_x86_emit_assembly_fragment(
      &packetization.schedule, &packetization.allocation, &builder));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  EXPECT_NE(output.find(".Lbb0:"), std::string::npos);
  EXPECT_EQ(output.find("vpdpbusd ymm"), std::string::npos);
  EXPECT_NE(output.find("ret"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

}  // namespace
}  // namespace loom
