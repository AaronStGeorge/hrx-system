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
    if (module_) {
      loom_module_free(module_);
      module_ = nullptr;
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
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

  void BuildSidecars(const char* body, iree_arena_allocator_t* arena,
                     loom_low_packetization_t* out_packetization) {
    std::string source =
        "target.snapshot @x86_snapshot {codegen_format = low_native, "
        "target_triple = \"x86_64-unknown-linux-gnu\", data_layout = \"\", "
        "artifact_format = elf, target_cpu = \"x86-64-v4\", "
        "target_features = \"+avx512f,+avx512bw,+avx512dq,+avx512vl\", "
        "default_pointer_bitwidth = 64, index_bitwidth = 64, "
        "offset_bitwidth = 64, memory_space_generic = 0, "
        "memory_space_global = 0, memory_space_workgroup = 0, "
        "memory_space_constant = 0, memory_space_private = 0, "
        "memory_space_host = 0, memory_space_descriptor = 4294967295}\n"
        "target.export @x86_export {export_symbol = \"x86_fragment\", "
        "abi = object_function, linkage = default, "
        "hal_binding_alignment = 0, hal_workgroup_size_x = 0, "
        "hal_workgroup_size_y = 0, hal_workgroup_size_z = 0, "
        "hal_flat_workgroup_size_min = 0, hal_flat_workgroup_size_max = 0, "
        "hal_buffer_resource_flags = 0}\n"
        "target.config @x86_config {contract_set_key = "
        "\"x86.avx512.core\", contract_feature_bits = 0}\n"
        "target.bundle @x86_target {snapshot = @x86_snapshot, export_plan = "
        "@x86_export, config = @x86_config}\n";
    source += body;
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

    const loom_op_t* low_function = FindFirstLowFunction(module_);
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
      "(reg<x86.zmm>, reg<x86.zmm>, reg<x86.zmm>) -> reg<x86.zmm>\n"
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

}  // namespace
}  // namespace loom
