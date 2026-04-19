// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/assembly.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/lower.h"
#include "loom/target/ir_records.h"
#include "loom/target/presets.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

class AmdgpuAssemblyTest : public ::testing::Test {
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
                        IREE_SV("amdgpu_assembly_test.loom"), &context_,
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

  loom_func_like_t FindFirstSemanticFunction(loom_module_t* module) {
    loom_block_t* block = loom_module_block(module);
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      loom_func_like_t function = loom_func_like_cast(module, op);
      if (loom_func_like_isa(function) && !loom_low_func_def_isa(op)) {
        return function;
      }
    }
    return (loom_func_like_t){0};
  }

  loom_symbol_ref_t SymbolRef(loom_module_t* module,
                              iree_string_view_t symbol_name) {
    loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_intern_string(module, symbol_name, &symbol_name_id));
    uint16_t symbol_id = loom_module_find_symbol(module, symbol_name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return loom_symbol_ref_t{.module_id = 0, .symbol_id = symbol_id};
  }

  void BuildSidecars(const char* body, iree_arena_allocator_t* arena,
                     loom_low_packetization_t* out_packetization) {
    std::string source =
        "target.preset @gfx11_target {key = \"amdgpu-gfx11\", source = "
        "@gfx11_fragment}\n";
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

  void LowerSource(const char* preset_key, const char* body,
                   loom_low_lower_result_t* out_lower_result) {
    std::string source = "target.preset @gfx_target {key = \"";
    source += preset_key;
    source += "\", source = @gfx_source}\n";
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

    loom_target_ir_bundle_storage_t bundle_storage = {};
    IREE_ASSERT_OK(loom_target_ir_bundle_from_symbol_name(
        module_, IREE_SV("gfx_target"), &bundle_storage));
    loom_low_lower_policy_registry_t policy_registry = {};
    loom_amdgpu_low_lower_policy_registry_initialize(&policy_registry);
    const loom_low_lower_policy_t* policy = nullptr;
    IREE_ASSERT_OK(loom_low_lower_policy_registry_lookup_for_bundle(
        &policy_registry, &bundle_storage.bundle, &policy));
    const loom_low_lower_options_t lower_options = {
        .target_ref = SymbolRef(module_, IREE_SV("gfx_target")),
        .bundle = &bundle_storage.bundle,
        .descriptor_registry = &target_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .policy = policy,
        .max_errors = 20,
    };
    IREE_ASSERT_OK(loom_low_lower_function(module_,
                                           FindFirstSemanticFunction(module_),
                                           &lower_options, out_lower_result));
  }

  void LowerSourceAndBuildSidecars(
      const char* preset_key, const char* body, iree_arena_allocator_t* arena,
      loom_low_packetization_t* out_packetization) {
    loom_low_lower_result_t lower_result = {};
    LowerSource(preset_key, body, &lower_result);
    EXPECT_EQ(lower_result.error_count, 0u);
    ASSERT_NE(lower_result.low_func_op, nullptr);
    ASSERT_NE(lower_result.abi_adapter_op, nullptr);

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

  void ExpectSourceLoweringEmitsFragment(const char* preset_key) {
    iree_arena_allocator_t sidecar_arena;
    iree_arena_initialize(&block_pool_, &sidecar_arena);
    loom_low_packetization_t packetization = {};
    LowerSourceAndBuildSidecars(preset_key,
                                "func.def @gfx_source(%lhs: i32, %rhs: i32, "
                                "%vlhs: vector<1xi32>, %vrhs: vector<1xi32>) "
                                "{\n"
                                "  %seven = scalar.constant 7 : i32\n"
                                "  %biased = scalar.addi %lhs, %seven : i32\n"
                                "  %sum = scalar.addi %biased, %rhs : i32\n"
                                "  %vsum = vector.addi %vlhs, %vrhs : "
                                "vector<1xi32>\n"
                                "  %vproduct = vector.muli %vsum, %vrhs : "
                                "vector<1xi32>\n"
                                "  func.return\n"
                                "}\n",
                                &sidecar_arena, &packetization);

    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_ASSERT_OK(loom_amdgpu_emit_assembly_fragment(
        &packetization.schedule, &packetization.allocation, &builder));
    const std::string output(iree_string_builder_view(&builder).data,
                             iree_string_builder_view(&builder).size);
    EXPECT_NE(output.find(".Lbb0:"), std::string::npos);
    EXPECT_NE(output.find("s_mov_b32 s"), std::string::npos);
    EXPECT_NE(output.find("s_add_u32 s"), std::string::npos);
    EXPECT_NE(output.find("v_add_u32 v"), std::string::npos);
    EXPECT_NE(output.find("v_mul_lo_u32 v"), std::string::npos);
    EXPECT_NE(output.find("s_endpgm"), std::string::npos);
    iree_string_builder_deinitialize(&builder);
    iree_arena_deinitialize(&sidecar_arena);
  }

  void ExpectSourceLoweringRejects(const char* preset_key, const char* body) {
    loom_low_lower_result_t lower_result = {};
    LowerSource(preset_key, body, &lower_result);
    EXPECT_GT(lower_result.error_count, 0u);
    EXPECT_EQ(lower_result.low_func_op, nullptr);
    EXPECT_EQ(lower_result.abi_adapter_op, nullptr);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t target_registry_ = {};
};

TEST_F(AmdgpuAssemblyTest, EmitsGfx11Fragment) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildSidecars(
      "low.func.def target(@gfx11_target) @gfx11_fragment(%s0 : "
      "reg<amdgpu.sgpr>, %s1 : reg<amdgpu.sgpr>, %v0 : "
      "reg<amdgpu.vgpr>, %v1 : reg<amdgpu.vgpr>, %a : "
      "reg<amdgpu.vgpr x4>, %b : reg<amdgpu.vgpr x4>, %acc : "
      "reg<amdgpu.vgpr x8>, %resource : reg<amdgpu.sgpr x4>, "
      "%soffset : reg<amdgpu.sgpr>, %vaddr : reg<amdgpu.vgpr>) {\n"
      "  %s_sum = low.op<amdgpu.s_add_u32>(%s0, %s1) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %v_sum = low.op<amdgpu.v_add_u32>(%v0, %v1) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  %smem = low.op<amdgpu.s_buffer_load_dword>(%resource, "
      "%soffset) {offset = 0} : (reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %vmem = low.op<amdgpu.buffer_load_dword>(%resource, %vaddr, "
      "%soffset) {offset = 4} : (reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>\n"
      "  %s_mix = low.op<amdgpu.s_add_u32>(%s_sum, %smem) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %v_mix = low.op<amdgpu.v_add_u32>(%v_sum, %vmem) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  %matrix0 = low.op<amdgpu.v_wmma_f32_16x16x16_f16>(%a, %b, "
      "%acc) : (reg<amdgpu.vgpr x4>, reg<amdgpu.vgpr x4>, "
      "reg<amdgpu.vgpr x8>) -> %acc as reg<amdgpu.vgpr x8>\n"
      "  %matrix1 = low.op<amdgpu.v_wmma_f32_16x16x16_f16>(%a, %b, "
      "%matrix0) : (reg<amdgpu.vgpr x4>, reg<amdgpu.vgpr x4>, "
      "reg<amdgpu.vgpr x8>) -> %matrix0 as reg<amdgpu.vgpr x8>\n"
      "  low.op<amdgpu.buffer_store_dword>(%v_mix, %resource, %vaddr, "
      "%soffset) {offset = 8} : (reg<amdgpu.vgpr>, "
      "reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.op<amdgpu.s_waitcnt>() {vmcnt = 0, lgkmcnt = 0} : ()\n"
      "  low.op<amdgpu.s_waitcnt_depctr>() {depctr = 0} : ()\n"
      "  low.op<amdgpu.s_wait_idle>() : ()\n"
      "  low.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_amdgpu_emit_assembly_fragment(
      &packetization.schedule, &packetization.allocation, &builder));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  EXPECT_NE(output.find(".Lbb0:"), std::string::npos);
  EXPECT_NE(output.find("s_add_u32 s"), std::string::npos);
  EXPECT_NE(output.find("v_add_u32 v"), std::string::npos);
  EXPECT_NE(output.find("s_buffer_load_dword s"), std::string::npos);
  EXPECT_NE(output.find("buffer_load_dword v"), std::string::npos);
  EXPECT_NE(output.find("buffer_store_dword v"), std::string::npos);
  EXPECT_NE(output.find("offset:8"), std::string::npos);
  EXPECT_NE(output.find("v_wmma_f32_16x16x16_f16 v["), std::string::npos);
  EXPECT_NE(output.find("s_waitcnt vmcnt(0) lgkmcnt(0)"), std::string::npos);
  EXPECT_NE(output.find("s_waitcnt_depctr depctr(0)"), std::string::npos);
  EXPECT_NE(output.find("s_wait_idle"), std::string::npos);
  EXPECT_NE(output.find("s_endpgm"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(AmdgpuAssemblyTest, EmitsFragmentsFromSourceLowering) {
  static constexpr const char* kPresetKeys[] = {
      "amdgpu-gfx950",
      "amdgpu-gfx11",
      "amdgpu-gfx12",
      "amdgpu-gfx1250",
  };
  for (const char* preset_key : kPresetKeys) {
    SCOPED_TRACE(preset_key);
    ExpectSourceLoweringEmitsFragment(preset_key);
  }
}

TEST_F(AmdgpuAssemblyTest, RejectsUnsupportedVectorSourceLowering) {
  ExpectSourceLoweringRejects(
      "amdgpu-gfx11",
      "func.def @gfx_source(%lhs: vector<2xi32>, %rhs: vector<2xi32>) {\n"
      "  %sum = vector.addi %lhs, %rhs : vector<2xi32>\n"
      "  func.return\n"
      "}\n");
}

}  // namespace
}  // namespace loom
