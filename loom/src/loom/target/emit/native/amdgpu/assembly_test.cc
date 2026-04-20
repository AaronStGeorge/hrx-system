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
#include "loom/target/arch/amdgpu/wait_plan.h"
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

  bool StringIdEquals(loom_string_id_t string_id, iree_string_view_t expected) {
    if (string_id == LOOM_STRING_ID_INVALID ||
        string_id >= module_->strings.count) {
      return false;
    }
    return iree_string_view_equal(module_->strings.entries[string_id],
                                  expected);
  }

  loom_value_id_t FindValueIdByName(const char* name) {
    iree_string_view_t expected_name = iree_make_cstring_view(name);
    for (iree_host_size_t i = 0; i < module_->values.count; ++i) {
      const loom_value_t* value =
          loom_module_value(module_, (loom_value_id_t)i);
      if (value->name_id == LOOM_STRING_ID_INVALID ||
          value->name_id >= module_->strings.count) {
        continue;
      }
      iree_string_view_t value_name = module_->strings.entries[value->name_id];
      if (iree_string_view_equal(value_name, expected_name)) {
        return (loom_value_id_t)i;
      }
    }
    return LOOM_VALUE_ID_INVALID;
  }

  void BuildShiftedCopySidecars(iree_arena_allocator_t* arena,
                                loom_low_packetization_t* out_packetization) {
    const char* body =
        "low.func.def target(@gfx11_target) @gfx11_fragment(%source : "
        "reg<amdgpu.sgpr x3>) {\n"
        "  %shifted = low.copy %source : reg<amdgpu.sgpr x3> -> "
        "reg<amdgpu.sgpr x3>\n"
        "  low.return\n"
        "}\n";
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

    loom_value_id_t source_value = FindValueIdByName("source");
    loom_value_id_t shifted_value = FindValueIdByName("shifted");
    ASSERT_NE(source_value, LOOM_VALUE_ID_INVALID);
    ASSERT_NE(shifted_value, LOOM_VALUE_ID_INVALID);
    const loom_low_allocation_fixed_value_t fixed_values[] = {
        {
            .value_id = source_value,
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = 0,
            .location_count = 3,
        },
        {
            .value_id = shifted_value,
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = 1,
            .location_count = 3,
        },
    };
    const loom_op_t* low_function = FindFirstLowFunction(module_);
    ASSERT_NE(low_function, nullptr);
    loom_low_packetization_options_t packetization_options = {
        .descriptor_registry = &target_registry_.registry,
        .allocation_fixed_values = fixed_values,
        .allocation_fixed_value_count = IREE_ARRAYSIZE(fixed_values),
    };
    IREE_ASSERT_OK(loom_low_packetize_function(module_, low_function,
                                               &packetization_options, arena,
                                               out_packetization));
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

  void BuildSidecars(const char* body, iree_arena_allocator_t* arena,
                     loom_low_packetization_t* out_packetization) {
    BuildSidecarsForPreset("amdgpu-gfx11", "gfx11_target", "gfx11_fragment",
                           body, arena, out_packetization);
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
                                "%vlhs: vector<1xi32>, %vrhs: vector<1xi32>, "
                                "%flhs: vector<1xf32>, %frhs: vector<1xf32>) "
                                "{\n"
                                "  %seven = scalar.constant 7 : i32\n"
                                "  %biased = scalar.addi %lhs, %seven : i32\n"
                                "  %sum = scalar.addi %biased, %rhs : i32\n"
                                "  %difference = scalar.subi %sum, %seven : "
                                "i32\n"
                                "  %vconst = vector.constant 42 : "
                                "vector<1xi32>\n"
                                "  %vsum = vector.addi %vlhs, %vrhs : "
                                "vector<1xi32>\n"
                                "  %vbiased = vector.addi %vsum, %vconst : "
                                "vector<1xi32>\n"
                                "  %vdifference = vector.subi %vbiased, "
                                "%vconst : vector<1xi32>\n"
                                "  %vproduct = vector.muli %vdifference, %vrhs "
                                ": vector<1xi32>\n"
                                "  %fsum = vector.addf %flhs, %frhs : "
                                "vector<1xf32>\n"
                                "  %fproduct = vector.mulf %fsum, %frhs : "
                                "vector<1xf32>\n"
                                "  %ffma = vector.fmaf %flhs, %frhs, "
                                "%fproduct : vector<1xf32>\n"
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
    EXPECT_NE(output.find("s_sub_u32 s"), std::string::npos);
    EXPECT_NE(output.find("v_mov_b32 v"), std::string::npos);
    EXPECT_NE(output.find("v_add_u32 v"), std::string::npos);
    EXPECT_NE(output.find("v_sub"), std::string::npos);
    EXPECT_NE(output.find("v_mul_lo_u32 v"), std::string::npos);
    EXPECT_NE(output.find("v_add_f32 v"), std::string::npos);
    EXPECT_NE(output.find("v_mul_f32 v"), std::string::npos);
    EXPECT_NE(output.find("v_fma_f32 v"), std::string::npos);
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
      "reg<amdgpu.vgpr x8>, %kernarg : reg<amdgpu.sgpr x2>, "
      "%resource : reg<amdgpu.sgpr x4>, %soffset : reg<amdgpu.sgpr>, "
      "%vaddr : reg<amdgpu.vgpr>) {\n"
      "  %s_sum = low.op<amdgpu.s_add_u32>(%s0, %s1) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %v_sum = low.op<amdgpu.v_add_u32>(%v0, %v1) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  %smem = low.op<amdgpu.s_buffer_load_dword>(%resource, "
      "%soffset) {offset = 0} : (reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %binding_ptr = low.op<amdgpu.s_load_dwordx2>(%kernarg, "
      "%soffset) {offset = 16} : (reg<amdgpu.sgpr x2>, "
      "reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr x2>\n"
      "  %vmem = low.op<amdgpu.buffer_load_dword>(%resource, %vaddr, "
      "%soffset) {offset = 4} : (reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>\n"
      "  %vmem_b128 = low.op<amdgpu.buffer_load_b128>(%resource, %vaddr, "
      "%soffset) {offset = 16} : (reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr x4>\n"
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
      "  low.op<amdgpu.buffer_store_b128>(%vmem_b128, %resource, %vaddr, "
      "%soffset) {offset = 32} : (reg<amdgpu.vgpr x4>, "
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
  EXPECT_NE(output.find("s_load_dwordx2 s"), std::string::npos);
  EXPECT_NE(output.find("buffer_load_dword v"), std::string::npos);
  EXPECT_NE(output.find("buffer_store_dword v"), std::string::npos);
  EXPECT_NE(output.find("buffer_load_b128 v["), std::string::npos);
  EXPECT_NE(output.find("buffer_store_b128 v["), std::string::npos);
  EXPECT_NE(output.find("offen offset:8"), std::string::npos);
  EXPECT_NE(output.find("offen offset:32"), std::string::npos);
  EXPECT_NE(output.find("v_wmma_f32_16x16x16_f16 v["), std::string::npos);
  EXPECT_NE(output.find("s_waitcnt vmcnt(0) lgkmcnt(0)"), std::string::npos);
  EXPECT_NE(output.find("s_waitcnt_depctr depctr(0)"), std::string::npos);
  EXPECT_NE(output.find("s_wait_idle"), std::string::npos);
  EXPECT_NE(output.find("s_endpgm"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(AmdgpuAssemblyTest, LowersSemanticBufferStoreToVmemPacket) {
  loom_low_lower_result_t lower_result = {};
  LowerSource("amdgpu-gfx11",
              "func.def @gfx_source(%output: buffer) {\n"
              "  %zero = index.constant 0 : offset\n"
              "  %view = buffer.view %output[%zero] : buffer -> "
              "view<1xi32, #dense>\n"
              "  %value = vector.constant 42 : vector<1xi32>\n"
              "  vector.store %value, %view[0] : vector<1xi32>, "
              "view<1xi32, #dense>\n"
              "  func.return\n"
              "}\n",
              &lower_result);
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

  bool saw_resource = false;
  bool saw_vaddr_zero = false;
  bool saw_soffset_zero = false;
  bool saw_store = false;
  loom_region_t* low_body = loom_low_func_def_body(lower_result.low_func_op);
  loom_block_t* entry_block = loom_region_entry_block(low_body);
  loom_op_t* op = nullptr;
  loom_block_for_each_op(entry_block, op) {
    if (loom_low_resource_isa(op)) {
      saw_resource = true;
    } else if (loom_low_const_isa(op)) {
      saw_vaddr_zero |= StringIdEquals(loom_low_const_opcode(op),
                                       IREE_SV("amdgpu.v_mov_b32"));
      saw_soffset_zero |= StringIdEquals(loom_low_const_opcode(op),
                                         IREE_SV("amdgpu.s_mov_b32"));
    } else if (loom_low_op_isa(op)) {
      if (StringIdEquals(loom_low_op_opcode(op),
                         IREE_SV("amdgpu.buffer_store_dword"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 4u);
        EXPECT_EQ(loom_low_op_results(op).count, 0u);
        saw_store = true;
      }
    }
  }
  EXPECT_TRUE(saw_resource);
  EXPECT_TRUE(saw_vaddr_zero);
  EXPECT_TRUE(saw_soffset_zero);
  EXPECT_TRUE(saw_store);
}

TEST_F(AmdgpuAssemblyTest, LowersSemanticBufferLoadAddStoreToVmemPackets) {
  loom_low_lower_result_t lower_result = {};
  LowerSource("amdgpu-gfx11",
              "func.def @gfx_source(%input: buffer, %output: buffer) {\n"
              "  %zero = index.constant 0 : offset\n"
              "  %input_view = buffer.view %input[%zero] : buffer -> "
              "view<1xi32, #dense>\n"
              "  %output_view = buffer.view %output[%zero] : buffer -> "
              "view<1xi32, #dense>\n"
              "  %loaded = vector.load %input_view[0] : "
              "view<1xi32, #dense> -> vector<1xi32>\n"
              "  %value = vector.constant 7 : vector<1xi32>\n"
              "  %sum = vector.addi %loaded, %value : vector<1xi32>\n"
              "  vector.store %sum, %output_view[0] : vector<1xi32>, "
              "view<1xi32, #dense>\n"
              "  func.return\n"
              "}\n",
              &lower_result);
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

  bool saw_load = false;
  bool saw_add = false;
  bool saw_store = false;
  uint32_t resource_count = 0;
  loom_region_t* low_body = loom_low_func_def_body(lower_result.low_func_op);
  loom_block_t* entry_block = loom_region_entry_block(low_body);
  loom_op_t* op = nullptr;
  loom_block_for_each_op(entry_block, op) {
    if (loom_low_resource_isa(op)) {
      ++resource_count;
    } else if (loom_low_op_isa(op)) {
      if (StringIdEquals(loom_low_op_opcode(op),
                         IREE_SV("amdgpu.buffer_load_dword"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 3u);
        EXPECT_EQ(loom_low_op_results(op).count, 1u);
        saw_load = true;
      } else if (StringIdEquals(loom_low_op_opcode(op),
                                IREE_SV("amdgpu.v_add_u32"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 2u);
        EXPECT_EQ(loom_low_op_results(op).count, 1u);
        saw_add = true;
      } else if (StringIdEquals(loom_low_op_opcode(op),
                                IREE_SV("amdgpu.buffer_store_dword"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 4u);
        EXPECT_EQ(loom_low_op_results(op).count, 0u);
        saw_store = true;
      }
    }
  }
  EXPECT_EQ(resource_count, 2u);
  EXPECT_TRUE(saw_load);
  EXPECT_TRUE(saw_add);
  EXPECT_TRUE(saw_store);
}

TEST_F(AmdgpuAssemblyTest,
       LowersSemanticWorkitemIndexedLoadAddStoreToVmemPackets) {
  loom_low_lower_result_t lower_result = {};
  LowerSource("amdgpu-gfx11",
              "func.def @gfx_source(%input: buffer, %output: buffer) {\n"
              "  %tid = kernel.workitem.id<x> : index\n"
              "  %zero = index.constant 0 : offset\n"
              "  %input_view = buffer.view %input[%zero] : buffer -> "
              "view<64xi32, #dense>\n"
              "  %output_view = buffer.view %output[%zero] : buffer -> "
              "view<64xi32, #dense>\n"
              "  %loaded = vector.load %input_view[%tid] : "
              "view<64xi32, #dense> -> vector<1xi32>\n"
              "  %value = vector.constant 7 : vector<1xi32>\n"
              "  %sum = vector.addi %loaded, %value : vector<1xi32>\n"
              "  vector.store %sum, %output_view[%tid] : vector<1xi32>, "
              "view<64xi32, #dense>\n"
              "  func.return\n"
              "}\n",
              &lower_result);
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

  bool saw_workitem_live_in = false;
  bool saw_byte_offset = false;
  bool saw_load = false;
  bool saw_add = false;
  bool saw_store = false;
  uint32_t resource_count = 0;
  loom_region_t* low_body = loom_low_func_def_body(lower_result.low_func_op);
  loom_block_t* entry_block = loom_region_entry_block(low_body);
  loom_op_t* op = nullptr;
  loom_block_for_each_op(entry_block, op) {
    if (loom_low_live_in_isa(op)) {
      saw_workitem_live_in |= StringIdEquals(loom_low_live_in_source(op),
                                             IREE_SV("amdgpu.workitem_id.x"));
    } else if (loom_low_resource_isa(op)) {
      ++resource_count;
    } else if (loom_low_op_isa(op)) {
      if (StringIdEquals(loom_low_op_opcode(op),
                         IREE_SV("amdgpu.v_mul_lo_u32"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 2u);
        EXPECT_EQ(loom_low_op_results(op).count, 1u);
        saw_byte_offset = true;
      } else if (StringIdEquals(loom_low_op_opcode(op),
                                IREE_SV("amdgpu.buffer_load_dword"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 3u);
        EXPECT_EQ(loom_low_op_results(op).count, 1u);
        saw_load = true;
      } else if (StringIdEquals(loom_low_op_opcode(op),
                                IREE_SV("amdgpu.v_add_u32"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 2u);
        EXPECT_EQ(loom_low_op_results(op).count, 1u);
        saw_add = true;
      } else if (StringIdEquals(loom_low_op_opcode(op),
                                IREE_SV("amdgpu.buffer_store_dword"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 4u);
        EXPECT_EQ(loom_low_op_results(op).count, 0u);
        saw_store = true;
      }
    }
  }
  EXPECT_EQ(resource_count, 2u);
  EXPECT_TRUE(saw_workitem_live_in);
  EXPECT_TRUE(saw_byte_offset);
  EXPECT_TRUE(saw_load);
  EXPECT_TRUE(saw_add);
  EXPECT_TRUE(saw_store);
}

TEST_F(AmdgpuAssemblyTest, LowersSemanticWorkitemIndexedWideCopyToB128Packets) {
  loom_low_lower_result_t lower_result = {};
  LowerSource("amdgpu-gfx11",
              "func.def @gfx_source(%input: buffer, %output: buffer) {\n"
              "  %tid = kernel.workitem.id<x> : index\n"
              "  %zero = index.constant 0 : offset\n"
              "  %input_view = buffer.view %input[%zero] : buffer -> "
              "view<64x4xi32, #dense>\n"
              "  %output_view = buffer.view %output[%zero] : buffer -> "
              "view<64x4xi32, #dense>\n"
              "  %loaded = vector.load %input_view[%tid, 0] : "
              "view<64x4xi32, #dense> -> vector<4xi32>\n"
              "  vector.store %loaded, %output_view[%tid, 0] : "
              "vector<4xi32>, view<64x4xi32, #dense>\n"
              "  func.return\n"
              "}\n",
              &lower_result);
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

  bool saw_workitem_live_in = false;
  bool saw_byte_offset = false;
  bool saw_wide_load = false;
  bool saw_wide_store = false;
  bool saw_dword_load = false;
  bool saw_dword_store = false;
  uint32_t resource_count = 0;
  loom_region_t* low_body = loom_low_func_def_body(lower_result.low_func_op);
  loom_block_t* entry_block = loom_region_entry_block(low_body);
  loom_op_t* op = nullptr;
  loom_block_for_each_op(entry_block, op) {
    if (loom_low_live_in_isa(op)) {
      saw_workitem_live_in |= StringIdEquals(loom_low_live_in_source(op),
                                             IREE_SV("amdgpu.workitem_id.x"));
    } else if (loom_low_resource_isa(op)) {
      ++resource_count;
    } else if (loom_low_op_isa(op)) {
      if (StringIdEquals(loom_low_op_opcode(op),
                         IREE_SV("amdgpu.v_mul_lo_u32"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 2u);
        EXPECT_EQ(loom_low_op_results(op).count, 1u);
        saw_byte_offset = true;
      } else if (StringIdEquals(loom_low_op_opcode(op),
                                IREE_SV("amdgpu.buffer_load_b128"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 3u);
        EXPECT_EQ(loom_low_op_results(op).count, 1u);
        saw_wide_load = true;
      } else if (StringIdEquals(loom_low_op_opcode(op),
                                IREE_SV("amdgpu.buffer_store_b128"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 4u);
        EXPECT_EQ(loom_low_op_results(op).count, 0u);
        saw_wide_store = true;
      } else if (StringIdEquals(loom_low_op_opcode(op),
                                IREE_SV("amdgpu.buffer_load_dword"))) {
        saw_dword_load = true;
      } else if (StringIdEquals(loom_low_op_opcode(op),
                                IREE_SV("amdgpu.buffer_store_dword"))) {
        saw_dword_store = true;
      }
    }
  }
  EXPECT_EQ(resource_count, 2u);
  EXPECT_TRUE(saw_workitem_live_in);
  EXPECT_TRUE(saw_byte_offset);
  EXPECT_TRUE(saw_wide_load);
  EXPECT_TRUE(saw_wide_store);
  EXPECT_FALSE(saw_dword_load);
  EXPECT_FALSE(saw_dword_store);
}

TEST_F(AmdgpuAssemblyTest, LowersSemanticWorkitemIndexedWideAddToLanePackets) {
  loom_low_lower_result_t lower_result = {};
  LowerSource("amdgpu-gfx11",
              "func.def @gfx_source(%lhs: buffer, %rhs: buffer, %output: "
              "buffer) {\n"
              "  %tid = kernel.workitem.id<x> : index\n"
              "  %zero = index.constant 0 : offset\n"
              "  %lhs_view = buffer.view %lhs[%zero] : buffer -> "
              "view<64x4xi32, #dense>\n"
              "  %rhs_view = buffer.view %rhs[%zero] : buffer -> "
              "view<64x4xi32, #dense>\n"
              "  %output_view = buffer.view %output[%zero] : buffer -> "
              "view<64x4xi32, #dense>\n"
              "  %lhs_loaded = vector.load %lhs_view[%tid, 0] : "
              "view<64x4xi32, #dense> -> vector<4xi32>\n"
              "  %rhs_loaded = vector.load %rhs_view[%tid, 0] : "
              "view<64x4xi32, #dense> -> vector<4xi32>\n"
              "  %sum = vector.addi %lhs_loaded, %rhs_loaded : vector<4xi32>\n"
              "  vector.store %sum, %output_view[%tid, 0] : vector<4xi32>, "
              "view<64x4xi32, #dense>\n"
              "  func.return\n"
              "}\n",
              &lower_result);
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

  uint32_t resource_count = 0;
  uint32_t wide_load_count = 0;
  uint32_t add_count = 0;
  uint32_t slice_count = 0;
  bool saw_concat = false;
  bool saw_wide_store = false;
  bool saw_dword_load = false;
  bool saw_dword_store = false;
  loom_region_t* low_body = loom_low_func_def_body(lower_result.low_func_op);
  loom_block_t* entry_block = loom_region_entry_block(low_body);
  loom_op_t* op = nullptr;
  loom_block_for_each_op(entry_block, op) {
    if (loom_low_resource_isa(op)) {
      ++resource_count;
    } else if (loom_low_slice_isa(op)) {
      EXPECT_GE(loom_low_slice_offset(op), 0);
      ++slice_count;
    } else if (loom_low_concat_isa(op)) {
      EXPECT_EQ(loom_low_concat_sources(op).count, 4u);
      saw_concat = true;
    } else if (loom_low_op_isa(op)) {
      if (StringIdEquals(loom_low_op_opcode(op),
                         IREE_SV("amdgpu.buffer_load_b128"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 3u);
        EXPECT_EQ(loom_low_op_results(op).count, 1u);
        ++wide_load_count;
      } else if (StringIdEquals(loom_low_op_opcode(op),
                                IREE_SV("amdgpu.v_add_u32"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 2u);
        EXPECT_EQ(loom_low_op_results(op).count, 1u);
        ++add_count;
      } else if (StringIdEquals(loom_low_op_opcode(op),
                                IREE_SV("amdgpu.buffer_store_b128"))) {
        EXPECT_EQ(loom_low_op_operands(op).count, 4u);
        EXPECT_EQ(loom_low_op_results(op).count, 0u);
        saw_wide_store = true;
      } else if (StringIdEquals(loom_low_op_opcode(op),
                                IREE_SV("amdgpu.buffer_load_dword"))) {
        saw_dword_load = true;
      } else if (StringIdEquals(loom_low_op_opcode(op),
                                IREE_SV("amdgpu.buffer_store_dword"))) {
        saw_dword_store = true;
      }
    }
  }
  EXPECT_EQ(resource_count, 3u);
  EXPECT_EQ(wide_load_count, 2u);
  EXPECT_EQ(slice_count, 8u);
  EXPECT_EQ(add_count, 4u);
  EXPECT_TRUE(saw_concat);
  EXPECT_TRUE(saw_wide_store);
  EXPECT_FALSE(saw_dword_load);
  EXPECT_FALSE(saw_dword_store);
}

TEST_F(AmdgpuAssemblyTest, EmitsPlannedWaitPackets) {
  struct Case {
    const char* preset_key;
    const char* load_wait;
    const char* store_wait;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", "s_waitcnt vmcnt(0) lgkmcnt(0)",
       "s_waitcnt vmcnt(0) lgkmcnt(0)"},
      {"amdgpu-gfx11", "s_waitcnt vmcnt(0) lgkmcnt(0)",
       "s_waitcnt vmcnt(0) lgkmcnt(0)"},
      {"amdgpu-gfx12", "s_wait_loadcnt loadcnt(0)",
       "s_wait_storecnt storecnt(0)"},
      {"amdgpu-gfx1250", "s_wait_loadcnt loadcnt(0)",
       "s_wait_storecnt storecnt(0)"},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    iree_arena_allocator_t sidecar_arena;
    iree_arena_initialize(&block_pool_, &sidecar_arena);
    loom_low_packetization_t packetization = {};
    BuildSidecarsForPreset(
        test_case.preset_key, "gfx_target", "wait_fragment",
        "low.func.def target(@gfx_target) @wait_fragment(%value : "
        "reg<amdgpu.vgpr>, %resource : reg<amdgpu.sgpr x4>, "
        "%soffset : reg<amdgpu.sgpr>, %vaddr : reg<amdgpu.vgpr>) {\n"
        "  %loaded = low.op<amdgpu.buffer_load_dword>(%resource, %vaddr, "
        "%soffset) {offset = 0} : (reg<amdgpu.sgpr x4>, "
        "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr>\n"
        "  low.op<amdgpu.buffer_store_dword>(%loaded, %resource, %vaddr, "
        "%soffset) {offset = 4} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
        "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
        "  low.return\n"
        "}\n",
        &sidecar_arena, &packetization);

    iree_string_builder_t raw_builder;
    iree_string_builder_initialize(iree_allocator_system(), &raw_builder);
    IREE_ASSERT_OK(loom_amdgpu_emit_assembly_fragment(
        &packetization.schedule, &packetization.allocation, &raw_builder));
    const std::string raw_output(iree_string_builder_view(&raw_builder).data,
                                 iree_string_builder_view(&raw_builder).size);
    EXPECT_EQ(raw_output.find("s_wait"), std::string::npos);
    iree_string_builder_deinitialize(&raw_builder);

    loom_amdgpu_wait_plan_t wait_plan = {};
    IREE_ASSERT_OK(loom_amdgpu_wait_plan_build(&packetization.schedule,
                                               &sidecar_arena, &wait_plan));
    loom_amdgpu_wait_packet_plan_t wait_packets = {};
    IREE_ASSERT_OK(loom_amdgpu_wait_packet_plan_build(
        &wait_plan, &sidecar_arena, &wait_packets));
    ASSERT_EQ(wait_packets.packet_count, 2u);

    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_ASSERT_OK(loom_amdgpu_emit_assembly_fragment_with_wait_packets(
        &packetization.schedule, &packetization.allocation, &wait_packets,
        &builder));
    const std::string output(iree_string_builder_view(&builder).data,
                             iree_string_builder_view(&builder).size);
    const size_t load_wait = output.find(test_case.load_wait);
    const size_t store = output.find("buffer_store_dword");
    const size_t store_wait = output.find(test_case.store_wait, store);
    const size_t endpgm = output.find("s_endpgm");
    EXPECT_NE(load_wait, std::string::npos);
    EXPECT_NE(store, std::string::npos);
    EXPECT_NE(store_wait, std::string::npos);
    EXPECT_NE(endpgm, std::string::npos);
    EXPECT_LT(load_wait, store);
    EXPECT_LT(store, store_wait);
    EXPECT_LT(store_wait, endpgm);
    iree_string_builder_deinitialize(&builder);
    iree_arena_deinitialize(&sidecar_arena);
  }
}

TEST_F(AmdgpuAssemblyTest, EmitsMaterializedCopies) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildSidecars(
      "low.func.def target(@gfx11_target) @gfx11_fragment(%s0 : "
      "reg<amdgpu.sgpr>, %s1 : reg<amdgpu.sgpr>, %a : "
      "reg<amdgpu.vgpr x4>, %b : reg<amdgpu.vgpr x4>, %acc : "
      "reg<amdgpu.vgpr x8>) {\n"
      "  %s_copy = low.copy %s0 : reg<amdgpu.sgpr> -> reg<amdgpu.sgpr>\n"
      "  %s_sum = low.op<amdgpu.s_add_u32>(%s_copy, %s1) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %s_second = low.op<amdgpu.s_add_u32>(%s0, %s_sum) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %v_copy = low.copy %a : reg<amdgpu.vgpr x4> -> "
      "reg<amdgpu.vgpr x4>\n"
      "  %matrix0 = low.op<amdgpu.v_wmma_f32_16x16x16_f16>(%v_copy, "
      "%b, %acc) : (reg<amdgpu.vgpr x4>, reg<amdgpu.vgpr x4>, "
      "reg<amdgpu.vgpr x8>) -> %acc as reg<amdgpu.vgpr x8>\n"
      "  %matrix1 = low.op<amdgpu.v_wmma_f32_16x16x16_f16>(%a, %b, "
      "%matrix0) : (reg<amdgpu.vgpr x4>, reg<amdgpu.vgpr x4>, "
      "reg<amdgpu.vgpr x8>) -> %matrix0 as reg<amdgpu.vgpr x8>\n"
      "  low.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_amdgpu_emit_assembly_fragment(
      &packetization.schedule, &packetization.allocation, &builder));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  EXPECT_NE(output.find("s_mov_b32 s"), std::string::npos);
  const size_t first_v_mov = output.find("v_mov_b32 v");
  EXPECT_NE(first_v_mov, std::string::npos);
  EXPECT_NE(output.find("v_mov_b32 v", first_v_mov + 1), std::string::npos);
  EXPECT_NE(output.find("v_wmma_f32_16x16x16_f16 v["), std::string::npos);
  EXPECT_NE(output.find("s_endpgm"), std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(AmdgpuAssemblyTest, SequencesOverlappingCopyBeforeClobber) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildShiftedCopySidecars(&sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_amdgpu_emit_assembly_fragment(
      &packetization.schedule, &packetization.allocation, &builder));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  const size_t move_s3 = output.find("s_mov_b32 s3, s2");
  const size_t move_s2 = output.find("s_mov_b32 s2, s1");
  const size_t move_s1 = output.find("s_mov_b32 s1, s0");
  EXPECT_NE(move_s3, std::string::npos);
  EXPECT_NE(move_s2, std::string::npos);
  EXPECT_NE(move_s1, std::string::npos);
  EXPECT_LT(move_s3, move_s2);
  EXPECT_LT(move_s2, move_s1);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(AmdgpuAssemblyTest, EmitsConcatRegisterCopies) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildSidecars(
      "low.func.def target(@gfx11_target) @gfx11_fragment(%r0 : "
      "reg<amdgpu.sgpr>, %r1 : reg<amdgpu.sgpr>, %r2 : reg<amdgpu.sgpr>, "
      "%r3 : reg<amdgpu.sgpr>, %value : reg<amdgpu.vgpr>, %vaddr : "
      "reg<amdgpu.vgpr>) {\n"
      "  %resource = low.concat(%r0, %r1, %r2, %r3) : (reg<amdgpu.sgpr>, "
      "reg<amdgpu.sgpr>, reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> "
      "reg<amdgpu.sgpr x4>\n"
      "  %sum0 = low.op<amdgpu.s_add_u32>(%r0, %r1) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %sum1 = low.op<amdgpu.s_add_u32>(%r2, %r3) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %soffset = low.op<amdgpu.s_add_u32>(%sum0, %sum1) : "
      "(reg<amdgpu.sgpr>, reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  low.op<amdgpu.buffer_store_dword>(%value, %resource, %vaddr, "
      "%soffset) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_amdgpu_emit_assembly_fragment(
      &packetization.schedule, &packetization.allocation, &builder));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  const size_t first_move = output.find("s_mov_b32 s");
  EXPECT_NE(first_move, std::string::npos);
  EXPECT_NE(output.find("s_mov_b32 s", first_move + 1), std::string::npos);
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
