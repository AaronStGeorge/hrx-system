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
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/hal_kernel_abi.h"
#include "loom/target/arch/amdgpu/hal_resource_materialization.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/wait_packets.h"
#include "loom/target/arch/amdgpu/wait_plan.h"
#include "loom/target/low_descriptor_registry.h"

namespace loom {
namespace {

class AmdgpuKernelAssemblyTest : public ::testing::Test {
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
                        IREE_SV("amdgpu_kernel_assembly_test.loom"), &context_,
                        &block_pool_, &parse_options, &module));
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
      if (loom_low_function_def_isa(op)) {
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

  void BuildSidecarsFromSource(const std::string& source,
                               iree_arena_allocator_t* arena,
                               loom_low_packetization_t* out_packetization) {
    ResetModule();
    module_ = ParseSource(source);
    VerifyAndPacketizeCurrentModule(arena, out_packetization);
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

  void BuildMaterializedHalResourceSidecarsForPreset(
      const char* preset_key, const char* target_symbol, const char* body,
      iree_host_size_t expected_materialized_resource_count,
      iree_host_size_t expected_fixed_value_count,
      loom_amdgpu_hal_kernel_abi_layout_t* out_abi_layout,
      iree_arena_allocator_t* arena,
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

    loom_op_t* low_function = FindFirstLowFunction(module_);
    ASSERT_NE(low_function, nullptr);
    loom_target_bundle_storage_t bundle_storage = {};
    ResolveFunctionTarget(low_function, &bundle_storage);
    const loom_low_descriptor_set_t* descriptor_set = nullptr;
    IREE_ASSERT_OK(loom_target_low_descriptor_set_select_for_bundle(
        &target_registry_.registry, &bundle_storage.bundle,
        LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        &descriptor_set));

    loom_amdgpu_hal_resource_materialization_result_t materialization = {};
    IREE_ASSERT_OK(loom_amdgpu_hal_resource_materialize(
        module_, low_function, &bundle_storage.bundle, descriptor_set,
        &materialization, arena));
    ASSERT_EQ(materialization.materialized_resource_count,
              expected_materialized_resource_count);
    ASSERT_TRUE(materialization.inserted_kernarg_segment_ptr_live_in);
    if (out_abi_layout != nullptr) {
      *out_abi_layout = materialization.abi_layout;
    }

    const loom_low_allocation_fixed_value_t* fixed_values = nullptr;
    iree_host_size_t fixed_value_count = 0;
    IREE_ASSERT_OK(loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
        module_, low_function, descriptor_set, &fixed_values,
        &fixed_value_count, arena));
    ASSERT_EQ(fixed_value_count, expected_fixed_value_count);

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
    ASSERT_EQ(verify_result.error_count, 0u);

    loom_low_packetization_options_t packetization_options = {
        .descriptor_registry = &target_registry_.registry,
        .allocation_fixed_values = fixed_values,
        .allocation_fixed_value_count = fixed_value_count,
    };
    IREE_ASSERT_OK(loom_low_packetize_function(module_, low_function,
                                               &packetization_options, arena,
                                               out_packetization));
  }

  void ResolveFunctionTarget(const loom_op_t* function_op,
                             loom_target_bundle_storage_t* out_storage) {
    loom_low_resolved_target_t target = {};
    IREE_ASSERT_OK(loom_low_resolve_function_target(
        module_, function_op, &target_registry_.registry,
        iree_diagnostic_emitter_t{}, &target));
    *out_storage = target.bundle_storage;
    loom_target_bundle_storage_rebind(out_storage);
  }

  void EmitKernelForPreset(const char* preset_key, const char* target_cpu,
                           std::string* out_output) {
    ASSERT_NE(out_output, nullptr);
    iree_arena_allocator_t sidecar_arena;
    iree_arena_initialize(&block_pool_, &sidecar_arena);
    loom_low_packetization_t packetization = {};
    BuildSidecarsForPreset(
        preset_key, "gfx_target",
        "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
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
  EXPECT_NE(output.find("  .amdhsa_user_sgpr_kernarg_segment_ptr 0\n"),
            std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_system_sgpr_workgroup_id_x 1\n"),
            std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_next_free_vgpr 0\n"), std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_next_free_sgpr "), std::string::npos);
  EXPECT_EQ(output.find("  .amdhsa_next_free_sgpr 0\n"), std::string::npos);
  EXPECT_NE(output.find(".end_amdhsa_kernel\n"), std::string::npos);
  EXPECT_NE(output.find(".amdgpu_metadata\n"), std::string::npos);
  EXPECT_NE(output.find("  amdhsa.target: "
                        "'amdgcn-amd-amdhsa--gfx1100'\n"),
            std::string::npos);
  EXPECT_NE(output.find("  amdhsa.kernels:\n"), std::string::npos);
  EXPECT_NE(output.find("    - .name: 'loom_kernel'\n"), std::string::npos);
  EXPECT_NE(output.find("      .symbol: 'loom_kernel.kd'\n"),
            std::string::npos);
  EXPECT_NE(output.find("      .wavefront_size: 32\n"), std::string::npos);
  EXPECT_NE(output.find("      .args: []\n"), std::string::npos);
  EXPECT_NE(output.find(".end_amdgpu_metadata\n"), std::string::npos);
}

TEST_F(AmdgpuKernelAssemblyTest, EmitsFixedSegmentSizesFromLowSlots) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildSidecarsForPreset(
      "amdgpu-gfx11", "gfx_target",
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
      "  low.return\n"
      "}\n"
      "low.slot @lds0 {align = 64, function = @loom_kernel, size = 128, "
      "space = lds}\n"
      "low.slot @lds1 {align = 16, function = @loom_kernel, size = 16, "
      "space = lds}\n"
      "low.slot @private0 {align = 16, function = @loom_kernel, size = 12, "
      "space = private}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_ASSERT_OK(loom_amdgpu_emit_kernel_assembly(&packetization.schedule,
                                                  &packetization.allocation,
                                                  &builder, &sidecar_arena));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);
  EXPECT_NE(output.find("  .amdhsa_group_segment_fixed_size 144\n"),
            std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_private_segment_fixed_size 12\n"),
            std::string::npos);
  EXPECT_NE(output.find("      .group_segment_fixed_size: 144\n"),
            std::string::npos);
  EXPECT_NE(output.find("      .private_segment_fixed_size: 12\n"),
            std::string::npos);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(AmdgpuKernelAssemblyTest, EmitsHalBufferResourceMetadata) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  loom_amdgpu_hal_kernel_abi_layout_t abi_layout = {};
  BuildMaterializedHalResourceSidecarsForPreset(
      "amdgpu-gfx11", "gfx_target",
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
      "  %binding = low.resource<hal_buffer_resource> {index = 0, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  %value = low.const<amdgpu.v_mov_b32> {imm32 = 1} : "
      "reg<amdgpu.vgpr>\n"
      "  %vaddr = low.const<amdgpu.v_mov_b32> {imm32 = 0} : "
      "reg<amdgpu.vgpr>\n"
      "  %zero = low.const<amdgpu.s_mov_b32> {imm32 = 0} : "
      "reg<amdgpu.sgpr>\n"
      "  low.op<amdgpu.buffer_store_dword>(%value, %binding, %vaddr, %zero) "
      "{offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      1, 1, &abi_layout, &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_amdgpu_kernel_assembly_options_t assembly_options = {
      .abi_layout = &abi_layout,
  };
  IREE_ASSERT_OK(loom_amdgpu_emit_kernel_assembly_with_options(
      &packetization.schedule, &packetization.allocation, &assembly_options,
      &builder, &sidecar_arena));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);

  EXPECT_NE(output.find("  .amdhsa_kernarg_size 8\n"), std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_user_sgpr_count 2\n"), std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_user_sgpr_kernarg_segment_ptr 1\n"),
            std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_next_free_sgpr "), std::string::npos);
  EXPECT_EQ(output.find("  .amdhsa_next_free_sgpr 0\n"), std::string::npos);
  EXPECT_NE(output.find("      .kernarg_segment_size: 8\n"), std::string::npos);
  EXPECT_NE(output.find("      .kernarg_segment_align: 8\n"),
            std::string::npos);
  EXPECT_NE(output.find("      .args:\n"), std::string::npos);
  EXPECT_NE(output.find("        - .name: 'binding0'\n"), std::string::npos);
  EXPECT_NE(output.find("          .offset: 0\n"), std::string::npos);
  EXPECT_NE(output.find("          .size: 8\n"), std::string::npos);
  EXPECT_NE(output.find("          .value_kind: global_buffer\n"),
            std::string::npos);
  EXPECT_NE(output.find("          .align: 8\n"), std::string::npos);
  EXPECT_NE(output.find("          .address_space: global\n"),
            std::string::npos);

  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(AmdgpuKernelAssemblyTest, EmitsWorkitemZDescriptorMode) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  loom_amdgpu_hal_kernel_abi_layout_t abi_layout = {};
  BuildMaterializedHalResourceSidecarsForPreset(
      "amdgpu-gfx11", "gfx_target",
      R"(low.kernel.def target(@gfx_target) @loom_kernel() {
  %packed_tid = low.live_in<amdgpu.workitem_id.packed.xyz> : reg<amdgpu.vgpr>
  %binding = low.resource<hal_buffer_resource> {index = 0, semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>
  %vaddr = low.const<amdgpu.v_mov_b32> {imm32 = 0} : reg<amdgpu.vgpr>
  %zero = low.const<amdgpu.s_mov_b32> {imm32 = 0} : reg<amdgpu.sgpr>
  low.op<amdgpu.buffer_store_dword>(%packed_tid, %binding, %vaddr, %zero) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)
  low.return
}
)",
      1, 2, &abi_layout, &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_amdgpu_kernel_assembly_options_t assembly_options = {
      .abi_layout = &abi_layout,
  };
  IREE_ASSERT_OK(loom_amdgpu_emit_kernel_assembly_with_options(
      &packetization.schedule, &packetization.allocation, &assembly_options,
      &builder, &sidecar_arena));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);

  EXPECT_NE(output.find("  .amdhsa_system_vgpr_workitem_id 2\n"),
            std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_next_free_vgpr "), std::string::npos);
  EXPECT_EQ(output.find("  .amdhsa_next_free_vgpr 0\n"), std::string::npos);

  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(AmdgpuKernelAssemblyTest, EmitsWorkgroupZDescriptorFlags) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  loom_amdgpu_hal_kernel_abi_layout_t abi_layout = {};
  BuildMaterializedHalResourceSidecarsForPreset(
      "amdgpu-gfx11", "gfx_target",
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
      "  %bid_z = low.live_in<" LOOM_AMDGPU_HAL_KERNEL_ABI_WORKGROUP_ID_Z_SOURCE
      "> : reg<amdgpu.sgpr>\n"
      "  %resource = low.resource<hal_buffer_resource> {index = 0, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  %value = low.const<amdgpu.v_mov_b32> {imm32 = 1} : "
      "reg<amdgpu.vgpr>\n"
      "  %vaddr = low.const<amdgpu.v_mov_b32> {imm32 = 0} : "
      "reg<amdgpu.vgpr>\n"
      "  low.op<amdgpu.buffer_store_dword>(%value, %resource, %vaddr, "
      "%bid_z) {offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      1, 2, &abi_layout, &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_amdgpu_kernel_assembly_options_t assembly_options = {
      .abi_layout = &abi_layout,
  };
  IREE_ASSERT_OK(loom_amdgpu_emit_kernel_assembly_with_options(
      &packetization.schedule, &packetization.allocation, &assembly_options,
      &builder, &sidecar_arena));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);

  EXPECT_NE(output.find("  .amdhsa_system_sgpr_workgroup_id_x 1\n"),
            std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_system_sgpr_workgroup_id_y 1\n"),
            std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_system_sgpr_workgroup_id_z 1\n"),
            std::string::npos);

  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(AmdgpuKernelAssemblyTest,
       EmitsMaterializedHalBufferStoreKernelForGfx11) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  loom_amdgpu_hal_kernel_abi_layout_t abi_layout = {};
  BuildMaterializedHalResourceSidecarsForPreset(
      "amdgpu-gfx11", "gfx_target",
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
      "  %binding = low.resource<hal_buffer_resource> {index = 0, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  %value = low.const<amdgpu.v_mov_b32> {imm32 = 42} : "
      "reg<amdgpu.vgpr>\n"
      "  %vaddr = low.const<amdgpu.v_mov_b32> {imm32 = 0} : "
      "reg<amdgpu.vgpr>\n"
      "  %zero = low.const<amdgpu.s_mov_b32> {imm32 = 0} : "
      "reg<amdgpu.sgpr>\n"
      "  low.op<amdgpu.buffer_store_dword>(%value, %binding, %vaddr, %zero) "
      "{offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      1, 1, &abi_layout, &sidecar_arena, &packetization);

  loom_amdgpu_wait_plan_t wait_plan = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_plan_build(&packetization.schedule,
                                             &sidecar_arena, &wait_plan));
  loom_amdgpu_wait_packet_plan_t wait_packets = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_packet_plan_build(&wait_plan, &sidecar_arena,
                                                    &wait_packets));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_amdgpu_kernel_assembly_options_t assembly_options = {
      .abi_layout = &abi_layout,
      .wait_packets = &wait_packets,
  };
  IREE_ASSERT_OK(loom_amdgpu_emit_kernel_assembly_with_options(
      &packetization.schedule, &packetization.allocation, &assembly_options,
      &builder, &sidecar_arena));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);

  const size_t load = output.find("s_load_dwordx2 s");
  const size_t store = output.find("buffer_store_dword v");
  const size_t wait_before_store = output.rfind("s_waitcnt", store);
  const size_t wait_after_store = output.find("s_waitcnt", store);
  const size_t endpgm = output.find("s_endpgm");
  EXPECT_NE(output.find("v_mov_b32 v"), std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_kernarg_size 8\n"), std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_user_sgpr_kernarg_segment_ptr 1\n"),
            std::string::npos);
  EXPECT_NE(output.find("      .args:\n"), std::string::npos);
  EXPECT_NE(output.find("        - .name: 'binding0'\n"), std::string::npos);
  EXPECT_NE(output.find("          .value_kind: global_buffer\n"),
            std::string::npos);
  EXPECT_NE(load, std::string::npos);
  EXPECT_NE(store, std::string::npos);
  EXPECT_NE(wait_before_store, std::string::npos);
  EXPECT_NE(wait_after_store, std::string::npos);
  EXPECT_NE(endpgm, std::string::npos);
  EXPECT_LT(load, wait_before_store);
  EXPECT_LT(wait_before_store, store);
  EXPECT_LT(store, wait_after_store);
  EXPECT_LT(wait_after_store, endpgm);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(AmdgpuKernelAssemblyTest, EmitsB128CopyKernelForGfx11) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  loom_amdgpu_hal_kernel_abi_layout_t abi_layout = {};
  BuildMaterializedHalResourceSidecarsForPreset(
      "amdgpu-gfx11", "gfx_target",
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
      "  %tid = low.live_in<" LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE
      "> : reg<amdgpu.vgpr>\n"
      "  %source = low.resource<hal_buffer_resource> {index = 0, semantic_type "
      "= hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  %target = low.resource<hal_buffer_resource> {index = 1, semantic_type "
      "= hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  %scale = low.const<amdgpu.v_mov_b32> {imm32 = 16} : "
      "reg<amdgpu.vgpr>\n"
      "  %byte_offset = low.op<amdgpu.v_mul_lo_u32>(%tid, %scale) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  %zero = low.const<amdgpu.s_mov_b32> {imm32 = 0} : "
      "reg<amdgpu.sgpr>\n"
      "  %loaded = low.op<amdgpu.buffer_load_b128>(%source, %byte_offset, "
      "%zero) {offset = 0} : (reg<amdgpu.sgpr x4>, reg<amdgpu.vgpr>, "
      "reg<amdgpu.sgpr>) -> reg<amdgpu.vgpr x4>\n"
      "  low.op<amdgpu.buffer_store_b128>(%loaded, %target, %byte_offset, "
      "%zero) {offset = 0} : (reg<amdgpu.vgpr x4>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      2, 2, &abi_layout, &sidecar_arena, &packetization);

  loom_amdgpu_wait_plan_t wait_plan = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_plan_build(&packetization.schedule,
                                             &sidecar_arena, &wait_plan));
  loom_amdgpu_wait_packet_plan_t wait_packets = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_packet_plan_build(&wait_plan, &sidecar_arena,
                                                    &wait_packets));
  ASSERT_GE(wait_packets.packet_count, 2u);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_amdgpu_kernel_assembly_options_t assembly_options = {
      .abi_layout = &abi_layout,
      .wait_packets = &wait_packets,
  };
  IREE_ASSERT_OK(loom_amdgpu_emit_kernel_assembly_with_options(
      &packetization.schedule, &packetization.allocation, &assembly_options,
      &builder, &sidecar_arena));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);

  const size_t load = output.find("buffer_load_b128 v[");
  const size_t store = output.find("buffer_store_b128 v[");
  const size_t wait_before_store = output.rfind("s_waitcnt", store);
  const size_t wait_after_store = output.find("s_waitcnt", store);
  const size_t endpgm = output.find("s_endpgm");
  EXPECT_NE(output.find("  .amdhsa_kernarg_size 16\n"), std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_user_sgpr_kernarg_segment_ptr 1\n"),
            std::string::npos);
  EXPECT_NE(output.find("  .amdhsa_system_vgpr_workitem_id 0\n"),
            std::string::npos);
  EXPECT_NE(output.find("        - .name: 'binding0'\n"), std::string::npos);
  EXPECT_NE(output.find("        - .name: 'binding1'\n"), std::string::npos);
  EXPECT_NE(output.find("          .offset: 8\n"), std::string::npos);
  EXPECT_NE(output.find("v_mul_lo_u32 v"), std::string::npos);
  EXPECT_NE(load, std::string::npos);
  EXPECT_NE(store, std::string::npos);
  EXPECT_NE(wait_before_store, std::string::npos);
  EXPECT_NE(wait_after_store, std::string::npos);
  EXPECT_NE(endpgm, std::string::npos);
  EXPECT_LT(load, wait_before_store);
  EXPECT_LT(wait_before_store, store);
  EXPECT_LT(store, wait_after_store);
  EXPECT_LT(wait_after_store, endpgm);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(AmdgpuKernelAssemblyTest, EmitsMemoryAluStressKernelForGfx11) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  loom_amdgpu_hal_kernel_abi_layout_t abi_layout = {};
  BuildMaterializedHalResourceSidecarsForPreset(
      "amdgpu-gfx11", "gfx_target",
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
      "  %tid = low.live_in<" LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE
      "> : reg<amdgpu.vgpr>\n"
      "  %source = low.resource<hal_buffer_resource> {index = 0, semantic_type "
      "= hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  %wide_target = low.resource<hal_buffer_resource> {index = 1, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  %scalar_target = low.resource<hal_buffer_resource> {index = 2, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  %zero_v = low.const<amdgpu.v_mov_b32> {imm32 = 0} : "
      "reg<amdgpu.vgpr>\n"
      "  %four = low.const<amdgpu.v_mov_b32> {imm32 = 4} : "
      "reg<amdgpu.vgpr>\n"
      "  %byte_offset = low.op<amdgpu.v_mul_lo_u32>(%tid, %four) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
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
      "}\n",
      3, 2, &abi_layout, &sidecar_arena, &packetization);

  loom_amdgpu_wait_plan_t wait_plan = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_plan_build(&packetization.schedule,
                                             &sidecar_arena, &wait_plan));
  loom_amdgpu_wait_packet_plan_t wait_packets = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_packet_plan_build(&wait_plan, &sidecar_arena,
                                                    &wait_packets));
  ASSERT_GE(wait_packets.packet_count, 2u);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  const loom_amdgpu_kernel_assembly_options_t assembly_options = {
      .abi_layout = &abi_layout,
      .wait_packets = &wait_packets,
  };
  IREE_ASSERT_OK(loom_amdgpu_emit_kernel_assembly_with_options(
      &packetization.schedule, &packetization.allocation, &assembly_options,
      &builder, &sidecar_arena));
  const std::string output(iree_string_builder_view(&builder).data,
                           iree_string_builder_view(&builder).size);

  const size_t load = output.find("buffer_load_b128 v[");
  const size_t add = output.find("v_add_u32 v");
  const size_t wide_store = output.find("buffer_store_b128 v[");
  const size_t scalar_store = output.find("buffer_store_dword v");
  const size_t wait_before_wide_store = output.rfind("s_waitcnt", wide_store);
  const size_t wait_after_scalar_store = output.find("s_waitcnt", scalar_store);
  const size_t endpgm = output.find("s_endpgm");
  EXPECT_NE(output.find("  .amdhsa_kernarg_size 24\n"), std::string::npos);
  EXPECT_NE(output.find("        - .name: 'binding0'\n"), std::string::npos);
  EXPECT_NE(output.find("        - .name: 'binding1'\n"), std::string::npos);
  EXPECT_NE(output.find("        - .name: 'binding2'\n"), std::string::npos);
  EXPECT_NE(output.find("v_mul_lo_u32 v"), std::string::npos);
  EXPECT_NE(load, std::string::npos);
  EXPECT_NE(add, std::string::npos);
  EXPECT_NE(wide_store, std::string::npos);
  EXPECT_NE(scalar_store, std::string::npos);
  EXPECT_NE(wait_before_wide_store, std::string::npos);
  EXPECT_NE(wait_after_scalar_store, std::string::npos);
  EXPECT_NE(endpgm, std::string::npos);
  EXPECT_LT(load, add);
  EXPECT_LT(add, wait_before_wide_store);
  EXPECT_LT(wait_before_wide_store, wide_store);
  EXPECT_LT(wide_store, scalar_store);
  EXPECT_LT(scalar_store, wait_after_scalar_store);
  EXPECT_LT(wait_after_scalar_store, endpgm);
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(AmdgpuKernelAssemblyTest, EmitsTargetIdsForCurrentAmdgpuPresets) {
  struct Case {
    const char* preset_key;
    const char* target_cpu;
    uint32_t wavefront_size;
  };
  const Case cases[] = {
      {"amdgpu-gfx950", "gfx950", 64},
      {"amdgpu-gfx11", "gfx1100", 32},
      {"amdgpu-gfx12", "gfx1200", 32},
      {"amdgpu-gfx1250", "gfx1250", 32},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.preset_key);
    std::string output;
    EmitKernelForPreset(test_case.preset_key, test_case.target_cpu, &output);
    EXPECT_NE(output.find(std::string("      .wavefront_size: ") +
                          std::to_string(test_case.wavefront_size) + "\n"),
              std::string::npos);
  }
}

TEST_F(AmdgpuKernelAssemblyTest, RejectsFunctionArgumentsBeforeAbiLowering) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildSidecarsForPreset(
      "amdgpu-gfx11", "gfx_target",
      "low.kernel.def target(@gfx_target) @loom_kernel(%arg: "
      "reg<amdgpu.sgpr>) {\n"
      "  low.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_amdgpu_emit_kernel_assembly(
                            &packetization.schedule, &packetization.allocation,
                            &builder, &sidecar_arena));
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(AmdgpuKernelAssemblyTest, RejectsNonDefaultLinkage) {
  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  BuildSidecarsFromSource(
      "target.profile @gfx_target preset(\"amdgpu-gfx11\") {linkage = "
      "\"dso_local\"}\n"
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
      "  low.return\n"
      "}\n",
      &sidecar_arena, &packetization);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_amdgpu_emit_kernel_assembly(
                            &packetization.schedule, &packetization.allocation,
                            &builder, &sidecar_arena));
  iree_string_builder_deinitialize(&builder);
  iree_arena_deinitialize(&sidecar_arena);
}

}  // namespace
}  // namespace loom
