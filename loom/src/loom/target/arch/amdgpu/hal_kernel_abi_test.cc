// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal_kernel_abi.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/ir_records.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class AmdgpuHalKernelAbiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
  }

  void TearDown() override {
    ResetModule();
    iree_arena_deinitialize(&arena_);
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
                        IREE_SV("amdgpu_hal_kernel_abi_test.loom"), &context_,
                        &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  const loom_op_t* FindFirstLowFunction() {
    loom_block_t* block = loom_module_block(module_);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (loom_low_func_def_isa(op)) {
        return op;
      }
    }
    return nullptr;
  }

  void BuildModule(const char* export_source_symbol, const char* function_body,
                   const char* resource_records) {
    std::string source =
        "target.snapshot @gfx1100 {artifact_format = elf, codegen_format = "
        "low_native, data_layout = \"\", default_pointer_bitwidth = 64, "
        "index_bitwidth = 32, memory_space_constant = 4, "
        "memory_space_descriptor = 7, memory_space_generic = 0, "
        "memory_space_global = 1, memory_space_host = 4294967295, "
        "memory_space_private = 5, memory_space_workgroup = 3, "
        "offset_bitwidth = 64, target_cpu = \"gfx1100\", "
        "target_features = \"\", target_triple = \"amdgcn-amd-amdhsa\"}\n"
        "target.export @gfx_export {abi = hal_kernel, export_symbol = "
        "\"loom_kernel\", hal_binding_alignment = 16, "
        "hal_buffer_resource_flags = 159744, hal_flat_workgroup_size_max = "
        "64, hal_flat_workgroup_size_min = 64, hal_workgroup_size_x = 64, "
        "hal_workgroup_size_y = 1, hal_workgroup_size_z = 1, linkage = "
        "default, source = @";
    source += export_source_symbol;
    source +=
        "}\n"
        "target.config @gfx_config {contract_feature_bits = 0, "
        "contract_set_key = \"amdgpu.gfx11.core\"}\n"
        "target.bundle @gfx_target {config = @gfx_config, export_plan = "
        "@gfx_export, snapshot = @gfx1100}\n";
    source += function_body;
    source += resource_records;

    ResetModule();
    module_ = ParseSource(source);
    ASSERT_NE(module_, nullptr);
  }

  void BuildLayout(loom_amdgpu_hal_kernel_abi_layout_t* out_layout) {
    ASSERT_NE(module_, nullptr);
    const loom_op_t* function_op = FindFirstLowFunction();
    ASSERT_NE(function_op, nullptr);
    loom_target_ir_bundle_storage_t storage = {};
    IREE_ASSERT_OK(loom_target_ir_bundle_from_symbol_name(
        module_, IREE_SV("gfx_target"), &storage));
    IREE_ASSERT_OK(loom_amdgpu_hal_kernel_abi_layout_from_low(
        module_, function_op, &storage.bundle, out_layout, &arena_));
  }

  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t arena_ = {};
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
};

TEST_F(AmdgpuHalKernelAbiTest, LaysOutOneHalBufferResource) {
  BuildModule("loom_kernel",
              "low.func.def target(@gfx_target) @loom_kernel() {\n"
              "  %binding = low.resource @binding0 : reg<amdgpu.sgpr x4>\n"
              "  low.return\n"
              "}\n",
              "low.abi.resource @binding0 {function = @loom_kernel, kind = "
              "hal_buffer_resource, index = 0, semantic_type = hal.buffer, "
              "abi_type = reg<amdgpu.sgpr x4>}\n");

  loom_amdgpu_hal_kernel_abi_layout_t layout = {};
  BuildLayout(&layout);

  EXPECT_EQ(layout.kernarg_segment_size, 8u);
  EXPECT_EQ(layout.kernarg_segment_alignment, 8u);
  EXPECT_TRUE(layout.uses_kernarg_segment_ptr);
  ASSERT_EQ(layout.resource_count, 1u);
  const loom_amdgpu_hal_kernarg_resource_t& resource = layout.resources[0];
  EXPECT_EQ(ToString(resource.name), "binding0");
  EXPECT_EQ(resource.binding_index, 0u);
  EXPECT_EQ(resource.kernarg_offset, 0u);
  EXPECT_EQ(resource.kernarg_size, 8u);
  EXPECT_EQ(resource.kernarg_alignment, 8u);
  EXPECT_TRUE(loom_type_is_dialect(resource.semantic_type));
  EXPECT_TRUE(loom_type_is_register(resource.abi_type));
  EXPECT_EQ(loom_type_register_unit_count(resource.abi_type), 4u);
}

TEST_F(AmdgpuHalKernelAbiTest, AllowsNoResources) {
  BuildModule("loom_kernel",
              "low.func.def target(@gfx_target) @loom_kernel() {\n"
              "  low.return\n"
              "}\n",
              "");

  loom_amdgpu_hal_kernel_abi_layout_t layout = {};
  BuildLayout(&layout);

  EXPECT_EQ(layout.kernarg_segment_size, 0u);
  EXPECT_EQ(layout.kernarg_segment_alignment, 8u);
  EXPECT_FALSE(layout.uses_kernarg_segment_ptr);
  EXPECT_EQ(layout.resource_count, 0u);
  EXPECT_EQ(layout.resources, nullptr);
}

TEST_F(AmdgpuHalKernelAbiTest, RejectsDuplicateBindingIndex) {
  BuildModule("loom_kernel",
              "low.func.def target(@gfx_target) @loom_kernel() {\n"
              "  low.return\n"
              "}\n",
              "low.abi.resource @binding0 {function = @loom_kernel, kind = "
              "hal_buffer_resource, index = 0, semantic_type = hal.buffer, "
              "abi_type = reg<amdgpu.sgpr x4>}\n"
              "low.abi.resource @binding1 {function = @loom_kernel, kind = "
              "hal_buffer_resource, index = 0, semantic_type = hal.buffer, "
              "abi_type = reg<amdgpu.sgpr x4>}\n");

  const loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_ir_bundle_storage_t storage = {};
  IREE_ASSERT_OK(loom_target_ir_bundle_from_symbol_name(
      module_, IREE_SV("gfx_target"), &storage));
  loom_amdgpu_hal_kernel_abi_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      loom_amdgpu_hal_kernel_abi_layout_from_low(
          module_, function_op, &storage.bundle, &layout, &arena_));
}

TEST_F(AmdgpuHalKernelAbiTest, RejectsMissingDenseBindingIndex) {
  BuildModule("loom_kernel",
              "low.func.def target(@gfx_target) @loom_kernel() {\n"
              "  low.return\n"
              "}\n",
              "low.abi.resource @binding1 {function = @loom_kernel, kind = "
              "hal_buffer_resource, index = 1, semantic_type = hal.buffer, "
              "abi_type = reg<amdgpu.sgpr x4>}\n");

  const loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_ir_bundle_storage_t storage = {};
  IREE_ASSERT_OK(loom_target_ir_bundle_from_symbol_name(
      module_, IREE_SV("gfx_target"), &storage));
  loom_amdgpu_hal_kernel_abi_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_hal_kernel_abi_layout_from_low(
          module_, function_op, &storage.bundle, &layout, &arena_));
}

TEST_F(AmdgpuHalKernelAbiTest, RejectsWrongAbiType) {
  BuildModule("loom_kernel",
              "low.func.def target(@gfx_target) @loom_kernel() {\n"
              "  low.return\n"
              "}\n",
              "low.abi.resource @binding0 {function = @loom_kernel, kind = "
              "hal_buffer_resource, index = 0, semantic_type = hal.buffer, "
              "abi_type = reg<amdgpu.sgpr x3>}\n");

  const loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_ir_bundle_storage_t storage = {};
  IREE_ASSERT_OK(loom_target_ir_bundle_from_symbol_name(
      module_, IREE_SV("gfx_target"), &storage));
  loom_amdgpu_hal_kernel_abi_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_hal_kernel_abi_layout_from_low(
          module_, function_op, &storage.bundle, &layout, &arena_));
}

TEST_F(AmdgpuHalKernelAbiTest, RejectsMissingResourceRecord) {
  BuildModule("loom_kernel",
              "low.func.def target(@gfx_target) @loom_kernel() {\n"
              "  %binding = low.resource @binding0 : reg<amdgpu.sgpr x4>\n"
              "  low.return\n"
              "}\n",
              "");

  const loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_ir_bundle_storage_t storage = {};
  IREE_ASSERT_OK(loom_target_ir_bundle_from_symbol_name(
      module_, IREE_SV("gfx_target"), &storage));
  loom_amdgpu_hal_kernel_abi_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_amdgpu_hal_kernel_abi_layout_from_low(
          module_, function_op, &storage.bundle, &layout, &arena_));
}

TEST_F(AmdgpuHalKernelAbiTest, RejectsExportSourceMismatch) {
  BuildModule("other_kernel",
              "low.func.def target(@gfx_target) @loom_kernel() {\n"
              "  low.return\n"
              "}\n",
              "");

  const loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_ir_bundle_storage_t storage = {};
  IREE_ASSERT_OK(loom_target_ir_bundle_from_symbol_name(
      module_, IREE_SV("gfx_target"), &storage));
  loom_amdgpu_hal_kernel_abi_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_amdgpu_hal_kernel_abi_layout_from_low(
          module_, function_op, &storage.bundle, &layout, &arena_));
}

}  // namespace
}  // namespace loom
