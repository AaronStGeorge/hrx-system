// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal_kernel_abi.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/arch/amdgpu/gfx11_descriptors.h"
#include "loom/target/arch/amdgpu/low_registry.h"

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
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_amdgpu_low_descriptor_registry_initialize(&target_registry_);
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

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  const loom_op_t* FindFirstLowFunction() {
    loom_block_t* block = loom_module_block(module_);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (loom_low_function_def_isa(op)) {
        return op;
      }
    }
    return nullptr;
  }

  void BuildModule(const char* function_body, const char* resource_records) {
    std::string source =
        "target.profile @gfx_target preset(\"amdgpu-gfx11\")\n";
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
    loom_target_bundle_storage_t storage = {};
    ResolveFunctionTarget(function_op, &storage);
    IREE_ASSERT_OK(loom_amdgpu_hal_kernel_abi_layout_from_low(
        module_, function_op, &storage.bundle,
        loom_amdgpu_gfx11_core_descriptor_set(), out_layout, &arena_));
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

  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t arena_ = {};
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t target_registry_ = {};
};

TEST_F(AmdgpuHalKernelAbiTest, LaysOutOneHalBufferResource) {
  BuildModule(
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
      "  %binding = low.resource<hal_buffer_resource> {index = 0, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  low.return\n"
      "}\n",
      "");

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

TEST_F(AmdgpuHalKernelAbiTest, LaysOutTwoHalBufferResources) {
  BuildModule(
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
      "  %input = low.resource<hal_buffer_resource> {index = 0, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  %output = low.resource<hal_buffer_resource> {index = 1, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  low.return\n"
      "}\n",
      "");

  loom_amdgpu_hal_kernel_abi_layout_t layout = {};
  BuildLayout(&layout);

  EXPECT_EQ(layout.kernarg_segment_size, 16u);
  EXPECT_EQ(layout.kernarg_segment_alignment, 8u);
  EXPECT_TRUE(layout.uses_kernarg_segment_ptr);
  ASSERT_EQ(layout.resource_count, 2u);
  EXPECT_EQ(ToString(layout.resources[0].name), "binding0");
  EXPECT_EQ(layout.resources[0].binding_index, 0u);
  EXPECT_EQ(layout.resources[0].kernarg_offset, 0u);
  EXPECT_EQ(layout.resources[1].binding_index, 1u);
  EXPECT_EQ(ToString(layout.resources[1].name), "binding1");
  EXPECT_EQ(layout.resources[1].kernarg_offset, 8u);
}

TEST_F(AmdgpuHalKernelAbiTest, AllowsNoResources) {
  BuildModule(
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
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
  BuildModule(
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
      "  %binding0 = low.resource<hal_buffer_resource> {index = 0, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  %binding1 = low.resource<hal_buffer_resource> {index = 0, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  low.return\n"
      "}\n",
      "");

  const loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_bundle_storage_t storage = {};
  ResolveFunctionTarget(function_op, &storage);
  loom_amdgpu_hal_kernel_abi_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      loom_amdgpu_hal_kernel_abi_layout_from_low(
          module_, function_op, &storage.bundle,
          loom_amdgpu_gfx11_core_descriptor_set(), &layout, &arena_));
}

TEST_F(AmdgpuHalKernelAbiTest, RejectsMissingDenseBindingIndex) {
  BuildModule(
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
      "  %binding1 = low.resource<hal_buffer_resource> {index = 1, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  low.return\n"
      "}\n",
      "");

  const loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_bundle_storage_t storage = {};
  ResolveFunctionTarget(function_op, &storage);
  loom_amdgpu_hal_kernel_abi_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_hal_kernel_abi_layout_from_low(
          module_, function_op, &storage.bundle,
          loom_amdgpu_gfx11_core_descriptor_set(), &layout, &arena_));
}

TEST_F(AmdgpuHalKernelAbiTest, RejectsWrongAbiType) {
  BuildModule(
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
      "  %binding0 = low.resource<hal_buffer_resource> {index = 0, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x3>\n"
      "  low.return\n"
      "}\n",
      "");

  const loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_bundle_storage_t storage = {};
  ResolveFunctionTarget(function_op, &storage);
  loom_amdgpu_hal_kernel_abi_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_hal_kernel_abi_layout_from_low(
          module_, function_op, &storage.bundle,
          loom_amdgpu_gfx11_core_descriptor_set(), &layout, &arena_));
}

TEST_F(AmdgpuHalKernelAbiTest, RejectsWrongImportKind) {
  BuildModule(
      "low.kernel.def target(@gfx_target) @loom_kernel() {\n"
      "  %binding = low.resource<vm_state> {index = 0, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  low.return\n"
      "}\n",
      "");

  const loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_bundle_storage_t storage = {};
  ResolveFunctionTarget(function_op, &storage);
  loom_amdgpu_hal_kernel_abi_layout_t layout = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_hal_kernel_abi_layout_from_low(
          module_, function_op, &storage.bundle,
          loom_amdgpu_gfx11_core_descriptor_set(), &layout, &arena_));
}

}  // namespace
}  // namespace loom
