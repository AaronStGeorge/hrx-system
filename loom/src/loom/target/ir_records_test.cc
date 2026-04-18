// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/ir_records.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/target/ops.h"

namespace loom {
namespace {

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class TargetIrRecordsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_op_registry_initialize_context(iree_allocator_system(),
                                                       &context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* ParseSource(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;

    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("target_ir_records_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  const loom_op_t* FindFirstOp(loom_module_t* module, loom_op_kind_t kind) {
    loom_block_t* block = loom_module_block(module);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (op->kind == kind) return op;
    }
    return nullptr;
  }

  loom_op_t* FindFirstMutableOp(loom_module_t* module, loom_op_kind_t kind) {
    loom_block_t* block = loom_module_block(module);
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (op->kind == kind) return op;
    }
    return nullptr;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(TargetIrRecordsTest, MaterializesWasmBundle) {
  const char source[] =
      "target.snapshot @wasm32 {codegen_format = wasm, target_triple = "
      "\"wasm32-unknown-unknown\", data_layout = \"\", artifact_format = "
      "wasm_binary, target_cpu = \"generic\", target_features = "
      "\"+simd128\", default_pointer_bitwidth = 32, index_bitwidth = 32, "
      "offset_bitwidth = 32, memory_space_generic = 0, memory_space_global = "
      "0, memory_space_workgroup = 4294967295, memory_space_constant = 0, "
      "memory_space_private = 4294967295, memory_space_host = 4294967295, "
      "memory_space_descriptor = 4294967295}\n"
      "target.export @wasm_export {export_symbol = \"matmul\", abi = "
      "wasm_function, linkage = default, hal_binding_alignment = 0, "
      "hal_workgroup_size_x = 0, hal_workgroup_size_y = 0, "
      "hal_workgroup_size_z = 0, hal_flat_workgroup_size_min = 0, "
      "hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = 0}\n"
      "target.config @wasm_config {contract_set_key = "
      "\"wasm.core.simd128\", contract_feature_bits = 7}\n"
      "target.bundle @wasm_target {snapshot = @wasm32, export_plan = "
      "@wasm_export, config = @wasm_config}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_target_ir_bundle_storage_t storage = {};
  IREE_ASSERT_OK(loom_target_ir_bundle_from_ops(
      module, FindFirstOp(module, LOOM_OP_TARGET_BUNDLE),
      FindFirstOp(module, LOOM_OP_TARGET_SNAPSHOT),
      FindFirstOp(module, LOOM_OP_TARGET_EXPORT),
      FindFirstOp(module, LOOM_OP_TARGET_CONFIG), &storage));

  EXPECT_EQ(ToString(storage.bundle.name), "wasm_target");
  EXPECT_EQ(storage.bundle.snapshot, &storage.snapshot);
  EXPECT_EQ(storage.snapshot.codegen_format, LOOM_TARGET_CODEGEN_FORMAT_WASM);
  EXPECT_EQ(storage.snapshot.artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY);
  EXPECT_EQ(ToString(storage.snapshot.target_triple), "wasm32-unknown-unknown");
  EXPECT_EQ(ToString(storage.snapshot.target_features), "+simd128");
  EXPECT_EQ(storage.snapshot.default_pointer_bitwidth, 32u);
  EXPECT_EQ(storage.snapshot.memory_spaces.workgroup, UINT32_MAX);
  EXPECT_EQ(storage.export_plan.abi_kind, LOOM_TARGET_ABI_WASM_FUNCTION);
  EXPECT_TRUE(iree_string_view_is_empty(storage.export_plan.source_symbol));
  EXPECT_EQ(ToString(storage.export_plan.export_symbol), "matmul");
  EXPECT_EQ(ToString(storage.config.contract_set_key), "wasm.core.simd128");
  EXPECT_EQ(storage.config.contract_feature_bits, 7u);

  loom_module_free(module);
}

TEST_F(TargetIrRecordsTest, MaterializesHalKernelBundle) {
  const char source[] =
      "target.snapshot @gfx1100 {codegen_format = low_native, target_triple = "
      "\"amdgcn-amd-amdhsa\", data_layout = \"amdgpu-layout\", "
      "artifact_format = elf, target_cpu = \"gfx1100\", target_features = "
      "\"+wavefrontsize32\", default_pointer_bitwidth = 64, index_bitwidth = "
      "32, offset_bitwidth = 64, memory_space_generic = 0, "
      "memory_space_global = 1, memory_space_workgroup = 3, "
      "memory_space_constant = 4, memory_space_private = 5, "
      "memory_space_host = 4294967295, memory_space_descriptor = 7}\n"
      "target.export @hal_export {export_symbol = \"kernel\", abi = "
      "hal_kernel, linkage = default, hal_binding_alignment = 16, "
      "hal_workgroup_size_x = 64, hal_workgroup_size_y = 2, "
      "hal_workgroup_size_z = 1, hal_flat_workgroup_size_min = 64, "
      "hal_flat_workgroup_size_max = 128, hal_buffer_resource_flags = "
      "159744}\n"
      "target.config @gfx_config {contract_set_key = \"amdgpu.gfx11.core\", "
      "contract_feature_bits = 0}\n"
      "target.bundle @gfx_target {snapshot = @gfx1100, export_plan = "
      "@hal_export, config = @gfx_config}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);

  loom_target_ir_bundle_storage_t storage = {};
  IREE_ASSERT_OK(loom_target_ir_bundle_from_ops(
      module, FindFirstOp(module, LOOM_OP_TARGET_BUNDLE),
      FindFirstOp(module, LOOM_OP_TARGET_SNAPSHOT),
      FindFirstOp(module, LOOM_OP_TARGET_EXPORT),
      FindFirstOp(module, LOOM_OP_TARGET_CONFIG), &storage));

  EXPECT_EQ(storage.snapshot.codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE);
  EXPECT_EQ(storage.snapshot.artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(ToString(storage.snapshot.target_cpu), "gfx1100");
  EXPECT_EQ(storage.snapshot.memory_spaces.descriptor, 7u);
  EXPECT_EQ(storage.export_plan.abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_EQ(storage.export_plan.hal_kernel.binding_alignment, 16u);
  EXPECT_EQ(storage.export_plan.hal_kernel.required_workgroup_size.x, 64u);
  EXPECT_EQ(storage.export_plan.hal_kernel.required_workgroup_size.y, 2u);
  EXPECT_EQ(storage.export_plan.hal_kernel.flat_workgroup_size_max, 128u);
  EXPECT_EQ(storage.export_plan.hal_kernel.buffer_resource_flags, 159744u);
  EXPECT_EQ(ToString(storage.config.contract_set_key), "amdgpu.gfx11.core");

  loom_module_free(module);
}

TEST_F(TargetIrRecordsTest, RejectsWrongRecordKind) {
  const char source[] =
      "target.config @config {contract_set_key = \"iree.vm.core\", "
      "contract_feature_bits = 0}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  const loom_op_t* config = FindFirstOp(module, LOOM_OP_TARGET_CONFIG);
  ASSERT_NE(config, nullptr);

  loom_target_snapshot_t snapshot = {};
  iree_status_t status =
      loom_target_ir_snapshot_from_op(module, config, &snapshot);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  loom_module_free(module);
}

TEST_F(TargetIrRecordsTest, CarriesFutureNonzeroExportAbiOrdinal) {
  const char source[] =
      "target.export @future_export {export_symbol = \"future\", abi = "
      "object_function, linkage = default, hal_binding_alignment = 0, "
      "hal_workgroup_size_x = 0, hal_workgroup_size_y = 0, "
      "hal_workgroup_size_z = 0, hal_flat_workgroup_size_min = 0, "
      "hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = 0}\n";
  loom_module_t* module = ParseSource(source);
  ASSERT_NE(module, nullptr);
  loom_op_t* export_op = FindFirstMutableOp(module, LOOM_OP_TARGET_EXPORT);
  ASSERT_NE(export_op, nullptr);
  loom_op_attrs(export_op)[loom_target_export_abi_ATTR_INDEX] =
      loom_attr_enum(250);

  loom_target_export_plan_t export_plan = {};
  IREE_ASSERT_OK(
      loom_target_ir_export_plan_from_op(module, export_op, &export_plan));
  EXPECT_EQ((uint32_t)export_plan.abi_kind, 250u);

  loom_module_free(module);
}

}  // namespace
}  // namespace loom
