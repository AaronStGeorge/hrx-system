// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal_resource_materialization.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/hal_kernel_abi.h"
#include "loom/target/arch/amdgpu/low_registry.h"
#include "loom/target/arch/amdgpu/wait_packets.h"
#include "loom/target/arch/amdgpu/wait_plan.h"
#include "loom/target/emit/native/amdgpu/encoding.h"
#include "loom/target/ir_records.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

class AmdgpuHalResourceMaterializationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
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
                        IREE_SV("amdgpu_hal_resource_materialization.loom"),
                        &context_, &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_op_t* FindFirstLowFunction() {
    loom_block_t* block = loom_module_block(module_);
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (loom_low_func_def_isa(op)) {
        return op;
      }
    }
    return nullptr;
  }

  std::string ModuleString(loom_string_id_t string_id) {
    if (string_id == LOOM_STRING_ID_INVALID ||
        string_id >= module_->strings.count) {
      return std::string();
    }
    iree_string_view_t value = module_->strings.entries[string_id];
    return std::string(value.data, value.size);
  }

  bool HasLowResourceOp() {
    loom_op_t* function_op = FindFirstLowFunction();
    if (function_op == nullptr) {
      return false;
    }
    loom_region_t* body = loom_low_func_def_body(function_op);
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(body, block_index);
      const loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (loom_low_resource_isa(op)) {
          return true;
        }
      }
    }
    return false;
  }

  int CountLowOpsWithOpcode(const char* opcode) {
    loom_op_t* function_op = FindFirstLowFunction();
    if (function_op == nullptr) {
      return 0;
    }
    int count = 0;
    loom_region_t* body = loom_low_func_def_body(function_op);
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(body, block_index);
      const loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (loom_low_op_isa(op) &&
            ModuleString(loom_low_op_opcode(op)) == opcode) {
          ++count;
        }
      }
    }
    return count;
  }

  int CountLowDescriptorOpsWithOpcode(const char* opcode) {
    loom_op_t* function_op = FindFirstLowFunction();
    if (function_op == nullptr) {
      return 0;
    }
    int count = 0;
    loom_region_t* body = loom_low_func_def_body(function_op);
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(body, block_index);
      const loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (loom_low_op_isa(op) &&
            ModuleString(loom_low_op_opcode(op)) == opcode) {
          ++count;
        } else if (loom_low_const_isa(op) &&
                   ModuleString(loom_low_const_opcode(op)) == opcode) {
          ++count;
        }
      }
    }
    return count;
  }

  bool HasLowConstWithI64Attr(const char* opcode, const char* attr_name,
                              int64_t value) {
    loom_op_t* function_op = FindFirstLowFunction();
    if (function_op == nullptr) {
      return false;
    }
    loom_region_t* body = loom_low_func_def_body(function_op);
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(body, block_index);
      const loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (!loom_low_const_isa(op) ||
            ModuleString(loom_low_const_opcode(op)) != opcode) {
          continue;
        }
        loom_named_attr_slice_t attrs = loom_low_const_attrs(op);
        for (iree_host_size_t i = 0; i < attrs.count; ++i) {
          if (ModuleString(attrs.entries[i].name_id) == attr_name &&
              attrs.entries[i].value.kind == LOOM_ATTR_I64 &&
              attrs.entries[i].value.i64 == value) {
            return true;
          }
        }
      }
    }
    return false;
  }

  int CountLiveInsWithSource(const char* source) {
    loom_op_t* function_op = FindFirstLowFunction();
    if (function_op == nullptr) {
      return 0;
    }
    int count = 0;
    loom_region_t* body = loom_low_func_def_body(function_op);
    const loom_block_t* entry_block = loom_region_const_entry_block(body);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(entry_block, op) {
      if (loom_low_live_in_isa(op) &&
          ModuleString(loom_low_live_in_source(op)) == source) {
        ++count;
      }
    }
    return count;
  }

  void BuildModule(const char* function_body, const char* resource_records) {
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
        "default, source = @loom_kernel}\n"
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

  void BuildBundle(loom_target_ir_bundle_storage_t* out_storage) {
    IREE_ASSERT_OK(loom_target_ir_bundle_from_symbol_name(
        module_, IREE_SV("gfx_target"), out_storage));
  }

  const loom_low_descriptor_set_t* SelectDescriptorSet(
      const loom_target_bundle_t* bundle) {
    const loom_low_descriptor_set_t* descriptor_set = nullptr;
    IREE_CHECK_OK(loom_target_low_descriptor_set_select_for_bundle(
        &target_registry_.registry, bundle,
        LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        &descriptor_set));
    return descriptor_set;
  }

  void VerifyModule() {
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
  }

  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t arena_ = {};
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t target_registry_ = {};
};

TEST_F(AmdgpuHalResourceMaterializationTest,
       MaterializesHalBufferResourceFromKernargPointer) {
  BuildModule(
      "low.func.def target(@gfx_target) @loom_kernel() {\n"
      "  %value = low.const<amdgpu.v_mov_b32> {imm32 = 42} : "
      "reg<amdgpu.vgpr>\n"
      "  %vaddr = low.const<amdgpu.v_mov_b32> {imm32 = 0} : "
      "reg<amdgpu.vgpr>\n"
      "  %binding = low.resource<hal_buffer_resource> {index = 0, "
      "semantic_type = hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  %zero = low.const<amdgpu.s_mov_b32> {imm32 = 0} : "
      "reg<amdgpu.sgpr>\n"
      "  low.op<amdgpu.buffer_store_dword>(%value, %binding, %vaddr, %zero) "
      "{offset = 0} : (reg<amdgpu.vgpr>, reg<amdgpu.sgpr x4>, "
      "reg<amdgpu.vgpr>, reg<amdgpu.sgpr>)\n"
      "  low.return\n"
      "}\n",
      "");

  loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_ir_bundle_storage_t bundle_storage = {};
  BuildBundle(&bundle_storage);
  const loom_low_descriptor_set_t* descriptor_set =
      SelectDescriptorSet(&bundle_storage.bundle);

  loom_amdgpu_hal_resource_materialization_result_t result = {};
  IREE_ASSERT_OK(loom_amdgpu_hal_resource_materialize(
      module_, function_op, &bundle_storage.bundle, descriptor_set, &result,
      &arena_));
  EXPECT_EQ(result.abi_layout.function_op, function_op);
  EXPECT_EQ(result.abi_layout.resource_count, 1u);
  EXPECT_EQ(result.abi_layout.kernarg_segment_size, 8u);
  EXPECT_TRUE(result.abi_layout.uses_kernarg_segment_ptr);
  EXPECT_EQ(result.materialized_resource_count, 1u);
  EXPECT_TRUE(result.inserted_kernarg_segment_ptr_live_in);
  EXPECT_FALSE(HasLowResourceOp());
  EXPECT_EQ(CountLiveInsWithSource(
                LOOM_AMDGPU_HAL_KERNEL_ABI_KERNARG_SEGMENT_PTR_SOURCE),
            1);
  EXPECT_EQ(CountLowOpsWithOpcode("amdgpu.s_load_dwordx2"), 1);

  const loom_low_allocation_fixed_value_t* fixed_values = nullptr;
  iree_host_size_t fixed_value_count = 0;
  IREE_ASSERT_OK(loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
      module_, function_op, descriptor_set, &fixed_values, &fixed_value_count,
      &arena_));
  ASSERT_EQ(fixed_value_count, 1u);
  EXPECT_EQ(fixed_values[0].location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  EXPECT_EQ(fixed_values[0].location_base, 0u);
  EXPECT_EQ(fixed_values[0].location_count, 2u);

  VerifyModule();
  loom_low_packetization_options_t packetization_options = {
      .descriptor_registry = &target_registry_.registry,
      .allocation_fixed_values = fixed_values,
      .allocation_fixed_value_count = fixed_value_count,
  };
  loom_low_packetization_t packetization = {};
  IREE_ASSERT_OK(loom_low_packetize_function(
      module_, function_op, &packetization_options, &arena_, &packetization));

  loom_amdgpu_wait_plan_t wait_plan = {};
  IREE_ASSERT_OK(loom_amdgpu_wait_plan_build(&packetization.schedule, &arena_,
                                             &wait_plan));
  loom_amdgpu_wait_packet_plan_t wait_packets = {};
  IREE_ASSERT_OK(
      loom_amdgpu_wait_packet_plan_build(&wait_plan, &arena_, &wait_packets));
  EXPECT_GT(wait_packets.packet_count, 0u);

  iree_const_byte_span_t text = iree_const_byte_span_empty();
  IREE_ASSERT_OK(loom_amdgpu_encode_instruction_stream_with_wait_packets(
      &packetization.schedule, &packetization.allocation, &wait_packets, &text,
      &arena_));
  ASSERT_GT(text.data_length, 16u);
  EXPECT_EQ(text.data[text.data_length - 4], 0x00u);
  EXPECT_EQ(text.data[text.data_length - 3], 0x00u);
  EXPECT_EQ(text.data[text.data_length - 2], 0xB0u);
  EXPECT_EQ(text.data[text.data_length - 1], 0xBFu);
}

TEST_F(AmdgpuHalResourceMaterializationTest,
       MaterializesHalBufferResourceRangeFromValidByteCount) {
  BuildModule(
      "low.func.def target(@gfx_target) @loom_kernel() {\n"
      "  %binding = low.resource<hal_buffer_resource> {index = 0, "
      "semantic_type = hal.buffer, valid_byte_count = 64} : "
      "reg<amdgpu.sgpr x4>\n"
      "  low.return\n"
      "}\n",
      "");

  loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_ir_bundle_storage_t bundle_storage = {};
  BuildBundle(&bundle_storage);
  const loom_low_descriptor_set_t* descriptor_set =
      SelectDescriptorSet(&bundle_storage.bundle);

  loom_amdgpu_hal_resource_materialization_result_t result = {};
  IREE_ASSERT_OK(loom_amdgpu_hal_resource_materialize(
      module_, function_op, &bundle_storage.bundle, descriptor_set, &result,
      &arena_));
  EXPECT_FALSE(HasLowResourceOp());
  EXPECT_TRUE(HasLowConstWithI64Attr("amdgpu.s_mov_b32", "imm32", 64));
  VerifyModule();
}

TEST_F(AmdgpuHalResourceMaterializationTest,
       PacketizationCleansDeadDivergentScalarRematerialization) {
  BuildModule(
      "low.func.def target(@gfx_target) @loom_kernel() -> "
      "(reg<amdgpu.vgpr>) {\n"
      "  %tid = low.live_in<" LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE
      "> : reg<amdgpu.vgpr>\n"
      "  %dead_mask = low.const<amdgpu.s_mov_b32> {imm32 = 255} : "
      "reg<amdgpu.sgpr>\n"
      "  %dead_shift = low.const<amdgpu.s_mov_b32> {imm32 = 3} : "
      "reg<amdgpu.sgpr>\n"
      "  %mask = low.const<amdgpu.v_mov_b32> {imm32 = 255} : "
      "reg<amdgpu.vgpr>\n"
      "  %masked = low.op<amdgpu.v_and_b32>(%tid, %mask) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  %shift = low.const<amdgpu.v_mov_b32> {imm32 = 3} : "
      "reg<amdgpu.vgpr>\n"
      "  %left = low.op<amdgpu.v_lshlrev_b32>(%shift, %masked) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  %mixed = low.op<amdgpu.v_xor_b32>(%left, %tid) : "
      "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
      "  low.return %mixed : reg<amdgpu.vgpr>\n"
      "}\n",
      "");

  loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  VerifyModule();
  EXPECT_EQ(CountLowDescriptorOpsWithOpcode("amdgpu.s_mov_b32"), 2);
  loom_target_ir_bundle_storage_t bundle_storage = {};
  BuildBundle(&bundle_storage);
  const loom_low_descriptor_set_t* descriptor_set =
      SelectDescriptorSet(&bundle_storage.bundle);

  const loom_low_allocation_fixed_value_t* fixed_values = nullptr;
  iree_host_size_t fixed_value_count = 0;
  IREE_ASSERT_OK(loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
      module_, function_op, descriptor_set, &fixed_values, &fixed_value_count,
      &arena_));
  loom_low_packetization_options_t packetization_options = {
      .descriptor_registry = &target_registry_.registry,
      .allocation_fixed_values = fixed_values,
      .allocation_fixed_value_count = fixed_value_count,
  };
  loom_low_packetization_t packetization = {};
  IREE_ASSERT_OK(loom_low_packetize_function(
      module_, function_op, &packetization_options, &arena_, &packetization));

  EXPECT_EQ(CountLowDescriptorOpsWithOpcode("amdgpu.s_mov_b32"), 0);
  EXPECT_EQ(CountLowDescriptorOpsWithOpcode("amdgpu.v_mov_b32"), 2);
  EXPECT_GT(packetization.schedule.scheduled_node_count, 0u);
}

TEST_F(AmdgpuHalResourceMaterializationTest,
       LeavesUnusedResourceAbiWithoutLiveIn) {
  BuildModule(
      "low.func.def target(@gfx_target) @loom_kernel() {\n"
      "  low.return\n"
      "}\n",
      "");

  loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_ir_bundle_storage_t bundle_storage = {};
  BuildBundle(&bundle_storage);
  const loom_low_descriptor_set_t* descriptor_set =
      SelectDescriptorSet(&bundle_storage.bundle);

  loom_amdgpu_hal_resource_materialization_result_t result = {};
  IREE_ASSERT_OK(loom_amdgpu_hal_resource_materialize(
      module_, function_op, &bundle_storage.bundle, descriptor_set, &result,
      &arena_));
  EXPECT_EQ(result.abi_layout.function_op, function_op);
  EXPECT_EQ(result.abi_layout.resource_count, 0u);
  EXPECT_EQ(result.materialized_resource_count, 0u);
  EXPECT_FALSE(result.inserted_kernarg_segment_ptr_live_in);
  EXPECT_EQ(CountLiveInsWithSource(
                LOOM_AMDGPU_HAL_KERNEL_ABI_KERNARG_SEGMENT_PTR_SOURCE),
            0);

  const loom_low_allocation_fixed_value_t* fixed_values = nullptr;
  iree_host_size_t fixed_value_count = 0;
  IREE_ASSERT_OK(loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
      module_, function_op, descriptor_set, &fixed_values, &fixed_value_count,
      &arena_));
  EXPECT_EQ(fixed_value_count, 0u);
  EXPECT_EQ(fixed_values, nullptr);
}

TEST_F(AmdgpuHalResourceMaterializationTest, FixesWorkitemIdXLiveInToVgprZero) {
  BuildModule(
      "low.func.def target(@gfx_target) @loom_kernel() {\n"
      "  %tid = low.live_in<" LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_X_SOURCE
      "> : reg<amdgpu.vgpr>\n"
      "  low.return\n"
      "}\n",
      "");

  loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_ir_bundle_storage_t bundle_storage = {};
  BuildBundle(&bundle_storage);
  const loom_low_descriptor_set_t* descriptor_set =
      SelectDescriptorSet(&bundle_storage.bundle);

  const loom_low_allocation_fixed_value_t* fixed_values = nullptr;
  iree_host_size_t fixed_value_count = 0;
  IREE_ASSERT_OK(loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
      module_, function_op, descriptor_set, &fixed_values, &fixed_value_count,
      &arena_));
  ASSERT_EQ(fixed_value_count, 1u);
  EXPECT_EQ(fixed_values[0].location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  EXPECT_EQ(fixed_values[0].location_base, 0u);
  EXPECT_EQ(fixed_values[0].location_count, 1u);

  VerifyModule();
}

TEST_F(AmdgpuHalResourceMaterializationTest,
       FixesWorkitemIdYZLiveInsToVgprOneAndTwo) {
  BuildModule(
      "low.func.def target(@gfx_target) @loom_kernel() {\n"
      "  %tid_y = low.live_in<" LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Y_SOURCE
      "> : reg<amdgpu.vgpr>\n"
      "  %tid_z = low.live_in<" LOOM_AMDGPU_HAL_KERNEL_ABI_WORKITEM_ID_Z_SOURCE
      "> : reg<amdgpu.vgpr>\n"
      "  low.return\n"
      "}\n",
      "");

  loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_ir_bundle_storage_t bundle_storage = {};
  BuildBundle(&bundle_storage);
  const loom_low_descriptor_set_t* descriptor_set =
      SelectDescriptorSet(&bundle_storage.bundle);

  const loom_low_allocation_fixed_value_t* fixed_values = nullptr;
  iree_host_size_t fixed_value_count = 0;
  IREE_ASSERT_OK(loom_amdgpu_hal_kernel_abi_fixed_values_from_low(
      module_, function_op, descriptor_set, &fixed_values, &fixed_value_count,
      &arena_));
  ASSERT_EQ(fixed_value_count, 2u);
  EXPECT_EQ(fixed_values[0].location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  EXPECT_EQ(fixed_values[0].location_base, 1u);
  EXPECT_EQ(fixed_values[0].location_count, 1u);
  EXPECT_EQ(fixed_values[1].location_kind,
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER);
  EXPECT_EQ(fixed_values[1].location_base, 2u);
  EXPECT_EQ(fixed_values[1].location_count, 1u);

  VerifyModule();
}

TEST_F(AmdgpuHalResourceMaterializationTest, RejectsUnsupportedResourceShape) {
  BuildModule(
      "low.func.def target(@gfx_target) @loom_kernel() {\n"
      "  %binding = low.resource<vm_state> {index = 0, semantic_type = "
      "hal.buffer} : reg<amdgpu.sgpr x4>\n"
      "  low.return\n"
      "}\n",
      "");

  loom_op_t* function_op = FindFirstLowFunction();
  ASSERT_NE(function_op, nullptr);
  loom_target_ir_bundle_storage_t bundle_storage = {};
  BuildBundle(&bundle_storage);
  const loom_low_descriptor_set_t* descriptor_set =
      SelectDescriptorSet(&bundle_storage.bundle);
  loom_amdgpu_hal_resource_materialization_result_t result = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_amdgpu_hal_resource_materialize(
                            module_, function_op, &bundle_storage.bundle,
                            descriptor_set, &result, &arena_));
}

}  // namespace
}  // namespace loom
