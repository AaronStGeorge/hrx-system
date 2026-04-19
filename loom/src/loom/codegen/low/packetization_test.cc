// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packetization.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/testing/context.h"

namespace loom {
namespace {

class LowPacketizationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
    loom_test_low_descriptor_registry_initialize(&target_registry_);
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
                        IREE_SV("low_packetization_test.loom"), &context_,
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

  void ParseAndVerify(const char* body) {
    std::string source =
        "target.snapshot @test_snapshot {codegen_format = low_native, "
        "target_triple = \"test-low\", data_layout = \"\", artifact_format = "
        "elf, target_cpu = \"test-low\", target_features = \"\", "
        "default_pointer_bitwidth = 64, index_bitwidth = 64, "
        "offset_bitwidth = 64, memory_space_generic = 0, "
        "memory_space_global = 0, memory_space_workgroup = 0, "
        "memory_space_constant = 0, memory_space_private = 0, "
        "memory_space_host = 0, memory_space_descriptor = 0}\n"
        "target.export @test_export {export_symbol = \"packetized\", abi = "
        "object_function, linkage = default, hal_binding_alignment = 0, "
        "hal_workgroup_size_x = 0, hal_workgroup_size_y = 0, "
        "hal_workgroup_size_z = 0, hal_flat_workgroup_size_min = 0, "
        "hal_flat_workgroup_size_max = 0, hal_buffer_resource_flags = 0}\n"
        "target.config @test_config {contract_set_key = \"test.low.core\", "
        "contract_feature_bits = 0}\n"
        "target.bundle @test_target {snapshot = @test_snapshot, export_plan = "
        "@test_export, config = @test_config}\n";
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
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_target_low_descriptor_registry_t target_registry_ = {};
};

TEST_F(LowPacketizationTest, BuildsMatchingScheduleAndAllocationSidecars) {
  ParseAndVerify(
      "low.func.def target(@test_target) @packetized(%lhs : reg<test.i32>, "
      "%rhs : reg<test.i32>) -> (reg<test.i32>) {\n"
      "  %c7 = low.const<test.const.i32> {i32_value = 7} : reg<test.i32>\n"
      "  %sum = low.op<test.add.i32>(%lhs, %c7) : "
      "(reg<test.i32>, reg<test.i32>) -> reg<test.i32>\n"
      "  %cmp = low.op<test.cmp.eq.i32>(%sum, %rhs) : "
      "(reg<test.i32>, reg<test.i32>) -> reg<test.i32>\n"
      "  low.return %cmp : reg<test.i32>\n"
      "}\n");
  const loom_op_t* low_function = FindFirstLowFunction(module_);
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  loom_low_packetization_options_t options = {
      .descriptor_registry = &target_registry_.registry,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE,
  };
  IREE_ASSERT_OK(loom_low_packetize_function(module_, low_function, &options,
                                             &sidecar_arena, &packetization));

  EXPECT_EQ(packetization.schedule.module, module_);
  EXPECT_EQ(packetization.allocation.module, module_);
  EXPECT_EQ(packetization.schedule.function_op, low_function);
  EXPECT_EQ(packetization.allocation.function_op, low_function);
  EXPECT_EQ(packetization.schedule.target.descriptor_set,
            packetization.allocation.target.descriptor_set);
  EXPECT_EQ(packetization.schedule.scheduled_node_count, 4u);
  EXPECT_GT(packetization.allocation.assignment_count, 0u);
  EXPECT_GT(packetization.schedule.pressure_step_count, 0u);
  IREE_EXPECT_OK(loom_low_packet_validate_sidecars(&packetization.schedule,
                                                   &packetization.allocation));

  loom_low_packet_view_t first_packet = {};
  IREE_ASSERT_OK(loom_low_packet_view_at(
      &packetization.schedule, &packetization.allocation, 0, &first_packet));
  EXPECT_NE(first_packet.node, nullptr);

  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(LowPacketizationTest, RejectsMissingDescriptorRegistry) {
  ParseAndVerify(
      "low.func.def target(@test_target) @packetized() {\n"
      "  low.return\n"
      "}\n");
  const loom_op_t* low_function = FindFirstLowFunction(module_);
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  loom_low_packetization_options_t options = {};
  iree_status_t status = loom_low_packetize_function(
      module_, low_function, &options, &sidecar_arena, &packetization);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_arena_deinitialize(&sidecar_arena);
}

TEST_F(LowPacketizationTest, RejectsNonLowFunction) {
  ParseAndVerify(
      "func.def @ordinary() {\n"
      "}\n"
      "low.func.def target(@test_target) @packetized() {\n"
      "  low.return\n"
      "}\n");
  const loom_op_t* ordinary_function = loom_module_block(module_)->first_op;
  ASSERT_NE(ordinary_function, nullptr);

  iree_arena_allocator_t sidecar_arena;
  iree_arena_initialize(&block_pool_, &sidecar_arena);
  loom_low_packetization_t packetization = {};
  loom_low_packetization_options_t options = {
      .descriptor_registry = &target_registry_.registry,
  };
  iree_status_t status = loom_low_packetize_function(
      module_, ordinary_function, &options, &sidecar_arena, &packetization);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_arena_deinitialize(&sidecar_arena);
}

}  // namespace
}  // namespace loom
