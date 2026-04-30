// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/frame.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/target/test/low_registry.h"

namespace loom {
namespace {

class LowPacketizationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_LOW, loom_low_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
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
                        IREE_SV("low_frame_test.loom"), &context_, &block_pool_,
                        &parse_options, &module));
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

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  void ParseAndVerify(const char* body) {
    std::string source = "target.profile @test_target preset(\"test-low\")\n";
    source += body;
    module_ = ParseSource(source);
    ASSERT_NE(module_, nullptr);

    loom_low_verify_options_t verify_options = {
        .descriptor_registry = &target_registry_.registry,
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

TEST_F(LowPacketizationTest, BuildsMatchingScheduleAndAllocationTables) {
  ParseAndVerify(
      "low.func.def target(@test_target) @packetized(%lhs: reg<test.i32>, "
      "%rhs: reg<test.i32>) -> (reg<test.i32>) {\n"
      "  %c7 = low.const<test.const.i32> {i32_value = 7} : reg<test.i32>\n"
      "  %sum = low.op<test.add.i32>(%lhs, %c7) : "
      "(reg<test.i32>, reg<test.i32>) -> reg<test.i32>\n"
      "  %cmp = low.op<test.cmp.eq.i32>(%sum, %rhs) : "
      "(reg<test.i32>, reg<test.i32>) -> reg<test.i32>\n"
      "  low.return %cmp : reg<test.i32>\n"
      "}\n");
  loom_op_t* low_function = FindFirstLowFunction(module_);
  ASSERT_NE(low_function, nullptr);

  iree_arena_allocator_t table_arena;
  iree_arena_initialize(&block_pool_, &table_arena);
  loom_low_emission_frame_t frame = {};
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &target_registry_.registry,
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE,
  };
  IREE_ASSERT_OK(loom_low_emission_frame_build(module_, low_function, &options,
                                               &table_arena, &frame));

  EXPECT_EQ(frame.schedule.module, module_);
  EXPECT_EQ(frame.allocation.module, module_);
  EXPECT_EQ(frame.schedule.function_op, low_function);
  EXPECT_EQ(frame.allocation.function_op, low_function);
  EXPECT_EQ(frame.schedule.target.descriptor_set,
            frame.allocation.target.descriptor_set);
  EXPECT_EQ(frame.schedule.scheduled_node_count, 4u);
  EXPECT_GT(frame.allocation.assignment_count, 0u);
  EXPECT_GT(frame.schedule.pressure_step_count, 0u);
  IREE_EXPECT_OK(
      loom_low_packet_validate_tables(&frame.schedule, &frame.allocation));

  loom_low_packet_view_t first_packet = {};
  IREE_ASSERT_OK(loom_low_packet_view_at(&frame.schedule, &frame.allocation, 0,
                                         &first_packet));
  EXPECT_NE(first_packet.node, nullptr);

  iree_arena_deinitialize(&table_arena);
}

TEST_F(LowPacketizationTest, RejectsNonLowFunction) {
  ParseAndVerify(
      "func.def @ordinary() {\n"
      "}\n"
      "low.func.def target(@test_target) @packetized() {\n"
      "  low.return\n"
      "}\n");
  loom_op_t* ordinary_function = loom_module_block(module_)->first_op;
  ASSERT_NE(ordinary_function, nullptr);

  iree_arena_allocator_t table_arena;
  iree_arena_initialize(&block_pool_, &table_arena);
  loom_low_emission_frame_t frame = {};
  loom_low_emission_frame_options_t options = {
      .descriptor_registry = &target_registry_.registry,
  };
  iree_status_t status = loom_low_emission_frame_build(
      module_, ordinary_function, &options, &table_arena, &frame);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  iree_arena_deinitialize(&table_arena);
}

}  // namespace
}  // namespace loom
