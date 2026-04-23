// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/lower.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/packetization.h"
#include "loom/codegen/low/source_selection.h"
#include "loom/codegen/low/source_workload.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/test/low_registry.h"
#include "loom/target/test/lower.h"
#include "loom/testing/context.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

struct ModuleDeleter {
  void operator()(loom_module_t* module) const { loom_module_free(module); }
};

using ModulePtr = std::unique_ptr<loom_module_t, ModuleDeleter>;

static uint32_t CountLowDescriptorOps(const loom_op_t* low_func_op) {
  uint32_t count = 0;
  const loom_region_t* body = loom_low_func_def_body(low_func_op);
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (loom_low_op_isa(op) || loom_low_const_isa(op)) {
        ++count;
      }
    }
  }
  return count;
}

class SourceLoweringStressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(loom_testing_context_initialize_all(iree_allocator_system(),
                                                       &context_));
    loom_test_low_descriptor_registry_initialize(&descriptor_registry_);
    loom_test_low_lower_policy_registry_initialize(&policy_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t VerifySourceModule(loom_module_t* module) {
    loom_verify_options_t options = {};
    loom_verify_result_t result = {};
    IREE_RETURN_IF_ERROR(loom_verify_module(module, &options, &result));
    if (result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "generated source failed verification");
    }
    return iree_ok_status();
  }

  iree_status_t VerifyLowModule(loom_module_t* module) {
    const loom_low_verify_options_t options = {
        .flags = LOOM_LOW_VERIFY_FLAG_VERIFY_DESCRIPTOR_REGISTRY,
        .descriptor_registry = &descriptor_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .max_errors = 20,
    };
    loom_low_verify_result_t result = {};
    IREE_RETURN_IF_ERROR(loom_low_verify_module(module, &options, &result));
    if (result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "generated low function failed verification");
    }
    return iree_ok_status();
  }

  iree_status_t LowerGeneratedModule(loom_module_t* module,
                                     loom_symbol_ref_t func_ref,
                                     iree_arena_allocator_t* arena,
                                     loom_low_lower_result_t* out_result) {
    const iree_string_view_t func_name =
        module->strings
            .entries[module->symbols.entries[func_ref.symbol_id].name_id];
    const loom_low_source_selection_options_t selection_options = {
        .func_symbol_name = func_name,
        .descriptor_registry = &descriptor_registry_.registry,
        .policy_registry = &policy_registry_,
    };
    loom_low_source_selection_t selection = {};
    IREE_RETURN_IF_ERROR(loom_low_select_source_func(module, &selection_options,
                                                     arena, &selection));
    const loom_low_lower_options_t lower_options = {
        .target_ref = selection.target_ref,
        .bundle = selection.target_bundle,
        .descriptor_registry = &descriptor_registry_.registry,
        .descriptor_requirements =
            LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION,
        .policy = selection.policy,
        .max_errors = 20,
    };
    return loom_low_lower_function(module, selection.func, &lower_options,
                                   out_result);
  }

  iree_status_t PacketizeGeneratedLowFunction(
      loom_module_t* module, loom_op_t* low_func_op,
      iree_arena_allocator_t* arena,
      loom_low_packetization_t* out_packetization) {
    const loom_low_packetization_options_t options = {
        .descriptor_registry = &descriptor_registry_.registry,
        .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE,
    };
    IREE_RETURN_IF_ERROR(loom_low_packetize_function(
        module, low_func_op, &options, arena, out_packetization));
    return loom_low_packet_validate_sidecars(&out_packetization->schedule,
                                             &out_packetization->allocation);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t descriptor_registry_ = {};
  loom_low_lower_policy_registry_t policy_registry_ = {};
};

TEST_F(SourceLoweringStressTest, GeneratedSupportedSourceLowersAndPacketizes) {
  loom_low_source_workload_counts_t aggregate_source_counts = {};
  uint32_t aggregate_low_descriptor_ops = 0;
  uint32_t aggregate_schedule_nodes = 0;
  uint32_t aggregate_assignments = 0;

  for (uint64_t seed = 0; seed < 16; ++seed) {
    SCOPED_TRACE(::testing::Message() << "seed=" << seed);
    loom_module_t* module_raw = nullptr;
    loom_symbol_ref_t func_ref = loom_symbol_ref_null();
    loom_low_source_workload_config_t workload_config =
        loom_low_source_workload_config_make(IREE_SV("test-low"), 1);
    IREE_ASSERT_OK(loom_low_source_workload_generate_seeded_module(
        seed, &workload_config, &context_, &block_pool_, &module_raw,
        &func_ref));
    ModulePtr module(module_raw);

    IREE_ASSERT_OK(VerifySourceModule(module.get()));
    ASSERT_LT(func_ref.symbol_id, module->symbols.count);
    const loom_symbol_t* func_symbol =
        &module->symbols.entries[func_ref.symbol_id];
    ASSERT_TRUE(loom_func_def_isa(func_symbol->defining_op));
    loom_low_source_workload_counts_t source_counts = {};
    loom_low_source_workload_count_func_ops(func_symbol->defining_op,
                                            &source_counts);
    aggregate_source_counts.scalar_integer_op_count +=
        source_counts.scalar_integer_op_count;
    aggregate_source_counts.scalar_constant_count +=
        source_counts.scalar_constant_count;
    aggregate_source_counts.vector_integer_op_count +=
        source_counts.vector_integer_op_count;
    aggregate_source_counts.index_madd_op_count +=
        source_counts.index_madd_op_count;

    iree_arena_allocator_t lowering_arena;
    iree_arena_initialize(&block_pool_, &lowering_arena);
    loom_low_lower_result_t lower_result = {};
    IREE_ASSERT_OK(LowerGeneratedModule(module.get(), func_ref, &lowering_arena,
                                        &lower_result));
    EXPECT_EQ(lower_result.error_count, 0u);
    EXPECT_EQ(lower_result.remark_count, 0u);
    ASSERT_NE(lower_result.low_func_op, nullptr);
    EXPECT_TRUE(loom_symbol_ref_is_valid(lower_result.low_func_ref));
    IREE_ASSERT_OK(VerifyLowModule(module.get()));

    aggregate_low_descriptor_ops +=
        CountLowDescriptorOps(lower_result.low_func_op);

    iree_arena_allocator_t packet_arena;
    iree_arena_initialize(&block_pool_, &packet_arena);
    loom_low_packetization_t packetization = {};
    IREE_ASSERT_OK(PacketizeGeneratedLowFunction(
        module.get(), lower_result.low_func_op, &packet_arena, &packetization));
    EXPECT_GT(packetization.schedule.scheduled_node_count, 0u);
    EXPECT_GT(packetization.allocation.assignment_count, 0u);
    aggregate_schedule_nodes +=
        (uint32_t)packetization.schedule.scheduled_node_count;
    aggregate_assignments +=
        (uint32_t)packetization.allocation.assignment_count;
    iree_arena_deinitialize(&packet_arena);
    iree_arena_deinitialize(&lowering_arena);
  }

  EXPECT_GT(aggregate_source_counts.scalar_integer_op_count, 0u);
  EXPECT_GT(aggregate_source_counts.scalar_constant_count, 0u);
  EXPECT_GT(aggregate_source_counts.vector_integer_op_count, 0u);
  EXPECT_GT(aggregate_source_counts.index_madd_op_count, 0u);
  EXPECT_GT(aggregate_low_descriptor_ops, 0u);
  EXPECT_GT(aggregate_schedule_nodes, 0u);
  EXPECT_GT(aggregate_assignments, 0u);
}

}  // namespace
}  // namespace loom
