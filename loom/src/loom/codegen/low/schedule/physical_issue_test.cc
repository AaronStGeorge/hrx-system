// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/physical_issue.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/schedule/dependency_index.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

class PhysicalIssueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    schedule_.target.descriptor_set = loom_test_low_core_descriptor_set();
    schedule_.target.schedule_model =
        loom_low_descriptor_set_lookup_schedule_model_by_id(
            schedule_.target.descriptor_set, UINT64_C(0x235c515ddcc8fd92));
    ASSERT_NE(schedule_.target.schedule_model, nullptr);
    schedule_.nodes = nodes_;
    schedule_.node_count = IREE_ARRAYSIZE(nodes_);
    schedule_.scheduled_node_indices = order_;
    schedule_.scheduled_node_count = IREE_ARRAYSIZE(order_);
    schedule_.dependency_index = &index_;

    iree_arena_allocator_t scratch;
    iree_arena_initialize(&pool_, &scratch);
    loom_low_schedule_dependency_graph_t graph;
    loom_low_schedule_dependency_graph_initialize(&graph);
    const loom_low_schedule_dependency_t dependency = {
        .producer_node = 0,
        .consumer_node = 1,
        .minimum_issue_separation_cycles = -2,
        .producer_attachment_index = UINT16_MAX,
        .consumer_attachment_index = UINT16_MAX,
        .producer_event_id = UINT16_MAX,
        .consumer_event_id = UINT16_MAX,
        .value_operand_index = 0,
        .kind = LOOM_LOW_SCHEDULE_DEPENDENCY_SSA,
    };
    IREE_ASSERT_OK(loom_low_schedule_dependency_graph_append(&graph, dependency,
                                                             &scratch));
    uint32_t indegrees[2] = {};
    loom_low_schedule_dependency_detail_index_t details;
    IREE_ASSERT_OK(loom_low_schedule_dependency_index_initialize(
        &graph, IREE_ARRAYSIZE(nodes_), &scratch, &arena_, indegrees, &index_,
        &details));
    iree_arena_deinitialize(&scratch);
    iree_arena_block_pool_trim(&pool_);
    IREE_ASSERT_OK(
        loom_low_physical_issue_initialize(&schedule_, &arena_, &issue_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Shared pool with graph scratch released before physical admission.
  iree_arena_block_pool_t pool_ = {};
  // Retained index and physical-issue state lifetime.
  iree_arena_allocator_t arena_ = {};
  // Same-block producer and consumer in accepted semantic order.
  loom_low_schedule_node_t nodes_[2] = {};
  // Packet-to-node mapping for the accepted schedule.
  uint32_t order_[2] = {0, 1};
  // Real grouped index built independently of the retired raw graph.
  loom_low_schedule_dependency_index_t index_ = {};
  // Native schedule facts consumed by physical admission, with no IR owners.
  loom_low_schedule_table_t schedule_ = {};
  // Admission state under test.
  loom_low_physical_issue_t issue_ = {};
};

TEST_F(PhysicalIssueTest, ReusedRegisterWaitsForPreviousPhysicalRead) {
  const auto* descriptors = schedule_.target.descriptor_set;
  const auto* model = schedule_.target.schedule_model;
  const uint32_t read_descriptor = loom_low_descriptor_set_lookup_descriptor(
      descriptors, IREE_SV("test.event.read.late.phys"));
  const uint32_t write_descriptor = loom_low_descriptor_set_lookup_descriptor(
      descriptors, IREE_SV("test.event.write.fast.phys"));
  ASSERT_NE(read_descriptor, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  ASSERT_NE(write_descriptor, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  const uint16_t read_registers[] = {0};
  uint16_t write_registers[] = {0, 1};
  const loom_low_physical_instruction_t instructions[] = {
      loom_low_physical_instruction_make(descriptors, model, read_descriptor,
                                         read_registers),
      loom_low_physical_instruction_make(descriptors, model, write_descriptor,
                                         write_registers),
  };
  ASSERT_EQ(loom_low_physical_issue_find_earliest_issue_cycle(
                &issue_, &instructions[0], 1, 10),
            10u);
  IREE_ASSERT_OK(
      loom_low_physical_issue_commit(&issue_, &instructions[0], 1, 10));
  // The read consumes the old contents three cycles after issue. A new SSA
  // value may reuse its register only after that physical read completes.
  EXPECT_EQ(loom_low_physical_issue_find_earliest_issue_cycle(
                &issue_, &instructions[1], 1, 11),
            13u);
  EXPECT_FALSE(
      loom_low_physical_issue_group_fits(descriptors, model, instructions, 2));
  write_registers[0] = 1;
  EXPECT_EQ(loom_low_physical_issue_find_earliest_issue_cycle(
                &issue_, &instructions[1], 1, 11),
            11u);
  EXPECT_TRUE(
      loom_low_physical_issue_group_fits(descriptors, model, instructions, 2));

  const uint32_t serial_ordinal = loom_low_descriptor_set_lookup_descriptor(
      descriptors, IREE_SV("test.resource.serial.i32"));
  ASSERT_NE(serial_ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  const loom_low_descriptor_view_t* serial_view =
      loom_low_descriptor_set_descriptor_view_at(descriptors, serial_ordinal);
  const loom_low_schedule_class_t* serial_class =
      loom_low_schedule_model_schedule_class(model, serial_view);
  const loom_low_schedule_class_t* explicit_resources[] = {serial_class,
                                                           serial_class};
  EXPECT_FALSE(loom_low_physical_issue_group_fits_with_resource_classes(
      descriptors, model, instructions, 2, explicit_resources,
      IREE_ARRAYSIZE(explicit_resources)));

  loom_low_physical_issue_t resource_issue = {};
  IREE_ASSERT_OK(
      loom_low_physical_issue_initialize(&schedule_, &arena_, &resource_issue));
  const loom_low_schedule_class_t* one_resource[] = {serial_class};
  EXPECT_EQ(
      loom_low_physical_issue_find_earliest_issue_cycle_with_resource_classes(
          &resource_issue, instructions, 1, one_resource,
          IREE_ARRAYSIZE(one_resource), 0),
      0u);
  IREE_ASSERT_OK(loom_low_physical_issue_commit_with_resource_classes(
      &resource_issue, instructions, 1, one_resource,
      IREE_ARRAYSIZE(one_resource), 0));
  EXPECT_EQ(
      loom_low_physical_issue_find_earliest_issue_cycle_with_resource_classes(
          &resource_issue, instructions, 1, one_resource,
          IREE_ARRAYSIZE(one_resource), 0),
      4u);
}

TEST_F(PhysicalIssueTest, RetainedSourceOrderSurvivesGraphScratchRelease) {
  // Hardware may admit a consumer before the producer's issue cycle, but the
  // semantic source dependency still requires producer-before-consumer order.
  loom_low_physical_issue_commit_source(&issue_, 0, 10);
  EXPECT_EQ(loom_low_physical_issue_source_ready_cycle(&issue_, 1), 10u);
  loom_low_physical_issue_reset_execution_state(&issue_);
  EXPECT_EQ(loom_low_physical_issue_source_ready_cycle(&issue_, 1), 10u);
}

}  // namespace
}  // namespace loom
