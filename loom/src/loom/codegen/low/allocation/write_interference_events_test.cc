// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/write_interference_events.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/allocation/assignment.h"
#include "loom/ir/types.h"

namespace loom {
namespace {

class WriteInterferenceEventsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    iree_arena_initialize(&pool_, &scratch_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&scratch_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Blocks shared by the persistent results and disposable solver workspace.
  iree_arena_block_pool_t pool_;
  // Event accesses and immutable snapshots.
  iree_arena_allocator_t arena_;
  // Decoder and dataflow state, reset before querying snapshots.
  iree_arena_allocator_t scratch_;
};

TEST_F(WriteInterferenceEventsTest, PartialOverwriteReleasesOnlyWrittenUnits) {
  for (uint32_t overwritten = 0; overwritten < 2; ++overwritten) {
    SCOPED_TRACE(overwritten);
    const loom_low_write_value_t values[] = {
        {
            .origin = 0,
            .fixed_base = 0,
            .construction = {.width = 2, .retained_start = 0},
        },
        {
            .origin = 1,
            .fixed_base = overwritten,
            .construction = {.width = 1, .retained_start = UINT32_MAX},
        },
        {
            .origin = 2,
            .fixed_base = UINT32_MAX,
            .construction = {.width = 1, .retained_start = UINT32_MAX},
        },
    };
    loom_low_write_event_t points[9] = {};
    for (auto& point : points) {
      point.access = UINT32_MAX;
    }
    loom_low_write_events_t events = {
        .points = points,
        .point_count = IREE_ARRAYSIZE(points),
    };
    IREE_ASSERT_OK(
        loom_low_write_events_append(&events, 1,
                                     {
                                         .value = 0,
                                         .source = LOOM_VALUE_ORDINAL_INVALID,
                                         .count = 2,
                                         .flags = LOOM_LOW_WRITE_ACCESS_RETAIN,
                                     },
                                     &arena_));
    // A memory result overwrites one known unit, without a hazardous ALU write.
    IREE_ASSERT_OK(
        loom_low_write_events_append(&events, 4,
                                     {
                                         .value = 1,
                                         .source = LOOM_VALUE_ORDINAL_INVALID,
                                         .count = 1,
                                         .flags = LOOM_LOW_WRITE_ACCESS_WRITE,
                                     },
                                     &arena_));
    IREE_ASSERT_OK(loom_low_write_events_append(
        &events, 7,
        {
            .value = 2,
            .source = LOOM_VALUE_ORDINAL_INVALID,
            .count = 1,
            .flags =
                LOOM_LOW_WRITE_ACCESS_WRITE | LOOM_LOW_WRITE_ACCESS_INTERFERE,
        },
        &arena_));
    loom_low_write_flow_block_t block = {.begin = 1, .end = 9};
    uint32_t worklist;
    const loom_low_write_flow_t flow = {
        .blocks = &block,
        .block_count = 1,
        .worklist = &worklist,
    };
    iree_host_size_t constraint_count = 0;
    IREE_ASSERT_OK(loom_low_write_events_solve(
        &events, values, IREE_ARRAYSIZE(values), 2, &flow, &arena_, &scratch_,
        &constraint_count));
    iree_arena_reset(&scratch_);

    EXPECT_EQ(constraint_count, 1u);
    EXPECT_EQ(points[1].retained.count, 0u);
    ASSERT_EQ(points[4].retained.count, 1u);
    const auto& before = events.retained[points[4].retained.start];
    EXPECT_EQ(before.value, 0u);
    EXPECT_EQ(before.offset, 0u);
    EXPECT_EQ(before.count, 2u);
    ASSERT_EQ(points[5].retained.count, 1u);
    const auto& after = events.retained[points[5].retained.start];
    EXPECT_EQ(after.value, 0u);
    EXPECT_EQ(after.offset, 1u - overwritten);
    EXPECT_EQ(after.count, 1u);
    // Vector-only gaps and writes at unknown locations reuse the snapshot.
    EXPECT_EQ(points[2].retained.start, points[4].retained.start);
    EXPECT_EQ(points[5].retained.start, points[8].retained.start);
    EXPECT_EQ(events.retained_count, 2u);

    const loom_low_allocation_assignment_t assignments[] = {
        {
            .unit_count = 1,
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = overwritten,
            .location_count = 1,
        },
        {
            .unit_count = 2,
            .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
            .location_base = 0,
            .location_count = 2,
        },
    };
    const uint32_t assignment_indices[] = {1, 0, UINT32_MAX};
    for (uint32_t location = 0; location < 3; ++location) {
      EXPECT_EQ(loom_low_write_events_temporary_conflicts(
                    &events, assignments, assignment_indices, 4, location),
                location < 2);
      EXPECT_EQ(loom_low_write_events_temporary_conflicts(
                    &events, assignments, assignment_indices, 5, location),
                location == 1u - overwritten);
    }
  }
}

}  // namespace
}  // namespace loom
