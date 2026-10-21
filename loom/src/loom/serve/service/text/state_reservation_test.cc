// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/service_test_fixture.h"
#include "loom/serve/service/text/source.h"
#include "loom/serve/service/text/state.h"

namespace {

// The same source bootstrap and native storage as model creation, without
// parameter activation or execution kernels. Existing checkpoint tests exercise
// the immediate wrapper through complete model inference.
class TextStateReservationTest
    : public loom_serve::testing::text::SourceTextServiceTest {
 protected:
  bool NeedsStandaloneModel() const override { return false; }

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(SourceTextServiceTest::SetUp());
    const auto options = ModelOptions();
    loom_serve_text_state_initialize(&options, host_allocator_, &state_);
    IREE_ASSERT_OK(loom_serve_text_source_initialize(
        loom_serve_host_target_environment(host_), &options, host_allocator_,
        &source_, &state_.storage));
    IREE_ASSERT_OK(loom_serve_device_create_execution(device_, &execution_));
    IREE_ASSERT_OK(
        loom_serve_text_state_allocate(&state_, device_, execution_));
    IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  }

  void TearDown() override {
    loom_serve_text_state_reservation_free(reservation_);
    loom_serve_text_state_retire(&state_, nullptr, nullptr);
    IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    loom_serve_text_state_await_retirement(&state_);
    IREE_EXPECT_OK(loom_serve_text_state_deinitialize(&state_));
    loom_serve_execution_release(execution_);
    loom_serve_text_source_deinitialize(&source_);
    SourceTextServiceTest::TearDown();
  }

  void Reserve(size_t index, size_t extent) {
    loom_serve_memory_admission_t admission;
    IREE_ASSERT_OK(loom_serve_text_state_row_submit_reserve(
        &state_.rows[index], LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, extent,
        nullptr, nullptr, &admission));
    ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
    IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
    loom_serve_text_state_row_complete_reserve(&state_.rows[index]);
  }

  // Owned source description outliving native state.
  loom_serve_text_source_t source_ = {};
  // Owned source-defined pooled state under test.
  loom_serve_text_state_t state_ = {};
  // Owned device timeline, shared by all rows of this state.
  loom_serve_execution_t* execution_ = nullptr;
  // Prepared continuation, cancelled by teardown if an assertion aborts.
  loom_serve_text_state_reservation_t* reservation_ = nullptr;
};

TEST_F(TextStateReservationTest, PreparationOwnsIdsWithoutSubmittingOrMapping) {
  auto* row = &state_.rows[0];
  const auto before = loom_serve_execution_frontier(execution_);
  const auto memory = state_.memory.statistics;
  const auto pages = state_.cache.pool.available;
  const auto recurrent = state_.recurrent.pool.available;
  IREE_ASSERT_OK(loom_serve_text_state_row_prepare_reserve(
      row, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 16, &reservation_));
  ASSERT_NE(reservation_, nullptr);
  EXPECT_EQ(state_.cache.pool.available, pages - 4);
  EXPECT_EQ(state_.recurrent.pool.available, recurrent - 1);
  EXPECT_EQ(row->block_count, 0u);
  EXPECT_EQ(row->reserved_extent, 0u);
  EXPECT_EQ(row->recurrent.slot, UINT32_MAX);
  EXPECT_EQ(state_.memory.statistics.committed_bytes, memory.committed_bytes);
  EXPECT_EQ(state_.memory.statistics.pending_bytes, memory.pending_bytes);
  const auto after = loom_serve_execution_frontier(execution_);
  for (size_t i = 0; i < IREE_ARRAYSIZE(before.values); ++i) {
    EXPECT_EQ(after.semaphores[i], before.semaphores[i]);
    EXPECT_EQ(after.values[i], before.values[i]);
  }

  iree_host_size_t range_count = 0;
  const auto* ranges =
      loom_serve_text_state_reservation_ranges(reservation_, &range_count);
  EXPECT_EQ(range_count != 0, state_.memory.pool != nullptr);
  for (size_t i = 0; i < range_count; ++i) {
    EXPECT_NE(ranges[i].buffer, nullptr);
    EXPECT_GT(ranges[i].length, 0u);
  }
  loom_serve_text_state_reservation_free(reservation_);
  reservation_ = nullptr;
  EXPECT_EQ(state_.cache.pool.available, pages);
  EXPECT_EQ(state_.recurrent.pool.available, recurrent);

  // Cancellation returns the plan's IDs in acquisition order; an immediate
  // request can use the full pool with no stale continuation ownership.
  ASSERT_NO_FATAL_FAILURE(Reserve(0, 96));
  EXPECT_EQ(row->reserved_extent, 96u);
  EXPECT_EQ(state_.cache.pool.available, pages - 24);
}

TEST_F(TextStateReservationTest,
       DeniedPreparedContinuationKeepsItsExactPlanThroughPeerProgress) {
  ASSERT_NO_FATAL_FAILURE(Reserve(0, 16));
  auto* row = &state_.rows[0];
  const std::vector<uint32_t> old_blocks(state_.cache.maps,
                                         state_.cache.maps + row->block_count);
  const auto pages = state_.cache.pool.available;
  const auto recurrent = state_.recurrent.pool.available;
  IREE_ASSERT_OK(loom_serve_text_state_row_prepare_reserve(
      row, LOOM_SERVE_TEXT_RESERVE_CONTINUE, nullptr, 32, &reservation_));
  ASSERT_NE(reservation_, nullptr);
  // The unconsumed private pages fund their replacement, not a second full
  // reservation. The prepared plan owns four additional pages only.
  EXPECT_EQ(state_.cache.pool.available, pages - 4);
  EXPECT_EQ(state_.recurrent.pool.available, recurrent);
  iree_host_size_t range_count = 0;
  const auto* ranges =
      loom_serve_text_state_reservation_ranges(reservation_, &range_count);
  std::vector<loom_serve_memory_range_t> planned;
  if (range_count) {
    planned.assign(ranges, ranges + range_count);
  }
  const auto deny = [](void* user_data, iree_host_size_t count,
                       const loom_serve_memory_range_t* ranges,
                       iree_hal_semaphore_t* readiness, uint64_t value,
                       loom_serve_memory_admission_t* out_admission,
                       loom_serve_memory_operation_t** out_operation) {
    *out_admission = LOOM_SERVE_MEMORY_ADMISSION_CAPACITY;
    *out_operation = nullptr;
    return iree_ok_status();
  };
  loom_serve_memory_admission_t admission;
  IREE_ASSERT_OK(loom_serve_text_state_reservation_submit(reservation_, deny,
                                                          nullptr, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_CAPACITY);
  EXPECT_EQ(row->reserved_extent, 16u);
  EXPECT_EQ(std::vector<uint32_t>(state_.cache.maps,
                                  state_.cache.maps + row->block_count),
            old_blocks);

  // Independent rows can acquire/submit/retire between planning and commit.
  // This does not replan the protected destination or steal its selected IDs.
  ASSERT_NO_FATAL_FAILURE(Reserve(1, 16));
  loom_serve_text_state_row_reset(&state_.rows[1]);
  IREE_ASSERT_OK(loom_serve_text_state_reservation_submit(reservation_, nullptr,
                                                          nullptr, &admission));
  ASSERT_EQ(admission, LOOM_SERVE_MEMORY_ADMISSION_AVAILABLE);
  iree_host_size_t submitted_count = 0;
  ranges =
      loom_serve_text_state_reservation_ranges(reservation_, &submitted_count);
  ASSERT_EQ(submitted_count, planned.size());
  for (size_t i = 0; i < submitted_count; ++i) {
    EXPECT_EQ(ranges[i].buffer, planned[i].buffer);
    EXPECT_EQ(ranges[i].offset, planned[i].offset);
    EXPECT_EQ(ranges[i].length, planned[i].length);
  }
  loom_serve_text_state_reservation_free(reservation_);
  reservation_ = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  loom_serve_text_state_row_complete_reserve(row);
  EXPECT_EQ(row->reserved_extent, 32u);
  EXPECT_EQ(row->block_count, 8u);
  EXPECT_EQ(state_.cache.pool.available, pages - 4);
  EXPECT_EQ(state_.recurrent.pool.available, recurrent);
  EXPECT_TRUE(
      std::equal(old_blocks.begin(), old_blocks.end(), state_.cache.maps));
}

}  // namespace
