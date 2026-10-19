// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/input_ring.h"

#include "iree/testing/gtest.h"

namespace {

constexpr uint64_t kDefaultTargetBytes = 32 * 1024 * 1024;

TEST(InputRingTest, LargeStorageSetIsNotDuplicated) {
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/0, kDefaultTargetBytes,
                /*materialized_storage_bytes=*/110729216,
                /*input_uses_per_batch=*/64),
            1u);
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/0, kDefaultTargetBytes,
                /*materialized_storage_bytes=*/
                UINT64_C(7) * 1024 * 1024 * 1024,
                /*input_uses_per_batch=*/64),
            1u);
}

TEST(InputRingTest, AutomaticCountApproachesByteTargetWithinBatch) {
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/0, kDefaultTargetBytes,
                /*materialized_storage_bytes=*/1024 * 1024,
                /*input_uses_per_batch=*/64),
            32u);
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/0, kDefaultTargetBytes,
                /*materialized_storage_bytes=*/1,
                /*input_uses_per_batch=*/64),
            64u);
}

TEST(InputRingTest, DisabledOrEmptyAutomaticRingUsesOneSet) {
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/0, /*requested_min_bytes=*/0,
                /*materialized_storage_bytes=*/1024,
                /*input_uses_per_batch=*/64),
            1u);
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/0, kDefaultTargetBytes,
                /*materialized_storage_bytes=*/0,
                /*input_uses_per_batch=*/64),
            1u);
}

TEST(InputRingTest, ExplicitCountIsExact) {
  EXPECT_EQ(iree_benchmark_loom_input_ring_select_count(
                /*requested_count=*/96, kDefaultTargetBytes,
                /*materialized_storage_bytes=*/110729216,
                /*input_uses_per_batch=*/64),
            96u);
}

}  // namespace
