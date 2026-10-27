// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/reasoning.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ReasoningTest, DistinguishesUnknownUnsupportedAndUnselectableReasoning) {
  loom_serve_reasoning_t reasoning = {};
  EXPECT_EQ(sizeof(reasoning), 3u);
  EXPECT_EQ(reasoning.support, LOOM_SERVE_REASONING_UNKNOWN);
  for (int32_t enabled : {0, 1}) {
    IREE_ASSERT_OK(loom_serve_reasoning_initialize(enabled, 0, 0, &reasoning));
    EXPECT_EQ(reasoning.support, enabled ? LOOM_SERVE_REASONING_SUPPORTED
                                         : LOOM_SERVE_REASONING_UNSUPPORTED);
    EXPECT_EQ(reasoning.efforts, 0);
    EXPECT_EQ(reasoning.default_effort, 0);
  }
}

TEST(ReasoningTest, EveryLadderRetainsItsOwnDefault) {
  const char* names[] = {"none", "minimal", "low", "medium",
                         "high", "xhigh",   "max"};
  for (int32_t effort = 0; effort < LOOM_SERVE_REASONING_EFFORT_COUNT;
       ++effort) {
    EXPECT_STREQ(loom_serve_reasoning_effort_name(
                     static_cast<loom_serve_reasoning_effort_t>(effort)),
                 names[effort]);
    for (int32_t mask = 2; mask < 128; ++mask) {
      loom_serve_reasoning_t reasoning = {};
      auto status =
          loom_serve_reasoning_initialize(1, mask, effort, &reasoning);
      if (mask & (1u << effort)) {
        IREE_ASSERT_OK(status);
        EXPECT_EQ(reasoning.support, LOOM_SERVE_REASONING_SUPPORTED);
        EXPECT_EQ(reasoning.efforts, mask);
        EXPECT_EQ(reasoning.default_effort, effort);
      } else {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
        EXPECT_EQ(reasoning.support, LOOM_SERVE_REASONING_UNKNOWN);
      }
    }
  }
}

TEST(ReasoningTest, RejectsMalformedSourceDeclarationsWithoutPublishingFacts) {
  const int32_t invalid[][3] = {{-1, 0, 0}, {2, 0, 0},   {0, 4, 2}, {0, 0, 2},
                                {1, -1, 0}, {1, 128, 0}, {1, 1, 0}, {1, 4, -1},
                                {1, 4, 7},  {1, 0, 2}};
  for (const auto& values : invalid) {
    loom_serve_reasoning_t reasoning = {LOOM_SERVE_REASONING_SUPPORTED, 45, 0};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_reasoning_initialize(
                              values[0], values[1], values[2], &reasoning));
    EXPECT_EQ(reasoning.support, LOOM_SERVE_REASONING_UNKNOWN);
    EXPECT_EQ(reasoning.efforts, 0);
    EXPECT_EQ(reasoning.default_effort, 0);
  }
}

}  // namespace
