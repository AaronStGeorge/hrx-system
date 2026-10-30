// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/abi/feedback.h"

#include "iree/testing/gtest.h"

namespace {

TEST(AmdgpuFeedbackAbiTest, PacketLengthRoundsUpToAbiAlignment) {
  EXPECT_EQ(loom_amdgpu_feedback_packet_length(0), 64u);
  EXPECT_EQ(loom_amdgpu_feedback_packet_length(1), 128u);
  EXPECT_EQ(loom_amdgpu_feedback_packet_length(63), 128u);
  EXPECT_EQ(loom_amdgpu_feedback_packet_length(64), 128u);
  EXPECT_EQ(loom_amdgpu_feedback_packet_length(65), 192u);
  EXPECT_EQ(loom_amdgpu_feedback_packet_length(
                LOOM_AMDGPU_FEEDBACK_PACKET_MAX_PAYLOAD_LENGTH),
            64u + 16u * 1024u);
}

}  // namespace
