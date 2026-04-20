// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/move_sequence.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

loom_low_move_location_t Location(uint32_t ordinal) {
  return loom_low_move_location_t{
      .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
      .location = ordinal,
  };
}

loom_low_move_t Move(uint32_t destination, uint32_t source) {
  return loom_low_move_t{
      .destination = Location(destination),
      .source = Location(source),
  };
}

std::string MoveString(const loom_low_move_location_t* destination,
                       const loom_low_move_location_t* source) {
  return std::to_string(destination->location) + "<-" +
         std::to_string(source->location);
}

iree_status_t RecordMove(void* user_data,
                         const loom_low_move_location_t* destination,
                         const loom_low_move_location_t* source) {
  auto* moves = static_cast<std::vector<std::string>*>(user_data);
  moves->push_back(MoveString(destination, source));
  return iree_ok_status();
}

std::vector<std::string> EmitMoves(loom_low_move_t* moves,
                                   iree_host_size_t move_count,
                                   const loom_low_move_location_t* temporary) {
  std::vector<std::string> emitted_moves;
  loom_low_move_sequence_options_t options = {
      .temporary_location = temporary,
      .emit_move =
          {
              .fn = RecordMove,
              .user_data = &emitted_moves,
          },
  };
  IREE_EXPECT_OK(loom_low_move_sequence_emit(moves, move_count, &options));
  return emitted_moves;
}

TEST(LowMoveSequenceTest, SkipsIdentityMoves) {
  loom_low_move_t moves[] = {
      Move(0, 0),
      Move(1, 1),
  };

  EXPECT_TRUE(EmitMoves(moves, IREE_ARRAYSIZE(moves), nullptr).empty());
}

TEST(LowMoveSequenceTest, EmitsIndependentMovesInInputOrder) {
  loom_low_move_t moves[] = {
      Move(4, 0),
      Move(5, 1),
  };

  EXPECT_THAT(EmitMoves(moves, IREE_ARRAYSIZE(moves), nullptr),
              ::testing::ElementsAre("4<-0", "5<-1"));
}

TEST(LowMoveSequenceTest, ReordersForwardClobberingShift) {
  loom_low_move_t moves[] = {
      Move(1, 0),
      Move(2, 1),
  };

  EXPECT_THAT(EmitMoves(moves, IREE_ARRAYSIZE(moves), nullptr),
              ::testing::ElementsAre("2<-1", "1<-0"));
}

TEST(LowMoveSequenceTest, KeepsBackwardShiftInInputOrder) {
  loom_low_move_t moves[] = {
      Move(0, 1),
      Move(1, 2),
  };

  EXPECT_THAT(EmitMoves(moves, IREE_ARRAYSIZE(moves), nullptr),
              ::testing::ElementsAre("0<-1", "1<-2"));
}

TEST(LowMoveSequenceTest, UsesTemporaryForCycle) {
  loom_low_move_t moves[] = {
      Move(0, 1),
      Move(1, 0),
  };
  const loom_low_move_location_t temporary = Location(9);

  EXPECT_THAT(EmitMoves(moves, IREE_ARRAYSIZE(moves), &temporary),
              ::testing::ElementsAre("9<-0", "0<-1", "1<-9"));
}

TEST(LowMoveSequenceTest, RejectsCycleWithoutTemporary) {
  loom_low_move_t moves[] = {
      Move(0, 1),
      Move(1, 0),
  };
  std::vector<std::string> emitted_moves;
  loom_low_move_sequence_options_t options = {
      .emit_move =
          {
              .fn = RecordMove,
              .user_data = &emitted_moves,
          },
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_low_move_sequence_emit(moves, IREE_ARRAYSIZE(moves), &options));
  EXPECT_TRUE(emitted_moves.empty());
}

TEST(LowMoveSequenceTest, RejectsDuplicateDestinations) {
  loom_low_move_t moves[] = {
      Move(0, 1),
      Move(0, 2),
  };
  std::vector<std::string> emitted_moves;
  loom_low_move_sequence_options_t options = {
      .emit_move =
          {
              .fn = RecordMove,
              .user_data = &emitted_moves,
          },
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_low_move_sequence_emit(moves, IREE_ARRAYSIZE(moves), &options));
  EXPECT_TRUE(emitted_moves.empty());
}

}  // namespace
}  // namespace loom
