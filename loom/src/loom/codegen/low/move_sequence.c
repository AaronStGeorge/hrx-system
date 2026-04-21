// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/move_sequence.h"

#include <inttypes.h>
#include <string.h>

bool loom_low_move_locations_equal(const loom_low_move_location_t* lhs,
                                   const loom_low_move_location_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         lhs->location == rhs->location &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id;
}

iree_status_t loom_low_move_location_from_assignment_unit(
    const loom_low_allocation_assignment_t* assignment, uint32_t unit_index,
    loom_low_move_location_t* out_location) {
  IREE_ASSERT_ARGUMENT(out_location);
  *out_location = (loom_low_move_location_t){0};
  if (assignment == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "move location assignment is required");
  }
  if (unit_index >= assignment->location_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "move location unit %" PRIu32
                            " exceeds assignment range of %" PRIu32,
                            unit_index, assignment->location_count);
  }
  if (assignment->location_base > UINT32_MAX - unit_index) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "move location unit exceeds uint32_t");
  }
  *out_location = (loom_low_move_location_t){
      .location_kind = assignment->location_kind,
      .value_class = assignment->value_class,
      .descriptor_reg_class_id = assignment->descriptor_reg_class_id,
      .location = assignment->location_base + unit_index,
  };
  return iree_ok_status();
}

static void loom_low_move_sequence_remove_move(loom_low_move_t* moves,
                                               iree_host_size_t* move_count,
                                               iree_host_size_t move_index) {
  if (move_index + 1 < *move_count) {
    memmove(&moves[move_index], &moves[move_index + 1],
            (*move_count - move_index - 1) * sizeof(*moves));
  }
  --*move_count;
}

static bool loom_low_move_sequence_destination_is_source(
    const loom_low_move_t* moves, iree_host_size_t move_count,
    iree_host_size_t move_index) {
  const loom_low_move_location_t* destination = &moves[move_index].destination;
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    if (loom_low_move_locations_equal(destination, &moves[i].source)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_move_sequence_find_move_by_destination(
    const loom_low_move_t* moves, iree_host_size_t move_count,
    const loom_low_move_location_t* destination,
    iree_host_size_t* out_move_index) {
  IREE_ASSERT_ARGUMENT(out_move_index);
  *out_move_index = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    if (loom_low_move_locations_equal(destination, &moves[i].destination)) {
      *out_move_index = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "parallel move cycle is malformed");
}

static iree_status_t loom_low_move_sequence_validate(
    const loom_low_move_t* moves, iree_host_size_t move_count,
    const loom_low_move_sequence_options_t* options) {
  if (move_count != 0 && moves == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parallel move rows are required");
  }
  if (options == NULL || options->emit_move.fn == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parallel move emitter is required");
  }
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    if (options->temporary_location != NULL &&
        (loom_low_move_locations_equal(options->temporary_location,
                                       &moves[i].destination) ||
         loom_low_move_locations_equal(options->temporary_location,
                                       &moves[i].source))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parallel move temporary location aliases move storage");
    }
    for (iree_host_size_t j = i + 1; j < move_count; ++j) {
      if (!loom_low_move_locations_equal(&moves[i].destination,
                                         &moves[j].destination)) {
        continue;
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parallel move rows contain duplicate destinations");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_move_sequence_emit_one(
    const loom_low_move_sequence_options_t* options,
    const loom_low_move_location_t* destination,
    const loom_low_move_location_t* source) {
  if (loom_low_move_locations_equal(destination, source)) {
    return iree_ok_status();
  }
  return options->emit_move.fn(options->emit_move.user_data, destination,
                               source);
}

static iree_status_t loom_low_move_sequence_emit_cycle(
    loom_low_move_t* moves, iree_host_size_t* move_count,
    const loom_low_move_sequence_options_t* options) {
  if (options->temporary_location == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "parallel move cycle requires a temporary "
                            "location");
  }

  const loom_low_move_location_t saved_location = moves[0].destination;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_emit_one(
      options, options->temporary_location, &saved_location));

  loom_low_move_location_t destination = moves[0].destination;
  loom_low_move_location_t source = moves[0].source;
  for (;;) {
    iree_host_size_t move_index = IREE_HOST_SIZE_MAX;
    IREE_RETURN_IF_ERROR(loom_low_move_sequence_find_move_by_destination(
        moves, *move_count, &destination, &move_index));
    if (loom_low_move_locations_equal(&source, &saved_location)) {
      IREE_RETURN_IF_ERROR(loom_low_move_sequence_emit_one(
          options, &destination, options->temporary_location));
      loom_low_move_sequence_remove_move(moves, move_count, move_index);
      return iree_ok_status();
    }

    IREE_RETURN_IF_ERROR(
        loom_low_move_sequence_emit_one(options, &destination, &source));
    loom_low_move_sequence_remove_move(moves, move_count, move_index);
    destination = source;
    IREE_RETURN_IF_ERROR(loom_low_move_sequence_find_move_by_destination(
        moves, *move_count, &destination, &move_index));
    source = moves[move_index].source;
  }
}

iree_status_t loom_low_move_sequence_emit(
    loom_low_move_t* moves, iree_host_size_t move_count,
    const loom_low_move_sequence_options_t* options) {
  IREE_RETURN_IF_ERROR(
      loom_low_move_sequence_validate(moves, move_count, options));

  iree_host_size_t i = 0;
  while (i < move_count) {
    if (loom_low_move_locations_equal(&moves[i].destination,
                                      &moves[i].source)) {
      loom_low_move_sequence_remove_move(moves, &move_count, i);
      continue;
    }
    ++i;
  }

  while (move_count != 0) {
    bool emitted_safe_move = false;
    for (iree_host_size_t move_index = 0; move_index < move_count;
         ++move_index) {
      if (loom_low_move_sequence_destination_is_source(moves, move_count,
                                                       move_index)) {
        continue;
      }
      const loom_low_move_t move = moves[move_index];
      IREE_RETURN_IF_ERROR(loom_low_move_sequence_emit_one(
          options, &move.destination, &move.source));
      loom_low_move_sequence_remove_move(moves, &move_count, move_index);
      emitted_safe_move = true;
      break;
    }
    if (emitted_safe_move) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_move_sequence_emit_cycle(moves, &move_count, options));
  }

  return iree_ok_status();
}
