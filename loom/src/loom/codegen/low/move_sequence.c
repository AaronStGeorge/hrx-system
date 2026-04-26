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

static bool loom_low_move_locations_share_storage_class(
    const loom_low_move_location_t* lhs, const loom_low_move_location_t* rhs) {
  return lhs->location_kind == rhs->location_kind &&
         lhs->descriptor_reg_class_id == rhs->descriptor_reg_class_id &&
         loom_liveness_value_class_equal(lhs->value_class, rhs->value_class);
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

static void loom_low_move_location_from_edge_copy_temporary(
    const loom_low_allocation_edge_copy_temporary_t* temporary,
    loom_low_move_location_t* out_location) {
  IREE_ASSERT_ARGUMENT(temporary);
  IREE_ASSERT_ARGUMENT(out_location);
  *out_location = (loom_low_move_location_t){
      .location_kind = temporary->location_kind,
      .value_class = temporary->value_class,
      .descriptor_reg_class_id = temporary->descriptor_reg_class_id,
      .location = temporary->location,
  };
}

iree_status_t loom_low_move_sequence_count_edge_copy_units(
    const loom_low_allocation_sidecar_t* allocation,
    const loom_low_allocation_edge_copy_group_t* group,
    iree_host_size_t* out_move_count) {
  IREE_ASSERT_ARGUMENT(allocation);
  IREE_ASSERT_ARGUMENT(group);
  IREE_ASSERT_ARGUMENT(out_move_count);
  *out_move_count = 0;
  if (group->copy_start > allocation->edge_copy_count ||
      group->copy_count > allocation->edge_copy_count - group->copy_start) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "edge-copy group range is outside allocation");
  }
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &allocation->edge_copies[group->copy_start + i];
    if (edge_copy->source_assignment_index >= allocation->assignment_count ||
        edge_copy->destination_assignment_index >=
            allocation->assignment_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "edge-copy references an assignment outside allocation");
    }
    const loom_low_allocation_assignment_t* source_assignment =
        &allocation->assignments[edge_copy->source_assignment_index];
    const loom_low_allocation_assignment_t* destination_assignment =
        &allocation->assignments[edge_copy->destination_assignment_index];
    if (source_assignment->location_count !=
        destination_assignment->location_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "edge-copy source and destination location counts differ");
    }
    if (source_assignment->location_count >
        IREE_HOST_SIZE_MAX - *out_move_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "edge-copy unit move count exceeds host size");
    }
    *out_move_count += source_assignment->location_count;
  }
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_populate_edge_copy_units(
    const loom_low_allocation_sidecar_t* allocation,
    const loom_low_allocation_edge_copy_group_t* group, loom_low_move_t* moves,
    iree_host_size_t move_count) {
  IREE_ASSERT_ARGUMENT(allocation);
  IREE_ASSERT_ARGUMENT(group);
  if (move_count != 0) {
    IREE_ASSERT_ARGUMENT(moves);
  }
  iree_host_size_t expected_move_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_count_edge_copy_units(
      allocation, group, &expected_move_count));
  if (move_count != expected_move_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "edge-copy move storage has %zu entries but needs "
                            "%zu",
                            move_count, expected_move_count);
  }
  iree_host_size_t move_index = 0;
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &allocation->edge_copies[group->copy_start + i];
    const loom_low_allocation_assignment_t* source_assignment =
        &allocation->assignments[edge_copy->source_assignment_index];
    const loom_low_allocation_assignment_t* destination_assignment =
        &allocation->assignments[edge_copy->destination_assignment_index];
    for (uint32_t unit_index = 0;
         unit_index < source_assignment->location_count; ++unit_index) {
      IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
          destination_assignment, unit_index, &moves[move_index].destination));
      IREE_RETURN_IF_ERROR(loom_low_move_location_from_assignment_unit(
          source_assignment, unit_index, &moves[move_index].source));
      ++move_index;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_move_sequence_populate_edge_copy_temporaries(
    const loom_low_allocation_sidecar_t* allocation,
    const loom_low_allocation_edge_copy_group_t* group,
    loom_low_move_location_t* temporary_locations,
    iree_host_size_t temporary_location_count) {
  IREE_ASSERT_ARGUMENT(allocation);
  IREE_ASSERT_ARGUMENT(group);
  if (temporary_location_count != 0) {
    IREE_ASSERT_ARGUMENT(temporary_locations);
  }
  if (temporary_location_count != group->temporary_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "edge-copy temporary storage has %zu entries but needs %" PRIu32,
        temporary_location_count, group->temporary_count);
  }
  if (group->temporary_start > allocation->edge_copy_temporary_count ||
      group->temporary_count >
          allocation->edge_copy_temporary_count - group->temporary_start) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "edge-copy temporary group range is outside allocation");
  }
  for (uint32_t i = 0; i < group->temporary_count; ++i) {
    loom_low_move_location_from_edge_copy_temporary(
        &allocation->edge_copy_temporaries[group->temporary_start + i],
        &temporary_locations[i]);
  }
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

static iree_status_t loom_low_move_sequence_find_temporary(
    const loom_low_move_sequence_options_t* options,
    const loom_low_move_location_t* storage_class,
    const loom_low_move_location_t** out_temporary_location) {
  IREE_ASSERT_ARGUMENT(out_temporary_location);
  *out_temporary_location = NULL;
  for (iree_host_size_t i = 0; i < options->temporary_location_count; ++i) {
    const loom_low_move_location_t* temporary =
        &options->temporary_locations[i];
    if (!loom_low_move_locations_share_storage_class(temporary,
                                                     storage_class)) {
      continue;
    }
    *out_temporary_location = temporary;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "parallel move cycle requires a temporary "
                          "location");
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
  if (options->temporary_location_count != 0 &&
      options->temporary_locations == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parallel move temporary locations are required");
  }
  for (iree_host_size_t i = 0; i < move_count; ++i) {
    for (iree_host_size_t j = 0; j < options->temporary_location_count; ++j) {
      const loom_low_move_location_t* temporary =
          &options->temporary_locations[j];
      if (loom_low_move_locations_equal(temporary, &moves[i].destination) ||
          loom_low_move_locations_equal(temporary, &moves[i].source)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "parallel move temporary location aliases move storage");
      }
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
  for (iree_host_size_t i = 0; i < options->temporary_location_count; ++i) {
    const loom_low_move_location_t* lhs = &options->temporary_locations[i];
    for (iree_host_size_t j = i + 1; j < options->temporary_location_count;
         ++j) {
      if (!loom_low_move_locations_equal(lhs,
                                         &options->temporary_locations[j])) {
        continue;
      }
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parallel move temporary locations contain duplicates");
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
  const loom_low_move_location_t saved_location = moves[0].destination;
  const loom_low_move_location_t* temporary_location = NULL;
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_find_temporary(
      options, &saved_location, &temporary_location));
  IREE_RETURN_IF_ERROR(loom_low_move_sequence_emit_one(
      options, temporary_location, &saved_location));

  loom_low_move_location_t destination = moves[0].destination;
  loom_low_move_location_t source = moves[0].source;
  for (;;) {
    iree_host_size_t move_index = IREE_HOST_SIZE_MAX;
    IREE_RETURN_IF_ERROR(loom_low_move_sequence_find_move_by_destination(
        moves, *move_count, &destination, &move_index));
    if (loom_low_move_locations_equal(&source, &saved_location)) {
      IREE_RETURN_IF_ERROR(loom_low_move_sequence_emit_one(
          options, &destination, temporary_location));
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
