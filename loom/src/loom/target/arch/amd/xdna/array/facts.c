// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/array/facts.h"

#include "loom/target/arch/amd/xdna/array_facts_tables.inl"

const loom_xdna_array_family_t* loom_xdna_npu2_array_family(void) {
  return &kLoomXdnaNpu2ArrayFamily;
}

const loom_xdna_tile_facts_t* loom_xdna_array_tile_facts(
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t coordinate) {
  for (uint8_t i = 0; i < family->tile_count; ++i) {
    const loom_xdna_tile_facts_t* facts = &family->tiles[i];
    if (coordinate.row >= facts->first_row &&
        coordinate.row < facts->first_row + facts->row_count) {
      return facts;
    }
  }
  IREE_ASSERT_UNREACHABLE("generated tile rows cover admitted coordinates");
  return NULL;
}

const loom_xdna_tile_facts_t* loom_xdna_array_tile_kind_facts(
    const loom_xdna_array_family_t* family, loom_xdna_tile_kind_t tile_kind) {
  for (uint8_t i = 0; i < family->tile_count; ++i) {
    const loom_xdna_tile_facts_t* facts = &family->tiles[i];
    if (facts->kind == tile_kind) {
      return facts;
    }
  }
  IREE_ASSERT_UNREACHABLE("generated tile kinds cover target planning");
  return NULL;
}

const loom_xdna_stream_port_range_t* loom_xdna_array_stream_port_range(
    const loom_xdna_array_family_t* family, loom_xdna_tile_kind_t tile_kind,
    loom_xdna_stream_direction_t direction, loom_xdna_stream_port_t port) {
  for (uint8_t i = 0; i < family->stream_port_count; ++i) {
    const loom_xdna_stream_port_range_t* range = &family->stream_ports[i];
    if (range->tile_kind == tile_kind && range->direction == direction &&
        range->port == port) {
      return range;
    }
  }
  IREE_ASSERT_UNREACHABLE("generated stream ports cover target routing");
  return NULL;
}

uint32_t loom_xdna_array_stream_destination_channels(
    const loom_xdna_array_family_t* family,
    const loom_xdna_stream_port_range_t* source, uint8_t source_channel,
    const loom_xdna_stream_port_range_t* destination) {
  return (family->stream_connections[source->connection_offset +
                                     source_channel] >>
          destination->ordinal) &
         (UINT32_MAX >> (32 - destination->count));
}

const loom_xdna_address_window_t* loom_xdna_array_load_window(
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t accessor, loom_xdna_tile_coordinate_t owner) {
  const loom_xdna_tile_facts_t* accessor_tile =
      loom_xdna_array_tile_facts(family, accessor);
  const loom_xdna_tile_facts_t* owner_tile =
      loom_xdna_array_tile_facts(family, owner);
  for (uint8_t i = 0; i < accessor_tile->memory.window_count; ++i) {
    const loom_xdna_address_window_t* window =
        &family->address_windows[accessor_tile->memory.window_start + i];
    if ((int32_t)accessor.column + window->owner_column_delta == owner.column &&
        (int32_t)accessor.row + window->owner_row_delta == owner.row &&
        window->owner_kind == owner_tile->kind) {
      return window;
    }
  }
  return NULL;
}
