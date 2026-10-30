// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/write_interference_events.h"

#include <string.h>

#include "iree/base/bitmap.h"
#include "loom/codegen/low/allocation/assignment.h"

// Called only for full construction arrays. Limiting logical capacity makes
// the next growth fail before an append can use UINT32_MAX as a row index.
static iree_status_t loom_low_write_grow_array(iree_arena_allocator_t* arena,
                                               iree_host_size_t count,
                                               iree_host_size_t element_size,
                                               iree_host_size_t* inout_capacity,
                                               void** inout_values) {
  if (count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "retained-write rows exceed u32 index capacity");
  }
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      arena, count, count + 1, element_size, inout_capacity, inout_values));
  *inout_capacity = iree_min(*inout_capacity, UINT32_MAX);
  return iree_ok_status();
}

iree_status_t loom_low_write_events_append(loom_low_write_events_t* events,
                                           uint32_t point,
                                           loom_low_write_access_t access,
                                           iree_arena_allocator_t* arena) {
  if (events->access_count == events->access_capacity) {
    IREE_RETURN_IF_ERROR(loom_low_write_grow_array(
        arena, events->access_count, sizeof(*events->accesses),
        &events->access_capacity, (void**)&events->accesses));
  }
  access.next = events->points[point].access;
  events->points[point].access = (uint32_t)events->access_count;
  events->accesses[events->access_count++] = access;
  return iree_ok_status();
}

// Known writes cut only the overwritten units. For ALU writes this boundary
// promises the physical planner's post-write dependency; memory results carry
// their ordinary asynchronous completion dependency before reuse.
// Returns whether a previously materialized snapshot must be rebuilt.
static bool loom_low_write_transfer(const loom_low_write_events_t* events,
                                    const loom_low_write_value_t* values,
                                    uint32_t point,
                                    const loom_low_write_retained_t* units,
                                    iree_bitmap_t active) {
  bool invalidated = false;
  const loom_low_write_event_t* event = &events->points[point];
  for (uint32_t a = event->access; a != UINT32_MAX;
       a = events->accesses[a].next) {
    const loom_low_write_access_t* access = &events->accesses[a];
    if (!iree_any_bit_set(access->flags, LOOM_LOW_WRITE_ACCESS_WRITE)) {
      continue;
    }
    if (access->source != LOOM_VALUE_ORDINAL_INVALID) {
      int64_t delta = 0;
      if (!loom_low_write_known_delta(values, access->value, access->source,
                                      &delta) ||
          delta + access->offset == access->source_offset) {
        continue;
      }
    }
    for (iree_host_size_t bit = iree_bitmap_find_first_set(active, 0);
         bit < active.bit_count;
         bit = iree_bitmap_find_first_set(active, bit + 1)) {
      const loom_low_write_retained_t* unit = &units[bit];
      int64_t delta = 0;
      if (loom_low_write_known_delta(values, access->value, unit->value,
                                     &delta) &&
          unit->offset >= delta + access->offset &&
          unit->offset < delta + access->offset + access->count) {
        iree_bitmap_reset(active, bit);
        invalidated = true;
      }
    }
  }
  for (uint32_t a = event->access; a != UINT32_MAX;
       a = events->accesses[a].next) {
    const loom_low_write_access_t* access = &events->accesses[a];
    if (iree_any_bit_set(access->flags, LOOM_LOW_WRITE_ACCESS_RETAIN)) {
      iree_bitmap_set_span(active,
                           values[access->value].construction.retained_start,
                           access->count);
      invalidated = true;
    }
  }
  return invalidated;
}

static iree_status_t loom_low_write_snapshot(
    loom_low_write_events_t* events, const loom_low_write_retained_t* units,
    iree_bitmap_t active, iree_arena_allocator_t* arena,
    loom_low_write_range_t* out_snapshot) {
  *out_snapshot =
      (loom_low_write_range_t){.start = (uint32_t)events->retained_count};
  for (iree_host_size_t bit = iree_bitmap_find_first_set(active, 0);
       bit < active.bit_count;) {
    loom_low_write_retained_t range = units[bit++];
    while (bit < active.bit_count && iree_bitmap_test(active, bit) &&
           units[bit].value == range.value &&
           units[bit].offset == range.offset + range.count) {
      ++range.count;
      ++bit;
    }
    if (events->retained_count == events->retained_capacity) {
      IREE_RETURN_IF_ERROR(loom_low_write_grow_array(
          arena, events->retained_count, sizeof(*events->retained),
          &events->retained_capacity, (void**)&events->retained));
    }
    events->retained[events->retained_count++] = range;
    ++out_snapshot->count;
    if (bit < active.bit_count) {
      bit = iree_bitmap_find_first_set(active, bit);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_write_count_constraints(
    const loom_low_write_events_t* events, uint32_t point,
    iree_host_size_t* inout_constraint_count) {
  const loom_low_write_event_t* event = &events->points[point];
  if (event->retained.count == 0) {
    return iree_ok_status();
  }
  for (uint32_t a = event->access; a != UINT32_MAX;
       a = events->accesses[a].next) {
    const loom_low_write_access_t* access = &events->accesses[a];
    if (iree_any_bit_set(access->flags, LOOM_LOW_WRITE_ACCESS_INTERFERE)) {
      const uint32_t write_count =
          access->source != LOOM_VALUE_ORDINAL_INVALID ? access->count : 1;
      const uint64_t row_count = (uint64_t)event->retained.count * write_count;
      if (row_count > UINT32_MAX - *inout_constraint_count) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "retained-write table exceeds u32 index capacity");
      }
      *inout_constraint_count += row_count;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_write_events_solve(
    loom_low_write_events_t* events, const loom_low_write_value_t* values,
    iree_host_size_t value_count, uint32_t retained_unit_count,
    const loom_low_write_flow_t* flow, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    iree_host_size_t* out_constraint_count) {
  *out_constraint_count = 0;
  loom_low_write_retained_t* units = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, retained_unit_count, sizeof(*units), (void**)&units));
  for (uint32_t v = 0; v < value_count; ++v) {
    const loom_low_write_value_t* value = &values[v];
    if (value->construction.retained_start == UINT32_MAX) {
      continue;
    }
    for (uint32_t u = 0; u < value->construction.width; ++u) {
      units[value->construction.retained_start + u] =
          (loom_low_write_retained_t){v, u, 1};
    }
  }
  const iree_host_size_t word_count =
      iree_bitmap_calculate_words(retained_unit_count);
  const iree_host_size_t byte_count = word_count * sizeof(uint64_t);
  uint64_t* incoming = NULL;
  uint64_t* active_words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, flow->block_count, byte_count, (void**)&incoming));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(scratch_arena, byte_count, (void**)&active_words));
  memset(incoming, 0, flow->block_count * byte_count);
  iree_bitmap_t active = {.bit_count = retained_unit_count,
                          .words = active_words};
  // All transfers are monotone gen/kill maps. Only changed incoming sets
  // require another evaluation. Seed every span: local reads can generate
  // retained state even without incoming bits or reachable predecessors.
  uint32_t* pending_next = flow->worklist;
  for (uint32_t b = 0; b < flow->block_count; ++b) {
    pending_next[b] = b + 1;
  }
  uint32_t pending_head = 0;
  uint32_t pending_tail = flow->block_count - 1;
  pending_next[pending_tail] = UINT32_MAX;
  while (pending_head != UINT32_MAX) {
    const uint32_t b = pending_head;
    pending_head = pending_next[b];
    // A self-link marks absence; queued links name a distinct next span or
    // UINT32_MAX at the tail. Removing membership before transfer permits
    // a self-loop to enqueue this span again without a separate bitmap.
    pending_next[b] = b;
    const loom_low_write_flow_block_t* block = &flow->blocks[b];
    memcpy(active_words, incoming + b * word_count, byte_count);
    for (uint32_t p = block->begin; p < block->end; ++p) {
      if (events->points[p].reset) {
        iree_bitmap_reset_all(active);
      }
      loom_low_write_transfer(events, values, p, units, active);
    }
    for (uint32_t s = 0; s < block->successor_count; ++s) {
      const uint32_t successor = flow->successors[block->successor_start + s];
      uint64_t* next = incoming + successor * word_count;
      bool changed = false;
      for (iree_host_size_t w = 0; w < word_count; ++w) {
        const uint64_t added = active_words[w] & ~next[w];
        next[w] |= added;
        changed |= added != 0;
      }
      if (changed && pending_next[successor] == successor) {
        if (pending_head == UINT32_MAX) {
          pending_head = successor;
        } else {
          pending_next[pending_tail] = successor;
        }
        pending_tail = successor;
        pending_next[successor] = UINT32_MAX;
      }
    }
  }
  for (uint32_t b = 0; b < flow->block_count; ++b) {
    const loom_low_write_flow_block_t* block = &flow->blocks[b];
    memcpy(active_words, incoming + b * word_count, byte_count);
    loom_low_write_range_t snapshot = {0};
    bool snapshot_invalidated = true;
    for (uint32_t p = block->begin; p < block->end; ++p) {
      if (events->points[p].reset) {
        iree_bitmap_reset_all(active);
        snapshot = (loom_low_write_range_t){0};
        snapshot_invalidated = false;
      }
      // The transfer owns every mutation of |active|. Preserve its snapshot
      // until that producer invalidates it instead of comparing or rebuilding
      // identical ranges at every point. A reset establishes the empty set.
      if (snapshot_invalidated) {
        IREE_RETURN_IF_ERROR(
            loom_low_write_snapshot(events, units, active, arena, &snapshot));
      }
      events->points[p].retained = snapshot;
      IREE_RETURN_IF_ERROR(
          loom_low_write_count_constraints(events, p, out_constraint_count));
      snapshot_invalidated =
          loom_low_write_transfer(events, values, p, units, active);
    }
  }
  return iree_ok_status();
}

bool loom_low_write_events_temporary_conflicts(
    const loom_low_write_events_t* events,
    const loom_low_allocation_assignment_t* assignments,
    const uint32_t* assignment_indices_by_origin, uint32_t write_point,
    uint32_t temporary_location) {
  const loom_low_write_range_t range = events->points[write_point].retained;
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_write_retained_t* retained =
        &events->retained[range.start + i];
    const loom_low_allocation_assignment_t* assignment =
        &assignments[assignment_indices_by_origin[retained->value]];
    const int64_t offset = (int64_t)temporary_location -
                           assignment->location_base - retained->offset;
    if (offset >= 0 && offset < retained->count) {
      return true;
    }
  }
  return false;
}
