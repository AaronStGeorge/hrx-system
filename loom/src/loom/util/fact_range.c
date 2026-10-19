// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_range.h"

#include <string.h>

static void loom_value_fact_range_fill(uint8_t* entries, iree_host_size_t count,
                                       iree_host_size_t element_size,
                                       const void* empty_entry) {
  if (!empty_entry) {
    memset(entries, 0, count * element_size);
  } else {
    for (iree_host_size_t i = 0; i < count; ++i) {
      memcpy(entries + i * element_size, empty_entry, element_size);
    }
  }
}

iree_status_t loom_value_fact_range_grow(iree_arena_allocator_t* arena,
                                         loom_value_id_t value_id,
                                         iree_host_size_t element_size,
                                         const void* empty_entry,
                                         loom_value_fact_range_t* range,
                                         void** entries) {
  if (loom_value_fact_range_contains(*range, value_id)) {
    return iree_ok_status();
  }
  loom_value_id_t first_value_id = value_id;
  uint64_t end_value_id = (uint64_t)value_id + 1;
  if (range->capacity) {
    first_value_id = iree_min(first_value_id, range->first_value_id);
    end_value_id = iree_max(end_value_id,
                            (uint64_t)range->first_value_id + range->capacity);
  }
  const uint64_t minimum_capacity = end_value_id - first_value_id;
  if (minimum_capacity > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fact range capacity overflow");
  }
  iree_host_size_t capacity = range->capacity;
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_grow_array(arena, 0, (iree_host_size_t)minimum_capacity,
                            element_size, &capacity, (void**)&storage));
  // Keep spare capacity at the growing edge. Descending definitions otherwise
  // prepend one slot at a time while repeatedly doubling an unused suffix.
  if (range->capacity && value_id < range->first_value_id) {
    first_value_id = end_value_id > capacity
                         ? (loom_value_id_t)(end_value_id - capacity)
                         : 0;
  }
  const iree_host_size_t prefix_count =
      range->capacity ? range->first_value_id - first_value_id : 0;
  loom_value_fact_range_fill(storage, prefix_count, element_size, empty_entry);
  if (range->capacity) {
    memcpy(storage + prefix_count * element_size, *entries,
           range->capacity * element_size);
  }
  const iree_host_size_t suffix_offset = prefix_count + range->capacity;
  loom_value_fact_range_fill(storage + suffix_offset * element_size,
                             capacity - suffix_offset, element_size,
                             empty_entry);
  *range = (loom_value_fact_range_t){first_value_id, capacity};
  *entries = storage;
  return iree_ok_status();
}
